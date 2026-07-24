"""IncrementalSession — L1+L2+L3 processing during live recording (latency
Wave 3, docs/incremental_capture_design.md).

CLINICAL-SAFETY CONTRACT (same as web/live_asr.py, strengthened, not
weakened): `partial_turns()` is UI-DISPLAY-ONLY. It must never feed L4, never
land in note.json, and must always render under the label "LIVE PREVIEW —
final transcript follows" with no structured fields. Unlike live_asr's
preview, partial_turns() DOES use the full production decode path (guards +
degeneration ladder — src.fast_asr.decode_windows_words), so its text quality
is the same as the final transcript; only `finalize()`'s output is
authoritative and may be persisted or extracted from.

Design (see docs/incremental_capture_design.md for the full rationale):
every `feed()` call may decode newly SETTLED audio — the portion of the
buffer more than `settle_margin_s` behind the growing edge — with the exact
same guarded, ladder-backed decode path production uses
(src.fast_asr.decode_windows_words), and FREEZES that decode: because the
production decode is deterministic (temperature=0.0,
condition_on_previous_text=False — src.config.FAST_ASR_DECODE_KWARGS),
re-decoding the same settled audio later would reproduce the same words, so
`finalize()` never re-runs it. `finalize()` only decodes the short remaining
tail and reconciles diarization, which is why stop-to-note collapses to
roughly (tail decode ∥ final diarize) instead of the full clip's ASR cost.

Diarization cannot be packed the same way: only a PROVISIONAL, still-changing
diarization exists while recording continues, so `partial_turns()` re-runs
pyannote on the WHOLE buffer each call (not incrementally) and `finalize()`
runs one more, final full-file pass. Both use the resident pipeline
(src/model_registry.py) rather than reloading per call.

L4 KV prefix warming (latency Wave 4): `warm_l4_prefix()` fires a background
Ollama call (src.l4_extract.warm_llm_prefix) with the turns settled so far,
priming the model's KV cache for the eventual extract() call at finalize()
time — never feeds L4 for real, purely a cache-priming side effect. Tracks
`kv_warm_wasted` (an int counter) whenever `finalize()`'s actual prompt hash
doesn't match the last warm's hash — never a correctness issue (extract()
always sends the right prompt regardless), just a missed optimization.

NOT built in this pass (see LEARNINGS.md's Wave 3 entry for why): the FastAPI
routes and browser capture-loop wiring that would actually call feed() during
a live recording. Today's API surface (POST /api/sessions,
POST /api/sessions/{sid}/process) is a single whole-file upload with no
session-scoped chunked ingestion — building routes with no real capture UI to
exercise them would be exactly the speculative scaffolding CLAUDE.md's
Development Rules warn against. This module is a rigorously tested backend
primitive (see tests/test_incremental.py's real-audio equivalence test);
wiring it into HTTP routes + the browser is deliberately left as follow-up
work, to be done alongside (and tested against) an actual capture screen.
"""

import io
import logging
import os
import tempfile
import threading

import librosa
import numpy as np
import soundfile as sf

from src import config, model_registry
from src.fast_asr import (
    Word,
    _assign_words_to_turns,
    _build_turns_with_roles,
    decode_windows_words,
    pack_duration_into_windows,
)
from src.l1_preprocess import TARGET_SR
from src.l2_diarize import diarize as _diarize
from src.types import Turn

logger = logging.getLogger(__name__)


class IncrementalSession:
    """Owns one consultation's growing audio buffer during live recording.

    Not thread-safe across `finalize()` and a concurrent `feed()` — callers
    must stop feeding before calling finalize() (the natural sequencing of
    "doctor presses stop"). `feed()` calls themselves ARE safe to interleave
    with `partial_turns()` polls from another thread.
    """

    def __init__(
        self,
        session_id: str,
        *,
        settle_margin_s: float | None = None,
        max_window_s: float | None = None,
    ):
        self.session_id = session_id
        self._settle_margin_s = (
            settle_margin_s
            if settle_margin_s is not None
            else config.INCREMENTAL_SETTLE_MARGIN_S
        )
        self._max_window_s = max_window_s if max_window_s is not None else config.WINDOW_MAX_SPAN_S
        self._lock = threading.Lock()
        self._audio = np.zeros(0, dtype=np.float32)
        self._decoded_up_to_s = 0.0  # end of settled audio already decoded
        self._settled_words: list[Word] = []
        # Latency Wave 4 (src/l4_extract.py's KV prefix warming): the hash of
        # the last prompt this session fired a background warm() for, and how
        # many times finalize()'s actual prompt didn't match it (the warm was
        # wasted — never wrong, since extract() always sends the correct
        # prompt regardless — just didn't help).
        self._last_l4_warm_hash: str | None = None
        self.kv_warm_wasted = 0

    def feed(self, wav_bytes: bytes) -> None:
        """Append a chunk of newly recorded audio; decode any newly settled span.

        Args:
            wav_bytes: Raw bytes of the NEW audio increment (not the whole
                buffer) in any format librosa can decode.
        """
        chunk, _ = librosa.load(io.BytesIO(wav_bytes), sr=TARGET_SR, mono=True)
        with self._lock:
            self._audio = np.concatenate([self._audio, chunk])
            self._decode_newly_settled_locked()

    def _decode_newly_settled_locked(self) -> None:
        """Decode complete max_window_s-sized chunks of settled backlog only.

        Deliberately does NOT decode on every feed() tick just because
        something settled: a decode call costs ~7-9s fixed regardless of how
        much audio it covers (src/fast_asr.py's Fix 2 rationale — the whole
        reason windowed decoding exists over one-call-per-segment). Firing a
        decode for every ~10s tick interval would create MORE, SMALLER calls
        than the batch path's ~28s-capped windows — worse, not better. So the
        backlog (settled-but-undecoded span) accumulates across feed() calls
        until it holds at least one full max_window_s chunk; any leftover
        remainder under that threshold waits for the next tick. finalize()
        does not use this gate — the tail must decode whatever backlog is
        left, however small, since no more audio is coming.
        """
        total_duration_s = len(self._audio) / TARGET_SR
        settled_end_s = max(0.0, total_duration_s - self._settle_margin_s)
        backlog = settled_end_s - self._decoded_up_to_s
        if backlog < self._max_window_s:
            return
        n_full_windows = int(backlog // self._max_window_s)
        windows = [
            (
                self._decoded_up_to_s + i * self._max_window_s,
                self._decoded_up_to_s + (i + 1) * self._max_window_s,
            )
            for i in range(n_full_windows)
        ]
        new_words = decode_windows_words(self._audio, TARGET_SR, windows)
        self._settled_words.extend(new_words)
        self._decoded_up_to_s = windows[-1][1]
        logger.info(
            "Session %s: settled %.1fs (%d new words, %d windows)",
            self.session_id,
            windows[-1][1] - windows[0][0],
            len(new_words),
            len(windows),
        )

    def _diarize_current_buffer(self, audio: np.ndarray):
        """Write `audio` to a temp WAV and run the resident pyannote pipeline.

        A temp-file round trip because src.l2_diarize.diarize() reads a path
        (see its own docstring: torchcodec is broken here, so it already
        works around in-memory audio itself) — not worth widening L2's
        stable contract for this one caller.
        """
        pipeline = model_registry.get_pyannote_pipeline()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            sf.write(tmp_path, audio, TARGET_SR, subtype="PCM_16")
            return _diarize(tmp_path, pipeline=pipeline)
        finally:
            os.unlink(tmp_path)

    def partial_turns(self) -> list[Turn]:
        """UI-DISPLAY-ONLY running transcript — see module docstring's contract.

        Re-diarizes the whole buffer so far (provisional — speaker labels may
        still change) and attributes the settled words decoded up to now.
        Returns an empty list if nothing has settled yet.
        """
        with self._lock:
            audio = self._audio.copy()
            words = list(self._settled_words)
        if not words:
            return []
        segments = self._diarize_current_buffer(audio)
        if not segments:
            return []
        raw_turns = _assign_words_to_turns(words, segments)
        return _build_turns_with_roles(raw_turns)

    def warm_l4_prefix(self) -> str | None:
        """Fire a background L4 KV-warm call with turns settled so far.

        Reuses partial_turns() for attribution (one diarization pass, same
        cost a UI-polling caller already pays) rather than a separate one.
        The actual Ollama call runs on a daemon thread — this method returns
        as soon as the prefix hash is computed, without waiting on the
        network call, so it is cheap enough to call after every settle event.

        Returns:
            The prefix hash of what was (attempted to be) warmed, or None if
            there is nothing settled yet to warm with.
        """
        from src import l4_extract

        turns = self.partial_turns()
        if not turns:
            return None
        prefix_hash = l4_extract.prompt_prefix_hash(turns)
        self._last_l4_warm_hash = prefix_hash
        threading.Thread(
            target=l4_extract.warm_llm_prefix, args=(turns,), daemon=True
        ).start()
        return prefix_hash

    def finalize(self) -> list[Turn]:
        """Authoritative final pass: tail decode ∥ final diarize, then attribute.

        The only unsettled audio left is the short tail (<= settle_margin_s
        plus whatever arrived since the last feed()) — decoding it and
        re-diarizing the full buffer run concurrently, so the stop-time cost
        is roughly max(tail decode, final diarize) instead of their sum.
        """
        with self._lock:
            audio = self._audio.copy()
            decoded_up_to_s = self._decoded_up_to_s
            settled_words = list(self._settled_words)

        total_duration_s = len(audio) / TARGET_SR
        tail_windows = [
            (decoded_up_to_s + s, decoded_up_to_s + e)
            for s, e in pack_duration_into_windows(
                total_duration_s - decoded_up_to_s, self._max_window_s
            )
        ]

        tail_words: list[Word] = []
        segments = []

        def _decode_tail():
            nonlocal tail_words
            tail_words = decode_windows_words(audio, TARGET_SR, tail_windows)

        def _final_diarize():
            nonlocal segments
            segments = self._diarize_current_buffer(audio)

        t_decode = threading.Thread(target=_decode_tail)
        t_diarize = threading.Thread(target=_final_diarize)
        t_decode.start()
        t_diarize.start()
        t_decode.join()
        t_diarize.join()

        all_words = settled_words + tail_words
        if not all_words or not segments:
            return []
        raw_turns = _assign_words_to_turns(all_words, segments)
        turns = _build_turns_with_roles(raw_turns)

        if turns and self._last_l4_warm_hash is not None:
            from src import l4_extract

            if l4_extract.prompt_prefix_hash(turns) != self._last_l4_warm_hash:
                self.kv_warm_wasted += 1
                logger.info(
                    "Session %s: kv_warm_wasted (finalize prompt != last warm)",
                    self.session_id,
                )

        return turns
