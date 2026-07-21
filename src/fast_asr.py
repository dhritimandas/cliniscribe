"""L3-fast — mlx-whisper transcription with a chunk-local hallucination-retry
ladder (see src.config.FAST_ASR_ENABLED for gating status).

Same contract semantics as src.l3_asr.transcribe: per-segment decode, a
script guard that force-corrects Arabic-script misdetections of Hindi, and
the doctor/patient role heuristic (reusing src.l3_asr._doctor_score and
src.l3_asr._contains_arabic_script directly rather than reimplementing them).

The engine itself is mlx-whisper on the Metal GPU, not faster-whisper: it is
~2-3x faster per eval/engine_study.py, at the cost of confident repetition-
loop hallucinations on some short/hard Hindi segments (documented there, not
threshold-fixable). This module's answer is chunk-local detect+retry, never a
fallback to the slow whole-recording method — see looks_degenerate and
_retry_ladder.

mlx_whisper.audio.load_audio shells out to the ffmpeg CLI, which is not
installed on this machine (see eval/engine_study.py module docstring). We
therefore never pass a path to mlx_whisper.transcribe: the wav is decoded
once via soundfile into an in-memory float32 array, and each segment (or
ladder retry window) is sliced from that array by hand.

v2 adds two fixes found by the v1 frozen-bench gate (outputs/fast_asr_gate.json,
FAIL on corpus/keyword WER):

Fix 1 -- language-allowlist guard (_decode_with_script_guard): v1's ladder
resolved every degenerate (repetition-loop) segment, yet the gate still
failed, because some segments decode into a WRONG but FLUENT language --
Portuguese/Turkish/Indonesian words interleaved with real Hindi/English --
which looks_degenerate cannot see (it is not a loop). mlx_whisper.transcribe's
returned `language` is now checked against config.ASR_LANGUAGE_ALLOWLIST on
every decode and force-corrected to "hi", belt-and-braces alongside the
existing Arabic-script guard.

Fix 2 -- window-packed decoding (fast_transcribe_windowed): every
mlx_whisper.transcribe() call costs a near-constant ~7-9s because Whisper
pads every input to a 30s window before encoding, regardless of how much
audio it actually contains (see config.WINDOW_MAX_SPAN_S). fast_transcribe
therefore scales with SEGMENT COUNT, not audio duration (12 segments = 104s
for 26s of audio). fast_transcribe_windowed packs diarized segments into
<=28s windows (_pack_segments_into_windows), decodes each window ONCE with
word timestamps, and re-attributes words to the original diarized segments
by midpoint overlap (_segment_index_for_midpoint / _assign_words_to_turns,
adapted from the whisperX-alignment helpers on
worktree-agent-a9241a6fe03e8de65:src/l3_asr.py -- see those functions'
docstrings for the attribution note). KNOWN RISK, carried over from the
project's earlier merged-window study (LEARNINGS.md, "Latency Phase"): a
window spanning many seconds can still absorb a code-switched Latin term into
the dominant Devanagari script (measured directly: eval.engine_study.merge_segments
collapses clip 431d272c, 20.608s duration, into one 20.402s single-speaker
window, decoded by both fw-large-v3+merged and mlx-large-v3+merged as
"लाप टेस्ट" for "lab test") -- eval/fast_asr_gate.py's frozen-bench gate is
the arbiter of whether this config ships.
"""

import logging
import re
import zlib
from collections.abc import Callable, Sequence

import numpy as np
import soundfile as sf

from src import config
from src.l3_asr import _contains_arabic_script, _doctor_score
from src.types import Segment, Turn

logger = logging.getLogger(__name__)

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_TOKEN_PUNCT_RE = re.compile(r"[.,?!;:।॥\"'()\[\]{}]")

# One transcribed word: (start, end, text), absolute clip-time seconds. A
# local alias, not a stable inter-stage contract like Segment/Turn
# (src/types.py) -- it never crosses a module boundary. Matches the alias
# on worktree-agent-a9241a6fe03e8de65:src/l3_asr.py (see fast_transcribe_windowed).
Word = tuple[float, float, str]


# ── Degeneration detector ────────────────────────────────────────────────────


def _tokens(text: str) -> list[str]:
    return _TOKEN_PUNCT_RE.sub(" ", text).split()


def _max_consecutive_phrase_repeats(tokens: list[str], n: int) -> int:
    """Largest number of times a length-`n` phrase repeats back-to-back.

    Non-overlapping windows: for tokens ["a","b","a","b","a","b"] and n=2,
    the phrase ("a","b") repeats 3 times. This (not a sliding-window ngram
    comparison) is what actually detects "phrase said K times in a row" —
    overlapping windows of a period-n loop alternate and never compare equal
    to their immediate neighbor.

    Args:
        tokens: Whitespace/punctuation-tokenized decode text.
        n: Phrase length in tokens.

    Returns:
        The largest consecutive-repeat count found (>= 1; 1 means no repeat).
    """
    if n <= 0 or n > len(tokens):
        return 0
    total = len(tokens)
    best = 1
    i = 0
    while i + n <= total:
        phrase = tokens[i : i + n]
        count = 1
        j = i + n
        while j + n <= total and tokens[j : j + n] == phrase:
            count += 1
            j += n
        best = max(best, count)
        i = j if count > 1 else i + 1
    return best


def _compression_ratio(text: str) -> float | None:
    """zlib byte-compression ratio, or None if `text` is too short to be meaningful."""
    raw = text.encode("utf-8")
    if len(raw) < config.DEGEN_MIN_LEN_FOR_RATIO:
        return None
    compressed = zlib.compress(raw)
    return len(raw) / len(compressed) if compressed else float("inf")


def looks_degenerate(text: str, audio_seconds: float) -> bool:
    """True if `text` shows signs of a Whisper repetition-loop hallucination.

    Four independent signals (any one firing is enough):
      (a) zlib compression ratio > config.DEGEN_COMPRESSION_RATIO — repeated
          text (word-level or glued sub-word garbage) compresses unusually well.
      (b) some 1-4 token phrase repeats >= config.DEGEN_NGRAM_MIN_REPEAT times
          consecutively (e.g. "college college college college").
      (c) chars-per-audio-second > config.DEGEN_MAX_CHARS_PER_SECOND — no
          hi/en/mr speech decodes this dense; catches glued-token loops with
          no whitespace, which (b) cannot see.
      (d) empty decode on a segment >= config.DEGEN_EMPTY_ON_VOICED_MIN_S long
          — silence-hallucination suppression eating real speech.

    Args:
        text: Decoded text for one segment (or ladder retry window).
        audio_seconds: Duration of the audio that produced `text`.

    Returns:
        True if the text should be treated as degenerate and retried.
    """
    stripped = text.strip()
    if not stripped:
        return audio_seconds >= config.DEGEN_EMPTY_ON_VOICED_MIN_S

    tokens = _tokens(stripped)
    for n in config.DEGEN_NGRAM_SIZES:
        if _max_consecutive_phrase_repeats(tokens, n) >= config.DEGEN_NGRAM_MIN_REPEAT:
            return True

    ratio = _compression_ratio(stripped)
    if ratio is not None and ratio > config.DEGEN_COMPRESSION_RATIO:
        return True

    return (
        audio_seconds > 0
        and len(stripped) / audio_seconds > config.DEGEN_MAX_CHARS_PER_SECOND
    )


def _degeneracy_score(text: str) -> float:
    """Continuous degeneracy proxy for ranking ladder candidates (higher = worse).

    Not used for the pass/fail decision (looks_degenerate is boolean) — only
    to pick the least-bad candidate when every ladder step still degenerates
    (see _retry_ladder step4).
    """
    stripped = text.strip()
    if not stripped:
        return float("inf")
    tokens = _tokens(stripped)
    max_repeat = max(
        (_max_consecutive_phrase_repeats(tokens, n) for n in config.DEGEN_NGRAM_SIZES),
        default=1,
    )
    ratio = _compression_ratio(stripped) or 1.0
    return max(
        max_repeat / config.DEGEN_NGRAM_MIN_REPEAT,
        ratio / config.DEGEN_COMPRESSION_RATIO,
    )


def _is_latin_dominant(text: str) -> bool:
    """True if `text` has more Latin than Devanagari letters.

    Used only to pick the give-up wrap label ("[unclear — VERIFY]" vs
    "[अस्पष्ट — VERIFY]").
    """
    return len(_LATIN_RE.findall(text)) > len(_DEVANAGARI_RE.findall(text))


# ── mlx-whisper per-segment decode ──────────────────────────────────────────


def _decode_window(
    audio: np.ndarray,
    sr: int,
    start: float,
    end: float,
    repo: str,
    *,
    language: str | None,
    decode_kwargs: dict,
) -> tuple[str, str | None]:
    """Slice [start, end) from `audio` in memory and decode it with mlx-whisper.

    Never passes a path to mlx_whisper.transcribe (see module docstring).

    Returns:
        (text, detected_language). `detected_language` is
        mlx_whisper.transcribe's reported `result["language"]" (the value
        `language` was forced to, if it was not None) -- used by the
        language-allowlist guard (Fix 1). None if the window was empty.
    """
    import mlx_whisper

    start_sample = max(0, round(start * sr))
    end_sample = min(len(audio), round(end * sr))
    window = audio[start_sample:end_sample]
    if window.size == 0:
        return "", None
    result = mlx_whisper.transcribe(
        window,
        path_or_hf_repo=repo,
        language=language,
        task="transcribe",
        word_timestamps=False,
        **decode_kwargs,
    )
    return result["text"].strip(), result.get("language")


def _decode_with_script_guard(
    audio: np.ndarray, sr: int, start: float, end: float, repo: str, decode_kwargs: dict
) -> str:
    """Decode a window, then re-decode with language="hi" if either guard fires.

    Two independent, belt-and-braces checks on the SAME initial decode:
      1. Script guard (src.config.ASR_SCRIPT_GUARD): mirrors
         src.l3_asr.transcribe's script guard exactly -- language=None
         auto-detection occasionally misclassifies Hindi as Urdu and writes
         Arabic script; we only support hi/en/mr, so Arabic script is always
         a misdetection.
      2. Language-allowlist guard (Fix 1, src.config.ASR_LANGUAGE_ALLOWLIST):
         auto-detection can also lock onto a wrong but FLUENT language (no
         Arabic script, no repetition loop -- invisible to both the script
         guard and looks_degenerate). Any detected language outside
         {hi, en, mr} is always a misdetection too.
    Only one re-decode ever fires (`elif`): if the script guard already
    caught it, the language guard's target fix (language="hi") is identical,
    so a second re-decode would be wasted work.
    """
    text, detected_language = _decode_window(
        audio, sr, start, end, repo, language=None, decode_kwargs=decode_kwargs
    )
    if config.ASR_SCRIPT_GUARD and _contains_arabic_script(text):
        logger.warning(
            "fast_asr script guard: Arabic-script decode at [%.2f, %.2f]s "
            "('%s') — re-decoding with language=hi",
            start,
            end,
            text[:40],
        )
        text, _ = _decode_window(
            audio, sr, start, end, repo, language="hi", decode_kwargs=decode_kwargs
        )
    elif (
        detected_language is not None
        and detected_language not in config.ASR_LANGUAGE_ALLOWLIST
    ):
        logger.warning(
            "fast_asr language guard: non-allowlist language '%s' detected at "
            "[%.2f, %.2f]s ('%s') — re-decoding with language=hi",
            detected_language,
            start,
            end,
            text[:40],
        )
        text, _ = _decode_window(
            audio, sr, start, end, repo, language="hi", decode_kwargs=decode_kwargs
        )
    return text


# ── Hallucination retry ladder ───────────────────────────────────────────────


def _retry_ladder(
    audio: np.ndarray,
    sr: int,
    seg: Segment,
    total_duration: float,
    *,
    initial_text: str,
) -> tuple[str, str]:
    """Chunk-local retries for one segment whose decode looked degenerate.

    Each step is re-checked by looks_degenerate; the ladder stops at the
    first step that produces clean text. If all three retries still
    degenerate, step4 wraps the least-degenerate candidate (by
    _degeneracy_score, over all 4 attempts) with a VERIFY marker instead of
    blocking or falling back to a slower whole-file decode.

    Args:
        audio: Full-clip float32 mono array (16 kHz).
        sr: Sample rate (16000).
        seg: The diarized segment that degenerated.
        total_duration: Full clip duration in seconds, for clamping step1's
            boundary shift.
        initial_text: The already-degenerate first decode (step0), kept as a
            give-up candidate.

    Returns:
        (final_text, step_name). step_name is one of "step1_boundary_shift",
        "step2_temp_condition", "step3_large_model", "step4_giveup" — logged
        by the caller as the ladder-firing record.
    """
    candidates = [initial_text]

    # Step 1: widen the segment boundaries — a measured loop-breaker.
    shift = config.RETRY_BOUNDARY_SHIFT_S
    start1 = max(0.0, seg.start - shift)
    end1 = min(total_duration, seg.end + shift)
    text1 = _decode_with_script_guard(
        audio, sr, start1, end1, config.FAST_ASR_MODEL, config.FAST_ASR_DECODE_KWARGS
    )
    candidates.append(text1)
    if not looks_degenerate(text1, end1 - start1):
        return text1, "step1_boundary_shift"

    # Step 2: allow conditioning on previous text, slightly warmer sampling.
    step2_kwargs = {
        "temperature": config.RETRY_TEMP_STEP2,
        "condition_on_previous_text": True,
    }
    text2 = _decode_with_script_guard(
        audio, sr, seg.start, seg.end, config.FAST_ASR_MODEL, step2_kwargs
    )
    candidates.append(text2)
    if not looks_degenerate(text2, seg.end - seg.start):
        return text2, "step2_temp_condition"

    # Step 3: escalate to the bigger mlx model, still on the Metal GPU.
    text3 = _decode_with_script_guard(
        audio,
        sr,
        seg.start,
        seg.end,
        config.FAST_ASR_FALLBACK_MODEL,
        config.FAST_ASR_DECODE_KWARGS,
    )
    candidates.append(text3)
    if not looks_degenerate(text3, seg.end - seg.start):
        return text3, "step3_large_model"

    # Step 4: give up gracefully — never block, never revert to whole-file decode.
    best = min(candidates, key=_degeneracy_score)
    marker = "[unclear — VERIFY]" if _is_latin_dominant(best) else "[अस्पष्ट — VERIFY]"
    return f"{marker} {best}".strip(), "step4_giveup"


# ── Speaker-role assignment (shared by both entry points) ────────────────────


def _build_turns_with_roles(
    raw_turns: list[tuple[str, str, float, float]],
) -> list[Turn]:
    """Score each speaker label with src.l3_asr._doctor_score and assign roles.

    Shared by fast_transcribe and fast_transcribe_windowed (identical logic to
    src.l3_asr.transcribe, duplicated here per that module's "do not touch"
    scope — only _doctor_score itself is imported, not reimplemented).

    Args:
        raw_turns: (speaker, text, start, end) tuples in chronological order.

    Returns:
        List of Turn(speaker_role, text, start, end); the speaker with the
        higher aggregate doctor-heuristic score is "DOCTOR", the other
        "PATIENT"; ties or a single speaker keep "UNKNOWN". Empty input
        returns an empty list.
    """
    if not raw_turns:
        return []

    speaker_scores: dict[str, int] = {}
    for speaker, text, _, _ in raw_turns:
        speaker_scores[speaker] = speaker_scores.get(speaker, 0) + _doctor_score(text)

    speakers = list(speaker_scores)
    if len(speakers) < 2:
        role_map = {speakers[0]: "UNKNOWN"}
    else:
        top_speaker = max(speakers, key=lambda s: speaker_scores[s])
        runner_up = [s for s in speakers if s != top_speaker]
        if speaker_scores[top_speaker] > max(speaker_scores[s] for s in runner_up):
            role_map = {top_speaker: "DOCTOR"}
            for s in runner_up:
                role_map[s] = "PATIENT"
        else:
            role_map = {s: "UNKNOWN" for s in speakers}

    return [
        Turn(
            speaker_role=role_map.get(speaker, "UNKNOWN"),
            text=text,
            start=start,
            end=end,
        )
        for speaker, text, start, end in raw_turns
    ]


# ── Window-packed decoding (Fix 2) ───────────────────────────────────────────


def _pack_segments_into_windows(
    segments: Sequence[Segment],
    max_window_s: float = 28.0,
    min_break_gap_s: float = 0.8,
) -> list[list[Segment]]:
    """Partition chronological diarized segments into decode windows.

    Pure function: no I/O, no model calls. Defaults match
    config.WINDOW_MAX_SPAN_S / config.WINDOW_MIN_BREAK_GAP_S at the time of
    writing; the production caller (fast_transcribe_windowed) passes those
    config values explicitly, read at call time (not baked in here as import-
    time defaults), so eval harnesses can override them the same way
    eval/engine_study.py's merge_segments does. Greedily packs consecutive
    segments (regardless of speaker — unlike eval.engine_study.merge_segments,
    which only merges same-speaker runs) into a window whose span
    ([first.start, last.end]) stays within `max_window_s`; a segment is never
    split across two windows.

    When the next segment would overflow the cap, the window must close.
    Preference is given to closing it at the most recent "natural" break
    inside the current run — a speaker change, or a gap >= `min_break_gap_s`
    — PROVIDED the carried-over remainder (the segments after that break,
    plus the overflowing segment) would itself still fit in one window; this
    keeps a forced cut from landing mid-utterance when a cleaner earlier
    break point is available. Otherwise the window closes at the hard span
    boundary (all segments accumulated so far).

    Args:
        segments: Diarized segments, chronologically ordered (as L2 emits
            them). Must be non-empty to receive resumeable windows.
        max_window_s: Maximum span of one decode window, in seconds.
        min_break_gap_s: Minimum silence gap that counts as a natural break.

    Returns:
        A list of non-empty segment groups, one per decode window, in
        chronological order. Empty input returns an empty list.
    """
    if not segments:
        return []

    windows: list[list[Segment]] = []
    current: list[Segment] = [segments[0]]
    preferred_cut: int | None = None  # index into `current`: latest natural break

    for seg in segments[1:]:
        if seg.end - current[0].start <= max_window_s:
            is_natural_break = (
                seg.speaker != current[-1].speaker
                or (seg.start - current[-1].end) >= min_break_gap_s
            )
            if is_natural_break:
                preferred_cut = len(current) - 1
            current.append(seg)
            continue

        # `seg` does not fit in `current` -- a break is forced.
        cut = len(current) - 1
        if preferred_cut is not None:
            remainder = current[preferred_cut + 1 :]
            if remainder and seg.end - remainder[0].start <= max_window_s:
                cut = preferred_cut

        windows.append(current[: cut + 1])
        current = current[cut + 1 :] + [seg]
        preferred_cut = None

    windows.append(current)
    return windows


def pack_duration_into_windows(
    total_duration_s: float, max_window_s: float = 28.0
) -> list[tuple[float, float]]:
    """Partition [0, total_duration_s) into <=max_window_s windows, no segments needed.

    Used by web/incremental.py (latency Wave 3): during live capture, only a
    PROVISIONAL diarization exists and keeps changing as more audio arrives,
    so windows cannot be packed from diarized segments the way
    _pack_segments_into_windows does for the batch path. This is safe because
    ASR and diarization are already decoupled here — a window's decode
    produces word timestamps that get re-attributed to whatever the FINAL
    diarization turns out to be (_assign_words_to_turns), so the window
    boundary only affects where a decode call is cut, never which speaker a
    word is attributed to.

    Args:
        total_duration_s: Total audio duration to cover, in seconds.
        max_window_s: Maximum span of one decode window, in seconds.

    Returns:
        List of (start, end) tuples covering [0, total_duration_s) with no
        gaps or overlaps, each span <= max_window_s. Empty if
        total_duration_s <= 0.
    """
    if total_duration_s <= 0:
        return []
    windows = []
    start = 0.0
    while start < total_duration_s:
        end = min(start + max_window_s, total_duration_s)
        windows.append((start, end))
        start = end
    return windows


# ── Word <-> segment attribution ─────────────────────────────────────────────
# Adapted from the whisperX-alignment helpers on
# worktree-agent-a9241a6fe03e8de65:src/l3_asr.py (unmerged branch; that
# module's own gate — "measure whole-file ASR before/after" — rejected
# whole-file decoding for a DIFFERENT reason, single-language-locked
# code-switch absorption over the WHOLE clip, not the windowed case this
# module scopes each decode to). The two functions below are copied
# essentially verbatim: they are pure, already unit-tested there, and
# reusable as-is for attributing one window's words to the diarized segments
# it spans.


def _segment_index_for_midpoint(midpoint: float, segments: Sequence[Segment]) -> int:
    """Return the index of the segment whose span contains `midpoint`.

    If no segment's [start, end) span contains it — a diarization gap — return
    the index of the segment with the nearest midpoint instead, so every word
    lands somewhere rather than being silently dropped.

    Args:
        midpoint: A word's temporal midpoint, (word.start + word.end) / 2.
        segments: Diarized segments to choose from. Must be non-empty.

    Returns:
        Index into `segments`.
    """
    for i, seg in enumerate(segments):
        if seg.start <= midpoint < seg.end:
            return i
    return min(
        range(len(segments)),
        key=lambda i: abs(midpoint - (segments[i].start + segments[i].end) / 2),
    )


def _assign_words_to_turns(
    words: Sequence[Word], segments: Sequence[Segment]
) -> list[tuple[str, str, float, float]]:
    """Assign transcribed words to diarized segments and group into raw turns.

    Each word is assigned to the segment whose span contains the word's
    midpoint; a word in a diarization gap goes to the nearest segment by
    midpoint distance (see `_segment_index_for_midpoint`). Consecutive words
    assigned to the same segment are joined into one raw turn — segments
    that receive zero words simply produce no turn.

    Args:
        words: (start, end, text) tuples, absolute clip-time seconds, in
            chronological order.
        segments: Diarized segments from L2, in any order.

    Returns:
        List of (speaker, joined_text, start, end) tuples, one per contiguous
        run of same-segment words, in chronological order (the order `words`
        was given in). `start`/`end` are the first/last assigned word's
        bounds, which may differ slightly from the segment's own nominal
        bounds.

    Raises:
        ValueError: `segments` is empty but `words` is not — a word cannot be
            assigned to speaker segments that don't exist.
    """
    if not words:
        return []
    if not segments:
        raise ValueError("segments must be non-empty when words is non-empty")

    turns: list[tuple[str, str, float, float]] = []
    current_idx: int | None = None
    current_words: list[str] = []
    current_start = current_end = 0.0

    for start, end, text in words:
        idx = _segment_index_for_midpoint((start + end) / 2, segments)
        if idx != current_idx:
            if current_words:
                turns.append(
                    (
                        segments[current_idx].speaker,
                        " ".join(current_words),
                        current_start,
                        current_end,
                    )
                )
            current_idx = idx
            current_words = []
            current_start = start
        current_words.append(text)
        current_end = end

    if current_words:
        turns.append(
            (
                segments[current_idx].speaker,
                " ".join(current_words),
                current_start,
                current_end,
            )
        )
    return turns


# ── Window decode + guards + ladder (word-timestamp-carrying) ───────────────
# Parallels _decode_window / _decode_with_script_guard / _retry_ladder above,
# but requests word_timestamps=True and threads the resulting words through
# every guard re-decode and every ladder step, so attribution never loses
# them. Kept separate from the per-segment functions rather than unified,
# because word_timestamps=True measurably slows every call (~4-5s extra on a
# 10s window — see LEARNINGS.md) and the per-segment path (config i) must
# report its own, unslowed latency for an honest comparison against this
# windowed path (config ii).


def _decode_window_words(
    audio: np.ndarray,
    sr: int,
    start: float,
    end: float,
    repo: str,
    *,
    language: str | None,
    decode_kwargs: dict,
) -> tuple[str, str | None, list[Word]]:
    """Like _decode_window, but with word_timestamps=True.

    Returns:
        (text, detected_language, words). `words` start/end are shifted from
        window-local time (0 = `start`) to absolute clip-time seconds.
    """
    import mlx_whisper

    start_sample = max(0, round(start * sr))
    end_sample = min(len(audio), round(end * sr))
    window = audio[start_sample:end_sample]
    if window.size == 0:
        return "", None, []
    result = mlx_whisper.transcribe(
        window,
        path_or_hf_repo=repo,
        language=language,
        task="transcribe",
        word_timestamps=True,
        **decode_kwargs,
    )
    words: list[Word] = [
        (start + w["start"], start + w["end"], w["word"].strip())
        for whisper_seg in result["segments"]
        for w in whisper_seg["words"]
        if w["word"].strip()
    ]
    return result["text"].strip(), result.get("language"), words


def _decode_window_words_with_guards(
    audio: np.ndarray, sr: int, start: float, end: float, repo: str, decode_kwargs: dict
) -> tuple[str, list[Word]]:
    """Word-timestamp-carrying counterpart of _decode_with_script_guard.

    Applies the same two guards (Arabic-script, then language-allowlist —
    see that function's docstring) to the SAME initial decode; whichever one
    fires re-decodes with language="hi" and its words replace the original
    decode's words.
    """
    text, detected_language, words = _decode_window_words(
        audio, sr, start, end, repo, language=None, decode_kwargs=decode_kwargs
    )
    if config.ASR_SCRIPT_GUARD and _contains_arabic_script(text):
        logger.warning(
            "fast_asr script guard (windowed): Arabic-script decode at "
            "[%.2f, %.2f]s ('%s') — re-decoding with language=hi",
            start,
            end,
            text[:40],
        )
        text, _, words = _decode_window_words(
            audio, sr, start, end, repo, language="hi", decode_kwargs=decode_kwargs
        )
    elif (
        detected_language is not None
        and detected_language not in config.ASR_LANGUAGE_ALLOWLIST
    ):
        logger.warning(
            "fast_asr language guard (windowed): non-allowlist language '%s' "
            "at [%.2f, %.2f]s ('%s') — re-decoding with language=hi",
            detected_language,
            start,
            end,
            text[:40],
        )
        text, _, words = _decode_window_words(
            audio, sr, start, end, repo, language="hi", decode_kwargs=decode_kwargs
        )
    return text, words


# Progress-reporting granularity fix: window-packing (Fix 2) cut on_progress
# from one call per DIARIZED SEGMENT down to one call per WINDOW, so a clip
# short enough to pack into a single window (the common case — most tier-2/3
# consults run under WINDOW_MAX_SPAN_S) reported exactly ONE datum, right at
# the window's completion — indistinguishable in practice from no progress at
# all, since it lands at (or after) the moment the stage itself ends and the
# caller clears the display. When a window's decode degenerates and the retry
# ladder fires, this shared helper lets each ladder step also report partial
# credit, so a retried window does not go dark for its (up to 3) extra decode
# calls either.
def _report_progress(
    on_progress: Callable[[float, float], None] | None,
    done_seconds: float,
    total_seconds: float,
) -> None:
    if on_progress is None:
        return
    try:
        on_progress(done_seconds, total_seconds)
    except Exception:
        logger.debug("on_progress callback failed (ignored)", exc_info=True)


# Fractional credit (of the CURRENT window's own audio span) reported after
# each ladder step resolves or falls through — approximate, not precise (a
# step's wall-clock cost does not scale with the window's audio duration),
# but strictly increasing step-to-step so the displayed percentage never
# jumps backward.
_LADDER_STEP_PROGRESS_FRACTION = {
    "step1_boundary_shift": 0.25,
    "step2_temp_condition": 0.5,
    "step3_large_model": 0.75,
}


def _retry_ladder_windowed(
    audio: np.ndarray,
    sr: int,
    window_start: float,
    window_end: float,
    total_duration: float,
    *,
    initial_text: str,
    initial_words: list[Word],
    on_progress: Callable[[float, float], None] | None = None,
    done_seconds_before_window: float = 0.0,
    total_seconds: float = 0.0,
) -> tuple[str, list[Word], str]:
    """Window-granularity counterpart of _retry_ladder.

    Identical 4-step logic and step names (see _retry_ladder's docstring for
    the rationale of each step); the only difference is every candidate
    carries its word timestamps alongside its text, so attribution can use
    whichever candidate is ultimately chosen.

    Args:
        audio: Full-clip float32 mono array (16 kHz).
        sr: Sample rate (16000).
        window_start, window_end: The decode window that degenerated.
        total_duration: Full clip duration in seconds, for clamping step1's
            boundary shift.
        initial_text, initial_words: The already-degenerate first decode
            (step0), kept as a give-up candidate.
        on_progress, done_seconds_before_window, total_seconds: optional
            per-step progress reporting (see _report_progress and
            _LADDER_STEP_PROGRESS_FRACTION above). `done_seconds_before_window`
            is every earlier window's already-completed audio; the caller
            (fast_transcribe_windowed) still reports this window's full span
            once the ladder returns, regardless of which step resolved it.

    Returns:
        (final_text, final_words, step_name).
    """
    candidates: list[tuple[str, list[Word]]] = [(initial_text, initial_words)]
    window_span = window_end - window_start

    def _report_step(step_name: str) -> None:
        _report_progress(
            on_progress,
            done_seconds_before_window + _LADDER_STEP_PROGRESS_FRACTION[step_name] * window_span,
            total_seconds,
        )

    # Step 1: widen the window boundaries — a measured loop-breaker.
    shift = config.RETRY_BOUNDARY_SHIFT_S
    start1 = max(0.0, window_start - shift)
    end1 = min(total_duration, window_end + shift)
    text1, words1 = _decode_window_words_with_guards(
        audio, sr, start1, end1, config.FAST_ASR_MODEL, config.FAST_ASR_DECODE_KWARGS
    )
    candidates.append((text1, words1))
    _report_step("step1_boundary_shift")
    if not looks_degenerate(text1, end1 - start1):
        return text1, words1, "step1_boundary_shift"

    # Step 2: allow conditioning on previous text, slightly warmer sampling.
    step2_kwargs = {
        "temperature": config.RETRY_TEMP_STEP2,
        "condition_on_previous_text": True,
    }
    text2, words2 = _decode_window_words_with_guards(
        audio, sr, window_start, window_end, config.FAST_ASR_MODEL, step2_kwargs
    )
    candidates.append((text2, words2))
    _report_step("step2_temp_condition")
    if not looks_degenerate(text2, window_end - window_start):
        return text2, words2, "step2_temp_condition"

    # Step 3: escalate to the bigger mlx model, still on the Metal GPU.
    text3, words3 = _decode_window_words_with_guards(
        audio,
        sr,
        window_start,
        window_end,
        config.FAST_ASR_FALLBACK_MODEL,
        config.FAST_ASR_DECODE_KWARGS,
    )
    candidates.append((text3, words3))
    _report_step("step3_large_model")
    if not looks_degenerate(text3, window_end - window_start):
        return text3, words3, "step3_large_model"

    # Step 4: give up gracefully — never block, never revert to a slower decode.
    best_text, best_words = min(candidates, key=lambda c: _degeneracy_score(c[0]))
    marker = (
        "[unclear — VERIFY]" if _is_latin_dominant(best_text) else "[अस्पष्ट — VERIFY]"
    )
    # The marker is not attached to any real word; a zero-width synthetic word
    # at the window's start carries it through _assign_words_to_turns so the
    # VERIFY flag survives into the final transcript instead of being
    # silently dropped (words, not this function's text return, are what
    # actually becomes turn text — see fast_transcribe_windowed).
    marked_words = [(window_start, window_start, marker)] + best_words
    return f"{marker} {best_text}".strip(), marked_words, "step4_giveup"


# ── Public entry points ──────────────────────────────────────────────────────


def fast_transcribe(
    wav_path: str,
    segments: list[Segment],
    on_progress=None,
) -> list[Turn]:
    """Transcribe each diarized segment with mlx-whisper and assign a speaker role.

    Args:
        wav_path: Path to a 16 kHz mono WAV file (output of L1).
        segments: Diarized segments from L2.
        on_progress: Optional callback(done_seconds, total_seconds), invoked
            after each segment (ladder retries included) resolves. Exceptions
            are swallowed — progress reporting must never break transcription.

    Returns:
        List of Turn(speaker_role, text, start, end) in chronological order.
        Role assignment is identical to src.l3_asr.transcribe: the speaker
        with the higher aggregate doctor-heuristic score is "DOCTOR", the
        other "PATIENT"; ties or a single speaker keep "UNKNOWN". Segments
        whose ladder gives up are still included, prefixed with a VERIFY
        marker (see _retry_ladder) rather than dropped.
    """
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"expected 16kHz mono wav from L1 preprocess, got {sr}Hz")
    total_duration = len(audio) / sr

    total_seconds = sum(max(0.0, s.end - s.start) for s in segments)
    done_seconds = 0.0

    raw_turns: list[tuple[str, str, float, float]] = []
    for seg in segments:
        text = _decode_with_script_guard(
            audio,
            sr,
            seg.start,
            seg.end,
            config.FAST_ASR_MODEL,
            config.FAST_ASR_DECODE_KWARGS,
        )
        if looks_degenerate(text, seg.end - seg.start):
            text, step = _retry_ladder(
                audio, sr, seg, total_duration, initial_text=text
            )
            logger.warning(
                "fast_asr ladder fired at [%.2f, %.2f]s — resolved by %s",
                seg.start,
                seg.end,
                step,
            )

        if text:
            raw_turns.append((seg.speaker, text, seg.start, seg.end))

        done_seconds += max(0.0, seg.end - seg.start)
        if on_progress is not None:
            try:
                on_progress(done_seconds, total_seconds)
            except Exception:
                logger.debug("on_progress callback failed (ignored)", exc_info=True)

    return _build_turns_with_roles(raw_turns)


def fast_transcribe_windowed(
    wav_path: str,
    segments: list[Segment],
    on_progress=None,
) -> list[Turn]:
    """Fix 2: decode packed <=28s windows instead of one call per segment.

    Same contract as fast_transcribe (see its docstring), but answers the
    measured per-call fixed cost (config.WINDOW_MAX_SPAN_S's docstring) by
    cutting the number of mlx_whisper.transcribe() calls from one-per-
    diarized-segment to one-per-window: segments are packed into contiguous
    windows (_pack_segments_into_windows), each window is decoded ONCE with
    word timestamps (both guards and the degeneration ladder still apply, at
    window granularity — _decode_window_words_with_guards,
    _retry_ladder_windowed), and the resulting words are re-attributed back
    to the original diarized segments by midpoint overlap
    (_assign_words_to_turns) before the shared role heuristic runs.

    Args:
        wav_path: Path to a 16 kHz mono WAV file (output of L1).
        segments: Diarized segments from L2.
        on_progress: Optional callback(done_seconds, total_seconds), invoked
            after each WINDOW (not each segment) resolves — coarser-grained
            than fast_transcribe's per-segment callback, since a window may
            span several segments. Exceptions are swallowed.

    Returns:
        List of Turn(speaker_role, text, start, end) in chronological order.
        A turn's start/end are its assigned words' bounds, which may differ
        slightly from the original segment's nominal bounds (see
        _assign_words_to_turns). Segments that receive zero words (all their
        audio fell in a window with no aligned speech) produce no turn —
        unlike fast_transcribe, which always emits a turn for every non-empty
        decode.
    """
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"expected 16kHz mono wav from L1 preprocess, got {sr}Hz")
    total_duration = len(audio) / sr

    if not segments:
        return []

    window_groups = _pack_segments_into_windows(
        segments,
        max_window_s=config.WINDOW_MAX_SPAN_S,
        min_break_gap_s=config.WINDOW_MIN_BREAK_GAP_S,
    )
    total_seconds = sum(max(0.0, s.end - s.start) for s in segments)
    done_seconds = 0.0

    all_words: list[Word] = []
    for group in window_groups:
        window_start, window_end = group[0].start, group[-1].end
        words = _decode_one_window_with_ladder(
            audio, sr, window_start, window_end, total_duration, on_progress=on_progress
        )
        all_words.extend(words)

        done_seconds += sum(max(0.0, s.end - s.start) for s in group)
        _report_progress(on_progress, done_seconds, total_seconds)

    raw_turns = _assign_words_to_turns(all_words, segments)
    return _build_turns_with_roles(raw_turns)


def _decode_one_window_with_ladder(
    audio: np.ndarray,
    sr: int,
    window_start: float,
    window_end: float,
    total_duration: float,
    *,
    on_progress: Callable[[float, float], None] | None = None,
    done_seconds_before_window: float = 0.0,
    total_seconds: float = 0.0,
) -> list[Word]:
    """Decode one window (guards + degeneration ladder), return its words.

    Factored out of fast_transcribe_windowed's loop body so
    decode_windows_words (segment-free, latency Wave 3's incremental capture)
    can share the exact same guard/ladder behavior per window without
    depending on diarized segments — this function never touches Segment.
    """
    text, words = _decode_window_words_with_guards(
        audio,
        sr,
        window_start,
        window_end,
        config.FAST_ASR_MODEL,
        config.FAST_ASR_DECODE_KWARGS,
    )
    if looks_degenerate(text, window_end - window_start):
        text, words, step = _retry_ladder_windowed(
            audio,
            sr,
            window_start,
            window_end,
            total_duration,
            initial_text=text,
            initial_words=words,
            on_progress=on_progress,
            done_seconds_before_window=done_seconds_before_window,
            total_seconds=total_seconds,
        )
        logger.warning(
            "fast_asr windowed ladder fired at [%.2f, %.2f]s — resolved by %s",
            window_start,
            window_end,
            step,
        )
    return words


def decode_windows_words(
    audio: np.ndarray, sr: int, windows: list[tuple[float, float]]
) -> list[Word]:
    """Decode a list of (start, end) windows (see pack_duration_into_windows),
    guards + ladder applied per window, and return all words concatenated in
    chronological order.

    Segment-free counterpart to fast_transcribe_windowed's per-window loop —
    used by web/incremental.py, which cannot pack windows from diarized
    segments (only a provisional, still-changing diarization exists during
    live capture). No progress callback: incremental capture reports its own
    progress via partial_turns(), not this per-window mechanism.
    """
    total_duration = len(audio) / sr
    all_words: list[Word] = []
    for window_start, window_end in windows:
        all_words.extend(
            _decode_one_window_with_ladder(audio, sr, window_start, window_end, total_duration)
        )
    return all_words
