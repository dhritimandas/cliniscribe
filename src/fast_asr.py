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
"""

import logging
import re
import zlib

import numpy as np
import soundfile as sf

from src import config
from src.l3_asr import _contains_arabic_script, _doctor_score
from src.types import Segment, Turn

logger = logging.getLogger(__name__)

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_TOKEN_PUNCT_RE = re.compile(r"[.,?!;:।॥\"'()\[\]{}]")


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
) -> str:
    """Slice [start, end) from `audio` in memory and decode it with mlx-whisper.

    Never passes a path to mlx_whisper.transcribe (see module docstring).
    """
    import mlx_whisper

    start_sample = max(0, round(start * sr))
    end_sample = min(len(audio), round(end * sr))
    window = audio[start_sample:end_sample]
    if window.size == 0:
        return ""
    result = mlx_whisper.transcribe(
        window,
        path_or_hf_repo=repo,
        language=language,
        task="transcribe",
        word_timestamps=False,
        **decode_kwargs,
    )
    return result["text"].strip()


def _decode_with_script_guard(
    audio: np.ndarray, sr: int, start: float, end: float, repo: str, decode_kwargs: dict
) -> str:
    """Decode a window, then re-decode with language="hi" if the guard fires.

    Mirrors src.l3_asr.transcribe's script guard exactly (see
    src.config.ASR_SCRIPT_GUARD docstring): language=None auto-detection
    occasionally misclassifies Hindi as Urdu and writes Arabic script; we
    only support hi/en/mr, so Arabic script is always a misdetection.
    """
    text = _decode_window(
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
        text = _decode_window(
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


# ── Public entry point ───────────────────────────────────────────────────────


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

    if not raw_turns:
        return []

    # Aggregate doctor-heuristic scores per speaker label (identical logic to
    # src.l3_asr.transcribe, duplicated here per that module's "do not touch"
    # scope — only _doctor_score itself is imported, not reimplemented).
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
