"""L3 — Multilingual ASR (Hindi/English/Marathi, code-switching) via faster-whisper."""

import gc
import logging
import re

import torch
from faster_whisper import WhisperModel

from src import config, telemetry
from src.types import Segment, Turn

# Arabic-script Unicode blocks (Arabic, Supplement, Extended-A, Presentation
# Forms A/B). We only support hi/en/mr (Latin or Devanagari); any Arabic-script
# text means Whisper's per-segment auto-detect misclassified Hindi as Urdu.
_ARABIC_SCRIPT_RE = re.compile(
    "[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]"
)


def _contains_arabic_script(text: str) -> bool:
    """True if text contains any Arabic-script character (see module docstring)."""
    return bool(_ARABIC_SCRIPT_RE.search(text))


# Doctor heuristic: bag-of-words score over transcribed text.
# Doctors tend to use question forms (eliciting symptoms) AND clinical terms
# (drug names, diagnoses). The speaker with the higher combined score is DOCTOR.
# This is a coarse approximation; replace with a trained classifier once we
# have enough labelled turns.
_DOCTOR_TOKENS: frozenset[str] = frozenset(
    {
        # Hindi/Urdu question words
        "kya", "kyun", "kab", "kahan", "kaun", "kaise", "kitna", "kitne", "kitni",
        # English question words that doctors use when eliciting history
        "what", "why", "when", "where", "who", "how", "which",
        "do", "does", "did", "is", "are", "was", "were", "have", "has",
        # Clinical / prescription vocabulary
        "prescription", "medicine", "medicines", "tablet", "tablets", "capsule",
        "dosage", "dose", "diagnosis", "symptoms", "symptom",
        "antibiotic", "antibiotics", "mg", "ml",
        "morning", "evening", "night", "days", "weeks",
        "blood", "test", "report", "lab",
        "bp", "pressure", "sugar", "diabetes", "infection",
        "pain", "chest", "breathing", "fever", "cough",
        # Romanised Hindi clinical terms
        "dawai", "dawa", "bukhar", "dard",
    }
)

logger = logging.getLogger(__name__)


def _doctor_score(text: str) -> int:
    """Count how many doctor-indicator tokens appear in text (case-insensitive)."""
    tokens = text.lower().split()
    return sum(1 for t in tokens if t.strip(".,?!।") in _DOCTOR_TOKENS)


def transcribe(
    wav_path: str,
    segments: list[Segment],
    model: WhisperModel | None = None,
    on_progress=None,
) -> list[Turn]:
    """Transcribe each diarized segment and assign a speaker role.

    Args:
        wav_path: Path to a 16 kHz mono WAV file (output of L1).
        segments: Diarized segments from L2.
        model: Optional preloaded WhisperModel. When provided it is used as-is
            and NOT released — for eval harnesses iterating many clips, where
            per-clip model loading dominates runtime. Production passes None:
            load, use, release (the 24 GB memory discipline).
        on_progress: Optional callback(done_seconds, total_seconds) invoked
            after each segment decodes — real transcription progress (decoded
            audio seconds over total segment seconds), used by the review
            frontend for the percent/ETA display. Exceptions are swallowed:
            progress reporting must never break transcription.

    Returns:
        List of Turn(speaker_role, text, start, end) in chronological order.
        Speaker roles are heuristically assigned: the speaker with more
        question-forms / medical terms is labelled "DOCTOR"; the other "PATIENT".
        Ties and single-speaker recordings keep role "UNKNOWN".

    Notes:
        faster-whisper CTranslate2 backend does not support MPS directly;
        device="cpu" uses Apple AMX optimisation on M-series via BLAS.
        language=None enables per-segment auto-detection for Hindi/English/
        Marathi code-switching. task="transcribe" is explicit to prevent
        translation even if Whisper internally detects a non-English segment.
        vad_filter (config.ASR_VAD_FILTER) is passed through but is a measured
        no-op here: faster-whisper ignores vad_filter whenever clip_timestamps
        is set, and this call always sets clip_timestamps for per-segment
        decoding. It provides no silence-hallucination protection — kept
        False (see src.config.ASR_VAD_FILTER for the full explanation).
        When config.ASR_SCRIPT_GUARD is True (default), a segment whose
        decode contains Arabic-script characters is re-decoded once with
        language="hi" forced — we only support hi/en/mr, so Arabic script is
        always a misdetection (see src.config.ASR_SCRIPT_GUARD).
    """
    owns_model = model is None
    if owns_model:
        with telemetry.timer("l3.model_load"):
            model = WhisperModel(config.ASR_MODEL, device="cpu", compute_type="int8")
        logger.info("L3: loaded faster-whisper %s", config.ASR_MODEL)

    def _decode(seg: Segment, language: str | None) -> str:
        gen, _ = model.transcribe(
            wav_path,
            language=language,
            task="transcribe",
            clip_timestamps=f"{seg.start},{seg.end}",
            # Read at call time (not import time) so eval studies can override
            # src.config.ASR_BEAM_SIZE / ASR_VAD_FILTER; pipeline and eval
            # share these values.
            beam_size=config.ASR_BEAM_SIZE,
            vad_filter=config.ASR_VAD_FILTER,
            word_timestamps=False,
        )
        return " ".join(chunk.text.strip() for chunk in gen).strip()

    total_seconds = sum(max(0.0, s.end - s.start) for s in segments)
    done_seconds = 0.0

    raw_turns: list[tuple[str, str, float, float]] = []  # (speaker, text, start, end)
    for seg in segments:
        text = _decode(seg, language=None)

        # Script guard: language=None occasionally misdetects Hindi as Urdu
        # and decodes the segment in Arabic script. We only support hi/en/mr
        # (Latin/Devanagari), so Arabic script is always a misdetection —
        # force a single re-decode with language="hi" (config.ASR_SCRIPT_GUARD).
        if config.ASR_SCRIPT_GUARD and _contains_arabic_script(text):
            logger.warning(
                "L3 script guard: Arabic-script decode at [%.2f, %.2f]s "
                "('%s') — re-decoding with language=hi",
                seg.start,
                seg.end,
                text[:40],
            )
            text = _decode(seg, language="hi")

        if text:
            raw_turns.append((seg.speaker, text, seg.start, seg.end))

        done_seconds += max(0.0, seg.end - seg.start)
        if on_progress is not None:
            try:
                on_progress(done_seconds, total_seconds)
            except Exception:
                logger.debug("on_progress callback failed (ignored)", exc_info=True)

    if owns_model:
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    if not raw_turns:
        return []

    # Aggregate doctor-heuristic scores per speaker label
    speaker_scores: dict[str, int] = {}
    for speaker, text, _, _ in raw_turns:
        speaker_scores[speaker] = speaker_scores.get(speaker, 0) + _doctor_score(text)

    speakers = list(speaker_scores)
    if len(speakers) < 2:
        # Only one speaker detected — can't assign roles confidently
        role_map = {speakers[0]: "UNKNOWN"}
    else:
        top_speaker = max(speakers, key=lambda s: speaker_scores[s])
        runner_up = [s for s in speakers if s != top_speaker]
        # Only assign DOCTOR if top speaker's score is strictly higher
        if speaker_scores[top_speaker] > max(speaker_scores[s] for s in runner_up):
            role_map = {top_speaker: "DOCTOR"}
            for s in runner_up:
                role_map[s] = "PATIENT"
        else:
            role_map = {s: "UNKNOWN" for s in speakers}

    logger.info(
        "L3: role assignment — %s",
        {s: f"{role_map.get(s, 'UNKNOWN')} (score={speaker_scores[s]})" for s in speakers},
    )

    return [
        Turn(
            speaker_role=role_map.get(speaker, "UNKNOWN"),
            text=text,
            start=start,
            end=end,
        )
        for speaker, text, start, end in raw_turns
    ]
