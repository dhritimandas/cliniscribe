"""L3 — Multilingual ASR (Hindi/English/Marathi, code-switching) via faster-whisper."""

import gc
import logging

import torch
from faster_whisper import WhisperModel

from src.types import Segment, Turn

ASR_MODEL = "large-v3"

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


def transcribe(wav_path: str, segments: list[Segment]) -> list[Turn]:
    """Transcribe each diarized segment and assign a speaker role.

    Args:
        wav_path: Path to a 16 kHz mono WAV file (output of L1).
        segments: Diarized segments from L2.

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
    """
    model = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")
    logger.info("L3: loaded faster-whisper %s", ASR_MODEL)

    raw_turns: list[tuple[str, str, float, float]] = []  # (speaker, text, start, end)
    for seg in segments:
        gen, _ = model.transcribe(
            wav_path,
            language=None,
            task="transcribe",
            clip_timestamps=f"{seg.start},{seg.end}",
            beam_size=5,
            word_timestamps=False,
        )
        text = " ".join(chunk.text.strip() for chunk in gen).strip()
        if text:
            raw_turns.append((seg.speaker, text, seg.start, seg.end))

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
