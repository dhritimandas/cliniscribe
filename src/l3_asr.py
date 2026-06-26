"""L3 — Multilingual ASR (Hindi/English/Marathi, code-switching) via faster-whisper."""

import logging

from src.types import Segment, Turn

logger = logging.getLogger(__name__)


def transcribe(wav_path: str, segments: list[Segment]) -> list[Turn]:
    """Transcribe each diarized segment and assign a speaker role.

    Args:
        wav_path: Path to a 16 kHz mono WAV file (output of L1).
        segments: Diarized segments from L2.

    Returns:
        List of Turn(speaker_role, text, start, end) in chronological order.
        Speaker roles are heuristically assigned: the speaker with more clinical
        vocabulary is labelled "DOCTOR"; the other "PATIENT".

    Notes:
        Model: faster-whisper large-v3 (CTranslate2, compute_type="int8").
        Language detection is per-segment (language=None) to handle code-switching.
        Do NOT pre-translate to English before downstream stages.

        Known failure mode: WER spikes at code-switch boundaries (Hindi↔English
        mid-sentence). This is the primary product metric to track.
    """
    raise NotImplementedError
