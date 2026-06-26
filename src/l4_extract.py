"""L4 — Clinical entity extraction via Qwen2.5-3B-Instruct (Ollama)."""

import logging

from src.types import ClinicalNote, Turn

logger = logging.getLogger(__name__)


def extract(turns: list[Turn]) -> ClinicalNote:
    """Extract structured clinical entities from the normalized transcript.

    Args:
        turns: Normalized, speaker-attributed turns from L3.5.

    Returns:
        ClinicalNote with fields: chief_complaint, history, diagnosis,
        medications, investigations, advice, follow_up, low_confidence_fields.

    Notes:
        Model: qwen2.5:3b-instruct quantized to 4-bit via Ollama.
        Drug names and doses MUST be validated against the CDSCO drug list.
        Unvalidated entries get validated=False and appear in low_confidence_fields.
        The model must never invent a dosage — flag unknown doses, do not fabricate.

        Memory: load Ollama client, run inference, release before any other
        large model loads. Never hold ASR and LLM resident simultaneously.
    """
    raise NotImplementedError
