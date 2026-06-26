"""L3.5 — Lay-term to clinical-concept normalization via parrotlet-e embeddings."""

import logging

from src.types import Turn

logger = logging.getLogger(__name__)


def normalize(turns: list[Turn]) -> list[Turn]:
    """Map lay medical terms in transcript text to canonical clinical concepts.

    Args:
        turns: Speaker-attributed transcript from L3.

    Returns:
        Turns with lay terms replaced by canonical clinical terms
        (e.g. "sugar" → "Type 2 Diabetes Mellitus", "bukhar" → "Fever").

    Notes:
        Model: ekacare/parrotlet-e (fine-tuned bge-m3 on multilingual medical
        term pairs aligned to SNOMED CT / UMLS). Embed candidate spans, resolve
        to nearest clinical concept by cosine similarity.

        This covers native and romanized scripts across Indic languages —
        the India-specific differentiator vs. US incumbents.
    """
    raise NotImplementedError
