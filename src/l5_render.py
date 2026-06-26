"""L5 — Draft prescription PDF rendering via reportlab."""

import logging

from src.types import ClinicalNote

logger = logging.getLogger(__name__)


def render(note: ClinicalNote) -> str:
    """Render a ClinicalNote as a draft prescription PDF.

    Args:
        note: Structured clinical note from L4.

    Returns:
        Path to the generated PDF file in outputs/.

    Notes:
        Output is clearly watermarked as an unverified draft.
        low_confidence_fields and unvalidated medications are visually flagged.
        The physician review-and-edit step is mandatory before any output
        reaches a patient record.
    """
    raise NotImplementedError
