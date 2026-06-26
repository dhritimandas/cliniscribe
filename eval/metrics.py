"""KARMA evaluation metrics: WER, Semantic WER, Keyword WER, DER."""

import logging

logger = logging.getLogger(__name__)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute standard WER between reference and hypothesis strings.

    Args:
        reference: Ground-truth transcript.
        hypothesis: ASR output transcript.

    Returns:
        WER as a float in [0, inf).
    """
    raise NotImplementedError


def keyword_wer(reference: str, hypothesis: str, keywords: list[str]) -> float:
    """Compute WER restricted to a set of clinically critical keywords.

    Args:
        reference: Ground-truth transcript.
        hypothesis: ASR output transcript.
        keywords: Drug names, dosages, vital signs to evaluate accuracy on.

    Returns:
        Keyword WER as a float in [0, inf).
    """
    raise NotImplementedError


def diarization_error_rate(reference_segments, hypothesis_segments) -> float:
    """Compute DER between reference and hypothesis diarization.

    Args:
        reference_segments: Ground-truth list of Segment.
        hypothesis_segments: Predicted list of Segment.

    Returns:
        DER as a float in [0, inf).
    """
    raise NotImplementedError
