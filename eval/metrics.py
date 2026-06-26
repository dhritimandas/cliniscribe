"""KARMA evaluation metrics: WER, Keyword WER, DER.

Phase A baseline ships WER and Keyword WER (the ASR patient-safety metric).
Semantic WER, DER, and concept-match land alongside the stages they score.
"""

import logging
import re

from jiwer import wer as _jiwer_wer

logger = logging.getLogger(__name__)

# Punctuation to strip before scoring: Latin sentence marks plus the Devanagari
# danda (।) and double danda (॥). Keeps tokens comparable across scripts.
_PUNCT_RE = re.compile(r"[.,?!;:।॥\"'()\[\]{}]")
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for fair scoring.

    Lowercasing is a no-op on Devanagari but matters for romanized/English
    tokens. The Devanagari danda is treated like a full stop.

    Args:
        text: Raw reference or hypothesis string.

    Returns:
        Normalized string with single spaces and no edge whitespace.
    """
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute standard WER between reference and hypothesis strings.

    Both strings are normalized first (case, punctuation, whitespace).

    Args:
        reference: Ground-truth transcript.
        hypothesis: ASR output transcript.

    Returns:
        WER as a float in [0, inf). 0.0 means identical after normalization.
        An empty reference returns 0.0 if the hypothesis is also empty,
        else 1.0 (every hypothesis word is an insertion error).
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _jiwer_wer(ref, hyp)


def keyword_hits(reference: str, hypothesis: str, keywords: list[str]) -> tuple[int, int]:
    """Count reference keywords and how many are missing from the hypothesis.

    A keyword "counts" only if it appears in the reference (gold spans should
    guarantee this, but we filter defensively). It is "missed" if its
    normalized form is not a substring of the normalized hypothesis.

    Args:
        reference: Ground-truth transcript.
        hypothesis: ASR output transcript.
        keywords: Surface forms of drugs, dosages, vitals (gold spans).

    Returns:
        (present, missed): number of keywords found in the reference, and how
        many of those are absent from the hypothesis. Enables micro-averaging
        across a corpus (sum missed / sum present).
    """
    norm_ref = normalize_text(reference)
    norm_hyp = normalize_text(hypothesis)
    present = [k for k in keywords if normalize_text(k) and normalize_text(k) in norm_ref]
    missed = sum(1 for k in present if normalize_text(k) not in norm_hyp)
    return len(present), missed


def keyword_wer(reference: str, hypothesis: str, keywords: list[str]) -> float:
    """Compute keyword error rate over clinically critical terms.

    Defined as 1 - recall: of the keywords that genuinely appear in the
    reference, the fraction NOT reproduced (as a normalized substring) in the
    hypothesis. This is the patient-safety view — a missed drug name or dosage
    is the error that matters, regardless of surrounding-word accuracy.

    Args:
        reference: Ground-truth transcript.
        hypothesis: ASR output transcript.
        keywords: Surface forms of drugs, dosages, vitals (gold spans).

    Returns:
        Keyword error rate in [0, 1]. 0.0 means every reference keyword was
        reproduced. Returns 0.0 when no keyword appears in the reference
        (nothing safety-critical to get wrong).
    """
    present, missed = keyword_hits(reference, hypothesis, keywords)
    if present == 0:
        return 0.0
    return missed / present


def corpus_word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    """Compute micro-averaged WER over a corpus (total edits / total words).

    This is the standard headline ASR number — more meaningful than averaging
    per-sample WER, which over-weights short utterances.

    Args:
        references: Ground-truth transcripts.
        hypotheses: ASR output transcripts, aligned to references by index.

    Returns:
        Corpus WER as a float in [0, inf). Pairs with an empty reference are
        dropped to avoid division by zero.
    """
    refs = [normalize_text(r) for r in references]
    hyps = [normalize_text(h) for h in hypotheses]
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    if not pairs:
        return 0.0
    kept_refs, kept_hyps = zip(*pairs)
    return _jiwer_wer(list(kept_refs), list(kept_hyps))


def diarization_error_rate(reference_segments, hypothesis_segments) -> float:
    """Compute DER between reference and hypothesis diarization.

    Args:
        reference_segments: Ground-truth list of Segment.
        hypothesis_segments: Predicted list of Segment.

    Returns:
        DER as a float in [0, inf).
    """
    raise NotImplementedError
