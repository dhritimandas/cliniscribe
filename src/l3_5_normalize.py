"""L3.5 — Lay-term to clinical-concept normalization via parrotlet-e embeddings."""

import gc
import logging
import os
from dataclasses import dataclass

import numpy as np

from src.concepts import CONCEPTS
from src.types import Turn

logger = logging.getLogger(__name__)

MODEL_ID = "ekacare/parrotlet-e"
COSINE_THRESHOLD = 0.65
MAX_NGRAM = 3  # unigrams through trigrams


@dataclass
class _Match:
    span: str
    start_word: int   # inclusive
    end_word: int     # exclusive
    concept_term: str
    snomed_id: str | None
    similarity: float


class _EmbeddingBackend:
    """Load parrotlet-e once, encode texts → unit-norm vectors, then release.

    Uses MPS if available, falls back to CPU. Mean pooling + L2 norm per the
    parrotlet-e model card (not CLS token).
    """

    def __init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        device_str = (
            "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        self._device = torch.device(device_str)
        hf_token = os.environ.get("HF_TOKEN")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
            self._model = AutoModel.from_pretrained(MODEL_ID, token=hf_token).to(
                self._device
            )
        except Exception as exc:
            exc_str = str(exc).lower()
            if "401" in exc_str or "gated" in exc_str or "unauthorized" in exc_str or "403" in exc_str:
                raise PermissionError(
                    "parrotlet-e is gated. Accept the terms at "
                    "https://huggingface.co/ekacare/parrotlet-e with your HF "
                    "account, then ensure HF_TOKEN is set in .env."
                ) from exc
            raise

        self._model.eval()
        logger.info(
            "L3.5 backend: %s on %s, hidden_size=%d",
            MODEL_ID,
            device_str,
            self._model.config.hidden_size,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts → L2-normalised embeddings, shape (N, hidden_size).

        Args:
            texts: List of text strings to encode.

        Returns:
            Float32 numpy array of shape (N, hidden_size).
        """
        import torch

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            output = self._model(**encoded)

        # Mean pooling — mask out padding tokens before averaging
        mask = encoded["attention_mask"].unsqueeze(-1).float()  # (N, seq, 1)
        embeddings = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-8)
        # L2 normalise → cosine similarity = dot product
        norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
        embeddings = (embeddings / norms).cpu().numpy().astype(np.float32)
        return embeddings

    def release(self) -> None:
        """Delete model weights and free device memory."""
        import torch

        del self._model, self._tokenizer
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def _ngrams(words: list[str], max_n: int) -> list[tuple[str, int, int]]:
    """Return (span_text, start_idx, end_idx_exclusive) for n=1..max_n."""
    spans = []
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            spans.append((" ".join(words[i : i + n]), i, i + n))
    return spans


def _best_non_overlapping(matches: list[_Match]) -> list[_Match]:
    """Greedy selection: highest-similarity first, drop overlapping matches."""
    selected: list[_Match] = []
    covered: set[int] = set()
    for m in sorted(matches, key=lambda m: m.similarity, reverse=True):
        word_indices = set(range(m.start_word, m.end_word))
        if word_indices & covered:
            continue
        selected.append(m)
        covered |= word_indices
    return sorted(selected, key=lambda m: m.start_word)


def _gloss_turn(turn: Turn, matches: list[_Match], words: list[str]) -> Turn:
    """Rebuild turn text with clinical term glossed in parentheses.

    e.g. 'sugar hai' → 'sugar (Type 2 Diabetes Mellitus) hai'
    """
    if not matches:
        return turn

    match_at: dict[int, _Match] = {m.start_word: m for m in matches}
    result: list[str] = []
    skip_to = 0

    for i, word in enumerate(words):
        if i < skip_to:
            continue
        if i in match_at:
            m = match_at[i]
            span_text = " ".join(words[m.start_word : m.end_word])
            result.append(f"{span_text} ({m.concept_term})")
            logger.info(
                "L3.5 gloss [%.2fs]: '%s' → '%s' (sim=%.3f)",
                turn.start,
                span_text,
                m.concept_term,
                m.similarity,
            )
            skip_to = m.end_word
        else:
            result.append(word)

    return Turn(
        speaker_role=turn.speaker_role,
        text=" ".join(result),
        start=turn.start,
        end=turn.end,
    )


def normalize(turns: list[Turn]) -> list[Turn]:
    """Map lay medical terms in transcript turns to canonical clinical concepts.

    Uses parrotlet-e (fine-tuned bge-m3) embeddings. Candidate spans (1–3
    words) from each turn are compared against canonical concept embeddings;
    matches above COSINE_THRESHOLD are glossed non-destructively in
    parentheses, e.g. ``sugar (Type 2 Diabetes Mellitus)``.

    Model is loaded, used, and released in one call — memory discipline.

    Args:
        turns: Speaker-attributed transcript from L3 (or earlier).

    Returns:
        Same-length list of Turns with lay terms glossed where matched.
    """
    if not turns:
        return turns

    backend = _EmbeddingBackend()

    # Reference: canonical concept terms are the embedding targets.
    # parrotlet-e is trained such that lay synonyms map close to canonical terms.
    concept_texts = [c.term for c in CONCEPTS]
    concept_matrix = backend.encode(concept_texts)  # (C, D)

    # Collect all candidate spans across all turns in one batch for efficiency
    all_spans: list[tuple[str, int, int]] = []   # (span, start_w, end_w)
    turn_span_offsets: list[tuple[int, int]] = []  # (global_start, global_end) per turn
    turn_words: list[list[str]] = []

    for turn in turns:
        words = turn.text.split()
        turn_words.append(words)
        if not words:
            turn_span_offsets.append((len(all_spans), len(all_spans)))
            continue
        cands = _ngrams(words, MAX_NGRAM)
        start_off = len(all_spans)
        all_spans.extend(cands)
        turn_span_offsets.append((start_off, len(all_spans)))

    normalized: list[Turn] = []

    if all_spans:
        span_texts = [s[0] for s in all_spans]
        span_matrix = backend.encode(span_texts)  # (S, D)
        sims = span_matrix @ concept_matrix.T      # (S, C)
        max_sims = sims.max(axis=1)                # (S,)
        best_concepts = sims.argmax(axis=1)        # (S,)

        for turn, words, (off_start, off_end) in zip(turns, turn_words, turn_span_offsets):
            if off_start == off_end:
                normalized.append(turn)
                continue

            matches: list[_Match] = []
            for j in range(off_start, off_end):
                sim = float(max_sims[j])
                if sim >= COSINE_THRESHOLD:
                    span, start_w, end_w = all_spans[j]
                    concept = CONCEPTS[best_concepts[j]]
                    matches.append(
                        _Match(
                            span=span,
                            start_word=start_w,
                            end_word=end_w,
                            concept_term=concept.term,
                            snomed_id=concept.snomed_id,
                            similarity=sim,
                        )
                    )

            glossed = _gloss_turn(turn, _best_non_overlapping(matches), words)
            normalized.append(glossed)
    else:
        normalized = list(turns)

    backend.release()
    return normalized
