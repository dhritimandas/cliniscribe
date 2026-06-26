"""Tests for L3.5 normalization: internal helpers and real-model integration."""

import numpy as np
import pytest

from src.l3_5_normalize import (
    COSINE_THRESHOLD,
    _Match,
    _best_non_overlapping,
    _gloss_turn,
    _ngrams,
)
from src.types import Turn


def _turn(text: str) -> Turn:
    return Turn(speaker_role="DOCTOR", text=text, start=0.0, end=1.0)


# ── _ngrams ──────────────────────────────────────────────────────────────────


def test_ngrams_unigrams_only() -> None:
    result = _ngrams(["a", "b", "c"], max_n=1)
    assert result == [("a", 0, 1), ("b", 1, 2), ("c", 2, 3)]


def test_ngrams_up_to_trigram() -> None:
    spans = _ngrams(["a", "b", "c"], max_n=3)
    span_texts = [s[0] for s in spans]
    assert "a b c" in span_texts
    assert "a b" in span_texts
    assert "a" in span_texts


def test_ngrams_empty_returns_empty() -> None:
    assert _ngrams([], max_n=3) == []


def test_ngrams_single_word() -> None:
    result = _ngrams(["sugar"], max_n=3)
    assert result == [("sugar", 0, 1)]


# ── _best_non_overlapping ────────────────────────────────────────────────────


def _match(start: int, end: int, sim: float, term: str = "X") -> _Match:
    return _Match(
        span=" ".join(["w"] * (end - start)),
        start_word=start,
        end_word=end,
        concept_term=term,
        snomed_id=None,
        similarity=sim,
    )


def test_non_overlapping_picks_highest_sim() -> None:
    # "bp" (word 0) and "high bp" (words 0–1) overlap; higher sim wins
    m1 = _match(0, 1, sim=0.80)
    m2 = _match(0, 2, sim=0.92)
    selected = _best_non_overlapping([m1, m2])
    assert len(selected) == 1
    assert selected[0].similarity == 0.92


def test_non_overlapping_keeps_disjoint() -> None:
    m1 = _match(0, 1, sim=0.80)  # word 0
    m2 = _match(2, 3, sim=0.75)  # word 2
    selected = _best_non_overlapping([m1, m2])
    assert len(selected) == 2


def test_non_overlapping_empty_input() -> None:
    assert _best_non_overlapping([]) == []


# ── _gloss_turn ──────────────────────────────────────────────────────────────


def test_gloss_single_match() -> None:
    words = ["patient", "has", "sugar"]
    m = _Match(span="sugar", start_word=2, end_word=3,
               concept_term="Type 2 Diabetes Mellitus", snomed_id="44054006",
               similarity=0.85)
    result = _gloss_turn(_turn("patient has sugar"), [m], words)
    assert result.text == "patient has sugar (Type 2 Diabetes Mellitus)"


def test_gloss_two_non_overlapping_matches() -> None:
    words = ["bp", "high", "and", "bukhar"]
    m1 = _Match(span="bp", start_word=0, end_word=1,
                concept_term="Hypertension", snomed_id="38341003", similarity=0.90)
    m2 = _Match(span="bukhar", start_word=3, end_word=4,
                concept_term="Fever", snomed_id="386661006", similarity=0.88)
    result = _gloss_turn(_turn("bp high and bukhar"), [m1, m2], words)
    assert "Hypertension" in result.text
    assert "Fever" in result.text


def test_gloss_bigram_match() -> None:
    words = ["chest", "pain"]
    m = _Match(span="chest pain", start_word=0, end_word=2,
               concept_term="Chest Pain", snomed_id="29857009", similarity=0.88)
    result = _gloss_turn(_turn("chest pain"), [m], words)
    assert result.text == "chest pain (Chest Pain)"


def test_gloss_no_matches_returns_original() -> None:
    t = _turn("hello there")
    result = _gloss_turn(t, [], ["hello", "there"])
    assert result.text == "hello there"
    assert result.speaker_role == t.speaker_role


def test_gloss_preserves_metadata() -> None:
    t = Turn(speaker_role="PATIENT", text="dard hai", start=1.5, end=3.0)
    words = ["dard", "hai"]
    m = _Match(span="dard", start_word=0, end_word=1,
               concept_term="Pain", snomed_id="22253000", similarity=0.80)
    result = _gloss_turn(t, [m], words)
    assert result.speaker_role == "PATIENT"
    assert result.start == 1.5
    assert result.end == 3.0


# ── Real model (slow) ─────────────────────────────────────────────────────────


@pytest.mark.slow
def test_sugar_maps_to_t2dm() -> None:
    """'sugar' in a turn must be glossed as Type 2 Diabetes Mellitus."""
    from src.l3_5_normalize import normalize

    turns = [_turn("patient has sugar problem")]
    result = normalize(turns)
    assert "Type 2 Diabetes Mellitus" in result[0].text


@pytest.mark.slow
def test_bp_maps_to_hypertension() -> None:
    from src.l3_5_normalize import normalize

    turns = [_turn("bp is very high")]
    result = normalize(turns)
    assert "Hypertension" in result[0].text


@pytest.mark.slow
def test_bukhar_maps_to_fever() -> None:
    from src.l3_5_normalize import normalize

    turns = [_turn("mujhe bukhar hai")]
    result = normalize(turns)
    assert "Fever" in result[0].text


@pytest.mark.slow
def test_below_threshold_leaves_text_unchanged() -> None:
    """A nonsense medical term should not match anything above threshold."""
    from src.l3_5_normalize import normalize

    turns = [_turn("xyzqrst blorp")]
    result = normalize(turns)
    # original text preserved (no gloss injected)
    assert result[0].text == "xyzqrst blorp"


@pytest.mark.slow
def test_empty_turns_returns_empty() -> None:
    from src.l3_5_normalize import normalize

    assert normalize([]) == []
