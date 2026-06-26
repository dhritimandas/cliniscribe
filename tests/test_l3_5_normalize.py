"""Tests for L3.5 normalization: internal helpers and real-model integration."""

import numpy as np
import pytest

from src.l3_5_normalize import (
    COSINE_THRESHOLD,
    HARDNEG_MARGIN,
    _Match,
    _best_non_overlapping,
    _gloss_turn,
    _ngrams,
    _passes_hardneg_gate,
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


# ── Hard-negative gate (fast, no model) ──────────────────────────────────────


def test_hardneg_gate_rejects_when_hardneg_within_margin() -> None:
    # concept_sim=0.70, hardneg_sim=0.68 → margin=0.02 < HARDNEG_MARGIN(0.05) → rejected
    assert not _passes_hardneg_gate(sim=0.70, max_hn_sim=0.68)


def test_hardneg_gate_accepts_when_hardneg_far() -> None:
    # concept_sim=0.85, hardneg_sim=0.60 → margin=0.25 > HARDNEG_MARGIN → accepted
    assert _passes_hardneg_gate(sim=0.85, max_hn_sim=0.60)


def test_hardneg_gate_rejects_at_exact_boundary() -> None:
    # Exactly at margin boundary (equal) is rejected (>=, not >)
    boundary = 0.70 - HARDNEG_MARGIN  # = 0.65
    assert not _passes_hardneg_gate(sim=0.70, max_hn_sim=boundary)


def test_hardneg_gate_accepts_just_past_boundary() -> None:
    # One epsilon below boundary → accepted
    boundary = 0.70 - HARDNEG_MARGIN  # = 0.65
    assert _passes_hardneg_gate(sim=0.70, max_hn_sim=boundary - 0.001)


def test_hardneg_margin_constant_value() -> None:
    # Verify the constant hasn't drifted from the documented value
    assert HARDNEG_MARGIN == 0.05


# ── New concepts exist in the table ──────────────────────────────────────────


def test_new_concepts_present() -> None:
    """All newly added concepts must be present in the CONCEPTS list."""
    from src.concepts import CONCEPTS

    terms = {c.term for c in CONCEPTS}
    expected_new = {
        "Abdominal Pain",
        "Back Pain",
        "Asthma",
        "Nausea",
        "Upper Respiratory Tract Infection",
        "Allergic Rhinitis",
        "Migraine",
        "Anxiety",
        "Loss of Appetite",
        "Fungal Infection",
    }
    assert expected_new <= terms, f"Missing concepts: {expected_new - terms}"


def test_all_concepts_have_snomed_id() -> None:
    """Every concept must have a SNOMED CT identifier (no silent omissions)."""
    from src.concepts import CONCEPTS

    missing = [c.term for c in CONCEPTS if c.snomed_id is None]
    assert not missing, f"Concepts without SNOMED ID: {missing}"


def test_hard_negatives_populated_for_key_concepts() -> None:
    """Concepts with known confusable everyday words must have hard negatives."""
    from src.concepts import CONCEPTS

    must_have_hardnegs = {
        "Common Cold",     # "cold weather"
        "Hypertension",    # "work stress"
        "Acid Reflux",     # "gas cylinder"
        "Type 2 Diabetes Mellitus",  # "sugar in tea"
    }
    concept_map = {c.term: c for c in CONCEPTS}
    for term in must_have_hardnegs:
        assert term in concept_map, f"Concept not found: {term}"
        assert concept_map[term].hard_negatives, f"No hard negatives for: {term}"


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


@pytest.mark.slow
def test_sardi_maps_to_common_cold() -> None:
    """'sardi' (Hindi: cold/flu) must still gloss to Common Cold despite hard negatives."""
    from src.l3_5_normalize import normalize

    turns = [_turn("mujhe sardi ho gayi hai")]
    result = normalize(turns)
    assert "Common Cold" in result[0].text


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Unigram 'cold' matches variant 'cold' at sim=1.0 with hardneg margin=0.37 — "
        "well above HARDNEG_MARGIN=0.05. The model treats 'cold' as intrinsically "
        "clinical; single-word context disambiguation requires sentence-level encoding "
        "(Phase D). Per-concept threshold tuning would address this."
    ),
)
def test_cold_weather_not_glossed_as_common_cold() -> None:
    """'cold' in a temperature/weather context should be blocked by hard negatives."""
    from src.l3_5_normalize import normalize

    turns = [_turn("it is very cold outside today")]
    result = normalize(turns)
    assert "(Common Cold)" not in result[0].text


@pytest.mark.slow
def test_pait_dard_maps_to_abdominal_pain() -> None:
    """New concept: 'pait mein dard' must gloss to Abdominal Pain."""
    from src.l3_5_normalize import normalize

    turns = [_turn("mujhe pait mein dard hai")]
    result = normalize(turns)
    assert "Abdominal Pain" in result[0].text


@pytest.mark.slow
def test_ghabrahat_maps_to_anxiety() -> None:
    """New concept: 'ghabrahat' (nervousness/anxiety) must gloss to Anxiety."""
    from src.l3_5_normalize import normalize

    turns = [_turn("bahut ghabrahat ho rahi hai")]
    result = normalize(turns)
    assert "Anxiety" in result[0].text
