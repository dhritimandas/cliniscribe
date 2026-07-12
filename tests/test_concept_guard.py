"""Adversarial suite for the concept matcher's near-collision hardening.

Mirrors src/drug_lexicon.py's testing philosophy: pin the SAFETY PROPERTIES
(never gloss an everyday word, never gloss an ambiguous span) with fast,
model-free tests wherever the property can be checked without embeddings,
and reserve @pytest.mark.slow (real parrotlet-e) for the handful of cases
that need actual model behavior.

Four disciplines under test here (see src/l3_5_normalize.py, src/config.py):
  1. Everyday-word guard (EVERYDAY_WORDS) — categorical, never-gloss list.
  2. Unigram spans require COSINE_THRESHOLD_UNIGRAM (higher than bigram+).
  3. Between-concept ambiguity margin (CONCEPT_AMBIGUITY_MARGIN).
  4. Programmatic one-edit-neighbor collision scan (generalizes the ten
     hard negatives added by commit 26c42ea's manual audit).
"""

import numpy as np
import pytest

from src import config
from src.concepts import CONCEPTS, EVERYDAY_WORDS
from src.l3_5_normalize import (
    MatchConfig,
    _is_devanagari,
    _passes_ambiguity_gate,
    _score_spans,
)

# ── Everyday-word guard ──────────────────────────────────────────────────────


def test_everyday_words_size_in_curated_range() -> None:
    """Curated set, not a downloaded stopword list — sanity-bound the size."""
    assert 60 <= len(EVERYDAY_WORDS) <= 150


@pytest.mark.parametrize(
    "word",
    [
        # Time words
        "हफ्ता", "हफ्ते", "दिन", "सुबह", "शाम", "आज", "week", "tomorrow",
        # Numbers/quantifiers
        "एक", "दो", "बार", "थोड़ा", "one", "two", "more",
        # Function/filler words
        "है", "नहीं", "और", "okay", "please",
        # Common verbs
        "खाना", "लेना", "करना", "take", "eat",
        # Kinship/person words
        "डॉक्टर", "भाई", "बेटा", "doctor", "patient",
    ],
)
def test_known_everyday_words_present(word: str) -> None:
    assert word in EVERYDAY_WORDS


def test_haphte_family_covered_by_everyday_word_guard_too() -> None:
    """Belt-and-braces: हफते (outputs/<session>, 2026-07-12 incident) is now
    blocked by BOTH the hard-negative margin (concepts.py) and the
    categorical everyday-word guard — two independent mechanisms."""
    for week_word in ("हफ्ता", "हफ्ते", "हफते", "हफ़्ते"):
        assert week_word in EVERYDAY_WORDS


def test_everyday_words_never_shadow_a_listed_concept_variant() -> None:
    """If a concept legitimately lists a common word as a variant, the
    variant listing wins (module docstring in src/concepts.py). Verified
    empirically: zero of ~110 single-word CONCEPTS variants collide with
    the curated everyday-word categories (time/number/filler/verb/kinship).
    EVERYDAY_WORDS itself is already variant-overlap-filtered at import
    time; this test guards against a future regression of that filter."""
    all_variants_lower = {v.lower() for c in CONCEPTS for v in c.variants}
    overlap = EVERYDAY_WORDS & all_variants_lower
    assert not overlap, f"Everyday-word guard shadows real concept variants: {overlap}"


# ── Ambiguity margin gate (unit tests, mirrors _passes_hardneg_gate's) ──────


def test_ambiguity_gate_rejects_when_within_margin() -> None:
    # best=0.70, second=0.68 -> margin=0.02 < CONCEPT_AMBIGUITY_MARGIN(0.03)
    assert not _passes_ambiguity_gate(best_sim=0.70, second_sim=0.68, margin=0.03)


def test_ambiguity_gate_accepts_when_far_apart() -> None:
    assert _passes_ambiguity_gate(best_sim=0.90, second_sim=0.40, margin=0.03)


def test_ambiguity_gate_rejects_at_exact_boundary() -> None:
    # margin computed from the SAME subtraction the gate performs internally,
    # so this is an exact tie regardless of floating-point representation.
    best_sim, second_sim = 0.70, 0.67
    margin = best_sim - second_sim
    assert not _passes_ambiguity_gate(best_sim, second_sim, margin)


def test_ambiguity_gate_accepts_just_past_boundary() -> None:
    best_sim, second_sim = 0.70, 0.67
    margin = (best_sim - second_sim) - 0.001
    assert _passes_ambiguity_gate(best_sim, second_sim, margin)


def test_ambiguity_gate_rejects_exact_tie() -> None:
    assert not _passes_ambiguity_gate(best_sim=0.85, second_sim=0.85, margin=0.0)


def test_concept_ambiguity_margin_constant_configured() -> None:
    assert config.CONCEPT_AMBIGUITY_MARGIN > 0.0
    assert config.COSINE_THRESHOLD_UNIGRAM > config.COSINE_THRESHOLD


# ── _score_spans: pure-numpy integration of all four gates ─────────────────
# No model calls — synthetic similarity matrices, real CONCEPTS indices.


def _cfg(**overrides) -> MatchConfig:
    base = dict(
        cosine_threshold=0.65,
        cosine_threshold_unigram=0.65,
        ambiguity_margin=0.0,
        everyday_words=frozenset(),
    )
    base.update(overrides)
    return MatchConfig(**base)


def test_score_spans_unigram_threshold_stricter_than_bigram() -> None:
    """Same cosine (0.70): a unigram span is rejected under a 0.75 unigram
    bar while a bigram span at the same score is accepted (0.65 bigram bar)."""
    n_concepts = len(CONCEPTS)
    all_spans = [("uni", 0, 1), ("bi gram", 1, 3)]
    turn_span_offsets = [(0, 2)]
    ref_ci_arr = np.array([0])
    ref_sims = np.array([[0.70], [0.70]])
    cfg = _cfg(cosine_threshold_unigram=0.75)

    matches = _score_spans(
        all_spans, turn_span_offsets, ref_sims, ref_ci_arr, None, None, n_concepts, cfg
    )[0]
    spans = {m.span for m in matches}
    assert "uni" not in spans
    assert "bi gram" in spans


def test_score_spans_ambiguity_margin_rejects_near_tie_between_concepts() -> None:
    """Span scores 0.80 vs concept 0 and 0.78 vs concept 1 — a genuine
    near-tie must be rejected outright, not resolved by argmax."""
    n_concepts = len(CONCEPTS)
    all_spans = [("span", 0, 1)]
    turn_span_offsets = [(0, 1)]
    ref_ci_arr = np.array([0, 1])
    ref_sims = np.array([[0.80, 0.78]])
    cfg = _cfg(ambiguity_margin=0.05)

    matches = _score_spans(
        all_spans, turn_span_offsets, ref_sims, ref_ci_arr, None, None, n_concepts, cfg
    )[0]
    assert matches == []


def test_score_spans_accepts_when_concepts_are_not_close() -> None:
    n_concepts = len(CONCEPTS)
    all_spans = [("span", 0, 1)]
    turn_span_offsets = [(0, 1)]
    ref_ci_arr = np.array([0, 1])
    ref_sims = np.array([[0.90, 0.30]])
    cfg = _cfg(ambiguity_margin=0.05)

    matches = _score_spans(
        all_spans, turn_span_offsets, ref_sims, ref_ci_arr, None, None, n_concepts, cfg
    )[0]
    assert len(matches) == 1
    assert matches[0].concept_term == CONCEPTS[0].term
    assert matches[0].runner_up_term == CONCEPTS[1].term
    assert matches[0].runner_up_similarity == pytest.approx(0.30)


def test_score_spans_everyday_word_guard_blocks_regardless_of_similarity() -> None:
    """A perfect (1.0) similarity match is still rejected if the span text
    is a curated everyday word — the categorical guard is unconditional."""
    n_concepts = len(CONCEPTS)
    all_spans = [("हफ्ते", 0, 1)]
    turn_span_offsets = [(0, 1)]
    ref_ci_arr = np.array([0])
    ref_sims = np.array([[1.0]])
    cfg = _cfg(everyday_words=frozenset({"हफ्ते"}))

    matches = _score_spans(
        all_spans, turn_span_offsets, ref_sims, ref_ci_arr, None, None, n_concepts, cfg
    )[0]
    assert matches == []


def test_score_spans_everyday_word_guard_is_case_insensitive_for_latin() -> None:
    n_concepts = len(CONCEPTS)
    all_spans = [("Take", 0, 1)]
    turn_span_offsets = [(0, 1)]
    ref_ci_arr = np.array([0])
    ref_sims = np.array([[1.0]])
    cfg = _cfg(everyday_words=frozenset({"take"}))

    matches = _score_spans(
        all_spans, turn_span_offsets, ref_sims, ref_ci_arr, None, None, n_concepts, cfg
    )[0]
    assert matches == []


def test_score_spans_hard_negative_gate_still_applies_unchanged() -> None:
    """The pre-existing hard-negative gate (commit 26c42ea) must still fire
    inside the refactored _score_spans, unchanged."""
    n_concepts = len(CONCEPTS)
    all_spans = [("span", 0, 1)]
    turn_span_offsets = [(0, 1)]
    ref_ci_arr = np.array([0])
    ref_sims = np.array([[0.70]])
    hardneg_idx_arr = np.array([0])
    # hard negative scores 0.68 -> margin 0.02 < HARDNEG_MARGIN(0.05) default
    hardneg_sims = np.array([[0.68]])
    cfg = _cfg()

    # concept 0 must actually have hard_negatives for the gate to engage.
    assert CONCEPTS[0].hard_negatives, "test assumes CONCEPTS[0] has hard negatives"
    matches = _score_spans(
        all_spans, turn_span_offsets, ref_sims, ref_ci_arr,
        hardneg_sims, hardneg_idx_arr, n_concepts, cfg,
    )[0]
    assert matches == []


# ── हांफ/हफ्ते real regression (found by this suite's own slow tests) ───────
# Root cause: हफ्ते was hard-negatived onto Shortness of Breath to fix the
# हफते incident (a UNIGRAM collision) — but हफ्ते's embedding is ALSO close
# to "हांफ रहे"/"हांफ रहे हैं" (a genuine, longer, more specific mention of
# the same concept), and that unigram hard negative was outscoring the
# concept match itself for the longer span (max_hn_sim > sim, not just
# within margin) — no hard-negative MARGIN value can fix an inverted
# relationship like that. A unigram hard negative has no business vetoing a
# longer, more specific span: it was curated to guard a UNIGRAM collision,
# not a phrase. Fixed by excluding hard negatives SHORTER than the query
# span from the max() (hardneg_word_counts) — the original unigram-vs-
# unigram collision (हफते alone) is untouched since word counts are equal.


def test_hard_negative_shorter_than_span_does_not_veto_a_longer_match() -> None:
    n_concepts = len(CONCEPTS)
    all_spans = [("span1 span2", 0, 2)]  # a 2-word span
    turn_span_offsets = [(0, 1)]
    ref_ci_arr = np.array([0])
    ref_sims = np.array([[0.82]])
    hardneg_idx_arr = np.array([0])
    hardneg_sims = np.array([[0.90]])  # would veto without the length exclusion
    hardneg_word_counts = np.array([1])  # the hard negative is only 1 word
    cfg = _cfg()

    assert CONCEPTS[0].hard_negatives, "test assumes CONCEPTS[0] has hard negatives"
    matches = _score_spans(
        all_spans, turn_span_offsets, ref_sims, ref_ci_arr,
        hardneg_sims, hardneg_idx_arr, n_concepts, cfg,
        hardneg_word_counts=hardneg_word_counts,
    )[0]
    assert len(matches) == 1
    assert matches[0].concept_term == CONCEPTS[0].term


def test_hard_negative_same_length_as_span_still_vetoes() -> None:
    """The original unigram-vs-unigram collision (e.g. हफते alone) must
    still be rejected — only STRICTLY SHORTER hard negatives are excluded."""
    n_concepts = len(CONCEPTS)
    all_spans = [("span1", 0, 1)]  # a 1-word span
    turn_span_offsets = [(0, 1)]
    ref_ci_arr = np.array([0])
    ref_sims = np.array([[0.82]])
    hardneg_idx_arr = np.array([0])
    hardneg_sims = np.array([[0.90]])
    hardneg_word_counts = np.array([1])  # same length as the query span
    cfg = _cfg()

    assert CONCEPTS[0].hard_negatives, "test assumes CONCEPTS[0] has hard negatives"
    matches = _score_spans(
        all_spans, turn_span_offsets, ref_sims, ref_ci_arr,
        hardneg_sims, hardneg_idx_arr, n_concepts, cfg,
        hardneg_word_counts=hardneg_word_counts,
    )[0]
    assert matches == []


def test_hardneg_word_counts_defaults_to_no_length_exclusion() -> None:
    """Omitting hardneg_word_counts (production default via
    _encode_reference_matrices) preserves the ORIGINAL unfiltered gate —
    every hard negative competes regardless of length."""
    n_concepts = len(CONCEPTS)
    all_spans = [("span1 span2", 0, 2)]
    turn_span_offsets = [(0, 1)]
    ref_ci_arr = np.array([0])
    ref_sims = np.array([[0.82]])
    hardneg_idx_arr = np.array([0])
    hardneg_sims = np.array([[0.90]])
    cfg = _cfg()

    assert CONCEPTS[0].hard_negatives, "test assumes CONCEPTS[0] has hard negatives"
    matches = _score_spans(
        all_spans, turn_span_offsets, ref_sims, ref_ci_arr,
        hardneg_sims, hardneg_idx_arr, n_concepts, cfg,
    )[0]
    assert matches == []


# ── Programmatic one-edit-neighbor collision scan ───────────────────────────
# Generalizes commit 26c42ea's manual audit: generate every one-edit
# Devanagari neighbor of every concept's single-word variants, and check
# that any neighbor which is a KNOWN real word (the everyday-word guard's
# curated list, union the ten-plus collision words that same manual audit
# already found and fixed) is present in that concept's hard_negatives.
# Catches a FUTURE variant addition that creates a new collision without a
# matching fix — this is exactly how this suite found and fixed an eleventh
# collision (Fungal Infection's दाद vs दान, see src/concepts.py).

_DEVA_ALPHABET = (
    "कखगघचछजझटठडढणतथदधनपफबभमयरलवशषसहािीुूेैोौंँःअआइईउऊएऐओऔ़्"
)

# The ten-plus common words the manual audit (commit 26c42ea) found and
# fixed as hard negatives, used here as "known real words" ground truth —
# EVERYDAY_WORDS deliberately excludes concept-adjacent NOUNS like these
# (a different collision class handled by per-concept hard negatives, not
# the categorical guard — see src/concepts.py's module docstring).
_KNOWN_COLLISION_WORDS: frozenset[str] = frozenset({
    "दवा", "दाल", "पेड़", "हांफते", "पेशा", "कब्र", "नाच", "कमरा", "दान",
    "भूल", "सांप",
})
_REAL_WORD_CORPUS: frozenset[str] = EVERYDAY_WORDS | _KNOWN_COLLISION_WORDS


def _one_edit_neighbors(word: str) -> set[str]:
    """Every Devanagari string one deletion/substitution/insertion from word."""
    neighbors: set[str] = set()
    for i in range(len(word)):
        neighbors.add(word[:i] + word[i + 1 :])
    for i in range(len(word)):
        for ch in _DEVA_ALPHABET:
            if ch != word[i]:
                neighbors.add(word[:i] + ch + word[i + 1 :])
    for i in range(len(word) + 1):
        for ch in _DEVA_ALPHABET:
            neighbors.add(word[:i] + ch + word[i:])
    neighbors.discard(word)
    return neighbors


def test_one_edit_neighbor_generator_correctness() -> None:
    """Sanity-check the generator itself against a known toy example."""
    neighbors = _one_edit_neighbors("का")
    assert "क" in neighbors    # deletion of ा
    assert "की" in neighbors  # substitution ा -> ी
    assert "काल" in neighbors  # insertion of ल
    assert "का" not in neighbors  # the word itself is never its own neighbor


def test_one_edit_neighbors_of_variants_are_hard_negatived() -> None:
    """For every concept's single-word Devanagari variant, every one-edit
    neighbor that is a real known word must be a hard negative on that
    SAME concept — a regression here means a new variant was added without
    checking it against common everyday/audit-known words."""
    failures = []
    for concept in CONCEPTS:
        hard_neg_set = set(concept.hard_negatives)
        for variant in concept.variants:
            if len(variant.split()) != 1 or not _is_devanagari(variant):
                continue
            real_neighbors = _one_edit_neighbors(variant) & _REAL_WORD_CORPUS
            for neighbor in real_neighbors:
                if neighbor not in hard_neg_set:
                    failures.append((concept.term, variant, neighbor))
    assert not failures, f"Uncovered one-edit collisions: {failures}"


@pytest.mark.parametrize(
    ("concept_term", "collision_word"),
    [
        ("Asthma", "दवा"),
        ("Fungal Infection", "दाल"),
        ("Fungal Infection", "दान"),
        ("Abdominal Pain", "पेड़"),
        ("Urinary Tract Infection", "पेशा"),
        ("Constipation", "कब्र"),
        ("Allergic Rhinitis", "नाच"),
        ("Back Pain", "कमरा"),
        ("Skin Rash", "दान"),
        ("Skin Rash", "cash"),
        ("Pain", "rain"),
        ("Common Cold", "gold"),
        ("Shortness of Breath", "सांप"),
    ],
)
def test_known_one_edit_collisions_are_hard_negatived(
    concept_term: str, collision_word: str
) -> None:
    """Pin every one-edit collision found by the commit 26c42ea audit plus
    this suite's programmatic scan (Fungal Infection/दान)."""
    concept = next(c for c in CONCEPTS if c.term == concept_term)
    assert collision_word in concept.hard_negatives


# ── सर्दी polysemy policy ────────────────────────────────────────────────────
# सर्दी means BOTH "common cold" (illness) and "winter/cold weather" in
# Hindi — the SAME string, not a one-edit neighbor. Concept matching here is
# SPAN-ONLY (see src/l3_5_normalize.py: candidate spans are encoded in
# isolation, never with their surrounding sentence — confirmed empirically
# in eval/gloss_audit.py, where the same word always gets the same cosine
# regardless of context), so this architecture CANNOT disambiguate "सर्दी
# में बहुत ठंड होती है" (winter) from "मुझे सर्दी हो गयी" (illness): both
# encode the identical unigram "सर्दी". Policy: सर्दी keeps its Common Cold
# variant listing (removing it would break the genuine illness case, see
# test_sardi_maps_to_common_cold in tests/test_l3_5_normalize.py) and relies
# on the higher unigram bar alone — which does NOT fix this specific case,
# because सर्दी is an exact variant string (cosine ~1.0), not a near-miss.
# This is a documented, accepted residual risk — same honesty pattern as the
# existing test_cold_weather_not_glossed_as_common_cold xfail.


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "सर्दी (winter/cold weather) and सर्दी (common cold, the illness) are "
        "the identical string — concept spans are encoded in isolation with "
        "no sentence context (verified in eval/gloss_audit.py), so this "
        "architecture cannot disambiguate them. The higher unigram bar does "
        "not help here because सर्दी is an exact variant match (cosine ~1.0), "
        "not a near-miss. Documented residual risk, not a bug in this phase."
    ),
)
def test_sardi_winter_weather_not_glossed_as_common_cold() -> None:
    from src.l3_5_normalize import normalize
    from src.types import Turn

    text = "सर्दी में बहुत ठंड होती है"
    turns = [Turn(speaker_role="PATIENT", text=text, start=0.0, end=1.0)]
    result = normalize(turns)
    assert "(Common Cold)" not in result[0].text
