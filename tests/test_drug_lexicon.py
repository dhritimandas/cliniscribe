"""Tests for src/drug_lexicon.py: lexicon merge integrity + fold-key matcher.

Adversarial focus (patient-safety, not coverage): the matcher must never
guess a WRONG drug. Near-collision brand/generic pairs must resolve to
themselves (or trigger the ambiguity guard), common Hindi/English words must
never canonicalize, and the historical incident set is checked honestly —
including the one case (जिफिट -> zifi) that the bound correctly does NOT
reach.
"""

import pytest

from src.cdsco import _APPROVED_DRUGS
from src.drug_lexicon import (
    DRUG_LEXICON,
    _ADDITIONAL_DRUGS,
    _distance_bound,
    _fold,
    _FOLD_INDEX,
    _levenshtein,
    canonicalize_drug_span,
)
from src.l3_5_normalize import _normalize_drug_text

# ── Lexicon merge integrity ──────────────────────────────────────────────────


def test_lexicon_merges_cdsco_without_duplication() -> None:
    """The CDSCO seed set must be a subset; additions must not repeat it."""
    assert _APPROVED_DRUGS <= DRUG_LEXICON
    assert _ADDITIONAL_DRUGS.isdisjoint(_APPROVED_DRUGS)


def test_lexicon_size_in_expected_range() -> None:
    """Additional knowledge-sourced entries land in the requested 350-500 band."""
    assert 350 <= len(_ADDITIONAL_DRUGS) <= 500


def test_no_exact_fold_collisions_in_shipped_lexicon() -> None:
    """No two distinct canonical drugs currently fold to the identical key.

    Documents the current state (546 lexicon entries, 546 distinct fold
    keys) rather than asserting collisions can never happen — if a future
    addition creates one, canonicalize_drug_span() already handles it safely
    (ambiguity guard on the exact tier too), but this test makes any new
    collision visible at review time rather than silently absorbed.
    """
    collisions = {k: v for k, v in _FOLD_INDEX.items() if len(v) > 1}
    assert collisions == {}


# ── Fold function ─────────────────────────────────────────────────────────


def test_fold_lowercases_and_keeps_spaces() -> None:
    # _fold keeps spaces (l4_extract's and drug_bench's word-window
    # comparisons need them); a despaced dictionary key is
    # _fold(text).replace(" ", "") — see canonicalize_drug_span below.
    assert _fold("Zerodol SP") == "zerodol sp"
    assert _fold("Zerodol SP").replace(" ", "") == "zerodolsp"


def test_fold_devanagari_x_digraph() -> None:
    # क्स -> x (_FOLD_DIGRAPHS) -> ks (the Latin-orthography x -> ks rule,
    # _apply_orthography) — so an English "norflox" and a Devanagari
    # "...लोक्स" spelling land on the same key. Shared convention with
    # src/l4_extract.py's fold_drug, which now imports this function.
    assert "ks" in _fold("क्सरे")


def test_fold_collapses_doubled_consonants() -> None:
    assert _fold("allegra") == _fold("alegra")


# ── Distance bound ───────────────────────────────────────────────────────


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 6, 7, 8])
def test_distance_bound_none_below_nine(length: int) -> None:
    # A 5-8-char distance-1 tier was tried and rejected: it produced 3
    # wrong-drug substitutions ("दो तीन" -> drotin, "से vital" -> revital,
    # a surname "पांडे" -> pan d) against 1 correct one on the 49-clip drug
    # bench. See _distance_bound's docstring and LEARNINGS.md.
    assert _distance_bound(length) is None


@pytest.mark.parametrize("length", [9, 12, 20])
def test_distance_bound_two_for_nine_plus(length: int) -> None:
    assert _distance_bound(length) == 2


def test_levenshtein_basic() -> None:
    assert _levenshtein("kitten", "sitting", max_dist=5) == 3


def test_levenshtein_length_filter_short_circuits() -> None:
    # |len diff| exceeds max_dist -> returns max_dist + 1 without computing.
    assert _levenshtein("abc", "abcdefgh", max_dist=1) == 2


# ── Near-collision adversarial pairs (design point 4) ───────────────────────
# Policy this implementation enforces: brand and generic names within the
# SAME drug family are kept as fully distinct canonical entries. The fuzzy
# tier merges two entries only when their fold keys are within the
# length-scaled edit-distance bound; family members are long-vs-short
# spellings (brand abbreviation vs full generic), so their length difference
# alone almost always exceeds the bound, which is what keeps them apart. A
# same-family match is only "acceptable" in the sense that it would require
# fold-EXACT equality (impossible here, given the length gap) — the matcher
# never fuzzily collapses a brand into its generic or vice versa.


def test_dolo_and_dolonex_resolve_independently() -> None:
    """dolo (paracetamol) and dolonex (piroxicam) are different drugs."""
    assert canonicalize_drug_span("dolo") == ("dolo", 1.0)
    assert canonicalize_drug_span("dolonex") == ("dolonex", 1.0)


def test_telma_family_resolves_independently() -> None:
    """telma / telma am / telma h / telmisartan never cross-contaminate."""
    assert canonicalize_drug_span("telma") == ("telma", 1.0)
    assert canonicalize_drug_span("telma am") == ("telma am", 1.0)
    assert canonicalize_drug_span("telma h") == ("telma h", 1.0)
    assert canonicalize_drug_span("telmisartan") == ("telmisartan", 1.0)


def test_aten_and_augmentin_absurd_distance_no_match() -> None:
    """aten (atenolol brand) is nowhere near augmentin; must not collide."""
    assert canonicalize_drug_span("aten") == ("aten", 1.0)
    assert canonicalize_drug_span("augmentin") == ("augmentin", 1.0)


def test_losar_and_losartan_resolve_independently() -> None:
    """losar (brand) and losartan (generic) — same family, not fold-exact."""
    assert canonicalize_drug_span("losar") == ("losar", 1.0)
    assert canonicalize_drug_span("losartan") == ("losartan", 1.0)


# ── Real ambiguity-guard firings discovered in this lexicon ─────────────────
# Found by scanning canonical fold-key pairs within the length-scaled bound
# (not synthetic): a digit-garbled "vitamin b<n>" span (9-char key, bound 2)
# is within bound of TEN distinct vitamin-letter canonicals, and a 1-letter
# blend of albendazole/mebendazole (11-char keys, bound 2) sits at distance 1
# from both — two different antiparasitics that happen to differ by one
# letter. Two distinct fold-index entries within bound is ambiguous by
# definition, regardless of whether the pair is clinically related.


def test_ambiguity_guard_fires_on_garbled_vitamin_digit() -> None:
    """A garbled vitamin-B digit is within bound of many distinct vitamins."""
    for span in ("vitamin b7", "vitamin b4", "vitamin b5", "vitamin bi"):
        assert canonicalize_drug_span(span) is None


def test_ambiguity_guard_fires_on_albendazole_mebendazole_blend() -> None:
    """A 1-letter blend of albendazole/mebendazole sits at distance 1 from both."""
    assert canonicalize_drug_span("aebendazole") is None
    # The two source drugs still resolve to themselves independently.
    assert canonicalize_drug_span("albendazole") == ("albendazole", 1.0)
    assert canonicalize_drug_span("mebendazole") == ("mebendazole", 1.0)


# ── Common words must never canonicalize (~30 innocents) ────────────────────

_INNOCENTS: tuple[str, ...] = (
    "पानी", "water", "doctor", "morning", "नहीं", "प्रॉब्लम",
    "evening", "night", "please", "patient", "medicine", "tablet",
    "the", "and", "take", "rest", "blood", "pressure", "test", "daily",
    "twice", "thrice", "days", "weeks", "month", "again", "because",
    "table", "would", "should", "could", "हाँ", "ठीक", "है",
)


@pytest.mark.parametrize("word", _INNOCENTS)
def test_innocents_never_canonicalize(word: str) -> None:
    assert canonicalize_drug_span(word) is None


# ── Historical incident set (design point 4) ────────────────────────────────


def test_naxdom_devanagari_incident_via_full_pipeline() -> None:
    """नैक्स्टोम -> naxdom: already owned by the curated tier (tier order
    means the new lexicon tier never has to reach it in production)."""
    result = _normalize_drug_text("और एक नैक्स्टोम 500 खाईएगा")
    assert "naxdom 500" in result


def test_zifi_incident_not_reached_report_honestly() -> None:
    """जिफिट -> zifi is NOT recovered: its 5-char fold key is below the
    length-9 fuzzy floor (exact-fold-only), and even under the originally
    proposed 5-8/distance-1 tier the edit distance was 2, over bound. This
    matches the prior LEARNINGS.md finding (~0.67 string similarity, below
    any safe substitution threshold) — the doctrine correctly declines this
    match rather than guessing the wrong drug.
    """
    assert canonicalize_drug_span("जिफिट") is None


def test_glycomate_recovered_via_lexicon_tier() -> None:
    """glycomate (misspelling) -> glycomet (real brand, metformin).

    Confidence moved from the pre-orthography 0.7778 (9-char key, distance 2)
    to 0.8 (10-char key "glaikomate", distance 2 from "glaikomet") once the
    y -> ai Latin-orthography rule folds "gly-" the same way Devanagari
    spells its /aɪ/ diphthong (ग्लाइ) — see _apply_orthography.
    """
    assert canonicalize_drug_span("glycomate") == ("glycomet", 0.8)


def test_gmenti_recovered_via_lexicon_tier_with_context() -> None:
    """'gmenti' alone is too short/ambiguous-radius; 'aur gmenti' (the real
    ASR window, per outputs/beam_study.json) reaches augmentin at distance 2
    (bound 2 for a 9-char key) — same outcome the pre-existing CDSCO-fuzzy
    tier already reached by coincidence of ratio; the new tier agrees.
    """
    assert canonicalize_drug_span("gmenti") is None
    result = canonicalize_drug_span("aur gmenti")
    assert result is not None
    assert result[0] == "augmentin"


# ── Digit exclusion from the fold key ───────────────────────────────────────


def test_digit_only_span_returns_none() -> None:
    assert canonicalize_drug_span("500") is None


def test_digit_tokens_excluded_from_key_but_drug_still_resolves() -> None:
    result = canonicalize_drug_span("naxdom 500")
    assert result is not None
    assert result[0] == "naxdom"
