"""Numerical-correctness tests for eval metrics (known inputs → known outputs)."""

import pytest

from eval.metrics import (
    corpus_word_error_rate,
    keyword_hits,
    keyword_wer,
    normalize_text,
    word_error_rate,
)


def test_normalize_lowercases_and_strips_punctuation() -> None:
    assert normalize_text("Daily, Three Times.") == "daily three times"


def test_normalize_treats_danda_as_space() -> None:
    assert normalize_text("लगाना। And") == "लगाना and"


def test_wer_identical_is_zero() -> None:
    assert word_error_rate("the cat sat", "the cat sat") == 0.0


def test_wer_one_substitution_of_three() -> None:
    assert word_error_rate("the cat sat", "the dog sat") == pytest.approx(1 / 3)


def test_wer_ignores_case_and_punctuation() -> None:
    assert word_error_rate("Daily, three times.", "daily three times") == 0.0


def test_wer_empty_reference_empty_hypothesis_is_zero() -> None:
    assert word_error_rate("", "") == 0.0


def test_wer_empty_reference_nonempty_hypothesis_is_one() -> None:
    assert word_error_rate("", "extra words") == 1.0


def test_keyword_hits_counts_present_and_missed() -> None:
    # both keywords are in the reference; hypothesis drops "daily"
    present, missed = keyword_hits(
        "augmentin daily three times", "augmentin 650", ["augmentin", "daily"]
    )
    assert (present, missed) == (2, 1)


def test_keyword_wer_half_missed() -> None:
    assert keyword_wer(
        "augmentin daily", "augmentin 650", ["augmentin", "daily"]
    ) == pytest.approx(0.5)


def test_keyword_wer_zero_when_no_keyword_in_reference() -> None:
    # keyword not in reference → nothing safety-critical to score → 0.0
    assert keyword_wer("hello world", "hello world", ["augmentin"]) == 0.0


def test_keyword_wer_matches_multiword_substring() -> None:
    assert keyword_wer(
        "give daily three times today", "daily three times", ["daily three times"]
    ) == 0.0


# ── Token-boundary matching fixtures (external-review false-positive cases) ──


def test_keyword_no_substring_match_inside_longer_token() -> None:
    # "ors" must NOT match inside "doctors" — the review's false-positive case.
    present, missed = keyword_hits("ors solution", "the doctors said", ["ors"])
    assert (present, missed) == (1, 1)


def test_keyword_dolo_does_not_match_dolores() -> None:
    # "dolo" (drug) must NOT match inside "dolores" (name).
    present, missed = keyword_hits("take dolo 650", "she met dolores", ["dolo"])
    assert (present, missed) == (1, 1)


def test_keyword_exact_token_still_matches() -> None:
    present, missed = keyword_hits("take dolo 650", "take dolo now", ["dolo"])
    assert (present, missed) == (1, 0)


def test_keyword_strict_no_match_on_glued_alphanumeric() -> None:
    # DELIBERATE strictness: "paracetamol" does not match "paracetamol500".
    # No glued tokens occur in the frozen bench; revisit only with evidence.
    present, missed = keyword_hits(
        "paracetamol dose", "gave paracetamol500 dose", ["paracetamol"]
    )
    assert (present, missed) == (1, 1)


def test_multiword_keyword_requires_contiguous_tokens() -> None:
    # "cough syrup" split by an intervening word is NOT a match.
    present, missed = keyword_hits(
        "take cough syrup daily", "the cough and syrup", ["cough syrup"]
    )
    assert (present, missed) == (1, 1)


def test_multiword_keyword_matches_contiguous_tokens() -> None:
    present, missed = keyword_hits(
        "take cough syrup daily", "use cough syrup twice", ["cough syrup"]
    )
    assert (present, missed) == (1, 0)


def test_keyword_devanagari_token_boundary() -> None:
    # Devanagari keyword must match as a whole token, not as a substring
    # of a longer Devanagari token ("दर्द" inside "सरदर्द" is no match).
    present, missed = keyword_hits("पेट में दर्द है", "उसको सरदर्द है", ["दर्द"])
    assert (present, missed) == (1, 1)
    present, missed = keyword_hits("पेट में दर्द है", "पेट में दर्द बहुत", ["दर्द"])
    assert (present, missed) == (1, 0)


def test_keyword_matches_at_string_edges() -> None:
    present, missed = keyword_hits("augmentin now", "augmentin", ["augmentin"])
    assert (present, missed) == (1, 0)
    present, missed = keyword_hits("take augmentin", "start augmentin", ["augmentin"])
    assert (present, missed) == (1, 0)


def test_keyword_hyphen_is_a_token_boundary() -> None:
    # A hyphenated compound genuinely contains the keyword at a boundary:
    # "stress" IS captured by "stress-related" (unlike "ors" in "doctors").
    present, missed = keyword_hits(
        "stress and fatigue", "it could be stress-related", ["stress"]
    )
    assert (present, missed) == (1, 0)


def test_keyword_presence_in_reference_uses_token_boundary_too() -> None:
    # Denominator discipline: "ors" inside reference word "doctors" does not
    # count the keyword as present — the population shrinks honestly.
    present, missed = keyword_hits("the doctors said", "the doctors said", ["ors"])
    assert (present, missed) == (0, 0)


def test_corpus_wer_micro_averages_over_words() -> None:
    # sample A: 1 error / 3 words; sample B: 0 errors / 2 words.
    # micro = total_errors / total_words = 1 / 5 = 0.2 (NOT mean of rates 0.167)
    refs = ["the cat sat", "good morning"]
    hyps = ["the dog sat", "good morning"]
    assert corpus_word_error_rate(refs, hyps) == pytest.approx(0.2)
