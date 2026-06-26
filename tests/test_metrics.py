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


def test_corpus_wer_micro_averages_over_words() -> None:
    # sample A: 1 error / 3 words; sample B: 0 errors / 2 words.
    # micro = total_errors / total_words = 1 / 5 = 0.2 (NOT mean of rates 0.167)
    refs = ["the cat sat", "good morning"]
    hyps = ["the dog sat", "good morning"]
    assert corpus_word_error_rate(refs, hyps) == pytest.approx(0.2)
