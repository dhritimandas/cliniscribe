"""Tests for L3 ASR: doctor-heuristic scoring and role assignment logic."""

import pytest

from src.l3_asr import _doctor_score
from src.types import Segment, Turn


def test_doctor_score_hindi_clinical_terms() -> None:
    assert _doctor_score("dawai do din ke liye") > 0


def test_doctor_score_english_question_forms() -> None:
    assert _doctor_score("how long have you had pain") > 2


def test_doctor_score_empty_text() -> None:
    assert _doctor_score("") == 0


def test_doctor_score_irrelevant_text() -> None:
    assert _doctor_score("hello hi yes okay") == 0


def test_transcribe_produces_turns_from_real_wav() -> None:
    """L3 must return at least one Turn for a real audio file."""
    import os

    from src.l3_asr import transcribe

    wav = "outputs/sample_00_16k.wav"
    if not os.path.exists(wav):
        pytest.skip("pre-processed WAV not found — run phase-A first")

    segments = [Segment(start=0.03, end=2.75, speaker="SPEAKER_00")]
    turns = transcribe(wav, segments)
    assert len(turns) >= 1
    assert all(isinstance(t, Turn) for t in turns)
    assert all(t.text.strip() for t in turns)


def test_role_unknown_for_single_speaker() -> None:
    """Role must be UNKNOWN when only one speaker label appears."""
    import os

    from src.l3_asr import transcribe

    wav = "outputs/sample_00_16k.wav"
    if not os.path.exists(wav):
        pytest.skip("pre-processed WAV not found — run phase-A first")

    segments = [Segment(start=0.03, end=2.75, speaker="SPEAKER_00")]
    turns = transcribe(wav, segments)
    assert all(t.speaker_role == "UNKNOWN" for t in turns)
