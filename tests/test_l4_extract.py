"""Tests for L4 extraction: dose-fabrication guard, CDSCO validation, fallback."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.types import Turn


def _turn(role: str, text: str) -> Turn:
    return Turn(speaker_role=role, text=text, start=0.0, end=1.0)


def _mock_ollama(content: dict):
    """Return a mock ollama.chat response for the given JSON content."""
    msg = MagicMock()
    msg.content = json.dumps(content)
    response = MagicMock()
    response.message = msg
    return response


_EMPTY_RESPONSE = {
    "chief_complaint": None,
    "history": None,
    "diagnosis": [],
    "medications": [],
    "investigations": [],
    "advice": None,
    "follow_up": None,
    "low_confidence_fields": [],
}


# ── CDSCO validation ─────────────────────────────────────────────────────────


def test_known_drug_is_validated() -> None:
    from src.cdsco import validate_drug

    assert validate_drug("paracetamol") is True
    assert validate_drug("Paracetamol") is True
    assert validate_drug("  AMOXICILLIN  ") is True


def test_unknown_drug_not_validated() -> None:
    from src.cdsco import validate_drug

    assert validate_drug("brandnewdrug123") is False


def test_validate_drug_empty_string() -> None:
    from src.cdsco import validate_drug

    assert validate_drug("") is False


# ── Dose-fabrication guard ────────────────────────────────────────────────────


def test_dose_null_when_not_stated() -> None:
    """When LLM correctly returns null dose, Medication.dose must be None."""
    from src.l4_extract import extract

    response_data = {
        **_EMPTY_RESPONSE,
        "chief_complaint": "fever",
        "medications": [
            {"drug": "paracetamol", "dose": None, "frequency": "twice daily", "duration": "3 days"}
        ],
    }
    turns = [_turn("DOCTOR", "take paracetamol twice daily for 3 days")]
    with patch("ollama.chat", return_value=_mock_ollama(response_data)):
        note = extract(turns)

    assert len(note.medications) == 1
    med = note.medications[0]
    assert med.dose is None
    assert med.frequency == "twice daily"
    # dose_unknown must appear in low_confidence_fields
    assert any("dose_unknown" in f for f in note.low_confidence_fields)


def test_dose_present_when_stated() -> None:
    """When LLM returns an explicit dose, it must be preserved."""
    from src.l4_extract import extract

    response_data = {
        **_EMPTY_RESPONSE,
        "medications": [
            {"drug": "ibuprofen", "dose": "400mg", "frequency": "three times daily", "duration": None}
        ],
    }
    turns = [_turn("DOCTOR", "ibuprofen 400mg three times daily")]
    with patch("ollama.chat", return_value=_mock_ollama(response_data)):
        note = extract(turns)

    assert note.medications[0].dose == "400mg"


# ── CDSCO flag propagation ────────────────────────────────────────────────────


def test_unvalidated_drug_flagged_in_low_confidence() -> None:
    from src.l4_extract import extract

    response_data = {
        **_EMPTY_RESPONSE,
        "medications": [
            {"drug": "unknownbrand42", "dose": "10mg", "frequency": None, "duration": None}
        ],
    }
    turns = [_turn("DOCTOR", "take unknownbrand42 10mg")]
    with patch("ollama.chat", return_value=_mock_ollama(response_data)):
        note = extract(turns)

    assert not note.medications[0].validated
    assert any("unvalidated" in f for f in note.low_confidence_fields)


def test_validated_drug_not_flagged() -> None:
    from src.l4_extract import extract

    response_data = {
        **_EMPTY_RESPONSE,
        "medications": [
            {"drug": "metformin", "dose": "500mg", "frequency": "twice daily", "duration": "30 days"}
        ],
    }
    turns = [_turn("DOCTOR", "metformin 500mg twice daily")]
    with patch("ollama.chat", return_value=_mock_ollama(response_data)):
        note = extract(turns)

    assert note.medications[0].validated
    assert not any("unvalidated" in f for f in note.low_confidence_fields)


# ── low_confidence_fields population ─────────────────────────────────────────


def test_llm_low_confidence_fields_preserved() -> None:
    """Fields the LLM marks uncertain must appear in the final note."""
    from src.l4_extract import extract

    response_data = {
        **_EMPTY_RESPONSE,
        "chief_complaint": "vague discomfort",
        "low_confidence_fields": ["chief_complaint"],
    }
    turns = [_turn("PATIENT", "feeling off")]
    with patch("ollama.chat", return_value=_mock_ollama(response_data)):
        note = extract(turns)

    assert "chief_complaint" in note.low_confidence_fields


# ── Malformed JSON fallback ───────────────────────────────────────────────────


def test_malformed_json_returns_minimal_note() -> None:
    """Both JSON parse failures must return a minimal, all-flagged note."""
    from src.l4_extract import extract

    bad_msg = MagicMock()
    bad_msg.content = "not json at all {"
    bad_resp = MagicMock()
    bad_resp.message = bad_msg

    turns = [_turn("DOCTOR", "something happened")]
    with patch("ollama.chat", return_value=bad_resp):
        note = extract(turns)

    assert note.chief_complaint is None
    assert len(note.medications) == 0
    assert "chief_complaint" in note.low_confidence_fields


# ── Live Ollama (slow) ────────────────────────────────────────────────────────


@pytest.mark.slow
def test_extract_live_dose_omitted() -> None:
    """Live call: dose must be null when transcript never states one."""
    from src.l4_extract import extract

    turns = [
        _turn("DOCTOR", "You have fever. Take paracetamol twice a day for three days."),
        _turn("PATIENT", "Okay doctor."),
    ]
    note = extract(turns)
    # At least one medication extracted
    assert len(note.medications) >= 1
    paracetamol_meds = [m for m in note.medications if "paracetamol" in m.drug.lower()]
    if paracetamol_meds:
        # No specific dose was stated — must be null
        assert paracetamol_meds[0].dose is None


@pytest.mark.slow
def test_extract_live_returns_valid_schema() -> None:
    from src.types import ClinicalNote

    from src.l4_extract import extract

    turns = [
        _turn("DOCTOR", "You have Type 2 Diabetes Mellitus. Take metformin 500mg twice daily."),
        _turn("DOCTOR", "Come back in two weeks for a follow-up."),
    ]
    note = extract(turns)
    assert isinstance(note, ClinicalNote)
    assert isinstance(note.low_confidence_fields, list)
    assert isinstance(note.diagnosis, list)
    assert isinstance(note.medications, list)
