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
    "symptoms": [],
    "vitals": [],
    "examination": None,
    "diagnosis": [],
    "medications": [],
    "investigations": [],
    "diagnostic_results": [],
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


# ── Extended schema: symptoms, vitals, exam, diagnostic results, timing ──────


def test_symptoms_extracted_with_qualifiers() -> None:
    """Symptoms array must populate name, severity, and finding_status."""
    from src.l4_extract import extract

    response_data = {
        **_EMPTY_RESPONSE,
        "symptoms": [
            {"name": "abdominal pain", "finding_status": "Present",
             "severity": "Moderate", "since": "2 days"},
            {"name": "nausea", "finding_status": "Present",
             "severity": None, "since": None},
        ],
    }
    turns = [_turn("PATIENT", "pet dard 2 din se, ji machal raha")]
    with patch("ollama.chat", return_value=_mock_ollama(response_data)):
        note = extract(turns)

    assert len(note.symptoms) == 2
    assert note.symptoms[0].name == "abdominal pain"
    assert note.symptoms[0].severity == "Moderate"
    assert note.symptoms[0].since == "2 days"
    assert note.symptoms[1].severity is None


def test_symptom_finding_status_absent_preserved() -> None:
    """A denied symptom must keep finding_status 'Absent'."""
    from src.l4_extract import extract

    response_data = {
        **_EMPTY_RESPONSE,
        "symptoms": [{"name": "vomiting", "finding_status": "Absent"}],
    }
    turns = [_turn("PATIENT", "no vomiting")]
    with patch("ollama.chat", return_value=_mock_ollama(response_data)):
        note = extract(turns)

    assert note.symptoms[0].finding_status == "Absent"


def test_symptom_finding_status_defaults_present() -> None:
    """When finding_status is omitted, it must default to 'Present'."""
    from src.l4_extract import extract

    response_data = {**_EMPTY_RESPONSE, "symptoms": [{"name": "fever"}]}
    turns = [_turn("PATIENT", "bukhar hai")]
    with patch("ollama.chat", return_value=_mock_ollama(response_data)):
        note = extract(turns)

    assert note.symptoms[0].finding_status == "Present"


def test_vitals_extracted() -> None:
    """Vitals must populate name and value-with-unit; valueless vitals dropped."""
    from src.l4_extract import extract

    response_data = {
        **_EMPTY_RESPONSE,
        "vitals": [
            {"name": "BP", "value": "130/90 mmHg"},
            {"name": "SpO2", "value": "97 %"},
            {"name": "Pulse", "value": ""},  # no measurement → dropped
        ],
    }
    turns = [_turn("DOCTOR", "BP 130 by 90, oxygen 97 percent")]
    with patch("ollama.chat", return_value=_mock_ollama(response_data)):
        note = extract(turns)

    assert len(note.vitals) == 2
    assert note.vitals[0].name == "BP"
    assert note.vitals[0].value == "130/90 mmHg"


def test_medication_timing_extracted() -> None:
    """Medication timing (before/after food) must be preserved."""
    from src.l4_extract import extract

    response_data = {
        **_EMPTY_RESPONSE,
        "medications": [
            {"drug": "pan d", "dose": None, "frequency": "three times a day",
             "timing": "before food", "duration": None}
        ],
    }
    turns = [_turn("DOCTOR", "pan d before food three times a day")]
    with patch("ollama.chat", return_value=_mock_ollama(response_data)):
        note = extract(turns)

    assert note.medications[0].timing == "before food"


def test_diagnosis_status_extracted() -> None:
    """Diagnosis status must be preserved when stated."""
    from src.l4_extract import extract

    response_data = {
        **_EMPTY_RESPONSE,
        "diagnosis": [
            {"term": "Acute Gastritis", "snomed_id": None, "status": "Suspected"}
        ],
    }
    turns = [_turn("DOCTOR", "looks like gastritis, suspected")]
    with patch("ollama.chat", return_value=_mock_ollama(response_data)):
        note = extract(turns)

    assert note.diagnosis[0].status == "Suspected"


def test_diagnostic_results_distinct_from_investigations() -> None:
    """Ordered tests and in-hand results must land in separate fields."""
    from src.l4_extract import extract

    response_data = {
        **_EMPTY_RESPONSE,
        "investigations": ["CBC", "LFT"],
        "diagnostic_results": ["Hb 9.2 g/dL", "fasting glucose 142 mg/dL"],
    }
    turns = [_turn("DOCTOR", "Hb is 9.2, get a CBC and LFT done")]
    with patch("ollama.chat", return_value=_mock_ollama(response_data)):
        note = extract(turns)

    assert note.investigations == ["CBC", "LFT"]
    assert note.diagnostic_results == ["Hb 9.2 g/dL", "fasting glucose 142 mg/dL"]


def test_examination_free_text_preserved() -> None:
    from src.l4_extract import extract

    response_data = {
        **_EMPTY_RESPONSE,
        "examination": "abdomen soft, mild epigastric tenderness",
    }
    turns = [_turn("DOCTOR", "abdomen soft, tender in epigastrium")]
    with patch("ollama.chat", return_value=_mock_ollama(response_data)):
        note = extract(turns)

    assert note.examination == "abdomen soft, mild epigastric tenderness"


def test_new_fields_default_empty_when_omitted() -> None:
    """Back-compat: a response omitting the new keys must yield empty collections."""
    from src.l4_extract import extract

    # Minimal response without any of the new keys
    response_data = {"chief_complaint": "fever", "medications": []}
    turns = [_turn("PATIENT", "fever")]
    with patch("ollama.chat", return_value=_mock_ollama(response_data)):
        note = extract(turns)

    assert note.symptoms == []
    assert note.vitals == []
    assert note.examination is None
    assert note.diagnostic_results == []


# ── Phase-B regression tests (deterministic, no Ollama) ─────────────────────


def test_null_crash_regression_i18() -> None:
    """Regression for i=18: model emitting null for list fields must not crash.

    Root cause: data.get("key", []) returns None when JSON has "key": null;
    None is not iterable. Fixed by using (data.get("key") or []).
    Asserts: no exception; returns a valid ClinicalNote with empty lists.
    """
    from src.l4_extract import _build_note
    from src.types import ClinicalNote

    data = {
        "chief_complaint": None,
        "history": None,
        "symptoms": None,
        "vitals": None,
        "diagnosis": None,
        "medications": None,
        "investigations": None,
        "diagnostic_results": None,
        "low_confidence_fields": None,
    }
    note = _build_note(data)
    assert isinstance(note, ClinicalNote)
    assert note.symptoms == []
    assert note.vitals == []
    assert note.diagnosis == []
    assert note.medications == []
    assert note.investigations == []
    assert note.diagnostic_results == []
    assert isinstance(note.low_confidence_fields, list)


def test_cdsco_tablet_paracetamol_validates() -> None:
    """Regression for i=0: 'Tablet paracetamol' must validate as True.

    Root cause: validate_drug() did exact set-membership; 'tablet paracetamol'
    was not in the set (only bare 'paracetamol'). Fixed by stripping dosage-form
    words before lookup.
    """
    from src.cdsco import validate_drug

    assert validate_drug("Tablet paracetamol") is True
    assert validate_drug("Tab paracetamol") is True
    assert validate_drug("Capsule amoxicillin") is True
    assert validate_drug("Syrup paracetamol") is True


def test_calibration_flags_hallucinated_diagnosis_i23() -> None:
    """Regression for i=23: diagnosis term with no transcript word overlap must be flagged.

    Root cause: model hallucinated 'Pulmonary Embolism' on an acne/skin consult
    transcript with full confidence; low_confidence_fields was empty.
    Fixed by post-extraction word-token overlap check.
    """
    from src.l4_extract import _build_note

    # Simulated acne consult transcript (face cream, pimples, redness, dizziness)
    transcript = (
        "[UNKNOWN]: face cream pimples redness dizziness low blood pressure"
    )
    data = {
        "chief_complaint": "pimples and redness on face",
        "history": None,
        "symptoms": [{"name": "redness", "finding_status": "Present"}],
        "vitals": [],
        "examination": None,
        "diagnosis": [
            {"term": "Pulmonary Embolism", "snomed_id": None, "status": "Confirmed"}
        ],
        "medications": [],
        "investigations": [],
        "diagnostic_results": [],
        "advice": None,
        "follow_up": None,
        "low_confidence_fields": [],
    }
    note = _build_note(data, transcript=transcript)
    assert any(
        "no_transcript_overlap" in f and "Pulmonary Embolism" in f
        for f in note.low_confidence_fields
    ), f"Expected 'Pulmonary Embolism' flagged; got: {note.low_confidence_fields}"


def test_calibration_does_not_flag_devanagari_transcript() -> None:
    """Devanagari transcripts must not trigger false hallucination flags.

    Cross-lingual: Hindi/Marathi transcript → English diagnosis term. We cannot
    compare scripts, so we skip the check entirely for Devanagari transcripts.
    """
    from src.l4_extract import _build_note

    devanagari_transcript = (
        "[UNKNOWN]: चेहरे पर मुंहासे हैं, लालिमा है, चक्कर आ रहे हैं"
    )
    data = {
        "chief_complaint": None,
        "history": None,
        "symptoms": [],
        "vitals": [],
        "examination": None,
        "diagnosis": [{"term": "Acne Vulgaris", "snomed_id": None, "status": None}],
        "medications": [],
        "investigations": [],
        "diagnostic_results": [],
        "advice": None,
        "follow_up": None,
        "low_confidence_fields": [],
    }
    note = _build_note(data, transcript=devanagari_transcript)
    assert not any(
        "no_transcript_overlap" in f for f in note.low_confidence_fields
    ), f"Should not flag Devanagari transcript; got: {note.low_confidence_fields}"


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
