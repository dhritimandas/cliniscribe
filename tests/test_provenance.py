"""Tests for web/provenance.py: token-overlap matcher and flag translation."""

from src.types import ClinicalNote, Diagnosis, Medication, Symptom, Turn, Vital
from web.provenance import flags_by_path, provenance_for_note


def _turn(text: str, i: int) -> Turn:
    return Turn(speaker_role="DOCTOR", text=text, start=float(i), end=float(i) + 1.0)


# ── provenance_for_note ──────────────────────────────────────────────────


def test_latin_field_matches_latin_turn() -> None:
    note = ClinicalNote(chief_complaint="fever since Monday", history=None)
    turns = [_turn("Patient reports fever since Monday", 0)]

    prov = provenance_for_note(note, turns)

    assert prov["chief_complaint"] == {
        "turn_index": 0,
        "start": 0.0,
        "end": 1.0,
        "snippet": "Patient reports fever since Monday",
    }


def test_devanagari_field_matches_devanagari_turn() -> None:
    note = ClinicalNote(chief_complaint="बुखार तीन दिन से", history=None)
    turns = [_turn("मरीज को बुखार तीन दिन से है", 0)]

    prov = provenance_for_note(note, turns)

    assert prov["chief_complaint"]["turn_index"] == 0


def test_cross_script_match_via_phonetic_fold() -> None:
    # Devanagari drug token vs a Latin-script turn containing the same token —
    # the coarse Devanagari->Latin fold ("टेंडो" -> "tendo") must bridge scripts.
    note = ClinicalNote(
        chief_complaint=None,
        history=None,
        medications=[
            Medication(
                drug="टेंडो",
                dose=None,
                frequency=None,
                timing=None,
                duration=None,
                validated=False,
            )
        ],
    )
    turns = [_turn("Doctor prescribed Tendo Life tablets twice daily", 0)]

    prov = provenance_for_note(note, turns)

    assert prov["medications[0].drug"]["turn_index"] == 0


def test_no_shared_token_leaves_field_absent() -> None:
    note = ClinicalNote(chief_complaint="headache", history=None)
    turns = [_turn("Patient has abdominal pain", 0)]

    prov = provenance_for_note(note, turns)

    assert "chief_complaint" not in prov


def test_short_tokens_below_three_chars_do_not_count() -> None:
    # "is" and "of" are shared but both <3 chars post-fold; must not match.
    note = ClinicalNote(chief_complaint="is of it", history=None)
    turns = [_turn("is of it too", 0)]

    prov = provenance_for_note(note, turns)

    assert "chief_complaint" not in prov


def test_empty_field_never_produces_provenance_entry() -> None:
    note = ClinicalNote(chief_complaint=None, history=None)
    turns = [_turn("anything", 0)]

    prov = provenance_for_note(note, turns)

    assert "chief_complaint" not in prov


def test_best_turn_picked_by_higher_token_overlap() -> None:
    note = ClinicalNote(chief_complaint="fever headache nausea", history=None)
    turns = [
        _turn("patient reports fever only", 0),
        _turn("patient reports fever headache and nausea today", 1),
    ]

    prov = provenance_for_note(note, turns)

    assert prov["chief_complaint"]["turn_index"] == 1


# ── flags_by_path ─────────────────────────────────────────────────────────


def test_drug_name_flag_resolves_to_its_row() -> None:
    note = ClinicalNote(
        chief_complaint=None,
        history=None,
        medications=[
            Medication(
                drug="Dolo 650",
                dose=None,
                frequency=None,
                timing=None,
                duration=None,
                validated=True,
            )
        ],
        low_confidence_fields=["medications.Dolo 650.dose_unknown"],
    )

    flags = flags_by_path(note)

    assert flags == {"medications[0].dose": "medications.Dolo 650.dose_unknown"}


def test_diagnosis_term_flag_resolves_to_its_row() -> None:
    note = ClinicalNote(
        chief_complaint=None,
        history=None,
        diagnosis=[Diagnosis(term="Pulmonary Embolism", snomed_id=None)],
        low_confidence_fields=["diagnosis.Pulmonary Embolism.no_transcript_overlap"],
    )

    flags = flags_by_path(note)

    assert flags == {
        "diagnosis[0].term": "diagnosis.Pulmonary Embolism.no_transcript_overlap"
    }


def test_symptom_and_vital_bare_name_flags_resolve_by_index() -> None:
    note = ClinicalNote(
        chief_complaint=None,
        history=None,
        symptoms=[Symptom(name="fever")],
        vitals=[Vital(name="BP", value="140/90 mmHg")],
        low_confidence_fields=["symptoms.fever", "vitals.BP"],
    )

    flags = flags_by_path(note)

    assert flags == {
        "symptoms[0].name": "symptoms.fever",
        "vitals[0].value": "vitals.BP",
    }


def test_bare_top_level_field_name_maps_to_whole_field_path() -> None:
    note = ClinicalNote(
        chief_complaint=None,
        history=None,
        low_confidence_fields=["chief_complaint", "medications"],
    )

    flags = flags_by_path(note)

    assert flags == {"chief_complaint": "chief_complaint", "medications": "medications"}


def test_unresolvable_flag_lands_in_general_bucket() -> None:
    note = ClinicalNote(
        chief_complaint=None,
        history=None,
        low_confidence_fields=["medications.Nonexistent Drug.dose_unknown"],
    )

    flags = flags_by_path(note)

    assert flags == {"_general": ["medications.Nonexistent Drug.dose_unknown"]}


def test_no_flags_returns_empty_dict() -> None:
    note = ClinicalNote(chief_complaint="fever", history=None, low_confidence_fields=[])

    assert flags_by_path(note) == {}
