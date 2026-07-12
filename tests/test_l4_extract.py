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


def test_non_dict_list_item_skipped_not_crashed_idx99() -> None:
    """Regression for idx=99 (live eval): a raw string mixed into a list field
    must be skipped, not crash the whole extraction.

    Root cause: on a live sample the model emitted a JSON formatting slip
    where "vitals" was a list containing genuine vital dicts followed by
    stray strings (a duplicate-key parsing artifact, e.g. "examination: null").
    _build_note's list comprehensions called .get() on every item
    unconditionally, so the first non-dict item raised
    AttributeError("'str' object has no attribute 'get'"), caught by the
    generic except-Exception branch in extract() and collapsing the whole
    note to _empty_note() — losing every genuinely-extracted field, not just
    the malformed one. Fixed by filtering to dict items before construction.
    """
    from src.l4_extract import _build_note

    data = {
        "chief_complaint": None,
        "history": "",
        "symptoms": "",  # garbled: string instead of list — already handled by `or []`
        "vitals": [
            {"name": "blood pressure", "value": "137/81"},
            {"name": "pulse", "value": "71"},
            "examination: null",  # malformed tail from a duplicate-key parse slip
            "diagnosis: null",
            "medications: null",
        ],
        "diagnosis": [
            {"term": "raised total cholesterol", "snomed_id": None, "status": None},
            "some garbled string",
        ],
        "medications": [
            {"drug": "Tablet Calcirix XT", "dose": None, "frequency": "once daily"},
            "garbled medication string",
        ],
        "investigations": [],
        "diagnostic_results": [],
        "low_confidence_fields": [],
    }
    note = _build_note(data)
    assert [v.name for v in note.vitals] == ["blood pressure", "pulse"]
    assert [d.term for d in note.diagnosis] == ["raised total cholesterol"]
    assert [m.drug for m in note.medications] == ["Tablet Calcirix XT"]


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


# ── Generic-term leak (B2): दवाई/"medicine" must never be a drug NAME ──────


def test_generic_drug_name_becomes_unnamed_medication() -> None:
    from src.l4_extract import _build_note

    data = {
        "medications": [
            {"drug": "medicine", "dose": None, "frequency": "twice daily"},
        ]
    }
    note = _build_note(data)
    assert note.medications[0].drug == "unnamed medication 1"
    assert note.medications[0].frequency == "twice daily"  # info preserved
    assert "medications.unnamed medication 1.unnamed" in note.low_confidence_fields


def test_generic_devanagari_drug_name_becomes_unnamed() -> None:
    from src.l4_extract import _build_note

    data = {"medications": [{"drug": "दवाई", "dose": None}]}
    note = _build_note(data)
    assert note.medications[0].drug == "unnamed medication 1"


def test_devanagari_generic_head_phrase_is_generic() -> None:
    # "डायबिटीज की दवाई" = "diabetes medicine" — no identifiable drug.
    from src.l4_extract import _is_generic_drug_name

    assert _is_generic_drug_name("डायबिटीज की दवाई") is True


def test_branded_compound_is_not_generic() -> None:
    from src.l4_extract import _is_generic_drug_name

    assert _is_generic_drug_name("Benadryl cough syrup") is False
    assert _is_generic_drug_name("Paracetamol 650") is False


def test_bare_form_words_are_generic() -> None:
    from src.l4_extract import _is_generic_drug_name

    for term in ("cough syrup", "tablet", "injection", "Medicines"):
        assert _is_generic_drug_name(term) is True, term


def test_two_unnamed_medications_get_distinct_flags() -> None:
    from src.l4_extract import _build_note

    data = {
        "medications": [
            {"drug": "medicine", "frequency": "1-0-1"},
            {"drug": "दवा", "frequency": "0-0-1"},
        ]
    }
    note = _build_note(data)
    assert [m.drug for m in note.medications] == [
        "unnamed medication 1",
        "unnamed medication 2",
    ]
    flags = [f for f in note.low_confidence_fields if f.endswith(".unnamed")]
    assert len(flags) == 2  # no collision


def test_unnamed_medication_not_double_flagged_unvalidated() -> None:
    from src.l4_extract import _build_note

    data = {"medications": [{"drug": "medicine"}]}
    note = _build_note(data)
    unvalidated = [f for f in note.low_confidence_fields if f.endswith(".unvalidated")]
    assert unvalidated == []


def test_real_drug_name_untouched_by_generic_filter() -> None:
    from src.l4_extract import _build_note

    data = {"medications": [{"drug": "Paracetamol", "dose": "650 mg"}]}
    note = _build_note(data)
    assert note.medications[0].drug == "Paracetamol"


# ── Prompt-example leakage guard ──────────────────────────────────────────────


def test_system_prompt_has_no_leaked_example_values() -> None:
    """Rule 9's old few-shot lab values leaked verbatim into unrelated notes.

    qwen2.5:3b copied "Hb is 9.2", "raised cholesterol", "HbA1c 9.1",
    "Vitamin D low", "Total IGE 2107" from the prompt into diagnostic_results
    on 6 of 24 frozen eval samples (idx 121, 131, 44, 49, 24, 140) — fabricated
    lab results a physician could act on. Tripwire: the prompt must never
    contain these concrete content-value examples again.
    """
    from src.l4_extract import _SYSTEM_PROMPT

    leaked_examples = [
        "Hb is 9.2",
        "raised cholesterol",
        "HbA1c 9.1",
        "Vitamin D low",
        "Total IGE 2107",
    ]
    for value in leaked_examples:
        assert value not in _SYSTEM_PROMPT, (
            f"Leaked example value {value!r} reintroduced into _SYSTEM_PROMPT"
        )


def test_system_prompt_drug_fidelity_rule_has_no_concrete_drug_names() -> None:
    """Rule 12 (drug name script/spelling fidelity) must stay concrete-value-
    free, same tripwire spirit as the lab-value leakage guard above — a 3B
    model treats concrete values in instructions as content to reuse.
    """
    from src.l4_extract import _SYSTEM_PROMPT

    for value in ("naxdom", "नक्सडम", "azithral", "paracetamol"):
        assert value.lower() not in _SYSTEM_PROMPT.lower(), (
            f"Concrete drug name {value!r} leaked into _SYSTEM_PROMPT"
        )


# ── Drug-name source-fidelity restoration (Bug B backstop) ─────────────────
#
# qwen2.5:3b sometimes re-spells a Latin-script drug name spoken in the
# transcript into Devanagari, or otherwise distorts its spelling, instead of
# copying it verbatim — CDSCO validation then fails on an invented spelling
# rather than the drug actually said. The backstop is a source-fidelity
# restoration (substitute what was actually said), never a lexicon guess, so
# there is no wrong-drug substitution risk to test for here.


def test_devanagari_respelling_restored_to_transcript_latin_spelling() -> None:
    """Real case (outputs/20260711-161726-201975/): transcript said Latin
    'naxdom 500'; qwen2.5:3b wrote Devanagari 'नक्सडम 500' — an invented
    spelling. 'naxdom' is not in the CDSCO list under either spelling, so
    this asserts the restoration itself (source fidelity), not a validation
    outcome the drug list cannot provide either way.
    """
    from src.cdsco import validate_drug
    from src.l4_extract import _build_note

    transcript = "[UNKNOWN]: hidek ke liye naxdom 500 recommend kar deta hoon"
    data = {"medications": [{"drug": "नक्सडम 500", "dose": None}]}
    note = _build_note(data, transcript=transcript)
    assert note.medications[0].drug == "naxdom 500"
    assert validate_drug("naxdom 500") is False
    assert validate_drug("नक्सडम 500") is False


def test_restoration_preserves_dose_digits_glued_to_name() -> None:
    """Dose-deletion regression guard: a matched window that lacks the LLM
    string's digit tokens must not silently drop the dose (see LEARNINGS.md
    Phase B Hardening item 2 — a prior fuzzy-match substitution deleted a
    dose the same way).
    """
    from src.l4_extract import _restore_drug_spelling

    transcript = "[DOCTOR]: aapko azithral chahiye roz ek baar\n[PATIENT]: theek hai"
    assert _restore_drug_spelling("Azithral500", transcript) == "azithral 500"


def test_restoration_skips_when_nothing_similar_in_transcript() -> None:
    from src.l4_extract import _restore_drug_spelling

    transcript = "[DOCTOR]: aapko azithral chahiye roz ek baar\n[PATIENT]: theek hai"
    assert _restore_drug_spelling("Xyzqqrandomdrug", transcript) == "Xyzqqrandomdrug"


def test_restoration_fixes_latin_latin_distortion() -> None:
    from src.l4_extract import _restore_drug_spelling

    transcript = "[DOCTOR]: main azithral de raha hoon roz ek baar\n[PATIENT]: theek hai"
    assert _restore_drug_spelling("Azihral", transcript) == "azithral"


def test_restoration_noop_on_empty_transcript() -> None:
    from src.l4_extract import _restore_drug_spelling

    assert _restore_drug_spelling("नक्सडम 500", "") == "नक्सडम 500"


# ── Grounding guard (invented-drug-name backstop) ───────────────────────────
#
# Real incident (outputs/20260711-184756-36c330/): the doctor said "naxdom
# 500" but Whisper wrote a spelling variant ("नैक्स्टोम"/"नेक्स्टोम") absent
# from the L3.5 curated table. L3.5 could not normalize it, and qwen2.5:3b
# then INVENTED "nasal spray" as the drug name — a full fabrication with no
# source in the transcript, which _restore_drug_spelling cannot catch (it
# only restores fold-matches >= 0.80; an invention matches nothing). The
# grounding guard is the systemic fix: any drug name below fold-similarity
# 0.60 to every transcript window is converted to the numbered "unnamed
# medication N" form instead of rendering as a real drug.


def test_invented_drug_name_becomes_unnamed_with_ungrounded_flag() -> None:
    """A drug name with no plausible source in the transcript must never
    render as a real drug — it is converted to an unnamed row instead."""
    from src.l4_extract import _build_note

    transcript = "[UNKNOWN]: aur ek naxdom 500 khaiyega"
    data = {"medications": [{"drug": "budecort inhaler", "dose": None}]}
    note = _build_note(data, transcript=transcript)
    assert note.medications[0].drug == "unnamed medication 1"
    assert "medications.unnamed medication 1.ungrounded" in note.low_confidence_fields


def test_generic_check_precedence_over_grounding_for_nasal_spray() -> None:
    """Real case: the LLM invented "nasal spray" as a drug name. It is now
    also a whole-string generic term (fix: defense in depth), and the generic
    check runs BEFORE the grounding guard (unchanged precedence) — so the row
    becomes unnamed via the .unnamed flag, and the .ungrounded path is never
    reached for this specific string. Both paths converge on the same safe
    outcome (never a real drug name), but only one flag fires.
    """
    from src.l4_extract import _build_note

    transcript = "[UNKNOWN]: aur ek naxdom 500 khaiyega"
    data = {"medications": [{"drug": "nasal spray", "dose": None}]}
    note = _build_note(data, transcript=transcript)
    assert note.medications[0].drug == "unnamed medication 1"
    flags = note.low_confidence_fields
    assert any(f.endswith(".unnamed") for f in flags)
    assert not any(f.endswith(".ungrounded") for f in flags)


def test_grounded_drug_name_left_untouched() -> None:
    """A drug name that DOES appear in the transcript must pass through
    unchanged — the guard must not flag real, spoken drug names."""
    from src.l4_extract import _build_note

    transcript = "[UNKNOWN]: paracetamol lijiye roz do baar"
    data = {"medications": [{"drug": "paracetamol", "dose": None}]}
    note = _build_note(data, transcript=transcript)
    assert note.medications[0].drug == "paracetamol"
    assert not any(f.endswith(".ungrounded") for f in note.low_confidence_fields)


def test_grounding_guard_skipped_on_empty_transcript() -> None:
    """Empty transcript => skip the guard entirely (tests/eval call
    _build_note without a transcript)."""
    from src.l4_extract import _build_note

    data = {"medications": [{"drug": "budecort inhaler", "dose": None}]}
    note = _build_note(data)
    assert note.medications[0].drug == "budecort inhaler"
    assert not any(f.endswith(".ungrounded") for f in note.low_confidence_fields)


# ── Dose-provenance flag (cross-attribution backstop) ───────────────────────
#
# Real incident (same session): the doctor never stated a paracetamol dose
# ("पैरसेट मॉल दिन में दो बार" — no number), but the LLM attached "500 mg" to
# paracetamol anyway — the 500 belongs to naxdom, spoken elsewhere in the
# transcript. This flag is flag-only: it never modifies the extracted dose,
# it only tells the reviewing physician to check.


def test_dose_unattributed_fires_when_dose_far_from_drug_mention() -> None:
    from src.l4_extract import _build_note

    transcript = (
        "[UNKNOWN]: paracetamol lijiye subah shaam roz\n"
        "[UNKNOWN]: aur ek naxdom 500 khaiyega roz ek baar"
    )
    data = {
        "medications": [
            {"drug": "paracetamol", "dose": "500 mg", "frequency": "twice daily"}
        ]
    }
    note = _build_note(data, transcript=transcript)
    assert note.medications[0].dose == "500 mg"  # value is never altered
    assert "medications.paracetamol.dose_unattributed" in note.low_confidence_fields


def test_dose_unattributed_does_not_fire_when_dose_adjacent_to_drug() -> None:
    from src.l4_extract import _build_note

    transcript = "[UNKNOWN]: paracetamol 500 mg subah shaam roz"
    data = {
        "medications": [
            {"drug": "paracetamol", "dose": "500 mg", "frequency": "twice daily"}
        ]
    }
    note = _build_note(data, transcript=transcript)
    assert note.medications[0].dose == "500 mg"
    assert not any(
        f.endswith(".dose_unattributed") for f in note.low_confidence_fields
    )


def test_dose_unattributed_skipped_for_unnamed_medication_rows() -> None:
    """An ungrounded/generic row has no real drug mention to be "near" — the
    dose-provenance check must not fire for it."""
    from src.l4_extract import _build_note

    transcript = "[UNKNOWN]: aur ek naxdom 500 khaiyega"
    data = {"medications": [{"drug": "nasal spray", "dose": "500 mg"}]}
    note = _build_note(data, transcript=transcript)
    assert note.medications[0].drug == "unnamed medication 1"
    assert not any(
        f.endswith(".dose_unattributed") for f in note.low_confidence_fields
    )


# ── Condition guard (category-error backstop) ───────────────────────────────
#
# Real incident (outputs/20260712-124506-715247, 2026-07-12): patient said
# "मेरा BP (Hypertension) भी हाई है" — L3.5 correctly glossed "BP" with its
# clinical name "(Hypertension)", and qwen2.5:3b then extracted the
# parenthetical itself as a MEDICATION drug name. The grounding guard passes
# this string (it IS in the transcript) — grounding catches inventions, not
# category errors. The condition guard drops the row entirely: it is a
# clinical condition, not a drug, and the information is already captured
# elsewhere in the note (history/diagnosis).


def test_condition_gloss_dropped_from_medications() -> None:
    from src.l4_extract import _build_note

    transcript = "[UNKNOWN]: मेरा BP (Hypertension) भी हाई है"
    data = {"medications": [{"drug": "(Hypertension) भी", "dose": None}]}
    note = _build_note(data, transcript=transcript)
    assert note.medications == []
    assert "medications.(Hypertension) भी.condition_in_rx" in note.low_confidence_fields


def test_condition_raw_english_variant_dropped() -> None:
    from src.l4_extract import _build_note

    data = {"medications": [{"drug": "Hypertension", "dose": None}]}
    note = _build_note(data, transcript="[UNKNOWN]: Hypertension noted")
    assert note.medications == []
    assert "medications.Hypertension.condition_in_rx" in note.low_confidence_fields


def test_condition_guard_precedence_over_generic_check_unaffected() -> None:
    """Generic-term check still runs first — a generic term must still become
    an unnamed row, not fall through to the condition check."""
    from src.l4_extract import _build_note

    data = {"medications": [{"drug": "दवाई", "dose": None}]}
    note = _build_note(data)
    assert note.medications[0].drug == "unnamed medication 1"


def test_condition_guard_does_not_drop_real_drug() -> None:
    """A real, grounded drug name must pass through untouched."""
    from src.l4_extract import _build_note

    transcript = "[UNKNOWN]: paracetamol lijiye roz do baar"
    data = {"medications": [{"drug": "paracetamol", "dose": None}]}
    note = _build_note(data, transcript=transcript)
    assert note.medications[0].drug == "paracetamol"
    assert not any(f.endswith(".condition_in_rx") for f in note.low_confidence_fields)


def test_condition_guard_does_not_false_positive_on_substring_brand() -> None:
    """A real brand name containing a condition-ish substring ("cheston
    cold" contains "cold") must not be dropped — the guard requires an
    EXACT fold match (or an exact-matching parenthetical gloss), never a
    substring match, precisely to avoid this false positive."""
    from src.l4_extract import _is_condition_term

    assert _is_condition_term("cheston cold") is False


def test_no_drug_lexicon_entry_matches_condition_set() -> None:
    """Collision check (brief requirement): no canonical drug-lexicon entry
    must fold-match a condition term. If this ever fires, drug-lexicon
    membership takes precedence (see _is_condition_term) — the failing
    entry would need to be handled there, not by shrinking the condition set.
    """
    from src.l4_extract import _CONDITION_TERMS, _fold_drug
    from src.drug_lexicon import DRUG_LEXICON

    collisions = [
        d for d in DRUG_LEXICON if _fold_drug(d).replace(" ", "") in _CONDITION_TERMS
    ]
    assert collisions == [], f"Drug entries collide with conditions: {collisions}"


def test_drug_lexicon_precedence_wins_over_condition_set(monkeypatch) -> None:
    """Direct test of the precedence rule itself (no natural collision exists
    today — see test above): if a fold key were ever a member of BOTH sets,
    drug-lexicon membership must win and the row must NOT be dropped.
    """
    import src.l4_extract as l4

    monkeypatch.setattr(l4, "_DRUG_LEXICON_FOLDS", frozenset({"hypertension"}))
    assert l4._is_condition_term("Hypertension") is False
