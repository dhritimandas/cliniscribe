"""Tests for L5 rendering: human-readable flag sentences and PDF output."""

import os

import pytest
from pypdf import PdfReader

import src.l5_render as l5_render
from src.l5_render import _flag_sentence, render
from src.types import ClinicalNote, Medication, Symptom, Vital

# Stub for web/translations.LABELS (contract: docs/frontend_contracts.md
# "Ownership of shared modules"). That module is built in parallel by another
# agent; this stub only covers the keys src/l5_render.py actually consumes,
# and is injected via monkeypatch — it is never written to web/translations.py.
_STUB_LABELS = {
    "hi": {
        "chief_complaint": "मुख्य शिकायत",
        "history": "इतिहास",
        "symptoms": "लक्षण",
        "vitals": "महत्वपूर्ण संकेत",
        "examination": "जांच",
        "diagnosis": "निदान",
        "medications": "दवाइयाँ",
        "drug": "दवा",
        "dose": "खुराक",
        "frequency": "आवृत्ति",
        "timing": "समय",
        "duration": "अवधि",
        "investigations": "जांच के आदेश",
        "diagnostic_results": "जांच परिणाम",
        "advice": "सलाह",
        "follow_up": "अनुवर्ती",
        "verify": "हस्ताक्षर से पहले जांचें",
        "draft_banner": "मसौदा — नैदानिक उपयोग के लिए नहीं",
        "doctor_name": "डॉक्टर",
        "reg_no": "पंजीकरण संख्या",
        "clinic_address": "क्लिनिक",
        "signed": "इलेक्ट्रॉनिक रूप से हस्ताक्षरित",
    },
    "mr": {
        "chief_complaint": "मुख्य तक्रार",
        "medications": "औषधे",
        "vitals": "जीवनावश्यक चिन्हे",
        "symptoms": "लक्षणे",
    },
}


@pytest.fixture
def stub_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a fixed LABELS dict so tests don't depend on web/translations.py."""
    monkeypatch.setattr(l5_render, "LABELS", _STUB_LABELS)


def test_dose_unknown_flag_is_plain_language() -> None:
    s = _flag_sentence("medications.Sibelium.dose_unknown")
    assert s == "Dose for Sibelium could not be confirmed — verify with the patient"


def test_unvalidated_flag_is_plain_language() -> None:
    s = _flag_sentence("medications.Xanovir.unvalidated")
    assert "CDSCO" in s and "Xanovir" in s and "." not in s.replace("— ", "")


def test_unnamed_medication_flag_is_plain_language() -> None:
    s = _flag_sentence("medications.unnamed medication 1.unnamed")
    assert "without a clear name" in s


def test_hallucination_flag_is_plain_language() -> None:
    s = _flag_sentence("diagnosis.Pulmonary Embolism.no_transcript_overlap")
    assert "Pulmonary Embolism" in s and "confirm" in s.lower()


def test_drug_in_diagnosis_flag_is_plain_language() -> None:
    s = _flag_sentence("diagnosis.naxdom 500.drug_in_diagnosis")
    assert "naxdom 500" in s and "medication" in s.lower() and "." not in s.replace("— ", "")


def test_drug_in_investigations_flag_is_plain_language() -> None:
    s = _flag_sentence("investigations.naxdom 500.drug_in_investigations")
    assert "naxdom 500" in s and "medication" in s.lower() and "." not in s.replace("— ", "")


def test_bare_field_flag_gets_label() -> None:
    assert _flag_sentence("chief_complaint").startswith("Chief complaint")
    assert _flag_sentence("follow_up").startswith("Follow-up")


def test_unknown_flag_pattern_renders_readable_fallback() -> None:
    s = _flag_sentence("something.new_pattern.we_added_later")
    assert "_" not in s  # no raw dotted/underscored path reaches the PDF


def test_no_dotted_paths_reach_the_pdf(tmp_path) -> None:
    """End-to-end: footer text in the rendered PDF contains no dotted flags."""
    note = ClinicalNote(
        chief_complaint="fever",
        history=None,
        medications=[
            Medication(
                drug="unnamed medication 1",
                dose=None,
                frequency="1-0-1",
                timing=None,
                duration=None,
                validated=False,
            )
        ],
        low_confidence_fields=[
            "medications.unnamed medication 1.unnamed",
            "medications.unnamed medication 1.dose_unknown",
            "chief_complaint",
        ],
    )
    out = str(tmp_path / "rx.pdf")
    path = render(note, out_path=out)
    assert os.path.exists(path)
    # Extract text layer and assert no dotted flag survived.
    from pypdf import PdfReader

    text = "".join(page.extract_text() for page in PdfReader(path).pages)
    assert "dose_unknown" not in text
    assert "medications.unnamed" not in text
    assert "could not be confirmed" in text


def _simple_note(**overrides) -> ClinicalNote:
    defaults = dict(
        chief_complaint="fever",
        history=None,
        medications=[
            Medication(
                drug="Paracetamol",
                dose="650 mg",
                frequency="1-0-1",
                timing="after food",
                duration="5 days",
                validated=True,
            )
        ],
    )
    defaults.update(overrides)
    return ClinicalNote(**defaults)


def test_render_positional_out_path_is_still_backward_compatible(tmp_path) -> None:
    """render(note, out_path) — the pre-existing two-positional-arg call shape."""
    out = str(tmp_path / "rx.pdf")
    path = render(_simple_note(), out)
    assert os.path.exists(path)


def test_default_render_still_contains_draft_banner(tmp_path) -> None:
    out = str(tmp_path / "rx.pdf")
    render(_simple_note(), out_path=out)
    text = "".join(page.extract_text() for page in PdfReader(out).pages)
    assert "DRAFT" in text


def test_lang_hi_label_appears_in_pdf_text_layer(stub_labels, tmp_path) -> None:
    out = str(tmp_path / "rx_hi.pdf")
    render(_simple_note(), out_path=out, lang="hi")
    text = "".join(page.extract_text() for page in PdfReader(out).pages)
    assert "दवाइयाँ" in text  # Hindi for "Medications"


def test_signed_pdf_has_doctor_details_and_no_draft(tmp_path) -> None:
    out = str(tmp_path / "rx_signed.pdf")
    doctor = {"name": "Dr. Asha Rao", "reg_no": "MH12345", "clinic": "Rao Clinic"}
    render(_simple_note(), out_path=out, signed=True, doctor=doctor)
    text = "".join(page.extract_text() for page in PdfReader(out).pages)
    assert "Dr. Asha Rao" in text
    assert "MH12345" in text
    assert "DRAFT" not in text


def test_signed_without_doctor_raises() -> None:
    with pytest.raises(ValueError):
        render(_simple_note(), signed=True)


@pytest.mark.parametrize("lang", ["en", "hi", "mr"])
def test_devanagari_symptom_value_renders_without_error(
    stub_labels, tmp_path, lang: str
) -> None:
    note = _simple_note(
        symptoms=[Symptom(name="बुखार", severity="Mild", since="3 days")]
    )
    out = str(tmp_path / f"rx_{lang}.pdf")
    path = render(note, out_path=out, lang=lang)
    assert os.path.exists(path)


def test_arabic_script_advice_renders_without_error_and_in_text_layer(
    tmp_path,
) -> None:
    """Bug 4: an Arabic/Urdu-script value (e.g. a legacy ASR misdetection)
    must render as visible glyphs, not crash and not vanish silently.

    Reportlab has no bidi/shaping engine, so the extracted glyph order is not
    guaranteed to match the logical reading order (known, documented
    limitation) — this only asserts that genuine Arabic-range characters
    made it into the PDF's text layer, not an exact substring match.
    """
    note = _simple_note(advice="اس کو دو بار لیں")
    out = str(tmp_path / "rx_arabic.pdf")
    path = render(note, out_path=out)
    assert os.path.exists(path)
    text = "".join(page.extract_text() for page in PdfReader(path).pages)
    assert any("؀" <= ch <= "ۿ" for ch in text)


def test_devanagari_drug_name_renders_in_medications_table(tmp_path) -> None:
    """Table cells (not just Paragraph fields) must route through the same
    script-run wrapping — a raw string in a Table cell renders in a single
    Latin-only font, so a Devanagari drug name would be tofu."""
    note = _simple_note(
        medications=[
            Medication(
                drug="नेक्स डॉम 500",
                dose="500 mg",
                frequency="1-0-1",
                timing="after food",
                duration="5 days",
                validated=True,
            )
        ]
    )
    out = str(tmp_path / "rx_deva_drug.pdf")
    path = render(note, out_path=out)
    assert os.path.exists(path)
    text = "".join(page.extract_text() for page in PdfReader(path).pages)
    assert any("ऀ" <= ch <= "ॿ" for ch in text)


def test_arabic_drug_name_renders_in_medications_table(tmp_path) -> None:
    note = _simple_note(
        medications=[
            Medication(
                drug="باراسیٹامول",
                dose="500 mg",
                frequency="1-0-1",
                timing="after food",
                duration="5 days",
                validated=False,
            )
        ]
    )
    out = str(tmp_path / "rx_arabic_drug.pdf")
    path = render(note, out_path=out)
    assert os.path.exists(path)
    text = "".join(page.extract_text() for page in PdfReader(path).pages)
    assert any("؀" <= ch <= "ۿ" for ch in text)


def test_hindi_medications_header_label_renders_in_table(
    stub_labels, tmp_path
) -> None:
    """Table header rows are the same defect class as data rows: _label()
    returns Devanagari for lang="hi"/"mr", so a plain-string header cell
    would render as tofu just like a plain-string drug-name cell.

    "दवा" ("drug", the header) is a substring of "दवाइयाँ" ("medications",
    the section heading, already wrapped before this fix) — so a plain
    `"दवा" in text` check would pass even without the header-row fix. This
    counts occurrences instead: 1 from the heading alone (pre-fix) vs. 2
    once the header cell itself also renders (post-fix).
    """
    out = str(tmp_path / "rx_hi_header.pdf")
    path = render(_simple_note(), out_path=out, lang="hi")
    assert os.path.exists(path)
    text = "".join(page.extract_text() for page in PdfReader(path).pages)
    assert text.count("दवा") >= 2


def test_devanagari_vital_value_renders_in_vitals_table(tmp_path) -> None:
    note = _simple_note(vitals=[Vital(name="BP", value="१२०/८० mmHg")])
    out = str(tmp_path / "rx_deva_vital.pdf")
    path = render(note, out_path=out)
    assert os.path.exists(path)
    text = "".join(page.extract_text() for page in PdfReader(path).pages)
    assert any("ऀ" <= ch <= "ॿ" for ch in text)


def test_mixed_devanagari_arabic_latin_string_wraps_each_script_run() -> None:
    """Each script run gets its own font face; Latin text passes through
    unwrapped — mixed-script safety in all three directions at once."""
    arabic_word = "بخار"  # بخار
    devanagari_word = "बुखार"  # बुखार
    wrapped = l5_render._wrap_scripts(f"Take {arabic_word} {devanagari_word} now")
    assert wrapped.count("<font") == 2
    assert 'face="NotoNaskhArabic"' in wrapped
    assert 'face="NotoSansDevanagari"' in wrapped
    assert wrapped.startswith("Take ")
    assert wrapped.endswith(" now")


def test_real_web_translations_module_wires_up_correctly(tmp_path) -> None:
    """Integration proof: web.translations (owned by another agent, built in
    parallel) is picked up as-is, with no monkeypatch, for lang="hi"."""
    pytest.importorskip("web.translations")
    out = str(tmp_path / "rx_hi_real.pdf")
    render(_simple_note(), out_path=out, lang="hi")
    text = "".join(page.extract_text() for page in PdfReader(out).pages)
    assert "दवाइयाँ" in text  # web.translations.TRANSLATIONS["hi"]["medications"]


def test_lang_en_labels_are_unaffected_by_web_translations_module() -> None:
    """render(note, lang="en") must stay byte-identical to pre-multilingual
    behavior even once web.translations exists and defines its own "en"
    wording (which differs, e.g. "verify": "VERIFY")."""
    assert l5_render._label("verify", "en") == "Items to verify before signing"
    assert l5_render._label("draft_banner", "en") == (
        "DRAFT — NOT FOR CLINICAL USE — PHYSICIAN REVIEW REQUIRED"
    )
