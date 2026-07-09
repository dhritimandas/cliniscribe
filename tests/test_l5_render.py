"""Tests for L5 rendering: human-readable flag sentences and PDF output."""

import os

from src.l5_render import _flag_sentence, render
from src.types import ClinicalNote, Medication


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
                drug="unnamed medication 1", dose=None, frequency="1-0-1",
                timing=None, duration=None, validated=False,
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
