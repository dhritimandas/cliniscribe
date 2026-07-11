"""L5 — Draft prescription PDF rendering via reportlab."""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.types import ClinicalNote

logger = logging.getLogger(__name__)

_WARN_COLOR = colors.HexColor("#D97706")  # amber — low-confidence flag
_DRAFT_COLOR = colors.HexColor("#DC2626")  # red — draft watermark
_SIGNED_COLOR = colors.HexColor("#374151")  # grayscale — signed header/footer

# Vendored Unicode fonts for non-Latin script runs. Both of these Noto static
# builds have NO Latin glyphs, so each must only ever wrap its own script's
# substrings, never a whole mixed-script string — see _wrap_scripts.
_DEVANAGARI_FONT_NAME = "NotoSansDevanagari"
_ARABIC_FONT_NAME = "NotoNaskhArabic"
_FONTS_DIR = Path(__file__).resolve().parent.parent / "web" / "fonts"
_DEVANAGARI_FONT_PATH = _FONTS_DIR / f"{_DEVANAGARI_FONT_NAME}-Regular.ttf"
_ARABIC_FONT_PATH = _FONTS_DIR / f"{_ARABIC_FONT_NAME}-Regular.ttf"


def _register_font(name: str, path: Path) -> bool:
    """Register a vendored TTF with reportlab; return whether it is usable."""
    if not path.exists():
        logger.warning(
            "%s font not found at %s; matching text will render with missing glyphs",
            name,
            path,
        )
        return False
    try:
        pdfmetrics.registerFont(TTFont(name, str(path)))
        return True
    except Exception:
        logger.warning("Could not register %s font at %s", name, path)
        return False


_DEVANAGARI_FONT_AVAILABLE = _register_font(_DEVANAGARI_FONT_NAME, _DEVANAGARI_FONT_PATH)
_ARABIC_FONT_AVAILABLE = _register_font(_ARABIC_FONT_NAME, _ARABIC_FONT_PATH)

# Devanagari: U+0900-U+097F. Arabic script (covers Urdu, which is written in
# the Arabic script): U+0600-U+06FF, U+0750-U+077F, U+08A0-U+08FF (Arabic
# Extended-A), U+FB50-U+FDFF and U+FE70-U+FEFF (Arabic Presentation Forms).
_DEVANAGARI_CHARS = "ऀ-ॿ"
_ARABIC_CHARS = "؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿"
_SCRIPT_RUN_RE = re.compile(f"(?P<deva>[{_DEVANAGARI_CHARS}]+)|(?P<arab>[{_ARABIC_CHARS}]+)")

try:
    from web.translations import TRANSLATIONS as LABELS
except ImportError:
    LABELS: dict[str, dict[str, str]] = {}

# English fallback for section/UI labels — used when web.translations is
# absent or missing a key. Values here are byte-identical to this module's
# pre-existing hardcoded strings, so lang="en" behavior is unchanged.
_EN_FALLBACK: dict[str, str] = {
    "chief_complaint": "Chief Complaint",
    "history": "History",
    "symptoms": "Symptoms",
    "vitals": "Vitals",
    "examination": "Examination",
    "diagnosis": "Diagnosis",
    "medications": "Medications",
    "drug": "Drug",
    "dose": "Dose",
    "frequency": "Frequency",
    "timing": "Timing",
    "duration": "Duration",
    "investigations": "Investigations (Ordered)",
    "diagnostic_results": "Diagnostic Results (Available)",
    "advice": "Advice",
    "follow_up": "Follow-up",
    "verify": "Items to verify before signing",
    "draft_banner": "DRAFT — NOT FOR CLINICAL USE — PHYSICIAN REVIEW REQUIRED",
    "doctor_name": "Doctor",
    "reg_no": "Reg. No.",
    "signed": "Signed electronically",
}

# Dotted low-confidence flags → clinician sentences for the PDF footer.
# Phrasing is deliberately neutral ("could not be confirmed", not "not stated
# in audio") — a missing value may be an extraction miss of something that WAS
# spoken; the PDF must not assert facts about the recording.
_FLAG_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"^medications\.(?P<x>.+)\.dose_unknown$"),
        "Dose for {x} could not be confirmed — verify with the patient",
    ),
    (
        re.compile(r"^medications\.(?P<x>.+)\.unvalidated$"),
        "'{x}' is not in the CDSCO drug list — verify the drug name",
    ),
    (
        re.compile(r"^medications\.(?P<x>.+)\.unnamed$"),
        "A medication was mentioned without a clear name ({x}) — identify it",
    ),
    (
        re.compile(r"^diagnosis\.(?P<x>.+)\.no_transcript_overlap$"),
        "Diagnosis '{x}' lacks clear support in the conversation — confirm",
    ),
    (
        re.compile(r"^symptoms\.(?P<x>.+)$"),
        "Symptom '{x}' could not be confirmed — verify with the patient",
    ),
    (
        re.compile(r"^vitals\.(?P<x>.+)$"),
        "Vital sign '{x}' could not be confirmed — re-measure if needed",
    ),
]
_FIELD_LABELS: dict[str, str] = {
    "chief_complaint": "Chief complaint",
    "history": "History",
    "examination": "Examination findings",
    "diagnosis": "Diagnosis",
    "medications": "Medications",
    "investigations": "Ordered investigations",
    "diagnostic_results": "Diagnostic results",
    "advice": "Advice",
    "follow_up": "Follow-up",
}


def _flag_sentence(flag: str) -> str:
    """Translate a dotted low-confidence flag into a clinician sentence."""
    for pattern, template in _FLAG_PATTERNS:
        m = pattern.match(flag)
        if m:
            return template.format(x=m.group("x"))
    if flag in _FIELD_LABELS:
        return f"{_FIELD_LABELS[flag]} could not be determined — complete manually"
    # Unknown pattern: render something readable rather than a dotted path.
    return flag.replace("_", " ").replace(".", " — ") + " (verify)"


def _label(key: str, lang: str) -> str:
    """Look up a section label for `lang`, falling back to English.

    lang="en" always resolves to this module's own English strings
    (_EN_FALLBACK), never web.translations' "en" table — this is what
    guarantees render(note) stays byte-identical to pre-multilingual
    behavior regardless of what wording the sibling module uses for "en".
    """
    if lang != "en":
        translated = LABELS.get(lang, {}).get(key)
        if translated:
            return translated
    return _EN_FALLBACK.get(key, key)


def _wrap_scripts(text: str) -> str:
    """Wrap Devanagari- and Arabic-script runs in their vendored font faces.

    Everything else is left untouched (rendered in the paragraph's default
    Latin-capable font). This is mixed-script safe in both directions for
    both scripts: the default font can't render Devanagari or Arabic, and
    neither vendored font has Latin glyphs — so a run must never be rendered
    whole in one font. Returns `text` unchanged when there is nothing
    Devanagari/Arabic to wrap (the common lang="en" path), which keeps that
    path byte-for-byte identical to before this function existed.

    Reportlab does not shape or bidi-reorder text: an Arabic-script run
    renders as isolated glyph forms in logical (left-to-right in the source
    string) order, not the visually-correct joined, right-to-left result a
    real Arabic text-shaping engine would produce. This is legible but not
    typographically correct — a known limitation, not something this
    function (or reportlab, without a shaping engine) can fix.
    """
    if not text or not _SCRIPT_RUN_RE.search(text):
        return text
    out: list[str] = []
    last = 0
    for m in _SCRIPT_RUN_RE.finditer(text):
        if m.start() > last:
            out.append(text[last : m.start()])
        run = m.group()
        if m.group("deva") is not None and _DEVANAGARI_FONT_AVAILABLE:
            out.append(f'<font face="{_DEVANAGARI_FONT_NAME}">{_xml_escape(run)}</font>')
        elif m.group("arab") is not None and _ARABIC_FONT_AVAILABLE:
            out.append(f'<font face="{_ARABIC_FONT_NAME}">{_xml_escape(run)}</font>')
        else:
            out.append(run)
        last = m.end()
    if last < len(text):
        out.append(text[last:])
    return "".join(out)


def render(
    note: ClinicalNote,
    out_path: str | None = None,
    *,
    lang: str = "en",
    signed: bool = False,
    doctor: dict[str, str] | None = None,
) -> str:
    """Render a ClinicalNote as a draft or signed prescription PDF.

    Draft mode (signed=False, the default) is watermarked DRAFT.
    low_confidence_fields and unvalidated medications are visually flagged in
    amber. Physician review is mandatory before this document reaches a
    patient record. Signed mode replaces the DRAFT banner with a grayscale
    doctor/clinic header and a signature line — no red anywhere.

    Args:
        note: Structured clinical note from L4.
        out_path: Destination PDF path. The pipeline passes the session-scoped
            path (outputs/<session_id>/draft_rx.pdf). Defaults to
            outputs/draft_rx.pdf for direct/dev use (overwritten per run).
        lang: Section-label language — "en", "hi", or "mr". Field values are
            always printed as-is (drug names are never translated).
        signed: If True, renders the signed header/footer instead of the
            DRAFT banner. Requires `doctor`.
        doctor: Required when signed=True — {"name", "reg_no", "clinic"}.

    Returns:
        Path to the generated PDF.
    """
    if signed and (
        not doctor or not all(k in doctor for k in ("name", "reg_no", "clinic"))
    ):
        raise ValueError(
            "signed=True requires doctor={'name': ..., 'reg_no': ..., 'clinic': ...}"
        )

    path = out_path or os.path.join("outputs", "draft_rx.pdf")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, spaceAfter=4)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, spaceAfter=2)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9, spaceAfter=2)
    warn = ParagraphStyle(
        "Warn", parent=styles["Normal"], fontSize=9, textColor=_WARN_COLOR
    )
    draft_style = ParagraphStyle(
        "Draft",
        parent=styles["Normal"],
        fontSize=10,
        textColor=_DRAFT_COLOR,
        alignment=1,  # centre
        spaceAfter=4,
        borderPad=3,
    )
    signed_style = ParagraphStyle(
        "Signed",
        parent=styles["Normal"],
        fontSize=9,
        textColor=_SIGNED_COLOR,
        alignment=1,  # centre
        spaceAfter=2,
    )

    low_conf = set(note.low_confidence_fields or [])
    date_str = datetime.now().strftime("%Y-%m-%d")

    def field(label: str, value: str | None, field_key: str | None = None) -> list:
        if not value:
            return []
        s = warn if (field_key and field_key in low_conf) else body
        flag = " ⚑" if (field_key and field_key in low_conf) else ""
        label_r, value_r = _wrap_scripts(label), _wrap_scripts(value)
        return [Paragraph(f"<b>{label_r}:</b> {value_r}{flag}", s), Spacer(1, 1 * mm)]

    if signed:
        doctor_name_label = _wrap_scripts(_label("doctor_name", lang))
        reg_no_label = _wrap_scripts(_label("reg_no", lang))
        doctor_name = _wrap_scripts(doctor["name"])
        reg_no = _wrap_scripts(doctor["reg_no"])
        story = [
            Paragraph(_wrap_scripts(doctor["clinic"]), signed_style),
            Paragraph(
                f"{doctor_name_label}: {doctor_name}"
                f"    {reg_no_label}: {reg_no}"
                f"    {date_str}",
                signed_style,
            ),
            Spacer(1, 2 * mm),
            Paragraph("CliniScribe — Consultation Note", h1),
            Spacer(1, 3 * mm),
        ]
    else:
        story = [
            Paragraph(_wrap_scripts(_label("draft_banner", lang)), draft_style),
            Paragraph("CliniScribe — Draft Consultation Note", h1),
            Spacer(1, 3 * mm),
        ]

    story += field(
        _label("chief_complaint", lang), note.chief_complaint, "chief_complaint"
    )
    story += field(_label("history", lang), note.history, "history")

    if note.symptoms:
        story.append(Paragraph(_wrap_scripts(_label("symptoms", lang)), h2))
        for s in note.symptoms:
            parts = [s.name]
            if s.finding_status != "Present":
                parts.append(f"[{s.finding_status}]")
            if s.severity:
                parts.append(s.severity)
            if s.since:
                parts.append(f"since {s.since}")
            key = f"symptoms.{s.name}"
            p_style = warn if key in low_conf else body
            flag = " ⚑" if key in low_conf else ""
            line = _wrap_scripts(", ".join(parts))
            story.append(Paragraph("• " + line + flag, p_style))
        story.append(Spacer(1, 2 * mm))

    # Always render vitals section with standard rows for Height/Weight/BP.
    _VITAL_ORDER = ["Height", "Weight", "BP", "Temperature", "SpO2", "Pulse"]
    extracted_vitals = {v.name: v for v in (note.vitals or [])}
    # Merge: standard rows first (in order), then any additional extracted vitals.
    ordered_names = _VITAL_ORDER + [
        n for n in extracted_vitals if n not in _VITAL_ORDER
    ]
    story.append(Paragraph(_wrap_scripts(_label("vitals", lang)), h2))
    data = [["Parameter", "Value"]]
    for name in ordered_names:
        v = extracted_vitals.get(name)
        flag = " ⚑" if f"vitals.{name}" in low_conf else ""
        value = v.value if v else "—"
        data.append([name + flag, value])
    t = Table(data, colWidths=[60 * mm, 80 * mm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F9FAFB")],
                ),
            ]
        )
    )
    story += [t, Spacer(1, 2 * mm)]

    story += field(_label("examination", lang), note.examination, "examination")

    if note.diagnosis:
        story.append(Paragraph(_wrap_scripts(_label("diagnosis", lang)), h2))
        for d in note.diagnosis:
            status = f" [{d.status}]" if d.status else ""
            snomed = f" (SNOMED: {d.snomed_id})" if d.snomed_id else ""
            flag = " ⚑" if f"diagnosis.{d.term}" in low_conf else ""
            p_style = warn if f"diagnosis.{d.term}" in low_conf else body
            term = _wrap_scripts(d.term)
            story.append(Paragraph(f"• {term}{status}{snomed}{flag}", p_style))
        story.append(Spacer(1, 2 * mm))

    if note.medications:
        story.append(Paragraph(_wrap_scripts(_label("medications", lang)), h2))
        data = [
            [
                _label("drug", lang),
                _label("dose", lang),
                _label("frequency", lang),
                _label("timing", lang),
                _label("duration", lang),
                "Validated",
            ]
        ]
        for m in note.medications:
            val_flag = "✓" if m.validated else "⚑ No"
            data.append(
                [
                    m.drug,
                    m.dose or "—",
                    m.frequency or "—",
                    m.timing or "—",
                    m.duration or "—",
                    val_flag,
                ]
            )
        t = Table(
            data,
            colWidths=[38 * mm, 22 * mm, 28 * mm, 22 * mm, 22 * mm, 18 * mm],
        )
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#F9FAFB")],
                    ),
                    ("TEXTCOLOR", (-1, 1), (-1, -1), _WARN_COLOR),
                ]
            )
        )
        story += [t, Spacer(1, 2 * mm)]

    if note.investigations:
        story.append(Paragraph(_wrap_scripts(_label("investigations", lang)), h2))
        for inv in note.investigations:
            story.append(Paragraph(f"• {_wrap_scripts(inv)}", body))
        story.append(Spacer(1, 2 * mm))

    if note.diagnostic_results:
        story.append(
            Paragraph(_wrap_scripts(_label("diagnostic_results", lang)), h2)
        )
        for res in note.diagnostic_results:
            story.append(Paragraph(f"• {_wrap_scripts(res)}", body))
        story.append(Spacer(1, 2 * mm))

    story += field(_label("advice", lang), note.advice, "advice")
    story += field(_label("follow_up", lang), note.follow_up, "follow_up")

    if low_conf:
        story.append(Spacer(1, 4 * mm))
        story.append(
            Paragraph(f"<b>{_wrap_scripts(_label('verify', lang))} (⚑):</b>", warn)
        )
        for flag in sorted(low_conf):
            story.append(Paragraph(f"• {_flag_sentence(flag)}", warn))

    if signed:
        signed_label = _wrap_scripts(_label("signed", lang))
        doctor_name = _wrap_scripts(doctor["name"])
        story.append(Spacer(1, 4 * mm))
        story.append(
            Paragraph(
                f"{signed_label} — {doctor_name}, {date_str}",
                signed_style,
            )
        )

    doc.build(story)
    logger.info(
        "L5: rendered %s PDF (lang=%s) → %s",
        "signed" if signed else "draft",
        lang,
        path,
    )
    return path
