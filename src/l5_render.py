"""L5 — Draft prescription PDF rendering via reportlab."""

import logging
import os
import tempfile

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.types import ClinicalNote

logger = logging.getLogger(__name__)

_WARN_COLOR = colors.HexColor("#D97706")   # amber — low-confidence flag
_DRAFT_COLOR = colors.HexColor("#DC2626")  # red — draft watermark


def render(note: ClinicalNote) -> str:
    """Render a ClinicalNote as a draft prescription PDF.

    Output is watermarked DRAFT. low_confidence_fields and unvalidated
    medications are visually flagged in amber. Physician review is mandatory
    before this document reaches a patient record.

    Args:
        note: Structured clinical note from L4.

    Returns:
        Absolute path to the generated PDF in outputs/.
    """
    os.makedirs("outputs", exist_ok=True)
    fd, path = tempfile.mkstemp(suffix="_draft_rx.pdf", dir="outputs")
    os.close(fd)

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

    low_conf = set(note.low_confidence_fields or [])

    def field(label: str, value: str | None, field_key: str | None = None) -> list:
        if not value:
            return []
        s = warn if (field_key and field_key in low_conf) else body
        flag = " ⚑" if (field_key and field_key in low_conf) else ""
        return [Paragraph(f"<b>{label}:</b> {value}{flag}", s), Spacer(1, 1 * mm)]

    story = [
        Paragraph("DRAFT — NOT FOR CLINICAL USE — PHYSICIAN REVIEW REQUIRED", draft_style),
        Paragraph("CliniScribe — Draft Consultation Note", h1),
        Spacer(1, 3 * mm),
    ]

    story += field("Chief Complaint", note.chief_complaint, "chief_complaint")
    story += field("History", note.history, "history")

    if note.symptoms:
        story.append(Paragraph("Symptoms", h2))
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
            story.append(Paragraph("• " + ", ".join(parts) + flag, p_style))
        story.append(Spacer(1, 2 * mm))

    # Always render vitals section with standard rows for Height/Weight/BP.
    _VITAL_ORDER = ["Height", "Weight", "BP", "Temperature", "SpO2", "Pulse"]
    extracted_vitals = {v.name: v for v in (note.vitals or [])}
    # Merge: standard rows first (in order), then any additional extracted vitals.
    ordered_names = _VITAL_ORDER + [
        n for n in extracted_vitals if n not in _VITAL_ORDER
    ]
    story.append(Paragraph("Vitals", h2))
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
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
            ]
        )
    )
    story += [t, Spacer(1, 2 * mm)]

    story += field("Examination", note.examination, "examination")

    if note.diagnosis:
        story.append(Paragraph("Diagnosis", h2))
        for d in note.diagnosis:
            status = f" [{d.status}]" if d.status else ""
            snomed = f" (SNOMED: {d.snomed_id})" if d.snomed_id else ""
            flag = " ⚑" if f"diagnosis.{d.term}" in low_conf else ""
            p_style = warn if f"diagnosis.{d.term}" in low_conf else body
            story.append(Paragraph(f"• {d.term}{status}{snomed}{flag}", p_style))
        story.append(Spacer(1, 2 * mm))

    if note.medications:
        story.append(Paragraph("Medications", h2))
        data = [["Drug", "Dose", "Frequency", "Timing", "Duration", "Validated"]]
        for m in note.medications:
            val_flag = "✓" if m.validated else "⚑ No"
            data.append([
                m.drug,
                m.dose or "—",
                m.frequency or "—",
                m.timing or "—",
                m.duration or "—",
                val_flag,
            ])
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
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
                    ("TEXTCOLOR", (-1, 1), (-1, -1), _WARN_COLOR),
                ]
            )
        )
        story += [t, Spacer(1, 2 * mm)]

    if note.investigations:
        story.append(Paragraph("Investigations (Ordered)", h2))
        for inv in note.investigations:
            story.append(Paragraph(f"• {inv}", body))
        story.append(Spacer(1, 2 * mm))

    if note.diagnostic_results:
        story.append(Paragraph("Diagnostic Results (Available)", h2))
        for res in note.diagnostic_results:
            story.append(Paragraph(f"• {res}", body))
        story.append(Spacer(1, 2 * mm))

    story += field("Advice", note.advice, "advice")
    story += field("Follow-up", note.follow_up, "follow_up")

    if low_conf:
        story.append(Spacer(1, 4 * mm))
        story.append(
            Paragraph(
                f"<b>Low-confidence fields (⚑ — physician must verify):</b> "
                + ", ".join(sorted(low_conf)),
                warn,
            )
        )

    doc.build(story)
    logger.info("L5: rendered draft PDF → %s", path)
    return path
