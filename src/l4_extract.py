"""L4 — Clinical entity extraction via Qwen2.5-3B-Instruct (Ollama)."""

import json
import logging

from src.cdsco import validate_drug
from src.types import ClinicalNote, Diagnosis, Medication, Symptom, Turn, Vital

logger = logging.getLogger(__name__)

_MODEL = "qwen2.5:3b-instruct"

_SYSTEM_PROMPT = """\
You are a clinical documentation assistant. Extract structured medical information \
from the consultation transcript below.

Output ONLY valid JSON matching this exact schema — no extra text, no markdown:
{
  "chief_complaint": "string or null",
  "history": "string or null",
  "symptoms": [{"name": "string", "finding_status": "Present | Absent", \
"severity": "Mild | Moderate | Severe or null", "since": "string or null"}],
  "vitals": [{"name": "string", "value": "string with unit"}],
  "examination": "string or null",
  "diagnosis": [{"term": "string", "snomed_id": "string or null", \
"status": "Confirmed | Suspected | Ruled out or null"}],
  "medications": [{"drug": "string", "dose": "string or null", \
"frequency": "string or null", "timing": "string or null", \
"duration": "string or null"}],
  "investigations": ["string"],
  "diagnostic_results": ["string"],
  "advice": "string or null",
  "follow_up": "string or null",
  "low_confidence_fields": ["string"]
}

RULES — follow every rule exactly:
1. Set a field to null if it is NOT explicitly stated. Never infer or invent values.
2. DOSE must be null unless the doctor stated a specific dose in the transcript. \
Do not supply a standard or typical dose. null means unknown.
3. Add the field name to low_confidence_fields whenever you are uncertain about \
a value or the value is absent but clinically expected.
4. When a clinical term appears in parentheses after a lay term \
(e.g. "sugar (Type 2 Diabetes Mellitus)"), extract the clinical term \
in parentheses.
5. Extract only what is spoken. Do not add clinical knowledge not present \
in the transcript.
6. Use the language of the transcript for text fields. Do not translate.
7. SYMPTOMS are complaints the patient reports (pain, fever, nausea). \
finding_status is "Present" by default, "Absent" only when a symptom is \
explicitly denied (e.g. "no vomiting"). severity and since are null unless stated.
8. VITALS are measured signs with a number and unit (BP, pulse, SpO2, \
temperature, weight). Only include a vital when a measured value is spoken. \
Never invent a measurement.
9. INVESTIGATIONS are tests the doctor ORDERS for later (e.g. "get a CBC"). \
DIAGNOSTIC_RESULTS are results already available in the consultation \
(e.g. "Hb is 9.2"). Do not put an ordered test in diagnostic_results.
10. examination is free text describing physical-exam findings \
(e.g. "abdomen soft, mild tenderness"). null if no exam is described.\
"""


def _turns_to_text(turns: list[Turn]) -> str:
    return "\n".join(f"[{t.speaker_role}]: {t.text}" for t in turns)


def _strip_fences(raw: str) -> str:
    """Remove markdown code fences that models sometimes add despite format=json."""
    s = raw.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines)
    return s.strip()


def _parse(raw: str) -> dict:
    return json.loads(_strip_fences(raw))


def _build_note(data: dict) -> ClinicalNote:
    # Guard with `or []`: model may emit null for list fields (e.g. "symptoms": null).
    # data.get("symptoms", []) returns None when the key is present with value null,
    # and None is not iterable — the `or []` converts None → [].
    symptoms = [
        Symptom(
            name=s.get("name", "").strip(),
            finding_status=(s.get("finding_status") or "Present").strip() or "Present",
            severity=s.get("severity") or None,
            since=s.get("since") or None,
        )
        for s in (data.get("symptoms") or [])
        if s.get("name", "").strip()
    ]

    vitals = [
        Vital(name=v.get("name", "").strip(), value=str(v.get("value") or "").strip())
        for v in (data.get("vitals") or [])
        if v.get("name", "").strip() and str(v.get("value") or "").strip()
    ]

    diagnosis = [
        Diagnosis(
            term=d.get("term", "").strip(),
            snomed_id=d.get("snomed_id"),
            status=d.get("status") or None,
        )
        for d in (data.get("diagnosis") or [])
        if d.get("term", "").strip()
    ]

    medications: list[Medication] = []
    for m in (data.get("medications") or []):
        drug = (m.get("drug") or "").strip()
        if not drug:
            continue
        medications.append(
            Medication(
                drug=drug,
                dose=m.get("dose") or None,
                frequency=m.get("frequency") or None,
                timing=m.get("timing") or None,
                duration=m.get("duration") or None,
                validated=validate_drug(drug),
            )
        )

    low_conf: list[str] = list(data.get("low_confidence_fields") or [])

    for med in medications:
        if not med.validated:
            flag = f"medications.{med.drug}.unvalidated"
            if flag not in low_conf:
                low_conf.append(flag)
        if med.dose is None:
            flag = f"medications.{med.drug}.dose_unknown"
            if flag not in low_conf:
                low_conf.append(flag)

    return ClinicalNote(
        chief_complaint=data.get("chief_complaint") or None,
        history=data.get("history") or None,
        symptoms=symptoms,
        vitals=vitals,
        examination=data.get("examination") or None,
        diagnosis=diagnosis,
        medications=medications,
        investigations=list(data.get("investigations") or []),
        diagnostic_results=list(data.get("diagnostic_results") or []),
        advice=data.get("advice") or None,
        follow_up=data.get("follow_up") or None,
        low_confidence_fields=low_conf,
    )


def _empty_note() -> ClinicalNote:
    return ClinicalNote(
        chief_complaint=None,
        history=None,
        low_confidence_fields=[
            "chief_complaint",
            "history",
            "diagnosis",
            "medications",
        ],
    )


def extract(turns: list[Turn]) -> ClinicalNote:
    """Extract a structured ClinicalNote from normalized transcript turns.

    Calls qwen2.5:3b-instruct via Ollama at temperature=0 with a JSON format
    constraint. Retries once on JSON parse failure, then returns a minimal
    note with all fields flagged low-confidence.

    Dose is null whenever the doctor did not state it explicitly — never
    fabricated. CDSCO validation flags unrecognised drug names.

    Args:
        turns: Normalised, speaker-attributed turns from L3.5.

    Returns:
        ClinicalNote with validated medications and populated
        low_confidence_fields where values are absent or uncertain.
    """
    import ollama

    transcript_text = _turns_to_text(turns)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Transcript:\n{transcript_text}"},
    ]

    for attempt in range(1, 3):
        try:
            response = ollama.chat(
                model=_MODEL,
                messages=messages,
                format="json",
                options={"temperature": 0},
            )
            raw = (
                response.message.content
                if hasattr(response, "message")
                else response["message"]["content"]
            )
            data = _parse(raw)
            note = _build_note(data)
            logger.info(
                "L4: extracted note (attempt %d): %d symptoms, %d vitals, "
                "%d diagnosis, %d meds, %d low_conf",
                attempt,
                len(note.symptoms),
                len(note.vitals),
                len(note.diagnosis),
                len(note.medications),
                len(note.low_confidence_fields),
            )
            return note
        except json.JSONDecodeError as exc:
            logger.warning("L4: JSON parse failed on attempt %d: %s", attempt, exc)
        except Exception as exc:
            logger.error("L4: unexpected error on attempt %d: %s", attempt, exc)
            return _empty_note()

    logger.error("L4: both attempts failed — returning minimal note")
    return _empty_note()
