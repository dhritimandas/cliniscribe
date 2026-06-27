"""Shared data contracts for inter-stage communication."""

from dataclasses import dataclass, field


@dataclass
class Segment:
    """A diarized audio span attributed to a single speaker."""

    start: float
    end: float
    speaker: str


@dataclass
class Turn:
    """A transcribed, speaker-attributed utterance."""

    speaker_role: str  # "DOCTOR" | "PATIENT" | "UNKNOWN"
    text: str
    start: float
    end: float


@dataclass
class Symptom:
    """A patient symptom with optional clinical qualifiers."""

    name: str
    finding_status: str = "Present"  # "Present" | "Absent" | "Unknown"
    severity: str | None = None  # "Mild" | "Moderate" | "Severe" | None
    since: str | None = None  # free-text onset, e.g. "3 days"


@dataclass
class Vital:
    """A measured body vital sign.

    The value is kept as a single string with its unit (e.g. "120/80 mmHg",
    "38.2 °C") rather than a parsed number — clinic dictation rarely separates
    them cleanly, and the reviewing physician reads the raw measurement.
    """

    name: str  # "BP", "Temperature", "SpO2", "Pulse", "Weight", ...
    value: str  # "120/80 mmHg", "98 %", "72 bpm"


@dataclass
class Medication:
    """A single medication entry from the consultation."""

    drug: str
    dose: str | None
    frequency: str | None
    timing: str | None  # "before food" | "after food" | "at bedtime" | None
    duration: str | None
    validated: bool  # True only if matched against CDSCO drug list


@dataclass
class Diagnosis:
    """A clinical diagnosis with optional SNOMED CT identifier."""

    term: str
    snomed_id: str | None
    status: str | None = None  # "Confirmed" | "Suspected" | "Ruled out" | None


@dataclass
class ClinicalNote:
    """Structured output of L4 entity extraction (spec §3.5).

    Field set is aligned to the EkaCare clinical-note rubric targets: symptoms,
    vitals, examination findings, and diagnostic results are first-class so the
    note can hold what the consultation actually contains. Free-text history
    still absorbs past/family/social history, which Indian tier-2/3 clinic
    transcripts rarely structure on tape.
    """

    chief_complaint: str | None
    history: str | None
    symptoms: list[Symptom] = field(default_factory=list)
    vitals: list[Vital] = field(default_factory=list)
    examination: str | None = None  # free-text physical exam findings
    diagnosis: list[Diagnosis] = field(default_factory=list)
    medications: list[Medication] = field(default_factory=list)
    investigations: list[str] = field(default_factory=list)  # tests ordered
    diagnostic_results: list[str] = field(default_factory=list)  # results in hand
    advice: str | None = None
    follow_up: str | None = None
    low_confidence_fields: list[str] = field(default_factory=list)
