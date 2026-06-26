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
class Medication:
    """A single medication entry from the consultation."""

    drug: str
    dose: str | None
    frequency: str | None
    duration: str | None
    validated: bool  # True only if matched against CDSCO drug list


@dataclass
class Diagnosis:
    """A clinical diagnosis with optional SNOMED CT identifier."""

    term: str
    snomed_id: str | None


@dataclass
class ClinicalNote:
    """Structured output of L4 entity extraction (spec §3.5)."""

    chief_complaint: str | None
    history: str | None
    diagnosis: list[Diagnosis] = field(default_factory=list)
    medications: list[Medication] = field(default_factory=list)
    investigations: list[str] = field(default_factory=list)
    advice: str | None = None
    follow_up: str | None = None
    low_confidence_fields: list[str] = field(default_factory=list)
