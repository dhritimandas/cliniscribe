"""Note-field to transcript-turn provenance, and low-confidence flag routing.

Two responsibilities (contract: docs/frontend_contracts.md "Provenance" and
"Flag translation" sections):

1. `provenance_for_note` — for each populated leaf field of a `ClinicalNote`,
   find the transcript turn whose text most overlaps it, so the review UI can
   show "this came from here" on tap. Never fabricates a match.
2. `flags_by_path` — L4's `low_confidence_fields` are NAME-keyed strings
   (e.g. "medications.Dolo 650.dose_unknown"); the UI and PATCH edits work
   with INDEX-keyed field paths (e.g. "medications[0].dose"). This module is
   the single place that translates between the two.
"""

import re
import unicodedata

from src.types import ClinicalNote, Turn

# Coarse Devanagari→Latin fold, mirrored from eval/drug_bench.py's _FOLD_MAP —
# same phonetic mapping, reused here for cross-script token comparison rather
# than exact-window drug matching.
_FOLD_MAP = {
    "क": "k",
    "ख": "kh",
    "ग": "g",
    "घ": "gh",
    "च": "ch",
    "छ": "chh",
    "ज": "j",
    "झ": "jh",
    "ट": "t",
    "ठ": "th",
    "ड": "d",
    "ढ": "dh",
    "त": "t",
    "थ": "th",
    "द": "d",
    "ध": "dh",
    "न": "n",
    "प": "p",
    "फ": "f",
    "ब": "b",
    "भ": "bh",
    "म": "m",
    "य": "y",
    "र": "r",
    "ल": "l",
    "व": "v",
    "श": "sh",
    "ष": "sh",
    "स": "s",
    "ह": "h",
    "ज़": "z",
    "फ़": "f",
    "ा": "a",
    "ि": "i",
    "ी": "i",
    "ु": "u",
    "ू": "u",
    "े": "e",
    "ै": "ai",
    "ो": "o",
    "ौ": "au",
    "ं": "n",
    "अ": "a",
    "आ": "aa",
    "इ": "i",
    "ई": "i",
    "उ": "u",
    "ऊ": "u",
    "ए": "e",
    "ऐ": "ai",
    "ओ": "o",
    "औ": "au",
    "्": "",
}
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")
_MIN_TOKEN_LEN = 3  # per contract: "≥1 shared token ≥3 chars"


def _fold(text: str) -> str:
    """Coarse phonetic fold: Devanagari→Latin, lowercase, alnum only."""
    folded = "".join(_FOLD_MAP.get(ch, ch) for ch in unicodedata.normalize("NFC", text))
    return _NON_ALNUM_RE.sub("", folded.lower())


def _folded_tokens(text: str) -> set[str]:
    """Whitespace-split tokens, folded, keeping only those ≥3 chars post-fold.

    Whitespace splitting works for both Devanagari and Latin script (both are
    space-delimited); the length threshold is applied post-fold so it is
    comparable across scripts.
    """
    return {
        folded for word in text.split() if len(folded := _fold(word)) >= _MIN_TOKEN_LEN
    }


def _leaf_fields(note: ClinicalNote) -> list[tuple[str, str]]:
    """Enumerate (field_path, value) for every populated editable leaf field.

    Field paths mirror the contract's fixed set exactly: top-level scalars,
    symptoms[i].name, vitals[i].value, diagnosis[i].term,
    medications[i].{drug,dose,frequency,timing,duration},
    investigations[i], diagnostic_results[i].
    """
    fields: list[tuple[str, str]] = []
    for name in ("chief_complaint", "history", "examination", "advice", "follow_up"):
        value = getattr(note, name)
        if value:
            fields.append((name, value))
    for i, s in enumerate(note.symptoms):
        if s.name:
            fields.append((f"symptoms[{i}].name", s.name))
    for i, v in enumerate(note.vitals):
        if v.value:
            fields.append((f"vitals[{i}].value", v.value))
    for i, d in enumerate(note.diagnosis):
        if d.term:
            fields.append((f"diagnosis[{i}].term", d.term))
    for i, m in enumerate(note.medications):
        for sub in ("drug", "dose", "frequency", "timing", "duration"):
            value = getattr(m, sub)
            if value:
                fields.append((f"medications[{i}].{sub}", value))
    for i, inv in enumerate(note.investigations):
        if inv:
            fields.append((f"investigations[{i}]", inv))
    for i, res in enumerate(note.diagnostic_results):
        if res:
            fields.append((f"diagnostic_results[{i}]", res))
    return fields


def _best_turn(value: str, turns: list[Turn]) -> dict[str, object] | None:
    """Return the best-overlapping turn's provenance entry, or None."""
    value_tokens = _folded_tokens(value)
    if not value_tokens:
        return None
    best_index, best_overlap = -1, 0
    for i, turn in enumerate(turns):
        overlap = len(value_tokens & _folded_tokens(turn.text))
        if overlap > best_overlap:
            best_index, best_overlap = i, overlap
    if best_index < 0:
        return None
    turn = turns[best_index]
    return {
        "turn_index": best_index,
        "start": turn.start,
        "end": turn.end,
        "snippet": turn.text,
    }


def provenance_for_note(
    note: ClinicalNote, turns: list[Turn]
) -> dict[str, dict[str, object]]:
    """Map each populated leaf field path to its best-matching transcript turn.

    Args:
        note: The structured clinical note.
        turns: Speaker-attributed transcript turns for the same session.

    Returns:
        `{field_path: {turn_index, start, end, snippet}}`, one entry per field
        with a qualifying match (≥1 shared folded token ≥3 chars). Fields with
        no match are simply absent — the UI must never fabricate provenance.
    """
    result: dict[str, dict[str, object]] = {}
    for path, value in _leaf_fields(note):
        match = _best_turn(value, turns)
        if match is not None:
            result[path] = match
    return result


# ── Flag translation (NAME-keyed low_confidence_fields → INDEX-keyed paths) ──

# Reason suffix (from src/l4_extract.py's flag vocabulary) → medication leaf
# field it concerns.
_MEDICATION_REASON_SUBFIELD = {
    "dose_unknown": "dose",
    "unvalidated": "drug",
    "unnamed": "drug",
}


def _resolve_flag(flag: str, note: ClinicalNote) -> str | None:
    """Resolve one NAME-keyed flag to an INDEX-keyed field path, or None."""
    parts = flag.split(".")
    if len(parts) == 1:
        return parts[0]  # bare field name -> whole-field path

    namespace = parts[0]
    if namespace == "medications" and len(parts) >= 3:
        name, reason = ".".join(parts[1:-1]), parts[-1]
        idx = next((i for i, m in enumerate(note.medications) if m.drug == name), None)
        if idx is None:
            return None
        subfield = _MEDICATION_REASON_SUBFIELD.get(reason, "drug")
        return f"medications[{idx}].{subfield}"
    if namespace == "diagnosis" and len(parts) >= 3:
        term = ".".join(parts[1:-1])
        idx = next((i for i, d in enumerate(note.diagnosis) if d.term == term), None)
        return f"diagnosis[{idx}].term" if idx is not None else None
    if namespace == "symptoms" and len(parts) >= 2:
        name = ".".join(parts[1:])
        idx = next((i for i, s in enumerate(note.symptoms) if s.name == name), None)
        return f"symptoms[{idx}].name" if idx is not None else None
    if namespace == "vitals" and len(parts) >= 2:
        name = ".".join(parts[1:])
        idx = next((i for i, v in enumerate(note.vitals) if v.name == name), None)
        return f"vitals[{idx}].value" if idx is not None else None
    return None  # unrecognised dotted namespace: unresolvable


def flags_by_path(note: ClinicalNote) -> dict[str, str | list[str]]:
    """Translate NAME-keyed low_confidence_fields into INDEX-keyed paths.

    Args:
        note: The structured clinical note carrying `low_confidence_fields`.

    Returns:
        `{field_path: raw_flag}` for every flag resolvable to a current field
        or row, plus `_general: [raw_flag, ...]` for flags that reference a
        name/term no longer present in the note (still surfaced in the
        review footer, never silently dropped).
    """
    result: dict[str, str] = {}
    general: list[str] = []
    for flag in note.low_confidence_fields:
        path = _resolve_flag(flag, note)
        if path is None:
            general.append(flag)
        else:
            result[path] = flag
    output: dict[str, str | list[str]] = dict(result)
    if general:
        output["_general"] = general
    return output
