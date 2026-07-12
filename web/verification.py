"""Background verification — a targeted "second listen" on safety-critical
fields only, run on a daemon thread after a fast-engine review is ready.

Design constraint (hard, from the product decision to ship fast-engine ASR
by default): the check must finish within an average appointment (<=90s) —
see src.config.VERIFY_BUDGET_S. A full accurate re-transcription
(src.l3_asr.transcribe on CPU) costs 20-60 minutes for a multi-minute
consult, which is not a background check, it's a different appointment. So
this module does NOT re-transcribe the whole clip and does NOT re-run L4
extraction. Instead:

1. Locate the transcript mention of every safety-critical field (medication
   drug + dose, vitals, diagnosis) via the SAME provenance machinery the
   review UI already uses (web.provenance.provenance_for_note).
2. Pad each mention +/-1.5s (config.VERIFY_SPAN_PAD_S) and merge overlapping
   mentions, so two fields spoken in the same breath cost one re-listen.
3. Greedily keep spans in priority order (drug > dose > vital > diagnosis)
   up to config.VERIFY_MAX_WINDOWS windows of config.VERIFY_MAX_WINDOW_S
   seconds each — whatever doesn't fit is reported as "unverified_by_budget",
   never silently dropped.
4. Re-decode the selected span(s) ONCE each with config.VERIFY_MODEL (the
   bigger, decorrelated mlx model — mlx-community/whisper-large-v3-mlx, not
   the turbo model the primary fast pass used), reusing src.fast_asr's
   window-decode machinery (script/language guard, degeneration ladder)
   directly rather than reimplementing it — src/fast_asr.py is out of scope
   to edit for this task, but Python does not enforce leading-underscore
   privacy and this codebase already imports another module's private
   helpers this way (src/fast_asr.py imports src.l3_asr._contains_arabic_script
   / _doctor_score directly); the same precedent is used here.
5. Fold-compare (digit-exact for dose) each field's current value against its
   re-decoded span text. A mismatch becomes a fields_differing entry; never
   overwrites note.json.

Doctor-edited fields (present in corrections.jsonl) are excluded from
verification entirely and reported as "doctor_resolved" — never re-flagged.

Memory discipline: run_verification is only ever started (by
web/app.py._run_pipeline_thread) AFTER pipeline.run() has returned for this
session — i.e. after L4 extraction finished and the fast-path Whisper/mlx
model has already been released. While this module's own mlx decode runs,
Ollama's Qwen instance may still be resident (config.EXTRACT_KEEP_ALIVE) from
the L4 call that just finished, but this module never calls Ollama (no L4
re-extraction here), so it is not double-loaded — only one model (this
module's own VERIFY_MODEL, on the Metal GPU) plus the already-idle, already-
warm Ollama server are resident at once.
"""

import difflib
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import soundfile as sf

from src import config
from src.fast_asr import (
    Word,
    _decode_window_words_with_guards,
    _pack_segments_into_windows,
    _retry_ladder_windowed,
    looks_degenerate,
)
from src.pipeline import OUTPUTS_ROOT
from src.types import ClinicalNote, Diagnosis, Medication, Segment, Symptom, Turn, Vital
from web.provenance import provenance_for_note

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))  # matches web/app.py's corrections.jsonl convention

# Safety-critical field priority, most urgent first — a budget-tight session
# keeps drug names over doses, doses over vitals, vitals over diagnosis.
_DRUG_PRIORITY = 0
_DOSE_PRIORITY = 1
_VITAL_PRIORITY = 2
_DIAGNOSIS_PRIORITY = 3

_EMPTY_RESULT_FIELDS: dict[str, Any] = {
    "coverage": "full",
    "verified_seconds": 0.0,
    "fields_differing": [],
    "doctor_resolved": [],
    "unverified_by_budget": [],
    "transcript_accurate": [],
}


# ── small JSON/session helpers (reimplemented, not imported from web/app.py —
# web/app.py imports THIS module to kick off verification, so the reverse
# import would be circular) ─────────────────────────────────────────────────


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: Any) -> None:
    """Atomic write (temp file + rename) — mirrors web/app.py's _write_json:
    GET .../verification may read this file while a run is still in
    progress, so a partial write must never be observable."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _now_iso() -> str:
    return datetime.now(_IST).isoformat(timespec="seconds")


def _note_from_dict(data: dict[str, Any]) -> ClinicalNote:
    """Reconstruct a ClinicalNote from note.json — mirrors web/app.py's
    _note_from_dict (reimplemented locally for the same circular-import
    reason as the JSON helpers above)."""
    return ClinicalNote(
        chief_complaint=data.get("chief_complaint"),
        history=data.get("history"),
        symptoms=[Symptom(**s) for s in data.get("symptoms", [])],
        vitals=[Vital(**v) for v in data.get("vitals", [])],
        examination=data.get("examination"),
        diagnosis=[Diagnosis(**d) for d in data.get("diagnosis", [])],
        medications=[Medication(**m) for m in data.get("medications", [])],
        investigations=list(data.get("investigations", [])),
        diagnostic_results=list(data.get("diagnostic_results", [])),
        advice=data.get("advice"),
        follow_up=data.get("follow_up"),
        low_confidence_fields=list(data.get("low_confidence_fields", [])),
    )


def _corrected_fields(session_dir: str) -> set[str]:
    """Field paths the doctor has already corrected (corrections.jsonl)."""
    path = os.path.join(session_dir, "corrections.jsonl")
    if not os.path.exists(path):
        return set()
    fields: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                fields.add(json.loads(line)["field"])
    return fields


# ── span collection (safety-critical fields → padded, merged time spans) ────


@dataclass
class _SafetyField:
    path: str
    value: str
    start: float
    end: float
    priority: int


@dataclass
class _MergedSpan:
    start: float
    end: float
    fields: list[_SafetyField] = field(default_factory=list)

    @property
    def priority(self) -> int:
        return min(f.priority for f in self.fields)


def _append_if_located(
    fields: list[_SafetyField],
    path: str,
    value: str,
    provenance: dict[str, dict[str, object]],
    priority: int,
) -> None:
    prov = provenance.get(path)
    if prov is None:
        return  # no located transcript mention — nothing to re-listen to
    fields.append(
        _SafetyField(
            path=path, value=value, start=float(prov["start"]), end=float(prov["end"]), priority=priority
        )
    )


def _collect_safety_fields(
    note: ClinicalNote, provenance: dict[str, dict[str, object]]
) -> list[_SafetyField]:
    """Enumerate safety-critical leaf fields that have a located transcript
    mention (medication drug + dose, vitals, diagnosis — the fields a wrong
    value on a printed prescription can actually harm a patient over).

    Free-text fields (chief_complaint, history, examination, advice,
    follow_up) and fields with no per-item clinical stakes (investigations,
    diagnostic_results) are intentionally out of scope — verifying
    everything would blow the <=90s budget the same way a full
    re-transcription does.
    """
    fields: list[_SafetyField] = []
    for i, m in enumerate(note.medications):
        if m.drug:
            _append_if_located(fields, f"medications[{i}].drug", m.drug, provenance, _DRUG_PRIORITY)
        if m.dose:
            _append_if_located(fields, f"medications[{i}].dose", m.dose, provenance, _DOSE_PRIORITY)
    for i, v in enumerate(note.vitals):
        if v.value:
            _append_if_located(fields, f"vitals[{i}].value", v.value, provenance, _VITAL_PRIORITY)
    for i, d in enumerate(note.diagnosis):
        if d.term:
            _append_if_located(fields, f"diagnosis[{i}].term", d.term, provenance, _DIAGNOSIS_PRIORITY)
    return fields


def _merge_safety_fields(
    fields: list[_SafetyField], total_duration: float, pad_s: float
) -> list[_MergedSpan]:
    """Pad each field's transcript span by `pad_s` on both sides (clamped to
    the clip) and merge any that overlap, so two fields mentioned in the same
    breath (a drug name and its dose) cost one re-listen, not two."""
    padded = sorted(
        ((max(0.0, f.start - pad_s), min(total_duration, f.end + pad_s), f) for f in fields),
        key=lambda t: t[0],
    )
    merged: list[_MergedSpan] = []
    for start, end, f in padded:
        if merged and start <= merged[-1].end:
            merged[-1].end = max(merged[-1].end, end)
            merged[-1].fields.append(f)
        else:
            merged.append(_MergedSpan(start=start, end=end, fields=[f]))
    return merged


def _select_within_budget(
    spans_by_priority: list[_MergedSpan], max_windows: int, max_window_s: float
) -> tuple[list[_MergedSpan], list[_MergedSpan]]:
    """Greedily accept merged spans in priority order (drug > dose > vital >
    diagnosis — see _MergedSpan.priority; ties broken by start time), keeping
    only those whose chronological repacking still fits within `max_windows`
    windows of <= `max_window_s` each.

    Reuses src.fast_asr._pack_segments_into_windows for the packing itself
    (same "pack into <=N-second windows, prefer natural breaks" logic already
    proven there; it has no dependency on its inputs being diarized speaker
    segments rather than safety-critical field spans).

    Returns:
        (selected, skipped) — `skipped` becomes verification.json's
        "unverified_by_budget" list.
    """
    selected: list[_MergedSpan] = []
    skipped: list[_MergedSpan] = []
    for span in spans_by_priority:
        candidate = sorted(selected + [span], key=lambda s: s.start)
        pseudo_segments = [Segment(start=s.start, end=s.end, speaker="") for s in candidate]
        windows = _pack_segments_into_windows(
            pseudo_segments, max_window_s=max_window_s, min_break_gap_s=config.WINDOW_MIN_BREAK_GAP_S
        )
        if len(windows) <= max_windows:
            selected = candidate
        else:
            skipped.append(span)
    return selected, skipped


# ── fold-comparison (drug/vital/diagnosis fold-matched, dose digit-exact) ────
# Mirrors src/l4_extract.py's _fold_drug/_DRUG_FOLD_MAP exactly (reimplemented
# here, not imported — src/l4_extract.py is out of scope to edit for this
# task, and this codebase's established convention is to reimplement small
# fold helpers per module rather than share them across boundaries; see that
# module's own docstring: "Adapted from eval/drug_bench.py's _fold
# (reimplemented here, not imported...)"). The "क्स"/"क्श" -> "x" digraph fix
# is a no-op on text that doesn't contain it, so it is safe to reuse for
# vital/diagnosis comparisons too rather than keeping a drug-only variant.
_FOLD_DIGRAPHS: tuple[tuple[str, str], ...] = (("क्स", "x"), ("क्श", "x"))
_FOLD_MAP: dict[str, str] = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "च": "ch", "छ": "chh",
    "ज": "j", "झ": "jh", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p",
    "फ": "f", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r",
    "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "ज़": "z", "फ़": "f", "ा": "a", "ि": "i", "ी": "i", "ु": "u",
    "ू": "u", "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ं": "n",
    "अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "्": "",
}
_FOLD_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]")
_FOLD_WINDOW_SIZES: tuple[int, ...] = (1, 2, 3, 4)
_FOLD_MATCH_MIN_SIMILARITY = 0.80  # same threshold as l4_extract.py's proven drug-fidelity restoration
_DIGIT_RE = re.compile(r"\d+")

# src/l3_5_normalize.py's concept-gloss pass rewrites a recognized lay term as
# "<lay term> (<concept term>)" (_gloss_turn), and L4 extraction sometimes
# copies that whole glossed string into diagnosis.term (real case hit during
# E2E verification: "सरदर्द (मिग्रेन)" for a spoken "सरदर्द"/headache). The
# "(<concept term>)" half is L3.5's OWN annotation, never literally spoken —
# comparing it against a raw re-decode (this module never re-runs L3.5)
# would flag nearly every glossed diagnosis as "different" regardless of
# whether the underlying value is actually right. Stripped before folding;
# a no-op on values with no trailing parenthetical.
_GLOSS_SUFFIX_RE = re.compile(r"\s*\([^()]*\)\s*$")


def _strip_gloss_suffix(value: str) -> str:
    return _GLOSS_SUFFIX_RE.sub("", value).strip()


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for digraph, latin in _FOLD_DIGRAPHS:
        text = text.replace(digraph, latin)
    folded = "".join(_FOLD_MAP.get(ch, ch) for ch in text)
    return _FOLD_NON_ALNUM_RE.sub("", folded.lower())


def _best_fold_ratio(value: str, span_text: str) -> float:
    """Best fold-similarity of `value` (gloss-suffix stripped, see
    _strip_gloss_suffix) against any contiguous 1-4 word window of
    `span_text` — `span_text` is the whole re-decoded span, which usually
    carries a few extra words of context around the field's own mention."""
    words = span_text.split()
    if not words:
        return 0.0
    value_fold = _fold(_strip_gloss_suffix(value))
    best = 0.0
    for n in _FOLD_WINDOW_SIZES:
        for i in range(len(words) - n + 1):
            ratio = difflib.SequenceMatcher(None, value_fold, _fold(" ".join(words[i : i + n]))).ratio()
            best = max(best, ratio)
    return best


def _dose_matches(dose: str, span_text: str) -> bool:
    """Dose fields compare digit-exact: every digit group in `dose` must
    appear among `span_text`'s own digit groups. Falls back to fold-
    similarity only when `dose` has no digits at all (e.g. "one tablet")."""
    dose_digits = _DIGIT_RE.findall(dose)
    if not dose_digits:
        return _best_fold_ratio(dose, span_text) >= _FOLD_MATCH_MIN_SIMILARITY
    span_digits = set(_DIGIT_RE.findall(span_text))
    return all(d in span_digits for d in dose_digits)


def _field_matches(safety_field: _SafetyField, span_text: str) -> bool:
    if safety_field.priority == _DOSE_PRIORITY:
        return _dose_matches(safety_field.value, span_text)
    return _best_fold_ratio(safety_field.value, span_text) >= _FOLD_MATCH_MIN_SIMILARITY


def _words_in_span(words: list[Word], start: float, end: float) -> str:
    """Join every word (absolute clip-time) whose midpoint falls in [start, end)."""
    return " ".join(text for w_start, w_end, text in words if start <= (w_start + w_end) / 2 < end)


# ── orchestration ────────────────────────────────────────────────────────────


def _run_verification_inner(session_dir: str) -> dict[str, Any]:
    note = _note_from_dict(_read_json(os.path.join(session_dir, "note.json")))
    transcript_path = os.path.join(session_dir, "transcript.json")
    turns = (
        [Turn(**t) for t in _read_json(transcript_path)] if os.path.exists(transcript_path) else []
    )
    provenance = provenance_for_note(note, turns)

    corrected = _corrected_fields(session_dir)
    candidates = _collect_safety_fields(note, provenance)
    candidate_paths = {f.path for f in candidates}
    doctor_resolved = sorted(candidate_paths & corrected)
    fields = [f for f in candidates if f.path not in corrected]
    remaining_paths = candidate_paths - corrected

    if not fields:
        return {**_EMPTY_RESULT_FIELDS, "doctor_resolved": doctor_resolved}

    wav_path = os.path.join(session_dir, "input_16k.wav")
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    total_duration = len(audio) / sr

    merged = _merge_safety_fields(fields, total_duration, config.VERIFY_SPAN_PAD_S)
    ordered = sorted(merged, key=lambda m: (m.priority, m.start))
    selected, skipped = _select_within_budget(ordered, config.VERIFY_MAX_WINDOWS, config.VERIFY_MAX_WINDOW_S)

    window_groups = (
        _pack_segments_into_windows(
            sorted((Segment(start=m.start, end=m.end, speaker="") for m in selected), key=lambda s: s.start),
            max_window_s=config.VERIFY_MAX_WINDOW_S,
            min_break_gap_s=config.WINDOW_MIN_BREAK_GAP_S,
        )
        if selected
        else []
    )

    verified_seconds = 0.0
    transcript_accurate: list[dict[str, Any]] = []
    span_text_by_bounds: dict[tuple[float, float], str] = {}

    for group in window_groups:
        window_start, window_end = group[0].start, group[-1].end
        text, words = _decode_window_words_with_guards(
            audio, sr, window_start, window_end, config.VERIFY_MODEL, config.FAST_ASR_DECODE_KWARGS
        )
        if looks_degenerate(text, window_end - window_start):
            text, words, step = _retry_ladder_windowed(
                audio,
                sr,
                window_start,
                window_end,
                total_duration,
                initial_text=text,
                initial_words=words,
            )
            logger.warning(
                "verification ladder fired at [%.2f, %.2f]s — resolved by %s", window_start, window_end, step
            )
        verified_seconds += window_end - window_start
        transcript_accurate.append({"start": window_start, "end": window_end, "text": text})
        for seg in group:
            span_text_by_bounds[(seg.start, seg.end)] = _words_in_span(words, seg.start, seg.end)

    fields_differing: list[dict[str, Any]] = []
    verified_paths: set[str] = set()
    for merged_span in selected:
        span_text = span_text_by_bounds.get((merged_span.start, merged_span.end), "")
        for f in merged_span.fields:
            verified_paths.add(f.path)
            if not _field_matches(f, span_text):
                fields_differing.append({"field": f.path, "fast_value": f.value, "accurate_value": span_text})

    unverified = sorted(remaining_paths - verified_paths)
    return {
        "coverage": "partial" if unverified else "full",
        "verified_seconds": round(verified_seconds, 2),
        "fields_differing": fields_differing,
        "doctor_resolved": doctor_resolved,
        "unverified_by_budget": unverified,
        "transcript_accurate": transcript_accurate,
    }


def run_verification(sid: str) -> dict[str, Any]:
    """Run the targeted second-listen for session `sid` and persist the result.

    Never raises — any failure is caught, logged, and reported as
    verification.json's `"state": "failed"` (with `"error"` set) so the
    caller's background thread can mirror it into status.json without its
    own try/except. See the module docstring for the full design and the
    memory-discipline note on when this must be called.

    Args:
        sid: Session ID; outputs/<sid>/ must already contain note.json and
            transcript.json (written by a completed pipeline.run() call).

    Returns:
        The same dict written to outputs/<sid>/verification.json.
    """
    session_dir = os.path.join(OUTPUTS_ROOT, sid)
    verification_path = os.path.join(session_dir, "verification.json")
    started = _now_iso()
    _write_json(
        verification_path,
        {"state": "running", "started": started, "finished": None, "error": None, **_EMPTY_RESULT_FIELDS},
    )
    try:
        fields_result = _run_verification_inner(session_dir)
        result = {"state": "done", "started": started, "finished": _now_iso(), "error": None, **fields_result}
    except Exception as exc:
        logger.exception("Session %s: background verification failed", sid)
        result = {
            "state": "failed",
            "started": started,
            "finished": _now_iso(),
            "error": str(exc),
            **_EMPTY_RESULT_FIELDS,
        }
    _write_json(verification_path, result)
    return result
