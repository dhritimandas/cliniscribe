"""FastAPI backend for the CliniScribe review frontend.

Implements every route in docs/frontend_contracts.md's API table. Offline
only: the sole non-localhost-Ollama network dependency is the local Ollama
server used for `/translate-advice`. Sessions are processed in a daemon
thread so `POST .../process` returns immediately; progress is mirrored into
`outputs/<session_id>/status.json` via `pipeline.run`'s `on_stage` callback.
"""

import dataclasses
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import config, pipeline
from src.l5_render import render
from src.types import ClinicalNote, Diagnosis, Medication, Symptom, Turn, Vital
from web import live_asr, verification
from web.provenance import flags_by_path, provenance_for_note
from web.translations import TRANSLATIONS

logger = logging.getLogger(__name__)

OUTPUTS_ROOT = pipeline.OUTPUTS_ROOT
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
_ALLOWED_AUDIO_EXTS = {"wav", "mp3"}
_SUPPORTED_LANGS = {"en", "hi", "mr"}
_IST = timezone(timedelta(hours=5, minutes=30))  # tier-2/3 India clinics only
_PROGRESS_WRITE_MIN_INTERVAL_S = 1.0  # throttle status.json progress writes

_status_lock = threading.Lock()

app = FastAPI(title="CliniScribe")


# ── request bodies ────────────────────────────────────────────────────────
class DoctorInfo(BaseModel):
    """Regulatory placeholder block shown on a signed prescription."""

    name: str
    reg_no: str
    clinic: str


class SignRequest(BaseModel):
    """Body of POST /api/sessions/{sid}/sign."""

    lang: str
    doctor: DoctorInfo


class Edit(BaseModel):
    """One field-path edit within a PATCH note request."""

    field: str
    old: Any
    new: Any


class PatchNoteRequest(BaseModel):
    """Body of PATCH /api/sessions/{sid}/note.

    `lang` is not in the contract's API table for this route, but the
    corrections.jsonl schema requires a `lang` per line; it is resolved here
    as an optional request-level field (default "en") — see the coordinator
    report for this ambiguity.
    """

    edits: list[Edit]
    lang: str = "en"


class TranslateAdviceRequest(BaseModel):
    """Body of POST /api/sessions/{sid}/translate-advice."""

    lang: str


class TranslateRequest(BaseModel):
    """Body of POST /api/sessions/{sid}/translate."""

    lang: str


# ── session/file helpers ──────────────────────────────────────────────────
def _session_dir(sid: str) -> str:
    """Return the session directory, or raise 404 if the session is unknown."""
    path = os.path.join(OUTPUTS_ROOT, sid)
    if not os.path.isdir(path):
        raise HTTPException(status_code=404, detail=f"Unknown session: {sid}")
    return path


def _status_path(sid: str) -> str:
    return os.path.join(OUTPUTS_ROOT, sid, "status.json")


def _note_path(sid: str) -> str:
    return os.path.join(OUTPUTS_ROOT, sid, "note.json")


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: Any) -> None:
    """Write JSON atomically (temp file + rename).

    status.json is polled every ~1s by the SPA while a daemon thread writes
    it from `on_stage` callbacks; a plain truncate-then-write left a window
    where a concurrent read observed a zero-byte file and raised
    `json.JSONDecodeError` (surfaced as a transient 500 on GET .../status).
    `os.replace` is atomic on the same filesystem, so readers only ever see
    the fully-written old or new content.
    """
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _note_from_dict(data: dict[str, Any]) -> ClinicalNote:
    """Reconstruct a ClinicalNote from note.json (written by dataclasses.asdict)."""
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


def _read_turns(sid: str) -> list[Turn]:
    path = os.path.join(OUTPUTS_ROOT, sid, "transcript.json")
    if not os.path.exists(path):
        return []
    return [Turn(**t) for t in _read_json(path)]


# ── POST /api/sessions ────────────────────────────────────────────────────
@app.post("/api/sessions", status_code=201)
async def create_session(audio: UploadFile = File(...)) -> dict[str, str]:
    """Create a session from an uploaded audio file; state starts at idle."""
    filename = audio.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _ALLOWED_AUDIO_EXTS:
        raise HTTPException(
            status_code=400, detail=f"Unsupported audio type: {ext or 'unknown'}"
        )

    sid = pipeline.new_session_id()
    session_dir = os.path.join(OUTPUTS_ROOT, sid)
    os.makedirs(session_dir, exist_ok=True)
    with open(os.path.join(session_dir, f"input.{ext}"), "wb") as f:
        f.write(await audio.read())
    _write_json(
        _status_path(sid),
        {"state": "idle", "stage": None, "stages_done": [], "error": None},
    )
    # A finished recording is uploaded here — the live-preview model (if it
    # was loaded during recording) is no longer needed; release it before
    # the production faster-whisper (L3) or Ollama (L4) models load, so the
    # load-one-release-one memory discipline holds at its peak.
    live_asr.release_model()
    return {"session_id": sid}


# ── POST /api/live/preview ────────────────────────────────────────────────
@app.post("/api/live/preview")
def live_preview(audio: UploadFile = File(...)) -> dict[str, str]:
    """UI-DISPLAY-ONLY transcription preview of the recorded-so-far audio.

    Clinical-safety contract (see web/live_asr.py): this text is never used
    by L4 and never lands in note.json. Not session-scoped — capture happens
    before a session exists. Debounced: an overlapping call while one is
    already in flight is rejected with 429 so the SPA's ~10s polling can
    never stack concurrent mlx-whisper decodes.

    A plain (non-async) route: mlx-whisper's decode is a blocking, CPU/GPU-
    bound call — declaring this `async def` and calling it directly would
    block the single event loop for the whole decode (9-15s), stalling every
    other request (including status polling for a session already
    processing). FastAPI runs plain `def` routes in its threadpool, so this
    call runs on a worker thread instead.
    """
    if not live_asr.available():
        raise HTTPException(status_code=503, detail="Live preview unavailable")
    if not live_asr.try_acquire():
        raise HTTPException(status_code=429, detail="Preview already in progress")
    try:
        text = live_asr.transcribe_preview(audio.file.read())
    finally:
        live_asr.release()
    return {"text": text}


# ── POST /api/sessions/{sid}/process ─────────────────────────────────────
def _make_progress_callbacks(sid: str):
    """Build the on_stage/on_progress pair pipeline.run uses to mirror L3
    transcription progress into status.json.

    Both callbacks share one `l3_start` timestamp (set by on_stage when
    "l3_asr" starts) so on_progress can compute `eta_seconds = elapsed_l3 *
    (1 - progress) / progress`. on_stage clears "progress"/"eta_seconds" from
    status.json when "l3_asr" ends, so the SPA's percent/ETA suffix
    disappears the moment the stage completes.
    """
    shared: dict[str, float | None] = {"l3_start": None, "last_write": 0.0}

    def on_stage(name: str, event: str) -> None:
        with _status_lock:
            status = _read_json(_status_path(sid))
            if event == "start":
                status["state"] = "processing"
                status["stage"] = name
                if name == "l3_asr":
                    shared["l3_start"] = time.monotonic()
            elif event == "end":
                done = status.get("stages_done", [])
                if name not in done:
                    done.append(name)
                status["stages_done"] = done
                if name == "l3_asr":
                    status.pop("progress", None)
                    status.pop("eta_seconds", None)
            _write_json(_status_path(sid), status)

    def on_progress(done_seconds: float, total_seconds: float) -> None:
        now = time.monotonic()
        if now - shared["last_write"] < _PROGRESS_WRITE_MIN_INTERVAL_S:
            return
        shared["last_write"] = now
        progress = done_seconds / total_seconds if total_seconds > 0 else 0.0
        eta_seconds = None
        if progress > 0 and shared["l3_start"] is not None:
            elapsed_l3 = now - shared["l3_start"]
            eta_seconds = round(elapsed_l3 * (1 - progress) / progress, 1)
        with _status_lock:
            status = _read_json(_status_path(sid))
            status["progress"] = round(progress, 4)
            status["eta_seconds"] = eta_seconds
            _write_json(_status_path(sid), status)

    return on_stage, on_progress


def _run_verification_thread(sid: str) -> None:
    """Run web.verification's targeted second-listen and mirror its result
    into status.json's "verification" key.

    Started (see _run_pipeline_thread) only AFTER pipeline.run() has already
    returned for this session — i.e. after L4 extraction has finished and the
    fast-path Whisper/mlx model has already been released (memory
    discipline: see web/verification.py's module docstring). Runs on its own
    daemon thread and never blocks review or signing.
    """
    with _status_lock:
        status = _read_json(_status_path(sid))
        status["verification"] = {"state": "running", "n_differing": 0, "coverage": None}
        _write_json(_status_path(sid), status)
    result = verification.run_verification(sid)  # never raises — see its docstring
    with _status_lock:
        status = _read_json(_status_path(sid))
        status["verification"] = {
            "state": result.get("state", "failed"),
            "n_differing": len(result.get("fields_differing", [])),
            "coverage": result.get("coverage"),
        }
        _write_json(_status_path(sid), status)


def _run_pipeline_thread(sid: str, in_path: str) -> None:
    on_stage, on_progress = _make_progress_callbacks(sid)
    asr_engine = "fast" if config.FAST_ASR_ENABLED else "accurate"
    try:
        pipeline.run(
            in_path,
            session_id=sid,
            on_stage=on_stage,
            on_progress=on_progress,
            asr_engine=asr_engine,
        )
        with _status_lock:
            status = _read_json(_status_path(sid))
            status.update(state="review", stage=None, error=None)
            _write_json(_status_path(sid), status)
        if asr_engine == "fast":
            # Background verification (web/verification.py) only ever makes
            # sense for the fast engine — the accurate engine's own L3 output
            # is already the slow, careful transcription this would recheck
            # against.
            threading.Thread(target=_run_verification_thread, args=(sid,), daemon=True).start()
    except Exception as exc:
        logger.exception("Session %s: pipeline failed", sid)
        with _status_lock:
            status = _read_json(_status_path(sid))
            status.update(state="error", error=str(exc))
            _write_json(_status_path(sid), status)


@app.post("/api/sessions/{sid}/process", status_code=202)
def process_session(sid: str) -> dict[str, str]:
    """Run the pipeline for a session in a daemon thread; returns immediately."""
    session_dir = _session_dir(sid)
    in_files = sorted(f for f in os.listdir(session_dir) if f.startswith("input."))
    if not in_files:
        raise HTTPException(
            status_code=404, detail=f"No input audio for session: {sid}"
        )
    in_path = os.path.join(session_dir, in_files[0])
    threading.Thread(
        target=_run_pipeline_thread, args=(sid, in_path), daemon=True
    ).start()
    return {"session_id": sid}


# ── GET /api/sessions/{sid}/status ───────────────────────────────────────
@app.get("/api/sessions/{sid}/status")
def get_status(sid: str) -> dict[str, Any]:
    """Return the current session state machine snapshot."""
    _session_dir(sid)
    with _status_lock:
        return _read_json(_status_path(sid))


# ── GET /api/sessions/{sid}/note ─────────────────────────────────────────
@app.get("/api/sessions/{sid}/note")
def get_note(sid: str) -> dict[str, Any]:
    """Return the note, transcript, per-field provenance, and resolved flags.

    `flags` is the INDEX-keyed dict from `flags_by_path` (what the SPA reads);
    `low_confidence_fields` rides along as the raw NAME-keyed list for
    fidelity with note.json (contract: docs/frontend_contracts.md GET note row).
    """
    _session_dir(sid)
    if not os.path.exists(_note_path(sid)):
        raise HTTPException(status_code=404, detail="Note not ready")
    note_data = _read_json(_note_path(sid))
    note = _note_from_dict(note_data)
    turns = _read_turns(sid)
    return {
        "note": note_data,
        "transcript": [dataclasses.asdict(t) for t in turns],
        "provenance": provenance_for_note(note, turns),
        "flags": flags_by_path(note),
        "low_confidence_fields": note.low_confidence_fields,
    }


# ── GET /api/sessions/{sid}/verification ─────────────────────────────────
@app.get("/api/sessions/{sid}/verification")
def get_verification(sid: str) -> dict[str, Any]:
    """Return the background verification result (web/verification.py) for a
    session — fields_differing, doctor_resolved, unverified_by_budget,
    coverage, and the safety-critical-span "checked transcript".

    404 before verification has ever started (no verification.json yet); the
    SPA only calls this once status.json's "verification" key first appears.
    """
    session_dir = _session_dir(sid)
    path = os.path.join(session_dir, "verification.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Verification not started")
    return _read_json(path)


# ── PATCH /api/sessions/{sid}/note ───────────────────────────────────────
_FIELD_PATH_RE = re.compile(
    r"^(?P<name>[a-z_]+)(?:\[(?P<idx>\d+)\])?(?:\.(?P<sub>[a-z_]+))?$"
)


def _apply_edit(note_data: dict[str, Any], field: str, new: Any) -> None:
    """Apply one dotted-with-index field-path edit in place; 400 if invalid."""
    match = _FIELD_PATH_RE.match(field)
    if not match or match.group("name") not in note_data:
        raise HTTPException(status_code=400, detail=f"Unknown field path: {field}")
    name, idx, sub = match.group("name"), match.group("idx"), match.group("sub")

    if idx is None:
        note_data[name] = new
        return
    rows = note_data[name]
    row_index = int(idx)
    if not isinstance(rows, list) or row_index >= len(rows):
        raise HTTPException(status_code=400, detail=f"Index out of range: {field}")
    if sub is None:
        rows[row_index] = new
    elif sub in rows[row_index]:
        rows[row_index][sub] = new
    else:
        raise HTTPException(status_code=400, detail=f"Unknown subfield: {field}")


@app.patch("/api/sessions/{sid}/note")
def patch_note(sid: str, body: PatchNoteRequest) -> dict[str, Any]:
    """Apply edits by field path; log each to corrections.jsonl; rewrite note.json."""
    _session_dir(sid)
    note_data = _read_json(_note_path(sid))
    corrections_path = os.path.join(OUTPUTS_ROOT, sid, "corrections.jsonl")
    with open(corrections_path, "a", encoding="utf-8") as f:
        for edit in body.edits:
            _apply_edit(note_data, edit.field, edit.new)
            line = {
                "ts": datetime.now(_IST).isoformat(timespec="seconds"),
                "field": edit.field,
                "old": edit.old,
                "new": edit.new,
                "lang": body.lang,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    _write_json(_note_path(sid), note_data)
    return note_data


# ── POST /api/sessions/{sid}/sign ────────────────────────────────────────
@app.post("/api/sessions/{sid}/sign")
def sign_session(sid: str, body: SignRequest) -> dict[str, str]:
    """Persist the signature block and regenerate the PDF without the DRAFT banner."""
    session_dir = _session_dir(sid)
    if body.lang not in _SUPPORTED_LANGS:
        raise HTTPException(
            status_code=400, detail=f"Unsupported language: {body.lang}"
        )
    note = _note_from_dict(_read_json(_note_path(sid)))
    doctor = body.doctor.model_dump()
    _write_json(
        os.path.join(session_dir, "signed.json"), {"lang": body.lang, "doctor": doctor}
    )
    render(
        note,
        out_path=os.path.join(session_dir, "draft_rx.pdf"),
        lang=body.lang,
        signed=True,
        doctor=doctor,
    )
    status = _read_json(_status_path(sid))
    status["state"] = "signed"
    _write_json(_status_path(sid), status)
    return {"pdf_url": f"/api/sessions/{sid}/pdf?lang={body.lang}"}


# ── GET /api/sessions/{sid}/pdf ───────────────────────────────────────────
@app.get("/api/sessions/{sid}/pdf")
def get_pdf(sid: str, lang: str = "en") -> FileResponse:
    """Render (fresh, reflecting the current note state) and return the PDF."""
    session_dir = _session_dir(sid)
    if lang not in _SUPPORTED_LANGS:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {lang}")
    if not os.path.exists(_note_path(sid)):
        raise HTTPException(status_code=404, detail="Note not ready")
    note = _note_from_dict(_read_json(_note_path(sid)))

    signed_path = os.path.join(session_dir, "signed.json")
    signed, doctor = False, None
    if os.path.exists(signed_path):
        signed_data = _read_json(signed_path)
        signed, doctor = True, signed_data["doctor"]

    pdf_path = os.path.join(session_dir, "draft_rx.pdf")
    render(note, out_path=pdf_path, lang=lang, signed=signed, doctor=doctor)
    return FileResponse(
        pdf_path, media_type="application/pdf", filename=os.path.basename(pdf_path)
    )


# ── POST /api/sessions/{sid}/translate ───────────────────────────────────
# Fields eligible for display-only machine translation. Medications
# (drug/dose/frequency/timing/duration) and vitals values are NEVER
# translated — dosage and numeric clinical safety (contract note).
_TRANSLATABLE_SCALAR_FIELDS = ("chief_complaint", "history", "examination", "advice", "follow_up")

_OLLAMA_LANG_NAMES = {"en": "English", "hi": "Hindi", "mr": "Marathi"}

# Script ranges used to decide whether a string is already in the requested
# target script, so target=en never burns an Ollama call translating text
# that is already Latin-script (identity translation) — same Devanagari
# range as src/l5_render.py; Arabic range per contract (covers Urdu, a
# legacy ASR misdetection some sessions carry in Latin-labelled fields).
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")


def _needs_translation_to_en(text: str) -> bool:
    """Whether `text` carries non-Latin script and so needs translating to en."""
    return bool(_DEVANAGARI_RE.search(text) or _ARABIC_RE.search(text))


def _translatable_note_values(note_data: dict[str, Any]) -> dict[str, str]:
    """Collect `{field_path: text}` for every note field eligible for translation."""
    values: dict[str, str] = {}
    for name in _TRANSLATABLE_SCALAR_FIELDS:
        v = note_data.get(name)
        if v:
            values[name] = v
    for i, s in enumerate(note_data.get("symptoms", [])):
        if s.get("name"):
            values[f"symptoms[{i}].name"] = s["name"]
    for i, d in enumerate(note_data.get("diagnosis", [])):
        if d.get("term"):
            values[f"diagnosis[{i}].term"] = d["term"]
    for i, inv in enumerate(note_data.get("investigations", [])):
        if inv:
            values[f"investigations[{i}]"] = inv
    for i, res in enumerate(note_data.get("diagnostic_results", [])):
        if res:
            values[f"diagnostic_results[{i}]"] = res
    return values


def _strip_json_fences(raw: str) -> str:
    """Remove markdown code fences models sometimes add despite a JSON instruction."""
    s = raw.strip()
    if s.startswith("```"):
        lines = s.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines)
    return s.strip()


def _translate_batch_via_ollama(texts: list[str], lang: str) -> list[str]:
    """Translate a list of strings in one batched Qwen (Ollama) call.

    Returns translations aligned index-for-index with `texts`. Numbers, units,
    and Latin-script drug names embedded in a sentence are instructed to
    survive verbatim — this function is never called with drug/dose/vitals
    fields in the first place (see `_translatable_note_values`).
    """
    if not texts:
        return []
    import ollama

    system = (
        f"Translate each string in the JSON array into {_OLLAMA_LANG_NAMES[lang]}. "
        "Preserve numbers, units, and any Latin-script drug names verbatim within "
        "each translated sentence. Output ONLY a JSON array of translated strings, "
        "same length and order as the input, no extra commentary."
    )
    response = ollama.chat(
        model=config.EXTRACT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(texts, ensure_ascii=False)},
        ],
        options={"temperature": 0},
        keep_alive=config.EXTRACT_KEEP_ALIVE,
    )
    content = (
        response.message.content
        if hasattr(response, "message")
        else response["message"]["content"]
    )
    translated = json.loads(_strip_json_fences(content))
    if not isinstance(translated, list) or len(translated) != len(texts):
        raise ValueError("Ollama translation batch response malformed")
    return [str(item) for item in translated]


def _translate_texts_robust(texts: list[str], lang: str) -> list[str]:
    """Translate `texts` into `lang`, aligned index-for-index; never raises.

    Tries one batched call first. If the batch call fails outright or comes
    back misaligned (wrong length — an occasional Qwen JSON-array slip), it
    falls back to translating each string individually; if an individual
    item's translation call also fails, that item's ORIGINAL text is kept
    (never blank, never a crash, never lets one bad item fail the group).
    """
    if not texts:
        return []
    try:
        translated = _translate_batch_via_ollama(texts, lang)
        if len(translated) == len(texts):
            return translated
        logger.warning(
            "Ollama batch translation length mismatch (%d texts, %d results); "
            "retrying item-by-item",
            len(texts),
            len(translated),
        )
    except Exception:
        logger.exception("Ollama batch translation failed; retrying item-by-item")

    results: list[str] = []
    for text in texts:
        try:
            item = _translate_batch_via_ollama([text], lang)
            results.append(item[0] if item else text)
        except Exception:
            logger.exception("Ollama per-item translation failed; keeping original")
            results.append(text)
    return results


def _translate_needed_indices(texts: list[str], lang: str) -> list[int]:
    """Indices of `texts` that actually need translating into `lang`.

    For lang="en", skip strings that carry no Devanagari/Arabic script (they
    are already effectively English/Latin — no point burning a Qwen call on
    an identity translation). For hi/mr, translate every non-empty string:
    the source could be English, Devanagari, or (per a legacy ASR
    misdetection some sessions carry) Arabic script, and script alone can't
    disambiguate Hindi from Marathi, so there is no cheap skip available.
    """
    if lang == "en":
        return [i for i, text in enumerate(texts) if _needs_translation_to_en(text)]
    return list(range(len(texts)))


def _translations_cache_path(sid: str, lang: str) -> str:
    return os.path.join(OUTPUTS_ROOT, sid, f"translations_{lang}.json")


@app.post("/api/sessions/{sid}/translate")
def translate_session(sid: str, body: TranslateRequest) -> dict[str, Any]:
    """Translate the note's free-text values and transcript turns for display.

    Target-language-absolute (contract fix): "en" is a real target, not a
    no-op — selecting English always shows English regardless of the note's
    source script, and hi/mr always show hi/mr. Display-only: note.json and
    transcript.json on disk always keep the source language; this only feeds
    the review screen's translated view. Results are cached per (session,
    lang) in `translations_<lang>.json` (including lang="en") and served from
    cache on repeat calls, so repeated en<->hi<->mr switching is idempotent.
    """
    _session_dir(sid)
    if body.lang not in _SUPPORTED_LANGS:
        raise HTTPException(
            status_code=400, detail=f"Unsupported language: {body.lang}"
        )

    cache_path = _translations_cache_path(sid, body.lang)
    if os.path.exists(cache_path):
        return _read_json(cache_path)

    note_data = _read_json(_note_path(sid))
    turns = _read_turns(sid)

    note_values = _translatable_note_values(note_data)
    paths = list(note_values)
    note_texts = [note_values[p] for p in paths]
    turn_texts = [turn.text for turn in turns]

    note_needs = _translate_needed_indices(note_texts, body.lang)
    translated_note_by_idx = dict(
        zip(
            note_needs,
            _translate_texts_robust([note_texts[i] for i in note_needs], body.lang),
        )
    )
    note_values_out = {paths[i]: text for i, text in translated_note_by_idx.items()}

    turn_needs = _translate_needed_indices(turn_texts, body.lang)
    translated_turn_by_idx = dict(
        zip(
            turn_needs,
            _translate_texts_robust([turn_texts[i] for i in turn_needs], body.lang),
        )
    )
    translated_turns = [
        translated_turn_by_idx.get(i, text) for i, text in enumerate(turn_texts)
    ]

    result = {"note_values": note_values_out, "transcript": translated_turns}
    _write_json(cache_path, result)
    return result


# ── POST /api/sessions/{sid}/translate-advice ────────────────────────────
@app.post("/api/sessions/{sid}/translate-advice")
def translate_advice(sid: str, body: TranslateAdviceRequest) -> dict[str, str | None]:
    """Translate note.advice on demand; thin wrapper over POST .../translate.

    Kept for the pre-existing contract (response shape `{advice}`). The
    per-field `advice_translations` cache in note.json is superseded by the
    batched `translations_<lang>.json` cache used by `/translate`.
    """
    _session_dir(sid)
    if body.lang not in _SUPPORTED_LANGS:
        raise HTTPException(
            status_code=400, detail=f"Unsupported language: {body.lang}"
        )
    note_data = _read_json(_note_path(sid))
    advice = note_data.get("advice")
    if not advice or body.lang == "en":
        return {"advice": advice}

    result = translate_session(sid, TranslateRequest(lang=body.lang))
    return {"advice": result["note_values"].get("advice", advice)}


# ── GET /api/translations ─────────────────────────────────────────────────
@app.get("/api/translations")
def get_translations() -> dict[str, dict[str, str]]:
    """Dump the static en/hi/mr UI label table."""
    return TRANSLATIONS


# Static SPA mount — registered last so it never shadows the /api/* routes
# above (Starlette matches routes in registration order). check_dir=False:
# web/static/index.html is owned and built by another agent and may not
# exist yet at import time.
app.mount(
    "/", StaticFiles(directory=STATIC_DIR, html=True, check_dir=False), name="static"
)
