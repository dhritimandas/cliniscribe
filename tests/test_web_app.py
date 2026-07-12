"""Tests for web/app.py — the review-frontend FastAPI backend.

pipeline.run and web.app.render are stubbed throughout: no ASR/diarization/
LLM models are loaded, and no real PDF is rendered.
"""

import json
import os
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import src.pipeline as pipeline
import web.app as app_module
import web.verification as verification_module
from src.types import ClinicalNote, Diagnosis, Medication, Vital


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient with cwd isolated to tmp_path so outputs/ never touches the repo.

    FAST_ASR_ENABLED is forced False here so the pre-existing status-machine
    tests (which stub pipeline.run but never write note.json/transcript.json)
    don't also kick off a real background verification thread — the tests
    that specifically exercise fast-engine + verification wiring re-enable it.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app_module.config, "FAST_ASR_ENABLED", False)
    return TestClient(app_module.app)


def _create_session(client: TestClient) -> str:
    response = client.post(
        "/api/sessions",
        files={"audio": ("consult.wav", b"RIFF-fake-wav-bytes", "audio/wav")},
    )
    assert response.status_code == 201
    return response.json()["session_id"]


def _write_note(sid: str, note: ClinicalNote) -> None:
    import dataclasses

    path = os.path.join("outputs", sid, "note.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(note), f)


def _write_transcript(sid: str, turns: list[dict]) -> None:
    path = os.path.join("outputs", sid, "transcript.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(turns, f)


# ── _write_json atomicity ──────────────────────────────────────────────────


def test_write_json_is_atomic_and_leaves_no_tmp_file(tmp_path) -> None:
    """Regression test: status.json is polled every ~1s while a daemon thread
    writes it; a non-atomic write let a poll observe a zero-byte file and
    raise JSONDecodeError. `os.replace` must leave only the final file."""
    path = os.path.join(tmp_path, "status.json")

    app_module._write_json(path, {"state": "processing", "stage": "l3_asr"})

    assert os.path.exists(path)
    assert not os.path.exists(f"{path}.tmp")
    with open(path, encoding="utf-8") as f:
        assert json.load(f) == {"state": "processing", "stage": "l3_asr"}


# ── session creation ───────────────────────────────────────────────────────


def test_create_session_writes_input_and_idle_status(client) -> None:
    sid = _create_session(client)

    assert os.path.exists(os.path.join("outputs", sid, "input.wav"))
    with open(os.path.join("outputs", sid, "status.json"), encoding="utf-8") as f:
        status = json.load(f)
    assert status == {"state": "idle", "stage": None, "stages_done": [], "error": None}


def test_create_session_rejects_unsupported_audio_type(client) -> None:
    response = client.post(
        "/api/sessions",
        files={"audio": ("consult.ogg", b"junk", "audio/ogg")},
    )
    assert response.status_code == 400


def test_create_session_releases_live_preview_model(client, monkeypatch) -> None:
    """Memory discipline: a finished recording upload must release any
    resident live-preview model before production models load (web/live_asr.py)."""
    calls = []
    monkeypatch.setattr(app_module.live_asr, "release_model", lambda: calls.append(1))

    _create_session(client)

    assert calls == [1]


# ── live preview (UI-display-only, gate-exempt) ───────────────────────────


def test_live_preview_returns_text_when_model_available(client, monkeypatch) -> None:
    monkeypatch.setattr(app_module.live_asr, "available", lambda: True)
    monkeypatch.setattr(app_module.live_asr, "try_acquire", lambda: True)
    monkeypatch.setattr(app_module.live_asr, "release", lambda: None)
    monkeypatch.setattr(
        app_module.live_asr,
        "transcribe_preview",
        lambda wav_bytes: "fever since three days",
    )

    response = client.post(
        "/api/live/preview",
        files={"audio": ("live.wav", b"RIFF-fake-wav-bytes", "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json() == {"text": "fever since three days"}


def test_live_preview_is_503_when_model_unavailable(client, monkeypatch) -> None:
    monkeypatch.setattr(app_module.live_asr, "available", lambda: False)

    response = client.post(
        "/api/live/preview", files={"audio": ("live.wav", b"junk", "audio/wav")}
    )
    assert response.status_code == 503


def test_live_preview_rejects_overlapping_calls_with_429(client, monkeypatch) -> None:
    monkeypatch.setattr(app_module.live_asr, "available", lambda: True)
    monkeypatch.setattr(app_module.live_asr, "try_acquire", lambda: False)

    response = client.post(
        "/api/live/preview", files={"audio": ("live.wav", b"junk", "audio/wav")}
    )
    assert response.status_code == 429


# ── process / status state machine ────────────────────────────────────────


def test_process_advances_status_through_stages_to_review(client, monkeypatch) -> None:
    sid = _create_session(client)

    def fake_run(in_path, session_id=None, *, on_stage=None, on_progress=None, asr_engine="accurate"):
        for stage in ("l1_preprocess", "l2_diarize", "l3_asr"):
            on_stage(stage, "start")
            on_stage(stage, "end")
        return os.path.join("outputs", session_id, "draft_rx.pdf")

    monkeypatch.setattr(pipeline, "run", fake_run)

    response = client.post(f"/api/sessions/{sid}/process")
    assert response.status_code == 202
    assert response.json() == {"session_id": sid}

    deadline = time.monotonic() + 2.0
    status = {}
    while time.monotonic() < deadline:
        status = client.get(f"/api/sessions/{sid}/status").json()
        if status["state"] == "review":
            break
        time.sleep(0.01)

    assert status["state"] == "review"
    assert status["stages_done"] == ["l1_preprocess", "l2_diarize", "l3_asr"]
    assert status["error"] is None


def test_process_records_error_state_on_pipeline_failure(client, monkeypatch) -> None:
    sid = _create_session(client)

    def failing_run(in_path, session_id=None, *, on_stage=None, on_progress=None, asr_engine="accurate"):
        on_stage("l1_preprocess", "start")
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "run", failing_run)

    client.post(f"/api/sessions/{sid}/process")

    deadline = time.monotonic() + 2.0
    status = {}
    while time.monotonic() < deadline:
        status = client.get(f"/api/sessions/{sid}/status").json()
        if status["state"] == "error":
            break
        time.sleep(0.01)

    assert status["state"] == "error"
    assert "boom" in status["error"]


def test_process_unknown_session_is_404(client) -> None:
    response = client.post("/api/sessions/does-not-exist/process")
    assert response.status_code == 404


def test_status_unknown_session_is_404(client) -> None:
    response = client.get("/api/sessions/does-not-exist/status")
    assert response.status_code == 404


# ── L3 transcription progress / ETA (status.json "progress"/"eta_seconds") ──


def test_progress_callbacks_compute_eta_throttle_and_clear_on_stage_end(
    tmp_path, monkeypatch
) -> None:
    """Unit test of _make_progress_callbacks's math, in isolation from the
    daemon thread — deterministic control of time.monotonic() throughout."""
    monkeypatch.chdir(tmp_path)
    sid = "progress-test"
    os.makedirs(os.path.join("outputs", sid))
    app_module._write_json(
        app_module._status_path(sid),
        {"state": "idle", "stage": None, "stages_done": [], "error": None},
    )

    clock = {"t": 1000.0}
    monkeypatch.setattr(app_module.time, "monotonic", lambda: clock["t"])

    on_stage, on_progress = app_module._make_progress_callbacks(sid)

    on_stage("l3_asr", "start")  # l3_start = 1000.0
    clock["t"] = 1010.0  # 10s of "decoding" elapsed
    on_progress(5.0, 10.0)  # 50% done -> eta = 10 * (1-0.5)/0.5 = 10.0

    status = app_module._read_json(app_module._status_path(sid))
    assert status["progress"] == 0.5
    assert status["eta_seconds"] == 10.0

    # Throttle: within 1s of the last write, a new datum must not overwrite.
    clock["t"] = 1010.5
    on_progress(9.0, 10.0)
    status_throttled = app_module._read_json(app_module._status_path(sid))
    assert status_throttled["progress"] == 0.5

    # >=1s later, the next datum writes fresh values.
    clock["t"] = 1012.0
    on_progress(9.0, 10.0)  # 90% done -> eta = 12 * (1-0.9)/0.9
    status_updated = app_module._read_json(app_module._status_path(sid))
    assert status_updated["progress"] == 0.9
    assert status_updated["eta_seconds"] == round(12.0 * 0.1 / 0.9, 1)

    # Stage end clears both fields — the SPA's %/ETA suffix must disappear.
    on_stage("l3_asr", "end")
    status_final = app_module._read_json(app_module._status_path(sid))
    assert "progress" not in status_final
    assert "eta_seconds" not in status_final


# ── GET note (note + transcript + provenance + flags) ────────────────────


def test_get_note_returns_note_transcript_provenance_and_flags(client) -> None:
    sid = _create_session(client)
    note = ClinicalNote(
        chief_complaint="fever since Monday",
        history=None,
        low_confidence_fields=["chief_complaint"],
    )
    _write_note(sid, note)
    turn = {
        "speaker_role": "DOCTOR",
        "text": "fever since Monday",
        "start": 0.0,
        "end": 2.0,
    }
    _write_transcript(sid, [turn])

    response = client.get(f"/api/sessions/{sid}/note")
    assert response.status_code == 200
    body = response.json()

    assert body["note"]["chief_complaint"] == "fever since Monday"
    assert body["transcript"] == [turn]
    assert body["provenance"]["chief_complaint"]["turn_index"] == 0
    # "flags" is the INDEX-keyed dict the SPA reads; "low_confidence_fields"
    # is the raw NAME-keyed list straight off the note (contract GET note row).
    assert body["flags"] == {"chief_complaint": "chief_complaint"}
    assert body["low_confidence_fields"] == ["chief_complaint"]


def test_get_note_before_processing_is_404(client) -> None:
    sid = _create_session(client)
    response = client.get(f"/api/sessions/{sid}/note")
    assert response.status_code == 404


# ── PATCH note ─────────────────────────────────────────────────────────────


def test_patch_note_appends_correction_and_rewrites_note(client) -> None:
    sid = _create_session(client)
    note = ClinicalNote(chief_complaint=None, history=None)
    _write_note(sid, note)

    response = client.patch(
        f"/api/sessions/{sid}/note",
        json={
            "edits": [{"field": "chief_complaint", "old": None, "new": "fever"}],
            "lang": "en",
        },
    )
    assert response.status_code == 200
    assert response.json()["chief_complaint"] == "fever"

    with open(os.path.join("outputs", sid, "note.json"), encoding="utf-8") as f:
        assert json.load(f)["chief_complaint"] == "fever"

    with open(os.path.join("outputs", sid, "corrections.jsonl"), encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 1
    line = lines[0]
    assert set(line) == {"ts", "field", "old", "new", "lang"}
    assert line["field"] == "chief_complaint"
    assert line["old"] is None
    assert line["new"] == "fever"
    assert line["lang"] == "en"


def test_patch_note_medication_subfield_edit(client) -> None:
    from src.types import Medication

    sid = _create_session(client)
    note = ClinicalNote(
        chief_complaint=None,
        history=None,
        medications=[
            Medication(
                drug="Dolo 650",
                dose=None,
                frequency=None,
                timing=None,
                duration=None,
                validated=True,
            )
        ],
    )
    _write_note(sid, note)

    response = client.patch(
        f"/api/sessions/{sid}/note",
        json={
            "edits": [{"field": "medications[0].dose", "old": None, "new": "650 mg"}]
        },
    )
    assert response.status_code == 200
    assert response.json()["medications"][0]["dose"] == "650 mg"


def test_patch_note_unknown_field_is_400(client) -> None:
    sid = _create_session(client)
    _write_note(sid, ClinicalNote(chief_complaint=None, history=None))

    response = client.patch(
        f"/api/sessions/{sid}/note",
        json={"edits": [{"field": "not_a_real_field", "old": None, "new": "x"}]},
    )
    assert response.status_code == 400


# ── sign ────────────────────────────────────────────────────────────────


def test_sign_calls_render_signed_and_updates_status(client, monkeypatch) -> None:
    sid = _create_session(client)
    _write_note(sid, ClinicalNote(chief_complaint="fever", history=None))

    captured = {}

    def fake_render(note, out_path=None, *, lang="en", signed=False, doctor=None):
        captured.update(lang=lang, signed=signed, doctor=doctor, out_path=out_path)
        with open(out_path, "wb") as f:
            f.write(b"%PDF-fake")
        return out_path

    monkeypatch.setattr(app_module, "render", fake_render)

    response = client.post(
        f"/api/sessions/{sid}/sign",
        json={
            "lang": "hi",
            "doctor": {"name": "Dr. Rao", "reg_no": "MH12345", "clinic": "Rao Clinic"},
        },
    )
    assert response.status_code == 200
    assert response.json() == {"pdf_url": f"/api/sessions/{sid}/pdf?lang=hi"}

    assert captured["signed"] is True
    assert captured["lang"] == "hi"
    assert captured["doctor"] == {
        "name": "Dr. Rao",
        "reg_no": "MH12345",
        "clinic": "Rao Clinic",
    }

    with open(os.path.join("outputs", sid, "signed.json"), encoding="utf-8") as f:
        signed_data = json.load(f)
    assert signed_data["doctor"]["name"] == "Dr. Rao"
    assert signed_data["lang"] == "hi"

    status = client.get(f"/api/sessions/{sid}/status").json()
    assert status["state"] == "signed"


def test_get_pdf_renders_current_note_state(client, monkeypatch) -> None:
    sid = _create_session(client)
    _write_note(sid, ClinicalNote(chief_complaint="fever", history=None))

    def fake_render(note, out_path=None, *, lang="en", signed=False, doctor=None):
        with open(out_path, "wb") as f:
            f.write(b"%PDF-fake")
        return out_path

    monkeypatch.setattr(app_module, "render", fake_render)

    response = client.get(f"/api/sessions/{sid}/pdf?lang=en")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"


# ── translate-advice (thin wrapper over /translate) ──────────────────────


def _stub_ollama_batch_translate(monkeypatch, translations_by_call: list[list[str]]):
    """Stub ollama.chat to return one JSON-array response per call, in order."""

    class _FakeMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    class _FakeResponse:
        def __init__(self, content: str) -> None:
            self.message = _FakeMessage(content)

    calls = []
    remaining = list(translations_by_call)

    def fake_chat(model, messages, **kwargs):
        calls.append((model, messages, kwargs))
        translated = remaining.pop(0)
        return _FakeResponse(json.dumps(translated, ensure_ascii=False))

    import ollama

    monkeypatch.setattr(ollama, "chat", fake_chat)
    return calls


def test_translate_advice_calls_stubbed_ollama_and_caches(client, monkeypatch) -> None:
    sid = _create_session(client)
    _write_note(
        sid, ClinicalNote(chief_complaint=None, history=None, advice="Drink more water")
    )

    calls = _stub_ollama_batch_translate(monkeypatch, [["ज़्यादा पानी पिएं"]])

    response = client.post(f"/api/sessions/{sid}/translate-advice", json={"lang": "hi"})
    assert response.status_code == 200
    assert response.json() == {"advice": "ज़्यादा पानी पिएं"}
    assert len(calls) == 1  # note-values batch only; no transcript turns exist

    # second call is served from translations_hi.json's cache — no further calls.
    response2 = client.post(
        f"/api/sessions/{sid}/translate-advice", json={"lang": "hi"}
    )
    assert response2.json() == {"advice": "ज़्यादा पानी पिएं"}
    assert len(calls) == 1

    with open(
        os.path.join("outputs", sid, "translations_hi.json"), encoding="utf-8"
    ) as f:
        cache = json.load(f)
    assert cache["note_values"]["advice"] == "ज़्यादा पानी पिएं"


def test_translate_advice_english_is_a_no_op(client) -> None:
    sid = _create_session(client)
    _write_note(
        sid, ClinicalNote(chief_complaint=None, history=None, advice="Drink more water")
    )

    response = client.post(f"/api/sessions/{sid}/translate-advice", json={"lang": "en"})
    assert response.json() == {"advice": "Drink more water"}


# ── translate (note values + transcript, batched) ────────────────────────


def test_translate_excludes_medications_and_vitals_from_prompt(client, monkeypatch) -> None:
    from src.types import Medication, Vital

    sid = _create_session(client)
    note = ClinicalNote(
        chief_complaint="fever",
        history=None,
        vitals=[Vital(name="BP", value="120/80 mmHg")],
        medications=[
            Medication(
                drug="Azithral",
                dose="500 mg",
                frequency="once daily",
                timing="after food",
                duration="5 days",
                validated=False,
            )
        ],
    )
    _write_note(sid, note)
    _write_transcript(
        sid,
        [
            {
                "speaker_role": "DOCTOR",
                "text": "Take Azithral 500 mg once daily.",
                "start": 0.0,
                "end": 2.0,
            }
        ],
    )

    calls = _stub_ollama_batch_translate(
        monkeypatch, [["बुखार"], ["Azithral 500 mg रोज़ एक बार लें।"]]
    )

    response = client.post(f"/api/sessions/{sid}/translate", json={"lang": "hi"})
    assert response.status_code == 200
    body = response.json()
    assert body["note_values"] == {"chief_complaint": "बुखार"}
    assert body["transcript"] == ["Azithral 500 mg रोज़ एक बार लें।"]

    # Patient-safety assertion: the note_values group (first Ollama call) must
    # never carry medication or vitals text — only "fever" was eligible.
    note_values_call_payload = calls[0][1][1]["content"]
    assert "500 mg" not in note_values_call_payload
    assert "120/80" not in note_values_call_payload
    assert "Azithral" not in note_values_call_payload


def test_translate_caches_to_disk_and_serves_repeat_calls(client, monkeypatch) -> None:
    sid = _create_session(client)
    _write_note(sid, ClinicalNote(chief_complaint="fever", history=None))
    _write_transcript(sid, [])

    calls = _stub_ollama_batch_translate(monkeypatch, [["बुखार"]])

    response1 = client.post(f"/api/sessions/{sid}/translate", json={"lang": "hi"})
    assert response1.json() == {"note_values": {"chief_complaint": "बुखार"}, "transcript": []}
    assert len(calls) == 1

    response2 = client.post(f"/api/sessions/{sid}/translate", json={"lang": "hi"})
    assert response2.json() == {"note_values": {"chief_complaint": "बुखार"}, "transcript": []}
    assert len(calls) == 1  # served from translations_hi.json, no new Ollama call

    assert os.path.exists(os.path.join("outputs", sid, "translations_hi.json"))


def test_translate_lang_en_skips_already_latin_script_text(client) -> None:
    """lang="en" is a real target now, not a blanket no-op — but text that is
    already Latin-script (no Devanagari/Arabic) needs no Ollama call at all."""
    sid = _create_session(client)
    _write_note(sid, ClinicalNote(chief_complaint="fever", history=None))

    response = client.post(f"/api/sessions/{sid}/translate", json={"lang": "en"})
    assert response.json() == {"note_values": {}, "transcript": []}


def test_translate_lang_en_translates_non_latin_source_text(client, monkeypatch) -> None:
    """Bug 1 (target-language-absolute): a Devanagari-script note value
    selecting English must actually translate to English, not pass through."""
    sid = _create_session(client)
    _write_note(sid, ClinicalNote(chief_complaint="बुखार", history=None))
    _write_transcript(
        sid,
        [{"speaker_role": "PATIENT", "text": "मुझे बुखार है", "start": 0.0, "end": 2.0}],
    )

    calls = _stub_ollama_batch_translate(monkeypatch, [["Fever"], ["I have a fever"]])

    response = client.post(f"/api/sessions/{sid}/translate", json={"lang": "en"})
    assert response.status_code == 200
    body = response.json()
    assert body["note_values"] == {"chief_complaint": "Fever"}
    assert body["transcript"] == ["I have a fever"]
    assert len(calls) == 2  # one batch call for note values, one for transcript


def test_translate_per_item_retry_on_batch_misalignment_and_partial_failure(
    client, monkeypatch
) -> None:
    """Per-item robustness (Bug 1 hardening): a misaligned batch response
    falls back to per-item calls; one item's translation failure yields that
    item's original text untranslated, never a blank value or a crash."""
    sid = _create_session(client)
    _write_note(
        sid,
        ClinicalNote(chief_complaint="बुखार", history="सरदर्द है", examination=None),
    )
    _write_transcript(sid, [])

    class _FakeMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    class _FakeResponse:
        def __init__(self, content: str) -> None:
            self.message = _FakeMessage(content)

    calls: list[list[str]] = []

    def fake_chat(model, messages, **kwargs):
        payload = json.loads(messages[1]["content"])
        calls.append(payload)
        if len(payload) == 2:
            # Simulate a misaligned batch translation (wrong length).
            return _FakeResponse(json.dumps(["only one item"]))
        # Per-item retry: one text succeeds, one raises to simulate failure.
        text = payload[0]
        if text == "सरदर्द है":
            raise RuntimeError("ollama boom")
        return _FakeResponse(json.dumps([f"translated: {text}"]))

    import ollama

    monkeypatch.setattr(ollama, "chat", fake_chat)

    response = client.post(f"/api/sessions/{sid}/translate", json={"lang": "hi"})
    assert response.status_code == 200
    body = response.json()
    # chief_complaint recovered via per-item retry; history kept as original
    # (untranslated, never blank) because its individual retry raised.
    assert body["note_values"] == {
        "chief_complaint": "translated: बुखार",
        "history": "सरदर्द है",
    }
    assert len(calls) == 3  # 1 batch attempt + 2 per-item retries


def test_translate_unsupported_language_is_400(client) -> None:
    sid = _create_session(client)
    _write_note(sid, ClinicalNote(chief_complaint="fever", history=None))

    response = client.post(f"/api/sessions/{sid}/translate", json={"lang": "fr"})
    assert response.status_code == 400


# ── translations ────────────────────────────────────────────────────────


def test_get_translations_dumps_full_table(client) -> None:
    response = client.get("/api/translations")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"en", "hi", "mr"}
    assert body["hi"]["medications"] == "दवाइयाँ"


# ── background verification wiring (web/app.py <-> web/verification.py) ───


def test_process_accurate_engine_never_starts_verification(client, monkeypatch) -> None:
    """FAST_ASR_ENABLED is False (the client fixture's default) — status.json
    must never gain a "verification" key for the accurate engine."""
    sid = _create_session(client)

    def fake_run(in_path, session_id=None, *, on_stage=None, on_progress=None, asr_engine="accurate"):
        assert asr_engine == "accurate"
        on_stage("l1_preprocess", "start")
        on_stage("l1_preprocess", "end")
        return os.path.join("outputs", session_id, "draft_rx.pdf")

    monkeypatch.setattr(pipeline, "run", fake_run)
    calls = []
    monkeypatch.setattr(verification_module, "run_verification", lambda sid: calls.append(sid))

    client.post(f"/api/sessions/{sid}/process")

    deadline = time.monotonic() + 2.0
    status = {}
    while time.monotonic() < deadline:
        status = client.get(f"/api/sessions/{sid}/status").json()
        if status["state"] == "review":
            break
        time.sleep(0.01)

    assert status["state"] == "review"
    time.sleep(0.05)  # verification would already have started by now if wired wrong
    assert calls == []
    assert "verification" not in client.get(f"/api/sessions/{sid}/status").json()


def test_process_fast_engine_starts_verification_and_mirrors_status(client, monkeypatch) -> None:
    """FAST_ASR_ENABLED=True: pipeline.run receives asr_engine="fast", and
    once review is reached, web.verification.run_verification is kicked off
    with its result mirrored into status.json's "verification" key."""
    monkeypatch.setattr(app_module.config, "FAST_ASR_ENABLED", True)
    sid = _create_session(client)

    def fake_run(in_path, session_id=None, *, on_stage=None, on_progress=None, asr_engine="accurate"):
        assert asr_engine == "fast"
        return os.path.join("outputs", session_id, "draft_rx.pdf")

    def fake_run_verification(sid):
        return {
            "state": "done",
            "coverage": "full",
            "fields_differing": [{"field": "medications[0].drug", "fast_value": "a", "accurate_value": "b"}],
        }

    monkeypatch.setattr(pipeline, "run", fake_run)
    monkeypatch.setattr(verification_module, "run_verification", fake_run_verification)

    client.post(f"/api/sessions/{sid}/process")

    deadline = time.monotonic() + 2.0
    status = {}
    while time.monotonic() < deadline:
        status = client.get(f"/api/sessions/{sid}/status").json()
        if status.get("verification", {}).get("state") == "done":
            break
        time.sleep(0.01)

    assert status["state"] == "review"
    assert status["verification"] == {"state": "done", "n_differing": 1, "coverage": "full"}


def test_get_verification_before_started_is_404(client) -> None:
    sid = _create_session(client)
    response = client.get(f"/api/sessions/{sid}/verification")
    assert response.status_code == 404


def test_get_verification_returns_written_file(client) -> None:
    sid = _create_session(client)
    payload = {"state": "done", "fields_differing": [], "coverage": "full"}
    with open(os.path.join("outputs", sid, "verification.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f)

    response = client.get(f"/api/sessions/{sid}/verification")
    assert response.status_code == 200
    assert response.json() == payload


# ── web/verification.py — targeted second-listen diff logic ──────────────


def _write_silence_wav(path: str, duration_s: float, sr: int = 16000) -> None:
    sf.write(path, np.zeros(int(duration_s * sr), dtype="float32"), sr, subtype="PCM_16")


def _fake_decode_factory(canned: list[tuple[float, float, str]]):
    """Fake for verification_module._decode_window_words_with_guards.

    `canned` is the "ground truth" (start, end, text) for each field's own
    accurate re-decode, independent of how the production code packs spans
    into windows. For a requested [start, end) window, returns the words of
    every canned span that falls entirely within it, each canned span's text
    split into one word per whitespace token, evenly spaced across its own
    [start, end) — so _words_in_span can recover exactly that span's text
    regardless of how the caller padded/merged/packed it.
    """

    def fake_decode(audio, sr, start, end, repo, decode_kwargs):
        words: list[tuple[float, float, str]] = []
        for span_start, span_end, text in canned:
            if span_start >= start and span_end <= end:
                tokens = text.split()
                if not tokens:
                    continue
                step = (span_end - span_start) / len(tokens)
                for i, tok in enumerate(tokens):
                    w_start = span_start + i * step
                    words.append((w_start, w_start + step, tok))
        words.sort(key=lambda w: w[0])
        return " ".join(w[2] for w in words), words

    return fake_decode


def _setup_verification_session(
    tmp_path,
    monkeypatch,
    *,
    note: ClinicalNote,
    turns: list[dict],
    canned_spans: list[tuple[float, float, str]],
    corrections: list[dict] | None = None,
    wav_duration_s: float = 150.0,
) -> str:
    monkeypatch.chdir(tmp_path)
    sid = "verify-test"
    session_dir = os.path.join("outputs", sid)
    os.makedirs(session_dir, exist_ok=True)
    _write_note(sid, note)
    _write_transcript(sid, turns)
    if corrections:
        with open(os.path.join(session_dir, "corrections.jsonl"), "w", encoding="utf-8") as f:
            for c in corrections:
                f.write(json.dumps(c) + "\n")
    _write_silence_wav(os.path.join(session_dir, "input_16k.wav"), wav_duration_s)
    monkeypatch.setattr(
        verification_module, "_decode_window_words_with_guards", _fake_decode_factory(canned_spans)
    )
    monkeypatch.setattr(verification_module, "looks_degenerate", lambda text, audio_seconds: False)
    return sid


def test_run_verification_identical_and_fold_equal_values_are_not_flagged(tmp_path, monkeypatch) -> None:
    """Identical notes -> zero differing; drug-field fold comparison
    ("naxdom 500" vs "नक्सडम 500") -> NOT flagged (fold-equal)."""
    note = ClinicalNote(
        chief_complaint=None,
        history=None,
        vitals=[Vital(name="BP", value="120/80 mmHg")],
        diagnosis=[Diagnosis(term="Viral fever", snomed_id=None)],
        medications=[
            Medication(
                drug="naxdom 500", dose="500 mg", frequency="twice daily",
                timing=None, duration=None, validated=True,
            )
        ],
    )
    turns = [
        {"speaker_role": "DOCTOR", "text": "give one naxdom 500 twice daily", "start": 2.0, "end": 4.0},
        {"speaker_role": "DOCTOR", "text": "BP is 120 over 80 mmHg", "start": 10.0, "end": 12.0},
        {"speaker_role": "DOCTOR", "text": "looks like viral fever prescribing accordingly", "start": 20.0, "end": 22.0},
    ]
    canned_spans = [
        (2.0, 4.0, "give one नक्सडम 500 twice daily"),  # Devanagari drug spelling, fold-equal
        (10.0, 12.0, "BP is 120 over 80 mmHg"),
        (20.0, 22.0, "looks like viral fever prescribing accordingly"),
    ]
    sid = _setup_verification_session(tmp_path, monkeypatch, note=note, turns=turns, canned_spans=canned_spans)

    result = verification_module.run_verification(sid)

    assert result["state"] == "done"
    assert result["fields_differing"] == []
    assert result["coverage"] == "full"
    assert result["unverified_by_budget"] == []
    assert result["doctor_resolved"] == []
    assert result["verified_seconds"] > 0


def test_run_verification_genuinely_different_value_is_flagged(tmp_path, monkeypatch) -> None:
    note = ClinicalNote(
        chief_complaint=None,
        history=None,
        medications=[
            Medication(
                drug="azithral", dose="250 mg", frequency="once daily",
                timing=None, duration=None, validated=True,
            )
        ],
    )
    turns = [
        {"speaker_role": "DOCTOR", "text": "also give azithral 250 once daily", "start": 2.0, "end": 4.0},
    ]
    canned_spans = [(2.0, 4.0, "also give augmentin 250 once daily")]
    sid = _setup_verification_session(tmp_path, monkeypatch, note=note, turns=turns, canned_spans=canned_spans)

    result = verification_module.run_verification(sid)

    assert result["state"] == "done"
    differing_fields = {d["field"] for d in result["fields_differing"]}
    assert differing_fields == {"medications[0].drug"}
    entry = result["fields_differing"][0]
    assert entry["fast_value"] == "azithral"
    assert "augmentin" in entry["accurate_value"]
    # dose ("250") still present verbatim in the re-decode -> digit-exact match, not flagged.


def test_run_verification_doctor_edited_field_excluded_and_resolved(tmp_path, monkeypatch) -> None:
    note = ClinicalNote(
        chief_complaint=None,
        history=None,
        medications=[
            Medication(
                drug="azithral", dose="250 mg", frequency="once daily",
                timing=None, duration=None, validated=True,
            )
        ],
    )
    turns = [
        {"speaker_role": "DOCTOR", "text": "also give azithral 250 once daily", "start": 2.0, "end": 4.0},
    ]
    canned_spans = [(2.0, 4.0, "also give augmentin 250 once daily")]  # would flag drug w/o the correction
    corrections = [
        {"ts": "2026-07-11T10:00:00+05:30", "field": "medications[0].drug", "old": "azithral", "new": "augmentin", "lang": "en"}
    ]
    sid = _setup_verification_session(
        tmp_path, monkeypatch, note=note, turns=turns, canned_spans=canned_spans, corrections=corrections
    )

    result = verification_module.run_verification(sid)

    assert result["fields_differing"] == []
    assert result["doctor_resolved"] == ["medications[0].drug"]
    assert result["unverified_by_budget"] == []  # excluded, not counted as unverified either


def test_run_verification_partial_coverage_marks_unverified_by_budget(tmp_path, monkeypatch) -> None:
    """Fields too far apart to share a window under VERIFY_MAX_WINDOWS=1 —
    the lower-priority one is dropped and reported, not silently discarded."""
    monkeypatch.setattr(verification_module.config, "VERIFY_MAX_WINDOWS", 1)
    note = ClinicalNote(
        chief_complaint=None,
        history=None,
        medications=[
            Medication(
                drug="paracetamol", dose=None, frequency=None, timing=None, duration=None, validated=True,
            )
        ],
        diagnosis=[Diagnosis(term="Viral fever", snomed_id=None)],
    )
    turns = [
        {"speaker_role": "DOCTOR", "text": "take paracetamol daily", "start": 2.0, "end": 4.0},
        {"speaker_role": "DOCTOR", "text": "looks like viral fever today", "start": 100.0, "end": 102.0},
    ]
    canned_spans = [
        (2.0, 4.0, "take paracetamol daily"),
        (100.0, 102.0, "looks like viral fever today"),
    ]
    sid = _setup_verification_session(
        tmp_path, monkeypatch, note=note, turns=turns, canned_spans=canned_spans, wav_duration_s=150.0
    )

    result = verification_module.run_verification(sid)

    assert result["coverage"] == "partial"
    assert result["unverified_by_budget"] == ["diagnosis[0].term"]  # lower priority than the drug
    assert result["fields_differing"] == []  # the drug that WAS checked matched


# ── web/verification.py — span collection / padding / merging / budget ────


def test_merge_safety_fields_merges_overlapping_and_keeps_distant_spans_separate() -> None:
    fields = [
        verification_module._SafetyField(path="medications[0].drug", value="x", start=10.0, end=11.0, priority=0),
        verification_module._SafetyField(path="medications[0].dose", value="y", start=10.5, end=11.5, priority=1),
        verification_module._SafetyField(path="diagnosis[0].term", value="z", start=100.0, end=101.0, priority=3),
    ]

    merged = verification_module._merge_safety_fields(fields, total_duration=200.0, pad_s=1.5)

    assert len(merged) == 2
    assert merged[0].start == pytest.approx(8.5)
    assert merged[0].end == pytest.approx(13.0)
    assert {f.path for f in merged[0].fields} == {"medications[0].drug", "medications[0].dose"}
    assert merged[1].fields[0].path == "diagnosis[0].term"
    # padding clamps to the clip, never goes negative or past total_duration
    assert merged[1].start == pytest.approx(98.5)
    assert merged[1].end == pytest.approx(102.5)


def test_merge_safety_fields_clamps_padding_to_clip_bounds() -> None:
    fields = [verification_module._SafetyField(path="vitals[0].value", value="x", start=0.2, end=0.4, priority=2)]

    merged = verification_module._merge_safety_fields(fields, total_duration=5.0, pad_s=1.5)

    assert merged[0].start == 0.0  # start - pad_s would be negative
    assert merged[0].end == pytest.approx(1.9)


def test_select_within_budget_prioritizes_drug_over_dose_over_vital_over_diagnosis() -> None:
    def span(path: str, priority: int, start: float) -> verification_module._MergedSpan:
        f = verification_module._SafetyField(path=path, value="x", start=start, end=start + 2.0, priority=priority)
        return verification_module._MergedSpan(start=start, end=start + 2.0, fields=[f])

    # 40s apart -- no two can ever share a <=28s window.
    spans_by_priority = [
        span("medications[0].drug", 0, 0.0),
        span("medications[0].dose", 1, 40.0),
        span("vitals[0].value", 2, 80.0),
        span("diagnosis[0].term", 3, 120.0),
    ]

    selected, skipped = verification_module._select_within_budget(
        spans_by_priority, max_windows=1, max_window_s=28.0
    )

    assert [s.fields[0].path for s in selected] == ["medications[0].drug"]
    assert {s.fields[0].path for s in skipped} == {
        "medications[0].dose", "vitals[0].value", "diagnosis[0].term",
    }


def test_select_within_budget_keeps_lower_priority_span_that_fits_a_used_window() -> None:
    """A later, LOWER-priority span that happens to overlap an already-
    selected window's budget is still accepted -- priority order decides who
    gets tried first, not who is allowed in at all."""
    def span(path: str, priority: int, start: float, end: float) -> verification_module._MergedSpan:
        f = verification_module._SafetyField(path=path, value="x", start=start, end=end, priority=priority)
        return verification_module._MergedSpan(start=start, end=end, fields=[f])

    spans_by_priority = [
        span("medications[0].drug", 0, 0.0, 2.0),
        span("vitals[0].value", 2, 3.0, 4.0),  # close enough to fit alongside the drug span
    ]

    selected, skipped = verification_module._select_within_budget(spans_by_priority, max_windows=1, max_window_s=28.0)

    assert {s.fields[0].path for s in selected} == {"medications[0].drug", "vitals[0].value"}
    assert skipped == []
