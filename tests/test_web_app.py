"""Tests for web/app.py — the review-frontend FastAPI backend.

pipeline.run and web.app.render are stubbed throughout: no ASR/diarization/
LLM models are loaded, and no real PDF is rendered.
"""

import json
import os
import time

import pytest
from fastapi.testclient import TestClient

import src.pipeline as pipeline
import web.app as app_module
from src.types import ClinicalNote


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient with cwd isolated to tmp_path so outputs/ never touches the repo."""
    monkeypatch.chdir(tmp_path)
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

    def fake_run(in_path, session_id=None, *, on_stage=None, on_progress=None):
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

    def failing_run(in_path, session_id=None, *, on_stage=None, on_progress=None):
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
