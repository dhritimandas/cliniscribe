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


# ── process / status state machine ────────────────────────────────────────


def test_process_advances_status_through_stages_to_review(client, monkeypatch) -> None:
    sid = _create_session(client)

    def fake_run(in_path, session_id=None, *, on_stage=None):
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

    def failing_run(in_path, session_id=None, *, on_stage=None):
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


def test_translate_english_is_a_no_op(client) -> None:
    sid = _create_session(client)
    _write_note(sid, ClinicalNote(chief_complaint="fever", history=None))

    response = client.post(f"/api/sessions/{sid}/translate", json={"lang": "en"})
    assert response.json() == {"note_values": {}, "transcript": []}


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
