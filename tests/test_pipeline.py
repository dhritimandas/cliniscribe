"""Session-scoping tests for pipeline.run — stages stubbed, no models loaded."""

import json
import os

import pytest

import src.pipeline as pipeline
from src.types import ClinicalNote, Segment, Turn

_TURN = Turn(speaker_role="DOCTOR", text="fever since Monday", start=0.0, end=2.0)
_NOTE = ClinicalNote(
    chief_complaint="fever",
    history=None,
    low_confidence_fields=[],
)


@pytest.fixture
def stubbed_stages(monkeypatch, tmp_path):
    """Stub every stage; run() should only orchestrate and persist."""
    monkeypatch.chdir(tmp_path)  # outputs/ lands in tmp, not the repo

    def fake_preprocess(in_path, *, denoise=False, out_dir="outputs"):
        os.makedirs(out_dir, exist_ok=True)
        wav = os.path.join(out_dir, "audio_16k.wav")
        with open(wav, "wb") as f:
            f.write(b"RIFF")
        return wav

    def fake_render(note, out_path=None):
        path = out_path or os.path.join("outputs", "draft_rx.pdf")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"%PDF")
        return path

    monkeypatch.setattr(pipeline, "preprocess", fake_preprocess)
    monkeypatch.setattr(pipeline, "diarize", lambda wav: [Segment(0.0, 2.0, "S0")])
    monkeypatch.setattr(pipeline, "transcribe", lambda wav, segs, on_progress=None: [_TURN])
    monkeypatch.setattr(pipeline, "normalize", lambda turns: turns)
    monkeypatch.setattr(pipeline, "extract", lambda turns: _NOTE)
    monkeypatch.setattr(pipeline, "render", fake_render)
    monkeypatch.setattr(pipeline, "warm_llm", lambda: None)  # no Ollama in tests
    return tmp_path


def test_run_writes_all_artifacts_under_one_session_dir(stubbed_stages) -> None:
    pdf_path = pipeline.run("consult.mp3", session_id="test-session")

    session_dir = os.path.join("outputs", "test-session")
    assert pdf_path == os.path.join(session_dir, "draft_rx.pdf")
    assert os.path.exists(os.path.join(session_dir, "audio_16k.wav"))
    assert os.path.exists(os.path.join(session_dir, "transcript.json"))
    assert os.path.exists(os.path.join(session_dir, "note.json"))
    assert os.path.exists(pdf_path)


def test_run_generates_session_id_when_omitted(stubbed_stages) -> None:
    pdf_path = pipeline.run("consult.mp3")

    # PDF sits directly inside outputs/<generated-session-id>/
    session_dir = os.path.dirname(pdf_path)
    assert os.path.dirname(session_dir) == "outputs"
    assert os.path.basename(session_dir)  # non-empty generated ID
    assert os.path.basename(pdf_path) == "draft_rx.pdf"


def test_persisted_transcript_and_note_are_valid_json(stubbed_stages) -> None:
    pipeline.run("consult.mp3", session_id="s2")

    with open(os.path.join("outputs", "s2", "transcript.json"), encoding="utf-8") as f:
        turns = json.load(f)
    assert turns == [
        {"speaker_role": "DOCTOR", "text": "fever since Monday", "start": 0.0, "end": 2.0}
    ]

    with open(os.path.join("outputs", "s2", "note.json"), encoding="utf-8") as f:
        note = json.load(f)
    assert note["chief_complaint"] == "fever"


def test_new_session_ids_are_unique() -> None:
    ids = {pipeline.new_session_id() for _ in range(50)}
    assert len(ids) == 50


def test_run_forwards_on_progress_to_transcribe(stubbed_stages, monkeypatch) -> None:
    """run()'s on_progress kwarg must reach transcribe() unchanged (L3 is the
    only stage that reports real sub-stage progress; see src/l3_asr.py)."""
    received_kwarg = {}

    def fake_transcribe(wav, segs, on_progress=None):
        received_kwarg["on_progress"] = on_progress
        if on_progress:
            on_progress(1.0, 2.0)
        return [_TURN]

    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)

    events = []
    pipeline.run(
        "consult.mp3",
        session_id="progress-test",
        on_progress=lambda done, total: events.append((done, total)),
    )

    assert received_kwarg["on_progress"] is not None
    assert events == [(1.0, 2.0)]


# ── asr_engine selection (fast + background check architecture) ──────────


def test_run_default_asr_engine_uses_accurate_transcribe(stubbed_stages, monkeypatch) -> None:
    """CLI behavior unchanged: no asr_engine kwarg -> src.l3_asr.transcribe."""
    calls = []
    monkeypatch.setattr(pipeline, "transcribe", lambda wav, segs, on_progress=None: (calls.append("accurate"), [_TURN])[1])
    monkeypatch.setattr(pipeline, "fast_transcribe_windowed", lambda wav, segs, on_progress=None: (calls.append("fast"), [_TURN])[1])

    pipeline.run("consult.mp3", session_id="engine-default")

    assert calls == ["accurate"]


def test_run_asr_engine_fast_uses_fast_transcribe_windowed(stubbed_stages, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(pipeline, "transcribe", lambda wav, segs, on_progress=None: (calls.append("accurate"), [_TURN])[1])
    monkeypatch.setattr(pipeline, "fast_transcribe_windowed", lambda wav, segs, on_progress=None: (calls.append("fast"), [_TURN])[1])

    pipeline.run("consult.mp3", session_id="engine-fast", asr_engine="fast")

    assert calls == ["fast"]


def test_run_rejects_unknown_asr_engine(stubbed_stages) -> None:
    with pytest.raises(ValueError):
        pipeline.run("consult.mp3", session_id="engine-bad", asr_engine="bogus")
