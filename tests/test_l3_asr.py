"""Tests for L3 ASR: doctor-heuristic scoring and role assignment logic."""

import pytest

from src import config
from src.l3_asr import _contains_arabic_script, _doctor_score, transcribe
from src.types import Segment, Turn


def test_doctor_score_hindi_clinical_terms() -> None:
    assert _doctor_score("dawai do din ke liye") > 0


def test_doctor_score_english_question_forms() -> None:
    assert _doctor_score("how long have you had pain") > 2


def test_doctor_score_empty_text() -> None:
    assert _doctor_score("") == 0


def test_doctor_score_irrelevant_text() -> None:
    assert _doctor_score("hello hi yes okay") == 0


def test_transcribe_produces_turns_from_real_wav() -> None:
    """L3 must return at least one Turn for a real audio file."""
    import os

    from src.l3_asr import transcribe

    wav = "outputs/sample_00_16k.wav"
    if not os.path.exists(wav):
        pytest.skip("pre-processed WAV not found — run phase-A first")

    segments = [Segment(start=0.03, end=2.75, speaker="SPEAKER_00")]
    turns = transcribe(wav, segments)
    assert len(turns) >= 1
    assert all(isinstance(t, Turn) for t in turns)
    assert all(t.text.strip() for t in turns)


def test_role_unknown_for_single_speaker() -> None:
    """Role must be UNKNOWN when only one speaker label appears."""
    import os

    from src.l3_asr import transcribe

    wav = "outputs/sample_00_16k.wav"
    if not os.path.exists(wav):
        pytest.skip("pre-processed WAV not found — run phase-A first")

    segments = [Segment(start=0.03, end=2.75, speaker="SPEAKER_00")]
    turns = transcribe(wav, segments)
    assert all(t.speaker_role == "UNKNOWN" for t in turns)


@pytest.mark.slow
def test_vad_filter_is_a_noop_on_silence_with_clip_timestamps() -> None:
    """Silence hallucinates identically regardless of config.ASR_VAD_FILTER.

    faster-whisper hallucinates text (a fixed, deterministic phrase for this
    model/beam/audio, not flaky) when asked to decode pure silence. The
    hypothesis was that vad_filter=True would suppress this. It does not:
    faster-whisper's own docs state "vad_filter will be ignored if
    clip_timestamps is used" (transcribe.py), and transcribe() always passes
    clip_timestamps for per-segment decoding — confirmed directly against
    WhisperModel.transcribe (vad_filter=True + explicit clip_timestamps still
    hallucinates; vad_filter=True + no clip_timestamps correctly returns zero
    segments). This test documents that reality: toggling
    config.ASR_VAD_FILTER must not change transcribe()'s output at all.
    Guards against a future dev "fixing" the flag without noticing it never
    took effect, and against a faster-whisper upgrade silently changing this
    interaction (which would be worth re-running the gate over).
    """
    import os

    import numpy as np
    import soundfile as sf

    from src import config
    from src.l3_asr import transcribe

    wav_path = "outputs/_test_silence_20s.wav"
    silence = np.zeros(20 * 16_000, dtype=np.float32)
    sf.write(wav_path, silence, 16_000, subtype="PCM_16")
    segments = [Segment(start=0.0, end=20.0, speaker="S0")]

    original = config.ASR_VAD_FILTER
    try:
        config.ASR_VAD_FILTER = False
        turns_no_vad = transcribe(wav_path, segments)
        config.ASR_VAD_FILTER = True
        turns_vad = transcribe(wav_path, segments)
    finally:
        config.ASR_VAD_FILTER = original
        os.remove(wav_path)

    print(f"vad_filter=False: {[t.text for t in turns_no_vad]}")
    print(f"vad_filter=True:  {[t.text for t in turns_vad]}")
    assert [t.text for t in turns_no_vad] == [t.text for t in turns_vad]


# ── Script guard (Arabic-script misdetection) ────────────────────────────────
#
# Real repro: session 20260710-230150-cef13a decoded spoken Hindi as Urdu in
# Arabic script ("نیکس ڈوم فائیو ہنڈریڈ" for "Naxdom five hundred"). We only
# support hi/en/mr (Latin/Devanagari), so Arabic script is always a
# misdetection. These tests mock WhisperModel so they run without loading the
# real model.


class _FakeChunk:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeWhisperModel:
    """Stub WhisperModel.transcribe: returns one queued text per call.

    Records every call's kwargs so tests can assert on `language` and call
    count (i.e. whether a re-decode happened).
    """

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.calls: list[dict] = []

    def transcribe(self, wav_path: str, **kwargs):
        self.calls.append(kwargs)
        text = self._texts[len(self.calls) - 1]
        return iter([_FakeChunk(text)]), None


def test_contains_arabic_script_detects_urdu_text() -> None:
    assert _contains_arabic_script("نیکس ڈوم فائیو ہنڈریڈ")
    assert not _contains_arabic_script("नैक्सडॉम फाइव हंड्रेड")
    assert not _contains_arabic_script("naxdom five hundred")


def test_script_guard_redecodes_arabic_script_segment() -> None:
    """Arabic-script first decode triggers exactly one re-decode with hi."""
    model = _FakeWhisperModel(
        ["نیکس ڈوم فائیو ہنڈریڈ", "नैक्सडॉम फाइव हंड्रेड"]
    )
    segments = [Segment(start=11.47, end=24.97, speaker="S0")]

    original = config.ASR_SCRIPT_GUARD
    try:
        config.ASR_SCRIPT_GUARD = True
        turns = transcribe("fake.wav", segments, model=model)
    finally:
        config.ASR_SCRIPT_GUARD = original

    assert len(model.calls) == 2
    assert model.calls[0]["language"] is None
    assert model.calls[1]["language"] == "hi"
    assert turns[0].text == "नैक्सडॉम फाइव हंड्रेड"


def test_script_guard_no_redecode_for_clean_devanagari() -> None:
    """Clean Devanagari decodes must not trigger a re-decode."""
    model = _FakeWhisperModel(["मुझे बुखार हो रहा है"])
    segments = [Segment(start=0.0, end=2.0, speaker="S0")]

    original = config.ASR_SCRIPT_GUARD
    try:
        config.ASR_SCRIPT_GUARD = True
        turns = transcribe("fake.wav", segments, model=model)
    finally:
        config.ASR_SCRIPT_GUARD = original

    assert len(model.calls) == 1
    assert turns[0].text == "मुझे बुखार हो रहा है"


def test_script_guard_no_redecode_for_clean_latin() -> None:
    """Clean Latin (English) decodes must not trigger a re-decode."""
    model = _FakeWhisperModel(["how long have you had this fever"])
    segments = [Segment(start=0.0, end=2.0, speaker="S0")]

    original = config.ASR_SCRIPT_GUARD
    try:
        config.ASR_SCRIPT_GUARD = True
        transcribe("fake.wav", segments, model=model)
    finally:
        config.ASR_SCRIPT_GUARD = original

    assert len(model.calls) == 1


def test_script_guard_off_leaves_arabic_script_unchanged() -> None:
    """With config.ASR_SCRIPT_GUARD off, Arabic-script text passes through."""
    model = _FakeWhisperModel(["نیکس ڈوم فائیو ہنڈریڈ"])
    segments = [Segment(start=0.0, end=2.0, speaker="S0")]

    original = config.ASR_SCRIPT_GUARD
    try:
        config.ASR_SCRIPT_GUARD = False
        turns = transcribe("fake.wav", segments, model=model)
    finally:
        config.ASR_SCRIPT_GUARD = original

    assert len(model.calls) == 1
    assert turns[0].text == "نیکس ڈوم فائیو ہنڈریڈ"
