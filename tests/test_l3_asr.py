"""Tests for L3 ASR: doctor-heuristic scoring and role assignment logic."""

import pytest

from src.l3_asr import _doctor_score
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
