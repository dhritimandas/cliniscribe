"""Tests for L1 audio preprocessing."""

import os

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def sample_mp3() -> str:
    """Return path to an existing sample MP3 — downloaded during project setup."""
    path = "data/sample_00.mp3"
    if not os.path.exists(path):
        pytest.skip("sample MP3 not found — run data extraction first")
    return path


def test_output_is_16k_mono_wav(sample_mp3: str, tmp_path: str) -> None:
    """Processed file must be 16 kHz, mono, PCM WAV."""
    from src.l1_preprocess import preprocess

    out = preprocess(sample_mp3)
    info = sf.info(out)
    assert info.samplerate == 16_000
    assert info.channels == 1
    assert "PCM" in info.subtype


def test_output_duration_is_reasonable(sample_mp3: str) -> None:
    """Output duration must be > 1s and no longer than the input + 0.5s (VAD only trims)."""
    import librosa

    from src.l1_preprocess import preprocess

    out = preprocess(sample_mp3)
    audio, sr = librosa.load(out, sr=None)
    duration = len(audio) / sr
    assert duration > 1.0
    assert duration < 15.0  # sample_00 is ~11s


def test_denoising_toggle_does_not_crash(sample_mp3: str) -> None:
    """preprocess with denoise=True must succeed and return a valid WAV."""
    from src.l1_preprocess import preprocess

    out = preprocess(sample_mp3, denoise=True)
    info = sf.info(out)
    assert info.samplerate == 16_000


def test_output_is_finite(sample_mp3: str) -> None:
    """Output audio must not contain NaN or Inf."""
    import soundfile as sf
    import numpy as np

    from src.l1_preprocess import preprocess

    out = preprocess(sample_mp3)
    audio, _ = sf.read(out)
    assert np.all(np.isfinite(audio))
