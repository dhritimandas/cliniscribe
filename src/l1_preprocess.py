"""L1 — Audio preprocessing: resample, optional denoise, VAD silence removal."""

import logging
import os

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf

TARGET_SR = 16_000
VAD_TOP_DB = 40  # dB below peak to treat as silence

logger = logging.getLogger(__name__)


def preprocess(in_path: str, *, denoise: bool = False, out_dir: str = "outputs") -> str:
    """Resample audio to 16 kHz mono WAV, optionally denoise, drop long silences.

    Args:
        in_path: Path to input audio file (any sample rate, mono or stereo).
        denoise: Apply stationary noise reduction (default OFF). EkaCare's
            denoising-impact study shows aggressive reduction can degrade WER;
            benchmark with/without before enabling for a given noise profile.
        out_dir: Directory for the processed WAV. The pipeline passes the
            session directory (outputs/<session_id>/) so all artifacts of one
            consultation share a location; defaults to outputs/ for direct use.

    Returns:
        Path to the processed 16 kHz mono WAV file under out_dir.
    """
    audio, _ = librosa.load(in_path, sr=TARGET_SR, mono=True)

    if denoise:
        audio = nr.reduce_noise(y=audio, sr=TARGET_SR, stationary=True).astype(np.float32)

    audio, _ = librosa.effects.trim(audio, top_db=VAD_TOP_DB)

    stem = os.path.splitext(os.path.basename(in_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_16k.wav")
    os.makedirs(out_dir, exist_ok=True)
    sf.write(out_path, audio, TARGET_SR, subtype="PCM_16")

    logger.info("L1: %s → %s (%.1fs)", in_path, out_path, len(audio) / TARGET_SR)
    return out_path
