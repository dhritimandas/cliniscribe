"""L1 — Audio preprocessing: resample, optional denoise, VAD silence removal."""

import logging

logger = logging.getLogger(__name__)


def preprocess(in_path: str) -> str:
    """Resample audio to 16 kHz mono WAV, optionally denoise, drop long silences.

    Args:
        in_path: Path to input audio file (any sample rate, mono or stereo).

    Returns:
        Path to the processed 16 kHz mono WAV file.

    Notes:
        Denoising is off by default. EkaCare's denoising-impact study shows
        aggressive reduction can degrade ASR accuracy. Benchmark WER with and
        without denoising before enabling per noise profile.
    """
    raise NotImplementedError
