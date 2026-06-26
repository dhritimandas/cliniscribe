"""L2 — Speaker diarization using pyannote/speaker-diarization-community-1."""

import gc
import logging
import os

import soundfile as sf
import torch

from src.types import Segment

DIARIZE_MODEL = "pyannote/speaker-diarization-community-1"

logger = logging.getLogger(__name__)


def diarize(wav_path: str) -> list[Segment]:
    """Segment audio by speaker using pyannote diarization.

    Args:
        wav_path: Path to a 16 kHz mono WAV file (output of L1).

    Returns:
        List of Segment(start, end, speaker) sorted by start time.
        Speaker labels are arbitrary strings (e.g. "SPEAKER_00").
        Role attribution (DOCTOR vs. PATIENT) happens in L3 after transcription.

    Notes:
        torchcodec is broken in this environment (no FFmpeg dylibs). We work
        around it by loading the WAV ourselves and passing a preloaded
        {'waveform': tensor, 'sample_rate': int} dict to the pipeline, as
        recommended in the pyannote warning message.

        Model is released (del + gc + mps.empty_cache) before returning so L3
        can load faster-whisper into the same memory budget.
    """
    from pyannote.audio import Pipeline

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN not set — required for pyannote model download")

    pipeline = Pipeline.from_pretrained(DIARIZE_MODEL, token=hf_token)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    pipeline.to(device)
    logger.info("L2: loaded %s on %s", DIARIZE_MODEL, device)

    audio_array, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    # soundfile returns (samples, channels); pyannote needs (channels, samples)
    waveform = torch.from_numpy(audio_array.T)
    audio_input = {"waveform": waveform, "sample_rate": sr}

    output = pipeline(audio_input)
    # community-1 returns DiarizeOutput; speaker_diarization is the Annotation object
    diarization = output.speaker_diarization

    segments = [
        Segment(start=seg.start, end=seg.end, speaker=label)
        for seg, _, label in diarization.itertracks(yield_label=True)
    ]
    segments.sort(key=lambda s: s.start)
    logger.info("L2: %d segments, %d speakers", len(segments), len({s.speaker for s in segments}))

    del pipeline, waveform, audio_input
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return segments
