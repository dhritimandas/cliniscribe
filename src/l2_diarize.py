"""L2 — Speaker diarization using pyannote/speaker-diarization-community-1."""

import logging

from src.types import Segment

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
        Model: pyannote/speaker-diarization-community-1. Requires HF_TOKEN and
        acceptance of pyannote T&C on HuggingFace. Load pipeline, run, del before
        loading ASR model — never hold both in memory simultaneously.

        Known failure mode: overlapping speech and rapid back-and-forth degrade
        DER. w2v-bert-2.0 backbone is a candidate upgrade if DER is unacceptable.
    """
    raise NotImplementedError
