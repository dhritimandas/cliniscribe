"""Phase A proof script: L1 → L2 → L3 on 3 sample audio files.

Runs each stage sequentially, reports peak RSS after each stage, and prints
a speaker-attributed transcript per sample.

Usage:
    source .venv/bin/activate
    python run_phase_a.py
"""

import gc
import logging
import os
import resource
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s — %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("phase_a")

SAMPLES = [
    "data/sample_00.mp3",
    "data/sample_01.mp3",
    "data/sample_02.mp3",
]


def peak_rss_mb() -> float:
    """Return peak RSS in MB (macOS: ru_maxrss is bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def run_sample(audio_path: str) -> None:
    from src.l1_preprocess import preprocess
    from src.l2_diarize import diarize
    from src.l3_asr import transcribe

    print(f"\n{'='*60}")
    print(f"SAMPLE: {audio_path}")
    print("=" * 60)

    wav_path = preprocess(audio_path)
    gc.collect()
    print(f"[L1 done] peak RSS: {peak_rss_mb():.0f} MB  →  {wav_path}")

    segments = diarize(wav_path)
    gc.collect()
    print(f"[L2 done] peak RSS: {peak_rss_mb():.0f} MB  →  {len(segments)} segments")

    turns = transcribe(wav_path, segments)
    gc.collect()
    print(f"[L3 done] peak RSS: {peak_rss_mb():.0f} MB  →  {len(turns)} turns")

    print("\nSPEAKER-ATTRIBUTED TRANSCRIPT:")
    for t in turns:
        print(f"  [{t.start:6.2f}s–{t.end:6.2f}s] {t.speaker_role:7s}: {t.text}")


def main() -> None:
    missing = [p for p in SAMPLES if not os.path.exists(p)]
    if missing:
        sys.exit(f"Missing sample files: {missing}\nRun data extraction first.")

    for sample in SAMPLES:
        run_sample(sample)

    print("\nPhase A complete.")


if __name__ == "__main__":
    main()
