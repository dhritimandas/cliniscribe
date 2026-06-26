"""Phase B proof script: L3.5 normalize + L4 extract on 3 Phase-A samples.

Picks up from the preprocessed 16kHz WAVs in outputs/ (produced by Phase A).
Runs L2 diarize → L3 transcribe → L3.5 normalize → L4 extract per sample,
then prints the structured ClinicalNote JSON.

Prerequisites:
  source .venv/bin/activate
  ollama serve                              # terminal 1
  ollama pull qwen2.5:3b-instruct           # once
  python run_phase_b.py
"""

import dataclasses
import gc
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s — %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("phase_b")

PREPROCESSED_WAVS = [
    "outputs/sample_00_16k.wav",
    "outputs/sample_01_16k.wav",
    "outputs/sample_02_16k.wav",
]


def _check_ollama() -> None:
    try:
        import ollama

        ollama.list()
    except Exception as exc:
        sys.exit(
            f"Ollama not reachable: {exc}\n"
            "Run: ollama serve\n"
            "Then: ollama pull qwen2.5:3b-instruct"
        )


def _note_to_dict(note) -> dict:
    """Serialise ClinicalNote dataclass to a plain dict for JSON output."""
    return dataclasses.asdict(note)


def run_sample(wav_path: str) -> None:
    from src.l2_diarize import diarize
    from src.l3_5_normalize import normalize
    from src.l3_asr import transcribe
    from src.l4_extract import extract

    sample_name = Path(wav_path).stem
    print(f"\n{'=' * 64}")
    print(f"SAMPLE: {wav_path}")
    print("=" * 64)

    logger.info("L2: diarizing %s", wav_path)
    segments = diarize(wav_path)
    gc.collect()
    logger.info("  → %d segments", len(segments))

    logger.info("L3: transcribing")
    turns = transcribe(wav_path, segments)
    gc.collect()
    logger.info("  → %d turns", len(turns))

    print("\nRAW TRANSCRIPT:")
    for t in turns:
        print(f"  [{t.start:6.2f}s–{t.end:6.2f}s] {t.speaker_role:7s}: {t.text}")

    logger.info("L3.5: normalizing")
    norm_turns = normalize(turns)
    gc.collect()

    print("\nNORMALIZED TRANSCRIPT:")
    for t in norm_turns:
        print(f"  [{t.start:6.2f}s–{t.end:6.2f}s] {t.speaker_role:7s}: {t.text}")

    logger.info("L4: extracting clinical entities")
    note = extract(norm_turns)
    gc.collect()

    print("\nCLINICAL NOTE (JSON):")
    print(json.dumps(_note_to_dict(note), indent=2, ensure_ascii=False))


def main() -> None:
    missing = [p for p in PREPROCESSED_WAVS if not Path(p).exists()]
    if missing:
        sys.exit(
            f"Missing preprocessed WAVs: {missing}\n"
            "Run run_phase_a.py first to generate them."
        )

    _check_ollama()

    for wav in PREPROCESSED_WAVS:
        run_sample(wav)

    print("\nPhase B complete.")


if __name__ == "__main__":
    main()
