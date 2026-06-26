"""Full CliniScribe pipeline: audio → draft prescription PDF.

Stages execute sequentially. Each model is loaded, used, and released before
the next stage begins — never hold ASR and LLM in memory simultaneously.
"""

import gc
import logging
import sys

from dotenv import load_dotenv

from src.l1_preprocess import preprocess
from src.l2_diarize import diarize
from src.l3_asr import transcribe
from src.l3_5_normalize import normalize
from src.l4_extract import extract
from src.l5_render import render

load_dotenv()
logger = logging.getLogger(__name__)


def run(in_path: str) -> str:
    """Run the full pipeline on an audio file and return the PDF path.

    Args:
        in_path: Path to the input audio file.

    Returns:
        Path to the generated draft prescription PDF.
    """
    logger.info("L1: preprocessing %s", in_path)
    wav_path = preprocess(in_path)
    gc.collect()

    logger.info("L2: diarizing %s", wav_path)
    segments = diarize(wav_path)
    gc.collect()

    logger.info("L3: transcribing %d segments", len(segments))
    turns = transcribe(wav_path, segments)
    gc.collect()

    logger.info("L3.5: normalizing %d turns", len(turns))
    turns = normalize(turns)
    gc.collect()

    logger.info("L4: extracting clinical entities")
    note = extract(turns)
    gc.collect()

    logger.info("L5: rendering prescription PDF")
    pdf_path = render(note)
    gc.collect()

    logger.info("Done: %s", pdf_path)
    return pdf_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    if len(sys.argv) != 2:
        print("Usage: python src/pipeline.py <audio_path>")
        sys.exit(1)
    print(run(sys.argv[1]))
