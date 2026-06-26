"""Run KARMA evaluation on EkaCare datasets for a given pipeline stage."""

import logging

logger = logging.getLogger(__name__)


def run_asr_eval() -> None:
    """Evaluate L3 ASR on ekacare/eka-medical-asr-evaluation-dataset.

    Scores WER and keyword WER against reference transcripts.
    Prints per-sample and aggregate results.
    """
    raise NotImplementedError


def run_extraction_eval() -> None:
    """Evaluate L4 extraction on ekacare/clinical_note_generation_dataset.

    Scores field-level extraction accuracy against ground-truth JSON.
    Prints per-field and aggregate results.
    """
    raise NotImplementedError


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    run_asr_eval()
