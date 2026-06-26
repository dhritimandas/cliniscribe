"""Run KARMA evaluation on EkaCare datasets for a given pipeline stage.

The ASR baseline runs the real L1->L2->L3 pipeline on each clip — not a
shortcut full-file transcription — so the number reflects the system as built,
including diarization-driven segment slicing. That is exactly the behavior the
"daily three times" code-switch error lives in, so any later ASR fix is scored
against an honest before-number on the same frozen set.
"""

import json
import logging
import os
import tempfile

from dotenv import load_dotenv

from eval.metrics import corpus_word_error_rate, keyword_hits, keyword_wer, word_error_rate
from src.l1_preprocess import preprocess
from src.l2_diarize import diarize
from src.l3_asr import transcribe

load_dotenv()
logger = logging.getLogger(__name__)

ASR_DATASET = "eka-medical-asr-dataset/hi/test-00000.parquet"
FROZEN_N = 10  # deterministic first-N rows; small for tractable CPU runtime
FROZEN_SET_PATH = "eval/frozen_set_asr.json"
RESULTS_PATH = "outputs/asr_baseline.json"
DRUG_CATEGORIES = {"drugs"}


def _keywords_from_entities(raw: str, *, drug_only: bool = False) -> list[str]:
    """Extract gold keyword surface forms from the medical_entities JSON column.

    Each entity is [surface_form, category, spans]. Duplicates (the dataset
    repeats some) are removed while preserving order.
    """
    entities = json.loads(raw)
    out: list[str] = []
    seen: set[str] = set()
    for surface, category, _spans in entities:
        if drug_only and category not in DRUG_CATEGORIES:
            continue
        if surface not in seen:
            seen.add(surface)
            out.append(surface)
    return out


def _pipeline_transcribe(audio_bytes: bytes) -> str:
    """Run L1->L2->L3 on one clip's raw bytes; return concatenated hypothesis."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        wav_path = preprocess(tmp_path)
        segments = diarize(wav_path)
        turns = transcribe(wav_path, segments)
    finally:
        os.unlink(tmp_path)
    return " ".join(t.text for t in turns).strip()


def run_asr_eval() -> None:
    """Evaluate L3 ASR on the frozen Hindi ASR set; print and save results.

    Scores corpus WER, plus micro-averaged keyword and drug-keyword error
    rates (the patient-safety metrics). Writes the frozen sample IDs for
    reproducibility and a full per-sample results JSON.
    """
    import pandas as pd

    df = pd.read_parquet(ASR_DATASET).head(FROZEN_N)
    os.makedirs(os.path.dirname(FROZEN_SET_PATH), exist_ok=True)
    with open(FROZEN_SET_PATH, "w") as f:
        json.dump({"dataset": ASR_DATASET, "md5_text": df["md5_text"].tolist()}, f, indent=2)

    per_sample: list[dict] = []
    refs: list[str] = []
    hyps: list[str] = []
    kw_present = kw_missed = drug_present = drug_missed = 0

    for i, (_, row) in enumerate(df.iterrows()):
        ref = row["text"]
        hyp = _pipeline_transcribe(row["audio"])
        kw = _keywords_from_entities(row["medical_entities"])
        kw_drug = _keywords_from_entities(row["medical_entities"], drug_only=True)

        p, m = keyword_hits(ref, hyp, kw)
        dp, dm = keyword_hits(ref, hyp, kw_drug)
        kw_present += p
        kw_missed += m
        drug_present += dp
        drug_missed += dm

        refs.append(ref)
        hyps.append(hyp)
        per_sample.append(
            {
                "id": row["md5_text"],
                "wer": round(word_error_rate(ref, hyp), 4),
                "keyword_wer": round(keyword_wer(ref, hyp, kw), 4),
                "reference": ref,
                "hypothesis": hyp,
                "keywords": kw,
            }
        )
        logger.info("[%d/%d] WER=%.3f  kwWER=%.3f", i + 1, FROZEN_N, per_sample[-1]["wer"], per_sample[-1]["keyword_wer"])

    summary = {
        "n": len(per_sample),
        "corpus_wer": round(corpus_word_error_rate(refs, hyps), 4),
        "keyword_wer_micro": round(kw_missed / kw_present, 4) if kw_present else 0.0,
        "drug_keyword_wer_micro": round(drug_missed / drug_present, 4) if drug_present else 0.0,
        "keywords_total": kw_present,
        "drug_keywords_total": drug_present,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({"summary": summary, "per_sample": per_sample}, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"ASR BASELINE — {summary['n']} clips ({ASR_DATASET})")
    print("=" * 60)
    print(f"Corpus WER            : {summary['corpus_wer']:.3f}")
    print(f"Keyword WER (micro)   : {summary['keyword_wer_micro']:.3f}  over {summary['keywords_total']} keywords")
    print(f"Drug Keyword WER      : {summary['drug_keyword_wer_micro']:.3f}  over {summary['drug_keywords_total']} drug terms")
    print(f"\nFull results: {RESULTS_PATH}")


def run_extraction_eval() -> None:
    """Evaluate L4 extraction on ekacare/clinical_note_generation_dataset.

    Scores field-level extraction accuracy against ground-truth JSON.
    Prints per-field and aggregate results.
    """
    raise NotImplementedError


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    run_asr_eval()
