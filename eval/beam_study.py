"""Beam-size reconciliation study: drug/keyword WER at beam 1 vs 5 (frozen bench).

Production l3_asr.py used beam_size=5 while the E8 eval numbers were taken at
beam_size=1. This study measures both on the frozen 10-clip Hindi bench with
the token-boundary scorer, so one value can be pinned in src.config.ASR_BEAM_SIZE
for BOTH pipeline and eval.

Method: per clip, L1 preprocess and L2 diarize run ONCE (identical across beam
settings); L3 transcribe runs once per beam via the real production transcribe()
with src.config.ASR_BEAM_SIZE overridden. Results are written incrementally to
outputs/beam_study.json so partial progress survives interruption.

Run:  PYTHONPATH=. python eval/beam_study.py
Parallel/resume: --rows 4 7 --out outputs/beam_study_a.json runs bench rows
[4,7) into a part file, skipping clips already present in it; merge part files
with --merge afterwards.
"""

import json
import logging
import os
import tempfile

from dotenv import load_dotenv

from eval.metrics import keyword_hits, word_error_rate
from eval.run_eval import _keywords_from_entities
from src import config
from src.l1_preprocess import preprocess
from src.l2_diarize import diarize
from src.l3_asr import transcribe

load_dotenv()
logger = logging.getLogger(__name__)

ASR_DATASET = "eka-medical-asr-dataset/hi/test-00000.parquet"
FROZEN_N = 10  # must match eval/frozen_set_asr.json (deterministic first-N rows)
BEAM_SIZES = (1, 5)
RESULTS_PATH = "outputs/beam_study.json"


def _summarise(results: dict) -> None:
    """Attach micro-averaged per-beam summary to a results dict in place."""
    results["summary"] = {}
    for beam in BEAM_SIZES:
        b = str(beam)
        kp = sum(c["beams"][b]["kw_present"] for c in results["per_clip"])
        km = sum(c["beams"][b]["kw_missed"] for c in results["per_clip"])
        dp = sum(c["beams"][b]["drug_present"] for c in results["per_clip"])
        dm = sum(c["beams"][b]["drug_missed"] for c in results["per_clip"])
        results["summary"][b] = {
            "keyword_wer_micro": round(km / kp, 4) if kp else 0.0,
            "drug_keyword_wer_micro": round(dm / dp, 4) if dp else 0.0,
            "keywords_total": kp,
            "drug_keywords_total": dp,
        }


def merge(part_paths: list[str], out_path: str = RESULTS_PATH) -> None:
    """Merge part files (from --rows runs) into one results file with summary."""
    merged: dict = {"beam_sizes": list(BEAM_SIZES), "per_clip": []}
    seen: set[str] = set()
    for p in part_paths:
        with open(p, encoding="utf-8") as f:
            part = json.load(f)
        for clip in part["per_clip"]:
            if clip["id"] not in seen:
                seen.add(clip["id"])
                merged["per_clip"].append(clip)
    _summarise(merged)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(json.dumps(merged["summary"], indent=2))
    print(f"Merged {len(merged['per_clip'])} clips -> {out_path}")


def main(row_start: int = 0, row_end: int = FROZEN_N, out_path: str = RESULTS_PATH) -> None:
    """Run the beam study on bench rows [row_start, row_end) with resume."""
    import pandas as pd

    df = pd.read_parquet(ASR_DATASET).head(FROZEN_N)
    results: dict = {"beam_sizes": list(BEAM_SIZES), "per_clip": []}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            results = json.load(f)
        results.pop("summary", None)  # recomputed at the end
    done_ids = {c["id"] for c in results["per_clip"]}

    def _flush() -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    for i, (_, row) in enumerate(df.iterrows()):
        if not (row_start <= i < row_end):
            continue
        if row["md5_text"] in done_ids:
            logger.info("[%d/%d] already done — skipping", i + 1, FROZEN_N)
            continue
        ref = row["text"]
        kw = _keywords_from_entities(row["medical_entities"])
        kw_drug = _keywords_from_entities(row["medical_entities"], drug_only=True)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(row["audio"])
            tmp_path = f.name
        try:
            wav_path = preprocess(tmp_path)
            segments = diarize(wav_path)

            clip_rec: dict = {"id": row["md5_text"], "beams": {}}
            for beam in BEAM_SIZES:
                config.ASR_BEAM_SIZE = beam
                turns = transcribe(wav_path, segments)
                hyp = " ".join(t.text for t in turns).strip()
                p, m = keyword_hits(ref, hyp, kw)
                dp, dm = keyword_hits(ref, hyp, kw_drug)
                clip_rec["beams"][str(beam)] = {
                    "wer": round(word_error_rate(ref, hyp), 4),
                    "kw_present": p,
                    "kw_missed": m,
                    "drug_present": dp,
                    "drug_missed": dm,
                    "hypothesis": hyp,
                }
                logger.info(
                    "[%d/%d] beam=%d WER=%.3f kw=%d/%d drug=%d/%d",
                    i + 1, FROZEN_N, beam,
                    clip_rec["beams"][str(beam)]["wer"], m, p, dm, dp,
                )
        finally:
            os.unlink(tmp_path)

        results["per_clip"].append(clip_rec)
        _flush()

    _summarise(results)
    _flush()

    print(json.dumps(results["summary"], indent=2))
    print(f"Full results: {out_path}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    parser = argparse.ArgumentParser(description="Beam-size reconciliation study.")
    parser.add_argument("--rows", nargs=2, type=int, metavar=("START", "END"),
                        default=[0, FROZEN_N], help="bench row range [START, END)")
    parser.add_argument("--out", default=RESULTS_PATH, help="results/part file path")
    parser.add_argument("--merge", nargs="+", metavar="PART",
                        help="merge part files into --out and exit")
    args = parser.parse_args()

    if args.merge:
        merge(args.merge, args.out)
    else:
        main(args.rows[0], args.rows[1], args.out)
