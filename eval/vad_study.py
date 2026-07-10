"""VAD-filter reconciliation study: silence-hallucination defense gate.

Diarized segments already exclude most non-speech (pyannote skips silence),
so Silero VAD (faster-whisper's vad_filter=True) may be redundant on clean
segments — but it protects against a diarized segment that itself spans a
silence stretch, where Whisper is known to hallucinate text. This study
measures whether turning vad_filter on regresses accuracy on the frozen
10-clip Hindi bench before it can ship (see src.config.ASR_VAD_FILTER).

Method: per clip, L1 preprocess and L2 diarize run ONCE (identical across VAD
settings, using a preloaded pyannote pipeline); L3 transcribe runs once per
VAD setting via the real production transcribe() (preloaded WhisperModel),
overriding src.config.ASR_VAD_FILTER the same way beam_study.py overrides
ASR_BEAM_SIZE — transcribe() reads the config value at call time, so this
measures exactly the code path that would ship. Results are written
incrementally to outputs/vad_study.json so partial progress survives
interruption.

Run:  PYTHONPATH=. python eval/vad_study.py
Parallel/resume: --rows 4 7 --out outputs/vad_study_a.json runs bench rows
[4,7) into a part file, skipping clips already present in it; merge part files
with --merge afterwards.
"""

import json
import logging
import os
import tempfile

from dotenv import load_dotenv

from eval.metrics import corpus_word_error_rate, keyword_hits, word_error_rate
from eval.run_eval import _keywords_from_entities
from src import config
from src.l1_preprocess import preprocess
from src.l2_diarize import diarize
from src.l3_asr import transcribe

load_dotenv()
logger = logging.getLogger(__name__)

ASR_DATASET = "eka-medical-asr-dataset/hi/test-00000.parquet"
FROZEN_N = 10  # must match eval/frozen_set_asr.json (deterministic first-N rows)
VAD_SETTINGS = (False, True)
RESULTS_PATH = "outputs/vad_study.json"


def _summarise(results: dict) -> None:
    """Attach micro-averaged per-setting summary to a results dict in place."""
    results["summary"] = {}
    for vad in VAD_SETTINGS:
        v = str(vad)
        kp = sum(c["vad"][v]["kw_present"] for c in results["per_clip"])
        km = sum(c["vad"][v]["kw_missed"] for c in results["per_clip"])
        dp = sum(c["vad"][v]["drug_present"] for c in results["per_clip"])
        dm = sum(c["vad"][v]["drug_missed"] for c in results["per_clip"])
        refs = [c["reference"] for c in results["per_clip"]]
        hyps = [c["vad"][v]["hypothesis"] for c in results["per_clip"]]
        results["summary"][v] = {
            "corpus_wer": round(corpus_word_error_rate(refs, hyps), 4),
            "keyword_wer_micro": round(km / kp, 4) if kp else 0.0,
            "drug_keyword_wer_micro": round(dm / dp, 4) if dp else 0.0,
            "keywords_total": kp,
            "drug_keywords_total": dp,
        }


def merge(part_paths: list[str], out_path: str = RESULTS_PATH) -> None:
    """Merge part files (from --rows runs) into one results file with summary."""
    merged: dict = {"vad_settings": list(VAD_SETTINGS), "per_clip": []}
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
    """Run the VAD study on bench rows [row_start, row_end) with resume.

    Models are loaded ONCE and passed to diarize()/transcribe() — on 10-20 s
    clips, per-clip model loading dominates runtime. Decode behavior is
    otherwise identical to production (same functions, same parameters).
    """
    import pandas as pd
    import torch
    from faster_whisper import WhisperModel
    from pyannote.audio import Pipeline

    df = pd.read_parquet(ASR_DATASET).head(FROZEN_N)
    results: dict = {"vad_settings": list(VAD_SETTINGS), "per_clip": []}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            results = json.load(f)
        results.pop("summary", None)  # recomputed at the end
    done_ids = {c["id"] for c in results["per_clip"]}

    def _flush() -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    hf_token = os.environ.get("HF_TOKEN")
    diarize_pipeline = Pipeline.from_pretrained(config.DIARIZE_MODEL, token=hf_token)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    diarize_pipeline.to(device)
    whisper = WhisperModel(config.ASR_MODEL, device="cpu", compute_type="int8")
    logger.info("Models resident: %s + %s", config.DIARIZE_MODEL, config.ASR_MODEL)

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
            segments = diarize(wav_path, pipeline=diarize_pipeline)

            clip_rec: dict = {"id": row["md5_text"], "reference": ref, "vad": {}}
            for vad in VAD_SETTINGS:
                config.ASR_VAD_FILTER = vad
                turns = transcribe(wav_path, segments, model=whisper)
                hyp = " ".join(t.text for t in turns).strip()
                p, m = keyword_hits(ref, hyp, kw)
                dp, dm = keyword_hits(ref, hyp, kw_drug)
                clip_rec["vad"][str(vad)] = {
                    "wer": round(word_error_rate(ref, hyp), 4),
                    "kw_present": p,
                    "kw_missed": m,
                    "drug_present": dp,
                    "drug_missed": dm,
                    "hypothesis": hyp,
                }
                logger.info(
                    "[%d/%d] vad=%s WER=%.3f kw=%d/%d drug=%d/%d",
                    i + 1, FROZEN_N, vad,
                    clip_rec["vad"][str(vad)]["wer"], m, p, dm, dp,
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
    parser = argparse.ArgumentParser(description="VAD-filter reconciliation study.")
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
