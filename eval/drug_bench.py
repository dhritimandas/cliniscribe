"""Larger frozen drug-keyword bench: normalization recovery at scale.

The committed 10-clip bench has only 7 drug terms, dominated by TRUE-DROP
acoustic misses — it cannot measure normalization recovery (0/7 recovered,
see eval/normalization_study.py). The source dataset has 241/320 clips with
drug keywords (809 instances, 791 Devanagari-rendered — the recoverable
class). This bench extends coverage deterministically: rows [ROW_START,
ROW_END) of the Hindi test parquet, i.e. the next 50 rows after the frozen
10 — no cherry-picking.

Two phases, cached between:
  asr   — L1→L2→L3 production path per clip; raw hypotheses cached to the
          results file (the expensive phase, run once).
  score — L3.5 normalize() applied to cached hypotheses; drug keywords
          scored raw vs normalized (strict + union, token-boundary scorer).
          Re-run this phase freely while iterating on the normalizer.

Run:  PYTHONPATH=. python eval/drug_bench.py           # asr (resumable) + score
      PYTHONPATH=. python eval/drug_bench.py --score-only
"""

import json
import logging
import os
import re
import tempfile
import unicodedata

from dotenv import load_dotenv

from eval.metrics import keyword_hits
from eval.run_eval import _keywords_from_entities
from src.l1_preprocess import preprocess
from src.l2_diarize import diarize
from src.l3_asr import transcribe
from src.types import Turn

load_dotenv()
logger = logging.getLogger(__name__)

ASR_DATASET = "eka-medical-asr-dataset/hi/test-00000.parquet"
ROW_START, ROW_END = 10, 60  # deterministic: next 50 rows after the frozen 10
RESULTS_PATH = "outputs/drug_bench.json"


def run_asr(
    row_start: int = ROW_START, row_end: int = ROW_END, out_path: str = RESULTS_PATH
) -> dict:
    """Transcribe bench rows via the production path; cache raw hypotheses.

    Models are loaded ONCE and passed to diarize()/transcribe() — on 10-20 s
    clips, per-clip model loading dominates runtime. Decode behavior is
    identical to production (same functions, same parameters).
    """
    import pandas as pd
    from faster_whisper import WhisperModel
    from pyannote.audio import Pipeline

    from src import config

    results: dict = {"rows": [row_start, row_end], "per_clip": []}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            results = json.load(f)
    done = {c["id"] for c in results["per_clip"]}

    import torch

    hf_token = os.environ.get("HF_TOKEN")
    pipeline = Pipeline.from_pretrained(config.DIARIZE_MODEL, token=hf_token)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    pipeline.to(device)
    whisper = WhisperModel(config.ASR_MODEL, device="cpu", compute_type="int8")
    logger.info("Models resident: %s + %s", config.DIARIZE_MODEL, config.ASR_MODEL)

    df = pd.read_parquet(ASR_DATASET)
    for i in range(row_start, row_end):
        row = df.iloc[i]
        if row["md5_text"] in done:
            continue
        kw_drug = _keywords_from_entities(row["medical_entities"], drug_only=True)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(row["audio"])
            tmp = f.name
        try:
            wav = preprocess(tmp)
            segments = diarize(wav, pipeline=pipeline)
            turns = transcribe(wav, segments, model=whisper)
            hyp = " ".join(t.text for t in turns).strip()
        finally:
            os.unlink(tmp)
        results["per_clip"].append(
            {"id": row["md5_text"], "reference": row["text"],
             "drug_keywords": kw_drug, "raw_hypothesis": hyp}
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("[row %d] %s: %d drug keywords, hyp %d chars",
                    i, row["md5_text"][:8], len(kw_drug), len(hyp))
    return results


# ── Script-symmetric fold (scorer correctness, not recovery) ────────────────
# Gold labels and hypotheses mix scripts and spacing for the SAME drug:
# gold 'तेंडोलाईफ' vs hyp 'टेंडो लाइफ'; gold 'एंटीबायोटिक्स' vs hyp 'antibiotics'.
# The ASR captured the drug — a script/spacing-blind scorer must not call it a
# miss. Fold = coarse Devanagari→Latin + lowercase + drop non-alnum; matching
# is EXACT equality of despaced folds over hypothesis token windows. No fuzzy
# matching here: fuzz is how the substring false-positive bug class returns.
_FOLD_MAP = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "च": "ch", "छ": "chh",
    "ज": "j", "झ": "jh", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p",
    "फ": "f", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r",
    "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "ज़": "z", "फ़": "f", "ा": "a", "ि": "i", "ी": "i", "ु": "u",
    "ू": "u", "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ं": "n",
    "अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "्": "",
}
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]")


def _fold(text: str) -> str:
    """Coarse phonetic fold: Devanagari→Latin, lowercase, alnum+space only."""
    folded = "".join(_FOLD_MAP.get(ch, ch) for ch in unicodedata.normalize("NFC", text))
    return _NON_ALNUM_RE.sub("", folded.lower())


def _folded_match(gold: str, hyp: str) -> bool:
    """True if gold's despaced fold equals some hyp token-window's despaced fold."""
    gold_key = _fold(gold).replace(" ", "")
    if not gold_key:
        return False
    hyp_tokens = _fold(hyp).split()
    max_n = min(len(hyp_tokens), len(gold.split()) + 2)
    for n in range(1, max_n + 1):
        for i in range(len(hyp_tokens) - n + 1):
            if "".join(hyp_tokens[i : i + n]) == gold_key:
                return True
    return False


def score(results: dict) -> None:
    """Score drug keywords raw vs normalized on cached hypotheses."""
    from src.l3_5_normalize import normalize

    tot = {"present": 0, "miss_raw": 0, "miss_folded": 0,
           "miss_strict": 0, "miss_union": 0}
    for clip in results["per_clip"]:
        ref, raw_hyp = clip["reference"], clip["raw_hypothesis"]
        kws = clip["drug_keywords"]
        norm_turns = normalize(
            [Turn(speaker_role="UNKNOWN", text=raw_hyp, start=0.0, end=0.0)]
        )
        norm_hyp = " ".join(t.text for t in norm_turns).strip()
        present, miss_raw = keyword_hits(ref, raw_hyp, kws)
        _, miss_strict = keyword_hits(ref, norm_hyp, kws)
        _, miss_union = keyword_hits(ref, f"{raw_hyp} {norm_hyp}", kws)
        # folded: raw-or-normalized hyp, script/spacing-symmetric exact match
        miss_folded = 0
        for kw in kws:
            p, m = keyword_hits(ref, f"{raw_hyp} {norm_hyp}", [kw])
            if p and m and not _folded_match(kw, f"{raw_hyp} {norm_hyp}"):
                miss_folded += 1
        clip["normalized_hypothesis"] = norm_hyp
        clip["drug"] = {"present": present, "missed_raw": miss_raw,
                        "missed_folded": miss_folded,
                        "missed_strict": miss_strict, "missed_union": miss_union}
        for k, v in (("present", present), ("miss_raw", miss_raw),
                     ("miss_folded", miss_folded),
                     ("miss_strict", miss_strict), ("miss_union", miss_union)):
            tot[k] += v

    def _rate(m: int) -> float:
        return round(m / tot["present"], 4) if tot["present"] else 0.0

    results["summary"] = {
        "clips": len(results["per_clip"]),
        "drug_keywords_total": tot["present"],
        "drug_wer_raw": _rate(tot["miss_raw"]),
        "drug_wer_normalized_strict": _rate(tot["miss_strict"]),
        "drug_wer_normalized_union": _rate(tot["miss_union"]),
        "drug_wer_folded": _rate(tot["miss_folded"]),
        "recovered_strict": tot["miss_raw"] - tot["miss_strict"],
        "recovered_union": tot["miss_raw"] - tot["miss_union"],
        "recovered_folded": tot["miss_raw"] - tot["miss_folded"],
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results["summary"], indent=2))
    print(f"Full results: {RESULTS_PATH}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    parser = argparse.ArgumentParser(description="Larger frozen drug-keyword bench.")
    parser.add_argument("--score-only", action="store_true",
                        help="re-score cached hypotheses (skip ASR)")
    parser.add_argument("--rows", nargs=2, type=int, metavar=("START", "END"),
                        default=[ROW_START, ROW_END], help="dataset row range [START, END)")
    parser.add_argument("--out", default=RESULTS_PATH, help="results/part file path")
    parser.add_argument("--merge", nargs="+", metavar="PART",
                        help="merge part files into --out, then score")
    args = parser.parse_args()

    if args.merge:
        merged: dict = {"rows": [ROW_START, ROW_END], "per_clip": []}
        seen: set[str] = set()
        for p in args.merge:
            with open(p, encoding="utf-8") as f:
                part = json.load(f)
            for clip in part["per_clip"]:
                if clip["id"] not in seen:
                    seen.add(clip["id"])
                    merged["per_clip"].append(clip)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        score(merged)
    elif args.score_only:
        with open(args.out, encoding="utf-8") as f:
            cached = json.load(f)
        score(cached)
    else:
        score(run_asr(args.rows[0], args.rows[1], args.out))
