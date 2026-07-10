"""L3.5 normalization impact on keyword/drug WER — committed frozen bench.

Replaces the E8 headline claim (drug WER 0.818 → 0.576, measured by an ad-hoc
uncommitted 15-clip script) with a reproducible measurement: apply the real
production normalize() to the saved beam-5 hypotheses from the beam study
(the production ASR path) and score with the token-boundary scorer.

Two post-normalization scores are reported, deliberately:
  strict — keyword must match the NORMALIZED text alone. This is the honest
      "system as shipped" number: L4 sees only the normalized transcript.
  union  — keyword matches raw OR normalized text (E8's monotone-recovery
      convention). Needed to compare against the legacy E8 method, and to
      separate real regressions from representation swaps (the drug
      normalizer SUBSTITUTES Devanagari → Latin, so a correctly-transcribed
      Devanagari drug can vanish from the normalized text while its Latin
      form appears).

Run:  PYTHONPATH=. python eval/normalization_study.py
Requires outputs/beam_study.json (from eval/beam_study.py).
"""

import json
import logging
import os

from dotenv import load_dotenv

from eval.beam_study import ASR_DATASET, FROZEN_N
from eval.metrics import keyword_hits
from eval.run_eval import _keywords_from_entities
from src.l3_5_normalize import normalize
from src.types import Turn

load_dotenv()
logger = logging.getLogger(__name__)

BEAM_STUDY_PATH = "outputs/beam_study.json"
PRODUCTION_BEAM = "5"  # config.ASR_BEAM_SIZE — the shipped decode setting
RESULTS_PATH = "outputs/normalization_study.json"


def main() -> None:
    """Score keyword/drug WER on raw vs normalized bench hypotheses."""
    import pandas as pd

    with open(BEAM_STUDY_PATH, encoding="utf-8") as f:
        beam = json.load(f)
    hyps = {c["id"]: c["beams"][PRODUCTION_BEAM]["hypothesis"] for c in beam["per_clip"]}

    df = pd.read_parquet(ASR_DATASET).head(FROZEN_N)

    per_clip: list[dict] = []
    tot = {k: 0 for k in (
        "kw_present", "kw_miss_raw", "kw_miss_strict", "kw_miss_union",
        "drug_present", "drug_miss_raw", "drug_miss_strict", "drug_miss_union",
    )}

    for _, row in df.iterrows():
        clip_id = row["md5_text"]
        ref = row["text"]
        raw_hyp = hyps[clip_id]
        kw = _keywords_from_entities(row["medical_entities"])
        kw_drug = _keywords_from_entities(row["medical_entities"], drug_only=True)

        norm_turns = normalize([Turn(speaker_role="UNKNOWN", text=raw_hyp, start=0.0, end=0.0)])
        norm_hyp = " ".join(t.text for t in norm_turns).strip()

        rec = {"id": clip_id, "raw_hypothesis": raw_hyp, "normalized_hypothesis": norm_hyp}
        for label, kws in (("kw", kw), ("drug", kw_drug)):
            present, miss_raw = keyword_hits(ref, raw_hyp, kws)
            _, miss_strict = keyword_hits(ref, norm_hyp, kws)
            # union: found in raw OR normalized → missed only if missed in both
            _, miss_norm_only = keyword_hits(ref, f"{raw_hyp} {norm_hyp}", kws)
            miss_union = miss_norm_only
            rec[label] = {
                "present": present,
                "missed_raw": miss_raw,
                "missed_strict": miss_strict,
                "missed_union": miss_union,
            }
            tot[f"{label}_present"] += present
            tot[f"{label}_miss_raw"] += miss_raw
            tot[f"{label}_miss_strict"] += miss_strict
            tot[f"{label}_miss_union"] += miss_union
        per_clip.append(rec)
        logger.info(
            "%s kw %d/%d→%d/%d(strict) drug %d/%d→%d/%d(strict)",
            clip_id[:8],
            rec["kw"]["missed_raw"], rec["kw"]["present"],
            rec["kw"]["missed_strict"], rec["kw"]["present"],
            rec["drug"]["missed_raw"], rec["drug"]["present"],
            rec["drug"]["missed_strict"], rec["drug"]["present"],
        )

    def _rate(miss: int, present: int) -> float:
        return round(miss / present, 4) if present else 0.0

    summary = {
        "beam": PRODUCTION_BEAM,
        "keyword_wer": {
            "raw": _rate(tot["kw_miss_raw"], tot["kw_present"]),
            "normalized_strict": _rate(tot["kw_miss_strict"], tot["kw_present"]),
            "normalized_union": _rate(tot["kw_miss_union"], tot["kw_present"]),
            "keywords_total": tot["kw_present"],
        },
        "drug_keyword_wer": {
            "raw": _rate(tot["drug_miss_raw"], tot["drug_present"]),
            "normalized_strict": _rate(tot["drug_miss_strict"], tot["drug_present"]),
            "normalized_union": _rate(tot["drug_miss_union"], tot["drug_present"]),
            "drug_keywords_total": tot["drug_present"],
        },
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_clip": per_clip}, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Full results: {RESULTS_PATH}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    main()
