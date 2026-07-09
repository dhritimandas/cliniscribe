"""Classify drug-keyword misses on the drug bench: what could recovery fix?

For every gold drug keyword missed in the RAW hypothesis, classify:

  cross_script — the drug is present in the hypothesis in the OTHER script
                 (gold Devanagari 'एंटीबायोटिक्स' vs hyp Latin 'antibiotics',
                 or vice versa). Captured by ASR; a scoring/normalization
                 symmetry gap, not an acoustic miss.
  distorted    — a phonetically similar token sequence exists in the
                 hypothesis (difflib ratio >= DISTORT_FUZZ after rough
                 transliteration). Potentially recoverable by the normalizer.
  true_drop    — no trace at all. Only better audio can fix these.

Uses the dev split ONLY (rows 10-44 → part files' clips with row < 45 are
approximated by taking dev clips from drug_bench_a.json plus the first 10 of
drug_bench_b.json); the holdout (last 15 clips) is never analysed here —
normalizer changes must be validated on it blind (advisor requirement).

Run:  PYTHONPATH=. python eval/drug_miss_analysis.py
"""

import difflib
import json
import re
import unicodedata

from eval.metrics import keyword_hits

DEV_A = "outputs/drug_bench_a.json"   # rows 10-34 (all dev)
DEV_B = "outputs/drug_bench_b.json"   # rows 35-59 (first 10 dev, last 15 HOLDOUT)
DEV_B_COUNT = 10
DISTORT_FUZZ = 0.55

_DEVA_RE = re.compile(r"[ऀ-ॿ]")

# Rough Devanagari→Latin fold for phonetic comparison (analysis only —
# the production normalizer has its own tiers; this is a coarse yardstick).
_TRANSLIT = {
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


def _fold(text: str) -> str:
    """Coarse phonetic fold: Devanagari→Latin, lowercase, alnum+space only."""
    out = []
    for ch in unicodedata.normalize("NFC", text):
        out.append(_TRANSLIT.get(ch, ch))
    s = "".join(out).lower()
    return re.sub(r"[^a-z0-9 ]", "", s)


def _windows(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)] or []


def classify(gold: str, hyp: str) -> tuple[str, str]:
    """Return (class, evidence) for one missed gold keyword."""
    gold_fold = _fold(gold)
    hyp_fold = _fold(hyp)
    if not gold_fold.strip():
        return "unfoldable", ""
    # cross_script: folded gold appears (as substring) in folded hyp
    if gold_fold.strip() and gold_fold.strip() in hyp_fold:
        return "cross_script", gold_fold.strip()
    # distorted: best fuzzy window over hyp tokens
    gt = gold_fold.split()
    ht = hyp_fold.split()
    best, best_win = 0.0, ""
    for n in range(max(1, len(gt) - 1), len(gt) + 2):
        for win in _windows(ht, n):
            r = difflib.SequenceMatcher(None, gold_fold, win).ratio()
            if r > best:
                best, best_win = r, win
    if best >= DISTORT_FUZZ:
        return "distorted", f"{best_win!r} (ratio {best:.2f})"
    return "true_drop", f"best {best_win!r} (ratio {best:.2f})"


def main() -> None:
    clips = json.load(open(DEV_A, encoding="utf-8"))["per_clip"]
    try:
        clips += json.load(open(DEV_B, encoding="utf-8"))["per_clip"][:DEV_B_COUNT]
    except FileNotFoundError:
        pass

    counts: dict[str, int] = {}
    total_present = 0
    for clip in clips:
        kws = clip["drug_keywords"]
        if not kws:
            continue
        ref, hyp = clip["reference"], clip["raw_hypothesis"]
        present, _ = keyword_hits(ref, hyp, kws)
        total_present += present
        for kw in kws:
            p, m = keyword_hits(ref, hyp, [kw])
            if p == 0 or m == 0:
                continue  # not in ref, or already matched raw
            cls, evidence = classify(kw, hyp)
            counts[cls] = counts.get(cls, 0) + 1
            print(f"[{cls:>12}] {clip['id'][:8]} gold={kw!r}  evidence={evidence}")

    print(f"\ndev clips analysed: {len(clips)}  drug keywords present: {total_present}")
    print("miss classes:", json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
