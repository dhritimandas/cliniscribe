"""B1 split study: attribute symptom/vital misses to extraction vs post-extraction.

For every missed symptom_name / body_vital_sign_* rubric criterion on the
English frozen samples, classify:

  extraction   — the gold value appears in the source transcript but NOWHERE
                 in the extracted note: the model failed to extract it.
  routing      — the gold value appears SOMEWHERE in the note (wrong field or
                 a string the eval matcher rejects): extracted but lost after
                 extraction. (The eval never reads the PDF, so any L5-side
                 loss is a subset of this class.)
  not_a_bug    — the gold value does not appear in the source transcript:
                 the rubric references chart data never spoken; nothing any
                 pipeline stage could extract. These cases are printed for
                 MANUAL REVIEW — transcript-scale fuzzy matching is weaker
                 than field-scale matching, so this class is spot-checked,
                 not trusted blindly.

Reads outputs/extraction_baseline.json (must contain full notes — produced by
eval/run_eval.py after the note-dump change). Pure analysis: no model calls.

Run:  PYTHONPATH=. python eval/split_study.py
"""

import json
import re

RESULTS_PATH = "outputs/extraction_baseline.json"
TARGET_CATS = {"symptom_name", "body_vital_sign_name", "body_vital_sign_value"}

_PUNCT_RE = re.compile(r"[^a-z0-9 ]")


def _normalise(s: object) -> str:
    return _PUNCT_RE.sub("", str(s).lower().strip())


def _extract_quoted(criterion: str) -> str:
    m = re.search(r"'([^']+)'", criterion)
    return m.group(1) if m else criterion


def _all_note_values(note: dict) -> list[str]:
    """Flatten every string value in the note across all fields."""
    vals: list[str] = []
    for key, v in note.items():
        if key == "low_confidence_fields":
            continue
        if isinstance(v, str):
            vals.append(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    vals.append(item)
                elif isinstance(item, dict):
                    vals.extend(str(x) for x in item.values() if x)
    return vals


def _gold_in(gold: str, candidates: list[str]) -> bool:
    """Substring or all-gold-tokens-covered match against any candidate."""
    g = _normalise(gold)
    if not g:
        return False
    gt = [t for t in g.split() if len(t) >= 3] or g.split()
    for c in candidates:
        cn = _normalise(c)
        if not cn:
            continue
        if g in cn or cn in g:
            return True
        ct = set(cn.split())
        if gt and all(
            any(t == w or (len(t) >= 3 and len(w) >= 3 and (t.startswith(w) or w.startswith(t)))
                for w in ct)
            for t in gt
        ):
            return True
    return False


def main() -> None:
    import pandas as pd

    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)

    # Transcripts come from the dataset (results store only the note).
    dfs = [
        pd.read_parquet(f"eka-clinical-note-generation-dataset/test-0000{i}.parquet")
        for i in (0, 1)
    ]
    df = pd.concat(dfs, ignore_index=True)

    split = {"extraction": [], "routing": [], "not_a_bug": []}
    for sample in results["per_sample"]:
        if sample["language"] != "english":
            continue
        transcript = str(df.loc[sample["idx"], "text"])
        note_vals = _all_note_values(sample["note"])
        for crit in sample["criteria"]:
            if crit["cat"] not in TARGET_CATS or crit["status"] != "missed":
                continue
            gold = _extract_quoted(crit["criterion"])
            entry = {"idx": sample["idx"], "cat": crit["cat"], "gold": gold}
            if _gold_in(gold, note_vals):
                split["routing"].append(entry)
            elif _gold_in(gold, [transcript]):
                split["extraction"].append(entry)
            else:
                split["not_a_bug"].append(entry)

    total = sum(len(v) for v in split.values())
    print(f"Missed symptom/vital criteria on EN samples: {total}")
    for cls, entries in split.items():
        print(f"\n== {cls}: {len(entries)} "
              f"({100 * len(entries) / total:.0f}%)" if total else cls)
        for e in entries:
            print(f"  idx={e['idx']:>3} [{e['cat']}] {e['gold']!r}")

    print("\nNOTE: not_a_bug entries above REQUIRE manual spot-check — confirm the "
          "term truly is absent from the transcript before closing them.")

    out = RESULTS_PATH.replace("extraction_baseline", "split_study")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
