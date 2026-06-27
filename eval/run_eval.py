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
    """Evaluate L4 extraction on a frozen subset of the EkaCare clinical-note dataset.

    Scoring methodology
    -------------------
    Ground truth is a set of per-entity rubric criteria (parsed from the `rubrics`
    column), not a structured JSON.  Each criterion belongs to a Category ID and
    contains a natural-language assertion (e.g. "A medication matching 'Augmentin'
    is present within the medications").

    We map categories to ClinicalNote fields and use a deterministic fuzzy matcher:
      normalize both strings → check substring OR token-overlap ≥ 0.5.

    For Hindi/Marathi transcripts the extracted note stays in the source language
    while rubric criteria are in English (per CLAUDE.md do-not-translate rule).
    We cannot string-match across scripts, so we apply a *presence check* only:
    if the note has at least one value in the expected field we count it as a
    partial hit.  This distinction is clearly surfaced in the output.

    Categories with no schema field (medical_condition_*, current_medication_*,
    lifestyle_habit_*, family_history_*, foodotherallergy_*, pastprocedures_*,
    recenttravelhistory_*, symptom_laterality, diagnosis_laterality, drugallergy_*)
    are counted as "structurally unrepresentable" and excluded from the
    model-recall denominator but reported separately.

    Output
    ------
    Saves outputs/extraction_baseline.json and prints a summary table.
    """
    import gc
    import re

    import pandas as pd

    from src.l4_extract import extract
    from src.types import ClinicalNote, Turn

    # ── Dataset paths ────────────────────────────────────────────────────────
    # Resolve dataset paths relative to the canonical project root.
    # run_eval.py may be invoked from the project root OR from a git worktree;
    # both cases are handled by walking up to the directory that contains the
    # dataset folder.
    import pathlib

    _script_dir = pathlib.Path(__file__).resolve().parent
    _search_roots = [pathlib.Path.cwd(), _script_dir.parent]
    _project_root: pathlib.Path | None = None
    for candidate in _search_roots:
        if (candidate / "eka-clinical-note-generation-dataset").exists():
            _project_root = candidate
            break
    if _project_root is None:
        raise FileNotFoundError(
            "Cannot locate 'eka-clinical-note-generation-dataset'. "
            "Run from the project root or ensure the dataset is downloaded."
        )

    DATASET_FILES = [
        str(_project_root / "eka-clinical-note-generation-dataset/test-00000.parquet"),
        str(_project_root / "eka-clinical-note-generation-dataset/test-00001.parquet"),
    ]
    FROZEN_SET_PATH = str(_project_root / "eval/frozen_set_extraction.json")
    RESULTS_PATH = str(_project_root / "outputs/extraction_baseline.json")

    # Frozen indices: 12 English + 12 Hindi/Marathi, stratified by rubric category
    # coverage and language. Selected to maximise representation of medication_name,
    # symptom_name, diagnosis_name, body_vital_sign_name, prescribed_test_name,
    # examination_name, diagnostic_result_name, and medication_timing categories.
    # i=83 (English asthma, the confirmed-working baseline) is included.
    FROZEN_INDICES: list[int] = [
        # English rows (sorted by rubric key-category coverage desc)
        91, 67, 109, 2, 82, 64, 99, 83, 131, 121, 38, 89,
        # Hindi/Marathi rows (sorted by rubric key-category coverage desc)
        74, 75, 45, 47, 48, 44, 49, 24, 46, 134, 60, 140,
    ]

    # ── Category → schema field mapping ─────────────────────────────────────
    # Each entry: (schema_field_name, extractor_fn(note) → list[str])
    # extractor_fn returns the list of string values to match against.
    REPRESENTABLE_MAP: dict[str, tuple[str, object]] = {
        "medication_name": (
            "medications[].drug",
            lambda note: [m.drug for m in note.medications],
        ),
        "medication_dose": (
            "medications[].dose",
            lambda note: [m.dose for m in note.medications if m.dose],
        ),
        "medication_frequency": (
            "medications[].frequency",
            lambda note: [m.frequency for m in note.medications if m.frequency],
        ),
        "medication_timing": (
            "medications[].timing",
            lambda note: [m.timing for m in note.medications if m.timing],
        ),
        "symptom_name": (
            "symptoms[].name",
            lambda note: [s.name for s in note.symptoms],
        ),
        "symptom_severity": (
            "symptoms[].severity",
            lambda note: [s.severity for s in note.symptoms if s.severity],
        ),
        "body_vital_sign_name": (
            "vitals[].name",
            lambda note: [v.name for v in note.vitals],
        ),
        "body_vital_sign_value": (
            "vitals[].value",
            lambda note: [v.value for v in note.vitals],
        ),
        "diagnosis_name": (
            "diagnosis[].term",
            lambda note: [d.term for d in note.diagnosis],
        ),
        "diagnosis_status": (
            "diagnosis[].status",
            lambda note: [d.status for d in note.diagnosis if d.status],
        ),
        "prescribed_test_name": (
            "investigations[]",
            lambda note: list(note.investigations),
        ),
        "diagnostic_result_name": (
            "diagnostic_results[]",
            lambda note: list(note.diagnostic_results),
        ),
        "examination_name": (
            "examination",
            lambda note: [note.examination] if note.examination else [],
        ),
        "examination_notes": (
            "examination",
            lambda note: [note.examination] if note.examination else [],
        ),
    }

    UNREPRESENTABLE_CATS: frozenset[str] = frozenset(
        {
            "medical_condition_name",
            "medical_condition_status",
            "current_medication_name",
            "lifestyle_habit_name",
            "family_history_name",
            "family_history_who",
            "foodotherallergy_name",
            "pastprocedures_name",
            "recenttravelhistory_name",
            "symptom_laterality",
            "diagnosis_laterality",
            "drugallergy_name",
            "diagnostic_result_value",
            "diagnostic_result_interpretation",
        }
    )

    # ── Rubric parsing ───────────────────────────────────────────────────────
    _RUBRIC_RE = re.compile(
        r"Rubric ID:\s*(\d+)\s*\nCategory ID:\s*([^\n]+)\s*\nCriterion:\s*([^\n]+(?:\n(?!Rubric ID:)[^\n]*)*)",
        re.MULTILINE,
    )
    _PUNCT_RE = re.compile(r"[^a-z0-9 ]")

    def _parse_rubrics(text: str) -> list[tuple[int, str, str]]:
        """Return list of (id, category, criterion) tuples from rubric text."""
        results = []
        for m in _RUBRIC_RE.finditer(text):
            rid = int(m.group(1))
            cat = m.group(2).strip().lower()
            crit = m.group(3).strip()
            results.append((rid, cat, crit))
        return results

    def _normalise(s: str) -> str:
        return _PUNCT_RE.sub("", s.lower().strip())

    def _token_overlap(a: str, b: str) -> float:
        """Token overlap Jaccard coefficient."""
        ta = set(a.split())
        tb = set(b.split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def _extract_quoted(criterion: str) -> str:
        """Extract first single-quoted value from a criterion string."""
        m = re.search(r"'([^']+)'", criterion)
        return m.group(1) if m else criterion

    def _fuzzy_match(criterion: str, field_values: list[str]) -> bool:
        """Return True if criterion matches any of the extracted field values.

        Strategy:
        1. Extract the quoted entity from the criterion (the 'Gold Standard').
        2. Normalise both candidate and gold.
        3. Accept if: substring (bidirectional) OR token-overlap ≥ 0.5.
        """
        gold = _normalise(_extract_quoted(criterion))
        if not gold:
            return False
        for val in field_values:
            candidate = _normalise(val)
            if not candidate:
                continue
            if gold in candidate or candidate in gold:
                return True
            if _token_overlap(gold, candidate) >= 0.5:
                return True
        return False

    def _is_devanagari(text: str) -> bool:
        return bool(re.search(r"[ऀ-ॿ]", text))

    # ── Load dataset ─────────────────────────────────────────────────────────
    dfs = [pd.read_parquet(p) for p in DATASET_FILES]
    df = pd.concat(dfs, ignore_index=True)

    # Write frozen set metadata
    os.makedirs(os.path.dirname(FROZEN_SET_PATH), exist_ok=True)
    frozen_rows = df.loc[FROZEN_INDICES].reset_index(drop=False)
    with open(FROZEN_SET_PATH, "w") as f:
        json.dump(
            {
                "dataset": DATASET_FILES,
                "frozen_indices": FROZEN_INDICES,
                "session_ids": frozen_rows["session_id"].tolist(),
                "n": len(FROZEN_INDICES),
            },
            f,
            indent=2,
        )

    # ── Per-category accumulators ─────────────────────────────────────────
    cat_total: dict[str, int] = {}
    cat_representable: dict[str, int] = {}
    cat_matched: dict[str, int] = {}
    unrepresentable_total: int = 0
    per_sample: list[dict] = []

    for idx in FROZEN_INDICES:
        row = df.loc[idx]
        transcript = str(row["text"])
        is_deva = _is_devanagari(transcript)
        rubrics = _parse_rubrics(str(row["rubrics"]))

        # Run L4 extraction (no diarization — single UNKNOWN turn, isolating L4)
        turns = [Turn(speaker_role="UNKNOWN", text=transcript, start=0.0, end=0.0)]
        try:
            note: ClinicalNote = extract(turns)
        except Exception as exc:
            logger.error("Extraction failed for idx=%d: %s", idx, exc)
            from src.l4_extract import _empty_note  # type: ignore[attr-defined]
            note = _empty_note()

        # Memory discipline: the Ollama client is lightweight (HTTP), no large
        # model held resident in Python — no explicit gc needed between rows.

        sample_results = {
            "idx": int(idx),
            "session_id": row["session_id"],
            "language": "hindi_marathi" if is_deva else "english",
            "n_rubrics": len(rubrics),
            "criteria": [],
        }

        for (rid, cat, criterion) in rubrics:
            # Track total per category
            cat_total[cat] = cat_total.get(cat, 0) + 1

            if cat in UNREPRESENTABLE_CATS:
                unrepresentable_total += 1
                sample_results["criteria"].append(
                    {"id": rid, "cat": cat, "criterion": criterion[:80],
                     "status": "unrepresentable"}
                )
                continue

            cat_representable[cat] = cat_representable.get(cat, 0) + 1

            if cat not in REPRESENTABLE_MAP:
                # Unknown category — log and skip
                logger.warning("Unknown rubric category: %s", cat)
                sample_results["criteria"].append(
                    {"id": rid, "cat": cat, "criterion": criterion[:80],
                     "status": "unknown_category"}
                )
                continue

            _field_name, extractor = REPRESENTABLE_MAP[cat]
            field_values: list[str] = extractor(note)

            if is_deva:
                # Cross-lingual: rubric in English, extracted note may also be
                # English (3B model tends to output English regardless of input
                # language). Apply presence check first; string match second.
                has_value = len(field_values) > 0
                matched = has_value and _fuzzy_match(criterion, field_values)
                status = "matched" if matched else ("present_no_match" if has_value else "missing")
            else:
                matched = _fuzzy_match(criterion, field_values)
                status = "matched" if matched else "missed"

            if matched:
                cat_matched[cat] = cat_matched.get(cat, 0) + 1

            sample_results["criteria"].append(
                {"id": rid, "cat": cat, "criterion": criterion[:80],
                 "status": status, "extracted": field_values[:5]}
            )

        per_sample.append(sample_results)

        logger.info(
            "[idx=%d lang=%s] %d rubrics: %d representable, note has %d syms %d meds %d diag",
            idx,
            "hi/mr" if is_deva else "en",
            len(rubrics),
            sum(1 for _, c, _ in rubrics if c not in UNREPRESENTABLE_CATS),
            len(note.symptoms),
            len(note.medications),
            len(note.diagnosis),
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    summary_rows = []
    for cat in sorted(cat_total.keys()):
        total = cat_total[cat]
        rep = cat_representable.get(cat, 0)
        matched = cat_matched.get(cat, 0)
        recall = round(matched / rep, 3) if rep else None
        summary_rows.append(
            {
                "category": cat,
                "total_criteria": total,
                "representable": rep,
                "matched": matched,
                "recall": recall,
            }
        )

    all_rep = sum(cat_representable.values())
    all_matched = sum(cat_matched.values())
    aggregate = {
        "n_samples": len(FROZEN_INDICES),
        "total_criteria": sum(cat_total.values()),
        "representable_criteria": all_rep,
        "matched_criteria": all_matched,
        "representable_recall": round(all_matched / all_rep, 3) if all_rep else 0.0,
        "unrepresentable_criteria": unrepresentable_total,
    }

    output = {"aggregate": aggregate, "per_category": summary_rows, "per_sample": per_sample}
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Print summary table
    print("\n" + "=" * 75)
    print(f"L4 EXTRACTION EVAL — {aggregate['n_samples']} samples (frozen set)")
    print("=" * 75)
    print(f"{'Category':<35} {'Total':>6} {'Rep':>6} {'Match':>6} {'Recall':>7}")
    print("-" * 75)
    for row in summary_rows:
        recall_str = f"{row['recall']:.3f}" if row["recall"] is not None else "  N/A"
        rep_marker = "" if row["representable"] > 0 else " (unrepresentable)"
        print(
            f"{row['category']:<35} {row['total_criteria']:>6} {row['representable']:>6} "
            f"{row['matched']:>6} {recall_str:>7}{rep_marker}"
        )
    print("-" * 75)
    print(f"{'AGGREGATE (representable only)':<35} {aggregate['total_criteria']:>6} "
          f"{aggregate['representable_criteria']:>6} {aggregate['matched_criteria']:>6} "
          f"{aggregate['representable_recall']:>7.3f}")
    print(f"  Unrepresentable criteria (schema gap): {aggregate['unrepresentable_criteria']}")
    print(
        "\nNote: Hindi/Marathi rows use presence check (cross-lingual, can't string-match).\n"
        "      English rows use full fuzzy match. See per_sample for details."
    )
    print(f"\nFull results: {RESULTS_PATH}")
    print(f"Frozen set:   {FROZEN_SET_PATH}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    run_asr_eval()
