"""Gloss audit: OLD vs NEW concept matcher over every cached transcript we own.

Mirrors the drug matcher's substitution audit (eval/drug_miss_analysis.py,
LEARNINGS.md "Drug Canonicalization Phase"): run the corpus through BOTH
matcher configurations, log every gloss applied with its runner-up and
source context, then diff the two runs so every DIFFERENCE — and every
everyday-word-adjacent gloss — is adjudicated by hand against the source
context rather than trusted from an aggregate accuracy number.

Corpus (four sources we actually have cached):
  extraction  — the 24 frozen eka-clinical-note-generation transcripts
                (eval/frozen_set_extraction.json's indices).
  drug_bench  — outputs/drug_bench.json's cached raw ASR hypotheses
                (49-clip bench).
  beam_study  — outputs/beam_study.json's clips, beam=5 hypothesis (matches
                production ASR_BEAM_SIZE — see src/config.py).
  session     — outputs/2026*/transcript.json's real session turns. These
                are POST-normalize (already glossed), so any existing
                "(...)" concept gloss is stripped first (see _strip_gloss) —
                the audit re-glosses from a clean slate. Assumption: no raw
                ASR output legitimately contains a "word (word)" pattern;
                true across every cached session inspected.

Efficiency: the model runs EXACTLY ONCE. Every turn across every source is
batched into one encode() call (src.l3_5_normalize._encode_turn_spans), and
the reference/hard-negative matrices are encoded once
(_encode_reference_matrices). Scoring under different MatchConfigs
(src.l3_5_normalize._score_spans) is pure numpy, so OLD vs NEW — and the
whole tuning sweep — costs nothing extra once the embeddings exist.

"OLD" = the matcher as it stood immediately before this hardening phase:
COSINE_THRESHOLD for every span length, no ambiguity margin, no
everyday-word guard. The hard-negative gate itself (commit 26c42ea) is
unchanged and active in both — EXCEPT for one bug fix uncovered while
verifying this phase's real-model gate tests: a hard negative SHORTER than
the query span (e.g. unigram हफ्ते vetoing the bigram "हांफ रहे हैं", a
genuine Shortness of Breath mention) is now excluded from that span's
hard-negative check (src.l3_5_normalize._score_spans's hardneg_word_counts).
This fix is orthogonal to the OLD-vs-NEW axis being measured here (it only
affects multi-word query spans compared against a SHORTER hard negative,
never the unigram candidates this audit tunes), so it is applied
identically in both "OLD" and "NEW_SUPERSET" below.

"NEW_SUPERSET" = everyday-word guard ON, but at the LOOSEST candidate
thresholds (cosine_threshold_unigram = COSINE_THRESHOLD, ambiguity_margin =
0.0). Because every new gate is strictly more restrictive as its constant
increases, this run's accepted glosses are a superset of every stricter NEW
config's output — so the full tuning sweep (build_tuning_table) is computed
by filtering this ONE run's logged (cosine, runner_up_cosine, is_unigram)
fields, with no further model calls.

Run:  PYTHONPATH=. python eval/gloss_audit.py
"""

import json
import pathlib
import re

import numpy as np
import pandas as pd

from src.concepts import CONCEPTS, EVERYDAY_WORDS
from src.l3_5_normalize import (
    COSINE_THRESHOLD,
    MatchConfig,
    _EmbeddingBackend,
    _encode_reference_matrices,
    _encode_turn_spans,
    _score_spans,
)
from src.types import Turn

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_GLOSS_RE = re.compile(r"\s\([^()]*\)")

RESULTS_PATH = _ROOT / "outputs/gloss_audit.json"


def _strip_gloss(text: str) -> str:
    """Remove an existing ' (Concept Name)' gloss suffix, if present."""
    return _GLOSS_RE.sub("", text)


def _load_extraction_texts() -> list[tuple[str, str]]:
    """(item_id, transcript_text) for the 24 frozen extraction-eval rows."""
    frozen_path = _ROOT / "eval/frozen_set_extraction.json"
    with open(frozen_path, encoding="utf-8") as f:
        frozen = json.load(f)
    dataset_dir = _ROOT / "eka-clinical-note-generation-dataset"
    dfs = [
        pd.read_parquet(dataset_dir / "test-00000.parquet"),
        pd.read_parquet(dataset_dir / "test-00001.parquet"),
    ]
    df = pd.concat(dfs, ignore_index=True)
    indices = frozen["frozen_indices"]
    return [(f"idx{idx}", str(df.loc[idx, "text"])) for idx in indices]


def _load_drug_bench_texts() -> list[tuple[str, str]]:
    """(item_id, raw_hypothesis) for the 49-clip frozen drug bench."""
    path = _ROOT / "outputs/drug_bench.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        (clip["id"][:8], clip["raw_hypothesis"])
        for clip in data["per_clip"]
        if clip.get("raw_hypothesis")
    ]


def _load_beam_study_texts(beam: str = "5") -> list[tuple[str, str]]:
    """(item_id, hypothesis) for the beam-study clips at the given beam size.

    beam="5" matches production ASR_BEAM_SIZE (src/config.py).
    """
    path = _ROOT / "outputs/beam_study.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for clip in data["per_clip"]:
        b = clip["beams"].get(beam)
        if b and b.get("hypothesis"):
            out.append((f"{clip['id'][:8]}_beam{beam}", b["hypothesis"]))
    return out


def _load_session_texts() -> list[tuple[str, str]]:
    """(item_id, turn_text) for every turn of every real session, gloss-stripped."""
    out = []
    for session_dir in sorted((_ROOT / "outputs").glob("2026*")):
        tpath = session_dir / "transcript.json"
        if not tpath.exists():
            continue
        with open(tpath, encoding="utf-8") as f:
            turns = json.load(f)
        for i, turn in enumerate(turns):
            text = _strip_gloss(str(turn.get("text", "")))
            if text.strip():
                out.append((f"{session_dir.name}#turn{i}", text))
    return out


def _gather_corpus() -> list[tuple[str, str, str]]:
    """(source, item_id, text) for every audit item across all four sources."""
    corpus: list[tuple[str, str, str]] = []
    corpus += [("extraction", i, t) for i, t in _load_extraction_texts()]
    corpus += [("drug_bench", i, t) for i, t in _load_drug_bench_texts()]
    corpus += [("beam_study", i, t) for i, t in _load_beam_study_texts()]
    corpus += [("session", i, t) for i, t in _load_session_texts()]
    return corpus


def _context(words: list[str], start_w: int, end_w: int, pad: int = 40) -> str:
    """±pad-char window of the joined-words text around one span."""
    text = " ".join(words)
    prefix = " ".join(words[:start_w])
    start_char = len(prefix) + (1 if prefix else 0)
    span_text = " ".join(words[start_w:end_w])
    end_char = start_char + len(span_text)
    return text[max(0, start_char - pad) : min(len(text), end_char + pad)]


def run_audit(cfgs: dict[str, MatchConfig]) -> tuple[dict[str, list[dict]], int]:
    """Score the whole corpus under each named MatchConfig; model runs once.

    Returns:
        (results, n_items) where results[cfg_name] is a flat list of every
        gloss applied under that config, each with span/concept/cosine/
        runner-up/context/is_unigram, and n_items is the corpus size (for
        the summary).
    """
    corpus = _gather_corpus()
    turns = [
        Turn(speaker_role="UNKNOWN", text=text, start=0.0, end=0.0)
        for _, _, text in corpus
    ]

    backend = _EmbeddingBackend()
    ref_matrix, ref_ci_arr, hardneg_matrix, hardneg_idx_arr, hardneg_word_counts = (
        _encode_reference_matrices(backend)
    )
    all_spans, turn_span_offsets, turn_words, span_matrix = _encode_turn_spans(
        backend, turns
    )
    backend.release()

    if span_matrix is None:
        return {name: [] for name in cfgs}, len(corpus)

    ref_sims = span_matrix @ ref_matrix.T
    hardneg_sims = (
        span_matrix @ hardneg_matrix.T if hardneg_matrix is not None else None
    )
    n_concepts = len(CONCEPTS)

    results: dict[str, list[dict]] = {}
    for cfg_name, cfg in cfgs.items():
        per_turn_matches = _score_spans(
            all_spans, turn_span_offsets, ref_sims, ref_ci_arr,
            hardneg_sims, hardneg_idx_arr, n_concepts, cfg,
            hardneg_word_counts=hardneg_word_counts,
        )
        entries = []
        zipped = zip(corpus, turn_words, per_turn_matches, strict=True)
        for (source, item_id, _text), words, matches in zipped:
            for m in matches:
                entries.append(
                    {
                        "source": source,
                        "id": item_id,
                        "span": m.span,
                        "is_unigram": (m.end_word - m.start_word) == 1,
                        "concept": m.concept_term,
                        "cosine": round(m.similarity, 4),
                        "runner_up": m.runner_up_term,
                        "runner_up_cosine": (
                            round(m.runner_up_similarity, 4)
                            if m.runner_up_similarity is not None
                            else None
                        ),
                        "context": _context(words, m.start_word, m.end_word),
                    }
                )
        results[cfg_name] = entries
    return results, len(corpus)


def _diff(old: list[dict], new: list[dict]) -> list[dict]:
    """Glosses present in `old` but absent from `new` (removed by hardening)."""

    def _key(e: dict) -> tuple:
        return (e["source"], e["id"], e["span"], e["concept"])

    new_keys = {_key(e) for e in new}
    return [e for e in old if _key(e) not in new_keys]


# ── Manual adjudication (2026-07-12) ────────────────────────────────────────
# Every DISTINCT (span, concept) unigram gloss the NEW-superset run produced
# (90 total — the full unigram universe any NEW config could ever gloss),
# read against its ±40-char context and labeled by hand:
#   correct   — the gloss matches the transcript's actual clinical meaning.
#   wrong     — the gloss is spurious or clinically misleading.
#   uncertain — genuinely ambiguous (counts as a loss when a threshold drops
#               it — "be conservative").
# Only spans NOT listed here are implicitly "correct" (the overwhelming
# majority — direct hits like बुखार/फीवर/fever, बीपी/BP, कफ/खांसी/cough,
# सुगर/शुगर/डायबिटीज/diabetes/sugar, दर्द/pain, दस्त, उल्टी, आर्थराइटिस, ...).
# This dict is the ground truth build_tuning_table() filters against.
ADJUDICATED_WRONG: frozenset[tuple[str, str]] = frozenset({
    # Cosine <= 0.721 — the crowded-short-span noise floor (all clearly wrong):
    ("योर", "Fever"),                       # "व्हाट्स योर नेम" (what's YOUR name)
    ("ब्लड", "Hypertension"),                # "ब्लड टेस्ट कराओ" (get a blood TEST)
    ("BPM,", "Hypertension"),                 # BPM = heart rate, not blood pressure
    ("blood", "Hypertension"),                # "Fasting blood SUGAR" (diabetes, not BP)
    ("DM", "Type 2 Diabetes Mellitus"),       # drug-name fragment: "Ompec DM"
    ("D.", "Diarrhea"),                       # list marker: "vitamin B, C, and D."
    ("वाइटिंग", "Vomiting"),                  # "party वेटिंग" (waiting), not vomiting
    ("संपल्या.", "Weakness"),                 # "गोळ्या संपल्या" (pills ran OUT)
    ("बसले", "Weakness"),                     # "बसले" = sat (down), not weak
    ("flu", "Fever"),                         # flu SHOT (vaccination), not fever
    ("stools", "Diarrhea"),                   # "are the stools NORMAL?" (negated)
    ("गैप", "Acid Reflux"),                   # dosing-schedule "gap", unrelated
    ("acid,", "Acid Reflux"),                 # URIC acid (lab value), not stomach acid
    # Cosine 0.78-0.85 — residual (see COSINE_THRESHOLD_UNIGRAM's comment,
    # src/config.py): a threshold high enough to drop these also drops
    # dozens of legitimate matches in the same band (कमजोर, कफ, दर्द-family,
    # डायबिटीज-family, "weakness.", "ache.").
    ("जळत", "Acid Reflux"),                   # "पाय जळत होते" (my LEGS were burning)
    ("खीस", "Cough"),                         # Marathi leg/knee word, not cough
    ("संपल्या", "Weakness"),                  # same as संपल्या. above, no period
    ("Tension", "Anxiety"),                   # "Tension headache" is a diagnosis
})
ADJUDICATED_UNCERTAIN: frozenset[tuple[str, str]] = frozenset({
    ("घबराने", "Anxiety"),  # "कुछ ऐसा घबराने की बात नहीं है" — negated ("nothing
                            # to be anxious about"); word IS anxiety-related but
                            # asserted absent, not present. Conservative: counts
                            # as a loss, not a win, when a threshold drops it.
})
# The named "eval recall depends on this" true-positive class the brief calls
# out explicitly, used to confirm no regression at the chosen threshold.
MUST_PRESERVE_UNIGRAMS: frozenset[tuple[str, str]] = frozenset({
    ("बीपी", "Hypertension"), ("बुखार", "Fever"), ("फीवर", "Fever"),
    ("खांसी", "Cough"), ("कफ", "Cough"), ("सुगर", "Type 2 Diabetes Mellitus"),
    ("शुगर", "Type 2 Diabetes Mellitus"), ("सर्दी", "Common Cold"),
    ("दर्द", "Pain"), ("उल्टी", "Vomiting"),
})


def build_tuning_table(
    new_superset_unigrams: list[dict], thresholds: list[float]
) -> list[dict]:
    """For each candidate COSINE_THRESHOLD_UNIGRAM, count wrong/lost/kept.

    Pure filtering over the ALREADY-COLLECTED new-superset run (ambiguity
    margin fixed at 0.0 in that run, so this table isolates the unigram
    threshold's effect only — see the module docstring's superset argument).

    Args:
        new_superset_unigrams: unigram entries from run_audit's "new_superset"
            config (ambiguity_margin=0.0, everyday_words=EVERYDAY_WORDS).
        thresholds: candidate COSINE_THRESHOLD_UNIGRAM values to try.

    Returns:
        One row per threshold: wrong remaining, uncertain remaining, and
        whether every MUST_PRESERVE_UNIGRAMS entry survives.
    """
    seen: dict[tuple[str, str], float] = {}
    for e in new_superset_unigrams:
        key = (e["span"], e["concept"])
        seen[key] = e["cosine"]

    rows = []
    for t in thresholds:
        kept = {k for k, cos in seen.items() if cos >= t}
        wrong_remaining = kept & ADJUDICATED_WRONG
        uncertain_remaining = kept & ADJUDICATED_UNCERTAIN
        preserved = MUST_PRESERVE_UNIGRAMS & kept
        lost_must_preserve = MUST_PRESERVE_UNIGRAMS - kept
        rows.append(
            {
                "threshold": t,
                "total_kept": len(kept),
                "wrong_remaining": len(wrong_remaining),
                "uncertain_remaining": len(uncertain_remaining),
                "must_preserve_kept": len(preserved),
                "must_preserve_lost": sorted(lost_must_preserve),
            }
        )
    return rows


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s — %(message)s"
    )

    old_cfg = MatchConfig(
        cosine_threshold=COSINE_THRESHOLD,
        cosine_threshold_unigram=COSINE_THRESHOLD,
        ambiguity_margin=0.0,
        everyday_words=frozenset(),
    )
    new_superset_cfg = MatchConfig(
        cosine_threshold=COSINE_THRESHOLD,
        cosine_threshold_unigram=COSINE_THRESHOLD,
        ambiguity_margin=0.0,
        everyday_words=EVERYDAY_WORDS,
    )

    results, n_items = run_audit({"old": old_cfg, "new_superset": new_superset_cfg})
    removed_by_guard = _diff(results["old"], results["new_superset"])

    print(f"Corpus items: {n_items}")
    print(f"OLD glosses: {len(results['old'])}")
    print(f"NEW-superset glosses: {len(results['new_superset'])}")
    print(f"Removed by everyday-word guard: {len(removed_by_guard)}")

    unigrams = [e for e in results["new_superset"] if e["is_unigram"]]
    n_unigram_pairs = len({(e["span"], e["concept"]) for e in unigrams})
    print(f"\nDistinct unigram (span, concept) candidates: {n_unigram_pairs}")

    print("\nTuning table (COSINE_THRESHOLD_UNIGRAM sweep, margin=0.0/guard=ON):")
    print(
        f"{'threshold':>10} {'kept':>6} {'wrong':>7} {'uncertain':>10} "
        f"{'must-preserve lost':>20}"
    )
    sweep = [0.65, 0.70, 0.721, 0.73, 0.76, 0.80, 0.86, 0.90]
    for row in build_tuning_table(unigrams, sweep):
        print(
            f"{row['threshold']:>10.3f} {row['total_kept']:>6} "
            f"{row['wrong_remaining']:>7} {row['uncertain_remaining']:>10} "
            f"{len(row['must_preserve_lost']):>20}"
        )

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_items": n_items,
                "old": results["old"],
                "new_superset": results["new_superset"],
                "removed_by_guard": removed_by_guard,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nFull results: {RESULTS_PATH}")
