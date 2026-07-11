"""Gate for src.fast_asr.fast_transcribe — the decisive measurement for
src.config.FAST_ASR_ENABLED. See that flag's docstring and LEARNINGS.md
"Deployment Latency Phase" for why every prior fast-engine attempt (plain
mlx-turbo/mlx-large-v3/fw-turbo, merged windows, decode-param tweaks, VAD
pre-slicing) failed this same accuracy bar outright.

Part A -- frozen bench (10 clips, `--bench`/`--score`): fast_transcribe()
decodes each clip's CACHED L2 diarization segments from
outputs/engine_study.json -- the same segments every engine this project has
benched has decoded, so results are apples-to-apples. Scored against the
fw-large-v3 production baseline also cached there (corpus_wer, keyword_wer,
drug_wer_folded). Gate:
    keyword_wer_micro <= baseline + eval.engine_study.KEYWORD_WER_TOLERANCE
    corpus_wer        <= baseline + eval.engine_study.CORPUS_WER_TOLERANCE
    drug_wer_folded   <= baseline (not worse)
Drug scoring applies the L3.5 normalize() pass and eval.drug_bench's
script-symmetric fold (_folded_match) over raw+normalized text combined --
the same formula eval/engine_study.py._score_engine uses, so the "drug
folded" number is directly comparable to the one already cached there.

Part B -- deployment clips (real UI sessions, `--session`/`--score`):
fast_transcribe() decodes each session's real diarized segments (cached for
20260710-230150-cef13a from a prior L2 run in outputs/engine_study.json;
freshly diarized for the other two -- no cache exists for them). Reports
wall time (this is where the "25s audio in <=20s" claim is proven or
disproven), an Arabic-script check, and a drug-keyword spot-check
post-normalize.

Run:
    PYTHONPATH=. python eval/fast_asr_gate.py --bench
    PYTHONPATH=. python eval/fast_asr_gate.py --session
    PYTHONPATH=. python eval/fast_asr_gate.py --score
"""

import argparse
import json
import logging
import os
import tempfile
import time

from dotenv import load_dotenv

from eval.drug_bench import _folded_match
from eval.engine_study import CORPUS_WER_TOLERANCE, KEYWORD_WER_TOLERANCE
from eval.metrics import corpus_word_error_rate, keyword_hits
from eval.run_eval import _keywords_from_entities
from src.fast_asr import fast_transcribe
from src.l1_preprocess import preprocess
from src.l3_5_normalize import normalize
from src.l3_asr import _contains_arabic_script
from src.types import Segment, Turn

load_dotenv()
logger = logging.getLogger(__name__)

ASR_DATASET = "eka-medical-asr-dataset/hi/test-00000.parquet"
FROZEN_N = 10  # must match eval/engine_study.py's frozen bench
ENGINE_STUDY_PATH = (
    "outputs/engine_study.json"  # cached segments + fw-large-v3 baseline summary
)
RESULTS_PATH = "outputs/fast_asr_gate.json"

# Deployment session clips (real UI uploads): wav path + the drug keyword
# each one's transcript actually contains, checked post-L3.5-normalize.
SESSION_CLIPS: dict[str, tuple[str, str]] = {
    "20260710-230150-cef13a": (
        "outputs/20260710-230150-cef13a/input_16k.wav",
        "naxdom",
    ),
    "20260711-184756-36c330": (
        "outputs/20260711-184756-36c330/input_16k.wav",
        "paracetamol",
    ),
    "20260711-160316-99d62b": (
        "outputs/20260711-160316-99d62b/input_16k.wav",
        "augmentin",
    ),
}
# The only session with cached L2 segments (outputs/engine_study.json's
# "session" block, from a prior run_session() there); the other two have no
# cache anywhere and are freshly diarized.
CEF13A_SESSION_ID = "20260710-230150-cef13a"
TARGET_WALL_S = 20.0  # brief: 25s audio must transcribe in <=20s, ladder included


# ── Results file plumbing (mirrors eval/engine_study.py's convention) ────────


def _load_results(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"per_clip": [], "sessions": {}}


def _flush(results: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _segments_from_cache(raw: list[dict]) -> list[Segment]:
    return [Segment(start=s["start"], end=s["end"], speaker=s["speaker"]) for s in raw]


class _LadderLogHandler(logging.Handler):
    """Captures src.fast_asr's "ladder fired" warnings for one decode call."""

    def __init__(self, sink: list[str]) -> None:
        super().__init__(level=logging.WARNING)
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "ladder fired" in message:
            self._sink.append(message)


def _timed_transcribe(
    wav_path: str, segments: list[Segment]
) -> tuple[list[Turn], float, list[str]]:
    """fast_transcribe(), wall-clock timed, with ladder firings captured."""
    ladder_log: list[str] = []
    handler = _LadderLogHandler(ladder_log)
    fast_asr_logger = logging.getLogger("src.fast_asr")
    fast_asr_logger.addHandler(handler)
    t0 = time.perf_counter()
    try:
        turns = fast_transcribe(wav_path, segments)
    finally:
        fast_asr_logger.removeHandler(handler)
    wall = time.perf_counter() - t0
    return turns, wall, ladder_log


def _drug_wer_folded(
    ref: str, raw_hyp: str, norm_hyp: str, kws: list[str]
) -> tuple[int, int]:
    """(present, missed) using the script-symmetric fold over raw+normalized
    text -- identical formula to eval/engine_study.py._drug_wer_folded, so
    numbers here are directly comparable to that baseline."""
    combined = f"{raw_hyp} {norm_hyp}"
    present = missed = 0
    for kw in kws:
        p, m = keyword_hits(ref, combined, [kw])
        if not p:
            continue
        present += 1
        if m and not _folded_match(kw, combined):
            missed += 1
    return present, missed


# ── Part A: frozen bench ─────────────────────────────────────────────────────


def run_bench(out_path: str = RESULTS_PATH) -> None:
    """Decode the frozen 10-clip bench with fast_transcribe() on cached L2 segments."""
    import pandas as pd

    with open(ENGINE_STUDY_PATH, encoding="utf-8") as f:
        engine_study = json.load(f)
    cached_by_id = {c["id"]: c for c in engine_study["per_clip"]}

    df = pd.read_parquet(ASR_DATASET).head(FROZEN_N)
    results = _load_results(out_path)
    done = {c["id"] for c in results["per_clip"]}

    for i, (_, row) in enumerate(df.iterrows()):
        clip_id = row["md5_text"]
        if clip_id in done:
            logger.info(
                "[%d/%d] %s already done -- skipping", i + 1, FROZEN_N, clip_id[:8]
            )
            continue
        cached = cached_by_id.get(clip_id)
        if cached is None or "segments" not in cached:
            raise ValueError(
                f"{clip_id[:8]} has no cached L2 segments in {ENGINE_STUDY_PATH} -- "
                "run eval/engine_study.py --bench at least once first"
            )
        segments = _segments_from_cache(cached["segments"])

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(row["audio"])
            tmp_path = f.name
        try:
            wav_path = preprocess(tmp_path)
        finally:
            os.unlink(tmp_path)

        turns, wall, ladder_log = _timed_transcribe(wav_path, segments)
        hyp = " ".join(t.text for t in turns).strip()

        results["per_clip"].append(
            {
                "id": clip_id,
                "hypothesis": hyp,
                "decode_wall_s": round(wall, 3),
                "ladder_log": ladder_log,
            }
        )
        logger.info(
            "[%d/%d] %s: %.2fs, %d ladder firing(s) -- %s",
            i + 1,
            FROZEN_N,
            clip_id[:8],
            wall,
            len(ladder_log),
            hyp[:80],
        )
        _flush(results, out_path)

    print(f"Bench results: {out_path}")


# ── Part B: deployment session clips ─────────────────────────────────────────


def run_session(out_path: str = RESULTS_PATH) -> None:
    """Decode every real UI session clip in SESSION_CLIPS with fast_transcribe()."""
    import soundfile as sf

    results = _load_results(out_path)
    sessions = results.setdefault("sessions", {})

    for session_id, (wav_path, _expected_drug) in SESSION_CLIPS.items():
        if session_id in sessions:
            logger.info("%s already done -- skipping", session_id)
            continue
        if not os.path.exists(wav_path):
            logger.warning("%s: %s not found -- skipping", session_id, wav_path)
            continue

        if session_id == CEF13A_SESSION_ID:
            with open(ENGINE_STUDY_PATH, encoding="utf-8") as f:
                engine_study = json.load(f)
            segments = _segments_from_cache(engine_study["session"]["segments"])
            segment_source = f"cached ({ENGINE_STUDY_PATH} session)"
        else:
            from src.l2_diarize import diarize

            t0 = time.perf_counter()
            segments = diarize(wav_path)
            diarize_wall = time.perf_counter() - t0
            segment_source = f"live diarize() ({diarize_wall:.2f}s)"

        duration_s = round(sf.info(wav_path).duration, 3)
        turns, wall, ladder_log = _timed_transcribe(wav_path, segments)
        hyp = " ".join(t.text for t in turns).strip()

        sessions[session_id] = {
            "duration_s": duration_s,
            "n_segments": len(segments),
            "segment_source": segment_source,
            "decode_wall_s": round(wall, 3),
            "ladder_log": ladder_log,
            "hypothesis": hyp,
            "arabic_script_present": _contains_arabic_script(hyp),
        }
        logger.info(
            "%s (%.2fs audio): decode=%.2fs, %d ladder firing(s)",
            session_id,
            duration_s,
            wall,
            len(ladder_log),
        )
        _flush(results, out_path)

    print(f"Session results: {out_path}")


# ── Scoring ──────────────────────────────────────────────────────────────────


def _score_bench(results: dict) -> dict:
    import pandas as pd

    with open(ENGINE_STUDY_PATH, encoding="utf-8") as f:
        baseline = json.load(f)["summary"]["fw-large-v3"]

    df = pd.read_parquet(ASR_DATASET).head(FROZEN_N)
    by_id = {c["id"]: c for c in results["per_clip"]}

    refs: list[str] = []
    hyps: list[str] = []
    per_clip_rows: list[dict] = []
    kw_present = kw_missed = 0
    drug_present = drug_missed = drug_present_f = drug_missed_f = 0

    for _, row in df.iterrows():
        clip_id = row["md5_text"]
        rec = by_id.get(clip_id)
        if rec is None:
            continue
        hyp, ref = rec["hypothesis"], row["text"]
        kw = _keywords_from_entities(row["medical_entities"])
        kw_drug = _keywords_from_entities(row["medical_entities"], drug_only=True)

        refs.append(ref)
        hyps.append(hyp)

        p, m = keyword_hits(ref, hyp, kw)
        kw_present += p
        kw_missed += m

        dp, dm = keyword_hits(ref, hyp, kw_drug)
        drug_present += dp
        drug_missed += dm

        norm_turns = normalize(
            [Turn(speaker_role="UNKNOWN", text=hyp, start=0.0, end=0.0)]
        )
        norm_hyp = " ".join(t.text for t in norm_turns).strip()
        dpf, dmf = _drug_wer_folded(ref, hyp, norm_hyp, kw_drug)
        drug_present_f += dpf
        drug_missed_f += dmf

        per_clip_rows.append(
            {
                "id": clip_id[:8],
                "wall_s": rec["decode_wall_s"],
                "ladder_firings": len(rec.get("ladder_log", [])),
                "keyword_wer": round(m / p, 4) if p else 0.0,
            }
        )

    n_scored = len(hyps)
    metrics = {
        "n_clips_scored": n_scored,
        "corpus_wer": round(corpus_word_error_rate(refs, hyps), 4) if refs else None,
        "keyword_wer_micro": round(kw_missed / kw_present, 4) if kw_present else 0.0,
        "drug_wer_raw": round(drug_missed / drug_present, 4) if drug_present else 0.0,
        "drug_wer_folded": round(drug_missed_f / drug_present_f, 4)
        if drug_present_f
        else 0.0,
        "mean_decode_wall_s": (
            round(sum(r["wall_s"] for r in per_clip_rows) / n_scored, 2)
            if n_scored
            else None
        ),
        "total_ladder_firings": sum(r["ladder_firings"] for r in per_clip_rows),
    }

    kw_max = round(baseline["keyword_wer_micro"] + KEYWORD_WER_TOLERANCE, 4)
    corpus_max = round(baseline["corpus_wer"] + CORPUS_WER_TOLERANCE, 4)
    kw_ok = n_scored == FROZEN_N and metrics["keyword_wer_micro"] <= kw_max
    corpus_ok = n_scored == FROZEN_N and metrics["corpus_wer"] <= corpus_max
    drug_ok = (
        n_scored == FROZEN_N
        and metrics["drug_wer_folded"] <= baseline["drug_wer_folded"]
    )
    gate = "PASS" if (kw_ok and corpus_ok and drug_ok) else "FAIL"
    if n_scored < FROZEN_N:
        gate = f"PARTIAL ({n_scored}/{FROZEN_N})"

    return {
        "baseline": baseline,
        "thresholds": {
            "keyword_wer_max": kw_max,
            "corpus_wer_max": corpus_max,
            "drug_wer_folded_max": baseline["drug_wer_folded"],
        },
        "metrics": metrics,
        "gate": gate,
        "per_clip": per_clip_rows,
    }


def _score_sessions(sessions: dict) -> dict:
    rows: dict[str, dict] = {}
    for session_id, (_, expected_drug) in SESSION_CLIPS.items():
        rec = sessions.get(session_id)
        if rec is None:
            continue
        hyp = rec["hypothesis"]
        norm_turns = normalize(
            [Turn(speaker_role="UNKNOWN", text=hyp, start=0.0, end=0.0)]
        )
        norm_hyp = " ".join(t.text for t in norm_turns).strip()
        drug_found = _folded_match(expected_drug, f"{hyp} {norm_hyp}")
        rows[session_id] = {
            "duration_s": rec["duration_s"],
            "decode_wall_s": rec["decode_wall_s"],
            "meets_20s_target": rec["decode_wall_s"] <= TARGET_WALL_S,
            "ladder_firings": len(rec.get("ladder_log", [])),
            "arabic_script_present": rec["arabic_script_present"],
            "expected_drug": expected_drug,
            "drug_found_post_normalize": drug_found,
        }
    return rows


def score(out_path: str = RESULTS_PATH) -> None:
    with open(out_path, encoding="utf-8") as f:
        results = json.load(f)

    bench_summary = _score_bench(results) if results.get("per_clip") else None
    session_summary = _score_sessions(results.get("sessions", {}))

    if bench_summary:
        b = bench_summary["metrics"]
        t = bench_summary["thresholds"]
        print(
            "\n=== Part A: frozen bench (fast_transcribe vs fw-large-v3 baseline) ==="
        )
        print(f"{'clip':<10} {'wall_s':>8} {'ladder':>7} {'kwWER':>8}")
        for row in bench_summary["per_clip"]:
            print(
                f"{row['id']:<10} {row['wall_s']:>8} {row['ladder_firings']:>7} {row['keyword_wer']:>8}"
            )
        print(f"\n{'metric':<18} {'fast_asr':>10} {'baseline':>10} {'required':>10}")
        print(
            f"{'corpus_wer':<18} {b['corpus_wer']!s:>10} {bench_summary['baseline']['corpus_wer']!s:>10} {'<= ' + str(t['corpus_wer_max']):>10}"
        )
        print(
            f"{'keyword_wer':<18} {b['keyword_wer_micro']!s:>10} {bench_summary['baseline']['keyword_wer_micro']!s:>10} {'<= ' + str(t['keyword_wer_max']):>10}"
        )
        print(
            f"{'drug_wer_folded':<18} {b['drug_wer_folded']!s:>10} {bench_summary['baseline']['drug_wer_folded']!s:>10} {'<= ' + str(t['drug_wer_folded_max']):>10}"
        )
        print(
            f"mean_decode_wall_s: {b['mean_decode_wall_s']}, total ladder firings: {b['total_ladder_firings']}"
        )
        print(f"GATE A: {bench_summary['gate']}")

    if session_summary:
        print("\n=== Part B: deployment session clips ===")
        for session_id, row in session_summary.items():
            print(
                f"{session_id}: duration={row['duration_s']}s decode={row['decode_wall_s']}s "
                f"(<=20s target: {'OK' if row['meets_20s_target'] else 'MISSED'}) "
                f"ladder_firings={row['ladder_firings']} arabic_script={row['arabic_script_present']} "
                f"drug '{row['expected_drug']}' found post-normalize: {row['drug_found_post_normalize']}"
            )

    results["summary"] = {"bench": bench_summary, "sessions": session_summary}
    _flush(results, out_path)
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s -- %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Gate for src.fast_asr.fast_transcribe."
    )
    parser.add_argument(
        "--bench", action="store_true", help="Part A: frozen 10-clip bench"
    )
    parser.add_argument(
        "--session", action="store_true", help="Part B: real deployment session clips"
    )
    parser.add_argument(
        "--score", action="store_true", help="score cached results (no models loaded)"
    )
    parser.add_argument("--out", default=RESULTS_PATH, help="results file path")
    args = parser.parse_args()

    if not (args.bench or args.session or args.score):
        parser.error("give at least one of --bench, --session, --score")
    if args.bench:
        run_bench(args.out)
    if args.session:
        run_session(args.out)
    if args.score:
        score(args.out)
