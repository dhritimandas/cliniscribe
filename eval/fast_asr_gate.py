"""Gate v2 for src.fast_asr.fast_transcribe / fast_transcribe_windowed -- the
decisive measurement between the two v2 fixes (language-allowlist guard,
window-packed decoding) against v1's FAIL verdict. v1's frozen-bench gate
(outputs/fast_asr_gate.json, kept as the historical record, untouched here)
FAILED outright: keyword_wer_micro 0.6429 > required 0.5814, corpus_wer
0.6694 > required 0.5388 -- drug_wer_folded 0.5714 <= required 0.8571 was
the only metric that already passed, despite the ladder resolving all 12
degenerate segments in that run. Root cause (src/fast_asr.py's module
docstring, Fix 1): some segments decoded into a wrong but FLUENT language
(Portuguese/Turkish/Indonesian), invisible to the degeneration detector.
See LEARNINGS.md's "Deployment Latency Phase" for the fixed ~7-9s per-call
cost Fix 2 (window-packed decoding) answers.

Two CONFIGS are benched side by side, BOTH running with the Fix 1 language
guard built unconditionally into src.fast_asr's decode-with-guard helpers
(there is no "guard off" mode to compare against):
    per_segment  src.fast_asr.fast_transcribe          (v1 engine + Fix 1)
    windowed     src.fast_asr.fast_transcribe_windowed (+ Fix 2)

Part A -- frozen bench (10 clips, `--bench --config {per_segment,windowed}`):
fast_transcribe()/fast_transcribe_windowed() decodes each clip's CACHED L2
diarization segments from outputs/engine_study.json -- the same segments
every engine this project has benched has decoded, so results are apples-to-
apples. Scored against the fw-large-v3 production baseline also cached
there (corpus_wer, keyword_wer, drug_wer_folded), identical gate formula to
v1:
    keyword_wer_micro <= baseline + eval.engine_study.KEYWORD_WER_TOLERANCE
    corpus_wer        <= baseline + eval.engine_study.CORPUS_WER_TOLERANCE
    drug_wer_folded   <= baseline (not worse)
Drug scoring applies the L3.5 normalize() pass and eval.drug_bench's
script-symmetric fold (_folded_match) over raw+normalized text combined, as
v1 did. For the windowed config only, per-clip rows also report ABSORPTION
CANDIDATES: reference keywords that are Latin-script, present in the
reference, missing from the raw windowed hypothesis, but recoverable by the
script-symmetric fold -- i.e. likely transliterated into Devanagari inside a
wide decode window, the SAME failure mode the project's earlier merged-
window study measured (LEARNINGS.md "Latency Phase"; "lab test" ->
"लाप टेस्ट", clip 431d272c). This is reported as a candidate count, not
asserted as proof -- it cannot distinguish absorption from a coincidental
fold match.

Part B -- deployment clips (real UI sessions, `--session --config ...`):
fast_transcribe()/fast_transcribe_windowed() decodes each session's real
diarized segments (cached for 20260710-230150-cef13a from a prior L2 run in
outputs/engine_study.json; diarized ONCE and cached in this module's own
results file for the other two, so both configs decode identical segments).
Reports wall time (this is where the "<=25s audio in <=20s" claim is proven
or disproven), an Arabic-script check, and a drug-keyword spot-check
post-normalize. The <=20s requirement is judged only on whichever config
wins Part A (see Part C).

Part C -- decision (`--score`): prints both configs' Part A tables and the
windowed config's absorption candidates, then whichever config PASSES Part
A's gate with the LOWER mean_decode_wall_s is the verdict; if both fail,
neither ships.

Run:
    PYTHONPATH=. python eval/fast_asr_gate.py --config per_segment --bench --session
    PYTHONPATH=. python eval/fast_asr_gate.py --config windowed --bench --session
    PYTHONPATH=. python eval/fast_asr_gate.py --score
"""

import argparse
import json
import logging
import os
import re
import tempfile
import time

from dotenv import load_dotenv

from eval.drug_bench import _folded_match
from eval.engine_study import CORPUS_WER_TOLERANCE, KEYWORD_WER_TOLERANCE
from eval.metrics import corpus_word_error_rate, keyword_hits
from eval.run_eval import _keywords_from_entities
from src.fast_asr import fast_transcribe, fast_transcribe_windowed
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
# v1's outputs/fast_asr_gate.json (FAIL) is left untouched as the historical
# record; its cached hypotheses predate Fix 1/Fix 2 and are not resumable
# under the current src/fast_asr.py, so v2 uses its own file.
RESULTS_PATH = "outputs/fast_asr_gate_v2.json"

CONFIG_FUNCS = {
    "per_segment": fast_transcribe,
    "windowed": fast_transcribe_windowed,
}

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
# cache anywhere and are diarized once (see _get_or_diarize_segments).
CEF13A_SESSION_ID = "20260710-230150-cef13a"
TARGET_WALL_S = 20.0  # brief: 25s audio must transcribe in <=20s, ladder included

_LATIN_KEYWORD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .\-]*$")


# ── Results file plumbing (mirrors eval/engine_study.py's convention) ────────


def _load_results(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {
        "configs": {name: {"per_clip": [], "sessions": {}} for name in CONFIG_FUNCS}
    }


def _flush(results: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _segments_from_cache(raw: list[dict]) -> list[Segment]:
    return [Segment(start=s["start"], end=s["end"], speaker=s["speaker"]) for s in raw]


def _get_or_diarize_segments(
    results: dict, session_id: str, wav_path: str, out_path: str
) -> tuple[list[Segment], str]:
    """Diarize a session's segments ONCE, cached in results['session_segments']
    so both configs decode IDENTICAL segments -- the same "same segments
    across engines" principle the frozen bench gets for free from
    outputs/engine_study.json's cache."""
    if session_id == CEF13A_SESSION_ID:
        with open(ENGINE_STUDY_PATH, encoding="utf-8") as f:
            engine_study = json.load(f)
        segments = _segments_from_cache(engine_study["session"]["segments"])
        return segments, f"cached ({ENGINE_STUDY_PATH} session)"

    cache = results.setdefault("session_segments", {})
    if session_id in cache:
        return _segments_from_cache(cache[session_id]), f"cached ({out_path})"

    from src.l2_diarize import diarize

    t0 = time.perf_counter()
    segments = diarize(wav_path)
    diarize_wall = time.perf_counter() - t0
    cache[session_id] = [
        {"start": s.start, "end": s.end, "speaker": s.speaker} for s in segments
    ]
    _flush(results, out_path)
    return segments, f"live diarize() ({diarize_wall:.2f}s)"


class _FastAsrLogHandler(logging.Handler):
    """Captures every WARNING src.fast_asr emits during one decode call --
    ladder firings ("... ladder fired ...") AND guard firings ("... script
    guard ...", "... language guard ..."), for both configs' log lines."""

    def __init__(self, sink: list[str]) -> None:
        super().__init__(level=logging.WARNING)
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self._sink.append(record.getMessage())


def _timed_transcribe(
    config_name: str, wav_path: str, segments: list[Segment]
) -> tuple[list[Turn], float, list[str]]:
    """fast_transcribe()/fast_transcribe_windowed(), wall-clock timed, with
    every src.fast_asr WARNING captured (see _FastAsrLogHandler)."""
    log_sink: list[str] = []
    handler = _FastAsrLogHandler(log_sink)
    fast_asr_logger = logging.getLogger("src.fast_asr")
    fast_asr_logger.addHandler(handler)
    func = CONFIG_FUNCS[config_name]
    t0 = time.perf_counter()
    try:
        turns = func(wav_path, segments)
    finally:
        fast_asr_logger.removeHandler(handler)
    wall = time.perf_counter() - t0
    return turns, wall, log_sink


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


def _is_latin_keyword(kw: str) -> bool:
    """True if `kw` is ASCII/Latin-script -- a candidate for code-switch
    absorption into Devanagari inside a wide decode window."""
    return bool(_LATIN_KEYWORD_RE.match(kw))


def _absorption_candidates(ref: str, raw_hyp: str, kws: list[str]) -> list[str]:
    """Latin-script reference keywords missing from the RAW hypothesis but
    recoverable by the script-symmetric fold -- likely transliterated into
    Devanagari rather than genuinely dropped. A candidate list, not proof:
    see module docstring."""
    return [
        kw
        for kw in kws
        if _is_latin_keyword(kw)
        and keyword_hits(ref, raw_hyp, [kw]) == (1, 1)
        and _folded_match(kw, raw_hyp)
    ]


# ── Part A: frozen bench ─────────────────────────────────────────────────────


def run_bench(config_name: str, out_path: str = RESULTS_PATH) -> None:
    """Decode the frozen 10-clip bench with the named config on cached L2 segments."""
    import pandas as pd

    with open(ENGINE_STUDY_PATH, encoding="utf-8") as f:
        engine_study = json.load(f)
    cached_by_id = {c["id"]: c for c in engine_study["per_clip"]}

    df = pd.read_parquet(ASR_DATASET).head(FROZEN_N)
    results = _load_results(out_path)
    config_results = results["configs"].setdefault(
        config_name, {"per_clip": [], "sessions": {}}
    )
    done = {c["id"] for c in config_results["per_clip"]}

    for i, (_, row) in enumerate(df.iterrows()):
        clip_id = row["md5_text"]
        if clip_id in done:
            logger.info(
                "[%d/%d] %s (%s) already done -- skipping",
                i + 1,
                FROZEN_N,
                clip_id[:8],
                config_name,
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

        turns, wall, log_sink = _timed_transcribe(config_name, wav_path, segments)
        hyp = " ".join(t.text for t in turns).strip()

        config_results["per_clip"].append(
            {
                "id": clip_id,
                "hypothesis": hyp,
                "decode_wall_s": round(wall, 3),
                "log": log_sink,
            }
        )
        logger.info(
            "[%d/%d] %s (%s): %.2fs, %d log line(s) -- %s",
            i + 1,
            FROZEN_N,
            clip_id[:8],
            config_name,
            wall,
            len(log_sink),
            hyp[:80],
        )
        _flush(results, out_path)

    print(f"Bench results ({config_name}): {out_path}")


# ── Part B: deployment session clips ─────────────────────────────────────────


def run_session(config_name: str, out_path: str = RESULTS_PATH) -> None:
    """Decode every real UI session clip in SESSION_CLIPS with the named config."""
    import soundfile as sf

    results = _load_results(out_path)
    config_results = results["configs"].setdefault(
        config_name, {"per_clip": [], "sessions": {}}
    )
    sessions = config_results["sessions"]

    for session_id, (wav_path, _expected_drug) in SESSION_CLIPS.items():
        if session_id in sessions:
            logger.info("%s (%s) already done -- skipping", session_id, config_name)
            continue
        if not os.path.exists(wav_path):
            logger.warning("%s: %s not found -- skipping", session_id, wav_path)
            continue

        segments, segment_source = _get_or_diarize_segments(
            results, session_id, wav_path, out_path
        )
        duration_s = round(sf.info(wav_path).duration, 3)
        turns, wall, log_sink = _timed_transcribe(config_name, wav_path, segments)
        hyp = " ".join(t.text for t in turns).strip()

        sessions[session_id] = {
            "duration_s": duration_s,
            "n_segments": len(segments),
            "segment_source": segment_source,
            "decode_wall_s": round(wall, 3),
            "log": log_sink,
            "hypothesis": hyp,
            "arabic_script_present": _contains_arabic_script(hyp),
        }
        logger.info(
            "%s (%s, %.2fs audio): decode=%.2fs, %d log line(s)",
            session_id,
            config_name,
            duration_s,
            wall,
            len(log_sink),
        )
        _flush(results, out_path)

    print(f"Session results ({config_name}): {out_path}")


# ── Scoring ──────────────────────────────────────────────────────────────────


def _score_bench(config_name: str, results: dict) -> dict:
    import pandas as pd

    with open(ENGINE_STUDY_PATH, encoding="utf-8") as f:
        baseline = json.load(f)["summary"]["fw-large-v3"]

    df = pd.read_parquet(ASR_DATASET).head(FROZEN_N)
    by_id = {c["id"]: c for c in results["configs"][config_name]["per_clip"]}

    refs: list[str] = []
    hyps: list[str] = []
    per_clip_rows: list[dict] = []
    kw_present = kw_missed = 0
    drug_present_f = drug_missed_f = 0

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

        norm_turns = normalize(
            [Turn(speaker_role="UNKNOWN", text=hyp, start=0.0, end=0.0)]
        )
        norm_hyp = " ".join(t.text for t in norm_turns).strip()
        dpf, dmf = _drug_wer_folded(ref, hyp, norm_hyp, kw_drug)
        drug_present_f += dpf
        drug_missed_f += dmf

        absorption = (
            _absorption_candidates(ref, hyp, kw) if config_name == "windowed" else []
        )
        log_lines = rec.get("log", [])

        ladder_firings = sum(1 for line in log_lines if "ladder fired" in line)
        guard_firings = sum(1 for line in log_lines if "guard" in line)
        per_clip_rows.append(
            {
                "id": clip_id[:8],
                "wall_s": rec["decode_wall_s"],
                "ladder_firings": ladder_firings,
                "guard_firings": guard_firings,
                "keyword_wer": round(m / p, 4) if p else 0.0,
                "absorption_candidates": absorption,
            }
        )

    n_scored = len(hyps)
    metrics = {
        "n_clips_scored": n_scored,
        "corpus_wer": round(corpus_word_error_rate(refs, hyps), 4) if refs else None,
        "keyword_wer_micro": round(kw_missed / kw_present, 4) if kw_present else 0.0,
        "drug_wer_folded": round(drug_missed_f / drug_present_f, 4)
        if drug_present_f
        else 0.0,
        "mean_decode_wall_s": (
            round(sum(r["wall_s"] for r in per_clip_rows) / n_scored, 2)
            if n_scored
            else None
        ),
        "total_ladder_firings": sum(r["ladder_firings"] for r in per_clip_rows),
        "total_guard_firings": sum(r["guard_firings"] for r in per_clip_rows),
        "total_absorption_candidates": sum(
            len(r["absorption_candidates"]) for r in per_clip_rows
        ),
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
        "config": config_name,
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


def _score_sessions(config_name: str, results: dict) -> dict:
    sessions = results["configs"][config_name].get("sessions", {})
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
        log_lines = rec.get("log", [])
        rows[session_id] = {
            "duration_s": rec["duration_s"],
            "decode_wall_s": rec["decode_wall_s"],
            "meets_20s_target": rec["decode_wall_s"] <= TARGET_WALL_S,
            "ladder_firings": sum(1 for line in log_lines if "ladder fired" in line),
            "guard_firings": sum(1 for line in log_lines if "guard" in line),
            "arabic_script_present": rec["arabic_script_present"],
            "expected_drug": expected_drug,
            "drug_found_post_normalize": drug_found,
        }
    return rows


def score(out_path: str = RESULTS_PATH) -> None:
    with open(out_path, encoding="utf-8") as f:
        results = json.load(f)

    bench_summaries: dict[str, dict] = {}
    session_summaries: dict[str, dict] = {}
    for config_name in CONFIG_FUNCS:
        cfg = results.get("configs", {}).get(config_name, {})
        if cfg.get("per_clip"):
            bench_summaries[config_name] = _score_bench(config_name, results)
        if cfg.get("sessions"):
            session_summaries[config_name] = _score_sessions(config_name, results)

    for config_name, bench_summary in bench_summaries.items():
        b = bench_summary["metrics"]
        t = bench_summary["thresholds"]
        print(f"\n=== Part A: frozen bench -- config={config_name} ===")
        print(
            f"{'clip':<10} {'wall_s':>8} {'ladder':>7} {'guard':>6} "
            f"{'kwWER':>8} {'absorb':>7}"
        )
        for row in bench_summary["per_clip"]:
            print(
                f"{row['id']:<10} {row['wall_s']:>8} {row['ladder_firings']:>7} "
                f"{row['guard_firings']:>6} {row['keyword_wer']:>8} "
                f"{len(row['absorption_candidates']):>7}"
            )
        print(f"\n{'metric':<18} {config_name:>14} {'baseline':>10} {'required':>10}")
        print(
            f"{'corpus_wer':<18} {b['corpus_wer']!s:>14} "
            f"{bench_summary['baseline']['corpus_wer']!s:>10} "
            f"{'<= ' + str(t['corpus_wer_max']):>10}"
        )
        print(
            f"{'keyword_wer':<18} {b['keyword_wer_micro']!s:>14} "
            f"{bench_summary['baseline']['keyword_wer_micro']!s:>10} "
            f"{'<= ' + str(t['keyword_wer_max']):>10}"
        )
        print(
            f"{'drug_wer_folded':<18} {b['drug_wer_folded']!s:>14} "
            f"{bench_summary['baseline']['drug_wer_folded']!s:>10} "
            f"{'<= ' + str(t['drug_wer_folded_max']):>10}"
        )
        print(
            f"mean_decode_wall_s: {b['mean_decode_wall_s']}, "
            f"ladder firings: {b['total_ladder_firings']}, "
            f"guard firings: {b['total_guard_firings']}, "
            f"absorption candidates: {b['total_absorption_candidates']}"
        )
        print(f"GATE A ({config_name}): {bench_summary['gate']}")
        if config_name == "windowed" and b["total_absorption_candidates"]:
            print(
                "Absorption candidates by clip (Latin keyword missing raw, "
                "fold-recovered):"
            )
            for row in bench_summary["per_clip"]:
                if row["absorption_candidates"]:
                    print(f"  {row['id']}: {row['absorption_candidates']}")

    for config_name, session_summary in session_summaries.items():
        print(f"\n=== Part B: deployment session clips -- config={config_name} ===")
        for session_id, row in session_summary.items():
            print(
                f"{session_id}: duration={row['duration_s']}s "
                f"decode={row['decode_wall_s']}s "
                f"(<=20s target: {'OK' if row['meets_20s_target'] else 'MISSED'}) "
                f"ladder={row['ladder_firings']} guard={row['guard_firings']} "
                f"arabic={row['arabic_script_present']} "
                f"drug '{row['expected_drug']}' found post-normalize: "
                f"{row['drug_found_post_normalize']}"
            )

    print("\n=== Part C: decision ===")
    passing = {name: s for name, s in bench_summaries.items() if s["gate"] == "PASS"}
    decision = None
    if not passing:
        print(
            "NEITHER config passed Part A's accuracy gate. FAST_ASR_ENABLED stays "
            "False; a full failure analysis is needed before another attempt."
        )
    else:
        decision = min(
            passing, key=lambda name: passing[name]["metrics"]["mean_decode_wall_s"]
        )
        print(
            f"Winner: {decision} (mean_decode_wall_s="
            f"{passing[decision]['metrics']['mean_decode_wall_s']})"
        )
        if decision in session_summaries:
            winner_rows = session_summaries[decision]
            all_ok = all(row["meets_20s_target"] for row in winner_rows.values())
            print(
                f"<=20s deployment target on the winner ({decision}): "
                f"{'MET' if all_ok else 'NOT MET'} for all session clips"
            )

    results["summary"] = {
        "bench": bench_summaries,
        "sessions": session_summaries,
        "decision": decision,
    }
    _flush(results, out_path)
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s -- %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Gate v2 for src.fast_asr (per_segment vs windowed)."
    )
    parser.add_argument(
        "--config",
        choices=sorted(CONFIG_FUNCS),
        help="which fast-ASR entry point to run (required for --bench/--session)",
    )
    parser.add_argument(
        "--bench", action="store_true", help="Part A: frozen 10-clip bench"
    )
    parser.add_argument(
        "--session", action="store_true", help="Part B: real deployment session clips"
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="score cached results for BOTH configs (no models loaded)",
    )
    parser.add_argument("--out", default=RESULTS_PATH, help="results file path")
    args = parser.parse_args()

    if not (args.bench or args.session or args.score):
        parser.error("give at least one of --bench, --session, --score")
    if (args.bench or args.session) and not args.config:
        parser.error("--bench/--session require --config {per_segment,windowed}")
    if args.bench:
        run_bench(args.config, args.out)
    if args.session:
        run_session(args.config, args.out)
    if args.score:
        score(args.out)
