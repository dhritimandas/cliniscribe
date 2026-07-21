"""Stop-to-note latency benchmark — the sub-10s goal's acceptance gate.

Measures perceived latency for a live-recorded consultation: the wall-clock
time from "recording stopped" to "clinical note is ready" (`stop_to_note_s`
in timings.json — see src.pipeline.run's `stop_monotonic_ts` parameter).

Today (before the incremental-capture waves land) there is no live capture
loop, so this benchmark exercises the batch entry point end to end and
reports `stop_to_note_s` measured from run() entry — the same "worst case,
nothing hidden during recording" number the post-stop budget table in the
latency plan is keyed against. Once W3 (incremental capture) ships, this
harness gains a `--simulate-recording` mode that feeds the fixture to
`web.incremental.IncrementalSession` in real-time chunks before calling
`finalize()`, so the reported number reflects the intended architecture
instead of the pre-W3 floor.

Fixture: a deterministic ~2.5 minute clip built by concatenating the
project's three sample clips (data/sample_00.mp3 + 01 + 02, ~44s combined)
with 1s silence gaps, repeated to reach the target duration. Concatenation
is content-deterministic (same bytes in, same bytes out) and cached under
outputs/cache/ so repeated benchmark runs don't rebuild it.

Usage:
    python -m bench.stop_to_note_bench                  # N=10, ~2.5 min fixture
    python -m bench.stop_to_note_bench --n 3 --fixture-seconds 150
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import statistics
import sys
import time

import numpy as np
import soundfile as sf

from src import pipeline

logger = logging.getLogger(__name__)

_SAMPLE_CLIPS = ["data/sample_00.mp3", "data/sample_01.mp3", "data/sample_02.mp3"]
_SILENCE_GAP_S = 1.0
_TARGET_SR = 16_000
_DEFAULT_FIXTURE_SECONDS = 150.0  # 2.5 min — inside the "2-3 min consult" range
_DEFAULT_N = 10
_CACHE_DIR = os.path.join("outputs", "cache")
_RESULTS_PATH = os.path.join("outputs", "stop_to_note_bench.json")


def _fixture_cache_path(target_seconds: float) -> str:
    key = hashlib.sha256(
        f"{_SAMPLE_CLIPS}|{_SILENCE_GAP_S}|{target_seconds}".encode()
    ).hexdigest()[:12]
    return os.path.join(_CACHE_DIR, f"stop_to_note_fixture_{key}.wav")


def build_fixture(target_seconds: float = _DEFAULT_FIXTURE_SECONDS) -> str:
    """Return the path to a deterministic ~target_seconds fixture WAV.

    Built by looping the concatenated sample clips (with silence gaps
    between repeats, not just between clips, so no decode window straddles
    a hard splice with zero context) until target_seconds is reached, then
    trimmed to exactly that length. Cached by content+target hash.
    """
    cache_path = _fixture_cache_path(target_seconds)
    if os.path.exists(cache_path):
        return cache_path

    import librosa

    clips = [librosa.load(p, sr=_TARGET_SR, mono=True)[0] for p in _SAMPLE_CLIPS]
    silence = np.zeros(int(_SILENCE_GAP_S * _TARGET_SR), dtype=np.float32)

    one_loop: list[np.ndarray] = []
    for clip in clips:
        one_loop.append(clip)
        one_loop.append(silence)
    loop_audio = np.concatenate(one_loop)

    target_samples = int(target_seconds * _TARGET_SR)
    reps = target_samples // len(loop_audio) + 1
    full = np.concatenate([loop_audio] * reps)[:target_samples]

    os.makedirs(_CACHE_DIR, exist_ok=True)
    sf.write(cache_path, full, _TARGET_SR, subtype="PCM_16")
    logger.info(
        "Built fixture: %.1fs from %d clips x%d reps -> %s",
        len(full) / _TARGET_SR,
        len(clips),
        reps,
        cache_path,
    )
    return cache_path


def _percentile(values: list[float], pct: float) -> float:
    if len(values) == 1:
        return values[0]
    ranked = sorted(values)
    idx = min(len(ranked) - 1, max(0, round(pct / 100 * (len(ranked) - 1))))
    return ranked[idx]


def run_bench(n: int = _DEFAULT_N, fixture_seconds: float = _DEFAULT_FIXTURE_SECONDS) -> dict:
    fixture_path = build_fixture(fixture_seconds)

    runs = []
    for i in range(n):
        session_id = f"bench-stop-to-note-{i:02d}"
        t0 = time.monotonic()
        pipeline.run(
            fixture_path,
            session_id=session_id,
            asr_engine="fast",
            stop_monotonic_ts=t0,
        )
        gc.collect()
        timings_path = os.path.join(pipeline.OUTPUTS_ROOT, session_id, "timings.json")
        with open(timings_path, encoding="utf-8") as f:
            timings = json.load(f)
        runs.append(timings)
        logger.info(
            "Run %d/%d: stop_to_note_s=%.2f peak_rss_mb=%.0f",
            i + 1,
            n,
            timings["stop_to_note_s"],
            timings["peak_rss_mb"],
        )

    stop_to_note = [r["stop_to_note_s"] for r in runs]
    stop_to_pdf = [r["stop_to_pdf_s"] for r in runs]
    peak_rss = [r["peak_rss_mb"] for r in runs]

    per_stage: dict[str, list[float]] = {}
    for r in runs:
        for stage, s in r["stages"].items():
            per_stage.setdefault(stage, []).append(s["wall_s"])

    summary = {
        "n": n,
        "fixture_seconds": fixture_seconds,
        "fixture_path": fixture_path,
        "stop_to_note_s": {
            "p50": _percentile(stop_to_note, 50),
            "p95": _percentile(stop_to_note, 95),
            "runs": stop_to_note,
        },
        "stop_to_pdf_s": {
            "p50": _percentile(stop_to_pdf, 50),
            "p95": _percentile(stop_to_pdf, 95),
            "runs": stop_to_pdf,
        },
        "peak_rss_mb": {
            "p50": _percentile(peak_rss, 50),
            "max": max(peak_rss),
            "runs": peak_rss,
        },
        "per_stage_wall_s_median": {
            stage: round(statistics.median(vals), 2) for stage, vals in per_stage.items()
        },
        "acceptance": {
            "target_p95_s": 10.0,
            "passed": _percentile(stop_to_note, 95) < 10.0,
        },
    }

    os.makedirs(os.path.dirname(_RESULTS_PATH), exist_ok=True)
    with open(_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=_DEFAULT_N)
    parser.add_argument("--fixture-seconds", type=float, default=_DEFAULT_FIXTURE_SECONDS)
    args = parser.parse_args()

    summary = run_bench(n=args.n, fixture_seconds=args.fixture_seconds)
    print(json.dumps(summary, indent=2))
    verdict = "PASS" if summary["acceptance"]["passed"] else "FAIL"
    print(
        f"\n{verdict}: stop_to_note_s p50={summary['stop_to_note_s']['p50']:.2f}s "
        f"p95={summary['stop_to_note_s']['p95']:.2f}s (target p95 < 10.0s)"
    )
    sys.exit(0 if summary["acceptance"]["passed"] else 1)


if __name__ == "__main__":
    main()
