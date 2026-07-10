"""Lightweight in-process instrumentation for pipeline latency reporting.

Stages record named durations (model loads, notable sub-steps) into a
module-level registry; the pipeline orchestrator resets it per run and folds
the values into its per-stage timing report. Stdlib-only, no behavior change
when nobody reads the registry.
"""

import resource
import sys
import time

_timings: dict[str, float] = {}


def reset() -> None:
    """Clear recorded timings (call at the start of a pipeline run)."""
    _timings.clear()


def record(key: str, seconds: float) -> None:
    """Record a named duration in seconds (last write wins per key)."""
    _timings[key] = round(seconds, 3)


def snapshot() -> dict[str, float]:
    """Return a copy of all recorded timings."""
    return dict(_timings)


def peak_rss_mb() -> float:
    """Process peak RSS in MB (ru_maxrss: bytes on macOS, KB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(peak / divisor, 1)


class timer:
    """Context manager: `with timer("l3.model_load"): ...` records elapsed."""

    def __init__(self, key: str):
        self.key = key

    def __enter__(self) -> "timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        record(self.key, time.perf_counter() - self._t0)
