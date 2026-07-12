"""Full CliniScribe pipeline: audio → draft prescription PDF.

Stages execute sequentially. Each model is loaded, used, and released before
the next stage begins — never hold ASR and LLM in memory simultaneously.

All artifacts of one consultation live under outputs/<session_id>/:
the preprocessed audio, the speaker-attributed transcript, the structured
note, the draft PDF, and (future) physician corrections share that one ID.
"""

import dataclasses
import gc
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import Callable

from dotenv import load_dotenv

import threading

from src import telemetry
from src.fast_asr import fast_transcribe_windowed
from src.l1_preprocess import preprocess
from src.l2_diarize import diarize
from src.l3_asr import transcribe
from src.l3_5_normalize import normalize
from src.l4_extract import extract, warm_llm
from src.l5_render import render
from src.types import Turn

load_dotenv()
logger = logging.getLogger(__name__)

OUTPUTS_ROOT = "outputs"


def new_session_id() -> str:
    """Return a sortable, collision-safe session ID (timestamp + 6 hex chars)."""
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def _write_turns(turns: list[Turn], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([dataclasses.asdict(t) for t in turns], f, ensure_ascii=False, indent=2)


def run(
    in_path: str,
    session_id: str | None = None,
    *,
    on_stage: Callable[[str, str], None] | None = None,
    on_progress: Callable[[float, float], None] | None = None,
    asr_engine: str = "accurate",
) -> str:
    """Run the full pipeline on an audio file and return the PDF path.

    Args:
        in_path: Path to the input audio file.
        session_id: Consultation session ID; generated when omitted. All
            artifacts are written under outputs/<session_id>/.
        on_stage: Optional callback invoked as `on_stage(name, event)` around
            each stage, with `event` in `{"start", "end"}` and `name` one of
            the stage_report keys (e.g. "l3_asr"). Used by the review-frontend
            backend to mirror progress into status.json. Default None keeps
            current behavior unchanged.
        on_progress: Optional callback passed through to L3's `transcribe()`
            as `on_progress(done_seconds, total_seconds)`, invoked after each
            segment decodes during L3 ASR. Used by the review-frontend backend
            for the percent/ETA display. Default None keeps current behavior.
        asr_engine: Which L3 transcription engine to use — "accurate"
            (src.l3_asr.transcribe, faster-whisper; the default, CLI behavior
            unchanged) or "fast" (src.fast_asr.fast_transcribe_windowed,
            mlx-whisper; see src.config.FAST_ASR_ENABLED). The web review
            frontend passes "fast" once that flag is on, then runs a
            background verification pass (web/verification.py) that
            re-checks safety-critical fields against a second, slower decode.

    Returns:
        Path to the generated draft prescription PDF
        (outputs/<session_id>/draft_rx.pdf).
    """
    if asr_engine not in ("accurate", "fast"):
        raise ValueError(f"asr_engine must be 'accurate' or 'fast', got {asr_engine!r}")
    transcribe_fn = fast_transcribe_windowed if asr_engine == "fast" else transcribe

    session_id = session_id or new_session_id()
    session_dir = os.path.join(OUTPUTS_ROOT, session_id)
    os.makedirs(session_dir, exist_ok=True)
    logger.info("Session %s → %s", session_id, session_dir)

    telemetry.reset()
    stage_report: dict[str, dict[str, float]] = {}

    def _staged(name: str, fn, *args, **kwargs):
        if on_stage:
            on_stage(name, "start")
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        gc.collect()
        stage_report[name] = {
            "wall_s": round(time.perf_counter() - t0, 2),
            "peak_rss_mb_so_far": telemetry.peak_rss_mb(),
        }
        if on_stage:
            on_stage(name, "end")
        return result

    logger.info("L1: preprocessing %s", in_path)
    wav_path = _staged("l1_preprocess", preprocess, in_path, out_dir=session_dir)

    logger.info("L2: diarizing %s", wav_path)
    segments = _staged("l2_diarize", diarize, wav_path)

    logger.info("L3: transcribing %d segments (engine=%s)", len(segments), asr_engine)
    turns = _staged("l3_asr", transcribe_fn, wav_path, segments, on_progress=on_progress)

    # Warm the LLM while L3.5 runs on CPU: Whisper was released inside
    # transcribe(), so only the (small) embedding model and Qwen coexist —
    # the load-one-release-one discipline holds at its peak.
    warm_thread = threading.Thread(target=warm_llm, daemon=True)
    with telemetry.timer("l4.warm_dispatch"):
        warm_thread.start()

    logger.info("L3.5: normalizing %d turns", len(turns))
    turns = _staged("l3_5_normalize", normalize, turns)
    _write_turns(turns, os.path.join(session_dir, "transcript.json"))

    with telemetry.timer("l4.warm_join_wait"):
        warm_thread.join(timeout=180)

    logger.info("L4: extracting clinical entities")
    note = _staged("l4_extract", extract, turns)
    with open(os.path.join(session_dir, "note.json"), "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(note), f, ensure_ascii=False, indent=2)

    logger.info("L5: rendering prescription PDF")
    pdf_path = _staged(
        "l5_render", render, note, out_path=os.path.join(session_dir, "draft_rx.pdf")
    )

    timings = {
        "stages": stage_report,
        "sub_timings": telemetry.snapshot(),  # model loads recorded by stages
        "total_wall_s": round(sum(s["wall_s"] for s in stage_report.values()), 2),
        "peak_rss_mb": telemetry.peak_rss_mb(),
    }
    with open(os.path.join(session_dir, "timings.json"), "w", encoding="utf-8") as f:
        json.dump(timings, f, indent=2)
    logger.info(
        "Timings: total %.1fs, peak RSS %.0f MB — %s",
        timings["total_wall_s"],
        timings["peak_rss_mb"],
        {k: v["wall_s"] for k, v in stage_report.items()},
    )

    logger.info("Done: %s", pdf_path)
    return pdf_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    if len(sys.argv) != 2:
        print("Usage: python src/pipeline.py <audio_path>")
        sys.exit(1)
    print(run(sys.argv[1]))
