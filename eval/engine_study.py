"""L3 ASR engine speed/accuracy probe: faster-whisper large-v3 (production
baseline, cached — never rerun here) vs three faster candidates, gated on the
frozen 10-clip Hindi bench's keyword/drug WER (token-boundary scorer, same as
eval/beam_study.py and eval/script_guard_bench.py).

Candidates
----------
fw-turbo      faster-whisper, model="large-v3-turbo". faster_whisper.utils
              resolves this to mobiuslabsgmbh/faster-whisper-large-v3-turbo
              (an existing CT2 conversion — no custom repo needed). Same CPU
              int8 path, same production transcribe() contract (script
              guard, beam_size, vad_filter no-op) — reused verbatim via
              src.l3_asr.transcribe(model=...); only the model kwarg differs.
mlx-large-v3  mlx-whisper on the Metal GPU, mlx-community/whisper-large-v3-mlx.
mlx-turbo     mlx-whisper on the Metal GPU, mlx-community/whisper-large-v3-turbo.

mlx-whisper is a DIFFERENT backend (a module-level mlx_whisper.transcribe()
call, not a WhisperModel object), so its per-segment decode is hand-rolled
below rather than reused from src.l3_asr.transcribe(). Two API facts, found
by reading the installed mlx-whisper==0.4.3 source (not assumed from docs):
  1. `clip_timestamps` IS supported (`transcribe(..., clip_timestamps="s,e")`)
     with faster-whisper-identical seek-window semantics, so no manual
     per-segment audio slicing was needed — contrary to this study's brief.
  2. `mlx_whisper.audio.load_audio` shells out to the `ffmpeg` CLI, which is
     NOT installed on this machine (the same ffmpeg gap that broke torchcodec
     for L2 — see LEARNINGS Part 1). We therefore decode the WAV ourselves
     once per clip via soundfile and pass an in-memory float32 array to
     `transcribe()`, never a path — the audio-decoupling rule L1 established.
mlx's decoder has no beam search; its defaults (undocumented by this probe's
brief, so recorded here): temperature fallback ladder
(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold=2.4,
logprob_threshold=-1.0, no_speech_threshold=0.6. This is stock Whisper
greedy/sampling decoding, not comparable to faster-whisper's beam_size=5 —
documented, not tuned, for this probe.

Method (mirrors eval/script_guard_bench.py: preloaded models, incremental
JSON, resumable per clip):
  1. `--engine {fw-turbo,mlx-large-v3,mlx-turbo} --bench` runs L1 preprocess +
     L2 diarize ONCE per clip (segments cached into the results file so every
     engine decodes identical segments), then the named engine's per-segment
     decode, recording wall time and the concatenated hypothesis.
  2. `--engine ... --session` runs the same engine once on the user's real
     session clip (outputs/20260710-230150-cef13a/input_16k.wav).
  3. `--score` (no models loaded) scores every engine present in the results
     file: corpus WER, keyword WER, drug WER (raw + folded-after-normalize,
     reusing eval.drug_bench's script-symmetric fold), against the
     fw-large-v3 baseline read directly from outputs/beam_study.json
     beams["5"] — cached, never rerun here (that decode already cost the
     ~10-20 min this study is trying to avoid repeating).

Merged-window follow-up (`--merge`): all three candidates above hallucinate
into repetition loops on short per-segment fragments (e.g. "झाल झाल झाल...").
`merge_segments()` (pure; unit-tested directly) joins adjacent same-speaker
diarized segments into windows (capped at MERGE_MAX_WINDOW_S, gap tolerance
MERGE_GAP_TOLERANCE_S) BEFORE decode, on the same cached L2 segments, so
per-window language auto-detection is preserved (unlike whole-file decode,
separately rejected — see LEARNINGS.md Latency Phase). `--engine fw-large-v3
--merge` is the CONTROL: the exact production model/beam on merged windows,
isolating the windowing effect from any engine swap. Results are stored under
"<engine>+merged" keys alongside the per-segment runs above, in the same
results file. `--score` gates merged configs on drug_wer_folded (must not
worsen vs. the fw-large-v3 baseline) rather than drug_wer_raw — see `_gate`.

Anti-hallucination decode-param probe (`--decode-config {a1,a2}`) — NEGATIVE
RESULT, both engines, both configs. `KNOWN_BAD_CLIP_IDS` (f2096fbd, 1ae62262)
are the two clips whose per-segment/merged hypotheses above show the worst
repetition loops; `probe_antihallu()` fail-fast-checks a decode-param config
against ONLY these two clips (reusing cached L2 segments — no diarization
pipeline needed) before any full-bench run would be considered.
`ANTIHALLU_CONFIGS`:
  a1 — temperature=0.0 (mlx) / [0.0] (fw), condition_on_previous_text=False.
  a2 — a1 + logprob_threshold=-0.3 (mlx) / log_prob_threshold=-0.3 (fw), a
       stricter (less negative) bound intended to force the silence-skip
       path to reject confidently-hallucinated near-silent segments.
Measured (mlx-whisper 0.4.3 source read directly, not assumed — see
`transcribe()` in `mlx_whisper/transcribe.py`): `temperature` as a scalar
(rather than the library default six-rung ladder 0.0,0.2,...,1.0) makes
`decode_with_fallback` run exactly one attempt, so `compression_ratio_threshold`
can no longer trigger escalation to a different sample — it and
`logprob_threshold`/`no_speech_threshold` then only gate the SEPARATE
silence-skip check (`should_skip`), not what text gets returned once a
window is decoded. Diagnostic decode of the two known-bad clips under STOCK
DEFAULT params (full ladder) showed the ladder is not the problem: on
1ae62262's dominant [3.2,19.7]s window (16.5s of genuine speech,
no_speech_prob=0.04 — confirmed NOT silence), the ladder already escalated
to temp=1.0 and STILL produced a repetition loop
("पैनियान से पैनियान से..."). a1 (mlx-large-v3, same window) loops
differently but still loops ("यह बात कर रहा है कि..." x13); a2 changes
nothing there either (logprob=-0.16, well above -0.3, so the silence-skip
override never engages). fw-turbo a1 on f2096fbd is actively WORSE than its
own default-param baseline: default fw-turbo decodes "चार कादिसे" cleanly at
that timestamp; a1 (temperature=[0.0] only) loops "चाहर चाहर चाहर..." ~28x
instead — disabling the ladder removed fw-turbo's escape valve without
removing the greedy decoder's tendency to loop. mlx-turbo a1 also still
loops on f2096fbd ("ट्ट्ट्ट्ट्ट्ट्ट्ट्ट्ट्ट्ट्...", "झाल झाल झाल..." x40+).
Conclusion: Probe A fails fast on both known-bad clips, on all three fast
engines, under both configs — no config was promoted to a full --bench run
(would cost the same 10-20 min this study exists to avoid, for a result
already determined on the clips that define the failure mode).

Probe B (mlx VAD pre-slicing) was ruled out WITHOUT building a pre-slicing
harness: `mlx_whisper.transcribe()` has no `vad_filter` parameter at all
(confirmed from the same source read — it only has `clip_timestamps`, which
is seek-window slicing, not silence detection), matching this probe's brief.
Silero VAD (`faster_whisper.vad.get_speech_timestamps`, already a project
dependency) was run directly on every segment of both known-bad clips: EVERY
segment implicated in a repetition loop is >=74% voiced (most are 100%
voiced, including the dominant 16.5s 1ae62262 window). VAD pre-slicing only
helps when hallucination is triggered by silence; these hallucinations occur
on audio VAD itself confirms is speech, so pre-slicing cannot suppress them.
Not implemented, since the one segment VAD *would* drop (1ae62262's trailing
0.39s, 0% voiced) already decodes to an uncontroversial filler word ("Uh-huh"
/ "आहा") in every engine tried — dropping it cannot move the gate.

Run:
  PYTHONPATH=. python eval/engine_study.py --engine fw-turbo --bench --session
  PYTHONPATH=. python eval/engine_study.py --engine mlx-large-v3 --bench --session
  PYTHONPATH=. python eval/engine_study.py --engine mlx-turbo --bench --session
  PYTHONPATH=. python eval/engine_study.py --engine fw-large-v3 --merge --bench --session
  PYTHONPATH=. python eval/engine_study.py --engine mlx-large-v3 --merge --bench --session
  PYTHONPATH=. python eval/engine_study.py --engine mlx-turbo --merge --bench --session
  PYTHONPATH=. python eval/engine_study.py --score
  PYTHONPATH=. python eval/engine_study.py --engine mlx-large-v3 --decode-config a1
  PYTHONPATH=. python eval/engine_study.py --engine fw-turbo --decode-config a2
"""

import argparse
import json
import logging
import os
import tempfile
import time

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

from eval.drug_bench import _folded_match
from eval.metrics import corpus_word_error_rate, keyword_hits
from eval.run_eval import _keywords_from_entities
from src import config
from src.l1_preprocess import preprocess
from src.l2_diarize import diarize
from src.l3_asr import _contains_arabic_script, transcribe
from src.types import Segment, Turn

load_dotenv()
logger = logging.getLogger(__name__)

ASR_DATASET = "eka-medical-asr-dataset/hi/test-00000.parquet"
FROZEN_N = 10  # must match eval/frozen_set_asr.json (deterministic first-N rows)
RESULTS_PATH = "outputs/engine_study.json"
BEAM_STUDY_PATH = "outputs/beam_study.json"  # fw-large-v3 baseline, beams["5"]
SESSION_WAV = "outputs/20260710-230150-cef13a/input_16k.wav"
SESSION_TIMINGS = "outputs/20260710-230150-cef13a/timings.json"

FW_TURBO_MODEL = "large-v3-turbo"
FW_MODEL_NAMES: dict[str, str] = {"fw-turbo": FW_TURBO_MODEL, "fw-large-v3": config.ASR_MODEL}
MLX_REPOS: dict[str, str] = {
    "mlx-large-v3": "mlx-community/whisper-large-v3-mlx",
    "mlx-turbo": "mlx-community/whisper-large-v3-turbo",
}
ENGINES: tuple[str, ...] = ("fw-turbo", "mlx-large-v3", "mlx-turbo")
# Merged-window follow-up (see merge_segments): "fw-large-v3+merged" is the
# CONTROL — the production model/beam, only the windowing changes — plus the
# three candidate engines from the per-segment study, all decoding merged
# windows instead of raw diarized segments.
MERGE_ENGINES: tuple[str, ...] = (
    "fw-large-v3+merged", "mlx-large-v3+merged", "mlx-turbo+merged", "fw-turbo+merged",
)
MERGE_MAX_WINDOW_S = 28.0
MERGE_GAP_TOLERANCE_S = 1.0
KEYWORD_WER_TOLERANCE = 0.01  # absolute, vs baseline keyword_wer_micro
CORPUS_WER_TOLERANCE = 0.02  # absolute, vs baseline corpus_wer

# Anti-hallucination decode-param probe (see module docstring for the
# measured, negative result). The two clips whose per-segment/merged
# hypotheses above show the worst repetition-loop hallucinations —
# `probe_antihallu` fail-fast-checks a candidate config against only these
# before any full --bench run would be worth its ~10-20 min cost.
KNOWN_BAD_CLIP_IDS: frozenset[str] = frozenset(
    {"f2096fbd33010dc7d81344b4ad2477e3", "1ae62262f1e93e81ff73e92bd9305f21"}
)
ANTIHALLU_CONFIGS: dict[str, dict[str, dict]] = {
    "a1": {
        "mlx": {"temperature": 0.0, "condition_on_previous_text": False},
        "fw": {"temperature": [0.0], "condition_on_previous_text": False},
    },
    "a2": {
        "mlx": {
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "logprob_threshold": -0.3,
        },
        "fw": {
            "temperature": [0.0],
            "condition_on_previous_text": False,
            "log_prob_threshold": -0.3,
        },
    },
}


# ── Results file plumbing ────────────────────────────────────────────────────


def _load_results(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"per_clip": [], "session": {}}


def _flush(results: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _clip_record(results: dict, clip_id: str) -> dict:
    for c in results["per_clip"]:
        if c["id"] == clip_id:
            return c
    rec = {"id": clip_id, "engines": {}}
    results["per_clip"].append(rec)
    return rec


def _segments_from_cache(rec: dict) -> list[Segment]:
    return [Segment(start=s["start"], end=s["end"], speaker=s["speaker"]) for s in rec["segments"]]


def _ensure_segments(rec: dict, audio_bytes: bytes, pipeline) -> tuple[str, np.ndarray]:
    """Return (wav_path, mono float32 array); cache L2 segments once per clip."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        wav_path = preprocess(tmp_path)
    finally:
        os.unlink(tmp_path)
    audio, sr = sf.read(wav_path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"expected 16kHz mono from L1 preprocess, got {sr}Hz")
    if "segments" not in rec:
        segments = diarize(wav_path, pipeline=pipeline)
        rec["segments"] = [{"start": s.start, "end": s.end, "speaker": s.speaker} for s in segments]
        rec["duration_s"] = round(len(audio) / sr, 3)
    return wav_path, audio


# ── Merged-window segmentation ───────────────────────────────────────────────


def merge_segments(
    segments: list[Segment], max_window_s: float = 28.0, gap_tolerance_s: float = 1.0
) -> list[Segment]:
    """Merge adjacent same-speaker diarized segments into decode windows.

    Pure function: no I/O, no model calls. Adjacent segments merge into one
    window [first.start, last.end] when they share a speaker label and the
    gap between them is within `gap_tolerance_s`, provided the merged window
    would not exceed `max_window_s`. `segments` must already be chronological
    (as L2 diarization emits them); this function does not sort or reorder.

    Args:
        segments: Diarized segments, chronologically ordered.
        max_window_s: Maximum span of one merged window, in seconds.
        gap_tolerance_s: Maximum gap between two same-speaker segments that
            still allows merging into the same window.

    Returns:
        A new list of Segment windows, chronologically ordered, each with
        `speaker` from the merged run and `start`/`end` spanning it.
    """
    if not segments:
        return []

    windows: list[Segment] = []
    run_speaker = segments[0].speaker
    run_start = segments[0].start
    run_end = segments[0].end

    for seg in segments[1:]:
        gap = seg.start - run_end
        merged_span = seg.end - run_start
        can_merge = (
            seg.speaker == run_speaker
            and gap <= gap_tolerance_s
            and merged_span <= max_window_s
        )
        if can_merge:
            run_end = seg.end
        else:
            windows.append(Segment(start=run_start, end=run_end, speaker=run_speaker))
            run_speaker, run_start, run_end = seg.speaker, seg.start, seg.end

    windows.append(Segment(start=run_start, end=run_end, speaker=run_speaker))
    return windows


# ── Per-engine decode ─────────────────────────────────────────────────────────


def _warm_fw(model_name: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device="cpu", compute_type="int8")


def _warm_mlx(repo: str) -> None:
    """Force model download + load once, outside any per-clip timing."""
    import mlx_whisper

    silence = np.zeros(16000, dtype=np.float32)
    mlx_whisper.transcribe(silence, path_or_hf_repo=repo, language="en", task="transcribe")


def _decode_fw(wav_path: str, segments: list[Segment], model) -> tuple[str, float]:
    """Reuse the real production transcribe() — same contract, different model."""
    t0 = time.perf_counter()
    turns = transcribe(wav_path, segments, model=model)
    wall = time.perf_counter() - t0
    return " ".join(t.text for t in turns).strip(), wall


def _mlx_decode_segment(
    audio: np.ndarray,
    seg: Segment,
    repo: str,
    *,
    language: str | None,
    decode_kwargs: dict | None = None,
) -> str:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=repo,
        language=language,
        task="transcribe",
        clip_timestamps=f"{seg.start},{seg.end}",
        word_timestamps=False,
        **(decode_kwargs or {}),
    )
    return result["text"].strip()


def _decode_mlx(
    audio: np.ndarray, segments: list[Segment], repo: str, *, decode_kwargs: dict | None = None
) -> tuple[str, float, int]:
    """Per-segment mlx-whisper decode with the same Arabic-script guard as L3.

    `decode_kwargs` (e.g. an ANTIHALLU_CONFIGS entry) are forwarded verbatim
    to every mlx_whisper.transcribe() call, including script-guard re-decodes.
    """
    t0 = time.perf_counter()
    texts: list[str] = []
    guard_fires = 0
    for seg in segments:
        text = _mlx_decode_segment(audio, seg, repo, language=None, decode_kwargs=decode_kwargs)
        if config.ASR_SCRIPT_GUARD and _contains_arabic_script(text):
            guard_fires += 1
            logger.warning(
                "mlx script guard: Arabic-script decode at [%.2f, %.2f]s ('%s') — "
                "re-decoding with language=hi", seg.start, seg.end, text[:40],
            )
            text = _mlx_decode_segment(audio, seg, repo, language="hi", decode_kwargs=decode_kwargs)
        if text:
            texts.append(text)
    wall = time.perf_counter() - t0
    return " ".join(texts).strip(), wall, guard_fires


def _fw_decode_segment_direct(wav_path: str, seg: Segment, model, decode_kwargs: dict) -> str:
    """Direct faster-whisper decode, bypassing src.l3_asr.transcribe().

    transcribe()'s contract only exposes beam_size/vad_filter (see its
    docstring); Probe A's temperature/condition_on_previous_text/
    log_prob_threshold knobs aren't part of that contract, so this probe
    calls WhisperModel.transcribe directly instead — used ONLY by
    probe_antihallu, never by the production-path fw-turbo/fw-large-v3 runs
    above (those correctly reuse transcribe() verbatim).
    """
    gen, _ = model.transcribe(
        wav_path,
        language=None,
        task="transcribe",
        clip_timestamps=f"{seg.start},{seg.end}",
        word_timestamps=False,
        **decode_kwargs,
    )
    return " ".join(chunk.text.strip() for chunk in gen).strip()


# ── Bench + session runners ──────────────────────────────────────────────────


def _diarize_pipeline():
    import torch
    from pyannote.audio import Pipeline

    hf_token = os.environ.get("HF_TOKEN")
    pipeline = Pipeline.from_pretrained(config.DIARIZE_MODEL, token=hf_token)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    pipeline.to(device)
    return pipeline


def run_bench(engine: str, out_path: str = RESULTS_PATH, merge: bool = False) -> None:
    """Run one engine over the frozen 10-clip bench; resumable per clip.

    When `merge` is True, the cached diarization segments are collapsed with
    `merge_segments` (production defaults: MERGE_MAX_WINDOW_S,
    MERGE_GAP_TOLERANCE_S) before decode, and results are stored under the
    "<engine>+merged" key so per-segment and merged-window runs coexist in
    the same results file. "fw-large-v3" is only a valid engine with
    merge=True — unmerged fw-large-v3 is the cached beam_study baseline and
    is never rerun here.
    """
    import pandas as pd

    if engine == "fw-large-v3" and not merge:
        raise ValueError("fw-large-v3 is only valid with merge=True (see run_bench docstring)")

    df = pd.read_parquet(ASR_DATASET).head(FROZEN_N)
    results = _load_results(out_path)
    pipeline = _diarize_pipeline()
    key = f"{engine}+merged" if merge else engine

    fw_model = None
    if engine in FW_MODEL_NAMES:
        logger.info("Loading faster-whisper %s ...", FW_MODEL_NAMES[engine])
        fw_model = _warm_fw(FW_MODEL_NAMES[engine])
    else:
        logger.info("Warming mlx-whisper %s ...", MLX_REPOS[engine])
        _warm_mlx(MLX_REPOS[engine])

    for i, (_, row) in enumerate(df.iterrows()):
        clip_id = row["md5_text"]
        rec = _clip_record(results, clip_id)
        if key in rec["engines"]:
            logger.info("[%d/%d] %s already done for %s — skipping", i + 1, FROZEN_N, clip_id[:8], key)
            continue

        wav_path, audio = _ensure_segments(rec, row["audio"], pipeline)
        segments = _segments_from_cache(rec)
        if merge:
            segments = merge_segments(segments, MERGE_MAX_WINDOW_S, MERGE_GAP_TOLERANCE_S)

        if engine in FW_MODEL_NAMES:
            hyp, wall = _decode_fw(wav_path, segments, fw_model)
            guard_fires = None
        else:
            hyp, wall, guard_fires = _decode_mlx(audio, segments, MLX_REPOS[engine])

        rec["engines"][key] = {
            "hypothesis": hyp,
            "decode_wall_s": round(wall, 3),
            "script_guard_fires": guard_fires,
        }
        if merge:
            rec["engines"][key]["n_windows"] = len(segments)
        logger.info("[%d/%d] %s %s: %.2fs — %s", i + 1, FROZEN_N, key, clip_id[:8], wall, hyp[:80])
        _flush(results, out_path)

    print(f"Full results: {out_path}")


def run_session(engine: str, out_path: str = RESULTS_PATH, merge: bool = False) -> None:
    """Run one engine once on the user's real session clip.

    See run_bench for the `merge` / "<engine>+merged" key convention.
    """
    if engine == "fw-large-v3" and not merge:
        raise ValueError("fw-large-v3 is only valid with merge=True (see run_bench docstring)")

    results = _load_results(out_path)
    session = results.setdefault("session", {})
    key = f"{engine}+merged" if merge else engine

    if "segments" not in session:
        pipeline = _diarize_pipeline()
        segs = diarize(SESSION_WAV, pipeline=pipeline)
        session["segments"] = [{"start": s.start, "end": s.end, "speaker": s.speaker} for s in segs]
        audio, sr = sf.read(SESSION_WAV, dtype="float32")
        session["duration_s"] = round(len(audio) / sr, 3)
        _flush(results, out_path)

    if key in session:
        logger.info("session already done for %s — skipping", key)
        return

    segments = [Segment(**s) for s in session["segments"]]
    if merge:
        segments = merge_segments(segments, MERGE_MAX_WINDOW_S, MERGE_GAP_TOLERANCE_S)
    audio, _ = sf.read(SESSION_WAV, dtype="float32")

    if engine in FW_MODEL_NAMES:
        model = _warm_fw(FW_MODEL_NAMES[engine])
        hyp, wall = _decode_fw(SESSION_WAV, segments, model)
        guard_fires = None
    else:
        _warm_mlx(MLX_REPOS[engine])
        hyp, wall, guard_fires = _decode_mlx(audio, segments, MLX_REPOS[engine])

    session[key] = {"hypothesis": hyp, "decode_wall_s": round(wall, 3), "script_guard_fires": guard_fires}
    if merge:
        session[key]["n_windows"] = len(segments)
    logger.info("session %s: %.2fs — %s", key, wall, hyp[:120])
    _flush(results, out_path)


def _clip_audio_for_probe(audio_bytes: bytes) -> tuple[str, np.ndarray]:
    """Preprocess one dataset row's raw audio bytes into a 16kHz mono wav +
    array, without diarizing. probe_antihallu reuses L2 segments already
    cached in the results file by a prior --bench run, so it never needs the
    diarization pipeline.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        wav_path = preprocess(tmp_path)
    finally:
        os.unlink(tmp_path)
    audio, sr = sf.read(wav_path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"expected 16kHz mono from L1 preprocess, got {sr}Hz")
    return wav_path, audio


def probe_antihallu(
    engine: str,
    config_name: str,
    clip_ids: frozenset[str] = KNOWN_BAD_CLIP_IDS,
    results_path: str = RESULTS_PATH,
) -> None:
    """Fail-fast anti-hallucination decode-param probe (Probe A).

    Decodes ONLY `clip_ids` (default KNOWN_BAD_CLIP_IDS) with the named
    ANTIHALLU_CONFIGS entry, reusing L2 segments already cached in
    `results_path` from a prior --bench run. Prints hypotheses for
    inspection; never writes to the results file and is never scored by
    score() — per this study's fail-fast discipline, only a config that
    survives here would be worth promoting to a full --bench run. See the
    module docstring for the measured (negative) result.
    """
    import pandas as pd

    if config_name not in ANTIHALLU_CONFIGS:
        raise ValueError(f"unknown decode-config: {config_name!r}")
    if engine in FW_MODEL_NAMES:
        decode_kwargs = ANTIHALLU_CONFIGS[config_name]["fw"]
    elif engine in MLX_REPOS:
        decode_kwargs = ANTIHALLU_CONFIGS[config_name]["mlx"]
    else:
        raise ValueError(f"unknown engine: {engine!r}")

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)
    by_id = {c["id"]: c for c in results["per_clip"]}

    df = pd.read_parquet(ASR_DATASET).head(FROZEN_N)
    rows = {row["md5_text"]: row for _, row in df.iterrows() if row["md5_text"] in clip_ids}

    model = _warm_fw(FW_MODEL_NAMES[engine]) if engine in FW_MODEL_NAMES else None
    if model is None:
        _warm_mlx(MLX_REPOS[engine])

    for clip_id, row in rows.items():
        rec = by_id.get(clip_id)
        if rec is None or "segments" not in rec:
            raise ValueError(
                f"{clip_id[:8]} has no cached L2 segments in {results_path} — "
                "run `--engine <any> --bench` at least once first"
            )
        segments = _segments_from_cache(rec)
        wav_path, audio = _clip_audio_for_probe(row["audio"])

        if model is not None:
            t0 = time.perf_counter()
            texts = [_fw_decode_segment_direct(wav_path, seg, model, decode_kwargs) for seg in segments]
            wall = time.perf_counter() - t0
            hyp = " ".join(t for t in texts if t).strip()
        else:
            hyp, wall, _ = _decode_mlx(audio, segments, MLX_REPOS[engine], decode_kwargs=decode_kwargs)

        print(f"[{engine} | {config_name}] {clip_id[:8]} ({wall:.1f}s): {hyp[:300]}")


# ── Scoring ──────────────────────────────────────────────────────────────────


def _fw_large_v3_baseline_hyps() -> dict[str, str]:
    """fw-large-v3 hypotheses at beam=5 — read from the cached beam study, never rerun."""
    with open(BEAM_STUDY_PATH, encoding="utf-8") as f:
        beam = json.load(f)
    return {c["id"]: c["beams"]["5"]["hypothesis"] for c in beam["per_clip"]}


def _drug_wer_folded(ref: str, raw_hyp: str, norm_hyp: str, kws: list[str]) -> tuple[int, int]:
    """(present, missed) using eval.drug_bench's script-symmetric fold over raw+normalized text."""
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


def _score_engine(engine: str, df, by_id: dict, baseline_hyps: dict[str, str]) -> dict:
    from src.l3_5_normalize import normalize

    refs: list[str] = []
    hyps: list[str] = []
    wall_times: list[float] = []
    kw_present = kw_missed = drug_present = drug_missed = 0
    drug_present_f = drug_missed_f = 0

    for _, row in df.iterrows():
        clip_id = row["md5_text"]
        if engine == "fw-large-v3":
            hyp = baseline_hyps.get(clip_id)
            wall = None
        else:
            eng_rec = by_id.get(clip_id, {}).get("engines", {}).get(engine)
            if eng_rec is None:
                continue
            hyp = eng_rec["hypothesis"]
            wall = eng_rec["decode_wall_s"]
        if hyp is None:
            continue

        ref = row["text"]
        kw = _keywords_from_entities(row["medical_entities"])
        kw_drug = _keywords_from_entities(row["medical_entities"], drug_only=True)

        refs.append(ref)
        hyps.append(hyp)
        if wall is not None:
            wall_times.append(wall)

        p, m = keyword_hits(ref, hyp, kw)
        kw_present += p
        kw_missed += m

        dp, dm = keyword_hits(ref, hyp, kw_drug)
        drug_present += dp
        drug_missed += dm

        norm_turns = normalize([Turn(speaker_role="UNKNOWN", text=hyp, start=0.0, end=0.0)])
        norm_hyp = " ".join(t.text for t in norm_turns).strip()
        dpf, dmf = _drug_wer_folded(ref, hyp, norm_hyp, kw_drug)
        drug_present_f += dpf
        drug_missed_f += dmf

    return {
        "n_clips_scored": len(hyps),
        "corpus_wer": round(corpus_word_error_rate(refs, hyps), 4) if refs else None,
        "keyword_wer_micro": round(kw_missed / kw_present, 4) if kw_present else 0.0,
        "drug_wer_raw": round(drug_missed / drug_present, 4) if drug_present else 0.0,
        "drug_wer_folded": round(drug_missed_f / drug_present_f, 4) if drug_present_f else 0.0,
        "keywords_total": kw_present,
        "drug_keywords_total": drug_present,
        "mean_decode_wall_s": round(sum(wall_times) / len(wall_times), 2) if wall_times else None,
    }


def _print_session_table(results: dict) -> None:
    session = results.get("session", {})
    if not session:
        return
    duration = session.get("duration_s")
    print(f"\nUser session clip ({SESSION_WAV}, duration={duration}s):")
    if os.path.exists(SESSION_TIMINGS):
        with open(SESSION_TIMINGS, encoding="utf-8") as f:
            t = json.load(f)
        anchor = round(
            t["stages"]["l3_asr"]["wall_s"] - t["sub_timings"].get("l3.model_load", 0.0), 2
        )
        print(
            f"  fw-large-v3 (production anchor, NOT rerun here): decode={anchor}s "
            f"(from {SESSION_TIMINGS}: l3_asr.wall_s - l3.model_load)"
        )
    for engine in ENGINES + MERGE_ENGINES:
        rec = session.get(engine)
        if rec is None:
            continue
        rtf = round(rec["decode_wall_s"] / duration, 2) if duration else None
        arabic_script = _contains_arabic_script(rec["hypothesis"])
        print(
            f"  {engine}: decode={rec['decode_wall_s']}s  RTF={rtf}x  "
            f"script_guard_fires={rec.get('script_guard_fires')}  "
            f"arabic_script_present={arabic_script}"
        )
        print(f"    hypothesis: {rec['hypothesis'][:200]}")


def _gate(engine: str, m: dict, baseline: dict) -> str:
    """PASS/FAIL/PARTIAL verdict for one engine against the fw-large-v3 baseline.

    Merged-window configs ("<engine>+merged") gate on drug_wer_folded not
    worsening (this study's brief); per-segment configs keep the original
    drug_wer_raw gate from the engine study this probe follows up on.
    """
    if m["n_clips_scored"] < FROZEN_N:
        return f"PARTIAL ({m['n_clips_scored']}/{FROZEN_N})"
    kw_ok = m["keyword_wer_micro"] <= baseline["keyword_wer_micro"] + KEYWORD_WER_TOLERANCE
    corpus_ok = m["corpus_wer"] <= baseline["corpus_wer"] + CORPUS_WER_TOLERANCE
    if engine.endswith("+merged"):
        drug_ok = m["drug_wer_folded"] <= baseline["drug_wer_folded"]
    else:
        drug_ok = m["drug_wer_raw"] <= baseline["drug_wer_raw"]
    return "PASS" if (kw_ok and drug_ok and corpus_ok) else "FAIL"


def score(results_path: str = RESULTS_PATH) -> None:
    """Score every engine present in the results file against the fw-large-v3 gate."""
    import pandas as pd

    df = pd.read_parquet(ASR_DATASET).head(FROZEN_N)
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)
    by_id = {c["id"]: c for c in results["per_clip"]}
    baseline_hyps = _fw_large_v3_baseline_hyps()

    present_engines = [
        e for e in ENGINES + MERGE_ENGINES if any(e in c.get("engines", {}) for c in results["per_clip"])
    ]
    order = ["fw-large-v3"] + present_engines

    table = {engine: _score_engine(engine, df, by_id, baseline_hyps) for engine in order}
    baseline = table["fw-large-v3"]

    header = (
        f"{'Engine':<20} {'CorpusWER':>10} {'KwWER':>8} {'DrugWER':>9} "
        f"{'DrugWERfold':>12} {'MeanDecodeS':>12} {'N':>4}  Gate"
    )
    print(f"\n{header}")
    for engine in order:
        m = table[engine]
        gate = "-" if engine == "fw-large-v3" else _gate(engine, m, baseline)
        print(
            f"{engine:<20} {m['corpus_wer']!s:>10} {m['keyword_wer_micro']!s:>8} "
            f"{m['drug_wer_raw']!s:>9} {m['drug_wer_folded']!s:>12} "
            f"{m['mean_decode_wall_s']!s:>12} {m['n_clips_scored']!s:>4}  {gate}"
        )

    _print_session_table(results)

    results["summary"] = table
    _flush(results, results_path)
    print(f"\nFull results: {results_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    parser = argparse.ArgumentParser(description="L3 ASR engine speed/accuracy probe.")
    parser.add_argument(
        "--engine", choices=ENGINES + ("fw-large-v3",), help="engine to run"
    )
    parser.add_argument("--bench", action="store_true", help="run on the frozen 10-clip bench")
    parser.add_argument("--session", action="store_true", help="run on the user's real session clip")
    parser.add_argument(
        "--merge",
        action="store_true",
        help=(
            "merge adjacent same-speaker diarized segments into windows "
            f"(max {MERGE_MAX_WINDOW_S}s, gap tolerance {MERGE_GAP_TOLERANCE_S}s) "
            "before decode; see merge_segments(). Required for --engine fw-large-v3."
        ),
    )
    parser.add_argument("--score", action="store_true", help="score cached results (no models loaded)")
    parser.add_argument(
        "--decode-config",
        choices=tuple(ANTIHALLU_CONFIGS),
        help=(
            "fail-fast anti-hallucination decode-param probe (Probe A): decode only "
            "KNOWN_BAD_CLIP_IDS with this ANTIHALLU_CONFIGS entry; see probe_antihallu()."
        ),
    )
    parser.add_argument("--out", default=RESULTS_PATH, help="results file path")
    args = parser.parse_args()

    if args.score:
        score(args.out)
    elif args.decode_config:
        if not args.engine or args.engine == "fw-large-v3":
            parser.error("--decode-config requires --engine {fw-turbo,mlx-large-v3,mlx-turbo}")
        probe_antihallu(args.engine, args.decode_config, results_path=args.out)
    elif not args.engine:
        parser.error("--engine is required unless --score is given")
    elif not (args.bench or args.session):
        parser.error("give --bench and/or --session")
    elif args.engine == "fw-large-v3" and not args.merge:
        parser.error("--engine fw-large-v3 requires --merge (unmerged fw-large-v3 is the cached baseline)")
    else:
        if args.bench:
            run_bench(args.engine, args.out, merge=args.merge)
        if args.session:
            run_session(args.engine, args.out, merge=args.merge)
