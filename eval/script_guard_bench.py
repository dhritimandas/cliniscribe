"""Script-guard no-op gate: the Arabic-script re-decode must not fire (or

otherwise alter output) on the frozen 10-clip Hindi bench, which never
produced Arabic-script text before this fix. This study runs the real
production path (L1 preprocess -> L2 diarize -> L3 transcribe, preloaded
models) with config.ASR_SCRIPT_GUARD on at production defaults
(ASR_BEAM_SIZE=5, ASR_VAD_FILTER=False) and records each clip's hypothesis
for a byte-identical diff against the cached pre-fix hypotheses in
outputs/vad_study.json (vad["False"], 3 clips) and outputs/beam_study.json
(beams["5"], all 10 clips). Any difference means the guard fired (or altered
behavior) somewhere it should not have — investigate before concluding.

Method mirrors eval/vad_study.py: L1/L2 run once per clip with a preloaded
pyannote pipeline; L3 runs once per clip with a preloaded WhisperModel.
Results are written incrementally to outputs/script_guard_bench.json so
partial progress survives interruption.

Run:  PYTHONPATH=. python eval/script_guard_bench.py
"""

import json
import logging
import os
import tempfile

from dotenv import load_dotenv

from src import config
from src.l1_preprocess import preprocess
from src.l2_diarize import diarize
from src.l3_asr import transcribe

load_dotenv()
logger = logging.getLogger(__name__)

ASR_DATASET = "eka-medical-asr-dataset/hi/test-00000.parquet"
FROZEN_N = 10  # must match eval/frozen_set_asr.json (deterministic first-N rows)
RESULTS_PATH = "outputs/script_guard_bench.json"
VAD_STUDY_PATH = "outputs/vad_study.json"
BEAM_STUDY_PATH = "outputs/beam_study.json"


def main(out_path: str = RESULTS_PATH) -> None:
    """Run the production path with the script guard on, on the frozen 10 clips."""
    import pandas as pd
    import torch
    from faster_whisper import WhisperModel
    from pyannote.audio import Pipeline

    df = pd.read_parquet(ASR_DATASET).head(FROZEN_N)
    results: dict = {"per_clip": []}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            results = json.load(f)
    done_ids = {c["id"] for c in results["per_clip"]}

    def _flush() -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(
        "config: ASR_SCRIPT_GUARD=%s ASR_BEAM_SIZE=%s ASR_VAD_FILTER=%s",
        config.ASR_SCRIPT_GUARD, config.ASR_BEAM_SIZE, config.ASR_VAD_FILTER,
    )

    hf_token = os.environ.get("HF_TOKEN")
    diarize_pipeline = Pipeline.from_pretrained(config.DIARIZE_MODEL, token=hf_token)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    diarize_pipeline.to(device)
    whisper = WhisperModel(config.ASR_MODEL, device="cpu", compute_type="int8")
    logger.info("Models resident: %s + %s", config.DIARIZE_MODEL, config.ASR_MODEL)

    for i, (_, row) in enumerate(df.iterrows()):
        if row["md5_text"] in done_ids:
            logger.info("[%d/%d] already done — skipping", i + 1, FROZEN_N)
            continue

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(row["audio"])
            tmp_path = f.name
        try:
            wav_path = preprocess(tmp_path)
            segments = diarize(wav_path, pipeline=diarize_pipeline)
            turns = transcribe(wav_path, segments, model=whisper)
            hyp = " ".join(t.text for t in turns).strip()
        finally:
            os.unlink(tmp_path)

        logger.info("[%d/%d] hypothesis: %s", i + 1, FROZEN_N, hyp[:80])
        results["per_clip"].append({"id": row["md5_text"], "hypothesis": hyp})
        _flush()

    print(f"Full results: {out_path}")


def diff(
    results_path: str = RESULTS_PATH,
    vad_study_path: str = VAD_STUDY_PATH,
    beam_study_path: str = BEAM_STUDY_PATH,
) -> None:
    """Diff script-guard-on hypotheses against cached pre-fix hypotheses."""
    with open(results_path, encoding="utf-8") as f:
        guard = {c["id"]: c["hypothesis"] for c in json.load(f)["per_clip"]}
    with open(vad_study_path, encoding="utf-8") as f:
        vad = {
            c["id"]: c["vad"]["False"]["hypothesis"]
            for c in json.load(f)["per_clip"]
        }
    with open(beam_study_path, encoding="utf-8") as f:
        beam = {
            c["id"]: c["beams"]["5"]["hypothesis"]
            for c in json.load(f)["per_clip"]
        }

    all_ok = True
    for clip_id, hyp in guard.items():
        against_vad = vad.get(clip_id)
        against_beam = beam.get(clip_id)
        vad_ok = against_vad is None or hyp == against_vad
        beam_ok = against_beam is None or hyp == against_beam
        ok = vad_ok and beam_ok
        all_ok &= ok
        print(f"{clip_id}: vad_match={vad_ok} beam_match={beam_ok} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            print(f"  guard_hyp: {hyp!r}")
            print(f"  vad_hyp:   {against_vad!r}")
            print(f"  beam_hyp:  {against_beam!r}")

    print("GATE PASSED" if all_ok else "GATE FAILED")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    parser = argparse.ArgumentParser(description="Script-guard no-op gate on the frozen bench.")
    parser.add_argument("--out", default=RESULTS_PATH, help="results file path")
    parser.add_argument("--diff", action="store_true", help="diff against cached studies and exit")
    args = parser.parse_args()

    if args.diff:
        diff(args.out)
    else:
        main(args.out)
