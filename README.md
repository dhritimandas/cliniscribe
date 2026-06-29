# CliniScribe

On-device clinical scribe for Indian tier-2/tier-3 clinics. It turns a recorded
consultation — Hindi, English, and Marathi, code-switched mid-sentence — into a
structured clinical note and a prescription PDF for the physician to review and sign.

Everything runs locally on a MacBook Air (M-series, 24 GB RAM). No cloud APIs are in
the production path: patient privacy, cost, and unreliable rural connectivity rule them
out.

## Pipeline

A sequential, end-of-consultation batch (not streaming):

```
audio → L1 preprocess → L2 diarize → L3 ASR → L3.5 normalize → L4 extract → L5 render → [PHYSICIAN REVIEW]
```

| Stage | Job | Component |
|---|---|---|
| L1 | Resample to 16 kHz mono, trim silence, optional denoise | librosa / soundfile / noisereduce |
| L2 | Diarization — "who spoke when" | pyannote speaker-diarization-community-1 |
| L3 | ASR — speech to text, in the source language | faster-whisper large-v3 |
| L3.5 | Normalize lay→clinical terms; fix drug spellings | parrotlet-e embeddings + curated tables |
| L4 | Extract structured `ClinicalNote` JSON | qwen2.5:3b-instruct (Ollama) |
| L5 | Render prescription PDF | reportlab |

**Design constraints that shape everything:** (1) *on-device, 24 GB* — models are loaded
one at a time and released before the next (`del model; gc.collect()`), which is why the
pipeline is batch; (2) *patient safety* — the metric that matters most is accuracy on
drug names, doses, and vitals, not overall transcription quality.

## Prerequisites

- macOS on Apple Silicon (validated target: MacBook Air M-series, 24 GB). Other
  platforms may work but are untested.
- Python 3.11+
- [Ollama](https://ollama.com) running locally, with the L4 model pulled:
  `ollama pull qwen2.5:3b-instruct`
- A HuggingFace token (for model and dataset downloads), placed in `.env` as
  `HF_TOKEN=...`

## Installation

```bash
git clone git@github.com:dhritimandas/cliniscribe.git
cd cliniscribe

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then add your HF_TOKEN
ollama pull qwen2.5:3b-instruct
```

Verify the HuggingFace token:

```bash
python -c "from huggingface_hub import whoami; import json; print(json.dumps(whoami(), indent=2))"
```

## Usage

```bash
# Run the full pipeline on one recording → PDF in outputs/
python src/pipeline.py path/to/consultation.wav

# Run the KARMA evaluation on the frozen eval sets
python eval/run_eval.py

# Run the test suite
pytest tests/ -v
```

Datasets (EkaCare, open access on HuggingFace) are pulled into `data/` on first use;
both `data/` and `outputs/` are gitignored.

## Repository structure

```
src/
  types.py           # Segment, Turn, ClinicalNote — shared dataclasses (stage contracts)
  l1_preprocess.py   # preprocess()  — audio cleanup
  l2_diarize.py      # diarize()     — speaker segmentation
  l3_asr.py          # transcribe()  — multilingual ASR
  l3_5_normalize.py  # normalize()   — lay→clinical + drug-name normalization
  l4_extract.py      # extract()     — structured note via LLM
  l5_render.py       # render()      — prescription PDF
  pipeline.py        # orchestrates all stages sequentially
  cdsco.py           # CDSCO drug-list validation
  concepts.py        # clinical concept table for L3.5
eval/
  metrics.py         # WER, Keyword WER, Drug-KW WER, DER
  run_eval.py        # KARMA evaluation on frozen sets
tests/               # mirrors src/
LEARNINGS.md         # engineering + research record (start here for depth)
```

## Documentation

[`LEARNINGS.md`](LEARNINGS.md) is the in-depth record: the methodology and transferable
principles, stage-by-stage engineering status, the full experiment log (E1–E9 with
numbers and root causes), and the product/schema decisions. Read it to understand *why*
the system is built the way it is and how every result was measured.

## Evaluation

Every model swap or prompt change is scored on a **frozen evaluation set** before it is
accepted (the KARMA framework: WER, Keyword WER, Drug-Keyword WER, DER, concept-match
accuracy). The guiding discipline — *measure before you tune*, and *suspect the ruler
before the model* — is documented in `LEARNINGS.md`, Part II.

## Status

Research MVP. Stages L1–L5 are implemented and evaluated on real EkaCare data; the
output is intended for physician review, not autonomous use.
