# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CliniScribe — an on-device clinical scribe MVP for Indian tier-2/tier-3 clinics.

**Pipeline (sequential, batch — end-of-consultation, not streaming):**
```
audio → L1 preprocess → L2 diarize → L3 ASR → L3.5 normalize → L4 extract → L5 render → [PHYSICIAN REVIEW]
```

**Hardware target:** MacBook Air M-series, 24 GB unified RAM. All models run locally. No cloud APIs in the production path.

## Pipeline Stages

| Stage | Function | Component | Runtime |
|---|---|---|---|
| L1 | Audio preprocessing (resample, denoise, VAD) | noisereduce, soundfile, librosa | CPU |
| L2 | Speaker diarization | pyannote/speaker-diarization-community-1 | CPU/Metal |
| L3 | Multilingual ASR (Hindi/English/Marathi, code-switch) | faster-whisper large-v3 | Metal |
| L3.5 | Lay→clinical vocab normalization | ekacare/parrotlet-e embeddings | CPU |
| L4 | Clinical entity extraction | qwen2.5:3b-instruct via Ollama | Local |
| L5 | Prescription PDF rendering | reportlab | CPU |

**Diarization model:** `pyannote/speaker-diarization-community-1` (not 3.1 — requires HF token and pyannote T&C acceptance).

## Memory Discipline

**Load one model, run it, release it (`del model; gc.collect()`). Never hold ASR and LLM resident simultaneously.**

The batch design exists to make sequential loading clean within 24 GB. This is not optional — violating it will OOM on target hardware.

## Module Interfaces (spec §7)

All inter-stage communication uses these stable contracts. Do not change signatures without updating all callers.

```python
# src/types.py — shared across all stages
from dataclasses import dataclass

@dataclass
class Segment:
    start: float
    end: float
    speaker: str

@dataclass
class Turn:
    speaker_role: str   # "DOCTOR" | "PATIENT" | "UNKNOWN"
    text: str
    start: float
    end: float

@dataclass
class Symptom:
    name: str
    finding_status: str = "Present"  # "Present" | "Absent" | "Unknown"
    severity: str | None = None      # "Mild" | "Moderate" | "Severe" | None
    since: str | None = None         # free-text onset, e.g. "3 days"

@dataclass
class Vital:
    name: str           # "BP", "Temperature", "SpO2", "Pulse", "Weight", ...
    value: str          # value with unit, e.g. "120/80 mmHg", "98 %"

@dataclass
class Medication:
    drug: str
    dose: str | None
    frequency: str | None
    timing: str | None      # "before food" | "after food" | "at bedtime" | None
    duration: str | None
    validated: bool     # True only if matched to CDSCO drug list

@dataclass
class Diagnosis:
    term: str
    snomed_id: str | None
    status: str | None = None  # "Confirmed" | "Suspected" | "Ruled out" | None

@dataclass
class ClinicalNote:
    chief_complaint: str | None
    history: str | None
    symptoms: list[Symptom]
    vitals: list[Vital]
    examination: str | None          # free-text physical exam findings
    diagnosis: list[Diagnosis]
    medications: list[Medication]
    investigations: list[str]        # tests ordered for later
    diagnostic_results: list[str]    # results already in hand (with values)
    advice: str | None
    follow_up: str | None
    low_confidence_fields: list[str]  # surfaced to reviewing physician
```

```python
def preprocess(in_path: str) -> str           # L1  → denoised 16kHz mono wav path
def diarize(wav_path: str) -> list[Segment]   # L2  → diarized segments
def transcribe(wav_path: str, segments: list[Segment]) -> list[Turn]  # L3 → attributed turns
def normalize(turns: list[Turn]) -> list[Turn]  # L3.5 → normalized turns
def extract(turns: list[Turn]) -> ClinicalNote  # L4  → structured JSON
def render(note: ClinicalNote) -> str           # L5  → pdf path
```

## L4 Output Schema (spec §3.5)

```json
{
  "chief_complaint": "string | null",
  "history": "string | null",
  "symptoms": [{
    "name": "string",
    "finding_status": "Present | Absent | Unknown",
    "severity": "Mild | Moderate | Severe | null",
    "since": "string | null"
  }],
  "vitals": [{ "name": "string", "value": "string (with unit)" }],
  "examination": "string | null",
  "diagnosis": [{
    "term": "string",
    "snomed_id": "string | null",
    "status": "Confirmed | Suspected | Ruled out | null"
  }],
  "medications": [{
    "drug": "string",
    "dose": "string | null",
    "frequency": "string | null",
    "timing": "string | null",
    "duration": "string | null",
    "validated": "boolean"
  }],
  "investigations": ["string"],
  "diagnostic_results": ["string"],
  "advice": "string | null",
  "follow_up": "string | null",
  "low_confidence_fields": ["string"]
}
```

Drug names and doses MUST be validated against the CDSCO list. Unvalidated entries get `"validated": false` and are flagged in `low_confidence_fields`.

**Schema scope:** field set is aligned to the EkaCare clinical-note rubric targets so the note can hold what consultations actually contain (symptoms ≈ 23% of rubric criteria, vitals ≈ 8%, exam ≈ 7%, diagnostic results ≈ 8%, medication timing ≈ 5%, diagnosis status ≈ 3%). Structured past/family/social/lifestyle history is intentionally left as free-text `history` — Indian tier-2/3 clinic transcripts rarely capture it on tape (≈ 10% of rubric criteria), and structuring empty fields invites LLM fabrication. `investigations` = tests ORDERED; `diagnostic_results` = results already AVAILABLE.

## Development Rules

**Build one stage at a time.** Each stage must run and be tested on real EkaCare data before the next begins. No speculative scaffolding.

**After each phase checkpoint:** Append a dated section to `LEARNINGS.md` written to teach a domain newcomer. Plain language; no jargon without a one-line definition. Required structure:
- **(a) What this phase does — 3 sentences.**
- **(b) The 2 hardest bugs and their root cause** (not just the symptom — the underlying reason).
- **(c) One thing that will matter when we fine-tune later.**

**When something fails:** State the root cause before proposing a fix.

**Do not pre-translate.** Run ASR and extraction in the source language. Translating Hindi/Marathi to English before downstream processing consistently degrades accuracy.

**Denoising is tunable, not always-on.** Benchmark WER with and without it per noise profile (see EkaCare denoising-impact study).

## Data

| Dataset | Content | Access |
|---|---|---|
| ekacare/clinical_note_generation_dataset | 156 Hindi/Marathi/English transcripts + ground-truth JSON | HuggingFace (open) |
| ekacare/eka-medical-asr-evaluation-dataset | Medical ASR audio + reference text | HuggingFace (open) |
| ekacare/denoising-impact-evaluation-dataset | 500 clinical recordings, varied SNR | HuggingFace (open) |

Local copies live in `data/` (gitignored). HF_TOKEN loaded from `.env`.

## Evaluation Metrics (KARMA framework)

- **WER** — baseline transcript accuracy
- **Semantic WER** — meaning-weighted; handles synonyms, abbreviations, code-switching
- **Keyword WER** — accuracy on drug names, dosages, vitals (patient-safety metric)
- **DER** — diarization error rate (L2)
- **Concept-match accuracy** — lay→clinical mapping accuracy (L3.5)

Every model swap or prompt change must be scored on a frozen eval set before merge. Wire metrics into `eval/` from the start.

## Commands

```bash
# Activate venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify HuggingFace token
python -c "from huggingface_hub import whoami; import json; print(json.dumps(whoami(), indent=2))"

# Run all tests
pytest tests/ -v

# Run a single stage test
pytest tests/test_l3_asr.py -v

# Check Ollama is running (required for L4)
ollama list

# Run full pipeline
python src/pipeline.py <wav_path>
```

## Project Structure

```
src/
  types.py          # Segment, Turn, ClinicalNote, shared dataclasses
  l1_preprocess.py  # preprocess()
  l2_diarize.py     # diarize()
  l3_asr.py         # transcribe()
  l3_5_normalize.py # normalize()
  l4_extract.py     # extract()
  l5_render.py      # render()
  pipeline.py       # orchestrates all stages sequentially
eval/
  metrics.py        # WER, Semantic WER, Keyword WER, DER
  run_eval.py       # runs KARMA evaluation on EkaCare datasets
data/               # local dataset copies (gitignored)
outputs/            # generated PDFs and intermediate artifacts (gitignored)
tests/              # mirrors src/ structure
LEARNINGS.md        # per-stage retrospectives (committed)
```

## Stage Completion Criteria

A stage is complete when:
1. It runs end-to-end on at least one real EkaCare sample without error.
2. Its tests pass (`pytest tests/test_<stage>.py`).
3. A `LEARNINGS.md` entry exists with failure modes documented.
