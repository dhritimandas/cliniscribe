# CliniScribe — Phase Learnings

Append a dated section after each phase checkpoint. Write for a domain
newcomer — plain language, no jargon without a one-line definition.

Format:

```
## Phase N — <name> (YYYY-MM-DD)

### (a) What this phase does
<3 sentences>

### (b) Hardest bugs
1. <bug> — root cause: <why, not just the symptom>
2. <bug> — root cause: <why, not just the symptom>

### (c) Fine-tuning hook
<one thing that will matter when we fine-tune later>
```

## Phase A — L1 preprocess + L2 diarize + L3 ASR (2026-06-26)

### (a) What this phase does
L1 resamples any input audio to 16 kHz mono WAV using librosa (Core Audio backend on macOS — no ffmpeg needed) and trims leading/trailing silence via energy-threshold VAD; denoising is a keyword-only toggle defaulting to OFF because stationary noise reduction can increase WER on clean clinic recordings. L2 runs pyannote/speaker-diarization-community-1 on MPS, passing audio as a preloaded `{'waveform': tensor, 'sample_rate': int}` dict because torchcodec (the preferred pyannote I/O backend) fails to link against the installed ffmpeg dylibs. L3 uses faster-whisper large-v3 on CPU (int8 quantization, ~3.6 GB peak RSS) and transcribes each diarized segment with language=None so Hindi, English, and Marathi are detected independently per segment rather than forcing a single global language.

### (b) Hardest bugs
1. `Pipeline.from_pretrained()` rejected `use_auth_token=` keyword — root cause: pyannote.audio >= 3.x switched to the standard HuggingFace `token=` parameter name; `use_auth_token` was silently removed upstream without a deprecation cycle, so the error was a `TypeError` rather than a warning.
2. `DiarizeOutput has no attribute 'itertracks'` — root cause: `pyannote/speaker-diarization-community-1` wraps its output in a `DiarizeOutput` dataclass rather than returning a bare `pyannote.core.Annotation` directly; the actual annotation is at `DiarizeOutput.speaker_diarization`. The pyannote docs only describe this for the 3.x family; the community model diverges silently.

### (c) Fine-tuning hook
The doctor-attribution heuristic (bag-of-words over question-forms + clinical terms) only works across ≥2 speakers. All EkaCare ASR evaluation clips are single-doctor utterances, so every turn was labelled UNKNOWN in Phase A tests. When training a proper role classifier, the label will need to come from full consultation recordings where doctor/patient turns alternate — the ASR eval dataset is not suitable for this and the clinical-note-generation dataset (156 transcripts with JSON ground truth) is the right source.

## Phase A — Teaching Notes (Q&A, 2026-06-26)

These four questions were asked after Phase A to consolidate understanding.
Kept verbatim with the answers because the *reasoning* is the lesson, not the
conclusion.

### Q1. Walk through `l3_asr.py` as if I've never used faster-whisper.

faster-whisper is a reimplementation of OpenAI's Whisper that runs on
**CTranslate2**, a C++ inference engine — *not* PyTorch. That one fact explains
the design.

- `WhisperModel("large-v3", device="cpu", compute_type="int8")` — `device="cpu"`
  (not MPS) because CTranslate2 has no Metal backend; on M-series it uses Apple's
  AMX matrix units via BLAS. `compute_type="int8"` quantizes weights to 8-bit
  integers, cutting RAM from ~10 GB (fp32) to ~3.6 GB — the tradeoff the 24 GB
  hardware target forces.
- `model.transcribe(...)` returns a **2-tuple** `(segments_generator, info)`. The
  generator is **lazy** — transcription runs only when iterated (in the
  `" ".join(...)`).
- `language=None` runs **language identification (LID)** per clip — the core of
  the code-switching strategy: each diarized slice gets its own language verdict.
- `task="transcribe"` pins same-language output so a Hindi segment never silently
  becomes English (project rule: do not pre-translate).
- `clip_timestamps="start,end"` restricts output to that window. Caveat: the logs
  showed the full file decoded on every call (the clip only restricts emitted
  text), so we re-decode N times per recording — an inefficiency to revisit.
- `beam_size=5` keeps the 5 best running hypotheses instead of greedy decoding.
- Cleanup `del model; gc.collect(); torch.mps.empty_cache()` — the empty_cache is
  effectively a no-op here (model was on CPU) but keeps the release discipline
  uniform across stages.

### Q2. The 3 most instructive bugs — what each taught about speech pipelines.

(The `use_auth_token`/`DiarizeOutput` bugs in section (b) above were generic
API-churn, not speech lessons. These three *are* speech lessons.)

1. **torchcodec couldn't link against ffmpeg → fed pyannote an in-memory waveform
   dict.** Lesson: audio decoding is a native-dependency minefield and it breaks
   *first*. Decode audio yourself once (librosa/soundfile) and pass models a
   `{'waveform', 'sample_rate'}` array, never a file path — decouple decoding
   from inference.
2. **faster-whisper ignored MPS and ran on CPU.** Lesson: "I have a GPU" ≠ "this
   model uses it." The inference *runtime* matters more than the chip — in one
   pipeline L2 (PyTorch) runs on MPS while L3 (CTranslate2) runs on CPU.
3. **Per-segment LID returned the wrong language at low confidence on short
   clips** (a sample_00 segment was detected `en` @ 0.67 when the audio is Hindi).
   Lesson: Whisper's LID degrades on short audio, and diarization hands you
   exactly that. The language verdict is shakiest at the short turns where
   code-switching happens.

### Q3. Was the predicted-hardest part actually the hardest?

The prediction recorded in the spec's L3 docstring was that **code-switch ASR
accuracy** would be the hard part ("WER spikes at code-switch boundaries... the
primary product metric"). That is probably right *about the product* — but it was
**not** what cost the most time in Phase A, and we could not even measure it yet.
What actually fought back was the **plumbing**: native audio decoding
(torchcodec/ffmpeg), backend mismatch (MPS vs CPU), and the pyannote API surface.
None of that is machine learning. This is the characteristic profile of
**on-device, local-first pipelines** — without a cloud API hiding decoding,
device placement, and quantization, those concerns front-load the schedule and
the ML risk is deferred until the `eval/` WER harness exists.

### Q4. A Hindi-English code-switch error in sample 2, and why the model is weak there.

First turn: `और डाइली फ्री टाइम्स`. Ground truth: **"daily three times."** Two
stacked failures at one switch point:
1. **Script:** English words transliterated into Devanagari (`डाइली`="daily",
   `टाइम्स`="times") instead of Latin.
2. **Lexical:** **"three" → "फ्री" ("free")** — `th` heard as `f`.

Why weak *exactly* there: every sample_02 segment was LID'd as Hindi at 0.92–0.98.
Once Whisper commits a segment to the `hi` language token, the decoder runs in
"Hindi mode" — biased toward Devanagari output and Hindi phonotactics. Standard
Hindi has **no /θ/** (the "th" in "three"), so a Hindi-conditioned decoder puts
near-zero probability on a /θ/-initial word and emits the nearest plausible
neighbor it *can* produce: /f/ → `फ्री`. The model breaks at the precise phoneme
that does not exist in the language it committed to. Contrast: later English terms
in the same recording ("lab test", "CBC", "chest x-ray", "after 8 days") survived,
because they sit inside a long Hindi-context segment with strong medical-English
priors, whereas "daily three times" was its own isolated 1.45 s segment with no
context to anchor it. This argues per-segment LID may *cause* some of these errors
by locking short English fragments into Hindi mode — a candidate fix is to bias
short-segment language from neighbouring turns.
