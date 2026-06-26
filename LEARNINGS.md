# CliniScribe — Phase Learnings

A running, teach-a-newcomer record of the build. Each part covers one stretch of
work in the order it happened: **build a stage → measure it → learn the concepts
needed for the next stage.** Plain language; any jargon is defined inline the
first time it appears. The required per-phase checkpoint format (what / hardest
bugs / fine-tuning hook) is preserved in the Appendix and used in Part 1.

Contents:
- **Part 1 — Phase A: audio → attributed transcript** (L1 preprocess, L2 diarize, L3 ASR)
- **Part 2 — Evaluation: measure before you tune** (the WER harness and the first baseline)
- **Part 3 — Concept layer: preparing for Phase B** (embeddings, normalization, LLM extraction)

---

# Part 1 — Phase A: Audio → Attributed Transcript (2026-06-26)

Stages L1→L2→L3 turn a raw audio file into a speaker-labelled, multilingual
transcript. *Diarization* = "who spoke when" (segmenting audio by speaker).
*ASR* = automatic speech recognition (audio → text). *Code-switching* = mixing
languages mid-sentence, e.g. Hindi and English, which Indian clinics do
constantly.

## (a) What this phase does
L1 resamples any input audio to 16 kHz mono WAV using librosa (Core Audio backend
on macOS — no ffmpeg needed) and trims leading/trailing silence via energy-threshold
VAD (voice-activity detection); denoising is a keyword-only toggle defaulting to OFF
because stationary noise reduction can *increase* word errors on clean clinic
recordings. L2 runs `pyannote/speaker-diarization-community-1` on the Mac GPU (MPS),
passing audio as a preloaded `{'waveform': tensor, 'sample_rate': int}` dict because
torchcodec (pyannote's preferred audio reader) fails to link against the installed
ffmpeg. L3 uses faster-whisper large-v3 on CPU (int8 quantization, ~3.6 GB peak RAM)
and transcribes each diarized segment with `language=None` so Hindi, English, and
Marathi are detected independently per segment rather than forcing one global language.

## (b) The two hardest bugs
1. **`Pipeline.from_pretrained()` rejected `use_auth_token=`** — root cause:
   pyannote.audio ≥ 3.x switched to the standard HuggingFace `token=` parameter and
   removed `use_auth_token` with no deprecation cycle, so it failed as a hard
   `TypeError`, not a warning.
2. **`DiarizeOutput has no attribute 'itertracks'`** — root cause: the community-1
   model wraps its result in a `DiarizeOutput` dataclass instead of returning a bare
   `pyannote.core.Annotation`; the annotation lives at
   `DiarizeOutput.speaker_diarization`. The public docs only describe the 3.x family,
   so the community model diverges silently.

## (c) Fine-tuning hook
The doctor/patient role heuristic (a bag-of-words score over question-forms and
clinical terms) only fires when ≥2 speakers are present. Every EkaCare ASR-eval clip
is a single-doctor utterance, so all turns came out `UNKNOWN`. A real role classifier
will need full consultations where doctor and patient turns alternate — the
clinical-note-generation dataset (156 transcripts with JSON ground truth), not the
ASR-eval set.

## How L3 works, for a newcomer
faster-whisper is a reimplementation of OpenAI's Whisper running on **CTranslate2**,
a C++ inference engine — *not* PyTorch. That one fact explains most of the design:

- `WhisperModel("large-v3", device="cpu", compute_type="int8")` — CPU, **not** the
  GPU, because CTranslate2 has no Metal backend; on M-series it falls back to Apple's
  AMX matrix units via BLAS. `compute_type="int8"` *quantizes* the weights to 8-bit
  integers (storing each number in 1 byte instead of 4), cutting RAM from ~10 GB to
  ~3.6 GB — the trade the 24 GB hardware budget forces.
- `model.transcribe(...)` returns `(segments_generator, info)`. The generator is
  **lazy** — no transcription runs until you iterate it.
- `language=None` runs **language identification (LID)** per clip — each diarized
  slice gets its own language verdict. This is the whole code-switching strategy.
- `task="transcribe"` pins same-language output, so a Hindi segment is never silently
  translated to English (a hard project rule).
- `beam_size=5` keeps the 5 best running hypotheses instead of greedily taking the
  single most likely next word — slower, more accurate.
- Cleanup is `del model; gc.collect()` so the next stage never shares RAM with this one.

## Three speech-pipeline lessons
The two bugs in (b) were generic library churn. These three are the real
speech-pipeline lessons:
1. **Audio decoding is a native-dependency minefield, and it breaks first.**
   torchcodec couldn't link ffmpeg, so we decode audio ourselves once
   (librosa/soundfile) and hand models an in-memory array, never a file path —
   decoupling decoding from inference.
2. **"I have a GPU" ≠ "this model uses it."** The inference *runtime* matters more
   than the chip: in one pipeline L2 (PyTorch) runs on the GPU while L3 (CTranslate2)
   runs on CPU.
3. **Whisper's language ID degrades on short audio — and diarization hands it exactly
   that.** A sub-second slice gives LID almost no signal, so the language verdict is
   shakiest at the short turns where code-switching happens. (A sample segment was
   tagged English at 0.67 confidence when the audio was Hindi.)

## What actually cost the most time
The spec predicted **code-switch ASR accuracy** would be the hard part. That's
probably right about the *product* — but it was **not** what consumed Phase A, and we
couldn't even measure it yet. What fought back was **plumbing**: native audio
decoding, CPU-vs-GPU runtime mismatch, and the pyannote API surface. None of that is
machine learning. This is the signature of **on-device, local-first pipelines** —
with no cloud API hiding decoding, device placement, and quantization, those
concerns front-load the schedule and the ML risk waits until the eval harness exists.

## A worked failure: "daily three times" — and how we'll fix it
The clearest code-switch error, from sample 2:

- **Reference (truth):** "daily three times."
- **ASR output:** `और डाइली फ्री टाइम्स`

Two failures stacked at one switch point:
1. **Script** — the English words were transliterated into Devanagari
   (`डाइली`="daily", `टाइम्स`="times") instead of written in Latin letters.
2. **Lexical** — **"three" → "फ्री" ("free")**: the "th" sound heard as "f".

**Why it breaks exactly there.** Every segment of this clip was LID'd as Hindi at
0.92–0.98 confidence. Once Whisper commits a segment to the `hi` language token, its
decoder runs in "Hindi mode" — biased toward Devanagari spelling and Hindi sound
patterns. Standard Hindi has **no /θ/** (the "th" in "three"), so a Hindi-conditioned
decoder assigns near-zero probability to a /θ/-initial word and emits the nearest
sound it *can* produce: /f/ → `फ्री`. The model fails at the precise phoneme that
doesn't exist in the language it committed to. Tellingly, later English terms in the
*same* recording ("lab test", "CBC", "chest x-ray", "after 8 days") survived — they
sit inside a long Hindi-context segment with strong medical-English priors, whereas
"daily three times" was its own isolated 1.45-second slice with no context to anchor
the language.

**The root cause is the slicing, not the model.** We feed Whisper one tiny segment at
a time, starving it of the ~30-second context window it needs for reliable language
ID and decoding. The fix, in priority order:
1. **Stop slicing — transcribe the whole file once, then assign words to speakers by
   overlapping word-level timestamps with the diarization** (the whisperX pattern).
   Whisper regains full context, and it also removes the wasteful re-decode-per-segment.
2. **Bias the decoder toward keywords** via `initial_prompt`/`hotwords` seeded with a
   clinical-English lexicon (drug names, "daily/twice/three times", OD/BD/TDS).
3. **Add a downstream safety net** in L3.5/L4: normalize garbled frequency terms, and
   flag any dose/frequency that can't be validated rather than guessing.

This fix is deferred until after the eval harness (Part 2) so it can be **measured**,
not guessed — see Part 2 for why that ordering is deliberate.

---

# Part 2 — Evaluation: Measure Before You Tune (2026-06-26)

## Why we built the eval harness before fixing the ASR error
You cannot responsibly fix "daily three times" without a number to move. Two unknowns
made a blind fix reckless: (1) *how bad is it, really?* — one eyeballed error is not a
WER number; and (2) *does the error even reach the final note?* — the product output
is the structured note, not the raw transcript, and L4 (the LLM) may recover or flag
it. The project rule is explicit: wire metrics in from the start and score every
change on a frozen set. So the harness came first; the ASR fix becomes a measured
experiment.

## The metrics, explained
**WER (Word Error Rate)** is edit distance at the word level, normalized by reference
length — the same edit distance used on strings, but counting whole words. Align the
reference against the hypothesis and count the minimum edits: **S**ubstitutions (wrong
word), **D**eletions (missing word), **I**nsertions (extra word).

    WER = (S + D + I) / N      where N = number of words in the reference

Worked example on the Augmentin case:

    REF:  give  augmentin   twice  daily        (N = 4)
    HYP:  give  augmenting  twice  —
                └ substitution      └ deletion
    S=1, D=1, I=0  →  WER = 2/4 = 0.50

Scale: 0.0 = perfect; 1.0 = as many errors as words; it can exceed 1.0 with many
insertions. Lower is better; clean English dictation is ~0.05–0.10.

**Keyword WER** = `1 − recall` over clinically critical terms (drugs, doses,
diagnostics) taken from the dataset's gold `medical_entities`. It asks: *of the terms
that matter, how many did we lose?* "**micro**" averaging pools across all clips
(total missed ÷ total present), so a clip with more keywords counts more; "macro"
would average each clip's rate equally. We report micro — it's the honest
patient-safety view.

**Drug Keyword WER** = the same, restricted to drug-category terms.

## Phase A ASR baseline (frozen Hindi 10-clip set)

| Metric              | Value | Notes                         |
|---------------------|-------|-------------------------------|
| Corpus WER          | 0.52  | edit errors over all words    |
| Keyword WER (micro) | 0.62  | over 42 clinical keywords     |
| Drug Keyword WER    | 1.00  | over 7 drug terms             |

**The key insight is the *correlation*, not any single number:**

    0.52 (all words)  <  0.62 (keywords)  <  1.00 (drug terms)

The error rate **climbs as the words get more clinically important.** The model is
*least* accurate on exactly the tokens that matter most — drug names, doses, clinical
terms — because those are the rare, English-origin, code-switched words it
transliterates or garbles, while it handles common Hindi filler fine. For a clinical
scribe that is the worst possible error distribution: **accurate where it's harmless,
wrong where it's dangerous.** A single global WER would have hidden this — which is
exactly why we split out keyword and drug metrics.

**Honest caveats (so the numbers are read correctly):**
- **N = 10** is small.
- The scorer is **strict on script**: it heard "daily" but wrote `डाइली` (Devanagari)
  → counted as an error even though it's phonetically right. So 0.52 mixes real errors
  with script-form mismatches and is *pessimistic*.
- **Drug WER 1.00 is partly a metric artifact**: it exact-matches the compound gold
  span "Augmentin 650 mg", and since the drug *name* garbled (`Augmentin → augmenting`)
  the whole span scores as missed — even though the **dose number 650 survived**. The
  real signal is "drug *names* garble," not "doses fail."

These are a **baseline to beat, not a verdict**. Planned refinements: script-folding
before scoring, and separating drug-name / dose / frequency into their own buckets so
we can attribute errors precisely (see Part 3's open question).

---

# Part 3 — Concept Layer: Preparing for Phase B (2026-06-26)

Phase B is L3.5 (normalize lay terms → clinical terms) and L4 (extract a structured
note). Two model types do the work — an **embedding model** and an **LLM** — and the
choice of which goes where is the crux. These notes build the concepts from scratch.

## What an embedding is
An **embedding** turns a piece of text into a list of numbers — a vector — that
represents its *meaning* as a **point in space**. The model is trained so texts with
similar meaning land near each other and unrelated texts land far apart. A typical
embedding has hundreds of dimensions (parrotlet-e: ~1024), but the intuition holds in 2-D.

**The "meaning map" analogy.** On a geographic map, *position* encodes *location* —
Mumbai and Pune are close, Mumbai and Delhi far. An embedding is the same idea with
the axes encoding *meaning* instead of geography. On this meaning-map, "sugar" (the
everyday word for diabetes), "diabetes", "high blood sugar", and "मधुमेह" all sit in
one neighborhood, while "fracture" sits in a distant district. Closeness is measured
by **cosine similarity** — the angle between two vectors; small angle = similar meaning.

**How the positions get there — learned, not hand-placed.** Nobody types in
coordinates, and there's no closed-form formula. It's **metric learning**, the same
recipe as face recognition: a network is trained with a **contrastive/triplet loss**
that pulls *positive pairs* together and pushes *negative pairs* apart, by gradient
descent over millions of examples. For text, ("sugar", "diabetes") is a positive pair
→ pulled together; ("sugar", "fracture") is negative → pushed apart. Humans curate
*which pairs should be close* (often from medical ontologies like UMLS/SNOMED); the
network learns *where the points go*. A generic model would put "sugar" near
"sucrose/dessert" — the food sense — so **parrotlet-e** (bge-m3 fine-tuned on medical
pairs) re-draws the neighborhood so the clinical sense wins, exactly like fine-tuning
an ImageNet backbone on medical images.

**Normalization then = nearest-neighbor lookup on the meaning-map:** pre-place the
canonical clinical vocabulary, embed the patient's phrase, return the nearest
canonical concept. This beats string-matching, where "sugar" and "diabetes" share no
letters but are meaning-neighbors.

## "Normalization" — three different meanings, disambiguated
The word is overloaded. Only the third is the pipeline stage:
1. **Image normalization** — rescaling signal values (pixels to [0,1], z-scoring,
   histogram equalization). A numeric operation on intensities.
2. **Vector normalization** — scaling a vector to unit length (÷ its L2 norm) so you
   can compare directions via cosine similarity. Happens *inside* the embedding math.
3. **Lexical / concept normalization (our L3.5 stage)** — mapping many *surface forms*
   of a concept to one **canonical form** ("sugar", "मधुमेह", "high blood sugar" →
   `Type 2 Diabetes Mellitus`). The NLP/database sense: collapse variants to a
   canonical key. The closest vision analogy is **canonicalization/registration** —
   mapping many variant inputs (lighting, angle, label) to one reference representation.

## Why embeddings for normalization but an LLM for extraction
They are good at opposite things, and each is dangerous in the other's job.
- **Embedding model = closed-world matcher.** Answers "*which known thing is this most
  like?*" Its output is always one of the N entries you pre-placed — it *cannot* return
  a concept outside the vocabulary, which is exactly what you want for normalization
  (a guaranteed-valid clinical term). But it can't read a sentence, handle negation, or
  build structured output.
- **LLM = open-world reasoner/generator.** Answers "*what is the structured meaning of
  all this?*" It reads a messy code-switched transcript and fills a schema (complaint,
  history, meds, follow-up), handling context and negation. That generative power is
  irreplaceable for extraction — and is also the hazard: it can produce things that
  were never in the input.

Rule of thumb: **"pick from a list" → embeddings; "compose structured output" → LLM.**

## Why "never invent a dose" is hard for an LLM specifically
An LLM is a **next-token predictor** trained to produce the most *plausible-sounding*
continuation — not the most *source-faithful* one. Fabrication isn't a defect bolted
on; it's the default behavior of a fluency engine asked to be complete. If the
transcript says "Augmentin de raha hoon" but never states a dose, the model has seen
"Augmentin" followed by "625 mg" thousands of times, so the statistically likely
continuation *is* a dose — and it fills the gap with its prior. Three compounding reasons:
1. **Objective = plausibility, not faithfulness.** Nothing in pretraining rewards
   silence when the source is silent.
2. **No native "I didn't see this."** By default the model doesn't separate *observed*
   from *guessed*; both come out as equally fluent, confident text.
3. **Priors are strongest where stakes are highest** — the most common drugs have the
   most entrenched drug→dose associations, so the model invents most confidently on the
   most standard medications.

Because you can't make the mechanism *want* to abstain, the mitigation is structural:
constrain the output schema, **validate every drug/dose against the CDSCO list**, and
force `validated:false` / `low_confidence_fields` when the source is silent.

## Open question, to be settled with data: where will extraction fail most?
Candidates: drug names, doses, frequencies, diagnoses. Current reasoning:
- **Doses and frequencies are the dangerous pair** because fabrication risk meets the
  *absence of an external validator* — there is no list to check "650 mg" or "twice
  daily" against, so an invented value goes uncaught. Frequencies may be worst:
  ASR already mangles them ("three" → `फ्री`, OD/BD/TDS) *and* they're trivially
  plausible to invent.
- **Drug names are frequently wrong too** (the ASR baseline shows `Augmentin →
  augmenting`), but a garbled drug **fails the CDSCO check** and gets flagged — it's
  *catchable*.
- The ASR baseline hints that drug **names** garble while dose **numbers** sometimes
  survive — but that's the ASR layer; dose **fabrication** is an L4 phenomenon the ASR
  metric structurally cannot see.

This is settled once L4 runs with metrics that **separate drug-name / dose / frequency
/ diagnosis** and, for doses/frequencies, measure **fabrication** (precision: did L4
emit a value the transcript never contained?), not just recall. That metric work is
the bridge from Part 2 into Phase B.

---

# Appendix — Per-phase checkpoint format

Every phase checkpoint appends a dated section in this structure (plain language, no
jargon without a one-line definition):

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
