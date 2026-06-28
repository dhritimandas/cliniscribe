# CliniScribe — Engineering & Research Learnings

A teaching-grade record of how this clinical-scribe MVP was built, measured, and
reasoned about. It is written so that a domain newcomer — or you, three months from
now, or a reviewer at a frontier lab — can read it end-to-end, understand every
decision, and reproduce the results.

## How to read this document

The old version of this file was a single chronological stream that mixed *what the
code does*, *how we run experiments*, and *what we decided to ship*. It was complete
but hard to navigate. This version separates those three concerns into layers, so you
can enter at the layer you need:

| If you want to… | Read |
|---|---|
| Understand the project in one minute | **Part I — The Project in 60 Seconds** |
| Learn *how we think* (the research discipline + the transferable principles) | **Part II — Methodology & Principles** |
| Know *where the code stands* right now, stage by stage | **Part III — Engineering Status** |
| Follow *what was actually tried*, in order, as a lab notebook | **Part IV — Experiment Log** |
| See *what we chose to build and deliberately not build*, and why | **Part V — Product & Schema Decisions** |
| Look up a term, or reproduce a result | **Appendix** (glossary + repro + checkpoint format) |

Cross-references use `[E#]` for experiments (Part IV) and `[P2.#]` for principles
(Part II). Jargon is defined inline on first use and collected in the glossary.

---

# Part I — The Project in 60 Seconds

**What.** CliniScribe is an on-device clinical scribe for Indian tier-2/tier-3 clinics.
A doctor records a 3–8 minute consultation (Hindi/English/Marathi, code-switched);
the system returns a structured clinical note and a prescription PDF that the
physician reviews and signs.

**Two constraints shape everything.**
1. **On-device, 24 GB.** Everything runs locally on a MacBook Air (M-series, 24 GB
   unified RAM). No cloud APIs in the production path — patient privacy, cost, and
   unreliable rural connectivity rule them out.
2. **Patient safety.** A wrong drug name or dose can harm someone. So the metric that
   matters most is not overall transcription quality — it is accuracy on the
   *safety-critical tokens*: drug names, doses, vitals.

**The pipeline** is a sequential, end-of-consultation batch (not streaming):

```
audio → L1 preprocess → L2 diarize → L3 ASR → L3.5 normalize → L4 extract → L5 render → [PHYSICIAN REVIEW]
```

| Stage | Job | Component |
|---|---|---|
| L1 | resample → 16 kHz mono, trim silence, optional denoise | librosa / soundfile / noisereduce |
| L2 | diarization — "who spoke when" | pyannote speaker-diarization-community-1 |
| L3 | ASR — speech → text, in the source language | faster-whisper large-v3 (CTranslate2) |
| L3.5 | normalize lay→clinical terms; fix drug spellings | parrotlet-e embeddings + curated tables |
| L4 | extract structured `ClinicalNote` JSON | qwen2.5:3b-instruct via Ollama |
| L5 | render prescription PDF | reportlab |

**Memory discipline (non-negotiable).** Load one model, run it, release it
(`del model; gc.collect()`). Never hold the ASR model and the LLM resident at once —
they do not both fit in 24 GB. The batch design exists precisely to make this clean.

**The prime directive.** *Measure before you tune.* Every model swap or prompt change
is scored on a **frozen evaluation set** before it is accepted. This single rule
generated most of the insights in this document — see Part II.

---

# Part II — Methodology & Principles

This is the "how we think" layer. The engineering details live in Parts III–IV; here
is the reasoning discipline that produced them, plus the transferable principles
worth carrying to the next project.

## 2.1 The prime directive — measure before you tune

You cannot responsibly change a system you cannot measure. Two questions make any
blind fix reckless: *how bad is it, really?* (one eyeballed error is not a number) and
*does the error even reach the product?* (the output is the structured note, not the
raw transcript — a later stage may recover or flag it). So the evaluation harness is
built **before** the first optimization, and every change becomes a *measured
experiment*, not a guess.

**Frozen eval set.** We lock a fixed set of test examples and never change it between
experiments. Analogy: to track weight loss you weigh yourself on the *same scale* each
morning — swap scales daily and any change might be the instrument, not you. A frozen
set is the same scale every time, so a moved number means the *change* moved it, not an
easier batch of examples. Our frozen sets: a 10-clip Hindi ASR set, a 15-clip / 33-drug
ASR set, and a 24-sample L4 set (12 English + 12 Hindi/Marathi).

## 2.2 The metrics — KARMA

| Metric | Plain meaning | Why it exists |
|---|---|---|
| **WER** | Word Error Rate = (Substitutions + Deletions + Insertions) ÷ reference words | baseline transcript quality |
| **Keyword WER** | `1 − recall` over clinically critical terms (drugs, doses, diagnostics) | the safety view |
| **Drug Keyword WER** | same, restricted to drug names | the sharpest safety view |
| **DER** | Diarization Error Rate — how often "who spoke" is wrong | L2 quality |
| **Concept-match** | did "sugar" correctly map to "diabetes"? | L3.5 quality |

WER scale: `0.0` = perfect, `1.0` = as many errors as words (can exceed 1.0 with many
insertions); clean English dictation is ~0.05–0.10. Worked example:

```
REF:  give  augmentin   twice  daily      (N = 4 reference words)
HYP:  give  augmenting  twice  —
            └ substitution      └ deletion
S=1, D=1, I=0  →  WER = 2/4 = 0.50
```

We report **micro** averaging for keyword WER (total missed ÷ total present, so a clip
with more keywords counts more) — the honest patient-safety pooling.

## 2.3 One change → one experiment → one measurement → one attribution

The core experimental loop:

```
1. FREEZE the eval set.
2. MEASURE a baseline — write the number down.
3. CHANGE exactly ONE thing.
4. RE-MEASURE on the SAME frozen set.
5. ATTRIBUTE the delta to that one change. Keep if better, revert if worse.
```

The discipline is in step 3: change *one* thing. When two fixes ship together, their
deltas are entangled forever. The cleanest example in this project is the
`medication_frequency` recovery [E9]: we measured a baseline (0.040), then the
**eval-matcher fix alone** (0.620), then the eval+prompt fix (0.938) — *as three
separate measurements*. That ordering is the only reason we can say "78% of the
apparent failure was a broken scorer, 22% was a real model gain." Had we changed both
at once, that attribution would be permanently lost.

## 2.4 The recurring lesson — suspect the ruler before the model

When a metric looks bad, the first question is **not** "how do I fix the model." It is:
*is the failure in the model, or in the measuring instrument (the schema, the scorer,
the eval set)?* This project hit that fork three times, and each time a large fraction
of the "model failure" was actually a measurement failure:

- **Schema coverage [E6]:** 64% of ground-truth rubric criteria targeted a field our
  output schema could not even hold. No prompt or fine-tune can emit a field that does
  not exist. *Fix the contract, not the model.*
- **Devanagari drug names [E8]:** Whisper heard "Augmentin" correctly but wrote it in
  Devanagari script; an exact Latin-script comparison scored it wrong. *Fix the
  normalizer/scorer, not the acoustic model.*
- **Dosing frequency notation [E9]:** the rubric used `1-0-1`, the model said "twice
  daily" — synonyms the exact-match scorer rejected. *Fix the matcher, not the model.*

If you skip this question you can spend weeks prompt-engineering a model that was
already correct, because your ruler was bent.

## 2.5 Principle — loanword-in-foreign-script is not transliteration (a lookup table is the only fix)

This is why the curated Devanagari→Latin **table** [E8, Tier 1] beats a
romanization-only ("transliterate then match") upper bound. The example: Whisper writes
the English word *sunscreen* phonetically in Devanagari as `सनस्क्रीन` (reads roughly
"sa-nas-kreen").

There are two categorically different normalization problems hiding under one word:

- **Native-word transliteration is rule-governed.** A Hindi word written in Devanagari
  has a *canonical* romanization because the Devanagari spelling **is** the word's
  phonological truth. Script-A → Script-B is a deterministic function (ITRANS, IAST).
  No table needed beyond the romanization rules; it generalizes to unseen words.

- **Loanword-in-foreign-script is convention-governed, not rule-governed.** `सनस्क्रीन`
  encodes the *Indian-accented English sound*, but the target label is the *English
  orthography* — and English spelling is famously non-phonetic and irregular ("sun" not
  "san", silent letters, "colonel"). There is **no rule** that maps the sound to the
  spelling. Romanizing `सनस्क्रीन` faithfully yields "sanaskreen" — the sound — which is
  **not** "sunscreen." Edit-distance fuzzy matching also fails: the surface gap is large,
  and a wrong drug may sit closer in edit distance than the right one.

Because the mapping is an arbitrary per-word convention, the **only** thing that bridges
it is *memorized pairs* — a lookup table now (`सनस्क्रीन → sunscreen`), a fine-tuned
model later. Rule-based methods (romanization) and metric-based methods (edit distance)
both have a ceiling **below 100%** on this class, structurally. That is why Tier 1 (the
table) can exceed the transliteration-based upper bound: for loanwords, transliteration
*cannot* produce the irregular target at all.

**Takeaway:** before choosing a normalization method, classify the problem. Rule-governed
variation → a rule (transliteration, stemming). Convention-governed variation → memorization
(a table or a learned mapping). Using a rule on a convention problem caps you below the
ceiling no matter how much you tune it.

## 2.6 Principle — two places to intervene: before vs after the decode bottleneck

`initial_prompt` biasing [E8, Goal 3] can touch acoustic misses that *post-processing
cannot*, because it acts **before** transcription, not after. Understanding why is
worth more than the (negative) result itself.

Whisper's decoder is **autoregressive**: at each step it samples the next token from
`P(token | audio-encoding, text-context-so-far)` — a blend of *acoustic evidence*
(cross-attention to the audio encoder) and a learned *language prior* over the running
text. The running text includes whatever you put in `initial_prompt`.

- **`initial_prompt` edits the prior, before the bottleneck.** Seeding the context with
  drug names shifts `P(token | context)`, so in an **acoustically ambiguous** region
  (where the audio underdetermines the word and several tokens are plausible), the
  posterior tips toward prompt-consistent tokens. A smeared "Augmentin" can resolve
  correctly *because* "Augmentin" primes the context. This re-weights the search **while
  the alternatives are still live.**

- **Post-processing edits the output, after the bottleneck.** It only sees the *final*
  text — after beam search/argmax already collapsed the distribution and discarded the
  alternatives. If the word was dropped or mangled beyond recognition, the information
  needed to recover it is simply gone. Post-processing can transform what survived; it
  cannot add back what the decode threw away.

That is the fundamental difference: one modifies the *generative process before the
information bottleneck*; the other transforms the *output after it*. It is also why the
two map cleanly onto the miss taxonomy in [P2.7]: `initial_prompt` can rescue
*DISTORTED-but-present* misses (ambiguous → tipped correct); nothing downstream can.

**Why it still backfired here.** The prior is **global and indiscriminate** — it raises
the entire English/Latin-script register *everywhere*, not just in drug regions. So in
segments where the audio clearly said a Hindi word, the English-biased prior overrode
correct acoustic evidence and **suppressed correctly-transcribed Devanagari terms**
(3 regressions for 1 recovery). The lever is real but unaimable on code-switched audio.

## 2.7 Principle — knowing when a lever is EXHAUSTED (distance-to-floor ≤ noise)

After normalization, the Latin drug-keyword WER sits at **0.576**, just **0.030 above
the acoustic floor of ~0.546** (the misses that are acoustic in nature — distorted or
absent — which *no text post-processing* can touch). How do you know the
text-normalization lever is done, rather than worth another weekend?

**The principle:** measure your progress against the **floor of the lever you are
pulling**, not against perfection (0). Every lever has a floor it cannot cross — text
post-processing cannot transcribe a word that is not in the audio. A lever is
**exhausted** when:

```
(distance to that lever's floor)  ≤  (measurement resolution / noise band)
```

Here the remaining headroom is 0.030 ≈ **one drug out of 33** (each drug = 3.0% of the
metric). That is *at* the resolution limit of a 33-sample bench — any "improvement" you
measure inside it is indistinguishable from which clips happened to land in the set.
Continuing to optimize is polishing noise. The correct move is to **switch levers** to
one whose floor is lower: the acoustic floor is crossed only by *better acoustics*
(domain fine-tuning for the distorted class; better microphones / SNR for the true-drop
class — see [P2.6], [E8]).

**To avoid over-working a near-solved stage:** compute each lever's floor *before* you
work it, and define "done for this lever" as reaching floor + noise band — then stop and
move to the lever with the lower floor.

## 2.8 Principle — anchor targets to demonstrated ceilings, not wishes

A target was once proposed: drive acoustic drug-miss from **54.5% → 10%** in one shot.
The best *demonstrated* system on comparable Indian medical ASR — Saaras (Sarvam's
cloud model, proprietary Indian corpus, far more resources than a local 3B-class setup)
— reaches only **~42%**. A 10% target therefore asks to **beat the best-resourced
system in the field by ~4×**, using a smaller, local, open stack.

**The principle:** before adopting a performance target, locate the **best demonstrated
result** in the field on a comparable task — the SOTA ceiling — and check your target
against it. A target *below* the demonstrated ceiling (i.e., better than anyone has
shown, especially anyone with more resources) is not "ambitious," it is **ungrounded**,
unless you can name the *specific new mechanism* (novel method, better data) by which
you will beat SOTA. Absent that mechanism, the right target is anchored *relative to the
ceiling*: e.g., "approach Saaras's 42% from our 54.5% floor," or "close X% of the
floor→ceiling gap." Targets pulled from a desired business outcome rather than a
demonstrated reference point burn effort chasing the impossible — and tempt you to cheat
the measurement to hit them (see [P2.10]).

## 2.9 Principle — small samples make aggressive targets meaningless

The 54.5% → 10% target lives on **33 drugs from 15 clips**. Beyond being ungrounded
[P2.8], it is *unmeasurable* at this sample size, for two compounding reasons.

**1. Resolution.** Each drug is `1/33 = 3.03%` of the metric. The metric can only take
values in ~3% steps. "10%" is not even cleanly expressible — the nearest reachable
values are 9.1% (3/33) and 12.1% (4/33). Sub-drug precision is meaningless.

**2. Confidence intervals dwarf the moves.** The 95% CI half-width for a proportion is
`≈ 1.96·√(p(1−p)/n)`:
- at the floor (p≈0.545, n=33): `±0.170` → the true rate is somewhere in **[37%, 72%]**.
- at the target (p≈0.10, n=33): `±0.102` → **[0%, 20%]**.

You do not even know your *starting point* to better than ±17%. Which 15 clips you drew
dominates the number more than your method does.

**What sample size would you actually need?** It depends on which move you want to
detect (two-proportion power, α=0.05, power 0.80):
- To detect the (ungrounded) **54.5% → 10%** move — a *huge* effect — you need only
  **~14 drugs per arm**. So small sample is *not* what makes the big target hard; the
  field ceiling [P2.8] is.
- To detect the **moves actually available** near the floor — roughly one drug, a ~3%
  change (e.g. 54.5% → 51.5%) — you need **~4,300 drugs per arm**.

That is the real meaninglessness: the *only detectable* target (the big jump) is
*physically unreachable*, while the *only reachable* targets (small, near-floor moves)
are *undetectable* at n=33. With 33 drugs you can neither legitimately hit the
aggressive target nor measure the legitimate progress that remains.

## 2.10 Principle — if only cheating reaches the target, the target is wrong

On this data the *only* paths to "10%" were **overfitting** (tune to the 15 specific
clips — memorize the test set) or **matcher-loosening** (relax what counts as a
"correct" drug match until misses become hits). Both were rejected as cheating.

**The principle:** when the only routes to a target corrupt the measurement, treat that
as *evidence the target is wrong*, not as a cue to be cleverer. This is Goodhart's Law —
"when a measure becomes a target, it ceases to be a good measure." Each cheat destroys
exactly the thing the metric protects:
- *Overfitting* = test-set leakage → the metric no longer predicts generalization.
- *Matcher-loosening* = the metric no longer means "clinically correct drug," which in a
  patient-safety tool means it stops protecting the patient.

A cheat does not remove the failure; it *relocates* it — from a visible miss now to an
invisible miss in production. For a clinical tool that invisible miss is a patient harm
you can no longer see. The correct response is to **re-anchor** the target to the
demonstrated frontier (floor [P2.7] + field ceiling [P2.8]), not to find a clever path.

---

# Part III — Engineering Status

Where the code stands, stage by stage. Bug *detail* lives in Part IV; this is the
scannable status layer.

| Stage | Status | Headline number / state |
|---|---|---|
| L1 preprocess | ✅ built | 16 kHz mono + VAD trim; denoise default **OFF** (it raises WER on clean audio) |
| L2 diarize | ✅ built | pyannote community-1 on MPS; role heuristic fires only with ≥2 speakers |
| L3 ASR | ✅ built | faster-whisper large-v3, int8, ~3.6 GB; per-segment language ID |
| L3.5 normalize | ✅ built | parrotlet-e gloss + 3-tier drug normalization; Latin drug-KW WER 0.818 → **0.576** |
| L4 extract | ✅ built | qwen2.5:3b; 24-sample baseline aggregate recall **0.306**; field recovery in [E9] |
| L5 render | ✅ built | reportlab PDF; standard vitals rows always shown; red draft banner |

**L1 — preprocess.** Resamples to 16 kHz mono WAV (librosa, Core Audio backend on
macOS — no ffmpeg). Trims leading/trailing silence via energy-threshold VAD. Denoising
is a keyword-only toggle, default OFF — stationary noise reduction *increases* WER on
clean clinic recordings (benchmark per noise profile before enabling).

**L2 — diarize.** `pyannote/speaker-diarization-community-1` on the Mac GPU (MPS). Audio
is handed in as a preloaded `{'waveform': tensor, 'sample_rate': int}` dict because
torchcodec (pyannote's preferred reader) fails to link the installed ffmpeg. Doctor/
patient role assignment is a bag-of-words heuristic that only fires with ≥2 speakers;
single-utterance ASR-eval clips all come out `UNKNOWN` (a real role classifier needs
full alternating consultations — the 156-transcript set, not the ASR-eval set).

**L3 — ASR.** faster-whisper large-v3 on **CTranslate2** (a C++ engine, *not* PyTorch —
this fact explains the design). `device="cpu"` because CTranslate2 has no Metal backend;
`compute_type="int8"` quantizes weights to 8-bit, cutting RAM ~10 GB → ~3.6 GB.
`language=None` runs per-segment language ID (the code-switch strategy); `task="transcribe"`
forbids silent translation; `beam_size=5` for accuracy. Cleanup `del model; gc.collect()`.

**L3.5 — normalize.** Two jobs: (1) gloss lay→clinical terms non-destructively via
parrotlet-e embeddings (`फीवर (Fever)`), 28-concept table with a hard-negative rejection
gate; (2) a 3-tier drug-name normalizer (curated table → ITRANS+CDSCO → guarded fuzzy)
wired into the live pipeline.

**L4 — extract.** qwen2.5:3b-instruct via Ollama at temperature 0, JSON-constrained,
one retry. Every drug validated against the CDSCO list; unrecognized drugs / missing
doses flagged in `low_confidence_fields`. Schema widened to the high-value SOAP fields
[E6]; field-routing precision is the current frontier [E9].

**L5 — render.** reportlab PDF. Standard vitals rows (Height/Weight/BP/Temperature/SpO2/
Pulse) always rendered, "—" when absent; medications in their own section; a 10 pt red
draft banner marks it as physician-review-pending.

---

# Part IV — Experiment Log

A chronological lab notebook. Each entry: what it did, the hardest bugs (root cause, not
just symptom), and the forward-looking hook. Principles extracted from these experiments
live in Part II and are cross-referenced.

## [E1] Phase A build — audio → attributed transcript (2026-06-26)

**What.** Stood up L1→L2→L3: raw audio to a speaker-labelled, multilingual transcript.

**Hardest bugs.**
1. `Pipeline.from_pretrained()` rejected `use_auth_token=` — pyannote.audio ≥ 3.x
   switched to HuggingFace's `token=` and removed the old arg with no deprecation cycle
   (hard `TypeError`, not a warning).
2. `DiarizeOutput has no attribute 'itertracks'` — the community-1 model wraps its
   result in a `DiarizeOutput` dataclass; the annotation lives at
   `DiarizeOutput.speaker_diarization`. Public docs describe only the 3.x family.

**What actually cost the most time.** Not ML — **plumbing**: native audio decoding
(torchcodec/ffmpeg linkage), CPU-vs-GPU runtime mismatch (L2 on GPU via PyTorch, L3 on
CPU via CTranslate2), and the pyannote API surface. This is the signature of on-device,
local-first pipelines: with no cloud API hiding decoding, device placement, and
quantization, those concerns front-load the schedule and the ML risk waits for the eval
harness. Three durable lessons: (a) audio decoding is a native-dependency minefield —
decode once ourselves and hand models an in-memory array, never a file path; (b) "I have
a GPU" ≠ "this model uses it" — the runtime decides; (c) Whisper's language ID degrades
on short audio, and diarization hands it exactly that.

**A worked failure ("daily three times").** Reference "daily three times" → ASR
`और डाइली फ्री टाइम्स`. Two failures stacked: the English words were transliterated into
Devanagari, and "three" became `फ्री` ("free") — the "th" /θ/ sound heard as /f/. Root
cause: Standard Hindi has no /θ/, so once Whisper committed the 1.45 s slice to the `hi`
language token, its decoder assigned near-zero probability to a /θ/-initial word and
emitted the nearest sound it *could* produce. Tellingly, English terms inside long
Hindi-context segments ("CBC", "chest x-ray") survived — the isolated short slice had no
context to anchor language ID. **The root cause is the slicing, not the model**: feeding
Whisper one tiny segment starves it of the ~30 s context window it needs. Deferred fix
(measured, not guessed): transcribe the whole file once and assign words to speakers by
timestamp overlap (the whisperX pattern).

**Hook.** The role heuristic needs full consultations to train a real doctor/patient
classifier — the 156-transcript clinical-note set, not the single-utterance ASR-eval set.

## [E2] ASR baseline — the climbing error rate (2026-06-26)

**What.** Built the WER harness *first* [P2.1] and took a baseline on the frozen Hindi
10-clip set.

**Result.**

| Metric | Value | Over |
|---|---|---|
| Corpus WER | 0.52 | all words |
| Keyword WER (micro) | 0.62 | 42 clinical keywords |
| Drug Keyword WER | 1.00 | 7 drug terms |

**The insight is the correlation, not any single number:** `0.52 < 0.62 < 1.00`. The
error rate **climbs as the words get more clinically important** — the model is *least*
accurate on exactly the tokens that matter most, because those are the rare,
English-origin, code-switched words it transliterates or garbles, while it handles common
Hindi filler fine. For a clinical scribe that is the worst possible error distribution:
**accurate where it's harmless, wrong where it's dangerous.** A single global WER would
have hidden this — which is why we split keyword and drug metrics out.

**Honest caveats.** N=10 is small; the scorer is strict on script (Devanagari "daily"
counts as an error though phonetically right), so 0.52 is *pessimistic*; Drug WER 1.00 is
partly a metric artifact (the compound span "Augmentin 650 mg" scores as fully missed
because the *name* garbled, even though the dose number 650 survived). A baseline to
beat, not a verdict.

## [E3] Concept layer — embeddings vs LLM (2026-06-26)

**What.** Designed the division of labor for Phase B before building it. The crux: which
model does which job.

**The reasoning.** An **embedding** turns text into a vector whose *position* encodes
*meaning* — a "meaning map" where "sugar", "diabetes", "madhumeh", and "मधुमेह" cluster
and "fracture" sits far away (cosine similarity = closeness). Positions are *learned* by
metric learning (contrastive/triplet loss pulling positive pairs together), not
hand-placed; parrotlet-e is bge-m3 fine-tuned on medical pairs so the *clinical* sense of
"sugar" wins over the *food* sense. So:

- **Embedding model = closed-world matcher.** Answers "which known thing is this most
  like?" Output is always one of the N pre-placed entries — *cannot* return a concept
  outside the vocabulary, which is exactly what normalization wants (a guaranteed-valid
  clinical term). Cannot read a sentence, handle negation, or build structure.
- **LLM = open-world reasoner.** Answers "what is the structured meaning of all this?"
  Reads a messy transcript and fills a schema, handling context and negation. That
  generative power is irreplaceable for extraction — and is also the hazard: it can emit
  things never in the input.

Rule of thumb: **"pick from a list" → embeddings; "compose structured output" → LLM.**

**Why "never invent a dose" is hard for an LLM specifically.** An LLM is a next-token
predictor trained for *plausibility*, not *source-faithfulness*. It has seen "Augmentin"
→ "625 mg" thousands of times, so when the transcript states no dose, the statistically
likely continuation *is* a dose. Fabrication is the default behavior of a fluency engine
asked to be complete; priors are strongest on the most common drugs (highest stakes).
You cannot make the mechanism *want* to abstain, so the mitigation is structural:
constrain the schema, validate every drug/dose against CDSCO, force `validated:false` /
`low_confidence_fields` when the source is silent.

## [E4] Phase B build — L3.5 + L4 (2026-06-26)

**What.** L3.5 embeds 1–3-word spans, glosses those above cosine 0.65 in parentheses
(`फीवर (Fever)`, sim 0.865 across scripts). L4 sends the glossed transcript to qwen2.5:3b,
extracts the `ClinicalNote` schema at temperature 0, runs CDSCO validation.

**Hardest bugs.**
1. `ModuleNotFoundError: transformers` at *runtime, not install time* — root cause:
   `transformers` was an implicit transitive dependency that happened to be visible
   outside the venv but not inside it. Fix: pin `transformers` and `torch` explicitly in
   `requirements.txt`; never rely on transitive deps.
2. Gloss fires on `फीवर` but not on `infection` — root cause: concept-*coverage*, not
   model quality. The embedding correctly places both; "infection" simply had no entry in
   the 18-concept table to map to. Adding the concept fixes it.

**Hook.** The cosine threshold (0.65) is a fixed precision/recall knob; tuning it needs
the concept-match metric and a per-language calibration on a frozen hold-out — or, better,
explicit hard-negative contrast examples (which [E5] added).

## [E5] Concept table v2 — hard negatives + coverage (2026-06-27)

**What.** Added a hard-negative rejection gate (the *margin test*) and expanded the table
18 → 28 concepts with SNOMED CT IDs.

**Hardest findings.**
1. **Canonical-term-only reference fails for colloquialisms** — "sugar" → "Type 2
   Diabetes Mellitus" scores only cosine 0.33 (the model rarely saw that bridge), while
   formal "madhumeh" → T2DM scores 0.75. Fix: put *variant* texts (including "sugar")
   directly in the reference matrix so "sugar" matches variant "sugar" at ~1.0.
2. **The margin gate is necessary but insufficient for unigrams** — "cold outside" is
   correctly rejected (margin −0.49), but the bare unigram "cold" passes (margin 0.37):
   the model treats "cold" as intrinsically clinical. A 5% global margin cannot separate
   "I have a cold" from "it is cold outside" at the unigram level; sentence-level encoding
   would, but needs a different inference architecture. Tracked for Phase D.

**Hook.** `HARDNEG_MARGIN` (0.05) and `COSINE_THRESHOLD` (0.65) are hand-tuned globals;
the "cold" case wants ~0.37 while valid matches pass at 0.09 — a single global margin is
incoherent. Per-concept thresholds calibrated on Phase D's concept-match metric are the
answer. Until then, favor higher recall (L4 + physician review are downstream catches).

## [E6] Schema coverage reality check — the contract was the ceiling (2026-06-27)

**What.** *Before* touching the model, measured what fraction of EkaCare ground-truth
rubric criteria target a field our `ClinicalNote` schema could not even *hold*. (A
*rubric* = one scored success criterion attached to a transcript; the dataset ships 2469
across 156 transcripts.)

**The finding: 64% (1585/2469) of all rubric criteria targeted a field the old schema
could not represent.** 153/156 transcripts had ≥1 structurally unsatisfiable rubric. This
is a **contract decision, not a model failure** [P2.4] — no prompt or fine-tune emits a
field that does not exist.

| Missing field group | Criteria | % |
|---|---|---|
| Symptoms (name/severity/laterality) | 580 | 23.5% |
| Vitals | 205 | 8.3% |
| Structured medical history | 197 | 8.0% |
| Diagnostic results (labs in hand) | 193 | 7.8% |
| Examination findings | 163 | 6.6% |
| Medication timing | 111 | 4.5% |
| Diagnosis status/laterality | 74 | 3.0% |
| Lifestyle/family/allergy/travel | 62 | 2.5% |

**What changed.** Extended the schema to the high-value SOAP fields (symptoms, vitals,
free-text examination, diagnosis.status, medication.timing, and `diagnostic_results`
*separate from* `investigations`) — covering ~84% of the missing criteria. Deliberately
left history unstructured — see [P5] for the fabrication-surface rationale. **The result
is that field-routing is now a *measurable* model-quality problem rather than a hidden
structural one** (e.g. "Hb 9.2" landing in both `vitals` and `diagnostic_results`).

## [E7] L4 defect fixes + repeatable scorer + baseline (2026-06-27)

**What.** Fixed two code bugs and two design gaps, narrowed a fabrication-encouraging
prompt rule, added a hallucination-calibration check, and built the repeatable L4 eval
(previously `raise NotImplementedError`). Produced the first 24-sample baseline.

**Hardest bugs.**
1. **`_build_note` crashes on null list fields, silently swallowing the whole note** —
   `data.get("symptoms", [])` returns `None` (not `[]`) when the key is present with value
   `null`; `dict.get(key, default)` only substitutes the default when the key is *absent*.
   `None` is not iterable → `TypeError`, swallowed by a bare `except` that returned an
   empty note. Fix: `(data.get("key") or [])`. (The asymmetry — some fields already had
   `or []` — was the diagnostic signal.)
2. **CDSCO validation rejects most real Indian prescriptions** — three compounding causes:
   exact-match on bare generics missed "Tablet paracetamol"; common Indian brands (Dolo,
   Meftal Spas, Pantop, Shelcal…) were absent from the seed set; and the false-unvalidated
   chain buried the `low_confidence_fields` signal it was meant to provide. Fix:
   dosage-form stripping, bidirectional substring/token matching, ~50 brand names added.

**Baseline (post-fix, frozen 24-sample set):**

```
Category                  Total  Rep  Match  Recall
medication_name             104  104     64   0.615
diagnosis_name               30   30     13   0.433
diagnosis_status             23   23      9   0.391
body_vital_sign_name         41   41     15   0.366
prescribed_test_name         37   37     14   0.378
symptom_name                 87   87     31   0.356
medication_timing            45   45     12   0.267
symptom_severity             12   12      3   0.250
examination_name             34   34      7   0.206
examination_notes            33   33      5   0.152
medication_frequency         86   86      5   0.058   ← attacked in [E9]
diagnostic_result_name       34   34      1   0.029   ← attacked in [E9]
medication_dose              24   24      0   0.000   ← dose-null policy [P5]
AGGREGATE                   726  630    193   0.306
```

Observations: medication_name (0.615) is strongest; `medication_frequency` and
`diagnostic_result_name` are the weakest extractable fields; 6/12 Hindi/Marathi samples
returned 0 extractions (a 3B Devanagari-segmentation capability gap — a future target).

**Hook.** The hallucination calibration (word overlap between diagnosis term and
transcript) is a floor, not a ceiling: it catches "Pulmonary Embolism" on an acne
transcript (no overlap) but not a confident "Hypertension" where "BP" was mentioned in
passing. The real signal is a *calibration loss* — train the model to assign low
probability to diagnosis tokens absent a supporting evidence phrase (extractive-QA
"no answer" analogue).

## [E8] ASR Drug Keyword WER — normalization + decoder biasing (2026-06-28)

**What.** Whisper writes English drug names in Devanagari script (`Augmentin` →
`ऑर्ग्यूमेंटिंग`), so a Latin-script match fails though the *sound* was captured. This
experiment measured the miss, built a normalizer to recover what text-processing can,
and characterized what it cannot.

**Goal 2 — shippable normalization (3 tiers).** (1) hand-curated Devanagari→Latin table
(111 entries) for English-phonetic loanwords [P2.5]; (2) ITRANS romanization + exact CDSCO
lookup for standard Devanagari spellings; (3) length-guarded fuzzy match (≥8-char CDSCO
candidate, threshold 0.82). On the frozen 15-clip / 33-drug set: **Latin WER 0.818 →
0.576**, closing **87.5% of the recoverable gap**.

**Goal 3 — acoustic miss decomposition + `initial_prompt`.** The residual misses split
into two kinds [P2.6]:
- **DISTORTED-but-present** — a phonetically similar token exists at that position (the
  decoder heard something, spelled it wrong). *Decoder-biasable.*
- **TRUE DROP** — no phonetic trace at all (the audio never triggered a nearby token).
  *Audio-quality bounded; unrecoverable by any text or prompt trick.*

Results (7-clip intersection, beam_size=1):
- Baseline acoustic WER 0.5714 (8/14 missed); decomposition **3 distorted : 5 true drop**.
- `initial_prompt` (~50 drug names): recovered **1/3 distorted, 0/5 true drop**.
- **Regressions: 3** — native Devanagari terms (`एंटीबायोटिक्स`, …) the baseline got right
  were suppressed by the English-biased prior [P2.6].
- **Net verdict: counterproductive** — biased WER 0.714 vs baseline 0.571. Killed early at
  7/15 by a pre-set kill condition. Do not ship an English-list `initial_prompt` into
  Hindi-dominant audio.
- **Irreducible residual:** 2 still-distorted (→ ASR fine-tuning) + 8 true-drop (→ better
  audio capture).

**Hardest bugs.**
1. **Normalization scoring produced negative recovery counts** — the "after" count only
   checked the normalized hypothesis; when a Devanagari gold label was substituted to
   Latin, the Devanagari form vanished and counted as a *new* miss → negative recovery.
   Fix: enforce the invariant `missed_after ⊆ missed_before` by checking both raw and
   normalized hypotheses. Recovery can now only improve or stay flat.
2. **faster-whisper hangs 10+ min after loading a 43 MB parquet** — CTranslate2
   initializes its allocator at `WhisperModel(...)` time; a prior large pandas/pyarrow
   allocation fragments memory and the buffer setup stalls. Fix: **model-first loading** —
   construct the model before any parquet read. Never hold a large native allocation when
   initializing CTranslate2.

**Hook.** Acoustic partition 3 distorted : 5 true drop. Domain-adaptive ASR fine-tuning
(on a larger Indian-clinic corpus — a *DISPLACE-style* domain set; "DISPLACE-M" here is
shorthand for that *category* of corpus, not a confirmed off-the-shelf dataset, so verify
availability before planning around it — and **not** the 15-clip bench, far too small
without overfitting [P2.9]) is the correct lever for the *distorted* class; the *true-drop*
class is bounded by microphone and SNR. Any future decoder biasing must be applied *selectively*
(only when the hypothesis is already Latin/English) to avoid the Devanagari-suppression
regression [P2.6].

## [E9] L4 field recovery — frequency + diagnostic results (2026-06-28)

**What.** The two weakest extractable fields — `medication_frequency` (0.040) and
`diagnostic_result_name` (0.067) — failed even on *clean English* samples, so the cause
was not ASR. Split the failure into *eval-canonicalization* vs *true model omission*
[P2.4], fixed both the scorer and the prompt, and measured the gains **separately** [P2.3].

**Result (English samples):**

| Fix | medication_frequency | diagnostic_result_name |
|---|---|---|
| Baseline (old eval + old prompt, 24-sample) | 2/50 = 0.040 | 1/15 = 0.067 |
| **Eval-fix only** (canon map + prefix match) | 31/50 = 0.620 | 3/15 = 0.200 |
| **Full fix** (eval + prompt, 6 repr. samples) | 30/32 = 0.938 | 5/13 = 0.385 |

**Attribution** [P2.3]: for `medication_frequency`, the dominant failure was a *broken
ruler* — the rubric uses Indian dosing notation `1-0-1` (morning-noon-night: `1-0-1` =
twice daily, `1-1-1` = thrice, `0-0-1` = once at night), the model says "twice daily";
synonyms the exact-match scorer rejected. The scorer fix alone recovered +0.58 of the
+0.90 total (≈78%). The prompt fix recovered the genuine omissions (time-of-day routed to
the wrong field). For `diagnostic_result_name`, the residual is a *true* 3B capacity limit
— the model does not reliably extract every lab value from a dense multi-system note.

**Hardest bugs.**
1. **Canon map key mismatch after normalization** — `_normalise()` strips punctuation, so
   `'1-0-0'` becomes `'100'` *before* the canonical lookup; the map had only `"1-0-0"`, so
   every X-X-X criterion missed. Fix: store **both** raw and post-normalization forms as
   keys. Lesson: when a normalizer runs before a lookup table, every key must be in its
   post-normalization form (or both forms stored).
2. **Time-of-day phrases routed to the wrong field** — the prompt defined `timing` with an
   "at bedtime" example but never *excluded* time-of-day; the 3B model pattern-matched "at
   night" → `timing` instead of `frequency`. Fix: a prescriptive rule listing which
   surface forms belong in `frequency` (`"at night"`, `"SOS"`, `"1-0-0"`) vs `timing`
   (meal context only), with worked examples. A 3B model needs the patterns, not an
   abstract principle.

**Hook.** A 3B model can reliably *detect* a dosing schedule but labels it inconsistently
(`0-0-1`, "once nightly", "at night" all for the same dose). A fine-tuned model should
emit a *canonical* form (X-X-X or SNOMED frequency codes), removing the need for a
hand-maintained synonym map and stabilizing the scorer, the PDF, and any pharmacy
integration.

---

# Part V — Product & Schema Decisions

The "what we chose to build and *not* build, and why" layer. These are deliberate product
decisions, recorded so they are not silently re-litigated.

**Schema scope = what consultations actually contain.** The `ClinicalNote` field set is
aligned to the high-value EkaCare rubric targets [E6]: symptoms (≈23% of criteria),
vitals (≈8%), examination (≈7%), diagnostic results (≈8%), medication timing (≈5%),
diagnosis status (≈3%). `investigations` = tests *ordered for later*; `diagnostic_results`
= results *already in hand* — the split matters for the note's clinical meaning.

**Structured history is deliberately free-text.** Past/family/social/lifestyle history
(≈10% of criteria) is left as a free-text `history` field, not the 8 structured sub-arrays
EkaCare offers. Two reasons: (1) Indian tier-2/3 transcripts are 3–8 minutes and rarely
take a systematic history on tape, so the fields would be empty almost always; (2) **an
empty structured field is a fabrication surface** — the same next-token-plausibility
hazard behind "never invent a dose" [E3]. **Adding a field has a cost, not just a benefit.**

**Dose-null policy = KEPT.** Dose is `null` unless explicitly stated; never inferred. The
dataset convention of reading "1 tablet" from the word "Tablet" is a scoring artifact, not
a clinical instruction — auto-filling a dose risks a 2×/5× overdose if the physician
rubber-stamps it. The physician review step exists to fill such gaps from clinical
judgment. This *costs* recall on `medication_dose` (0.000) by design.

**Do-not-translate policy = KEPT.** Run ASR and extraction in the source language;
pre-translating Hindi/Marathi to English degrades accuracy and loses code-switch nuance.
The cross-lingual eval gap (English rubric vs source-language note) is a *measurement
limitation* [P2.4], handled by a Devanagari presence check in the scorer — not a model
failure.

**Denoising default = OFF.** Stationary noise reduction raises WER on clean recordings;
it is a per-noise-profile toggle, benchmarked before enabling, never always-on.

---

# Appendix

## A. Glossary

- **ASR** — Automatic Speech Recognition (audio → text).
- **Diarization** — segmenting audio by *who* is speaking ("who spoke when").
- **Code-switching** — mixing languages mid-sentence (Hindi + English), constant in
  Indian clinics.
- **WER / Keyword WER / Drug-KW WER** — see [P2.2].
- **DER** — Diarization Error Rate.
- **Embedding** — a vector whose position encodes meaning; similarity = cosine of the
  angle between vectors. See [E3].
- **LID** — Language IDentification (Whisper's per-segment language verdict).
- **CTranslate2** — the C++ inference engine behind faster-whisper (not PyTorch; CPU/AMX,
  no Metal).
- **Quantization (int8)** — storing weights as 8-bit integers to cut RAM (~10 GB → 3.6 GB).
- **CDSCO** — India's drug regulator; its approved-drug list is the validation dictionary.
- **SNOMED CT** — a clinical terminology; provides canonical concept IDs.
- **Rubric** — one scored success criterion attached to a transcript in the EkaCare set.
- **DISTORTED-but-present / TRUE DROP** — the two acoustic-miss kinds. See [E8], [P2.6].
- **Acoustic floor** — the WER no text post-processing can cross (distorted + dropped
  misses). See [P2.7].

## B. Reproduce it

```bash
source .venv/bin/activate
pip install -r requirements.txt
ollama list                      # qwen2.5:3b-instruct must be present for L4
pytest tests/ -v                 # all stage tests
python src/pipeline.py <wav>     # full pipeline on one file
python eval/run_eval.py          # KARMA eval on the frozen sets
```

Frozen sets: 10-clip Hindi ASR; 15-clip / 33-drug ASR; 24-sample L4 (12 EN + 12 HI/MR,
indices in `eval/run_eval.py:FROZEN_INDICES`). Datasets live in `data/` (gitignored),
HF_TOKEN in `.env`. Key files: `src/l3_5_normalize.py` (3-tier normalizer),
`src/l4_extract.py` (prompt rules + `_build_note`), `eval/run_eval.py` (rubric scorer +
frequency canon map).

## C. Per-phase checkpoint format

Every phase checkpoint appends an experiment entry to Part IV in this structure (plain
language, no jargon without a one-line definition):

```
## [E#] <name> (YYYY-MM-DD)

**What.** <3 sentences>

**Hardest bugs.**
1. <bug> — root cause: <why, not just the symptom>
2. <bug> — root cause: <why, not just the symptom>

**Hook.** <one thing that will matter when we fine-tune later>
```
