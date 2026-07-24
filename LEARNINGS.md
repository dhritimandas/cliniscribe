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

## Phase B — L3.5 Normalize + L4 Extract (2026-06-26)

### (a) What this phase does
L3.5 loads the parrotlet-e embedding model (a multilingual medical encoder,
567 MB, running on MPS), embeds candidate word spans (1–3 words) from each
transcript turn, and compares them against embeddings of canonical clinical
terms; spans above cosine 0.65 are glossed non-destructively in parentheses
(e.g., `फीवर (Fever)`). L4 sends the glossed transcript to qwen2.5:3b-instruct
via Ollama, extracts the `ClinicalNote` JSON schema at temperature=0, and runs
every drug name through a CDSCO-approved-drug lookup, setting `validated=false`
and adding a `low_confidence_fields` entry for any unrecognised drug or missing
dose. The two stages hand off through the same `list[Turn]` → `ClinicalNote`
contract established in Phase A.

### (b) Hardest bugs

1. **`ModuleNotFoundError: No module named 'transformers'` at runtime, not at
   install time** — root cause: pyannote.audio pulls PyTorch as a dependency,
   and PyTorch coexists with the system miniforge install which has
   `transformers` in its own site-packages. When running inside the project
   venv, the system packages are not visible, so `import transformers` silently
   worked during early prototyping (outside the venv) but failed on the first
   in-venv L3.5 call. Adding `transformers` and `torch` explicitly to
   `requirements.txt` is the fix; implicit transitive dependencies cannot be
   relied on.

2. **L3.5 gloss fires on `फीवर` but not on `infection`** — root cause: both
   are in the model's embedding space, but `infection` does not appear in our
   18-concept table. The concept table is symptom-oriented (fever, cough, pain);
   generic diagnoses like "infection" have no canonical entry to map to. The
   embedding model is working correctly — it correctly maps `फीवर` (Devanagari
   "fever") to `Fever` (sim=0.865) across script boundaries. The failure for
   `infection` is concept-coverage, not model quality. Adding an "Infection"
   concept (SNOMED 40733004) to `src/concepts.py` would catch it.

### (c) Fine-tuning hook
The cosine threshold (0.65) is a fixed constant that controls the precision/
recall trade-off of concept matching. A threshold that is too low causes false
positives (common words like "cold" in "feeling cold" gloss to "Common Cold");
too high means genuine lay terms get missed. Fine-tuning the threshold requires
the **concept-match accuracy metric** from the KARMA framework (Phase D), which
scores whether each matched span truly belongs to the glossed concept. During
fine-tuning, the right move is not to tune the model weights but to tune the
threshold per language (English, Hindi romanized, Devanagari) by running the
metric on a frozen hold-out set of annotated transcript–concept pairs. An
alternative — and potentially more powerful — approach is to add known-hard
negatives (common words near but below the clinical boundary) as explicit
contrast examples when expanding `src/concepts.py`.

---

## Phase B Enhancement — Concept Table v2: Hard Negatives + Coverage Expansion (2026-06-27)

### (a) What this enhancement does
Two problems surfaced after running Phase B on real EkaCare data. First, common English
words like "cold", "gas", and "tension" are exact surface-form matches to concept
variants, so they gloss correctly in clinical context ("I have a cold") but would also
fire on non-clinical text ("it's cold outside", "gas cylinder"). Second, the 18-concept
table had no entries for Abdominal Pain, Nausea, Asthma, Anxiety, Migraine, Back Pain,
Fungal Infection, Allergic Rhinitis, URTI, or Loss of Appetite — all high-frequency
conditions in the EkaCare 156-transcript dataset. This enhancement adds a hard-negative
rejection gate (the margin test) and expands the table from 18 to 28 concepts with
SNOMED CT identifiers for all entries.

### (b) Hardest design decisions

1. **Canonical-term-only reference fails for colloquial abbreviations** — root cause:
   parrotlet-e (the medical embedding model) cannot bridge the gap between a colloquial
   abbreviation and its canonical expansion. "sugar" → "Type 2 Diabetes Mellitus" scores
   cosine=0.33 (below the 0.65 threshold), because the model was trained on medical text
   where "sugar" rarely co-occurs with T2DM in a way that builds a direct bridge. By
   contrast, formal Hindi ("madhumeh" → T2DM) scores 0.75 and cross-language paraphrase
   ("high blood pressure" → "Hypertension") scores 0.76 — both well above threshold. The
   fix is to include all variant texts alongside canonical terms in the reference matrix.
   "sugar" then matches variant "sugar" at sim~1.0, which maps to T2DM. The hard-negative
   gate (see below) is what prevents this exact-match-to-anything behavior from causing
   false positives.

2. **Hard-negative gate is necessary but insufficient for unigram ambiguity** — the
   margin test (`concept_sim > hn_sim + HARDNEG_MARGIN`) correctly rejects context-heavy
   spans: "cold outside" scores 0.428 to Common Cold but 0.919 to hard-negative "it's
   cold outside", so margin = −0.49 → **rejected**. But the unigram "cold" scores 1.0 to
   the variant and only 0.630 to hard-negative "cold weather" — margin = 0.37, which
   passes easily. The model, trained on medical text, treats the bare word "cold" as
   intrinsically clinical; a 5% margin cannot distinguish "I have a cold" from "it is
   cold outside" at the unigram level. Sentence-level encoding (encoding the whole
   sentence context, not just the span) would fix this but requires a different inference
   architecture. This is a known limitation tracked for Phase D.

### (c) Fine-tuning hook
`HARDNEG_MARGIN` (0.05) and `COSINE_THRESHOLD` (0.65) are both hand-tuned constants.
The diagnostic above shows that "cold" disambiguation alone would require a margin of
~0.37 — 7× the global setting — while valid clinical matches like "pait mein" →
Abdominal Pain pass with margin=0.09. This spread makes a single global margin
incoherent. Per-concept thresholds, calibrated against Phase D's concept-match accuracy
metric on a frozen hold-out set, are the right answer. Until then, err on the side of a
lower threshold (higher recall) since L4 and the physician review step are downstream
correction layers.

---

## Phase B Enhancement — ClinicalNote Schema Coverage (2026-06-27)

### Why this exists: the contract, not the model, was the ceiling
Before fixing anything in the LLM, we measured what fraction of the EkaCare
ground-truth rubrics target a field our `ClinicalNote` schema could not even
*hold*. A **rubric** here is one scored success criterion attached to a transcript
(e.g. "a symptom matching 'nausea' is present in the symptoms array"). The dataset
ships 2469 such criteria across 156 transcripts. We grouped each by the field it
targets and asked a model-independent question: *if our extractor were perfect,
could the schema even represent the answer?*

**The finding: 64% (1585/2469) of all rubric criteria target a field the old schema
could not represent.** 153 of 156 transcripts had at least one structurally
unsatisfiable rubric. This is a **contract decision, not a model failure** — no
amount of prompt tuning or fine-tuning can emit a field that does not exist in the
output schema. The breakdown:

| Missing field group | Criteria | % of all rubrics |
|---|---|---|
| Symptoms (name / severity / laterality) | 580 | 23.5% |
| Vitals (BP, SpO2, pulse, ...) | 205 | 8.3% |
| Structured medical history | 197 | 8.0% |
| Diagnostic results (labs in hand) | 193 | 7.8% |
| Examination findings | 163 | 6.6% |
| Medication timing (before/after food) | 111 | 4.5% |
| Diagnosis status / laterality | 74 | 3.0% |
| Lifestyle / family / allergy / travel | 62 | 2.5% |

### What we changed and what we deliberately left out
We extended the schema to cover the **clinically high-value SOAP fields**: a
`symptoms` array (name, finding_status, severity, since), a `vitals` array
(name + value-with-unit), free-text `examination`, structured `diagnosis.status`,
`medication.timing`, and a `diagnostic_results` list **separate from**
`investigations`. The split matters: *investigations* are tests the doctor **orders
for later**; *diagnostic_results* are values **already available** in the room
("Hb is 9.2"). These six additions cover ~1326 of the 1585 missing criteria (84%).

We **intentionally did not** structure past/family/social/lifestyle history into the
8 sub-arrays the EkaCare schema offers (~10% of criteria). Two reasons: (1) Indian
tier-2/3 clinic transcripts are 3-8 minutes and rarely take a systematic social or
family history on tape, so those fields would be empty almost always; (2) empty
structured fields are an *invitation* for the LLM to fabricate — the same
next-token-plausibility hazard that makes "never invent a dose" hard (see Part 3).
Free-text `history` absorbs what little of it appears. **Adding a field has a cost,
not just a benefit: every optional structured field is a fabrication surface.**

### The honest result: schema is sufficient, model precision now becomes measurable
A live smoke test through qwen2.5:3b on a transcript exercising every new field
confirmed each field is reachable and populates. But it also exposed the *next*
problem, which is now a **measurable** model-quality issue rather than a hidden
structural one:
- **Field-classification ambiguity**: "Hb 9.2" landed in **both** `vitals` and
  `diagnostic_results`. The model does not reliably distinguish a measured vital sign
  from a lab result, even with explicit prompt rules. (Hb is a lab result.)
- **Recall misses**: medication timing "before food" leaked into the free-text
  `advice` field instead of `medications[].timing`; a reported symptom (nausea) and a
  *denied* one ("no vomiting" → finding_status Absent) were both dropped; palpation
  findings never reached `examination`.

The point of the schema fix is exactly this: these are now **scorable against the
rubrics**. Before, "nausea missing" and "schema has no symptoms array" were
indistinguishable in the final score; now the first is a recall number we can move
with prompting or fine-tuning, and the second no longer exists.

### Fine-tuning hook
With the contract widened, the next gains are **field-routing precision**, not
coverage. The dataset's own rubric guidance is lenient here ("INFORMATION PRESENCE
OVER FIELD LOCATION" — frequency stated inside `instruction` still scores as a
match), so the scoring tolerates the timing-in-advice leak. But for a clean EMR
hand-off the physician needs fields in their right slots. The fine-tuning signal is
the per-category rubric score (symptom_name vs vital vs diagnostic_result), which
isolates *recall* (did we extract it at all?) from *routing* (did it go in the right
field?). The vital-vs-result confusion in particular wants either few-shot examples
contrasting the two, or a post-extraction reclassifier keyed on whether a number has
a reference range.

---

## Phase B Error Analysis — Defect Fixes and Repeatable L4 Scorer (2026-06-27)

### (a) What this phase does
Running L4 directly on the EkaCare 156-transcript dataset surfaced four confirmed
defects — two code bugs and two design gaps — plus a missing evaluation harness.
This phase fixes the two code bugs (a silent null-crash and a CDSCO false-rejection
cascade), narrows a prompt instruction that was encouraging fabrication, adds a
post-extraction hallucination calibration check, and builds the repeatable L4
evaluation that was previously `raise NotImplementedError`. After the fixes a
frozen 24-sample eval set (12 English, 12 Hindi/Marathi) can be run to produce
a before/after per-category recall comparison.

### (b) Hardest bugs

1. **`_build_note` crashes on null list fields, silently swallowing an entire note**
   — root cause: `data.get("symptoms", [])` returns `None` (not `[]`) when the
   JSON key is present with value `null` (e.g., `"symptoms": null`). Python's
   `dict.get(key, default)` only substitutes the default when the key is **absent**;
   a key with an explicit `null` value is present and returns `None`. `None` is not
   iterable, so the list comprehension raises `TypeError`. This was caught by the
   bare `except Exception` in `extract()`, which returned `_empty_note()` — losing
   all extracted data with no visible error. The symptom was observed on transcript
   i=18 (Dolo 650 fever case): medications and symptoms were fully absent from the
   returned note. The fix is `(data.get("key") or [])` — the `or` converts `None`
   to `[]` regardless of whether the key was absent or explicitly null.
   Note: `investigations`, `diagnostic_results`, and `low_confidence_fields` already
   had `or []` guards — that asymmetry in the old code was the diagnostic signal.

2. **CDSCO validation rejects the majority of real Indian prescriptions**
   — root cause: three compounding issues, all arising from a too-narrow design:
   (a) the lookup was exact set-membership on bare generic names, but the model
   frequently prepends the dosage form ("Tablet paracetamol") which is not in the
   set even though "paracetamol" is; (b) common Indian branded drugs (Dolo, Moxclav,
   Bifilac, Meftal Spas, Grenil, Ultracet, Pantop, Shelcal, Foracort, Limcee,
   Pan D, Asthalin, etc.) were absent from the seed set entirely — every brand-name
   prescription was flagged unvalidated; (c) the false-unvalidated chain then
   triggered `medications.<drug>.unvalidated` entries in `low_confidence_fields`
   for every medication, burying the signal that flag was supposed to provide. The
   fix adds dosage-form stripping before lookup, bidirectional substring and
   token-overlap matching, and ~50 common Indian brand names to the seed set.

### (c) Fine-tuning hook
The diagnosis hallucination calibration added here (word-token overlap between
the diagnosis term and the transcript) is a necessary floor but not sufficient.
It catches "Pulmonary Embolism" on an acne transcript because no word overlaps.
It does **not** catch a model that confidently adds "Hypertension" to a transcript
where "BP" or "blood pressure" were mentioned in passing without any diagnosis being
stated — because the words do overlap. The correct fine-tuning signal is a
**calibration loss**: train the model to assign low probability to diagnosis tokens
when no supporting evidence phrase appears in the context window. This is analogous
to a reading-comprehension extractive QA model being trained to output "no answer"
when the answer is not in the passage. Until then, the word-overlap check + the
physician review layer are the safety net.

**Baseline eval results (post-fix, frozen 24-sample set — 12 EN + 12 HI/MR):**

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
medication_frequency         86   86      5   0.058
diagnostic_result_name       34   34      1   0.029
medication_dose              24   24      0   0.000   ← dose null rule (see policy)
AGGREGATE                   726  630    193   0.306
Unrepresentable criteria: 96/726 = 13.2% (schema gap, not model failure)
```

Key observations:
- medication_name recall (0.615) is the strongest — the model extracts drug names well
- medication_dose recall (0.000) is a known consequence of the dose-null policy (correct)
- medication_frequency (0.058) and diagnostic_result_name (0.029) are the weakest extractable fields — both require the model to produce structured strings that can fuzzy-match English rubric criteria from a source-language transcript
- Hindi/Marathi samples: 6 of 12 returned 0 extractions (model capability gap at 3B scale on Devanagari entity segmentation) — this is the primary next-phase investigation target

**Policy decisions documented here (not changed unilaterally):**

*Dose null rule (prompt rule 2):* KEPT. The dataset convention of inferring "1 tablet"
from the word "Tablet" is a scoring artefact for LLM judges, not a clinical
instruction. Fabricating a dose that was not stated risks a 2× or 5× overdose if
the physician rubber-stamps the auto-fill. Rubric-match gain here is at the cost
of patient safety. Recommendation: keep `dose=null` as the explicit default; the
physician review step exists precisely to fill gaps like this from their clinical
judgment.

*Do-not-translate rule (prompt rule 6):* KEPT. Pre-translating Hindi/Marathi to
English before extraction consistently degrades accuracy and loses code-switch
nuance. The cross-lingual recall gap in the evaluation (rubric in English, note
extracted in source language) is a **measurement limitation**, not a model failure
— addressed in the scorer by a presence check for Devanagari rows.

---

## Phase B Enhancement — ASR Drug Keyword WER: Normalization Shippable Path + Decoder Biasing (2026-06-28)

### (a) What this phase does
The ASR stage (faster-whisper large-v3) misses most drug names because Indian clinical
speech is heavily code-switched: doctors say drug names in English inside Hindi sentences.
Whisper writes those English names in Devanagari script ("Augmentin" → `ऑर्ग्यूमेंटिंग`),
so an exact Latin-script match fails even though the *pronunciation* was captured. This
phase measures the miss precisely, builds a three-tier normalization pipeline that
recovers what can be recovered by text post-processing alone, and characterizes what
cannot — classifying the residual acoustic misses into two kinds and testing whether
seeding the decoder with a drug-name vocabulary (`initial_prompt`) closes any of the gap.

**Goal 2 — shippable normalization path.** A three-tier pipeline on the faster-whisper
hypothesis: (1) a hand-curated Devanagari→Latin table (111 entries) for English-phonetic
loanwords Whisper renders in Devanagari, (2) ITRANS romanization + exact CDSCO lookup for
standard Devanagari spellings, (3) length-guarded fuzzy matching (≥8-char CDSCO candidate,
threshold 0.82) for near-misses. Measured on the frozen Hindi-15 set (33 drug terms
across 15 clips): Latin WER = 0.818 → 0.576 after normalization, closing 87.5% of the
recoverable gap between the model's acoustic output and the Latin-surface label.

**Goal 3 — acoustic miss decomposition + initial_prompt biasing.** Not every miss is the
same. *DISTORTED-but-present* misses have a phonetically similar token in the hypothesis
at that position (the decoder heard something, just spelled it wrong or distorted it) —
these are decoder-biasable. *TRUE DROP* misses leave no phonetic trace at all (the audio
never triggered a token near the drug name) — these are audio-quality or coverage-bounded
regardless of post-processing or prompt tricks. Seeding faster-whisper with ~50 common
Indian clinic drug names via `initial_prompt` shifts the decoder's token priors before
the beam search runs; the idea is that a distorted drug has a higher probability of
resolving to the correct token when that token appears in the context window.

**Results (7-clip intersection, beam_size=1, DISTORT_FUZZ=0.42):**

- Baseline acoustic WER on 7 clips: **0.5714** (8/14 drug terms missed).
- Decomposition of the 8 baseline acoustic misses:
  - **(a) DISTORTED-but-present: 3** — clip3 "medicine" (ratio 0.421), clip18 "cough syrup"
    (0.421), clip18 "Paracetamol 625 mg tablet" (0.486).
  - **(b) TRUE DROP: 5** — clip1 "sunscreen" (0.353), clip3 "medicines" (0.400),
    clip3 "Fluconazole" (0.400), clip18 "Benadryl cough syrup" (0.400), clip18 "10 ml" (0.333).
- initial_prompt recovery on 7 clips: **1/3 distorted recovered** (Paracetamol 625mg tablet),
  **0/5 true drops recovered** (expected — no token proximity means prompt context cannot help).
- **Regressions: 3 new misses** introduced by the biased pass —
  clip5 "वैद ऋषि का अशकल्प", clip13 "एंटीबायोटिक्स", clip16 "एंटीबायोटिक".
  All three are native Hindi/Devanagari terms that the baseline transcribed correctly;
  the English-heavy prompt biased the decoder toward Latin-script output and suppressed them.
- **Net verdict: initial_prompt is counterproductive.** Biased acoustic WER = 0.714 vs
  baseline 0.571 on the same 7 clips. Run terminated early at 7/15 by design (kill condition:
  WER worse at 7/15). Do not ship `initial_prompt` with an English drug list into a
  Hindi-dominant transcript — it trades 3 Hindi drug recoveries for 3 Hindi regressions and
  only 1 distorted-class recovery.
- **Irreducible residual (after best attempt):** 2 still-distorted misses → addressable only
  by ASR fine-tuning (DISPLACE-M on a larger domain corpus, not this 15-clip set);
  8 true-drop misses → addressable only by better audio capture (mic placement, SNR).

### (b) Hardest bugs

1. **Normalization scoring produced negative recovery counts** — root cause: the original
   `score(ref, hyp, kws)` function checked only `norm_hyp` for the "after" count. When
   a Devanagari gold label (e.g. `एंटीबायोटिक्स`) was substituted to Latin (`antibiotics`)
   in `norm_hyp` by the normalization pipeline, the Devanagari form was no longer found
   → counted as a new miss → recovery went negative on clips where Devanagari drugs were
   in the reference. Fix: `missed_after ⊆ missed_before` invariant enforced by checking
   both the raw hypothesis and the normalized hypothesis for each miss:
   `missed_after = [k for k in missed_before if k not in nhyp_norm]`. Recovery can now
   only improve or stay the same, never regress.

2. **faster-whisper large-v3 hangs for 10+ minutes after loading a 43 MB parquet
   file** — root cause: CTranslate2 (the C++ inference engine behind faster-whisper)
   initializes its thread pool and memory allocator at `WhisperModel(...)` time. When
   a 43 MB parquet containing audio byte-arrays has already been loaded by pandas
   (which invokes pyarrow and allocates large native memory blocks), CTranslate2 sees
   a fragmented allocator state and spends >10 minutes on the tensor buffer setup.
   On a clean process, the same model loads in ~60s. Fix: model-first loading — call
   `WhisperModel(...)` before any `import pandas` or parquet read. The rule: never
   hold a large native-memory allocation when initializing CTranslate2.

### (c) Fine-tuning hook
On the 7-clip intersection, the acoustic miss partition is **3 distorted : 5 true drop**
(37.5% : 62.5%). **DISPLACE-M** (domain-adaptive fine-tuning on a larger corpus of Indian
clinic audio — not the current ~15-clip bench, which is too small to fine-tune on without
overfitting) is the correct next lever for the distorted class: it teaches the ASR model
which phoneme confusions matter clinically (e.g. the Devanagari /ṭ/-initial form of
"Augmentin" should produce "augmentin"). For the true-drop class (62.5% of residual
misses), the ceiling is audio quality and mic placement — no model change can recover a
drug name that was never acoustically present. The `initial_prompt` result adds a
constraint: any decoder-biasing technique using a Latin-script drug list must be applied
selectively only when the hypothesis is already in Latin/English, or it will suppress
correctly-transcribed Devanagari terms in mixed-language consultations.

---

## Phase B Extension — L4 Field Recovery: medication_frequency + diagnostic_result_name (2026-06-28)

### (a) What this phase does
Two L4 output fields — `medication_frequency` and `diagnostic_result_name` — scored
near-zero on English evaluation samples (0.040 and 0.067) despite clean ASR. The
analysis split failures into two distinct categories: *eval-canonicalization failures*
(the rubric uses `X-X-X` dosing notation like `1-0-1`, the model outputs English phrases
like `"twice daily"` — synonymous but unmatched by exact string comparison) and *true
model omissions* (time-of-day phrases being placed in the wrong JSON field, lab results
not extracted at all). We fixed both the eval matcher (bidirectional frequency canon map
+ prefix abbreviation matching) and the L4 system prompt (two new rules), then measured
the combined gain on 6 representative English samples.

**Results summary (English samples):**

| Fix | medication_frequency | diagnostic_result_name |
|---|---|---|
| Baseline (old eval + old prompt, 24-sample) | 2/50 = 0.040 | 1/15 = 0.067 |
| Eval-fix only (canon map + prefix match, old prompt, 24-sample) | 31/50 = 0.620 | 3/15 = 0.200 |
| Full fix (new eval + new prompt, 6 representative samples) | 30/32 = 0.938 | 5/13 = 0.385 |

The 2 remaining frequency misses in 30/32 are both from a single complex hospital note
(`idx=91`): model returns `"in the afternoon"` and `"3 mg in the night"` as frequency
values, which contain the correct information but don't match any earlier canon key.
Adding `"in the night"` and `"in the afternoon"` to the canonical map (done after the
eval run) resolves them. The `diagnostic_result_name` residual (8/13 still missing)
are true model omissions — the 3B model doesn't reliably extract all lab values from a
dense multi-system consultation; this is a model capacity limit, not a matcher problem.

### (b) Hardest bugs

1. **Canon map key mismatch after normalisation** — root cause: `_normalise()` strips
   punctuation via `re.compile(r'[^a-z0-9 ]')`, turning `'1-0-0'` into `'100'`.
   The frequency canon map had `"1-0-0": "once_daily"` but `_canonical_freq()` was
   called *after* `_normalise()`, so it received `'100'` and found no key. Every
   X-X-X rubric criterion was therefore treated as unmatched even when the model
   produced a semantically correct English synonym. Fix: store both the raw and
   normalised forms as keys — `"1-0-0": "once_daily"` AND `"100": "once_daily"` (and
   similarly for all other notation variants). Lesson: when a normalisation function is
   applied to strings before they hit a lookup table, every key in that table must be
   in its post-normalisation form, or both forms must be stored.

2. **Time-of-day phrases routing to the wrong JSON field** — root cause: the original
   L4 prompt defined `timing` as `"before food | after food | at bedtime | None"` but
   never explicitly excluded time-of-day phrases. The 3B model pattern-matched `"at
   night"` to the `"at bedtime"` example in `timing` without understanding the
   semantic distinction between meal-relative context (timing) and dosing schedule
   (frequency). Result: `Sibelium: freq=None, timing='at night'` instead of
   `freq='once at night', timing=None`. Fix: Rule 11 in the updated system prompt
   explicitly lists which surface forms belong in `frequency` (time-of-day: `"at night"`,
   `"SOS"`, `"in the morning"`; dosing notation: `"BD/SOS"`, `"1-0-0"`) versus
   `timing` (meal context only: `"before food"`, `"after food"`, `"with food"`), with
   worked examples. The rule must be prescriptive and list surface forms, not just
   state an abstract principle — a 3B model needs the example patterns.

### (c) Fine-tuning hook
The frequency field shows a 3B model can reliably *detect* that a dosing schedule is
being stated, but labels it inconsistently: the same once-daily-at-night dose can appear
as `"0-0-1"`, `"once nightly"`, `"once at night"`, `"at night"`, or `"0-0-1 in the
night"` across different consultations. A fine-tuned model should be trained to output
a *canonical form* (standardize on `X-X-X` notation or a fixed phrase set like SNOMED
Frequency codes) — this would make the downstream scorer, PDF renderer, and any
integration with pharmacy dispensing systems more reliable without requiring a
bidirectional synonym map that must be maintained by hand.

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

---

## Phase B Hardening — Honest rulers, context windows, and the drug bench (2026-07-10)

### (a) What this phase does
This phase repaired the measurement instruments (the "rulers") and the two worst
production defects they had been hiding, then built a drug-keyword bench large
enough to say something real about drug-name recovery. The eval scorer moved from
substring matching (which counted "ors" as found inside "doctors") to
token-boundary matching, the L4 extractor got the context window it silently
lacked, and a 49-clip frozen drug bench replaced a 7-term sample that could not
measure recovery at all. Every fix was measured on frozen data before and after,
with the drug-bench holdout kept blind to all fix decisions.

### (b) Hardest bugs

1. **Six of twelve Hindi/Marathi consultations extracted nothing — blamed on the
   model, caused by a config default.** Root cause: `extract()` never set
   Ollama's `num_ctx`, and the default (4096 tokens) silently truncates long
   Devanagari transcripts (a 5.4k-character transcript is ~5,000 tokens — 
   Devanagari costs roughly one token per character). The truncation cut off the
   system prompt containing the JSON schema, so the model returned literally
   `{}`. The failure correlated perfectly with transcript LENGTH, not script or
   language — that correlation, visible in ten minutes of static analysis, was
   the tell that pointed away from the "3B Devanagari capability gap" hypothesis
   the previous phase had recorded. Fix: `EXTRACT_NUM_CTX=16384`. All five
   previously-empty samples now produce full notes; frozen-set aggregate recall
   rose 0.306 → 0.495. Lesson (fourth occurrence): when a model "fails", first
   prove the model actually SAW the input.

2. **The drug normalizer's new fuzzy tier deleted a dose.** Extending fuzzy
   CDSCO matching to Latin spans made "paracetamol 625" match the canonical
   entry "paracetamol" (similarity 0.88) — and the substitution replaced the
   span, silently discarding "625". Root cause: span-replacement semantics
   assumed the canonical form carries at least as much information as the span
   it replaces, which is false whenever the canonical is a prefix of a
   drug+dose phrase. On a prescription, a deleted dose is a patient-safety
   incident, not a rounding error. Fix: a substitution may never remove a digit
   token that does not survive in the canonical form; regression-tested. Lesson:
   any automatic text substitution in a clinical pipeline needs an information-
   preservation invariant, not just a similarity threshold.

### (c) Fine-tuning hook
The 49-clip drug bench decomposes drug misses into classes with different owners:
distorted-but-present forms dominate (15 of 18 dev misses), and a measured subset
is text-recoverable (dev drug WER 0.750 → 0.600 via the Latin-span tier plus
script-symmetric scoring). The residual distorted class — Devanagari brand
variants like जिफिट for ज़ीफी (Zifi) at ~0.67 similarity — sits BELOW any safe
substitution threshold: mapping it by text risks substituting the wrong drug.
That class is precisely what ASR fine-tuning on a domain corpus should fix, and
the bench now provides the frozen before-number and metric ladder
(raw → normalized → folded) to prove whether it did. Open hypothesis for a fresh
bench slice (rows 60+): extending the Latin fuzzy tier's lexicon from CDSCO-only
to include curated brand values would recover 1-char Latin brand distortions
("glycomate" → Glycomet); it was discovered by inspecting the holdout, so it
must be validated on data neither dev nor holdout has touched.

---

## Phase B Hardening (2) — Long measurements on a laptop, and closing the original 0/7 (2026-07-10)

### (a) What this phase does
This phase made multi-hour benchmark runs survivable on a MacBook that sleeps,
restarts its tooling, and throttles background work — and then used the finished
bench to answer the question that started it: are the original seven missed drug
terms still missed? Compute now runs detached from the assistant session with
per-clip caching, so any interruption costs at most one clip. The verdict on the
original seven: one recovered, and the other six are provably not text-recoverable
— a decomposition, not an excuse.

### (b) Hardest bugs

1. **Background workers kept dying, three different ways, and each looked like
   the same mystery.** Root causes, once separated: (i) benchmark processes were
   children of the assistant session, so every session restart reaped them; (ii)
   macOS puts orphaned "nohup" processes into background quality-of-service — a
   scheduling class that confines them to efficiency cores, silently capping each
   at ~87% of one core; (iii) laptop sleep paused what survived overnight (5 clips
   progressed in 8 hours). No single observation distinguished these — the fix
   required treating them as a stack: detach the process tree (nohup + disown, so
   the work outlives the session), lift the QoS clamp (taskpolicy), and hold the
   machine awake for the run's duration (caffeinate with a time cap). Lesson:
   "the job died" is not a root cause; process lifecycle, scheduler class, and
   power state fail independently and must be ruled out independently.

2. **A completion check that could never fire.** The waiter tested "50 clips
   cached" because the bench spans 50 dataset rows — but the dataset contains a
   duplicate transcript hash, so 50 rows yield 49 unique clips. The workers all
   exited successfully while the monitor waited forever for a 50th clip that
   does not exist. Root cause: conflating row COUNT with unique IDENTITY —
   the same class of error as double-counting a patient who registered twice.
   Lesson: completion conditions must be derived from the same identity key the
   work is deduplicated by, never from the input's nominal size.

### (c) Fine-tuning hook
The original seven misses are now fully attributed: 1 recovered by the Latin-span
tier + fold-symmetric scoring (Augmentin 650 mg), 3 are gold-labeling artifacts
(generic words — "medicine(s)" — that our own prescription layer refuses to treat
as drug names), 2 are true acoustic drops (sunscreen, Fluconazole — better
microphone, not better models), and 1 is an out-of-lexicon Ayurvedic compound.
Combined with the 49-clip bench (dev drug WER 0.750 → 0.600 after text fixes),
the text-processing ceiling is now measured on two frozen sets. Everything above
that ceiling and below the true-drop floor is the ASR fine-tuning target, and the
per-class attribution means a future fine-tune can be scored on exactly the class
it claims to fix — distorted-but-present drug tokens — rather than on an average
that mixes in unfixable misses.

---

## Latency Phase — Instrument first, and the fix that failed its own gate (2026-07-10)

### (a) What this phase does
This phase measured where the pipeline's time actually goes before optimizing
anything, then tested three latency levers against a hard rule: no accuracy
change. Instrumentation (per-stage wall clock, model-load times, peak memory,
written per session) showed transcription is 82–92% of end-to-end time and
everything else is almost irrelevant. One lever shipped (warming the LLM during
an earlier CPU-light stage), and two were measured and rejected — including the
one we had believed in for weeks.

### (b) Hardest bugs

1. **Whole-file ASR — the designed fix — regressed the patient-safety metric
   and was rejected.** The plan (recorded in HANDOFF since Phase A) was to stop
   decoding per diarized segment and transcribe the whole file once, attaching
   words to speakers by timestamp. Implemented and measured on the frozen set:
   overall word error IMPROVED (0.52 → 0.43) but keyword error — drug names,
   clinical terms — regressed badly (0.57 → 0.79). Root cause: one whole-file
   pass locks the clip into a single detected language, so English clinical
   terms embedded in Hindi speech get phonetically absorbed into Devanagari
   ("lab test" → "लाप टेस्ट"). The per-segment decoding we wanted to remove was
   accidentally PROTECTING code-switched terms, because each short segment
   re-detects its own language. The documented flag for this (multilingual=True)
   measured byte-identical — inert on clips short enough to fit one internal
   chunk. Lesson: an improvement on the headline metric can be a regression on
   the metric that matters; and a "textbook fix" is a hypothesis until measured.

2. **Attribution under noise: the totals moved the wrong way while the change
   worked.** After shipping the LLM warm-up, end-to-end totals LOOKED 14–16%
   worse — because transcription (untouched by the change) drifted with thermal
   state between runs, swamping the real gain. The honest resolution was a
   variance envelope: isolate the changed stage and repeat it — warm inference
   spread was 0.4s across runs while the cold-start penalty was 12.8s, so the
   improvement is ~30x the noise floor. Lesson: when the dominant stage is
   noisy, never present end-to-end totals as evidence about a non-dominant
   stage; measure the changed stage against its own repeat-variance.

### (c) Fine-tuning hook
The latency ceiling is transcription itself: ~20x real-time on CPU with the
current model, and both safe software levers are exhausted (whole-file decoding
fails the accuracy gate; more threads are slower on this chip). The remaining
latency moves are (1) hiding transcription inside recording time — designed,
deferred until a capture UI exists (docs/incremental_capture_design.md) — and
(2) a smaller or quantized ASR model, which is an ACCURACY decision, not a
latency one: any model swap must clear the frozen keyword/drug-WER gate first,
and the code-switch absorption failure above predicts exactly where a smaller
model will break.

---

## L4/L5 Defects Phase — Attribute before you fix (2026-07-10)

### (a) What this phase does
This phase took the three worst product-facing defects — missing symptoms and
vitals, generic words appearing as drug names on prescriptions, and
developer-notation warnings in the PDF — and attributed each failure to its true
layer before touching any code. The attribution changed the fix list: half the
"model failures" were the evaluation matcher, none were the PDF renderer, and
none were absent from the audio. Fixes then landed where the evidence pointed:
matcher synonym canons, a generic-term guard with dose/frequency preservation,
and plain-language footer sentences.

### (b) Hardest bugs

1. **The automated failure classifier called 6 of 7 cases "not in the
   transcript" — and manual review overturned every one of them.** Root cause:
   the transcript-scan matcher was synonym-blind. "Peripheral oxygen saturation"
   IS in the transcript — as "SpO2"; "136/88 mmHg" is there as "136" and "88"
   spoken separately. An automated attribution tool inherits every blind spot of
   the matcher it is built from, so its "not-a-bug" class — the one that closes
   issues — is exactly where its errors concentrate. The advisor-mandated manual
   spot-check of that class was the only thing standing between us and closing
   six real, fixable misses as dataset noise. Lesson: when a classifier's output
   decides what gets IGNORED, audit that class by hand, always.

2. **Flag strings collided when two medications had no name.** Converting
   generic drug mentions ("medicine", "दवाई") to "unnamed medication" made two
   such rows produce identical low-confidence flags — and the dedup logic
   silently dropped one, so a prescription with two unnamed drugs would warn the
   physician about only one. Root cause: flags were keyed by drug NAME, and the
   fix made names non-unique. Caught in advisor review before shipping; fixed by
   numbering ("unnamed medication 1", "unnamed medication 2"). Lesson: any
   transformation that maps distinct entities onto one label must check every
   downstream consumer that assumed the label was a key.

### (c) Fine-tuning hook
The corrected split (57% extraction / 43% matcher / 0% rendering) gives the
first clean per-layer error budget for symptoms and vitals. The extraction
misses cluster on a specific behavior — vitals spoken as ranges ("BP has been
130 to 140") that the model does not lift into the vitals list — which is a
concrete, few-shot-teachable pattern for either prompt work or a future
fine-tune, and the matcher canons mean any gain will now actually show up in
the score instead of being eaten by synonym mismatches.

---

## Frontend Phase — The glue layer is where safety signals die (2026-07-10)

### (a) What this phase does
This phase built the offline review frontend (record → review → edit → sign →
PDF) through coordinator-written contracts and parallel implementer subagents,
then verified the whole flow in a real browser against the real pipeline. The
verification pass — not the implementation pass — found the bugs that mattered:
a physician-facing "verify this" badge that never rendered, a status file that
intermittently returned garbage mid-write, and, upstream, an extraction prompt
that was quietly fabricating laboratory results.

### (b) Hardest bugs

1. **The extraction prompt's own examples became patient data.** Six of
   twenty-four frozen consultations contained lab values copied VERBATIM from
   the instruction prompt's illustrative examples ("Hb is 9.2", "HbA1c 9.1",
   "Total IGE 2107") into their notes as if measured. Root cause: a 3-billion-
   parameter model does not reliably distinguish "example of the format" from
   "content to reuse" — concrete values in instructions are training data for
   the very next generation. The fix (abstract shape descriptions, zero
   concrete values, a tripwire test on the prompt text) cost real recall
   (0.495 → 0.457 aggregate) — and part of that loss was itself revealing:
   the old diagnostic-results recall was partly CREDIT FOR FABRICATIONS that
   happened to match other samples' rubrics. Lesson: in any extraction prompt
   for a small model, every concrete value is a candidate hallucination; and a
   metric can be inflated by the very failure it should catch.

2. **The textbook defense that was structurally inert.** Whisper hallucinates
   text on silence (a 20-second silent WAV deterministically produces a
   Norwegian subtitle credit). The documented fix — vad_filter=True — was
   probed under a WER gate and found to do literally nothing in our pipeline:
   faster-whisper silently ignores the flag whenever clip_timestamps is set,
   and our per-segment decoding always sets it. Four independent confirmations
   (including byte-identical hallucinated output with the flag on and off).
   Lesson: a mitigation you haven't watched fire is a hypothesis, not a
   defense — the same lesson as multilingual=True in the latency phase, one
   layer deeper: this flag LOOKED shippable because it passed the WER gate
   with zero delta, and zero delta was precisely the proof it did nothing.

### (c) Fine-tuning hook
The review frontend now logs every physician correction as a structured diff
(field, old, new, timestamp) in outputs/<session>/corrections.jsonl. That file
is the future fine-tuning dataset this project has been missing: it captures
exactly the model-output → clinician-accepted-truth pairs, per field, in the
source language, with provenance back to the transcript — accumulating
passively during real use, at zero annotation cost.

---

## Deployment Latency Phase — Four negative results and the lever that remained (2026-07-11)

### (a) What this phase does
This phase attacked the deployment-killing fact that transcription runs ten to
twenty times slower than the audio it processes, under a hard rule: the
patient-safety keyword metric may not regress. Four escalating attempts to
speed up decoding — faster engines, merged decode windows, anti-hallucination
decode parameters, and voice-activity pre-slicing — were each measured on the
frozen bench and each failed the gate, together proving the slowness is not a
configuration problem. What shipped instead: honest progress ("42% · ~2 min
left") computed from real per-segment completion, and a live in-recording
transcript preview on the fast-but-ungated engine, walled off from all
clinical content by contract.

### (b) Hardest bugs

1. **The same clips broke every fast engine, and every knob, for the same
   reason.** Lighter decoders (large-v3-turbo, MLX builds) hallucinate
   repetition loops ("झाल झाल झाल…") on short or acoustically hard Hindi
   segments. Root-caused past the point of doubt: on the worst window the
   decoder escalated through its entire temperature ladder to 1.0 and still
   looped with high confidence (avg logprob −0.16) on audio that voice-activity
   detection confirms is 100% speech. Not silence-triggered, not
   threshold-fixable, not sampling-fixable — a property of the acoustic model
   itself on this audio class. And the structural alternatives are already
   closed: long windows absorb code-switched clinical terms into the dominant
   script (whole-file AND merged-window results, including the same-model
   control), so the slow per-segment baseline sits at the only measured point
   that protects drug-name accuracy. Lesson: when independent fixes keep
   failing on the same inputs, the inputs are telling you the failure lives
   below every layer you can configure — the honest next lever is the model,
   not another knob.

2. **A one-word async mistake froze the whole application, invisibly.** The
   live-preview route was declared `async def` with a blocking 10-second model
   decode inside — which stalls FastAPI's single event loop, freezing every
   concurrent request (including the status polling that tells the doctor the
   system is alive) for the duration of each preview. Nothing crashed; nothing
   logged; the app just went unresponsive in exactly the moments it was doing
   the most work. Found only because the debounce test could never produce a
   real overlap. Fix: a plain `def` route, which the framework dispatches to a
   worker thread. Lesson: in async servers, blocking work inside an async
   handler is a silent, total outage — and it is undetectable unless a test
   genuinely exercises concurrency.

### (c) Fine-tuning hook
The four negative results convert the ASR fine-tuning argument from "would be
nice" to "is the only remaining accuracy-safe speed lever." The target is now
precise: a distilled or fine-tuned model must fix confident repetition-loop
degeneration on short Hindi clinical segments — the bench clips that break
every stock engine (f2096fbd, 1ae62262) are the acceptance test, the frozen
bench is the gate, and the live-preview architecture means even a modest
fine-tuned model that clears the gate can slot into BOTH the preview and the
final pass, collapsing stop-to-note latency to near the extraction floor.

---

## Final Regression Phase — The bugs that only a full pass finds (2026-07-11)

### (a) What this phase does
Before pushing the frontend-hardening wave, one agent re-drove the entire
product in a real browser against the real pipeline — fresh upload, live
progress, review, provenance, editing, signing, tri-lingual PDFs, plus the
original Urdu-triggering audio re-processed end-to-end. Nine of eleven checks
passed; the two that failed were bugs no unit test and no per-fix verification
had caught, because each lived in the seam BETWEEN two things that were
individually correct.

### (b) Hardest bugs

1. **Two correct close paths, one shared stale state.** The provenance panel
   (per-field) and the transcript drawer (global) were synchronized on open
   but not on close: every drawer-close path reset the drawer's state but not
   the panels'. Result: after closing the drawer, re-clicking the same
   "source" button read its panel as already-open and silently did nothing —
   a dead button, no error, no log. Root cause: two UI elements representing
   one user-facing concept ("I am looking at this field's source") held
   independent state variables with no invariant tying them together. Lesson:
   when two widgets open together, closing either must reconcile both — and
   the test that finds this is "do it twice", which almost no one writes.

2. **The extractor re-spelled a drug the transcript had gotten right.** The
   transcript contained the Latin string "naxdom 500" verbatim; the 3B
   extraction model, generating its JSON, wrote the drug as a Devanagari
   spelling of its own invention — which matched no lexicon variant and
   failed CDSCO validation. Every upstream fix (script guard, curated
   spellings) was keyed to ASR-stage misspellings; nothing guaranteed the
   LLM preserves source spelling during generation. Fix: a deterministic
   source-fidelity backstop — if an extracted drug name has no fold-match in
   the transcript, substitute the transcript's own best-matching surface form
   (never a lexicon guess, so no wrong-drug risk), digit-preserving. Lesson:
   in an LLM pipeline, every generation step is a potential re-spelling of
   safety-critical tokens; fidelity to source must be enforced by
   deterministic code at the boundary, not assumed from instructions.

### (c) Fine-tuning hook
The restoration backstop doubles as a measurement instrument: every time it
fires, it logs a case where the extractor altered a safety-critical surface
form. Those logs, joined with corrections.jsonl, give a labeled dataset of
exactly the token-fidelity failures an extraction fine-tune (or a
constrained-decoding scheme locking drug spans to transcript substrings)
should be evaluated against.

---

## Drug Canonicalization Phase — Skeletons, crowded streets, and the audit the metric couldn't do (2026-07-11)

### (a) What this phase does
This phase replaced the losing game of hand-listing every possible drug
misspelling with a matcher that absorbs variants automatically: every drug
name (546 now, up from 163) is reduced to a phonetic "sound skeleton" —
script, vowel marks, doubling, and spacing stripped away, leaving the
consonant backbone that speech recognition reliably preserves — and incoming
words are matched skeleton-to-skeleton, with a bounded allowance for
distortion. The same phase built the safety apparatus that makes fuzzy
matching tolerable in medicine at all: an exhaustive substitution audit, a
length floor, and an ambiguity guard. It also hardened extraction so that a
drug name the model invents (like "nasal spray" for नैक्स्टोम) can never
render as a real drug.

### (b) Hardest bugs

1. **The matcher turned "two-three" into a drug, and the accuracy score
   could not see it.** With the initially specified tolerance (one letter of
   slack from skeleton length 5), the new tier converted दो तीन ("two-three")
   into Drotin, a fragment around "vital" into Revital, and a doctor's
   surname (Pandey) into Pan-D — three invented drugs. The headline accuracy
   number was byte-identical with and without them, because that metric only
   asks "how many REAL drugs did we find?" — inserting a wrong drug removes
   no real one. Root cause, in two layers: (i) metric blindness — recall
   cannot see false alarms; and (ii) the geometry of short words — the space
   of short skeletons is crowded (everyday words live there), so any short
   drug skeleton has innocent neighbors one step away, while long skeletons
   sit in nearly empty space where a near-miss is almost certainly the drug
   itself. The catch came from a dedicated substitution audit: run the whole
   test set with the new tier ON and OFF, diff to isolate every substitution
   the new code made, and adjudicate each one against the recording's
   human-written answer key (similarity explains why the machine guessed;
   only the answer key says whether the guess is TRUE). The fix was a rule,
   not a blacklist: raise the floor — no fuzzy matching below skeleton
   length 9, exact match only — accepting the loss of one correct short
   match (टिल्मा→Telma) to eliminate all three wrong ones, because the cost
   matrix is asymmetric: an unmatched drug shows as "unnamed — VERIFY" and
   costs the doctor five seconds; a wrong drug prints on a prescription.
   Re-audit after the fix: zero wrong substitutions across all 59 cached
   recordings. Second rail: if a skeleton sits within tolerance of TWO
   different drugs (albendazole/mebendazole, one letter apart, both real),
   the system refuses to choose and flags instead.

2. **The fast engine hallucinated fluent Portuguese on Hindi speech — and
   the repetition detector was structurally blind to it.** The fast-ASR
   gate run surfaced a hallucination class beyond the documented repetition
   loops: confident, fluent, wrong-language text ("Obrigada", Turkish,
   Indonesian) on Hindi clinical audio. It isn't repetitive, dense, or
   empty, so every lexical signal (compression ratio, n-gram repeats,
   chars/sec) passes it. Root cause of the blindness: the detector was
   built from the failure taxonomy we had OBSERVED, and a new failure class
   sat outside it. The mitigating fact discovered in the same run: the
   engine itself reports its detected language — the misdetection is
   sitting in the metadata, so an allowlist (we support exactly hi/en/mr)
   plus a forced-Hindi re-decode catches it deterministically, the same
   pattern that fixed the Urdu-script incident. Bonus root cause from the
   same gate: ~9-10s FIXED cost per decode call regardless of audio length
   (Whisper pads every input to a 30-second window), so latency scales
   with segment COUNT, not duration — the fix is fewer, fuller windows,
   not a faster model.

### (c) Fine-tuning hook
The substitution audit is reusable as a standing harness: any future change
to drug matching (or a fine-tuned extraction model) can be diff-audited the
same way — every changed drug decision logged and adjudicated against gold —
turning "did the score move?" into "show me every action you took."
Meanwhile the audit's false-alarm examples (दो तीन→Drotin class) are exactly
the hard negatives a learned drug-NER or constrained-decoding scheme should
train against, and the wrong-language hallucination clips join the
repetition-loop clips (f2096fbd, 1ae62262) in the acceptance set any
fine-tuned ASR must clear.

---

## Fast Transcription Phase — The 90-second constraint that redesigned the checker (2026-07-12)

### (a) What this phase does
This phase shipped the speed the clinic actually needs: the web app now
transcribes with the window-packed fast engine (a 28-second recording's
speech-to-text runs in ~16 seconds; the full note lands in 65-77 seconds),
protected by two guards proven earlier — the wrong-language allowlist and the
repetition-loop ladder — plus a background "second listen" that re-checks
only the safety-critical seconds of audio. The decisive design force was a
product constraint, not a technical one: appointments run 2-3 minutes, so a
checker that takes longer than ~90 seconds is furniture. That single sentence
from the user invalidated the obvious design (re-run the accurate engine on
everything: 20-60 minutes) and produced a better one.

### (b) Hardest bugs

1. **The obvious verification design was uselessly correct.** Re-transcribing
   the whole recording with the accurate engine gives the best possible
   second opinion — arriving half an hour after the patient has left. Root
   cause: engineering for maximum verification quality without pricing the
   clinical workflow's time budget; correctness that misses its deadline is
   indistinguishable from absence. The redesign inverts the question from
   "how do we verify everything?" to "what is worth verifying inside 90
   seconds?" — answer: the seconds of audio that produced drugs, doses,
   vitals, and diagnoses (the note's provenance already knows them), padded,
   merged, packed into one decode window, checked by a bigger decorrelated
   model, with a hard budget cap and an honest "partial check" label when
   spans overflow it. Measured: 57-59 seconds, full coverage, on the real
   incident session. Lesson: a verifier is a product feature with a latency
   SLO, not an offline benchmark — design it from the deadline backward.

2. **The checker's first real run flagged almost every diagnosis — because
   of our own annotations.** The pipeline glosses lay terms with clinical
   ones ("बुखार (Fever)"), so diagnosis VALUES carry a suffix that no raw
   re-decode of the audio will ever contain; fold-comparison saw permanent
   disagreement. Root cause: comparing a value from one representational
   layer (post-gloss) against text from another (raw decode) — the two sides
   of a diff must be brought to the same representation before comparing.
   Fixed by stripping the gloss suffix pre-comparison; caught only because
   the E2E ran on real session data rather than synthetic fixtures. Lesson:
   any diff across pipeline stages must normalize both sides to a common
   form first, and false-positive floods in a verifier are as damaging as
   misses — doctors stop reading flags that cry wolf.

### (c) Fine-tuning hook
Every background-check disagreement is now logged with both readings (fast
vs checked span text), and every "use checked version" tap is a
doctor-adjudicated label between two ASR hypotheses on identical audio —
accumulating exactly the preference data a future ASR fine-tune or reranker
needs, at zero annotation cost, in the deployment domain the frozen bench
can't represent.

---

## QA Hardening Phase — Category errors and the incident suite (2026-07-12)

### (a) What this phase does
This phase closed the user-reported defect round with a QA doctrine: every
real production failure becomes a permanent, model-free regression test
(tests/test_incidents.py), so no fixed bug can silently return. The marquee
fix: "my BP is high" had produced a MEDICATION row reading "(Hypertension)
भी" — a category error the grounding guard rightly passed (the string WAS
spoken; grounding catches inventions, not miscategorization). A deterministic
condition guard now drops any drug-field value that fold-matches the clinical
concepts table, with real-drug precedence proven collision-free across all
546 lexicon entries. The same round restored and expanded live progress to
all three pipeline stages, removed the note-appearance dead time, and pinned
the print dialog to the review page.

### (b) Hardest bugs

1. **Every guard was right, and the condition still landed in the Rx table.**
   The generic-term guard, grounding guard, and lexicon each did their job —
   none of them owns the question "is this string a DISEASE?". Root cause: a
   taxonomy gap between defenses, each built from a previous incident's shape
   (invented names, generic words, misspellings) — while the model found a
   fourth shape: correctly-transcribed, well-grounded, wrong CATEGORY.
   Defenses built from incident shapes will always trail the model's
   creativity by one shape; the countermeasure is a defense per FIELD
   SEMANTICS (what may a drug field contain?) rather than per failure story —
   plus the incident suite so each new shape is at least never repeated.

2. **The progress display existed, worked, and was never visible.** The fast
   engine reports progress once per decode window; short clinic clips pack
   into ONE window, so the single progress event raced the stage-end event
   that clears it — technically alive, observably dead (934 of 934 DOM
   samples empty). Root cause: an interface contract ("callback fires during
   the stage") that silently degenerated when the implementation's
   granularity (per-window) collapsed to one unit. The fix blends calibrated
   expectation (per-stage medians from real session history, shown instantly,
   capped at 95%) with true events wherever they exist. Lesson: progress
   reporting is a product surface with its own liveness requirement — test
   "is it VISIBLE at t=2s?", not "does the callback fire?".

### (c) Fine-tuning hook
The BP incident is the clearest argument yet for constrained extraction: the
model had the right information in the right fields (history captured "BP is
high") and STILL emitted a spurious medication row. A future fine-tune or
constrained-decoding scheme should be evaluated not just on recall but on a
category-confusion matrix (condition-in-Rx, drug-in-diagnosis, etc.) — and
tests/test_incidents.py plus the corrections flywheel now accumulate exactly
those labeled confusions from real use.

## Concept Matcher Rebuild Phase — The audit corpus picks the threshold (2026-07-12)

### (a) What this phase does
The concept matcher is the L3.5 pass that spots lay medical words in the
transcript ("sugar", "बीपी") and glosses them with the clinical concept they
mean ("Type 2 Diabetes Mellitus", "Hypertension") using embedding similarity —
a score of how close two phrases are in meaning. This phase transplanted the
drug matcher's near-collision disciplines onto that pass: a curated
everyday-word ban list (time words, numbers, fillers, verbs, kinship terms can
never gloss, whatever their score), a higher similarity bar for single-word
spans (the crowded, collision-prone class), and an ambiguity gate that refuses
to gloss a span sitting between two different concepts. Every threshold was
chosen from a hand-adjudicated audit of 173 real transcripts (90 distinct
single-word gloss candidates read in context), which cut wrong glosses from 17
to 4 while preserving all 10 must-keep true positives — बीपी, बुखार, खांसी,
शुगर and friends all still gloss.

### (b) Hardest bugs

1. **The guard we added last phase was silently vetoing a genuine symptom.**
   Last phase added हफ्ते ("week") as a hard negative on Shortness of Breath,
   to stop the one-edit collision with हांफते ("panting"). This phase's gate
   verification found the pre-existing test for the GENUINE case — "हांफ रहे
   हैं" (is panting) — already failing on the inherited code: the हफ्ते
   hard-negative embedding scored higher (0.88–0.94) against the panting
   phrase than the phrase's own concept match (0.82–0.88), so the guard built
   to protect the concept was suppressing its real mentions. Root cause: a
   hard negative added for a one-WORD collision was allowed to compete
   against spans of any length, and short common words sit so centrally in
   embedding space that they out-score specific multi-word phrases. Fix:
   a hard negative shorter than the query span is excluded from the veto —
   unigram-vs-unigram collisions (the case the guard exists for) are
   untouched. Lesson: every guard needs a regression test for the case it
   must NOT fire on, written the day the guard lands.

2. **The eleventh collision the manual audit missed.** The previous phase's
   audit hand-listed ten one-edit neighbor pairs (दमा↔दवा, हफते↔हांफते …).
   Rebuilding, we replaced the hand list with a programmatic scan: generate
   every concept variant's one-edit neighbors and check each against a common
   Hindi word list. It found दाद (ringworm) ↔ दान (donation) — a pair the
   manual pass never considered, even though दान was ALREADY a hard negative
   on a different concept (Skin Rash) for a different collision. Root cause:
   manual enumeration finds the collisions a human thinks to check;
   the space of one-edit neighbors is mechanical and should be searched
   mechanically. The scan now runs as a permanent test, so a newly added
   variant with a dangerous neighbor fails CI instead of shipping.

### (c) Fine-tuning hook
Every residual wrong gloss is the same failure: the matcher encodes each span
in isolation, so it cannot see that "acid" meant uric acid (a lab value), that
"blood" meant a blood TEST, or that "Tension" was part of "Tension headache" —
and no threshold separates these from real mentions, because the word alone IS
ambiguous. The audit artifact (outputs/gloss_audit.json — every candidate with
its context window, score, runner-up, and a hand adjudication) is exactly the
labeled dataset a context-aware fine-tune of parrotlet-e needs: train the
encoder to score span-IN-SENTENCE rather than span-alone, and this entire
residual class (including the accepted सर्दी winter/illness polysemy) becomes
learnable instead of unreachable.

## Drug Canonicalization Phase 2 — The spelling that lies, and one fold to rule them all (2026-07-12)

### (a) What this phase does
A real session prescribed नौरफलोक्स (norflox) and एजित्रोमाइसिन
(azithromycin), and the app failed to recognize either as a known drug — the
Rx showed Devanagari names marked "unvalidated", and the translation layer
later mangled them into different drugs entirely. This phase made the drug
fold script-symmetric: the fold (the function that squashes a word down to a
phonetic "sound skeleton" so Hindi and English spellings of the same drug can
be compared) now normalizes ENGLISH orthography too — x→ks, th→t, ph→f,
c→s before e/i/y else k, y→ai — because English spelling encodes sound
irregularly and skeletons previously met only when a drug's English spelling
happened to be phonetic (paracetamol matched; norflox never could). On top of
that, the note builder now displays the lexicon's canonical Latin name
whenever a drug resolves (dose digits preserved, CDSCO validation against
the canonical, a DISTINCT review flag when the match was fuzzy rather than
exact), so drug names default to English on the prescription.

### (b) Hardest bugs

1. **The skeletons were phonetic on one side only.** fold(नौरफलोक्स) gave
   "naurfloks" — roughly how it sounds — but fold("norflox") stayed
   "norflox", because the Latin side passed through nearly raw. "ks" vs "x"
   is obviously the same sound to a human and edit-distance-4 to the
   matcher. Root cause: the fold was built outward from Devanagari (map each
   letter to its sound) and nobody made the return journey — English
   orthography is itself a lossy encoding of sound (x=/ks/, soft c=/s/,
   th≈/t/) and needs its own normalization layer before the two scripts can
   meet. The giveaway pattern for this bug class: matching succeeds exactly
   when English spelling is phonetic and fails otherwise — a systematic
   asymmetry, not missing lexicon entries.

2. **Three hand-maintained copies of the fold, and the fix that would have
   re-created the bug downstream.** The fold existed in three modules
   (lexicon, L4 extractor, eval scorer), already drifted once before. The
   advisor caught the trap before implementation: patch only the lexicon's
   copy, and the L4 grounding guard — which re-folds every displayed drug
   name to verify it actually appears in the transcript — would fold
   "azithromycin" and the Devanagari transcript with the OLD rules, see low
   similarity, and demote a correctly-resolved drug to "unnamed medication".
   Root cause: duplicated safety-critical logic can't stay consistent by
   discipline alone; the fix and the guard must share one definition of
   "sounds like". The unification commit landed FIRST, and the blocker
   scenario is now a permanent regression test.

Two safety moments worth recording. The all-pairs collision scan over the
546-drug lexicon under the new fold surfaced exactly one new exact collision
— "vitamin c" and "vitamin k" (c→k!) — fixed with a lone-token-c exception;
and the fuzzy-floor length threshold was re-DERIVED (rules like th→t shorten
skeletons, shifting the length distribution it was tuned on), not just
re-run. And एजित्रोमाइसिन turned out to sit within tolerance of BOTH
azithromycin and erythromycin — two real antibiotics, genuinely ambiguous
because the Devanagari rendering drops the aspirate that separates them —
so the ambiguity guard refused, we did NOT loosen it, and the resolution
went into the curated exact-match table instead. Wrong drug is worse than
unknown drug, still.

### (c) Fine-tuning hook
The azithromycin/erythromycin ambiguity is an ASR-layer problem wearing a
matcher costume: the distinguishing phonetics exist in the audio but the
Devanagari transcription collapses them. An ASR fine-tune scored on drug
keywords should be evaluated specifically on aspirate/nukta preservation in
drug names. Separately, the hand-written orthography rules are a tiny
grapheme-to-phoneme model built from five incident classes — if the lexicon
grows past what hand rules cover, a learned G2P (or phonetic embeddings)
could replace them, gated by exactly the same collision-scan +
substitution-audit protocol this phase used.

## Translation Guard Phase — A prompt instruction is not a safety mechanism (2026-07-12)

### (a) What this phase does
The translation layer was the last unguarded door for drug names: the note on
disk stored the right drugs, but the LLM that translates the review screen
phonetically guessed Devanagari drug names into DIFFERENT English drugs —
एजित्रोमाइसिन (azithromycin) became "acetaminophen", and नौरफलोक्स (norflox,
an antibiotic) became "naloxone", an opioid-overdose drug. This phase wraps
every string sent to the translator in deterministic masking: drug spans are
found by fold-matching (against the note's own medication list and the full
lexicon), swapped for indexed placeholders the model must copy through
unchanged, and restored afterwards as canonical Latin names — with any string
whose placeholders don't survive intact falling back to its untranslated
original, because showing the source language beats showing a guess. The
same phase auto-warms all three language caches in the background the moment
the note is ready (language switching drops from minutes to milliseconds),
and an edit now drops its field from every warmed cache — so a doctor's
correction is shown verbatim in every language view, never overwritten by a
stale translation and never machine-translated itself.

### (b) Hardest bugs

1. **The protection existed, as a sentence.** The translator prompt already
   said "preserve Latin-script drug names verbatim" — and the incident drugs
   were Devanagari, so the sentence simply didn't apply, and a 3B model fills
   the gap with its best phonetic guess. Root cause: safety expressed as an
   INSTRUCTION to a model rather than a MECHANISM around it — an instruction
   has a scope you didn't enumerate (here: one script of two), degrades with
   model size, and fails silently. The replacement mechanism is enumerable
   end to end: find spans deterministically, verify each placeholder appears
   exactly once after translation (count-only checks pass when the model
   drops one and duplicates another), fall back per-string on any violation.
   Same family as the glue-layer lesson: note.json was CORRECT on disk; only
   the rendered translation lied, so no pipeline metric could have caught it.

2. **Globally ambiguous, locally certain.** The general lexicon correctly
   REFUSES एजित्रोमाइसिन (within edit tolerance of both azithromycin and
   erythromycin — the guard from the previous phase doing its job), which
   would have left the most dangerous span unmasked. The resolution: scope
   the candidate set — when matching against the note's OWN prescription
   list (three drugs, only one of them an -mycin), the same fuzzy match is
   unambiguous. Context shrinks ambiguity; the same string can be
   unresolvable against 546 candidates and certain against 3. The dual of
   this bug also appeared: a still-Devanagari medication value could
   fuzzy-shadow a better whole-lexicon match under longest-span-wins, fixed
   by letting only Latin (already-canonicalized) medication values join
   tier-1 matching — two detection tiers with different trust levels must
   not compete as equals.

### (c) Fine-tuning hook
Masking treats "drug span in Devanagari free text" as a permanent fact of
life; an ASR fine-tune that learns to emit drug names in Latin script at
transcription time (they are Latin-script brands being read aloud) would
shrink the masked surface toward zero and remove the cross-script fuzzy
step entirely. And the placeholder protocol doubles as an exact evaluation
metric for any future translator swap: placeholder survival rate per model —
a drug-name-safety bench that costs nothing beyond the runs themselves.

## Latency Phase 2 — Stop-to-Note Instrumentation: the sub-10s baseline (2026-07-20)

### (a) What this phase does
This phase does not change any model or pipeline stage — it adds the measurement
the rest of the sub-10s latency effort is gated on. `pipeline.run()` gains an
optional `stop_monotonic_ts` (the moment "recording stopped"; the web flow's
proxy is upload completion, since there is no live capture loop yet) and writes
`stop_to_note_s` / `stop_to_pdf_s` into `timings.json` alongside the existing
per-stage wall times. A new harness, `bench/stop_to_note_bench.py`, builds a
deterministic ~2.5-minute fixture (the project's three sample clips, looped
with 1s silence gaps, cached by content hash) and runs the real production
pipeline (`asr_engine="fast"`) N times, reporting p50/p95 against the <10s
acceptance target. The honest baseline this phase establishes: **p50 = 348.9s,
p95 = 365.7s** (N=3) — a FAIL, exactly as expected, since none of the
optimization waves (residency, incremental capture, L4 fast path) have shipped
yet. The number this phase's harness will be re-run against as each wave lands.

### (b) Hardest bugs

1. **Ollama was running 100% CPU (~1 token/sec) — a root cause, not a fluke,
   and one this project already hit once before without diagnosing it.**
   §8e of an earlier working handoff notes "Ollama appeared CPU-only (~1 tok/s)
   during wave-B E2E — verify the running ollama is the native arm64 build next
   session," but that session moved on without confirming why. This time: a
   30s smoke clip's L4 stage took 372s (should be ~13-28s), and `ollama ps`
   showed `100% CPU`. The server log's own GPU discovery line was the proof —
   `msg="discovering available GPUs..."` → `msg="inference compute" id=cpu
   library=cpu` — Metal was never even detected as a candidate device, not
   merely unused. Root cause, precisely: this machine's `ollama` was installed
   via `brew install ollama` (the Homebrew-core formula), which packages the
   CLI-only, CPU-only build — a long-standing Homebrew-core packaging gap, not
   an architecture mismatch (the binary WAS native arm64; "verify arm64" in
   the earlier note was the wrong diagnostic question). The fix is a different
   package entirely: `brew install --cask ollama-app` installs the official
   Ollama.app bundle (Metal-enabled `llama-server` backend), which the CLI
   formula does not provide no matter how it's reinstalled or upgraded.
   Verified after the fix: `ollama ps` reports `100% GPU`, and a real L4 call
   on this fixture landed at 28.48s — matching the historical "cold 28.5s"
   figure almost exactly. **Anyone resuming this project on a fresh machine
   must install the `ollama-app` cask, not the `ollama` formula** — this is
   now the second time the CLI-only formula silently produced a 40-70x latency
   regression that looked like a model or pipeline problem until the GPU
   discovery log was actually read.

2. **A benchmark fixture built by looping identical clips manufactures its
   own worst case, and it does so deterministically enough to look like a
   real finding if you don't check.** The first N=3 baseline run showed L3 ASR
   at ~241s median for a 150s clip — 25-30x the ~7-9s/window fixed cost the
   fast engine is supposed to have. The degeneration retry ladder
   (`src.fast_asr`) fired at exactly the same two window boundaries in all
   three runs — `[0.03, 23.25]s` and `[118.17, 142.88]s` — both of which sit
   at the fixture's loop-splice points (silence-gap joins between repeated
   copies of the same three sample clips). One of the two exhausts the ladder
   entirely (`step4_giveup`, its most expensive rung). Root cause: the
   degeneration detector is tuned to catch the model repeating itself — and a
   fixture built by literally repeating the same audio at fixed intervals is
   exactly the input that heuristic is designed to flag, whether or not the
   underlying decode would have been fine on natural, non-repeating speech at
   that duration. This means **the L3 ASR component of this baseline is not a
   trustworthy per-window latency number** — it is inflated by an artifact of
   how the fixture was built, not a property of the production decode path.
   The other stages' numbers (L2 diarize, L3.5 normalize, L4 extract, L5
   render) do not share this defect since they don't re-trigger per-window
   retry logic on loop boundaries the same way.

### (c) Fine-tuning hook
Not applicable in the ML sense — this phase changed no model. The forward
hook is methodological: **do not accept this baseline's L3 ASR figure, or any
future bench run on this same looped fixture, as the acceptance number for
Wave 3's incremental-capture work.** The fixture needs non-repeating audio of
target duration before it can honestly gate a latency claim — either more
distinct EkaCare clips concatenated (no repeats), or, better, the
deployment-mic recordings HANDOFF's open item 0b already calls out as needed
(the frozen bench is phone-quality EkaCare audio and has previously been shown
to diverge from clinic-mic behavior; the same domain gap likely applies to
retry-ladder trigger rates, not just WER). Until then, treat L2/L3.5/L4/L5's
baseline figures as load-bearing and L3's as a known-inflated placeholder.

## Latency Phase 3 — Model Residency: the CLAUDE.md carve-out, spent on the right thing (2026-07-20)

### (a) What this phase does
This phase amends CLAUDE.md's "load one model, release it — never hold ASR
and LLM resident simultaneously" rule for the fast engine only (user-approved
carve-out, recorded in CLAUDE.md itself): `src/model_registry.py` keeps
pyannote's diarization `Pipeline` and parrotlet's `_EmbeddingBackend` loaded
across sessions within one server process, instead of `diarize()`/`normalize()`
each loading and releasing their model on every single call. `src/pipeline.py`
fetches both from the registry and passes them through (`pipeline=`, `backend=`
kwargs both already existed as optional params on `diarize()`; `normalize()`
gained one) whenever `asr_engine="fast"` and `config.FAST_ENGINE_RESIDENT` is
True — the accurate CLI path is untouched regardless of the flag. A second,
independent change lands in the same phase: `l3_5_normalize._encode_reference_matrices()`
re-encodes every CONCEPTS canonical term/variant/hard-negative through the
model on EVERY call, even though that content never changes between sessions —
this is now served from a content-addressed disk cache
(`_get_reference_matrices`, keyed by MODEL_ID + a hash of the CONCEPTS table,
so it self-invalidates on any concept edit) unconditionally, independent of
the residency flag. mlx-whisper needed no new code at all — `mlx_whisper.
transcribe.ModelHolder` already caches at module scope for free; the only
change was to stop `web/app.py`'s `create_session()` from deliberately
clearing that cache on every upload when resident mode is active.

**Measured, on the real pipeline (fast engine, 30s fixture, two consecutive
sessions in one process — see LEARNINGS' own verification script, not
committed):**

| Stage | Before (Wave-1 baseline, same fixture) | After: 1st call (cold) | After: 2nd call (warm) |
|---|---|---|---|
| L3.5 normalize | 21.8–25.4s | 3.39s | **2.77s** |
| stop_to_note_s | 123.2s | 89.9s | **68.6s** |

L3.5 normalize dropped **~8x** — the single largest lever in the whole
latency plan, exactly matching an independent, much earlier finding already
sitting in this project's own (gitignored) working handoff: "L3.5 latency
(~19s) is now the biggest stop-to-review item — parrotlet model residency /
preload during ASR would cut most of it." That note was written 2026-07-12
and sat unactioned for over a week; this phase is the fix it was asking for.
L2 diarize's per-stage number (7.6–8.9s) is **not directly comparable** to
pre-Wave-2 numbers: `model_registry.get_pyannote_pipeline()` is called
*before* the L2 stage timer starts, so model-load time that used to be
attributed to the `l2_diarize` bucket is now attributed to nothing (absorbed
into `stop_to_note_s` but invisible in the per-stage breakdown) on a cold
call, and genuinely doesn't exist on a warm one. Anyone diffing per-stage
JSON across this phase boundary needs to know that, or L2 will look like it
regressed when it didn't.

### (b) Hardest bugs

1. **The residency wiring silently broke the "stubbed stages, no models
   loaded" invariant of the existing pipeline test suite — not because the
   stub functions had the wrong signature, but because a THIRD, unstubbed
   call path was introduced.** `pipeline.run()`'s new resident-kwargs
   construction (`model_registry.get_pyannote_pipeline()` /
   `get_embedding_backend()`) happens *before* `diarize()`/`normalize()` are
   even invoked — so `tests/test_pipeline.py`'s `stubbed_stages` fixture,
   which monkeypatches `pipeline.diarize` and `pipeline.normalize` directly,
   did nothing to prevent a REAL pyannote HF download attempt from firing
   inside a supposedly fully-mocked unit test (visible as a real HTTP call in
   the failing test's captured stderr). Root cause: residency added a second,
   independent way to reach the real model-loading code, and only one of the
   two paths was covered by the existing stub discipline. Fix: the fixture
   now also stubs `pipeline.model_registry.get_pyannote_pipeline` and
   `get_embedding_backend`, mirroring how every other stage is already
   stubbed — the general lesson being that adding a new call site to
   already-mocked functionality means finding and updating every fixture
   that mocks the OLD call site, not just checking the new code compiles.

2. **The Ollama Metal defect from Latency Phase 2 was not a one-off — it
   directly gated whether this phase's residency numbers could be trusted at
   all, and very nearly produced a second round of the same mistake.** The
   real two-session verification run for this phase depends on L4 extract
   being fast (otherwise it would dominate and mask the L3.5 signal this
   phase is actually testing). Because the `ollama-app` cask fix from Phase 2
   was already in place, L4 measured 23.44s cold / 14.45s warm here — sane
   numbers that didn't need re-diagnosing. This is recorded not because
   anything new broke, but because it is the second LEARNINGS entry in two
   phases where an unrelated-looking latency phase's validity turned out to
   depend on that fix being present — worth flagging for whoever next resets
   this environment.

### (c) Fine-tuning hook
Not applicable — no model changed. The forward hook: the reference-matrices
disk cache and backend residency are independent levers, and this phase
proves the disk cache alone (which requires no held-open device memory, no
CLAUDE.md carve-out, and works even on the accurate/CPU path) already
captures most of the win — re-encoding hundreds of reference texts was the
dominant cost, not the model weights themselves being reloaded. If a future
change needs to claw back memory budget, dropping backend residency while
keeping the reference-matrix cache is a safe, low-cost fallback that gives up
only a fraction of this phase's gain, not all of it.

## Latency Phase 4 — Incremental Capture: a backend primitive, and two bugs the equivalence test earned its keep on (2026-07-20)

### (a) What this phase does
`web/incremental.py`'s `IncrementalSession` (`feed` / `partial_turns` /
`finalize`) implements docs/incremental_capture_design.md's L1+L2+L3 backend
with one deliberate departure from that design: instead of one whole-file
re-decode at stop, settled windows are decoded once during `feed()` calls
and FROZEN — production's fast-path decode is deterministic
(`temperature=0.0`, no conditioning), so re-decoding identical audio would
reproduce identical text, making a whole-file re-decode provably redundant
work. `finalize()` decodes only the un-settled tail, concurrently with one
final full-file pyannote pass (two threads, joined) — the actual mechanism
behind collapsing stop-time cost to roughly `max(tail decode, final
diarize)`. A new segment-free windowing primitive
(`src.fast_asr.pack_duration_into_windows` + `decode_windows_words`) makes
this possible: diarization is only PROVISIONAL during live capture and keeps
changing, so windows are cut by fixed duration instead of diarized natural
breaks, relying on the fact that word-to-speaker attribution was already
decoupled from decode-window boundaries (`_assign_words_to_turns`) before
this phase touched anything. This wave is deliberately scoped to the backend
primitive only — no FastAPI routes or browser capture-loop wiring, since
nothing exists yet to drive them (see docs/incremental_capture_design.md's
"Preconditions before HTTP/browser wiring").

### (b) Hardest bugs

1. **A benchmark-quality real-audio equivalence test caught a real design
   flaw that every unit test with a fake decoder had no way to see.** The
   first version of `_decode_newly_settled_locked` decoded on every single
   `feed()` tick, however little new audio had settled. On an 11.3s test
   clip with a 4s settle margin, this produced 2-5 SECOND decode windows —
   and the real-audio equivalence test's word-overlap against the batch path
   collapsed to 31% (batch words were genuinely different content:
   `{medicines, put, giving, morning...}` vs incremental's `{argumenting,
   subhash, reasons...}` on the same audio). Root cause: firing a decode call
   for every ~10s tick interval creates MORE, SMALLER mlx_whisper.transcribe()
   calls than the batch path's ~28s-capped windows — directly working
   against Fix 2's whole rationale (src/fast_asr.py's module docstring: a
   decode call costs a near-constant ~7-9s regardless of how much audio it
   covers, so fewer/bigger calls is the entire point of windowing). Every
   pure-Python unit test with a faked `decode_windows_words` passed
   throughout, because none of them could observe that windows were
   pathologically small — only feeding real audio through the real decode
   path and comparing against a real reference exposed it. Fixed by
   accumulating settled-but-undecoded backlog across `feed()` calls until it
   holds at least one full `max_window_s` chunk, packing as many complete
   windows as the backlog supports in ONE `decode_windows_words` call, and
   leaving any sub-window remainder for the next tick — `finalize()`'s tail
   decode is exempt from this gate since no more audio is coming. Overlap
   jumped to 69% on a properly-proportioned second test (realistic ~70s
   fixture, production settle margin, ~25s feed ticks) — the residual gap is
   the ACCEPTED, DESIGNED-FOR divergence between incremental's fixed
   28s-boundary cuts and batch's diarization-natural-break cuts (documented
   in the original latency plan's risk register), not a defect.

2. **`src/model_registry.py` depended on a side effect from a module it
   might never be imported alongside.** `get_pyannote_pipeline()` needs
   `HF_TOKEN`, which `src/pipeline.py` loads via `load_dotenv()` at import
   time — but `model_registry.py` itself never called `load_dotenv()`, so it
   silently relied on SOME other already-imported module having done so
   first. The real-audio equivalence test imports `web.incremental` directly
   (never `src.pipeline`), so `HF_TOKEN` was genuinely absent from
   `os.environ` when `finalize()`'s background diarize thread ran, surfacing
   as `OSError: HF_TOKEN not set` inside a `PytestUnhandledThreadExceptionWarning`
   rather than a clean top-level failure (background-thread exceptions don't
   propagate to the caller — another reason this class of bug hides easily).
   Root cause: an import-order dependency masquerading as "it just works" —
   any module that NEEDS an environment variable should load it itself,
   not assume a sibling module's import already did. Fixed by adding
   `load_dotenv()` directly to `model_registry.py`, mirroring
   `src/pipeline.py`'s own pattern; the module is now self-sufficient
   regardless of what else has been imported.

### (c) Fine-tuning hook
Not applicable to model weights — this phase is architecture, not a model
change. The forward hook is about test methodology: bug #1 was invisible to
every fast unit test and only surfaced through a slow, real-model
equivalence test — a concrete argument for keeping at least one real-audio
integration test per major architectural change, even though (per this
project's own testing discipline) the default suite must stay fast and
fake-decoder-based. The two-tier pattern used here — extensive fast unit
tests for logic, one marked-slow real-audio test for the assembly — is the
template worth reusing for Wave 5/6's ASR-adjacent changes, which carry the
same class of risk (a fake decoder cannot see a pathological windowing
decision, only a real one can).

## Latency Phase 5 — L4 Compact Output: a real gate result, and why it stays off (2026-07-21)

### (a) What this phase does
Adds a compact prompt variant (`_SYSTEM_PROMPT_COMPACT` — `_SYSTEM_PROMPT`
plus one rule instructing the model to omit any null/empty field rather than
emit it explicitly) and `config.EXTRACT_COMPACT_OUTPUT` to switch `extract()`
between them; `_build_note` needs no change since it already treats a
missing key the same as an explicit null via `data.get(field) or default`.
Also adds L4 KV-prefix warming: `src.l4_extract.warm_llm_prefix()` fires the
exact system+transcript prompt `extract()` will eventually send, with
minimal generation, so Ollama's KV cache is primed before the real call;
`web/incremental.py`'s `IncrementalSession.warm_l4_prefix()` calls it with
turns settled so far (reusing `partial_turns()`'s diarization pass, not a
second one) and tracks `kv_warm_wasted` — incremented whenever `finalize()`'s
actual prompt hash doesn't match the last warm's (never a correctness bug,
since `extract()` always sends the right prompt regardless — just a missed
optimization). `prompt_prefix_hash()` is shared by both extract() and the
warm path so the two can never silently drift on prompt construction.

**Gate result on the frozen extraction eval (`eval/frozen_set_extraction.json`,
24 samples, 630 representable rubric criteria) — real, and not a clean pass:**

| | Verbose (baseline) | Compact |
|---|---|---|
| Aggregate recall | 0.481 (303/630) | 0.479 (302/630) |
| medication_name | 66/104 | 70/104 |
| medication_frequency | 55/86 | 51/86 |
| prescribed_test_name | 13/37 | 18/37 |
| symptom_name | 55/87 | 46/87 |

The aggregate is a wash (630 criteria, a 1-criterion difference). Per-category,
`symptom_name` drops 9 points (10.3% relative) while `medication_name` and
`prescribed_test_name` each gain — not uniform non-regression. Inspecting the
actual extracted symptom strings per sample shows this is mostly generic
3B-instruct prompt-sensitivity (case changes, "Upper abdominal pain" →
"Abdominal pain", extra symptoms caught in some rows, one real semantic miss
at idx=140 — "pain in the back" → "Pain in the chest") rather than a
mechanism specific to omitting empty fields; there is no design reason field
omission would selectively hurt symptom extraction. **Decision: ship the
capability, fully built and tested, with `EXTRACT_COMPACT_OUTPUT = False` by
default.** At N=24 this evidence cannot distinguish "compact output is
accuracy-neutral" from "compact output trades some symptom recall for some
medication/test recall" — and flipping a default that touches symptom
extraction, a clinically load-bearing field, on evidence this ambiguous is
not a call to make unilaterally. This is the same "measure before tune"
discipline the project has applied to every other latency lever — a mixed
gate result is a real finding, documented like a win, not quietly rounded up
to a pass.

### (b) Hardest bugs
No implementation bugs this phase — the gate machinery (compact prompt,
`prompt_prefix_hash`, `warm_llm_prefix`, the `--compact` eval CLI flag) all
worked correctly on the first real run. The "bug," such as it is, was almost
reaching for the aggregate number alone and calling it a pass — 0.481 vs
0.479 looks like textbook non-regression until the per-category breakdown is
read, which is where the real signal (and the real question about whether to
ship) actually lives. Worth recording as a process note: an aggregate gate
number over a small, heterogeneous category mix can hide exactly the kind of
per-field tradeoff that matters most in a clinical-safety context.

### (c) Fine-tuning hook
Not applicable to model weights. The forward hook: if compact output is
revisited, the right next step is a LARGER frozen sample (N=24 is too small
to separate "real effect" from "3B-model noise" on any single category with
~40-90 criteria) or a repeated-run variance study (same prompt, multiple
temperature=0 calls, to establish how much category-level recall naturally
wobbles run-to-run before attributing a shift to the prompt change at all).
Until then, the compact prompt and `EXTRACT_COMPACT_NUM_PREDICT` stay
available behind the flag for anyone who wants to trade the ambiguity for
the latency win in a specific deployment, but production defaults to the
verbose prompt this project's existing eval baselines were measured against.
