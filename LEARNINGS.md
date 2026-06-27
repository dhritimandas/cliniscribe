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
