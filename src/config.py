"""Central configuration: tunables shared by the pipeline and the eval harness.

Every value here is a measured decision, not a default — see LEARNINGS.md for
the study behind each. Eval and production MUST read the same values so that
benchmark numbers describe the system as shipped.
"""

# ── Model identifiers (one per stage) ────────────────────────────────────────
ASR_MODEL = "large-v3"  # L3 — faster-whisper
DIARIZE_MODEL = "pyannote/speaker-diarization-community-1"  # L2 — needs HF token + T&C
NORMALIZE_MODEL = "ekacare/parrotlet-e"  # L3.5 — concept embeddings
EXTRACT_MODEL = "qwen2.5:3b-instruct"  # L4 — via Ollama

# ── L3 ASR decoding ──────────────────────────────────────────────────────────
# Beam width reconciled by eval/beam_study.py on the frozen 10-clip Hindi bench
# (drug/keyword WER, token-boundary scorer): beam=5 keyword WER 0.5714 vs
# beam=1 0.7381 — see outputs/beam_study.json for the measurement.
ASR_BEAM_SIZE = 5

# NOT a working silence-hallucination defense — kept False. faster-whisper's
# own docs (transcribe.py: "vad_filter will be ignored if clip_timestamps is
# used") mean this flag is a no-op at our call site, which always passes
# clip_timestamps for per-segment decoding. Measured on the frozen bench
# (eval/vad_study.py) and reproduced directly against faster-whisper: with an
# explicit clip_timestamps, vad_filter=True does not change transcribe()'s
# output at all, byte-for-byte, including on 20s of pure silence that Silero
# VAD itself correctly flags as speech-free (see
# tests/test_l3_asr.py::test_vad_filter_is_a_noop_on_silence_with_clip_timestamps).
# Do not flip this to True expecting protection — it does nothing while
# per-segment clip_timestamps decoding is in place.
ASR_VAD_FILTER = False

# Whisper's per-segment language auto-detection (language=None) occasionally
# misclassifies spoken Hindi as Urdu and writes the segment in Arabic script
# (observed in production session 20260710-230150-cef13a: "نیکس ڈوم فائیو
# ہنڈریڈ" for "Naxdom five hundred"). We only support hi/en/mr — all Latin or
# Devanagari — so any Arabic-script output is always a misdetection. When
# True, transcribe() re-decodes the affected segment once with language="hi"
# forced. See src/l3_asr.py::_contains_arabic_script.
ASR_SCRIPT_GUARD = True

# ── L3.5 concept matching (E5 study: hard-negative gate) ─────────────────────
COSINE_THRESHOLD = 0.65  # min cosine similarity for a lay→clinical concept match
HARDNEG_MARGIN = 0.05  # span must beat its hardest hard-negative by this margin

# ── L3.5 concept matching — near-collision hardening (Concept Matcher
# Rebuild Phase, tuned by eval/gloss_audit.py against the frozen 24-sample
# extraction transcripts + 49-clip drug bench + 10-clip beam study + real
# session transcripts — 173 items, 90 distinct unigram gloss candidates
# adjudicated by hand; see the tuning table + adjudication in the phase
# report). Transplants the drug matcher's length-floor and ambiguity-guard
# disciplines onto the concept pass:
#   - Unigram spans are the crowded, one-edit-collision-prone class (हफते↔
#     हांफते, दमा↔दवा, ...), so they must clear a HIGHER bar than bigram+
#     spans, which get contextual support from a neighbor word. Empirically,
#     every CLEAR wrong unigram gloss found in the audit (योर->Fever, ब्लड/
#     blood->Hypertension, BPM->Hypertension, DM->T2DM [a drug-name
#     fragment, "Ompec DM"], D.->Diarrhea [a list marker "B, C, and D."],
#     वाइटिंग->Vomiting, बसले/संपल्या->Weakness, flu->Fever, stools/
#     गैप/acid->Acid Reflux) scored <= 0.721; the first preserved true
#     positive (बीपी->Hypertension) scores 0.733 — 0.73 sits in that gap.
#     A residual class of 4 unigrams (जळत->Acid Reflux, खीस->Cough,
#     bare संपल्या->Weakness, Tension->Anxiety) still gloss wrong at
#     0.78-0.85; pushing the bar there would ALSO drop dozens of legitimate
#     matches in the same band (कमजोर, कफ, दर्द-family, डायबिटीज-family,
#     "weakness.", "ache.", ...) — documented residual risk, same class as
#     the सर्दी policy (span-only embeddings can't see sentence context).
COSINE_THRESHOLD_UNIGRAM = 0.73
#   - If the best- and second-best-scoring CONCEPTS for a span are within
#     this margin, the span is genuinely ambiguous between two concepts and
#     must not be glossed (mirrors src/drug_lexicon.py's ambiguity guard:
#     "if a skeleton sits within tolerance of TWO drugs, refuse to choose").
#     0.03 causes zero regressions on the audit corpus (its smallest
#     observed margin for any accepted gloss is 0.065, a Migraine-vs-
#     Headache case already resolved by other means) — kept as prophylactic
#     defense-in-depth, not because it fired here, exactly like
#     src/drug_lexicon.py's albendazole/mebendazole tie-breaker.
CONCEPT_AMBIGUITY_MARGIN = 0.03

# ── L4 extraction context ─────────────────────────────────────────────────────
# Ollama's default context truncated long HI/MR transcripts and produced empty
# notes: a 5.4k-char Devanagari transcript measures 5006 prompt tokens (bench
# max 9.4k chars ≈ ~9k tokens), so 8192 is not enough headroom. Measured in
# outputs/l4_ctx_diag.json: idx-74 yields {} at default ctx, a full note at 8192.
EXTRACT_NUM_CTX = 16384
# Bound generation: degenerate unbounded JSON outputs were observed running
# 40+ min on CPU during diagnosis; a full note is < ~800 tokens.
EXTRACT_NUM_PREDICT = 2048
# Keep Qwen resident between calls (consecutive consultations in a clinic
# session) instead of Ollama's default unload; also enables the L3.5-time
# preload. Whisper is always released before the LLM loads (pipeline order).
EXTRACT_KEEP_ALIVE = "15m"

# ── L3-fast ASR (mlx-whisper, shipped) ───────────────────────────────────────
# Chunk-local detect+retry engine (src/fast_asr.py), built to answer the
# Deployment Latency Phase's four negative results (see LEARNINGS.md): faster
# engines/merged windows/decode-param tweaks/VAD pre-slicing all failed the
# frozen-bench gate outright. eval/fast_asr_gate.py's frozen 10-clip bench
# (out-of-domain phone-call-style Hindi) never reported a PASS for either
# engine config (outputs/fast_asr_gate_v2.json: both FAIL). The shipping
# decision overrides that gate on the strength of a DIFFERENT measurement in
# the same file — the "windowed" engine's numbers on the three actual
# deployment-style clinic clips: ~15-20s decode (vs. the accurate path's
# multi-minute wall) and improved drug capture post-normalize (2/3 clips,
# including the naxdom session, vs. 1/3 for per-segment) — the frozen bench's
# corpus_wer regression traces to whole-clip windows drifting into English
# paraphrase on code-switched Hindi, a failure mode the background
# verification pass below (web/verification.py) exists specifically to catch
# on safety-critical fields without paying a full accurate re-transcription's
# cost. Production always uses fast_transcribe_windowed for this reason (see
# src/pipeline.py's asr_engine="fast" branch) regardless of FAST_ASR_MODE
# below, which predates this decision and is now inert.
FAST_ASR_ENABLED = True

FAST_ASR_MODEL = "mlx-community/whisper-large-v3-turbo"  # per-segment primary decode
FAST_ASR_FALLBACK_MODEL = "mlx-community/whisper-large-v3-mlx"  # retry-ladder step3
# temperature=0.0 (scalar, not the 6-rung fallback ladder) + no conditioning:
# measured no worse than the stock ladder on the known-bad clips (both still
# loop either way — see eval/engine_study.py probe_antihallu), and faster.
FAST_ASR_DECODE_KWARGS: dict = {"temperature": 0.0, "condition_on_previous_text": False}

# Degeneration-detector thresholds (src/fast_asr.py::looks_degenerate), tuned
# against the real repetition-loop hallucinations cached in
# outputs/engine_study.json ("college college college...", "झाल झाल झाल...")
# with zero false positives across all 20 fw-large-v3 hypotheses in
# outputs/beam_study.json (see tests/test_fast_asr.py for the sweep).
DEGEN_COMPRESSION_RATIO = 2.4  # zlib text-compression ratio above this = looping
DEGEN_MIN_LEN_FOR_RATIO = 20  # chars; below this, zlib overhead makes the ratio noisy
DEGEN_NGRAM_SIZES = (1, 2, 3, 4)  # phrase lengths checked for back-to-back repetition
DEGEN_NGRAM_MIN_REPEAT = 4  # same phrase repeated >= this many times consecutively
DEGEN_MAX_CHARS_PER_SECOND = 30.0  # implausible decode density for hi/en/mr speech
DEGEN_EMPTY_ON_VOICED_MIN_S = 1.0  # empty decode on a segment this long+ is suspect

# Retry-ladder step knobs (src/fast_asr.py::_retry_ladder). Step1's boundary
# shift is a measured loop-breaker; step2's temperature/conditioning change is
# the other cheap lever tried before escalating to a bigger model in step3.
RETRY_BOUNDARY_SHIFT_S = 0.4
RETRY_TEMP_STEP2 = 0.2

# ── L3-fast v2, Fix 1: language-allowlist guard ──────────────────────────────
# mlx-whisper's per-window language auto-detection occasionally locks onto a
# WRONG but fluent language -- not a repetition loop, so looks_degenerate()
# cannot see it. Observed on the v1 frozen-bench gate (outputs/fast_asr_gate.json):
# "Obrigada" (Portuguese), "işte sulta" / "Bu, sayıda" (Turkish), "saya
# menikmati" (Indonesian) decoded confidently over Hindi speech. We only
# support hi/en/mr; any other detected language token is always a
# misdetection. Checked on every decode, belt-and-braces alongside
# ASR_SCRIPT_GUARD (which only catches the Arabic-script case) -- see
# src/fast_asr.py::_decode_with_script_guard.
ASR_LANGUAGE_ALLOWLIST = frozenset({"hi", "en", "mr"})

# ── L3-fast v2, Fix 2: window-packed decoding ────────────────────────────────
# Measured root cause of the ~10-20x-realtime slowdown: mlx-whisper pads every
# input to a 30s window before encoding regardless of content length (its
# transcribe() always calls pad_or_trim(mel, N_FRAMES, ...) where N_FRAMES is
# a fixed 30s), so one decode call costs a near-constant ~7-9s whether it
# decodes 2s or 25s of audio -- cost is per WINDOW, not per second (measured
# directly on warm calls; see LEARNINGS.md). fast_transcribe_windowed() packs
# diarized segments into <= WINDOW_MAX_SPAN_S windows and decodes each ONCE,
# cutting call count from one-per-segment to one-per-window.
WINDOW_MAX_SPAN_S = 28.0
# Minimum silence gap -- or any speaker change -- preferred as a window break
# point when a window must close before reaching the cap (see
# src/fast_asr.py::_pack_segments_into_windows).
WINDOW_MIN_BREAK_GAP_S = 0.8

# ── Background verification (targeted second listen, web/verification.py) ───
# After a fast-engine review reaches the doctor, a daemon thread re-decodes
# ONLY safety-critical spans (drug names+doses, vitals, diagnosis) — not a
# full accurate re-transcription, which costs 20-60 min on CPU for a
# multi-minute consult, far past the budget below.
VERIFY_MODEL = FAST_ASR_FALLBACK_MODEL  # bigger mlx model, decorrelated from the primary turbo decode
VERIFY_SPAN_PAD_S = 1.5  # seconds of context padded on each side of a field's transcript mention
VERIFY_MAX_WINDOW_S = 28.0  # per-window cap, same as WINDOW_MAX_SPAN_S
VERIFY_BUDGET_S = 90.0  # hard wall-clock budget for the whole verification pass
VERIFY_DECODE_S_PER_WINDOW_ESTIMATE = 48.0  # measured: one full 28s window on VERIFY_MODEL
# A second window (56s of audio, ~2x the decode cost) is only attempted when
# its ESTIMATED cost still fits the budget below. With the measured
# per-window cost above, 2 windows (~96s) does not, so verification currently
# always resolves to a single packed window; written generically so a faster
# model or a larger budget could unlock a second window later.
VERIFY_MAX_WINDOWS = 2 if 2 * VERIFY_DECODE_S_PER_WINDOW_ESTIMATE <= VERIFY_BUDGET_S else 1

# Which fast-ASR entry point to use once FAST_ASR_ENABLED flips True: either
# "per_segment" (src.fast_asr.fast_transcribe) or "windowed"
# (src.fast_asr.fast_transcribe_windowed) -- whichever passes
# eval/fast_asr_gate.py's frozen-bench gate fastest. NEITHER passed the v2
# gate (outputs/fast_asr_gate_v2.json, both engines run 2026-07-11):
# per_segment  corpus_wer 0.6613 > 0.5388, keyword_wer 0.6429 > 0.5814
#              (drug_wer_folded 0.5714 <= 0.8571 -- the only metric that passed)
# windowed     corpus_wer 0.9059 > 0.5388, keyword_wer 0.6905 > 0.5814
#              (drug_wer_folded 0.7143 <= 0.8571 -- also passed)
# windowed's much worse corpus_wer traces to whole-clip windows drifting into
# English PARAPHRASE/translation of code-switched Hindi speech rather than
# transcription (e.g. "सर दर्द" -> "How severe is your heart?"), a distinct
# and worse failure mode than the anticipated Devanagari-absorption risk (zero
# absorption candidates were found -- see eval/fast_asr_gate.py). This value
# is a placeholder: src/pipeline.py's asr_engine="fast" branch always calls
# fast_transcribe_windowed directly (see the shipping-decision note above) and
# never reads this value — kept for eval/ harnesses that still compare both
# configs, not as a live production switch.
FAST_ASR_MODE = "per_segment"

# ── Latency Wave 2: fast-engine model residency (src/model_registry.py) ─────
# CLAUDE.md's "load one model, release it — never hold ASR and LLM resident
# simultaneously" rule was written for the accurate CPU path (faster-whisper
# large-v3, ~3GB int8, coexisting with Qwen would risk OOM on 24GB). The fast
# engine's models are far smaller (pyannote ~0.5GB, parrotlet ~1GB, mlx-turbo
# ~1.6GB; Qwen lives in Ollama's separate process either way) — resident
# together they measure well inside the budget below. This flag is the
# carve-out: it keeps pyannote's Pipeline and parrotlet's _EmbeddingBackend
# loaded across sessions (src/model_registry.py) instead of reloading them
# every pipeline.run() call, cutting L2 and most of L3.5's wall time. Gated to
# asr_engine="fast" only in src/pipeline.py — the accurate CLI path is
# unaffected regardless of this flag's value.
FAST_ENGINE_RESIDENT = True
# Enforcement mechanism replacing the blanket rule for the fast engine: a
# measured peak-RSS ceiling (bench/stop_to_note_bench.py asserts against it)
# rather than an unconditional "never both" rule. ~7-8GB resident + headroom
# on the 24GB target machine.
RESIDENT_PEAK_RSS_BUDGET_MB = 14000
