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

# ── L3-fast ASR (mlx-whisper, gated) ─────────────────────────────────────────
# Chunk-local detect+retry engine (src/fast_asr.py), built to answer the
# Deployment Latency Phase's four negative results (see LEARNINGS.md): faster
# engines/merged windows/decode-param tweaks/VAD pre-slicing all failed the
# frozen-bench gate outright. This flag stays False — the production path
# keeps using src.l3_asr.transcribe — until eval/fast_asr_gate.py reports a
# PASS; the caller (src/pipeline.py) flips it, this file does not.
FAST_ASR_ENABLED = False

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
