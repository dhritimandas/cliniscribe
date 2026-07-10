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
