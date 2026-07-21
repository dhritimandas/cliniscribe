"""Process-lifetime model residency for the fast engine (latency Wave 2).

CLAUDE.md's "load one model, release it — never hold ASR and LLM resident
simultaneously" rule still applies unconditionally to the accurate CPU path
(src/l3_asr.py, faster-whisper large-v3). This module exists only to serve
the fast engine (src/config.py: FAST_ENGINE_RESIDENT), whose models are small
enough to stay resident together across sessions within one long-running
server process — see src/config.py's FAST_ENGINE_RESIDENT docstring for the
budget this replaces the blanket rule with.

mlx-whisper needs no entry here: mlx_whisper.transcribe.ModelHolder already
caches the loaded model at module scope inside the mlx_whisper library
itself, for free, across every call in the process — the only thing that was
defeating that cache was web/app.py's create_session() unconditionally
calling live_asr.release_model() on every upload (fixed alongside this
module, gated on FAST_ENGINE_RESIDENT).
"""

import gc
import logging
import os
import threading

import torch
from dotenv import load_dotenv

from src import config

load_dotenv()  # HF_TOKEN for pyannote — self-sufficient regardless of import
# order; do not rely on src.pipeline's own load_dotenv() side effect (bit us
# once: a test importing web.incremental directly, never src.pipeline, hit
# "HF_TOKEN not set" inside a background diarize thread — see LEARNINGS.md).
logger = logging.getLogger(__name__)

_lock = threading.Lock()
_pyannote_pipeline = None
_embedding_backend = None


def get_pyannote_pipeline():
    """Return a process-resident pyannote Pipeline, loading it on first use."""
    global _pyannote_pipeline
    with _lock:
        if _pyannote_pipeline is None:
            from pyannote.audio import Pipeline

            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                raise EnvironmentError(
                    "HF_TOKEN not set — required for pyannote model download"
                )
            pipeline = Pipeline.from_pretrained(config.DIARIZE_MODEL, token=hf_token)
            device = (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
            pipeline.to(device)
            _pyannote_pipeline = pipeline
            logger.info("model_registry: pyannote resident on %s", device)
        return _pyannote_pipeline


def get_embedding_backend():
    """Return a process-resident parrotlet-e _EmbeddingBackend, loading on first use."""
    global _embedding_backend
    with _lock:
        if _embedding_backend is None:
            from src.l3_5_normalize import _EmbeddingBackend

            _embedding_backend = _EmbeddingBackend()
            logger.info("model_registry: parrotlet embedding backend resident")
        return _embedding_backend


def reset() -> None:
    """Drop all resident models and free device memory.

    Not called in normal server operation (models stay resident for the
    process lifetime by design) — for test isolation and explicit shutdown.
    """
    global _pyannote_pipeline, _embedding_backend
    with _lock:
        if _embedding_backend is not None:
            _embedding_backend.release()
        _pyannote_pipeline = None
        _embedding_backend = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
