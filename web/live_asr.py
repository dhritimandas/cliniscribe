"""L3-preview — UI-display-only live transcription during recording.

mlx-whisper (mlx-community/whisper-large-v3-turbo, RTF ~0.65 on target
hardware) re-decodes the ENTIRE recorded-so-far audio buffer roughly every
10 seconds while the doctor is still talking, purely so the capture screen
can show that recording/transcription is actually working.

CLINICAL-SAFETY UI CONTRACT (advisor-mandated, non-negotiable): this output
NEVER feeds L4, never lands in note.json, and never appears on the review
screen. It is reassurance that capture works, not clinical content — doctors
anchor on wrong drug names even in clearly labeled previews. Callers must
render it as running text only, under the label "LIVE PREVIEW — final
transcript follows", with no structured fields.

Memory: the mlx model (~1.5 GB) loads lazily on the first preview call and
stays resident for the rest of the recording; `release_model()` drops it
once recording stops (called from web/app.py's create_session). The
production faster-whisper model (src/l3_asr.py) is never loaded while this
one is resident, so the load-one-release-one discipline still holds at its
peak: mlx-preview during recording, faster-whisper during L3, never both.
"""

import gc
import io
import logging
import threading

import librosa

logger = logging.getLogger(__name__)

MODEL_ID = "mlx-community/whisper-large-v3-turbo"
SAMPLE_RATE = 16_000

try:
    import mlx_whisper
    from mlx_whisper.transcribe import ModelHolder

    _AVAILABLE = True
except ImportError:
    mlx_whisper = None
    ModelHolder = None
    _AVAILABLE = False
    logger.warning("mlx_whisper not installed — live preview disabled")

# Debounce guard: only one preview decode may run at a time. A non-blocking
# acquire lets the caller reject overlapping requests (429) instead of
# queuing them, so the SPA's periodic polling never stacks concurrent decodes.
_preview_lock = threading.Lock()


def available() -> bool:
    """Whether the live-preview model can be used (mlx_whisper import succeeded)."""
    return _AVAILABLE


def try_acquire() -> bool:
    """Non-blocking: True if this call may run a preview now, False if one is
    already in flight."""
    return _preview_lock.acquire(blocking=False)


def release() -> None:
    """Release the debounce lock acquired by a successful `try_acquire()`."""
    _preview_lock.release()


def transcribe_preview(wav_bytes: bytes) -> str:
    """Whole-buffer preview transcription of the recorded-so-far WAV audio.

    UI-DISPLAY ONLY — see module docstring's clinical-safety contract. Caller
    must hold the debounce lock (`try_acquire`/`release`) around this call.
    temperature=0 and condition_on_previous_text=False keep each call an
    independent, deterministic whole-buffer decode (no continuation state
    carried across the growing buffer's re-decoded prefix).
    """
    audio, _ = librosa.load(io.BytesIO(wav_bytes), sr=SAMPLE_RATE, mono=True)
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=MODEL_ID,
        temperature=0,
        condition_on_previous_text=False,
    )
    return result["text"].strip()


def release_model() -> None:
    """Best-effort: drop the cached mlx-whisper model once recording stops.

    `ModelHolder` (mlx_whisper.transcribe) caches the loaded model at module
    scope across calls; clearing it lets the ~1.5 GB of weights be garbage
    collected before the production faster-whisper (L3) or Ollama (L4)
    models load. A no-op if the model was never loaded (no recording used
    the preview, or mlx_whisper is unavailable).
    """
    if not _AVAILABLE or ModelHolder.model is None:
        return
    ModelHolder.model = None
    ModelHolder.model_path = None
    gc.collect()
