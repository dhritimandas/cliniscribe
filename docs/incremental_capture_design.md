# Design note: incremental processing during recording

**Status (2026-07-20): backend primitive implemented; HTTP/browser wiring
still deferred.** `web/incremental.py`'s `IncrementalSession` (`feed` /
`partial_turns` / `finalize`) exists, is unit-tested, and is verified against
the batch pipeline on real audio (equivalence test in
`tests/test_incremental.py`) — but no FastAPI route calls `feed()` yet, and
no capture-screen JS streams audio chunks to one. Today's upload API (`POST
/api/sessions`) is still a single whole-file upload; wiring a session-scoped
chunked-ingestion endpoint plus the browser-side capture loop is real,
separate work that needs an actual capture UI to test against — building
those routes now, with nothing driving them, would be exactly the
speculative scaffolding CLAUDE.md's Development Rules warn against. This
note now also records one deliberate DEPARTURE from the original design
below: **frozen windows, not whole-file re-decode at stop** (see "Departure
from the original reconciliation design").

## Departure from the original reconciliation design

The original design (this note, as first written) called for one final
whole-file re-decode of the ENTIRE buffer at stop, discarding all partial
results. The shipped implementation instead **freezes** each settled
window's decode and never re-decodes it: production's fast-path decode is
deterministic (`temperature=0.0`, `condition_on_previous_text=False` —
`src.config.FAST_ASR_DECODE_KWARGS`), so re-decoding identical audio with
identical parameters reproduces identical text — a whole-file re-decode
would be provably redundant work for every window whose audio hasn't
changed since it settled. `finalize()` decodes only the short remaining
tail (the audio still within the settle margin) and reconciles diarization
with one final full-file pyannote pass, run concurrently with the tail
decode. This is the actual mechanism behind "stop-to-note collapses to
roughly (tail decode ∥ final diarize)" below — not a full re-decode hidden
by warm caches, but no re-decode at all for settled audio.

## Why this matters (step-1 evidence, 2026-07-10)

Instrumented baseline on 3 real clips: **L3 ASR is 82–95% of end-to-end wall
time**; every other stage combined is under 15%. A consultation lasts minutes
while the doctor talks — that is exactly the idle compute window L3 needs.
Hiding transcription inside recording time turns perceived latency into
(tail-processing + L3.5 + L4 + L5) ≈ under a minute, without touching model
quality.

## Design (as shipped — see the departure note above for what changed)

**Granularity.** Segment-free, duration-based windows (`pack_duration_into_windows`,
`src/fast_asr.py`), not the batch path's diarization-driven packing — during
recording only a PROVISIONAL diarization exists and keeps changing, so
windows cannot be cut at diarized natural breaks the way the batch path
does. This sidesteps the two known failure modes of naive chunked decoding
(context starvation at chunk boundaries, diarization instability on short
windows) differently than originally planned: not by re-decoding the whole
buffer, but by decoding fixed ≤28s (`config.WINDOW_MAX_SPAN_S`) chunks once
each and re-attributing their words to whichever diarization is available at
read time (provisional for `partial_turns()`, final for `finalize()`) —
word-level attribution is fully decoupled from decode-window boundaries
already (`_assign_words_to_turns`), so this re-attribution is safe.

**Settling, not whole-buffer reconciliation.** `feed()` decodes complete
`max_window_s`-sized chunks of "settled" backlog — audio more than
`config.INCREMENTAL_SETTLE_MARGIN_S` behind the buffer's growing edge — and
NEVER redecodes them once decoded (see the departure note: production decode
is deterministic, so a re-decode would reproduce identical text). A decode
call is deliberately withheld until a full window's worth of backlog has
accumulated, not fired on every tick, matching the fixed ~7-9s-per-call cost
`fast_transcribe_windowed`'s own windowing was built to amortize (firing on
every ~10s tick would create MORE, SMALLER calls than the batch path's
~28s-capped windows — worse, not better; caught by the real-audio
equivalence test during implementation, see LEARNINGS.md's Wave 3 entry).

**Reconciliation at stop.** `finalize()` decodes only the un-settled tail
and runs the final full-file pyannote pass CONCURRENTLY (two threads,
joined) — not sequentially, and not a whole-file ASR re-decode. Partial
results (`partial_turns()`) exist ONLY for UI display, never to feed L4 —
the clinical-safety contract is in `web/incremental.py`'s module docstring
and enforced by a dedicated test.

**API shape (implemented, matches the original design).**
```python
session = IncrementalSession(session_id)      # settle_margin_s/max_window_s from config
session.feed(wav_bytes)                        # append the NEW audio increment; may trigger a decode
partial = session.partial_turns()              # latest attributed turns (UI only)
turns = session.finalize()                     # authoritative final pass
```
`IncrementalSession` does not hold its own model copies — it calls
`src.model_registry.get_pyannote_pipeline()` / `get_embedding_backend()`
(latency Wave 2) for the resident pyannote pipeline, and mlx-whisper needs no
explicit residency at all (`mlx_whisper.transcribe.ModelHolder` already
caches at module scope for free — see Wave 2's LEARNINGS entry).

**L4 warm-up.** Not yet wired into `IncrementalSession` — `warm_llm()`
firing during incremental capture (rather than at L3.5 time, as the batch
pipeline does today) is part of the still-deferred HTTP/browser wiring, not
this backend primitive.

## Preconditions before HTTP/browser wiring (the remaining work)

1. A capture UI that streams audio chunks to a session-scoped endpoint —
   does not exist yet; today's capture screen posts to the stateless
   `/api/live/preview` (no session, UI-display-only) then a single whole-file
   `POST /api/sessions` at the end.
2. New session-scoped routes: something like `POST /api/sessions/{sid}/feed`
   calling `IncrementalSession.feed()`, and stop calling `.finalize()`
   followed by the existing `pipeline` L3.5→L4→L5 stages via a
   `precomputed_turns`-style entry point (not yet built either).
3. Measure actual idle headroom during recording on target hardware — the
   decode/diarize work inside `feed()` must not starve the audio capture
   thread.
4. Wire `warm_llm()` into the capture loop once the above exists, so L4 is
   warm by the time `finalize()` reaches it (mirroring the batch pipeline's
   L3.5-time warm dispatch).
