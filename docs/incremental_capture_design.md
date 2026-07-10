# Design note: incremental processing during recording (deferred)

**Status: NOT implemented — deliberately.** The latency goal's own gate
("only if step-1 shows recording-time compute is idle") is unmeasured: no
capture UI exists yet, so there is no recording time to hide compute inside.
Building the API now would be speculative scaffolding (CLAUDE.md Development
Rules). This note records the design so implementation can start the day a
capture screen exists.

## Why this matters (step-1 evidence, 2026-07-10)

Instrumented baseline on 3 real clips: **L3 ASR is 82–95% of end-to-end wall
time**; every other stage combined is under 15%. A consultation lasts minutes
while the doctor talks — that is exactly the idle compute window L3 needs.
Hiding transcription inside recording time turns perceived latency into
(tail-processing + L3.5 + L4 + L5) ≈ under a minute, without touching model
quality.

## Design

**Granularity.** Process completed audio in growing whole-file passes, not
streaming chunks: every N seconds (N ≈ 30), run the pipeline's L1+L2+L3 on
the ENTIRE recorded-so-far buffer, replacing the previous partial result.
This deliberately re-decodes earlier audio — the compute waste hides inside
recording time, and it sidesteps the two known failure modes of chunked
decoding: context starvation at chunk boundaries (the "daily three times"
class) and diarization instability on short windows.

**Reconciliation at stop.** When recording stops, one final whole-file pass
over the complete audio produces the authoritative transcript; all partial
results are discarded. Partial results exist ONLY to warm caches and to give
the capture UI live progress ("transcribed so far…"), never to feed L4.

**Interaction with whole-file ASR (implemented).** Fully compatible: each
partial pass IS a whole-file pass over the current buffer. No per-segment
slicing anywhere.

**API shape.**
```python
session = IncrementalSession(session_id)      # owns models, keeps them warm
session.feed(wav_bytes)                        # append audio; may trigger a pass
partial = session.partial_turns()              # latest attributed turns (UI only)
turns = session.finalize()                     # authoritative final pass
```
`IncrementalSession` holds pyannote + Whisper resident for its lifetime
(justified: capture is interactive; release both before L4 loads, preserving
load-one-release-one at its peak).

**L4 warm-up.** `finalize()` fires the existing `warm_llm()` thread as soon
as the final ASR pass starts — by L4 time the model is resident (implemented
in the batch pipeline already).

## Preconditions before implementing

1. A capture UI exists (review-frontend goal) with a real recording loop.
2. Measure actual idle headroom during recording on target hardware — the
   passes must not starve the audio capture thread.
3. Re-measure pass latency with whole-file ASR + any cpu_threads tuning in
   place; N (pass interval) is chosen from those numbers, not guessed.
