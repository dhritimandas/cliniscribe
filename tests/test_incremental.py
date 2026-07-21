"""Tests for web/incremental.py — IncrementalSession (latency Wave 3).

Fast tests fake decode_windows_words/diarize (no models loaded). One real
end-to-end test is marked @pytest.mark.slow (real mlx-whisper + pyannote).
"""

import numpy as np
import pytest

import web.incremental as incremental
from src.types import Segment, Turn


def _fake_words(start: float, end: float, label: str) -> list[tuple[float, float, str]]:
    """One synthetic word spanning the whole window, tagged with its bounds
    so assertions can tell which window produced it."""
    return [(start, end, f"{label}[{start:.1f}-{end:.1f}]")]


@pytest.fixture
def session(monkeypatch):
    # max_window_s set explicitly (small) so tests can trigger a decode with
    # short, fast-to-read durations — see _decode_newly_settled_locked's
    # docstring: a decode only fires once the settled backlog reaches a full
    # max_window_s chunk, never on every feed() tick.
    s = incremental.IncrementalSession("test-session", settle_margin_s=5.0, max_window_s=5.0)
    return s


def _silence_bytes(seconds: float, sr: int = 16000) -> bytes:
    """Fake WAV bytes librosa.load can decode: real silence, real header."""
    import io

    import soundfile as sf

    audio = np.zeros(int(seconds * sr), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_feed_below_settle_margin_decodes_nothing(session, monkeypatch):
    calls = []
    monkeypatch.setattr(incremental, "decode_windows_words", lambda audio, sr, windows: (calls.append(windows), [])[1])

    session.feed(_silence_bytes(3.0))  # 3s < 5s settle margin

    assert calls == []
    assert session._decoded_up_to_s == 0.0


def test_feed_below_max_window_backlog_decodes_nothing_even_past_settle_margin(session, monkeypatch):
    """settle_margin_s=5 makes some audio "settled", but max_window_s=5 means
    a decode call only fires once a FULL window's worth has backed up — a
    settled-but-sub-window backlog must wait, not trigger a small decode."""
    calls = []
    monkeypatch.setattr(incremental, "decode_windows_words", lambda audio, sr, windows: (calls.append(windows), [])[1])

    session.feed(_silence_bytes(8.0))  # settled_end = 8 - 5 = 3.0s < max_window_s=5.0

    assert calls == []
    assert session._decoded_up_to_s == 0.0


def test_feed_past_settle_margin_decodes_one_full_window(session, monkeypatch):
    calls = []

    def fake_decode(audio, sr, windows):
        calls.append(windows)
        return [w for start, end in windows for w in _fake_words(start, end, "settled")]

    monkeypatch.setattr(incremental, "decode_windows_words", fake_decode)

    session.feed(_silence_bytes(12.0))  # settled_end = 12 - 5 = 7.0s -> 1 full 5.0s window, 2.0s left as backlog

    assert calls == [[(0.0, 5.0)]]
    assert session._decoded_up_to_s == 5.0
    assert len(session._settled_words) == 1


def test_feed_does_not_redecode_already_settled_audio(session, monkeypatch):
    calls = []

    def fake_decode(audio, sr, windows):
        calls.append(windows)
        return [w for start, end in windows for w in _fake_words(start, end, "settled")]

    monkeypatch.setattr(incremental, "decode_windows_words", fake_decode)

    session.feed(_silence_bytes(12.0))  # settled_end=7.0 -> decodes [0,5.0), backlog 2.0s left
    session.feed(_silence_bytes(6.0))  # total 18s, settled_end=13.0, backlog since 5.0 = 8.0s -> 1 full window [5,10), 3.0s left

    assert calls == [[(0.0, 5.0)], [(5.0, 10.0)]]
    assert session._decoded_up_to_s == 10.0
    assert len(session._settled_words) == 2


def test_feed_decodes_multiple_full_windows_in_one_call_when_backlog_is_large(session, monkeypatch):
    """A big enough backlog (e.g. a slow-polling client) must pack into
    several max_window_s chunks in ONE decode_windows_words call, not one
    call per chunk — matches src.fast_asr's own windowed-decode contract."""
    calls = []

    def fake_decode(audio, sr, windows):
        calls.append(windows)
        return []

    monkeypatch.setattr(incremental, "decode_windows_words", fake_decode)

    session.feed(_silence_bytes(20.0))  # settled_end = 20-5 = 15.0 -> 3 full 5.0s windows

    assert calls == [[(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]]
    assert session._decoded_up_to_s == 15.0


def test_partial_turns_empty_before_anything_settles(session, monkeypatch):
    monkeypatch.setattr(incremental, "decode_windows_words", lambda audio, sr, windows: [])

    session.feed(_silence_bytes(3.0))

    assert session.partial_turns() == []


def test_partial_turns_diarizes_whole_buffer_and_attributes_settled_words(session, monkeypatch):
    def fake_decode(audio, sr, windows):
        return [w for start, end in windows for w in _fake_words(start, end, "settled")]

    diarize_calls = []

    def fake_diarize_buffer(self, audio):
        diarize_calls.append(len(audio))
        return [Segment(start=0.0, end=100.0, speaker="S0")]

    monkeypatch.setattr(incremental, "decode_windows_words", fake_decode)
    monkeypatch.setattr(
        incremental.IncrementalSession, "_diarize_current_buffer", fake_diarize_buffer
    )

    session.feed(_silence_bytes(12.0))
    turns = session.partial_turns()

    assert len(diarize_calls) == 1
    assert len(turns) == 1
    assert isinstance(turns[0], Turn)
    assert "settled[0.0-5.0]" in turns[0].text


def test_finalize_combines_settled_and_tail_words(session, monkeypatch):
    def fake_decode(audio, sr, windows):
        return [w for start, end in windows for w in _fake_words(start, end, "decoded")]

    def fake_diarize_buffer(self, audio):
        return [Segment(start=0.0, end=100.0, speaker="S0")]

    monkeypatch.setattr(incremental, "decode_windows_words", fake_decode)
    monkeypatch.setattr(
        incremental.IncrementalSession, "_diarize_current_buffer", fake_diarize_buffer
    )

    session.feed(_silence_bytes(12.0))  # settles+decodes [0, 5.0), decoded_up_to_s=5.0
    turns = session.finalize()  # tail backlog [5.0, 12.0) packs into [5,10) + [10,12)

    assert len(turns) == 1
    assert "decoded[0.0-5.0]" in turns[0].text
    assert "decoded[5.0-10.0]" in turns[0].text
    assert "decoded[10.0-12.0]" in turns[0].text


def test_finalize_runs_tail_decode_and_diarize_both(session, monkeypatch):
    decode_called = []
    diarize_called = []

    def fake_decode(audio, sr, windows):
        decode_called.append(windows)
        return []

    def fake_diarize_buffer(self, audio):
        diarize_called.append(len(audio))
        return [Segment(start=0.0, end=100.0, speaker="S0")]

    monkeypatch.setattr(incremental, "decode_windows_words", fake_decode)
    monkeypatch.setattr(
        incremental.IncrementalSession, "_diarize_current_buffer", fake_diarize_buffer
    )

    session.feed(_silence_bytes(3.0))  # nothing settles yet (< 5s margin)
    session.finalize()

    assert len(decode_called) == 1  # the tail decode inside finalize()
    assert len(diarize_called) == 1  # the final diarize inside finalize()


def test_finalize_with_no_audio_returns_empty_list(session, monkeypatch):
    monkeypatch.setattr(incremental, "decode_windows_words", lambda audio, sr, windows: [])
    monkeypatch.setattr(
        incremental.IncrementalSession, "_diarize_current_buffer", lambda self, audio: []
    )

    assert session.finalize() == []


def test_partial_turns_never_touches_l4_or_note_json(session, monkeypatch):
    """Clinical-safety contract, checked structurally: partial_turns()'s
    return type is exactly list[Turn] — the same shape L3 produces before L3.5/
    L4 ever run — so nothing about this call path can reach extract() or a
    note.json write without a caller deliberately doing so elsewhere."""
    monkeypatch.setattr(incremental, "decode_windows_words", lambda audio, sr, windows: [(0.0, 1.0, "hi")])
    monkeypatch.setattr(
        incremental.IncrementalSession,
        "_diarize_current_buffer",
        lambda self, audio: [Segment(start=0.0, end=100.0, speaker="S0")],
    )

    session.feed(_silence_bytes(12.0))
    result = session.partial_turns()

    assert isinstance(result, list)
    assert all(isinstance(t, Turn) for t in result)


# ── Real end-to-end equivalence smoke test ─────────────────────────────────


@pytest.mark.slow
def test_incremental_finalize_matches_batch_pipeline_on_real_clip():
    """Feed a realistic ~70s fixture in ~25s chunks (simulating tick-interval
    uploads during recording) at the PRODUCTION settle margin, then compare
    against the batch path (diarize() + fast_transcribe_windowed()) on the
    same audio.

    Proportions matter here: an earlier version of this test used an 11s
    clip with a 4s settle margin, which forced 2-5s decode windows — far
    smaller than either path's real ~28s window cap — and surfaced a real
    design bug (_decode_newly_settled_locked was decoding on every feed()
    tick instead of batching up to max_window_s; see LEARNINGS.md's Wave 3
    entry). At THESE proportions (settle_margin_s=10, the production
    default, on a clip several multiples of the 28s cap), incremental
    windows land at fixed 28s-multiple boundaries while batch's windows
    land at the nearest NATURAL break before the cap — a real, ACCEPTED
    residual difference (the plan's risk register: "window boundary choice
    only affects decode-window placement... provisional diarization is safe
    to use for it" — deliberately not chasing natural breaks during live
    capture, since only a still-changing provisional diarization exists
    then). Measured: this divergence alone costs real word overlap (69%
    observed here, vs 31% for the pre-fix tiny-window bug on an 11s clip) —
    the threshold below is set to catch a regression back to that bug's
    failure mode, not to demand boundary-identical transcripts, which
    fixed-duration windowing cannot promise by design.
    """
    import librosa

    from bench.stop_to_note_bench import build_fixture
    from src.fast_asr import fast_transcribe_windowed
    from src.l1_preprocess import TARGET_SR, preprocess
    from src.l2_diarize import diarize as batch_diarize

    fixture_path = build_fixture(target_seconds=70.0)
    wav_path = preprocess(fixture_path, out_dir="outputs/test_incremental_tmp")
    audio, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    total_s = len(audio) / TARGET_SR

    # --- Incremental path: ~25s feeds at the production settle margin.
    sess = incremental.IncrementalSession("equivalence-check")  # config defaults
    tick_s = 25.0
    pos = 0
    while pos < total_s:
        end = min(pos + tick_s, total_s)
        sess.feed(_to_wav_bytes(audio[int(pos * TARGET_SR) : int(end * TARGET_SR)], TARGET_SR))
        pos = end
    incremental_turns = sess.finalize()

    # --- Batch path on the identical preprocessed audio.
    batch_segments = batch_diarize(wav_path)
    batch_turns = fast_transcribe_windowed(wav_path, batch_segments)

    assert incremental_turns, "incremental path produced no turns on real audio"
    assert batch_turns, "batch path produced no turns on real audio (bad fixture?)"

    incremental_words = set(" ".join(t.text for t in incremental_turns).lower().split())
    batch_words = set(" ".join(t.text for t in batch_turns).lower().split())
    overlap = incremental_words & batch_words
    overlap_ratio = len(overlap) / max(1, len(batch_words))

    assert overlap_ratio >= 0.6, (
        f"incremental vs batch word overlap too low ({overlap_ratio:.0%}) — "
        f"incremental={incremental_words}\nbatch={batch_words}"
    )


def _to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    import io

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
