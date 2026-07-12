"""Unit tests for eval.engine_study.merge_segments — the pure merged-window
segmentation function behind the merged-window ASR decode probe (see
eval/engine_study.py module docstring for the hypothesis being tested) — and
for the anti-hallucination decode-param probe's kwarg-forwarding plumbing
(Probe A; see the module docstring's "Anti-hallucination decode-param probe"
section for the measured, negative result these unit tests do not re-derive)."""

import sys
import types

import numpy as np

from eval.engine_study import (
    ANTIHALLU_CONFIGS,
    KNOWN_BAD_CLIP_IDS,
    _decode_mlx,
    _fw_decode_segment_direct,
    _mlx_decode_segment,
    merge_segments,
)
from src.types import Segment


def test_merges_adjacent_same_speaker_within_gap_tolerance() -> None:
    segments = [
        Segment(start=0.0, end=2.0, speaker="S0"),
        Segment(start=2.3, end=5.0, speaker="S0"),
    ]
    windows = merge_segments(segments, max_window_s=28.0, gap_tolerance_s=1.0)
    assert windows == [Segment(start=0.0, end=5.0, speaker="S0")]


def test_speaker_change_breaks_the_merge() -> None:
    segments = [
        Segment(start=0.0, end=2.0, speaker="S0"),
        Segment(start=2.1, end=5.0, speaker="S1"),
    ]
    windows = merge_segments(segments, max_window_s=28.0, gap_tolerance_s=1.0)
    assert windows == segments


def test_gap_exceeding_tolerance_breaks_the_merge() -> None:
    segments = [
        Segment(start=0.0, end=2.0, speaker="S0"),
        Segment(start=5.0, end=8.0, speaker="S0"),  # 3.0s gap > 1.0s tolerance
    ]
    windows = merge_segments(segments, max_window_s=28.0, gap_tolerance_s=1.0)
    assert windows == segments


def test_max_window_cap_is_enforced() -> None:
    # Three same-speaker segments, each individually mergeable by gap, but
    # the third would push the window past a 10s cap — it must start a new
    # window instead.
    segments = [
        Segment(start=0.0, end=4.0, speaker="S0"),
        Segment(start=4.5, end=8.0, speaker="S0"),
        Segment(start=8.5, end=12.0, speaker="S0"),
    ]
    windows = merge_segments(segments, max_window_s=10.0, gap_tolerance_s=1.0)
    assert windows == [
        Segment(start=0.0, end=8.0, speaker="S0"),
        Segment(start=8.5, end=12.0, speaker="S0"),
    ]


def test_chronological_ordering_is_preserved_across_multiple_runs() -> None:
    segments = [
        Segment(start=0.0, end=2.0, speaker="S0"),
        Segment(start=2.2, end=4.0, speaker="S1"),
        Segment(start=4.1, end=6.0, speaker="S1"),
        Segment(start=6.2, end=8.0, speaker="S0"),
    ]
    windows = merge_segments(segments, max_window_s=28.0, gap_tolerance_s=1.0)
    assert [w.speaker for w in windows] == ["S0", "S1", "S0"]
    assert [(w.start, w.end) for w in windows] == [(0.0, 2.0), (2.2, 6.0), (6.2, 8.0)]


def test_empty_segments_returns_empty_list() -> None:
    assert merge_segments([]) == []


def test_single_segment_is_returned_as_one_window() -> None:
    seg = Segment(start=1.0, end=3.0, speaker="S0")
    assert merge_segments([seg]) == [seg]


# ── Anti-hallucination decode-param probe (Probe A) — kwarg plumbing ────────


def test_known_bad_clip_ids_is_the_documented_pair() -> None:
    """Guards the constant against silent drift/typos — probe_antihallu's
    fail-fast check is only meaningful against these two specific clips."""
    assert KNOWN_BAD_CLIP_IDS == {
        "f2096fbd33010dc7d81344b4ad2477e3",
        "1ae62262f1e93e81ff73e92bd9305f21",
    }


def test_antihallu_configs_have_expected_engine_keys() -> None:
    for name, config in ANTIHALLU_CONFIGS.items():
        assert set(config) == {"mlx", "fw"}, name
        assert config["mlx"]["condition_on_previous_text"] is False
        assert config["fw"]["condition_on_previous_text"] is False


def test_mlx_decode_segment_forwards_decode_kwargs() -> None:
    """decode_kwargs must reach mlx_whisper.transcribe() verbatim."""
    calls: list[dict] = []

    def fake_transcribe(audio, **kwargs):
        calls.append(kwargs)
        return {"text": "ok"}

    fake_module = types.SimpleNamespace(transcribe=fake_transcribe)
    original = sys.modules.get("mlx_whisper")
    sys.modules["mlx_whisper"] = fake_module
    try:
        seg = Segment(start=0.0, end=1.0, speaker="S0")
        text = _mlx_decode_segment(
            np.zeros(16000, dtype=np.float32),
            seg,
            "fake-repo",
            language=None,
            decode_kwargs=ANTIHALLU_CONFIGS["a1"]["mlx"],
        )
    finally:
        if original is not None:
            sys.modules["mlx_whisper"] = original
        else:
            del sys.modules["mlx_whisper"]

    assert text == "ok"
    assert len(calls) == 1
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["condition_on_previous_text"] is False


def test_decode_mlx_forwards_decode_kwargs_to_every_call_including_script_guard() -> None:
    """decode_kwargs must reach both the initial decode AND any script-guard
    re-decode triggered by an Arabic-script first pass."""
    calls: list[dict] = []
    texts = iter(["نیکس ڈوم", "नैक्सडॉम"])  # Arabic-script first, clean re-decode second

    def fake_transcribe(audio, **kwargs):
        calls.append(kwargs)
        return {"text": next(texts)}

    fake_module = types.SimpleNamespace(transcribe=fake_transcribe)
    original = sys.modules.get("mlx_whisper")
    sys.modules["mlx_whisper"] = fake_module
    try:
        segments = [Segment(start=0.0, end=1.0, speaker="S0")]
        hyp, _, guard_fires = _decode_mlx(
            np.zeros(16000, dtype=np.float32),
            segments,
            "fake-repo",
            decode_kwargs=ANTIHALLU_CONFIGS["a2"]["mlx"],
        )
    finally:
        if original is not None:
            sys.modules["mlx_whisper"] = original
        else:
            del sys.modules["mlx_whisper"]

    assert guard_fires == 1
    assert hyp == "नैक्सडॉम"
    assert len(calls) == 2
    assert all(c["logprob_threshold"] == -0.3 for c in calls)


def test_fw_decode_segment_direct_forwards_decode_kwargs() -> None:
    """decode_kwargs must reach WhisperModel.transcribe() verbatim (fw path)."""

    class _FakeChunk:
        def __init__(self, text: str) -> None:
            self.text = text

    class _FakeModel:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def transcribe(self, wav_path, **kwargs):
            self.calls.append(kwargs)
            return iter([_FakeChunk("ok")]), None

    model = _FakeModel()
    seg = Segment(start=0.0, end=1.0, speaker="S0")
    text = _fw_decode_segment_direct("fake.wav", seg, model, ANTIHALLU_CONFIGS["a1"]["fw"])

    assert text == "ok"
    assert len(model.calls) == 1
    assert model.calls[0]["temperature"] == [0.0]
    assert model.calls[0]["condition_on_previous_text"] is False
