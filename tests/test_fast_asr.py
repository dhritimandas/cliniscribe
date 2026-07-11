"""Tests for src.fast_asr: the degeneration detector, the retry ladder, and
the fast_transcribe() contract.

Detector fixtures are real hallucinated hypotheses, not invented text:
'college college...' and 'झाल झाल...' are copied verbatim from cached
mlx-turbo/fw-turbo decodes in outputs/engine_study.json (clip IDs cited
inline). The glued-repeat fixture ("ट्ट्ट्ट्..." looping) is documented in
eval/engine_study.py's module docstring (mlx-turbo `a1` probe on clip
f2096fbd) but was never persisted to JSON — probe_antihallu() intentionally
never writes results (see its docstring) — so it is reproduced here from
that prose record, extended to a realistic length, rather than grepped from
a file.

test_zero_false_positives_on_frozen_bench_baseline runs the detector over
every hypothesis in outputs/beam_study.json (the fw-large-v3 production
baseline, 10 clips x beam sizes 1 and 5 = 20 texts) and asserts zero false
positives — except clip 9ea0bcce's beam=5 hypothesis, which the detector
correctly flags: it ends in "हुआ है" repeated 7 times consecutively, a real
repetition-loop hallucination baked into that cached baseline hypothesis
itself. That is a true positive on genuinely degenerate text, not a false
positive on clean text — see the test for the exact repeated substring.
"""

import sys
import types

import numpy as np
import pytest
import soundfile as sf

from src import config
from src.fast_asr import (
    _degeneracy_score,
    _is_latin_dominant,
    _max_consecutive_phrase_repeats,
    _retry_ladder,
    fast_transcribe,
    looks_degenerate,
)
from src.types import Segment, Turn

# ── Real hallucination fixtures (see module docstring for provenance) ───────

# outputs/engine_study.json, clip d3da2c34..., engine "mlx-turbo": "college"
# repeated ~223 times consecutively before trailing off into English.
COLLEGE_LOOP = "college " * 223 + "And, uh... medicines there on Metra. Thank you."
COLLEGE_LOOP_DURATION_S = 11.072

# outputs/engine_study.json, clip f2096fbd..., engine "fw-turbo".
JHAAL_LOOP_8X = (
    "Skin allergy, fungal allergy, infection. झाल झाल झाल झाल झाल झाल झाल झाल "
    "कब से हैं आपको? चार कादिसे. चार साल से है झाल क्या इश्यू हो रहा है राश्यस हो रही है आपको"
)
JHAAL_LOOP_8X_DURATION_S = 10.592

# outputs/engine_study.json, clip b26524ae..., engine "mlx-turbo": exactly 4
# consecutive repeats — the minimum that config.DEGEN_NGRAM_MIN_REPEAT catches.
JHAAL_LOOP_4X = (
    "बन नगय तो बिएस्ट झाल झाल झाल झाल और इसके साथ बुखार या उल्टी कुछ हो रहा है नहीं नहीं थोड़ा सा"
)
JHAAL_LOOP_4X_DURATION_S = 11.4

# outputs/engine_study.json, clip ce76bfdd..., engine "fw-turbo": only 3
# consecutive repeats — must stay BELOW the >=4 threshold (real clip, not a
# known-bad one; the ladder should never fire on this).
JHAAL_LOOP_3X_CLEAN = (
    "Thank you. This infection has been caused by the infection. Thank you. "
    "And this is why we use medication. झाल झाल झाल अज़ I'll put sunscreen on. "
    "दो संस्क्रीन दे रहा हूं"
)
JHAAL_LOOP_3X_CLEAN_DURATION_S = 11.232

# outputs/engine_study.json, clip 1ae62262..., engine "mlx-large-v3": "प्रति"
# repeated ~35 times.
PRASTUTI_LOOP = (
    "कि इसके करण हो गई है यह stress related या migraine भी है " + "प्रति " * 35 + "आहा"
)
PRASTUTI_LOOP_DURATION_S = 20.4

# eval/engine_study.py module docstring ("Anti-hallucination decode-param
# probe" section): mlx-turbo `a1` config on clip f2096fbd looped
# "ट्ट्ट्ट्ट्ट्ट्ट्ट्ट्ट्ट्ट्..." — a single glued (whitespace-free) run, the
# kind word-level n-gram checks cannot see. Not in outputs/engine_study.json
# (probe_antihallu never persists), so reproduced from the docstring record.
GLUED_REPEAT_LOOP = "ट्" * 200

# outputs/beam_study.json, clip 9ea0bcce..., beam="5": a real hallucination
# living in the cached fw-large-v3 PRODUCTION baseline itself (7 consecutive
# repeats of the bigram "हुआ है"). Excluded from the "clean" set on purpose —
# see test_zero_false_positives_on_frozen_bench_baseline.
KNOWN_BASELINE_LOOP_CLIP_ID = "9ea0bcce"
KNOWN_BASELINE_LOOP_BEAM = "5"


# ── Degeneration detector ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,duration",
    [
        (COLLEGE_LOOP, COLLEGE_LOOP_DURATION_S),
        (JHAAL_LOOP_8X, JHAAL_LOOP_8X_DURATION_S),
        (JHAAL_LOOP_4X, JHAAL_LOOP_4X_DURATION_S),
        (PRASTUTI_LOOP, PRASTUTI_LOOP_DURATION_S),
        (GLUED_REPEAT_LOOP, 2.0),
    ],
)
def test_detects_known_real_hallucination_loops(text: str, duration: float) -> None:
    assert looks_degenerate(text, duration) is True


def test_does_not_flag_three_consecutive_repeats_below_threshold() -> None:
    """Exactly 3 repeats of 'झाल' — below DEGEN_NGRAM_MIN_REPEAT (4) — must
    stay clean; a stressed patient saying a word 3 times is real speech."""
    assert (
        looks_degenerate(JHAAL_LOOP_3X_CLEAN, JHAAL_LOOP_3X_CLEAN_DURATION_S) is False
    )


def test_empty_text_on_long_voiced_segment_is_degenerate() -> None:
    assert looks_degenerate("", config.DEGEN_EMPTY_ON_VOICED_MIN_S + 0.5) is True


def test_empty_text_on_brief_segment_is_not_flagged() -> None:
    assert looks_degenerate("", 0.2) is False


def test_clean_conversational_text_is_not_flagged() -> None:
    text = "हलो मुझे बुखार हो रहा है और सरदर्द हो रहा है तो मैं क्या करूं एक पैरासिटामोल खा लेना"
    assert looks_degenerate(text, 8.0) is False


def test_zero_false_positives_on_frozen_bench_baseline() -> None:
    """Sweep the detector over every hypothesis in outputs/beam_study.json
    (fw-large-v3, the production baseline — 10 clips x 2 beam sizes) and
    assert no false positives, except the one genuine hallucination the
    detector correctly finds in that cache (see module docstring)."""
    import json
    import os

    path = "outputs/beam_study.json"
    if not os.path.exists(path):
        pytest.skip(f"{path} not present in this environment")
    with open(path, encoding="utf-8") as f:
        beam_study = json.load(f)

    false_positives: list[str] = []
    for clip in beam_study["per_clip"]:
        clip_id = clip["id"][:8]
        for beam in ("1", "5"):
            text = clip["beams"][beam]["hypothesis"]
            is_known_loop = (
                clip_id == KNOWN_BASELINE_LOOP_CLIP_ID
                and beam == KNOWN_BASELINE_LOOP_BEAM
            )
            flagged = looks_degenerate(text, audio_seconds=15.0)
            if flagged and not is_known_loop:
                false_positives.append(f"{clip_id} beam={beam}")
            if is_known_loop:
                assert flagged is True, "expected the known baseline loop to be caught"

    assert false_positives == [], f"unexpected false positives: {false_positives}"


def test_max_consecutive_phrase_repeats_counts_non_overlapping_runs() -> None:
    tokens = ["a", "b", "a", "b", "a", "b", "c", "d"]
    assert _max_consecutive_phrase_repeats(tokens, 2) == 3  # "a b" x3, then "c d"
    assert _max_consecutive_phrase_repeats(tokens, 1) == 1  # no single-token repeat


def test_is_latin_dominant() -> None:
    assert _is_latin_dominant("this is unclear text") is True
    assert _is_latin_dominant("यह अस्पष्ट पाठ है") is False


def test_degeneracy_score_ranks_clean_text_below_looping_text() -> None:
    clean = "हलो मुझे बुखार हो रहा है"
    assert _degeneracy_score(clean) < _degeneracy_score(COLLEGE_LOOP)


# ── Retry ladder (fake decoder — no model load) ──────────────────────────────


def _patch_decode_window(
    monkeypatch: pytest.MonkeyPatch, responses: list[str]
) -> list[dict]:
    """Monkeypatch src.fast_asr._decode_window to return `responses` in order
    and record every call's arguments."""
    calls: list[dict] = []
    it = iter(responses)

    def fake(audio, sr, start, end, repo, *, language, decode_kwargs):
        calls.append(
            {
                "start": start,
                "end": end,
                "repo": repo,
                "language": language,
                "decode_kwargs": dict(decode_kwargs),
            }
        )
        return next(it)

    monkeypatch.setattr("src.fast_asr._decode_window", fake)
    return calls


def test_ladder_stops_at_step1_boundary_shift(monkeypatch: pytest.MonkeyPatch) -> None:
    recovered_text = (
        "chaar saal se hai, kya issue ho raha hai aapko"  # clean -- step1 succeeds
    )
    calls = _patch_decode_window(monkeypatch, [recovered_text])
    seg = Segment(start=5.0, end=8.0, speaker="S0")
    text, step = _retry_ladder(
        np.zeros(16000, dtype=np.float32),
        16000,
        seg,
        total_duration=30.0,
        initial_text=JHAAL_LOOP_4X,
    )
    assert step == "step1_boundary_shift"
    assert text == recovered_text
    assert len(calls) == 1
    assert calls[0]["start"] == pytest.approx(5.0 - config.RETRY_BOUNDARY_SHIFT_S)
    assert calls[0]["end"] == pytest.approx(8.0 + config.RETRY_BOUNDARY_SHIFT_S)
    assert calls[0]["repo"] == config.FAST_ASR_MODEL


def test_boundary_shift_clamps_at_clip_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_decode_window(monkeypatch, ["clean text"])
    seg = Segment(start=0.1, end=29.9, speaker="S0")
    _retry_ladder(
        np.zeros(16000, dtype=np.float32),
        16000,
        seg,
        total_duration=30.0,
        initial_text=JHAAL_LOOP_4X,
    )
    assert calls[0]["start"] == 0.0  # would be -0.3 unshifted; clamped to 0
    assert calls[0]["end"] == 30.0  # would be 30.3 unshifted; clamped to total_duration


def test_ladder_falls_through_to_step2(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_decode_window(monkeypatch, [JHAAL_LOOP_4X, "clean recovered text"])
    seg = Segment(start=1.0, end=4.0, speaker="S0")
    text, step = _retry_ladder(
        np.zeros(16000, dtype=np.float32),
        16000,
        seg,
        total_duration=30.0,
        initial_text=JHAAL_LOOP_8X,
    )
    assert step == "step2_temp_condition"
    assert text == "clean recovered text"
    assert len(calls) == 2
    assert (
        calls[1]["start"] == 1.0 and calls[1]["end"] == 4.0
    )  # step2 uses original bounds
    assert calls[1]["decode_kwargs"]["temperature"] == config.RETRY_TEMP_STEP2
    assert calls[1]["decode_kwargs"]["condition_on_previous_text"] is True


def test_ladder_falls_through_to_step3_bigger_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_decode_window(
        monkeypatch, [JHAAL_LOOP_4X, JHAAL_LOOP_8X, "clean at last"]
    )
    seg = Segment(start=1.0, end=4.0, speaker="S0")
    text, step = _retry_ladder(
        np.zeros(16000, dtype=np.float32),
        16000,
        seg,
        total_duration=30.0,
        initial_text=PRASTUTI_LOOP,
    )
    assert step == "step3_large_model"
    assert text == "clean at last"
    assert len(calls) == 3
    assert calls[2]["repo"] == config.FAST_ASR_FALLBACK_MODEL


def test_ladder_gives_up_and_wraps_least_degenerate_hindi_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # All three retries still degenerate; step2's output has the lowest
    # _degeneracy_score of the 4 candidates (exactly at the repeat-count
    # threshold, vs. the much longer loops elsewhere) and must be chosen.
    least_bad = "ठीक है ठीक है ठीक है ठीक है"  # 4x bigram repeat -> score 1.0
    calls = _patch_decode_window(monkeypatch, [COLLEGE_LOOP, least_bad, PRASTUTI_LOOP])
    seg = Segment(start=1.0, end=4.0, speaker="S0")
    text, step = _retry_ladder(
        np.zeros(16000, dtype=np.float32),
        16000,
        seg,
        total_duration=30.0,
        initial_text=JHAAL_LOOP_8X,
    )
    assert step == "step4_giveup"
    assert len(calls) == 3
    assert text.startswith("[अस्पष्ट — VERIFY] ")
    assert least_bad in text


def test_ladder_gives_up_and_wraps_latin_dominant_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # score 1.0, same tier as the Hindi case above
    latin_least_bad = "unclear unclear unclear unclear speech fragment"
    _patch_decode_window(monkeypatch, [COLLEGE_LOOP, latin_least_bad, PRASTUTI_LOOP])
    seg = Segment(start=1.0, end=4.0, speaker="S0")
    text, step = _retry_ladder(
        np.zeros(16000, dtype=np.float32),
        16000,
        seg,
        total_duration=30.0,
        initial_text=JHAAL_LOOP_8X,
    )
    assert step == "step4_giveup"
    assert text.startswith("[unclear — VERIFY] ")


# ── fast_transcribe contract test (fake mlx_whisper module) ──────────────────


@pytest.fixture
def silent_wav(tmp_path) -> str:
    path = tmp_path / "silent_16k.wav"
    sf.write(str(path), np.zeros(16000 * 10, dtype=np.float32), 16000)
    return str(path)


def _install_fake_mlx_whisper(
    monkeypatch: pytest.MonkeyPatch, texts_by_language: dict
) -> list[dict]:
    """Install a fake mlx_whisper module keyed by the `language` kwarg, matching
    the sys.modules mocking convention already used in tests/test_engine_study.py."""
    calls: list[dict] = []

    def fake_transcribe(audio, **kwargs):
        calls.append(kwargs)
        return {"text": texts_by_language[kwargs["language"]]}

    fake_module = types.SimpleNamespace(transcribe=fake_transcribe)
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_module)
    return calls


def test_fast_transcribe_returns_turn_list_shape(
    monkeypatch: pytest.MonkeyPatch, silent_wav: str
) -> None:
    _install_fake_mlx_whisper(
        monkeypatch, {None: "how long have you had fever", "hi": "n/a"}
    )
    segments = [Segment(start=0.0, end=2.0, speaker="SPEAKER_00")]
    turns = fast_transcribe(silent_wav, segments)
    assert len(turns) == 1
    assert isinstance(turns[0], Turn)
    assert turns[0].text == "how long have you had fever"
    assert turns[0].start == 0.0 and turns[0].end == 2.0


def test_fast_transcribe_applies_doctor_patient_role_heuristic(
    monkeypatch: pytest.MonkeyPatch, silent_wav: str
) -> None:
    calls_seen: list[str] = []

    def fake_transcribe(audio, **kwargs):
        # First call -> doctor segment, second -> patient segment.
        text = "what medicine dosage tablet" if not calls_seen else "hi hello okay"
        calls_seen.append(text)
        return {"text": text}

    fake_module = types.SimpleNamespace(transcribe=fake_transcribe)
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_module)

    segments = [
        Segment(start=0.0, end=2.0, speaker="SPEAKER_00"),
        Segment(start=2.0, end=4.0, speaker="SPEAKER_01"),
    ]
    turns = fast_transcribe(silent_wav, segments)
    assert len(turns) == 2
    assert turns[0].speaker_role == "DOCTOR"
    assert turns[1].speaker_role == "PATIENT"


def test_fast_transcribe_script_guard_fires_on_arabic_script(
    monkeypatch: pytest.MonkeyPatch, silent_wav: str
) -> None:
    calls = _install_fake_mlx_whisper(
        monkeypatch, {None: "نیکس ڈوم فائیو ہنڈریڈ", "hi": "नैक्सडॉम 500"}
    )
    segments = [Segment(start=0.0, end=2.0, speaker="SPEAKER_00")]
    turns = fast_transcribe(silent_wav, segments)
    assert len(turns) == 1
    assert turns[0].text == "नैक्सडॉम 500"
    assert len(calls) == 2
    assert calls[0]["language"] is None
    assert calls[1]["language"] == "hi"


def test_fast_transcribe_on_progress_reports_completion(
    monkeypatch: pytest.MonkeyPatch, silent_wav: str
) -> None:
    _install_fake_mlx_whisper(monkeypatch, {None: "okay fine", "hi": "n/a"})
    progress: list[tuple[float, float]] = []
    segments = [
        Segment(start=0.0, end=2.0, speaker="SPEAKER_00"),
        Segment(start=2.0, end=5.0, speaker="SPEAKER_00"),
    ]
    fast_transcribe(
        silent_wav,
        segments,
        on_progress=lambda done, total: progress.append((done, total)),
    )
    assert progress == [(2.0, 5.0), (5.0, 5.0)]


def test_fast_transcribe_no_segments_returns_empty_list(silent_wav: str) -> None:
    assert fast_transcribe(silent_wav, []) == []
