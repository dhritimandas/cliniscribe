"""Incident regression suite — every REAL production/eval incident fixed in
this codebase, consolidated in one place as a permanent "never again" record.

Each test below documents one incident: the date/session (or eval artifact)
it was found in, and a one-line root cause. All tests are model-free (direct
function calls / mocks only) so this file runs in well under a second and
can gate every commit, not just release branches.
"""

from src.drug_lexicon import canonicalize_drug_span
from src.l3_5_normalize import _normalize_drug_text
from src.l3_asr import _contains_arabic_script
from src.l4_extract import _build_note


# ── (1) nextom/nasal-spray — outputs/20260711-184756-36c330, 2026-07-11 ─────
# Root cause: Whisper wrote a naxdom spelling variant ("नैक्स्टोम") absent from
# L3.5's curated Devanagari table, so L3.5 could not normalize it; qwen2.5:3b
# then INVENTED "nasal spray" as the drug name — a fabrication with no source
# in the transcript. Fixed in two places: L3.5's expanded fold-lexicon tier
# now recovers the spelling variant, and L4's grounding guard converts any
# drug name with no plausible transcript source to an unnamed row.


def test_naxdom_devanagari_variant_normalizes_with_dose() -> None:
    result = _normalize_drug_text("और एक नैक्स्टोम 500 खाईएगा")
    assert "naxdom 500" in result


def test_invented_nasal_spray_becomes_unnamed_medication() -> None:
    transcript = "[UNKNOWN]: aur ek naxdom 500 khaiyega"
    data = {"medications": [{"drug": "nasal spray", "dose": None}]}
    note = _build_note(data, transcript=transcript)
    assert note.medications[0].drug == "unnamed medication 1"
    assert any(f.endswith(".unnamed") for f in note.low_confidence_fields)


# ── (2) dose cross-attribution — outputs/20260711-184756-36c330, 2026-07-11 ─
# Root cause: the doctor never stated a paracetamol dose ("पैरसेट मॉल दिन में
# दो बार" — no number), but qwen2.5:3b glued a neighbouring drug's dose onto
# it ("naxdom 500" -> paracetamol dose "500 mg"). Fixed with a flag-only
# dose-provenance check: a dose whose digits appear nowhere near the drug's
# own transcript mention is flagged, never altered.


def test_paracetamol_dose_cross_attributed_from_naxdom_flagged() -> None:
    transcript = (
        "[UNKNOWN]: paracetamol lijiye subah shaam roz\n"
        "[UNKNOWN]: aur ek naxdom 500 khaiyega roz ek baar"
    )
    data = {
        "medications": [
            {"drug": "paracetamol", "dose": "500 mg", "frequency": "twice daily"}
        ]
    }
    note = _build_note(data, transcript=transcript)
    assert note.medications[0].dose == "500 mg"  # value is never altered
    assert "medications.paracetamol.dose_unattributed" in note.low_confidence_fields


# ── (3) BP/Hypertension category error — outputs/20260712-124506-715247,
# 2026-07-12 ──────────────────────────────────────────────────────────────
# Root cause: patient said "मेरा BP (Hypertension) भी हाई है" — L3.5's
# concept-glosser correctly annotated "BP" with its clinical gloss
# "(Hypertension)", but qwen2.5:3b then extracted the parenthetical itself,
# "(Hypertension) भी", as a MEDICATION drug name. The grounding guard
# correctly passed it through (the string IS in the transcript verbatim —
# grounding catches inventions, not category errors). Fixed with a condition
# guard: a drug string that fold-matches a clinical condition term (from
# src/concepts.py's CONCEPTS table) is dropped from medications entirely —
# the information is already captured elsewhere (history/diagnosis).


def test_bp_hypertension_gloss_produces_no_medication_row() -> None:
    transcript = "[UNKNOWN]: मेरा BP (Hypertension) भी हाई है"
    data = {"medications": [{"drug": "(Hypertension) भी", "dose": None}]}
    note = _build_note(data, transcript=transcript)
    assert note.medications == []
    assert "medications.(Hypertension) भी.condition_in_rx" in note.low_confidence_fields


def test_bp_hypertension_raw_english_variant_produces_no_medication_row() -> None:
    """Same category error, without the parenthetical gloss punctuation —
    the LLM extracting the bare English clinical term must be caught too."""
    data = {"medications": [{"drug": "Hypertension", "dose": None}]}
    note = _build_note(data, transcript="[UNKNOWN]: Hypertension noted")
    assert note.medications == []
    assert "medications.Hypertension.condition_in_rx" in note.low_confidence_fields


# ── (4) Urdu script misdetection — outputs/20260710-230150-cef13a,
# 2026-07-10 ──────────────────────────────────────────────────────────────
# Root cause: faster-whisper's per-segment language auto-detect misclassified
# spoken Hindi as Urdu and decoded the segment in Arabic script ("نیکس ڈوم
# فائیو ہنڈریڈ" for "Naxdom five hundred"). We only support hi/en/mr
# (Latin/Devanagari), so Arabic script is always a misdetection. Fixed with
# an L3 script guard that re-decodes any Arabic-script segment once, forcing
# language="hi". Full re-decode mechanics (mocked WhisperModel, call-count
# assertions) are exercised in tests/test_l3_asr.py — replicated here is only
# the minimal detection-function assertion, to keep this suite model-free.


def test_arabic_script_naxdom_misdetection_is_detected() -> None:
    assert _contains_arabic_script("نیکس ڈوم فائیو ہنڈریڈ")
    assert not _contains_arabic_script("नैक्सडॉम फाइव हंड्रेड")
    assert not _contains_arabic_script("naxdom five hundred")


# ── (5) दो तीन → Drotin wrong-substitution class — 49-clip frozen drug
# bench, 2026-07-11 (Drug Canonicalization Phase) ───────────────────────────
# Root cause: an initially-shipped 5-8-char/distance-1 fuzzy tier in
# src/drug_lexicon.py treated Hindi number phrases and short proper nouns as
# drug-name candidates — "दो तीन" ("two-three") fuzzy-matched to Drotin, a
# fragment of "vital" to Revital, and a doctor's surname (Pandey) to Pan-D.
# Wrong drug is worse than unknown, so that tier was dropped rather than
# patched with a per-word denylist (see _distance_bound's docstring and
# LEARNINGS.md "Drug Canonicalization Phase"). Fixed: fuzzy matching is now
# exact-fold-only below a 9-character key.


def test_do_teen_number_phrase_never_becomes_a_drug() -> None:
    assert canonicalize_drug_span("दो तीन") is None
    result = _normalize_drug_text("दो तीन 500 khao")
    assert "drotin" not in result.lower()
