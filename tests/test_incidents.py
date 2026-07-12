"""Incident regression suite — every REAL production/eval incident fixed in
this codebase, consolidated in one place as a permanent "never again" record.

Each test below documents one incident: the date/session (or eval artifact)
it was found in, and a one-line root cause. All tests are model-free (direct
function calls / mocks only) so this file runs in well under a second and
can gate every commit, not just release branches.
"""

from src.concepts import CONCEPTS, EVERYDAY_WORDS
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


# ── (6) candra-o fold gap — नैक्स्टॉम, outputs/<session>, 2026-07-12 ────────
# Root cause: ASR wrote "एक नैक्स्टॉम 500 एक" using candra-o (ॉ), a Devanagari
# vowel sign absent from all three "mirrored" fold-map implementations
# (eval/drug_bench.py, src/l4_extract.py, src/drug_lexicon.py) — they listed
# ा ि ी ु ू े ै ो ौ ं but not ॉ/ॅ/ृ/ऑ/ऍ/ँ/ः, so candra-o (and its siblings)
# silently dropped out of the fold key instead of contributing "o". The
# curated table already had नैक्स्टोम (regular ो) but not the candra-o
# spelling. Fixed: all three fold maps gained the 7 missing entries (see
# tests/test_fold_parity.py for the cross-implementation guard) and the
# candra-o spelling was added to _DEVA_CURATED alongside its regular-o twin.


def test_naxdom_candra_o_variant_normalizes_with_dose() -> None:
    result = _normalize_drug_text("एक नैक्स्टॉम 500 एक")
    assert "naxdom 500" in result


# ── (7) filler-word glue on a drug span — same incident as (6) ─────────────
# Root cause: even after (6)'s fix restores the drug name inside the
# transcript, an LLM extraction can still copy Hindi filler words glued to
# both ends of the drug span into the "drug" field verbatim (rule 12 says
# copy exactly what's written) — "एक नैक्स्टॉम 500 एक" ("one naxdom 500 one")
# was extracted as a single drug name. Fixed with a conservative span-
# isolation step in L4: filler tokens are stripped from the edges only, and
# the drug is trimmed/canonicalized ONLY when the remaining inner span
# resolves via the curated table, the expanded lexicon, or CDSCO — an
# unmatched name is never truncated.


def test_filler_glued_drug_span_isolates_to_dose_and_name() -> None:
    data = {"medications": [{"drug": "एक नैक्स्टॉम 500 एक", "dose": None}]}
    note = _build_note(
        data, transcript="[UNKNOWN]: एक नैक्स्टॉम 500 एक खा लो"
    )
    assert note.medications[0].drug == "naxdom 500"


def test_filler_glued_dolo_isolates_cleanly() -> None:
    data = {"medications": [{"drug": "dolo 650 le lena", "dose": None}]}
    note = _build_note(data, transcript="[UNKNOWN]: dolo 650 le lena subah")
    assert note.medications[0].drug == "dolo 650"


def test_unmatched_garbled_drug_name_is_never_truncated() -> None:
    """Conservative guard: no inner match exists, so the whole string stays —
    trimming an unmatched name would silently discard information."""
    data = {"medications": [{"drug": "एक झिनझिनझान 500 एक", "dose": None}]}
    note = _build_note(
        data, transcript="[UNKNOWN]: एक झिनझिनझान 500 एक खा लो"
    )
    assert note.medications[0].drug == "एक झिनझिनझान 500 एक"


# ── (8) हफते/Shortness-of-Breath near-collision, outputs/<session>,
# 2026-07-12 ─────────────────────────────────────────────────────────────
# Root cause: transcript "एक हफते के लिए" ("for one week") was glossed
# "(Shortness of Breath)" — हफते ("week") is one edit from हांफते
# ("huffing/panting", a genuine near-synonym of this concept), and the
# embedding model conflated the two. Fixed by adding हफ्ता/हफ्ते/हफते/हफ़्ते as
# hard_negatives on the Shortness of Breath concept (src/concepts.py). The
# full behavioral check requires the real parrotlet-e embedding model and
# lives in tests/test_l3_5_normalize.py under @pytest.mark.slow
# (test_haphte_week_not_glossed_as_shortness_of_breath /
# test_haanphte_genuine_case_still_glosses); this is the fast, model-free
# regression guard that the DATA fix itself never regresses.


def test_haphte_family_present_as_shortness_of_breath_hard_negative() -> None:
    sob = next(c for c in CONCEPTS if c.term == "Shortness of Breath")
    for week_word in ("हफ्ता", "हफ्ते", "हफते", "हफ़्ते"):
        assert week_word in sob.hard_negatives


def test_haphte_family_also_blocked_by_everyday_word_guard() -> None:
    """Concept Matcher Rebuild Phase (2026-07-12): this incident is now
    caught by TWO independent mechanisms — the per-concept hard negative
    above, AND the categorical EVERYDAY_WORDS guard (src/concepts.py),
    which blocks हफ्ता/हफ्ते regardless of similarity to ANY concept, not
    just Shortness of Breath. See tests/test_concept_guard.py."""
    for week_word in ("हफ्ता", "हफ्ते", "हफते", "हफ़्ते"):
        assert week_word in EVERYDAY_WORDS


# ── (9) drug name in diagnosis/investigations — same screenshots as (6)/(7),
# 2026-07-12 ─────────────────────────────────────────────────────────────
# Root cause: "one naxdom 500 one" (the same filler-glued drug span) appeared
# not only in medications but also in the DIAGNOSIS and INVESTIGATIONS
# fields of the same extraction. A drug name is not a diagnosis or a test —
# but it also isn't safe to silently move or delete (the doctor may have
# meant something else entirely by that row). Fixed with a flag-only,
# symmetric check: a diagnosis term or investigation string that
# fold-matches a known drug (after stripping filler/dose tokens, same tiers
# as (7)) is flagged, never altered.


def test_drug_name_in_diagnosis_flagged_not_removed() -> None:
    data = {"diagnosis": [{"term": "one naxdom 500 one"}]}
    note = _build_note(data)
    assert note.diagnosis[0].term == "one naxdom 500 one"  # never moved/deleted
    assert "diagnosis.one naxdom 500 one.drug_in_diagnosis" in note.low_confidence_fields


def test_drug_name_in_investigations_flagged_not_removed() -> None:
    data = {"investigations": ["one naxdom 500 one"]}
    note = _build_note(data)
    assert note.investigations == ["one naxdom 500 one"]  # never moved/deleted
    assert (
        "investigations.one naxdom 500 one.drug_in_investigations"
        in note.low_confidence_fields
    )


# ── (10) Hindi frequency phrase surfaced verbatim in the note, outputs/
# <session>, 2026-07-12 ────────────────────────────────────────────────────
# Root cause: the model correctly extracted frequency "दो बार दिन में"
# verbatim from a Hindi transcript (rule 6: use the transcript's own
# language) — but the note's frequency/timing fields are excluded from the
# translate route by design (dosing schedules are patient-safety text and
# must never be LLM-translated), so an English-reading physician saw only
# the Hindi phrase. Fixed with a deterministic, exact-match canonicalization
# table in L4 (never fuzzy, never model-based): recognized phrases (English
# and Hindi) map to one canonical English display phrase; clinical notation
# ("1-0-1", "BD", ...) and any unrecognized phrase pass through unchanged.


def test_hindi_frequency_phrase_canonicalizes_to_english() -> None:
    data = {
        "medications": [
            {"drug": "paracetamol", "dose": "500 mg", "frequency": "दो बार दिन में"}
        ]
    }
    note = _build_note(data, transcript="[UNKNOWN]: paracetamol 500 mg दो बार दिन में")
    assert note.medications[0].frequency == "twice a day"


def test_notation_frequency_passes_through_unchanged() -> None:
    data = {
        "medications": [{"drug": "augmentin", "dose": "625 mg", "frequency": "1-0-1"}]
    }
    note = _build_note(data, transcript="[UNKNOWN]: augmentin 625 mg 1-0-1")
    assert note.medications[0].frequency == "1-0-1"


def test_unrecognized_frequency_phrase_passes_through_unchanged() -> None:
    data = {
        "medications": [
            {"drug": "augmentin", "dose": "625 mg", "frequency": "कुछ अजीब सा"}
        ]
    }
    note = _build_note(data, transcript="[UNKNOWN]: augmentin 625 mg कुछ अजीब सा")
    assert note.medications[0].frequency == "कुछ अजीब सा"


def test_hindi_timing_phrase_canonicalizes_to_english() -> None:
    data = {
        "medications": [
            {"drug": "paracetamol", "dose": "500 mg", "timing": "खाने के बाद"}
        ]
    }
    note = _build_note(data, transcript="[UNKNOWN]: paracetamol 500 mg खाने के बाद")
    assert note.medications[0].timing == "after food"


# ── (11) Devanagari drug names stayed Devanagari and unvalidated, outputs/
# 20260712-194649-763e13, 2026-07-12 ────────────────────────────────────────
# Root cause: src/drug_lexicon.py's fold turned Devanagari into a PHONETIC
# Latin skeleton, but the Latin lexicon keys kept ENGLISH ORTHOGRAPHY ('x'
# never became 'ks'; 'th'/'ph'/'ch'/'c'/'y' never normalized) — a real drug
# spoken in Devanagari (नौरफलोक्स, एजित्रोमाइसिन) folded nowhere near its own
# English lexicon entry (norflox, azithromycin). Fixed with a Latin-
# orthography normalization pass, applied identically to BOTH sides, that
# was previously duplicated three ways (src/drug_lexicon.py,
# src/l4_extract.py, eval/drug_bench.py) and is now ONE shared fold (see
# tests/test_fold_parity.py) — required, not optional: the display fix
# below replaces a Devanagari drug name with its canonical Latin spelling
# BEFORE the grounding guard checks it against the (still-Devanagari)
# transcript, so the guard's own fold must be the SAME fold that resolved
# the name, or a correctly-resolved real drug gets wrongly demoted to an
# "unnamed medication" row.


def test_norflox_devanagari_resolves_and_canonicalizes_for_display() -> None:
    """नौरफलोक्स -> norflox via the expanded-lexicon fuzzy fold tier (0.7778
    confidence, below 1.0) — fuzzy display substitutions get their own
    low-confidence flag (advisor ruling 4), distinct from .unvalidated."""
    assert canonicalize_drug_span("नौरफलोक्स") == ("norflox", 0.7778)

    data = {"medications": [{"drug": "नौरफलोक्स", "dose": None}]}
    note = _build_note(
        data,
        transcript="[UNKNOWN]: नौरफलोक्स, रात में एक बार सोने से पहले, ठीक है?",
    )
    assert note.medications[0].drug == "norflox"
    assert note.medications[0].validated is True
    assert "medications.norflox.canonicalized_fuzzy" in note.low_confidence_fields


def test_azithromycin_no_aspirate_spelling_canonicalizes_with_dose() -> None:
    """एजित्रोमाइसिन 500 -> azithromycin 500, dose digit preserved.

    This specific ASR spelling (missing the थ aspirate AND the ज़ nukta) is
    NOT recovered by the general fuzzy-fold tier: its fold key sits within
    the fuzzy bound of BOTH azithromycin and erythromycin (two genuinely
    distinct antibiotics) — a real ambiguity from the missing information,
    not a rule gap. canonicalize_drug_span() correctly declines (returns
    None) rather than guess; the fix is a curated-table entry (the same
    zero-risk mechanism the नैक्सडॉम family already uses for a known, seen
    distortion), checked before the fuzzy tier.
    """
    assert canonicalize_drug_span("एजित्रोमाइसिन") is None  # ambiguous, declined

    result = _normalize_drug_text("एजित्रोमाइसिन 500, दिन में दो बार")
    assert "azithromycin 500" in result

    data = {"medications": [{"drug": "एजित्रोमाइसिन 500", "dose": None}]}
    note = _build_note(
        data, transcript="[UNKNOWN]: एजित्रोमाइसिन 500, दिन में दो बार,"
    )
    assert note.medications[0].drug == "azithromycin 500"
    assert note.medications[0].validated is True
    assert not any(
        f.endswith(".canonicalized_fuzzy") for f in note.low_confidence_fields
    )


def test_canonical_display_survives_the_grounding_guard_against_devanagari() -> None:
    """The advisor's blocker scenario, as a regression test: _isolate_drug_span
    canonicalizes a filler-glued Devanagari span to its Latin display name
    BEFORE the grounding guard runs (src/l4_extract.py's medications loop).
    With three separately-hand-maintained folds, the guard's OWN fold could
    disagree with the one that resolved the name, folding the Latin
    canonical far from the still-Devanagari transcript and wrongly demoting
    a correctly-resolved real drug to "unnamed medication N". With one
    shared fold (src/drug_lexicon.py's, imported by src/l4_extract.py), the
    guard agrees with the resolution and the row survives.
    """
    data = {"medications": [{"drug": "एक एजित्रोमाइसिन 500 एक", "dose": None}]}
    note = _build_note(
        data, transcript="[UNKNOWN]: एक एजित्रोमाइसिन 500 एक खा लो"
    )
    assert note.medications[0].drug == "azithromycin 500"
    assert not any(f.endswith(".ungrounded") for f in note.low_confidence_fields)

    data2 = {"medications": [{"drug": "एक नौरफलोक्स एक", "dose": None}]}
    note2 = _build_note(data2, transcript="[UNKNOWN]: एक नौरफलोक्स एक खा लो")
    assert note2.medications[0].drug == "norflox"
    assert not any(f.endswith(".ungrounded") for f in note2.low_confidence_fields)
