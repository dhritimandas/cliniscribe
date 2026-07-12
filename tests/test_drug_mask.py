"""Tests for web/drug_mask.py — deterministic drug-span masking around
machine translation.

Root cause under test: outputs/20260712-194649-763e13, where a translator
LLM asked (only in the prompt, not enforced) to preserve drug names verbatim
phonetically guessed "acetaminophen" for एजित्रोमाइसिन (azithromycin) and
"naloxone" for नौरफलोक्स (norflox). These tests never call Ollama — masking/
restoring is pure string logic; the translator itself is exercised in
tests/test_web_app.py with a stubbed `ollama.chat`.
"""

from web.drug_mask import DrugSpan, find_drug_spans, mask, restore

# ── find_drug_spans: detection tiers ───────────────────────────────────────


def test_verbatim_medication_match() -> None:
    """A medication name mentioned verbatim in Latin script is found."""
    spans = find_drug_spans("Take Azithral once daily.", medication_drugs=["Azithral"])
    assert len(spans) == 1
    assert spans[0].original == "Azithral"
    assert spans[0].canonical == "Azithral"


def test_cross_script_fuzzy_match_against_medication_list() -> None:
    """The real incident: एजित्रोमाइसिन (an ASR spelling missing the थ aspirate
    and ज़ nukta) is globally ambiguous between azithromycin and erythromycin
    (canonicalize_drug_span correctly declines it, see src/drug_lexicon.py),
    but the note itself only prescribed azithromycin — restricting the fuzzy
    candidate set to the note's own medications resolves it unambiguously."""
    from src.drug_lexicon import canonicalize_drug_span

    assert canonicalize_drug_span("एजित्रोमाइसिन") is None  # ambiguous globally

    spans = find_drug_spans(
        "एजित्रोमाइसिन 500, दिन में दो बार", medication_drugs=["azithromycin"]
    )
    assert len(spans) == 1
    assert spans[0].canonical == "azithromycin"
    assert spans[0].dose_digits == "500"
    assert spans[0].original == "एजित्रोमाइसिन 500"


def test_lexicon_only_drug_not_in_medications_list() -> None:
    """A drug mentioned in free text that never made it into the medications
    list (e.g. only in advice/transcript) is still caught via the general
    lexicon tier — नौरफलोक्स -> norflox, no medication list involved."""
    spans = find_drug_spans("नौरफलोक्स, रात में एक बार सोने से पहले", medication_drugs=())
    assert len(spans) == 1
    assert spans[0].canonical == "norflox"
    assert spans[0].original == "नौरफलोक्स"


def test_no_medication_list_and_globally_ambiguous_spelling_is_unmatched() -> None:
    """Without a medication list to disambiguate, a globally ambiguous
    spelling is correctly left unmatched (no span at all) — never guessed."""
    spans = find_drug_spans("एजित्रोमाइसिन 500", medication_drugs=())
    assert spans == ()


def test_devanagari_medication_value_does_not_shadow_a_better_lexicon_match() -> None:
    """Regression (outputs/20260712-194649-763e13, pre-previous-wave state):
    a medication whose OWN drug field is still Devanagari (not yet
    canonicalized) cannot function as "the resolved canonical name" — it
    must not be used for tier-1 matching at all, only Latin-script
    medication values are. Before this guard, a stray raw-Devanagari
    medication value could fuzzy-match a slightly LONGER window (e.g.
    including an adjacent "एक") than the general lexicon's correct,
    shorter match, and longest-span-wins would then restore the raw
    Devanagari text (a no-op) instead of the lexicon's real "norflox"."""
    text = "और एक नौरफलोक्स, रात में एक बार"
    spans = find_drug_spans(text, medication_drugs=["नौरफलोक्स"])
    assert len(spans) == 1
    assert spans[0].canonical == "norflox"
    assert spans[0].original == "नौरफलोक्स"


def test_dose_digits_included_in_span_and_restored() -> None:
    spans = find_drug_spans("azithromycin 500 daily", medication_drugs=["azithromycin"])
    assert len(spans) == 1
    assert spans[0].original == "azithromycin 500"
    assert spans[0].dose_digits == "500"

    m = mask("azithromycin 500 daily", medication_drugs=["azithromycin"])
    assert m.masked == "DRUGSPAN0 daily"
    restored = restore("DRUGSPAN0 daily", m.spans)
    assert restored == "azithromycin 500 daily"


def test_dose_digit_far_away_is_not_attached() -> None:
    """A digit several tokens later is not swept into the drug span."""
    spans = find_drug_spans(
        "azithromycin daily for 500 rupees", medication_drugs=["azithromycin"]
    )
    assert len(spans) == 1
    assert spans[0].dose_digits is None
    assert spans[0].original == "azithromycin"


def test_longest_span_wins_on_overlap() -> None:
    """"augmentin duo" (a 2-word lexicon entry) wins over the shorter
    "augmentin"-only match at the same starting position."""
    spans = find_drug_spans("Take augmentin duo twice daily.", medication_drugs=())
    assert len(spans) == 1
    assert spans[0].original == "augmentin duo"
    assert spans[0].canonical == "augmentin duo"


def test_no_spans_in_plain_text() -> None:
    assert find_drug_spans("मुझे बुखार है", medication_drugs=()) == ()
    assert find_drug_spans("", medication_drugs=()) == ()


# ── mask / restore: placeholder protocol ───────────────────────────────────


def test_mask_produces_indexed_placeholders_in_order() -> None:
    m = mask(
        "paracetamol khaana, azithromycin 500 lena",
        medication_drugs=["paracetamol", "azithromycin"],
    )
    assert m.masked == "DRUGSPAN0 khaana, DRUGSPAN1 lena"
    assert len(m.spans) == 2


def test_restore_happy_path() -> None:
    m = mask("Take azithromycin 500 daily.", medication_drugs=["azithromycin"])
    translated = "रोज़ DRUGSPAN0 लें।"
    assert restore(translated, m.spans) == "रोज़ azithromycin 500 लें।"


def test_restore_reordered_placeholder_is_ok() -> None:
    """Each index appearing exactly once is all that matters — order in the
    translated sentence may legitimately differ (e.g. word-order swaps)."""
    m = mask(
        "paracetamol and azithromycin 500",
        medication_drugs=["paracetamol", "azithromycin"],
    )
    reordered = "DRUGSPAN1 and DRUGSPAN0"
    assert restore(reordered, m.spans) == "azithromycin 500 and paracetamol"


def test_restore_dropped_placeholder_is_violation() -> None:
    m = mask(
        "paracetamol and azithromycin 500",
        medication_drugs=["paracetamol", "azithromycin"],
    )
    # DRUGSPAN1 missing entirely.
    dropped = "DRUGSPAN0 and something else"
    assert restore(dropped, m.spans) is None


def test_restore_duplicated_placeholder_is_violation() -> None:
    m = mask(
        "paracetamol and azithromycin 500",
        medication_drugs=["paracetamol", "azithromycin"],
    )
    # DRUGSPAN0 appears twice, DRUGSPAN1 never appears — a naive count-only
    # check (2 placeholders total, matches spans count) would miss this.
    duplicated = "DRUGSPAN0 and DRUGSPAN0"
    assert restore(duplicated, m.spans) is None


def test_restore_no_placeholders_survive_at_all_is_violation() -> None:
    """The exact incident failure mode: the model translates straight through
    and no placeholder survives — must be treated as a violation, not a
    silent pass-through."""
    m = mask("Take azithromycin 500 daily.", medication_drugs=["azithromycin"])
    assert restore("Take acetaminophen daily.", m.spans) is None


def test_restore_with_no_spans_returns_translated_unchanged() -> None:
    assert restore("कोई दवा नहीं", ()) == "कोई दवा नहीं"


def test_restore_canonical_for_resolved_span() -> None:
    span = DrugSpan(start=0, end=5, original="xxxxx", canonical="azithromycin")
    assert restore("DRUGSPAN0", (span,)) == "azithromycin"


def test_restore_original_verbatim_for_unresolved_span() -> None:
    """Restore rule for a drug-shaped span that could not be safely
    identified: show the original text verbatim, never invent a name. No
    live detection tier currently emits `canonical=None` (both tiers only
    ever produce a span when they HAVE resolved a canonical name) — this
    exercises the `restore()` contract directly, as the defensive branch it
    is."""
    span = DrugSpan(start=0, end=6, original="किसी दवा", canonical=None)
    assert restore("DRUGSPAN0", (span,)) == "किसी दवा"
