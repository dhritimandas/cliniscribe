"""L4 — Clinical entity extraction via Qwen2.5-3B-Instruct (Ollama)."""

import difflib
import json
import logging
import re
import unicodedata

from src import config
from src.cdsco import validate_drug
from src.concepts import CONCEPTS
from src.drug_lexicon import DRUG_LEXICON, canonicalize_drug_span
from src.l3_5_normalize import _DEVA_CURATED, _LATIN_CURATED
from src.types import ClinicalNote, Diagnosis, Medication, Symptom, Turn, Vital

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a clinical documentation assistant. Extract structured medical information \
from the consultation transcript below.

Output ONLY valid JSON matching this exact schema — no extra text, no markdown:
{
  "chief_complaint": "string or null",
  "history": "string or null",
  "symptoms": [{"name": "string", "finding_status": "Present | Absent", \
"severity": "Mild | Moderate | Severe or null", "since": "string or null"}],
  "vitals": [{"name": "string", "value": "string with unit"}],
  "examination": "string or null",
  "diagnosis": [{"term": "string", "snomed_id": "string or null", \
"status": "Confirmed | Suspected | Ruled out or null"}],
  "medications": [{"drug": "string", "dose": "string or null", \
"frequency": "string or null", "timing": "string or null", \
"duration": "string or null"}],
  "investigations": ["string"],
  "diagnostic_results": ["string"],
  "advice": "string or null",
  "follow_up": "string or null",
  "low_confidence_fields": ["string"]
}

RULES — follow every rule exactly:
1. Set a field to null if it is NOT explicitly stated. Never infer or invent values.
2. DOSE must be null unless the doctor stated a specific dose in the transcript. \
Do not supply a standard or typical dose. null means unknown.
3. Add the field name to low_confidence_fields whenever you are uncertain about \
a value or the value is absent but clinically expected.
4. If the transcript already contains a clinical synonym in parentheses \
— a lay term followed by its clinical name, e.g. "<lay term> (<clinical term>)" \
— extract that parenthetical term. Do NOT add parenthetical clinical terms \
yourself — only use what is explicitly written in the transcript. This rule \
exists for L3.5-normalised transcripts; if no parenthetical is present, \
extract only what is said.
5. Extract only what is spoken. Do not add clinical knowledge not present \
in the transcript.
6. Use the language of the transcript for text fields. Do not translate.
7. SYMPTOMS are complaints the patient reports (pain, fever, nausea). \
finding_status is "Present" by default, "Absent" only when a symptom is \
explicitly denied (e.g. "no vomiting"). severity and since are null unless stated.
8. VITALS are measured signs with a number and unit (BP, pulse, SpO2, \
temperature, weight). Only include a vital when a measured value is spoken. \
Never invent a measurement.
9. INVESTIGATIONS are tests the doctor ORDERS for later (e.g. "get a CBC"). \
DIAGNOSTIC_RESULTS are results already available in the consultation — a test \
name paired with its stated value or qualitative finding, exactly as spoken. \
Extract each as a separate string. Do not put ordered tests in \
diagnostic_results; do not put lab results in history. Never copy a value \
from these instructions — only extract values actually spoken in the \
transcript below.
10. examination is free text describing physical-exam findings \
(e.g. "abdomen soft, mild tenderness"). null if no exam is described.
11. FREQUENCY is the dosing schedule — how often and when during the day: \
"once daily", "twice daily", "BD", "TDS", "SOS", "once at night", \
"once in the morning", "1-0-1". Extract it whenever a dosing schedule is stated. \
TIMING is ONLY for meal-relative context: "before food", "after food", "with food". \
Time-of-day phrases ("at night", "SOS", "in the morning") and dosing notation \
("1-0-0", "BD/SOS") belong in frequency, not timing.
12. DRUG names must be copied EXACTLY as written in the transcript — the same \
characters, in the same script. Never transliterate, translate, or re-spell a \
drug name into a different script or a different spelling than what appears \
in the transcript.\
"""


def _turns_to_text(turns: list[Turn]) -> str:
    return "\n".join(f"[{t.speaker_role}]: {t.text}" for t in turns)


def _strip_fences(raw: str) -> str:
    """Remove markdown code fences that models sometimes add despite format=json."""
    s = raw.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines)
    return s.strip()


def _parse(raw: str) -> dict:
    return json.loads(_strip_fences(raw))


# Generic terms that must never populate a drug NAME on a prescription:
# "the doctor gave medicine" is not a prescribable entry. Matching is
# whole-string after normalisation ("syrup" is generic; "Benadryl cough
# syrup" is not). Devanagari "X की दवा(ई)" phrases are caught by the
# generic-head rule in _is_generic_drug_name.
_GENERIC_DRUG_TERMS: frozenset[str] = frozenset(
    {
        "medicine", "medicines", "medication", "medications",
        "tablet", "tablets", "capsule", "capsules",
        "syrup", "cough syrup", "injection", "gel", "cream",
        "ointment", "drops", "painkiller", "pain killer",
        "nasal spray", "spray",
        "dawai", "dawa",
        "दवाई", "दवा", "दवाइयां", "टैबलेट", "जेल", "इंजेक्शन", "सिरप",
    }
)
# Devanagari generic head nouns: a name ENDING in one of these ("डायबिटीज की
# दवाई" = "diabetes medicine") is generic unless some token validates in CDSCO.
_GENERIC_HEAD_TOKENS: frozenset[str] = frozenset({"दवाई", "दवा", "दवाइयां"})


def _is_generic_drug_name(drug: str) -> bool:
    """Return True if the extracted drug string names no identifiable drug."""
    norm = " ".join(drug.lower().split())
    if norm in _GENERIC_DRUG_TERMS:
        return True
    tokens = norm.split()
    if tokens and tokens[-1] in _GENERIC_HEAD_TOKENS:
        # "X की दवाई" — generic unless any token is a real CDSCO drug.
        return not any(validate_drug(t) for t in tokens if len(t) >= 4)
    return False


_TOKEN_RE = re.compile(r"[a-z]+", re.IGNORECASE)
_MIN_OVERLAP_TOKEN_LEN = 4  # ignore short words (conjunctions, articles, etc.)
# Speaker-role prefix tokens that appear in every turn and must be excluded
# from the hallucination-overlap check.
_ROLE_TOKENS: frozenset[str] = frozenset({"doctor", "patient", "unknown"})


def _transcript_tokens(transcript: str) -> set[str]:
    """Return lowercase alpha tokens ≥ _MIN_OVERLAP_TOKEN_LEN, excluding role prefixes."""
    return (
        {t.lower() for t in _TOKEN_RE.findall(transcript) if len(t) >= _MIN_OVERLAP_TOKEN_LEN}
        - _ROLE_TOKENS
    )


def _diagnosis_has_overlap(term: str, transcript_tokens: set[str]) -> bool:
    """Return True if ≥1 token of the diagnosis term appears in the transcript.

    When transcript_tokens is empty (which happens for Devanagari-only transcripts
    after stripping role tokens), returns True — conservative, no false flagging.
    Limitation: cross-lingual mapping (Hindi term → English diagnosis) is not
    assessed; Devanagari transcripts are skipped entirely.
    """
    if not transcript_tokens:
        return True  # no Latin content to compare against; skip flagging
    term_tokens = {t.lower() for t in _TOKEN_RE.findall(term) if len(t) >= _MIN_OVERLAP_TOKEN_LEN}
    if not term_tokens:
        return True  # diagnosis term has no long tokens; can't assess
    return bool(term_tokens & transcript_tokens)


# ── Drug-name source-fidelity restoration (deterministic backstop) ─────────
# qwen2.5:3b sometimes re-spells a Latin-script drug name spoken in the
# transcript into Devanagari, or otherwise distorts its spelling, instead of
# copying it verbatim (real case: transcript "naxdom 500" -> LLM "नक्सडम 500").
# CDSCO validation then fails on an invented spelling. This is a source-
# fidelity restoration, not a lexicon lookup: we only ever substitute text
# that was actually said in the transcript, so there is no wrong-drug
# substitution risk the way there would be with a fuzzy drug-list match.

# Adapted from eval/drug_bench.py's _fold (reimplemented here, not imported —
# eval/ and src/ are separate module boundaries). "क्स"/"क्श" are common
# Devanagari digraphs for the English "x" sound in loanwords (टैक्स, बॉक्स,
# एक्स-रे); folded to "x" before the per-character pass, since the
# per-character map alone renders them as "ks", which folds far from the
# Latin spelling of names that use "x".
_DRUG_FOLD_DIGRAPHS: tuple[tuple[str, str], ...] = (("क्स", "x"), ("क्श", "x"))
_DRUG_FOLD_MAP: dict[str, str] = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "च": "ch", "छ": "chh",
    "ज": "j", "झ": "jh", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p",
    "फ": "f", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r",
    "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "ज़": "z", "फ़": "f", "ा": "a", "ि": "i", "ी": "i", "ु": "u",
    "ू": "u", "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ं": "n",
    "अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "्": "",
    # Candra vowels + vocalic r + candrabindu/visarga — see src/drug_lexicon.py's
    # _FOLD_MAP for the same addition and tests/test_fold_parity.py for the
    # cross-implementation parity guard.
    "ॉ": "o", "ॅ": "e", "ृ": "ri", "ऑ": "o", "ऍ": "e", "ँ": "n", "ः": "",
}
_DRUG_FOLD_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]")
_DRUG_WINDOW_SIZES: tuple[int, ...] = (1, 2, 3, 4)
_DRUG_RESTORE_MIN_SIMILARITY = 0.80
_ROLE_TAG_RE = re.compile(r"\[(?:DOCTOR|PATIENT|UNKNOWN)\]:\s*")


def _fold_drug(text: str) -> str:
    """Coarse phonetic fold: Devanagari→Latin, lowercase, alnum+space only."""
    text = unicodedata.normalize("NFC", text)
    for digraph, latin in _DRUG_FOLD_DIGRAPHS:
        text = text.replace(digraph, latin)
    folded = "".join(_DRUG_FOLD_MAP.get(ch, ch) for ch in text)
    return _DRUG_FOLD_NON_ALNUM_RE.sub("", folded.lower())


def _transcript_word_windows(transcript: str) -> list[str]:
    """Return every contiguous window (sizes 1-4) of transcript surface words.

    Role-tag prefixes ("[DOCTOR]: ") are stripped first so they never enter a
    window's surface text.
    """
    words = _ROLE_TAG_RE.sub("", transcript).split()
    windows: list[str] = []
    for n in _DRUG_WINDOW_SIZES:
        for i in range(len(words) - n + 1):
            windows.append(" ".join(words[i : i + n]))
    return windows


def _restore_drug_spelling(drug: str, transcript: str) -> str:
    """Replace an LLM-re-spelled drug string with the transcript's own spelling.

    Returns `drug` unchanged when it already fold-matches a transcript window
    exactly, or when no window reaches fold-similarity >= 0.80 (nothing safe
    to substitute). Otherwise returns the transcript's own surface text for
    the best-matching window. Digits in `drug` (a dose glued onto the drug
    name) are always preserved, even if the matched window doesn't include
    them — never lose dose info to a restoration.
    """
    windows = _transcript_word_windows(transcript)
    if not windows:
        return drug

    drug_fold_despaced = _fold_drug(drug).replace(" ", "")
    if drug_fold_despaced and any(
        _fold_drug(w).replace(" ", "") == drug_fold_despaced for w in windows
    ):
        return drug  # already an exact fold match — nothing to restore

    best_window, best_ratio = "", 0.0
    drug_fold = _fold_drug(drug)
    for window in windows:
        ratio = difflib.SequenceMatcher(None, drug_fold, _fold_drug(window)).ratio()
        if ratio > best_ratio:
            best_window, best_ratio = window, ratio

    if best_ratio < _DRUG_RESTORE_MIN_SIMILARITY:
        return drug  # nothing close enough in the transcript — keep as-is

    digit_tokens = re.findall(r"\d+", drug)
    missing_digits = [d for d in digit_tokens if d not in re.findall(r"\d+", best_window)]
    if missing_digits:
        return f"{best_window} {' '.join(missing_digits)}".strip()
    return best_window


# ── Drug span isolation (filler-word-glue backstop) ─────────────────────────
# Real incident (outputs/<session>, 2026-07-12): the extractor returned
# drug="एक नैक्स्टॉम 500 एक" ("one naxdom 500 one") — Hindi filler words glued
# onto both ends of a real drug span by the LLM. Conservative by design: this
# only TRIMS when an inner span (after stripping filler tokens) resolves via
# one of the three read-only recognition tiers below; an unmatched/garbled
# name is never truncated (a false trim would silently drop information a
# reviewing physician can't recover from the note).
_DRUG_FILLER_TOKENS: frozenset[str] = frozenset(
    {"एक", "one", "और", "and", "भी", "ले", "लेना", "खा", "खाना", "le", "lena"}
)


def _is_drug_filler_token(token: str) -> bool:
    return token.lower() in _DRUG_FILLER_TOKENS


def _strip_filler_edges(tokens: list[str]) -> list[str]:
    """Return `tokens` with leading/trailing filler words removed (never the
    middle — filler-glue in this incident class only wraps a span, it doesn't
    interrupt it)."""
    start, end = 0, len(tokens)
    while start < end and _is_drug_filler_token(tokens[start]):
        start += 1
    while end > start and _is_drug_filler_token(tokens[end - 1]):
        end -= 1
    return tokens[start:end]


def _resolve_known_drug_name(name_span: str) -> str | None:
    """Read-only drug recognition: curated table -> expanded lexicon -> CDSCO.

    Same three tiers L3.5's drug pass uses (src/l3_5_normalize.py), reused
    here only to ANSWER "does this span name a known drug" — never to
    re-spell a drug name L4 itself extracted (rule 12 forbids that); callers
    decide what to do with a hit.
    """
    curated = _DEVA_CURATED.get(name_span) or _LATIN_CURATED.get(name_span.lower())
    if curated:
        return curated
    lexicon_hit = canonicalize_drug_span(name_span)
    if lexicon_hit is not None:
        return lexicon_hit[0]
    if validate_drug(name_span):
        return name_span
    return None


def _names_a_known_drug(text: str) -> bool:
    """True if `text` (after stripping filler words and dose digits) resolves
    to a known drug — used to flag (never move or delete) a drug name that
    surfaced in diagnosis or investigations instead of medications."""
    name_tokens = [t for t in _strip_filler_edges(text.split()) if not t.isdigit()]
    if not name_tokens:
        return False
    return _resolve_known_drug_name(" ".join(name_tokens)) is not None


def _isolate_drug_span(drug: str) -> str:
    """Trim filler-word glue around a drug name, only when an inner match exists.

    Strips leading/trailing filler tokens, then resolves the remaining
    non-digit tokens against `_resolve_known_drug_name`. On a hit, returns the
    resolved canonical name plus any digit tokens from the trimmed span (dose
    survives). On no hit — either nothing to strip, or the inner span doesn't
    resolve either — returns `drug` unchanged.
    """
    tokens = drug.split()
    if len(tokens) <= 1:
        return drug

    core = _strip_filler_edges(tokens)
    if core == tokens:
        return drug  # no filler at either edge -- nothing to isolate

    name_tokens = [t for t in core if not t.isdigit()]
    digit_tokens = [t for t in core if t.isdigit()]
    if not name_tokens:
        return drug

    canonical = _resolve_known_drug_name(" ".join(name_tokens))
    if canonical is None:
        return drug  # inner span doesn't resolve either -- never truncate

    return " ".join([canonical, *digit_tokens]) if digit_tokens else canonical


# ── Condition guard (spurious medication-row backstop) ─────────────────────
# Real incident (outputs/20260712-124506-715247, 2026-07-12): the patient
# said "मेरा BP (Hypertension) भी हाई है" — L3.5's concept-glosser correctly
# annotated "BP" with its clinical gloss "(Hypertension)" — but qwen2.5:3b
# then extracted the parenthetical itself, "(Hypertension) भी", as a
# MEDICATION drug name. The grounding guard above correctly passed it (the
# string IS in the transcript; grounding catches inventions, not category
# errors). This is a category error, not a fabrication: a clinical
# CONDITION, not a drug, and the information is already captured elsewhere
# in the note (history/diagnosis) — so the row is dropped entirely rather
# than converted to an unnamed row (unlike a generic term, a condition row
# carries no salvageable prescription info).
_CONDITION_PARTICLES: frozenset[str] = frozenset({"भी", "है", "का", "की", "को"})
_CONDITION_PUNCT_RE = re.compile(r"[()।॥.,!?]")
_CONDITION_PAREN_RE = re.compile(r"\(([^()]+)\)")

# Modest supplement of common condition words not necessarily covered by
# CONCEPTS' variants — CONCEPTS (src/concepts.py) does the heavy lifting.
_CONDITION_SUPPLEMENT: frozenset[str] = frozenset({
    "hypertension", "diabetes", "fever", "cough", "cold", "asthma",
    "migraine", "anemia", "arthritis", "allergy", "infection",
    "बुखार", "खांसी", "जुकाम", "दमा", "माइग्रेन", "एनीमिया", "गठिया",
    "एलर्जी", "संक्रमण", "उच्च रक्तचाप", "मधुमेह",
})


def _strip_condition_noise(text: str) -> str:
    """Strip gloss punctuation/parens and trailing Hindi particles.

    "(Hypertension) भी" -> "Hypertension" — isolates the clinical term from
    surrounding gloss punctuation and grammatical particles a concept-gloss
    or LLM extraction may carry along with it.
    """
    despunctuated = _CONDITION_PUNCT_RE.sub(" ", text)
    tokens = [t for t in despunctuated.split() if t not in _CONDITION_PARTICLES]
    return " ".join(tokens)


def _build_condition_terms() -> frozenset[str]:
    """Fold-normalized condition-term set: CONCEPTS terms/variants + supplement."""
    terms: set[str] = set(_CONDITION_SUPPLEMENT)
    for concept in CONCEPTS:
        terms.add(concept.term)
        terms.update(concept.variants)
    folded: set[str] = set()
    for term in terms:
        key = _fold_drug(term).replace(" ", "")
        if key:
            folded.add(key)
    return frozenset(folded)


def _build_drug_lexicon_folds() -> frozenset[str]:
    """Fold-normalized keys of every canonical drug name (precedence check)."""
    folded: set[str] = set()
    for drug in DRUG_LEXICON:
        key = _fold_drug(drug).replace(" ", "")
        if key:
            folded.add(key)
    return frozenset(folded)


# Built once at import — CONCEPTS/DRUG_LEXICON are static tables, no per-call cost.
_CONDITION_TERMS: frozenset[str] = _build_condition_terms()
# Drug-lexicon membership takes precedence over the condition set (see
# _is_condition_term): a brand/generic name that happens to fold-match a
# condition term is never dropped. No collision exists in the current
# 546-entry lexicon (test_no_drug_lexicon_entry_matches_condition_set), but
# the precedence check stays as a structural guarantee, not a fact about
# today's lexicon contents.
_DRUG_LEXICON_FOLDS: frozenset[str] = _build_drug_lexicon_folds()


def _is_condition_term(drug: str) -> bool:
    """Return True if `drug` names a clinical condition, not a medication.

    Checks the whole string (after stripping gloss punctuation/particles)
    for an exact fold match against the condition set, then any
    parenthetical gloss content within it (e.g. "BP (Hypertension)"). Drug-
    lexicon membership always wins: a real drug name is never dropped even
    if it also happens to fold-match a condition term.
    """
    cleaned_fold = _fold_drug(_strip_condition_noise(drug)).replace(" ", "")
    if cleaned_fold in _DRUG_LEXICON_FOLDS:
        return False
    if cleaned_fold in _CONDITION_TERMS:
        return True
    for m in _CONDITION_PAREN_RE.finditer(drug):
        gloss_fold = _fold_drug(_strip_condition_noise(m.group(1))).replace(" ", "")
        is_condition = gloss_fold and gloss_fold in _CONDITION_TERMS
        if is_condition and gloss_fold not in _DRUG_LEXICON_FOLDS:
            return True
    return False


# ── Grounding guard + dose-provenance flag (invented-drug-name backstop) ────
# _restore_drug_spelling recovers a drug name the LLM MIS-spelled but actually
# said (source-fidelity restoration). This guard catches the other failure
# mode: the LLM INVENTING a drug name with no plausible source in the
# transcript at all (real case: transcript never says "nasal spray"; the LLM
# fabricated it after failing to normalize a distorted brand name). No
# restoration is safe here — there is nothing in the transcript to restore
# from — so the row is converted to the numbered "unnamed medication N" form
# instead, the same machinery the generic-term guard already uses.
_DRUG_GROUND_MIN_SIMILARITY = 0.60
_DOSE_PROXIMITY_TOKENS = 6
_DOSE_DIGIT_RE = re.compile(r"\d+")


def _transcript_words(transcript: str) -> list[str]:
    """Role-tag-stripped surface word tokens of a transcript string."""
    return _ROLE_TAG_RE.sub("", transcript).split()


def _best_fold_match(drug: str, words: list[str]) -> tuple[int, int, float]:
    """Return (start, end, ratio) of the words[]-window (sizes 1-4, see
    _DRUG_WINDOW_SIZES) with the highest fold-similarity to `drug`.

    Returns (0, 0, 0.0) when `words` is empty — no plausible match to report.
    """
    best_start, best_end, best_ratio = 0, 0, 0.0
    drug_fold = _fold_drug(drug)
    for n in _DRUG_WINDOW_SIZES:
        for i in range(len(words) - n + 1):
            ratio = difflib.SequenceMatcher(
                None, drug_fold, _fold_drug(" ".join(words[i : i + n]))
            ).ratio()
            if ratio > best_ratio:
                best_start, best_end, best_ratio = i, i + n, ratio
    return best_start, best_end, best_ratio


def _dose_digits_near_drug(dose: str, drug: str, words: list[str]) -> bool:
    """Return True if a digit token of `dose` appears in `words` within
    _DOSE_PROXIMITY_TOKENS of the drug's best fold-match span.

    Real case: transcript never states a paracetamol dose, but the LLM glued
    a neighbouring drug's dose onto it ("naxdom 500" -> paracetamol dose
    "500 mg"). Returns True (no flag) when `dose` has no digits or `words` is
    empty — nothing to check, never a false positive.
    """
    digit_tokens = _DOSE_DIGIT_RE.findall(dose)
    if not digit_tokens or not words:
        return True
    start, end, _ = _best_fold_match(drug, words)
    lo = max(0, start - _DOSE_PROXIMITY_TOKENS)
    hi = min(len(words), end + _DOSE_PROXIMITY_TOKENS)
    nearby = words[lo:hi]
    return any(digit in w for digit in digit_tokens for w in nearby)


def _iter_dicts(items: list, field_name: str) -> list[dict]:
    """Filter a list field to well-formed dict items, skipping malformed ones.

    A malformed model response (e.g. a raw string mixed into a structured list
    field — observed on a live sample where "vitals" contained a garbled
    ["...", "examination: null", "diagnosis: null", ...] tail from a JSON
    formatting slip) must degrade to a partial note, never crash the whole
    extraction. Mirrors the tolerance investigations/diagnostic_results
    already get via _coerce_str.
    """
    valid: list[dict] = []
    for item in items:
        if isinstance(item, dict):
            valid.append(item)
        else:
            logger.warning(
                "L4: non-dict item in %s skipped: %s", field_name, repr(item)[:80]
            )
    return valid


# ── Frequency/timing canonicalization (deterministic, no LLM translation) ──
# Real incident (outputs/<session>, 2026-07-12): the model correctly extracted
# frequency "दिन में दो बार" verbatim (rule 6: use the transcript's own
# language) but the English note view then showed only Hindi — the translate
# route deliberately excludes medications[].* (dosing text is patient-safety
# text; it must never be LLM-translated), which over-covers frequency/timing.
# Fixed here with an EXACT-lookup map (never fuzzy, never model-based) from
# recognized phrases to one canonical English phrase per group. An
# unrecognized phrase — including clinical notation like "1-0-1" or "BD",
# which is already language-neutral and needs no translation — is returned
# unchanged. The original transcript phrase remains recoverable via the
# session transcript; this field only ever holds the display form.
_FREQUENCY_CANON: dict[str, str] = {
    # English phrasing variants -> one canonical phrase per group (mirrors
    # eval/run_eval.py's _FREQ_CANON groups; see LEARNINGS Phase B — E9).
    # Kept in the system prompt's own "___ daily" idiom (rule 11's examples)
    # so already-English extractions are never rewritten into new wording —
    # only recognized (an already-correct value must never appear to change).
    "once daily": "once daily", "once a day": "once daily",
    "one daily": "once daily", "1 daily": "once daily",
    "once in the morning": "once daily", "one in the morning": "once daily",
    "once daily morning": "once daily", "in the morning": "once daily",
    "once at night": "once daily", "once nightly": "once daily",
    "once daily at night": "once daily", "at night": "once daily",
    "at bedtime": "once daily", "in the night": "once daily",
    "in the afternoon": "once daily", "at noon": "once daily",
    "once at noon": "once daily",
    "twice daily": "twice daily", "twice a day": "twice daily",
    "two times a day": "twice daily", "2 times a day": "twice daily",
    "morning and night": "twice daily", "morning and evening": "twice daily",
    "three times a day": "three times daily",
    "three times daily": "three times daily",
    "thrice a day": "three times daily", "thrice daily": "three times daily",
    "3 times a day": "three times daily",
    "as needed": "as needed", "when needed": "as needed",
    "if needed": "as needed",
    # Hindi phrases (this incident's fixture) — time-of-day dosing schedule,
    # per rule 11 ("time-of-day phrases ... belong in frequency, not timing").
    "दिन में दो बार": "twice a day", "दो बार दिन में": "twice a day",
    "दिन में एक बार": "once daily", "एक बार": "once daily",
    "दो बार": "twice a day",
    "तीन बार": "three times a day", "तीन बार दिन में": "three times a day",
    "सुबह शाम": "twice a day",
    "रोज़": "once daily", "रोज": "once daily", "हर रोज": "once daily",
    "रात को": "once daily", "सुबह": "once daily", "सोते समय": "once daily",
    "हफ्ते में एक बार": "once a week",
}

# Meal-relative phrases stay in TIMING, not frequency (rule 11: "TIMING is
# ONLY for meal-relative context").
_TIMING_CANON: dict[str, str] = {
    "खाने के बाद": "after food",
    "खाने से पहले": "before food",
    "खाली पेट": "empty stomach",
}


def _canonicalize_phrase(value: str | None, canon: dict[str, str]) -> str | None:
    """Exact-match, whitespace-normalized canonicalization; unmatched values
    pass through unchanged. Never fuzzy, never model-based — see the module
    section comment above for why dosing-schedule text is never guessed."""
    if value is None:
        return None
    return canon.get(" ".join(value.split()), value)


def _build_note(data: dict, transcript: str = "") -> ClinicalNote:
    # Guard with `or []`: model may emit null for list fields (e.g. "symptoms": null).
    # data.get("symptoms", []) returns None when the key is present with value null,
    # and None is not iterable — the `or []` converts None → [].
    symptoms = [
        Symptom(
            name=s.get("name", "").strip(),
            finding_status=(s.get("finding_status") or "Present").strip() or "Present",
            severity=s.get("severity") or None,
            since=s.get("since") or None,
        )
        for s in _iter_dicts(data.get("symptoms") or [], "symptoms")
        if s.get("name", "").strip()
    ]

    vitals = [
        Vital(name=v.get("name", "").strip(), value=str(v.get("value") or "").strip())
        for v in _iter_dicts(data.get("vitals") or [], "vitals")
        if v.get("name", "").strip() and str(v.get("value") or "").strip()
    ]

    diagnosis = [
        Diagnosis(
            term=d.get("term", "").strip(),
            snomed_id=d.get("snomed_id"),
            status=d.get("status") or None,
        )
        for d in _iter_dicts(data.get("diagnosis") or [], "diagnosis")
        if d.get("term", "").strip()
    ]

    medications: list[Medication] = []
    low_conf: list[str] = list(data.get("low_confidence_fields") or [])
    n_unnamed = 0
    transcript_words = _transcript_words(transcript)
    for m in _iter_dicts(data.get("medications") or [], "medications"):
        drug = (m.get("drug") or "").strip()
        if not drug:
            continue
        if _is_generic_drug_name(drug):
            # A generic term ("medicine", "दवाई") must never appear as a drug
            # NAME on a prescription. Keep the row — its dose/frequency/
            # duration may be real — but mark it unnamed and low-confidence.
            # Numbered so multiple unnamed rows get distinct flags.
            n_unnamed += 1
            label = f"unnamed medication {n_unnamed}"
            logger.warning(
                "L4: generic drug name %r converted to %r", drug, label
            )
            flag = f"medications.{label}.unnamed"
            if flag not in low_conf:
                low_conf.append(flag)
            drug = label
        elif _is_condition_term(drug):
            # A clinical condition/gloss extracted as a medication carries no
            # salvageable prescription info, and the information is already
            # captured elsewhere in the note (history/diagnosis) — drop the
            # row entirely rather than converting it to an unnamed row (see
            # the "Condition guard" section above for the real incident).
            logger.warning(
                "L4: condition term %r dropped from medications (condition_in_rx)",
                drug,
            )
            flag = f"medications.{drug}.condition_in_rx"
            if flag not in low_conf:
                low_conf.append(flag)
            continue
        else:
            restored = _restore_drug_spelling(drug, transcript)
            if restored != drug:
                logger.info(
                    "L4: restored drug spelling %r -> %r (source-fidelity backstop)",
                    drug, restored,
                )
                drug = restored
            isolated = _isolate_drug_span(drug)
            if isolated != drug:
                logger.info(
                    "L4: isolated drug span %r -> %r (filler-glue backstop)",
                    drug, isolated,
                )
                drug = isolated
            # Grounding guard: an invented drug name matches nothing in the
            # transcript, however loosely. _restore_drug_spelling only fixes
            # MIS-spellings of something actually said — a full invention
            # (real case: LLM fabricated "nasal spray" out of thin air) has
            # no transcript source to restore from, so it must never render
            # as a real drug name. Skipped when transcript is empty (tests/
            # eval call _build_note without one).
            if transcript_words:
                _, _, ground_ratio = _best_fold_match(drug, transcript_words)
                if ground_ratio < _DRUG_GROUND_MIN_SIMILARITY:
                    n_unnamed += 1
                    label = f"unnamed medication {n_unnamed}"
                    logger.warning(
                        "L4: ungrounded drug name %r (best transcript "
                        "similarity %.2f) converted to %r",
                        drug, ground_ratio, label,
                    )
                    flag = f"medications.{label}.ungrounded"
                    if flag not in low_conf:
                        low_conf.append(flag)
                    drug = label
        medications.append(
            Medication(
                drug=drug,
                dose=m.get("dose") or None,
                frequency=_canonicalize_phrase(m.get("frequency") or None, _FREQUENCY_CANON),
                timing=_canonicalize_phrase(m.get("timing") or None, _TIMING_CANON),
                duration=m.get("duration") or None,
                validated=validate_drug(drug),
            )
        )

    def _coerce_str(item: object) -> str:
        """Coerce a diagnostic_results or investigations item to plain string.

        The model occasionally returns dicts (e.g. {"term": "..."}) in list
        fields that should contain strings.  Extract the most informative key
        rather than repr the dict.
        """
        if isinstance(item, dict):
            return str(item.get("term") or item.get("name") or item.get("value") or item)
        return str(item)

    # Hallucination calibration: flag any diagnosis whose term shares no word
    # with the transcript. This catches the most egregious fabrications (e.g.
    # "Pulmonary Embolism" on an acne consult) without requiring a domain model.
    # Limitation 1: token comparison is Latin-script only. For Devanagari/Hindi/
    # Marathi transcripts the check is SKIPPED (transcript_tokens is empty after
    # stripping role tags) — the model correctly maps Hindi→English clinical terms,
    # and penalising those would create false positives we cannot distinguish from
    # real hallucinations at this layer. Per CLAUDE.md: do not pre-translate.
    # Limitation 2: a 3B model can confidently hallucinate a term that happens to
    # share a word with the transcript (e.g. "Diabetes" in a non-diabetes consult
    # where the word was spoken in context). Treat this as a floor, not a ceiling.
    transcript_tokens = _transcript_tokens(transcript)
    for diag in diagnosis:
        if not _diagnosis_has_overlap(diag.term, transcript_tokens):
            flag = f"diagnosis.{diag.term}.no_transcript_overlap"
            if flag not in low_conf:
                low_conf.append(flag)
                logger.warning(
                    "L4 calibration: diagnosis %r has no token overlap with transcript — flagged low-confidence",
                    diag.term,
                )
        if _names_a_known_drug(diag.term):
            # Symmetric to the condition-in-rx guard, opposite direction: a
            # drug name surfaced in DIAGNOSIS instead of medications (real
            # incident, same screenshots as the filler-glue defect: "one
            # naxdom 500 one" appeared in both diagnosis and investigations).
            # Flag-only — never move or delete; the doctor decides.
            flag = f"diagnosis.{diag.term}.drug_in_diagnosis"
            if flag not in low_conf:
                low_conf.append(flag)
                logger.warning(
                    "L4: diagnosis term %r fold-matches a drug lexicon entry — flagged (never moved)",
                    diag.term,
                )

    investigations = [_coerce_str(x) for x in (data.get("investigations") or [])]
    for inv in investigations:
        if _names_a_known_drug(inv):
            flag = f"investigations.{inv}.drug_in_investigations"
            if flag not in low_conf:
                low_conf.append(flag)
                logger.warning(
                    "L4: investigation %r fold-matches a drug lexicon entry — flagged (never moved)",
                    inv,
                )

    for med in medications:
        if not med.validated and not med.drug.startswith("unnamed medication"):
            # unnamed rows already carry a .unnamed flag; .unvalidated would
            # double-flag the same problem.
            flag = f"medications.{med.drug}.unvalidated"
            if flag not in low_conf:
                low_conf.append(flag)
        if med.dose is None:
            flag = f"medications.{med.drug}.dose_unknown"
            if flag not in low_conf:
                low_conf.append(flag)
        elif not med.drug.startswith("unnamed medication") and not _dose_digits_near_drug(
            med.dose, med.drug, transcript_words
        ):
            # Dose-provenance flag (never alters the extracted value): a dose
            # far from every mention of its own drug is likely cross-
            # attributed from a neighbouring medication (real case: a
            # "naxdom 500" dose glued onto "paracetamol"). unnamed rows are
            # skipped — there is no drug mention in the transcript to be
            # "near" once the name itself is a fabrication.
            flag = f"medications.{med.drug}.dose_unattributed"
            if flag not in low_conf:
                low_conf.append(flag)

    return ClinicalNote(
        chief_complaint=data.get("chief_complaint") or None,
        history=data.get("history") or None,
        symptoms=symptoms,
        vitals=vitals,
        examination=data.get("examination") or None,
        diagnosis=diagnosis,
        medications=medications,
        investigations=investigations,
        diagnostic_results=[_coerce_str(x) for x in (data.get("diagnostic_results") or [])],
        advice=data.get("advice") or None,
        follow_up=data.get("follow_up") or None,
        low_confidence_fields=low_conf,
    )


def _empty_note() -> ClinicalNote:
    return ClinicalNote(
        chief_complaint=None,
        history=None,
        low_confidence_fields=[
            "chief_complaint",
            "history",
            "diagnosis",
            "medications",
        ],
    )


def warm_llm() -> None:
    """Preload the extraction model into the Ollama server (non-fatal).

    Called on a background thread during L3.5 — after L3 has released
    Whisper, so the load-one-release-one memory discipline holds. The
    options MUST match extract()'s: Ollama restarts the model runner when
    num_ctx changes, which would waste the warm-up.
    """
    import ollama

    try:
        ollama.generate(
            model=config.EXTRACT_MODEL,
            prompt="",
            keep_alive=config.EXTRACT_KEEP_ALIVE,
            options={"num_ctx": config.EXTRACT_NUM_CTX},
        )
        logger.info("L4: model warmed (%s)", config.EXTRACT_MODEL)
    except Exception as exc:
        logger.warning("L4: warm-up failed (non-fatal): %s", exc)


def extract(turns: list[Turn]) -> ClinicalNote:
    """Extract a structured ClinicalNote from normalized transcript turns.

    Calls qwen2.5:3b-instruct via Ollama at temperature=0 with a JSON format
    constraint. Retries once on JSON parse failure, then returns a minimal
    note with all fields flagged low-confidence.

    Dose is null whenever the doctor did not state it explicitly — never
    fabricated. CDSCO validation flags unrecognised drug names.

    Args:
        turns: Normalised, speaker-attributed turns from L3.5.

    Returns:
        ClinicalNote with validated medications and populated
        low_confidence_fields where values are absent or uncertain.
    """
    import ollama

    transcript_text = _turns_to_text(turns)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Transcript:\n{transcript_text}"},
    ]

    for attempt in range(1, 3):
        try:
            response = ollama.chat(
                model=config.EXTRACT_MODEL,
                messages=messages,
                format="json",
                keep_alive=config.EXTRACT_KEEP_ALIVE,
                options={
                    "temperature": 0,
                    # Without num_ctx, Ollama's default truncates long HI/MR
                    # transcripts → empty {} notes (see config.EXTRACT_NUM_CTX).
                    "num_ctx": config.EXTRACT_NUM_CTX,
                    "num_predict": config.EXTRACT_NUM_PREDICT,
                },
            )
            raw = (
                response.message.content
                if hasattr(response, "message")
                else response["message"]["content"]
            )
            data = _parse(raw)
            note = _build_note(data, transcript=transcript_text)
            logger.info(
                "L4: extracted note (attempt %d): %d symptoms, %d vitals, "
                "%d diagnosis, %d meds, %d low_conf",
                attempt,
                len(note.symptoms),
                len(note.vitals),
                len(note.diagnosis),
                len(note.medications),
                len(note.low_confidence_fields),
            )
            return note
        except json.JSONDecodeError as exc:
            logger.warning("L4: JSON parse failed on attempt %d: %s", attempt, exc)
        except Exception as exc:
            logger.error("L4: unexpected error on attempt %d: %s", attempt, exc)
            return _empty_note()

    logger.error("L4: both attempts failed — returning minimal note")
    return _empty_note()
