"""Deterministic drug-span masking around machine translation.

Root cause (real incident, outputs/20260712-194649-763e13): the translator
prompt in web/app.py only *asked* the LLM to preserve Latin-script drug names
verbatim — a Devanagari drug name (एजित्रोमाइसिन, azithromycin) had no
protection at all and the model phonetically guessed a different, dangerous
drug (acetaminophen; नौरफलोक्स/norflox became naloxone, an opioid-overdose
drug). Prompt instructions are hope, not protection.

This module replaces every drug-name span in a string with an indexed,
untranslatable placeholder (``DRUGSPAN0``, ``DRUGSPAN1``, ...) before the
string goes to the translator, and restores the placeholders afterwards —
to the drug's canonical Latin name if one was resolved, or to the original
span text verbatim if the drug could not be safely identified (never guess).
Restoration verifies each placeholder index survived translation exactly
once; any violation (dropped or duplicated index) falls the WHOLE string
back to its untranslated original — count-only checks are insufficient
because a small model can drop one placeholder while duplicating another.

Detection has two tiers, longest-span-wins on overlap:
  1. The session's own `medications[].drug` values, LATIN-SCRIPT ONES ONLY
     (a value that is itself still Devanagari cannot function as "the
     resolved canonical name" — see `find_drug_spans`) — despaced fold-
     equality (src.drug_lexicon's unified fold) bridges Devanagari source
     text to a Latin medication name already resolved by L4. Exact fold
     match first; if that misses, the SAME bounded-edit-distance tier
     `canonicalize_drug_span` uses is retried, but restricted to this
     session's OWN medication names rather than the whole lexicon. A real
     ASR spelling folds close to more than one *globally* similar drug (e.g.
     एजित्रोमाइसिन sits within edit distance 2 of both azithromycin and
     erythromycin — canonicalize_drug_span correctly declines, ambiguity is
     real) but the note itself only prescribed one of them: restricting the
     fuzzy match's candidate set to the note's own medications removes that
     ambiguity without weakening the global lexicon's guard.
  2. The general drug lexicon (`canonicalize_drug_span`) — catches drugs
     mentioned in free text that never made it into the medications list.
Adjacent dose digits (e.g. the "500" in "azithromycin 500") are folded into
the same span so the translator never sees them split from the drug name.
"""

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from src.drug_lexicon import (
    _distance_bound,
    _fold,
    _levenshtein,
    canonicalize_drug_span,
)

# Devanagari + Latin + digit "word" characters — punctuation (commas, danda
# "।", etc.) is never part of a token, so spans never swallow trailing
# punctuation from the source sentence.
_TOKEN_RE = re.compile(r"[A-Za-z0-9ऀ-ॿ]+")
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_MAX_WINDOW = 3  # matches src/l3_5_normalize.py's own drug-span window size
_MAX_DIGIT_GAP_CHARS = 3  # "azithromycin 500" gap is 1 char (a space)
_PLACEHOLDER_RE = re.compile(r"DRUGSPAN(\d+)")


def _fold_key(text: str) -> str:
    """Despaced fold key — the dictionary-lookup form of the unified fold."""
    return _fold(text).replace(" ", "")


@dataclass(frozen=True)
class DrugSpan:
    """One detected drug-name occurrence in a source string.

    Attributes:
        start: Start character offset in the source string.
        end: End character offset (exclusive), including any adjacent dose
            digits folded into the span.
        original: The exact source substring `text[start:end]`.
        canonical: Canonical Latin drug name to restore, or None if the span
            was flagged as drug-shaped but could not be safely identified —
            in which case `original` is restored verbatim instead.
        dose_digits: Adjacent dose-digit text (e.g. "500") to append after
            `canonical` on restore; None if no digit was adjacent or if
            `canonical` is None (the digits are already part of `original`).
    """

    start: int
    end: int
    original: str
    canonical: str | None
    dose_digits: str | None = None


@dataclass(frozen=True)
class MaskedText:
    """A source string with drug spans replaced by indexed placeholders.

    `spans[i]` is what placeholder `DRUGSPAN{i}` in `masked` should restore
    to.
    """

    masked: str
    spans: tuple[DrugSpan, ...]


def _fuzzy_medication_match(
    key: str, medication_fold_keys: dict[str, str]
) -> str | None:
    """Bounded edit-distance match of `key` against the note's OWN drug names.

    Same distance bound as `canonicalize_drug_span` (see `_distance_bound`),
    but the candidate set is restricted to this session's medications
    instead of the whole lexicon — a real ASR spelling (e.g. एजित्रोमाइसिन)
    can sit within that bound of two genuinely different drugs GLOBALLY
    (azithromycin and erythromycin) while the note itself only prescribed
    one of them, so restricting the candidates removes the ambiguity
    without weakening the global guard. Declines (returns None) if the
    restricted set is STILL ambiguous (two distinct prescribed drugs both
    within bound) — wrong drug is worse than unknown, same rule as
    `canonicalize_drug_span`.
    """
    bound = _distance_bound(len(key))
    if bound is None:
        return None
    best_dist: dict[str, int] = {}
    for cand_key, drug in medication_fold_keys.items():
        dist = _levenshtein(key, cand_key, bound)
        if dist > bound:
            continue
        if drug not in best_dist or dist < best_dist[drug]:
            best_dist[drug] = dist
    if not best_dist:
        return None
    min_dist = min(best_dist.values())
    closest = {drug for drug, dist in best_dist.items() if dist == min_dist}
    return next(iter(closest)) if len(closest) == 1 else None


def _candidate_spans(
    tokens: list[re.Match[str]], medication_fold_keys: dict[str, str]
) -> list[tuple[int, int, str]]:
    """All (start, end, canonical) windows that resolve to a known drug.

    Windows containing a digit-only token are skipped (dose numbers are
    attached separately, after overlap resolution — see `_find_spans`).
    Tried in order per window: exact fold match against the note's own
    medications, bounded fuzzy match against the same restricted set, then
    the general lexicon (`canonicalize_drug_span`).
    """
    candidates: list[tuple[int, int, str]] = []
    n = len(tokens)
    for size in range(1, _MAX_WINDOW + 1):
        for i in range(n - size + 1):
            window = tokens[i : i + size]
            if any(t.group().isdigit() for t in window):
                continue
            window_text = " ".join(t.group() for t in window)
            key = _fold_key(window_text)
            canonical = medication_fold_keys.get(key)
            if canonical is None and key:
                canonical = _fuzzy_medication_match(key, medication_fold_keys)
            if canonical is None:
                resolved = canonicalize_drug_span(window_text)
                canonical = resolved[0] if resolved is not None else None
            if canonical is not None:
                candidates.append((window[0].start(), window[-1].end(), canonical))
    return candidates


def _resolve_overlaps(
    candidates: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    """Longest-span-wins: greedily keep non-overlapping spans, longest first."""
    candidates = sorted(candidates, key=lambda c: (-(c[1] - c[0]), c[0]))
    selected: list[tuple[int, int, str]] = []
    for start, end, canonical in candidates:
        if any(start < s_end and end > s_start for s_start, s_end, _ in selected):
            continue
        selected.append((start, end, canonical))
    return sorted(selected, key=lambda s: s[0])


def _attach_adjacent_dose(
    text: str, tokens: list[re.Match[str]], start: int, end: int
) -> tuple[int, str | None]:
    """Extend `end` to include an immediately-following dose-digit token."""
    next_tok = next((t for t in tokens if t.start() >= end), None)
    if (
        next_tok is not None
        and next_tok.group().isdigit()
        and next_tok.start() - end <= _MAX_DIGIT_GAP_CHARS
    ):
        return next_tok.end(), next_tok.group()
    return end, None


def find_drug_spans(
    text: str, medication_drugs: Sequence[str] = ()
) -> tuple[DrugSpan, ...]:
    """Find every drug-name span in `text`, longest-span-wins on overlap.

    Args:
        text: Source string (Devanagari, Latin, or mixed) to scan.
        medication_drugs: The session's own `medications[].drug` values —
            checked first via despaced fold-equality (bridges a Devanagari
            source mention to the note's already-resolved Latin name). Only
            entries already in Latin script are used: a medication value
            that is itself still Devanagari (a not-yet-canonicalized or
            pre-fix note) cannot function as "the resolved canonical name"
            — matching against it would only echo the same Devanagari text
            back, and its fold key can spuriously outrank a shorter, better
            general-lexicon match (tier 2) under longest-span-wins.

    Returns:
        Non-overlapping spans in left-to-right order.
    """
    tokens = list(_TOKEN_RE.finditer(text))
    if not tokens:
        return ()

    medication_fold_keys = {
        _fold_key(drug): drug
        for drug in medication_drugs
        if drug and not _DEVANAGARI_RE.search(drug)
    }
    candidates = _candidate_spans(tokens, medication_fold_keys)
    if not candidates:
        return ()

    spans = []
    for start, end, canonical in _resolve_overlaps(candidates):
        end, dose_digits = _attach_adjacent_dose(text, tokens, start, end)
        spans.append(
            DrugSpan(
                start=start,
                end=end,
                original=text[start:end],
                canonical=canonical,
                dose_digits=dose_digits,
            )
        )
    return tuple(spans)


def mask(text: str, medication_drugs: Sequence[str] = ()) -> MaskedText:
    """Replace every detected drug span in `text` with an indexed placeholder."""
    spans = find_drug_spans(text, medication_drugs)
    if not spans:
        return MaskedText(text, ())

    parts: list[str] = []
    cursor = 0
    for i, span in enumerate(spans):
        parts.append(text[cursor : span.start])
        parts.append(f"DRUGSPAN{i}")
        cursor = span.end
    parts.append(text[cursor:])
    return MaskedText("".join(parts), spans)


def restore(translated: str, spans: tuple[DrugSpan, ...]) -> str | None:
    """Restore placeholders in a translated string; None on any violation.

    Each of `DRUGSPAN0` .. `DRUGSPAN{len(spans)-1}` must appear in
    `translated` EXACTLY once — not just the same total count, since a
    small model can drop one placeholder while duplicating another. Any
    departure from that (missing, duplicated, or an unexpected extra index)
    is treated as a translation failure for this string; the caller is
    expected to fall back to the untranslated original.

    Args:
        translated: The translator's output for a string produced by `mask`.
        spans: The spans returned alongside that string's `MaskedText`.

    Returns:
        `translated` with every placeholder replaced (canonical name when
        resolved, original span text verbatim otherwise), or None if the
        placeholder survival check fails.
    """
    if not spans:
        return translated

    counts = Counter(int(m) for m in _PLACEHOLDER_RE.findall(translated))
    if counts != Counter({i: 1 for i in range(len(spans))}):
        return None

    def _replacement(match: re.Match[str]) -> str:
        span = spans[int(match.group(1))]
        if span.canonical is None:
            return span.original
        if span.dose_digits is None:
            return span.canonical
        return f"{span.canonical} {span.dose_digits}"

    return _PLACEHOLDER_RE.sub(_replacement, translated)
