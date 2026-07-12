"""L3.5 — Post-ASR normalization: drug-term transliteration + concept glossing.

Two passes in sequence:
1. Drug normalization (no model): 4-tier Devanagari→Latin pipeline (curated
   table → expanded-lexicon fold match → exact CDSCO → length-guarded fuzzy
   match).
2. Concept normalization (parrotlet-e): lay symptom/condition terms glossed
   with canonical clinical names + SNOMED IDs.
"""

import difflib
import gc
import logging
import os
import re
from dataclasses import dataclass

import numpy as np

from src import config
from src.cdsco import _APPROVED_DRUGS
from src.concepts import CONCEPTS, EVERYDAY_WORDS
from src.drug_lexicon import canonicalize_drug_span
from src.types import Turn

logger = logging.getLogger(__name__)

# ── Pass 1: Devanagari drug-name normalization ─────────────────────────────
# Constants: tuned on frozen Hindi-15 set (vaani-large-v3 + faster-whisper).

_DRUG_FUZZY_THRESHOLD = 0.82
_DRUG_FUZZY_MIN_LEN = 8  # CDSCO candidate must be ≥8 chars to enter fuzzy

# Hand-curated Devanagari → Latin table (longest-match wins).
# Covers English-phonetic drug names Whisper renders in Devanagari script.
_DEVA_CURATED: dict[str, str] = {
    "मेडिसिन्स": "medicines",
    "ऑर्ग्यूमेंटिंग": "augmentin",
    "मेडिसिन": "medicine",
    "डीओ टेबलेट": "DO tablet",
    "टेबलेट": "tablet",
    "टैबलेट": "tablet",
    "ऑग्मेंट": "augmentin",
    "ऑग्युमेंट": "augmentin",
    "ऑर्ग्यूमेंट": "augmentin",
    "दवाई": "medicine",
    "दवा": "medicine",
    "इंजेक्शन": "injection",
    "इंजेक्‍शन": "injection",
    "कैप्सूल": "capsule",
    "सिरप": "syrup",
    "कफ सिरप": "cough syrup",
    "जेल": "gel",
    "एंटीबायोटिक": "antibiotic",
    "एंटीबायोटिक्स": "antibiotics",
    "सनस्क्रीन": "sunscreen",
    "मलहम": "ointment",
    "क्रीम": "cream",
    "ड्रॉप्स": "drops",
    "सस्पेंशन": "suspension",
    "टिंचर": "tincture",
    "पैरासिटामोल": "paracetamol",
    "पैरासिटमोल": "paracetamol",
    "पैरासेट मूल": "paracetamol",
    "पैरसेट मॉल": "paracetamol",
    "पैरासेट मॉल": "paracetamol",
    "फ्लूकोनाज़ोल": "fluconazole",
    "फ्लुकोनाज़ोल": "fluconazole",
    "फ्लूकोनाज़ोल 150": "fluconazole 150",
    "मेट्रोनिडाज़ोल": "metronidazole",
    "एजिथ्रोमाइसिन": "azithromycin",
    "एजिथ्रोमायसिन": "azithromycin",
    # No-aspirate ASR spelling (missing थ AND the ज़ nukta) — real incident,
    # outputs/20260712-194649-763e13, 2026-07-12. The expanded-lexicon fold
    # tier (src/drug_lexicon.py) cannot recover this one safely: its fold
    # key sits within the fuzzy bound of BOTH azithromycin and erythromycin
    # (two genuinely distinct antibiotics), a real ambiguity from the missing
    # nukta/aspirate, not a rule gap — canonicalize_drug_span() correctly
    # returns None rather than guess. Curated here instead, the same
    # zero-risk mechanism the नैक्सडॉम spelling family already uses for a
    # known, seen distortion.
    "एजित्रोमाइसिन": "azithromycin",
    "एमोक्सिसिलिन": "amoxicillin",
    "अमोक्सिसिलिन": "amoxicillin",
    "आइबुप्रोफेन": "ibuprofen",
    "आईबुप्रोफेन": "ibuprofen",
    "ओमेप्राज़ोल": "omeprazole",
    "ओमेप्रेज़ोल": "omeprazole",
    "पैंटोप्राज़ोल": "pantoprazole",
    "मेटफॉर्मिन": "metformin",
    "डिक्लोफेनैक": "diclofenac",
    "डाइक्लोफेनेक": "diclofenac",
    "सेटिरीज़ीन": "cetirizine",
    "सेटिरिज़ीन": "cetirizine",
    "रेनिटिडीन": "ranitidine",
    "सेफिक्सिम": "cefixime",
    "लेवोसाल्बुटामोल": "levosalbutamol",
    "साल्बुटामोल": "salbutamol",
    "मोन्टेलुकास्ट": "montelukast",
    "टेल्मिसार्टन": "telmisartan",
    "एम्लोडिपिन": "amlodipine",
    "अम्लोडिपिन": "amlodipine",
    "एटोर्वास्टेटिन": "atorvastatin",
    "लोसार्टन": "losartan",
    "वारफेरिन": "warfarin",
    "एस्पिरिन": "aspirin",
    "बीटामेथासोन": "betamethasone",
    "डिक्सीसाइक्लिन": "doxycycline",
    "डॉक्सीसाइक्लिन": "doxycycline",
    "क्लोनाज़ेपाम": "clonazepam",
    "अल्प्राज़ोलम": "alprazolam",
    "रैनिटिडीन": "ranitidine",
    "ड्रोटावेरिन": "drotaverine",
    "मेफेनामिक एसिड": "mefenamic acid",
    "मेफेनामिक": "mefenamic acid",
    "ट्रामाडोल": "tramadol",
    "विटामिन सी": "vitamin c",
    "विटामिन डी": "vitamin d",
    "विटामिन डी3": "vitamin d3",
    "कैल्शियम कार्बोनेट": "calcium carbonate",
    "मल्टीविटामिन": "multivitamin",
    "आयरन": "iron",
    "फोलिक एसिड": "folic acid",
    "जिंक": "zinc",
    "प्रोबायोटिक": "probiotic",
    "ओमेगा 3": "omega 3",
    "फ्लुटिकासोन": "fluticasone",
    "बुडेसोनाइड": "budesonide",
    "टर्बुटालिन": "terbutaline",
    "क्लोरफेनिरामाइन": "chlorpheniramine",
    "डेक्सट्रोमेथोर्फन": "dextromethorphan",
    "गुआइफेनेसिन": "guaifenesin",
    "कोडीन": "codeine",
    "डोलो": "dolo",
    "डोलो 650": "dolo 650",
    "कैल्पोल": "calpol",
    "ग्लाइकोमेट": "glycomet",
    "ग्लाइकोमेट जीपी": "glycomet gp",
    "ओमनीजेल": "omnigel",
    "ओम्नि जेल": "omnigel",
    "वोवेरान": "voveran",
    "पैंटोप": "pantop",
    "लिमसी": "limcee",
    "शेल्कल": "shelcal",
    "अस्थालिन": "asthalin",
    "लेवोलिन": "levolin",
    "फोराकोर्ट": "foracort",
    "ड्रोटिन": "drotin",
    "ज़ीफी": "zifi",
    "ज़िफी": "zifi",
    "मेफ्टल": "meftal",
    "मेफ्टल स्पास": "meftal spas",
    "अल्ट्रासेट": "ultracet",
    "पैन डी": "pan d",
    "पैन-डी": "pan d",
    "मॉक्सीक्लाव": "moxclav",
    "बाइफिलैक": "bifilac",
    "नैक्सडॉम": "naxdom",
    "नक्सडोम": "naxdom",
    "नेक्सडोम": "naxdom",
    "नेक्स डॉम": "naxdom",
    "नैक्सडॉम 500": "naxdom 500",
    "नेक्स डॉम 500": "naxdom 500",
    "नैक्सडॉम 250": "naxdom 250",
    "नैक्स्टोम": "naxdom",
    "नेक्स्टोम": "naxdom",
    "नैक्स्टोम 500": "naxdom 500",
    "नेक्स्टोम 500": "naxdom 500",
    # Candra-o (ॉ) variant of the same spelling — outputs/<session>, 2026-07-12:
    # "एक नैक्स्टॉम 500 एक" escaped normalization because candra-o was absent
    # from every fold map (see src/drug_lexicon.py's _FOLD_MAP); fixed there
    # too, but the curated tier stays the zero-risk fast path for this
    # already-seen distortion.
    "नैक्स्टॉम": "naxdom",
    "नेक्स्टॉम": "naxdom",
    "नैक्स्टॉम 500": "naxdom 500",
    "नेक्स्टॉम 500": "naxdom 500",
}

_DEVA_RE = re.compile(r"[ऀ-ॿ]")


def _is_devanagari(token: str) -> bool:
    return bool(_DEVA_RE.search(token))


def _itrans_romanize(text: str) -> str:
    """ITRANS romanization with English-loanword post-processing."""
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate

        r = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS).lower()
    except Exception:
        return text.lower()
    r = r.replace("ph", "f")
    r = r.replace("ai", "a")
    r = re.sub(r"a$", "", r)
    return r


def _normalize_roman(r: str) -> str:
    """Additional normalization for CDSCO exact lookup."""
    r = r.replace("aa", "a").replace("ii", "i").replace("uu", "u")
    r = re.sub(r"([bcdfghjklmnpqrstvwxyz])\1", r"\1", r)
    r = re.sub(r"sh", "s", r)
    r = re.sub(r"([aeiou])n$", r"\1", r)
    return r.strip()


# Pre-build CDSCO lookup tables at import time (fast, no I/O).
_CDSCO_NORM_EXACT: dict[str, str] = {}
for _drug in _APPROVED_DRUGS:
    if not _drug.strip():
        continue
    _r = _itrans_romanize(_drug.lower())
    _n = _normalize_roman(_r)
    _CDSCO_NORM_EXACT[_n] = _drug
    _CDSCO_NORM_EXACT[_n.replace(" ", "")] = _drug

_CDSCO_FUZZY_LIST: list[tuple[str, str]] = [
    (d.lower().replace(" ", ""), d)
    for d in _APPROVED_DRUGS
    if len(d.replace(" ", "")) >= _DRUG_FUZZY_MIN_LEN
]


def _cdsco_exact(roman: str) -> str | None:
    n1 = _normalize_roman(roman)
    return _CDSCO_NORM_EXACT.get(n1) or _CDSCO_NORM_EXACT.get(n1.replace(" ", ""))


def _cdsco_fuzzy(roman: str) -> str | None:
    flat = roman.replace(" ", "")
    best_name, best_r = None, 0.0
    for norm_key, canonical in _CDSCO_FUZZY_LIST:
        r = difflib.SequenceMatcher(None, flat, norm_key).ratio()
        if r > best_r:
            best_r, best_name = r, canonical
    return best_name if best_r >= _DRUG_FUZZY_THRESHOLD else None


# Latin spans are only fuzzy-matched when they contain a plausible drug-like
# token: alphabetic, >= 5 chars, and not a common clinical/conversation word.
# Whisper writes distorted drug names in Latin too ("gmenti 625" for
# "Augmentin 625"); those spans previously bypassed all tiers because the
# drug pass examined Devanagari-containing windows only.
_LATIN_SPAN_STOPWORDS: frozenset[str] = frozenset(
    {
        "tablet", "tablets", "capsule", "capsules", "syrup", "injection",
        "medicine", "medicines", "doctor", "patient", "morning", "evening",
        "night", "before", "after", "fever", "infection", "pressure",
        "sugar", "blood", "test", "tests", "report", "daily", "times",
        "twice", "thrice", "water", "days", "weeks", "month", "months",
        "please", "problem", "little", "right", "there", "these", "those",
        "about", "again", "because", "table", "would", "should", "could",
    }
)


def _latin_span_candidate(span: list[str]) -> bool:
    """True if an all-Latin span is worth trying against the CDSCO tiers."""
    alpha = [t for t in span if t.isalpha()]
    if not any(len(t) >= 5 for t in alpha):
        return False
    return not any(t.lower() in _LATIN_SPAN_STOPWORDS for t in alpha)


# Hand-curated Latin→Latin table for brand names Whisper distorts even when it
# stays in Latin script (naxdom is not in CDSCO, so the CDSCO tiers below can
# never recover it). Checked first, same as _DEVA_CURATED for Devanagari spans.
_LATIN_CURATED: dict[str, str] = {
    "nextom": "naxdom",
}


def _lexicon_hit(span: list[str], span_text: str) -> str | None:
    """Try the expanded-lexicon fold matcher for one span, digit-guarded.

    Returns the canonical drug name only if canonicalize_drug_span() found
    an unambiguous match AND every digit token in the span survives in that
    canonical (same dose-preservation invariant as the other tiers). None
    otherwise — the caller falls through to the next tier.
    """
    hit = canonicalize_drug_span(span_text)
    if hit is None:
        return None
    canonical, _confidence = hit
    span_digits = [t for t in span if any(ch.isdigit() for ch in t)]
    if any(d not in canonical for d in span_digits):
        return None
    return canonical


def _normalize_drug_text(text: str) -> str:
    """Apply 4-tier drug normalization to a single string.

    Processes windows of 3, 2, 1 tokens (longest match wins). Windows with
    Devanagari tokens go through curated-table → expanded-lexicon fold match
    (_lexicon_hit) → ITRANS+CDSCO-exact → CDSCO-fuzzy. All-Latin windows that
    look drug-like (see _latin_span_candidate) go through curated-table
    (_LATIN_CURATED) → expanded-lexicon fold match → CDSCO-exact →
    CDSCO-fuzzy with the same thresholds — Whisper distorts Latin drug names
    too, and brands absent from CDSCO (e.g. naxdom) can only ever be
    recovered via the curated table or the expanded lexicon. The lexicon
    tier absorbs spelling variants the curated tables have never seen
    (fold-key match, ambiguity-guarded — see src/drug_lexicon.py); the
    curated tables stay as the zero-risk fast path for known distortions.
    Already-covered positions are skipped.

    Args:
        text: Raw ASR hypothesis string.

    Returns:
        String with matched drug spans replaced by their canonical forms.
    """
    tokens = text.split()
    hits: dict[tuple[int, int], str] = {}  # (start, end) → latin

    for window in (3, 2, 1):
        for i in range(len(tokens) - window + 1):
            span = tokens[i : i + window]
            if any(s <= i < e or s < i + window <= e for (s, e) in hits):
                continue

            span_text = " ".join(span)

            if any(_is_devanagari(t) for t in span):
                latin = _DEVA_CURATED.get(span_text.strip())
                if latin:
                    hits[(i, i + window)] = latin
                    continue

                latin = _lexicon_hit(span, span_text)
                if latin:
                    hits[(i, i + window)] = latin
                    continue

                roman = _itrans_romanize(span_text)
                latin = _cdsco_exact(roman)
                if latin:
                    hits[(i, i + window)] = latin
                    continue

                latin = _cdsco_fuzzy(roman)
                if latin:
                    hits[(i, i + window)] = latin
                continue

            if _latin_span_candidate(span):
                roman = span_text.lower()
                latin = _LATIN_CURATED.get(roman)
                if latin is None:
                    latin = _lexicon_hit(span, span_text)
                if latin is None:
                    latin = _cdsco_exact(roman)
                if latin is None:
                    latin = _cdsco_fuzzy(roman)
                if latin is None or latin.lower() == roman:
                    # No hit, or span already equals the canonical form.
                    continue
                # Substitution must never delete dose information: every
                # digit token in the span must survive in the canonical
                # ("paracetamol 625" → "paracetamol" would drop the dose).
                span_digits = [t for t in span if any(ch.isdigit() for ch in t)]
                if any(d not in latin for d in span_digits):
                    continue
                hits[(i, i + window)] = latin

    if not hits:
        return text

    result = list(tokens)
    offset = 0
    for (s, e), latin in sorted(hits.items()):
        sa, ea = s - offset, e - offset
        logger.debug("L3.5 drug: '%s' → '%s'", " ".join(result[sa:ea]), latin)
        result[sa:ea] = [latin]
        offset += (e - s) - 1
    return " ".join(result)


# ── Pass 2: parrotlet-e concept normalization ──────────────────────────────
# Model id and thresholds live in src/config.py; re-exported here so existing
# importers (tests) keep working.
MODEL_ID = config.NORMALIZE_MODEL
COSINE_THRESHOLD = config.COSINE_THRESHOLD
HARDNEG_MARGIN = config.HARDNEG_MARGIN
COSINE_THRESHOLD_UNIGRAM = config.COSINE_THRESHOLD_UNIGRAM
CONCEPT_AMBIGUITY_MARGIN = config.CONCEPT_AMBIGUITY_MARGIN
MAX_NGRAM = 3           # unigrams through trigrams
_ENCODE_BATCH_SIZE = 256  # backend.encode() chunk size — see its docstring


@dataclass
class _Match:
    span: str
    start_word: int   # inclusive
    end_word: int     # exclusive
    concept_term: str
    snomed_id: str | None
    similarity: float
    # Second-best-scoring DIFFERENT concept for this span, and its cosine —
    # populated by _score_spans, used by the ambiguity gate and surfaced to
    # eval/gloss_audit.py for adjudication. None when there is only one
    # concept in play (defensive; CONCEPTS always has 2+ entries in practice).
    runner_up_term: str | None = None
    runner_up_similarity: float | None = None


@dataclass(frozen=True)
class MatchConfig:
    """Tunable gates for one scoring pass over precomputed similarities.

    Bundled so eval/gloss_audit.py can score the SAME embeddings under an
    OLD (pre-hardening) and a NEW (current) configuration without re-running
    the model — see that module's module docstring.
    """

    cosine_threshold: float          # bigram+ spans
    cosine_threshold_unigram: float  # unigram spans (higher bar)
    ambiguity_margin: float          # best-vs-runner-up CONCEPT margin
    everyday_words: frozenset[str]   # spans that can never gloss, verbatim


def _default_match_config() -> MatchConfig:
    """Production config: current src/config.py values, guards ON."""
    return MatchConfig(
        cosine_threshold=COSINE_THRESHOLD,
        cosine_threshold_unigram=COSINE_THRESHOLD_UNIGRAM,
        ambiguity_margin=CONCEPT_AMBIGUITY_MARGIN,
        everyday_words=EVERYDAY_WORDS,
    )


class _EmbeddingBackend:
    """Load parrotlet-e once, encode texts → unit-norm vectors, then release.

    Uses MPS if available, falls back to CPU. Mean pooling + L2 norm per the
    parrotlet-e model card (not CLS token).
    """

    def __init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        device_str = (
            "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        self._device = torch.device(device_str)
        hf_token = os.environ.get("HF_TOKEN")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
            self._model = AutoModel.from_pretrained(MODEL_ID, token=hf_token).to(
                self._device
            )
        except Exception as exc:
            exc_str = str(exc).lower()
            if "401" in exc_str or "gated" in exc_str or "unauthorized" in exc_str or "403" in exc_str:
                raise PermissionError(
                    "parrotlet-e is gated. Accept the terms at "
                    "https://huggingface.co/ekacare/parrotlet-e with your HF "
                    "account, then ensure HF_TOKEN is set in .env."
                ) from exc
            raise

        self._model.eval()
        logger.info(
            "L3.5 backend: %s on %s, hidden_size=%d",
            MODEL_ID,
            device_str,
            self._model.config.hidden_size,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts → L2-normalised embeddings, shape (N, hidden_size).

        Chunks large inputs into batches of _ENCODE_BATCH_SIZE. A single
        forward pass over tens of thousands of spans (e.g.
        eval/gloss_audit.py's whole-corpus batch) OOMs the MPS backend;
        production transcripts have never been large enough to hit this, but
        the chunking is unconditional so neither caller has to know about it.

        Args:
            texts: List of text strings to encode.

        Returns:
            Float32 numpy array of shape (N, hidden_size).
        """
        import torch

        if len(texts) > _ENCODE_BATCH_SIZE:
            chunks = [
                self.encode(texts[i : i + _ENCODE_BATCH_SIZE])
                for i in range(0, len(texts), _ENCODE_BATCH_SIZE)
            ]
            return np.concatenate(chunks, axis=0)

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            output = self._model(**encoded)

        # Mean pooling — mask out padding tokens before averaging
        mask = encoded["attention_mask"].unsqueeze(-1).float()  # (N, seq, 1)
        embeddings = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-8)
        # L2 normalise → cosine similarity = dot product
        norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
        embeddings = (embeddings / norms).cpu().numpy().astype(np.float32)
        return embeddings

    def release(self) -> None:
        """Delete model weights and free device memory."""
        import torch

        del self._model, self._tokenizer
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def _passes_hardneg_gate(sim: float, max_hn_sim: float) -> bool:
    """Return True if the span is far enough above its closest hard negative.

    A match is accepted only when the concept similarity exceeds the best
    hard-negative similarity by more than HARDNEG_MARGIN. Boundary (equal)
    is treated as rejection.

    Args:
        sim: Cosine similarity of the candidate span to the matched concept.
        max_hn_sim: Maximum cosine similarity of the span to any hard negative
            of that concept.
    """
    return max_hn_sim < sim - HARDNEG_MARGIN


def _passes_ambiguity_gate(best_sim: float, second_sim: float, margin: float) -> bool:
    """Return True if the best-scoring concept is far enough ahead of the runner-up.

    Mirrors src/drug_lexicon.py's ambiguity guard ("if a skeleton sits within
    tolerance of TWO different drugs ... refuse to choose"): a span whose
    best and second-best CONCEPT scores are within `margin` is genuinely
    ambiguous and must not be glossed. Boundary (equal) is treated as
    rejection, consistent with _passes_hardneg_gate.

    Args:
        best_sim: Cosine similarity of the span to its best-matching concept.
        second_sim: Cosine similarity of the span to the second-best,
            DIFFERENT concept (``-inf`` if there is only one concept).
        margin: Minimum required lead of best over second-best.
    """
    return best_sim - second_sim > margin


def _ngrams(words: list[str], max_n: int) -> list[tuple[str, int, int]]:
    """Return (span_text, start_idx, end_idx_exclusive) for n=1..max_n."""
    spans = []
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            spans.append((" ".join(words[i : i + n]), i, i + n))
    return spans


def _best_non_overlapping(matches: list[_Match]) -> list[_Match]:
    """Greedy selection: highest-similarity first, drop overlapping matches."""
    selected: list[_Match] = []
    covered: set[int] = set()
    for m in sorted(matches, key=lambda m: m.similarity, reverse=True):
        word_indices = set(range(m.start_word, m.end_word))
        if word_indices & covered:
            continue
        selected.append(m)
        covered |= word_indices
    return sorted(selected, key=lambda m: m.start_word)


def _gloss_turn(turn: Turn, matches: list[_Match], words: list[str]) -> Turn:
    """Rebuild turn text with clinical term glossed in parentheses.

    e.g. 'sugar hai' → 'sugar (Type 2 Diabetes Mellitus) hai'
    """
    if not matches:
        return turn

    match_at: dict[int, _Match] = {m.start_word: m for m in matches}
    result: list[str] = []
    skip_to = 0

    for i, word in enumerate(words):
        if i < skip_to:
            continue
        if i in match_at:
            m = match_at[i]
            span_text = " ".join(words[m.start_word : m.end_word])
            result.append(f"{span_text} ({m.concept_term})")
            logger.info(
                "L3.5 gloss [%.2fs]: '%s' → '%s' (sim=%.3f)",
                turn.start,
                span_text,
                m.concept_term,
                m.similarity,
            )
            skip_to = m.end_word
        else:
            result.append(word)

    return Turn(
        speaker_role=turn.speaker_role,
        text=" ".join(result),
        start=turn.start,
        end=turn.end,
    )


_RefMatrices = tuple[
    np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None
]


def _encode_reference_matrices(backend: _EmbeddingBackend) -> _RefMatrices:
    """Encode the CONCEPTS reference + hard-negative texts.

    Split out of normalize() so eval/gloss_audit.py can encode ONCE and score
    multiple MatchConfigs against the same embeddings (the model call is the
    expensive part; scoring is pure numpy).

    Returns:
        (ref_matrix, ref_ci_arr, hardneg_matrix, hardneg_idx_arr,
        hardneg_word_counts) — the latter three are None if no concept has
        any hard negatives. hardneg_word_counts (one int per hard-negative
        row) lets _score_spans skip a hard negative that is SHORTER than the
        query span (see its docstring — a unigram hard negative like हफ्ते
        must not veto a longer, more specific span like "हांफ रहे हैं").
    """
    # Reference matrix: canonical term + all variants per concept.
    # Using only canonical terms (e.g. "Type 2 Diabetes Mellitus") fails for
    # colloquial abbreviations like "sugar" (sim=0.33) and "bp" (sim=0.45),
    # which are well below COSINE_THRESHOLD. Including variants covers these
    # exact/near-exact matches while canonical terms remain the anchors for
    # paraphrases and cross-lingual forms the model generalises well.
    ref_texts: list[str] = []
    ref_ci: list[int] = []
    for ci, concept in enumerate(CONCEPTS):
        ref_texts.append(concept.term)
        ref_ci.append(ci)
        for v in concept.variants:
            ref_texts.append(v)
            ref_ci.append(ci)
    ref_matrix = backend.encode(ref_texts)       # (R, D)
    ref_ci_arr = np.array(ref_ci)                # (R,)

    # Hard-negative matrix: one row per hard-negative text; parallel array
    # records which concept index each hard-negative belongs to.
    hardneg_texts: list[str] = []
    hardneg_concept_idx: list[int] = []
    for ci, concept in enumerate(CONCEPTS):
        for hn in concept.hard_negatives:
            hardneg_texts.append(hn)
            hardneg_concept_idx.append(ci)
    hardneg_matrix: np.ndarray | None = None
    hardneg_idx_arr: np.ndarray | None = None
    hardneg_word_counts: np.ndarray | None = None
    if hardneg_texts:
        hardneg_matrix = backend.encode(hardneg_texts)          # (H, D)
        hardneg_idx_arr = np.array(hardneg_concept_idx)         # (H,)
        counts = [len(hn.split()) for hn in hardneg_texts]
        hardneg_word_counts = np.array(counts)                  # (H,)

    return ref_matrix, ref_ci_arr, hardneg_matrix, hardneg_idx_arr, hardneg_word_counts


_TurnSpans = tuple[
    list[tuple[str, int, int]],
    list[tuple[int, int]],
    list[list[str]],
    np.ndarray | None,
]


def _encode_turn_spans(backend: _EmbeddingBackend, turns: list[Turn]) -> _TurnSpans:
    """Build and encode every 1..MAX_NGRAM candidate span across all turns.

    Returns:
        (all_spans, turn_span_offsets, turn_words, span_matrix). span_matrix
        is None when there are no spans at all (every turn empty).
    """
    all_spans: list[tuple[str, int, int]] = []    # (span, start_w, end_w)
    turn_span_offsets: list[tuple[int, int]] = []  # (global_start, end) per turn
    turn_words: list[list[str]] = []

    for turn in turns:
        words = turn.text.split()
        turn_words.append(words)
        if not words:
            turn_span_offsets.append((len(all_spans), len(all_spans)))
            continue
        cands = _ngrams(words, MAX_NGRAM)
        start_off = len(all_spans)
        all_spans.extend(cands)
        turn_span_offsets.append((start_off, len(all_spans)))

    if not all_spans:
        return all_spans, turn_span_offsets, turn_words, None

    span_texts = [s[0] for s in all_spans]
    span_matrix = backend.encode(span_texts)  # (S, D)
    return all_spans, turn_span_offsets, turn_words, span_matrix


def _per_concept_max_sims(
    sims: np.ndarray, ref_ci_arr: np.ndarray, n_concepts: int
) -> np.ndarray:
    """Collapse (S, R) span-vs-reference sims to (S, C) span-vs-CONCEPT sims.

    A concept may have many reference rows (canonical term + variants); the
    concept's score for a span is the max over its own rows. Used by the
    between-concept ambiguity gate, which must compare CONCEPTS, not raw
    reference rows (two references of the SAME concept are not "ambiguity").
    """
    out = np.full((sims.shape[0], n_concepts), -np.inf, dtype=np.float32)
    for ci in range(n_concepts):
        mask = ref_ci_arr == ci
        if mask.any():
            out[:, ci] = sims[:, mask].max(axis=1)
    return out


def _score_spans(
    all_spans: list[tuple[str, int, int]],
    turn_span_offsets: list[tuple[int, int]],
    ref_sims: np.ndarray,
    ref_ci_arr: np.ndarray,
    hardneg_sims: np.ndarray | None,
    hardneg_idx_arr: np.ndarray | None,
    n_concepts: int,
    cfg: MatchConfig,
    hardneg_word_counts: np.ndarray | None = None,
) -> list[list[_Match]]:
    """Apply every gate to precomputed similarities; return matches per turn.

    Pure numpy — no model calls, no I/O — so eval/gloss_audit.py can score
    the SAME embeddings under different MatchConfigs, and tests/test_
    concept_guard.py can exercise the ambiguity gate with hand-built arrays.

    Gates applied, in order, per candidate span:
      1. Everyday-word guard: span text is a curated common word -> reject.
      2. Cosine threshold: COSINE_THRESHOLD_UNIGRAM for 1-word spans,
         COSINE_THRESHOLD for 2+ word spans.
      3. Between-concept ambiguity margin: best CONCEPT must lead the
         second-best DIFFERENT concept by more than cfg.ambiguity_margin.
      4. Hard-negative margin (existing E5 gate). A hard negative SHORTER
         than the query span is excluded from this check — a unigram hard
         negative (e.g. हफ्ते, added to guard the UNIGRAM "हफते" collision)
         must not veto a longer, more specific span ("हांफ रहे हैं", a
         genuine Shortness of Breath mention one hard-negative-comparison
         away from हफ्ते in embedding space). Equal-length comparisons are
         unaffected — the original unigram-vs-unigram collision this gate
         was built for still applies exactly as before.

    Args:
        all_spans: (span_text, start_word, end_word) for every candidate,
            flattened across all turns.
        turn_span_offsets: (start, end) index range into all_spans per turn.
        ref_sims: (S, R) span-vs-reference cosine similarities.
        ref_ci_arr: (R,) concept index per reference row.
        hardneg_sims: (S, H) span-vs-hard-negative cosine similarities, or
            None if no concept has any hard negatives.
        hardneg_idx_arr: (H,) concept index per hard-negative row.
        n_concepts: len(CONCEPTS).
        cfg: Thresholds/guards for this pass.
        hardneg_word_counts: (H,) word count per hard-negative row. None
            (the default) disables the length exclusion — every hard
            negative competes regardless of length, the original behavior.

    Returns:
        One list of accepted, non-overlapping _Match per turn (same order
        and length as turn_span_offsets).
    """
    max_sims = ref_sims.max(axis=1)             # (S,) best ref similarity per span
    best_refs = ref_sims.argmax(axis=1)         # (S,) which reference matched best
    best_concepts = ref_ci_arr[best_refs]       # (S,) concept index for best ref
    per_concept_sims = _per_concept_max_sims(ref_sims, ref_ci_arr, n_concepts)  # (S, C)

    per_turn_matches: list[list[_Match]] = []
    for off_start, off_end in turn_span_offsets:
        matches: list[_Match] = []
        for j in range(off_start, off_end):
            span, start_w, end_w = all_spans[j]

            # (1) Everyday-word guard — unconditional, before any scoring.
            if span.strip().lower() in cfg.everyday_words:
                continue

            sim = float(max_sims[j])
            is_unigram = (end_w - start_w) == 1
            threshold = (
                cfg.cosine_threshold_unigram if is_unigram else cfg.cosine_threshold
            )
            if sim < threshold:
                continue

            ci = int(best_concepts[j])
            concept = CONCEPTS[ci]

            # (2) Between-concept ambiguity margin.
            concept_row = per_concept_sims[j].copy()
            concept_row[ci] = -np.inf
            second_ci = int(concept_row.argmax()) if n_concepts > 1 else -1
            second_sim = (
                float(concept_row[second_ci]) if second_ci >= 0 else float("-inf")
            )
            if not _passes_ambiguity_gate(sim, second_sim, cfg.ambiguity_margin):
                logger.debug(
                    "L3.5 rejected '%s': ambiguous %s(%.3f) vs %s(%.3f)",
                    span, concept.term, sim,
                    CONCEPTS[second_ci].term if second_ci >= 0 else "n/a", second_sim,
                )
                continue

            # (3) Hard-negative rejection gate: accept only if the span is
            # at least HARDNEG_MARGIN more similar to the concept than to
            # any of its hard negatives — excluding hard negatives shorter
            # than this span (see docstring).
            if hardneg_sims is not None and concept.hard_negatives:
                hn_mask = hardneg_idx_arr == ci          # (H,) bool
                if hardneg_word_counts is not None:
                    span_word_count = end_w - start_w
                    hn_mask = hn_mask & (hardneg_word_counts >= span_word_count)
                if hn_mask.any():
                    max_hn_sim = float(hardneg_sims[j, hn_mask].max())
                    if not _passes_hardneg_gate(sim, max_hn_sim):
                        logger.debug(
                            "L3.5 rejected '%s': concept_sim=%.3f hardneg_sim=%.3f",
                            span, sim, max_hn_sim,
                        )
                        continue

            matches.append(
                _Match(
                    span=span,
                    start_word=start_w,
                    end_word=end_w,
                    concept_term=concept.term,
                    snomed_id=concept.snomed_id,
                    similarity=sim,
                    runner_up_term=CONCEPTS[second_ci].term if second_ci >= 0 else None,
                    runner_up_similarity=second_sim if second_ci >= 0 else None,
                )
            )

        per_turn_matches.append(_best_non_overlapping(matches))
    return per_turn_matches


def normalize(turns: list[Turn]) -> list[Turn]:
    """Map lay medical terms in transcript turns to canonical clinical concepts.

    Uses parrotlet-e (fine-tuned bge-m3) embeddings. Candidate spans (1–3
    words) are compared against a combined reference of canonical terms +
    variants; matches above threshold pass an ambiguity-margin gate and a
    hard-negative rejection gate before being glossed non-destructively, e.g.
    ``sugar (Type 2 Diabetes Mellitus)`` — see _score_spans for the full gate
    order and src/config.py for the tuned thresholds.

    Model is loaded, used, and released in one call — memory discipline.

    Args:
        turns: Speaker-attributed transcript from L3 (or earlier).

    Returns:
        Same-length list of Turns with lay terms glossed where matched.
    """
    if not turns:
        return turns

    # Pass 1: Devanagari drug-name normalization (no model, always runs first).
    turns = [
        Turn(
            speaker_role=t.speaker_role,
            text=_normalize_drug_text(t.text),
            start=t.start,
            end=t.end,
        )
        for t in turns
    ]

    # Pass 2: lay-term concept glossing via parrotlet-e embeddings.
    backend = _EmbeddingBackend()
    ref_matrix, ref_ci_arr, hardneg_matrix, hardneg_idx_arr, hardneg_word_counts = (
        _encode_reference_matrices(backend)
    )
    all_spans, turn_span_offsets, turn_words, span_matrix = _encode_turn_spans(
        backend, turns
    )

    normalized: list[Turn] = []
    if span_matrix is not None:
        ref_sims = span_matrix @ ref_matrix.T  # (S, R)
        hardneg_sims = (
            span_matrix @ hardneg_matrix.T if hardneg_matrix is not None else None
        )  # (S, H)
        per_turn_matches = _score_spans(
            all_spans, turn_span_offsets, ref_sims, ref_ci_arr,
            hardneg_sims, hardneg_idx_arr, len(CONCEPTS), _default_match_config(),
            hardneg_word_counts=hardneg_word_counts,
        )
        zipped = zip(turns, turn_words, per_turn_matches, strict=True)
        for turn, words, matches in zipped:
            normalized.append(_gloss_turn(turn, matches, words))
    else:
        normalized = list(turns)

    backend.release()
    return normalized
