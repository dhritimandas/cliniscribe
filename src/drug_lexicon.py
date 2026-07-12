"""Expanded Indian drug lexicon + variant-absorbing fold-key matcher.

Replaces reactive per-variant dictionary growth (a new hand-typed curated
entry for every ASR misspelling) with ONE canonical entry per drug plus a
matcher that folds any spelling variant to a coarse phonetic key and looks it
up — exact first, then bounded edit distance. The curated tables in
src/l3_5_normalize.py stay as the fast, zero-risk path for historically-seen
distortions; this module is the tier that catches distortions nobody has
seen yet, without inventing a new dictionary entry for each one.

Lexicon provenance: the CDSCO seed set (src/cdsco.py, 163 entries) merged
with ~380 additional real Indian prescription drugs — common NLEM-class
generics and top prescribed brands — knowledge-sourced from common Indian
prescription patterns; verify against CDSCO/NLEM publications before
regulatory use. Real drugs only, no inventions. One canonical spelling per
drug; spelling variants are absorbed by the fold matcher, not hand-listed.

AMBIGUITY GUARD (the core safety property): if a query fold key is within
the allowed edit-distance bound of MORE THAN ONE distinct canonical drug,
canonicalize_drug_span() returns None rather than guessing. A wrong drug
substitution is a patient-safety incident; leaving a span unmatched (so it
falls through to the next tier, or is left as-is) is not.
"""

import logging
import re
import unicodedata

from src.cdsco import _APPROVED_DRUGS

logger = logging.getLogger(__name__)

# ── Additional canonical drug names (knowledge-sourced, see module docstring) ─
# Grouped by category for readability only — canonicalize_drug_span() treats
# the lexicon as one flat namespace. No spelling variants: brand + generic
# both present where both are actually prescribed under those names, but
# never two spellings of the same name (e.g. not both "amoxycillin" forms —
# CDSCO already carries the one spelling variant pair it shipped with).

_ANTIMICROBIALS: frozenset[str] = frozenset({
    "azithral", "azicip", "clindac", "taxim", "taxim o", "monocef", "oflox",
    "zenflox", "norflox tz", "metrogyl", "unasyn", "augmentin duo", "clavam",
    "amoxyclav", "cefpod", "ceftum", "zifi", "cifran", "ciprobid", "doxt sl",
    "doxt", "vibramycin", "roxid", "levoflox", "ceftas", "amoxil", "cefolac",
    "ceftriaxone", "cefuroxime", "cefotaxime", "cefaclor", "cefdinir",
    "cefepime", "amikacin", "gentamicin", "vancomycin", "linezolid",
    "erythromycin", "clarithromycin", "roxithromycin", "tetracycline",
    "minocycline", "cotrimoxazole", "oz", "ampicillin", "cloxacillin",
    "flucloxacillin", "penicillin", "benzathine penicillin", "netilmicin",
    "colistin", "meropenem", "imipenem", "piperacillin tazobactam",
    "moxifloxacin", "gatifloxacin", "sparfloxacin", "nalidixic acid",
    "furazolidone", "secnidazole", "tinidazole", "ornidazole",
})

_ANALGESICS: frozenset[str] = frozenset({
    "zerodol", "zerodol sp", "zerodol p", "hifenac", "aceclo", "ibugesic",
    "flexon", "sumo", "nise", "etoshine", "etoricoxib", "dolonex",
    "piroxicam", "brufen", "disprin", "saridon", "crocin", "etodolac",
    "diclomol", "celecoxib", "ketorolac", "indomethacin", "tapentadol",
    "codeine", "naxdom", "meloxicam", "ketoprofen", "flurbiprofen",
    "diacerein", "glucosamine", "chondroitin", "colchicine", "febuxostat",
    "allopurinol",
})

_GI: frozenset[str] = frozenset({
    "pantocid", "digene", "gelusil", "eno", "cremaffin", "duphalac", "looz",
    "smuth", "unienzyme", "aciloc", "cyclopam", "spasmonil", "drotin",
    "enterogermina", "sporlac", "lactobacillus", "domstal", "ondem",
    "perinorm", "metoclopramide", "dicyclomine", "lansoprazole", "rabium",
    "razo", "omez", "pan", "pantodac", "sucralfate", "famotidine",
    "misoprostol", "loperamide", "racecadotril", "drotaverine", "bisacodyl",
    "ispaghula", "mosapride", "pancreatin", "ursodeoxycholic acid",
    "rifaximin",
})

_RESPIRATORY_ENT: frozenset[str] = frozenset({
    "sinarest", "ascoril", "grilinctus", "cheston cold", "cheston",
    "montek lc", "montek", "montair lc", "wikoryl", "delcon", "ambrodil",
    "mucinac", "zedex", "alkasol", "cital", "niftas", "uribid", "levocet",
    "alerid", "okacet", "avil", "corex", "phensedyl", "benadryl",
    "diphenhydramine", "promethazine", "bromhexine", "ambroxol",
    "guaifenesin", "deriphyllin", "duolin", "budecort", "otrivin",
    "nasivion", "xylometazoline", "oxymetazoline", "fluticasone",
    "mometasone", "triamcinolone", "hydroxyzine", "desloratadine",
    "ketotifen", "calpol t", "bilastine", "rupatadine", "olopatadine",
    "azelastine", "beclomethasone", "formoterol", "indacaterol",
    "glycopyrronium", "roflumilast",
})

_ANTIDIABETIC: frozenset[str] = frozenset({
    "glycomet gp", "glimestar", "janumet", "istamet", "galvus met",
    "lantus", "mixtard", "gluconorm", "repaglinide", "voglibose",
    "trajenta", "linagliptin", "forxiga", "jardiance", "glucobay",
    "acarbose", "huminsulin", "pioglitazone", "canagliflozin",
    "remogliflozin", "saxagliptin", "exenatide", "liraglutide", "victoza",
})

_CARDIOVASCULAR: frozenset[str] = frozenset({
    "telma am", "telma h", "amlokind", "losar", "aten", "metolar", "ciplar",
    "propranolol", "concor", "cardace", "envas", "amlopres", "storvas",
    "atorva", "rosuvas", "clopilet", "ecosprin av", "spironolactone",
    "torsemide", "dytor", "aldactone", "candesartan", "olmesartan",
    "indapamide", "chlorthalidone", "methyldopa", "labetalol", "diltiazem",
    "verapamil", "isosorbide dinitrate", "isosorbide mononitrate",
    "nitroglycerin", "digoxin", "amiodarone", "prazosin", "terazosin",
    "carvedilol", "nebivolol", "eplerenone",
})

_STEROIDS_DERM: frozenset[str] = frozenset({
    "betnovate", "quadriderm", "surfaz", "surfaz sn", "itrafirst",
    "itraconazole", "t bact", "mupirocin", "soframycin", "framycetin",
    "clobetasol", "fluocinolone", "panderm", "desonide", "calamine",
    "permethrin", "terbinafine", "luliconazole", "sertaconazole",
    "eberconazole", "povidone iodine", "silver sulfadiazine",
    "fusidic acid", "neomycin", "salicylic acid",
})

_THYROID: frozenset[str] = frozenset({"thyrox"})

_VITAMINS_SUPPLEMENTS: frozenset[str] = frozenset({
    "zincovit", "evion", "orofer", "orofer xt", "livogen", "dexorange",
    "supradyn", "a to z", "revital", "shelcal 500", "gemcal", "uprise d3",
    "becadexamin", "polybion", "fefol", "autrin", "methylcobalamin",
    "pyridoxine", "thiamine", "biotin", "vitamin b6", "vitamin b1",
    "multivitamin", "omega 3", "cyanocobalamin", "riboflavin", "niacin",
    "pantothenic acid", "vitamin b2", "vitamin b3", "vitamin a",
    "vitamin e", "vitamin k", "zinc sulphate", "magnesium",
    "ferrous sulphate", "ferrous fumarate", "ferric carboxymaltose",
})

_ANTIPARASITIC_ANTIVIRAL: frozenset[str] = frozenset({
    "lariago", "falcigo", "artesunate", "albendazole", "zentel",
    "ivermectin", "acyclovir", "valcivir", "valacyclovir", "mebendazole",
    "praziquantel", "diethylcarbamazine", "oseltamivir", "ganciclovir",
    "valganciclovir", "lamivudine", "zidovudine", "tenofovir", "efavirenz",
})

_NEURO_PSYCH: frozenset[str] = frozenset({
    "alprax", "restyl", "calmpose", "diazepam", "clonazepam", "gabapentin",
    "pregabalin", "lyrica", "amitriptyline", "escitalopram", "nexito",
    "sizodon", "risperidone", "sodium valproate", "carbamazepine",
    "phenytoin", "levetiracetam", "topiramate", "baclofen", "tizanidine",
    "chlorzoxazone", "duloxetine", "venlafaxine", "sertraline",
    "fluoxetine", "haloperidol", "olanzapine", "quetiapine", "lithium",
})

_ELECTROLYTES_ENZYMES: frozenset[str] = frozenset({
    "electral", "aristozyme", "vizylac",
})

_URO_GYNAE: frozenset[str] = frozenset({
    "veltam", "volini", "finasteride", "dutasteride", "oxybutynin",
    "solifenacin", "mifepristone", "clomiphene", "progesterone",
    "medroxyprogesterone", "norethisterone", "ethinylestradiol",
    "unwanted 72", "i pill",
})

_OPHTHALMIC_ANESTHETIC: frozenset[str] = frozenset({
    "timolol", "hypromellose", "lignocaine", "bupivacaine",
})

# Category tuples kept for reporting only (not part of the lookup API).
_CATEGORIES: tuple[tuple[str, frozenset[str]], ...] = (
    ("antimicrobials", _ANTIMICROBIALS),
    ("analgesics", _ANALGESICS),
    ("gi", _GI),
    ("respiratory_ent", _RESPIRATORY_ENT),
    ("antidiabetic", _ANTIDIABETIC),
    ("cardiovascular", _CARDIOVASCULAR),
    ("steroids_derm", _STEROIDS_DERM),
    ("thyroid", _THYROID),
    ("vitamins_supplements", _VITAMINS_SUPPLEMENTS),
    ("antiparasitic_antiviral", _ANTIPARASITIC_ANTIVIRAL),
    ("neuro_psych", _NEURO_PSYCH),
    ("electrolytes_enzymes", _ELECTROLYTES_ENZYMES),
    ("uro_gynae", _URO_GYNAE),
    ("ophthalmic_anesthetic", _OPHTHALMIC_ANESTHETIC),
)

_ADDITIONAL_DRUGS: frozenset[str] = frozenset().union(*(c[1] for c in _CATEGORIES))

# The full canonical namespace: CDSCO seed set + additional knowledge-sourced
# drugs, merged (not duplicated — union is a no-op for any name already in
# _APPROVED_DRUGS).
DRUG_LEXICON: frozenset[str] = _APPROVED_DRUGS | _ADDITIONAL_DRUGS


# ── Fold: coarse Devanagari→Latin phonetic key, shared by both scripts ──────
# Adapted from the _fold_drug approach in src/l4_extract.py (same digraph
# handling for the क्स/क्श → x loanword sound), extended with the
# _normalize_roman-style collapses from src/l3_5_normalize.py (aa/ii/uu →
# single vowel, doubled-consonant collapse, ph → f) so Devanagari-origin and
# Latin-origin spans land in the SAME fold-key space and one index serves
# both. Matras and nukta are already erased by the character map (short/long
# vowel signs collapse to the same Latin vowel; nukta consonants like ज़/फ़
# map straight to z/f).
_FOLD_DIGRAPHS: tuple[tuple[str, str], ...] = (("क्स", "x"), ("क्श", "x"))
_FOLD_MAP: dict[str, str] = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "च": "ch", "छ": "chh",
    "ज": "j", "झ": "jh", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p",
    "फ": "f", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r",
    "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "ज़": "z", "फ़": "f", "ा": "a", "ि": "i", "ी": "i", "ु": "u",
    "ू": "u", "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ं": "n",
    "अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "्": "",
    # Candra vowels + vocalic r + candrabindu/visarga — absent from the
    # original map, which let ASR spellings using them (e.g. नैक्स्टॉम with
    # candra-o ॉ) escape every drug-normalization tier. See LEARNINGS.md /
    # tests/test_fold_parity.py.
    "ॉ": "o", "ॅ": "e", "ृ": "ri", "ऑ": "o", "ऍ": "e", "ँ": "n", "ः": "",
}
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")  # despaced: space is not kept either
_DOUBLE_CONSONANT_RE = re.compile(r"([bcdfghjklmnpqrstvwxyz])\1")
_DIGIT_TOKEN_RE = re.compile(r"^\d+$")


def _fold(text: str) -> str:
    """Coarse phonetic fold to a despaced Latin key.

    Args:
        text: A drug-name span, Devanagari, Latin, or mixed.

    Returns:
        Lowercase alnum-only string with no spaces — the fold key.
    """
    text = unicodedata.normalize("NFC", text)
    for digraph, latin in _FOLD_DIGRAPHS:
        text = text.replace(digraph, latin)
    folded = "".join(_FOLD_MAP.get(ch, ch) for ch in text).lower()
    folded = _NON_ALNUM_RE.sub("", folded)
    folded = folded.replace("ph", "f")
    folded = folded.replace("aa", "a").replace("ii", "i").replace("uu", "u")
    folded = _DOUBLE_CONSONANT_RE.sub(r"\1", folded)
    return folded


# Fold-key index: fold_key -> set of canonical drugs sharing that exact key.
# A key with >1 canonical is a genuine exact-fold collision (two different
# drugs that fold identically) — treated as ambiguous, same as a fuzzy
# collision, rather than picking one arbitrarily.
_FOLD_INDEX: dict[str, set[str]] = {}
for _drug in DRUG_LEXICON:
    _key = _fold(_drug)
    if not _key:
        continue
    _FOLD_INDEX.setdefault(_key, set()).add(_drug)


def _distance_bound(key_len: int) -> int | None:
    """Max edit distance allowed for a query fold key of this length.

    Returns:
        None below length 9 (exact-fold-only). 2 for length 9+.

    A length 5-8 / distance-1 tier was tried and rejected by the drug-bench
    GATE (see LEARNINGS.md): on the 49-clip bench it produced 3 wrong-drug
    substitutions and only 1 correct one — "दो तीन" (a number phrase, "two-
    three") -> drotin, "से vital" -> revital, and a physician's surname
    "पांडे" (Pandey) -> pan d. All three false positives landed in the
    5-8-char range, where the combinatorial space of short, unrelated
    Hindi/English words and names is large enough that a single edit is not
    a safe signal. Length 9+ / distance-2 produced zero wrong-drug events on
    the same bench (e.g. "aur gmenti" -> augmentin, a real distortion of a
    gold drug). Wrong drug is worse than unknown, so the shorter tier is
    dropped rather than patched with a per-word denylist — a denylist is
    exactly the reactive per-variant growth this module replaces.
    """
    if key_len < 9:
        return None
    return 2


def _levenshtein(a: str, b: str, max_dist: int) -> int:
    """Edit distance between a and b, length-filtered against max_dist.

    Args:
        a: First string (the query fold key).
        b: Second string (a candidate fold key).
        max_dist: If |len(a) - len(b)| already exceeds this, the true
            distance is skipped and max_dist + 1 is returned (cheap reject).

    Returns:
        The edit distance, or max_dist + 1 if it provably exceeds max_dist.
    """
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def canonicalize_drug_span(text: str) -> tuple[str, float] | None:
    """Canonicalize a drug-name span against the expanded lexicon.

    Digit-only tokens (dose numbers) are excluded from the fold key — the
    caller (src/l3_5_normalize.py) is responsible for verifying any dose
    digits in the original span survive in the returned canonical before
    substituting; this function only resolves the drug name itself.

    Tries an exact fold-key hit first, then bounded edit distance (see
    _distance_bound). Returns None — rather than a guess — whenever the
    match is ambiguous (2+ distinct canonical drugs within the bound) or
    when no candidate is within bound. Wrong drug is worse than unknown.

    Args:
        text: A drug-name candidate span (already isolated by the caller;
            Devanagari, Latin, or mixed script).

    Returns:
        (canonical_name, confidence) with confidence 1.0 for an unambiguous
        exact match, or 1 - distance / key_length for an edit-distance
        match; None if there is no safe match.
    """
    tokens = [t for t in text.split() if not _DIGIT_TOKEN_RE.match(t)]
    if not tokens:
        return None
    key = _fold(" ".join(tokens))
    if not key:
        return None

    exact = _FOLD_INDEX.get(key)
    if exact is not None:
        if len(exact) == 1:
            return next(iter(exact)), 1.0
        logger.debug("drug_lexicon: ambiguous exact fold '%s' -> %s", key, sorted(exact))
        return None

    bound = _distance_bound(len(key))
    if bound is None:
        return None

    best_dist: dict[str, int] = {}
    for cand_key, canonicals in _FOLD_INDEX.items():
        dist = _levenshtein(key, cand_key, bound)
        if dist > bound:
            continue
        for canonical in canonicals:
            if canonical not in best_dist or dist < best_dist[canonical]:
                best_dist[canonical] = dist

    if not best_dist:
        return None
    if len(best_dist) > 1:
        logger.debug("drug_lexicon: ambiguous fuzzy '%s' -> %s", key, sorted(best_dist))
        return None

    (canonical, dist), = best_dist.items()
    return canonical, round(1 - dist / len(key), 4)
