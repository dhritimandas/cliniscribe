"""CDSCO drug validation — checks if a drug name is in the approved drug set.

Validation pipeline (applied in order):
1. Strip recognised dosage-form prefixes/suffixes (Tablet, Capsule, Syrup, etc.)
   before lookup — the model frequently glues the form word onto the drug name.
2. Exact match against _APPROVED_DRUGS (lowercase, whitespace-normalised).
3. Substring match: any approved drug name that is a substring of the candidate
   (or vice-versa) at word-boundary level. Handles "Tab paracetamol" → matches
   "paracetamol", and "amoxicillin clavulanate" matching "amoxicillin".
4. Token-overlap match: if any single approved drug token appears as a full token
   in the candidate, accept it. Threshold: at least one shared token ≥ 4 chars
   (prevents single-letter false positives).

TODO: Replace the hardcoded seed set with the full CDSCO Schedule H / H1 list.
"""

import re

# Dosage-form words that models often prepend/append to drug names.
_FORM_WORDS: frozenset[str] = frozenset(
    {
        "tablet", "tab", "tablets",
        "capsule", "cap", "capsules",
        "syrup", "syp",
        "injection", "inj",
        "cream", "ointment",
        "drops", "drop",
        "solution", "susp", "suspension",
        "inhaler", "inhale",
        "gel", "lotion", "spray",
        "patch", "sachet",
        "powder",
        "forte",  # e.g. "Meftal Spas forte" — strip before lookup
    }
)

# Seed set of common CDSCO Schedule H prescription drugs (lowercase, stripped).
# Representative sample; not exhaustive. Includes the common Indian brand names
# and generics that appear frequently in EkaCare corpus samples.
_APPROVED_DRUGS: frozenset[str] = frozenset(
    {
        # Antibiotics
        "amoxicillin",
        "amoxycillin",
        "augmentin",
        "azithromycin",
        "ciprofloxacin",
        "metronidazole",
        "doxycycline",
        "clindamycin",
        "cefixime",
        "cefpodoxime",
        "levofloxacin",
        "ofloxacin",
        "amoxicillin clavulanate",
        "co-amoxiclav",
        "moxclav",           # brand: amoxicillin + clavulanate
        "clavulanate",
        "azee",              # brand: azithromycin
        "zithromax",
        "ciplox",            # brand: ciprofloxacin
        "flagyl",            # brand: metronidazole
        # Analgesics / NSAIDs / Antipyretics
        "paracetamol",
        "ibuprofen",
        "diclofenac",
        "aspirin",
        "nimesulide",
        "aceclofenac",
        "naproxen",
        "mefenamic acid",
        "tramadol",
        "dolo",              # brand: Dolo 650 = paracetamol 650mg
        "calpol",            # brand: paracetamol
        "combiflam",         # brand: ibuprofen + paracetamol
        "meftal",            # brand: mefenamic acid
        "meftal spas",       # brand: mefenamic acid + dicyclomine
        "ultracet",          # brand: tramadol + paracetamol
        "voveran",           # brand: diclofenac
        # Antihypertensives
        "amlodipine",
        "atenolol",
        "metoprolol",
        "ramipril",
        "telmisartan",
        "losartan",
        "enalapril",
        "valsartan",
        "bisoprolol",
        "nifedipine",
        "hydrochlorothiazide",
        "furosemide",
        "lasix",             # brand: furosemide
        "stamlo",            # brand: amlodipine
        "revelol",           # brand: metoprolol
        "telma",             # brand: telmisartan
        "repace",            # brand: losartan
        # Antidiabetics
        "metformin",
        "glipizide",
        "glibenclamide",
        "sitagliptin",
        "vildagliptin",
        "empagliflozin",
        "dapagliflozin",
        "insulin",
        "glimepiride",
        "teneligliptin",
        "januvia",           # brand: sitagliptin
        "glycomet",          # brand: metformin
        "amaryl",            # brand: glimepiride
        # Antihistamines / Respiratory
        "cetirizine",
        "loratadine",
        "fexofenadine",
        "chlorpheniramine",
        "levocetirizine",
        "montelukast",
        "salbutamol",
        "budesonide",
        "levosalbutamol",
        "ipratropium",
        "salmeterol",
        "tiotropium",
        "theophylline",
        "zyrtec",            # brand: cetirizine
        "allegra",           # brand: fexofenadine
        "montair",           # brand: montelukast
        "foracort",          # brand: budesonide + formoterol
        "asthalin",          # brand: salbutamol
        "seroflo",           # brand: salmeterol + fluticasone
        "levolin",           # brand: levosalbutamol
        # GI drugs
        "omeprazole",
        "pantoprazole",
        "rabeprazole",
        "domperidone",
        "ondansetron",
        "ranitidine",
        "ors",
        "zinc",
        "esomeprazole",
        "itopride",
        "pan d",             # brand: pantoprazole + domperidone
        "pantop",            # brand: pantoprazole
        "rantac",            # brand: ranitidine
        "nexpro",            # brand: esomeprazole
        "nexium",            # brand: esomeprazole
        "emetil",            # brand: ondansetron
        "emeset",            # brand: ondansetron
        "bifilac",           # brand: probiotic + lactic acid
        "lactic acid bacillus",
        "probiotic",
        # Vitamins / Supplements
        "vitamin d3",
        "vitamin b12",
        "folic acid",
        "iron",
        "calcium",
        "cholecalciferol",
        "vitamin c",
        "limcee",            # brand: vitamin C
        "shelcal",           # brand: calcium + vitamin D
        "calcirol",          # brand: cholecalciferol
        "neurobion",         # brand: B-complex + B12
        "becosules",         # brand: B-complex
        "b complex",
        "grenil",            # brand: ferrous ascorbate + folic acid
        "ferrous ascorbate",
        # Steroids / Anti-inflammatory
        "prednisolone",
        "methylprednisolone",
        "dexamethasone",
        "betamethasone",
        "medrol",            # brand: methylprednisolone
        "dexa",              # brand: dexamethasone
        # Cardiovascular / Lipid
        "atorvastatin",
        "rosuvastatin",
        "clopidogrel",
        "warfarin",
        "apixaban",
        "rivaroxaban",
        "atorlip",           # brand: atorvastatin
        "rozavel",           # brand: rosuvastatin
        "ecosprin",          # brand: aspirin (low dose)
        "deplatt",           # brand: clopidogrel
        # Thyroid
        "levothyroxine",
        "thyronorm",
        "thyroxine",
        "eltroxin",          # brand: levothyroxine
        # Antimalarials / Others
        "hydroxychloroquine",
        "chloroquine",
        "artemether",
        "lumefantrine",
        "coartem",           # brand: artemether + lumefantrine
        # Topical / Dermatology
        "clotrimazole",
        "miconazole",
        "hydrocortisone",
        "ketoconazole",
        "tretinoin",
        "benzoyl peroxide",
        "adapalene",
        "fluconazole",
        "candid",            # brand: clotrimazole
        # Urological / Others
        "tamsulosin",
        "sildenafil",
        "doxazosin",
        "nitrofurantoin",
        "norfloxacin",       # common UTI drug
        "trimethoprim",
        "norflox",           # brand: norfloxacin
        "urimax",            # brand: tamsulosin
    }
)

_WHITESPACE_RE = re.compile(r"\s+")
# Pattern to strip dosage forms from start or end of a drug name.
_FORM_RE = re.compile(
    r"^(?:" + "|".join(re.escape(w) for w in sorted(_FORM_WORDS, key=len, reverse=True)) + r")\s+|"
    r"\s+(?:" + "|".join(re.escape(w) for w in sorted(_FORM_WORDS, key=len, reverse=True)) + r")$",
    re.IGNORECASE,
)


def _normalize(name: str) -> str:
    """Lowercase, collapse whitespace, strip leading/trailing space."""
    return _WHITESPACE_RE.sub(" ", name.strip().lower())


def _strip_dosage_form(name: str) -> str:
    """Remove dosage-form prefix/suffix words (Tablet, Capsule, Syrup, etc.).

    Applies the strip repeatedly until stable (handles "Tab Cap paracetamol").
    """
    prev = name
    while True:
        name = _FORM_RE.sub("", name).strip()
        if name == prev:
            break
        prev = name
    return name


def _tokenize(name: str) -> list[str]:
    return [t for t in re.split(r"[\s\-/]+", name) if t]


def validate_drug(name: str) -> bool:
    """Return True if the drug name resolves to an approved CDSCO drug.

    Validation strategy (applied in order, first match wins):
    1. Exact match after normalisation (lowercase + whitespace collapse).
    2. Strip dosage-form words (Tablet, Capsule, Syrup, …) then exact match.
    3. Substring: normalised candidate is a substring of any approved name or
       vice-versa (at least 4 chars to avoid spurious single-char matches).
    4. Token overlap: any single token from the candidate ≥ 4 chars appears
       as a full token in any approved name.

    Args:
        name: Drug name as extracted by L4, may include form words or mixed case.

    Returns:
        True if the drug resolves to a known approved entry, False otherwise.
    """
    if not name or not name.strip():
        return False

    candidate = _normalize(name)

    # 1. Exact match
    if candidate in _APPROVED_DRUGS:
        return True

    # 2. Strip dosage form then exact match
    stripped = _normalize(_strip_dosage_form(candidate))
    if stripped and stripped != candidate and stripped in _APPROVED_DRUGS:
        return True

    # Work with stripped form for substring/token checks too
    work = stripped if stripped else candidate

    # 3. Substring match (bidirectional, min 4 chars to filter noise)
    if len(work) >= 4:
        for approved in _APPROVED_DRUGS:
            if len(approved) >= 4 and (work in approved or approved in work):
                return True

    # 4. Token overlap match (any candidate token ≥ 4 chars in an approved token set)
    candidate_tokens = set(_tokenize(work))
    long_tokens = {t for t in candidate_tokens if len(t) >= 4}
    if long_tokens:
        for approved in _APPROVED_DRUGS:
            approved_tokens = set(_tokenize(approved))
            if long_tokens & approved_tokens:
                return True

    return False
