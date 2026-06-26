"""CDSCO drug validation — checks if a drug name is in the approved drug set.

TODO: Replace the hardcoded seed set with the full CDSCO Schedule H / H1 list.
TODO: Add fuzzy matching to handle ASR garbling (e.g. 'Augmentin' → 'augmentin').
"""

import re

# Seed set of common CDSCO Schedule H prescription drugs (lowercase, stripped).
# Representative sample; not exhaustive.
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
        # Analgesics / NSAIDs
        "paracetamol",
        "ibuprofen",
        "diclofenac",
        "aspirin",
        "nimesulide",
        "aceclofenac",
        "naproxen",
        "mefenamic acid",
        "tramadol",
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
        # Antihistamines
        "cetirizine",
        "loratadine",
        "fexofenadine",
        "chlorpheniramine",
        "levocetirizine",
        "montelukast",
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
        # Vitamins / supplements
        "vitamin d3",
        "vitamin b12",
        "folic acid",
        "iron",
        "calcium",
        "cholecalciferol",
        # Respiratory
        "salbutamol",
        "budesonide",
        "levosalbutamol",
        "ipratropium",
        "salmeterol",
        "tiotropium",
        "theophylline",
        # Steroids
        "prednisolone",
        "methylprednisolone",
        "dexamethasone",
        "betamethasone",
        # Cardiovascular / lipid
        "atorvastatin",
        "rosuvastatin",
        "clopidogrel",
        "warfarin",
        "apixaban",
        "rivaroxaban",
        # Thyroid
        "levothyroxine",
        "thyronorm",
        "thyroxine",
        # Antimalarials / others
        "hydroxychloroquine",
        "chloroquine",
        "artemether",
        "lumefantrine",
    }
)

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(name: str) -> str:
    return _WHITESPACE_RE.sub(" ", name.strip().lower())


def validate_drug(name: str) -> bool:
    """Return True if the drug name appears in the CDSCO approved-drug seed set.

    Args:
        name: Drug name as extracted by L4 (may be mixed case, extra whitespace).

    Returns:
        True if found in the approved list, False otherwise.
    """
    return _normalize(name) in _APPROVED_DRUGS
