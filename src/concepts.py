"""Canonical clinical concept table for L3.5 lay-term normalization.

Each Concept carries the authoritative clinical label, an optional SNOMED CT
identifier, and the lay/romanized surface forms expected in Indian clinical
speech (Hindi romanized, Marathi romanized, and English colloquial).

The variants list exists for documentation and testing; the runtime matcher
(l3_5_normalize.py) embeds canonical terms as the reference and lets the
embedding model handle semantic proximity.
"""

from dataclasses import dataclass, field


@dataclass
class Concept:
    """A canonical clinical concept with known lay-term variants."""

    term: str
    snomed_id: str | None
    variants: list[str] = field(default_factory=list)


CONCEPTS: list[Concept] = [
    Concept(
        term="Type 2 Diabetes Mellitus",
        snomed_id="44054006",
        variants=[
            "sugar", "shakkar", "madhumeh", "blood sugar", "sugar ki bimari",
            "diabetes", "high sugar", "sugar high", "shakar",
        ],
    ),
    Concept(
        term="Hypertension",
        snomed_id="38341003",
        variants=[
            "bp", "high bp", "blood pressure", "high blood pressure", "bp high",
            "uchha raktachap", "BP", "tension",
        ],
    ),
    Concept(
        term="Pain",
        snomed_id="22253000",
        variants=["dard", "pain", "dukh", "ache", "peedha", "dard hona", "takleef"],
    ),
    Concept(
        term="Fever",
        snomed_id="386661006",
        variants=["bukhar", "fever", "taap", "tez bukhar", "bukhaar", "jwar", "buxar"],
    ),
    Concept(
        term="Cough",
        snomed_id="49727002",
        variants=[
            "khansi", "khaasi", "cough", "khansi hona", "sukhi khansi", "dry cough",
            "khasi",
        ],
    ),
    Concept(
        term="Common Cold",
        snomed_id="82272006",
        variants=[
            "cold", "sardi", "zukam", "nasal congestion", "nose block", "runny nose",
            "sardi zukam", "nazla",
        ],
    ),
    Concept(
        term="Diarrhea",
        snomed_id="62315008",
        variants=[
            "loose motions", "diarrhea", "dast", "loose motion", "patle dast",
            "ulti dast", "latrine", "motions",
        ],
    ),
    Concept(
        term="Vomiting",
        snomed_id="422400008",
        variants=[
            "ulti", "vomiting", "nausea", "ji machalna", "jee machlana", "vomit",
            "ulti hona",
        ],
    ),
    Concept(
        term="Acid Reflux",
        snomed_id="698065002",
        variants=[
            "acidity", "gas", "pait me jalan", "jalan", "chest burn",
            "heartburn", "acid reflux", "gas trouble",
        ],
    ),
    Concept(
        term="Headache",
        snomed_id="25064002",
        variants=[
            "sar dard", "headache", "sir dard", "migraine", "sar mein dard",
            "sir mein dard",
        ],
    ),
    Concept(
        term="Weakness",
        snomed_id="13791008",
        variants=[
            "kamzori", "weakness", "thakaan", "fatigue", "kamjori", "thaka hua",
            "kamzoori",
        ],
    ),
    Concept(
        term="Urinary Tract Infection",
        snomed_id="68566005",
        variants=[
            "uti", "urine infection", "peshab mein jalan", "peshab infection",
            "urinary infection",
        ],
    ),
    Concept(
        term="Skin Rash",
        snomed_id="271807003",
        variants=[
            "rash", "khujli", "itching", "daane", "chakte", "skin problem",
            "allergy",
        ],
    ),
    Concept(
        term="Joint Pain",
        snomed_id="57676002",
        variants=[
            "joint pain", "gathiya", "ghutne ka dard", "jodo mein dard",
            "arthritis", "ghutna dard",
        ],
    ),
    Concept(
        term="Chest Pain",
        snomed_id="29857009",
        variants=[
            "chest pain", "seene mein dard", "seena dard", "chest dard",
            "sine mein dard",
        ],
    ),
    Concept(
        term="Shortness of Breath",
        snomed_id="267036007",
        variants=[
            "saans lena", "breathlessness", "saans phulna", "saans ki takleef",
            "dyspnea", "saans faulna",
        ],
    ),
    Concept(
        term="Constipation",
        snomed_id="14760008",
        variants=[
            "kabz", "constipation", "pet saaf nahi", "kabziyat", "qabz",
        ],
    ),
    Concept(
        term="Throat Pain",
        snomed_id="162397003",
        variants=[
            "gale mein dard", "throat pain", "gala kharab", "gale mein kharash",
            "throat infection",
        ],
    ),
]
