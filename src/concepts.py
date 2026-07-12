"""Canonical clinical concept table for L3.5 lay-term normalization.

Each Concept carries the authoritative clinical label, an optional SNOMED CT
identifier, lay/romanized surface forms (documentation + testing), and
hard negatives.

Hard negatives are non-clinical phrases that share surface form or semantic
proximity with the concept but represent everyday non-medical usage. The
runtime matcher uses them to reject false-positive matches via a margin test:
a span is accepted only if it scores more similar to the canonical term than
to any of the concept's hard negatives (see l3_5_normalize.py).

The variants list has TWO roles:
- **Runtime**: included alongside the canonical term in the reference matrix.
  Necessary for abbreviations where the embedding model cannot bridge the gap
  alone (e.g. "sugar" → "Type 2 Diabetes Mellitus" cosine=0.33 without
  variants; "sugar" → "sugar" variant ~1.0 with variants). Devanagari forms
  also help for script-level coverage.
- **Documentation**: records known surface forms and romanised variants for
  human review and future eval.

The hard_negatives field guards against common-word false positives triggered
by short variants (e.g. "cold" in variants for Common Cold). The runtime
matcher applies a margin test and rejects matches that are too close to a hard
negative of the matched concept.
"""

from dataclasses import dataclass, field


@dataclass
class Concept:
    """A canonical clinical concept with lay-term variants and hard negatives."""

    term: str
    snomed_id: str | None
    variants: list[str] = field(default_factory=list)
    hard_negatives: list[str] = field(default_factory=list)


# ── Everyday-word guard (concept-matcher hardening, mirrors the drug
# matcher's near-collision discipline) ──────────────────────────────────────
# A curated set of common Hindi/English conversation words that can NEVER be
# glossed as a clinical concept, regardless of embedding similarity — the
# categorical counterpart to the drug matcher's length-floor rule (some spans
# live in "crowded" everyday-word space where similarity alone is not a safe
# signal). Scoped to five closed classes: time words, numbers/quantifiers,
# function/filler words, common verbs, and kinship/person words — NOT the
# concept-adjacent nouns already covered by per-concept hard_negatives
# (दवा, दाल, पेड़, कमरा, दान, कब्र, नाच, पेशा, भूल, सांप — see the audit in
# commit 26c42ea), which is a different collision class with a different
# fix (a margin test against a specific concept, not a blanket ban).
#
# Deliberately Devanagari + English only (no romanized-Hindi tier): keeps the
# list close to the ~60-100 entry target while covering the two scripts our
# ASR actually emits (config.py: "we only support hi/en/mr ... all Latin or
# Devanagari"). Romanized-Hindi fillers ("subah", "aaj") are a known residual
# gap, mitigated (not eliminated) by the higher unigram bar and the
# hard-negative margin gate — not by this list.
#
# Runtime rule: if a concept legitimately lists one of these words as a
# variant, the variant listing wins (see EVERYDAY_WORDS below, which removes
# any such overlap at import time). Checked empirically: zero of the ~110
# single-word CONCEPTS variants collide with this list (see
# tests/test_concept_guard.py::test_everyday_words_documented_variant_overlap).
_TIME_WORDS: frozenset[str] = frozenset({
    "हफ्ता", "हफ्ते", "हफते", "हफ़्ते", "दिन", "रात", "सुबह", "शाम", "कल",
    "आज", "साल", "महीना", "महीने",
    "week", "weeks", "day", "days", "night", "morning", "evening",
    "tomorrow", "yesterday", "today", "year", "month", "months",
})
_NUMBER_WORDS: frozenset[str] = frozenset({
    "एक", "दो", "तीन", "चार", "पांच", "दस", "बार", "थोड़ा", "ज्यादा", "कम",
    "one", "two", "three", "four", "five", "more", "less", "little",
    "times", "again",
})
_FILLER_WORDS: frozenset[str] = frozenset({
    "और", "तो", "हाँ", "हां", "नहीं", "है", "हैं", "था", "थी", "थे", "भी",
    "ही", "जी",
    "okay", "ok", "yes", "no", "please", "uh", "um",
})
_VERB_WORDS: frozenset[str] = frozenset({
    "खाना", "खाओ", "लेना", "लो", "करना", "करो", "जाना", "जाओ", "पीना",
    "सोना", "आना", "देना",
    "take", "eat", "do", "go", "drink", "sleep", "give", "come",
})
_KINSHIP_WORDS: frozenset[str] = frozenset({
    "डॉक्टर", "भाई", "बेटा", "बेटी", "माँ", "पापा", "पत्नी", "पति", "बहन",
    "doctor", "patient", "brother", "sister", "mother", "father", "wife",
    "husband", "son", "daughter", "sir", "madam",
})
_EVERYDAY_WORDS_CURATED: frozenset[str] = (
    _TIME_WORDS | _NUMBER_WORDS | _FILLER_WORDS | _VERB_WORDS | _KINSHIP_WORDS
)


CONCEPTS: list[Concept] = [
    # ── Chronic / metabolic ──────────────────────────────────────────────────
    Concept(
        term="Type 2 Diabetes Mellitus",
        snomed_id="44054006",
        variants=[
            "sugar", "shakkar", "madhumeh", "blood sugar", "sugar ki bimari",
            "diabetes", "high sugar", "sugar high", "shakar",
            "शुगर", "मधुमेह",
        ],
        hard_negatives=[
            "sugar in tea", "sweet food", "jaggery", "mishri", "sugar cane",
            "meetha khana",
        ],
    ),
    Concept(
        term="Hypertension",
        snomed_id="38341003",
        variants=[
            "bp", "high bp", "blood pressure", "high blood pressure", "bp high",
            "uchha raktachap", "BP", "tension",
        ],
        hard_negatives=[
            "work stress", "exam pressure", "tension at home", "job stress",
            "mental tension", "padhne ka pressure",
        ],
    ),
    # ── Symptoms ─────────────────────────────────────────────────────────────
    Concept(
        term="Pain",
        snomed_id="22253000",
        variants=[
            "dard", "pain", "dukh", "ache", "peedha", "dard hona", "takleef",
            "दर्द", "दर्द होना",
        ],
        hard_negatives=[
            "emotional pain", "painful memory", "heartbreak", "pain in the neck",
            # Audit (2026-07-12): "rain" is one edit from the "pain" variant —
            # monsoon small talk is common in Indian clinic conversation.
            "rain", "it is raining",
        ],
    ),
    Concept(
        term="Fever",
        snomed_id="386661006",
        variants=[
            "bukhar", "fever", "taap", "tez bukhar", "bukhaar", "jwar", "buxar",
            "बुखार", "तेज़ बुखार", "फीवर",
        ],
        # Fever is low false-positive risk in clinical speech; minimal hard negatives
        hard_negatives=[
            "fever pitch", "high temperature room",
        ],
    ),
    Concept(
        term="Cough",
        snomed_id="49727002",
        variants=[
            "khansi", "khaasi", "cough", "khansi hona", "sukhi khansi", "dry cough",
            "khasi", "खाँसी", "खांसी", "खाँसना",
        ],
        hard_negatives=[
            "cough syrup brand", "cough drop candy",
        ],
    ),
    Concept(
        term="Common Cold",
        snomed_id="82272006",
        variants=[
            "cold", "sardi", "zukam", "nasal congestion", "nose block", "runny nose",
            "sardi zukam", "nazla", "जुकाम", "सर्दी",
        ],
        hard_negatives=[
            "cold water", "cold weather", "cold drink", "feeling cold",
            "ice cold", "it's cold outside", "thanda pani",
            # Audit (2026-07-12): "gold" is one edit from the "cold" variant.
            "gold", "gold price", "gold jewellery",
        ],
    ),
    Concept(
        term="Diarrhea",
        snomed_id="62315008",
        variants=[
            "loose motions", "diarrhea", "dast", "loose motion", "patle dast",
            "ulti dast", "latrine", "motions", "दस्त", "पतले दस्त",
        ],
        hard_negatives=[
            "loose clothes", "loose ends", "motion picture",
        ],
    ),
    Concept(
        term="Vomiting",
        snomed_id="422400008",
        variants=[
            "ulti", "vomiting", "ji machalna", "jee machlana", "vomit",
            "ulti hona", "उल्टी", "जी मचलाना",
        ],
        hard_negatives=[
            "vomit-inducing smell", "disgusting",
        ],
    ),
    Concept(
        term="Nausea",
        snomed_id="422587007",
        variants=[
            "nausea", "ji machal raha", "queasy", "uneasy stomach", "jee ghubrana",
            "meetli aa rahi", "मितली",
        ],
        hard_negatives=[
            "nauseous about the news", "sick of work",
        ],
    ),
    Concept(
        term="Acid Reflux",
        snomed_id="698065002",
        variants=[
            "acidity", "gas", "pait me jalan", "jalan", "chest burn",
            "heartburn", "acid reflux", "gas trouble", "जलन", "एसिडिटी",
        ],
        hard_negatives=[
            "gas cylinder", "cooking gas", "car gas", "gas leak", "gas bill",
            "gas station",
        ],
    ),
    Concept(
        term="Headache",
        snomed_id="25064002",
        variants=[
            "sar dard", "headache", "sir dard", "sar mein dard",
            "sir mein dard", "सिर दर्द", "सर दर्द", "सर में दर्द",
        ],
        hard_negatives=[
            "administrative headache", "logistics headache", "this is a headache",
        ],
    ),
    Concept(
        term="Migraine",
        snomed_id="37796009",
        variants=[
            "migraine", "aadha sir dard", "half head pain", "migrain",
            "one side headache", "aadhe sir ka dard", "आधा सिर दर्द",
        ],
        hard_negatives=[
            "migraine trigger food",
        ],
    ),
    Concept(
        term="Weakness",
        snomed_id="13791008",
        variants=[
            "kamzori", "weakness", "thakaan", "fatigue", "kamjori", "thaka hua",
            "kamzoori", "कमज़ोरी", "थकान",
        ],
        hard_negatives=[
            "academic weakness", "weak argument", "wifi weak", "structural weakness",
            "weakness in character",
        ],
    ),
    Concept(
        term="Urinary Tract Infection",
        snomed_id="68566005",
        variants=[
            "uti", "urine infection", "peshab mein jalan", "peshab infection",
            "urinary infection", "पेशाब में जलन",
        ],
        hard_negatives=[
            "urine test result normal",
            # Audit (2026-07-12): पेशा ("occupation/profession") is one edit
            # from पेशाब ("urine") — a doctor's own history-taking question
            # ("आपका पेशा क्या है?") could risk collision.
            "पेशा", "आपका पेशा क्या है",
        ],
    ),
    Concept(
        term="Skin Rash",
        snomed_id="271807003",
        variants=[
            "rash", "khujli", "itching", "daane", "chakte", "skin problem",
            "allergy", "खुजली", "दाने",
        ],
        hard_negatives=[
            "skin care routine", "face wash", "skin cream", "moisturizer",
            "rash decision",
            # Audit (2026-07-12): दान ("donation") is one edit from the दाने
            # variant; "cash" is one edit from the "rash" variant (cash
            # payment talk is routine in an Indian clinic visit).
            "दान", "दान देना", "cash", "cash payment",
        ],
    ),
    Concept(
        term="Fungal Infection",
        snomed_id="414561005",
        variants=[
            "fungal", "ringworm", "daad", "daad khujli", "tinea",
            "fungal infection", "daad ki bimari", "दाद",
        ],
        hard_negatives=[
            "fungal growth on bread", "mold", "mushroom",
            # Audit (2026-07-12): दाल ("lentils") is one edit from दाद
            # (ringworm) — one of the most common everyday food words.
            "दाल", "दाल चावल",
            # Concept Matcher Rebuild audit (2026-07-12, programmatic one-edit
            # scan, tests/test_concept_guard.py): दान ("donation") is ALSO one
            # edit from दाद — missed by the original manual audit in commit
            # 26c42ea, which only checked common everyday NOUNS by hand;
            # दान was already flagged as a Skin Rash hard negative (दाने
            # collision) but never cross-checked against Fungal Infection.
            "दान", "दान देना", "रक्तदान",
        ],
    ),
    Concept(
        term="Allergic Rhinitis",
        snomed_id="61582004",
        variants=[
            "nasal allergy", "nose allergy", "dust allergy", "season allergy",
            "allergic rhinitis", "nak se paani", "naak bahna", "नाक बहना",
        ],
        hard_negatives=[
            "perfume allergy test", "food allergy",
            # Audit (2026-07-12): नाच ("dance") is one edit from the नाक
            # ("nose") root of the "नाक बहना" variant.
            "नाच", "नाच गाना",
        ],
    ),
    Concept(
        term="Joint Pain",
        snomed_id="57676002",
        variants=[
            "joint pain", "gathiya", "ghutne ka dard", "jodo mein dard",
            "arthritis", "ghutna dard", "जोड़ों में दर्द", "घुटने का दर्द",
        ],
        hard_negatives=[
            "joint venture", "joint meeting", "joint effort",
        ],
    ),
    Concept(
        term="Chest Pain",
        snomed_id="29857009",
        variants=[
            "chest pain", "seene mein dard", "seena dard", "chest dard",
            "sine mein dard", "सीने में दर्द",
        ],
        hard_negatives=[
            "chest of drawers", "treasure chest",
        ],
    ),
    Concept(
        term="Shortness of Breath",
        snomed_id="267036007",
        variants=[
            "saans lena", "breathlessness", "saans phulna", "saans ki takleef",
            "dyspnea", "saans faulna", "सांस फूलना", "सांस की तकलीफ",
        ],
        hard_negatives=[
            "short on time", "run out of breath after exercise",
            # Real incident (outputs/<session>, 2026-07-12): "एक हफते के लिए"
            # ("for one week") was glossed "(Shortness of Breath)" — हफ्ते
            # ("week") is one edit from हांफते ("huffing/panting", a genuine
            # near-synonym of this concept), so the embedding model conflated
            # them. All common spellings guarded, not just the one that fired.
            "हफ्ता", "हफ्ते", "हफते", "हफ़्ते",
            # Audit bonus (same one-edit-collision class): सांप ("snake") is
            # one edit from सांस ("breath"), a listed variant.
            "सांप",
        ],
    ),
    Concept(
        term="Constipation",
        snomed_id="14760008",
        variants=[
            "kabz", "constipation", "pet saaf nahi", "kabziyat", "qabz",
            "कब्ज़", "पेट साफ नहीं",
        ],
        hard_negatives=[
            "constipated bureaucracy", "system is constipated",
            # Audit (2026-07-12): कब्र ("grave") is one edit from कब्ज
            # (constipation) — a common word in family-history/bereavement talk.
            "कब्र", "कब्र में",
        ],
    ),
    Concept(
        term="Throat Pain",
        snomed_id="162397003",
        variants=[
            "gale mein dard", "throat pain", "gala kharab", "gale mein kharash",
            "throat infection", "गले में दर्द", "गला खराब",
        ],
        hard_negatives=[
            "throat singing", "clear your throat",
        ],
    ),
    # ── New concepts from EkaCare dataset ────────────────────────────────────
    Concept(
        term="Abdominal Pain",
        snomed_id="21522001",
        variants=[
            "pait mein dard", "stomach pain", "pet dard", "tummy ache",
            "abdominal pain", "pet mein dard", "pait dard", "पेट में दर्द",
            "पेट दर्द",
        ],
        hard_negatives=[
            "stomach for adventure", "can't stomach this",
            # Audit (2026-07-12): पेड़ ("tree") is one edit from पेट
            # ("stomach"), a listed variant.
            "पेड़", "पेड़ के नीचे",
        ],
    ),
    Concept(
        term="Back Pain",
        snomed_id="161891005",
        variants=[
            "kamar dard", "peeth dard", "back dard", "lower back pain",
            "back pain", "spine pain", "kamar mein dard", "कमर दर्द",
            "पीठ दर्द",
        ],
        hard_negatives=[
            "back to work", "back of the room",
            # Audit (2026-07-12): कमरा ("room") is one edit from कमर
            # ("waist/lower back") — a very common word in clinic instructions
            # ("मरीज़ को कमरे में ले जाओ").
            "कमरा", "कमरे में जाओ",
        ],
    ),
    Concept(
        term="Asthma",
        snomed_id="195967001",
        variants=[
            "dama", "saans ki bimari", "asthma", "asthma attack",
            "breathlessness attack", "inhaler use", "phuphus ki bimari",
            "दमा",
        ],
        hard_negatives=[
            "asthmatic performance", "short of breath after running",
            # Audit (2026-07-12): दवा ("medicine") is one edit from दमा
            # (asthma) — दवा is one of the highest-frequency words in ANY
            # consultation, making this the highest-risk finding of the audit.
            "दवा", "दवा लो", "दवा खाओ",
        ],
    ),
    Concept(
        term="Upper Respiratory Tract Infection",
        snomed_id="54150009",
        variants=[
            "urti", "throat infection", "gale ka infection", "nose throat problem",
            "upper respiratory infection", "respiratory tract infection",
        ],
        hard_negatives=[
            "respiratory rate normal",
        ],
    ),
    Concept(
        term="Anxiety",
        snomed_id="48694002",
        variants=[
            "ghabrahat", "ghabrao", "nervousness", "panic", "anxiety attack",
            "tension", "chinta", "घबराहट", "चिंता",
        ],
        hard_negatives=[
            "exam anxiety is normal", "anxious about results",
            "financial anxiety", "work anxiety",
        ],
    ),
    Concept(
        term="Loss of Appetite",
        snomed_id="79890006",
        variants=[
            "bhook nahi", "not feeling hungry", "no appetite", "khana nahi",
            "bhukh nahi lagti", "loss of appetite", "भूख नहीं",
        ],
        hard_negatives=[
            "appetite for success", "no appetite for risk",
            # Audit (2026-07-12): भूल ("forgot/mistake") is one edit from भूख
            # ("hunger"), a listed variant.
            "भूल", "मैं भूल गया",
        ],
    ),
]


def _all_variant_strings_lower() -> frozenset[str]:
    """Every CONCEPTS variant string, lower-cased, any span length.

    Used only to strip everyday-word/variant overlaps at import time — see
    EVERYDAY_WORDS below. Multi-word variants are included too so a variant
    listing wins even if (hypothetically) it matched a guard entry exactly.
    """
    return frozenset(v.lower() for c in CONCEPTS for v in c.variants)


# The runtime guard set: curated everyday words minus any word a concept
# legitimately lists as a variant (variant listing wins — see module
# docstring). Empirically empty today; computed defensively so a future
# CONCEPTS edit can never silently make a real lay term unglossable.
EVERYDAY_WORDS: frozenset[str] = _EVERYDAY_WORDS_CURATED - _all_variant_strings_lower()
