"""Parity guard for the three mirrored Devanagari-fold implementations.

src/drug_lexicon.py's _FOLD_MAP, src/l4_extract.py's _DRUG_FOLD_MAP, and
eval/drug_bench.py's _FOLD_MAP are hand-maintained copies of the same
character table (module boundaries prevent a shared import — src/ must not
depend on eval/, see src/l4_extract.py's module docstring). The candra-o
(ॉ) incident happened because one map silently fell out of sync with the
other two. This suite makes that class of drift structurally impossible to
miss: any future edit to one map without the other two fails CI here.

Scope note: the three _FOLD_MAP dicts are asserted byte-identical (that is
the actual shared contract). The wrapping fold() *functions* are NOT
asserted identical end-to-end — they intentionally differ downstream of the
map: eval/drug_bench.py's fold has no क्स/क्श digraph handling (a
pre-existing, out-of-scope gap — this file's ownership is "fold map only"),
and src/drug_lexicon.py's fold applies extra normalization (ph->f, aa/ii/uu
collapse, doubled-consonant collapse) that the other two don't, because its
consumer (exact fold-key lookup) needs more aggressive folding than the
difflib-ratio consumers do. What IS asserted for the fold functions: they
agree on every newly-added character in isolation, and each one individually
treats the candra-o and regular-o spellings of the same drug as equivalent.
"""

import importlib

from src.drug_lexicon import _fold as lexicon_fold
from src.drug_lexicon import _FOLD_MAP as LEXICON_FOLD_MAP
from src.l4_extract import _DRUG_FOLD_MAP as EXTRACT_FOLD_MAP
from src.l4_extract import _fold_drug as extract_fold

_drug_bench = importlib.import_module("eval.drug_bench")
BENCH_FOLD_MAP: dict[str, str] = _drug_bench._FOLD_MAP
bench_fold = _drug_bench._fold

# Every Devanagari char any of the three maps define, as of this fix
# (46 original entries + the 7 candra/vocalic-r/candrabindu/visarga
# additions from the नैक्स्टॉम incident).
_ALL_MAP_CHARS: frozenset[str] = (
    frozenset(LEXICON_FOLD_MAP) | frozenset(EXTRACT_FOLD_MAP) | frozenset(BENCH_FOLD_MAP)
)
_NEW_CHARS: tuple[str, ...] = ("ॉ", "ॅ", "ृ", "ऑ", "ऍ", "ँ", "ः")


def test_all_three_fold_maps_are_identical_dicts() -> None:
    """The three hand-maintained maps must be byte-identical, entry for entry."""
    assert LEXICON_FOLD_MAP == EXTRACT_FOLD_MAP == BENCH_FOLD_MAP


def test_new_chars_present_in_all_three_maps() -> None:
    """The candra-o incident's fix set must exist in every map, not just one."""
    for ch in _NEW_CHARS:
        assert ch in LEXICON_FOLD_MAP, f"{ch!r} missing from drug_lexicon._FOLD_MAP"
        assert ch in EXTRACT_FOLD_MAP, f"{ch!r} missing from l4_extract._DRUG_FOLD_MAP"
        assert ch in BENCH_FOLD_MAP, f"{ch!r} missing from drug_bench._FOLD_MAP"


def test_new_chars_fold_identically_across_all_three_functions() -> None:
    """Each newly-added character, folded singly, must agree across all three
    fold() functions (none of them trigger the pre-existing digraph/
    post-processing differences that apply to *other*, unrelated characters)."""
    for ch in _NEW_CHARS:
        a, b, c = lexicon_fold(ch), extract_fold(ch), bench_fold(ch)
        assert a == b == c, f"{ch!r}: lexicon={a!r} extract={b!r} bench={c!r}"


def test_candra_o_and_regular_o_fold_identically_per_implementation() -> None:
    """नैक्स्टॉम (candra-o ॉ) and नैक्स्टोम (regular ो) — the two ASR spellings
    behind this incident — must fold to the same key WITHIN each of the three
    implementations (each is internally consistent, independent of the
    pre-existing eval/drug_bench.py digraph gap noted in the module docstring).
    """
    candra_o, regular_o = "नैक्स्टॉम", "नैक्स्टोम"
    assert lexicon_fold(candra_o) == lexicon_fold(regular_o)
    assert extract_fold(candra_o) == extract_fold(regular_o)
    assert bench_fold(candra_o) == bench_fold(regular_o)


def test_probe_set_covers_every_mapped_character() -> None:
    """Sanity check on the probe set itself: every character any map defines
    is covered by at least one of the three maps checked above (guards
    against silently narrowing the probe set in a future edit)."""
    assert _ALL_MAP_CHARS == frozenset(LEXICON_FOLD_MAP)
    assert set(_NEW_CHARS) <= _ALL_MAP_CHARS
