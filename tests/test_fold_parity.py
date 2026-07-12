"""Unification guard for the (formerly three, now one) Devanagari-fold
implementation.

Until the drug-Latin-canon fix, src/drug_lexicon.py, src/l4_extract.py, and
eval/drug_bench.py each hand-maintained their own copy of the same
Devanagari->Latin character map — the candra-o (ॉ) incident happened
because one map silently fell out of sync with the other two. Rather than
keep three copies in parity, there is now exactly ONE fold implementation
(src/drug_lexicon.py's `_fold`), imported by the other two modules. This
suite pins the unification itself (same function object, not just
equal-valued copies) so a future edit cannot reintroduce a second
implementation without this file catching it.
"""

import eval.drug_bench as drug_bench
from src.drug_lexicon import _fold as lexicon_fold
from src.l4_extract import _fold_drug as extract_fold

_NEW_CHARS: tuple[str, ...] = ("ॉ", "ॅ", "ृ", "ऑ", "ऍ", "ँ", "ः")


def test_l4_extract_imports_the_shared_fold() -> None:
    """src/l4_extract.py must not reimplement its own fold — it imports
    src/drug_lexicon.py's, aliased locally as _fold_drug."""
    assert extract_fold is lexicon_fold


def test_drug_bench_imports_the_shared_fold() -> None:
    """eval/drug_bench.py must not reimplement its own fold either."""
    assert drug_bench._fold is lexicon_fold


def test_new_chars_present_in_the_shared_map() -> None:
    """The candra-o incident's fix set must still be in the one shared map."""
    from src.drug_lexicon import _FOLD_MAP

    for ch in _NEW_CHARS:
        assert ch in _FOLD_MAP, f"{ch!r} missing from the shared _FOLD_MAP"


def test_candra_o_and_regular_o_fold_identically() -> None:
    """नैक्स्टॉम (candra-o ॉ) and नैक्स्टोम (regular ो) — the two ASR spellings
    behind the candra-o incident — must fold to the same key."""
    candra_o, regular_o = "नैक्स्टॉम", "नैक्स्टोम"
    assert lexicon_fold(candra_o) == lexicon_fold(regular_o)


def test_bench_scorer_now_shares_the_ks_digraph_handling() -> None:
    """Regression pin for a real gap the unification closed: eval/
    drug_bench.py's own former fold had no क्स/क्श digraph handling (see
    _FOLD_DIGRAPHS in src/drug_lexicon.py) — क्श would fold character-by-
    character to "ksh" instead of the digraph's "ks" -> "ks" (via the
    Latin-orthography x -> ks rule). Since bench now imports the shared
    fold, it picks up the same digraph handling automatically.
    """
    assert drug_bench._fold("क्श") == lexicon_fold("क्श")
    assert "ksh" not in drug_bench._fold("क्श")
