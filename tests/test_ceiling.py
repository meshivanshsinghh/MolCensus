"""Recall-ceiling rules (scripts/analysis/ceiling_*.py) on made-up molecules. No private data. Run: pytest -q"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import ceiling_evidence as ce  # noqa: E402

M = Chem.MolFromSmiles
QUINAZOLINE = M("*c1ccc2ncnc(N*)c2c1")  # 11-atom core with an R at C6 and one on the amine N
LINKER = M("*OCCOCCOCCO[C@H]1CCN(*)C1")  # a linker fragment cut at both ends, as MolGlyph writes it


@pytest.mark.parametrize("smiles, expected", [
    ("COc1ccc2ncnc(Nc3ccccc3)c2c1", True),     # R1 = OMe, R2 = Ph
    ("c1ccc2ncnc(Nc3ccccc3)c2c1", True),       # R1 = H
    ("COc1ccc2ncnc(Nc3ccccc3)c2c1Cl", False),  # a Cl where the scaffold has no R
    ("COc1ccc2ncnc(N(C)c3ccccc3)c2c1", True),  # two groups on the R-bearing N
    ("Cc1ccc2ncnc(C)c2c1", False),             # no N: the core is not there
])
def test_table_match_scaffold(smiles, expected):
    assert ce.table_match(QUINAZOLINE, M(smiles)) is expected


def test_table_match_fragment_and_size_floor():
    assert ce.table_match(LINKER, M("c1ccccc1OCCOCCOCCOC1CCN(C(=O)C)C1"))
    assert not ce.table_match(LINKER, M("c1ccccc1OCCOCCOCCOC1CCN(C(=O)C)C1C"))  # methyl on the ring
    assert not ce.table_match(M("*c1ccccc1"), M("Clc1ccccc1"))  # 6-atom core: below MIN_CORE_ATOMS
    assert not ce.table_match(M("*CCCCCCCC*"), M("C1CCCCCCC1"))  # a bond between core atoms the core lacks
    assert ce.table_match(M("*CCCCCCCC*"), M("C1CCCCCCCC1"))  # the two R groups joined through one more atom


def test_table_search_runs_in_a_child():
    search = ce.TableSearch(timeout_s=60)
    try:
        assert search("COc1ccc2ncnc(Nc3ccccc3)c2c1", ["*CC", "*c1ccc2ncnc(N*)c2c1"]) == 1
        assert search("CCO", ["*c1ccc2ncnc(N*)c2c1"]) is None
        assert search("CCO", []) is None
    finally:
        search.close()


@pytest.mark.parametrize("new_crop, table, close_read, bucket", [
    (True, False, False, 2), (True, True, False, 2), (False, True, False, 3), (False, False, True, 1),
    (True, False, True, None), (False, True, True, None), (False, False, False, None),
])
def test_layer2_bucket(new_crop, table, close_read, bucket):
    assert ce.layer2_bucket(new_crop, table, close_read) == bucket


@pytest.mark.parametrize("name, kind, needle", [
    ("17q", "number and letter", "17q"), ("Z12", "letter and number", "Z12"), ("7", "number", None),
    ("unnamed cmpd 9 in Scheme 8", "unnamed, scheme given", "Scheme 8"), ("Intermediate 99", "intermediate", None),
    ("Examplinib", "name or code", "Examplinib"),
])
def test_label_kind(name, kind, needle):
    assert ce.label_kind(name)[0] == kind and ce.searchable(name) == needle


# ---- Layer 3 (ceiling_judges.py) ----------------------------------------------------------------------
import ceiling_judges as cj  # noqa: E402
import pandas as pd  # noqa: E402


def test_cohen_kappa():
    assert cj.cohen_kappa(list("AACC"), list("AACC")) == 1.0
    assert cj.cohen_kappa(list("ACAC"), list("AACC")) == 0.0  # agreement no better than chance
    assert abs(cj.cohen_kappa(list("AAAC"), list("AACC")) - 0.5) < 1e-9  # observed 0.75, chance 0.5
    assert cj.cohen_kappa(list("AAA"), list("AAA")) == 1.0  # one shared category


@pytest.mark.parametrize("claude, gemini, trusted, bucket", [
    ("C", "C", True, 3), ("B", "B", True, 2), ("C", "C", False, None), ("A", "C", True, None),
    ("E", "E", True, None), (None, "C", True, None),
])
def test_accepted(claude, gemini, trusted, bucket):
    assert cj.accepted(claude, gemini, trusted) == bucket


def test_parse_answers():
    text = '```json\n{"answers": [{"id": "M101", "answer": "c", "reason": " a  table "}, {"id": "M999", "answer": "A"},' \
           ' {"id": "M102", "answer": "Z"}]}\n```'
    assert cj.parse_answers(text, ["M101", "M102"]) == {"M101": {"answer": "C", "crop": None, "page": None, "reason": "a table"}}
    assert cj.parse_answers("not json", ["M101"]) == {}


def test_accuracy_counts_missing_as_wrong():
    judged = pd.DataFrame({"known": ["A", "A", "C", ""], "known_source": ["found", "found", "2b", ""],
                           "claude": ["A", None, "C", "B"]})
    s = cj.accuracy(judged, "claude")
    assert (s["correct"], s["n"], s["by_source"]) == (2, 3, {"2b": (1, 1), "found": (1, 2)})
