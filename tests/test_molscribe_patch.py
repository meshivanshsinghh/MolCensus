"""Test the fix for MolScribe's abbreviation expansion. Run with: pytest -q"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_abbreviation_expands_after_patch():
    pytest.importorskip("molscribe")
    from rdkit import Chem

    import molscribe.chemistry as chemistry
    from molcensus.recognizers.molscribe_runner import patch_abbreviation_bug

    patch_abbreviation_bug()
    patch_abbreviation_bug()  # a second call (MolScribe loaded twice in one run) must be safe
    mol = Chem.RWMol(Chem.MolFromSmiles("*CN"))  # "Me" label bonded to CH2-NH2
    Chem.SetAtomAlias(mol.GetAtomWithIdx(0), "Me")
    smiles, _ = chemistry._expand_functional_group(mol.GetMol(), {})
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)) == Chem.MolToInchiKey(Chem.MolFromSmiles("CCN"))
