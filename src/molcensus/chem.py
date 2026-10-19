"""Chemistry helpers built on RDKit.

Molecules are compared by InChIKey rather than by canonical SMILES string, because
canonical SMILES can change between RDKit versions while the InChIKey does not.
The first 14 characters of an InChIKey encode only the connectivity (the skeleton);
the rest adds stereochemistry and protonation (Heller et al., 2015). Matching on the
first block alone gives a stereo-blind score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")  # invalid predicted SMILES are expected; keep the log readable


@dataclass
class MoleculeFeatures:
    valid: bool
    canonical_smiles: str | None = None
    inchikey: str | None = None       # full key: connectivity + stereo + protonation
    skeleton_key: str | None = None   # first InChIKey block: connectivity only
    fingerprint: Any = None           # Morgan bit vector for Tanimoto similarity


class Featurizer:
    """Turns SMILES strings into the identifiers and fingerprints the scorer needs."""

    def __init__(self, radius: int = 2, n_bits: int = 2048):
        self._generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    def featurize(self, smiles: Any) -> MoleculeFeatures:
        if not isinstance(smiles, str) or not smiles.strip():
            return MoleculeFeatures(valid=False)
        molecule = Chem.MolFromSmiles(smiles.strip())
        if molecule is None:
            return MoleculeFeatures(valid=False)

        try:
            inchikey = Chem.MolToInchiKey(molecule) or None  # empty string when InChI fails (e.g. "*" atoms)
        except Exception:
            inchikey = None

        return MoleculeFeatures(
            valid=True,
            canonical_smiles=Chem.MolToSmiles(molecule),
            inchikey=inchikey,
            skeleton_key=inchikey.split("-")[0] if inchikey else None,
            fingerprint=self._generator.GetFingerprint(molecule),
        )

    @staticmethod
    def similarities(fingerprint: Any, others: list[Any]) -> list[float]:
        """Tanimoto similarity of one fingerprint against a list of fingerprints."""
        if fingerprint is None or not others:
            return []
        return list(DataStructs.BulkTanimotoSimilarity(fingerprint, others))
