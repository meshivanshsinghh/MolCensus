"""Run MolScribe (Qian et al., 2023) on molecule crops.

MolScribe reads one cropped molecule image and returns SMILES plus a confidence score.
It cannot tell that a crop has no molecule in it, so crop quality comes from the detector.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import pandas as pd

from . import PREDICTION_COLUMNS

log = logging.getLogger(__name__)

TOOL_NAME = "molscribe"


def patch_abbreviation_bug() -> None:
    """Fix MolScribe's abbreviation expansion (Me, Ph, OMe, ...), which returns '<invalid>' on macOS.

    `_expand_functional_group` deletes the bonds around an abbreviation, then reads their bond
    order from the deleted bond objects. That is undefined behavior in RDKit. It seems to work on
    Linux, but on macOS it returns 0 and the step fails with KeyError: 0. This reads the bond
    orders before the bonds are deleted and changes nothing else in MolScribe.
    """
    import inspect

    import molscribe.chemistry as chemistry

    if getattr(chemistry, "_molcensus_patched", False):  # MolScribe can be loaded twice in one run
        return
    source = inspect.getsource(chemistry._expand_functional_group)
    replacements = [
        ("bonds = atom.GetBonds()",
         "bonds = atom.GetBonds(); bond_types = [int(b.GetBondTypeAsDouble()) for b in bonds]"),
        ("for adjacent_atom, bond in zip(adjacent_atoms, bonds):",
         "for adjacent_atom, bond_type in zip(adjacent_atoms, bond_types):"),
        ("adjacent_atom.SetNumRadicalElectrons(int(bond.GetBondTypeAsDouble()))",
         "adjacent_atom.SetNumRadicalElectrons(bond_type)"),
    ]
    for old, new in replacements:
        if old not in source:
            raise RuntimeError(f"MolScribe source changed; abbreviation patch no longer applies: {old!r}")
        source = source.replace(old, new)
    exec(source, chemistry.__dict__)  # redefines the function inside MolScribe's own module
    chemistry._molcensus_patched = True


def run_molscribe(crops: pd.DataFrame, checkpoint_path: Path, device: str) -> pd.DataFrame:
    """Predict SMILES for every crop. A failed crop is recorded with its error instead of stopping the run."""
    import torch
    from molscribe import MolScribe

    patch_abbreviation_bug()

    model = MolScribe(str(checkpoint_path), device=torch.device(device))
    log.info("MolScribe loaded on %s; predicting %d crops", device, len(crops))

    records = []
    for position, crop in enumerate(crops.itertuples(index=False), start=1):
        started = time.perf_counter()
        smiles, confidence, error = "", math.nan, ""
        try:
            output = model.predict_image_file(crop.image_path, return_atoms_bonds=False, return_confidence=True)
            if isinstance(output, dict):  # current MolScribe API
                smiles = output.get("smiles") or ""
                confidence = float(output.get("confidence", math.nan))
            else:  # older releases return (smiles, molfile)
                smiles = output[0] or ""
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        records.append(
            {
                "tool": TOOL_NAME, "crop_id": crop.crop_id, "paper_id": crop.paper_id, "page": crop.page,
                "pred_smiles": smiles, "pred_label": "", "confidence": confidence,
                "runtime_s": round(time.perf_counter() - started, 3), "error": error,
            }
        )
        if position % 25 == 0:
            log.info("MolScribe: %d/%d crops done", position, len(crops))

    return pd.DataFrame(records, columns=PREDICTION_COLUMNS)
