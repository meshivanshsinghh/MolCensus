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


def run_molscribe(crops: pd.DataFrame, checkpoint_path: Path, device: str) -> pd.DataFrame:
    """Predict SMILES for every crop. A failed crop is recorded with its error instead of stopping the run."""
    import torch
    from molscribe import MolScribe

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
