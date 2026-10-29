"""Run a crop reader that lives in its own Python environment (MolSight, MolGlyph).

Their pinned packages clash with MolScribe's (albumentations, timm), so each one runs as a
subprocess with its own Python. A small worker script, <tool>_worker.py next to this file, reads a
CSV of crops and writes a CSV of SMILES. Only the worker imports the tool; this module moves files.

Worker contract:
    <python> <tool>_worker.py --checkpoint PATH --device cpu --crops IN.csv --out OUT.csv
    IN.csv:  crop_id, image_path
    OUT.csv: crop_id, pred_smiles, confidence, runtime_s, error   (one row per crop, written as it goes)

The worker runs from the tool's repo folder with that folder on PYTHONPATH. Its progress lines go
straight to this run's console. Rows it wrote before a crash are kept; crops without a row get the
crash as their error.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from ..config import resolve_path
from . import PREDICTION_COLUMNS

log = logging.getLogger(__name__)


def run_crop_reader(tool: str, crops: pd.DataFrame, settings: dict, checkpoint: Path, worker: Path | None = None) -> pd.DataFrame:
    """Run <tool>_worker.py on every crop and return predictions in the shared format."""
    repo_dir = resolve_path(settings.get("repo_dir"))
    python_exe = resolve_path(settings.get("python_executable"))
    if repo_dir is None or not repo_dir.is_dir() or python_exe is None or not python_exe.exists():
        raise FileNotFoundError(
            f"{tool}: repo_dir ({settings.get('repo_dir')}) or python_executable ({settings.get('python_executable')}) "
            "not found. See 'Readers in their own environment' in docs/usage.md."
        )
    worker = worker or Path(__file__).with_name(f"{tool}_worker.py")

    with tempfile.TemporaryDirectory() as tmp:
        crops_csv, out_csv = Path(tmp) / "crops.csv", Path(tmp) / "predictions.csv"
        crops[["crop_id", "image_path"]].to_csv(crops_csv, index=False)
        command = [
            str(python_exe), str(worker), "--checkpoint", str(checkpoint), "--device", str(settings.get("device", "cpu")),
            "--crops", str(crops_csv), "--out", str(out_csv),
        ]
        log.info("%s: %d crops, worker in %s", tool, len(crops), python_exe)
        error = ""
        try:
            completed = subprocess.run(
                command, cwd=repo_dir, env={**os.environ, "PYTHONPATH": str(repo_dir)},
                timeout=int(settings.get("timeout_s", 36000)),
            )
            if completed.returncode != 0:
                error = f"{tool} worker exit code {completed.returncode}"
        except (subprocess.TimeoutExpired, OSError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        if error:
            log.warning("%s: %s", tool, error)
        output = (
            pd.read_csv(out_csv, dtype={"crop_id": str, "pred_smiles": str, "error": str})
            if out_csv.exists() and out_csv.stat().st_size else pd.DataFrame(columns=["crop_id"])
        )

    output = output.drop_duplicates("crop_id", keep="last")
    merged = crops[["crop_id", "paper_id", "page"]].astype({"crop_id": str}).merge(
        output, on="crop_id", how="left", indicator=True
    )
    lost = merged["_merge"] == "left_only"
    merged.loc[lost, "error"] = error or "worker wrote no row for this crop"
    merged = merged.assign(
        tool=tool, pred_label="",
        pred_smiles=merged["pred_smiles"].fillna("") if "pred_smiles" in merged else "",
        error=merged["error"].fillna(""),
    )
    for column in ("confidence", "runtime_s"):
        if column not in merged:
            merged[column] = float("nan")
    return merged[PREDICTION_COLUMNS]
