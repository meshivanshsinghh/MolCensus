"""Run BioMiner (Yan et al., 2026) end to end on whole PDFs.

BioMiner parses the PDF with MinerU, finds molecules with MolDetV2, reads them with
MolGlyph, and uses a multimodal LLM to link labels and enumerate R-groups. It is run
through its own command-line entry point (example_open_source.py) inside its own
conda environment, so this wrapper calls it as a subprocess and only reads its output.

Requirements (from the BioMiner README): one 24 GB GPU in API mode, the MinerU,
MolDet and OCSR servers started from the BioMiner repo, and an MLLM API key in
BioMiner's own config file. This stage is disabled in configs/default.yaml until
GPU access is available.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

from ..config import expand
from . import PREDICTION_COLUMNS

log = logging.getLogger(__name__)

TOOL_NAME = "biominer"


def _read_tables_with_smiles(output_dir: Path) -> list[pd.DataFrame]:
    """Collect every CSV/TSV/JSON table in BioMiner's output folder that has a SMILES column."""
    tables = []
    for path in sorted(output_dir.rglob("*")):
        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                table = pd.read_csv(path)
            elif suffix == ".tsv":
                table = pd.read_csv(path, sep="\t")
            elif suffix == ".json":
                table = pd.read_json(path)
            else:
                continue
        except (ValueError, UnicodeDecodeError, pd.errors.ParserError):
            continue  # not a table we can use
        if any(str(column).lower() == "smiles" for column in table.columns):
            tables.append(table)
    return tables


def _to_predictions(tables: list[pd.DataFrame], paper_id: int, runtime_s: float) -> list[dict]:
    """Convert BioMiner's bioactivity rows to one prediction per unique (label, SMILES) pair.

    BioMiner repeats a ligand once per bioactivity measurement, so rows are deduplicated.
    Ligands reported without a bioactivity value may never appear in its output.
    """
    pairs = []
    for table in tables:
        columns = {str(column).lower(): column for column in table.columns}
        for _, row in table.iterrows():
            smiles = str(row[columns["smiles"]]).strip()
            label = str(row[columns["ligand"]]).strip() if "ligand" in columns else ""
            if smiles and smiles.lower() != "nan":
                pairs.append((label, smiles))

    unique_pairs = list(dict.fromkeys(pairs))  # keeps first-seen order
    return [
        {
            "tool": TOOL_NAME, "crop_id": f"biominer_p{paper_id}_{index:03d}", "paper_id": paper_id,
            "page": None, "pred_smiles": smiles, "pred_label": label, "confidence": math.nan,
            "runtime_s": round(runtime_s / max(len(unique_pairs), 1), 3), "error": "",
        }
        for index, (label, smiles) in enumerate(unique_pairs, start=1)
    ]


def run_biominer(pdfs: dict[int, Path], settings: dict, output_root: Path) -> pd.DataFrame:
    """Call BioMiner once per PDF and gather its SMILES into the shared prediction format."""
    repo_text = expand(str(settings.get("repo_dir", "")))
    if not repo_text or not (Path(repo_text) / "example_open_source.py").exists():
        raise FileNotFoundError(
            "BioMiner repo not found. Clone https://github.com/jiaxianyan/BioMiner, follow its install steps, "
            "and set BIOMINER_DIR in .env."
        )
    repo_dir = Path(repo_text)
    python_exe = expand(str(settings.get("python_executable", ""))) or "python"

    environment = {**os.environ, "PYTHONPATH": str(repo_dir)}
    records = []
    for paper_id, pdf_path in pdfs.items():
        paper_output = output_root / str(paper_id)
        shutil.rmtree(paper_output, ignore_errors=True)  # do not rescore tables left by an earlier run
        paper_output.mkdir(parents=True)
        command = [
            python_exe, "example_open_source.py",
            f"--config_path={settings['config_path']}",
            f"--pdf={pdf_path}",
            f"--output_dir={paper_output}",
        ]
        log.info("BioMiner: paper %s", paper_id)
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command, cwd=repo_dir, env=environment, capture_output=True, text=True,
                timeout=int(settings.get("timeout_s", 3600)),
            )
            (paper_output / "biominer_stdout.log").write_text(completed.stdout + "\n" + completed.stderr)
            error = "" if completed.returncode == 0 else f"exit code {completed.returncode}; see biominer_stdout.log"
        except (subprocess.TimeoutExpired, OSError) as exc:  # one slow or broken paper must not stop the run
            error = f"{type(exc).__name__}: {exc}"
        runtime = time.perf_counter() - started

        if error:
            records.append(
                {
                    "tool": TOOL_NAME, "crop_id": f"biominer_p{paper_id}_failed", "paper_id": paper_id,
                    "page": None, "pred_smiles": "", "pred_label": "", "confidence": math.nan,
                    "runtime_s": round(runtime, 1), "error": error,
                }
            )
            continue

        records.extend(_to_predictions(_read_tables_with_smiles(paper_output), paper_id, runtime))

    return pd.DataFrame(records, columns=PREDICTION_COLUMNS)
