"""Run BioMiner (Yan et al., 2026) end to end on whole PDFs.

BioMiner parses the PDF with MinerU, finds molecules with MolDetV2, reads them with
MolGlyph, and uses a multimodal LLM to link labels and enumerate R-groups. It is run
through its own command-line entry point (example_open_source.py) inside its own
conda environment, so this wrapper calls it as a subprocess and only reads its output.

BioMiner writes its tables to <output_dir>/extraction_<text model>/<name>/, where <name> is the
PDF file name up to the first dot. Two of them are used here:

- <name>_structure_merged_output_*.csv: every structure the MLLM linked to a compound label
  (columns ligand, smiles). These are the "biominer" predictions.
- <name>_merge_*.csv: only the structures that were also linked to a bioactivity value
  (columns protein, ligand, type, value, unit, smiles). These are saved as a second tool,
  "biominer_bioactivity", and scored separately.

The structure tables are written before BioMiner's final merge step, and that step can fail
(it raises ValueError when labels cannot be mapped to SMILES). The structure tables are then kept,
and each kept row notes the failure in its error column.

Each paper's output folder also gets run_info.json: the MLLM names from BioMiner's config, the
error if any, and how many MLLM calls failed. BioMiner tries each call once and carries on with an
empty answer, so a rate-limited run looks normal unless that count is checked.

Requirements (from the BioMiner README): one 24 GB GPU in API mode, the MinerU,
MolDet and OCSR servers started from the BioMiner repo, and an MLLM API key in
BioMiner's own config file. This stage is disabled in configs/default.yaml until
GPU access is available.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd
import yaml

from ..config import expand
from . import PREDICTION_COLUMNS

log = logging.getLogger(__name__)

TOOL_NAME = "biominer"
BIOACTIVITY_TOOL_NAME = "biominer_bioactivity"

# (tool, file pattern under the paper's output folder); {name} is BioMiner's name for the paper.
OUTPUT_TABLES = [
    (TOOL_NAME, "extraction_*/{name}/{name}_structure_merged_output_*.csv"),
    (BIOACTIVITY_TOOL_NAME, "extraction_*/{name}/{name}_merge_*.csv"),
]


def _model_names(config_file: Path) -> dict:
    """The MLLM names in BioMiner's config. They are logged and saved with every run."""
    try:
        model = yaml.safe_load(config_file.read_text()).get("model") or {}
    except (OSError, yaml.YAMLError, AttributeError) as exc:
        log.warning("BioMiner: cannot read model names from %s (%s)", config_file, exc)
        model = {}
    return {"text_model": model.get("text_mllm_type", "unknown"), "vision_model": model.get("vision_mllm_type", "unknown")}


def _read_tables(paper_output: Path, pattern: str) -> list[pd.DataFrame]:
    """Read every table that matches the pattern. A file that cannot be read is logged and skipped."""
    tables = []
    for path in sorted(paper_output.glob(pattern)):
        if path.stem.endswith("_clean_new"):  # copies that BioMiner's BioVista evaluator writes next to the originals
            continue
        try:
            tables.append(pd.read_csv(path, dtype=str))
        except Exception as exc:  # one unreadable file must not stop the run
            log.warning("BioMiner: cannot read %s (%s: %s)", path.name, type(exc).__name__, exc)
    return tables


def _to_predictions(tables: list[pd.DataFrame], tool: str, paper_id: int, runtime_s: float, error: str = "") -> list[dict]:
    """One prediction per unique (label, SMILES) pair.

    The bioactivity table repeats a ligand once per measurement, so rows are deduplicated.
    """
    pairs = []
    for table in tables:
        columns = {str(column).lower(): column for column in table.columns}
        if "smiles" not in columns:
            continue
        for _, row in table.iterrows():
            smiles = str(row[columns["smiles"]]).strip()
            label = str(row[columns["ligand"]]).strip() if "ligand" in columns else ""
            if smiles and smiles.lower() != "nan":
                pairs.append(("" if label.lower() == "nan" else label, smiles))

    unique_pairs = list(dict.fromkeys(pairs))  # keeps first-seen order
    return [
        {
            "tool": tool, "crop_id": f"{tool}_p{paper_id}_{index:03d}", "paper_id": paper_id,
            "page": None, "pred_smiles": smiles, "pred_label": label, "confidence": math.nan,
            "runtime_s": round(runtime_s / max(len(unique_pairs), 1), 3), "error": error,
        }
        for index, (label, smiles) in enumerate(unique_pairs, start=1)
    ]


def _failure_row(tool: str, paper_id: int, runtime_s: float, error: str) -> dict:
    return {
        "tool": tool, "crop_id": f"{tool}_p{paper_id}_failed", "paper_id": paper_id,
        "page": None, "pred_smiles": "", "pred_label": "", "confidence": math.nan,
        "runtime_s": round(runtime_s, 1), "error": error,
    }


def run_biominer(pdfs: dict[int, Path], settings: dict, output_root: Path) -> dict[str, pd.DataFrame]:
    """Call BioMiner once per PDF. Returns predictions for "biominer" and "biominer_bioactivity"."""
    repo_text = expand(str(settings.get("repo_dir", "")))
    if not repo_text or not (Path(repo_text) / "example_open_source.py").exists():
        raise FileNotFoundError(
            "BioMiner repo not found. Clone https://github.com/jiaxianyan/BioMiner, follow its install steps, "
            "and set BIOMINER_DIR in .env."
        )
    repo_dir = Path(repo_text)
    python_exe = expand(str(settings.get("python_executable", ""))) or "python"

    environment = {**os.environ, "PYTHONPATH": str(repo_dir)}
    models = _model_names(repo_dir / settings["config_path"])
    log.info("BioMiner models: text %s, vision %s (config %s)", models["text_model"], models["vision_model"], settings["config_path"])
    records = {tool: [] for tool, _ in OUTPUT_TABLES}
    for paper_id, pdf_path in pdfs.items():
        paper_output = output_root / str(paper_id)
        shutil.rmtree(paper_output, ignore_errors=True)  # do not rescore tables left by an earlier run
        paper_output.mkdir(parents=True)
        command = [
            python_exe, "example_open_source.py",
            f"--config_path={settings['config_path']}",
            f"--pdf={pdf_path}",
            f"--output_dir={paper_output}",
            *shlex.split(expand(str(settings.get("extra_args", "")))),  # e.g. --biovista_evaluate
        ]
        log.info("BioMiner: paper %s", paper_id)
        started = time.perf_counter()
        error, api_failures = "", None
        try:
            completed = subprocess.run(
                command, cwd=repo_dir, env=environment, capture_output=True, text=True,
                timeout=int(settings.get("timeout_s", 3600)),
            )
            output = completed.stdout + "\n" + completed.stderr
            (paper_output / "biominer_stdout.log").write_text(output)
            # BioMiner tries each MLLM call once and uses an empty answer if it fails (for example on a
            # rate limit), so failed calls lose content silently. Count them from its log line. With
            # scripts/aws/patch_biominer.py it retries first and prints that line only for a call that
            # failed every try, so the count is always "calls lost".
            api_failures = output.count(" try failed:")
            if api_failures:
                log.warning("BioMiner: paper %s: %d MLLM calls failed; their content is missing", paper_id, api_failures)
            if completed.returncode != 0:
                error = f"exit code {completed.returncode}; see biominer_stdout.log"
        except (subprocess.TimeoutExpired, OSError) as exc:  # one slow or broken paper must not stop the run
            error = f"{type(exc).__name__}: {exc}"
        runtime = time.perf_counter() - started
        if error:
            log.warning("BioMiner: paper %s: %s", paper_id, error)

        # Tables written before a failure are still used; a missing table becomes one error row.
        name = pdf_path.name.split(".")[0]
        n_rows = {}
        for tool, pattern in OUTPUT_TABLES:
            tables = _read_tables(paper_output, pattern.format(name=name))
            if tables:
                rows = _to_predictions(tables, tool, paper_id, runtime, f"partial run: {error}" if error else "")
            else:
                rows = [_failure_row(tool, paper_id, runtime, error or "no readable output table")]
            records[tool].extend(rows)
            n_rows[tool] = len(rows) if tables else 0

        run_info = {
            "paper_id": paper_id, **models, "config_path": str(settings["config_path"]), "error": error,
            "api_failures": api_failures, "runtime_s": round(runtime, 1), "n_predictions": n_rows,
        }
        (paper_output / "run_info.json").write_text(json.dumps(run_info, indent=2))

    return {tool: pd.DataFrame(rows, columns=PREDICTION_COLUMNS) for tool, rows in records.items()}
