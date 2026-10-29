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

Each paper's output folder also gets run_info.json: the MLLM names and key tier from BioMiner's
config, the error if any, how many MLLM calls failed, and the input, output and total tokens (for cost). BioMiner
tries each call once and carries on with an empty answer, so a rate-limited run looks normal unless
that count is checked. A paper with run_info.json counts as finished: a rerun reuses it.

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
import signal
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


def _mllm_settings(config_file: Path) -> dict:
    """The MLLM names and key tier in BioMiner's config. They are logged and saved with every run."""
    try:
        config = yaml.safe_load(config_file.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        log.warning("BioMiner: cannot read %s (%s)", config_file, exc)
        config = {}
    model = config.get("model") or {}
    return {
        "text_model": model.get("text_mllm_type", "unknown"),
        "vision_model": model.get("vision_mllm_type", "unknown"),
        "tier": config.get("molcensus_tier", "unknown"),  # written by scripts/aws/biominer_gemini.py
    }


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


class _Stopped(Exception):
    """Raised on SIGTERM, which run_all.sh's stage time limits (GNU timeout) send."""


def _stop(signum, frame):
    raise _Stopped


def _token_totals(output: str) -> dict:
    """Add up the MOLCENSUS_USAGE lines that patched BioMiner prints, one per answered MLLM call.
    All None when there are none (BioMiner without scripts/aws/patch_biominer.py)."""
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "mllm_calls_answered": 0}
    for line in output.splitlines():
        if not line.startswith("MOLCENSUS_USAGE "):
            continue
        try:
            usage = json.loads(line.split(" ", 1)[1])
        except ValueError:
            continue
        totals["input_tokens"] += int(usage.get("input") or 0)
        totals["output_tokens"] += int(usage.get("output") or 0)
        totals["total_tokens"] += int(usage.get("total") or 0)
        totals["mllm_calls_answered"] += 1
    return totals if totals["mllm_calls_answered"] else dict.fromkeys(totals)


def _collect(records: dict, paper_output: Path, name: str, paper_id: int, runtime: float, error: str) -> dict:
    """Add one paper's predictions. Tables written before a failure are used; a missing table becomes one error row."""
    n_rows = {}
    for tool, pattern in OUTPUT_TABLES:
        tables = _read_tables(paper_output, pattern.format(name=name))
        if tables:
            rows = _to_predictions(tables, tool, paper_id, runtime, f"partial run: {error}" if error else "")
        else:
            rows = [_failure_row(tool, paper_id, runtime, error or "no readable output table")]
        records[tool].extend(rows)
        n_rows[tool] = len(rows) if tables else 0
    return n_rows


def _finished_run(info_file: Path) -> dict | None:
    """run_info.json of a paper that finished in an earlier run, or None."""
    try:
        return json.loads(info_file.read_text())
    except (OSError, ValueError):
        return None


def run_biominer(pdfs: dict[int, Path], settings: dict, output_root: Path, force: bool = False) -> dict[str, pd.DataFrame]:
    """Call BioMiner once per PDF. Returns predictions for "biominer" and "biominer_bioactivity".

    A paper whose folder already has run_info.json finished in an earlier run and is reused, not rerun
    (force=True reruns it). On SIGTERM (a stage time limit), the paper in progress is stopped and the
    predictions of the papers already finished are returned; a rerun then continues with the rest.
    """
    repo_text = expand(str(settings.get("repo_dir", "")))
    if not repo_text or not (Path(repo_text) / "example_open_source.py").exists():
        raise FileNotFoundError(
            "BioMiner repo not found. Clone https://github.com/jiaxianyan/BioMiner, follow its install steps, "
            "and set BIOMINER_DIR in .env."
        )
    repo_dir = Path(repo_text)
    python_exe = expand(str(settings.get("python_executable", ""))) or "python"

    environment = {**os.environ, "PYTHONPATH": str(repo_dir)}
    models = _mllm_settings(repo_dir / settings["config_path"])
    log.info("BioMiner models: text %s, vision %s, key tier %s (config %s)",
             models["text_model"], models["vision_model"], models["tier"], settings["config_path"])
    records = {tool: [] for tool, _ in OUTPUT_TABLES}
    previous_handler = signal.signal(signal.SIGTERM, _stop)
    paper_id = None
    try:
        for paper_id, pdf_path in pdfs.items():
            paper_output = output_root / str(paper_id)
            name = pdf_path.name.split(".")[0]
            earlier = None if force else _finished_run(paper_output / "run_info.json")
            if earlier:
                log.info("BioMiner: paper %s finished in an earlier run; reusing it (--force reruns it)", paper_id)
                _collect(records, paper_output, name, paper_id, earlier.get("runtime_s", 0.0), earlier.get("error", ""))
                continue

            shutil.rmtree(paper_output, ignore_errors=True)  # do not rescore tables left by an unfinished run
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
            error, api_failures, tokens = "", None, _token_totals("")
            try:
                completed = subprocess.run(
                    command, cwd=repo_dir, env=environment, capture_output=True, text=True,
                    timeout=int(settings.get("timeout_s", 3600)),
                )
                output = completed.stdout + "\n" + completed.stderr
                (paper_output / "biominer_stdout.log").write_text(output)
                if completed.returncode < 0:  # killed by a signal (a stage time limit): this paper is not finished
                    raise _Stopped
                # BioMiner tries each MLLM call once and uses an empty answer if it fails (for example on a
                # rate limit), so failed calls lose content silently. Count them from its log line. With
                # scripts/aws/patch_biominer.py it retries first and prints that line only for a call that
                # failed every try, so the count is always "calls lost".
                api_failures = output.count(" try failed:")
                tokens = _token_totals(output)
                if api_failures:
                    log.warning("BioMiner: paper %s: %d MLLM calls failed; their content is missing", paper_id, api_failures)
                if completed.returncode != 0:
                    error = f"exit code {completed.returncode}; see biominer_stdout.log"
            except (subprocess.TimeoutExpired, OSError) as exc:  # one slow or broken paper must not stop the run
                error = f"{type(exc).__name__}: {exc}"
            runtime = time.perf_counter() - started
            if error:
                log.warning("BioMiner: paper %s: %s", paper_id, error)

            n_rows = _collect(records, paper_output, name, paper_id, runtime, error)
            run_info = {
                "paper_id": paper_id, **models, "config_path": str(settings["config_path"]), "error": error,
                "api_failures": api_failures, **tokens, "runtime_s": round(runtime, 1), "n_predictions": n_rows,
            }
            (paper_output / "run_info.json").write_text(json.dumps(run_info, indent=2))  # marks the paper finished
    except _Stopped:
        log.warning("BioMiner: stopped by SIGTERM during paper %s; finished papers are kept and a rerun continues "
                    "with the rest", paper_id)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)

    return {tool: pd.DataFrame(rows, columns=PREDICTION_COLUMNS) for tool, rows in records.items()}
