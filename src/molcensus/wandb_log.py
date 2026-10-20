"""Optional: publish benchmark metrics to Weights & Biases.

No model is trained here, so W&B is used only as a place to browse and share results.
Only metrics and counts are uploaded. Predicted and ground-truth SMILES stay on the
machine, because the course ground truth is private sponsor data.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def publish(results_dir: Path, config: dict, project: str, entity: str | None = None) -> str:
    """Create one W&B run with pooled metrics as the summary and the result tables and figure attached."""
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is not installed. Run `pip install wandb`, then `wandb login`.") from exc

    per_paper_file = results_dir / "per_paper_metrics.csv"
    per_paper = (
        pd.read_csv(per_paper_file, dtype={"paper_id": str}) if per_paper_file.exists()
        else pd.DataFrame(columns=["tool", "paper_id"])
    )
    comparison_file = results_dir / "comparison.csv"
    public_file = results_dir / "public" / "public_metrics.csv"
    environment_file = results_dir / "environment.json"
    environment = json.loads(environment_file.read_text()) if environment_file.exists() else {}
    public = pd.read_csv(public_file) if public_file.exists() else None
    tools = sorted(set(per_paper["tool"]) | (set(public["tool"]) if public is not None else set()))

    run = wandb.init(
        project=project,
        entity=entity or os.environ.get("WANDB_ENTITY") or None,
        job_type="eval",
        name=f"{'+'.join(tools)}__{config['detection']['method']}",
        config={
            "papers": config["papers"],
            "tools": tools,
            "detection_method": config["detection"]["method"],
            "render_dpi": config["render"]["dpi"],
            "scoring": config["scoring"],
            "environment": environment,
        },
    )

    # Pooled metrics become the run summary, namespaced by tool (e.g. "molscribe/exact_recall").
    for row in per_paper[per_paper["paper_id"] == "ALL"].to_dict(orient="records"):
        tool = row.pop("tool")
        row.pop("paper_id")
        run.summary.update({f"{tool}/{name}": value for name, value in row.items()})

    if public is not None:  # public crop check: accuracy per (tool, dataset)
        for row in public.to_dict(orient="records"):
            run.summary.update({f"public/{row['tool']}/{row['dataset']}/{k}": v for k, v in row.items()
                                if k not in ("tool", "dataset")})

    tables = {}
    if len(per_paper):
        tables["per_paper_metrics"] = per_paper
    if comparison_file.exists():
        tables["comparison"] = pd.read_csv(comparison_file)
    crops_file = results_dir / "crops_per_paper.csv"
    if crops_file.exists():
        tables["crops_per_paper"] = pd.read_csv(crops_file)
    if public is not None:
        tables["public_check"] = public
    run.log({name: wandb.Table(dataframe=frame) for name, frame in tables.items()})

    for figure_name in ("recall_by_paper", "public_check"):
        figure = results_dir / "figures" / f"{figure_name}.png"
        if figure.exists():
            run.log({f"figure_{figure_name}": wandb.Image(str(figure))})

    url = run.url
    run.finish()
    log.info("Published metrics to W&B: %s", url)
    return url
