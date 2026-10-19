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

    per_paper = pd.read_csv(results_dir / "per_paper_metrics.csv", dtype={"paper_id": str})
    comparison = pd.read_csv(results_dir / "comparison.csv")
    environment_file = results_dir / "environment.json"
    environment = json.loads(environment_file.read_text()) if environment_file.exists() else {}
    tools = sorted(per_paper["tool"].unique())

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

    tables = {"per_paper_metrics": per_paper, "comparison": comparison}
    crops_file = results_dir / "crops_per_paper.csv"
    if crops_file.exists():
        tables["crops_per_paper"] = pd.read_csv(crops_file)
    run.log({name: wandb.Table(dataframe=frame) for name, frame in tables.items()})

    figure = results_dir / "figures" / "recall_by_paper.png"
    if figure.exists():
        run.log({"recall_by_paper": wandb.Image(str(figure))})

    url = run.url
    run.finish()
    log.info("Published metrics to W&B: %s", url)
    return url
