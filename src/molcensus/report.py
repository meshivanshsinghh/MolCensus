"""Write the side-by-side comparison, a JSON summary, a Markdown results page and a figure."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, no display needed
import matplotlib.pyplot as plt
import pandas as pd

log = logging.getLogger(__name__)

# Metrics shown in the comparison table, with readable names.
HEADLINE_METRICS = {
    "n_predictions": "Predictions",
    "validity_rate": "Valid SMILES rate",
    "exact_precision": "Exact precision",
    "exact_recall": "Exact recall",
    "exact_f1": "Exact F1",
    "stereo_blind_recall": "Stereo-blind recall",
    "mean_best_tanimoto": "Mean best Tanimoto",
    "near_match_recall": "Near-match recall (Tanimoto >= threshold)",
    "mean_confidence": "Mean confidence",
}

PACKAGES = ["rdkit", "pymupdf", "ultralytics", "torch", "MolScribe", "pandas", "numpy", "huggingface_hub"]


def comparison_table(per_paper: pd.DataFrame) -> pd.DataFrame:
    """Pooled (ALL) metrics with one column per tool."""
    pooled = per_paper[per_paper["paper_id"] == "ALL"].set_index("tool")
    table = pooled[list(HEADLINE_METRICS)].T
    table.index = [HEADLINE_METRICS[name] for name in table.index]
    table.index.name = "metric"
    return table


def recall_figure(per_paper: pd.DataFrame, gt_counts: pd.Series, out_path: Path) -> None:
    """Grouped bars: exact and stereo-blind recall per paper and tool."""
    papers = per_paper[per_paper["paper_id"] != "ALL"]
    paper_ids = sorted(papers["paper_id"].unique(), key=int)
    tools = sorted(papers["tool"].unique())
    series = [(tool, metric) for tool in tools for metric in ("exact_recall", "stereo_blind_recall")]
    width = 0.8 / len(series)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for offset, (tool, metric) in enumerate(series):
        values = [
            papers[(papers["tool"] == tool) & (papers["paper_id"] == pid)][metric].squeeze() for pid in paper_ids
        ]
        positions = [i + (offset - (len(series) - 1) / 2) * width for i in range(len(paper_ids))]
        label = f"{tool} ({'exact' if metric == 'exact_recall' else 'stereo-blind'})"
        ax.bar(positions, values, width, label=label)

    ax.set_xticks(range(len(paper_ids)))
    ax.set_xticklabels([f"Paper {pid}\n(n={int(gt_counts.get(int(pid), 0))})" for pid in paper_ids])
    ax.set_ylabel("Recall of ground-truth molecules")
    ax.set_ylim(0, 1)
    ax.set_title("Recall by paper (n = ground-truth molecules in the paper)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def git_state(repo_root: Path) -> dict:
    """Commit hash and whether the working tree had uncommitted changes when results were written."""
    try:
        def git(*args: str) -> str:
            return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=True).stdout.strip()

        return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": None}


def package_versions(repo_root: Path) -> dict:
    versions = {"python": platform.python_version(), "platform": platform.platform(), **git_state(repo_root)}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _markdown_table(frame: pd.DataFrame, float_format: str = "{:.3f}") -> str:
    """Small Markdown table writer (avoids an extra dependency on `tabulate`)."""
    def fmt(value):
        if isinstance(value, float):
            if pd.isna(value):
                return ""
            if value.is_integer() and abs(value) > 1:  # counts such as number of predictions
                return str(int(value))
            return float_format.format(value)
        return str(value)

    header = "| " + " | ".join([frame.index.name or ""] + [str(c) for c in frame.columns]) + " |"
    divider = "|" + "---|" * (len(frame.columns) + 1)
    # itertuples keeps each column's own type (iterrows would turn integer counts into floats).
    rows = ["| " + " | ".join([str(row[0])] + [fmt(v) for v in row[1:]]) + " |" for row in frame.itertuples(name=None)]
    return "\n".join([header, divider, *rows])


def write_reports(per_paper: pd.DataFrame, ground_truth: pd.DataFrame, config: dict, results_dir: Path) -> None:
    """Write comparison.csv, summary_metrics.json, RESULTS.md, environment.json and the figure."""
    results_dir.mkdir(parents=True, exist_ok=True)
    gt_counts = ground_truth.groupby("paper_id").size()

    comparison = comparison_table(per_paper)
    comparison.to_csv(results_dir / "comparison.csv")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "detection_method": config["detection"]["method"],
        "near_match_threshold": config["scoring"]["near_match_threshold"],
        "tools": {
            tool: {
                "overall": rows[rows["paper_id"] == "ALL"].drop(columns=["tool", "paper_id"]).iloc[0].to_dict(),
                "per_paper": {
                    row["paper_id"]: row.drop(labels=["tool", "paper_id"]).to_dict()
                    for _, row in rows[rows["paper_id"] != "ALL"].iterrows()
                },
            }
            for tool, rows in per_paper.groupby("tool")
        },
    }
    (results_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    (results_dir / "environment.json").write_text(json.dumps(package_versions(config["repo_root"]), indent=2))

    figure_path = results_dir / "figures" / "recall_by_paper.png"
    recall_figure(per_paper, gt_counts, figure_path)

    recall_by_paper = per_paper[per_paper["paper_id"] != "ALL"].pivot(
        index="paper_id", columns="tool", values="exact_recall"
    )
    recall_by_paper = recall_by_paper.loc[sorted(recall_by_paper.index, key=int)]
    recall_by_paper.insert(0, "GT molecules", [int(gt_counts.get(int(p), 0)) for p in recall_by_paper.index])
    recall_by_paper.index.name = "paper"

    markdown = "\n\n".join(
        [
            "# MolCensus results",
            f"Generated {summary['generated_at']} with detection method `{summary['detection_method']}`. "
            "All numbers are pooled over the six course papers unless noted. Definitions are in the README.",
            "## Overall comparison",
            _markdown_table(comparison),
            "## Exact recall by paper",
            _markdown_table(recall_by_paper),
            "![Recall by paper](figures/recall_by_paper.png)",
        ]
    )
    (results_dir / "RESULTS.md").write_text(markdown + "\n")
    log.info("Reports written to %s", results_dir)
