"""Write the side-by-side comparison, a JSON summary, a Markdown results page and figures."""

from __future__ import annotations

import json
import logging
import math
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

# Interval columns added to the comparison table (written by metrics.add_intervals).
INTERVAL_METRICS = {
    f"{rate}_{kind}_{end}": f"{label} 95% {'CI (molecules)' if kind == 'ci' else 'CI (papers)'} {end}"
    for rate, label in [("exact_recall", "Exact recall"), ("exact_precision", "Exact precision"),
                        ("stereo_blind_recall", "Stereo-blind recall")]
    for kind in ("ci", "boot")
    for end in ("low", "high")
}

PACKAGES = ["rdkit", "pymupdf", "ultralytics", "torch", "MolScribe", "pandas", "numpy", "huggingface_hub"]


def comparison_table(per_paper: pd.DataFrame) -> pd.DataFrame:
    """Pooled (ALL) metrics with one column per tool, including the 95% interval bounds."""
    pooled = per_paper[per_paper["paper_id"] == "ALL"].set_index("tool")
    names = {**HEADLINE_METRICS, **{k: v for k, v in INTERVAL_METRICS.items() if k in pooled.columns}}
    table = pooled[[name for name in names if name in pooled.columns]].T
    table.index = [names[name] for name in table.index]
    table.index.name = "metric"
    return table


def _fmt_rate(value: float) -> str:
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f}"


def _with_interval(row: pd.Series, rate: str) -> str:
    """'0.381 [0.317, 0.449]' plus the paper-bootstrap interval when available."""
    text = f"{_fmt_rate(row[rate])} [{_fmt_rate(row.get(f'{rate}_ci_low'))}, {_fmt_rate(row.get(f'{rate}_ci_high'))}]"
    boot_low, boot_high = row.get(f"{rate}_boot_low"), row.get(f"{rate}_boot_high")
    if boot_low is not None and not (isinstance(boot_low, float) and math.isnan(boot_low)):
        text += f"; papers [{_fmt_rate(boot_low)}, {_fmt_rate(boot_high)}]"
    return text


def headline_table(per_paper: pd.DataFrame) -> pd.DataFrame:
    """Readable pooled results per tool, each rate with its 95% intervals."""
    pooled = per_paper[per_paper["paper_id"] == "ALL"]
    rows = {}
    for _, row in pooled.iterrows():
        rows[row["tool"]] = {
            "Exact recall": _with_interval(row, "exact_recall"),
            "Exact precision": _with_interval(row, "exact_precision"),
            "Stereo-blind recall": _with_interval(row, "stereo_blind_recall"),
            "Valid SMILES rate": _fmt_rate(row["validity_rate"]),
            "Mean best Tanimoto": _fmt_rate(row["mean_best_tanimoto"]),
            "Predictions / unique GT molecules": f"{int(row['n_predictions'])} / {int(row['n_gt_unique'])}",
        }
    table = pd.DataFrame(rows)
    table.index.name = "metric"
    return table


def _error_bars(frame: pd.DataFrame, rate: str) -> list[list[float]] | None:
    low, high = f"{rate}_ci_low", f"{rate}_ci_high"
    if low not in frame.columns:
        return None
    values = frame[rate].astype(float)
    return [list((values - frame[low].astype(float)).clip(lower=0)), list((frame[high].astype(float) - values).clip(lower=0))]


def recall_figure(per_paper: pd.DataFrame, gt_counts: pd.Series, out_path: Path) -> None:
    """Grouped bars: exact and stereo-blind recall per paper and tool, with 95% Wilson intervals."""
    papers = per_paper[per_paper["paper_id"] != "ALL"]
    paper_ids = sorted(papers["paper_id"].unique(), key=int)
    tools = sorted(papers["tool"].unique())
    series = [(tool, metric) for tool in tools for metric in ("exact_recall", "stereo_blind_recall")]
    width = 0.8 / len(series)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for offset, (tool, metric) in enumerate(series):
        rows = pd.concat([papers[(papers["tool"] == tool) & (papers["paper_id"] == pid)] for pid in paper_ids])
        positions = [i + (offset - (len(series) - 1) / 2) * width for i in range(len(paper_ids))]
        label = f"{tool} ({'exact' if metric == 'exact_recall' else 'stereo-blind'})"
        ax.bar(positions, rows[metric].astype(float), width, label=label,
               yerr=_error_bars(rows, metric), capsize=2, error_kw={"elinewidth": 0.8})

    ax.set_xticks(range(len(paper_ids)))
    ax.set_xticklabels([f"Paper {pid}\n(n={int(gt_counts.get(int(pid), 0))})" for pid in paper_ids])
    ax.set_ylabel("Recall of ground-truth molecules")
    ax.set_ylim(0, 1.05)
    ax.set_title("Recall by paper with 95% Wilson intervals (n = ground-truth molecules)")
    ax.legend(fontsize=8, frameon=True, facecolor="white", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def public_figure(public: pd.DataFrame, out_path: Path) -> None:
    """Exact accuracy on each public set with its 95% interval, next to the published number."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(6, 3.8))
    labels = [f"{row.tool}\n{row.dataset.upper()} (n={row.n_scored})" for row in public.itertuples()]
    positions = list(range(len(public)))
    ax.bar(positions, public["exact_accuracy"], 0.5, color="#4C72B0",
           yerr=[public["exact_accuracy"] - public["exact_ci_low"], public["exact_ci_high"] - public["exact_accuracy"]],
           capsize=4)
    handles = [Patch(color="#4C72B0", label="Measured here (95% CI)")]
    published = public["published_exact"].astype(float)
    if published.notna().any():
        ax.scatter(positions, published, marker="D", color="black", zorder=3)
        handles.append(Line2D([], [], marker="D", color="black", linestyle="none", label="Published"))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.75, len(public) - 0.25)  # keep bars narrow even with one dataset
    ax.set_ylabel("Exact-match accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Public crop check")
    ax.legend(handles=handles, fontsize=8, loc="lower right", frameon=True, facecolor="white", framealpha=1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def public_table(public: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in public.itertuples():
        within = {True: "yes", False: "no"}.get(row.published_within_ci, "n/a")
        rows.append(
            {
                "tool / dataset": f"{row.tool} / {row.dataset.upper()}",
                "images": int(row.n_scored),
                "exact accuracy [95% CI]": f"{row.exact_accuracy:.3f} [{row.exact_ci_low:.3f}, {row.exact_ci_high:.3f}]",
                "published": _fmt_rate(row.published_exact),
                "published inside CI": within,
                "stereo-blind accuracy": _fmt_rate(row.stereo_blind_accuracy),
                "chiral exact (n)": f"{_fmt_rate(row.chiral_exact_accuracy)} ({int(row.n_chiral)})",
                "valid SMILES": _fmt_rate(row.validity_rate),
                "mean Tanimoto": _fmt_rate(row.mean_tanimoto),
                "seconds / image": _fmt_rate(row.mean_runtime_s),
            }
        )
    return pd.DataFrame(rows).set_index("tool / dataset")


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


def write_reports(per_paper: pd.DataFrame | None, ground_truth: pd.DataFrame, config: dict, results_dir: Path) -> None:
    """Write comparison.csv, summary_metrics.json, RESULTS.md, environment.json and the figures."""
    results_dir.mkdir(parents=True, exist_ok=True)
    gt_counts = ground_truth.groupby("paper_id").size()
    public_file = results_dir / "public" / "public_metrics.csv"
    public = pd.read_csv(public_file) if public_file.exists() else None

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "detection_method": config["detection"]["method"],
        "near_match_threshold": config["scoring"]["near_match_threshold"],
        "tools": {},
    }
    sections = [
        "# MolCensus results",
        f"Generated {summary['generated_at']} with detection method `{summary['detection_method']}`. "
        "Rates are pooled over the six course papers unless noted, with 95% intervals: Wilson intervals "
        "treating molecules as independent, and a bootstrap over papers. Definitions are in the README.",
    ]

    if per_paper is not None:
        comparison_table(per_paper).to_csv(results_dir / "comparison.csv")
        summary["tools"] = {
            tool: {
                "overall": rows[rows["paper_id"] == "ALL"].drop(columns=["tool", "paper_id"]).iloc[0].to_dict(),
                "per_paper": {
                    row["paper_id"]: row.drop(labels=["tool", "paper_id"]).to_dict()
                    for _, row in rows[rows["paper_id"] != "ALL"].iterrows()
                },
            }
            for tool, rows in per_paper.groupby("tool")
        }
        recall_figure(per_paper, gt_counts, results_dir / "figures" / "recall_by_paper.png")

        recall_by_paper = per_paper[per_paper["paper_id"] != "ALL"].copy()
        recall_by_paper["exact recall [95% CI]"] = [
            f"{r.exact_recall:.3f} [{r.exact_recall_ci_low:.3f}, {r.exact_recall_ci_high:.3f}]"
            for r in recall_by_paper.itertuples()
        ]
        by_paper = recall_by_paper.pivot(index="paper_id", columns="tool", values="exact recall [95% CI]")
        by_paper = by_paper.loc[sorted(by_paper.index, key=int)]
        by_paper.insert(0, "GT molecules", [int(gt_counts.get(int(p), 0)) for p in by_paper.index])
        by_paper.index.name = "paper"
        sections += [
            "## Course papers: overall",
            _markdown_table(headline_table(per_paper)),
            "## Course papers: exact recall by paper",
            _markdown_table(by_paper),
            "![Recall by paper](figures/recall_by_paper.png)",
        ]

    if public is not None:
        summary["public_check"] = public.to_dict(orient="records")
        public_figure(public, results_dir / "figures" / "public_check.png")
        sections += [
            "## Public crop check",
            "Single-molecule public images with known SMILES (MolScribe evaluation sets). The check "
            "passes when the published accuracy falls inside our 95% interval.",
            _markdown_table(public_table(public)),
            "![Public crop check](figures/public_check.png)",
        ]

    (results_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    (results_dir / "environment.json").write_text(json.dumps(package_versions(config["repo_root"]), indent=2))
    (results_dir / "RESULTS.md").write_text("\n\n".join(sections) + "\n")
    log.info("Reports written to %s", results_dir)
