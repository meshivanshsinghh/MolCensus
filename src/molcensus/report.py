"""Write the comparison table, a JSON summary, RESULTS.md and the figures."""

from __future__ import annotations

import json
import logging
import math
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

from . import figures
from .metrics import OUTCOME_COLUMNS

log = logging.getLogger(__name__)

# Metrics in comparison.csv, with readable names.
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

INTERVAL_METRICS = {
    f"{rate}_{kind}_{end}": f"{label} 95% {'CI (molecules)' if kind == 'ci' else 'CI (papers)'} {end}"
    for rate, label in [("exact_recall", "Exact recall"), ("exact_precision", "Exact precision"),
                        ("stereo_blind_recall", "Stereo-blind recall")]
    for kind in ("ci", "boot")
    for end in ("low", "high")
}

OUTCOME_NAMES = {
    "n_found_exact": "found exactly",
    "n_stereo_only": "right except stereo",
    "n_near_miss": "near miss",
    "n_related": "related molecule read",
    "n_unrelated": "nothing similar read",
}

PACKAGES = ["rdkit", "pymupdf", "ultralytics", "torch", "MolScribe", "pandas", "numpy", "huggingface_hub"]


def comparison_table(per_paper: pd.DataFrame) -> pd.DataFrame:
    """Pooled metrics with one column per tool, including the interval bounds."""
    pooled = per_paper[per_paper["paper_id"] == "ALL"].set_index("tool")
    names = {**HEADLINE_METRICS, **{k: v for k, v in INTERVAL_METRICS.items() if k in pooled.columns}}
    table = pooled[[name for name in names if name in pooled.columns]].T
    table.index = [names[name] for name in table.index]
    table.index.name = "metric"
    return table


def _fmt(value: float) -> str:
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f}"


def _with_interval(row: pd.Series, rate: str) -> str:
    """'0.381 [0.317, 0.449]', plus the paper-bootstrap interval when there is one."""
    text = f"{_fmt(row[rate])} [{_fmt(row.get(f'{rate}_ci_low'))}, {_fmt(row.get(f'{rate}_ci_high'))}]"
    boot_low, boot_high = row.get(f"{rate}_boot_low"), row.get(f"{rate}_boot_high")
    if boot_low is not None and not (isinstance(boot_low, float) and math.isnan(boot_low)):
        text += f"; papers [{_fmt(boot_low)}, {_fmt(boot_high)}]"
    return text


def headline_table(per_paper: pd.DataFrame) -> pd.DataFrame:
    pooled = per_paper[per_paper["paper_id"] == "ALL"]
    rows = {}
    for _, row in pooled.iterrows():
        rows[row["tool"]] = {
            "Exact recall": _with_interval(row, "exact_recall"),
            "Exact precision": _with_interval(row, "exact_precision"),
            "Stereo-blind recall": _with_interval(row, "stereo_blind_recall"),
            "Valid SMILES rate": _fmt(row["validity_rate"]),
            "Mean best Tanimoto": _fmt(row["mean_best_tanimoto"]),
            "Predictions / unique GT molecules": f"{int(row['n_predictions'])} / {int(row['n_gt_unique'])}",
        }
    table = pd.DataFrame(rows)
    table.index.name = "metric"
    return table


def recall_by_paper_table(per_paper: pd.DataFrame) -> pd.DataFrame:
    papers = per_paper[per_paper["paper_id"] != "ALL"].copy()
    papers["exact recall [95% CI]"] = [
        f"{r.exact_recall:.3f} [{r.exact_recall_ci_low:.3f}, {r.exact_recall_ci_high:.3f}]" for r in papers.itertuples()
    ]
    table = papers.pivot(index="paper_id", columns="tool", values="exact recall [95% CI]")
    table = table.loc[sorted(table.index, key=int)]
    counts = papers.drop_duplicates("paper_id").set_index("paper_id")["n_gt_unique"]
    table.insert(0, "unique GT molecules", [int(counts[p]) for p in table.index])
    table.index.name = "paper"
    return table


def outcome_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Counts per outcome for one tool, one row per paper plus the pooled row."""
    columns = [c for c in OUTCOME_COLUMNS if c in rows.columns]
    table = rows.set_index("paper_id")[["n_gt_unique", *columns]].astype(int)
    table = table.rename(columns={"n_gt_unique": "unique GT molecules", **OUTCOME_NAMES})
    table.index.name = "paper"
    return table


def public_table(public: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in public.itertuples():
        rows.append(
            {
                "tool / dataset": f"{row.tool} / {row.dataset.upper()}",
                "images": int(row.n_scored),
                "exact accuracy [95% CI]": f"{row.exact_accuracy:.3f} [{row.exact_ci_low:.3f}, {row.exact_ci_high:.3f}]",
                "published": _fmt(row.published_exact),
                "published inside CI": {True: "yes", False: "no"}.get(row.published_within_ci, "n/a"),
                "stereo-blind accuracy": _fmt(row.stereo_blind_accuracy),
                "chiral exact (n)": f"{_fmt(row.chiral_exact_accuracy)} ({int(row.n_chiral)})",
                "valid SMILES": _fmt(row.validity_rate),
                "mean Tanimoto": _fmt(row.mean_tanimoto),
                "seconds / image": _fmt(row.mean_runtime_s),
            }
        )
    return pd.DataFrame(rows).set_index("tool / dataset")


def git_state(repo_root: Path) -> dict:
    """Commit hash, and whether the code (not results/) had uncommitted changes."""
    try:
        def git(*args: str) -> str:
            return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=True).stdout.strip()

        dirty = git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)results")
        return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(dirty)}
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
    def fmt(value):
        if isinstance(value, float):
            if pd.isna(value):
                return ""
            if value.is_integer() and abs(value) > 1:
                return str(int(value))
            return float_format.format(value)
        return str(value)

    header = "| " + " | ".join([frame.index.name or ""] + [str(c) for c in frame.columns]) + " |"
    divider = "|" + "---|" * (len(frame.columns) + 1)
    # itertuples keeps integer columns as integers; iterrows would turn them into floats.
    rows = ["| " + " | ".join([str(row[0])] + [fmt(v) for v in row[1:]]) + " |" for row in frame.itertuples(name=None)]
    return "\n".join([header, divider, *rows])


def write_reports(per_paper: pd.DataFrame | None, config: dict, results_dir: Path) -> None:
    """Write comparison.csv, summary_metrics.json, environment.json, RESULTS.md and the figures."""
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"
    public_file = results_dir / "public" / "public_metrics.csv"
    public = pd.read_csv(public_file) if public_file.exists() else None
    crops_file = results_dir / "crops_per_paper.csv"
    crops = pd.read_csv(crops_file).set_index("paper_id")["n_crops"] if crops_file.exists() else None
    scoring = config["scoring"]
    thresholds = (scoring["near_match_threshold"], scoring.get("related_threshold", 0.5))

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "detection_method": config["detection"]["method"],
        "near_match_threshold": thresholds[0],
        "related_threshold": thresholds[1],
        "tools": {},
    }
    sections = [
        "# MolCensus results",
        f"Generated {summary['generated_at']} with detection method `{summary['detection_method']}`. "
        "Rates are pooled over the papers unless noted. Brackets are 95% intervals: a Wilson interval "
        "over molecules, then a bootstrap over papers. Definitions are in docs/methods.md.",
    ]

    figures.overview_figure(per_paper, public, figures_dir / "overview.png")
    sections.append("![Overview](figures/overview.png)")

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
        figures.recall_figure(per_paper, figures_dir / "recall_by_paper.png")
        sections += [
            "## Course papers: overall",
            _markdown_table(headline_table(per_paper)),
            "## Course papers: exact recall by paper",
            _markdown_table(recall_by_paper_table(per_paper)),
            "![Recall by paper](figures/recall_by_paper.png)",
            "## What happened to each ground-truth molecule",
            f"Each unique molecule is counted once, in its best outcome. A near miss has a prediction with "
            f"Tanimoto at least {thresholds[0]:g}; a related molecule is between {thresholds[1]:g} and "
            f"{thresholds[0]:g}.",
        ]
        for tool, rows in per_paper.groupby("tool"):
            if not set(OUTCOME_COLUMNS) <= set(rows.columns):
                continue
            figures.outcome_figure(rows, crops, thresholds, figures_dir / f"outcomes_{tool}.png")
            sections += [_markdown_table(outcome_table(rows)), f"![Outcomes for {tool}](figures/outcomes_{tool}.png)"]

    if public is not None:
        summary["public_check"] = public.to_dict(orient="records")
        sections += [
            "## Public crop check",
            "Public single-molecule images with known SMILES (MolScribe's evaluation sets). The check "
            "passes when the published accuracy falls inside our 95% interval.",
            _markdown_table(public_table(public)),
        ]

    (results_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    (results_dir / "environment.json").write_text(json.dumps(package_versions(config["repo_root"]), indent=2))
    (results_dir / "RESULTS.md").write_text("\n\n".join(sections) + "\n")
    log.info("Reports written to %s", results_dir.name)
