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


# Chart colors: categorical slots 1 and 2 of a colorblind-checked palette, plus muted ink for text.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def _error_bars(frame: pd.DataFrame, rate: str, low: str | None = None, high: str | None = None):
    low, high = low or f"{rate}_ci_low", high or f"{rate}_ci_high"
    if low not in frame.columns:
        return None
    values = frame[rate].astype(float)
    return [list((values - frame[low].astype(float)).clip(lower=0)), list((frame[high].astype(float) - values).clip(lower=0))]


def _new_axes(width: float, height: float):
    """Figure with a plain light surface, no top/right frame and a faint value grid."""
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    return fig, ax


def _save(fig, out_path: Path) -> None:
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def recall_figure(per_paper: pd.DataFrame, out_path: Path) -> None:
    """Exact recall per paper and tool, with 95% Wilson intervals and the value on each bar."""
    papers = per_paper[per_paper["paper_id"] != "ALL"]
    paper_ids = sorted(papers["paper_id"].unique(), key=int)
    tools = sorted(papers["tool"].unique())
    width = 0.6 / len(tools)

    fig, ax = _new_axes(8, 4)
    for offset, tool in enumerate(tools):
        rows = papers[papers["tool"] == tool].set_index("paper_id").loc[paper_ids].reset_index()
        positions = [i + (offset - (len(tools) - 1) / 2) * width for i in range(len(paper_ids))]
        ax.bar(positions, rows["exact_recall"], width * 0.9, color=SERIES_COLORS[offset % len(SERIES_COLORS)],
               label=tool, yerr=_error_bars(rows, "exact_recall"), capsize=3,
               error_kw={"elinewidth": 0.8, "ecolor": MUTED, "capthick": 0.8})
        for x, value, high in zip(positions, rows["exact_recall"], rows["exact_recall_ci_high"]):
            ax.text(x, high + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=8, color=INK)

    first = papers[papers["tool"] == tools[0]].set_index("paper_id")
    ax.set_xticks(range(len(paper_ids)))
    ax.set_xticklabels([f"Paper {pid}\n{int(first.loc[pid, 'n_gt_unique'])} molecule{'s' if first.loc[pid, 'n_gt_unique'] != 1 else ''}" for pid in paper_ids])
    ax.set_ylabel("Exact recall", color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_title("Exact recall by paper (bars: 95% Wilson interval)", color=INK, loc="left")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    if len(tools) > 1:
        ax.legend(frameon=False, fontsize=8)
    _save(fig, out_path)


def overview_figure(per_paper: pd.DataFrame | None, public: pd.DataFrame | None, out_path: Path) -> None:
    """Accuracy on clean public crops next to recall on whole papers, each with its 95% interval."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rows = []  # (label, value, low, high, color slot, published)
    if public is not None:
        for row in public.sort_values("dataset", ascending=False).itertuples():
            rows.append((f"{row.dataset.upper()} crops, {row.tool} ({row.n_scored} images)", row.exact_accuracy,
                         row.exact_ci_low, row.exact_ci_high, 0, row.published_exact))
    if per_paper is not None:
        for row in per_paper[per_paper["paper_id"] == "ALL"].itertuples():
            rows.append((f"Whole papers, {row.tool} ({int(row.n_gt_unique)} molecules)", row.exact_recall,
                         row.exact_recall_ci_low, row.exact_recall_ci_high, 1, math.nan))
    if not rows:
        return

    fig, ax = _new_axes(8, 0.6 * len(rows) + 1.3)
    positions = list(range(len(rows)))[::-1]  # first row on top
    for y, (label, value, low, high, slot, published) in zip(positions, rows):
        ax.barh(y, value, 0.55, color=SERIES_COLORS[slot])
        ax.errorbar(value, y, xerr=[[value - low], [high - value]], fmt="none", ecolor=MUTED, elinewidth=0.8, capsize=3)
        ax.text(high + 0.015, y, f"{value:.3f}", va="center", fontsize=9, color=INK)
        if not math.isnan(published):
            ax.plot(published, y, marker="D", markersize=6, color=INK, markeredgecolor=SURFACE, zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Share of molecules recovered exactly", color=INK)
    ax.set_title("Clean crops vs. whole papers", color=INK, loc="left")
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    handles = []
    if public is not None:
        handles += [Patch(color=SERIES_COLORS[0], label="Public single-molecule crops (accuracy)"),
                    Line2D([], [], marker="D", color=INK, linestyle="none", label="Published accuracy")]
    if per_paper is not None:
        n_papers = per_paper.loc[per_paper["paper_id"] != "ALL", "paper_id"].nunique()
        handles.insert(1, Patch(color=SERIES_COLORS[1], label=f"Whole papers, {n_papers} pooled (recall)"))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.22), ncol=3)
    _save(fig, out_path)


def public_figure(public: pd.DataFrame, out_path: Path) -> None:
    """Exact accuracy on each public set with its 95% interval, next to the published number."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = _new_axes(6, 3.6)
    labels = [f"{row.dataset.upper()}, {row.tool}\n{row.n_scored} images" for row in public.itertuples()]
    positions = list(range(len(public)))
    ax.bar(positions, public["exact_accuracy"], 0.45, color=SERIES_COLORS[0],
           yerr=_error_bars(public, "exact_accuracy", "exact_ci_low", "exact_ci_high"), capsize=4,
           error_kw={"elinewidth": 0.8, "ecolor": MUTED, "capthick": 0.8})
    handles = [Patch(color=SERIES_COLORS[0], label="Measured here (95% CI)")]
    published = public["published_exact"].astype(float)
    if published.notna().any():
        ax.plot(positions, published, "D", markersize=7, color=INK, markeredgecolor=SURFACE, zorder=3)
        handles.append(Line2D([], [], marker="D", color=INK, linestyle="none", label="Published"))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.75, len(public) - 0.25)  # keep bars narrow even with one dataset
    ax.set_ylabel("Exact-match accuracy", color=INK)
    ax.set_ylim(0, 1.15)
    ax.set_title("Public crop check", color=INK, loc="left")
    ax.legend(handles=handles, fontsize=8, loc="upper left", frameon=False)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    _save(fig, out_path)


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
    """Commit hash and whether tracked code had uncommitted changes when results were written.

    results/ is left out: a run always rewrites it, so it would always look dirty.
    """
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


def write_reports(per_paper: pd.DataFrame | None, config: dict, results_dir: Path) -> None:
    """Write comparison.csv, summary_metrics.json, RESULTS.md, environment.json and the figures."""
    results_dir.mkdir(parents=True, exist_ok=True)
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
        "Rates are pooled over the papers unless noted, with 95% intervals: Wilson intervals "
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
        recall_figure(per_paper, results_dir / "figures" / "recall_by_paper.png")

        recall_by_paper = per_paper[per_paper["paper_id"] != "ALL"].copy()
        recall_by_paper["exact recall [95% CI]"] = [
            f"{r.exact_recall:.3f} [{r.exact_recall_ci_low:.3f}, {r.exact_recall_ci_high:.3f}]"
            for r in recall_by_paper.itertuples()
        ]
        by_paper = recall_by_paper.pivot(index="paper_id", columns="tool", values="exact recall [95% CI]")
        by_paper = by_paper.loc[sorted(by_paper.index, key=int)]
        unique_counts = recall_by_paper.drop_duplicates("paper_id").set_index("paper_id")["n_gt_unique"]
        by_paper.insert(0, "unique GT molecules", [int(unique_counts[p]) for p in by_paper.index])
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

    overview_figure(per_paper, public, results_dir / "figures" / "overview.png")
    sections.insert(2, "![Overview](figures/overview.png)")

    (results_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    (results_dir / "environment.json").write_text(json.dumps(package_versions(config["repo_root"]), indent=2))
    (results_dir / "RESULTS.md").write_text("\n\n".join(sections) + "\n")
    log.info("Reports written to %s", results_dir)
