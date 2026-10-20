"""Figures for RESULTS.md and the README."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, no display
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

# Colorblind-safe series colors, then text, axis and background colors.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

TOOL_NAMES = {"molscribe": "MolScribe", "biominer": "BioMiner"}

# Outcome buckets from metrics.OUTCOME_COLUMNS, best to worst: dark to light.
OUTCOME_STYLE = {
    "n_found_exact": ("Found exactly", "#2a78d6", "white"),
    "n_stereo_only": ("Right except stereochemistry", "#5598e7", "white"),
    "n_near_miss": ("Near miss", "#9ec5f4", INK),
    "n_related": ("Related molecule read", "#cde2fb", INK),
    "n_unrelated": ("Nothing similar read", "#dcdbd6", INK),
}


def _axes(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE, layout="constrained")
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _interval(values: pd.Series, low: pd.Series, high: pd.Series) -> list[list[float]]:
    """Error bar lengths below and above each value."""
    values = values.astype(float)
    return [list((values - low.astype(float)).clip(lower=0)), list((high.astype(float) - values).clip(lower=0))]


def overview_figure(
    per_paper: pd.DataFrame | None, public: pd.DataFrame | None, out_path: Path, width: float = 8, row_height: float = 0.55
) -> None:
    """Accuracy on clean public crops next to recall on whole papers, with 95% intervals."""
    tools = set(public["tool"] if public is not None else []) | set(per_paper["tool"] if per_paper is not None else [])
    if not tools:
        return
    one_tool = len(tools) == 1

    def label(tool: str, text: str) -> str:
        return text if one_tool else f"{TOOL_NAMES.get(tool, tool)}, {text}"

    rows = []  # label, value, low, high, color, published value
    if public is not None:
        for r in public.sort_values("dataset", ascending=False).itertuples():
            rows.append((label(r.tool, f"{r.dataset.upper()} crops\n{r.n_scored} images"),
                         r.exact_accuracy, r.exact_ci_low, r.exact_ci_high, BLUE, r.published_exact))
    if per_paper is not None:
        for r in per_paper[per_paper["paper_id"] == "ALL"].itertuples():
            rows.append((label(r.tool, f"Whole papers\n{int(r.n_gt_unique)} molecules"),
                         r.exact_recall, r.exact_recall_ci_low, r.exact_recall_ci_high, ORANGE, math.nan))

    fig, ax = _axes(width, row_height * len(rows) + 1.2)
    positions = list(range(len(rows)))[::-1]
    for y, (_, value, low, high, color, published) in zip(positions, rows):
        ax.barh(y, value, 0.55, color=color)
        ax.errorbar(value, y, xerr=[[value - low], [high - value]], fmt="none", ecolor=MUTED, elinewidth=0.8, capsize=3)
        ax.text(high + 0.015, y, f"{value:.3f}", va="center", fontsize=9, color=INK)
        if not math.isnan(published):
            ax.plot(published, y, "D", markersize=6, color=INK, markeredgecolor=SURFACE, zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Share of molecules recovered exactly", color=INK)
    title = "Clean crops vs. whole papers"
    ax.set_title(f"{TOOL_NAMES.get(min(tools), min(tools))}: {title.lower()}" if one_tool else title, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.8)

    handles = []
    if public is not None:
        handles += [Patch(color=BLUE, label="Public single-molecule crops (accuracy)"),
                    Line2D([], [], marker="D", color=INK, linestyle="none", label="Published accuracy")]
    if per_paper is not None:
        n_papers = per_paper.loc[per_paper["paper_id"] != "ALL", "paper_id"].nunique()
        handles.insert(1, Patch(color=ORANGE, label=f"Whole papers, recall over {n_papers} papers"))
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="outside lower center", ncol=len(handles))
    _save(fig, out_path)


def outcome_figure(
    rows: pd.DataFrame,
    crops: pd.Series | None,
    thresholds: tuple[float, float],
    out_path: Path,
    width: float = 8,
    row_height: float = 0.36,
) -> None:
    """One stacked bar per paper: what happened to each of its ground-truth molecules."""
    near, related = thresholds
    papers = rows[rows["paper_id"] != "ALL"].sort_values("paper_id", key=lambda ids: ids.astype(int))
    rows = pd.concat([papers, rows[rows["paper_id"] == "ALL"]])
    columns = [c for c in OUTCOME_STYLE if c in rows.columns and (c != "n_stereo_only" or rows[c].sum() > 0)]

    labels = []
    for r in rows.itertuples():
        name = "All papers" if r.paper_id == "ALL" else f"Paper {r.paper_id}"
        detail = f"{int(r.n_gt_unique)} molecule{'s' if r.n_gt_unique != 1 else ''}"
        if crops is not None:
            n_crops = int(crops.sum()) if r.paper_id == "ALL" else int(crops.get(int(r.paper_id), 0))
            detail += f", {n_crops} crops"
        labels.append(f"{name}\n{detail}")

    fig, ax = _axes(width, row_height * len(rows) + 1.2)
    positions = [len(rows) - i + (0 if pid == "ALL" else 0.4) for i, pid in enumerate(rows["paper_id"])]
    left = pd.Series(0.0, index=rows.index)
    for column in columns:
        name, color, text_color = OUTCOME_STYLE[column]
        share = rows[column] / rows["n_gt_unique"].where(rows["n_gt_unique"] > 0)
        ax.barh(positions, share, 0.62, left=left, color=color, edgecolor=SURFACE, linewidth=1)
        for y, x, width, count in zip(positions, left, share, rows[column]):
            if width >= 0.04:
                ax.text(x + width / 2, y, str(int(count)), ha="center", va="center", fontsize=8, color=text_color)
        left += share.fillna(0)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Share of the paper's ground-truth molecules", color=INK)
    tool = TOOL_NAMES.get(rows["tool"].iloc[0], rows["tool"].iloc[0])
    ax.set_title(f"What happened to each ground-truth molecule ({tool})", color=INK, loc="left")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)

    legend_names = {
        "n_stereo_only": "Right except stereo",
        "n_near_miss": f"Near miss (Tanimoto ≥ {near:g})",
        "n_related": f"Related ({related:g} to {near:g})",
        "n_unrelated": f"Nothing similar (< {related:g})",
    }
    handles = [Patch(color=OUTCOME_STYLE[c][1], label=legend_names.get(c, OUTCOME_STYLE[c][0])) for c in columns]
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="outside lower center", ncol=len(handles))
    _save(fig, out_path)


def recall_figure(per_paper: pd.DataFrame, out_path: Path) -> None:
    """Exact recall per paper and tool, with 95% Wilson intervals."""
    papers = per_paper[per_paper["paper_id"] != "ALL"]
    paper_ids = sorted(papers["paper_id"].unique(), key=int)
    tools = sorted(papers["tool"].unique())
    width = 0.6 / len(tools)

    fig, ax = _axes(8, 4)
    for offset, tool in enumerate(tools):
        rows = papers[papers["tool"] == tool].set_index("paper_id").loc[paper_ids].reset_index()
        positions = [i + (offset - (len(tools) - 1) / 2) * width for i in range(len(paper_ids))]
        ax.bar(positions, rows["exact_recall"], width * 0.9, color=[BLUE, ORANGE][offset % 2], label=TOOL_NAMES.get(tool, tool),
               yerr=_interval(rows["exact_recall"], rows["exact_recall_ci_low"], rows["exact_recall_ci_high"]),
               capsize=3, error_kw={"elinewidth": 0.8, "ecolor": MUTED, "capthick": 0.8})
        for x, value, high in zip(positions, rows["exact_recall"], rows["exact_recall_ci_high"]):
            ax.text(x, high + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=8, color=INK)

    counts = papers.drop_duplicates("paper_id").set_index("paper_id")["n_gt_unique"]
    ax.set_xticks(range(len(paper_ids)))
    ax.set_xticklabels([f"Paper {pid}\n{int(counts[pid])} molecule{'s' if counts[pid] != 1 else ''}" for pid in paper_ids])
    ax.set_ylabel("Exact recall", color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_title("Exact recall by paper (bars: 95% Wilson interval)", color=INK, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    if len(tools) > 1:
        ax.legend(frameon=False, fontsize=8)
    _save(fig, out_path)
