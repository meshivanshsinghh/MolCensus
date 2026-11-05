"""Build the public report, docs/index.html, from scripts/report/report_data.json.

    python scripts/report/collect.py      # once, on the machine with the private local results
    python scripts/report/build_site.py   # anywhere: needs only report_data.json

INCLUDE_COURSE (environment variable, default "true") controls the course-paper results. Course content
is per-paper totals only: no SMILES, crops, page images, compound names or ground truth. With
INCLUDE_COURSE=false the page keeps only the public-data results.

The page is one HTML file with inline CSS and inline SVG figures, so GitHub Pages serves it as is.
"""

from __future__ import annotations

import html
import json
import math
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "scripts" / "report" / "report_data.json"
OUT = ROOT / "docs" / "index.html"
INCLUDE_COURSE = os.environ.get("INCLUDE_COURSE", "true").strip().lower() == "true"
REPO = "https://github.com/meshivanshsinghh/MolCensus"
REPORT_DATE = "5 November 2026"

# Teammates' numbers, from the group report (Syeda and Obaidullah, 2026) and the reports it cites.
OBAIDULLAH_READERS = {  # Table 3: exact-match accuracy (%) on cropped images; his crop counts in brackets
    "datasets": ["USPTO (5,704)", "UOB (5,740)", "JPO (449)", "CLEF (977)", "Wild (2,371)"],
    "MolScribe": [92.1, 87.3, 59.5, 80.8, 60.1], "MolNexTR": [86.0, 85.9, 51.2, 76.9, 54.5],
    "DECIMER": [56.9, 86.4, 39.4, 72.2, 37.8], "MolVec": [88.3, 80.2, 66.6, 84.7, 35.6],
    "MolGlyph": [72.5, 87.0, 57.5, 72.1, 63.2], "OCSRGlyph": [94.0, 86.8, 51.9, 90.7, 42.7],
}
OBAIDULLAH_BIOMINER = {"course": {"precision": 72.4, "recall": 34.5, "f1": 46.8},
                       "biovista": {"precision": 62.7, "recall": 58.7, "f1": 60.6, "papers": 145, "molecules": 2304},
                       "biovista_full_vs_table": [74, 40]}
SYEDA_BIOVISTA = {"MolGlyph": 67, "MolNexTR": 53, "MolScribe": 48, "n": 100}  # Table 1, strict canonical isomeric SMILES

# Published accuracies (fractions): MolScribe from Qian et al. (2023), Table 2; MolSight-base from its README;
# MolScribe and MolGlyph on BioVista from BioMiner's README at commit 17c6161 (BioMiner's matching rule).
PUBLISHED = {("uspto", "molscribe"): 0.926, ("uob", "molscribe"): 0.879, ("clef", "molscribe"): 0.889,
             ("acs", "molscribe"): 0.719, ("uspto", "molsight"): 0.912,
             ("biovista", "molscribe"): 0.455, ("biovista", "molglyph"): 0.764}

# AWS spend, checked in Cost Explorer on 2026-11-05. The account's usage is recorded at $0.00, so these rows give the
# same usage at us-east-2 on-demand list prices: Cost Explorer's g5.2xlarge hours for 3 and 4 November (3.14 and 8.33),
# and the g6.2xlarge run of 5 November from its start (03:00:49 UTC) to its stop after the upload (about 05:48 UTC).
COSTS = [  # (item, basis, US dollars at list price)
    ("Gemini for BioMiner", "1,059,125 input, 86,175 output and 457,281 thinking tokens", 6.48),
    ("Gemini for the recall-ceiling judge", "25 calls; thinking tokens were not recorded, so the real cost is higher", 0.80),
    ("AWS GPU instances", "11.5 hours of g5.2xlarge at $1.212 and about 2.8 hours of g6.2xlarge at $0.9776", 16.64),
    ("AWS disks, snapshots, IP and S3", "a 150 GB disk for about 2.4 days, snapshot storage, public IPv4 hours (approximate)", 1.20),
]

READER_NAMES = {"molscribe": "MolScribe", "molsight": "MolSight", "molglyph": "MolGlyph"}
PAPERS = ["1", "2", "4", "6", "16", "19"]
MINUS = "−"


# ---------- formatting ----------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pc(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def interval(r: dict) -> str:
    return f"{100 * r['low']:.1f} to {100 * r['high']:.1f}%"


def with_ci(r: dict) -> str:
    return f"{pc(r['value'])} <span class=ci>{interval(r)}</span>"


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def table(headers: list[str], rows: list[list], note: str = "", numeric_from: int = 1) -> str:
    head = "".join(f"<th scope=col{' class=num' if i >= numeric_from else ''}>{h}</th>" for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f"<td{' class=num' if i >= numeric_from else ''}>{c}</td>" for i, c in enumerate(row)) + "</tr>"
                   for row in rows)
    note_html = f"<p class=table-note>{note}</p>" if note else ""
    return f"<div class=table-wrap><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{note_html}"


def legend(items: list[tuple[str, str]]) -> str:
    """An HTML legend: a swatch per series, so identity never rests on color alone in the chart."""
    return "<div class=legend>" + "".join(f"<span><i class='sw {cls}'></i>{esc(name)}</span>" for cls, name in items) + "</div>"


def svg(width: int, height: float, body: str, label: str, variant: str = "") -> str:
    return (f'<svg class="chart {variant}" viewBox="0 0 {width} {height:.0f}" role="img" aria-label="{esc(label)}" '
            f'xmlns="http://www.w3.org/2000/svg">{body}</svg>')


def both(make, wide: int = 760) -> str:
    """The same figure for a wide and a narrow screen (CSS shows one). Figures in a half column use wide=520."""
    return make(wide, "wide") + make(360, "narrow")


def figure(inner: str, caption: str, cls: str = "") -> str:
    return f'<figure class="{cls}">{inner}<figcaption>{caption}</figcaption></figure>'


def bar_path(x: float, y: float, w: float, h: float, r: float = 4.0, round_end: bool = True) -> str:
    """A bar from x to x + w with a rounded data end and a square baseline end."""
    if w <= 0:
        return ""
    r = min(r, w, h / 2) if round_end else 0
    if r <= 0:
        return f"M{x:.1f},{y:.1f}h{w:.1f}v{h:.1f}h{-w:.1f}z"
    return (f"M{x:.1f},{y:.1f}h{w - r:.1f}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r:.1f}"
            f"a{r},{r} 0 0 1 {-r},{r}h{-(w - r):.1f}z")


# ---------- figures ----------

def icon(kind: str, x: float, y: float) -> str:
    """Small line icons drawn at (x, y), about 22 px square."""
    if kind == "page":
        return (f'<path d="M{x},{y}h14l6,6v16h-20z M{x + 14},{y}v6h6" class=ico />'
                f'<path d="M{x + 4},{y + 11}h12 M{x + 4},{y + 15}h12 M{x + 4},{y + 19}h8" class=ico />')
    if kind == "find":
        return f'<circle cx="{x + 9}" cy="{y + 9}" r="7" class=ico /><path d="M{x + 14},{y + 14}l7,7" class=ico />'
    if kind == "read":
        return (f'<polygon points="{x + 11},{y} {x + 20},{y + 5} {x + 20},{y + 15} {x + 11},{y + 20} {x + 2},{y + 15} {x + 2},{y + 5}" class=ico />'
                f'<path d="M{x + 11},{y + 20}v4" class=ico />')
    if kind == "score":
        return f'<circle cx="{x + 11}" cy="{y + 11}" r="10" class=ico /><path d="M{x + 6},{y + 11}l4,4l7,-8" class=ico />'
    if kind == "link":
        return (f'<rect x="{x}" y="{y + 6}" width="10" height="10" rx="3" class=ico /><rect x="{x + 12}" y="{y + 6}" width="10" height="10" rx="3" class=ico />'
                f'<path d="M{x + 7},{y + 11}h8" class=ico />')
    return ""


def pipeline(width: int, variant: str) -> str:
    """The two pipelines: crop readers and BioMiner, as rows of steps."""
    lanes = [
        ("Crop readers", "s1", [
            ("page", "Render pages", "PyMuPDF, 300 dpi", "104 pages"),
            ("find", "Find drawings", "MolDetV2 detector", "237 drawings"),
            ("read", "Read each drawing", "MolScribe, MolSight|and MolGlyph", "one SMILES each"),
            ("score", "Score each paper", "RDKit, against|the answer key", "recall, precision"),
        ]),
        ("BioMiner", "s2", [
            ("page", "Page layout", "MinerU", "text, figures, tables"),
            ("find", "Find drawings", "MolDetV2 detector", "with page context"),
            ("read", "Read each drawing", "MolGlyph", "one SMILES each"),
            ("link", "Link and fill tables", "Gemini 3.5 Flash", "labelled structures"),
        ]),
    ]
    narrow = width < 480
    parts, y = [], 8
    for lane, cls, steps in lanes:
        parts.append(f'<text x="8" y="{y + 14}" class="t1 strong">{lane}</text>')
        y += 26
        if narrow:
            for i, (kind, title, tool, out) in enumerate(steps):
                parts.append(f'<rect x="8" y="{y}" width="{width - 16}" height="58" rx="10" class="card {cls}-tint"/>'
                             f'<rect x="8" y="{y}" width="5" height="58" rx="2" class="{cls}"/>'
                             + icon(kind, 24, y + 17) +
                             f'<text x="58" y="{y + 22}" class="t1 strong">{i + 1}. {title}</text>'
                             f'<text x="58" y="{y + 40}" class="t2 small">{tool.replace("|", " ") if len(tool) > 24 else tool + " · " + out}</text>')
                y += 58
                if i < len(steps) - 1:
                    parts.append(f'<path d="M{width / 2},{y + 2}v10" class=arrow /><path d="M{width / 2 - 4},{y + 8}l4,5l4,-5" class=arrow />')
                    y += 16
            y += 18
        else:
            gap, box_w = 22, (width - 16 - 3 * 22) / 4
            box_h = 134
            for i, (kind, title, tool, out) in enumerate(steps):
                x = 8 + i * (box_w + gap)
                lines = "".join(f'<text x="{x + 14:.1f}" y="{y + 82 + 16 * k}" class="t2 small">{t}</text>' for k, t in enumerate(tool.split("|")))
                parts.append(f'<rect x="{x:.1f}" y="{y}" width="{box_w:.1f}" height="{box_h}" rx="12" class="card {cls}-tint"/>'
                             f'<rect x="{x:.1f}" y="{y}" width="{box_w:.1f}" height="5" rx="2" class="{cls}"/>'
                             + icon(kind, x + 14, y + 18) +
                             f'<text x="{x + 46:.1f}" y="{y + 34}" class="t3 small">STEP {i + 1}</text>'
                             f'<text x="{x + 14:.1f}" y="{y + 62}" class="t1 strong">{title}</text>' + lines +
                             f'<text x="{x + 14:.1f}" y="{y + box_h - 12}" class="t3 small">{out}</text>')
                if i < len(steps) - 1:
                    ax = x + box_w + 4
                    parts.append(f'<path d="M{ax:.1f},{y + 58}h{gap - 8}" class=arrow /><path d="M{ax + gap - 12:.1f},{y + 54}l5,4l-5,4" class=arrow />')
            y += box_h + 24
    return svg(width, y, "".join(parts), "The two pipelines MolCensus runs", variant)


def explainer(width: int, variant: str) -> str:
    """Why R-group tables defeat a crop reader: a made-up core with two open positions and a three-row table."""
    narrow = width < 480

    def ring(cx: float, cy: float, r: float = 26):
        pts = [(cx + r * math.cos(math.radians(90 + 60 * k)), cy - r * math.sin(math.radians(90 + 60 * k))) for k in range(6)]
        path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + "Z"
        return f'<path d="{path}" class=bond /><circle cx="{cx}" cy="{cy}" r="{0.55 * r:.1f}" class=bond-thin />', pts

    def core(cx: float, cy: float, labels: tuple[str, str]) -> str:
        g, pts = ring(cx, cy)
        a, b = pts[1], pts[5]
        return (g + f'<path d="M{a[0]:.1f},{a[1]:.1f} l-22,-12" class=bond /><path d="M{b[0]:.1f},{b[1]:.1f} l22,-12" class=bond />'
                f'<text x="{a[0] - 30:.1f}" y="{a[1] - 14:.1f}" class="t1 atom" text-anchor="end">{labels[0]}</text>'
                f'<text x="{b[0] + 30:.1f}" y="{b[1] - 14:.1f}" class="t1 atom">{labels[1]}</text>')

    rows = [("a", "H", "CH₃"), ("b", "F", "OCH₃"), ("c", "Cl", "CF₃")]
    if narrow:
        parts, y = [], 10
        parts.append(f'<text x="12" y="{y + 12}" class="t3 small">WHAT THE PAPER PRINTS</text>')
        parts.append(core(110, y + 74, ("R¹", "R²")))
        tx = 200
        parts.append(f'<rect x="{tx}" y="{y + 30}" width="148" height="92" rx="8" class=card />')
        for j, h in enumerate(["", "R¹", "R²"]):
            parts.append(f'<text x="{tx + 14 + j * 46}" y="{y + 50}" class="t2 small strong">{h}</text>')
        for i, (c, r1, r2) in enumerate(rows):
            for j, v in enumerate([c, r1, r2]):
                parts.append(f'<text x="{tx + 14 + j * 46}" y="{y + 72 + i * 18}" class="t1 small">{v}</text>')
        y += 140
        parts.append(f'<rect x="12" y="{y}" width="{width - 24}" height="60" rx="10" class="card s2-tint"/>'
                     f'<text x="26" y="{y + 25}" class="t1 strong">A crop reader</text>'
                     f'<text x="26" y="{y + 44}" class="t2 small">sees one drawing: the core with open positions</text>')
        y += 72
        parts.append(f'<rect x="12" y="{y}" width="{width - 24}" height="60" rx="10" class="card s1-tint"/>'
                     f'<text x="26" y="{y + 25}" class="t1 strong">A table reader</text>'
                     f'<text x="26" y="{y + 44}" class="t2 small">builds compounds a, b and c, row by row</text>')
        return svg(width, y + 72, "".join(parts), "Why R-group tables defeat a crop reader", variant)

    if width < 640:
        parts = [f'<text x="12" y="20" class="t3 small">WHAT THE PAPER PRINTS</text>',
                 f'<rect x="8" y="30" width="{width - 16}" height="170" rx="14" class=card />']
        parts.append(core(120, 128, ("R¹", "R²")))
        tx = 250
        for j, h in enumerate(["", "R¹", "R²"]):
            parts.append(f'<text x="{tx + j * 64}" y="70" class="t2 small strong">{h}</text>')
        parts.append(f'<path d="M{tx - 8},78h196" class=rule />')
        for i, (c, r1, r2) in enumerate(rows):
            for j, v in enumerate([c, r1, r2]):
                parts.append(f'<text x="{tx + j * 64}" y="{100 + i * 24}" class="t1">{v}</text>')
        parts.append(f'<text x="{width / 2:.0f}" y="190" class="t3 small" text-anchor="middle">one drawing, three compounds</text>')
        parts.append(f'<text x="12" y="228" class="t3 small">WHAT EACH KIND OF TOOL GETS</text>')
        card_w = (width - 16 - 12) / 2
        for k, (cls, title, lines) in enumerate([
                ("s2", "Crop reader", ["sees one drawing: the core", "with open positions, so it", "finds none of the three"]),
                ("s1", "Table reader, like BioMiner", ["fills in each row of the", "table, so it can produce", "compounds a, b and c"])]):
            x = 8 + k * (card_w + 12)
            parts.append(f'<rect x="{x:.1f}" y="238" width="{card_w:.1f}" height="110" rx="14" class="card {cls}-tint"/>'
                         f'<text x="{x + 16:.1f}" y="266" class="t1 strong">{title}</text>'
                         + "".join(f'<text x="{x + 16:.1f}" y="{290 + 18 * m}" class="t2 small">{t}</text>' for m, t in enumerate(lines)))
        return svg(width, 356, "".join(parts), "Why R-group tables defeat a crop reader", variant)

    parts = [f'<text x="20" y="22" class="t3 small">WHAT THE PAPER PRINTS</text>',
             f'<text x="430" y="22" class="t3 small">WHAT EACH KIND OF TOOL GETS</text>']
    parts.append(f'<rect x="10" y="34" width="380" height="190" rx="14" class=card />')
    parts.append(core(110, 140, ("R¹", "R²")))
    tx = 220
    for j, h in enumerate(["", "R¹", "R²"]):
        parts.append(f'<text x="{tx + j * 50}" y="80" class="t2 small strong">{h}</text>')
    parts.append(f'<path d="M{tx - 8},88h150" class=rule />')
    for i, (c, r1, r2) in enumerate(rows):
        for j, v in enumerate([c, r1, r2]):
            parts.append(f'<text x="{tx + j * 50}" y="{110 + i * 24}" class="t1">{v}</text>')
    parts.append(f'<text x="200" y="208" class="t3 small" text-anchor="middle">one drawing, three compounds</text>')
    parts.append(f'<path d="M396,129h22" class=arrow /><path d="M412,125l6,4l-6,4" class=arrow />')
    parts.append(f'<rect x="428" y="34" width="322" height="88" rx="14" class="card s2-tint"/>'
                 f'<text x="446" y="64" class="t1 strong">Crop reader</text>'
                 f'<text x="446" y="86" class="t2">reads the core and its open positions,</text>'
                 f'<text x="446" y="106" class="t2">so it finds none of the three compounds</text>')
    parts.append(f'<rect x="428" y="136" width="322" height="88" rx="14" class="card s1-tint"/>'
                 f'<text x="446" y="166" class="t1 strong">Table reader, like BioMiner</text>'
                 f'<text x="446" y="188" class="t2">fills in each row of the table,</text>'
                 f'<text x="446" y="208" class="t2">so it can produce compounds a, b and c</text>')
    return svg(width, 236, "".join(parts), "Why R-group tables defeat a crop reader", variant)


def waffle(counts: list[tuple[str, str, int]], columns: int = 22) -> str:
    """One square per molecule, colored by what happened to it."""
    cell, gap = 16, 3
    total = sum(n for _, _, n in counts)
    rows_ = math.ceil(total / columns)
    parts, i = [], 0
    for cls, name, n in counts:
        for _ in range(n):
            x, y = (i % columns) * (cell + gap), (i // columns) * (cell + gap)
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="3" class="{cls}"><title>{esc(name)}</title></rect>')
            i += 1
    w, h = columns * (cell + gap) - gap, rows_ * (cell + gap) - gap
    return svg(w, h, "".join(parts), "Each of the 219 course molecules, by what happened to it", "waffle")


def stacked(rows: list[dict], segments: list[tuple[str, str, str]], width: int, variant: str) -> str:
    """Horizontal stacked bars of counts per paper, with a 2px gap between segments."""
    narrow = width < 480
    left, right = (12, width - 46) if narrow else (96, width - 50)
    scale = (right - left) / max(sum(r[key] for key, _, _ in segments) for r in rows)
    parts, y, bar_h = [], 4, 16
    for row in rows:
        if narrow:
            parts.append(f'<text x="{left}" y="{y + 11}" class="t2 small">{esc(row["label"])}</text>')
            y += 16
        else:
            parts.append(f'<text x="{left - 12}" y="{y + 12}" class="t2" text-anchor="end">{esc(row["label"])}</text>')
        x = left
        present = [(k, c, n) for k, c, n in segments if row[k]]
        for i, (key, cls, name) in enumerate(present):
            w = row[key] * scale
            last = i == len(present) - 1
            parts.append(f'<g><title>{esc(row["label"])}: {name}, {row[key]}</title>'
                         f'<path d="{bar_path(x, y, max(w - (0 if last else 2), 0.8), bar_h, round_end=last)}" class="{cls}"/></g>')
            x += w
        parts.append(f'<text x="{x + 6:.1f}" y="{y + 12}" class="t1 num small">{sum(row[k] for k, _, _ in segments)}</text>')
        y += bar_h + (12 if narrow else 9)
    return svg(width, y, "".join(parts), "Course molecules per paper by what happened to them", variant)


def pr_map(points: list[dict], width: int, variant: str) -> str:
    """Precision against recall for every tool, with 95% intervals and faint lines of equal F1."""
    narrow = width < 480
    left, top = 62, 14
    plot_w, plot_h = width - left - (16 if narrow else 150), (260 if narrow else 330)
    x_max = 0.5

    def X(r: float) -> float:
        return left + r / x_max * plot_w

    def Y(p: float) -> float:
        return top + (1 - p) * plot_h

    parts = []
    for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        parts.append(f'<path d="M{X(t):.1f},{top}V{top + plot_h}" class=grid /><text x="{X(t):.1f}" y="{top + plot_h + 18}" class="t3 small num" text-anchor="middle">{int(t * 100)}%</text>')
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        parts.append(f'<path d="M{left},{Y(t):.1f}H{left + plot_w}" class=grid /><text x="{left - 8}" y="{Y(t) + 4:.1f}" class="t3 small num" text-anchor="end">{int(t * 100)}%</text>')
    for f in (0.3, 0.4, 0.5):
        pts = [(r, f * r / (2 * r - f)) for r in [f / 2 + 0.005 * k for k in range(1, 200)] if 0 < f * r / (2 * r - f) <= 1 and r <= x_max]
        d = "M" + " L".join(f"{X(r):.1f},{Y(p):.1f}" for r, p in pts)
        lx, ly = pts[-1]
        parts.append(f'<path d="{d}" class=iso /><text x="{X(lx) - 4:.1f}" y="{Y(ly) - 6:.1f}" class="t3 tiny" text-anchor="end">F1 {f:.1f}</text>')
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{top + plot_h + 38}" class="t2 small" text-anchor="middle">Recall: share of the paper\'s molecules found</text>'
                 f'<text x="10" y="{top + plot_h / 2:.1f}" class="t2 small" text-anchor="middle" dominant-baseline="hanging" transform="rotate(-90 10 {top + plot_h / 2:.1f})">Precision</text>')
    for p in points:
        cx, cy = X(p["r"]), Y(p["p"])
        tip = esc(f'{p["name"]}: recall {pc(p["r"])}, precision {pc(p["p"])}')
        whisk = ""
        if p.get("rl") is not None:
            whisk = (f'<path d="M{X(p["rl"]):.1f},{cy:.1f}H{X(p["rh"]):.1f}M{cx:.1f},{Y(p["pl"]):.1f}V{Y(p["ph"]):.1f}" class="ci-line {p["cls"]}-stroke"/>')
        mark = (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" class="{p["cls"]} ring"/>' if not p.get("hollow")
                else f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" class="hollow {p["cls"]}-stroke"/>')
        dx, dy, anchor = p.get("label_narrow", p.get("label_at", (10, 4, "start"))) if narrow else p.get("label_at", (10, 4, "start"))
        label = "" if narrow and p.get("narrow_hide") else f'<text x="{cx + dx:.1f}" y="{cy + dy:.1f}" class="t1 small strong" text-anchor="{anchor}">{esc(p["short"] if narrow else p["name"])}</text>'
        parts.append(f'<g><title>{tip}</title>{whisk}{mark}{label}</g>')
    return svg(width, top + plot_h + 48, "".join(parts), "Precision against recall for each tool on the course papers", variant)


def dumbbell(rows: list[dict], width: int, variant: str) -> str:
    """Per structure type: linked recall, then all-extracted recall, with the published value as a diamond."""
    narrow = width < 480
    left = 14 if narrow else 150
    right = width - 30
    x_max = 0.7

    def X(v: float) -> float:
        return left + v / x_max * (right - left)

    parts, y = [], 8
    for t in [0, 0.2, 0.4, 0.6]:
        parts.append(f'<path d="M{X(t):.1f},4V{(len(rows)) * (72 if narrow else 50) + 8}" class=grid />')
    for row in rows:
        if narrow:
            parts.append(f'<text x="{left}" y="{y + 12}" class="t1 small strong">{esc(row["name"])}</text>')
            y += 22
        else:
            parts.append(f'<text x="{left - 14}" y="{y + 18}" class="t1 strong" text-anchor="end">{esc(row["name"])}</text>')
        cy = y + 14
        a, b, pub = X(row["linked"]), X(row["all"]), X(row["published"])
        parts.append(f'<g><title>{esc(row["name"])}: linked {pc(row["linked"])}, all extracted {pc(row["all"])}, published {pc(row["published"])}</title>'
                     f'<path d="M{a:.1f},{cy}H{b:.1f}" class=gap-line />'
                     f'<circle cx="{a:.1f}" cy="{cy}" r="7" class="s1 ring"/><circle cx="{b:.1f}" cy="{cy}" r="7" class="s2 ring"/>'
                     f'<path d="M{pub:.1f},{cy - 8}l8,8l-8,8l-8,-8z" class="s3 ring"/></g>')
        lo, hi = sorted([(a, pc(row["linked"], 0)), (b, pc(row["all"], 0))])
        if hi[0] - lo[0] < 34:  # too close for two labels
            parts.append(f'<text x="{(lo[0] + hi[0]) / 2:.1f}" y="{cy + 24}" class="t2 tiny num" text-anchor="middle">{lo[1]} and {hi[1]}</text>')
        else:
            parts.append(f'<text x="{lo[0]:.1f}" y="{cy + 24}" class="t2 tiny num" text-anchor="middle">{lo[1]}</text>'
                         f'<text x="{hi[0]:.1f}" y="{cy + 24}" class="t2 tiny num" text-anchor="middle">{hi[1]}</text>')
        parts.append(f'<text x="{pub:.1f}" y="{cy - 12}" class="t3 tiny num" text-anchor="middle">{pc(row["published"], 0)}</text>')
        y += 72 if narrow else 50
    for t in [0, 0.2, 0.4, 0.6]:
        parts.append(f'<text x="{X(t):.1f}" y="{y + 12}" class="t3 tiny num" text-anchor="middle">{int(t * 100)}%</text>')
    return svg(width, y + 18, "".join(parts), "BioVista recall by structure type", variant)


def forest(rows: list[dict], width: int, variant: str) -> str:
    """Differences in precision with their paper-bootstrap 95% intervals, against a zero line."""
    narrow = width < 480
    left, right = (14, width - 20) if narrow else (190, width - 30)
    lo_v, hi_v = -0.15, 0.25

    def X(v: float) -> float:
        return left + (v - lo_v) / (hi_v - lo_v) * (right - left)

    parts, y = [], 10
    n = len(rows)
    zero_bottom = 10 + n * (64 if narrow else 44)
    parts.append(f'<path d="M{X(0):.1f},4V{zero_bottom}" class=zero />')
    for row in rows:
        if narrow:
            parts.append(f'<text x="{left}" y="{y + 10}" class="t1 small strong">{esc(row["name"])}</text>')
            y += 18
        else:
            parts.append(f'<text x="{left - 14}" y="{y + 18}" class="t1 small" text-anchor="end">{esc(row["name"])}</text>')
        cy = y + 14
        parts.append(f'<g><title>{esc(row["name"])}: {row["label"]}</title>'
                     f'<path d="M{X(row["low"]):.1f},{cy}H{X(row["high"]):.1f}" class="ci-line s1-stroke wide"/>'
                     f'<circle cx="{X(row["est"]):.1f}" cy="{cy}" r="7" class="s1 ring"/></g>'
                     f'<text x="{X(row["est"]):.1f}" y="{cy - 12}" class="t1 tiny num" text-anchor="middle">{row["est_label"]}</text>')
        y += 46 if narrow else 44
    for t in [-0.1, 0, 0.1, 0.2]:
        lab = ("+" if t > 0 else MINUS if t < 0 else "") + f"{abs(round(t * 100))}"
        parts.append(f'<text x="{X(t):.1f}" y="{y + 10}" class="t3 tiny num" text-anchor="middle">{lab}</text>')
    parts.append(f'<text x="{(left + right) / 2:.1f}" y="{y + 28}" class="t2 tiny" text-anchor="middle">points of precision, agreement minus confidence</text>')
    return svg(width, y + 36, "".join(parts), "Precision gap between reader agreement and MolScribe's confidence", variant)


def slope(left_vals: dict, right_vals: dict, width: int, variant: str) -> str:
    """Rank of each reader on USPTO patent images and on the course papers."""
    narrow = width < 480
    xl, xr = (90, width - 90) if narrow else (round(width * 0.31), round(width * 0.69))
    colors = {"molscribe": "s1", "molsight": "s2", "molglyph": "s3"}
    rank_l = {t: i for i, t in enumerate(sorted(left_vals, key=lambda t: -left_vals[t]))}
    rank_r = {t: i for i, t in enumerate(sorted(right_vals, key=lambda t: -right_vals[t]))}

    def Y(rank: int) -> float:
        return 56 + rank * 54

    parts = [f'<text x="{xl}" y="22" class="t2 small strong" text-anchor="middle">USPTO patents</text>',
             f'<text x="{xr}" y="22" class="t2 small strong" text-anchor="middle">Course papers</text>',
             f'<text x="{xl}" y="38" class="t3 tiny" text-anchor="middle">accuracy per image</text>',
             f'<text x="{xr}" y="38" class="t3 tiny" text-anchor="middle">recall per molecule</text>']
    for t in colors:
        yl, yr = Y(rank_l[t]), Y(rank_r[t])
        parts.append(f'<g><title>{READER_NAMES[t]}: {pc(left_vals[t])} on USPTO, {pc(right_vals[t])} on the course papers</title>'
                     f'<path d="M{xl},{yl}L{xr},{yr}" class="slope-line {colors[t]}-stroke"/>'
                     f'<circle cx="{xl}" cy="{yl}" r="7" class="{colors[t]} ring"/><circle cx="{xr}" cy="{yr}" r="7" class="{colors[t]} ring"/></g>'
                     f'<text x="{xl - 14}" y="{yl + 4}" class="t1 small" text-anchor="end">{"" if narrow else READER_NAMES[t] + "  "}{pc(left_vals[t])}</text>'
                     f'<text x="{xr + 14}" y="{yr + 4}" class="t1 small">{pc(right_vals[t])}{"" if narrow else "  " + READER_NAMES[t]}</text>')
    return svg(width, Y(2) + 26, "".join(parts), "Reader ranking on patents and on real papers", variant)


def number_line(marks: list[dict], band: tuple[float, float], width: int, variant: str) -> str:
    """Values on a shared 80 to 100% scale, with the canonical-SMILES interval as a band."""
    left, right = 20, width - 20
    lo_v, hi_v = 0.80, 1.00

    def X(v: float) -> float:
        return left + (v - lo_v) / (hi_v - lo_v) * (right - left)

    parts = [f'<path d="M{left},70H{right}" class=axis-line />']
    for t in [0.80, 0.85, 0.90, 0.95, 1.00]:
        parts.append(f'<path d="M{X(t):.1f},66V74" class=axis-line /><text x="{X(t):.1f}" y="90" class="t3 tiny num" text-anchor="middle">{int(t * 100)}%</text>')
    parts.append(f'<rect x="{X(band[0]):.1f}" y="62" width="{X(band[1]) - X(band[0]):.1f}" height="16" rx="4" class="s1-tint band"/>')
    narrow = width < 480
    for m in marks:
        x = X(m["v"])
        up = m.get("up", True)
        ty = 40 if up else 112
        parts.append(f'<g><title>{esc(m["name"])}: {pc(m["v"])}</title><circle cx="{x:.1f}" cy="70" r="7" class="{m["cls"]} ring"/>'
                     f'<path d="M{x:.1f},{62 if up else 78}V{ty + (6 if up else -14)}" class=leader /></g>'
                     f'<text x="{x:.1f}" y="{ty - (14 if up else 0)}" class="t1 small strong" text-anchor="middle">{pc(m["v"])}</text>'
                     f'<text x="{x:.1f}" y="{ty + (0 if up else 14)}" class="t2 tiny" text-anchor="middle">{esc(m["short"] if narrow else m["name"])}</text>')
    return svg(width, 132, "".join(parts), "MolScribe on the UOB images under two matching rules", variant)


def cost_bar(width: int, variant: str) -> str:
    total = sum(v for *_, v in COSTS)
    classes = ["s1", "s4", "s2", "s3"]
    parts, x, left, right = [], 12, 12, width - 12
    scale = (right - left) / total
    for i, ((name, _, v), cls) in enumerate(zip(COSTS, classes)):
        w = v * scale
        last = i == len(COSTS) - 1
        parts.append(f'<g><title>{esc(name)}: ${v:.2f}</title><path d="{bar_path(x, 8, w - (0 if last else 2), 26, round_end=last)}" class="{cls}"/></g>')
        x += w
    return svg(width, 42, "".join(parts), "Cloud cost of the project at list price", variant)


# ---------- page pieces ----------

def tiles(items: list[tuple[str, str]]) -> str:
    return "<div class=tiles>" + "".join(f"<div class=tile><div class=tile-num>{n}</div><div class=tile-label>{lab}</div></div>"
                                         for n, lab in items) + "</div>"


def finding(anchor: str, number: int, title: str, lead: str, body: str, caveat: str) -> str:
    return (f'<section id="{anchor}" class=finding><div class=wrap>'
            f'<div class=kicker>Finding {number}</div><h2>{title}</h2><p class=lead>{lead}</p>{body}'
            f'<div class=caveat><span>Caveat</span>{caveat}</div></div></section>')


def finding_checks(d: dict) -> str:
    c = d["biominer_run"]["calls"]
    cards = [
        ("Mac, 19 October", "34.8%", "MolScribe on ACS journal images, against 71.9% in its paper",
         "It returned no molecule for any drawing with an abbreviation such as Me or Ph. The expansion step read bond "
         "orders from bonds it had already deleted, which fails on macOS, and MolScribe hid the error.",
         "A three-line patch. ACS accuracy went to 73.1%."),
        ("AWS GPU, 3 November", "0.0%", "MolScribe and MolSight on every public image set",
         "The machine image put its own cuDNN on the library path, so PyTorch mixed two versions and every convolution "
         "failed. The usual GPU check still passed.",
         "Setup now runs one real convolution per environment. USPTO went back to 92.9% and 91.4%."),
        ("AWS GPU, 3 November", "exit 0", "MinerU, BioMiner's layout tool, after a crash",
         "A model file was missing from the current download. MinerU crashed and still reported success.",
         "The setup check now looks for MinerU's text output instead of its exit code."),
        ("AWS GPU, 4 November", "21 refused", "Gemini calls refused for quota in BioMiner's smoke test",
         "A key meant to be paid was held to free-tier limits of 20 requests a day. BioMiner carries on with an empty "
         "answer when a call fails, so the run looked normal.",
         f"The wrapper counts failed calls and the smoke test stops the run. On Vertex AI: {c['answered']} calls, none lost."),
    ]
    grid = "<div class=incidents>" + "".join(
        f"<article class=incident><div class=inc-when>{when}</div><div class=inc-num>{num}</div><div class=inc-what>{what}</div>"
        f"<p>{why}</p><div class=inc-fix><span>Check added</span>{fix}</div></article>" for when, num, what, why, fix in cards) + "</div>"
    return finding("checks", 1, "Every major failure was silent",
                   "Four failures in this project ran to the end without an error message. Each was caught only "
                   "because a check compared the output with an answer known in advance.", grid,
                   "These checks cover the failures that happened. A fault that still produces believable numbers on the "
                   "public sets would get through.")


def finding_losses(d: dict) -> str:
    rc, found = d["recall_ceiling"], d["readers_found_inchikey"]["by_paper"]
    rows = []
    for p in PAPERS:
        b = rc["by_paper"].get(p, {})
        rows.append({"label": f"Paper {p}", "found": found[p]["found"], "table": b.get("3", 0), "reading": b.get("1", 0),
                     "unresolved": b.get("5", 0), "n": found[p]["n"]})
    total = {k: sum(r[k] for r in rows) for k in ("found", "table", "reading", "unresolved", "n")}
    segments = [("found", "s1", "Found by a reader"), ("table", "s2", "Only in an R-group table"),
                ("reading", "s3", "Drawn, but misread"), ("unresolved", "s4", "Unresolved")]
    j, strict = rc["judges"], rc["strict"]
    waffle_fig = figure(legend([(cls, f"{name} ({total[key]})") for key, cls, name in segments]) +
                        waffle([(cls, name, total[key]) for key, cls, name in segments]),
                        f"Each square is one of the {total['n']} distinct course molecules (InChIKey).", "waffle-fig")
    per_paper = figure(both(lambda w, v: stacked(rows, segments, w, v)),
                       "The same molecules by paper. Papers 4 and 19 hold 164 of them.")
    body = f"""
<div class=split>
<div>
<p>Three crop readers, MolScribe, MolSight and MolGlyph, read the same 237 drawings that the MolDetV2 detector cut
out of the six papers. Matched by InChIKey (a fixed-length code computed from a molecule, so two spellings of the same
molecule match), they found {total['found']} of the {total['n']} molecules between them.</p>
<p>Every one of the {rc['missed']} misses was then traced to a cause. Automatic tests went first: a second detection
pass at a much lower threshold, an exact substructure test of every reading with open positions, and a similarity
test for near misses. Two AI judges, Claude and Gemini 3.5 Flash, looked at the rest using the page images. Their
answers counted only when both agreed, and only after both passed a blind test on molecules with known answers.</p>
</div>
<div>{waffle_fig}</div>
</div>
{tiles([(pc(j['claude'][0] / j['claude'][1]), f"Claude correct on {j['claude'][1]} known cases"),
        (pc(j['gemini'][0] / j['gemini'][1]), f"Gemini correct on {j['gemini'][1]} known cases"),
        (f"{rc['kappa_all']:.2f}", "Cohen's kappa between the judges (1 is perfect agreement)"),
        ("34 to 37%", "recall ceiling for any reader that works one drawing at a time")])}
{per_paper}
<p>No missed molecule turned out to be a full drawing the detector skipped, and none was missing from the figures.
The two reading losses are large ring molecules in paper 16, drawn under a colored heatmap. Between 75 and 81 of the
219 molecules are drawn in full anywhere in the papers. Better reading could add at most 2 to 8 molecules; the rest
need a tool that reads tables. One rule withdrew an automatic label when both judges contradicted it. Without that
rule the totals are {strict.get('3', 0)} table losses, {strict.get('1', 0)} reading losses and {strict.get('5', 0)} unresolved.</p>"""
    return finding("losses", 2, "Almost every missed molecule lives in an R-group table",
                   f"{total['table']} of the {rc['missed']} molecules the crop readers missed are never drawn as a whole. "
                   "They exist only as rows of a table under a shared core.", body,
                   "The automatic table test treats any symbol a reader could not expand as an open position, so an odd "
                   "abbreviation can pass for a table compound; the judges caught four such cases. Claude also wrote the "
                   "evidence code, so the judges are not fully independent of the analysis.")


def course_points(d: dict) -> list[dict]:
    """The precision-recall map's points: BioMiner's three rules, the readers and Obaidullah's run."""
    s = d["biominer_course"]["scores"]

    def pt(key: str, name: str, short: str, cls: str, at=(10, 4, "start"), hide=False) -> dict:
        r, p = s[key]["ALL"]["recall"], s[key]["ALL"]["precision"]
        return {"name": name, "short": short, "cls": cls, "r": r["value"], "p": p["value"], "rl": r["low"], "rh": r["high"],
              "pl": p["low"], "ph": p["high"], "label_at": at, "narrow_hide": hide}

    ob = OBAIDULLAH_BIOMINER["course"]
    return [pt("no_star", "BioMiner, every whole structure", "BioMiner, all", "s1", (12, -10, "start")),
            pt("linked", "BioMiner, linked only", "linked", "s1", (-12, -10, "end")),
            pt("all_extracted", "BioMiner, with table pieces", "with pieces", "s1", (12, 4, "start")),
            {**pt("molglyph", "MolGlyph", "MolGlyph", "s2", (12, -6, "start")), "label_narrow": (-12, -8, "end")},
            pt("molsight", "MolSight", "MolSight", "s2", (12, 6, "start"), hide=True),
            pt("molscribe", "MolScribe", "MolScribe", "s2", (-12, 12, "end")),
            {"name": "Obaidullah's BioMiner run", "short": "Obaidullah", "cls": "s3", "r": ob["recall"] / 100, "p": ob["precision"] / 100,
             "hollow": True, "label_at": (12, 4, "start")}]


def finding_biominer(d: dict) -> str:
    bv = d["biominer_biovista"]["by_type"]
    order = [("full", "Full structures"), ("chiral", "Chiral structures"), ("R-group", "R-group tables"), ("all", "All structures")]
    db_rows = [{"name": name, "linked": bv[k]["linked"]["mean_recall"], "all": bv[k]["no_star"]["mean_recall"],
                "published": bv[k]["published_recall"]} for k, name in order]
    gap_rows = []
    for k, name in order:
        e = bv[k]
        gap = e["published_recall"] - e["linked"]["mean_recall"]
        gap_rows.append([name, e["linked"]["recall"]["n"], pc(e["linked"]["mean_recall"]), pc(e["no_star"]["mean_recall"]),
                         pc(e["published_recall"]), pc((e["no_star"]["mean_recall"] - e["linked"]["mean_recall"]) / gap, 0)])
    full, rg = bv["full"], bv["R-group"]
    ob_bv = OBAIDULLAH_BIOMINER["biovista"]
    biovista = f"""
<h3>On BioVista, most of the gap is linking</h3>
<p>BioVista (Yan et al., 2026) is BioMiner's own benchmark, with every structure in each paper labelled. On 12 of its
papers I scored the structures by type twice: once as BioMiner links them to compound labels, and once counting every
whole structure its reader found. For full structures, counting what MolGlyph read but Gemini did not link lifts
recall from {pc(full['linked']['mean_recall'])} to {pc(full['no_star']['mean_recall'])}, close to the published
{pc(full['published_recall'])}. Every linked full structure is one of these readings, so no linker could do better with
the same detector and reader. For R-group tables almost nothing changes, because those compounds exist only after the
language model fills in the table.</p>
{figure(legend([("s1", "Linked to a label"), ("s2", "Every whole structure read"), ("s3", "Published, BioMiner-Instruct")]) +
        both(lambda w, v: dumbbell(db_rows, w, v)), "Mean recall over 12 BioVista papers. Published values average 450 papers.")}
{table(["Structure type", "Labels", "Linked", "All extracted", "Published", "Gap closed"], gap_rows)}
<p>On 145 BioVista papers, Obaidullah's BioMiner run reached {ob_bv['recall']}% recall at {ob_bv['precision']}% precision,
close to the published result (Obaidullah, 2026). That fits the reading above: the shortfall here is mostly the
language model doing the linking.</p>"""
    plain = ("<p class=plain>BioMiner ran with Gemini 3.5 Flash on Google Cloud Vertex AI, not with its own fine-tuned "
             "model, BioMiner-Instruct, and on 12 BioVista papers rather than the 450 its authors used.</p>")
    caveat = ("Twelve papers against 450, and they were not chosen to match BioVista's test set. A BioMiner-Instruct run "
              "on the same papers would separate the model's share of the gap from the sample's.")
    if not INCLUDE_COURSE:
        return finding("biominer", 3, "BioMiner's table step",
                       "BioMiner reads a whole PDF and fills in R-group tables, which crop readers cannot do.",
                       plain + biovista, caveat)

    c = d["biominer_course"]
    s = c["scores"]

    ob = OBAIDULLAH_BIOMINER["course"]
    points = course_points(d)
    names = [("no_star", "BioMiner, every whole structure"), ("linked", "BioMiner, linked only"),
             ("all_extracted", "BioMiner, with table pieces"), ("molglyph", "MolGlyph"), ("molsight", "MolSight"), ("molscribe", "MolScribe")]
    course_table = [[name, with_ci(s[k]["ALL"]["recall"]), with_ci(s[k]["ALL"]["precision"]), pc(s[k]["ALL"]["f1"]),
                     s[k]["ALL"]["precision"]["n"]] for k, name in names]
    course_table.append(["Obaidullah's BioMiner run", f"{ob['recall']}%", f"{ob['precision']}%", f"{ob['f1']}%", "about 105"])
    head, only, counts = s["no_star"]["ALL"], c["only_biominer"], c["counts"]
    body = f"""{plain}
<p>BioMiner (Yan et al., 2026) works on the whole PDF. MinerU finds the layout, MolDetV2 the drawings and MolGlyph reads
them, then a language model ties each structure to its compound label and fills in the tables. What counts as a
BioMiner prediction turned out to matter. BioMiner itself keeps only the {counts['pipeline_rows']} structures it linked to
a label. Its reader found {counts['unlinked']} more, and {counts['unlinked_star']} of those are table scaffolds and
substituent pieces rather than molecules. The headline counts every whole structure, linked or not.</p>
{figure(legend([("s1", "BioMiner"), ("s2", "Crop readers"), ("s3", "Obaidullah's run")]) + both(lambda w, v: pr_map(points, w, v)),
        "Precision against recall on the six course papers (canonical SMILES), with 95% Wilson intervals. Up and to the "
        "right is better; faint curves mark equal F1.")}
<div class=split>
<div class=side>
{tiles([(pc(head['recall']['value']), "recall, every whole structure"), (pc(head['precision']['value']), "precision, the same rule"),
        (str(sum(only.values())), f"molecules no crop reader found: {only['4']} in paper 4, {only['19']} in paper 19")])}
</div>
<div>
<p>All {c['only_biominer_from_rgroup']} of those molecules came from filling in R-group tables, which is BioMiner's
real contribution over a crop reader. Obaidullah's BioMiner run on the same papers reached {ob['recall']}% recall at
{ob['precision']}% precision (Obaidullah, 2026), which matches ours once every structure read is counted.</p>
</div>
</div>
<details><summary>All course numbers, with intervals</summary>
{table(["Prediction rule", "Recall", "Precision", "F1", "Predictions"], course_table)}
<p class=table-note>Leaving out the pieces with open positions also drops two correct paper-19 molecules whose answer key
itself writes a generic group as an open position, so this rule finds 75 molecules and the version with pieces 77.</p>
</details>"""
    return finding("biominer", 3, "BioMiner's table step finds what crop readers cannot",
                   f"Counting every whole structure BioMiner reads gives {pc(head['recall']['value'])} recall at "
                   f"{pc(head['precision']['value'])} precision on the course papers, the best balance of any tool here.",
                   body + biovista, caveat)


def finding_agreement(d: dict) -> str:
    sel, diff = d["agreement"]["selections"], d["agreement"]["differences"]

    def signed(x: float) -> str:
        return ("+" if x >= 0 else MINUS) + f"{abs(100 * x):.1f}"

    rows = [{"name": f"All three agree vs top {sel['all_three']['crops']}", "est": diff["a_minus_c"]["estimate"],
             "low": diff["a_minus_c"]["low"], "high": diff["a_minus_c"]["high"], "est_label": signed(diff["a_minus_c"]["estimate"]),
             "label": f"{signed(diff['a_minus_c']['estimate'])} points, 95% interval {signed(diff['a_minus_c']['low'])} to {signed(diff['a_minus_c']['high'])}"},
            {"name": f"Two agree vs top {sel['two_agree']['crops']}", "est": diff["b_minus_c"]["estimate"],
             "low": diff["b_minus_c"]["low"], "high": diff["b_minus_c"]["high"], "est_label": signed(diff["b_minus_c"]["estimate"]),
             "label": f"{signed(diff['b_minus_c']['estimate'])} points, 95% interval {signed(diff['b_minus_c']['low'])} to {signed(diff['b_minus_c']['high'])}"}]
    t_rows = []
    for name, k in [("All three readers agree", "all_three"), (f"MolScribe's top {sel['top_conf_a']['crops']} by confidence", "top_conf_a"),
                    ("At least two readers agree", "two_agree"), (f"MolScribe's top {sel['top_conf_b']['crops']} by confidence", "top_conf_b")]:
        t_rows.append([name, sel[k]["crops"], with_ci(sel[k]["precision"]), f"{sel[k]['precision']['k']} of {sel[k]['precision']['n']}"])
    ahead = sum(1 for p in PAPERS if sel["top_conf_a"]["per_paper"][p][1] and sel["all_three"]["per_paper"][p][1]
                and sel["all_three"]["per_paper"][p][0] / sel["all_three"]["per_paper"][p][1]
                > sel["top_conf_a"]["per_paper"][p][0] / sel["top_conf_a"]["per_paper"][p][1])
    both_n = sum(1 for p in PAPERS if sel["top_conf_a"]["per_paper"][p][1] and sel["all_three"]["per_paper"][p][1])
    ob = OBAIDULLAH_READERS
    ob_rows = [[r] + [f"{v:.1f}" for v in ob[r]] for r in ["MolScribe", "MolNexTR", "DECIMER", "MolVec", "MolGlyph", "OCSRGlyph"]]
    body = f"""
<div class=split>
<div>
<p>A reader's confidence is its own estimate that a reading is right; of the three, only MolScribe reports one. All three
readers wrote the same molecule for {sel['all_three']['crops']} of the 237 drawings, and {pc(sel['all_three']['precision']['value'])}
of those molecules were correct. MolScribe's {sel['top_conf_a']['crops']} most confident drawings gave
{pc(sel['top_conf_a']['precision']['value'])}. With two readers agreeing the figures are {pc(sel['two_agree']['precision']['value'])}
and {pc(sel['top_conf_b']['precision']['value'])}.</p>
<p>Molecules from the same paper succeed or fail together, so the test resamples whole papers: 10,000 draws of the six
papers, the same draws for both selections. Agreement leads in {ahead} of the {both_n} papers where both selections have
predictions, but both intervals cross zero.</p>
</div>
<div>{figure(both(lambda w, v: forest(rows, w, v), 520), "Gap in precision with a 95% paper-bootstrap interval (InChIKey rule).")}</div>
</div>
{table(["Selection", "Drawings", "Precision", "Correct"], t_rows)}
<h3>Why agreement could help</h3>
<p>Obaidullah benchmarked six readers on five public image sets (Obaidullah, 2026), and the best reader changes with the
source. OCSRGlyph led on USPTO patents and CLEF but came fourth of six on MolRecBench-Wild journal images, where MolGlyph
led. Readers that fail on different images are exactly what an agreement filter needs, and the VERDICT preprint reports
the same direction on real documents (Guan et al., 2026).</p>
{table(["Reader"] + ob["datasets"], ob_rows, "Obaidullah (2026): exact-match accuracy (%) on cropped public images, with his crop counts.")}"""
    return finding("agreement", 4, "Agreement between readers beats confidence, so far",
                   "When all three readers write the same molecule for a drawing, it is right more often than MolScribe's "
                   "most confident readings. Six papers are not yet enough to show the gap is real.", body,
                   "More papers, and more readers such as the six in Obaidullah's benchmark, would settle it.")


def finding_public(d: dict) -> str:
    pub, t = d["public"], d["tautomers_uob_molscribe"]
    sets = [("uspto", "USPTO"), ("uob", "UOB"), ("clef", "CLEF"), ("acs", "ACS"), ("biovista", "BioVista")]
    rows = []
    for key, name in sets:
        for i, tool in enumerate(["molscribe", "molsight", "molglyph"]):
            v = pub[key][tool]
            published = PUBLISHED.get((key, tool))
            mate = ""
            if key in ("uspto", "uob", "clef") and tool != "molsight":
                mate = f"{OBAIDULLAH_READERS[READER_NAMES[tool]][['uspto', 'uob', 'jpo', 'clef'].index(key)]:.1f}%"
            if key == "biovista" and READER_NAMES[tool] in SYEDA_BIOVISTA:
                mate = f"{SYEDA_BIOVISTA[READER_NAMES[tool]]}%"
            rows.append([f"{name} ({v['n']:,})" if i == 0 else "", READER_NAMES[tool], pc(v["inchikey"]["value"]),
                         with_ci(v["smiles"]), pc(published) if published else "", mate])
    uob = pub["uob"]
    ms_band = (uob["molscribe"]["smiles"]["low"], uob["molscribe"]["smiles"]["high"])
    marks = [{"name": "InChIKey rule", "short": "InChIKey", "v": uob["molscribe"]["inchikey"]["value"], "cls": "s2", "up": True},
             {"name": "canonical SMILES rule", "short": "SMILES", "v": uob["molscribe"]["smiles"]["value"], "cls": "s1", "up": True},
             {"name": "published (Qian et al., 2023)", "short": "published", "v": PUBLISHED[("uob", "molscribe")], "cls": "s3", "up": False}]
    mg, ms = pub["biovista"]["molglyph"]["smiles"], pub["biovista"]["molscribe"]["smiles"]
    s_mg, s_ms = wilson(SYEDA_BIOVISTA["MolGlyph"], 100), wilson(SYEDA_BIOVISTA["MolScribe"], 100)
    note = ("Accuracy per image; canonical SMILES with its 95% Wilson interval. Published: MolScribe from Qian et al. (2023), "
            "MolSight from its README, BioVista from BioMiner's README (BioMiner's own matching rule). Teammates: Obaidullah "
            "(2026) on USPTO, UOB and CLEF with his own crop sets; Syeda (2026) on a 100-image BioVista sample.")
    body = ""
    if INCLUDE_COURSE:
        readers = d["course_readers_inchikey"]
        g, s_, p = d["readers_found_inchikey"]["molglyph_vs_molscribe"]
        left_vals = {k: pub["uspto"][k]["inchikey"]["value"] for k in READER_NAMES}
        right_vals = {k: readers[k]["recall"]["value"] for k in READER_NAMES}
        body += f"""
<div class=split>
<div>{figure(legend([("s1", "MolScribe"), ("s2", "MolSight"), ("s3", "MolGlyph")]) + both(lambda w, v: slope(left_vals, right_vals, w, v), 520),
             "Rank of each reader, best at the top.")}</div>
<div>
<p>On USPTO patent images MolScribe is the best of the three readers and MolGlyph the worst. On the course papers the
order reverses: MolGlyph finds the most molecules and MolScribe the fewest. Paired molecule by molecule, MolGlyph found
{g} that MolScribe missed and MolScribe found {s_} that MolGlyph missed (exact McNemar test, p = {p:.4f}).
Obaidullah's six-reader table shows the same pattern, with MolGlyph trailing on USPTO and leading on journal images.</p>
</div>
</div>"""
    body += f"""
<h3>One scoring choice moves a public score by nine points</h3>
<p>A tautomer is one of several forms of a molecule that differ only in where a hydrogen atom sits. The standard InChIKey
treats them as one molecule; canonical SMILES does not. On UOB, MolScribe scores {pc(uob['molscribe']['inchikey']['value'])}
by InChIKey and {pc(uob['molscribe']['smiles']['value'])} by canonical SMILES. Of the {t['n']} images that match only by
InChIKey, {t['tautomer']} differ only by tautomer, so the drawings and the labels probably show different forms. Under
canonical SMILES the published 87.9% falls inside our interval.</p>
{figure(legend([("s2", "InChIKey"), ("s1", "Canonical SMILES, with its 95% interval"), ("s3", "Published")]) + both(lambda w, v: number_line(marks, ms_band, w, v)),
        "MolScribe on the 5,740 UOB images.")}
<h3>Our numbers match the team's</h3>
<p>On the same 5,740 UOB images, Obaidullah measured 87.3% for MolScribe and 87.0% for MolGlyph (Obaidullah, 2026); we
measure {pc(uob['molscribe']['smiles']['value'])} and {pc(uob['molglyph']['smiles']['value'])}. On USPTO and CLEF his image sets
are larger than ours, and the numbers differ by up to seven points. On BioVista, Syeda's 100-image sample gave
MolScribe {SYEDA_BIOVISTA['MolScribe']}% and MolGlyph {SYEDA_BIOVISTA['MolGlyph']}% (Syeda, 2026). Our MolScribe score on all
{ms['n']:,} images, {pc(ms['value'])}, falls inside her interval ({100 * s_ms[0]:.1f} to {100 * s_ms[1]:.1f}%). Our MolGlyph
score, {pc(mg['value'])}, is above hers ({100 * s_mg[0]:.1f} to {100 * s_mg[1]:.1f}%), and the reason is not known yet.</p>
<details><summary>Every public set, every reader</summary>
{table(["Image set", "Reader", "InChIKey", "Canonical SMILES", "Published", "Teammates"], rows, note, numeric_from=2)}
</details>"""
    lead = ("Public image sets rank the readers differently from real papers, and the rule for matching molecules can move "
            "a public score by nine points." if INCLUDE_COURSE else
            "The rule for matching molecules can move a public score by nine points.")
    return finding("public", 5, "Public benchmarks do not predict real papers" if INCLUDE_COURSE else "How public scores depend on the rules",
                   lead, body, "Public crops are clean, single, already-cut drawings. They test reading, not finding or "
                   "table work, and our image sets are not identical to the published ones or to our teammates'.")


def how_section() -> str:
    return f"""
<section id=how class=alt><div class=wrap>
<div class=kicker>How it works</div><h2>Two pipelines, one answer key</h2>
<p class=lead>Each paper is rendered, its drawings are found and read, and the results are compared with a list of every
molecule the paper reports. BioMiner runs the same idea end to end, with a language model that links structures to labels.</p>
{figure(both(pipeline), "Both pipelines run on the same PDFs. The crop readers share one set of 237 drawings.", "wide-fig")}
<div class=cols3>
<div><h4>Scoring</h4><p>Drawings do not line up one to one with the answer key, so each paper's distinct predicted
molecules are compared with its distinct reference molecules as two sets. Recall is the share of reference molecules
found; precision is the share of predictions that are right.</p></div>
<div><h4>Uncertainty</h4><p>Every rate gets a 95% Wilson interval. Pooled rates also get a paper bootstrap, which
resamples the six papers and is wider, because molecules from one paper succeed or fail together.</p></div>
<div><h4>Data</h4><p>Six private course papers with 220 molecules; public crops from USPTO, UOB, CLEF and ACS; 8,861
BioVista crops and 12 BioVista papers. Matching uses InChIKey and canonical SMILES from RDKit.</p></div>
</div>
</div></section>"""


def timeline(d: dict) -> str:
    run = d["biominer_run"]
    events = [
        ("19 Oct", "First full run on a laptop", "The pipeline ran end to end on a MacBook CPU. The public check caught a "
         "MolScribe bug on macOS, and a patch fixed it that night."),
        ("20 Oct", "First results", "MolScribe found 51 of 219 course molecules, with intervals and a breakdown of what "
         "happened to each miss."),
        ("29 Oct", "More readers, more data", "MolSight and MolGlyph joined in their own environments. BioVista labels and "
         "the AWS setup scripts went in."),
        ("3 Nov", "Getting a GPU", "At least 98 attempts to start a GPU instance in us-east-2 were refused for lack of "
         "capacity before a g5.2xlarge came up. Its first run scored two readers at zero, and the cuDNN fix followed."),
        ("4 Nov", "The overnight run", "Three readers read 21,643 public images in 373 minutes. BioMiner was stopped by the "
         "free-tier Gemini quota."),
        ("4 Nov", "Keys out", "Gemini moved to Vertex AI. The instance now signs in to Google through workload identity "
         "federation, with no key stored on it."),
        ("5 Nov", "BioMiner end to end", f"Six course papers and 12 BioVista papers in about 70 minutes on one L4 GPU: "
         f"{run['calls']['answered']} Gemini calls, none lost."),
        ("5 Nov", "Tracing the misses", "Every missed molecule was sorted into a cause, with automatic tests and two AI judges."),
        ("5 Nov", "Published", "Code and report went public. All cloud resources were then deleted."),
    ]
    items = "".join(f"<li><div class=tl-date>{date}</div><div class=tl-body><h4>{title}</h4><p>{text}</p></div></li>"
                    for date, title, text in events)
    total = sum(v for *_, v in COSTS)
    cost_rows = [[name, basis, f"${v:.2f}"] for name, basis, v in COSTS] + [["Total", "", f"<strong>${total:.2f}</strong>"]]
    c = run["calls"]
    return f"""
<section id=timeline class=alt><div class=wrap>
<div class=kicker>How it got done</div><h2>Eighteen days, two GPU runs</h2>
<div class=split>
<div><ol class=timeline>{items}</ol></div>
<div class=side>
<h4>The GPU runs</h4>
{table(["Run", "Hardware", "What ran"], [["4&nbsp;Nov", "g5.2xlarge, NVIDIA A10G", "Three readers on 21,643 public images"],
                                          ["5&nbsp;Nov", "g6.2xlarge, NVIDIA L4", f"BioMiner on 18 papers, {c['answered']} Gemini calls"]], numeric_from=9)}
<h4>What it cost at list price</h4>
{figure(both(cost_bar) + legend([("s1", "Gemini, BioMiner"), ("s4", "Gemini, judge"), ("s2", "AWS GPUs"), ("s3", "AWS storage")]),
        f"About ${total:.0f} in all.")}
{table(["Item", "Basis", "Cost"], cost_rows, "AWS Cost Explorer records this account's usage at $0.00 (it is a member account of an "
       "AWS organization), so the AWS rows give the same usage at us-east-2 on-demand prices. Gemini was billed to a Google "
       "Cloud free-trial credit, at $1.50 per million input tokens and $9.00 per million output and thinking tokens.", numeric_from=2)}
</div>
</div>
</div></section>"""


def limitations() -> str:
    items = [
        ("Small sample", "Six papers, two of them with 1 and 4 molecules. Papers 4 and 19 hold 164 of the 219 molecules "
         "and dominate every pooled number." if INCLUDE_COURSE else "Six course papers only, and 12 BioVista papers."),
        ("Unchecked answer key", "The key lists each paper's molecules without saying where they appear, and it was not "
         "audited for this project. Five entries are themselves written with open positions."),
        ("Not BioMiner's own model", "BioMiner ran with Gemini 3.5 Flash instead of BioMiner-Instruct, on 12 BioVista "
         "papers instead of 450."),
        ("AI judges", "The recall-ceiling labels rest on automatic tests and two AI judges, one of which wrote the analysis "
         "code. No label was set by hand."),
        ("One run each", "Every stage ran once. MolScribe on the laptop and on the GPU gave identical SMILES, the only "
         "repeat. Detection has no ground-truth boxes."),
    ]
    cards = "".join(f"<div class=limit><h4>{t}</h4><p>{p}</p></div>" for t, p in items)
    return f'<section id=limits><div class=wrap><div class=kicker>Limits</div><h2>What these results cannot tell you</h2><div class=limits>{cards}</div></div></section>'


def team() -> str:
    people = [
        ("SS", "Shivansh Singh", "Built MolCensus: the pipeline, the GPU runs on AWS, the BioMiner and recall-ceiling "
         "analyses and this report."),
        ("MO", "Muhammad Obaidullah", "Benchmarked six readers on five public image sets and three whole-PDF systems, "
         "including BioMiner on 145 BioVista papers."),
        ("MS", "Maljha Fatima Syeda", "Ran the matched BioVista recognition study on 100 images and the error analysis "
         "of valid but wrong readings."),
    ]
    cards = "".join(f"<div class=person><div class=avatar>{ini}</div><div><h4>{name}</h4><p>{role}</p></div></div>"
                    for ini, name, role in people)
    tools = ["MolScribe", "MolSight", "MolGlyph", "MolDetV2", "BioMiner", "MinerU", "RDKit", "PyMuPDF", "Gemini 3.5 Flash",
             "Amazon EC2", "Google Cloud Vertex AI"]
    chips = "".join(f"<span class=chip>{t}</span>" for t in tools)
    refs = [
        "Guan, Y., et al. (2026). VERDICT: Agreement beats pixel-space verification in real-document OCSR [Preprint]. arXiv.",
        "Heller, S. R., et al. (2015). InChI, the IUPAC International Chemical Identifier. Journal of Cheminformatics, 7, 23.",
        "Obaidullah, M. (2026). Benchmarking existing tools for extracting SMILES from paper PDFs [Unpublished course report]. Northeastern University.",
        "Qian, Y., et al. (2023). MolScribe: Robust molecular structure recognition with image-to-graph generation. Journal of Chemical Information and Modeling, 63(7), 1925-1934.",
        "Syeda, M. F. (2026). Benchmarking optical chemical structure recognition models on the BioVista dataset [Unpublished course report]. Northeastern University.",
        "Syeda, M. F., and Obaidullah, M. (2026). Benchmarking molecular structure extraction from scientific PDFs: Group findings and Phase 2 plan [Course report]. Northeastern University.",
        "Yan, J., et al. (2026). BioMiner: A multi-modal system for automated mining of protein-ligand bioactivity data from literature [Preprint]. arXiv.",
    ]
    return f"""
<section id=team class=alt><div class=wrap>
<div class=kicker>People and tools</div><h2>Who did what</h2>
<p class=lead>MolCensus is a project for AAI 6630, Applied Computer Vision, at Northeastern University (fall 2026).
Teammates' numbers are credited wherever they appear.</p>
<div class=people>{cards}</div>
<h4>Built with</h4><div class=chips>{chips}</div>
<details><summary>References</summary><ul class=refs>{''.join(f'<li>{r}</li>' for r in refs)}</ul></details>
</div></section>"""


CSS = """
:root{--bg:#ffffff;--alt:#f6f7f9;--ink:#0f172a;--ink2:#475569;--muted:#64748b;--line:#e5e7eb;--grid:#eceef1;
--accent:#1c5cab;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;
--s1t:#eaf2fc;--s2t:#fdeee7;--s3t:#e6f6f0;--s4t:#fdf4e0}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;scroll-behavior:smooth;scroll-padding-top:64px}
body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.65 Inter,system-ui,-apple-system,"Segoe UI",sans-serif;
-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:2px}
.wrap{max-width:1080px;margin:0 auto;padding:0 20px}
nav.top{position:sticky;top:0;z-index:10;background:rgba(255,255,255,.92);backdrop-filter:saturate(180%) blur(8px);
border-bottom:1px solid var(--line)}
nav.top .wrap{display:flex;align-items:center;gap:22px;height:56px}
nav.top .brand{font-weight:800;color:var(--ink);text-decoration:none;letter-spacing:-.01em;margin-right:auto}
nav.top a.link{color:var(--ink2);text-decoration:none;font-size:.92rem}
nav.top a.link:hover{color:var(--ink)}
.btn{display:inline-flex;align-items:center;gap:8px;padding:9px 16px;border-radius:999px;font-weight:600;font-size:.93rem;
text-decoration:none;border:1px solid var(--line);color:var(--ink);background:#fff}
.btn.primary{background:var(--ink);color:#fff;border-color:var(--ink)}
@media (max-width:760px){nav.top a.link{display:none}}
header.hero{padding:72px 0 40px;background:radial-gradient(1100px 380px at 80% -10%,#eaf2fc 0%,rgba(255,255,255,0) 60%),
radial-gradient(800px 300px at 0% 0%,#fdf4e0 0%,rgba(255,255,255,0) 55%)}
.eyebrow{display:inline-block;font-size:.8rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--accent);
background:var(--s1t);padding:4px 10px;border-radius:999px;margin-bottom:18px}
h1{font-size:clamp(2.3rem,5vw,3.6rem);line-height:1.05;letter-spacing:-.03em;margin:0 0 18px;font-weight:800;max-width:16ch}
.hero p.sub{font-size:1.2rem;color:var(--ink2);max-width:60ch;margin:0 0 26px}
.hero .actions{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:40px}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0}
.tile{background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px 18px 16px;box-shadow:0 1px 2px rgba(15,23,42,.04)}
.tile-num{font-size:2rem;font-weight:800;letter-spacing:-.02em;line-height:1.1}
.tile-label{color:var(--ink2);font-size:.9rem;margin-top:6px;line-height:1.4}
.side .tiles{grid-template-columns:1fr 1fr 1fr}
@media (max-width:860px){.tiles,.side .tiles{grid-template-columns:1fr 1fr}}
section{padding:72px 0}
section.alt{background:var(--alt)}
.kicker{font-size:.8rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin-bottom:8px}
h2{font-size:clamp(1.6rem,3vw,2.2rem);line-height:1.15;letter-spacing:-.02em;margin:0 0 14px;font-weight:800;max-width:28ch}
h3{font-size:1.3rem;letter-spacing:-.01em;margin:48px 0 10px}
h4{font-size:1rem;margin:0 0 6px}
p{margin:0 0 16px;max-width:72ch}
.lead{font-size:1.15rem;color:var(--ink2);max-width:64ch;margin-bottom:28px}
.plain{background:var(--s1t);border:1px solid #cfe0f6;border-radius:12px;padding:12px 16px;max-width:none}
.split{display:grid;grid-template-columns:1fr 1fr;gap:36px;align-items:start}
@media (max-width:860px){.split{grid-template-columns:1fr}}
.cols3{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;margin-top:28px}
@media (max-width:760px){.cols3{grid-template-columns:1fr}}
figure{margin:20px 0;background:#fff;border:1px solid var(--line);border-radius:16px;padding:18px 18px 12px;
box-shadow:0 1px 2px rgba(15,23,42,.04)}
figcaption{font-size:.86rem;color:var(--muted);margin-top:8px}
.chart{display:block;width:100%;height:auto}
.chart.narrow{display:none}
@media (max-width:560px){.chart.wide{display:none}.chart.narrow{display:block}}
.waffle{max-width:420px;margin:10px auto 0}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.84rem;color:var(--ink2);margin:0 0 12px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px}
.chart text{font-family:Inter,system-ui,sans-serif;font-size:13px}
.chart.narrow text{font-size:14px}
.chart .small{font-size:12px}.chart .tiny{font-size:11px}.chart.narrow .small{font-size:13px}.chart.narrow .tiny{font-size:12px}
.chart .atom{font-size:17px;font-weight:600}
.t1{fill:var(--ink)}.t2{fill:var(--ink2)}.t3{fill:var(--muted)}.strong{font-weight:600}.num{font-variant-numeric:tabular-nums}
.s1{fill:var(--s1);background:var(--s1)}.s2{fill:var(--s2);background:var(--s2)}.s3{fill:var(--s3);background:var(--s3)}.s4{fill:var(--s4);background:var(--s4)}
.s1-tint{fill:var(--s1t)}.s2-tint{fill:var(--s2t)}.s3-tint{fill:var(--s3t)}.s4-tint{fill:var(--s4t)}
.s1-stroke{stroke:var(--s1)}.s2-stroke{stroke:var(--s2)}.s3-stroke{stroke:var(--s3)}
.card{fill:#fff;stroke:var(--line);stroke-width:1}
.ico{fill:none;stroke:var(--ink2);stroke-width:1.6;stroke-linejoin:round;stroke-linecap:round}
.arrow{fill:none;stroke:#94a3b8;stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round}
.bond{fill:none;stroke:var(--ink);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.bond-thin{fill:none;stroke:var(--ink);stroke-width:1.6}
.rule{stroke:var(--line);stroke-width:1}
.grid{stroke:var(--grid);stroke-width:1;fill:none}
.iso{fill:none;stroke:#cbd5e1;stroke-width:1;stroke-dasharray:3 4}
.ring{stroke:#fff;stroke-width:2}
.hollow{fill:#fff;stroke-width:2.5}
.ci-line{stroke-width:1.6;fill:none;opacity:.55}.ci-line.wide{stroke-width:6;opacity:.35;stroke-linecap:round}
.gap-line{stroke:#cbd5e1;stroke-width:4;stroke-linecap:round}
.zero{stroke:var(--ink2);stroke-width:1;stroke-dasharray:4 3}
.slope-line{stroke-width:3;fill:none;opacity:.85}
.axis-line{stroke:#94a3b8;stroke-width:1}
.leader{stroke:#cbd5e1;stroke-width:1}
.band{opacity:1}
.incidents{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:8px}
@media (max-width:960px){.incidents{grid-template-columns:1fr 1fr}}
@media (max-width:560px){.incidents{grid-template-columns:1fr}}
.incident{background:#fff;border:1px solid var(--line);border-radius:16px;padding:18px;display:flex;flex-direction:column;
box-shadow:0 1px 2px rgba(15,23,42,.04)}
.inc-when{font-size:.78rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.inc-num{font-size:2rem;font-weight:800;letter-spacing:-.02em;color:var(--s2);margin:6px 0 2px}
.inc-what{font-weight:600;font-size:.95rem;line-height:1.35;margin-bottom:10px}
.incident p{font-size:.9rem;color:var(--ink2);margin:0 0 14px;line-height:1.5}
.inc-fix{margin-top:auto;font-size:.88rem;background:var(--s3t);border-radius:10px;padding:10px 12px;line-height:1.45}
.inc-fix span,.caveat span{display:block;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--ink2);margin-bottom:2px}
.caveat{margin-top:28px;border-left:3px solid var(--s4);background:var(--s4t);border-radius:0 10px 10px 0;padding:12px 16px;
color:var(--ink2);font-size:.93rem;max-width:80ch}
.table-wrap{overflow-x:auto;margin:12px 0 18px;-webkit-overflow-scrolling:touch;background:#fff;border:1px solid var(--line);border-radius:12px}
table{border-collapse:collapse;width:100%;font-size:.9rem;font-variant-numeric:tabular-nums}
th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
tr:last-child td{border-bottom:0}
th{font-weight:600;color:var(--ink2);background:#fafafa;white-space:nowrap;font-size:.84rem}
td.num,th.num{text-align:right}
.ci{display:block;color:var(--muted);font-size:.8rem;white-space:nowrap}
.table-note{font-size:.84rem;color:var(--muted);margin:-6px 0 18px;max-width:none}
details{margin:18px 0;border:1px solid var(--line);border-radius:12px;background:#fff;padding:4px 16px}
details summary{cursor:pointer;font-weight:600;padding:10px 0;color:var(--ink)}
details[open]{padding-bottom:12px}
.timeline{list-style:none;margin:0;padding:0;position:relative}
.timeline:before{content:"";position:absolute;left:83px;top:6px;bottom:6px;width:2px;background:var(--line)}
.timeline li{display:grid;grid-template-columns:62px 1fr;gap:42px;padding:0 0 22px;position:relative}
.timeline li:after{content:"";position:absolute;left:76px;top:6px;width:10px;height:10px;border-radius:50%;background:#fff;border:3px solid var(--s1)}
.tl-date{white-space:nowrap}
.tl-date{font-size:.82rem;font-weight:700;color:var(--muted);text-align:right;padding-top:2px}
.tl-body h4{margin-bottom:2px}.tl-body p{font-size:.92rem;color:var(--ink2);margin:0}
.limits{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:10px}
@media (max-width:860px){.limits{grid-template-columns:1fr 1fr}}
@media (max-width:560px){.limits{grid-template-columns:1fr}}
.limit{border:1px solid var(--line);border-radius:14px;padding:16px 18px}
.limit p{font-size:.92rem;color:var(--ink2);margin:0}
.people{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:8px 0 28px}
@media (max-width:860px){.people{grid-template-columns:1fr}}
.person{display:flex;gap:14px;background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px}
.avatar{flex:none;width:44px;height:44px;border-radius:50%;background:var(--s1t);color:var(--accent);font-weight:700;
display:flex;align-items:center;justify-content:center}
.person p{font-size:.9rem;color:var(--ink2);margin:0}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 10px}
.chip{background:#fff;border:1px solid var(--line);border-radius:999px;padding:5px 12px;font-size:.86rem;color:var(--ink2)}
.refs{padding-left:18px;font-size:.88rem;color:var(--ink2)}.refs li{margin:6px 0}
code{font:.88em "JetBrains Mono",ui-monospace,Menlo,monospace;background:var(--alt);padding:.1em .35em;border-radius:5px}
footer{border-top:1px solid var(--line);padding:28px 0 40px;color:var(--muted);font-size:.86rem}
footer .wrap{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}
"""


def page(d: dict) -> str:
    course = INCLUDE_COURSE
    rc = d["recall_ceiling"]
    if course:
        found = d["readers_found_inchikey"]["by_paper"]
        n_found, n_all = sum(v["found"] for v in found.values()), sum(v["n"] for v in found.values())
        hero_tiles = tiles([("220", "distinct molecules in six real papers"),
                            (pc(n_found / n_all, 0), "of them found by three crop readers together"),
                            (f"{sum(b.get('3', 0) for b in rc['by_paper'].values())} / {rc['missed']}",
                             "missed molecules that exist only in R-group tables"),
                            (str(sum(d["biominer_course"]["only_biominer"].values())), "molecules only BioMiner found, by filling in tables")])
    else:
        hero_tiles = tiles([("21,643", "public images read by three readers"), ("12", "BioVista papers run through BioMiner"),
                            ("4", "silent failures caught by checks"), (f"${sum(v for *_, v in COSTS):.0f}", "of cloud compute at list price")])
    sections = [finding_checks(d)]
    if course:
        sections.append(finding_losses(d))
    sections.append(finding_biominer(d))
    if course:
        sections.append(finding_agreement(d))
    sections.append(finding_public(d))
    why = f"""
<section id=problem><div class=wrap>
<div class=split>
<div>
<div class=kicker>The problem</div><h2>Clean crops are not real papers</h2>
<p>Tools that read chemical drawings, called OCSR tools for optical chemical structure recognition, turn each drawing
into SMILES, a line of text that spells out a molecule's atoms and bonds. They are usually tested on single drawings
already cut out of patents, where the best of them are right more than nine times in ten.</p>
<p>Medicinal chemistry papers are harder. Molecules sit inside reaction schemes and figures, and a whole series of
compounds is often printed once, as a shared core with a table of substituents. MolCensus measures what that does to
the tools, paper by paper, against a list of every molecule each paper reports.</p>
<p class=table-note>The six course papers and their answer key are private, so this page reports totals per paper and
never shows a molecule, a drawing or a compound name. The drawing on the right is made up.</p>
</div>
<div>{figure(both(explainer, 520), "How one printed table turns into several compounds, and why a crop reader cannot see them.")}</div>
</div>
</div></section>"""
    return f"""<!doctype html>
<html lang=en>
<head>
<meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>MolCensus</title>
<meta name=description content="How well chemical structure recognition works on whole research papers: a benchmark of three crop readers and BioMiner.">
<link rel=preconnect href="https://fonts.googleapis.com"><link rel=preconnect href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel=stylesheet>
<style>{CSS}</style>
</head>
<body>
<nav class=top><div class=wrap>
<a class=brand href="#">MolCensus</a>
<a class=link href="#problem">Problem</a><a class=link href="#how">How it works</a><a class=link href="#checks">Findings</a>
<a class=link href="#timeline">Timeline</a><a class=link href="#team">Team</a>
<a class="btn primary" href="{REPO}">GitHub</a>
</div></nav>
<header class=hero><div class=wrap>
<span class=eyebrow>Benchmark · AAI 6630 · Fall 2026</span>
<h1>Reading molecules from whole papers</h1>
<p class=sub>MolCensus tests chemical structure recognition on real medicinal chemistry papers instead of clean single
images, and traces every molecule the tools miss to a cause.</p>
<div class=actions><a class="btn primary" href="#checks">See the findings</a><a class=btn href="{REPO}">Code on GitHub</a>
<a class=btn href="{REPO}/blob/main/docs/methods.md">Methods</a></div>
{hero_tiles}
</div></header>
{why}
{how_section()}
{''.join(sections)}
{timeline(d)}
{limitations()}
{team()}
<footer><div class=wrap><span>Results as of {REPORT_DATE}. Built by <code>scripts/report/build_site.py</code> from
<code>scripts/report/report_data.json</code>.</span><span>MIT license · <a href="{REPO}">{REPO.replace('https://', '')}</a></span></div></footer>
</body>
</html>
"""


SVG_STYLE = """<style>
text{font-family:-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;font-size:13px}
.small{font-size:12px}.tiny{font-size:11px}.atom{font-size:17px;font-weight:600}.strong{font-weight:600}
.t1{fill:#0f172a}.t2{fill:#475569}.t3{fill:#64748b}
.s1{fill:#2a78d6}.s2{fill:#eb6834}.s3{fill:#1baf7a}.s4{fill:#eda100}
.s1-tint{fill:#eaf2fc}.s2-tint{fill:#fdeee7}.s1-stroke{stroke:#2a78d6}.s2-stroke{stroke:#eb6834}.s3-stroke{stroke:#1baf7a}
.card{fill:#fff;stroke:#e5e7eb}.ico{fill:none;stroke:#475569;stroke-width:1.6;stroke-linejoin:round;stroke-linecap:round}
.arrow{fill:none;stroke:#94a3b8;stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round}
.grid{stroke:#eceef1;fill:none}.iso{fill:none;stroke:#cbd5e1;stroke-dasharray:3 4}.ring{stroke:#fff;stroke-width:2}
.hollow{fill:#fff;stroke-width:2.5}.ci-line{stroke-width:1.6;fill:none;opacity:.55}
</style><rect width="100%" height="100%" fill="#ffffff"/>"""


def standalone(markup: str) -> str:
    """One page figure as a standalone SVG file, with its styles inlined, for the README."""
    head, body = markup.split(">", 1)
    view = head.split('viewBox="')[1].split('"')[0].split()
    width, height = int(float(view[2])), int(float(view[3]))
    body = re.sub(r'class=([A-Za-z0-9_-]+)', r'class="\1"', body.rsplit("</svg>", 1)[0])  # XML needs quoted attributes
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">{SVG_STYLE}{body}</svg>\n'


def readme_figures(d: dict) -> None:
    images = ROOT / "docs" / "images"
    (images / "pipeline.svg").write_text(standalone(pipeline(760, "")))
    if not INCLUDE_COURSE:
        return
    rc, found = d["recall_ceiling"], d["readers_found_inchikey"]["by_paper"]
    totals = [("s1", "Found by a reader", sum(v["found"] for v in found.values())),
              ("s2", "Only in an R-group table", sum(b.get("3", 0) for b in rc["by_paper"].values())),
              ("s3", "Drawn, but misread", sum(b.get("1", 0) for b in rc["by_paper"].values())),
              ("s4", "Unresolved", sum(b.get("5", 0) for b in rc["by_paper"].values()))]
    legend_svg = "".join(f'<rect x="{12 + 220 * (i % 2)}" y="{8 + 20 * (i // 2)}" width="11" height="11" rx="3" class="{cls}"/>'
                         f'<text x="{28 + 220 * (i % 2)}" y="{18 + 20 * (i // 2)}" class="t2 small">{name} ({n})</text>'
                         for i, (cls, name, n) in enumerate(totals))
    (images / "where_molecules_are_lost.svg").write_text(_waffle_card(waffle(totals, columns=22), legend_svg))
    (images / "precision_recall.svg").write_text(standalone(pr_map(course_points(d), 760, "")))


def _waffle_card(grid: str, legend_svg: str) -> str:
    head, body = grid.split(">", 1)
    view = head.split('viewBox="')[1].split('"')[0].split()
    w, h = int(float(view[2])), int(float(view[3]))
    body = body.rsplit("</svg>", 1)[0]
    width, height = w + 24, h + 66
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">{SVG_STYLE}'
            f'{legend_svg}<g transform="translate(12,54)">{body}</g></svg>\n')


def main() -> None:
    data = json.loads(DATA.read_text())
    OUT.write_text(page(data))
    readme_figures(data)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes, course results {'included' if INCLUDE_COURSE else 'left out'})")


if __name__ == "__main__":
    main()
