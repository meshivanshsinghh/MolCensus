"""Build the public report, docs/index.html, from scripts/report/report_data.json.

    python scripts/report/collect.py      # once, on the machine with the private local results
    python scripts/report/build_site.py   # anywhere: needs only report_data.json

INCLUDE_COURSE (environment variable, default "true") controls the course-paper results. Course content
is per-paper totals only: no SMILES, crops, page images, compound names or ground truth. With
INCLUDE_COURSE=false the page keeps only the public-data results.

The page is one HTML file with inline CSS and inline SVG charts, so GitHub Pages serves it as is.
"""

from __future__ import annotations

import html
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "scripts" / "report" / "report_data.json"
OUT = ROOT / "docs" / "index.html"
INCLUDE_COURSE = os.environ.get("INCLUDE_COURSE", "true").strip().lower() == "true"
REPO = "https://github.com/meshivanshsinghh/molcensus"
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

READER_NAMES = {"molscribe": "MolScribe", "molsight": "MolSight", "molglyph": "MolGlyph"}
PAPERS = ["1", "2", "4", "6", "16", "19"]

# AWS spend, checked in Cost Explorer on 2026-11-05. The account's usage is recorded at $0.00, so these rows give the
# same usage at us-east-2 on-demand list prices: Cost Explorer's g5.2xlarge hours for 3 and 4 November (3.14 and 8.33),
# and the g6.2xlarge run of 5 November from its start (03:00:49 UTC) to its stop after the upload (about 05:48 UTC).
AWS_BILL: dict | None = {
    "rows": [
        ("AWS GPU instances, list price", 16.64,
         "11.5 hours of g5.2xlarge at $1.212 and about 2.8 hours of g6.2xlarge at $0.9776 an hour"),
        ("AWS disks, snapshots, IP address and S3, list price", 1.20,
         "a 150 GB gp3 disk for about 2.4 days, snapshot storage, public IPv4 hours and 1 GB in S3 (approximate)"),
    ],
    "note": ("AWS Cost Explorer records this account's usage at $0.00; it is a member account of an AWS organization. "
             "The AWS rows are what the same usage costs at on-demand list prices. The Gemini calls were billed to a "
             "Google Cloud free-trial credit. In all, the project used about $25 of cloud services at list price."),
}


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
    return f"{pc(r['value'])} <span class=ci>({interval(r)})</span>"


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def table(headers: list[str], rows: list[list], caption: str = "", numeric_from: int = 1, stack: bool = False) -> str:
    head = "".join(f"<th scope=col{' class=num' if i >= numeric_from else ''}>{h}</th>" for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f"<td{' class=num' if i >= numeric_from else ''}>{c}</td>" for i, c in enumerate(row)) + "</tr>"
                   for row in rows)
    note = f"<p class=table-note>{caption}</p>" if caption else ""
    return f'<div class="table-wrap{" stack" if stack else ""}"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{note}'


# ---------- charts (inline SVG, drawn twice: wide and narrow) ----------

def bar_path(x: float, y: float, w: float, h: float, r: float = 4.0, round_end: bool = True) -> str:
    """A bar from x to x + w with a rounded data end and a square baseline end."""
    if w <= 0:
        return ""
    r = min(r, w, h / 2) if round_end else 0
    if r <= 0:
        return f"M{x:.1f},{y:.1f}h{w:.1f}v{h:.1f}h{-w:.1f}z"
    return (f"M{x:.1f},{y:.1f}h{w - r:.1f}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r:.1f}"
            f"a{r},{r} 0 0 1 {-r},{r}h{-(w - r):.1f}z")


def svg(width: int, height: float, body: str, label: str, variant: str) -> str:
    return (f'<svg class="chart {variant}" viewBox="0 0 {width} {height:.0f}" role="img" aria-label="{esc(label)}" '
            f'xmlns="http://www.w3.org/2000/svg">{body}</svg>')


def legend(items: list[tuple[str, str]], x: float, y: float, width: int) -> tuple[str, float]:
    """Swatch-and-name legend that wraps; returns (svg, height used)."""
    out, cx, cy = [], x, y
    for cls, name in items:
        w = 22 + 7.2 * len(name)
        if cx + w > width and cx > x:
            cx, cy = x, cy + 20
        out.append(f'<rect x="{cx:.1f}" y="{cy - 9:.1f}" width="12" height="12" rx="3" class="{cls}"/>'
                   f'<text x="{cx + 17:.1f}" y="{cy + 1:.1f}" class="t2">{esc(name)}</text>')
        cx += w + 14
    return "".join(out), cy - y + 22


def hbars(rows: list[dict], width: int, x_max: float, ticks: list[float], fmt, label: str, variant: str,
          legend_items: list[tuple[str, str]] | None = None) -> str:
    """Horizontal bars from a zero baseline, optional 95% whiskers, the value at the tip.

    rows: {"label", "value", "cls"} plus optional "low", "high", "group" (a heading row), "tip" (hover text).
    Wide charts put labels left of the bars; narrow ones put them above.
    """
    narrow = width < 480
    longest = max((len(r["label"]) for r in rows if not r.get("group")), default=0)
    left = 12 if narrow else min(260, 22 + 7.0 * longest)
    right = width - 70
    scale = (right - left) / x_max
    parts, y = [], 8.0
    if legend_items:
        g, h = legend(legend_items, left if not narrow else 12, y + 12, width)
        parts.append(g)
        y += h + 6
    top = y
    bar_h = 14
    for row in rows:
        if row.get("group"):
            parts.append(f'<text x="12" y="{y + 16:.1f}" class="t1 strong">{esc(row["group"])}</text>')
            y += 24
            continue
        if narrow:
            parts.append(f'<text x="{left}" y="{y + 11:.1f}" class="t2">{esc(row["label"])}</text>')
            y += 16
        else:
            parts.append(f'<text x="{left - 10}" y="{y + 11:.1f}" class="t2" text-anchor="end">{esc(row["label"])}</text>')
        w = row["value"] * scale
        tip = esc(row.get("tip", f'{row["label"]}: {fmt(row["value"])}'))
        mark = f'<path d="{bar_path(left, y, w, bar_h)}" class="{row["cls"]}"/>'
        if row.get("low") is not None:
            lo, hi, cy = left + row["low"] * scale, left + row["high"] * scale, y + bar_h / 2
            mark += (f'<path d="M{lo:.1f},{cy:.1f}H{hi:.1f}M{lo:.1f},{cy - 4:.1f}V{cy + 4:.1f}M{hi:.1f},{cy - 4:.1f}V{cy + 4:.1f}" '
                     f'class="whisker"/>')
        tip_x = left + max(w, (row.get("high") or 0) * scale) + 6
        parts.append(f'<g><title>{tip}</title><rect x="{left}" y="{y - 3:.1f}" width="{right - left + 60}" height="{bar_h + 6}" '
                     f'class="hit"/>{mark}<text x="{tip_x:.1f}" y="{y + 11:.1f}" class="t1 num">{fmt(row["value"])}</text></g>')
        y += bar_h + (14 if narrow else 10)
    axis = [f'<path d="M{left},{top - 2:.1f}V{y - 4:.1f}" class="axis"/>']
    for t in ticks:
        x = left + t * scale
        axis.append(f'<path d="M{x:.1f},{top - 2:.1f}V{y - 4:.1f}" class="grid"/>'
                    f'<text x="{x:.1f}" y="{y + 10:.1f}" class="t3 num" text-anchor="middle">{fmt(t)}</text>')
    return svg(width, y + 18, "".join(axis) + "".join(parts), label, variant)


def stacked(rows: list[dict], segments: list[tuple[str, str, str]], width: int, label: str, variant: str) -> str:
    """Horizontal stacked bars of counts, one per row, with a 2px surface gap between segments."""
    narrow = width < 480
    left = 12 if narrow else 120
    right = width - 46
    x_max = max(sum(r[key] for key, _, _ in segments) for r in rows)
    scale = (right - left) / x_max
    g, h = legend([(cls, name) for _, cls, name in segments], 12 if narrow else left, 20, width)
    parts, y = [g], 20 + h + 4
    bar_h = 16
    for row in rows:
        if narrow:
            parts.append(f'<text x="{left}" y="{y + 11:.1f}" class="t2">{esc(row["label"])}</text>')
            y += 16
        else:
            parts.append(f'<text x="{left - 10}" y="{y + 12:.1f}" class="t2" text-anchor="end">{esc(row["label"])}</text>')
        x = left
        present = [(key, cls, name) for key, cls, name in segments if row[key]]
        for i, (key, cls, name) in enumerate(present):
            w = row[key] * scale
            last = i == len(present) - 1
            gap = 0 if last else 2
            parts.append(f'<g><title>{esc(row["label"])}: {name}, {row[key]}</title>'
                         f'<path d="{bar_path(x, y, max(w - gap, 0.8), bar_h, round_end=last)}" class="{cls}"/></g>')
            x += w
        total = sum(row[key] for key, _, _ in segments)
        parts.append(f'<text x="{x + 6:.1f}" y="{y + 12:.1f}" class="t1 num">{total}</text>')
        y += bar_h + (14 if narrow else 10)
    return svg(width, y + 4, "".join(parts), label, variant)


def chart(make) -> str:
    """The same chart drawn for a wide and a narrow screen; CSS shows one of them."""
    return make(640, "wide") + make(360, "narrow")


# ---------- the page ----------

def section(anchor: str, number: str, title: str, body: str) -> str:
    head = f'<span class=num-label>{number}</span> ' if number else ""
    return f'<section id="{anchor}"><h2>{head}{title}</h2>{body}</section>'


def figure(svg_markup: str, caption: str) -> str:
    return f"<figure>{svg_markup}<figcaption>{caption}</figcaption></figure>"


def finding_checks(d: dict) -> str:
    run = d["biominer_run"]
    calls = run["calls"]
    return section("checks", "1", "Installation checks catch silent failures", f"""
<p class=claim>The four failures below each ran to the end without an error message, and each was caught only because a
check compared the output with a known answer.</p>
<p>The first run on a Mac scored MolScribe at 34.8% on the ACS journal images, against 71.9% in its paper. MolScribe
returned no molecule for any drawing that uses an abbreviation such as Me or Ph. The step that expands an
abbreviation reads bond orders from bonds it has already deleted, which happens to fail on macOS, and MolScribe
hides the error. A three-line patch brought ACS accuracy to 73.1%.</p>
<p>The first GPU run on AWS gave MolScribe and MolSight a score of zero on every public image set. The machine image
put its own copy of cuDNN, NVIDIA's library of neural-network routines, on the library search path. PyTorch loaded
half of its cuDNN from its own package and half from that folder, and every convolution failed. The usual test,
<code>torch.cuda.is_available()</code>, still reported a working GPU. Setup now runs one real convolution in each
Python environment. After the fix, 200 USPTO patent images gave MolScribe 92.9% and MolSight 91.4%, against 92.6% and
91.2% in their papers.</p>
<p>MinerU, the PDF layout tool inside BioMiner, crashed on a missing model file and still exited with status 0, the
code for success. Its setup check now passes only if MinerU actually wrote its text output.</p>
<p>BioMiner sends parts of each paper to a large language model (LLM). It tries each call once and carries on with an
empty answer if the call fails, so a rate-limited run looks normal. In the first run, a Google AI Studio key that was
meant to be paid turned out to be held to free-tier limits of 5 requests a minute and 20 a day. Eight calls were
answered and 21 were refused for quota. Because the wrapper counts failed calls, the smoke test (one short paper run
before the real stages) failed at its 45-minute limit, and the BioMiner stages were skipped instead of being scored on
empty answers. The second run reached Gemini through Google Cloud Vertex AI, signed in from the AWS instance without
any stored key: {calls['answered']} calls were answered, {calls['retries']} were retried after a short rate limit, and
none were lost.</p>
<p class=caveat>These checks cover the failures we hit. A fault that still produces believable numbers on the public sets
would get through.</p>""")


def finding_losses(d: dict) -> str:
    rc, found = d["recall_ceiling"], d["readers_found_inchikey"]["by_paper"]
    rows = []
    for p in PAPERS:
        b = rc["by_paper"].get(p, {})
        rows.append({"label": f"Paper {p}", "found": found[p]["found"], "table": b.get("3", 0), "reading": b.get("1", 0),
                     "unresolved": b.get("5", 0), "n": found[p]["n"]})
    segments = [("found", "s1", "Found by a reader"), ("table", "s2", "Only in an R-group table"),
                ("reading", "s3", "Drawn but misread"), ("unresolved", "s4", "Unresolved")]
    total_found = sum(r["found"] for r in rows)
    n_total = sum(r["n"] for r in rows)
    j = rc["judges"]
    strict = rc["strict"]
    table_rows = [[r["label"].replace("Paper ", ""), r["n"], r["found"], r["table"], r["reading"], r["unresolved"]] for r in rows]
    table_rows.append(["All", n_total, total_found, sum(r["table"] for r in rows), sum(r["reading"] for r in rows),
                       sum(r["unresolved"] for r in rows)])
    return section("losses", "2", "Where molecules are lost", f"""
<p class=claim>On the course papers the crop readers are close to the most they can find one drawing at a time: 138 of
the 146 molecules they missed exist only as rows of R-group tables.</p>
<p>An R-group table is a drawing of a shared core with labelled open positions (R<sub>1</sub>, R<sub>2</sub>), plus a
table that says what sits at each position for each compound. The compound itself is never drawn. Three readers,
MolScribe, MolSight and MolGlyph, read the same 237 drawings that the MolDetV2 detector cut out of the six papers.
Scored by InChIKey (a fixed-length code computed from a molecule, so that two ways of writing the same molecule match),
they found {total_found} of the {n_total} distinct molecules between them, or {pc(total_found / n_total)}.</p>
<p>Each of the {rc['missed']} missed molecules was sorted into one cause. Automatic tests came first: a second detection
pass at a much lower threshold, an exact substructure test of every reading with open positions against each missed
molecule, and a similarity test for near misses. The remaining molecules went to two AI judges, Claude and Gemini 3.5
Flash, who saw the same page images and instructions. Before their answers counted, both were tested blind on the
{j['claude'][1]} molecules whose answer was already known. Claude got {j['claude'][0]} right ({pc(j['claude'][0] / j['claude'][1])})
and Gemini {j['gemini'][0]} ({pc(j['gemini'][0] / j['gemini'][1])}). Their agreement over all 219 molecules was a Cohen's
kappa of {rc['kappa_all']:.3f}; kappa measures agreement beyond chance, with 1 for perfect agreement and 0 for chance. A
judge label was used only when both judges gave it.</p>
{figure(chart(lambda w, v: stacked(rows, segments, w, "Course molecules per paper by what happened to them", v)),
        "Distinct molecules per paper (InChIKey), by what happened to them. Paper 19 holds 104 of the 219.")}
{table(["Paper", "Molecules", "Found by a reader", "Only in an R-group table", "Drawn but misread", "Unresolved"], table_rows)}
<p>No missed molecule turned out to be drawn in full but missed by the detector, and none was absent from the figures.
The two reading losses are both in paper 16, whose large ring molecules are drawn under a colored heatmap. One rule
withdrew an automatic label when both judges contradicted it; it fired five times. Under the strict rule, without that
withdrawal, the totals are {strict.get('3', 0)} table losses, {strict.get('1', 0)} reading losses and
{strict.get('5', 0)} unresolved.</p>
<p>So between 75 and 81 of the 219 molecules are drawn in full somewhere in the papers, a recall ceiling of 34.2% to
37.0% for any reader that works one drawing at a time. Better reading could add at most 2 to 8 molecules. Finding more
needs a tool that reads tables.</p>
<p class=caveat>The automatic table test treats every symbol a reader could not expand as an open position, so a drawn
molecule with an unusual abbreviation can pass for a table compound; the judges caught four such cases. The judges are
not independent of the analysis: Claude wrote the evidence code. Gemini's answers vary from run to run, and the cached
answers are the ones used. The second detection pass found no missed drawing, which is evidence, not proof, that
detection loses nothing.</p>""")


def finding_biominer(d: dict) -> str:
    bv = d["biominer_biovista"]
    by_type = bv["by_type"]
    order = [("full", "Full structures"), ("chiral", "Chiral structures"), ("R-group", "R-group tables"), ("all", "All structures")]
    bv_rows = []
    for key, name in order:
        e = by_type[key]
        bv_rows += [{"group": name},
                    {"label": "Linked", "value": e["linked"]["mean_recall"], "cls": "s1"},
                    {"label": "All extracted", "value": e["no_star"]["mean_recall"], "cls": "s2"},
                    {"label": "Published", "value": e["published_recall"], "cls": "s3",
                     "tip": f"{name}: published BioMiner-Instruct, {pc(e['published_recall'])}"}]
    bv_table = []
    for key, name in order:
        e = by_type[key]
        gap = e["published_recall"] - e["linked"]["mean_recall"]
        closed = (e["no_star"]["mean_recall"] - e["linked"]["mean_recall"]) / gap if gap > 0 else None
        bv_table.append([name, e["linked"]["recall"]["n"], pc(e["linked"]["mean_recall"]), pc(e["no_star"]["mean_recall"]),
                         pc(e["published_recall"]), pc(closed, 0) if closed is not None else "n/a"])
    full = by_type["full"]
    rgroup = by_type["R-group"]
    allx = by_type["all"]
    biovista = f"""
<h3>BioVista: finding a structure vs linking it to a label</h3>
<p>BioVista (Yan et al., 2026) is BioMiner's own benchmark of papers with every structure labelled. On 12 of its papers,
the structures MolGlyph read inside BioMiner were scored by type, once as linked to a compound label and once counting
every whole structure read. The published figures are means over papers, so these are too.</p>
{figure(chart(lambda w, v: hbars(bv_rows, w, 1.0, [0, 0.25, 0.5, 0.75, 1.0], lambda x: pc(x, 0) if x in (0, 0.25, 0.5, 0.75, 1.0) else pc(x),
                                  "BioVista recall by structure type", v,
                                  [("s1", "Linked"), ("s2", "All extracted, without * pieces"), ("s3", "Published, BioMiner-Instruct")])),
        "Mean recall over 12 BioVista papers by structure type. Published values average over 450 papers with BioMiner-Instruct.")}
{table(["Structure type", "Labels", "Linked", "All extracted", "Published", "Gap closed"], bv_table, numeric_from=1)}
<p>For full structures, counting what MolGlyph read but Gemini did not link closes most of the gap to the published
result: mean recall rises from {pc(full['linked']['mean_recall'])} to {pc(full['no_star']['mean_recall'])}, against
{pc(full['published_recall'])}. Every structure BioMiner links as a full structure is one of these readings, so no
linking step could do better than this count with the same detector and reader. For R-group tables almost nothing
changes ({pc(rgroup['linked']['mean_recall'])} to {pc(rgroup['no_star']['mean_recall'])}, against
{pc(rgroup['published_recall'])}). Those molecules exist only after the table is filled in, which is itself a step done
by the language model. Precision over all structures stays near half ({pc(allx['linked']['precision']['value'])} linked,
{pc(allx['no_star']['precision']['value'])} counting everything read), against {pc(allx['published_precision'])}
published. On 145 BioVista papers, Obaidullah's BioMiner run reached {OBAIDULLAH_BIOMINER['biovista']['recall']}% recall at
{OBAIDULLAH_BIOMINER['biovista']['precision']}% precision, close to the published result (Obaidullah, 2026), and he found
{OBAIDULLAH_BIOMINER['biovista_full_vs_table'][0]}% recall for fully drawn compounds against
{OBAIDULLAH_BIOMINER['biovista_full_vs_table'][1]}% for table-defined ones.</p>"""
    caveat = f"""
<p class=caveat>These are 12 papers, not 450, and they were not chosen to match BioVista's test set. BioMiner's own
evaluator, run on the same 12 papers with its matching rule, gives linked means within half a point of ours, so the
matching rule does not explain the gap. A run with BioMiner-Instruct on the same papers would separate the model's
share from the sample's.</p>"""
    intro = """
<p class=plain>BioMiner ran with Gemini 3.5 Flash on Google Cloud Vertex AI, not with its own fine-tuned model,
BioMiner-Instruct, and on 12 BioVista papers rather than the 450 its authors used.</p>
<p>BioMiner (Yan et al., 2026) processes a whole PDF. MinerU finds the page layout, MolDetV2 finds the drawings, MolGlyph
reads them, and a language model ties each structure to its compound label and fills in R-group tables.</p>"""
    if not INCLUDE_COURSE:
        return section("biominer", "3", "BioMiner's R-group step", intro + biovista + caveat)

    c = d["biominer_course"]
    s = c["scores"]
    names = [("no_star", "All extracted, without * pieces", "s1"), ("linked", "Linked", "s1"), ("all_extracted", "All extracted", "s1"),
             ("molglyph", "MolGlyph", "s2"), ("molsight", "MolSight", "s2"), ("molscribe", "MolScribe", "s2")]

    def rows_for(metric: str) -> list[dict]:
        return [{"label": name, "value": s[k]["ALL"][metric]["value"], "low": s[k]["ALL"][metric]["low"],
                 "high": s[k]["ALL"][metric]["high"], "cls": cls,
                 "tip": f"{name}: {metric} {pc(s[k]['ALL'][metric]['value'])} ({interval(s[k]['ALL'][metric])})"}
                for k, name, cls in names]

    def course_chart(metric: str, title: str) -> str:
        return chart(lambda w, v: hbars(rows_for(metric), w, 1.0, [0, 0.5, 1.0], lambda x: pc(x, 0) if x in (0, 0.5, 1.0) else pc(x),
                                        f"Course papers: {title}", v, [("s1", "BioMiner"), ("s2", "Crop readers")]))

    course_table = [[name, with_ci(s[k]["ALL"]["recall"]), with_ci(s[k]["ALL"]["precision"]), pc(s[k]["ALL"]["f1"]),
                     s[k]["ALL"]["precision"]["n"]] for k, name, _ in names]
    ob = OBAIDULLAH_BIOMINER["course"]
    course_table.append(["Obaidullah's BioMiner run", f"{ob['recall']}%", f"{ob['precision']}%", f"{ob['f1']}%", "about 105"])
    head = s["no_star"]["ALL"]
    only = c["only_biominer"]
    counts = c["counts"]
    course = f"""
<p class=claim>BioMiner fills in R-group tables, so it finds molecules that no crop reader can. Counting every whole
structure it reads gives the best balance of recall and precision on the course papers.</p>
<p>What counts as a BioMiner prediction matters, and three rules are reported. Linked structures are what BioMiner itself
keeps: the structures its language model tied to a compound label. On the course papers that is {counts['pipeline_rows']}
structures. All extracted structures adds every structure MolGlyph read inside BioMiner, linked or not
({counts['unlinked']} more). The headline rule drops from those the {counts['unlinked_star']} pieces with open positions
(written with <code>*</code> atoms), which are table scaffolds and substituent fragments rather than molecules. Matching
here is by canonical isomeric SMILES (RDKit's standard spelling of a molecule, including its stereochemistry marks), over 220 distinct
molecules.</p>
{figure(course_chart('recall', 'recall'), "Recall on the six course papers, with 95% Wilson intervals.")}
{figure(course_chart('precision', 'precision'), "Precision on the six course papers, with 95% Wilson intervals.")}
{table(["Prediction rule", "Recall", "Precision", "F1", "Predictions"], course_table)}
<p>Under the headline rule BioMiner finds {head['recall']['k']} of 220 molecules ({pc(head['recall']['value'])}) with
{pc(head['precision']['value'])} of its predictions correct. The best crop reader, MolGlyph, finds
{s['molglyph']['ALL']['recall']['k']} at {pc(s['molglyph']['ALL']['precision']['value'])} precision. Obaidullah's BioMiner run
on the same six papers reached {ob['recall']}% recall at {ob['precision']}% precision (Obaidullah, 2026). His figures imply
about 105 predictions and match ours when every structure read is counted, rather than the linked ones alone; his
model has not been confirmed and is probably BioMiner-Instruct. Dropping the pieces with open positions also drops two correct molecules
from paper 19 whose answer key itself writes a generic group as an open position, which is why the headline rule finds
75 molecules and the all-extracted rule 77.</p>
<p>BioMiner found {sum(only.values())} molecules that none of the three crop readers found, {only['4']} in paper 4 and
{only['19']} in paper 19. All {c['only_biominer_from_rgroup']} came from filling in R-group tables. That is BioMiner's own
contribution over a crop reader. Recall on those two papers is still low, 38.3% and 25.0% at best, because most of
their table compounds were not filled in.</p>"""
    return section("biominer", "3", "BioMiner's R-group step", intro + course + biovista + caveat)


def finding_agreement(d: dict) -> str:
    a = d["agreement"]
    sel, diff = a["selections"], a["differences"]
    groups = [("All three readers agree", "all_three", "top_conf_a"), ("At least two readers agree", "two_agree", "top_conf_b")]
    rows = []
    for name, x, y in groups:
        rows += [{"group": f"{name} ({sel[x]['crops']} drawings)"},
                 {"label": "Agreement", "value": sel[x]["precision"]["value"], "low": sel[x]["precision"]["low"],
                  "high": sel[x]["precision"]["high"], "cls": "s1"},
                 {"label": "Top confidence", "value": sel[y]["precision"]["value"], "low": sel[y]["precision"]["low"],
                  "high": sel[y]["precision"]["high"], "cls": "s2"}]
    t_rows = []
    for name, x, y in groups:
        for label, k in [(name, x), (f"MolScribe's top {sel[y]['crops']} by confidence", y)]:
            t_rows.append([label, sel[k]["crops"], with_ci(sel[k]["precision"]), f"{sel[k]['precision']['k']} of {sel[k]['precision']['n']}",
                           pc(sel[k]["recall"]["value"])])
    d1, d2 = diff["a_minus_c"], diff["b_minus_c"]
    ahead = sum(1 for p in PAPERS if sel["top_conf_a"]["per_paper"][p][1] and sel["all_three"]["per_paper"][p][1]
                and sel["all_three"]["per_paper"][p][0] / sel["all_three"]["per_paper"][p][1]
                > sel["top_conf_a"]["per_paper"][p][0] / sel["top_conf_a"]["per_paper"][p][1])
    both = sum(1 for p in PAPERS if sel["top_conf_a"]["per_paper"][p][1] and sel["all_three"]["per_paper"][p][1])
    ob = OBAIDULLAH_READERS
    ob_rows = [[r] + [f"{v:.1f}" for v in ob[r]] for r in ["MolScribe", "MolNexTR", "DECIMER", "MolVec", "MolGlyph", "OCSRGlyph"]]

    def signed(x: float) -> str:
        return ("+" if x >= 0 else "−") + f"{abs(100 * x):.1f}"

    return section("agreement", "4", "Reader agreement vs confidence", f"""
<p class=claim>When all three readers write the same molecule for a drawing, it is right more often than MolScribe's most
confident readings, but six papers are not enough to show that the difference is real.</p>
<p>A reader's confidence is its own estimate that a reading is right; of the three readers only MolScribe reports one.
All three readers agreed on {sel['all_three']['crops']} of the 237 drawings. Of the distinct molecules from those drawings,
{pc(sel['all_three']['precision']['value'])} were correct ({sel['all_three']['precision']['k']} of
{sel['all_three']['precision']['n']}). MolScribe's {sel['top_conf_a']['crops']} most confident drawings gave
{pc(sel['top_conf_a']['precision']['value'])}. With at least two readers agreeing ({sel['two_agree']['crops']} drawings) the
figure was {pc(sel['two_agree']['precision']['value'])}, against {pc(sel['top_conf_b']['precision']['value'])} for
MolScribe's top {sel['top_conf_b']['crops']}.</p>
{figure(chart(lambda w, v: hbars(rows, w, 1.0, [0, 0.5, 1.0], lambda x: pc(x, 0) if x in (0, 0.5, 1.0) else pc(x),
                                  "Precision of agreement vs confidence", v, [("s1", "Reader agreement"), ("s2", "MolScribe's top confidence, same number of drawings")])),
        "Share of distinct molecules that are correct, with 95% Wilson intervals (InChIKey rule).")}
{table(["Selection", "Drawings", "Precision", "Correct", "Recall"], t_rows)}
<p>Molecules from the same paper succeed or fail together, so the test resamples whole papers rather than molecules.
Over 10,000 draws of the six papers with replacement, using the same draws for both selections, the gap in precision
for the {sel['all_three']['crops']}-drawing comparison is {signed(d1['estimate'])} points, with a 95% interval from
{signed(d1['low'])} to {signed(d1['high'])}; {pc(d1['share_le_0'])} of the draws favor confidence or tie. For the
{sel['two_agree']['crops']}-drawing comparison it is {signed(d2['estimate'])} points ({signed(d2['low'])} to
{signed(d2['high'])}). Agreement leads in {ahead} of the {both} papers where both selections have predictions.</p>
<p>Obaidullah's benchmark of six readers on five public image sets shows why agreement could help (Obaidullah, 2026): the
best reader changes with the source of the images. OCSRGlyph was best on USPTO patents and on CLEF, and fourth of six on
MolRecBench-Wild journal images, where MolGlyph led. Readers that fail on different images are what an agreement filter
needs. The VERDICT preprint reports the same direction on real documents (Guan et al., 2026).</p>
{table(["Reader"] + ob["datasets"], ob_rows, "Obaidullah (2026): exact-match accuracy (%) on cropped public images, his crop counts")}
<p class=caveat>The direction is consistent, but both intervals include zero. More papers, and more readers such as the six
in Obaidullah's benchmark, are what it would take to settle it.</p>""")


def finding_public(d: dict) -> str:
    pub = d["public"]
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
    bv_rule = pub["biovista"]
    caption = ("Accuracy per image, with 95% Wilson intervals under canonical SMILES. USPTO holds patent drawings, UOB and "
               "CLEF are older OCSR test sets, ACS holds journal figures and BioVista crops come from BioMiner's papers. Published: MolScribe from Qian et al. (2023), MolSight from its README, BioVista from "
               f"BioMiner's README, which used BioMiner's own matching rule (under it we measure "
               f"{pc(bv_rule['molscribe']['biominer_rule']['value'])} for MolScribe and {pc(bv_rule['molglyph']['biominer_rule']['value'])} "
               "for MolGlyph). Teammates: Obaidullah (2026) on USPTO, UOB and CLEF, with his own crop sets of 5,704, 5,740 and 977 "
               "images; Syeda (2026) on a 100-image BioVista sample.")
    t = d["tautomers_uob_molscribe"]
    uob = pub["uob"]
    mg, ms = pub["biovista"]["molglyph"]["smiles"], pub["biovista"]["molscribe"]["smiles"]
    s_mg, s_ms = wilson(SYEDA_BIOVISTA["MolGlyph"], 100), wilson(SYEDA_BIOVISTA["MolScribe"], 100)
    body = f"""
<p class=claim>Scores on public image sets do not predict which reader does best on real papers, and one scoring choice
about tautomers moves a public score by nine points.</p>
<p>Public sets show one molecule per image with a known answer, so accuracy is simply the share of images read
correctly. The table scores each reader by InChIKey and by canonical isomeric SMILES, next to the published value and to
our teammates' runs.</p>
{table(["Image set", "Reader", "InChIKey", "Canonical SMILES", "Published", "Teammates"], rows, caption, numeric_from=2, stack=True)}
<p>A tautomer is one of several forms of a molecule that differ only in where a hydrogen atom sits and that turn into
each other easily. The standard InChIKey treats them as one molecule; canonical SMILES does not. On the UOB set MolScribe
scores {pc(uob['molscribe']['inchikey']['value'])} by InChIKey and {pc(uob['molscribe']['smiles']['value'])} by canonical
SMILES. Of the {t['n']} images that match only by InChIKey, {t['tautomer']} differ only by tautomer, {t['stereo']} by
stereo marks, {t['charge']} by charge and {t['other']} in some other way, so the drawing and the label probably show
different tautomers. Under canonical SMILES MolScribe's interval includes its published 87.9%.</p>
<p>The same rule reproduces our teammates' numbers. On the same 5,740 UOB images, Obaidullah measured 87.3% for MolScribe and
87.0% for MolGlyph (Obaidullah, 2026); we measure {pc(uob['molscribe']['smiles']['value'])} and
{pc(uob['molglyph']['smiles']['value'])}. On USPTO and CLEF his image sets are larger than ours (5,704 and 977 crops
against 5,633 and 904), and the numbers differ by up to seven points. On BioVista, Syeda's sample of 100 images gave
MolScribe {SYEDA_BIOVISTA['MolScribe']}% and MolGlyph {SYEDA_BIOVISTA['MolGlyph']}% (Syeda, 2026). Our MolScribe score on all
{ms['n']:,} images, {pc(ms['value'])}, is inside the 95% interval of her sample ({100 * s_ms[0]:.1f} to {100 * s_ms[1]:.1f}%).
Our MolGlyph score, {pc(mg['value'])}, is above hers ({100 * s_mg[0]:.1f} to {100 * s_mg[1]:.1f}%), and the reason has not been
found.</p>"""
    if INCLUDE_COURSE:
        readers = d["course_readers_inchikey"]
        g, s_, p = d["readers_found_inchikey"]["molglyph_vs_molscribe"]
        colors = {"molscribe": "s1", "molsight": "s2", "molglyph": "s3"}
        uspto_rows = [{"label": READER_NAMES[t], "value": pub["uspto"][t]["inchikey"]["value"], "cls": colors[t]} for t in colors]
        course_rows = [{"label": READER_NAMES[t], "value": readers[t]["recall"]["value"], "low": readers[t]["recall"]["low"],
                        "high": readers[t]["recall"]["high"], "cls": colors[t]} for t in colors]
        fmt = lambda x: pc(x, 0) if x in (0, 0.5, 1.0) else pc(x)  # noqa: E731
        body += f"""
<h3>The ranking flips on real papers</h3>
<p>On USPTO patent images MolScribe is the best of the three readers ({pc(pub['uspto']['molscribe']['inchikey']['value'])})
and MolGlyph the worst ({pc(pub['uspto']['molglyph']['inchikey']['value'])}). On the course papers the order reverses:
MolGlyph finds the most molecules ({pc(readers['molglyph']['recall']['value'])}) and MolScribe the fewest
({pc(readers['molscribe']['recall']['value'])}). Paired molecule by molecule, MolGlyph found {g} molecules that MolScribe
missed and MolScribe found {s_} that MolGlyph missed (exact McNemar test, p = {p:.4f}). Obaidullah's six-reader table
shows the same pattern: MolGlyph trails on USPTO but leads on MolRecBench-Wild journal images.</p>
<div class=pair>
{figure(chart(lambda w, v: hbars(uspto_rows, w if w < 480 else 320, 1.0, [0, 0.5, 1.0], fmt, "USPTO accuracy by reader", "narrow" if w < 480 else "wide")),
        "USPTO patent images: accuracy (InChIKey).")}
{figure(chart(lambda w, v: hbars(course_rows, w if w < 480 else 320, 1.0, [0, 0.5, 1.0], fmt, "Course-paper recall by reader", "narrow" if w < 480 else "wide")),
        "Course papers: recall (InChIKey), with 95% Wilson intervals.")}
</div>"""
    body += """
<p class=caveat>Public crops are clean, single, already-cut drawings. They test reading, not finding or table work, and our
image sets are not identical to the published ones or to our teammates'.</p>"""
    return section("public", "5", "Public benchmarks vs real papers", body)


def methods(d: dict) -> str:
    return section("methods", "", "Methods", f"""
<p>The pipeline renders each PDF page at 300 dpi with PyMuPDF, finds molecule drawings with MolDetV2 (a YOLO detector,
confidence 0.4), and passes each cut-out drawing, called a crop, to a reader that writes it as a SMILES string. SMILES is
a line of text that spells out a molecule's atoms and bonds. RDKit turns each string into a molecule or rejects it as
invalid. BioMiner runs separately on whole PDFs.</p>
<p>Scoring is per paper. Crops do not line up one to one with the answer key, so each paper's distinct predicted molecules
are compared with its distinct reference molecules as two sets. Recall is the share of reference molecules found;
precision is the share of distinct valid predictions that are reference molecules. Two matching rules are used: the
full InChIKey, and the canonical isomeric SMILES. Each rate gets a 95% Wilson interval, which treats molecules as
independent, and pooled rates also get a paper bootstrap interval, which resamples the six papers 2,000 times and is
wider because molecules from one paper succeed or fail together. Paired comparisons between two tools use the exact
McNemar test, a binomial test on the molecules only one of the two found.</p>
<p>The public image sets are USPTO, UOB, CLEF and ACS from MolScribe's evaluation archive, and the 8,861 BioVista crops,
scored one image at a time. The recall-ceiling labels and the BioMiner prediction rules are described in the
findings above. Details, file formats and every threshold are in <a href="{REPO}/blob/main/docs/methods.md">docs/methods.md</a>.</p>""")


def run_facts(d: dict) -> str:
    run = d["biominer_run"]
    c = run["calls"]
    stages = {s["stage"]: s for s in run["stages"]}
    rows = [
        ["4 November 2026", "g5.2xlarge, one NVIDIA A10G", "Crop readers on 21,643 public images; course crops", "373 minutes for the public sets",
         "None (BioMiner skipped)"],
        ["5 November 2026", "g6.2xlarge, one NVIDIA L4", "BioMiner on 6 course and 12 BioVista papers",
         f"{stages['biominer_course']['minutes']} and {stages['biominer_biovista']['minutes']} minutes", f"{c['answered']} answered, {c['lost']} lost"],
    ]
    costs = [["Gemini 3.5 Flash for BioMiner, list price",
              f"{c['input']:,} input, {c['output']:,} output and {run['thinking']:,} thinking tokens", f"${run['cost_usd']:.2f}"],
             ["Gemini 3.5 Flash for the recall-ceiling judge, list price",
              "25 calls, 440,202 input and 15,762 output tokens; its thinking tokens were not recorded, so the real cost is higher", "$0.80"]]
    if AWS_BILL:
        costs += [[name, note, f"${value:.2f}"] for name, value, note in AWS_BILL["rows"]]
    note = (f"<p>{AWS_BILL['note']}</p>" if AWS_BILL else "")
    return section("runs", "", "Run facts and costs", f"""
<p>All GPU work ran on Amazon EC2 in us-east-2. The crop readers and the course detection also ran once on a Mac CPU;
MolScribe gave identical SMILES on all 237 course crops on both machines.</p>
{table(["Run", "Instance", "What ran", "Time", "Gemini calls"], rows, numeric_from=9)}
<p>List prices for Gemini 3.5 Flash are $1.50 per million input tokens and $9.00 per million output and thinking
tokens; thinking tokens are the model's own reasoning, billed as output.</p>
{table(["Item", "Basis", "Cost"], costs, numeric_from=2)}
{note}""")


def limitations() -> str:
    return section("limits", "", "Limitations", """
<p>Six papers is a small sample. Two of them hold 1 and 4 molecules, so their per-paper scores swing widely, and the
paper bootstrap intervals are wide because the papers differ so much. Papers 4 and 19 hold 164 of the 219 molecules and
dominate every pooled number.</p>
<p>The answer key lists each paper's molecules without saying where they appear, and nobody checked it against the papers
for this project. Five of its entries are themselves written with open positions.</p>
<p>BioMiner ran with Gemini 3.5 Flash, not BioMiner-Instruct, and on 12 BioVista papers instead of 450. Its BioVista
numbers say how this setup does on those papers, not how BioMiner does in general.</p>
<p>The recall-ceiling labels rest on automatic tests and on two AI judges, one of which is the assistant that wrote the
analysis code. No label was set by hand.</p>
<p>Every stage ran once. The only repeat is MolScribe on the Mac CPU and on the GPU, which gave the same SMILES. There is no
ground truth for detection boxes, so detection is checked only indirectly.</p>""")


def credits() -> str:
    return section("credits", "", "Credits and sources", f"""
<p>MolCensus was built by Shivansh Singh for AAI 6630, Applied Computer Vision, at Northeastern University (fall 2026).
Teammates Muhammad Obaidullah and Maljha Fatima Syeda ran the multi-reader, whole-PDF and BioVista benchmarks whose
numbers appear above; they are credited where they appear. The code and the full write-ups are at
<a href="{REPO}">{REPO.replace('https://', '')}</a>.</p>
<ul class=refs>
<li>Guan, Y., et al. (2026). VERDICT: Agreement beats pixel-space verification in real-document OCSR [Preprint]. arXiv.</li>
<li>Heller, S. R., et al. (2015). InChI, the IUPAC International Chemical Identifier. Journal of Cheminformatics, 7, 23.</li>
<li>Obaidullah, M. (2026). Benchmarking existing tools for extracting SMILES from paper PDFs [Unpublished course report].
Northeastern University.</li>
<li>Qian, Y., et al. (2023). MolScribe: Robust molecular structure recognition with image-to-graph generation. Journal of
Chemical Information and Modeling, 63(7), 1925-1934.</li>
<li>Syeda, M. F. (2026). Benchmarking optical chemical structure recognition models on the BioVista dataset [Unpublished
course report]. Northeastern University.</li>
<li>Syeda, M. F., and Obaidullah, M. (2026). Benchmarking molecular structure extraction from scientific PDFs: Group
findings and Phase 2 plan [Course report]. Northeastern University.</li>
<li>Yan, J., et al. (2026). BioMiner: A multi-modal system for automated mining of protein-ligand bioactivity data from
literature [Preprint]. arXiv. MolGlyph and the BioVista benchmark come from the same work.</li>
<li>Tools: MolScribe, MolSight (hustvl/MolSight, MolSight-base), MolGlyph, MolDetV2 (UniParser/MolDetv2), BioMiner, MinerU,
RDKit and PyMuPDF.</li>
</ul>""")


CSS = """
:root{color-scheme:light;--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#6f6d68;--grid:#e1e0d9;
--axis:#c3c2b7;--line:rgba(11,11,11,.12);--link:#1c5cab;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--code:#f0efec}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#a3a198;--grid:#2c2c2a;--axis:#383835;--line:rgba(255,255,255,.12);--link:#86b6ef;--s1:#3987e5;--s2:#d95926;
--s3:#199e70;--s4:#c98500;--code:#262624}}
:root[data-theme=dark]{color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#a3a198;--grid:#2c2c2a;
--axis:#383835;--line:rgba(255,255,255,.12);--link:#86b6ef;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--code:#262624}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main,header,footer{max-width:46rem;margin:0 auto;padding:0 16px}
header{padding-top:3rem;padding-bottom:1rem}
h1{font-size:2rem;line-height:1.2;margin:0 0 .5rem;letter-spacing:-.01em}
.lede{font-size:1.1rem;color:var(--ink2);margin:0 0 1rem}
h2{font-size:1.35rem;line-height:1.3;margin:3rem 0 1rem;padding-top:1.5rem;border-top:1px solid var(--line)}
h3{font-size:1.1rem;margin:2rem 0 .75rem}
.num-label{display:inline-block;min-width:1.6em;color:var(--muted);font-variant-numeric:tabular-nums}
p{margin:0 0 1rem}
.claim{font-size:1.08rem;font-weight:600}
.plain{border-left:3px solid var(--s1);padding:.4rem 0 .4rem .9rem;color:var(--ink)}
.caveat{color:var(--ink2);font-size:.95rem;border-left:3px solid var(--axis);padding-left:.9rem}
a{color:var(--link)}
code{font:.9em ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--code);padding:.05em .3em;border-radius:4px}
nav ol{padding-left:1.3rem;margin:.5rem 0 0}
nav li{margin:.3rem 0}
.findings{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:1rem 1.25rem;margin:1.5rem 0}
.findings h2{border:0;margin:0 0 .5rem;padding:0;font-size:1.05rem}
.findings ol{margin:0;padding-left:1.3rem}
.findings li{margin:.45rem 0}
figure{margin:1.5rem 0;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:.75rem .75rem .5rem}
figcaption{font-size:.88rem;color:var(--ink2);margin-top:.25rem}
.chart{display:block;width:100%;height:auto}
.chart.narrow{display:none}
@media (max-width:560px){.chart.wide{display:none}.chart.narrow{display:block}}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.pair figure{margin:.5rem 0}
@media (max-width:560px){.pair{grid-template-columns:1fr}}
.chart text{font:13px system-ui,-apple-system,"Segoe UI",sans-serif}
.chart.narrow text{font-size:14px}
.chart .t1{fill:var(--ink)}.chart .t2{fill:var(--ink2)}.chart .t3{fill:var(--muted);font-size:12px}
.chart .strong{font-weight:600}.chart .num{font-variant-numeric:tabular-nums}
.chart .s1{fill:var(--s1)}.chart .s2{fill:var(--s2)}.chart .s3{fill:var(--s3)}.chart .s4{fill:var(--s4)}
.chart .grid{stroke:var(--grid);stroke-width:1;fill:none}.chart .axis{stroke:var(--axis);stroke-width:1;fill:none}
.chart .whisker{stroke:var(--ink2);stroke-width:1.5;fill:none}
.chart .hit{fill:transparent}.chart g:hover .hit{fill:var(--grid);opacity:.35}
.table-wrap{overflow-x:auto;margin:1rem 0 1.5rem;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.9rem;font-variant-numeric:tabular-nums}
caption{text-align:left;color:var(--ink2);font-size:.88rem;padding-bottom:.4rem}
th,td{padding:.45rem .6rem;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;color:var(--ink2);white-space:nowrap}
td.num,th.num{text-align:right}
.ci{color:var(--ink2);white-space:nowrap}
.stack .ci{display:block;font-size:.82rem}
.table-note{font-size:.85rem;color:var(--ink2);margin:-.75rem 0 1.5rem}
.refs{padding-left:1.2rem;font-size:.92rem;color:var(--ink2)}
.refs li{margin:.4rem 0}
footer{color:var(--muted);font-size:.85rem;padding-top:2rem;padding-bottom:3rem}
"""


def page(d: dict) -> str:
    course = INCLUDE_COURSE
    head = d["biominer_course"]["scores"]["no_star"]["ALL"] if course else None
    summary = [
        '<li><a href="#checks">Installation checks catch silent failures.</a> Four failures ran to the end without an error '
        'and were caught only by comparing output with known answers.</li>']
    if course:
        summary += [
            '<li><a href="#losses">Where molecules are lost.</a> 138 of the 146 molecules the crop readers missed exist only as '
            'rows of R-group tables, so reading one drawing at a time can reach at most 34% to 37% recall on these papers.</li>',
            f'<li><a href="#biominer">BioMiner\'s R-group step.</a> Counting every whole structure BioMiner reads gives '
            f'{pc(head["recall"]["value"])} recall at {pc(head["precision"]["value"])} precision on the course papers, and its '
            'table step found 12 molecules no crop reader found.</li>',
            '<li><a href="#agreement">Reader agreement vs confidence.</a> Molecules on which three readers agree are right '
            'more often than MolScribe\'s most confident readings, but six papers cannot yet show the difference is real.</li>']
    else:
        summary.append('<li><a href="#biominer">BioMiner\'s R-group step.</a> On 12 BioVista papers, most of the shortfall on full '
                       'structures comes from linking structures to labels, not from finding them.</li>')
    summary.append('<li><a href="#public">Public benchmarks vs real papers.</a> Public image sets rank the readers '
                   f'{"differently from real papers" if course else "differently from one another"}, and a scoring choice about '
                   'tautomers moves one public score by nine points.</li>')
    intro = f"""
<p>Chemistry papers show molecules as drawings. To search or compare them, a program has to turn each drawing back
into a molecule. Tools that do this are called OCSR tools, for optical chemical structure recognition. They are usually
tested on single drawings already cut out of patents, where the best of them are right more than 90% of the time.
MolCensus tests them on whole papers instead. It renders every page, finds each drawing, reads it, and compares the
result with a list of every molecule the paper reports.</p>
<p>The test set is six medicinal chemistry papers from a Northeastern University course (AAI 6630, fall 2026), with 220
distinct molecules{'' if course else ' (not shown on this page)'}, plus public image sets and 12 papers from the
BioVista benchmark. The six papers and their answer key are private, so this page reports totals per paper and never
shows a molecule, a drawing or a compound name.</p>"""
    sections = [finding_checks(d)]
    if course:
        sections.append(finding_losses(d))
    sections.append(finding_biominer(d))
    if course:
        sections.append(finding_agreement(d))
    sections += [finding_public(d), methods(d), run_facts(d), limitations(), credits()]
    return f"""<!doctype html>
<html lang=en>
<head>
<meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<meta name=color-scheme content="light dark">
<title>MolCensus report</title>
<meta name=description content="A benchmark of tools that read chemical structure drawings, run on whole research papers.">
<style>{CSS}</style>
</head>
<body>
<header>
<h1>MolCensus</h1>
<p class=lede>A benchmark of tools that turn chemical drawings in research papers into machine-readable molecules,
tested on whole papers instead of single clean images.</p>
</header>
<main>
{intro}
<div class=findings><h2>Five findings</h2><ol>{''.join(summary)}</ol></div>
{''.join(sections)}
</main>
<footer>Results as of {REPORT_DATE}. Built by <code>scripts/report/build_site.py</code> from
<code>scripts/report/report_data.json</code>. Code under the MIT license at <a href="{REPO}">{REPO.replace('https://', '')}</a>.</footer>
</body>
</html>
"""


def main() -> None:
    data = json.loads(DATA.read_text())
    OUT.write_text(page(data))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes, course results {'included' if INCLUDE_COURSE else 'left out'})")


if __name__ == "__main__":
    main()
