"""Score BioMiner's structures from a run downloaded from S3, on the course papers and the BioVista papers.

Prints counts and scores only, never a SMILES string, so its output is safe to paste into docs.

    python scripts/analysis/biominer_scores.py outputs/s3/20261105T043509Z

What counts as a BioMiner prediction (two rules, both reported):
- "linked structures", the pipeline's rule (biominer_runner.py): BioMiner's _structure_merged_output_ table,
  every structure the LLM tied to a compound label (full structures plus enumerated R-group tables).
- "all extracted structures": those plus every structure MolGlyph read in BioMiner's OCSR output
  (<task>_ocsr_output.json), whether or not the LLM tied it to a label.
- "all extracted, no * pieces": the same without structures that contain * atoms (R-group scaffolds and
  substituent pieces, which BioMiner reads only to enumerate R-group tables). Few of them can match: only 5 of
  the 222 course ground-truth rows contain * atoms (papers 2 and 19).
What counts as a match (two rules): canonical isomeric SMILES (primary) and InChIKey, as in
docs/aws_run_2026-11-04.md. Each paper's distinct predictions and distinct ground-truth molecules are
compared as sets, as in src/molcensus/metrics.py.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from molcensus.chem import Featurizer  # noqa: E402
from molcensus.config import load_config, resolve_path  # noqa: E402
from molcensus.data import load_ground_truth  # noqa: E402
from molcensus.metrics import featurize_column, paper_bootstrap, wilson_interval  # noqa: E402

RULES = {"smiles": "canonical_smiles", "inchikey": "match_key"}
READERS = ["molscribe", "molsight", "molglyph"]
FEATURIZER = Featurizer()

# BioMiner's structure tables under extraction_<model>/<name>/ ({name} is the PDF name up to the first dot).
TABLES = {
    "full": "{name}_structure_full_output_*.csv",
    "part": "{name}_structure_part_output_*.csv",
    "merged": "{name}_structure_merged_output_*.csv",
    "bioactivity": "{name}_merge_*.csv",
}

# Obaidullah's BioMiner run on the six course papers (group report, Tables 5 and 6), in percent.
# His model is not confirmed yet (probably BioMiner-Instruct).
OBAIDULLAH = {"precision": 72.4, "recall": 34.5, "f1": 46.8, "per_paper_recall": {1: 100, 2: 26, 4: 40, 6: 0, 16: 50, 19: 23}}

# BioMiner README at commit 17c6161 ("Evaluation on BioVista"): BioMiner-Instruct on the 450 test papers,
# structure-only recall and precision averaged over papers, BioMiner's own SMILES rule.
PUBLISHED_BIOVISTA = {"all": (0.580, 0.615), "full": (0.654, 0.630), "R-group": (0.339, 0.390), "chiral": (0.454, 0.467)}


# ---------- reading BioMiner's outputs ----------

def _smiles_in(node) -> list:
    """Every 'smiles' value in BioMiner's nested OCSR JSON (a list of papers, each a list of boxes)."""
    if isinstance(node, dict):
        return [node.get("smiles")] if "smiles" in node else [v for value in node.values() for v in _smiles_in(value)]
    if isinstance(node, list):
        return [v for item in node for v in _smiles_in(item)]
    return []


def paper_outputs(paper_dir: Path) -> dict[str, list]:
    """One paper's BioMiner structures by kind: SMILES lists for full, part, merged, bioactivity and ocsr."""
    outputs = {kind: [] for kind in [*TABLES, "ocsr"]}
    for folder in sorted(p for p in paper_dir.glob("extraction_*/*") if p.is_dir()):
        for kind, pattern in TABLES.items():
            for path in sorted(folder.glob(pattern.format(name=folder.name))):
                if path.stem.endswith("_clean_new"):  # copies BioMiner's BioVista evaluator writes
                    continue
                table = pd.read_csv(path, dtype=str)
                columns = {str(c).lower(): c for c in table.columns}
                if "smiles" in columns:
                    outputs[kind] += table[columns["smiles"]].tolist()
    for path in sorted(paper_dir.glob("*_ocsr_output.json")):
        outputs["ocsr"] += _smiles_in(json.loads(path.read_text()))
    return outputs


def run_outputs(folder: Path) -> dict[int, dict[str, list]]:
    """paper id -> paper_outputs, for every paper folder with a run_info.json."""
    return {int(p.parent.name): paper_outputs(p.parent) for p in sorted(folder.glob("*/run_info.json"))}


def as_frame(by_paper: dict[int, list]) -> pd.DataFrame:
    rows = [(paper, smiles) for paper, values in by_paper.items() for smiles in values]
    return pd.DataFrame(rows, columns=["paper_id", "smiles"])


def keyed(frame: pd.DataFrame, rule: str, column: str = "smiles") -> dict[int, set]:
    """paper id -> the set of distinct valid molecules under one rule."""
    keys: dict[int, set] = {int(p): set() for p in frame["paper_id"].unique()}
    if frame.empty:
        return keys
    features = featurize_column(frame[column], FEATURIZER)
    for paper, valid, key in zip(frame["paper_id"], features["valid"], features[RULES[rule]]):
        if valid and isinstance(key, str):
            keys[int(paper)].add(key)
    return keys


def keys_of(values: list, rule: str) -> set:
    """The distinct valid molecules in one list of SMILES."""
    return keyed(pd.DataFrame({"paper_id": 0, "smiles": values}), rule).get(0, set()) if values else set()


def whole_only(by_paper: dict[int, set]) -> dict[int, set]:
    """Drop structures with * atoms. Their keys keep the *: InChI fails on them, so they fall back to SMILES keys."""
    return {p: {k for k in keys if "*" not in k} for p, keys in by_paper.items()}


def union(*sets_by_paper: dict[int, set]) -> dict[int, set]:
    papers = {p for by_paper in sets_by_paper for p in by_paper}
    return {p: set().union(*(by_paper.get(p, set()) for by_paper in sets_by_paper)) for p in papers}


# ---------- scoring ----------

def score(pred: dict[int, set], truth: dict[int, set], papers: list[int], bootstrap_rounds: int = 2000) -> pd.DataFrame:
    """Per-paper and pooled recall, precision and F1, with Wilson intervals and a paper bootstrap on ALL."""
    rows = [{"paper_id": str(p), "n_gt": len(truth.get(p, set())), "n_pred": len(pred.get(p, set())),
             "tp": len(pred.get(p, set()) & truth.get(p, set()))} for p in papers]
    per = pd.DataFrame(rows)
    per = pd.concat([per, pd.DataFrame([{"paper_id": "ALL", **per[["n_gt", "n_pred", "tp"]].sum().to_dict()}])],
                    ignore_index=True)
    for rate, denominator in [("recall", "n_gt"), ("precision", "n_pred")]:
        per[rate] = [tp / n if n else math.nan for tp, n in zip(per["tp"], per[denominator])]
        bounds = [wilson_interval(int(tp), int(n)) for tp, n in zip(per["tp"], per[denominator])]
        per[f"{rate}_low"], per[f"{rate}_high"] = zip(*bounds)
        boot = paper_bootstrap(per, "tp", denominator, rounds=bootstrap_rounds)
        per[f"{rate}_boot"] = [boot if p == "ALL" else None for p in per["paper_id"]]
    p, r = per["precision"].fillna(0), per["recall"].fillna(0)
    per["f1"] = [2 * a * b / (a + b) if a + b else 0.0 for a, b in zip(p, r)]
    return per


def exact_mcnemar(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value: a binomial test on the discordant pairs."""
    n, k = only_a + only_b, min(only_a, only_b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def paired(a: dict[int, set], b: dict[int, set], truth: dict[int, set], papers: list[int]) -> tuple[int, int, float]:
    """Ground-truth molecules found only by a, only by b, and the exact McNemar p-value."""
    only_a = sum(len(truth.get(p, set()) & a.get(p, set()) - b.get(p, set())) for p in papers)
    only_b = sum(len(truth.get(p, set()) & b.get(p, set()) - a.get(p, set())) for p in papers)
    return only_a, only_b, exact_mcnemar(only_a, only_b)


# ---------- printing ----------

def cell(row, rate: str, ci: bool = True, count: bool = False) -> str:
    if math.isnan(row[rate]):
        return "-"
    text = f"{row[rate]:.3f}" + (f" [{row[f'{rate}_low']:.3f}, {row[f'{rate}_high']:.3f}]" if ci else "")
    return text + (f" ({row['tp']} of {row['n_gt' if rate == 'recall' else 'n_pred']})" if count else "")


def score_table(scores: dict[str, pd.DataFrame], compact: bool = False) -> list[str]:
    """Rows = papers, columns = tools or splits; each cell recall / precision.

    compact: per-paper cells give counts instead of intervals (for columns with different labels, as in BioVista's splits).
    """
    names = list(scores)
    lines = ["| Paper | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for index, paper in enumerate(scores[names[0]]["paper_id"]):
        rows = [scores[n].iloc[index] for n in names]
        pooled = paper == "ALL"
        label = ("**Pooled**" if pooled else paper) + ("" if compact else f" ({rows[0]['n_gt']})")
        lines.append(f"| {label} | " + " | ".join(f"{cell(r, 'recall', pooled or not compact, pooled or compact)} / "
                                                   f"{cell(r, 'precision', pooled or not compact, pooled or compact)}"
                                                   for r in rows) + " |")
    ends = [scores[n].iloc[-1] for n in names]
    lines.append("| Pooled, paper bootstrap | " + " | ".join(
        f"[{r['recall_boot'][0]:.3f}, {r['recall_boot'][1]:.3f}] / [{r['precision_boot'][0]:.3f}, {r['precision_boot'][1]:.3f}]"
        for r in ends) + " |")
    lines.append("| Pooled F1 | " + " | ".join(f"{r['f1']:.3f}" for r in ends) + " |")
    return lines


def count_table(outputs: dict[int, dict[str, list]], wrapper_rows: dict[int, int]) -> list[str]:
    """Distinct structures (canonical SMILES) in each kind of BioMiner output, per paper."""
    lines = ["| Paper | Pipeline rows | OCSR boxes | OCSR distinct | Full table | R-group table | Linked (merged) | "
             "With activity | OCSR not linked | Linked not in OCSR | All extracted |", "|---" * 11 + "|"]
    totals = [0] * 10
    for paper, kinds in sorted(outputs.items()):
        sets = {kind: keys_of(values, "smiles") for kind, values in kinds.items()}
        values = [wrapper_rows.get(paper, 0), len(kinds["ocsr"]),
                  len(sets["ocsr"]), len(sets["full"]), len(sets["part"]), len(sets["merged"]), len(sets["bioactivity"]),
                  len(sets["ocsr"] - sets["merged"]), len(sets["merged"] - sets["ocsr"]), len(sets["ocsr"] | sets["merged"])]
        totals = [t + v for t, v in zip(totals, values)]
        lines.append(f"| {paper} | " + " | ".join(map(str, values)) + " |")
    lines.append("| **Total** | " + " | ".join(f"**{t}**" for t in totals) + " |")
    return lines


# ---------- course papers ----------

def course(run: Path, predictions_dir: Path, ground_truth_csv: Path, papers: list[int]) -> None:
    outputs = run_outputs(run / "outputs/work/biominer_output")
    pipeline = pd.read_csv(run / "results/aws/predictions/biominer_predictions.csv", dtype={"pred_smiles": str})
    pipeline = pipeline.rename(columns={"pred_smiles": "smiles"})
    wrapper_rows = pipeline[pipeline["smiles"].notna()].groupby("paper_id").size().to_dict()

    print("## Course papers: BioMiner output files\n")
    print("Distinct structures by canonical SMILES. \"Pipeline rows\" are the (label, SMILES) pairs the pipeline kept.\n")
    print("\n".join(count_table(outputs, wrapper_rows)))

    truth_frame = load_ground_truth(ground_truth_csv, papers)
    readers = {t: pd.read_csv(predictions_dir / f"{t}_predictions.csv", dtype={"pred_smiles": str}).rename(columns={"pred_smiles": "smiles"})
               for t in READERS}
    ocsr = as_frame({p: k["ocsr"] for p, k in outputs.items()})
    merged = as_frame({p: k["merged"] for p, k in outputs.items()})

    for rule in RULES:
        truth = keyed(truth_frame, rule, "gt_smiles")
        tools = {"BioMiner (linked)": keyed(pipeline, rule)}
        same = {p: k for p, k in keyed(merged, rule).items() if k} == {p: k for p, k in tools["BioMiner (linked)"].items() if k}
        tools["BioMiner (all extracted)"] = union(tools["BioMiner (linked)"], keyed(ocsr, rule))
        tools["BioMiner (all extracted, no * pieces)"] = whole_only(tools["BioMiner (all extracted)"])
        tools.update({t: keyed(readers[t], rule) for t in READERS})
        scores = {t: score(pred, truth, papers) for t, pred in tools.items()}

        print(f"\n## Course papers, {rule} rule: recall / precision [95% Wilson]\n")
        print(f"Pipeline predictions equal the merged tables' structures: {same}.\n")
        print("\n".join(score_table(scores)))

        # Why "no * pieces" loses recall: a few ground-truth entries are themselves written with * atoms.
        starred = truth_frame[truth_frame["gt_smiles"].str.contains("*", regex=False)]
        print(f"\n### Ground-truth entries with * atoms ({rule}): which tools found them\n")
        print("| Paper | Entry | * atoms | " + " | ".join(tools) + " |\n" + "|---" * (len(tools) + 3) + "|")
        for _, row in starred.iterrows():
            key = keys_of([row["gt_smiles"]], rule)
            print(f"| {row['paper_id']} | {row['gt_name']} | {row['gt_smiles'].count('*')} | "
                  + " | ".join("yes" if key & pred.get(row["paper_id"], set()) else "" for pred in tools.values()) + " |")

        print(f"\n### Paired with the readers ({rule}): found only by BioMiner vs only by the reader, exact McNemar p\n")
        for name in ["BioMiner (linked)", "BioMiner (all extracted)"]:
            header = f"| {name} vs | " + " | ".join(READERS) + " |"
            print(header + "\n" + "|---" * (len(READERS) + 1) + "|")
            for scope in [papers, *[[p] for p in papers]]:
                label = "Pooled" if len(scope) > 1 else f"Paper {scope[0]}"
                cells = [paired(tools[name], tools[t], truth, scope) for t in READERS]
                print(f"| {label} | " + " | ".join(f"{a} vs {b}, p = {p:.2g}" for a, b, p in cells) + " |")
            print()

        any_reader = union(*(tools[t] for t in READERS))
        print(f"### Molecules BioMiner found that no reader found ({rule})\n")
        print("| Paper | Linked | All extracted | Found by any reader | Found by any reader or BioMiner (linked) |\n|---|---|---|---|---|")
        for p in papers:
            g = truth.get(p, set())
            print(f"| {p} | {len(g & tools['BioMiner (linked)'].get(p, set()) - any_reader.get(p, set()))} | "
                  f"{len(g & tools['BioMiner (all extracted)'].get(p, set()) - any_reader.get(p, set()))} | "
                  f"{len(g & any_reader.get(p, set()))} | {len(g & (any_reader.get(p, set()) | tools['BioMiner (linked)'].get(p, set())))} |")

        if rule == "smiles":
            ours = scores["BioMiner (linked)"]
            print("\n### Next to Obaidullah's BioMiner run (group report Tables 5 and 6, model not confirmed)\n")
            print("| Paper | Ours (linked) | Ours (all extracted) | Ours (all extracted, no * pieces) | Obaidullah |\n|---|---|---|---|---|")
            alls, whole = scores["BioMiner (all extracted)"], scores["BioMiner (all extracted, no * pieces)"]
            for index, paper in enumerate(ours["paper_id"][:-1]):
                print(f"| {paper}, recall | {100 * ours['recall'][index]:.1f} | {100 * alls['recall'][index]:.1f} | "
                      f"{100 * whole['recall'][index]:.1f} | {OBAIDULLAH['per_paper_recall'][int(paper)]} |")
            for rate in ["precision", "recall", "f1"]:
                print(f"| Pooled {rate} | {100 * ours[rate].iloc[-1]:.1f} | {100 * alls[rate].iloc[-1]:.1f} | "
                      f"{100 * whole[rate].iloc[-1]:.1f} | {OBAIDULLAH[rate]} |")
            print(f"| Distinct predictions | {ours['n_pred'].iloc[-1]} | {alls['n_pred'].iloc[-1]} | {whole['n_pred'].iloc[-1]} | "
                  f"about {round(OBAIDULLAH['recall'] / 100 * ours['n_gt'].iloc[-1] / (OBAIDULLAH['precision'] / 100))} (implied) |")


# ---------- BioVista papers ----------

def biovista_labels(labels_dir: Path, name: str) -> pd.DataFrame:
    """One paper's structure labels with BioMiner's splits: R-group (has a backbone), chiral ('@'), full (no backbone)."""
    labels = pd.read_csv(labels_dir / f"{name}_structure.csv", dtype=str)
    labels = labels[labels["smiles"].notna()]
    labels["rgroup"] = labels["backbone"].notna()
    labels["chiral"] = labels["smiles"].str.contains("@")
    return labels


def _chiral(values: list) -> list:
    return [s for s in values if isinstance(s, str) and "@" in s]


# split -> (its labels, the BioMiner structures scored against them), paired as BioMiner's own evaluator
# pairs them (BioMiner/runner/metric_fn.py, evaluate_structure_bioactivity_split_step).
SPLITS = {
    "all": (lambda labels: labels, lambda out: out["merged"]),
    "full": (lambda labels: labels[~labels["rgroup"]], lambda out: out["full"]),
    "R-group": (lambda labels: labels[labels["rgroup"]], lambda out: out["part"]),
    "chiral": (lambda labels: labels[labels["chiral"]], lambda out: _chiral(out["merged"])),
    "all, all extracted": (lambda labels: labels, lambda out: out["merged"] + out["ocsr"]),
    "all, all extracted, no * pieces": (lambda labels: labels, lambda out: _no_star(out["merged"] + out["ocsr"])),
}

def _no_star(values: list) -> list:
    return [s for s in values if isinstance(s, str) and "*" not in s]


# For the by-type comparison: which BioMiner structures each rule scores against one type's labels.
# "Linked, any table" isolates BioMiner's pairing; "all extracted, no * pieces" adds what OCSR read but the
# LLM did not tie to a label. Chiral labels are always scored against structures with '@', as BioMiner does.
TYPE_RULES = {
    "linked, BioMiner's pairing": lambda out, split: SPLITS[split][1](out),
    "linked, any table": lambda out, split: (_chiral if split == "chiral" else list)(out["merged"]),
    "all extracted, no * pieces": lambda out, split: (_chiral if split == "chiral" else list)(_no_star(out["merged"] + out["ocsr"])),
}
TYPES = ["all", "full", "R-group", "chiral"]

STRUCTURE_KEYS = {"all": "structure", "full": "entire_structure", "R-group": "scaffold_structure", "chiral": "chiral_structure"}


def biominer_evaluator(log_text: str) -> dict:
    """BioMiner's own --biovista_evaluate numbers for one paper, from its 'Average <key>: <value>' lines."""
    values = dict(re.findall(r"^Average (\S+): (-?[\d.e-]+)$", log_text, flags=re.M))
    found = {}
    for split, key in STRUCTURE_KEYS.items():
        for rate in ("recall", "precision"):
            value = values.get(f"{rate}_{key}_list")
            if value is not None and float(value) != -1:
                found[(split, rate)] = float(value)
    return found


def biovista(run: Path, labels_dir: Path) -> None:
    folder = run / "outputs/work_biovista/biominer_output"
    outputs = run_outputs(folder)
    names = {p: next(f.name for f in (folder / str(p)).glob("extraction_*/*") if f.is_dir()) for p in outputs}
    papers = sorted(outputs)

    for rule in RULES:
        print(f"\n## BioVista papers, {rule} rule: recall (found of labels) / precision (of distinct predictions)\n")
        scores = {}
        for split, (pick_labels, pick_structures) in SPLITS.items():
            truth = {p: keys_of(pick_labels(biovista_labels(labels_dir, names[p]))["smiles"].tolist(), rule) for p in papers}
            pred = {p: keys_of(pick_structures(outputs[p]), rule) for p in papers}
            scores[split] = score(pred, truth, papers)
        print("\n".join(score_table(scores, compact=True)))
        print("\nPaper mean (as BioMiner averages), recall / precision:",
              "; ".join(f"{s}: {macro(df, 'recall'):.3f} / {macro(df, 'precision'):.3f}" for s, df in scores.items()))

        print(f"\n### BioVista by type ({rule}): linking vs finding. Recall: pooled [95% Wilson] (found of labels); mean over papers\n")
        print("| Type | " + " | ".join(TYPE_RULES) + " | Published (BioMiner-Instruct, mean) |\n" + "|---" * (len(TYPE_RULES) + 2) + "|")
        by_type = {}
        for split in TYPES:
            truth = {p: keys_of(SPLITS[split][0](biovista_labels(labels_dir, names[p]))["smiles"].tolist(), rule) for p in papers}
            by_type[split] = {name: score({p: keys_of(pick(outputs[p], split), rule) for p in papers}, truth, papers)
                              for name, pick in TYPE_RULES.items()}
            print(f"| {split} | " + " | ".join(f"{cell(df.iloc[-1], 'recall', count=True)}; mean {macro(df, 'recall'):.3f}"
                                               for df in by_type[split].values()) + f" | {PUBLISHED_BIOVISTA[split][0]:.3f} |")
        for split in ("all", "chiral"):  # full and R-group: OCSR structures have no type, so no per-type precision
            print(f"| {split}, precision | " + " | ".join(f"{cell(df.iloc[-1], 'precision', count=True)}; mean {macro(df, 'precision'):.3f}"
                                                          for df in by_type[split].values()) + f" | {PUBLISHED_BIOVISTA[split][1]:.3f} |")

    print("\n## BioVista: BioMiner's own --biovista_evaluate output (its SMILES rule), recall / precision\n")
    evaluated = {p: biominer_evaluator((folder / str(p) / "biominer_stdout.log").read_text(errors="replace")) for p in papers}
    splits = list(STRUCTURE_KEYS)
    print("| Paper | " + " | ".join(splits) + " |\n" + "|---" * (len(splits) + 1) + "|")
    for p in papers:
        print(f"| {p} | " + " | ".join(f"{evaluated[p].get((s, 'recall'), float('nan')):.3f} / "
                                       f"{evaluated[p].get((s, 'precision'), float('nan')):.3f}" for s in splits) + " |")
    means = {s: tuple(_mean([evaluated[p][(s, r)] for p in papers if (s, r) in evaluated[p]]) for r in ("recall", "precision"))
             for s in splits}
    print("| **Mean over papers** | " + " | ".join(f"**{means[s][0]:.3f} / {means[s][1]:.3f}**" for s in splits) + " |")
    print("| Published (BioMiner-Instruct, 450 papers) | " + " | ".join(f"{PUBLISHED_BIOVISTA[s][0]:.3f} / {PUBLISHED_BIOVISTA[s][1]:.3f}"
                                                                         for s in splits) + " |")
    print("Papers with evaluator output:", sum(bool(evaluated[p]) for p in papers), "of", len(papers))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def macro(per: pd.DataFrame, rate: str) -> float:
    """Mean of per-paper rates over papers with labels in the split; precision 0 for a paper with no predictions."""
    papers = per[(per["paper_id"] != "ALL") & (per["n_gt"] > 0)]
    return _mean(papers[rate].fillna(0).tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path, help="the run folder downloaded from S3")
    parser.add_argument("--predictions", type=Path, default=ROOT / "results/aws/predictions",
                        help="the crop readers' course predictions (default: results/aws/predictions)")
    parser.add_argument("--config", default=ROOT / "configs/default.yaml", help="for the ground-truth path and paper list")
    args = parser.parse_args()
    config = load_config(args.config)
    course(args.run, args.predictions, resolve_path(config["paths"]["ground_truth_csv"]), list(config["papers"]))
    biovista(args.run, ROOT / "data/biovista/labels")


if __name__ == "__main__":
    main()
