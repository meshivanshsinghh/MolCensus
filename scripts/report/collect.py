"""Collect the numbers for the public report from MolCensus's local results.

Needs the private local files: the course ground truth, the readers' course predictions, the BioMiner
outputs downloaded from S3 and the recall-ceiling labels. Writes scripts/report/report_data.json, which
holds numbers only, at most per paper: no SMILES, no compound names, no crops or page images.
build_site.py renders the page from that file alone.

    python scripts/report/collect.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import biominer_facts as facts  # noqa: E402
import biominer_scores as bm  # noqa: E402
from molcensus.chem import Featurizer  # noqa: E402
from molcensus.config import load_config, resolve_path  # noqa: E402
from molcensus.data import load_ground_truth  # noqa: E402
from molcensus.metrics import paper_bootstrap, score_tool, wilson_interval  # noqa: E402

RDLogger.DisableLog("rdApp.*")
AWS = ROOT / "results" / "aws"
BIOMINER_RUN = ROOT / "outputs" / "s3" / "20261105T043509Z"
READERS = ["molscribe", "molsight", "molglyph"]
PAPERS = [1, 2, 4, 6, 16, 19]
OUT = ROOT / "scripts" / "report" / "report_data.json"


def rate(tp: int, n: int) -> dict:
    """A proportion with its 95% Wilson interval."""
    low, high = wilson_interval(int(tp), int(n))
    return {"k": int(tp), "n": int(n), "value": tp / n if n else None, "low": low, "high": high}


def rows(per: pd.DataFrame) -> dict:
    """biominer_scores.score() output -> per-paper and pooled recall and precision (counts only)."""
    result = {}
    for _, r in per.iterrows():
        entry = {"recall": rate(r["tp"], r["n_gt"]), "precision": rate(r["tp"], r["n_pred"]), "f1": r["f1"]}
        if r["paper_id"] == "ALL":
            entry["recall_boot"], entry["precision_boot"] = list(r["recall_boot"]), list(r["precision_boot"])
        result[str(r["paper_id"])] = entry
    return result


# ---------- public sets ----------

def public_sets() -> dict:
    """Per reader and dataset: InChIKey and canonical-SMILES accuracy, and BioMiner's rule on BioVista."""
    metrics = pd.read_csv(AWS / "public" / "public_metrics.csv")
    out = {}
    for tool in READERS:
        scored = pd.read_csv(AWS / "public" / f"{tool}_scored.csv",
                             usecols=["dataset", "gt_valid", "exact_match", "canonical_smiles", "gt_canonical_smiles", "biominer_rule_match"])
        scored = scored[scored["gt_valid"]]
        same = scored["canonical_smiles"].notna() & (scored["canonical_smiles"] == scored["gt_canonical_smiles"])
        for dataset, group in scored.groupby("dataset"):
            m = metrics[(metrics["tool"] == tool) & (metrics["dataset"] == dataset)].iloc[0]
            out.setdefault(dataset, {})[tool] = {
                "n": int(len(group)),
                "inchikey": rate(int(group["exact_match"].sum()), len(group)),
                "smiles": rate(int(same[group.index].sum()), len(group)),
                "biominer_rule": rate(round(m["biominer_rule_accuracy"] * m["n_biominer_rule"]), int(m["n_biominer_rule"])),
            }
    return out


def tautomer_breakdown(tool: str = "molscribe", dataset: str = "uob") -> dict:
    """Images that match by InChIKey but not by canonical SMILES, sorted by what differs."""
    s = pd.read_csv(AWS / "public" / f"{tool}_scored.csv", usecols=["dataset", "gt_valid", "exact_match", "canonical_smiles",
                                                                     "gt_canonical_smiles", "pred_smiles", "gt_smiles"])
    s = s[(s["dataset"] == dataset) & s["gt_valid"] & s["exact_match"] & (s["canonical_smiles"] != s["gt_canonical_smiles"])]
    enumerator, uncharger = rdMolStandardize.TautomerEnumerator(), rdMolStandardize.Uncharger()

    def form(smiles: str, step) -> str:
        return Chem.MolToSmiles(step(Chem.MolFromSmiles(smiles)))

    counts = {"tautomer": 0, "stereo": 0, "charge": 0, "other": 0}
    for pred, gt in zip(s["pred_smiles"], s["gt_smiles"]):
        # Stereo first: RDKit's tautomer canonicalizer also drops stereo marks.
        if Chem.MolToSmiles(Chem.MolFromSmiles(pred), isomericSmiles=False) == Chem.MolToSmiles(Chem.MolFromSmiles(gt), isomericSmiles=False):
            counts["stereo"] += 1
        elif form(pred, enumerator.Canonicalize) == form(gt, enumerator.Canonicalize):
            counts["tautomer"] += 1
        elif form(pred, uncharger.uncharge) == form(gt, uncharger.uncharge):
            counts["charge"] += 1
        else:
            counts["other"] += 1
    return {"tool": tool, "dataset": dataset, "n": int(len(s)), **counts}


# ---------- course papers ----------

def course_truth():
    config = load_config(ROOT / "configs" / "default.yaml")
    return load_ground_truth(resolve_path(config["paths"]["ground_truth_csv"]), PAPERS)


def reader_predictions() -> dict[str, pd.DataFrame]:
    return {t: pd.read_csv(AWS / "predictions" / f"{t}_predictions.csv", dtype={"pred_smiles": str}).rename(columns={"pred_smiles": "smiles"})
            for t in READERS}


def biominer_course(truth_frame: pd.DataFrame) -> dict:
    """BioMiner on the course papers under its three prediction rules, the readers beside it, canonical SMILES."""
    outputs = bm.run_outputs(BIOMINER_RUN / "outputs/work/biominer_output")
    linked = bm.keyed(bm.as_frame({p: o["merged"] for p, o in outputs.items()}), "smiles")
    everything = bm.union(linked, bm.keyed(bm.as_frame({p: o["ocsr"] for p, o in outputs.items()}), "smiles"))
    tools = {"linked": linked, "all_extracted": everything, "no_star": bm.whole_only(everything)}
    readers = {t: bm.keyed(frame, "smiles") for t, frame in reader_predictions().items()}
    truth = bm.keyed(truth_frame, "smiles", "gt_smiles")
    any_reader = bm.union(*readers.values())
    counts = {"pipeline_rows": 0, "ocsr_boxes": 0, "unlinked": 0, "unlinked_star": 0, "unlinked_correct": 0, "enumerated": 0}
    for p, o in outputs.items():
        merged, ocsr = bm.keys_of(o["merged"], "smiles"), bm.keys_of(o["ocsr"], "smiles")
        counts["pipeline_rows"] += len(merged)
        counts["ocsr_boxes"] += len(o["ocsr"])
        counts["unlinked"] += len(ocsr - merged)
        counts["unlinked_star"] += sum("*" in k for k in ocsr - merged)
        counts["unlinked_correct"] += len((ocsr - merged) & truth.get(p, set()))
        counts["enumerated"] += len(bm.keys_of(o["part"], "smiles"))
    only_biominer = {str(p): len(truth.get(p, set()) & linked.get(p, set()) - any_reader.get(p, set())) for p in PAPERS}
    return {
        "scores": {name: rows(bm.score(pred, truth, PAPERS)) for name, pred in {**tools, **readers}.items()},
        "only_biominer": only_biominer,
        "only_biominer_from_rgroup": sum(len(truth.get(p, set()) & bm.keys_of(outputs[p]["part"], "smiles") - any_reader.get(p, set()))
                                         for p in PAPERS),
        "found_by_any_reader": {str(p): len(truth.get(p, set()) & any_reader.get(p, set())) for p in PAPERS},
        "found_by_anyone": sum(len(truth.get(p, set()) & (any_reader.get(p, set()) | everything.get(p, set()))) for p in PAPERS),
        "starred_truth": int(truth_frame["gt_smiles"].str.contains("*", regex=False).sum()),
        "counts": counts,
    }


def biominer_biovista() -> dict:
    """Recall by label type on the 12 BioVista papers: linked vs all extracted (no * pieces) vs published."""
    folder = BIOMINER_RUN / "outputs/work_biovista/biominer_output"
    outputs = bm.run_outputs(folder)
    names = {p: next(f.name for f in (folder / str(p)).glob("extraction_*/*") if f.is_dir()) for p in outputs}
    papers = sorted(outputs)
    labels_dir = ROOT / "data" / "biovista" / "labels"
    result = {"papers": len(papers), "by_type": {}}
    for split in bm.TYPES:
        truth = {p: bm.keys_of(bm.SPLITS[split][0](bm.biovista_labels(labels_dir, names[p]))["smiles"].tolist(), "smiles") for p in papers}
        entry = {"published_recall": bm.PUBLISHED_BIOVISTA[split][0], "published_precision": bm.PUBLISHED_BIOVISTA[split][1]}
        for name in ["linked, BioMiner's pairing", "all extracted, no * pieces"]:
            per = bm.score({p: bm.keys_of(bm.TYPE_RULES[name](outputs[p], split), "smiles") for p in papers}, truth, papers)
            key = "linked" if name.startswith("linked") else "no_star"
            entry[key] = {"recall": rate(per["tp"].iloc[-1], per["n_gt"].iloc[-1]), "precision": rate(per["tp"].iloc[-1], per["n_pred"].iloc[-1]),
                          "mean_recall": bm.macro(per, "recall"), "mean_precision": bm.macro(per, "precision")}
        result["by_type"][split] = entry
    evaluated = [bm.biominer_evaluator((folder / str(p) / "biominer_stdout.log").read_text(errors="replace")) for p in papers]
    result["evaluator_mean"] = {s: [bm._mean([e[(s, r)] for e in evaluated if (s, r) in e]) for r in ("recall", "precision")]
                                for s in bm.STRUCTURE_KEYS}
    return result


def agreement(truth_frame: pd.DataFrame) -> dict:
    """Reader agreement vs MolScribe's confidence on the 237 course crops (InChIKey rule), with a paired paper bootstrap."""
    config = load_config(ROOT / "configs" / "aws_score.yaml")
    sc = config["scoring"]
    featurizer = Featurizer(sc["fingerprint_radius"], sc["fingerprint_bits"])
    preds = {t: pd.read_csv(AWS / "predictions" / f"{t}_predictions.csv", dtype={"pred_smiles": str, "pred_label": str}) for t in READERS}
    scored = {t: pd.read_csv(AWS / "scored" / f"{t}_scored_predictions.csv", dtype={"paper_id": str}) for t in READERS}
    crops = pd.DataFrame({t: scored[t].set_index("crop_id")["match_key"] for t in READERS})

    def majority(row):
        for t in READERS:
            if isinstance(row[t], str) and (row == row[t]).sum() >= 2:
                return t
        return None

    def per_paper(selection: pd.DataFrame) -> pd.DataFrame:
        _, _, pp = score_tool(selection.assign(tool="x"), truth_frame, featurizer, sc["near_match_threshold"], 10, 0,
                              sc["related_threshold"], tool="x")
        pp = pp.assign(paper_id=pp["paper_id"].astype(str))
        return pp[pp["paper_id"] != "ALL"].set_index("paper_id").loc[[str(p) for p in PAPERS], ["exact_tp", "n_pred_unique", "n_gt_unique"]]

    all3 = crops[[isinstance(r.molscribe, str) and r.molscribe == r.molsight == r.molglyph for r in crops.itertuples()]].index
    winner = crops.apply(majority, axis=1).dropna()
    by_confidence = preds["molscribe"].sort_values("confidence", ascending=False, kind="stable")
    selections = {
        "all_three": (len(all3), per_paper(preds["molscribe"][preds["molscribe"]["crop_id"].isin(all3)])),
        "top_conf_a": (len(all3), per_paper(by_confidence.head(len(all3)))),
        "two_agree": (len(winner), per_paper(pd.concat([preds[t][preds[t]["crop_id"].isin(winner[winner == t].index)] for t in READERS]))),
        "top_conf_b": (len(winner), per_paper(by_confidence.head(len(winner)))),
    }
    rng = np.random.default_rng(0)
    picks = rng.integers(0, len(PAPERS), size=(10_000, len(PAPERS)))

    def boot(s: pd.DataFrame):
        tp, n = s["exact_tp"].to_numpy(float)[picks].sum(1), s["n_pred_unique"].to_numpy(float)[picks].sum(1)
        return np.divide(tp, n, out=np.full(len(tp), np.nan), where=n > 0)

    out = {"selections": {}, "differences": {}}
    for name, (n_crops, s) in selections.items():
        out["selections"][name] = {"crops": n_crops, "precision": rate(s["exact_tp"].sum(), s["n_pred_unique"].sum()),
                                   "recall": rate(s["exact_tp"].sum(), s["n_gt_unique"].sum()),
                                   "per_paper": {p: [int(t), int(n)] for p, t, n in zip(s.index, s["exact_tp"], s["n_pred_unique"])}}
    for name, (a, c) in {"a_minus_c": ("all_three", "top_conf_a"), "b_minus_c": ("two_agree", "top_conf_b")}.items():
        sa, sc_ = selections[a][1], selections[c][1]
        diff = boot(sa) - boot(sc_)
        low, high = np.nanpercentile(diff, [2.5, 97.5])
        out["differences"][name] = {"estimate": sa["exact_tp"].sum() / sa["n_pred_unique"].sum() - sc_["exact_tp"].sum() / sc_["n_pred_unique"].sum(),
                                    "low": float(low), "high": float(high), "share_le_0": float(np.nanmean(diff <= 0))}
    return out


def course_readers_inchikey() -> dict:
    """The readers' pooled InChIKey scores, from the pipeline's own per-paper metrics."""
    per = pd.read_csv(AWS / "per_paper_metrics.csv", dtype={"paper_id": str})
    out = {}
    for tool in READERS:
        r = per[(per["tool"] == tool) & (per["paper_id"] == "ALL")].iloc[0]
        out[tool] = {"recall": rate(r["exact_tp"], r["n_gt_unique"]), "precision": rate(r["exact_tp"], r["n_pred_unique"]),
                     "recall_boot": [r["exact_recall_boot_low"], r["exact_recall_boot_high"]]}
    return out


def readers_found_inchikey() -> dict:
    """Per paper: distinct molecules (InChIKey) and how many any reader found; MolGlyph vs MolScribe, paired."""
    found = {}
    for tool in READERS:
        m = pd.read_csv(AWS / "scored" / f"{tool}_ground_truth_matches.csv", dtype={"paper_id": str})
        found[tool] = m[m["gt_valid"]].drop_duplicates(["paper_id", "gt_match_key"]).set_index(["paper_id", "gt_match_key"])["found_exact"]
    f = pd.DataFrame(found)
    by_paper = {p: {"n": int(len(g)), "found": int(g.any(axis=1).sum())} for p, g in f.groupby(level=0)}
    only_glyph, only_scribe = int((f["molglyph"] & ~f["molscribe"]).sum()), int((f["molscribe"] & ~f["molglyph"]).sum())
    return {"by_paper": by_paper, "molglyph_vs_molscribe": [only_glyph, only_scribe, bm.exact_mcnemar(only_glyph, only_scribe)]}


def recall_ceiling() -> dict:
    """Bucket totals per paper (final and strict rule), and the judges' accuracy and agreement."""
    labels = pd.read_csv(AWS / "scored" / "recall_ceiling.csv", dtype={"paper_id": str})
    final = {p: {str(b): int(n) for b, n in g["bucket"].value_counts().items()} for p, g in labels.groupby("paper_id")}
    # The strict rule keeps a layer 2 label even when both judges contradicted it (docs/recall_ceiling.md).
    withdrawn = (labels["bucket"] == 5) & labels["layer2_bucket"].notna()
    strict = labels["bucket"].astype(float).where(~withdrawn, labels["layer2_bucket"]).astype(int).value_counts().to_dict()
    tables = (ROOT / "outputs" / "ceiling" / "tables.md").read_text()
    judge = {name: [int(a), int(b)] for name, a, b in re.findall(r"- (claude|gemini): (\d+)/(\d+) =", tables)}
    kappas = re.findall(r"kappa (\d\.\d+)", tables)
    return {"missed": int(len(labels)), "by_paper": final, "strict": {str(k): int(v) for k, v in strict.items()},
            "judges": judge, "kappa_all": float(kappas[0]), "kappa_known": float(kappas[1]), "kappa_layer3": float(kappas[2])}


def run_facts() -> dict:
    rows_ = {stage: facts.paper_rows(BIOMINER_RUN / folder) for stage, folder in facts.PAPER_FOLDERS.items()}
    smoke = facts.call_counts((BIOMINER_RUN / "runs/biominer_smoke_output.log").read_text(errors="replace"))
    everything = [smoke] + [r for rs in rows_.values() for r in rs]
    total = {k: sum(r[k] for r in everything) for k in ("answered", "retries", "lost", "input", "output", "total")}
    return {
        "stages": facts.stages((BIOMINER_RUN / "runs/run_all.log").read_text()),
        "course_minutes": sum(r["minutes"] for r in rows_["biominer_course"]),
        "biovista_minutes": sum(r["minutes"] for r in rows_["biominer_biovista"]),
        "calls": total, "thinking": total["total"] - total["input"] - total["output"], "cost_usd": facts.cost(total),
        "cost_by_stage": {"smoke": facts.cost(smoke), **{s: facts.cost(facts.add_up(r, s)) for s, r in rows_.items()}},
        "models": sorted({m for rs in rows_.values() for r in rs for m in r["models"]} | set(smoke["models"])),
    }


def clean(value):
    """numpy numbers -> Python numbers, NaN -> None, tuples -> lists, so the JSON is plain and valid."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, bool, np.bool_)):
        return value if isinstance(value, bool) else int(value)
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(value) else round(float(value), 6)
    return value


def main() -> None:
    truth = course_truth()
    data = {
        "public": public_sets(),
        "tautomers_uob_molscribe": tautomer_breakdown(),
        "course_readers_inchikey": course_readers_inchikey(),
        "biominer_course": biominer_course(truth),
        "biominer_biovista": biominer_biovista(),
        "agreement": agreement(truth),
        "recall_ceiling": recall_ceiling(),
        "readers_found_inchikey": readers_found_inchikey(),
        "biominer_run": run_facts(),
    }
    OUT.write_text(json.dumps(clean(data), indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
