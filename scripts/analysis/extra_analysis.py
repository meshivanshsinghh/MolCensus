"""Paired McNemar, canonical-SMILES rescoring of the public sets, and agreement vs confidence.
Prints counts and scores only, never SMILES. Run from the repo root with the repo's .venv."""

import itertools
import math
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, "src")
from molcensus import data  # noqa: E402
from molcensus.chem import Featurizer  # noqa: E402
from molcensus.config import load_config  # noqa: E402
from molcensus.metrics import score_tool, wilson_interval  # noqa: E402

RDLogger.DisableLog("rdApp.*")
TOOLS = ["molscribe", "molsight", "molglyph"]
AWS = Path("results/aws")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar: binomial test of min(b, c) out of b + c at p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


assert mcnemar_exact(0, 6) == 2 / 64 and mcnemar_exact(3, 3) == 1.0  # self-check

# 1. Paired comparison on the course molecules (the scorer's unique, valid ground truth).
found = None
for tool in TOOLS:
    t = pd.read_csv(AWS / "scored" / f"{tool}_ground_truth_matches.csv", dtype={"paper_id": str})
    t = t[t.gt_valid].drop_duplicates(["paper_id", "gt_match_key"])[["paper_id", "gt_match_key", "found_exact"]]
    t = t.rename(columns={"found_exact": tool})
    found = t if found is None else found.merge(t, on=["paper_id", "gt_match_key"], validate="1:1")
print("== 1. McNemar", len(found), "molecules;", {t: int(found[t].sum()) for t in TOOLS})
for scope in ["ALL", "1", "4", "19"]:
    f = found if scope == "ALL" else found[found.paper_id == scope]
    for a, b in itertools.combinations(TOOLS, 2):
        only_a, only_b = int((f[a] & ~f[b]).sum()), int((~f[a] & f[b]).sum())
        both, neither = int((f[a] & f[b]).sum()), int((~f[a] & ~f[b]).sum())
        print(f"  {scope:>3} n {len(f):3d} {a} vs {b}: only {a} {only_a}, only {b} {only_b}, both {both}, neither {neither}, p {mcnemar_exact(only_a, only_b):.4f}")

# 2. Public sets: InChIKey (as scored) vs canonical isomeric SMILES, same images (reference readable by RDKit).
def canonical(s):
    mol = Chem.MolFromSmiles(s.strip()) if isinstance(s, str) and s.strip() else None
    return Chem.MolToSmiles(mol) if mol is not None else None  # isomeric by default

print("== 2. Public sets: InChIKey vs canonical isomeric SMILES")
for tool in TOOLS:
    s = pd.read_csv(AWS / "public" / f"{tool}_scored.csv", dtype={"pred_smiles": str, "gt_smiles": str})
    for dataset in ["uspto", "uob", "clef", "acs"]:
        d = s[(s.dataset == dataset) & s.gt_valid]
        gt_c, pred_c = d.gt_smiles.map(canonical), d.pred_smiles.map(canonical)
        smiles_hit = (pred_c == gt_c) & pred_c.notna()
        n, ik, sm = len(d), int(d.exact_match.sum()), int(smiles_hit.sum())
        lo, hi = wilson_interval(sm, n)
        print(f"  {tool:9s} {dataset:5s} n {n:5d} InChIKey {ik / n:.3f} | canonical SMILES {sm / n:.3f} [{lo:.3f}, {hi:.3f}]"
              f" | InChIKey only {int((d.exact_match & ~smiles_hit).sum())}, SMILES only {int((~d.exact_match & smiles_hit).sum())}")

# 3. Agreement vs confidence on the course crops, scored with the pipeline's own scorer.
config = load_config("configs/aws_score.yaml")
truth = data.load_ground_truth(config["paths"]["ground_truth_csv"], config["papers"])
sc = config["scoring"]
featurizer = Featurizer(sc["fingerprint_radius"], sc["fingerprint_bits"])
preds = {t: pd.read_csv(AWS / "predictions" / f"{t}_predictions.csv", dtype={"pred_smiles": str, "pred_label": str}) for t in TOOLS}
keys = {t: pd.read_csv(AWS / "scored" / f"{t}_scored_predictions.csv", dtype={"pred_smiles": str}).set_index("crop_id").match_key for t in TOOLS}
crops = pd.DataFrame(keys)  # one row per crop, each reader's match key (None if invalid)
assert len(crops) == 237


def pooled(selection: pd.DataFrame, name: str) -> str:
    _, _, per_paper = score_tool(selection.assign(tool=name), truth, featurizer, sc["near_match_threshold"],
                                 sc.get("bootstrap_rounds", 2000), sc.get("seed", 0), sc.get("related_threshold", 0.5), tool=name)
    r = per_paper[per_paper.paper_id == "ALL"].iloc[0]
    return (f"recall {r.exact_recall:.3f} [{r.exact_recall_ci_low:.3f}, {r.exact_recall_ci_high:.3f}] ({r.exact_tp} of {r.n_gt_unique}), "
            f"precision {r.exact_precision:.3f} [{r.exact_precision_ci_low:.3f}, {r.exact_precision_ci_high:.3f}] ({r.exact_tp} of {r.n_pred_unique})")


def majority(row) -> str | None:
    """The tool whose valid key at least two readers share, else None."""
    for t in TOOLS:
        if isinstance(row[t], str) and (row == row[t]).sum() >= 2:
            return t
    return None


all3 = crops[[isinstance(r.molscribe, str) and r.molscribe == r.molsight == r.molglyph for r in crops.itertuples()]].index
winner = crops.apply(majority, axis=1).dropna()
two = winner.index
by_two = pd.concat([preds[t][preds[t].crop_id.isin(winner[winner == t].index)] for t in TOOLS])
ms = preds["molscribe"].sort_values("confidence", ascending=False, kind="stable")
print("== 3. Agreement vs confidence (237 crops)")
print(f"  (a) all three agree: {len(all3)} crops; {pooled(preds['molscribe'][preds['molscribe'].crop_id.isin(all3)], 'all3')}")
print(f"  (b) at least two agree: {len(two)} crops; {pooled(by_two, 'two')}")
for label, n in [("(c) MolScribe top-confidence, as many as (a)", len(all3)), ("(c) MolScribe top-confidence, as many as (b)", len(two))]:
    cut = ms.confidence.iloc[n - 1]
    ties = int((ms.confidence.iloc[n:] == cut).sum())
    print(f"  {label}: {n} crops (cutoff confidence {cut:.4f}, {ties} more crops tied at it); {pooled(ms.head(n), 'top')}")
for t in TOOLS:
    print(f"  {t} alone: 237 crops; {pooled(preds[t], t)}")
print(f"  found by any reader: {pooled(pd.concat(preds.values()), 'any')}")
