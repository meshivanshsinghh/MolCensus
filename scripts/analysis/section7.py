"""Section 7 of docs/aws_run_2026-11-04.md. Prints counts and scores only, never SMILES.
Writes the per-molecule recall-ceiling draft to results/aws/scored/ (gitignored). Run from the repo root."""

import collections
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

sys.path.insert(0, "src")
from molcensus import data  # noqa: E402
from molcensus.chem import Featurizer  # noqa: E402
from molcensus.config import load_config  # noqa: E402
from molcensus.metrics import score_tool, set_match, wilson_interval  # noqa: E402

RDLogger.DisableLog("rdApp.*")
TOOLS = ["molscribe", "molsight", "molglyph"]
PAPERS = ["1", "2", "4", "6", "16", "19"]
AWS = Path("results/aws")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
TAUT, UNCHARGE = rdMolStandardize.TautomerEnumerator(), rdMolStandardize.Uncharger()


def mol(s):
    return Chem.MolFromSmiles(s.strip()) if isinstance(s, str) and s.strip() else None


def could_contain(big, small) -> bool:
    """Is small (nearly) a piece of big? Fingerprint containment: share of small's Morgan bits also set in big.
    ponytail: a draft-label proxy; an exact substructure search hung for 20+ minutes on one large molecule."""
    if small.GetNumHeavyAtoms() > big.GetNumHeavyAtoms():
        return False
    return DataStructs.TverskySimilarity(GEN.GetFingerprint(small), GEN.GetFingerprint(big), 1.0, 0.0) >= 0.9


def why(pred, gt) -> str:
    """Why two molecules with the same InChIKey have different canonical SMILES."""
    mp, mg = mol(pred), mol(gt)
    smi = Chem.MolToSmiles
    if smi(mp, isomericSmiles=False) == smi(mg, isomericSmiles=False):
        return "stereo"
    if smi(TAUT.Canonicalize(mp)) == smi(TAUT.Canonicalize(mg)):
        return "tautomer"
    if smi(UNCHARGE.uncharge(mp)) == smi(UNCHARGE.uncharge(mg)):
        return "charge"
    return "other"


scored = {t: pd.read_csv(AWS / "scored" / f"{t}_scored_predictions.csv", dtype={"paper_id": str, "pred_smiles": str}) for t in TOOLS}
truth_rows = pd.read_csv(AWS / "scored" / "molscribe_ground_truth_matches.csv", dtype={"paper_id": str})
truth_rows = truth_rows[truth_rows.gt_valid]

# 1. Course papers: canonical isomeric SMILES vs InChIKey.
print("== 1. Course papers, InChIKey vs canonical SMILES")
for tool in TOOLS:
    p = scored[tool][scored[tool].valid]
    rows, ik_only, reasons = [], 0, collections.Counter()
    for paper in PAPERS:
        pp, tt = p[p.paper_id == paper], truth_rows[truth_rows.paper_id == paper]
        ik, sm = set_match(pp.match_key, tt.gt_match_key), set_match(pp.canonical_smiles, tt.gt_canonical_smiles)
        rows.append((paper, ik, sm))
        # Ground-truth molecules found under InChIKey but not under SMILES, and why.
        smiles_found = set(pp.canonical_smiles)
        for _, g in tt.drop_duplicates("gt_match_key").iterrows():
            hits = pp[pp.match_key == g.gt_match_key]
            if len(hits) and g.gt_canonical_smiles not in smiles_found and not (tt[tt.gt_match_key == g.gt_match_key].gt_canonical_smiles.isin(smiles_found)).any():
                ik_only += 1
                reasons[why(hits.pred_smiles.iloc[0], g.gt_smiles)] += 1
    for paper, ik, sm in rows + [("ALL", *[{k: sum(r[i][k] for r in rows) for k in ("tp", "n_gt_unique", "n_pred_unique")} for i in (1, 2)])]:
        rl, rh = wilson_interval(sm["tp"], sm["n_gt_unique"])
        pl, ph = wilson_interval(sm["tp"], sm["n_pred_unique"])
        print(f"  {tool:9s} {paper:>3}: InChIKey {ik['tp']}/{ik['n_gt_unique']} R {ik['tp'] / ik['n_gt_unique']:.3f} P {ik['tp'] / max(ik['n_pred_unique'], 1):.3f} (of {ik['n_pred_unique']})"
              f" | SMILES {sm['tp']}/{sm['n_gt_unique']} R {sm['tp'] / sm['n_gt_unique']:.3f} [{rl:.3f}, {rh:.3f}] P {sm['tp'] / max(sm['n_pred_unique'], 1):.3f} [{pl:.3f}, {ph:.3f}] (of {sm['n_pred_unique']})")
    print(f"  {tool}: molecules found only under InChIKey: {ik_only} {dict(reasons)}")

# 2. Paired paper bootstrap of the precision differences, (a) - (c) and (b) - (c).
config = load_config("configs/aws_score.yaml")
truth = data.load_ground_truth(config["paths"]["ground_truth_csv"], config["papers"])
sc = config["scoring"]
featurizer = Featurizer(sc["fingerprint_radius"], sc["fingerprint_bits"])
preds = {t: pd.read_csv(AWS / "predictions" / f"{t}_predictions.csv", dtype={"pred_smiles": str, "pred_label": str}) for t in TOOLS}
crops = pd.DataFrame({t: scored[t].set_index("crop_id").match_key for t in TOOLS})


def majority(row):
    for t in TOOLS:
        if isinstance(row[t], str) and (row == row[t]).sum() >= 2:
            return t
    return None


def per_paper(selection: pd.DataFrame) -> pd.DataFrame:
    _, _, pp = score_tool(selection.assign(tool="x"), truth, featurizer, sc["near_match_threshold"], 10, 0, sc["related_threshold"], tool="x")
    pp = pp.assign(paper_id=pp.paper_id.astype(str))
    return pp[pp.paper_id != "ALL"].set_index("paper_id").loc[PAPERS, ["exact_tp", "n_pred_unique"]]


all3 = crops[[isinstance(r.molscribe, str) and r.molscribe == r.molsight == r.molglyph for r in crops.itertuples()]].index
winner = crops.apply(majority, axis=1).dropna()
ms = preds["molscribe"].sort_values("confidence", ascending=False, kind="stable")
sel = {
    "a": per_paper(preds["molscribe"][preds["molscribe"].crop_id.isin(all3)]),
    "b": per_paper(pd.concat([preds[t][preds[t].crop_id.isin(winner[winner == t].index)] for t in TOOLS])),
    "c_a": per_paper(ms.head(len(all3))),
    "c_b": per_paper(ms.head(len(winner))),
}
print("== 2. Paired paper bootstrap (10,000 resamples, seed 0) of the precision difference")
for name, s in sel.items():
    print(f"  {name} per paper tp/pred: {[f'{int(t)}/{int(n)}' for t, n in zip(s.exact_tp, s.n_pred_unique)]}")
rng = np.random.default_rng(0)
picks = rng.integers(0, len(PAPERS), size=(10_000, len(PAPERS)))


def boot_precision(s):
    tp, n = s.exact_tp.to_numpy(float)[picks].sum(1), s.n_pred_unique.to_numpy(float)[picks].sum(1)
    return np.divide(tp, n, out=np.full(len(tp), np.nan), where=n > 0)


for label, x, y in [("(a) - (c), 59 crops", "a", "c_a"), ("(b) - (c), 119 crops", "b", "c_b")]:
    point = sel[x].exact_tp.sum() / sel[x].n_pred_unique.sum() - sel[y].exact_tp.sum() / sel[y].n_pred_unique.sum()
    diff = boot_precision(sel[x]) - boot_precision(sel[y])
    lo, hi = np.nanpercentile(diff, [2.5, 97.5])
    print(f"  {label}: difference {point:+.3f}, 95% interval [{lo:+.3f}, {hi:+.3f}], share of resamples <= 0: {np.nanmean(diff <= 0):.3f}, undefined resamples {int(np.isnan(diff).sum())}")

# 3. MolScribe vs MolGlyph, found only by each, every paper.
found = {}
for t in TOOLS:
    m = pd.read_csv(AWS / "scored" / f"{t}_ground_truth_matches.csv", dtype={"paper_id": str})
    found[t] = m[m.gt_valid].drop_duplicates(["paper_id", "gt_match_key"]).set_index(["paper_id", "gt_match_key"]).found_exact
f = pd.DataFrame(found)
print("== 3. MolScribe vs MolGlyph per paper (only MolScribe, only MolGlyph, both, neither)")
for paper in PAPERS + ["ALL"]:
    g = f if paper == "ALL" else f.loc[paper]
    print(f"  {paper:>3} n {len(g):3d}: {int((g.molscribe & ~g.molglyph).sum())}, {int((~g.molscribe & g.molglyph).sum())}, {int((g.molscribe & g.molglyph).sum())}, {int((~g.molscribe & ~g.molglyph).sum())}")

# 4. Recall-ceiling draft for the molecules no reader found, from the readers' outputs on the 237 crops.
gen = rdFingerprintGenerator.GetMorganGenerator(radius=sc["fingerprint_radius"], fpSize=sc["fingerprint_bits"])
stereo_blind = pd.DataFrame({t: pd.read_csv(AWS / "scored" / f"{t}_ground_truth_matches.csv", dtype={"paper_id": str})
                             .query("gt_valid").drop_duplicates(["paper_id", "gt_match_key"]).set_index(["paper_id", "gt_match_key"]).found_stereo_blind for t in TOOLS})
missed = truth_rows.drop_duplicates(["paper_id", "gt_match_key"]).set_index(["paper_id", "gt_match_key"]).loc[f.index[~f.any(axis=1)]].reset_index()
all_preds = pd.concat([scored[t][scored[t].valid].assign(reader=t) for t in TOOLS])
all_preds["mol"] = all_preds.pred_smiles.map(mol)
all_preds = all_preds[all_preds.mol.notna()]
all_preds["markush"] = [any(a.GetAtomicNum() == 0 for a in m.GetAtoms()) for m in all_preds.mol]
def core_of(m):
    """The drawing with its * atoms removed, with ring info set so it can be fingerprinted."""
    core = Chem.DeleteSubstructs(m, Chem.MolFromSmarts("[#0]"))
    core.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(core)
    return core


all_preds["core"] = [core_of(m) if k else m for m, k in zip(all_preds.mol, all_preds.markush)]
all_preds["fp"] = [gen.GetFingerprint(m) for m in all_preds.mol]
found_keys = set(truth_rows.gt_match_key[truth_rows.set_index(["paper_id", "gt_match_key"]).index.isin(f.index[f.any(axis=1)])])

rows = []
for _, g in missed.iterrows():
    gm = mol(g.gt_smiles)
    n_atoms, fp = gm.GetNumHeavyAtoms(), gen.GetFingerprint(gm)
    pp = all_preds[all_preds.paper_id == g.paper_id]
    sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, list(pp.fp))) if len(pp) else np.array([])
    full = pp[~pp.markush & ~pp.match_key.isin(found_keys)]  # complete molecules that are not another ground-truth molecule read right
    full_sims = sims[(~pp.markush & ~pp.match_key.isin(found_keys)).to_numpy()] if len(pp) else np.array([])
    best_full = full.iloc[int(full_sims.argmax())] if len(full) else None
    size_close = best_full is not None and abs(best_full.mol.GetNumHeavyAtoms() - n_atoms) <= max(3, 0.15 * n_atoms)
    cores = pp[pp.markush & (pp.core.map(lambda c: c.GetNumHeavyAtoms()) >= max(8, 0.3 * n_atoms))]  # a lone ring is no evidence
    core_hit = cores[[could_contain(gm, c) for c in cores.core]]
    parts = full[[(m.GetNumHeavyAtoms() >= max(8, 0.4 * n_atoms)) and (m.GetNumHeavyAtoms() <= n_atoms - 4) and could_contain(gm, m) for m in full.mol]]
    best = float(full_sims.max()) if len(full_sims) else 0.0
    if stereo_blind.loc[(g.paper_id, g.gt_match_key)].any():
        label, evidence = "complete drawing", "a reader got the structure right except stereo"
    elif best >= 0.85 and size_close:
        label, evidence = "complete drawing", f"near miss: Tanimoto {best:.2f}, similar size ({best_full.reader}, crop {best_full.crop_id})"
    elif len(core_hit):
        label, evidence = "R-group table, placeholder or linker box", f"its core is a Markush drawing in crop {core_hit.crop_id.iloc[0]} ({core_hit.reader.iloc[0]})"
    elif len(parts):
        label, evidence = "R-group table, placeholder or linker box", f"only part of it is drawn: a {parts.mol.iloc[0].GetNumHeavyAtoms()}-atom piece of its {n_atoms} atoms (crop {parts.crop_id.iloc[0]})"
    elif best >= 0.5:
        label, evidence = "unclear", f"only related structures read: best Tanimoto {best:.2f}"
    else:
        label, evidence = "no evidence in any crop", f"best Tanimoto {best:.2f}: text only, or the drawing was not detected"
    rows.append({"paper_id": g.paper_id, "gt_name": g.gt_name, "draft_label": label, "evidence": evidence, "status": "DRAFT, check by hand", "checked_label": ""})

draft = pd.DataFrame(rows)
draft.to_csv(AWS / "scored" / "recall_ceiling_draft.csv", index=False)
print(f"== 4. Recall-ceiling draft: {len(draft)} molecules -> {AWS / 'scored' / 'recall_ceiling_draft.csv'}")
print(pd.crosstab(draft.paper_id, draft.draft_label, margins=True, margins_name="total").reindex(PAPERS + ["total"], fill_value=0).to_string())
