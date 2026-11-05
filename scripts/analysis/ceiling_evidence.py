"""Recall ceiling, layers 1 and 2: what the ground truth records, and automatic evidence (docs/recall_ceiling.md).

Writes outputs/ceiling/evidence.csv (gitignored), one row per unique valid ground-truth molecule of the six
course papers. Prints counts only, never SMILES. ceiling_run.py runs this and every later step.

Layer 2 has three tests. Each one is positive evidence for one bucket:
  a. Detection (bucket 2). MolDetV2 runs again at confidence LOW_THRESHOLD. A new crop (below the pipeline's
     threshold, and mostly outside every pipeline crop) that MolScribe reads as the molecule, by InChIKey or
     canonical SMILES, shows the molecule is drawn in full but was not cropped.
  b. Tables (bucket 3). A reading with * atoms (R-groups or attachment points) whose core, the reading
     without its * atoms, is an exact substructure of the molecule, and the molecule branches off that core
     only where the * atoms were. The molecule is then built from a scaffold or a fragment in the paper.
  c. Reading (bucket 1). A pipeline crop's reading with Tanimoto >= TANIMOTO_MIN to the molecule, when the
     molecule is the reading's single closest ground-truth molecule in the paper and the reading is not
     itself another ground-truth molecule. The molecule was cropped and misread.
"""

from __future__ import annotations

import multiprocessing
import re
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from molcensus.chem import Featurizer  # noqa: E402
from molcensus.config import load_config  # noqa: E402

RDLogger.DisableLog("rdApp.*")
CONFIG = ROOT / "configs" / "aws_score.yaml"
AWS = ROOT / "results" / "aws"
OUT = ROOT / "outputs" / "ceiling"
READERS = ["molscribe", "molsight", "molglyph"]
LOW_THRESHOLD = 0.05   # MolDetV2 confidence for the extra detection pass; the pipeline uses 0.4
NEW_MAX_INSIDE = 0.5   # a low-threshold box counts as new if at most this share of it lies inside a pipeline crop
TANIMOTO_MIN = 0.8     # test c
MIN_CORE_ATOMS = 8     # test b: a smaller core, such as a lone ring, matches almost anything
TABLE_TIMEOUT_S = 30   # test b: time limit per molecule for all its substructure searches


# ---- Layer 1: what the ground-truth CSV records -------------------------------------------------------

def label_kind(name: str) -> tuple[str, str]:
    """What a ground-truth name says about the molecule: (kind of label, scheme number if one is named)."""
    scheme = re.fullmatch(r"unnamed cmpd \d+ in Scheme (\d+)", name)
    if scheme:
        return "unnamed, scheme given", scheme.group(1)
    for kind, pattern in [("number", r"\d+"), ("number and letter", r"\d+[a-z]"),
                          ("letter and number", r"[A-Z]\d+"), ("intermediate", r"Intermediate \d+")]:
        if re.fullmatch(pattern, name):
            return kind, ""
    return "name or code", ""


def searchable(name: str) -> str | None:
    """The text to look for in the PDF, or None when it would match everywhere (a plain number)."""
    kind, scheme = label_kind(name)
    if kind == "unnamed, scheme given":
        return f"Scheme {scheme}"
    return None if kind in ("number", "intermediate") else name


def text_pages(pdf: Path, needles: set[str]) -> dict[str, list[int]]:
    """Pages whose text layer holds each needle as a whole token (so 17q does not match 117q or 17qa)."""
    import fitz  # PyMuPDF

    found = {needle: [] for needle in needles}
    with fitz.open(pdf) as document:
        for number, page in enumerate(document, start=1):
            text = " ".join(page.get_text().split())
            for needle in needles:
                if re.search(rf"(?<![\w-]){re.escape(needle)}(?![\w-])", text):
                    found[needle].append(number)
    return found


# ---- Layer 2a: a second detection pass ----------------------------------------------------------------

def inside_share(box, others: pd.DataFrame) -> float:
    """Largest share of box's area that lies inside one of the other boxes."""
    area = max((box.x2 - box.x1) * (box.y2 - box.y1), 1)
    best = 0.0
    for other in others.itertuples():
        w = min(box.x2, other.x2) - max(box.x1, other.x1)
        h = min(box.y2, other.y2) - max(box.y1, other.y1)
        best = max(best, max(w, 0) * max(h, 0) / area)
    return best


def low_threshold_crops(config: dict) -> pd.DataFrame:
    """MolDetV2 on the same page images at LOW_THRESHOLD. Crops go to outputs/ceiling/crops_lowthr/."""
    index = OUT / "crops_lowthr_index.csv"
    if index.exists():
        return pd.read_csv(index)
    from molcensus.detect import detect_with_moldet
    from molcensus.weights import fetch_weights

    paths, moldet = config["paths"], config["detection"]["moldet"]
    weights = fetch_weights(moldet["hf_repo"], moldet["weights_file"], paths["models_dir"], moldet.get("local_weights", ""))
    pages = pd.read_csv(paths["work_dir"] / "pages_index.csv")
    crops = detect_with_moldet(pages, weights, OUT / "crops_lowthr", moldet["imgsz"], LOW_THRESHOLD, moldet["pad_px"], "cpu")
    crops["crop_id"] = "low_" + crops.crop_id  # its own names: the same page can number its boxes differently
    default = pd.read_csv(paths["work_dir"] / "crops_index.csv")
    crops["inside_pipeline_crop"] = [
        round(inside_share(box, default[(default.paper_id == box.paper_id) & (default.page == box.page)]), 3)
        for box in crops.itertuples()]
    crops["new"] = (crops.det_confidence < moldet["conf_threshold"]) & (crops.inside_pipeline_crop <= NEW_MAX_INSIDE)
    crops.to_csv(index, index=False)
    return crops


def read_new_crops(config: dict, crops: pd.DataFrame) -> pd.DataFrame:
    """MolScribe on the new crops, on this machine's CPU."""
    path = OUT / "lowthr_molscribe_predictions.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"pred_smiles": str})
    from molcensus.recognizers.molscribe_runner import run_molscribe
    from molcensus.weights import fetch_weights

    molscribe = config["recognizers"]["molscribe"]
    checkpoint = fetch_weights(molscribe["hf_repo"], molscribe["checkpoint_file"], config["paths"]["models_dir"],
                               molscribe.get("local_checkpoint", ""))
    predictions = run_molscribe(crops[crops.new], checkpoint, "cpu")
    predictions.to_csv(path, index=False)
    return predictions


# ---- Layer 2b: scaffolds and fragments with * atoms ---------------------------------------------------

def core_of(scaffold: Chem.Mol) -> tuple[Chem.Mol, set[int]]:
    """The scaffold without its * atoms, and the core atoms that carried one (where branching is allowed)."""
    dummies = [atom.GetIdx() for atom in scaffold.GetAtoms() if atom.GetAtomicNum() == 0]
    carriers = {n.GetIdx() for d in dummies for n in scaffold.GetAtomWithIdx(d).GetNeighbors() if n.GetAtomicNum()}
    core = Chem.RWMol(scaffold)
    for index in sorted(dummies, reverse=True):
        core.RemoveAtom(index)
    kept = [i for i in range(scaffold.GetNumAtoms()) if i not in set(dummies)]  # old index of each new atom
    core = core.GetMol()
    core.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(core)  # RDKit's matcher checks ring membership, so it needs fresh ring data
    return core, {new for new, old in enumerate(kept) if old in carriers}


def table_match(scaffold: Chem.Mol, target: Chem.Mol, max_matches: int = 10_000) -> bool:
    """Is target an instance of scaffold? Its core must match exactly, with no extra bonds between core
    atoms, and target may branch off the core only at atoms that carried a * in the scaffold."""
    core, carriers = core_of(scaffold)
    if core.GetNumAtoms() < MIN_CORE_ATOMS or core.GetNumAtoms() > target.GetNumAtoms():
        return False
    params = Chem.SubstructMatchParameters()
    params.maxMatches, params.uniquify = max_matches, False
    for match in target.GetSubstructMatches(core, params):
        inside = set(match)
        branches_ok = all(n.GetIdx() in inside for q, t in enumerate(match) if q not in carriers
                          for n in target.GetAtomWithIdx(t).GetNeighbors())
        extra_bonds = sum(1 for b in target.GetBonds() if b.GetBeginAtomIdx() in inside and b.GetEndAtomIdx() in inside)
        if branches_ok and extra_bonds == core.GetNumBonds():
            return True
    return False


def _first_table_match(target_smiles: str, scaffold_smiles: list[str]) -> int | None:
    target = Chem.MolFromSmiles(target_smiles)
    for k, smiles in enumerate(scaffold_smiles):
        if table_match(Chem.MolFromSmiles(smiles), target):
            return k
    return None


class TableSearch:
    """Runs _first_table_match in a child process with a time limit, because one RDKit substructure search
    cannot be interrupted and can run for many minutes. The child is replaced only after a timeout.
    "spawn", not "fork": forking after MolScribe's torch threads have started can hang on macOS."""

    def __init__(self, timeout_s: float = TABLE_TIMEOUT_S):
        self.timeout_s, self.pool = timeout_s, None

    def __call__(self, target_smiles: str, scaffold_smiles: list[str]):
        """Index of the first scaffold the target is an instance of, None, or "timeout"."""
        if not scaffold_smiles:
            return None
        if self.pool is None:
            self.pool = multiprocessing.get_context("spawn").Pool(1)
            self.pool.apply(len, ([],))  # wait for the child to start, so start-up is not charged to a molecule
        job = self.pool.apply_async(_first_table_match, (target_smiles, scaffold_smiles))
        try:
            return job.get(self.timeout_s)
        except multiprocessing.TimeoutError:
            self.close()
            return "timeout"

    def close(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool = None


# ---- Decision -----------------------------------------------------------------------------------------

def layer2_bucket(new_crop: bool, table: bool, close_read: bool) -> int | None:
    """Bucket from layer 2 alone, or None when nothing fired or the evidence conflicts (left to layer 3).
    An exact reading in a new crop proves a full drawing, and a full drawing outranks a table; but a close
    reading of a pipeline crop (bucket 1) contradicts both a new crop (2) and a table (3)."""
    if close_read:
        return None if (new_crop or table) else 1
    if new_crop:
        return 2
    return 3 if table else None


# ---- Main ---------------------------------------------------------------------------------------------

def readings(config: dict, featurizer: Featurizer) -> pd.DataFrame:
    """Every valid reading: the three readers on the 237 pipeline crops, MolScribe on the new crops."""
    frames = [pd.read_csv(AWS / "predictions" / f"{reader}_predictions.csv", dtype={"pred_smiles": str}).assign(crops="pipeline")
              for reader in READERS]
    low = low_threshold_crops(config)
    frames.append(read_new_crops(config, low).assign(crops="new"))
    table = pd.concat(frames, ignore_index=True)
    table = table.merge(low[["crop_id", "det_confidence"]], on="crop_id", how="left")
    table["paper_id"] = table.paper_id.astype(str)
    table["mol"] = [Chem.MolFromSmiles(s) if isinstance(s, str) and s.strip() else None for s in table.pred_smiles]
    table = table[table.mol.notna()].copy()
    features = [featurizer.featurize(s) for s in table.pred_smiles]
    table["match_key"] = [f.inchikey for f in features]
    table["canonical"] = [f.canonical_smiles for f in features]
    table["fp"] = [f.fingerprint for f in features]
    table["n_dummy"] = [sum(a.GetAtomicNum() == 0 for a in m.GetAtoms()) for m in table.mol]
    return table


def molecules(featurizer: Featurizer) -> pd.DataFrame:
    """The 219 unique valid ground-truth molecules, with found = some reader read it exactly."""
    found = None
    for reader in READERS:
        m = pd.read_csv(AWS / "scored" / f"{reader}_ground_truth_matches.csv", dtype={"paper_id": str})
        m = m[m.gt_valid].groupby(["paper_id", "gt_match_key"]).agg(
            gt_name=("gt_name", lambda s: " / ".join(dict.fromkeys(s))), gt_smiles=("gt_smiles", "first"),
            gt_canonical_smiles=("gt_canonical_smiles", "first"), found=("found_exact", "any"))
        found = m if found is None else found.assign(found=found.found | m.found)
    found = found.reset_index()
    found["fp"] = [featurizer.featurize(s).fingerprint for s in found.gt_smiles]
    return found


def main() -> pd.DataFrame:
    config = load_config(CONFIG)
    scoring = config["scoring"]
    featurizer = Featurizer(scoring["fingerprint_radius"], scoring["fingerprint_bits"])
    gt = molecules(featurizer)
    reads = readings(config, featurizer)
    low = low_threshold_crops(config)

    # Layer 1: labels, and where they appear in the PDF text (a pointer for the judges, not evidence).
    kinds = [label_kind(n.split(" / ")[0]) for n in gt.gt_name]
    gt["label_kind"], gt["scheme"] = [k for k, _ in kinds], [s for _, s in kinds]
    gt["text_pages"], gt["searched"] = "", False
    for paper, rows in gt.groupby("paper_id"):
        needles = {searchable(n.split(" / ")[0]) for n in rows.gt_name} - {None}
        hits = text_pages(config["paths"]["pdf_dir"] / f"{paper}.pdf", needles)
        for i in rows.index:
            needle = searchable(gt.at[i, "gt_name"].split(" / ")[0])
            gt.at[i, "text_pages"] = " ".join(map(str, hits.get(needle, []))) if needle else ""
        gt.loc[rows.index, "searched"] = [searchable(n.split(" / ")[0]) is not None for n in rows.gt_name]

    # 2c candidates: each plain pipeline reading points at its single closest ground-truth molecule.
    closest = []
    for r in reads[(reads.crops == "pipeline") & (reads.n_dummy == 0)].itertuples():
        paper_gt = gt[gt.paper_id == r.paper_id]
        if r.match_key in set(paper_gt.gt_match_key):
            continue  # a correct reading of a ground-truth molecule, not a misread of another one
        sims = DataStructs.BulkTanimotoSimilarity(r.fp, list(paper_gt.fp))
        order = sorted(range(len(sims)), key=sims.__getitem__, reverse=True)
        if len(order) > 1 and sims[order[0]] == sims[order[1]]:
            continue  # a tie: no single molecule it is closest to
        closest.append((paper_gt.index[order[0]], sims[order[0]], r.crop_id, r.tool))
    closest = pd.DataFrame(closest, columns=["gt_index", "tanimoto", "crop_id", "reader"])

    scaffolds = reads[reads.n_dummy > 0]
    search = TableSearch()
    rows = []
    for i, g in gt.iterrows():
        row = {"paper_id": g.paper_id, "gt_match_key": g.gt_match_key, "gt_name": g.gt_name, "found": bool(g.found),
               "label_kind": g.label_kind, "scheme": g.scheme, "text_pages": g.text_pages, "label_searched": g.searched,
               "new_crop": "", "table": "", "table_status": "", "close_read": "", "best_close_tanimoto": float("nan")}
        if not g.found:
            same = reads[(reads.paper_id == g.paper_id) & (reads.crops == "new")]
            hit = same[(same.match_key == g.gt_match_key) | (same.canonical == g.gt_canonical_smiles)]
            if len(hit):
                h = hit.sort_values("det_confidence", ascending=False).iloc[0]
                row["new_crop"] = f"{h.crop_id} (detector confidence {h.det_confidence:.2f})"
            cand = scaffolds[scaffolds.paper_id == g.paper_id]
            status = search(g.gt_smiles, list(cand.pred_smiles))
            row["table_status"] = "timeout" if status == "timeout" else ("match" if status is not None else "no match")
            if isinstance(status, int):
                s = cand.iloc[status]
                n_core = core_of(s.mol)[0].GetNumAtoms()
                row["table"] = f"{s.tool} in {s.crop_id}, a {n_core}-atom core of its {Chem.MolFromSmiles(g.gt_smiles).GetNumHeavyAtoms()} atoms"
            mine = closest[closest.gt_index == i]
            if len(mine):
                best = mine.sort_values("tanimoto", ascending=False).iloc[0]
                row["best_close_tanimoto"] = round(best.tanimoto, 3)
                if best.tanimoto >= TANIMOTO_MIN:
                    row["close_read"] = f"{best.reader} in {best.crop_id}, Tanimoto {best.tanimoto:.2f}"
            row["layer2_bucket"] = layer2_bucket(bool(row["new_crop"]), bool(row["table"]), bool(row["close_read"]))
        rows.append(row)
    search.close()

    evidence = pd.DataFrame(rows)
    evidence["layer2_bucket"] = evidence.layer2_bucket.astype("Int64")
    evidence.to_csv(OUT / "evidence.csv", index=False)

    missed = evidence[~evidence.found]
    print(f"Layer 1: {len(evidence)} molecules, {int(evidence.found.sum())} found, {len(missed)} missed")
    print("  label kinds (missed):", missed.label_kind.value_counts().to_dict())
    print(f"Layer 2a: {len(low)} low-threshold boxes, {int(low.new.sum())} new; "
          f"{int((low.det_confidence >= 0.4).sum())} at >= 0.4, of which {int(((low.det_confidence >= 0.4) & (low.inside_pipeline_crop >= 0.95)).sum())} match a pipeline crop")
    print("  new-crop hits:", int((missed.new_crop != "").sum()))
    print("Layer 2b: table status", missed.table_status.value_counts().to_dict())
    t = missed.best_close_tanimoto
    print(f"Layer 2c: close reads >= {TANIMOTO_MIN}: {int((t >= TANIMOTO_MIN).sum())}; in [0.75, 0.80): {int(((t >= 0.75) & (t < 0.8)).sum())}, "
          f"[0.80, 0.85): {int(((t >= 0.8) & (t < 0.85)).sum())}")
    print("Layer 2 buckets:", missed.layer2_bucket.value_counts(dropna=False).to_dict())
    return evidence


if __name__ == "__main__":
    main()
