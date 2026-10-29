"""Public crop check: run recognizers on public molecule images whose correct SMILES are known.

This is the installation check from the 1.1 proposal. Before trusting a tool's score on the
course papers, confirm that it reproduces its published accuracy on public data. The images come
from MolScribe's evaluation archive (`real.zip`, 184 MB, in the MIT-licensed Hugging Face repo
`yujieq/MolScribe`), which holds the CLEF, UOB, USPTO, Staker and ACS benchmark sets.

Unlike the course papers, every image here shows exactly one molecule with one known answer, so
scoring is per image: is the predicted molecule the same as the reference molecule?

The BioVista OCSR set (8,861 crops from 335 papers, Yan et al., 2026) is a separate 61 MB download
from the Google Drive link in BioMiner's BioVista README. Its authors report accuracy on all images and
split into full structures, chiral ones and Markush ones (with * atoms), and they match SMILES with
their own rule (chem.biominer_smiles_match), so BioVista rows are checked against that rule.
"""

from __future__ import annotations

import logging
import math
import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .chem import biominer_smiles_match
from .metrics import featurize_column, wilson_interval

log = logging.getLogger(__name__)

IMAGE_COLUMNS = ("file_path", "image_path", "path", "image", "filename")
SMILES_COLUMNS = ("SMILES", "smiles", "canonical_smiles")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp"}

BIOVISTA_OCSR_URL = "https://drive.usercontent.google.com/download?id=11iCH0j7U9iFgwXahjkWZouNyqATfEPoL&export=download&confirm=t"
# Datasets whose published accuracies used BioMiner's matching rule instead of an exact InChIKey match.
BIOMINER_RULE_DATASETS = {"biovista"}


def fetch_archive(hf_repo: str, archive: str, dest_dir: Path) -> Path:
    """Download and unzip the archive once. Later runs reuse the extracted files."""
    marker = dest_dir / f".{archive}.extracted"
    if marker.exists():
        return dest_dir

    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s from huggingface.co/%s (one time)", archive, hf_repo)
    zip_path = hf_hub_download(repo_id=hf_repo, filename=archive, local_dir=str(dest_dir))
    with zipfile.ZipFile(zip_path) as bundle:
        bundle.extractall(dest_dir)
    marker.write_text(archive)
    return dest_dir


def fetch_biovista_ocsr(dest_dir: Path) -> Path:
    """Download and unpack BioVista's OCSR set once (ocsr/data/*.png, ocsr/labels/gd_SMIELS.csv)."""
    root = dest_dir / "ocsr"
    if (root / "labels" / "gd_SMIELS.csv").exists():
        return root
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / "ocsr.tar.gz"
    log.info("Downloading the BioVista OCSR set (61 MB, one time)")
    urllib.request.urlretrieve(BIOVISTA_OCSR_URL, archive)
    with tarfile.open(archive) as bundle:
        bundle.extractall(dest_dir, filter="data")  # refuses absolute paths and ".." in the archive
    return root


def load_biovista(root: Path, sample_size: int, seed: int) -> pd.DataFrame:
    """BioVista OCSR labels. Row (pdf_name, index, molminer_id "P<page>-<n>") shows image
    data/<pdf_name>_image_molminer_bbox_<page>-<index>.png; rows are matched by name, not by order."""
    table = pd.read_csv(root / "labels" / "gd_SMIELS.csv", index_col=0)
    page = table["molminer_id"].str.extract(r"^P(\d+)-", expand=False)
    image_ids = (table["pdf_name"] + "_" + page + "-" + table["index"].astype(str)).reset_index(drop=True)
    image_paths = pd.Series([
        str(root / "data" / f"{pdf}_image_molminer_bbox_{pg}-{idx}.png")
        for pdf, pg, idx in zip(table["pdf_name"], page, table["index"])
    ])
    return _select_items("biovista", image_ids, image_paths, table["smiles"].reset_index(drop=True), sample_size, seed, "gd_SMIELS.csv")


def find_dataset_csv(root: Path, name: str) -> Path:
    """Locate <name>.csv anywhere under root (case-insensitive), skipping macOS metadata folders."""
    candidates = [
        path for path in root.rglob("*.csv")
        if path.stem.lower() == name.lower() and "__MACOSX" not in path.parts
    ]
    if not candidates:
        available = sorted({p.stem for p in root.rglob("*.csv") if "__MACOSX" not in p.parts})
        raise FileNotFoundError(f"No {name}.csv under {root}. Datasets found: {available}")
    return min(candidates, key=lambda path: len(path.parts))  # the shallowest match


def _first_present(columns, options) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for option in options:
        if option.lower() in lookup:
            return lookup[option.lower()]
    return None


def _resolve_image_paths(relative_paths: pd.Series, bases: list[Path]) -> pd.Series:
    """Pick the base folder under which most of the listed images exist, and build absolute paths."""
    probe = relative_paths.dropna().astype(str).head(20)
    best_base = max(bases, key=lambda base: sum((base / p).exists() for p in probe))
    return relative_paths.astype(str).map(lambda p: str((best_base / p).resolve()))


def load_dataset(csv_path: Path, name: str, root: Path, sample_size: int, seed: int) -> pd.DataFrame:
    """Read one benchmark CSV and return a fixed random sample of (image path, reference SMILES) rows."""
    table = pd.read_csv(csv_path)
    smiles_column = _first_present(table.columns, SMILES_COLUMNS)
    if smiles_column is None:
        raise ValueError(f"{csv_path.name} has no SMILES column (columns: {list(table.columns)})")
    image_ids = table["image_id"].astype(str) if "image_id" in table.columns else table.index.astype(str)

    image_column = _first_present(table.columns, IMAGE_COLUMNS)
    if image_column is not None:
        bases = [csv_path.parent, csv_path.parent.parent, root, root.parent]
        image_paths = _resolve_image_paths(table[image_column], bases)
    else:
        # No path column: match image files to image_id by file name.
        by_stem = {p.stem: p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
        image_paths = image_ids.map(lambda i: str(by_stem[i]) if i in by_stem else "")

    return _select_items(name, image_ids, image_paths, table[smiles_column], sample_size, seed, csv_path.name)


def _select_items(
    name: str, image_ids: pd.Series, image_paths: pd.Series, smiles: pd.Series, sample_size: int, seed: int, source: str
) -> pd.DataFrame:
    """Keep the images that exist, then take a fixed random sample of them (0 = all)."""
    items = pd.DataFrame(
        {
            "dataset": name,
            "image_id": image_ids.values,
            "crop_id": [f"{name}_{i}" for i in image_ids.values],
            "image_path": image_paths.values,
            "gt_smiles": smiles.astype(str).str.strip().values,
        }
    )
    found = items["image_path"].map(lambda p: bool(p) and Path(p).exists())
    if not found.all():
        log.warning("%s: %d of %d listed images not found; they are skipped", name, (~found).sum(), len(items))
    items = items[found].reset_index(drop=True)

    if sample_size and len(items) > sample_size:
        items = items.sample(n=sample_size, random_state=seed).sort_values("crop_id").reset_index(drop=True)
    log.info("%s: %d images selected from %s", name, len(items), source)
    return items


def score_items(predictions: pd.DataFrame, items: pd.DataFrame, featurizer) -> pd.DataFrame:
    """Per-image comparison of the predicted molecule with the reference molecule."""
    merged = items.merge(
        predictions[["tool", "crop_id", "pred_smiles", "confidence", "runtime_s", "error"]], on="crop_id", how="left"
    )
    merged = merged.join(featurize_column(merged["pred_smiles"], featurizer))
    merged = merged.join(featurize_column(merged["gt_smiles"], featurizer).add_prefix("gt_"))

    both_valid = merged["valid"] & merged["gt_valid"]
    merged["exact_match"] = both_valid & (merged["match_key"] == merged["gt_match_key"])
    merged["stereo_blind_match"] = both_valid & (merged["skeleton_key"] == merged["gt_skeleton_key"])
    merged["tanimoto"] = [
        (featurizer.similarities(pred_fp, [gt_fp]) or [0.0])[0] if ok else 0.0
        for pred_fp, gt_fp, ok in zip(merged["fingerprint"], merged["gt_fingerprint"], both_valid)
    ]
    merged["chiral"] = merged["gt_smiles"].str.contains("@", regex=False)
    merged["biominer_rule_match"] = [biominer_smiles_match(p, g) for p, g in zip(merged["pred_smiles"], merged["gt_smiles"])]
    return merged.drop(columns=["fingerprint", "gt_fingerprint"])


def _summary_groups(scored: pd.DataFrame):
    """(tool, dataset, rows). BioVista also gets the full / chiral / Markush splits its authors report,
    defined as in BioMiner's evaluator: Markush = reference has a * atom, chiral = it has an @."""
    for (tool, dataset), group in scored.groupby(["tool", "dataset"]):
        yield tool, dataset, group
        if dataset in BIOMINER_RULE_DATASETS:
            markush = group["gt_smiles"].str.contains("*", regex=False)
            yield tool, f"{dataset}_full", group[~markush]
            yield tool, f"{dataset}_chiral", group[group["chiral"]]
            yield tool, f"{dataset}_markush", group[markush]


def summarize(scored: pd.DataFrame, published_exact: dict) -> pd.DataFrame:
    """One row of metrics per (tool, dataset), with 95% Wilson intervals on the accuracies."""
    rows = []
    for tool, dataset, group in _summary_groups(scored):
        reference = group[group["gt_valid"]]  # images whose reference SMILES RDKit can read
        n = len(reference)
        exact = int(reference["exact_match"].sum())
        blind = int(reference["stereo_blind_match"].sum())
        chiral = reference[reference["chiral"]]
        exact_low, exact_high = wilson_interval(exact, n)
        blind_low, blind_high = wilson_interval(blind, n)
        # BioMiner's evaluator keeps every reference except empty ones and a bare "C", so its denominator differs.
        biominer_rows = group[~group["gt_smiles"].str.lower().isin(["nan", "", "c"])]
        biominer_hits = int(biominer_rows["biominer_rule_match"].sum())
        biominer_low, biominer_high = wilson_interval(biominer_hits, len(biominer_rows))
        rule = "biominer" if dataset.split("_")[0] in BIOMINER_RULE_DATASETS else "exact"
        check_low, check_high = (biominer_low, biominer_high) if rule == "biominer" else (exact_low, exact_high)
        published = published_exact.get(dataset)
        rows.append(
            {
                "tool": tool,
                "dataset": dataset,
                "n_images": len(group),
                "n_scored": n,
                "validity_rate": float(group["valid"].mean()) if len(group) else 0.0,
                "exact_accuracy": exact / n if n else 0.0,
                "exact_ci_low": exact_low,
                "exact_ci_high": exact_high,
                "stereo_blind_accuracy": blind / n if n else 0.0,
                "stereo_blind_ci_low": blind_low,
                "stereo_blind_ci_high": blind_high,
                "n_chiral": len(chiral),
                "chiral_exact_accuracy": float(chiral["exact_match"].mean()) if len(chiral) else math.nan,
                "mean_tanimoto": float(reference["tanimoto"].mean()) if n else 0.0,
                "mean_confidence": float(group["confidence"].mean()) if group["confidence"].notna().any() else math.nan,
                "mean_runtime_s": float(group["runtime_s"].mean()) if group["runtime_s"].notna().any() else math.nan,
                "n_biominer_rule": len(biominer_rows),
                "biominer_rule_accuracy": biominer_hits / len(biominer_rows) if len(biominer_rows) else 0.0,
                "biominer_rule_ci_low": biominer_low,
                "biominer_rule_ci_high": biominer_high,
                "published_exact": published if published is not None else math.nan,
                "published_rule": rule,
                "published_within_ci": (check_low <= published <= check_high) if published is not None else None,
            }
        )
    return pd.DataFrame(rows)


def run_public_check(config: dict, models_dir: Path, out_dir: Path) -> pd.DataFrame | None:
    """Download (once), sample, run every crop-capable enabled recognizer, score, and save the results."""
    from .chem import Featurizer
    from .recognizers import CROP_TOOLS, run_crop_tool

    settings = config["public_check"]
    sample_size, seed = int(settings.get("sample_size", 0)), int(settings.get("seed", 0))
    public_dir = config["paths"]["public_data_dir"]
    frames = []
    for name in settings["datasets"]:
        if name == "biovista":
            frames.append(load_biovista(fetch_biovista_ocsr(public_dir / "biovista"), sample_size, seed))
        else:
            root = fetch_archive(settings["hf_repo"], settings["archive"], public_dir)
            frames.append(load_dataset(find_dataset_csv(root, name), name, root, sample_size, seed))
    items = pd.concat(frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops = items.assign(paper_id=items["dataset"], page=None)

    scoring = config["scoring"]
    featurizer = Featurizer(scoring["fingerprint_radius"], scoring["fingerprint_bits"])
    summaries = []

    for tool in CROP_TOOLS:
        tool_settings = config["recognizers"].get(tool, {})
        if not tool_settings.get("enabled"):
            continue
        try:  # one tool that cannot start must not stop the others
            predictions = run_crop_tool(tool, crops, tool_settings, models_dir)
        except Exception:
            log.exception("%s failed on the public check; continuing with the next tool", tool)
            continue
        predictions.to_csv(out_dir / f"{tool}_predictions.csv", index=False)
        scored = score_items(predictions, items, featurizer)
        scored["image_path"] = scored["image_path"].map(lambda path: os.path.relpath(path, config["repo_root"]))
        scored.to_csv(out_dir / f"{tool}_scored.csv", index=False)
        summaries.append(summarize(scored, settings.get("published_exact", {}).get(tool, {})))

    # BioMiner reads whole PDFs, not single-molecule images, so it has no crop-level run.
    if not summaries:
        log.warning("No crop-capable recognizer is enabled; skipping the public check")
        return None

    metrics = pd.concat(summaries, ignore_index=True)
    metrics.to_csv(out_dir / "public_metrics.csv", index=False)
    for row in metrics.itertuples(index=False):
        rule = "biominer_rule" if row.published_rule == "biominer" else "exact"
        log.info(
            "%s on %s: exact %.3f; %s %.3f [%.3f, %.3f] (published %s) over %d images",
            row.tool, row.dataset, row.exact_accuracy, rule, getattr(row, f"{rule}_accuracy"),
            getattr(row, f"{rule}_ci_low"), getattr(row, f"{rule}_ci_high"),
            "n/a" if math.isnan(row.published_exact) else f"{row.published_exact:.3f}", row.n_scored,
        )
    return metrics
