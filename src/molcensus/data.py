"""Load the course ground truth and check that the benchmark PDFs are present."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Column names in the course file -> names used everywhere in this pipeline.
GROUND_TRUTH_COLUMNS = {
    "paperID": "paper_id",
    "name of molecule": "gt_name",
    "canonical_SMILES": "gt_smiles",
}


def load_ground_truth(csv_path: Path, papers: list[int]) -> pd.DataFrame:
    """Read the ground-truth CSV and keep only the papers being benchmarked."""
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found at {csv_path}. Copy the course CSV into data/ (see data/README.md)."
        )
    table = pd.read_csv(csv_path)
    missing = set(GROUND_TRUTH_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"Ground-truth CSV is missing columns: {sorted(missing)}")

    table = table.rename(columns=GROUND_TRUTH_COLUMNS)[list(GROUND_TRUTH_COLUMNS.values())]
    table["paper_id"] = table["paper_id"].astype(int)
    table["gt_name"] = table["gt_name"].astype(str).str.strip()
    table["gt_smiles"] = table["gt_smiles"].astype(str).str.strip()
    table = table[table["paper_id"].isin(papers)].reset_index(drop=True)

    log.info("Loaded %d ground-truth molecules across %d papers", len(table), table["paper_id"].nunique())
    return table


def find_pdfs(pdf_dir: Path, papers: list[int]) -> dict[int, Path]:
    """Map each paper ID to its PDF (expected name: <paper_id>.pdf)."""
    pdfs = {}
    for paper_id in papers:
        pdf_path = pdf_dir / f"{paper_id}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing {pdf_path}. Put the course PDFs in {pdf_dir} (see data/README.md).")
        pdfs[paper_id] = pdf_path
    return pdfs


def sha256_of(path: Path) -> str:
    """SHA-256 of a file, so anyone rerunning the benchmark can confirm they have identical inputs."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(pdfs: dict[int, Path], ground_truth_csv: Path, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """One row per input file with its hash and the number of ground-truth molecules it covers."""
    counts = ground_truth.groupby("paper_id").size()
    rows = [
        {
            "file": pdf_path.name,
            "paper_id": paper_id,
            "size_bytes": pdf_path.stat().st_size,
            "sha256": sha256_of(pdf_path),
            "n_ground_truth_molecules": int(counts.get(paper_id, 0)),
        }
        for paper_id, pdf_path in pdfs.items()
    ]
    rows.append(
        {
            "file": ground_truth_csv.name,
            "paper_id": "",
            "size_bytes": ground_truth_csv.stat().st_size,
            "sha256": sha256_of(ground_truth_csv),
            "n_ground_truth_molecules": len(ground_truth),
        }
    )
    return pd.DataFrame(rows)
