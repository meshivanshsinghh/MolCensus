"""Convert other datasets' labels into the MolCensus ground-truth format.

The ground-truth CSV needs the columns paperID (an integer), name of molecule and canonical_SMILES
(see data.py). Extra columns are kept for reference; the loader ignores them.

    PYTHONPATH=src python -m molcensus.convert biovista <labels folder> <out.csv>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def biovista(labels_dir: Path) -> pd.DataFrame:
    """BioVista structure labels -> ground-truth rows, one per labelled structure.

    BioVista stores one <index>_<pdb id>_structure.csv per paper, with columns ligand (the compound
    label; several labels are separated by ";"), smiles, and backbone (filled when the compound is
    defined by a scaffold plus R groups). Format from BioMiner's evaluator (runner/metric_fn.py,
    load_labels_and_results, commit 17c6161). paperID is the BioVista index: 68 for 68_6r8r.pdf.
    """
    frames = []
    for path in sorted(labels_dir.glob("*_structure.csv")):
        name = path.name.removesuffix("_structure.csv")
        table = pd.read_csv(path, dtype=str)
        frames.append(
            pd.DataFrame(
                {
                    "paperID": int(name.split("_")[0]),
                    "name of molecule": table["ligand"].fillna("").str.strip(),
                    "canonical_SMILES": table["smiles"].str.strip(),
                    "pdf_name": name,
                    "biovista_type": table["backbone"].notna().map({True: "scaffold + R group", False: "full"}),
                }
            )
        )
    if not frames:
        raise FileNotFoundError(f"No *_structure.csv files in {labels_dir}")
    rows = pd.concat(frames, ignore_index=True)
    missing = rows["canonical_SMILES"].isna() | (rows["canonical_SMILES"] == "")
    if missing.any():
        log.warning("BioVista: %d label rows have no SMILES and are left out", int(missing.sum()))
    return rows[~missing].reset_index(drop=True)


CONVERTERS = {"biovista": biovista}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", choices=sorted(CONVERTERS))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = CONVERTERS[args.source](args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output, index=False)
    log.info("Wrote %d rows for %d papers to %s", len(rows), rows["paperID"].nunique(), args.output)


if __name__ == "__main__":
    main()
