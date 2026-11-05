"""Label converters: their output must load as MolCensus ground truth. Run with: pytest -q"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from molcensus import convert  # noqa: E402
from molcensus.data import find_pdfs, load_ground_truth  # noqa: E402


def test_biovista_labels_load_as_ground_truth(tmp_path):
    # Columns as BioMiner's evaluator reads them: ligand, smiles, backbone (filled for scaffold + R group).
    pd.DataFrame({"ligand": ["4x;4y", "10", None], "smiles": ["CCO", "c1ccccc1C", None],
                  "backbone": [None, "c1ccccc1[*]", None]}).to_csv(tmp_path / "68_6r8r_structure.csv", index=False)
    pd.DataFrame({"ligand": ["1"], "smiles": ["CCN"], "backbone": [None]}).to_csv(tmp_path / "5_6un3_structure.csv", index=False)

    rows = convert.biovista(tmp_path)
    assert len(rows) == 3  # the row without SMILES is left out
    assert rows.set_index("name of molecule").loc["10", "biovista_type"] == "scaffold + R group"
    assert rows.set_index("name of molecule").loc["4x;4y", "pdf_name"] == "68_6r8r"

    rows.to_csv(tmp_path / "gt.csv", index=False)
    truth = load_ground_truth(tmp_path / "gt.csv", [5, 68])
    assert sorted(truth["paper_id"].unique()) == [5, 68] and len(truth) == 3


def test_find_pdfs_accepts_biovista_names(tmp_path):
    (tmp_path / "1.pdf").write_bytes(b"%PDF")
    (tmp_path / "68_6r8r.pdf").write_bytes(b"%PDF")
    assert {k: v.name for k, v in find_pdfs(tmp_path, [1, 68]).items()} == {1: "1.pdf", 68: "68_6r8r.pdf"}
