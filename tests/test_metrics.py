"""Tests for the scoring logic. Run with: pytest -q"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from molcensus.metrics import score_tool, set_match  # noqa: E402


def test_set_match_counts_unique_keys():
    result = set_match(["A", "A", "B", None, "X"], ["A", "B", "C", "D"])
    assert result["tp"] == 2
    assert result["precision"] == pytest.approx(2 / 3)   # A, B, X predicted; A, B correct
    assert result["recall"] == pytest.approx(2 / 4)


def test_set_match_handles_empty_predictions():
    result = set_match([], ["A"])
    assert result["precision"] == 0.0 and result["recall"] == 0.0 and result["f1"] == 0.0


@dataclass
class FakeFeatures:
    valid: bool
    canonical_smiles: str = None
    inchikey: str = None
    skeleton_key: str = None
    fingerprint: str = None


class FakeFeaturizer:
    """Treats a SMILES-like string as its own key. 'X@' means stereo variant of 'X'. '!' is unparseable."""

    def featurize(self, smiles):
        if not isinstance(smiles, str) or not smiles or "!" in smiles:
            return FakeFeatures(valid=False)
        skeleton = smiles.replace("@", "")
        return FakeFeatures(True, smiles, inchikey=smiles, skeleton_key=skeleton, fingerprint=smiles)

    @staticmethod
    def similarities(fingerprint, others):
        if fingerprint is None:
            return []
        return [1.0 if fingerprint == other else 0.5 if fingerprint.replace("@", "") == other.replace("@", "") else 0.0
                for other in others]


def _predictions(rows):
    return pd.DataFrame(
        [{"tool": "fake", "crop_id": f"c{i}", "paper_id": paper, "page": 1, "pred_smiles": smi,
          "pred_label": "", "confidence": 0.9, "runtime_s": 0.1, "error": ""} for i, (paper, smi) in enumerate(rows)]
    )


def test_score_tool_exact_stereo_blind_and_validity():
    truth = pd.DataFrame({"paper_id": [1, 1, 1, 2], "gt_name": ["a", "b", "c", "d"], "gt_smiles": ["A", "B@", "C", "D"]})
    # Paper 1: A correct, B read without stereo, one invalid output. Paper 2: wrong molecule.
    preds = _predictions([(1, "A"), (1, "B"), (1, "!!"), (2, "Z")])

    scored, matches, per_paper = score_tool(preds, truth, FakeFeaturizer())
    paper1 = per_paper[per_paper["paper_id"] == "1"].iloc[0]
    pooled = per_paper[per_paper["paper_id"] == "ALL"].iloc[0]

    assert paper1["validity_rate"] == pytest.approx(2 / 3)
    assert paper1["exact_tp"] == 1 and paper1["exact_recall"] == pytest.approx(1 / 3)
    assert paper1["stereo_blind_tp"] == 2 and paper1["stereo_blind_recall"] == pytest.approx(2 / 3)
    assert pooled["exact_recall"] == pytest.approx(1 / 4)       # 1 of 4 ground-truth molecules
    assert pooled["exact_precision"] == pytest.approx(1 / 3)    # A, B, Z are the unique valid predictions
    assert bool(matches.set_index("gt_name").loc["a", "found_exact"]) is True
    assert bool(matches.set_index("gt_name").loc["b", "found_stereo_blind"]) is True
    assert math.isnan(scored.set_index("crop_id").loc["c2", "best_tanimoto"])  # invalid prediction


def test_rdkit_featurizer_matches_equivalent_smiles():
    pytest.importorskip("rdkit")
    from molcensus.chem import Featurizer

    featurizer = Featurizer()
    kekule = featurizer.featurize("C1=CC=CC=C1O")
    aromatic = featurizer.featurize("Oc1ccccc1")
    assert kekule.valid and kekule.inchikey == aromatic.inchikey   # same phenol, different SMILES
    assert not featurizer.featurize("C1CC").valid                   # unclosed ring
    l_ala, d_ala = featurizer.featurize("C[C@@H](N)C(=O)O"), featurizer.featurize("C[C@H](N)C(=O)O")
    assert l_ala.inchikey != d_ala.inchikey and l_ala.skeleton_key == d_ala.skeleton_key


# ---- confidence intervals ----------------------------------------------------------------------

from molcensus.metrics import add_intervals, paper_bootstrap, wilson_interval  # noqa: E402


def test_wilson_interval_known_values():
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-3) and high == pytest.approx(0.7634, abs=1e-3)
    low, high = wilson_interval(0, 10)
    assert low == 0.0 and high == pytest.approx(0.2775, abs=1e-3)
    assert all(math.isnan(v) for v in wilson_interval(0, 0))


def test_paper_bootstrap_is_reproducible_and_bounded():
    per_paper = pd.DataFrame(
        {"paper_id": ["1", "2", "3", "ALL"], "exact_tp": [8, 1, 3, 12], "n_gt_unique": [10, 20, 5, 35]}
    )
    first = paper_bootstrap(per_paper, "exact_tp", "n_gt_unique", rounds=500, seed=0)
    second = paper_bootstrap(per_paper, "exact_tp", "n_gt_unique", rounds=500, seed=0)
    assert first == second and 0.0 <= first[0] <= 12 / 35 <= first[1] <= 1.0


def test_add_intervals_puts_bootstrap_only_on_pooled_row():
    truth = pd.DataFrame({"paper_id": [1, 1, 1, 2], "gt_name": ["a", "b", "c", "d"], "gt_smiles": ["A", "B@", "C", "D"]})
    preds = _predictions([(1, "A"), (1, "B"), (2, "Z")])
    _, _, per_paper = score_tool(preds, truth, FakeFeaturizer(), bootstrap_rounds=200)
    pooled = per_paper[per_paper["paper_id"] == "ALL"].iloc[0]
    paper1 = per_paper[per_paper["paper_id"] == "1"].iloc[0]
    assert pooled["exact_recall_ci_low"] <= pooled["exact_recall"] <= pooled["exact_recall_ci_high"]
    assert not math.isnan(pooled["exact_recall_boot_low"]) and math.isnan(paper1["exact_recall_boot_low"])


# ---- public crop check -------------------------------------------------------------------------

from molcensus.public_crops import find_dataset_csv, load_dataset, score_items, summarize  # noqa: E402


def _fake_public_archive(root):
    """real/<name>.csv with file_path values relative to the parent folder, like MolScribe's layout."""
    images = root / "real" / "uspto"
    images.mkdir(parents=True)
    rows = []
    for i in range(6):
        (images / f"img{i}.png").write_bytes(b"png")
        rows.append({"image_id": f"img{i}", "file_path": f"real/uspto/img{i}.png", "SMILES": ["A", "B@", "C"][i % 3]})
    pd.DataFrame(rows).to_csv(root / "real" / "uspto.csv", index=False)


def test_public_loader_resolves_paths_and_samples(tmp_path):
    _fake_public_archive(tmp_path)
    csv = find_dataset_csv(tmp_path, "USPTO")
    items = load_dataset(csv, "uspto", tmp_path, sample_size=4, seed=0)
    assert len(items) == 4 and all(Path(p).exists() for p in items["image_path"])
    assert load_dataset(csv, "uspto", tmp_path, sample_size=4, seed=0)["crop_id"].tolist() == items["crop_id"].tolist()


def test_public_scoring_counts_exact_stereo_and_published_check(tmp_path):
    _fake_public_archive(tmp_path)
    items = load_dataset(find_dataset_csv(tmp_path, "uspto"), "uspto", tmp_path, sample_size=0, seed=0)
    answers = {"A": "A", "B@": "B", "C": "!!"}  # right, stereo lost, unreadable
    preds = pd.DataFrame(
        [{"tool": "fake", "crop_id": r.crop_id, "pred_smiles": answers[r.gt_smiles], "confidence": 0.9,
          "runtime_s": 0.5, "error": ""} for r in items.itertuples()]
    )
    metrics = summarize(score_items(preds, items, FakeFeaturizer()), {"uspto": 0.3})
    row = metrics.iloc[0]
    assert row["n_scored"] == 6
    assert row["exact_accuracy"] == pytest.approx(2 / 6)
    assert row["stereo_blind_accuracy"] == pytest.approx(4 / 6)
    assert row["validity_rate"] == pytest.approx(4 / 6)
    assert bool(row["published_within_ci"]) is True
