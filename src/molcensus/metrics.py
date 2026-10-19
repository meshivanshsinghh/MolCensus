"""Score predicted SMILES against the course ground truth, paper by paper.

Predictions from a paper are compared only with that paper's ground truth. Crops are not
aligned one-to-one with ground-truth entries, so exact matching is done on sets of unique
molecules per paper (the same approach as paper-level precision/recall in the anchor study):

  precision = matched unique predictions / unique valid predictions
  recall    = matched unique ground-truth molecules / unique ground-truth molecules

Metrics reported for each tool and paper, plus an "ALL" row pooled over papers:
  validity_rate         share of predictions RDKit can parse
  exact_*               match on the full InChIKey (stereochemistry must be right)
  stereo_blind_*        match on the first InChIKey block (connectivity only)
  mean_best_tanimoto    for each ground-truth molecule, its highest Tanimoto similarity to
                        any valid prediction from the same paper, averaged
  near_match_recall     share of ground-truth molecules with best Tanimoto >= threshold
  mean_confidence       average tool-reported confidence over valid predictions
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def set_match(predicted_keys: Iterable, truth_keys: Iterable) -> dict:
    """Precision, recall and F1 between two sets of molecule identifiers (empty values are ignored)."""
    predicted = {key for key in predicted_keys if isinstance(key, str) and key}  # drops None and NaN
    truth = {key for key in truth_keys if isinstance(key, str) and key}
    true_positives = len(predicted & truth)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_pred_unique": len(predicted), "n_gt_unique": len(truth), "tp": true_positives,
        "precision": precision, "recall": recall, "f1": f1,
    }


def _featurize_column(smiles: pd.Series, featurizer) -> pd.DataFrame:
    """Featurize each distinct SMILES once and return identifier columns aligned with the input."""
    cache = {value: featurizer.featurize(value) for value in smiles.dropna().unique()}
    empty = featurizer.featurize("")
    features = [cache.get(value, empty) for value in smiles]

    def key_with_fallback(feature, key):
        # A valid molecule without an InChIKey (e.g. it contains "*" atoms) still counts as a
        # distinct, unmatched prediction instead of silently dropping out of the precision count.
        if not feature.valid:
            return None
        return key or f"SMILES:{feature.canonical_smiles}"

    def column(values: list) -> pd.Series:
        # dtype=object keeps None as None (newer pandas would otherwise turn it into NaN).
        return pd.Series(values, index=smiles.index, dtype=object)

    return pd.DataFrame(
        {
            "valid": pd.Series([feature.valid for feature in features], index=smiles.index, dtype=bool),
            "canonical_smiles": column([feature.canonical_smiles for feature in features]),
            "match_key": column([key_with_fallback(feature, feature.inchikey) for feature in features]),
            "skeleton_key": column([key_with_fallback(feature, feature.skeleton_key) for feature in features]),
            "fingerprint": column([feature.fingerprint for feature in features]),
        }
    )


def score_tool(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    featurizer,
    near_match_threshold: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score one tool. Returns (scored predictions, ground-truth match table, per-paper metrics)."""
    tool = predictions["tool"].iloc[0] if len(predictions) else "unknown"

    preds = predictions.join(_featurize_column(predictions["pred_smiles"], featurizer))
    truth = ground_truth.join(
        _featurize_column(ground_truth["gt_smiles"], featurizer).add_prefix("gt_")
    )

    scored_parts, truth_parts, metric_rows = [], [], []
    for paper_id in sorted(truth["paper_id"].unique()):
        paper_preds = preds[preds["paper_id"] == paper_id].copy()
        paper_truth = truth[truth["paper_id"] == paper_id].copy()
        valid_preds = paper_preds[paper_preds["valid"]]

        gt_keys = set(paper_truth["gt_match_key"].dropna())
        gt_skeletons = set(paper_truth["gt_skeleton_key"].dropna())

        # Per-prediction view: does it match any ground-truth molecule, and how close is the nearest one?
        paper_preds["exact_match"] = paper_preds["match_key"].isin(gt_keys)
        paper_preds["stereo_blind_match"] = paper_preds["skeleton_key"].isin(gt_skeletons)
        truth_fps = paper_truth["gt_fingerprint"].tolist()
        best_scores, best_names = [], []
        for fingerprint in paper_preds["fingerprint"]:
            sims = featurizer.similarities(fingerprint, truth_fps)
            if sims:
                best = int(np.argmax(sims))
                best_scores.append(sims[best])
                best_names.append(paper_truth["gt_name"].iloc[best])
            else:
                best_scores.append(np.nan)
                best_names.append("")
        paper_preds["best_tanimoto"] = best_scores
        paper_preds["best_gt_name"] = best_names

        # Per-ground-truth view: was it found exactly, found ignoring stereo, and what was the closest prediction?
        pred_keys = set(valid_preds["match_key"].dropna())
        pred_skeletons = set(valid_preds["skeleton_key"].dropna())
        pred_fps = valid_preds["fingerprint"].tolist()
        paper_truth["found_exact"] = paper_truth["gt_match_key"].isin(pred_keys)
        paper_truth["found_stereo_blind"] = paper_truth["gt_skeleton_key"].isin(pred_skeletons)
        gt_best, gt_best_crop = [], []
        for fingerprint in paper_truth["gt_fingerprint"]:
            sims = featurizer.similarities(fingerprint, pred_fps)
            if sims:
                best = int(np.argmax(sims))
                gt_best.append(sims[best])
                gt_best_crop.append(valid_preds["crop_id"].iloc[best])
            else:
                gt_best.append(0.0)
                gt_best_crop.append("")
        paper_truth["best_tanimoto"] = gt_best
        paper_truth["best_crop_id"] = gt_best_crop

        metric_rows.append(_paper_metrics(tool, str(paper_id), paper_preds, paper_truth, near_match_threshold))
        scored_parts.append(paper_preds)
        truth_parts.append(paper_truth)

    scored = pd.concat(scored_parts, ignore_index=True) if scored_parts else preds.iloc[0:0]
    truth_table = pd.concat(truth_parts, ignore_index=True) if truth_parts else truth.iloc[0:0]
    per_paper = pd.DataFrame(metric_rows)
    per_paper = pd.concat(
        [per_paper, pd.DataFrame([_pooled_metrics(tool, per_paper, scored, truth_table, near_match_threshold)])],
        ignore_index=True,
    )

    # Fingerprint objects are only needed in memory.
    scored = scored.drop(columns=["fingerprint"])
    truth_table = truth_table.drop(columns=["gt_fingerprint"])
    return scored, truth_table, per_paper


def _unique_truth(paper_truth: pd.DataFrame) -> pd.DataFrame:
    """Ground truth with duplicate molecules removed, so a molecule listed twice is counted once."""
    return paper_truth.drop_duplicates(subset="gt_match_key")


def _paper_metrics(tool: str, paper_id: str, preds: pd.DataFrame, truth: pd.DataFrame, threshold: float) -> dict:
    valid = preds[preds["valid"]]
    exact = set_match(valid["match_key"], truth["gt_match_key"])
    blind = set_match(valid["skeleton_key"], truth["gt_skeleton_key"])
    unique_truth = _unique_truth(truth)
    return {
        "tool": tool,
        "paper_id": paper_id,
        "n_gt": len(truth),
        "n_gt_unique": exact["n_gt_unique"],
        "n_predictions": len(preds),
        "n_valid": len(valid),
        "validity_rate": len(valid) / len(preds) if len(preds) else 0.0,
        "n_pred_unique": exact["n_pred_unique"],
        "exact_tp": exact["tp"],
        "exact_precision": exact["precision"],
        "exact_recall": exact["recall"],
        "exact_f1": exact["f1"],
        "stereo_blind_tp": blind["tp"],
        "stereo_blind_precision": blind["precision"],
        "stereo_blind_recall": blind["recall"],
        "stereo_blind_f1": blind["f1"],
        "mean_best_tanimoto": float(unique_truth["best_tanimoto"].mean()) if len(unique_truth) else 0.0,
        "near_match_recall": float((unique_truth["best_tanimoto"] >= threshold).mean()) if len(unique_truth) else 0.0,
        "mean_confidence": float(valid["confidence"].mean()) if valid["confidence"].notna().any() else np.nan,
    }


def _pooled_metrics(tool: str, per_paper: pd.DataFrame, preds: pd.DataFrame, truth: pd.DataFrame, threshold: float) -> dict:
    """Pool counts over all papers (micro-average). Sets stay per paper, so summing true positives is valid."""
    def ratio(numerator: str, denominator: str) -> float:
        total = per_paper[denominator].sum()
        return float(per_paper[numerator].sum() / total) if total else 0.0

    def f1(precision: float, recall: float) -> float:
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    exact_p, exact_r = ratio("exact_tp", "n_pred_unique"), ratio("exact_tp", "n_gt_unique")
    blind_p, blind_r = ratio("stereo_blind_tp", "n_pred_unique"), ratio("stereo_blind_tp", "n_gt_unique")
    unique_truth = truth.drop_duplicates(subset=["paper_id", "gt_match_key"])
    valid = preds[preds["valid"]]
    return {
        "tool": tool,
        "paper_id": "ALL",
        "n_gt": int(per_paper["n_gt"].sum()),
        "n_gt_unique": int(per_paper["n_gt_unique"].sum()),
        "n_predictions": int(per_paper["n_predictions"].sum()),
        "n_valid": int(per_paper["n_valid"].sum()),
        "validity_rate": ratio("n_valid", "n_predictions"),
        "n_pred_unique": int(per_paper["n_pred_unique"].sum()),
        "exact_tp": int(per_paper["exact_tp"].sum()),
        "exact_precision": exact_p,
        "exact_recall": exact_r,
        "exact_f1": f1(exact_p, exact_r),
        "stereo_blind_tp": int(per_paper["stereo_blind_tp"].sum()),
        "stereo_blind_precision": blind_p,
        "stereo_blind_recall": blind_r,
        "stereo_blind_f1": f1(blind_p, blind_r),
        "mean_best_tanimoto": float(unique_truth["best_tanimoto"].mean()) if len(unique_truth) else 0.0,
        "near_match_recall": float((unique_truth["best_tanimoto"] >= threshold).mean()) if len(unique_truth) else 0.0,
        "mean_confidence": float(valid["confidence"].mean()) if valid["confidence"].notna().any() else np.nan,
    }
