"""MolCensus end-to-end run: PDFs -> page images -> molecule crops -> SMILES -> metrics -> reports.

Usage:
    python run_pipeline.py                                  # run every step
    python run_pipeline.py --steps score,report             # rerun only scoring and reports
    python run_pipeline.py --steps public,report            # only the public crop check
    python run_pipeline.py --config configs/default.yaml --papers 1 4
    python run_pipeline.py --steps report --wandb           # publish metrics to W&B

Each step reads the files written by the step before it, so any step can be rerun on its own.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from molcensus import data, detect, render, report, weights  # noqa: E402
from molcensus.config import load_config  # noqa: E402

STEPS = ["prepare", "render", "detect", "recognize", "score", "public", "report"]
log = logging.getLogger("pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config (relative to repo root)")
    parser.add_argument("--steps", default=",".join(STEPS), help=f"comma-separated subset of {STEPS}")
    parser.add_argument("--papers", type=int, nargs="*", help="override the paper list in the config")
    parser.add_argument("--force", action="store_true", help="re-render pages even if images already exist")
    parser.add_argument("--wandb", action="store_true", help="publish metrics (no SMILES) to Weights & Biases")
    parser.add_argument("--wandb-project", default="molcensus", help="W&B project name")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity (defaults to WANDB_ENTITY or your account)")
    return parser.parse_args()


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode="a")],
    )


def enabled_tools(config: dict) -> list[str]:
    return [name for name, settings in config["recognizers"].items() if settings.get("enabled")]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    papers = args.papers or config["papers"]
    steps = [step.strip() for step in args.steps.split(",") if step.strip()]
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise SystemExit(f"Unknown steps: {sorted(unknown)}. Choose from {STEPS}.")

    work_dir, results_dir = paths["work_dir"], paths["results_dir"]
    predictions_dir = results_dir / "predictions"
    scored_dir = results_dir / "scored"
    for folder in (work_dir, predictions_dir, scored_dir):
        folder.mkdir(parents=True, exist_ok=True)
    setup_logging(results_dir / "run_log.txt")
    log.info("Steps %s on papers %s (config: %s)", steps, papers, config["config_file"].name)

    # The private course inputs are only needed for the paper steps, so "public,report" runs on a fresh clone.
    if set(steps) & {"prepare", "render", "detect", "recognize", "score"}:
        ground_truth = data.load_ground_truth(paths["ground_truth_csv"], papers)
        pdfs = data.find_pdfs(paths["pdf_dir"], papers)

    # 1. Record exactly which input files were used.
    if "prepare" in steps:
        data.build_manifest(pdfs, paths["ground_truth_csv"], ground_truth).to_csv(
            results_dir / "data_manifest.csv", index=False
        )

    # 2. Render every PDF page to an image.
    pages_index = work_dir / "pages_index.csv"
    if "render" in steps:
        pages = render.render_pages(pdfs, work_dir / "pages", config["render"]["dpi"], force=args.force)
        pages.to_csv(pages_index, index=False)

    # 3. Detect molecules and save one crop per molecule.
    crops_index = work_dir / "crops_index.csv"
    if "detect" in steps:
        detection = config["detection"]
        crops_dir = work_dir / f"crops_{detection['method']}"
        if detection["method"] == "moldet":
            moldet = detection["moldet"]
            weights_path = weights.fetch_weights(
                moldet["hf_repo"], moldet["weights_file"], paths["models_dir"], moldet.get("local_weights", "")
            )
            crops = detect.detect_with_moldet(
                pd.read_csv(pages_index), weights_path, crops_dir,
                moldet["imgsz"], moldet["conf_threshold"], moldet["pad_px"], moldet["device"],
            )
        elif detection["method"] == "embedded":
            embedded = detection["embedded"]
            crops = detect.detect_embedded_images(pdfs, crops_dir, embedded["min_width"], embedded["min_height"])
        else:
            raise SystemExit(f"Unknown detection method: {detection['method']}")
        crops.to_csv(crops_index, index=False)
        # Crop counts per paper (no file paths) go to results so they can be committed.
        # Start from the ground truth so a paper with no crops shows 0 instead of disappearing.
        ground_truth.groupby("paper_id").size().rename("n_ground_truth").to_frame().join(
            crops.groupby("paper_id").size().rename("n_crops")
        ).fillna({"n_crops": 0}).astype(int)[["n_crops", "n_ground_truth"]].to_csv(results_dir / "crops_per_paper.csv")

    # 4. Run each enabled recognizer and save its predictions.
    if "recognize" in steps:
        for tool in enabled_tools(config):
            settings = config["recognizers"][tool]
            if tool == "molscribe":
                from molcensus.recognizers.molscribe_runner import run_molscribe

                checkpoint = weights.fetch_weights(
                    settings["hf_repo"], settings["checkpoint_file"], paths["models_dir"],
                    settings.get("local_checkpoint", ""),
                )
                predictions = run_molscribe(pd.read_csv(crops_index), checkpoint, settings["device"])
            elif tool == "biominer":
                from molcensus.recognizers.biominer_runner import run_biominer

                predictions = run_biominer(pdfs, settings, work_dir / "biominer_output")
            else:
                log.warning("No runner for recognizer '%s'; skipping", tool)
                continue
            predictions.to_csv(predictions_dir / f"{tool}_predictions.csv", index=False)
            log.info("%s: %d predictions saved", tool, len(predictions))

    # 5. Score every tool that has a predictions file.
    per_paper_file = results_dir / "per_paper_metrics.csv"
    if "score" in steps:
        from molcensus.chem import Featurizer
        from molcensus.metrics import score_tool

        scoring = config["scoring"]
        featurizer = Featurizer(scoring["fingerprint_radius"], scoring["fingerprint_bits"])
        all_metrics = []
        for tool in enabled_tools(config):
            prediction_file = predictions_dir / f"{tool}_predictions.csv"
            if not prediction_file.exists():
                log.warning("No predictions for %s yet; run the recognize step first", tool)
                continue
            predictions = pd.read_csv(prediction_file, dtype={"pred_smiles": str, "pred_label": str})
            scored, truth_matches, per_paper = score_tool(
                predictions, ground_truth, featurizer, scoring["near_match_threshold"],
                scoring.get("bootstrap_rounds", 2000), scoring.get("seed", 0),
            )
            scored.to_csv(scored_dir / f"{tool}_scored_predictions.csv", index=False)
            truth_matches.to_csv(scored_dir / f"{tool}_ground_truth_matches.csv", index=False)
            all_metrics.append(per_paper)
            pooled = per_paper[per_paper["paper_id"] == "ALL"].iloc[0]
            log.info(
                "%s: exact recall %.3f, exact precision %.3f, stereo-blind recall %.3f, mean best Tanimoto %.3f",
                tool, pooled["exact_recall"], pooled["exact_precision"],
                pooled["stereo_blind_recall"], pooled["mean_best_tanimoto"],
            )
        if all_metrics:
            pd.concat(all_metrics, ignore_index=True).to_csv(per_paper_file, index=False)

    # 6. Public crop check: does each tool reproduce its published accuracy on public images?
    if "public" in steps and config.get("public_check"):
        from molcensus.public_crops import run_public_check

        run_public_check(config, paths["models_dir"], results_dir / "public")

    # 7. Side-by-side comparison, JSON summary, RESULTS.md and figures.
    if "report" in steps:
        per_paper = pd.read_csv(per_paper_file, dtype={"paper_id": str}) if per_paper_file.exists() else None
        if per_paper is None and not (results_dir / "public" / "public_metrics.csv").exists():
            raise SystemExit("No metrics found. Run the score or public step first.")
        report.write_reports(per_paper, config, results_dir)

    # 8. Optional: publish the metrics to W&B (python run_pipeline.py --steps report --wandb).
    if args.wandb:
        from molcensus.wandb_log import publish

        publish(results_dir, config, args.wandb_project, args.wandb_entity)

    log.info("Done.")


if __name__ == "__main__":
    main()
