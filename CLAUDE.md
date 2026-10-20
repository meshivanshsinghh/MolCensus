# CLAUDE.md: MolCensus

This file briefs an AI coding agent (Claude Code) on the project. Read it, then `README.md`,
before changing anything.

## What this project is

MolCensus benchmarks optical chemical structure recognition (OCSR) tools on whole scientific
PDFs. It renders PDF pages, detects molecule drawings, converts each one to SMILES with one or
more tools, and scores every tool against a ground-truth list of the molecules in each paper.

- Course: AAI 6630 Applied Computer Vision, Northeastern University, Fall 2026 (XN project).
- Phase 1 deliverables: Assignment 1.2 (this code, graded on structure, readability, README and
  reproducibility; the instructor reads the code but does not run it) and Assignment 1.3
  (a 2–3 page report on the results).
- Owner: Shivansh Singh. The first version was written with AI help. The owner is learning how
  it fits together and extending it, so explain what you change and why.

## Current state (2026-10-19)

- Working code: render (PyMuPDF), detect (MolDetV2 through `ultralytics`, plus an
  embedded-image fallback), recognize (MolScribe on CPU), score (validity, InChIKey exact match,
  stereo-blind match, Tanimoto, per-paper and pooled P/R/F1, Wilson and paper-bootstrap 95%
  intervals), public crop check (MolScribe on public USPTO and ACS images vs. published accuracy),
  report (CSV, JSON, Markdown, PNG), optional W&B upload (metrics only).
- `configs/embedded.yaml` reruns the course papers with embedded-image detection, to compare detectors.
- Wired in but never run: BioMiner end to end. It needs a 24 GB GPU, its three model servers and
  an MLLM API key. It is disabled in `configs/default.yaml`.
- The scoring logic, intervals and public check have unit tests. The full pipeline has not been confirmed end to end on the
  owner's Mac yet; check `results/run_log.txt` for the latest run.

## Commands

```bash
uv venv --python 3.10 && source .venv/bin/activate && uv pip install -r requirements.txt
python run_pipeline.py --papers 16          # quick test on the smallest paper
python run_pipeline.py                      # full run
python run_pipeline.py --steps score,report # rescore saved predictions
python run_pipeline.py --steps report --wandb
pytest -q
```

## How the code fits together

`run_pipeline.py` runs six steps in order. Each step reads files the previous one wrote:

| Step | Module | Output file it hands on |
|---|---|---|
| prepare | `src/molcensus/data.py` | `results/data_manifest.csv` |
| render | `src/molcensus/render.py` | `outputs/work/pages_index.csv` |
| detect | `src/molcensus/detect.py` | `outputs/work/crops_index.csv` |
| recognize | `src/molcensus/recognizers/*_runner.py` | `results/predictions/<tool>_predictions.csv` |
| score | `src/molcensus/metrics.py` (uses `chem.py`) | `results/per_paper_metrics.csv` |
| public | `src/molcensus/public_crops.py` | `results/public/public_metrics.csv` |
| report | `src/molcensus/report.py`, `wandb_log.py` | `results/comparison.csv`, `RESULTS.md`, figure |

Every recognizer returns a DataFrame with the columns in `PREDICTION_COLUMNS`
(`src/molcensus/recognizers/__init__.py`). Scoring and reporting work on that schema alone, so a
new tool needs no changes there.

## Rules

- **Privacy.** Never commit `data/` (the PDFs are copyrighted and the ground truth is private
  sponsor data), `results/predictions/`, `results/scored/`, or any other file containing
  predicted or ground-truth SMILES for the six papers. W&B uploads stay metrics-only.
- **No hardcoded paths or secrets.** Paths go through `configs/default.yaml` and
  `config.resolve_path`. Machine-specific values and keys go in `.env` (never committed).
- **Never invent results.** Do not write numbers for a tool or dataset that has not run.
- **Keep the prediction schema stable.** Change `PREDICTION_COLUMNS` only if every runner and
  the scorer are updated together.
- **Compare molecules by InChIKey,** not canonical SMILES strings (those vary across RDKit versions).
- **One bad item must not stop a run.** Record the error for that item and keep going.
- **Python 3.10.** MolScribe pins `numpy<2`, `timm==0.4.12` and `albumentations==1.1.0`.
- **Plain language** in comments, docs and commit messages. Short sentences, no hype.
- Run `pytest -q` after changes to scoring, and update `README.md` when behavior changes.

## Open work, highest priority first

1. **More public data.** The public crop check (`public_crops.py`) covers USPTO and ACS from
   MolScribe's `real.zip`. CLEF, UOB and Staker are in the same archive and only need adding to
   `public_check.datasets`. The BioVista subset (whole papers with ground truth) is the next step
   for paper-level public results.
2. **A second recognizer** (MolSight or MolNexTR) for a real side-by-side comparison. Follow
   "Add a recognizer" in the README. If its dependencies clash with MolScribe's pins, run it in
   its own environment as a subprocess, like the BioMiner wrapper.
3. **Recognizer registry,** so a new tool needs no edit to `run_pipeline.py`.
4. **Packaging:** `pyproject.toml` and a `molcensus` command-line entry point.
5. **CI:** a GitHub Actions workflow that runs `pytest` (install RDKit so the RDKit test runs).
6. **BioMiner** end to end, once GPU access exists (the course HPC request uses partition
   `courses-gpu`).
7. Later: detection scoring against hand-drawn boxes, R-group enumeration, and a "recall
   ceiling" annotation marking which ground-truth molecules are drawn in full.

## Facts about the six course papers

- 222 ground-truth rows, 220 unique molecules. Paper 19 has 105, paper 4 has 61, paper 2 has 34,
  paper 1 has 17, paper 16 has 4 and paper 6 has 1.
- Papers 4 and 19 define many compounds through R-group tables, letter placeholders and linker
  boxes instead of full drawings, so a crop-by-crop reader cannot reach 100% recall on them.
- Paper 1's 17 molecules are all named drugs drawn in full. It is the cleanest test case.
- Paper 16's 4 molecules are macrocycles drawn with a colored importance heatmap over the atoms.
