# Using MolCensus

Setup, configuration, outputs and how to extend the benchmark. For what the metrics mean, see
[methods.md](methods.md).

## Pipeline steps

`run_pipeline.py` runs the steps below in order. Each step writes files that the next one reads,
so any step can be rerun on its own, for example `--steps score,report`.

| Step | Code | Reads | Writes |
|---|---|---|---|
| prepare | `data.py` | ground-truth CSV, PDFs | `results/data_manifest.csv` (SHA-256 of every input) |
| render | `render.py` | PDFs | `outputs/work/pages/<paper>/page_###.png`, `pages_index.csv` |
| detect | `detect.py` | page images | one PNG per molecule, `crops_index.csv`, `results/crops_per_paper.csv` |
| recognize | `recognizers/*_runner.py` | crops (or PDFs for BioMiner) | `results/predictions/<tool>_predictions.csv` |
| score | `chem.py`, `metrics.py` | predictions, ground truth | `results/per_paper_metrics.csv`, `results/scored/` |
| public | `public_crops.py` | MolScribe's public evaluation images | `results/public/public_metrics.csv` and per-image files |
| report | `report.py`, `figures.py` | metrics | `comparison.csv`, `summary_metrics.json`, `RESULTS.md`, figures |

```bash
python run_pipeline.py                                   # every step, all six papers
python run_pipeline.py --papers 16 --steps prepare,render,detect,recognize,score,report
python run_pipeline.py --steps score,report              # rescore saved predictions
python run_pipeline.py --steps public,report             # only the public crop check
```

## Configuration

Everything lives in `configs/default.yaml`: input paths, which papers to run, render resolution,
detector settings, which recognizers are enabled, the public check and scoring settings. Relative
paths are resolved from the repository root, so the same file works on any machine.
Machine-specific values (only needed for BioMiner) go in a `.env` file copied from `.env.example`.
A config can start with `extends: default.yaml` and list only what it changes, as
`configs/embedded.yaml` does.

If MolDetV2 cannot be installed, set `detection.method: embedded`. That saves the raster images
embedded in each PDF instead. It needs no model, but it misses structures drawn as vector graphics
and can hand MolScribe whole multi-molecule figures, so expect lower scores.

## Outputs

| File | Contents |
|---|---|
| `results/RESULTS.md` | Readable summary with all tables and figures |
| `results/comparison.csv` | Pooled metrics side by side, one column per tool, with interval bounds |
| `results/per_paper_metrics.csv` | Every metric and outcome count for every tool and paper, plus an `ALL` row |
| `results/summary_metrics.json` | The same numbers in JSON, with run settings |
| `results/crops_per_paper.csv` | Detected crops compared with the ground-truth count, per paper |
| `results/public/public_metrics.csv` | Public check: accuracy per tool and set, interval, published value, pass or fail |
| `results/public/molscribe_*.csv` | Per-image predictions and match flags for the public images |
| `results/figures/*.png` | Overview, recall by paper, and the outcome breakdown for each tool |
| `results/data_manifest.csv` | SHA-256 hash of every input file |
| `results/environment.json` | Python and package versions, git commit, whether the tree was clean |
| `results/run_log.txt` | Log of every run (local only, git-ignored) |
| `results/predictions/`, `results/scored/` | Per-molecule files with SMILES for the course papers (local only, git-ignored) |

Page images and crops go to `outputs/work/` and are not committed.

## Detection comparison

`configs/embedded.yaml` repeats the course-paper run with one change: molecules come from the raster
images embedded in each PDF instead of MolDetV2 detections. Comparing the two runs shows how much of
the score depends on detection alone. This run has not been done yet.

```bash
python run_pipeline.py --config configs/embedded.yaml --steps prepare,detect,recognize,score,report
```

Results go to `results/embedded_detection/`.

## Add a recognizer

1. Create `src/molcensus/recognizers/<tool>_runner.py` with a function that takes the crops table
   (or the PDFs, for a whole-document tool) and returns a DataFrame with exactly the columns in
   `PREDICTION_COLUMNS` (`recognizers/__init__.py`). Catch errors per item and record them in the
   `error` column, so one bad crop never stops a run.
2. Add a block for the tool under `recognizers:` in `configs/default.yaml` with `enabled: true` and
   its settings (weights repository, device and so on).
3. Add a branch for the tool in the `recognize` step of `run_pipeline.py`.
4. Add its dependencies to `requirements.txt`, or run it in its own environment as a subprocess
   (like the BioMiner wrapper) if they clash with MolScribe's pins.

Scoring and reporting pick up every tool that has a predictions file, so the new tool gets its own
column in `comparison.csv` and its own bars in the figures with no further changes.

## Add a dataset

The paper-level loader in `data.py` expects a CSV with `paperID`, `name of molecule` and
`canonical_SMILES` columns, and PDFs named `<paperID>.pdf`. Any set of papers with that layout runs
as is: point `paths` in a new config at it. For more public single-molecule sets, add `clef`, `uob`
or `staker` to `public_check.datasets`; they are already in the downloaded archive.

## Publish to Weights & Biases (optional)

No model is trained here, so W&B is only a place to browse and share results. Not used yet.

```bash
wandb login                                   # once
python run_pipeline.py --steps report --wandb # metrics, tables and figures; no SMILES
```

One run is created in the `molcensus` project (`--wandb-project` and `--wandb-entity` change this).

## Enable BioMiner (needs a GPU)

1. Clone https://github.com/jiaxianyan/BioMiner and follow its installation steps (two conda
   environments, model weights).
2. Put the MLLM API key in BioMiner's `BioMiner/config/default_open_source.yaml`.
3. Start its MinerU, MolDet and OCSR servers.
4. Copy `.env.example` to `.env` and set `BIOMINER_DIR` and `BIOMINER_PYTHON`.
5. Set `recognizers.biominer.enabled: true`.

The wrapper calls `example_open_source.py --pdf <paper> --output_dir <dir>` for each paper and reads
every output table that has a `smiles` column.

## Troubleshooting

- **`rdkit` import errors after installing MolScribe.** Some MolScribe versions pull in the older
  `rdkit-pypi` package. Run `pip uninstall -y rdkit-pypi && pip install --force-reinstall rdkit`.
- **NumPy 2 errors.** Make sure `numpy<2` is installed, as pinned in `requirements.txt`.
- **`No module named 'torch._six'` when importing MolScribe.** Its pinned `timm` predates PyTorch 2.
  Run `pip install "torch==1.13.1" "torchvision==0.14.1"`.
- **401/403 downloading MolDetV2.** The repository was not gated as of October 2026. If that changes,
  accept the license at https://huggingface.co/UniParser/MolDetv2 and run `hf auth login`, or set
  `detection.method: embedded`.
- **MolScribe returns `<invalid>` for drawings with Me, Ph or OMe labels.** This is the abbreviation
  bug in [methods.md](methods.md#a-molscribe-bug-the-check-caught). `run_molscribe` patches it
  automatically. If it raises "abbreviation patch no longer applies", the installed MolScribe
  changed and the patch needs a look.
- **No GPU.** Every enabled stage runs on CPU. Set `device: mps` or `device: cuda` in the config
  only if your torch build supports it.
