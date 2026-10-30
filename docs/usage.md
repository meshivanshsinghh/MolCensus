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

## Readers in their own environment (MolSight, MolGlyph)

MolSight and MolGlyph pin packages that clash with MolScribe's, so each runs in its own Python
environment. `recognizers/subprocess_reader.py` writes the crop list to a CSV, runs
`recognizers/<tool>_worker.py` with that tool's Python from the tool's repo folder, and reads the
SMILES back. Both readers work on the course crops and in the public check, like MolScribe.

**MolSight** (base checkpoint, `Robert-zwr/MolSight`, downloaded into `models/` on first use):

```bash
git clone https://github.com/hustvl/MolSight.git tools/MolSight && git -C tools/MolSight checkout 86e979b
uv venv --python 3.10 tools/molsight-env
uv pip install --python tools/molsight-env/bin/python -r tools/MolSight/requirements.txt "datasets>=3" "numpy<2"
```

Its requirements are unpinned. Without the last two pins the resolver picks `datasets` 1.1.1, which
fails with current pyarrow, and NumPy 2, which the old `rdkit-pypi` build cannot load cleanly.
Then set `recognizers.molsight.enabled: true`. On 5 course crops on a Mac CPU it took 0.6 to 1.7 s
per crop.

**MolGlyph** runs inside BioMiner's environment through BioMiner's own `MolGlyph` class and the same
caption-to-SMILES steps BioMiner uses, so it needs the BioMiner clone and `BIOMINER_DIR` /
`BIOMINER_PYTHON` in `.env`. Its weights are gated: accept the terms at
https://huggingface.co/jiaxianustc/MolGlyph and run `hf auth login`. Then set
`recognizers.molglyph.enabled: true`. It has not been run on this Mac (no BioMiner environment here).

## Add a recognizer

1. For a crop reader, add its name to `CROP_TOOLS` in `recognizers/__init__.py`. If it can run in
   this environment, add a branch to `run_crop_tool`; otherwise write `recognizers/<tool>_worker.py`
   following the contract in `subprocess_reader.py`. For a whole-document tool, write
   `recognizers/<tool>_runner.py` and add a branch in the `recognize` step of `run_pipeline.py`.
   Every runner returns exactly the columns in `PREDICTION_COLUMNS`. Catch errors per item and record
   them in the `error` column, so one bad crop never stops a run.
2. Add a block for the tool under `recognizers:` in `configs/default.yaml` with `enabled: true` and
   its settings (weights repository, device and so on).
3. Add its dependencies to `requirements.txt`, or give it its own environment if they clash with
   MolScribe's pins.

Scoring and reporting pick up every tool that has a predictions file, so the new tool gets its own
column in `comparison.csv` and its own bars in the figures with no further changes.

## Add a dataset

The paper-level loader in `data.py` expects a CSV with `paperID`, `name of molecule` and
`canonical_SMILES` columns, and PDFs named `<paperID>.pdf` or `<paperID>_<anything>.pdf`. Any set of
papers with that layout runs as is: point `paths` in a new config at it.

`src/molcensus/convert.py` turns other label formats into that CSV:

```bash
PYTHONPATH=src python -m molcensus.convert biovista <folder of *_structure.csv> data/biovista/ground_truth.csv
```

BioVista papers keep BioVista's names, `<index>_<pdb id>.pdf` (for example `68_6r8r.pdf`), because
BioMiner's evaluator reads the PDB ID from the file name; the index becomes `paperID`. The BioVista
structure labels go in `data/biovista/labels/`. Use the "Bioactivity Triplet Extraction" link in
BioMiner's main README: the one in its `BioVista/README.md` is dead (both links are in
[methods.md](methods.md)).

For more public single-molecule sets, add `clef`, `uob` or `staker` (already in the downloaded
archive) or `biovista` (downloaded on first use) to `public_check.datasets`.

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
2. Give it an MLLM. MolCensus uses Gemini through Gemini's OpenAI-compatible endpoint.
   `scripts/aws/biominer_gemini.py --keys <keys file> --biominer <clone> --check` writes
   `BioMiner/config/molcensus_free.yaml` (free-tier key, public papers) and `molcensus_paid.yaml`
   (paid-tier key, course papers) inside the clone, both with `gemini-3.5-flash`, then sends one test
   request per key shaped like BioMiner's own. `run_all.sh` sends the course PDFs only when its
   permission flag file exists. The keys file and the configs stay outside this repo. BioMiner picks
   its Gemini prompts for any model name without "gpt".
3. Start its MinerU, MolDet and OCSR servers.
4. Copy `.env.example` to `.env` and set `BIOMINER_DIR` and `BIOMINER_PYTHON`.
5. Set `recognizers.biominer.enabled: true` and `tier: free` or `tier: paid` (that picks
   `molcensus_<tier>.yaml`; `configs/aws.yaml` sets it for the course papers with `course_tier`).

Each paper's output folder gets `run_info.json` with the model names, the key tier, the number of
MLLM calls lost after all retries, and the input, output and total tokens of the answered calls.
Unpatched BioMiner tries each call once and carries on with an empty answer, so check the lost-call
number: a rate-limited run otherwise looks normal, just with fewer molecules. A paper that already has
`run_info.json` counts as finished: a rerun reuses it (`--force` reruns it). On SIGTERM, for example
from a time limit, the wrapper stops the paper in progress and saves the predictions of the finished
ones.

The wrapper calls `example_open_source.py --pdf <paper> --output_dir <dir>` for each paper and reads
two tables from `<dir>/extraction_<model>/<name>/`:

- `<name>_structure_merged_output_*.csv` (every structure linked to a label) becomes
  `results/predictions/biominer_predictions.csv`.
- `<name>_merge_*.csv` (only structures with a bioactivity value) becomes
  `results/predictions/biominer_bioactivity_predictions.csv`, scored as its own tool.

If BioMiner fails after writing the structure table (its final merge step can raise), the structure
rows are kept and their `error` column starts with `partial run:`. A missing or unreadable table
becomes one error row for that paper.

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
