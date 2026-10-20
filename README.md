# MolCensus

**Count every molecule drawn in a paper, then check which ones each tool finds.**

MolCensus is a benchmarking framework for optical chemical structure recognition (OCSR) on whole
scientific PDFs. You give it PDFs and a ground-truth list of the molecules drawn in them. It
renders the pages, finds each structure drawing, converts it to SMILES with the tools you choose,
and scores every tool against the ground truth with one fixed set of metrics.

It started as Assignment 1.2 (benchmarking code) for AAI 6630 Applied Computer Vision at
Northeastern University, Fall 2026, as part of the XN project on extracting chemical structures
from papers. Author: Shivansh Singh.

## Why it exists

Published OCSR scores mostly come from cropped patent images. Journal papers are harder: they
draw molecules inside schemes and tables, define compound series through R-group tables, and use
colored or annotated figures. MolScribe, for example, reaches 92.6% exact match on USPTO patent
images but 71.9% on ACS journal figures (Tang et al., 2026). MolCensus measures tools on the
papers themselves, at the paper level, so different tools can be compared on the same footing.

## Status

| Tool | Role | Status |
|---|---|---|
| MolDetV2 (BioMiner's detector) | Finds molecule drawings on page images | Runs on CPU |
| MolScribe (Qian et al., 2023) | Converts one cropped drawing to SMILES, with a confidence score | Runs on CPU |
| BioMiner end to end (Yan et al., 2026) | Whole PDF to labeled SMILES | Wired in and **disabled**. Needs one 24 GB GPU; waiting on course HPC access. |

Every tool is also checked on public data first: MolScribe runs on a fixed sample of public USPTO
patent images and ACS journal images with known SMILES, and its accuracy is compared with the
published number (see "Public crop check").

No numbers are reported for a tool until it has actually run.

## Quick start

```bash
git clone https://github.com/<your-username>/MolCensus.git
cd MolCensus

# Python 3.10 environment (uv fetches Python 3.10 for you; install it with `brew install uv`)
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import rdkit, fitz, ultralytics, torch, molscribe; print('environment OK')"

# Put the six course PDFs in data/pdfs/ and the ground-truth CSV in data/ (see data/README.md)

python run_pipeline.py --papers 16    # quick test on the smallest paper
python run_pipeline.py                # full run: six papers plus the public crop check
```

The first full run downloads the public check images once (184 MB, into `data/public/`).

Results land in `results/`. Start with `results/RESULTS.md`.

Using conda instead of uv: `conda create -n molcensus python=3.10 -y`, `conda activate molcensus`,
`pip install -r requirements.txt`.

## How it works

```
PDFs ──render──▶ page images ──detect──▶ molecule crops ──recognize──▶ SMILES + confidence ──score──▶ metrics ──┐
                 (PyMuPDF)               (MolDetV2)                    (MolScribe; BioMiner        (RDKit)         │
                                                                        reads PDFs directly)                       ├──report──▶ tables, JSON, figures
public images with known SMILES ──public──▶ per-image predictions and accuracy, compared with published ───────────┘   (optional: W&B)
```

Each step writes files that the next step reads, so any step can be rerun on its own
(`--steps score,report`, for example).

| Step | Code | Reads | Writes |
|---|---|---|---|
| prepare | `data.py` | ground-truth CSV, PDFs | `results/data_manifest.csv` (SHA-256 of every input) |
| render | `render.py` | PDFs | `outputs/work/pages/<paper>/page_###.png`, `pages_index.csv` |
| detect | `detect.py` | page images | one PNG per molecule, `crops_index.csv`, `results/crops_per_paper.csv` |
| recognize | `recognizers/*_runner.py` | crops (or PDFs for BioMiner) | `results/predictions/<tool>_predictions.csv` |
| score | `chem.py`, `metrics.py` | predictions, ground truth | `results/per_paper_metrics.csv`, `results/scored/` |
| public | `public_crops.py` | MolScribe's public evaluation images | `results/public/public_metrics.csv` and per-image files |
| report | `report.py`, `wandb_log.py` | metrics | `comparison.csv`, `summary_metrics.json`, `RESULTS.md`, figures |

## Repository layout

```
.
├── run_pipeline.py              # entry point: runs every step in order
├── configs/default.yaml         # all paths and settings (nothing is hardcoded in the code)
├── configs/embedded.yaml        # same run with embedded-image detection (detection comparison)
├── requirements.txt             # dependencies
├── .env.example                 # template for machine-specific paths (BioMiner only)
├── CLAUDE.md                    # project brief for AI coding agents (Claude Code)
├── data/                        # course PDFs and ground truth go here (not committed)
│   └── README.md
├── src/molcensus/
│   ├── config.py                # loads the YAML, resolves paths relative to the repo root
│   ├── data.py                  # ground-truth loading, PDF lookup, input manifest
│   ├── render.py                # PDF pages -> PNG
│   ├── detect.py                # MolDetV2 detection and cropping (+ embedded-image fallback)
│   ├── weights.py               # downloads model weights from Hugging Face
│   ├── recognizers/
│   │   ├── __init__.py          # the prediction schema every tool must follow
│   │   ├── molscribe_runner.py  # MolScribe: SMILES + confidence per crop
│   │   └── biominer_runner.py   # BioMiner: runs its CLI per PDF, parses its SMILES tables
│   ├── chem.py                  # RDKit: validity, InChIKeys, Morgan fingerprints
│   ├── metrics.py               # all metrics, per paper and pooled
│   ├── public_crops.py          # public crop check: download, sample, run, per-image scoring
│   ├── report.py                # comparison table, JSON, RESULTS.md, figures
│   └── wandb_log.py             # optional upload of metrics to Weights & Biases
├── tests/test_metrics.py        # unit tests for the scoring logic
└── results/                     # committed outputs (SMILES-free)
```

## Configuration

Everything lives in `configs/default.yaml`: input paths, which papers to run, render resolution,
detector settings, which recognizers are enabled, and scoring settings. Relative paths are
resolved from the repository root, so the same config works on any machine. Machine-specific
values (only needed for BioMiner) go in a `.env` file copied from `.env.example`.

**Fallback detector.** If MolDetV2 cannot be installed, set `detection.method: embedded`. That
saves the raster images embedded in each PDF instead. It needs no model, but it misses structures
drawn as vector graphics and can hand MolScribe whole multi-molecule figures, so expect lower scores.

## Outputs

| File | Contents |
|---|---|
| `results/RESULTS.md` | Readable summary: overall comparison and recall by paper |
| `results/comparison.csv` | Side-by-side pooled metrics, one column per tool |
| `results/per_paper_metrics.csv` | Every metric for every tool and paper, plus an `ALL` row |
| `results/summary_metrics.json` | Same numbers in JSON, with run settings |
| `results/crops_per_paper.csv` | Detected crops compared with the ground-truth count, per paper |
| `results/figures/recall_by_paper.png` | Exact and stereo-blind recall by paper, with 95% intervals |
| `results/public/public_metrics.csv` | Public crop check: accuracy per tool and dataset, 95% interval, published value, pass/fail |
| `results/public/molscribe_predictions.csv`, `molscribe_scored.csv` | Per-image predictions and match flags for the public images |
| `results/figures/public_check.png` | Public accuracy with 95% intervals next to the published numbers |
| `results/data_manifest.csv` | SHA-256 hash of every input file |
| `results/environment.json` | Python and package versions, git commit, and whether the tree was clean |
| `results/run_log.txt` | Log of every run |
| `results/predictions/<tool>_predictions.csv` | Raw tool output: crop ID, paper, page, SMILES, label, confidence, runtime, errors (local only) |
| `results/scored/<tool>_scored_predictions.csv` | Each prediction with validity, InChIKey, match flags and nearest ground-truth molecule (local only) |
| `results/scored/<tool>_ground_truth_matches.csv` | Each ground-truth molecule: found exactly, found ignoring stereo, best Tanimoto, closest crop (local only) |

Page images and crops go to `outputs/work/` and are not committed.

## Privacy

The six course PDFs are copyrighted, and their ground truth is private sponsor data. Neither is
committed. The files marked "local only" contain predicted or ground-truth SMILES for those
papers, so they are written to `results/` on your machine but are git-ignored. Everything
committed for the course papers is SMILES-free: metrics, counts, figures and run metadata. The
public check files in `results/public/` are committed, because their images and SMILES are public.
The W&B upload follows the same rule.

## Metrics

Crops are not matched one-to-one with ground-truth entries. Scoring therefore compares the set of
unique predicted molecules from a paper with the set of unique ground-truth molecules from that
same paper, as in the anchor study's paper-level precision and recall (Gulluoglu et al., 2026).

- **Validity rate:** share of predictions that RDKit can parse into a molecule.
- **Exact precision / recall / F1:** matching on the full InChIKey, so stereochemistry must also be correct. Molecules are compared by InChIKey rather than canonical SMILES because canonical SMILES strings can differ between RDKit versions.
- **Stereo-blind precision / recall / F1:** matching on the first InChIKey block, which encodes connectivity only (Heller et al., 2015). The difference from exact recall measures stereochemistry errors.
- **Mean best Tanimoto:** for each ground-truth molecule, the highest Tanimoto similarity (Morgan fingerprints, radius 2, 2048 bits) to any valid prediction from the same paper, averaged. This gives partial credit for near misses.
- **Near-match recall:** share of ground-truth molecules whose best Tanimoto is at least 0.85 (set in the config).
- **Mean confidence:** the tool's own confidence, averaged over valid predictions.

**Uncertainty.** With about 220 molecules in six papers, every recall and precision comes with two
95% intervals. The Wilson interval treats molecules as independent. The paper bootstrap (pooled row
only) resamples the six papers with replacement and recomputes the pooled rate; it is wider and more
honest, because molecules from the same paper tend to succeed or fail together.

Precision counts unique valid predictions, so a molecule drawn twice in a paper counts once.
Invalid outputs are not in the precision denominator; they are reported separately through the
validity rate. The `ALL` row pools counts over papers (micro-average).

## Public crop check

Before trusting a tool's score on the course papers, MolCensus checks that the installed tool
reproduces its published accuracy on public data. This was the installation check planned in the
setup proposal.

- **Data:** MolScribe's evaluation archive `real.zip` (184 MB) from the MIT-licensed Hugging Face
  repository `yujieq/MolScribe`. It holds the CLEF, UOB, USPTO, Staker and ACS sets; CLEF, UOB and
  USPTO originally come from the `Kohulan/OCSR_Review` collection, and ACS was collected by the
  MolScribe authors. It is downloaded once into `data/public/` and never committed.
- **Default sets:** USPTO (patent drawings) and ACS (journal figures, the closest match to the course
  papers), 300 images each as a fixed random sample (`public_check` in the config).
- **Scoring:** each image shows one molecule with one known answer, so accuracy is per image: exact
  match on the full InChIKey, stereo-blind match, accuracy on chiral molecules, validity, Tanimoto,
  confidence and seconds per image, each accuracy with a 95% Wilson interval.
- **Pass criterion:** the published accuracy (MolScribe: 92.6% on USPTO, 71.9% on ACS; Qian et al.,
  2023) falls inside our 95% interval. InChIKey matching can be slightly more lenient than the
  canonical-SMILES matching used in the paper (it treats some tautomers as identical), so small
  differences are expected.

```bash
python run_pipeline.py --steps public,report   # only the public check
```

## Detection comparison

`configs/embedded.yaml` repeats the course-paper run with one change: molecules come from the raster
images embedded in each PDF instead of MolDetV2 detections. Comparing the two runs shows how much of
the final score depends on detection alone.

```bash
python run_pipeline.py --config configs/embedded.yaml --steps prepare,detect,recognize,score,report
```

Results go to `results/embedded_detection/`.

## Extending MolCensus

### Add a recognizer

1. Create `src/molcensus/recognizers/<tool>_runner.py` with a function that takes the crops table
   (or the PDFs, for a whole-document tool) and returns a DataFrame with exactly the columns in
   `PREDICTION_COLUMNS` (`recognizers/__init__.py`). Catch errors per item and record them in the
   `error` column, so one bad crop never stops a run.
2. Add a block for the tool under `recognizers:` in `configs/default.yaml` with `enabled: true`
   and its settings (weights repo, device, and so on).
3. Add a branch for the tool in the `recognize` step of `run_pipeline.py`.
4. Add its dependencies to `requirements.txt` (or document a separate environment if they conflict).

Scoring and reporting pick up every enabled tool that has a predictions file, so the new tool gets
its own column in `comparison.csv` and its own bars in the figure with no further changes.

### Add a dataset

The paper-level loader in `data.py` expects a CSV with `paperID`, `name of molecule` and
`canonical_SMILES` columns, and PDFs named `<paperID>.pdf`. Any set of papers with that layout
runs as is: point `paths` in a new config at it. Crop datasets (one image per molecule with its
SMILES) need a crop loader and crop-level scoring, which are the first roadmap item.

## Publishing results to Weights & Biases (optional)

No model is trained here, so W&B is used only as a place to browse and share the results.

```bash
wandb login                                   # once
python run_pipeline.py --steps report --wandb # uploads metrics, tables and the figure
```

One run is created in the `molcensus` project (`--wandb-project` and `--wandb-entity` change
this). Its summary holds the pooled metrics for each tool. The per-paper metrics, comparison
table, crop counts and recall figure are attached. No SMILES are uploaded.

## Enabling BioMiner (needs a GPU)

1. Clone https://github.com/jiaxianyan/BioMiner and follow its installation steps (two conda environments, model weights).
2. Put the MLLM API key in BioMiner's `BioMiner/config/default_open_source.yaml`.
3. Start its MinerU, MolDet and OCSR servers.
4. Copy `.env.example` to `.env` and set `BIOMINER_DIR` and `BIOMINER_PYTHON`.
5. Set `recognizers.biominer.enabled: true`.

The wrapper calls `example_open_source.py --pdf <paper> --output_dir <dir>` for each paper and
reads every output table that has a `smiles` column.

## Known limitations

- Paper 6 has 1 ground-truth molecule and paper 16 has 4, so their per-paper scores swing widely. Pooled numbers are more reliable.
- Many ground-truth molecules in papers 4 and 19 are defined through R-group tables, placeholders and linker boxes rather than drawn in full. A crop-by-crop reader like MolScribe cannot recover those without Markush enumeration, which BioMiner adds.
- BioMiner reports ligands that have bioactivity values. Molecules without a reported value may be missing from its output even when it reads them correctly.
- Detection quality limits everything downstream, because MolScribe cannot reject a crop that has no molecule.

## Roadmap

1. More public sets: CLEF, UOB and Staker are already in the archive (add them to
   `public_check.datasets`), plus the BioVista subset of whole papers.
2. A second recognizer (MolSight or MolNexTR) for a wider side-by-side comparison.
3. A recognizer registry, so adding a tool needs no edit to `run_pipeline.py`.
4. Packaging (`pyproject.toml`) and a `molcensus` command-line tool.
5. Continuous integration that runs the tests on every push.
6. BioMiner end to end, once GPU access is available.
7. Detection scoring against hand-drawn boxes, and R-group enumeration.

## Troubleshooting

- **`rdkit` import errors after installing MolScribe.** Some MolScribe versions pull in the older `rdkit-pypi` package. Run `pip uninstall -y rdkit-pypi && pip install --force-reinstall rdkit`.
- **NumPy 2 errors.** Make sure `numpy<2` is installed, as pinned in `requirements.txt`.
- **`No module named 'torch._six'` when importing MolScribe.** Its pinned `timm` predates PyTorch 2. Run `pip install "torch==1.13.1" "torchvision==0.14.1"`.
- **401/403 error downloading MolDetV2.** Open https://huggingface.co/UniParser/MolDetv2, accept the license while logged in, then run `huggingface-cli login`. Or set `detection.method: embedded` to skip the detector.
- **No GPU.** Every enabled stage runs on CPU. Set `device: mps` or `device: cuda` in the config only if your torch build supports it.

## References

- Gulluoglu, H. S. A., et al. (2026). Practical use of advanced AI frameworks on real-life scientific problems: Three case studies. *bioRxiv*. https://doi.org/10.64898/2026.06.23.734132
- Heller, S. R., McNaught, A., Pletnev, I., Stein, S., & Tchekhovskoi, D. (2015). InChI, the IUPAC International Chemical Identifier. *Journal of Cheminformatics, 7*, 23. https://doi.org/10.1186/s13321-015-0068-4
- Qian, Y., Guo, J., Tu, Z., Li, Z., Coley, C. W., & Barzilay, R. (2023). MolScribe: Robust molecular structure recognition with image-to-graph generation. *Journal of Chemical Information and Modeling, 63*(7), 1925–1934. https://doi.org/10.1021/acs.jcim.2c01480
- Tang, H., Dang, X., & Wang, J. (2026). Fine-tuning DeepSeek-OCR-2 for molecular structure recognition. *arXiv*. https://arxiv.org/abs/2604.03476
- Yan, J., et al. (2026). BioMiner: A multi-modal system for automated mining of protein-ligand bioactivity data from literature. *arXiv*. https://arxiv.org/abs/2604.21508
- MolDetV2 weights: https://huggingface.co/UniParser/MolDetv2
- Public check images: https://huggingface.co/yujieq/MolScribe (`real.zip`, MIT license)

## License

MolCensus code is released under the MIT License (see `LICENSE`). The models it runs keep their
own licenses: MolScribe and BioMiner are MIT licensed, and the MolDetV2 weights are for
non-commercial use only.
