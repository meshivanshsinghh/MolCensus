# Methods

How MolCensus scores a tool, how it reports uncertainty, and what the public check found.

## Scoring the course papers

Crops don't line up one-to-one with ground-truth entries: one drawing can be cut into several crops,
and one crop can show several molecules. So for each paper, scoring compares the set of unique
predicted molecules with the set of unique ground-truth molecules, the same paper-level precision and
recall used in the anchor study (Gulluoglu et al., 2026).

| Metric | Definition |
|---|---|
| Validity rate | Share of predictions RDKit can parse into a molecule |
| Exact precision / recall / F1 | Match on the full InChIKey, so stereochemistry must also be right |
| Stereo-blind precision / recall / F1 | Match on the first InChIKey block, which encodes connectivity only (Heller et al., 2015) |
| Mean best Tanimoto | For each ground-truth molecule, the highest Tanimoto similarity (Morgan, radius 2, 2048 bits) to any valid prediction from the same paper, averaged |
| Near-match recall | Share of ground-truth molecules with best Tanimoto of at least 0.85 |
| Mean confidence | The tool's own confidence, averaged over valid predictions |

Molecules are compared by InChIKey rather than canonical SMILES, because canonical SMILES strings can
differ between RDKit versions. InChIKey is slightly lenient on tautomers. Molecules with `*` atoms
have no InChIKey and are matched by canonical SMILES instead. Precision counts unique valid
predictions, so a molecule drawn twice counts once; invalid outputs are left out of precision and
show up in the validity rate instead. The `ALL` row adds up the counts over papers (micro-average).

## What happened to each molecule

Recall says how many molecules were found, not what happened to the rest. Each unique ground-truth
molecule is therefore put in exactly one bucket, the best one that applies:

| Bucket | Meaning |
|---|---|
| Found exactly | Full InChIKey matches a prediction from the same paper |
| Right except stereochemistry | Only the first InChIKey block matches |
| Near miss | Best Tanimoto of at least 0.85: a prediction that is almost the same molecule |
| Related molecule read | Best Tanimoto from 0.5 to 0.85: for example a shared core or a close analog |
| Nothing similar read | Best Tanimoto below 0.5 |

Both thresholds are in the config (`near_match_threshold`, `related_threshold`). The buckets add up
to the number of unique ground-truth molecules, and they are what the outcome chart shows.

## Uncertainty

Six papers and about 220 molecules is a small sample, so every recall and precision gets two 95%
intervals. The Wilson interval treats molecules as independent. The bootstrap interval, on the
pooled row only, resamples the six papers with replacement 2,000 times and recomputes the pooled
rate. It is wider, and more honest, because molecules from the same paper tend to succeed or fail
together.

## Public crop check

Before trusting a tool's score on the course papers, MolCensus checks that the installed tool
reproduces its published accuracy on public data. The images come from MolScribe's evaluation
archive `real.zip` (184 MB, from the MIT-licensed Hugging Face repository `yujieq/MolScribe`), which
holds the CLEF, UOB, USPTO, Staker and ACS sets. It is downloaded once into `data/public/` and never
committed.

The default run uses USPTO (patent drawings) and ACS (journal figures, the closest match to the
course papers), each a fixed random sample of 300 images. Every image shows one molecule with a known
answer, so accuracy is per image: exact match, stereo-blind match, chiral accuracy, validity,
Tanimoto, confidence and seconds per image. The check passes when the published accuracy (92.6% on
USPTO and 71.9% on ACS; Qian et al., 2023) falls inside our 95% Wilson interval.

**BioVista OCSR** (Yan et al., 2026) adds 8,861 crops from 335 papers: 4,844 full structures, 4,017
Markush structures (the reference has `*` atoms) and 2,230 chiral ones (the reference has `@`). It is
a separate 61 MB download from the Google Drive link in BioMiner's `BioVista/README.md`. Its labels
(`ocsr/labels/gd_SMIELS.csv`: `pdf_name`, `index`, `molminer_id`, `smiles`) are matched to the images
`ocsr/data/<pdf_name>_image_molminer_bbox_<page>-<index>.png` by name. BioMiner's own evaluator pairs
labels and predictions by row order instead, and the label file is not sorted by paper, so we do not
copy that step. The published BioVista values (BioMiner README, commit `17c6161`) were computed with
BioMiner's matching rule: neutralize charges, compare canonical SMILES, and treat the first `[1*]` as a
plain `*`. MolCensus scores every image with both rules and checks BioVista's published values against
BioMiner's rule; `exact_accuracy` stays our InChIKey rule. The summary has one row for all images and
one each for the full, chiral and Markush splits.

A sample of 300 BioVista images (seed 0) with MolScribe on the Mac CPU gave 0.457 under BioMiner's
rule [0.401, 0.513] against the published 0.455, and 0.450 under our rule. The full, chiral and Markush
splits also fell inside their intervals. This was a check of the setup, not a result; the full set
runs on the GPU instance.

**BioVista paper labels.** Paper-level scoring and BioMiner's `--biovista_evaluate` need two more
downloads, both kept locally in `data/biovista/` and never committed:

| Download | Google Drive link | Status on 2026-10-29 |
|---|---|---|
| Bioactivity Triplet Extraction (`labels/`: 500 `<name>_structure.csv`, 503 `<name>_data.csv`, 83 `.xlsx`) | https://drive.google.com/file/d/14icAlg_jEy3uE7FOcDD-EYn3UIKQxC__/view (BioMiner's main README) | works |
| The same, as linked from BioMiner's `BioVista/README.md` | https://drive.google.com/file/d/1xITA_Mub9u1MCZjlgs26qL_dMCrzS9Rn/view | dead: HTTP 404 |
| Structure-Bioactivity Annotation (`structure_annotation/`: PDBbind index files and ligand structures for 501 PDB IDs, 46 MB) | https://drive.google.com/file/d/1cLC3NePtxctIurnCVaNLw0JzBWYrWTVW/view (both READMEs) | works |

Both READMEs are at BioMiner commit `17c6161`. The structure files have the columns `smiles`, `ligand`,
`backbone` and `groups` (`backbone` is filled for compounds defined by a scaffold plus R groups).
Three `_data.csv` files and their `.xlsx` copies (`398_6j5t`, `404_6jyw`, `411_6iqj`) carry PDB IDs
that do not match BioVista's paper list, which has `398_6j5l`, `404_6iyw` and `411_6iqg`; those three
have full pairs. Each of the 83 `.xlsx` files has a bioactivity table in its first sheet; the other two sheets are empty. Every paper
in BioVista's `file_names.txt` has a structure file. `convert.py` turns them into the ground-truth
format; 68_6r8r gives 50 unique molecules.

Reference SMILES that RDKit cannot read are left out. In ACS that is 47 of the 300 labels, mostly
placeholders such as `[R]`, `[Ar]` or `[X]`; in USPTO it is 5 malformed labels. That leaves 253 ACS
and 295 USPTO images. The published numbers used the authors' own handling of these labels, so the
comparison is on a slightly different set.

## A MolScribe bug the check caught

The first run on a Mac failed the ACS check: 0.348 exact accuracy against the published 0.719, with
only 42% valid SMILES. USPTO passed.

Tracing one failed image showed that MolScribe read every atom correctly, including the
abbreviations `Me` and `Ph`, but still returned `<invalid>`. The cause is in MolScribe's
`_expand_functional_group` (`molscribe/chemistry.py`), which turns an abbreviation into atoms. It
deletes the bonds around the label and then reads their bond order from the deleted bond objects.
That is undefined behavior in RDKit. It seems to work on Linux, but on macOS it returns 0, the step
fails with `KeyError: 0`, and MolScribe hides the error. The crash is the same with every RDKit
release from 2021.09 to 2026.09, so pinning RDKit does not help. Patent drawings rarely use
abbreviations, which is why USPTO was not affected.

`patch_abbreviation_bug()` in `recognizers/molscribe_runner.py` reads the bond orders before the
bonds are deleted. It changes only how MolScribe writes its own prediction as SMILES, not the
reference SMILES, the scoring, the thresholds or the sample, and it stops with an error if a new
MolScribe version changes that code. `tests/test_molscribe_patch.py` covers it.

| | Images or molecules | Before fix | After fix | Published |
|---|---|---|---|---|
| USPTO exact accuracy | 295 images | 0.922 | 0.922 | 0.926 |
| ACS exact accuracy | 253 images | 0.348 | 0.731 | 0.719 |
| Course papers, valid SMILES | 237 crops | 0.806 | 0.932 | |
| Course papers, exact recall | 219 molecules | 0.192 | 0.233 | |

## The AWS GPU run (2026-11-03 and 2026-11-04)

The GPU results in `results/aws/` come from one AWS instance, set up and run as described in
`scripts/aws/README.md`.

| | |
|---|---|
| Instance | `g5.2xlarge` in us-east-2: one NVIDIA A10G (23,028 MiB), driver 595.112.07 |
| AMI | `ami-06a00337bcd7341f7`, "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20261030" |
| MolScribe and MolSight | Python 3.10.12, torch 2.14.1+cu130 with its own cuDNN 9.24.0.43 (`.venv` and `tools/molsight-env`) |
| MolGlyph and BioMiner | BioMiner commit `17c6161`, its conda environment: Python 3.10.14, torch 2.6.0+cu124 |
| MolSight | commit `86e979b`, checkpoint MolSight-base |

**cuDNN mismatch.** The first run (2026-11-03) gave MolScribe and MolSight 0.000 on every public set.
Every image failed with `cudnnFinalize failed ... CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`, so neither
reader wrote a SMILES. The AMI puts its own cuDNN 9.23.2 (`/usr/local/cuda-13.2/lib`) on
`LD_LIBRARY_PATH`, from `/etc/profile.d/dlami.sh`. torch 2.14.1 loaded the core of cuDNN
(`libcudnn.so.9`, `libcudnn_graph.so.9`) from its own wheel, then the engine libraries
(`libcudnn_engines_*.so.9.23.2`) from the AMI's folder, and every convolution failed.
`torch.cuda.is_available()` still returned True. MolGlyph runs in BioMiner's conda environment, whose
convolutions worked with the AMI's path set, so it was not affected.

The fix (commit `b8a42f4`): `run_all.sh` unsets `LD_LIBRARY_PATH` for every stage and gives it back only
to the BioMiner servers. `setup.sh` now runs one convolution on the GPU in each environment instead
of only checking `torch.cuda.is_available()`. A crop-reader stage now fails if a reader wrote no SMILES at all.
On 200 USPTO images (198 scored), MolScribe then gave 0.929 [0.885, 0.957] against the published
0.926, and MolSight 0.914 [0.867, 0.946] against the published 0.912.

**MinerU OCR model.** MinerU, inside BioMiner, still loads
`models/OCR/paddleocr_torch/ch_PP-OCRv3_det_infer.pth`, which the current
`opendatalab/PDF-Extract-Kit-1.0` model pack no longer has. `setup.sh` downloads it from commit
`fe3643d1b873284fca2d90c3d731ffbc182b3566` (`fe3643d1b8`). On 2026-11-03 `magic-pdf` exited 0 after a
traceback, so the MinerU smoke test now passes only if it wrote a Markdown file.

**Gemini quota failure.** BioMiner gave no results on 2026-11-04. The smoke test (BioMiner's example
`68_6r8r.pdf`, `gemini-3.5-flash`, the key set by `course_tier: paid`) reached its 45-minute
limit. 8 MLLM calls were answered. 21 attempts were refused with HTTP 429 `RESOURCE_EXHAUSTED` on
Google's free-tier metric `generate_content_free_tier_requests`: 1 on the 5-requests-per-minute limit
and 20 on the 20-requests-per-day limit. The errors name the model `gemini-3.6-flash`. Another 10
attempts got HTTP 503 ("high demand"). 4 calls failed after all 6 tries. So that key was held to free-tier
limits, although both keys had passed a one-call test during setup. Because the smoke test did not
pass, `run_all.sh` skipped BioMiner on the course papers and on BioVista. MolRecBench-Wild was
skipped too, because its data file was not on the instance.

## Privacy

The six course PDFs are copyrighted and their ground truth is private sponsor data. Neither is
committed, and neither is any file with predicted or ground-truth SMILES for those papers
(`results/predictions/`, `results/scored/`). What is committed for the course papers is SMILES-free:
metrics, counts, figures and run metadata. The public-check files in `results/public/` are committed,
because their images and SMILES are public. The optional W&B upload follows the same rule.

## Known limitations

- Papers 6 and 16 have 1 and 4 molecules, so their per-paper scores swing widely.
- Papers 4 and 19 define many compounds through R-group tables, placeholders and linker boxes. A
  crop-by-crop reader can't recover those without R-group enumeration.
- There is no detection ground truth, so missed detections and misreadings can't be separated.
- MolScribe can't reject a crop that has no complete molecule, so detection errors flow straight
  through.
- BioMiner's predictions are the structures its MLLM linked to a compound label (the
  `_structure_merged_output_` table). A structure it read but could not link to a label is not in
  that table. Its final `_merge_` table keeps only structures with a bioactivity value; it is scored
  separately as `biominer_bioactivity`.
- Everything ran once: on the Mac CPU, and the crop readers again on one AWS GPU run (above).
