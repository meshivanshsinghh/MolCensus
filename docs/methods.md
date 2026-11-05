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

## The BioMiner run (2026-11-05)

BioMiner ran end to end once, on a `g6.2xlarge` in us-east-2 with one NVIDIA L4. It was BioMiner commit
`17c6161`, patched by `scripts/aws/patch_biominer.py`. The language model was Gemini 3.5 Flash on Google Cloud
Vertex AI, through its OpenAI-compatible endpoint, not BioMiner's own BioMiner-Instruct. The instance signed in
to Google through AWS-to-Google workload identity federation, with no key stored on it. The API reported
`google/gemini-3.5-flash` for every call. Settings: temperature 0, JSON output for image calls. A call is retried
up to 5 times after a server error, a connection error, or a rate limit (429) that asks for a wait of 300 s or
less; a 429 asking for longer gives up at once.

| | |
|---|---|
| Times | smoke test 3 min; six course papers 22 min (1.6 to 5.6 per paper); 12 BioVista papers 43 min (1.7 to 6.1 per paper) |
| Calls | 330 answered, 4 retried after short 429s, 0 lost |
| Tokens | 1,059,125 input, 86,175 output and 457,281 thinking (total minus input minus output) |
| Cost | $6.48 at list price: $1.50 per million input tokens, $9.00 per million output and thinking tokens |

**What counts as a BioMiner prediction.** The pipeline's rule, "linked structures", is BioMiner's
`_structure_merged_output_` table: the full structures and R-group enumerations that the language model tied to a
compound label. It does not depend on activity values. The analysis in `scripts/analysis/biominer_scores.py`
adds two rules. "All extracted structures" also counts every structure in BioMiner's OCSR output
(`<task>_ocsr_output.json`), whether or not it was linked. "All extracted, without `*` pieces" leaves out
structures with `*` atoms, which are table scaffolds and substituent pieces. That variant also drops two
paper-19 molecules that are drawn with a "Heteroaryl/Aryl" label, which the answer key itself writes as `*`;
that is why its recall is 0.341 and not 0.350.

Course papers, canonical SMILES rule (220 molecules):

| Rule | Recall | Precision | F1 |
|---|---|---|---|
| Linked | 0.214 [0.165, 0.272] (47) | 0.810 (47 of 58) | 0.338 |
| All extracted | 0.350 (77) | 0.445 (of 173) | 0.392 |
| All extracted, without `*` pieces | 0.341 (75) | 0.789 (of 95) | 0.476 |

BioMiner found 12 molecules that none of the three crop readers found (9 in paper 4, 3 in paper 19), all by
enumerating R-group tables. Obaidullah's BioMiner run on the same papers (72.4% precision, 34.5% recall, 46.8%
F1; group report, Table 5) is consistent with counting every structure BioMiner read. His model is not confirmed.

**BioVista, linking vs finding.** On 12 BioVista papers (415 labelled structures), as means over papers under
canonical SMILES:

| Type | Linked | All extracted, without `*` pieces | Published (BioMiner-Instruct, 450 papers) | Share of the gap closed |
|---|---|---|---|---|
| Full structures | 0.361 | 0.594 | 0.654 | 80% |
| Chiral | 0.221 | 0.369 | 0.454 | about two thirds |
| R-group tables | 0.283 | 0.286 | 0.339 | almost none |

Every full-table structure is an OCSR read, so for full structures "all extracted" is a ceiling for any linking
step. R-group-table structures are enumerations (150 of 153), so no OCSR read can make up for them. Pooled
structure recall is 0.313 and precision 0.522. BioMiner's own evaluator, as means over papers, gives 0.338 and
0.526, against the published 0.580 and 0.615. Details: `docs/biominer_flash_2026-11-05.md`.

## Recall ceiling

The 146 course molecules that no crop reader found were sorted into five buckets: reading loss (drawn in full
and cropped, but misread), detection loss (drawn in full, not cropped), table loss (defined only through an
R-group table, placeholder or linker box), not in the figures, or unresolved. No label was set by hand.

Automatic evidence came first:
- a second MolDetV2 pass at confidence 0.05, with MolScribe on the 38 new crops (an exact InChIKey match means
  detection loss);
- an exact substructure test of every reading with `*` atoms against each missed molecule, allowing branches only
  at the `*` positions (table loss);
- a pipeline crop whose reading has Tanimoto of at least 0.8 to a missed molecule, when that molecule is the
  reading's closest ground-truth molecule (reading loss).

The rest went to two AI judges, Claude and Gemini 3.5 Flash. They saw the same page images, with the crops
marked, and the same instructions. First they were tested blind on 213 molecules with a known answer: Claude
scored 0.967 and Gemini 0.930. Their agreement was Cohen's kappa = 0.904 over all 219 molecules. A judge label
counted only when both agreed. A layer 2 label that both judges contradicted was withdrawn; without that rule
(the strict rule) the totals are 142 table loss, 3 reading loss and 1 unresolved.

Result: 138 table loss, 2 reading loss, 0 detection loss, 0 not in the figures, 6 unresolved. So 75 to 81 of the
219 molecules (0.342 to 0.370) are drawn in full, against 73 (0.333) found by any of the three readers. On these
papers recall is limited by how compounds are defined, not by detection or reading. Details:
`docs/recall_ceiling.md`.

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
- BioMiner's predictions in the pipeline are the structures its language model linked to a compound label
  (the `_structure_merged_output_` table). A structure it read but did not link is not in that table; the
  "all extracted" rules above count those too. Its final `_merge_` table keeps only structures with a
  bioactivity value; it is scored separately as `biominer_bioactivity`.
- BioMiner ran with Gemini 3.5 Flash, not BioMiner-Instruct, and on 12 BioVista papers rather than 450.
- Everything ran once: on the Mac CPU, the crop readers again on one AWS GPU run, and BioMiner on another.
