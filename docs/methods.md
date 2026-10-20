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
- BioMiner only reports ligands with bioactivity values, so molecules without one may be missing
  from its output even when it reads them correctly.
- Everything ran once, on CPU.
