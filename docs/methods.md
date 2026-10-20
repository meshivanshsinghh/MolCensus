# Methods

How MolCensus scores tools, how it handles uncertainty, and what the public check found.

## Scoring the course papers

Crops do not line up one-to-one with ground-truth entries. For each paper, scoring therefore
compares the set of unique predicted molecules with the set of unique ground-truth molecules, as in
the anchor study's paper-level precision and recall (Gulluoglu et al., 2026).

| Metric | Definition |
|---|---|
| Validity rate | Share of predictions RDKit can parse into a molecule |
| Exact precision / recall / F1 | Match on the full InChIKey, so stereochemistry must also be right |
| Stereo-blind precision / recall / F1 | Match on the first InChIKey block, which encodes connectivity only (Heller et al., 2015) |
| Mean best Tanimoto | For each ground-truth molecule, the highest Tanimoto similarity (Morgan, radius 2, 2048 bits) to any valid prediction from the same paper, averaged |
| Near-match recall | Share of ground-truth molecules with best Tanimoto at least 0.85 |
| Mean confidence | The tool's own confidence, averaged over valid predictions |

- Molecules are compared by InChIKey, not canonical SMILES, because canonical SMILES strings can
  differ between RDKit versions. InChIKey is slightly lenient on tautomers.
- Molecules with `*` atoms have no InChIKey. They are matched by canonical SMILES instead.
- Precision counts unique valid predictions, so a molecule drawn twice counts once. Invalid outputs
  are left out of precision and reported through the validity rate.
- The `ALL` row pools counts over papers (micro-average).

## Uncertainty

Six papers and about 220 molecules is a small sample, so every recall and precision gets two 95%
intervals:

- **Wilson interval:** treats molecules as independent.
- **Bootstrap over papers** (pooled row only): resamples the six papers with replacement 2,000
  times and recomputes the pooled rate. It is wider and more honest, because molecules from the
  same paper tend to succeed or fail together.

## Public crop check

Before trusting a tool's score on the course papers, MolCensus checks that the installed tool
reproduces its published accuracy on public data.

- **Data:** MolScribe's evaluation archive `real.zip` (184 MB, MIT-licensed Hugging Face repository
  `yujieq/MolScribe`), with the CLEF, UOB, USPTO, Staker and ACS sets. It is downloaded once into
  `data/public/` and never committed.
- **Sets used:** USPTO (patent drawings) and ACS (journal figures, closest to the course papers),
  each a fixed random sample of 300 images.
- **Scoring:** one molecule per image, so accuracy is per image: exact match on the full InChIKey,
  stereo-blind match, chiral accuracy, validity, Tanimoto, confidence and seconds per image.
- **Pass criterion:** the published accuracy (92.6% USPTO, 71.9% ACS; Qian et al., 2023) falls
  inside our 95% Wilson interval.
- **Left out:** reference SMILES that RDKit cannot read. In ACS that is 47 of 300 labels, mostly
  placeholders such as `[R]`, `[Ar]` or `[X]`; in USPTO it is 5 malformed labels. This leaves 253
  ACS and 295 USPTO images. The published numbers used the authors' own handling of these, so the
  comparison is on a slightly different set.

## A MolScribe bug the check caught

The first run on a Mac failed the ACS check: 0.348 exact accuracy against the published 0.719, with
only 42% valid SMILES. USPTO passed.

Tracing one failed image showed that MolScribe read every atom correctly, including the
abbreviations `Me` and `Ph`, but returned `<invalid>`. The cause is in MolScribe's
`_expand_functional_group` (`molscribe/chemistry.py`), which turns an abbreviation into atoms. It
deletes the bonds around the label and then reads their bond order from the deleted bond objects.
That is undefined behavior in RDKit. It seems to have worked on the authors' Linux machines, but on
macOS it returns 0, the step fails with `KeyError: 0`, and MolScribe hides the error. The crash is
the same with every RDKit release from 2021.09 to 2026.09, so pinning RDKit does not help. Patent
drawings rarely use abbreviations, which is why USPTO was not affected.

`patch_abbreviation_bug()` in `recognizers/molscribe_runner.py` reads the bond orders before the
bonds are deleted. It changes only how MolScribe writes its own prediction as SMILES, not the
reference SMILES, scoring, thresholds or sample. It stops with an error if a new MolScribe version
changes that code. `tests/test_molscribe_patch.py` covers it.

| | Images scored | Before fix | After fix | Published |
|---|---|---|---|---|
| USPTO exact accuracy | 295 | 0.922 | 0.922 | 0.926 |
| ACS exact accuracy | 253 | 0.348 | 0.731 | 0.719 |
| Course papers, valid SMILES | 237 crops | 0.806 | 0.932 | |
| Course papers, exact recall | 219 molecules | 0.192 | 0.233 | |

## Privacy

The six course PDFs are copyrighted and their ground truth is private sponsor data. Neither is
committed, and neither is any file with predicted or ground-truth SMILES for those papers
(`results/predictions/`, `results/scored/`). Everything committed for the course papers is
SMILES-free: metrics, counts, figures and run metadata. Public-check files in `results/public/` are
committed, because their images and SMILES are public. The optional W&B upload follows the same rule.

## Known limitations

- Papers 6 and 16 have 1 and 4 molecules, so their per-paper scores swing widely.
- Papers 4 and 19 define many compounds through R-group tables, placeholders and linker boxes. A
  crop-by-crop reader cannot recover those without R-group enumeration.
- There is no detection ground truth, so missed detections and misreadings cannot be separated.
- MolScribe cannot reject a crop that has no molecule, so detection errors flow straight through.
- BioMiner reports only ligands with bioactivity values, so molecules without one may be missing
  from its output even when it reads them correctly.
- Everything ran once, on CPU.
