# Data

The pipeline expects the course benchmark files here. They are not committed: the PDFs are
copyrighted journal articles, and the ground truth is course material.

```
data/
├── ground_truth_SMILES_confirmed_1_2_4_6_16_19.csv   # columns: paperID, name of molecule, canonical_SMILES
└── pdfs/
    ├── 1.pdf
    ├── 2.pdf
    ├── 4.pdf
    ├── 6.pdf
    ├── 16.pdf
    └── 19.pdf
```

Both are available from the AAI 6630 course page. The `prepare` step writes
`results/data_manifest.csv` with the SHA-256 hash of every input file, so you can confirm
you are scoring the same files.
