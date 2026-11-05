# MolCensus: BioMiner with Gemini 3.5 Flash, AWS run of 2026-11-05

This file sums up the first complete BioMiner run: what ran, what it cost, why it produced only 58
predictions on the course papers, and how it scores on the six course papers and on 12 BioVista
papers. It holds scores and counts only, never a SMILES string.

The numbers come from two scripts, run on the Mac from the downloaded run folder:

```bash
python scripts/analysis/biominer_facts.py outputs/s3/20261105T043509Z    # times, Gemini calls, tokens, cost
python scripts/analysis/biominer_scores.py outputs/s3/20261105T043509Z   # output counts, scores, paired tests
```

## 1. What ran

| Item | Value |
|---|---|
| S3 folder | `20261105T043509Z` |
| Instance | g6.2xlarge (one NVIDIA L4, 24 GB), us-east-2 |
| BioMiner | commit `17c6161`, patched by `scripts/aws/patch_biominer.py` (one GPU, Vertex AI client, retries) |
| LLM | Gemini 3.5 Flash (`gemini-3.5-flash`) on Google Cloud Vertex AI, OpenAI-compatible endpoint, signed in through AWS-to-Google workload identity federation (no key on the instance) |
| Model the API reported | `google/gemini-3.5-flash`, for all 330 answered calls |
| OCSR inside BioMiner | MolGlyph, on MolDetV2 boxes, with MinerU layout |

| Stage | Started (UTC) | Minutes | Outcome |
|---|---|---|---|
| course_readers | | | done in an earlier run, skipped |
| biominer_smoke (BioMiner's example paper `68_6r8r`) | 04:35:09 | 3 | passed, no lost calls |
| biominer_course (6 course papers) | 04:38:40 | 22 | done |
| public_readers | | | done in an earlier run, skipped |
| biominer_biovista (12 BioVista papers, with `--biovista_evaluate`) | 05:00:55 | 43 | done |
| molrecbench_wild | 05:44:22 | | skipped: no data file |

The upload to S3 was checked at 05:44:43, about 70 minutes after the start. Then the instance
stopped itself. Per paper, BioMiner took 1.6 to 5.6 minutes on the course papers and 1.7 to 6.1
minutes on the BioVista papers. For BioVista the plan had estimated 3 to 10 minutes per paper.

## 2. Download and copy to the Mac

The run folder was downloaded to `outputs/s3/20261105T043509Z/` (gitignored). File counts match
the bucket, folder by folder:

| Folder | Bucket | Mac |
|---|---|---|
| `results/aws/` | 13 | 13 |
| `results/aws_biovista/` | 3 | 3 |
| `outputs/work/biominer_output/` (6 course papers) | 1,408 | 1,408 |
| `outputs/work_biovista/biominer_output/` (12 BioVista papers) | 2,758 | 2,758 |
| `runs/` | 259 | 259 |
| `setup.log`, `UPLOAD_COMPLETE` | 2 | 2 |
| **Total** | **4,443** | **4,443** |

`UPLOAD_COMPLETE` says "4442 files, checked 2026-11-05T05:44:43Z". It does not count itself. The
instance's state was `stopped` after the download.

Next, only files not already on the Mac were copied into the places that
`scripts/aws/README.md` ("Copy back from S3") uses:

- **`results/`: 5 new files.** These are the BioMiner predictions for the course papers and for
  BioVista, plus the BioVista run log. The crop readers' prediction files in S3 are byte-identical
  to the Mac copies.
- **`outputs/work/biominer_output/` and `outputs/work_biovista/biominer_output/`:** all 1,408 and
  2,758 files were new.
- **`outputs/aws_logs/runs/`: 41 new files.**

33 files already on the Mac differ from the S3 copy and were **kept as they were**:

- **`results/aws/run_log.txt`.**
- **Eight logs in `outputs/aws_logs/runs/`:** `run_all.log`, `biominer_smoke.log`,
  `biominer_smoke_output.log`, `biominer_course.log`, `biominer_biovista.log`, `server_mineru.log`,
  `server_moldet.log` and `server_ocsr.log`.
- **24 files in `outputs/aws_logs/runs/biominer_smoke/`.** That folder holds the stopped smoke test
  of 2026-11-04, and this run reused it.

The Mac copies are the 2026-11-04 versions. This run's versions are in
`outputs/s3/20261105T043509Z/runs/`, which is what the analysis reads. Nothing from `outputs/` or
from a `predictions/` folder is committed.

## 3. Run facts: Gemini calls, tokens and cost

**Counting rules:**
- **Calls.** "Answered" counts the calls Gemini answered. A retry is a failed try that was tried
  again. A lost call failed every try.
- **Thinking tokens** are total minus input minus output. The Vertex endpoint counts the model's
  thinking in `total_tokens` but not in `completion_tokens`.
- **Cost** is at list price: $1.50 per million input tokens and $9.00 per million output and
  thinking tokens.

**Course papers**

| Paper | Minutes | Calls answered | Retries | Lost | Input | Output | Thinking | Cost (USD) |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.1 | 17 | 0 | 0 | 52,251 | 5,074 | 24,090 | 0.341 |
| 2 | 4.9 | 18 | 0 | 0 | 77,406 | 4,521 | 27,568 | 0.405 |
| 4 | 5.0 | 26 | 0 | 0 | 68,639 | 4,905 | 36,226 | 0.473 |
| 6 | 1.8 | 9 | 0 | 0 | 31,256 | 647 | 10,734 | 0.149 |
| 16 | 1.6 | 13 | 0 | 0 | 42,494 | 175 | 7,099 | 0.129 |
| 19 | 5.6 | 31 | 0 | 0 | 89,942 | 8,466 | 42,161 | 0.591 |
| **Total** | **22.1** | **114** | **0** | **0** | **361,988** | **23,788** | **147,878** | **2.088** |

**BioVista papers**

| Paper | Minutes | Calls answered | Retries | Lost | Input | Output | Thinking | Cost (USD) |
|---|---|---|---|---|---|---|---|---|
| 40 | 3.8 | 15 | 0 | 0 | 49,135 | 5,473 | 31,153 | 0.403 |
| 65 | 4.5 | 20 | 0 | 0 | 80,235 | 7,235 | 33,795 | 0.490 |
| 67 | 4.7 | 24 | 0 | 0 | 50,549 | 7,075 | 35,245 | 0.457 |
| 68 | 3.3 | 15 | 0 | 0 | 37,194 | 5,808 | 19,822 | 0.286 |
| 106 | 1.7 | 9 | 0 | 0 | 24,740 | 3,503 | 11,203 | 0.169 |
| 153 | 2.0 | 14 | 0 | 0 | 33,173 | 2,460 | 14,305 | 0.201 |
| 156 | 1.9 | 10 | 0 | 0 | 36,129 | 2,300 | 13,241 | 0.194 |
| 163 | 4.0 | 20 | 0 | 0 | 60,813 | 6,950 | 27,065 | 0.397 |
| 239 | 3.0 | 17 | 0 | 0 | 42,938 | 2,971 | 22,372 | 0.292 |
| 297 | 2.8 | 20 | 0 | 0 | 68,339 | 1,488 | 18,544 | 0.283 |
| 312 | 5.2 | 17 | 1 | 0 | 99,242 | 5,151 | 28,740 | 0.454 |
| 471 | 6.1 | 20 | 3 | 0 | 77,452 | 6,155 | 33,158 | 0.470 |
| **Total** | **43.0** | **201** | **4** | **0** | **659,939** | **56,569** | **288,643** | **4.097** |

**Smoke test** (paper `68_6r8r`): 15 calls answered, none retried or lost, 37,198 input, 5,818
output and 20,760 thinking tokens, $0.295.

**Whole run:**

| | Value |
|---|---|
| Calls answered | 330 |
| Retries | 4 |
| Lost calls | 0 |
| Input tokens | 1,059,125 |
| Output tokens | 86,175 |
| Thinking tokens | 457,281 |
| Cost at list price | **$6.48** ($0.30 smoke, $2.09 course, $4.10 BioVista) |

- **All four retries were short 429s.** Each was a `RESOURCE_EXHAUSTED` answer that gave no wait
  time, on BioVista papers 312 (1) and 471 (3). All four succeeded on a later try. No 429 asked for
  a long wait, so the new give-up rule never fired.
- **Thinking is the largest share of the cost.** It made up 84% of the output-side tokens, and
  $4.12 of the $6.48 (64%).
- **No paper had a BioMiner error.** Every run_info.json records the endpoint as `vertex`.

## 4. Why only 58 predictions on the course papers

The pipeline (`src/molcensus/recognizers/biominer_runner.py`) reads BioMiner's
`_structure_merged_output_` table. The question was whether that drops structures BioMiner
extracted but did not link to an activity value. Every kind of BioMiner output was counted, paper
by paper. Counts are distinct structures by canonical SMILES:

| Paper | Pipeline rows | OCSR boxes | OCSR distinct | Full-structure table | R-group table | Linked (merged table) | With an activity value | Read by OCSR, not linked | Linked, not in OCSR | All extracted |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 17 | 18 | 17 | 17 | 0 | 17 | 10 | 0 | 0 | 17 |
| 2 | 5 | 84 | 31 | 5 | 0 | 5 | 0 | 26 | 0 | 31 |
| 4 | 13 | 75 | 55 | 3 | 10 | 13 | 10 | 52 | 10 | 65 |
| 6 | 2 | 2 | 2 | 2 | 0 | 2 | 1 | 0 | 0 | 2 |
| 16 | 7 | 8 | 8 | 7 | 0 | 7 | 0 | 1 | 0 | 8 |
| 19 | 14 | 52 | 47 | 11 | 3 | 14 | 4 | 36 | 3 | 50 |
| **Total** | **58** | **239** | **160** | **45** | **13** | **58** | **25** | **115** | **13** | **173** |

Where these come from:
- **OCSR boxes:** every molecule box MolGlyph read inside BioMiner, in `biominer_upgrade_ocsr_output.json`.
- **Full-structure and R-group tables:** what the LLM tied to a compound label, as whole drawings
  or as R-group enumerations. The merged table is the two together.
- **With an activity value:** the `_merge_` table, saved as the separate tool `biominer_bioactivity`.

**What this shows:**

- **The pipeline does not drop structures for lacking an activity value.** The merged table holds
  every structure the LLM tied to a compound label, whether or not it has an activity value. 33 of
  the 58 have none. The pipeline's 58 rows are exactly the merged tables' 58 structures.
- **It does leave out structures that the LLM tied to no label: 115 of them.**
  - These are mostly in papers 4 (52), 19 (36) and 2 (26).
  - 78 of the 115 contain `*` atoms. They are R-group scaffolds and substituent pieces, which
    BioMiner reads in order to enumerate R-group tables. They are not whole molecules.
  - 30 of the 115 are ground-truth molecules.
- **The linking step works in both directions.** 13 linked structures are not in the OCSR output at
  all: 10 in paper 4 and 3 in paper 19. BioMiner built them by combining a scaffold with its
  substituents.

So a second rule is added in the analysis script. The pipeline's rule is unchanged.

- **"Linked structures"** is the pipeline's rule, as before: the merged table.
- **"All extracted structures"** is new: the merged table plus every structure in BioMiner's OCSR
  output, linked or not.
- **"All extracted, no `*` pieces"** is the same without structures that contain `*` atoms. Five
  of the 222 ground-truth rows contain `*` atoms (one in paper 2, four in paper 19). BioMiner's
  OCSR matched two of the paper 19 ones, so this variant loses those two finds (see "Why leaving
  out the `*` pieces lowers recall" in section 5).

## 5. Course papers: scores

The primary rule matches a prediction and a ground-truth molecule when their RDKit canonical
isomeric SMILES are equal. Under this rule the six papers have 220 distinct molecules. Each paper's
distinct predictions are compared with its distinct molecules as two sets. Cells give the rate and
its 95% Wilson interval. "Paper bootstrap" resamples the six papers 2,000 times (seed 0).

The crop readers' rows repeat section 7 of `docs/aws_run_2026-11-04.md`. They match it exactly,
which also checks the scorer.

**Recall (ground-truth molecules found)**

| Paper (molecules) | BioMiner, linked | BioMiner, all extracted | All extracted, no `*` pieces | MolScribe | MolSight | MolGlyph | Obaidullah's BioMiner |
|---|---|---|---|---|---|---|---|
| 1 (17) | 1.000 [0.816, 1.000] | 1.000 [0.816, 1.000] | 1.000 [0.816, 1.000] | 0.765 [0.527, 0.904] | 0.941 [0.730, 0.990] | 1.000 [0.816, 1.000] | 1.00 |
| 2 (34) | 0.088 [0.030, 0.230] | 0.265 [0.146, 0.431] | 0.265 [0.146, 0.431] | 0.235 [0.124, 0.400] | 0.206 [0.103, 0.368] | 0.265 [0.146, 0.431] | 0.26 |
| 4 (60) | 0.200 [0.118, 0.318] | 0.383 [0.271, 0.510] | 0.383 [0.271, 0.510] | 0.183 [0.106, 0.299] | 0.100 [0.047, 0.201] | 0.217 [0.131, 0.336] | 0.40 |
| 6 (1) | 0.000 [0.000, 0.793] | 0.000 [0.000, 0.793] | 0.000 [0.000, 0.793] | 0.000 [0.000, 0.793] | 0.000 [0.000, 0.793] | 0.000 [0.000, 0.793] | 0.00 |
| 16 (4) | 0.500 [0.150, 0.850] | 0.500 [0.150, 0.850] | 0.500 [0.150, 0.850] | 0.250 [0.046, 0.699] | 0.250 [0.046, 0.699] | 0.500 [0.150, 0.850] | 0.50 |
| 19 (104) | 0.125 [0.075, 0.202] | 0.250 [0.177, 0.341] | 0.231 [0.160, 0.320] | 0.163 [0.105, 0.246] | 0.221 [0.152, 0.310] | 0.192 [0.128, 0.278] | 0.23 |
| **Pooled (220)** | **0.214** [0.165, 0.272] (47) | **0.350** [0.290, 0.415] (77) | **0.341** [0.281, 0.406] (75) | **0.227** [0.177, 0.287] (50) | **0.241** [0.189, 0.302] (53) | **0.277** [0.222, 0.340] (61) | **0.345** |
| Pooled, paper bootstrap | [0.118, 0.558] | [0.256, 0.655] | [0.241, 0.655] | [0.169, 0.473] | [0.128, 0.541] | [0.201, 0.610] | |

**Precision (over each tool's distinct valid predictions)**

| Paper | BioMiner, linked | BioMiner, all extracted | All extracted, no `*` pieces | MolScribe | MolSight | MolGlyph | Obaidullah's BioMiner |
|---|---|---|---|---|---|---|---|
| 1 | 1.000 [0.816, 1.000] | 1.000 [0.816, 1.000] | 1.000 [0.816, 1.000] | 0.765 [0.527, 0.904] | 0.941 [0.730, 0.990] | 1.000 [0.816, 1.000] | |
| 2 | 0.600 [0.231, 0.882] | 0.290 [0.161, 0.466] | 0.692 [0.424, 0.873] | 0.235 [0.124, 0.400] | 0.206 [0.103, 0.368] | 0.300 [0.167, 0.479] | |
| 4 | 0.923 [0.667, 0.986] | 0.354 [0.249, 0.475] | 0.793 [0.616, 0.902] | 0.169 [0.097, 0.278] | 0.095 [0.044, 0.193] | 0.224 [0.136, 0.347] | |
| 6 | 0.000 [0.000, 0.658] | 0.000 [0.000, 0.658] | 0.000 [0.000, 0.658] | 0.000 [0.000, 0.658] | 0.000 [0.000, 0.658] | 0.000 [0.000, 0.658] | |
| 16 | 0.286 [0.082, 0.641] | 0.250 [0.071, 0.591] | 0.250 [0.071, 0.591] | 0.200 [0.036, 0.624] | 0.125 [0.022, 0.471] | 0.250 [0.071, 0.591] | |
| 19 | 0.929 [0.685, 0.987] | 0.520 [0.385, 0.652] | 0.923 [0.759, 0.979] | 0.370 [0.245, 0.514] | 0.479 [0.345, 0.617] | 0.417 [0.288, 0.557] | |
| **Pooled** | **0.810** [0.691, 0.891] (of 58) | **0.445** [0.373, 0.520] (of 173) | **0.789** [0.697, 0.859] (of 95) | **0.296** [0.232, 0.369] (of 169) | **0.308** [0.244, 0.381] (of 172) | **0.374** [0.304, 0.451] (of 163) | **0.724** |
| Pooled, paper bootstrap | [0.510, 0.940] | [0.309, 0.667] | [0.559, 0.910] | [0.179, 0.507] | [0.116, 0.596] | [0.234, 0.652] | |
| **Pooled F1** | 0.338 | 0.392 | 0.476 | 0.257 | 0.270 | 0.319 | 0.468 |

**InChIKey rule.** The six papers have 219 distinct molecules under this rule. It changes only
papers 2 and 4:

| | Linked | All extracted | All extracted, no `*` pieces |
|---|---|---|---|
| Pooled recall | 0.219 [0.169, 0.279] (48 of 219) | 0.365 [0.304, 0.431] (80) | 0.356 [0.296, 0.422] (78) |
| Pooled precision | 0.828 [0.711, 0.904] (48 of 58) | 0.462 [0.390, 0.537] (80 of 173) | 0.821 [0.732, 0.885] (78 of 95) |
| Paper 2 recall | 0.121 (4 of 33) | 0.333 (11 of 33) | 0.333 |
| Paper 4 recall | 0.200 | 0.400 (24 of 60) | 0.400 |

The readers' InChIKey scores are those of section 2 of `docs/aws_run_2026-11-04.md`.

### Why leaving out the `*` pieces lowers recall from 0.350 to 0.341

The drop is two molecules, 77 to 75 of 220, both in paper 19 (entries 19-c and 19-d below). Five
ground-truth entries contain `*` atoms. This is which tools found each one; the entries are numbered
here instead of named, because the compound names belong to the private answer key:

| Paper | Entry | `*` atoms | BioMiner, linked | BioMiner, all extracted | All extracted, no `*` pieces | MolScribe | MolSight | MolGlyph |
|---|---|---|---|---|---|---|---|---|
| 2 | 2-a | 5 | | | | | | |
| 19 | 19-a | 1 | | | | | yes | |
| 19 | 19-b | 1 | | | | | yes | |
| 19 | **19-c** | 1 | | **yes** | | | yes | |
| 19 | **19-d** | 1 | | **yes** | | yes | yes | yes |

**Why they matched:**
- **One generic group.** The two molecules are drawn side by side in one scheme of paper 19. Each
  has the text label "Heteroaryl/Aryl" in place of one group. The ground truth writes that group
  as a single `*` atom bonded to a carbon.
- **MolGlyph read both drawings correctly,** inside BioMiner, and it also wrote the label as one
  `*` in the same place. On one it read the label text as "Hinyl"; on the other it read no
  label. Either way, the canonical SMILES equal the ground truth's.
- **The LLM tied neither drawing to a compound label.** So both are in "all extracted" but not in
  the linked table.

**Why the variant loses them:**
- **The variant drops every structure with a `*` atom.** These two are whole drawn molecules with
  one generic group, not R-group scaffold pieces, but the SMILES alone cannot tell the two kinds
  apart.
- **Under InChIKey the same two are lost** (80 to 78). InChI cannot handle `*` atoms, so both
  rules compare these entries by SMILES.
- **Both are also found by crop readers,** so the count of molecules only BioMiner found does not
  change.
- **BioMiner found none of the other three `*` entries.** The paper 2 entry is a small piece with
  five `*` atoms.

### Next to Obaidullah's BioMiner run

Obaidullah's numbers are from the group report, Tables 5 and 6:
- internal precision 72.4%, recall 34.5% and F1 46.8%;
- per-paper recall: paper 1 100, paper 2 26, paper 4 40, paper 6 0, paper 16 50, paper 19 23.

His model is not confirmed yet; it is probably BioMiner-Instruct.

- **His figures imply about 105 predictions.** 34.5% of 220 is about 76 molecules found, and 76 at
  72.4% precision is about 105 predictions. That matches the 105 he reported.
- **His recall matches ours under "all extracted", not "linked."** Per paper, his recall is within
  2 points of ours under "all extracted", on every paper. Under "linked", our recall is 17.2 points
  lower on paper 2, 20 on paper 4 and 10.5 on paper 19.
- **"All extracted, no `*` pieces" comes closest overall.** It has 95 predictions with 75 correct:
  precision 0.789, recall 0.341, F1 0.476. His are 0.724, 0.345 and 0.468.
- **So his count most likely included structures that the LLM did not tie to a label.** This
  cannot be confirmed without his output files. A stronger linking model is the other possible
  explanation.

### Paired comparison with the crop readers

Each cell gives the ground-truth molecules found only by BioMiner vs only by the reader. Then comes
the two-sided exact McNemar p-value, a binomial test on those two counts. The rule is canonical
SMILES. Under InChIKey a few counts move by one or two, and no conclusion below changes. The pooled
values become: linked vs MolGlyph 13 vs 29 (p = 0.020), all extracted vs MolScribe 31 vs 2
(p = 1.3e-07), and all extracted vs MolSight 29 vs 3 (p = 2.6e-06).

| Scope | Linked vs MolScribe | Linked vs MolSight | Linked vs MolGlyph | All extracted vs MolScribe | All extracted vs MolSight | All extracted vs MolGlyph |
|---|---|---|---|---|---|---|
| Pooled | 20 vs 23, p = 0.76 | 17 vs 23, p = 0.43 | 13 vs 27, p = 0.038 | 29 vs 2, p = 4.6e-07 | 27 vs 3, p = 8.4e-06 | 17 vs 1, p = 0.00014 |
| Paper 1 | 4 vs 0, p = 0.12 | 1 vs 0, p = 1 | 0 vs 0, p = 1 | 4 vs 0, p = 0.12 | 1 vs 0, p = 1 | 0 vs 0, p = 1 |
| Paper 2 | 0 vs 5, p = 0.062 | 0 vs 4, p = 0.12 | 0 vs 6, p = 0.031 | 2 vs 1, p = 1 | 2 vs 0, p = 0.5 | 0 vs 0, p = 1 |
| Paper 4 | 10 vs 9, p = 1 | 10 vs 4, p = 0.18 | 10 vs 11, p = 1 | 13 vs 1, p = 0.0018 | 17 vs 0, p = 1.5e-05 | 11 vs 1, p = 0.0063 |
| Paper 16 | 1 vs 0, p = 1 | 1 vs 0, p = 1 | 0 vs 0, p = 1 | 1 vs 0, p = 1 | 1 vs 0, p = 1 | 0 vs 0, p = 1 |
| Paper 19 | 5 vs 9, p = 0.42 | 5 vs 15, p = 0.041 | 3 vs 10, p = 0.092 | 9 vs 0, p = 0.0039 | 6 vs 3, p = 0.51 | 6 vs 0, p = 0.031 |

Paper 6 is 0 vs 0 in every pair.

- **No reader differs significantly from BioMiner under the pipeline's rule.** After a Bonferroni
  correction for the six pooled tests (threshold 0.0083), "linked" differs from none of the three
  readers. MolGlyph's lead (27 vs 13, p = 0.038) does not pass.
- **Under "all extracted", BioMiner finds more than every reader, and this passes the
  correction.** It comes mainly from papers 4 and 19.
- **Part of that is the same reader.** "All extracted" includes MolGlyph's own reads of BioMiner's
  boxes, so against MolGlyph it partly compares two crop sets of the same reader.
- **The test assumes molecules are independent.** They cluster by paper, as the wide paper-bootstrap
  intervals show.

### Molecules BioMiner found that no crop reader found

| Paper | Found by BioMiner (linked) but by no reader | Found by any reader | Found by any reader or BioMiner |
|---|---|---|---|
| 1 | 0 | 17 | 17 |
| 2 | 0 | 10 | 10 |
| 4 | **9** | 15 | 24 |
| 6 | 0 | 0 | 0 |
| 16 | 0 | 2 | 2 |
| 19 | **3** | 26 | 29 |
| **Total** | **12** | **70** | **82** |

- **All 12 come from BioMiner's R-group enumeration.** They are in the R-group table, not in the
  OCSR output. On the papers that define compounds through R-group tables (4 and 19), this is what
  BioMiner adds to a crop reader.
- **"All extracted" adds no further molecule that no reader found.** The 30 correct structures
  outside the merged table were all found by at least one crop reader.
- **Under InChIKey the counts are the same.** 9 and 3 found only by BioMiner, 73 by any reader
  (as in section 6 of the earlier doc), and 85 in all.
- **138 of the 220 molecules are still found by no tool.**

## 6. BioVista papers: scores

The 12 BioVista papers are 40, 65, 67, 68, 106, 153, 156, 163, 239, 297, 312 and 471. Their labels
are in `data/biovista/labels`: 415 distinct labelled structures by canonical SMILES.

The labels are split the way BioMiner's own evaluator splits them (`BioMiner/runner/metric_fn.py`):

| Split | Labels | Scored against | Distinct labels |
|---|---|---|---|
| Full structures | rows with no backbone | BioMiner's full-structure table | 163 |
| R-group tables | rows with a backbone | BioMiner's R-group table | 256 |
| Chiral | SMILES with `@` | BioMiner's linked structures with `@` | 211 |
| All | every row | the merged table | 415 |

The splits overlap a little: a molecule can sit both in a table and in a full drawing. The canonical
SMILES rule and the InChIKey rule give the same numbers on these papers.

**Pooled over the 12 papers** (recall and precision with 95% Wilson intervals, then the paper bootstrap)

| Split | Recall | Precision | Paper bootstrap (recall / precision) | Mean over papers (recall / precision) |
|---|---|---|---|---|
| All, linked | 0.313 [0.271, 0.359] (130 of 415) | 0.522 [0.460, 0.583] (130 of 249) | [0.176, 0.481] / [0.314, 0.737] | 0.338 / 0.516 |
| Full structures | 0.350 [0.281, 0.426] (57 of 163) | 0.582 [0.483, 0.674] (57 of 98) | [0.246, 0.525] / [0.415, 0.844] | 0.361 / 0.640 |
| R-group tables | 0.289 [0.237, 0.347] (74 of 256) | 0.484 [0.406, 0.562] (74 of 153) | [0.085, 0.524] / [0.158, 0.762] | 0.283 / 0.356 |
| Chiral | 0.209 [0.159, 0.268] (44 of 211) | 0.297 [0.230, 0.375] (44 of 148) | [0.057, 0.410] / [0.073, 0.590] | 0.221 / 0.327 |
| All, all extracted | 0.407 [0.361, 0.455] (169 of 415) | 0.324 [0.286, 0.366] (169 of 521) | [0.256, 0.583] / [0.209, 0.452] | 0.455 / 0.357 |
| All, all extracted, no `*` pieces | 0.407 [0.361, 0.455] (169 of 415) | 0.537 [0.481, 0.591] (169 of 315) | [0.256, 0.583] / [0.363, 0.715] | 0.455 / 0.523 |

**Per paper** (all splits, linked; found of labels, then correct of distinct predictions)

| Paper | Recall | Precision | BioMiner's evaluator (recall / precision) |
|---|---|---|---|
| 40 | 0.625 (25 of 40) | 0.862 (25 of 29) | 0.634 / 0.862 |
| 65 | 0.089 (4 of 45) | 0.129 (4 of 31) | 0.085 / 0.129 |
| 67 | 0.238 (10 of 42) | 0.357 (10 of 28) | 0.217 / 0.333 |
| 68 | 0.580 (29 of 50) | 0.853 (29 of 34) | 0.618 / 0.857 |
| 106 | 0.818 (18 of 22) | 0.947 (18 of 19) | 0.818 / 0.947 |
| 153 | 0.762 (16 of 21) | 1.000 (16 of 16) | 0.762 / 1.000 |
| 156 | 0.000 (0 of 17) | 0.000 (0 of 17) | 0.000 / 0.000 |
| 163 | 0.024 (1 of 42) | 0.167 (1 of 6) | 0.023 / 0.167 |
| 239 | 0.122 (6 of 49) | 0.545 (6 of 11) | 0.115 / 0.545 |
| 297 | 0.385 (5 of 13) | 0.625 (5 of 8) | 0.385 / 0.714 |
| 312 | 0.057 (2 of 35) | 0.071 (2 of 28) | 0.036 / 0.062 |
| 471 | 0.359 (14 of 39) | 0.636 (14 of 22) | 0.359 / 0.696 |

### BioMiner's own `--biovista_evaluate` output, against the published numbers

The stage ran BioMiner with `--biovista_evaluate`, and all 12 papers have its output. Its numbers
differ slightly from ours, for two reasons:
- **Its matching rule:** it neutralizes charges and treats the first `[1*]` as `*` (see `chem.py`).
- **Its denominators:** it counts rows, not distinct molecules.

Like the published figures, these are means over papers of structure-only recall / precision.

| Split | This run (12 papers, Gemini 3.5 Flash) | Published (450 papers, BioMiner-Instruct) |
|---|---|---|
| All | 0.338 / 0.526 | 0.580 / 0.615 |
| Full structures ("entire") | 0.358 / 0.650 | 0.654 / 0.630 |
| R-group tables ("scaffold") | 0.288 / 0.356 | 0.339 / 0.390 |
| Chiral | 0.219 / 0.331 | 0.454 / 0.467 |

The published numbers are from BioMiner's README at commit `17c6161` ("Evaluation on BioVista").
They were made with BioMiner-Instruct, BioMiner's own fine-tuned model, not Gemini, on BioVista's
450 test papers. This run used Gemini 3.5 Flash on 12 papers. The paper samples differ, and so do
the models.

- **Precision is close to the published value:** 0.526 against 0.615 overall.
- **Recall is well below:** 0.338 against 0.580. The gap is largest for full structures (0.358
  against 0.654) and for chiral ones (0.219 against 0.454). For R-group tables it is small (0.288
  against 0.339).
- **The papers vary widely.** Papers 106, 153, 40 and 68 reach recall 0.58 to 0.82. Papers 156,
  163, 312 and 65 stay below 0.1, and paper 156 got 0 of 17. With 12 papers, the paper-bootstrap
  interval for pooled recall runs from 0.18 to 0.48.

### Linking versus finding, by type

How much of the recall gap comes from BioMiner not *finding* a structure, and how much from Gemini
not *linking* a structure it found to a compound label? To see, the same label types are scored
two ways:

- **Linked:** BioMiner's structure tables, paired by type as above.
- **All extracted, no `*` pieces:** the linked structures plus every structure BioMiner's OCSR
  read, without `*` atoms, linked or not. Chiral labels are scored only against structures with
  `@`, as BioMiner does.

Scoring each type against all linked tables, instead of only its own table, gives exactly the same
numbers on these 12 papers. So BioMiner's pairing by type is not a factor.

**Recall:** pooled [95% Wilson] (found of labels), then the mean over papers. The published values
are means over papers.

| Type (labels) | Linked | All extracted, no `*` pieces | Published (BioMiner-Instruct, 450 papers) |
|---|---|---|---|
| All (415) | 0.313 [0.271, 0.359] (130); mean 0.338 | 0.407 [0.361, 0.455] (169); mean 0.455 | 0.580 |
| Full structures (163) | 0.350 [0.281, 0.426] (57); mean 0.361 | 0.583 [0.506, 0.656] (95); mean 0.594 | 0.654 |
| R-group tables (256) | 0.289 [0.237, 0.347] (74); mean 0.283 | 0.293 [0.241, 0.351] (75); mean 0.286 | 0.339 |
| Chiral (211) | 0.209 [0.159, 0.268] (44); mean 0.221 | 0.294 [0.237, 0.359] (62); mean 0.369 | 0.454 |

**The gap in mean recall,** and how much of it counting the unlinked structures closes:

| Type | Gap to published (linked) | Closed by counting unlinked structures | Left |
|---|---|---|---|
| All | 0.242 | 0.117 (48%) | 0.125 |
| Full structures | 0.293 | 0.233 (80%) | 0.060 |
| R-group tables | 0.056 | 0.003 (5%) | 0.053 |
| Chiral | 0.233 | 0.148 (64%) | 0.085 |

**Precision** stays about the same when the unlinked structures are added:
- **All:** 0.522 [0.460, 0.583] (130 of 249) linked, against 0.537 [0.481, 0.591] (169 of 315).
- **Chiral:** 0.297 [0.230, 0.375] (44 of 148), against 0.335 [0.271, 0.406] (62 of 185).
- **Published:** 0.615 (all) and 0.467 (chiral).

Precision by type is given only for these two. OCSR reads carry no full-or-R-group type, so the
other two types have no precision.

**What this shows:**

- **On full structures, "all extracted" is a ceiling for any linking step.** Every structure in
  BioMiner's full-structure table is one of its OCSR reads, with no exception in all 18 papers.
  So no linking step could keep more full structures than "all extracted" holds.
- **Full structures:** about 80% of the gap is linking. MolGlyph read the molecule, but Gemini did
  not tie it to a compound label. What is left, 0.06, is small enough that the different paper
  samples could explain it.
- **Chiral structures:** about two thirds of the gap is linking.
- **R-group tables:** almost none of the gap is linking in this sense. These molecules exist only
  after BioMiner enumerates a table, which is itself an LLM step: 150 of the 153 structures in
  BioVista's R-group tables are enumerations, not OCSR reads. The remaining 0.05 is in the
  enumeration.
- **Overall,** about half the gap closes. The rest comes mostly from the R-group tables, which hold
  256 of the 415 labels on these papers.
- **Caveats.** These are 12 papers against 450, and the rule is canonical SMILES. BioMiner's own
  rule gave linked means within 0.005 of these (see the evaluator table above).

## 7. What this means

- **The Vertex AI route works.** 330 Gemini calls, none lost, in about 70 minutes, for $6.48 at
  list price. Four short rate limits were retried and passed. The instance needed no key.
- **The 58 predictions are a counting rule, not a bug.** The pipeline keeps only the structures
  the LLM tied to a compound label. On the course papers that is a careful, precise list: 81% of
  BioMiner's linked structures are right, against 30 to 37% for the crop readers. But it finds
  only 21% of the molecules, fewer than any crop reader.
- **The other count matches Obaidullah's.** Counting every whole structure BioMiner read gives 34%
  recall at 79% precision. That is close to his 34.5% and 72.4%.
- **BioMiner's own contribution is R-group enumeration.** It found 12 molecules that no crop reader
  found, 9 in paper 4 and 3 in paper 19, all by filling in R-group tables. Recall on those two
  papers is still low, at 38% and 25% at best.
- **On BioVista, Gemini 3.5 Flash in BioMiner's linking role recalls fewer structures than the
  published BioMiner-Instruct.** Mean structure recall is 0.34 against 0.58, while precision is
  close: 0.53 against 0.62.
- **Most of that gap is linking, not finding.** On full structures, MolGlyph had read most of the
  missing molecules, but Gemini did not tie them to a compound label. Counting them closes 80% of
  the gap: mean recall 0.59 against the published 0.65. On chiral structures it closes about two
  thirds. On R-group tables it closes almost nothing: there the shortfall is in BioMiner's
  enumeration of the tables, which is also an LLM step.
- **Twelve papers is a small sample.** Part of what is left could be the different papers.

## 8. Notes for methods.md

These are facts to add to `docs/methods.md`; that file was not edited here.

- **The BioMiner run:** S3 folder `20261105T043509Z`, g6.2xlarge (NVIDIA L4), BioMiner commit
  `17c6161`. The LLM was Gemini 3.5 Flash on Vertex AI (OpenAI-compatible endpoint), signed in
  through AWS-to-Google workload identity federation. The API reported `google/gemini-3.5-flash`
  for every call. Settings: temperature 0, and JSON output for image calls. Up to 5 retries for
  server errors, connection errors and 429s that ask for a wait of 300 s or less.
- **Times:** smoke test 3 min; course papers 22 min (1.6 to 5.6 per paper); BioVista papers 43 min
  (1.7 to 6.1 per paper).
- **Calls and cost:** 330 calls answered, 4 retried (all short 429s), 0 lost. 1.06 million input,
  0.09 million output and 0.46 million thinking tokens: $6.48 at list price.
- **BioMiner predictions on the course papers.** "Linked structures" (the pipeline's rule) is the
  `_structure_merged_output_` table: full structures and R-group enumerations that the LLM tied to
  a compound label. It does not depend on activity values. "All extracted structures" adds every
  structure in BioMiner's OCSR output (`<task>_ocsr_output.json`). The variant "no `*` pieces"
  leaves out R-group scaffolds and substituent pieces.
- **Course scores (canonical SMILES rule):**

  | Rule | Recall | Precision | F1 |
  |---|---|---|---|
  | Linked | 0.214 [0.165, 0.272] (47 of 220) | 0.810 (47 of 58) | 0.338 |
  | All extracted | 0.350 (77) | 0.445 (of 173) | 0.392 |
  | All extracted, no `*` pieces | 0.341 (75) | 0.789 (of 95) | 0.476 |

- **Molecules only BioMiner found:** 12 that no crop reader found (9 in paper 4, 3 in paper 19),
  all from R-group enumeration.
- **BioVista,** 12 papers, 415 labelled structures. Pooled structure recall 0.313, precision 0.522.
  BioMiner's own evaluator, as means over papers: 0.338 / 0.526, against the published 0.580 /
  0.615 for BioMiner-Instruct on 450 papers.
- **Linking vs finding on BioVista** (means over papers, canonical SMILES):

  | Type | Linked | All extracted, no `*` pieces | Published | Share of the gap closed |
  |---|---|---|---|---|
  | Full structures | 0.361 | 0.594 | 0.654 | 80% |
  | Chiral | 0.221 | 0.369 | 0.454 | about two thirds |
  | R-group tables | 0.283 | 0.286 | 0.339 | almost none |

  Every full-table structure is an OCSR read, so for full structures "all extracted" is a ceiling
  for any linking step. R-group-table structures are enumerations (150 of 153), so no OCSR read
  can make up for them.
- **The "no `*` pieces" variant also drops whole molecules drawn with a generic group.** On the
  course papers these are two paper-19 molecules (19-c and 19-d above), each drawn with a
  "Heteroaryl/Aryl" label that the ground truth writes as `*`. That is why its recall is 0.341,
  not 0.350.
- **Obaidullah's BioMiner numbers** (72.4 / 34.5 / 46.8) are consistent with counting every
  structure BioMiner read, not only the linked ones. His model is not yet confirmed.
