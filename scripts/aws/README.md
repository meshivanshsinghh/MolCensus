# GPU run on AWS

One trip to a single-GPU instance runs everything that needs a GPU: three crop readers (MolScribe,
MolSight, MolGlyph) on the course crops and on the full public sets, and BioMiner end to end on
BioVista papers and, with permission, on the course papers. Scoring of the course papers happens
back on the Mac. **The ground-truth CSV never leaves the Mac**: the instance uses a header-only
placeholder (`placeholder_ground_truth.csv`).

| File | What it does |
|---|---|
| `setup.sh` | Installs everything on a fresh instance. Safe to rerun. Log: `~/setup.log` |
| `run_all.sh` | Runs stages a to e, then stops the instance. Logs: `~/runs/` |
| `patch_biominer.py` | Local BioMiner edits: one GPU, retries for failed MLLM calls (called by setup.sh) |
| `biominer_gemini.py` | Writes BioMiner's Gemini configs from the keys file and tests each key (called by setup.sh) |
| `../../configs/aws*.yaml` | Instance configs; `aws_score.yaml` is for scoring on the Mac |

## Before launch (on the Mac)

1. AWS: quota of at least 8 vCPUs for "Running On-Demand G and VT instances" in your region, the AWS
   CLI logged in (`aws configure`), and a billing alarm.
2. Hugging Face: accept the terms at https://huggingface.co/jiaxianustc/MolGlyph (it is gated; the
   dry run here got 401 without that) and create a read token.
3. Gemini: a free-tier key (public papers) and, only once the course allows sending its PDFs to
   Google, a paid key. Every BioMiner run uses `gemini-3.5-flash` (set in `biominer_gemini.py`):
   in a test on 2026-10-29, `gemini-3.8-flash` returned 503 "high demand" twice, while
   `gemini-3.5-flash` read an image with `response_format: json_object` correctly after one 503.
4. The keys file, outside the repo: `~/.molcensus/molcensus_keys.env` (folder mode 700, file mode
   600). It starts as a template with these names and empty values; fill them in with an editor:

   | Name | What |
   |---|---|
   | `HF_TOKEN` | Hugging Face read token (MolGlyph's terms accepted first) |
   | `GEMINI_API_KEY_FREE` | Gemini key, free tier: public papers only |
   | `GEMINI_API_KEY_PAID` | Gemini key, paid tier: course papers; leave empty until permitted |

5. BioVista PDFs you want BioMiner to read go in `data/biovista/pdfs/`, named `<index>_<pdb id>.pdf`
   (for example `68_6r8r.pdf`). The two in BioMiner's `example/pdfs/` are added on the instance.
   BioVista's labels go in `data/biovista/labels/` (500 `<name>_structure.csv` plus `<name>_data.csv`)
   and `data/biovista/structure_annotation/` (`labels/INDEX_general_PL_*.2020`, `data/<pdb id>/`).

## Launch

g6.2xlarge (L4, about $0.98/h on demand in us-east-1) or g5.2xlarge (A10G, about $1.21/h); both have
24 GB of GPU memory. Check current prices. `--instance-initiated-shutdown-behavior stop` matters:
`run_all.sh` ends with `shutdown`, and with "terminate" the disk and all results would be deleted.

```bash
REGION=us-east-1
AMI=$(aws ssm get-parameter --region $REGION --query Parameter.Value --output text \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id)
aws ec2 create-key-pair --region $REGION --key-name molcensus --query KeyMaterial --output text > ~/.ssh/molcensus.pem
chmod 600 ~/.ssh/molcensus.pem
SG=$(aws ec2 create-security-group --region $REGION --group-name molcensus-ssh --description "SSH for MolCensus" --query GroupId --output text)
aws ec2 authorize-security-group-ingress --region $REGION --group-id $SG --protocol tcp --port 22 --cidr "$(curl -s https://checkip.amazonaws.com)/32"
ID=$(aws ec2 run-instances --region $REGION --image-id $AMI --instance-type g6.2xlarge \
  --key-name molcensus --security-group-ids $SG --instance-initiated-shutdown-behavior stop \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=molcensus}]' \
  --query 'Instances[0].InstanceId' --output text)
aws ec2 wait instance-status-ok --region $REGION --instance-ids $ID
HOST=ubuntu@$(aws ec2 describe-instances --region $REGION --instance-ids $ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
SSH="ssh -i ~/.ssh/molcensus.pem"
```

The key pair and security group only need creating once. After a stop and start, the public IP
changes: rerun the `HOST=` line.

## Copy up

```bash
cd ~/Professional/MolCensus
# Code: only tracked files, so nothing private (data/, results/predictions/, results/scored/) goes up.
git archive --format=tar HEAD | $SSH $HOST 'mkdir -p molcensus && tar -x -C molcensus'
scp -i ~/.ssh/molcensus.pem ~/.molcensus/molcensus_keys.env $HOST:molcensus_keys.env
# The 237 course crops and their index (run_all.sh rewrites the Mac paths in it).
$SSH $HOST 'mkdir -p molcensus/outputs/work molcensus/data/biovista/pdfs'
rsync -a -e "$SSH" outputs/work/crops_moldet outputs/work/crops_index.csv $HOST:molcensus/outputs/work/
# BioVista PDFs and labels (labels/ and structure_annotation/, for --biovista_evaluate in stage c).
rsync -az --exclude download/ -e "$SSH" data/biovista/ $HOST:molcensus/data/biovista/
# Only with permission to send the course PDFs to Gemini (stage d is skipped otherwise):
# rsync -a -e "$SSH" data/pdfs/*.pdf $HOST:molcensus/data/pdfs/ && $SSH $HOST touch COURSE_PDFS_PERMISSION_GIVEN
```

Never copy up `data/ground_truth_*.csv`, `results/predictions/` or `results/scored/`.

## Setup (30 to 60 minutes, mostly BioMiner's conda environment)

```bash
$SSH -t $HOST "tmux new -s setup 'bash molcensus/scripts/aws/setup.sh; bash'"
```

It ends with `=== setup.sh finished`. If it stops, fix the cause and run the same command again;
finished steps are skipped. If `torch CUDA` fails, the AMI's NVIDIA driver is older than the CUDA
build of torch from PyPI: install torch from `https://download.pytorch.org/whl/cu126` into
`molcensus/.venv` and rerun. A failed Gemini check only warns, because the crop-reader stages do not
need Gemini.

## Run (in tmux, so it survives a dropped connection)

```bash
$SSH $HOST "mkdir -p runs && tmux new -d -s run 'bash molcensus/scripts/aws/run_all.sh 2>&1 | tee -a runs/run_all.log'"
```

| Stage | What | Needs | Estimate |
|---|---|---|---|
| a. course_readers | MolScribe, MolSight, MolGlyph on the 237 course crops | crops copied up | about 10 min |
| b. public_readers | the three readers on all of USPTO, UOB, CLEF, ACS and BioVista OCSR (21,643 images), scored on the instance | downloads itself | 1.5 to 3.5 h |
| c. biominer_biovista | BioMiner servers, smoke test on `example/pdfs/68_6r8r.pdf`, then every BioVista PDF | free Gemini key | 15 min plus 3 to 10 min per paper |
| d. biominer_course | BioMiner on the six course PDFs | paid key, PDFs, `~/COURSE_PDFS_PERMISSION_GIVEN` | 20 to 60 min |
| e. molrecbench_wild | the three readers on MolRecBench-Wild | `data/public/molrecbench_wild/molrecbench_wild.csv` | 20 to 50 min |

The estimates are extrapolated from Mac CPU timings, not measured on a GPU. The instance stops
itself after the last stage, or after 10 hours at the latest (`sudo shutdown -c` cancels that).
Stage c copies `data/biovista/labels/` to `BioVista/bioactivity_extraction/labels/` and
`data/biovista/structure_annotation/` to `BioVista/structure_annotation/` inside the BioMiner clone,
then adds `--biovista_evaluate`. If either folder is missing it runs without that flag. Where the two
label sets come from is in docs/methods.md (BioVista paragraph).

## Check progress

```bash
$SSH $HOST 'tail -n 15 runs/run_all.log; ls runs/*.done 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'
$SSH $HOST 'tail -n 5 runs/public_readers.log'                   # one stage's log
$SSH $HOST 'cat molcensus/outputs/work*/biominer_output/*/run_info.json'   # BioMiner: models, errors, lost MLLM calls
$SSH -t $HOST 'tmux attach -t run'                                # watch live; detach with Ctrl-b d
```

To redo a stage, delete its `runs/<stage>.done` and start `run_all.sh` again; done stages are skipped.

## Copy back

If the instance has stopped, start it first:
`aws ec2 start-instances --region $REGION --instance-ids $ID && aws ec2 wait instance-status-ok --region $REGION --instance-ids $ID`,
then rerun the `HOST=` line.

```bash
cd ~/Professional/MolCensus
for d in aws aws_biovista aws_molrecbench; do                 # predictions and public metrics
  rsync -a -e "$SSH" $HOST:molcensus/results/$d results/
done
# Raw BioMiner tables, logs and run_info.json. Kept in separate folders: BioVista and course paper IDs can overlap.
rsync -a -e "$SSH" $HOST:molcensus/outputs/work/biominer_output/ outputs/work/biominer_output/
rsync -a -e "$SSH" $HOST:molcensus/outputs/work_biovista/biominer_output/ outputs/work_biovista/biominer_output/
mkdir -p outputs/aws_logs
rsync -a -e "$SSH" $HOST:runs/ outputs/aws_logs/runs/
rsync -a -e "$SSH" $HOST:setup.log outputs/aws_logs/
```

A stage that did not run has no folder; rsync then prints an error for it and carries on.

`results/aws*/predictions/` and `outputs/` are git-ignored: the course predictions contain SMILES.
`results/aws/public/` holds only public data and can be committed after a look.

## Score on the Mac

```bash
python run_pipeline.py --config configs/aws_score.yaml --steps score,report --tools molscribe,molsight,molglyph,biominer
```

This writes `results/aws/per_paper_metrics.csv`, `comparison.csv` and `RESULTS.md`, with one column per
tool, including `biominer_bioactivity`.

## Clean up

```bash
$SSH $HOST 'rm -rf molcensus/data/pdfs molcensus/outputs/work/crops_moldet molcensus_keys.env'
aws ec2 terminate-instances --region $REGION --instance-ids $ID   # also deletes its disk
```
