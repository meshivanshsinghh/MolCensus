# GPU run on AWS

One trip to a single-GPU instance runs everything that needs a GPU: three crop readers (MolScribe,
MolSight, MolGlyph) on the course crops and on the full public sets, and BioMiner end to end on the
course papers (paid Gemini key, with permission) and on BioVista papers (free key). Scoring of the
course papers happens back on the Mac. **The ground-truth CSV never leaves the Mac**: the instance
uses a header-only placeholder (`placeholder_ground_truth.csv`).

| File | What it does |
|---|---|
| `setup.sh` | Installs everything on a fresh instance. Safe to rerun. Log: `~/setup.log` |
| `run_all.sh` | Runs the stages in order, then stops the instance. Logs: `~/runs/` |
| `patch_biominer.py` | Local BioMiner edits: one GPU, retries for failed MLLM calls, token counts (called by setup.sh) |
| `biominer_gemini.py` | Writes BioMiner's two Gemini configs (free key, paid key) and tests both keys (called by setup.sh) |
| `../../configs/aws*.yaml` | Instance configs; `aws_score.yaml` is for scoring on the Mac |

## Before launch (on the Mac)

1. AWS: install AWS CLI 2.32.0 or later (`aws --version`), sign in, and confirm who you are:

   ```bash
   aws login                     # short-term credentials, valid for at most 12 hours; add --remote if no browser opens
   aws sts get-caller-identity
   ```

   Also set a billing alarm, and check (read-only) the quota of at least 8 vCPUs for "Running
   On-Demand G and VT instances". It shows 8 in us-east-2, which the commands below use:

   ```bash
   for r in us-east-1 us-east-2; do
     echo "$r: $(aws service-quotas get-service-quota --region $r --service-code ec2 --quota-code L-DB2E81BA --query Quota.Value --output text)"
   done
   ```

2. Hugging Face: accept the terms at https://huggingface.co/jiaxianustc/MolGlyph (it is gated; the
   dry run here got 401 without that) and create a read token.
3. Gemini: two keys. The free-tier key reads the BioVista papers; the paid-tier key reads the six
   course papers. `course_tier: paid` in `configs/aws.yaml` picks the key for the course papers, and
   the smoke test uses the same key as stage d. Both keys use `gemini-3.5-flash` (set in
   `biominer_gemini.py`): in a test on 2026-10-29, `gemini-3.8-flash` returned 503 "high demand"
   twice, while `gemini-3.5-flash` read an image with `response_format: json_object` correctly after
   one 503.
4. The keys file, outside the repo: `~/.molcensus/molcensus_keys.env` (folder mode 700, file mode
   600). It starts as a template with these names and empty values; fill them in with an editor:

   | Name | What |
   |---|---|
   | `HF_TOKEN` | Hugging Face read token (MolGlyph's terms accepted first) |
   | `GEMINI_API_KEY_FREE` | Gemini key, free tier: the BioVista papers |
   | `GEMINI_API_KEY_PAID` | Gemini key, paid tier: the smoke test and the six course papers |

5. BioVista PDFs you want BioMiner to read go in `data/biovista/pdfs/`, named `<index>_<pdb id>.pdf`
   (for example `68_6r8r.pdf`). The two in BioMiner's `example/pdfs/` are added on the instance.
   BioVista's labels go in `data/biovista/labels/` (500 `<name>_structure.csv` plus `<name>_data.csv`)
   and `data/biovista/structure_annotation/` (`labels/INDEX_general_PL_*.2020`, `data/<pdb id>/`).
6. Course papers: stage d sends the six course PDFs to Gemini (the `course_tier` key) only if the
   permission file `~/COURSE_PDFS_PERMISSION_GIVEN` exists on the instance. It lives at the same path
   on the Mac (outside the repo) and records who approved it, how, when and for what; "Copy up" sends
   it. Stage d prints it into its log, and each paper's `run_info.json` records the tier actually
   used. If you switch `course_tier` to `free`, also change the Scope line in
   `~/COURSE_PDFS_PERMISSION_GIVEN` to "free tier" to match.

## Launch

These commands are written for bash. In zsh (the macOS default), `$SSH` does not split into `ssh`
and its options, so type `bash` first in that terminal tab.

g6.2xlarge (L4) or g5.2xlarge (A10G); both have 24 GB of GPU memory. In us-east-1 they cost about
$0.98 and $1.21 per hour on demand; check the current us-east-2 prices.
`--instance-initiated-shutdown-behavior stop` matters: `run_all.sh` ends with `shutdown`, and with
"terminate" the disk and all results would be deleted. Both blocks below are safe to rerun.

One time: the key pair and the security group, each created only if it does not exist yet. The key is
written to a temporary file first, so a failed call never leaves an empty `.pem`. If AWS still has a
key pair named `molcensus` but `~/.ssh/molcensus.pem` is gone, delete the old one first
(`aws ec2 delete-key-pair --region $REGION --key-name molcensus`).

```bash
REGION=us-east-2
if [ ! -f ~/.ssh/molcensus.pem ]; then
  (umask 077 && aws ec2 create-key-pair --region $REGION --key-name molcensus --query KeyMaterial --output text \
    > ~/.ssh/molcensus.pem.new && mv ~/.ssh/molcensus.pem.new ~/.ssh/molcensus.pem) || rm -f ~/.ssh/molcensus.pem.new
fi
if ! aws ec2 describe-security-groups --region $REGION --filters Name=group-name,Values=molcensus-ssh \
     --query 'SecurityGroups[0].GroupId' --output text | grep -q '^sg-'; then
  SG=$(aws ec2 create-security-group --region $REGION --group-name molcensus-ssh --description "SSH for MolCensus" --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --region $REGION --group-id $SG --protocol tcp --port 22 --cidr "$(curl -s https://checkip.amazonaws.com)/32"
fi
```

Each launch:

```bash
REGION=us-east-2
SG=$(aws ec2 describe-security-groups --region $REGION --filters Name=group-name,Values=molcensus-ssh --query 'SecurityGroups[0].GroupId' --output text)
AMI=$(aws ssm get-parameter --region $REGION --query Parameter.Value --output text \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id)
ID=$(aws ec2 run-instances --region $REGION --image-id $AMI --instance-type g6.2xlarge \
  --key-name molcensus --security-group-ids $SG --instance-initiated-shutdown-behavior stop \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=molcensus}]' \
  --query 'Instances[0].InstanceId' --output text)
aws ec2 wait instance-status-ok --region $REGION --instance-ids $ID
HOST=ubuntu@$(aws ec2 describe-instances --region $REGION --instance-ids $ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
SSH="ssh -i ~/.ssh/molcensus.pem"
```

If your IP address changed since the security group was made, add the new one first (see "Reconnect
from a new terminal"). If `run-instances` fails with
`InsufficientInstanceCapacity`, wait a few minutes and run it again, or add `--subnet-id` with a subnet
in another zone (list them: `aws ec2 describe-subnets --region $REGION --query 'Subnets[].[SubnetId,AvailabilityZone]' --output text`).

### Reconnect from a new terminal

These commands are written for bash. In zsh (the macOS default), `$SSH` does not split into `ssh`
and its options, so type `bash` first in that terminal tab.

```bash
aws login                     # the sign-in lasts at most 12 hours
REGION=us-east-2
SSH="ssh -i ~/.ssh/molcensus.pem"
ID=$(aws ec2 describe-instances --region $REGION \
  --filters Name=tag:Name,Values=molcensus Name=instance-state-name,Values=running,stopped \
  --query 'Reservations[].Instances[].InstanceId' --output text)
HOST=ubuntu@$(aws ec2 describe-instances --region $REGION --instance-ids $ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "$ID $HOST"   # a stopped instance has no address ("None"): start it first (see "Copy back")
```

If SSH times out because your network (and so your IP address) changed, add the new address:

```bash
SG=$(aws ec2 describe-security-groups --region $REGION --filters Name=group-name,Values=molcensus-ssh --query 'SecurityGroups[0].GroupId' --output text)
aws ec2 authorize-security-group-ingress --region $REGION --group-id $SG --protocol tcp --port 22 --cidr "$(curl -s https://checkip.amazonaws.com)/32"
```

## Copy up

```bash
cd ~/Professional/MolCensus
# Code: only tracked files, so nothing private (data/, results/predictions/, results/scored/) goes up.
git archive --format=tar HEAD | $SSH $HOST 'mkdir -p molcensus && tar -x -C molcensus'
scp -i ~/.ssh/molcensus.pem ~/.molcensus/molcensus_keys.env $HOST:molcensus_keys.env
$SSH $HOST chmod 600 molcensus_keys.env
# The 237 course crops and their index (run_all.sh rewrites the Mac paths in it).
$SSH $HOST 'mkdir -p molcensus/outputs/work molcensus/data/biovista/pdfs molcensus/data/pdfs'
rsync -a -e "$SSH" outputs/work/crops_moldet outputs/work/crops_index.csv $HOST:molcensus/outputs/work/
# BioVista PDFs and labels (labels/ and structure_annotation/, for --biovista_evaluate in stage c).
rsync -az --exclude download/ -e "$SSH" data/biovista/ $HOST:molcensus/data/biovista/
# The six course PDFs and the permission file. Without the file, stage d skips, with no error.
rsync -a -e "$SSH" data/pdfs/*.pdf $HOST:molcensus/data/pdfs/
scp -i ~/.ssh/molcensus.pem ~/COURSE_PDFS_PERMISSION_GIVEN $HOST:COURSE_PDFS_PERMISSION_GIVEN
$SSH $HOST 'ls molcensus/data/pdfs; cat COURSE_PDFS_PERMISSION_GIVEN'   # check both arrived
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

What to watch first: the smoke test passing, then stage d starting its first course paper.

```bash
$SSH $HOST 'grep -E "biominer_smoke|biominer_course" runs/run_all.log; tail -n 1 runs/biominer_smoke.log'
$SSH $HOST 'grep "BioMiner: paper" runs/biominer_course.log | head -n 3'
```

You want `[biominer_smoke] done after N min`, a last smoke-log line `smoke test passed with the paid
key; MLLM calls lost after all retries: 0` (at most 2 is allowed), then `[biominer_course] started`
and `BioMiner: paper 1` in stage d's log. If the smoke test fails (for example because more than 2
calls were lost, as with a refused key), stages d and c are skipped; BioMiner's own output is in
`runs/biominer_smoke_output.log`.

| Order | Stage | What | Needs | Estimate | Time limit |
|---|---|---|---|---|---|
| 1 | a. course_readers | MolScribe, MolSight, MolGlyph on the 237 course crops | crops copied up | about 10 min | none |
| 2 | biominer_smoke | start the BioMiner servers, then BioMiner on `example/pdfs/68_6r8r.pdf`; fails if more than 2 MLLM calls are lost | `course_tier` key (paid) | 15 to 45 min | 45 min |
| 3 | d. biominer_course | BioMiner on the six course PDFs | smoke test passed, `course_tier` key (paid), PDFs, `~/COURSE_PDFS_PERMISSION_GIVEN` | 20 to 60 min | 2 h |
| | | servers stopped, GPU memory checked free | | | |
| 4 | b. public_readers | the three readers on all of USPTO, UOB, CLEF, ACS and BioVista OCSR (21,643 images), scored on the instance | downloads itself | 1.5 to 3.5 h | none |
| 5 | c. biominer_biovista | BioMiner on every BioVista PDF | smoke test passed, free key | 3 to 10 min per paper | 3 h |
| | | servers stopped | | | |
| 6 | e. molrecbench_wild | the three readers on MolRecBench-Wild | `data/public/molrecbench_wild/molrecbench_wild.csv` | 20 to 50 min | none |

The estimates are extrapolated from Mac CPU timings, not measured on a GPU. A stage that reaches its
time limit stops cleanly: the papers it finished keep their output and `run_info.json`, no done-marker
is written, and rerunning `run_all.sh` continues with the unfinished papers. The instance stops itself
after the last stage, or after 10 hours at the latest (`sudo shutdown -c` cancels that); if every stage
runs to its limit or upper estimate, stage e may not fit.
Stage c copies `data/biovista/labels/` to `BioVista/bioactivity_extraction/labels/` and
`data/biovista/structure_annotation/` to `BioVista/structure_annotation/` inside the BioMiner clone,
then adds `--biovista_evaluate`. If either folder is missing it runs without that flag. Where the two
label sets come from is in docs/methods.md (BioVista paragraph).

## Check progress

```bash
$SSH $HOST 'tail -n 15 runs/run_all.log; ls runs/*.done 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'
$SSH $HOST 'tail -n 5 runs/public_readers.log'                   # one stage's log
$SSH $HOST 'cat molcensus/outputs/work*/biominer_output/*/run_info.json'   # BioMiner: tier, model, errors, lost calls, tokens
$SSH -t $HOST 'tmux attach -t run'                                # watch live; detach with Ctrl-b d
```

To redo a stage, delete its `runs/<stage>.done` and start `run_all.sh` again; done stages are skipped.
BioMiner papers that already have a `run_info.json` are reused, not rerun; delete a paper's folder to
rerun it.

For cost, each paper's `run_info.json` has `input_tokens` and `output_tokens` (all answered MLLM calls
added up), plus `total_tokens`. If `total_tokens` is larger than the two together, the rest is most
likely the model's thinking tokens; check Google's pricing for how they are billed. The values are
`null` when no call was answered.

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
aws ec2 wait instance-terminated --region $REGION --instance-ids $ID
aws ec2 describe-volumes --region $REGION --query 'Volumes[].[VolumeId,State,Size,CreateTime]' --output table   # should list none
```
