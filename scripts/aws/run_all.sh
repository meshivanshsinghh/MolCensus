#!/usr/bin/env bash
sudo shutdown -h +600  # hard stop after 10 hours if the run hangs; the end of the script replaces it (cancel with: sudo shutdown -c)
# Every GPU stage, in order. Start it inside tmux after setup.sh (see scripts/aws/README.md):
#   bash ~/molcensus/scripts/aws/run_all.sh 2>&1 | tee -a ~/runs/run_all.log
#
# Order: a. crop readers on the course crops; BioMiner servers and smoke test; d. BioMiner on the
# course papers; servers stopped; b. crop readers on the public sets; c. BioMiner on BioVista papers;
# servers stopped; e. MolRecBench-Wild. Every BioMiner stage calls Gemini the same way, through Google
# Cloud Vertex AI, signed in by AWS workload identity federation (no key on the instance; see
# molcensus_vertex.py), so the smoke test checks the route stages d and c use.
#
# Each stage writes ~/runs/<stage>.log and, if it succeeds, ~/runs/<stage>.done. A done stage is
# skipped on a rerun (delete its .done file to redo it). A stage whose data is missing is skipped. A
# failed stage does not stop the next one. Time limits: smoke test 45 min, stage d 2 h, stage c 3 h.
# At its limit a stage stops, keeps the papers it finished and writes no done-marker, so a rerun
# continues with the unfinished papers. The time estimates assume one A10G or L4 GPU; they are
# extrapolated from the Mac CPU timings, not measured on a GPU.
set -u
# The Deep Learning AMI puts its own cuDNN (/usr/local/cuda-13.2/lib) on LD_LIBRARY_PATH
# (/etc/profile.d/dlami.sh). The pip torch in .venv and molsight-env then loads part of cuDNN from
# there and part from its own copy, and every convolution fails with
# CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH: on 2026-11-03 MolScribe and MolSight scored 0 that way.
# So the stages run without it. Only the BioMiner servers get it back: MinerU's paddle has only
# been run with it.
AMI_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
unset LD_LIBRARY_PATH
MOLCENSUS=$(cd "$(dirname "$0")/../.." && pwd)
BIOMINER=$HOME/BioMiner
BIOMINER_PY=$HOME/miniconda3/envs/BioMiner/bin/python
PY=$MOLCENSUS/.venv/bin/python
RUNS=$HOME/runs
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)  # this run's folder in the S3 bucket (upload_results.sh)
PERMISSION_FLAG=$HOME/COURSE_PDFS_PERMISSION_GIVEN  # who approved sending the course PDFs to Gemini; copied up from the Mac
READERS=molscribe,molsight,molglyph
SERVERS='scripts/(mineru|moldet|ocsr)_server.py'
SKIP=3     # exit code a stage returns when its data is missing
LIMIT=124  # exit code a stage returns when it reached its time limit (as GNU timeout does)
mkdir -p "$RUNS"
cd "$MOLCENSUS" || exit 1
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_DOWNLOAD_TIMEOUT=180  # stage b downloads the public sets from Hugging Face
VERTEX_ENV=$HOME/vertex.env  # Google project, service account and federation config (no secret), copied up from the Mac
VERTEX_CONFIG=BioMiner/config/molcensus_vertex.yaml  # written by biominer_gemini.py in setup.sh
if [ -f "$VERTEX_ENV" ]; then set -a; . "$VERTEX_ENV"; set +a; fi  # exported: BioMiner's Gemini client signs in with it

stage() {  # stage <name> <estimate> <function>
  local name=$1 estimate=$2 fn=$3 started=$SECONDS code
  if [ -e "$RUNS/$name.done" ]; then echo "[$name] done earlier, skipped"; return; fi
  echo "[$name] started $(date +%T), estimate $estimate, log $RUNS/$name.log"
  ( "$fn" ) > "$RUNS/$name.log" 2>&1  # subshell: a stage's cd and exports stay inside it
  code=$?
  case $code in
    0) touch "$RUNS/$name.done"; echo "[$name] done after $(( (SECONDS - started) / 60 )) min" ;;
    "$SKIP") echo "[$name] skipped: $(tail -1 "$RUNS/$name.log")" ;;
    "$LIMIT") echo "[$name] stopped at its time limit after $(( (SECONDS - started) / 60 )) min; finished papers" \
                   "are kept and no done-marker was written, so a rerun of run_all.sh continues it" ;;
    *) echo "[$name] FAILED (exit $code) after $(( (SECONDS - started) / 60 )) min; see $RUNS/$name.log" ;;
  esac
}

limited() {  # limited <duration> <command...>
  # GNU timeout sends SIGTERM to the command's whole process group (run_pipeline.py and the BioMiner
  # process under it), then SIGKILL 60 s later. run_pipeline.py saves the papers it finished on SIGTERM.
  timeout --kill-after=60 "$@"
  local code=$?
  if [ "$code" = 124 ] || [ "$code" = 137 ]; then return "$LIMIT"; fi
  return "$code"
}

has_smiles() {  # has_smiles <predictions.csv>: at least one SMILES. A reader that fails on every crop still writes a row per crop.
  "$PY" -c "import sys, pandas as pd; sys.exit(0 if pd.read_csv(sys.argv[1])['pred_smiles'].notna().any() else 1)" "$1" 2> /dev/null
}

server_up() {  # 200 from /health, or 404 if this litserve version has no /health route but is listening
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' "localhost:$1/health")
  [ "$code" = 200 ] || [ "$code" = 404 ]
}

start_servers() {  # BioMiner's MinerU (8002), MolDetV2 (8001) and MolGlyph OCSR (8003) servers
  local entry name port
  for entry in mineru:8002 moldet:8001 ocsr:8003; do
    name=${entry%:*}; port=${entry#*:}
    # setsid: each server leads its own process group, so a stage's time limit never kills it by
    # accident and stop_servers can stop it together with its worker processes.
    server_up "$port" || (cd "$BIOMINER" && PYTHONPATH=./ LD_LIBRARY_PATH=$AMI_LD_LIBRARY_PATH setsid nohup "$BIOMINER_PY" "scripts/${name}_server.py" \
      > "$RUNS/server_$name.log" 2>&1 < /dev/null &)
  done
  for entry in mineru:8002 moldet:8001 ocsr:8003; do
    name=${entry%:*}; port=${entry#*:}
    for _ in $(seq 180); do server_up "$port" && break; sleep 5; done  # up to 15 minutes each
    server_up "$port" || { echo "$name server did not start; see $RUNS/server_$name.log"; return 1; }
    echo "$name server is up on port $port"
  done
}

stop_servers() {  # stop the servers and their workers, then make sure nothing is left on the GPU
  local pid apps
  for pid in $(pgrep -f "$SERVERS"); do kill -TERM -- "-$(ps -o pgid= -p "$pid" | tr -d ' ')" 2> /dev/null; done
  for _ in $(seq 30); do pgrep -f "$SERVERS" > /dev/null || break; sleep 2; done
  for pid in $(pgrep -f "$SERVERS"); do kill -KILL -- "-$(ps -o pgid= -p "$pid" | tr -d ' ')" 2> /dev/null; done
  sleep 5
  # Between stages nothing else should be on the GPU, so anything still there is a leftover.
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2> /dev/null | tr -d ' ')
  if [ -n "$apps" ]; then
    echo "GPU still held by: $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';'); killing"
    for pid in $apps; do kill -KILL "$pid" 2> /dev/null; done
    sleep 5
  fi
  echo "BioMiner servers stopped; GPU memory in use: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2> /dev/null)"
}

# a. Three crop readers on the 237 course crops copied from the Mac.
course_readers() {
  [ -f outputs/work/crops_index.csv ] && [ -d outputs/work/crops_moldet ] || { echo "no course crops in outputs/work"; return $SKIP; }
  # crops_index.csv holds the Mac's absolute paths; point them at this copy.
  "$PY" - <<'EOF' || return 1
import pathlib
import pandas as pd
index = pathlib.Path("outputs/work/crops_index.csv")
crops = pd.read_csv(index)
here = pathlib.Path("outputs/work/crops_moldet").resolve()
crops["image_path"] = [str(here / path.split("/crops_moldet/", 1)[1]) for path in crops["image_path"]]
missing = sum(not pathlib.Path(p).exists() for p in crops["image_path"])
crops.to_csv(index, index=False)
print(f"{len(crops)} crops, {missing} missing after rewriting paths")
raise SystemExit(1 if missing else 0)
EOF
  "$PY" run_pipeline.py --config configs/aws.yaml --steps recognize --tools "$READERS"
  for tool in ${READERS//,/ }; do has_smiles "results/aws/predictions/${tool}_predictions.csv" || { echo "no $tool SMILES"; return 1; }; done
}

vertex_ready() {  # the BioMiner config for Vertex AI and the sign-in it needs
  [ -f "$BIOMINER/$VERTEX_CONFIG" ] || { echo "no $VERTEX_CONFIG: run setup.sh (biominer_gemini.py writes it)"; return 1; }
  [ -n "${VERTEX_WIF_CONFIG:-}" ] || { echo "no VERTEX_WIF_CONFIG: copy ~/.molcensus/vertex.env up to $VERTEX_ENV"; return 1; }
}

smoke_check() {  # smoke_check <BioMiner output folder> <BioMiner output log>
  # Passes only with a structure table and at most 2 MLLM calls lost after all retries. A refused
  # sign-in or a spent quota loses every call, and stage d's results would then be kept by the rerun rule.
  local lost
  find "$1" -name '*_structure_merged_output_*.csv' | grep -q . || { echo "smoke test wrote no structure table"; return 1; }
  lost=$(grep -c ' try failed:' "$2")
  [ "$lost" -le 2 ] || { echo "smoke test FAILED: $lost MLLM calls lost after all retries (more than 2); check the Vertex AI sign-in and quota"; return 1; }
  echo "smoke test passed (Gemini on Vertex AI); MLLM calls lost after all retries: $lost"
}

# BioMiner servers, then a smoke test on BioMiner's own example, limit 45 minutes.
biominer_smoke() {
  vertex_ready || return $SKIP
  start_servers || return 1
  # Start empty: BioMiner reuses MLLM answers it finds in its output folder, so a rerun would otherwise
  # reuse the empty answers of a refused key without calling Gemini again.
  rm -rf "$RUNS/biominer_smoke"
  limited 45m bash -c "cd '$BIOMINER' && PYTHONPATH=./ '$BIOMINER_PY' example_open_source.py \
    --config_path=$VERTEX_CONFIG --pdf=example/pdfs/68_6r8r.pdf --output_dir='$RUNS/biominer_smoke'" \
    > "$RUNS/biominer_smoke_output.log" 2>&1 \
    || { local code=$?; echo "smoke test did not finish (exit $code); see $RUNS/biominer_smoke_output.log"; return $code; }
  smoke_check "$RUNS/biominer_smoke" "$RUNS/biominer_smoke_output.log"
}

# d. BioMiner on the six course PDFs (gemini-3.5-flash on Vertex AI), only with the permission file, limit 2 hours.
biominer_course() {
  [ -e "$RUNS/biominer_smoke.done" ] || { echo "the BioMiner smoke test has not passed"; return $SKIP; }
  [ -f "$PERMISSION_FLAG" ] || { echo "no permission flag $PERMISSION_FLAG"; return $SKIP; }
  ls data/pdfs/*.pdf > /dev/null 2>&1 || { echo "no course PDFs in data/pdfs"; return $SKIP; }
  vertex_ready || return $SKIP
  echo "Permission record ($PERMISSION_FLAG):"; cat "$PERMISSION_FLAG"; echo
  start_servers || return 1
  limited 2h "$PY" run_pipeline.py --config configs/aws.yaml --steps recognize --tools biominer || return $?
  [ -s results/aws/predictions/biominer_predictions.csv ] || { echo "no course BioMiner predictions"; return 1; }
}

# b. Three crop readers on the full public sets, scored here (public data, safe to commit later).
public_readers() {
  "$PY" run_pipeline.py --config configs/aws.yaml --steps public --tools "$READERS"
  [ -s results/aws/public/public_metrics.csv ] || { echo "no public_metrics.csv"; return 1; }
  for tool in ${READERS//,/ }; do has_smiles "results/aws/public/${tool}_predictions.csv" || { echo "no $tool SMILES"; return 1; }; done
}

# c. BioMiner on BioVista papers, limit 3 hours.
biominer_biovista() {
  [ -e "$RUNS/biominer_smoke.done" ] || { echo "the BioMiner smoke test has not passed"; return $SKIP; }
  vertex_ready || return $SKIP
  mkdir -p data/biovista/pdfs
  cp -n "$BIOMINER"/example/pdfs/*.pdf data/biovista/pdfs/  # 40_6s8a and 68_6r8r, shipped with BioMiner
  local papers
  papers=$(ls data/biovista/pdfs/*.pdf | xargs -n1 basename | cut -d_ -f1 | sort -n | tr '\n' ' ')
  # --biovista_evaluate needs BioVista's labels inside the BioMiner clone, where its config looks for them:
  # the bioactivity/structure labels (data/biovista/labels) and the PDBbind index plus ligand files
  # (data/biovista/structure_annotation), both copied up from the Mac. Without them it would fail every paper.
  if [ -d data/biovista/labels ] && [ -d data/biovista/structure_annotation/labels ]; then
    mkdir -p "$BIOMINER/BioVista/bioactivity_extraction/labels" "$BIOMINER/BioVista/structure_annotation"
    cp -a data/biovista/labels/. "$BIOMINER/BioVista/bioactivity_extraction/labels/"
    cp -a data/biovista/structure_annotation/. "$BIOMINER/BioVista/structure_annotation/"
  fi
  if [ -d "$BIOMINER/BioVista/bioactivity_extraction/labels" ] && [ -d "$BIOMINER/BioVista/structure_annotation/labels" ] \
     && [ -d "$BIOMINER/BioVista/structure_annotation/data" ]; then
    echo "BioVista labels in place; running with --biovista_evaluate"
    export BIOMINER_EXTRA_ARGS=--biovista_evaluate
  else
    echo "BioVista labels not in $BIOMINER/BioVista; running without --biovista_evaluate"
  fi
  start_servers || return 1
  limited 3h "$PY" run_pipeline.py --config configs/aws_biovista.yaml --steps recognize --tools biominer --papers $papers || return $?
  [ -s results/aws_biovista/predictions/biominer_predictions.csv ] || { echo "no BioVista predictions"; return 1; }
}

# e. MolRecBench-Wild, once its converter has written the public-check CSV.
molrecbench_wild() {
  [ -f data/public/molrecbench_wild/molrecbench_wild.csv ] || { echo "no data/public/molrecbench_wild/molrecbench_wild.csv"; return $SKIP; }
  "$PY" run_pipeline.py --config configs/aws_molrecbench.yaml --steps public --tools "$READERS"
  [ -s results/aws_molrecbench/public/public_metrics.csv ] || { echo "no public_metrics.csv"; return 1; }
  for tool in ${READERS//,/ }; do has_smiles "results/aws_molrecbench/public/${tool}_predictions.csv" || { echo "no $tool SMILES"; return 1; }; done
}

n_biovista=$(( $(ls data/biovista/pdfs/*.pdf 2>/dev/null | grep -v -e '/40_6s8a.pdf' -e '/68_6r8r.pdf' | wc -l) + 2 ))
echo "run_all.sh started $(date). After the last stage the results go to S3 (folder $RUN_STAMP) and the instance"
echo "stops; if the upload fails it stops 8 hours later, and if run_all.sh hangs, 10 hours from now"
echo "Plan (estimates, not measured on this GPU; limits in brackets):"
echo "  a. course_readers     237 crops x 3 readers                                         about 10 min"
echo "     biominer_smoke     servers, then BioMiner on example/pdfs/68_6r8r.pdf (Vertex AI)    15 to 45 min [45 min]"
echo "  d. biominer_course    6 course papers (Vertex AI), only with $PERMISSION_FLAG            20 to 60 min [2 h]"
echo "     (servers stopped)"
echo "  b. public_readers     21,643 images (USPTO, UOB, CLEF, ACS, BioVista) x 3 readers    1.5 to 3.5 h"
echo "  c. biominer_biovista  $n_biovista BioVista papers (Vertex AI) at 3 to 10 min each          $(( 3 * n_biovista )) to $(( 10 * n_biovista )) min [3 h]"
echo "     (servers stopped)"
echo "  e. molrecbench_wild   5,029 images x 3 readers, if converted                        20 to 50 min"

stage course_readers "about 10 min" course_readers
stage biominer_smoke "15 to 45 min, limit 45 min" biominer_smoke
stage biominer_course "20 to 60 min, limit 2 h" biominer_course
stop_servers  # stage b runs the crop readers on the GPU
stage public_readers "1.5 to 3.5 h" public_readers
stage biominer_biovista "$(( 3 * n_biovista )) to $(( 10 * n_biovista )) min, limit 3 h" biominer_biovista
stop_servers  # stage e runs the crop readers on the GPU
stage molrecbench_wild "20 to 50 min" molrecbench_wild

# Upload the results to S3 and stop at once if every file arrived. If not, stay up 8 more hours so
# the results can be copied back by hand: GPUs are scarce, and a stopped instance may not start again.
# The 8-hour stop is set first (it replaces the 10-hour one), so it also holds if the upload hangs.
sudo shutdown -c
sudo shutdown -h +480
STOP_AT=$(date -d '+480 min')
echo "run_all.sh finished $(date). Outputs: results/aws*, outputs/work*/biominer_output. Uploading to S3."
if bash "$MOLCENSUS/scripts/aws/upload_results.sh" "$RUN_STAMP"; then
  echo "$(tail -n 1 "$RUNS/upload.log"). Stopping the instance now."
  sudo shutdown -h now
else
  echo "$(tail -n 1 "$RUNS/upload.log") (more in $RUNS/upload.log)"
  echo "The instance stops at $STOP_AT (8 hours after the last stage; cancel with: sudo shutdown -c)."
fi
