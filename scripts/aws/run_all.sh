#!/usr/bin/env bash
sudo shutdown -h +600  # hard stop after 10 hours whatever happens below (cancel with: sudo shutdown -c)
# Every GPU stage, in order. Start it inside tmux after setup.sh (see scripts/aws/README.md):
#   bash ~/molcensus/scripts/aws/run_all.sh 2>&1 | tee -a ~/runs/run_all.log
#
# Each stage writes ~/runs/<stage>.log and, if it succeeds, ~/runs/<stage>.done. A done stage is
# skipped on a rerun (delete its .done file to redo it). A stage whose data is missing is skipped. A
# failed stage does not stop the next one. The time estimates assume one A10G or L4 GPU; they are
# extrapolated from the Mac CPU timings, not measured on a GPU.
set -u
MOLCENSUS=$(cd "$(dirname "$0")/../.." && pwd)
BIOMINER=$HOME/BioMiner
BIOMINER_PY=$HOME/miniconda3/envs/BioMiner/bin/python
PY=$MOLCENSUS/.venv/bin/python
RUNS=$HOME/runs
PERMISSION_FLAG=$HOME/COURSE_PDFS_PERMISSION_GIVEN  # who approved sending the course PDFs to Gemini; copied up from the Mac
READERS=molscribe,molsight,molglyph
SKIP=3  # exit code a stage returns when its data is missing
mkdir -p "$RUNS"
cd "$MOLCENSUS" || exit 1
export NO_ALBUMENTATIONS_UPDATE=1

stage() {  # stage <name> <estimate> <function>
  local name=$1 estimate=$2 fn=$3 started=$SECONDS code
  if [ -e "$RUNS/$name.done" ]; then echo "[$name] done earlier, skipped"; return; fi
  echo "[$name] started $(date +%T), estimate $estimate, log $RUNS/$name.log"
  ( "$fn" ) > "$RUNS/$name.log" 2>&1  # subshell: a stage's cd and exports stay inside it
  code=$?
  case $code in
    0) touch "$RUNS/$name.done"; echo "[$name] done after $(( (SECONDS - started) / 60 )) min" ;;
    "$SKIP") echo "[$name] skipped: $(tail -1 "$RUNS/$name.log")" ;;
    *) echo "[$name] FAILED (exit $code) after $(( (SECONDS - started) / 60 )) min; see $RUNS/$name.log" ;;
  esac
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
    server_up "$port" || (cd "$BIOMINER" && PYTHONPATH=./ nohup "$BIOMINER_PY" "scripts/${name}_server.py" > "$RUNS/server_$name.log" 2>&1 &)
  done
  for entry in mineru:8002 moldet:8001 ocsr:8003; do
    name=${entry%:*}; port=${entry#*:}
    for _ in $(seq 180); do server_up "$port" && break; sleep 5; done  # up to 15 minutes each
    server_up "$port" || { echo "$name server did not start; see $RUNS/server_$name.log"; return 1; }
    echo "$name server is up on port $port"
  done
}

stop_servers() { pkill -f 'scripts/(mineru|moldet|ocsr)_server.py'; sleep 5; }

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
  for tool in ${READERS//,/ }; do [ -s "results/aws/predictions/${tool}_predictions.csv" ] || { echo "no $tool predictions"; return 1; }; done
}

# b. Three crop readers on the full public sets, scored here (public data, safe to commit later).
public_readers() {
  "$PY" run_pipeline.py --config configs/aws.yaml --steps public --tools "$READERS"
  [ -s results/aws/public/public_metrics.csv ] || { echo "no public_metrics.csv"; return 1; }
}

# c. BioMiner smoke test, then BioMiner on BioVista papers (free Gemini key).
biominer_biovista() {
  [ -f "$BIOMINER/BioMiner/config/molcensus_free.yaml" ] || { echo "no molcensus_free.yaml: no free Gemini key"; return $SKIP; }
  start_servers || return 1
  (cd "$BIOMINER" && PYTHONPATH=./ "$BIOMINER_PY" example_open_source.py --config_path=BioMiner/config/molcensus_free.yaml \
    --pdf=example/pdfs/68_6r8r.pdf --output_dir="$RUNS/biominer_smoke") || { echo "smoke test failed"; return 1; }
  find "$RUNS/biominer_smoke" -name '*_structure_merged_output_*.csv' | grep -q . || { echo "smoke test wrote no structure table"; return 1; }
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
  "$PY" run_pipeline.py --config configs/aws_biovista.yaml --steps recognize --tools biominer --papers $papers
  [ -s results/aws_biovista/predictions/biominer_predictions.csv ] || { echo "no BioVista predictions"; return 1; }
}

# d. BioMiner on the six course PDFs (free Gemini key, gemini-3.5-flash), only with the permission flag file.
biominer_course() {
  [ -f "$PERMISSION_FLAG" ] || { echo "no permission flag $PERMISSION_FLAG"; return $SKIP; }
  ls data/pdfs/*.pdf > /dev/null 2>&1 || { echo "no course PDFs in data/pdfs"; return $SKIP; }
  [ -f "$BIOMINER/BioMiner/config/molcensus_free.yaml" ] || { echo "no molcensus_free.yaml: no free Gemini key"; return $SKIP; }
  echo "Permission record ($PERMISSION_FLAG):"; cat "$PERMISSION_FLAG"; echo
  start_servers || return 1
  "$PY" run_pipeline.py --config configs/aws.yaml --steps recognize --tools biominer
  [ -s results/aws/predictions/biominer_predictions.csv ] || { echo "no course BioMiner predictions"; return 1; }
}

# e. MolRecBench-Wild, once its converter has written the public-check CSV.
molrecbench_wild() {
  [ -f data/public/molrecbench_wild/molrecbench_wild.csv ] || { echo "no data/public/molrecbench_wild/molrecbench_wild.csv"; return $SKIP; }
  "$PY" run_pipeline.py --config configs/aws_molrecbench.yaml --steps public --tools "$READERS"
  [ -s results/aws_molrecbench/public/public_metrics.csv ] || { echo "no public_metrics.csv"; return 1; }
}

n_biovista=$(( $(ls data/biovista/pdfs/*.pdf 2>/dev/null | grep -v -e '/40_6s8a.pdf' -e '/68_6r8r.pdf' | wc -l) + 2 ))
echo "run_all.sh started $(date); instance stops at the latest 10 hours from now"
echo "Plan (estimates, not measured on this GPU):"
echo "  a. course_readers     237 crops x 3 readers                          about 10 min"
echo "  b. public_readers     21,643 images (USPTO, UOB, CLEF, ACS, BioVista) x 3 readers, scored   1.5 to 3.5 h"
echo "  c. biominer_biovista  servers, smoke test, $n_biovista BioVista papers at 3 to 10 min each   $(( 15 + 3 * n_biovista )) to $(( 25 + 10 * n_biovista )) min"
echo "  d. biominer_course    6 papers at 3 to 10 min each, only with $PERMISSION_FLAG   20 to 60 min"
echo "  e. molrecbench_wild   5,029 images x 3 readers, if converted             20 to 50 min"

stage course_readers "about 10 min" course_readers
stage public_readers "1.5 to 3.5 h" public_readers
stage biominer_biovista "$(( 15 + 3 * n_biovista )) to $(( 25 + 10 * n_biovista )) min" biominer_biovista
stage biominer_course "20 to 60 min" biominer_course
stop_servers  # free the GPU for the crop readers in stage e
stage molrecbench_wild "20 to 50 min" molrecbench_wild

echo "run_all.sh finished $(date). Outputs: results/aws*, outputs/work*/biominer_output. Stopping the instance."
sudo shutdown -h now
