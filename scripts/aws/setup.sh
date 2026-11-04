#!/usr/bin/env bash
# One-time setup of an AWS GPU instance (Ubuntu 22.04 with the NVIDIA driver, for example the AWS
# Deep Learning Base GPU AMI) for MolCensus, BioMiner, MolSight and MolGlyph.
#
# Safe to rerun: every step checks whether it is already done. Everything goes to ~/setup.log.
#   bash ~/molcensus/scripts/aws/setup.sh
#
# Needs ~/molcensus_keys.env, copied up from the Mac (never committed):
#   HF_TOKEN=...              Hugging Face read token; accept MolGlyph's terms on its model page first
#   GEMINI_API_KEY_FREE=...   public papers (stage c and the smoke test)
#   GEMINI_API_KEY_PAID=...   the six course papers (stage d, only with run_all.sh's permission flag)
# The Gemini model (gemini-3.5-flash) is set in biominer_gemini.py, not here.
set -uo pipefail
exec > >(tee -a "$HOME/setup.log") 2>&1
echo "=== setup.sh started $(date -u +%FT%TZ)"

MOLCENSUS=$(cd "$(dirname "$0")/../.." && pwd)
BIOMINER=$HOME/BioMiner
KEYS=$HOME/molcensus_keys.env
CONDA=$HOME/miniconda3
BIOMINER_COMMIT=17c6161
MOLSIGHT_COMMIT=86e979b
PY=$MOLCENSUS/.venv/bin/python
BIOMINER_PY=$CONDA/envs/BioMiner/bin/python
export PATH=$HOME/.local/bin:$CONDA/bin:$PATH

step() { echo; echo "--- $(date +%T) $*"; }
fail() { echo "SETUP FAILED: $*"; exit 1; }
gpu_check() {  # gpu_check <python>: one convolution on the GPU, without the AMI's LD_LIBRARY_PATH, as run_all.sh runs it
  # torch.cuda.is_available() alone passed on 2026-11-03 while every convolution failed (see run_all.sh).
  env -u LD_LIBRARY_PATH "$1" -c "import torch; torch.nn.Conv2d(3, 8, 3).cuda()(torch.zeros(1, 3, 32, 32, device='cuda')); \
print('torch', torch.__version__, 'cuDNN', torch.backends.cudnn.version(), torch.cuda.get_device_name(0))"
}

[ -f "$KEYS" ] || fail "missing $KEYS (see scripts/aws/README.md)"
chmod 600 "$KEYS"
set -a; . "$KEYS"; set +a  # HF_TOKEN for the gated MolGlyph download
export HF_HUB_DOWNLOAD_TIMEOUT=180  # seconds per network read; huggingface_hub's default is 10

step "GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || fail "nvidia-smi failed: no GPU driver"

step "System packages"
# poppler-utils: pdfinfo and pdftoppm, which BioMiner's pdf2image calls.
command -v tmux >/dev/null && command -v git >/dev/null && command -v pdfinfo >/dev/null \
  || { sudo apt-get update -qq && sudo apt-get install -y -qq tmux git curl poppler-utils; } || fail "system packages"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

step "MolCensus environment (MolScribe, scoring)"
[ -x "$PY" ] || uv venv --python 3.10 "$MOLCENSUS/.venv" || fail "uv venv"
uv pip install -q --python "$PY" -r "$MOLCENSUS/requirements.txt" || fail "MolCensus requirements"
gpu_check "$PY" || fail "torch CUDA"

step "MolSight environment (commit $MOLSIGHT_COMMIT)"
[ -d "$MOLCENSUS/tools/MolSight/.git" ] || git clone -q https://github.com/hustvl/MolSight.git "$MOLCENSUS/tools/MolSight"
git -C "$MOLCENSUS/tools/MolSight" checkout -q "$MOLSIGHT_COMMIT"
[ -x "$MOLCENSUS/tools/molsight-env/bin/python" ] || uv venv --python 3.10 "$MOLCENSUS/tools/molsight-env"
# Its requirements are unpinned; these two pins are explained in docs/usage.md.
uv pip install -q --python "$MOLCENSUS/tools/molsight-env/bin/python" \
  -r "$MOLCENSUS/tools/MolSight/requirements.txt" "datasets>=3" "numpy<2" || fail "MolSight requirements"
gpu_check "$MOLCENSUS/tools/molsight-env/bin/python" || fail "torch CUDA (MolSight)"

step "Miniconda"
[ -x "$CONDA/bin/conda" ] || {
  curl -fsSLo /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh &&
  bash /tmp/miniconda.sh -b -p "$CONDA"; } || fail "Miniconda install"
# BioMiner's environment file uses the anaconda and defaults channels, which need Anaconda's terms accepted.
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes

step "BioMiner (commit $BIOMINER_COMMIT) and its conda environment (20 to 40 minutes the first time)"
[ -d "$BIOMINER/.git" ] || git clone -q https://github.com/jiaxianyan/BioMiner.git "$BIOMINER"
git -C "$BIOMINER" checkout -q -- . 2>/dev/null  # undo our own edits before checkout; they are reapplied below
git -C "$BIOMINER" checkout -q "$BIOMINER_COMMIT"
[ -x "$BIOMINER_PY" ] || conda env create -q -f "$BIOMINER/main_environment.yml" || fail "BioMiner conda environment"

step "MinerU models"
if [ ! -f "$HOME/.molcensus_mineru_models" ]; then
  [ -f "$HOME/magic-pdf.json" ] || cp "$BIOMINER/magic-pdf.json" "$HOME/magic-pdf.json"
  # download_models.py fetches from ModelScope and rewrites the model folders in ~/magic-pdf.json.
  if (cd "$BIOMINER" && "$BIOMINER_PY" download_models.py); then
    touch "$HOME/.molcensus_mineru_models"
  else
    echo "ModelScope download failed; using the Hugging Face copy jiaxianustc/BioMiner-MinerU-Model"
    "$PY" - <<'EOF' && touch "$HOME/.molcensus_mineru_models" || fail "MinerU models"
import json, pathlib
from huggingface_hub import snapshot_download
root = pathlib.Path(snapshot_download("jiaxianustc/BioMiner-MinerU-Model", local_dir=str(pathlib.Path.home() / "mineru_models")))
config_path = pathlib.Path.home() / "magic-pdf.json"
config = json.loads(config_path.read_text())
config["models-dir"] = str(root / "models") if (root / "models").is_dir() else str(root)
layout = [p for p in root.rglob("layoutreader") if p.is_dir()]
if layout:
    config["layoutreader-model-dir"] = str(layout[0])
config["device-mode"] = "cuda"
config_path.write_text(json.dumps(config, indent=4))
print("models-dir", config["models-dir"], "| layoutreader-model-dir", config.get("layoutreader-model-dir"))
EOF
  fi
fi
# MinerU still loads this OCR model, but the latest PDF-Extract-Kit pack dropped it; take it from an older commit.
"$PY" - <<'EOF' || fail "MinerU OCR model"
import json, pathlib, shutil
from huggingface_hub import hf_hub_download
name = "models/OCR/paddleocr_torch/ch_PP-OCRv3_det_infer.pth"
models_dir = json.loads((pathlib.Path.home() / "magic-pdf.json").read_text())["models-dir"]
target = pathlib.Path(models_dir) / name.removeprefix("models/")
if not target.exists():
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(hf_hub_download("opendatalab/PDF-Extract-Kit-1.0", name, revision="fe3643d1b873284fca2d90c3d731ffbc182b3566"), target)
print("OCR model", target)
EOF

step "Weights (MolDetV2 and MolGlyph for BioMiner; MolScribe, MolSight-base, MolGlyph for the crop readers)"
"$PY" - "$BIOMINER" "$MOLCENSUS/models" <<'EOF' || fail "weights (MolGlyph is gated: accept its terms and check HF_TOKEN)"
import os, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
biominer, models = Path(sys.argv[1]), Path(sys.argv[2])
hf_hub_download("UniParser/MolDetv2", "moldet_v2_yolo11n_960_doc.pt", local_dir=str(biominer / "scripts"))
for repo, name in [("yujieq/MolScribe", "swin_base_char_aux_1m680k.pth"),
                   ("Robert-zwr/MolSight", "pubchem_uspto_smiles_edges_10.pth"),
                   ("jiaxianustc/MolGlyph", "molglyph_large.pt")]:
    hf_hub_download(repo, name, local_dir=str(models))
link = biominer / "scripts" / "molglyph_large.pt"  # one copy on disk, used by BioMiner's OCSR server too
if not link.exists():
    link.symlink_to(models / "molglyph_large.pt")
print("weights ready")
EOF

step "Local BioMiner edits: one GPU, and retries for failed MLLM calls"
"$PY" "$MOLCENSUS/scripts/aws/patch_biominer.py" "$BIOMINER" || fail "BioMiner patch"

step "BioMiner configs for Gemini, and one test call per key"
"$BIOMINER_PY" "$MOLCENSUS/scripts/aws/biominer_gemini.py" --keys "$KEYS" --biominer "$BIOMINER" --check \
  || echo "WARNING: a Gemini check failed (see above). BioMiner stages will lose content until this is fixed."

step "MolCensus .env"
printf 'BIOMINER_DIR=%s\nBIOMINER_PYTHON=%s\n' "$BIOMINER" "$BIOMINER_PY" > "$MOLCENSUS/.env"
cat "$MOLCENSUS/.env"

step "MinerU smoke test on BioMiner's example PDF"
if [ ! -f "$HOME/.molcensus_mineru_smoke" ]; then
  rm -rf /tmp/mineru_smoke
  (cd "$BIOMINER" && "$CONDA/envs/BioMiner/bin/magic-pdf" -p example/pdfs/40_6s8a.pdf -o /tmp/mineru_smoke)
  # magic-pdf can exit 0 after a traceback (it did on 2026-11-03), so passing means it wrote the Markdown.
  if find /tmp/mineru_smoke -name '*.md' | grep -q .; then
    touch "$HOME/.molcensus_mineru_smoke"
  else
    echo "WARNING: MinerU smoke test wrote no .md file; BioMiner falls back to plain text without MinerU"
  fi
fi

step "Versions"
git -C "$MOLCENSUS" log -1 --format='MolCensus %h %s'
git -C "$BIOMINER" log -1 --format='BioMiner %h'
git -C "$MOLCENSUS/tools/MolSight" log -1 --format='MolSight %h'
"$PY" --version; "$BIOMINER_PY" --version
df -h "$HOME" | tail -1
echo "=== setup.sh finished $(date -u +%FT%TZ)"
