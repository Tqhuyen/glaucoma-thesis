#!/usr/bin/env bash
# Raw 200^3 experiments on a rented GPU (vast.ai / any CUDA box).
#
#   export HF_TOKEN=hf_xxx WANDB_API_KEY=wandb_xxx
#   bash scripts/vast_raw200.sh raw200-resnext 8 1 8
#   bash scripts/vast_raw200.sh raw200-pyramid 8 1 8
#
# Args: run_name epochs batch_size grad_accum [architecture]
# Data is downloaded once into $DATA and reused by later runs on the same box.
set -euo pipefail

RUN="${1:-raw200-resnext}"
EPOCHS="${2:-8}"
BS="${3:-1}"
ACCUM="${4:-8}"
ARCH="${5:-}"
DATA="${DATA:-/workspace/data/raw200}"
OUT="${OUT:-/workspace/training_runs}"
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"

case "$RUN" in
  raw200-resnext*) ARCH="${ARCH:-pyramid_resnext3d}" ;;
  raw200-pyramid*) ARCH="${ARCH:-pyramid3dcnn}" ;;
  raw200-thinhead*) ARCH="${ARCH:-pyramid3dcnn}" ;;
  raw200-simple*) ARCH="${ARCH:-simple3dcnn}" ;;
  *) ARCH="${ARCH:-pyramid_resnext3d}" ;;
esac
SEED="${SEED:-42}"

: "${HF_TOKEN:?export HF_TOKEN first}"
: "${WANDB_API_KEY:?export WANDB_API_KEY first}"

cd "$REPO"
nvidia-smi
python - <<'PY'
import importlib.util
need = [m for m in ("yaml", "sklearn", "tensorboard", "wandb", "huggingface_hub",
                    "dotenv", "tqdm", "pandas", "matplotlib", "pynvml", "psutil")
        if importlib.util.find_spec(m) is None]
print("[setup] missing:", need or "none")
PY
python -m pip install -q pyyaml scikit-learn tensorboard wandb huggingface_hub python-dotenv tqdm pandas matplotlib pynvml psutil

if [ ! -f "$DATA/Training_volumes.npy" ]; then
  echo "[data] downloading raw 200^3 to $DATA"
  mkdir -p "$DATA"
  python - "$DATA" <<'PY'
import os, sys
from huggingface_hub import hf_hub_download
root = sys.argv[1]
for split in ("Training", "Validation", "Test"):
    for name in (f"{split}_volumes.npy", f"{split}_labels.npy"):
        hf_hub_download("tqhuyen/harvard-oct-glaucoma-200", name, repo_type="dataset",
                        local_dir=root, token=os.environ["HF_TOKEN"])
        print("[data]", name, flush=True)
PY
fi

HP=gap
case "$RUN" in raw200-thinhead*) HP=depth_attention ;; esac
if [ "$ARCH" = "pyramid_resnext3d" ]; then FDIM=128; else FDIM=64; fi

mkdir -p "$OUT"
python -m pipeline.train --cfg configs/glaucoma.yaml --set \
  run_name="$RUN" seed="$SEED" model.architecture="$ARCH" data.data_dir="$DATA" \
  model.fusion_dim="$FDIM" model.head_pool="$HP" \
  logging.wandb="${WANDB_ENABLED:-true}" logging.tensorboard=true output_dir="$OUT" \
  train.lr=0.0001 train.epochs="$EPOCHS" train.batch_size="$BS" \
  train.grad_accum_steps="$ACCUM" train.eval_every_steps=100 \
  train.eval_test_every_steps=300 train.save_every_steps=300 \
  train.early_stop_patience=6
