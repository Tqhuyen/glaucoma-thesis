#!/usr/bin/env bash
# vast.ai launcher — model-agnostic.
# Usage: bash scripts/vast_setup.sh <config_name> [download|smoke|train|resume]
#   config_name = glaucoma | base  (looks up configs/<name>.yaml)
# Example:
#   bash scripts/vast_setup.sh glaucoma download
#   bash scripts/vast_setup.sh glaucoma train
set -euo pipefail

MODEL_CFG="${1:-base}"
CFG_FILE="configs/${MODEL_CFG}.yaml"
shift 2>/dev/null || true
MODE="${1:-train}"

if [ ! -f "$CFG_FILE" ]; then
  echo "ERROR: config file not found: $CFG_FILE"
  echo "Available: configs/*.yaml"
  exit 1
fi

PERSIST=/workspace/training_runs
mkdir -p "$PERSIST"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN not set. Gated/private datasets will fail."
  echo "         export HF_TOKEN=hf_xxx"
fi

echo "== GPU =="
nvidia-smi || { echo "No GPU visible — wrong instance type?"; exit 1; }

echo "== Installing deps =="
pip install -q -e ".[glaucoma]"

start_tensorboard() {
  if ! tmux has-session -t tb 2>/dev/null; then
    tmux new-session -d -s tb "tensorboard --logdir $PERSIST --port 6006 --bind_all"
    echo "TensorBoard live on port 6006 (tunnel it: ssh -L 6006:localhost:6006 ...)"
  fi
}

case "$MODE" in
  download)
    python -m pipeline.download --cfg "$CFG_FILE"
    ;;
  smoke)
    python -m pipeline.train --cfg "$CFG_FILE" --smoke --output-dir /tmp/smoke
    echo "Smoke test PASSED. Next: bash scripts/vast_setup.sh $MODEL_CFG train"
    ;;
  train)
    start_tensorboard
    tmux new-session -d -s train \
      "python -m pipeline.train --cfg $CFG_FILE --output-dir $PERSIST 2>&1 | tee $PERSIST/train.log"
    echo "Training in tmux 'train'.  Watch: tmux attach -t train  |  tail -f $PERSIST/train.log"
    ;;
  resume)
    start_tensorboard
    tmux new-session -d -s train \
      "python -m pipeline.train --cfg $CFG_FILE --output-dir $PERSIST --resume 2>&1 | tee -a $PERSIST/train.log"
    echo "Resumed in tmux 'train'."
    ;;
  *)
    echo "Usage: bash scripts/vast_setup.sh <config_name> [download|smoke|train|resume]"
    exit 1
    ;;
esac
