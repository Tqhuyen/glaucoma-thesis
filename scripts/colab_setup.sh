#!/usr/bin/env bash
# Colab CLI launcher — one-shot setup & train.
# Usage: bash scripts/colab_setup.sh <config_name>
#   config_name = glaucoma | base  (looks up configs/<name>.yaml)
# Example: bash scripts/colab_setup.sh glaucoma
#
# Before running:
#   1. Runtime -> Change runtime type -> GPU (T4 minimum for bf16)
#   2. Set HF_TOKEN in Colab Secrets (key icon -> HF_TOKEN -> hf_xxx)
#      Or: export HF_TOKEN=hf_xxx before running this script
set -euo pipefail

CFG_NAME="${1:-glaucoma}"
CFG_FILE="configs/${CFG_NAME}.yaml"

if [ ! -f "$CFG_FILE" ]; then
  echo "ERROR: config file not found: $CFG_FILE"
  echo "Available: configs/*.yaml"
  exit 1
fi

echo "=== GPU Check ==="
nvidia-smi || { echo "No GPU — go to Runtime -> Change runtime type -> GPU"; exit 1; }

if [ -z "${HF_TOKEN:-}" ]; then
  echo "=== Attempting HF_TOKEN from Colab secrets ==="
  HF_TOKEN=$(python -c "
try:
    from google.colab import userdata
    print(userdata.get('HF_TOKEN'))
except Exception:
    pass
" 2>/dev/null || echo "")
  if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN
    echo "HF_TOKEN loaded from Colab secrets."
  else
    echo "WARNING: HF_TOKEN not found in env or Colab secrets."
    echo "  Colab:  key icon (left sidebar) -> Add secret -> name: HF_TOKEN, value: hf_xxx"
    echo "  Or: export HF_TOKEN=hf_xxx"
  fi
fi

echo "=== Installing dependencies ==="
pip install -q -e ".[glaucoma]"

echo "=== Smoke test (~2 min, safe to fail) ==="
python -m pipeline.train --cfg "$CFG_FILE" --smoke --output-dir /tmp/smoke 2>&1 || {
  echo "Smoke test had issues but continuing — check output above."
}

echo "=== Starting TensorBoard ==="
python -c "
try:
    import tensorboard
    %load_ext tensorboard
    %tensorboard --logdir outputs
except Exception as e:
    print(f'TensorBoard inline failed (may need notebook cell): {e}')
" 2>/dev/null || echo "Start TensorBoard in a notebook cell: %tensorboard --logdir outputs"

echo "=== Training: $CFG_FILE ==="
python -m pipeline.train --cfg "$CFG_FILE"

echo "=== Done ==="
echo "To resume after disconnect: bash scripts/colab_setup.sh $CFG_NAME --resume"
