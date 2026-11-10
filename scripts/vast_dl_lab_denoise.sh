#!/usr/bin/env bash
# Post-suite task: deep-learning-laboratory Urban100 sigma-25 seed-2 denoising run
# (notebooks/urban100_s25_seed_2.ipynb, ADM Tweedie + R2R, 88 images). Headless via nbconvert.
# Outputs: $POST_OUT/outputs/{scores.csv,tests.csv,summary.json,DONE} and executed.ipynb.
set -euo pipefail

LAB="${LAB:-/workspace/deep-learning-laboratory}"
POST_OUT="${POST_OUT:-/workspace/dl_lab_run}"
NOTEBOOK="${NOTEBOOK:-notebooks/urban100_s25_seed_2.ipynb}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ ! -d "$LAB/.git" ]; then
  git clone --depth 1 https://github.com/Tqhuyen/deep-learning-laboratory.git "$LAB"
else
  git -C "$LAB" pull --ff-only
fi
python -m pip install -q numpy pandas pillow nbconvert ipykernel

mkdir -p "$POST_OUT"
cp "$LAB/$NOTEBOOK" "$POST_OUT/task.ipynb"
cd "$POST_OUT"
echo "[post] executing $NOTEBOOK in $POST_OUT at $(date -u +%FT%TZ)"
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  task.ipynb --output executed.ipynb
echo "[post] finished at $(date -u +%FT%TZ)"
ls -la outputs
