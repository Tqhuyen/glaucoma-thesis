#!/usr/bin/env bash
# Post task: report assets (per-sample glaucoma predictions, case images, saliency; HSI restorations) into
# $POST_OUT/outputs before the controller uploads and destroys the box.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
export POST_OUT="${POST_OUT:-/workspace/report_assets}"
export HSI_CODE="${HSI_CODE:-/workspace/deep-learning-laboratory/hsi_mphsir}"
export HSI_DATA="${HSI_DATA:-/workspace/hsi_data}" HSI_RUN="${HSI_RUN:-/workspace/hsi_run}"
export MPHSIR_DIR="${MPHSIR_DIR:-/workspace/MP-HSIR}"
mkdir -p "$POST_OUT/outputs"
cd "$HERE/.."
python scripts/report_assets.py --runs /workspace/training_runs --data /workspace/data/raw200 \
  --out "$POST_OUT/outputs" 2>&1 | tee "$POST_OUT/outputs/assets.log"
