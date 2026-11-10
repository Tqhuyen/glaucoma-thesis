#!/usr/bin/env bash
# Post-suite chain: Urban100 denoising notebook, then the MP-HSIR hyperspectral pilot
# (deep-learning-laboratory/hsi_mphsir/run_vast.sh) until DEADLINE_EPOCH minus a margin for uploads and self-destroy.
# Urban100 outputs are uploaded here because the controller only uploads POST_OUT after the whole chain.
#
# Env: HF_TOKEN, RESULTS_REPO, DEADLINE_EPOCH (set by the controller); POST_OUT (Urban100 dir);
#      HSI_MARGIN_MIN (default 25).
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LAB="${LAB:-/workspace/deep-learning-laboratory}"
export POST_OUT="${POST_OUT:-/workspace/dl_lab_run}"
: "${DEADLINE_EPOCH:?missing DEADLINE_EPOCH}"

bash "$HERE/vast_dl_lab_denoise.sh" || echo "[chain] urban100 exited non-zero"
python - <<'PY' || echo "[chain] urban100 upload failed"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo, out = os.environ["RESULTS_REPO"], os.environ["POST_OUT"]
api.upload_folder(repo_id=repo, repo_type="model", folder_path=out, path_in_repo="post/urban100-s25-seed2",
                  allow_patterns=["outputs/**", "task.ipynb", "executed.ipynb"], commit_message="urban100 outputs")
print(f"[chain] uploaded urban100 -> hf://{repo}/post/urban100-s25-seed2", flush=True)
PY

git -C "$LAB" pull --ff-only || echo "[chain] lab pull failed; using existing checkout"
export HSI_DEADLINE=$(( DEADLINE_EPOCH - ${HSI_MARGIN_MIN:-25} * 60 ))
echo "[chain] hsi pilot until $(date -u -d @"$HSI_DEADLINE" +%FT%TZ)"
bash "$LAB/hsi_mphsir/run_vast.sh" || echo "[chain] hsi pilot exited non-zero"
