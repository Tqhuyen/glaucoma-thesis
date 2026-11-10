#!/usr/bin/env bash
# Sequential raw-200^3 suite for one rented GPU. Always destroys its own vast.ai
# instance on exit (success, failure, or time cap) so billing cannot run on idle.
#
# Env: HF_TOKEN WANDB_API_KEY (required); MAX_MINUTES (hard wall-clock cap, default 1800);
#      EPOCHS (default 8); SUITE ("name:seed,..."); RESULTS_REPO (private HF repo for run outputs);
#      WAIT_PID + WAIT_RUN (adopt an already-running training process before starting SUITE);
#      DEADLINE_EPOCH (absolute unix time cap, overrides MAX_MINUTES);
#      CONTAINER_ID/CONTAINER_API_KEY (set by vast.ai).
set -uo pipefail

REPO_URL="${REPO_URL:-https://github.com/Tqhuyen/glaucoma-thesis.git}"
WORK="${WORK:-/workspace}"
export OUT="${OUT:-$WORK/training_runs}"
MAX_MINUTES="${MAX_MINUTES:-1800}"
EPOCHS="${EPOCHS:-8}"
SUITE="${SUITE:-raw200-resnext:42,raw200-simple:42,raw200-pyramid:42,raw200-resnext:43}"
SUITE="${SUITE//,/ }"
export RESULTS_REPO="${RESULTS_REPO:-tqhuyen/glaucoma-raw200-runs}"
export LOG="$WORK/suite.log"
mkdir -p "$WORK"
exec > >(tee -a "$LOG") 2>&1
START=$(date +%s)
DEADLINE="${DEADLINE_EPOCH:-$((START + MAX_MINUTES * 60))}"

upload_run() {
  local run="$1"
  [ -d "$OUT/$run" ] || { echo "[suite] nothing to upload for $run"; return 0; }
  RUN="$run" python - <<'PY' || echo "[suite] upload failed for $run"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ["RESULTS_REPO"]
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
run = os.environ["RUN"]
api.upload_folder(repo_id=repo, repo_type="model", folder_path=os.path.join(os.environ["OUT"], run),
                  path_in_repo=f"runs/{run}", ignore_patterns=["checkpoints/*"],
                  commit_message=f"{run} outputs")
print(f"[suite] uploaded {run} -> hf://{repo}/runs/{run}", flush=True)
PY
}

upload_log() {
  python - <<'PY' || true
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ["RESULTS_REPO"]
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
api.upload_file(repo_id=repo, repo_type="model", path_or_fileobj=os.environ["LOG"],
                path_in_repo="suite.log", commit_message="suite log")
PY
}

finish() {
  echo "[suite] finishing at $(date -u +%FT%TZ), elapsed $(( ($(date +%s) - START) / 60 )) min"
  upload_log
  if [ -n "${CONTAINER_ID:-}" ] && [ -n "${CONTAINER_API_KEY:-}" ]; then
    pip install -q vastai
    if vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" 2>&1 | tee -a "$LOG" | grep -qiE "error|failed"; then
      echo "[suite] destroy refused; stopping instance instead"
      vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" 2>&1 | tee -a "$LOG" || true
    fi
  fi
  echo "[suite] stopping container so GPU billing ends even if the API calls failed"
  sync
  kill -TERM 1 2>/dev/null || true
}
trap finish EXIT

: "${HF_TOKEN:?missing HF_TOKEN}"
: "${WANDB_API_KEY:?missing WANDB_API_KEY}"

nvidia-smi || { echo "[suite] no GPU visible"; exit 1; }
pip install -q wandb huggingface_hub
if timeout 90 python -c "import wandb; wandb.Api(timeout=30).viewer.username" >/dev/null 2>&1; then
  export WANDB_ENABLED=true
else
  export WANDB_ENABLED=false
  echo "[suite] W&B unreachable from this host; outputs go to hf://$RESULTS_REPO and are replayed to W&B later"
fi
command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git; }
if [ ! -d "$WORK/repo/.git" ]; then git clone --depth 1 "$REPO_URL" "$WORK/repo"; else git -C "$WORK/repo" pull --ff-only; fi
cd "$WORK/repo"

if [ -n "${WAIT_PID:-}" ]; then
  echo "[suite] adopting running process $WAIT_PID (${WAIT_RUN:-?})"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
      echo "[suite] deadline hit while waiting"
      kill -TERM "$WAIT_PID"
      break
    fi
    sleep 30
  done
  if [ -n "${WAIT_RUN:-}" ]; then upload_run "$WAIT_RUN"; fi
fi

for item in $SUITE; do
  name="${item%%:*}"
  seed="${item##*:}"
  run="${name}-s${seed}"
  remaining=$(( DEADLINE - $(date +%s) ))
  if [ "$remaining" -lt 1800 ]; then
    echo "[suite] skip $run: only $((remaining / 60)) min left under cap"
    continue
  fi
  echo "[suite] start $run (cap ${remaining}s)"
  SEED="$seed" timeout "${remaining}s" bash scripts/vast_raw200.sh "$run" "$EPOCHS" 1 8 || echo "[suite] $run exited non-zero"
  upload_run "$run"
done
echo "[suite] training suite done"

if [ -n "${POST_TASK:-}" ]; then
  remaining=$(( DEADLINE - $(date +%s) ))
  export POST_OUT="${POST_OUT:-$WORK/post_task}"
  if [ "$remaining" -lt 1800 ]; then
    echo "[suite] skip post task $POST_TASK: only $((remaining / 60)) min left under cap"
  else
    echo "[suite] post task $POST_TASK (cap ${remaining}s, out $POST_OUT)"
    timeout "${remaining}s" bash "$POST_TASK" || echo "[suite] post task exited non-zero"
    POST_NAME="${POST_NAME:-post-task}" python - <<'PY' || echo "[suite] post upload failed"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo, out = os.environ["RESULTS_REPO"], os.environ["POST_OUT"]
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
api.upload_folder(repo_id=repo, repo_type="model", folder_path=out, path_in_repo=f"post/{os.environ['POST_NAME']}",
                  allow_patterns=["outputs/*", "executed.ipynb"], commit_message=f"{os.environ['POST_NAME']} outputs")
print(f"[suite] uploaded post task -> hf://{repo}/post/{os.environ['POST_NAME']}", flush=True)
PY
  fi
fi
echo "[suite] done"
