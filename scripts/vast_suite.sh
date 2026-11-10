#!/usr/bin/env bash
# Sequential raw-200^3 suite for one rented GPU. Always destroys its own vast.ai
# instance on exit (success, failure, or time cap) so billing cannot run on idle.
#
# Env: HF_TOKEN WANDB_API_KEY (required); MAX_MINUTES (hard wall-clock cap, default 1800);
#      EPOCHS (default 8); SUITE ("name:seed ..."); CONTAINER_ID/CONTAINER_API_KEY (set by vast.ai).
set -uo pipefail

REPO_URL="${REPO_URL:-https://github.com/Tqhuyen/glaucoma-thesis.git}"
WORK="${WORK:-/workspace}"
MAX_MINUTES="${MAX_MINUTES:-1800}"
EPOCHS="${EPOCHS:-8}"
SUITE="${SUITE:-raw200-resnext:42,raw200-simple:42,raw200-pyramid:42,raw200-resnext:43}"
SUITE="${SUITE//,/ }"
export LOG="$WORK/suite.log"
mkdir -p "$WORK"
exec > >(tee -a "$LOG") 2>&1
START=$(date +%s)

finish() {
  echo "[suite] finishing at $(date -u +%FT%TZ), elapsed $(( ($(date +%s) - START) / 60 )) min"
  python - <<'PY' || true
import os, wandb
run = wandb.init(project="glaucoma-thesis", name="vast-suite-log", job_type="log")
art = wandb.Artifact("vast-suite-log", type="log")
art.add_file(os.environ["LOG"])
run.log_artifact(art)
run.finish()
PY
  if [ -n "${CONTAINER_ID:-}" ] && [ -n "${CONTAINER_API_KEY:-}" ]; then
    pip install -q vastai
    vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
  fi
}
trap finish EXIT

: "${HF_TOKEN:?missing HF_TOKEN}"
: "${WANDB_API_KEY:?missing WANDB_API_KEY}"

nvidia-smi || { echo "[suite] no GPU visible"; exit 1; }
pip install -q wandb
command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git; }
if [ ! -d "$WORK/repo/.git" ]; then git clone --depth 1 "$REPO_URL" "$WORK/repo"; else git -C "$WORK/repo" pull --ff-only; fi
cd "$WORK/repo"

for item in $SUITE; do
  name="${item%%:*}"
  seed="${item##*:}"
  run="${name}-s${seed}"
  remaining=$(( MAX_MINUTES * 60 - ($(date +%s) - START) ))
  if [ "$remaining" -lt 1800 ]; then
    echo "[suite] skip $run: only $((remaining / 60)) min left under cap"
    continue
  fi
  echo "[suite] start $run (cap ${remaining}s)"
  SEED="$seed" timeout "${remaining}s" bash scripts/vast_raw200.sh "$run" "$EPOCHS" 1 8 || echo "[suite] $run exited non-zero"
done
echo "[suite] done"
