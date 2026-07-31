#!/usr/bin/env bash
# Expose TensorBoard at a FREE public HTTPS URL — no SSH tunnel, no localhost.
#
#   bash scripts/tunnel_dashboard.sh [logdir]      # default: /workspace/training_runs
#
# Uses Cloudflare Quick Tunnel: no account, no signup. Prints a URL like
#   https://random-words.trycloudflare.com
# Open it from any device. URL is unguessable but PUBLIC — anyone with the
# link can view your metrics. Fine for loss curves; don't put secrets in run names.
# The URL changes each restart (quick tunnels are ephemeral by design).
set -euo pipefail

LOGDIR="${1:-/workspace/training_runs}"

# Install cloudflared if missing (~20 MB static binary)
if ! command -v cloudflared >/dev/null 2>&1; then
  echo "[tunnel] installing cloudflared..."
  ARCH=$(uname -m); case "$ARCH" in x86_64) ARCH=amd64;; aarch64) ARCH=arm64;; esac
  curl -sL -o /usr/local/bin/cloudflared \
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${ARCH}"
  chmod +x /usr/local/bin/cloudflared
fi

# TensorBoard in tmux (idempotent)
if ! tmux has-session -t tb 2>/dev/null; then
  tmux new-session -d -s tb "tensorboard --logdir $LOGDIR --port 6006 --bind_all"
  echo "[tunnel] TensorBoard started on :6006 (logdir=$LOGDIR)"
fi

# Tunnel in tmux; URL appears in the log
tmux kill-session -t tunnel 2>/dev/null || true
tmux new-session -d -s tunnel \
  "cloudflared tunnel --url http://localhost:6006 2>&1 | tee /tmp/tunnel.log"

echo "[tunnel] waiting for public URL..."
for _ in $(seq 1 30); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/tunnel.log | head -n1 || true)
  [ -n "${URL:-}" ] && break
  sleep 1
done

if [ -n "${URL:-}" ]; then
  echo ""
  echo "=================================================="
  echo "  Dashboard live at:  $URL"
  echo "  Open from any device. Ctrl-C safe (runs in tmux)."
  echo "=================================================="
else
  echo "URL not found yet — check: tail -f /tmp/tunnel.log"
fi
