#!/usr/bin/env bash
# Single-node multi-GPU launcher. Auto-detects GPU count.
# Usage: bash scripts/launch_ddp.sh configs/glaucoma.yaml
set -euo pipefail

CFG="${1:-configs/glaucoma.yaml}"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)

echo "Launching DDP with $NGPU GPUs on config $CFG"

torchrun --standalone --nproc_per_node="$NGPU" -m pipeline.train --cfg "$CFG"
