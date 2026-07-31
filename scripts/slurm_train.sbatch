#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=2                  # scale me
#SBATCH --gres=gpu:8               # GPUs per node
#SBATCH --ntasks-per-node=1        # ONE torchrun per node; it spawns 8 workers
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --signal=B:SIGTERM@120     # SIGTERM 120s before walltime -> we checkpoint
#SBATCH --output=logs/%x_%j.out

# Enterprise cluster entrypoint. Preemption-safe: the trainer catches SIGTERM,
# checkpoints, exits 0; requeue with the same command + --resume.

export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN in your environment or ~/.bashrc}
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500

srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --node_rank="$SLURM_NODEID" \
  --nproc_per_node=8 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  -m src.train --cfg configs/base.yaml --output-dir /shared/training_runs --resume
