"""Distributed (DDP) plumbing. The contract:

- `torchrun` sets RANK / WORLD_SIZE / LOCAL_RANK -> we init NCCL process group.
- Plain `python -m pipeline.train` -> world_size=1, everything degrades to single-GPU/CPU.
- Rank 0 is the ONLY rank that logs, saves, prints, or touches the hub.

Same train.py runs on: laptop CPU, Colab T4, 8xA100 node, multi-node SLURM cluster.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed() -> tuple[int, int, int]:
    if not is_distributed():
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=30),
    )
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int = None) -> bool:
    if rank is not None:
        return rank == 0
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


@contextmanager
def main_process_first():
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        dist.barrier()
    yield
    if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
        dist.barrier()
