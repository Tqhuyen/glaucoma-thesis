"""Prefetch the dataset to local cache at max bandwidth, BEFORE renting/starting the GPU.

Usage:
    export HF_TOKEN=hf_xxx
    python -m pipeline.download --cfg configs/base.yaml
"""
from __future__ import annotations

import argparse
import time

from datasets import load_dataset

from .utils import load_config, optimal_num_proc, setup_hf_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    args = p.parse_args()

    cfg = load_config(args.cfg)
    token = setup_hf_env(cfg.get("hf", {}))
    num_proc = optimal_num_proc(cfg["data"].get("num_proc", "auto"))

    t0 = time.time()
    print(f"[download] {cfg['data']['dataset']} with num_proc={num_proc}, hf_transfer=on")
    ds = load_dataset(cfg["data"]["dataset"], token=token, num_proc=num_proc)
    print(f"[download] done in {time.time() - t0:.1f}s")
    for split, d in ds.items():
        print(f"  {split}: {len(d):,} rows")
    print("[download] cached under ~/.cache/huggingface — training will reuse it instantly.")


if __name__ == "__main__":
    main()
