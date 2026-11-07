"""Step-time / VRAM comparison: Simple3DCNN vs Pyramid3DCNN on random volumes.

Synthetic tensors only (no dataset), same hidden widths and pooling as
``configs/glaucoma.yaml``. Run the same command on a Colab T4 / local GPU and
compare; CPU numbers are only relative compute, not training speed.

    python scripts/bench_pyramid.py --res 96 --batch-sizes 1,2 --steps 3
"""

import argparse
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.glaucoma.model import build_model  # noqa: E402
from pipeline.config import validate_config  # noqa: E402
from pipeline.utils import load_config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bench(architecture, cfg, bs, res, steps, amp_dtype, warmup=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {**cfg, "model": {**cfg["model"], "architecture": architecture}}
    validate_config(cfg)
    torch.cuda.empty_cache()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model = build_model(cfg).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(bs, 1, res, res, res, device=device)
    y = torch.randint(0, 2, (bs,), device=device)
    dtype = getattr(torch, amp_dtype)
    use_amp = device.type == "cuda" and dtype is not None

    started = None
    for iteration in range(warmup + steps):
        if iteration == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
        with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
            loss = torch.nn.functional.cross_entropy(model(x)["logits"], y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()

    result = {
        "architecture": architecture,
        "batch_size": bs,
        "res": res,
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
        "ms_per_step": round((time.perf_counter() - started) / steps * 1000, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2) if device.type == "cuda" else None,
    }
    del model, optimizer, x, y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default=os.path.join(ROOT, "configs", "glaucoma.yaml"))
    p.add_argument("--res", type=int, default=96)
    p.add_argument("--batch-sizes", default="1,2")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--architectures", default="simple3dcnn,pyramid3dcnn")
    p.add_argument("--amp", default="bf16" if torch.cuda.is_available() else "none")
    args = p.parse_args()

    cfg = load_config(args.cfg)
    dtype = None if args.amp == "none" else args.amp
    results = []
    for bs in [int(v) for v in args.batch_sizes.split(",")]:
        for architecture in args.architectures.split(","):
            results.append(bench(architecture, cfg, bs, args.res, args.steps, dtype))
            print(json.dumps(results[-1]))

    print("\n| architecture | batch | res | params (M) | ms/step | peak VRAM (GB) |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r['architecture']} | {r['batch_size']} | {r['res']} | {r['params_m']} | "
            f"{r['ms_per_step']} | {r['peak_vram_gb'] if r['peak_vram_gb'] is not None else '—'} |"
        )


if __name__ == "__main__":
    main()
