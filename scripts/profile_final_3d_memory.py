"""Bounded CUDA measurements of the actual 3D encoder, not a full-model fit claim."""

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path


def worker(args):
    import time

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    torch.cuda.set_per_process_memory_fraction(0.55)
    torch.backends.cudnn.benchmark = False
    source = Path(__file__).with_name("final_model.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", "") in {"gnorm", "ResXBlock3D", "Enc3DResNeXt"}]
    scope = {"torch": torch, "nn": nn, "F": F}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    encoder = scope["Enc3DResNeXt"]().cuda()
    head = nn.Linear(192, 2).cuda()
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=5e-5, foreach=False)
    scaler = torch.cuda.amp.GradScaler()
    x = torch.randn(args.batch, 1, args.res, args.res, args.res, device="cuda")
    labels = torch.arange(args.batch, device="cuda") % 2

    def early(value):
        value = encoder.stem(value)
        for block in encoder.stages[:2]:
            value = block(value)
        return value

    def forward(value):
        if args.checkpoint:
            value = checkpoint(early, value, use_reentrant=False)
            for block in encoder.stages[2:]:
                value = checkpoint(block, value, use_reentrant=False)
            return encoder.pool(value).flatten(1)
        return encoder(value)

    started = time.perf_counter()
    try:
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = F.cross_entropy(head(forward(x)), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()
        result = {
            "status": "ok",
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            "seconds_two_steps": time.perf_counter() - started,
        }
    except torch.cuda.OutOfMemoryError:
        result = {"status": "oom_under_55_percent_allocator_cap"}
    result.update(
        res=args.res,
        batch=args.batch,
        checkpoint=args.checkpoint,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        scope="3D encoder + linear head; no MaxViT",
    )
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cuda-python")
    parser.add_argument("--res", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if not args.cuda_python:
        parser.error("--cuda-python required")
    import numpy as np
    import wandb

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline.utils import load_env_file

    load_env_file()
    output = Path("outputs/memory_profile_local")
    output.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project="glaucoma-thesis",
        name="local-3d-memory-profile",
        mode="offline",
        dir=str(output),
        config={"synthetic": True, "full_model": False, "allocator_fraction": 0.55},
    )
    results, estimates = [], []
    try:
        for batch, use_checkpoint in ((2, False), (1, False), (1, True)):
            for resolution in (32, 48, 64, 80):
                command = [
                    args.cuda_python,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--res",
                    str(resolution),
                    "--batch",
                    str(batch),
                ]
                if use_checkpoint:
                    command.append("--checkpoint")
                process = subprocess.run(command, capture_output=True, text=True, timeout=120)
                if process.returncode:
                    raise RuntimeError(process.stderr)
                result = json.loads(process.stdout.strip().splitlines()[-1])
                results.append(result)
                print(result, flush=True)
                run.log(result)
                if result["status"] != "ok":
                    break
            subset = [
                r for r in results if r["batch"] == batch and r["checkpoint"] == use_checkpoint and r["status"] == "ok"
            ]
            if len(subset) >= 3:
                a, b = np.polyfit([r["res"] ** 3 for r in subset], [r["peak_allocated_gib"] for r in subset], 1)
                estimate = {
                    "batch": batch,
                    "checkpoint": use_checkpoint,
                    "extrapolated_3d_only_200_gib": float(a * 200**3 + b),
                    "warning": "Cubic extrapolation only; excludes MaxViT and full-model overlap. Workspace may change.",
                }
                estimates.append(estimate)
                print(estimate, flush=True)
        report = {
            "measurements": results,
            "estimates": estimates,
            "limitations": "Old local CUDA stack; 3D-only synthetic microbenchmarks, not full 200^3 training. "
            "Offline W&B. Drive sync pending: no verified Drive mount on local Windows.",
        }
        (output / "measurements.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        run.summary.update({"measurements": len(results), "full_model_verified": False})
    finally:
        run.finish()


if __name__ == "__main__":
    main()
