"""Device benchmark for the controls-96 specs (synthetic tensors, no dataset).

Run the same command locally and on a Colab T4 to compare step time, throughput
and peak VRAM. Example:

    python scripts/controls_bench.py --batch-sizes 1,2,4,8 --steps 3
"""

import argparse
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import controls_model as cm  # noqa: E402

SPECS = {
    "P": dict(use_3d=True, n_2d=2, fusion="crossgate", gate_fixed=False),
    "B1": dict(use_3d=True, n_2d=1, fusion="crossgate", gate_fixed=False),
    "B2": dict(use_3d=True, n_2d=1, fusion="crossgate", gate_fixed=False),
    "B3": dict(use_3d=False, n_2d=2, fusion="concat", gate_fixed=False),
    "C1": dict(use_3d=True, n_2d=2, fusion="concat", gate_fixed=False),
    "C2": dict(use_3d=True, n_2d=2, fusion="crossgate", gate_fixed=True),
}


def configure():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def bench(spec, bs, steps, res3d, res2d, amp_dtype, warmup):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    model = cm.ControlsModel(
        n_2d=spec["n_2d"],
        D=256,
        enc2d="maxvit_tiny_rw_224",
        enc3d_features=(32, 64, 128, 192),
        enc2d_pretrained=False,
        view_indices=tuple(range(spec["n_2d"])),
        fusion=spec["fusion"],
        use_3d=spec["use_3d"],
        gate_fixed=spec["gate_fixed"],
    ).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(bs, 1, res3d, res3d, res3d, device=device)
    views = torch.randn(bs, spec["n_2d"], 1, res2d, res2d, device=device)
    target = torch.randint(0, 2, (bs,), device=device)
    dtype = getattr(torch, amp_dtype)

    for iteration in range(warmup + steps):
        if iteration == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            loss = torch.nn.functional.cross_entropy(model(x, views), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    per_step = (time.perf_counter() - started) / steps
    peak = torch.cuda.max_memory_reserved() / 2 ** 30 if device.type == "cuda" else 0.0
    params = sum(p.numel() for p in model.parameters())
    del model, optimizer, x, views, target
    torch.cuda.empty_cache()
    gc.collect()
    return per_step, peak, params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", default="P,B1,B2,B3,C1,C2")
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--res3d", type=int, default=96)
    parser.add_argument("--res2d", type=int, default=224)
    parser.add_argument("--amp-dtype", default="bfloat16")
    parser.add_argument("--train-n", type=int, default=2100, help="samples per epoch, for time estimates")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    configure()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"device={name} type={device} torch={torch.__version__} amp={args.amp_dtype}")
    results = []
    for code in [s.strip() for s in args.specs.split(",") if s.strip()]:
        for bs in [int(b) for b in args.batch_sizes.split(",") if b.strip()]:
            try:
                per_step, peak, params = bench(SPECS[code], bs, args.steps, args.res3d, args.res2d,
                                               args.amp_dtype, args.warmup)
            except torch.cuda.OutOfMemoryError:
                print(f"{code:3s} bs={bs} OOM")
                results.append({"spec": code, "batch_size": bs, "oom": True})
                continue
            epoch_min = args.train_n * per_step / bs / 60
            print(f"{code:3s} bs={bs} step_s={per_step:.3f} peak_reserved_GiB={peak:.2f} "
                  f"epoch_min={epoch_min:.1f} run20_h={20 * epoch_min / 60:.1f} params={params}")
            results.append({"spec": code, "batch_size": bs, "step_s": per_step, "peak_reserved_gib": peak,
                            "epoch_min": epoch_min, "run20_h": 20 * epoch_min / 60, "params": params})
    if args.out:
        json.dump({"device": name, "torch": torch.__version__, "amp": args.amp_dtype, "results": results},
                  open(args.out, "w"), indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
