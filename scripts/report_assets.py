"""Generate report assets on the GPU box before it is destroyed.

Glaucoma: per-sample test probabilities for every raw-200^3 checkpoint, B-scan / en-face images of selected cases
(agreement, disagreement, all-wrong) and input-gradient saliency for two models.
HSI (optional, needs MPHSIR_DIR + HSI_DATA + HSI_RUN): pseudo-RGB clean / noisy / restored images and error maps for
the official, control and physics checkpoints on identical noisy inputs, plus real HyRank scenes.

    python scripts/report_assets.py --runs /workspace/training_runs --data /workspace/data/raw200 --out OUT
"""

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RUNS = ["raw200-simple-s42", "raw200-pyramid-s42", "raw200-resnext-s42", "raw200-resnext-s43", "raw200-pyramid-e15-s42"]


def to_png(array, path, low=None, high=None):
    low = np.percentile(array, 1) if low is None else low
    high = np.percentile(array, 99) if high is None else high
    img = np.clip((array - low) / max(high - low, 1e-8), 0, 1)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def heat_png(array, path):
    import matplotlib.cm as cm

    a = array / max(np.percentile(array, 99.5), 1e-8)
    Image.fromarray((cm.inferno(np.clip(a, 0, 1))[..., :3] * 255).astype(np.uint8)).save(path)


def load_model(run_dir, device):
    from models import get_model_builder

    cfg = json.loads((run_dir / "config_used.json").read_text())
    net = get_model_builder(cfg["model"]["type"])(cfg)
    net.load_state_dict(torch.load(run_dir / "best_model.pt", map_location="cpu"))
    return net.to(device).eval()


def logits_of(net, x):
    out = net(x)
    return out["logits"] if isinstance(out, dict) else out


def glaucoma(args, device):
    out = args.out / "glaucoma"
    out.mkdir(parents=True, exist_ok=True)
    volumes = np.load(args.data / "Test_volumes.npy", mmap_mode="r")
    labels = np.load(args.data / "Test_labels.npy").astype(int)
    probs = {}
    for run in RUNS:
        run_dir = args.runs / run
        if not (run_dir / "best_model.pt").exists():
            continue
        net = load_model(run_dir, device)
        p = np.zeros(len(labels), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(labels), 4):
                x = torch.from_numpy(np.asarray(volumes[start : start + 4], dtype=np.float32) / 255.0).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p[start : start + 4] = torch.softmax(logits_of(net, x).float(), 1)[:, 1].cpu().numpy()
        probs[run] = p
        print(f"[assets] {run}: acc@0.5={((p > 0.5) == labels).mean():.4f}", flush=True)
        del net
        torch.cuda.empty_cache()
    np.savez(out / "test_probs.npz", labels=labels, **probs)

    correct = {r: (p > 0.5) == labels for r, p in probs.items()}
    best, base = "raw200-pyramid-e15-s42", "raw200-simple-s42"
    others = [r for r in probs if r != best]
    rng = np.random.default_rng(0)
    picks = {}

    def pick(name, mask, k=2):
        idx = np.flatnonzero(mask)
        if len(idx):
            for i in rng.choice(idx, min(k, len(idx)), replace=False):
                picks[int(i)] = name

    all_right = np.all([correct[r] for r in probs], 0)
    all_wrong = ~np.any([correct[r] for r in probs], 0)
    pick("all models correct (glaucoma)", all_right & (labels == 1), 1)
    pick("all models correct (normal)", all_right & (labels == 0), 1)
    pick("only Pyramid-e15 correct", correct[best] & ~np.any([correct[r] for r in others], 0))
    pick("Pyramid-e15 correct, Simple wrong", correct[best] & ~correct[base])
    pick("Pyramid-e15 wrong, Simple correct", ~correct[best] & correct[base])
    pick("all models wrong", all_wrong)

    saliency_nets = {r: load_model(args.runs / r, device) for r in (best, base) if r in probs}
    cases = []
    for i, why in picks.items():
        vol = np.asarray(volumes[i], dtype=np.float32)
        to_png(vol[vol.shape[0] // 2], out / f"case{i}_bscan.png")
        to_png(vol.mean(1), out / f"case{i}_enface.png")
        entry = {"index": i, "label": int(labels[i]), "why": why, "probs": {r: float(probs[r][i]) for r in probs}}
        for r, net in saliency_nets.items():
            x = torch.from_numpy(vol / 255.0).unsqueeze(0).to(device).requires_grad_(True)
            logits_of(net, x)[0, 1].backward()
            heat_png(x.grad.abs()[0].mean(1).cpu().numpy(), out / f"case{i}_saliency_{r}.png")
        cases.append(entry)
    (out / "cases.json").write_text(json.dumps(cases, indent=1))


HSI_SETS = {
    "WDC": "WDC_test_256",
    "Xiongan": "xiongan_test_1",
    "Chikusei": "chikusei_test_1",
    "PaviaU": "paviau_test",
    "Houston": "houston_test",
}
HSI_CONDITIONS = ["gauss50", "case2_stripe", "case4_mixture", "physics", "physics_heavy"]
RGB = [41, 25, 12]


def rgb(cube, low, high):
    img = np.stack([cube[b] for b in RGB], -1)
    return Image.fromarray((np.clip((img - low) / (high - low), 0, 1) * 255).astype(np.uint8))


def hsi(args, device):
    import os

    sys.path.insert(0, str(Path(os.environ["HSI_CODE"])))
    import evaluate
    import model as mphsir
    import noise

    import data as hsi_data

    root, run = Path(os.environ["HSI_DATA"]), Path(os.environ["HSI_RUN"])
    out = args.out / "hsi"
    out.mkdir(parents=True, exist_ok=True)
    nets = {"official": mphsir.build(root / "Remote_sensing.ckpt", device).eval()}
    for arm in ("control", "physics"):
        if (run / arm / "last.pt").exists():
            net = mphsir.build(root / "Remote_sensing.ckpt", device)
            net.load_state_dict(torch.load(run / arm / "last.pt", map_location=device))
            nets[arm] = net.eval()
    records = []
    for set_name, stem in HSI_SETS.items():
        path = root / "test" / set_name / f"{stem}.mat"
        if not path.exists():
            continue
        clean = torch.from_numpy(evaluate.crop64(hsi_data.load_cube(path))).unsqueeze(0).to(device)
        clean = clean[..., :256, :256]
        c = clean[0].cpu().numpy()
        low, high = np.percentile(c[RGB], 1), np.percentile(c[RGB], 99)
        rgb(c, low, high).save(out / f"{set_name}_clean.png")
        for condition in HSI_CONDITIONS:
            prompt_id, apply = noise.EVAL_CONDITIONS[condition]
            seed = zlib.crc32(f"{set_name}/{stem}/{condition}/report".encode())
            noisy = apply(clean.clone(), torch.Generator(device=device).manual_seed(seed))
            rgb(noisy[0].cpu().numpy(), low, high).save(out / f"{set_name}_{condition}_noisy.png")
            rec = {
                "set": set_name,
                "condition": condition,
                "psnr_noisy": evaluate.psnr_per_band(noisy.clamp(0, 1), clean.clamp(0, 1)),
            }
            for tag, net in nets.items():
                restored = evaluate.restore(net, noisy, torch.tensor([prompt_id], device=device)).clamp(0, 1)
                r = restored[0].cpu().numpy()
                rgb(r, low, high).save(out / f"{set_name}_{condition}_{tag}.png")
                heat_png(np.abs(r - c).mean(0), out / f"{set_name}_{condition}_{tag}_err.png")
                rec[tag] = {
                    "psnr": evaluate.psnr_per_band(restored, clean.clamp(0, 1)),
                    "sam": evaluate.sam_degrees(restored, clean.clamp(0, 1)),
                }
            records.append(rec)
            print(f"[assets] hsi {set_name} {condition}", flush=True)
    for path in sorted((root / "real").glob("*.npy")):
        cube = torch.from_numpy(np.load(path)).unsqueeze(0).to(device)
        c = cube[0].cpu().numpy()
        low, high = np.percentile(c[RGB], 1), np.percentile(c[RGB], 99)
        rgb(c, low, high).save(out / f"real_{path.stem}_input.png")
        to_png(c[2], out / f"real_{path.stem}_input_band412.png")
        for tag, net in nets.items():
            r = evaluate.restore(net, cube, torch.tensor([1], device=device)).clamp(0, 1)[0].cpu().numpy()
            rgb(r, low, high).save(out / f"real_{path.stem}_{tag}.png")
            to_png(r[2], out / f"real_{path.stem}_{tag}_band412.png", *np.percentile(c[2], [1, 99]))
    (out / "records.json").write_text(json.dumps(records, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-hsi", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda")
    if not (args.out / "glaucoma" / "cases.json").exists():
        glaucoma(args, device)
    if not args.skip_hsi:
        hsi(args, device)


if __name__ == "__main__":
    main()
