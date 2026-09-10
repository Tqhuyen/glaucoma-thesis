import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resolution_study import _box1d  # noqa: E402

VIEWS = ["aip_full", "slab_aip", "slab_mip"]


def depth_axis(vol):
    f = vol.astype(np.float32)
    stds = [float(f.mean(axis=tuple(i for i in range(3) if i != ax)).std()) for ax in range(3)]
    return int(np.argmax(stds))


def to_depth_last(vol, dz):
    if dz == 2:
        return vol
    others = [i for i in range(3) if i != dz]
    return np.transpose(vol, tuple(others) + (dz,))


def project_views(dvol, half=16):
    S = dvol.shape[2]
    half = min(int(half), max(2, S // 8))
    prof = dvol.mean(axis=(0, 1)).astype(np.float32)
    peak = int(prof.argmax())
    lo, hi = max(0, peak - half), min(S, peak + half + 1)
    out = []
    for m in VIEWS:
        if m == "aip_full":
            out.append(dvol.mean(axis=2))
        elif m == "slab_aip":
            out.append(dvol[:, :, lo:hi].mean(axis=2))
        elif m == "slab_mip":
            out.append(dvol[:, :, lo:hi].max(axis=2))
    return np.stack([np.round(o).clip(0, 255).astype(np.uint8) for o in out], axis=0)


def _box2d(a, k):
    return _box1d(_box1d(a, 0, k), 1, k)


def ssim2d(a, b, data_range=255.0, win=7):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    ux, uy = _box2d(a, win), _box2d(b, win)
    vx = _box2d(a * a, win) - ux * ux
    vy = _box2d(b * b, win) - uy * uy
    vxy = _box2d(a * b, win) - ux * uy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    return float(np.mean(s))


def pearson(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def spearman(a, b):
    ra = np.argsort(np.argsort(a.ravel())).astype(np.float64)
    rb = np.argsort(np.argsort(b.ravel())).astype(np.float64)
    return pearson(ra, rb)


def norm_mutual_info(a, b, bins=32):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    hist, _, _ = np.histogram2d(a, b, bins=bins)
    p = hist / max(hist.sum(), 1)
    px = p.sum(1, keepdims=True)
    py = p.sum(0, keepdims=True)
    nz = p > 0
    mi = float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())
    ha = float(-(px[px > 0] * np.log(px[px > 0])).sum())
    hb = float(-(py[py > 0] * np.log(py[py > 0])).sum())
    return float(2 * mi / (ha + hb)) if (ha + hb) > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "glaucoma_all_200"))
    ap.add_argument("--split", default="Validation")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default=os.path.join("figures", "view_analysis"))
    args = ap.parse_args()

    vol_path = os.path.join(args.data, f"{args.split}_volumes.npy")
    vols = np.load(vol_path, mmap_mode="r")
    n = min(args.n, len(vols))
    idx = np.random.default_rng(0).choice(len(vols), size=n, replace=False)

    pairs = [(0, 1), (0, 2), (1, 2)]
    acc = {f"{VIEWS[i]}|{VIEWS[j]}": {"pearson": [], "spearman": [], "ssim": [], "nmi": [], "mad": []}
           for i, j in pairs}
    samples = []
    for c, j in enumerate(idx):
        raw = np.asarray(vols[j])
        if raw.ndim == 4:
            raw = raw[0]
        raw = np.ascontiguousarray(raw)
        dz = depth_axis(raw)
        views = project_views(to_depth_last(raw, dz))
        if c < 3:
            samples.append(views)
        for i, k in pairs:
            a, b = views[i].astype(np.float32), views[k].astype(np.float32)
            key = f"{VIEWS[i]}|{VIEWS[k]}"
            acc[key]["pearson"].append(pearson(a, b))
            acc[key]["spearman"].append(spearman(a, b))
            acc[key]["ssim"].append(ssim2d(a, b))
            acc[key]["nmi"].append(norm_mutual_info(a, b))
            acc[key]["mad"].append(float(np.mean(np.abs(a - b))))

    summary = {}
    for key, d in acc.items():
        summary[key] = {m: {"mean": float(np.mean(v)), "std": float(np.std(v))} for m, v in d.items()}
    summary["n_volumes"] = n

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "view_redundancy.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"n_volumes={n}")
    for key, d in summary.items():
        if key == "n_volumes":
            continue
        print(f"{key:24s} pearson={d['pearson']['mean']:.3f} spearman={d['spearman']['mean']:.3f} "
              f"ssim={d['ssim']['mean']:.3f} NMI={d['nmi']['mean']:.3f} MAD={d['mad']['mean']:.1f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(len(samples), 3, figsize=(11, 3.6 * len(samples)))
        axs = np.atleast_2d(axs)
        for r, views in enumerate(samples):
            for c in range(3):
                axs[r, c].imshow(views[c], cmap="gray")
                axs[r, c].set_title(VIEWS[c], fontsize=9)
                axs[r, c].axis("off")
        fig.suptitle("En-face views from 3 OCT volumes (rows) — aip_full / slab_aip / slab_mip")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "view_examples.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        labels = list(acc.keys())
        means = [np.mean(acc[k]["pearson"]) for k in labels]
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.bar(labels, means, color="#4c72b0")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("mean Pearson r")
        ax.set_title("Pairwise similarity between en-face views")
        for i, v in enumerate(means):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "view_pairwise_pearson.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote figures to", args.out)
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
