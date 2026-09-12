"""Identity-aware XAI artifacts without touching the frozen final-training bundle.

The consolidated Harvard OCT volumes carry no per-sample file name, so a plain
Grad-CAM PNG cannot be traced back to a scan. This module resolves the dataset
row to its original stem (verified against the Harvard-GF summary CSV and the
consolidated ``progress.json`` for ``tqhuyen/harvard-oct-glaucoma-200``), embeds
the identity in every file name and in ``*_xai_meta.json``, and saves the raw
saliency tensors so later analysis never has to recover heatmaps from PNGs.
"""

import json
from pathlib import Path

import numpy as np

SPLIT_OFFSETS = {"Training": 0, "Validation": 2100, "Test": 2400}
SOURCE_REPO = "tqhuyen/harvard-oct-glaucoma-200"
SOURCE_REVISION = "939a38876b7b9313162842ef2d44b7edc2b57020"


def resolve_stem(split, index, *, offsets=None):
    offsets = SPLIT_OFFSETS if offsets is None else offsets
    if split not in offsets:
        raise ValueError(f"Unknown split: {split}")
    index = int(index)
    if index < 0:
        raise ValueError("index must be nonnegative")
    return f"data_{offsets[split] + index + 1:04d}"


def sample_identity(split, index, label, *, offsets=None):
    return {
        "split": split,
        "index": int(index),
        "stem": resolve_stem(split, index, offsets=offsets),
        "label": int(label),
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
    }


def identity_prefix(identity):
    return f"{identity['split']}_idx{identity['index']}_{identity['stem']}_label{identity['label']}"


def _heatmap_slice(array):
    array = np.asarray(array, dtype=np.float32)
    return array[array.shape[0] // 2] if array.ndim == 3 else array


def _overlay(image, heat, alpha=0.42):
    scale = float(np.max(heat)) or 1.0
    heat = np.clip(np.asarray(heat, dtype=np.float32) / scale, 0, 1)
    base = np.asarray(image, dtype=np.float32)
    lo, hi = float(np.percentile(base, 1)), float(np.percentile(base, 99))
    base = np.clip((base - lo) / max(hi - lo, 1e-6), 0, 1)
    base = np.stack([base] * 3, -1)
    heat = _resize_nearest(heat, base.shape[:2])
    colored = _jet(heat)
    return np.clip((1 - alpha) * base + alpha * colored, 0, 1)


def _resize_nearest(array, shape):
    rows = np.clip((np.arange(shape[0]) * array.shape[0] / shape[0]).astype(int), 0, array.shape[0] - 1)
    cols = np.clip((np.arange(shape[1]) * array.shape[1] / shape[1]).astype(int), 0, array.shape[1] - 1)
    return array[rows][:, cols]


def _jet(values):
    r = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    return np.stack([r, g, b], -1)


def save_xai_identified(model, dataset, artifacts, run, *, split, index, smoke=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    import wandb
    from scripts import final_model as fm

    model.eval()
    device = next(model.parameters()).device
    x, v, target = dataset[index]
    x, v, target = x[None].to(device), v[None].to(device), int(target)
    identity = sample_identity(split, index, target)
    prefix = identity_prefix(identity)
    tensors = {}

    def save_png(array, name):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(np.atleast_2d(array), cmap="hot")
        ax.set_title(f"{name} | {identity['stem']} ({split}[{index}], label={target})")
        path = artifacts.local / f"{prefix}_{name}.png"
        fig.savefig(path)
        plt.close(fig)
        artifacts.sync(path)
        run.log({f"xai/{name}": wandb.Image(str(path))})
        return path

    def save_overlay(image, name):
        heat = _heatmap_slice(tensors[name])
        patch = _overlay(image, heat)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(patch)
        ax.set_title(f"{name} overlay | {identity['stem']} (label={target})")
        ax.axis("off")
        path = artifacts.local / f"{prefix}_{name}_overlay.png"
        fig.savefig(path)
        plt.close(fig)
        artifacts.sync(path)
        run.log({f"xai/{name}_overlay": wandb.Image(str(path))})

    def record(name, array):
        array = np.asarray(array, dtype=np.float32)
        tensors[name] = array
        save_png(_heatmap_slice(array), name)

    module = model.conv if smoke else model.gcam3d_module()
    cam, _, _ = fm.grad_cam(module, model, x, v, target=target, is_3d=True)
    tensors["gradcam3d"] = np.asarray(cam, dtype=np.float32)
    save_png(_heatmap_slice(cam), "gradcam3d")
    if not smoke:
        for i in range(v.shape[1]):
            cam2d, _, _ = fm.grad_cam(model.gcam2d_module(i), model, x, v, target=target, is_3d=False)
            record(f"gradcam2d_{i}", cam2d)
            save_overlay(v[0, i, 0].detach().cpu().numpy(), f"gradcam2d_{i}")
    record("occlusion", fm.occlusion_sensitivity(model, x, v, target, n=2 if smoke else 4)[0])
    record("integrated_gradients", fm.integrated_gradients(model, x, v, target, steps=2 if smoke else 16))
    meta = {"identity": identity, "tensors": {name: list(array.shape) for name, array in tensors.items()}}
    if not smoke:
        weights, gate = fm.crossgate_attention(model, x, v)
        names, drops = fm.branch_drop_importance(model, x, v, target)
        names = ["all_2d_inputs", *names[1:]]
        meta["fusion"] = {"gate": float(gate), "branch_drop": {name: float(drop) for name, drop in zip(names, drops)}}
        meta["fusion"]["attention"] = (
            {f"2D-{i}": float(value) for i, value in enumerate(np.asarray(weights).ravel())}
            if weights is not None
            else {}
        )
        run.summary.update({f"xai/{key}": value for key, value in meta["fusion"]["branch_drop"].items()})
    artifacts.save(tensors, f"{prefix}_saliency_tensors.pt")
    meta_path = artifacts.local / f"{prefix}_xai_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    artifacts.sync(meta_path)
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return meta


def load_identity(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["identity"]
