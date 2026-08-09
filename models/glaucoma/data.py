import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _robust_stats(volume):
    """Per-volume percentile clip + z-score (robust to OCT outliers)."""
    lo, hi = np.percentile(volume, (1, 99))
    v = np.clip(volume, lo, hi)
    mu, sd = float(v.mean()), float(v.std()) + 1e-8
    return v, mu, sd


class GlaucomaNpyDataset(Dataset):
    def __init__(self, data_dir, split, cache_in_ram=False, normalize="minmax"):
        self.labels = np.load(os.path.join(data_dir, f"{split}_labels.npy"))
        vp = os.path.join(data_dir, f"{split}_volumes.npy")
        if cache_in_ram:
            self.volumes = np.load(vp)
        else:
            self.volumes = np.load(vp, mmap_mode="r")
        self.normalize = normalize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = np.asarray(self.volumes[idx], dtype=np.float32)
        if self.normalize == "robust":
            x, _, _ = _robust_stats(x)
        elif self.normalize != "none":
            x = x / 255.0
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return torch.from_numpy(np.ascontiguousarray(x)), y


def build_dataloaders(cfg, smoke=False):
    dcfg = cfg["data"]
    tcfg = cfg["train"]

    data_dir = dcfg.get("data_dir")
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "glaucoma_all"
        )
    data_dir = os.path.normpath(data_dir)

    bs = cfg["smoke"]["batch_size"] if smoke else tcfg["batch_size"]

    nw = os.cpu_count() or 2
    if dcfg.get("num_workers") is not None and dcfg["num_workers"] != "auto":
        nw = int(dcfg["num_workers"])
    nw = max(2, min(nw, 8))

    cache = dcfg.get("cache_in_ram", False)
    normalize = dcfg.get("normalize", "minmax")
    train_ds = GlaucomaNpyDataset(data_dir, "Training", cache_in_ram=cache, normalize=normalize)
    val_ds = GlaucomaNpyDataset(data_dir, "Validation", cache_in_ram=cache, normalize=normalize)
    test_ds = GlaucomaNpyDataset(data_dir, "Test", cache_in_ram=cache, normalize=normalize)

    if smoke:
        limit = cfg["smoke"]["max_train_samples"]
        train_ds.labels = train_ds.labels[:limit]
        train_ds.volumes = train_ds.volumes[:limit]
        limit_v = cfg["smoke"]["max_val_samples"]
        val_ds.labels = val_ds.labels[:limit_v]
        val_ds.volumes = val_ds.volumes[:limit_v]

    common = dict(
        batch_size=bs,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
    )

    g = torch.Generator()
    g.manual_seed(cfg["seed"])

    train_loader = DataLoader(train_ds, shuffle=True, generator=g, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    print(
        f"[glaucoma] {len(train_ds)} train | {len(val_ds)} val | "
        f"{len(test_ds)} test | bs={bs} | workers={nw}"
    )
    return train_loader, val_loader, test_loader
