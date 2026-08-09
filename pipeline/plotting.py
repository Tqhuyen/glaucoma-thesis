"""Render train/val/test curves from metrics.jsonl -> PNG (+ optional Drive copy).

Called automatically at the end of every run when `logging.plot_curves: true`.
The PNG + metrics.jsonl + best_model.pt can be synced to Google Drive via
`logging.drive_sync_dir` (set to e.g. /content/drive/MyDrive/thesis/runs/<run>).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe (Colab, vast.ai, SLURM)
import matplotlib.pyplot as plt


def load_metrics(out_dir: Path) -> dict:
    """Read metrics.jsonl into {step: {metric: value}} for the given run."""
    path = out_dir / "metrics.jsonl"
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            rows[int(rec["step"])] = {k: v for k, v in rec.items() if k not in ("step", "time")}
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return rows


def render_curves(out_dir: Path, tag: str = "training_curves") -> Path | None:
    """Plot train/val/test loss + accuracy + F1 from metrics.jsonl -> PNG.

    Returns the PNG path, or None if there is nothing to plot.
    """
    rows = load_metrics(out_dir)
    if not rows:
        return None

    steps = sorted(rows)

    def series(key):
        return [(s, rows[s][key]) for s in steps if key in rows[s]]

    train_loss, val_loss, test_loss = series("train/loss"), series("val/loss"), series("test/loss")
    val_acc, test_acc, train_acc = series("val/acc"), series("test/acc"), series("train/acc")
    val_f1, test_f1, train_f1 = series("val/f1"), series("test/f1"), series("train/f1")

    if not (train_loss or val_loss or train_acc):
        return None

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    def plot_series(ax, series_data, label, color):
        if series_data:
            ax.plot([s for s, _ in series_data], [v for _, v in series_data],
                    label=label, color=color, marker="o", ms=2.5, lw=1.2)
            ax.legend()

    plot_series(axes[0], train_loss, "train loss", "tab:blue")
    plot_series(axes[0], val_loss, "val loss", "tab:orange")
    plot_series(axes[0], test_loss, "test loss", "tab:green")
    axes[0].set_title("Loss (lower is better)")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)

    plot_series(axes[1], train_acc, "train acc", "tab:blue")
    plot_series(axes[1], val_acc, "val acc", "tab:orange")
    plot_series(axes[1], test_acc, "test acc", "tab:green")
    axes[1].set_title("Accuracy (higher is better)")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)

    plot_series(axes[2], train_f1, "train f1", "tab:blue")
    plot_series(axes[2], val_f1, "val f1", "tab:orange")
    plot_series(axes[2], test_f1, "test f1", "tab:green")
    axes[2].set_title("F1 (higher is better)")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("f1")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(out_dir.name)
    fig.tight_layout()
    out_png = out_dir / f"{tag}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def sync_to_drive(out_dir: Path, drive_dir: str, tag: str = "training_curves") -> Path | None:
    """Copy run artifacts (PNG, metrics.jsonl, best_model.pt) to Google Drive.

    Returns the Drive directory Path, or None if drive_sync_dir is empty.
    """
    if not drive_dir:
        return None
    dst = Path(drive_dir)
    dst.mkdir(parents=True, exist_ok=True)

    candidates = [out_dir / f"{tag}.png", out_dir / "metrics.jsonl", out_dir / "best_model.pt"]
    copied = []
    for src in candidates:
        if src.exists():
            shutil.copy2(src, dst / src.name)
            copied.append(str(dst / src.name))

    if copied:
        print(f"[drive] synced {len(copied)} artifact(s) -> {dst}")
        for c in copied:
            print(f"  - {c}")
    return dst


def render_and_sync(out_dir: Path, cfg: dict, tag: str = "training_curves") -> Path | None:
    """Plot curves and (optionally) copy artifacts to Drive. Returns the PNG path."""
    lcfg = cfg.get("logging", {})
    png = render_curves(out_dir, tag=tag) if lcfg.get("plot_curves", True) else None
    if lcfg.get("drive_sync_dir"):
        drive = os.path.expandvars(lcfg.get("drive_sync_dir", ""))
        if drive:
            sync_to_drive(out_dir, drive, tag=tag)
    return png
