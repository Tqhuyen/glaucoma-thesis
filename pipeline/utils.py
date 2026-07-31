"""Utilities that prevent the 4 most common remote-training bugs:
1. Non-reproducible runs        -> seed_everything()
2. Silent config typos          -> load_config() with strict key checks
3. Lost work on disconnect      -> atomic checkpoint save/load
4. Wrong device / dtype         -> resolve_device(), resolve_amp_dtype()
"""
from __future__ import annotations

import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, value = item.split("=", 1)
        _set_nested(cfg, key, value)
    return cfg


def _set_nested(cfg: dict, dotted_key: str, raw_value: str) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node:
            raise KeyError(f"Unknown config section '{k}' in '{dotted_key}'")
        node = node[k]
    leaf = keys[-1]
    if leaf not in node:
        raise KeyError(f"Unknown config key '{dotted_key}'. Check spelling against the YAML.")
    node[leaf] = yaml.safe_load(raw_value)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("WARNING: no GPU detected — training on CPU (fine for smoke tests only).")
    return torch.device("cpu")


def resolve_amp_dtype(device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def atomic_save(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        torch.save(state, tmp.name)
        tmp_path = tmp.name
    shutil.move(tmp_path, path)


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return ckpts[-1] if ckpts else None


def prune_checkpoints(ckpt_dir: Path, keep: int) -> None:
    ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for old in ckpts[:-keep]:
        old.unlink(missing_ok=True)


def optimal_num_proc(cfg_value) -> int:
    if cfg_value != "auto":
        return int(cfg_value)
    try:
        import psutil
        cores = psutil.cpu_count(logical=False) or os.cpu_count() or 2
    except ImportError:
        cores = os.cpu_count() or 2
    return max(1, cores)


def optimal_num_workers(cfg_value) -> int:
    if cfg_value != "auto":
        return int(cfg_value)
    return max(2, min(os.cpu_count() or 2, 8))


def setup_hf_env(hf_cfg: dict) -> str | None:
    if hf_cfg.get("fast_transfer", True):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_cfg.get("use_auth", False) and not token:
        print(
            "WARNING: hf.use_auth=true but HF_TOKEN env var is not set.\n"
            "  Public datasets will still work; gated/private ones will 401.\n"
            "  Fix:  export HF_TOKEN=hf_xxx   (vast.ai)  |  Colab secrets (key icon) -> HF_TOKEN"
        )
    return token


def write_run_manifest(out_dir: Path) -> None:
    import platform
    import subprocess
    import sys

    def sh(cmd):
        try:
            return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"

    manifest = {
        "git_sha": sh("git rev-parse HEAD"),
        "git_dirty": bool(sh("git status --porcelain")),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
    }
    import json
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


def detect_environment() -> str:
    if "COLAB_RELEASE_TAG" in os.environ or Path("/content").exists():
        return "colab"
    if Path("/workspace").exists():
        return "vast"
    return "local"
