"""Run the repository training pipeline on Modal (used when Kaggle GPU quota is gone).

    python -m modal run scripts/modal_train.py --experiment smoke
    python -m modal run scripts/modal_train.py --experiment resnext
    python -m modal run scripts/modal_train.py --experiment thinhead
    python -m modal run scripts/modal_train.py --experiment aniso96x200

Data and outputs live on the Modal Volume ``glaucoma-thesis-vol``; secrets come
from the Modal secret ``glaucoma-secrets`` (HF_TOKEN, WANDB_API_KEY).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("glaucoma-thesis")
VOL = modal.Volume.from_name("glaucoma-thesis-vol", create_if_missing=True)
HF_REPO = "tqhuyen/harvard-oct-glaucoma-200"
HF_REPO_96 = "tqhuyen/harvard-oct-glaucoma-200-bilateral-96"
DATA = Path("/vol/data")
OUT = Path("/vol/outputs")

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.1",
        "numpy<2",
        "pyyaml",
        "scikit-learn",
        "tensorboard",
        "wandb",
        "huggingface_hub",
        "python-dotenv",
        "tqdm",
        "pandas",
        "matplotlib",
        "pynvml",
        "psutil",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/Tqhuyen/glaucoma-thesis.git /repo",
    )
)

SECRETS = [modal.Secret.from_name("glaucoma-secrets")]
VOLUMES = {"/vol": VOL}


def _hf_download(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset", token=os.environ["HF_TOKEN"])


@app.function(image=IMAGE, volumes=VOLUMES, secrets=SECRETS, timeout=1800)
def prepare_smoke() -> str:
    import numpy as np

    root = DATA / "smoke"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for split, n in (("Training", 40), ("Validation", 8), ("Test", 8)):
        y = np.array([i % 2 for i in range(n)], dtype=np.int64)
        v = rng.integers(30, 70, size=(n, 1, 32, 32, 32), dtype=np.uint8)
        v[y == 1] += 120
        np.save(root / f"{split}_volumes.npy", v)
        np.save(root / f"{split}_labels.npy", y)
    VOL.commit()
    return str(root)


@app.function(image=IMAGE, volumes=VOLUMES, secrets=SECRETS, timeout=3600)
def prepare_bilateral96() -> str:
    root = DATA / "bilateral96"
    root.mkdir(parents=True, exist_ok=True)
    for split in ("Training", "Validation", "Test"):
        vol = _hf_download(HF_REPO_96, f"{split}_volumes_dn96.npy")
        lab = _hf_download(HF_REPO_96, f"{split}_labels.npy")
        shutil.copyfile(vol, root / f"{split}_volumes.npy")
        shutil.copyfile(lab, root / f"{split}_labels.npy")
    VOL.commit()
    return str(root)


@app.function(image=IMAGE, volumes=VOLUMES, secrets=SECRETS, timeout=3 * 3600)
def prepare_aniso96x200() -> str:
    import numpy as np
    import torch
    import torch.nn.functional as F

    root = DATA / "aniso96x200"
    root.mkdir(parents=True, exist_ok=True)
    for split in ("Training", "Validation", "Test"):
        src = _hf_download(HF_REPO, f"{split}_volumes.npy")
        lab = _hf_download(HF_REPO, f"{split}_labels.npy")
        raw = np.load(src, mmap_mode="r")
        out = np.lib.format.open_memmap(
            root / f"{split}_volumes.npy", mode="w+", dtype=np.uint8, shape=(len(raw), 1, 96, 96, 200)
        )
        for i in range(len(raw)):
            x = torch.from_numpy(np.asarray(raw[i], dtype=np.float32) / 255.0)
            y = F.interpolate(x[None], size=(96, 96, 200), mode="trilinear", align_corners=False)[0]
            out[i] = (y.clamp(0, 1) * 255).round().to(torch.uint8).numpy()[None]
            if (i + 1) % 500 == 0:
                print(split, i + 1, "/", len(raw), flush=True)
        out.flush()
        del out, raw
        shutil.copyfile(lab, root / f"{split}_labels.npy")
        for p in (src, lab):
            Path(p).unlink(missing_ok=True)
    VOL.commit()
    return str(root)


@app.function(image=IMAGE, gpu="T4", volumes=VOLUMES, secrets=SECRETS, timeout=4 * 3600)
def train(
    run_name: str,
    architecture: str,
    data_dir: str,
    epochs: int = 12,
    batch_size: int = 4,
    grad_accum: int = 2,
    head_pool: str = "gap",
    fusion_dim: int = 64,
    smoke: bool = False,
) -> dict:
    overrides = [
        f"run_name={run_name}",
        f"model.architecture={architecture}",
        f"data.data_dir={data_dir}",
        "logging.wandb=true",
        "logging.tensorboard=true",
        f"output_dir={OUT}",
        f"model.head_pool={head_pool}",
        f"model.fusion_dim={fusion_dim}",
        "train.lr=0.0001",
        f"train.epochs={epochs}",
        f"train.batch_size={batch_size}",
        f"train.grad_accum_steps={grad_accum}",
        "train.eval_every_steps=100",
        "train.eval_test_every_steps=300",
        "train.save_every_steps=300",
        "train.early_stop_patience=6",
    ]
    cmd = [sys.executable, "-m", "pipeline.train", "--cfg", "configs/glaucoma.yaml"]
    if smoke:
        cmd.append("--smoke")
        overrides += ["train.epochs=1", "train.eval_every_steps=2", "train.eval_test_every_steps=4"]
    cmd += ["--set"] + overrides
    print(" ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONPATH": "/repo", "HF_XET_HIGH_PERFORMANCE": "1"}
    subprocess.run(cmd, cwd="/repo", env=env, check=True)
    VOL.commit()
    result = OUT / run_name / "test_results.json"
    return {"run": run_name, "test_results": result.read_text() if result.exists() else None}


@app.local_entrypoint()
def main(
    experiment: str = "smoke",
    run_name: str = "",
    epochs: int = 12,
    batch_size: int = 4,
    grad_accum: int = 2,
):
    if experiment == "smoke":
        data = prepare_smoke.remote()
        print(
            train.remote(
                run_name=run_name or "modal-smoke",
                architecture="pyramid3dcnn",
                data_dir=data,
                epochs=1,
                batch_size=2,
                grad_accum=1,
                head_pool="gap",
                smoke=True,
            )
        )
        return

    data = prepare_bilateral96.remote() if experiment != "aniso96x200" else prepare_aniso96x200.remote()
    if experiment == "resnext":
        print(
            train.remote(
                run_name=run_name or "pyramid-resnext-bilateral96-s42",
                architecture="pyramid_resnext3d",
                data_dir=data,
                epochs=epochs,
                batch_size=batch_size,
                grad_accum=grad_accum,
                head_pool="gap",
                fusion_dim=128,
            )
        )
    elif experiment == "thinhead":
        print(
            train.remote(
                run_name=run_name or "pyramid-thinhead-bilateral96-s42",
                architecture="pyramid3dcnn",
                data_dir=data,
                epochs=epochs,
                batch_size=batch_size,
                grad_accum=grad_accum,
                head_pool="depth_attention",
                fusion_dim=64,
            )
        )
    elif experiment == "aniso96x200":
        print(
            train.remote(
                run_name=run_name or "pyramid-aniso96x200-s42",
                architecture="pyramid3dcnn",
                data_dir=data,
                epochs=epochs,
                batch_size=batch_size,
                grad_accum=grad_accum,
                head_pool="gap",
                fusion_dim=64,
            )
        )
    else:
        raise ValueError(f"unknown experiment: {experiment}")
