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
def prepare_raw200() -> str:
    root = DATA / "raw200"
    root.mkdir(parents=True, exist_ok=True)
    for split in ("Training", "Validation", "Test"):
        vol = _hf_download(HF_REPO, f"{split}_volumes.npy")
        lab = _hf_download(HF_REPO, f"{split}_labels.npy")
        shutil.copyfile(vol, root / f"{split}_volumes.npy")
        shutil.copyfile(lab, root / f"{split}_labels.npy")
        for p in (vol, lab):
            Path(p).unlink(missing_ok=True)
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


@app.function(image=IMAGE, gpu="T4", cpu=8, memory=32768, volumes=VOLUMES, secrets=SECRETS, timeout=900)
def probe(architecture: str = "pyramid_resnext3d", batch_size: int = 2, steps: int = 3) -> dict:
    import time

    import torch

    sys.path.insert(0, "/repo")
    from models.glaucoma.model import build_model
    from pipeline.utils import load_config

    cfg = load_config("/repo/configs/glaucoma.yaml")
    cfg["model"]["architecture"] = architecture
    if architecture == "pyramid_resnext3d":
        cfg["model"]["fusion_dim"] = 128
    model = build_model(cfg).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(batch_size, 1, 200, 200, 200, device="cuda")
    y = torch.randint(0, 2, (batch_size,), device="cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    for i in range(steps + 1):
        if i == 1:
            torch.cuda.synchronize()
            started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(model(x)["logits"], y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return {
        "architecture": architecture,
        "batch_size": batch_size,
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
        "sec_per_microbatch": round((time.perf_counter() - started) / steps, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }


@app.function(image=IMAGE, gpu="T4", cpu=8, memory=32768, volumes=VOLUMES, secrets=SECRETS, timeout=12 * 3600)
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
    if data_dir == "raw200":
        local = Path("/tmp/raw200")
        local.mkdir(parents=True, exist_ok=True)
        for split in ("Training", "Validation", "Test"):
            for name in (f"{split}_volumes.npy", f"{split}_labels.npy"):
                dst = local / name
                if not dst.exists():
                    from huggingface_hub import hf_hub_download

                    hf_hub_download(
                        HF_REPO, name, repo_type="dataset", local_dir=str(local), token=os.environ["HF_TOKEN"]
                    )
        data_dir = str(local)
        print("[data] raw200 ready on local disk", flush=True)

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
    out_dir = OUT / run_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
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
    probe_batch: int = 2,
):
    if experiment == "probe-resnext":
        print(probe.remote(architecture="pyramid_resnext3d", batch_size=probe_batch))
        return
    if experiment == "probe-pyramid":
        print(probe.remote(architecture="pyramid3dcnn", batch_size=probe_batch))
        return

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

    if experiment.startswith("raw200"):
        data = "raw200"
    elif experiment == "aniso96x200":
        data = prepare_aniso96x200.remote()
    else:
        data = prepare_bilateral96.remote()

    specs = {
        "resnext": ("pyramid_resnext3d", "pyramid-resnext-bilateral96-s42", 128, "gap"),
        "thinhead": ("pyramid3dcnn", "pyramid-thinhead-bilateral96-s42", 64, "depth_attention"),
        "raw200-resnext": ("pyramid_resnext3d", "pyramid-resnext-raw200-s42b", 128, "gap"),
        "raw200-pyramid": ("pyramid3dcnn", "pyramid-raw200-s42b", 64, "gap"),
        "aniso96x200": ("pyramid3dcnn", "pyramid-aniso96x200-s42", 64, "gap"),
    }
    if experiment not in specs:
        raise ValueError(f"unknown experiment: {experiment}")
    architecture, default_run, fusion_dim, head_pool = specs[experiment]
    print(
        train.remote(
            run_name=run_name or default_run,
            architecture=architecture,
            data_dir=data,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum=grad_accum,
            head_pool=head_pool,
            fusion_dim=fusion_dim,
        )
    )
