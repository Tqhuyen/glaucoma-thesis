"""Kaggle platform adapter. Importing this module never initializes CUDA."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts import controls_kaggle_data as kd
from scripts import controls_model as cm
from scripts import controls_training as ct
from scripts import final_model as fm
from scripts import final_training as ft
from scripts.controls_kaggle_data import make_datasets, require_disk, run_lock, sha256, write_json

SPECS = [
    dict(
        code=code,
        use_3d=code != "B3",
        n_2d=len(views),
        view_indices=views,
        fusion="concat" if code in ("B3", "C1") else "crossgate",
        gate_fixed=code == "C2",
    )
    for code, views in (
        ("P", [0, 1]),
        ("B1", [0]),
        ("B2", [1]),
        ("B3", [0, 1]),
        ("C1", [0, 1]),
        ("C2", [0, 1]),
        ("B4", [0, 1]),
    )
]
OWNED_CHILDREN = {}
RECOVERY_FILES = {
    "run_identity.pt",
    "run_identity.json",
    "last.pt",
    "best_weights.pt",
    "steps.jsonl",
    "history.json",
    "report.json",
    "logits.npz",
    "logits_receipt.json",
    "completed.json",
    "report.csv",
    "initial_stage.json",
}


def default_config():
    smoke = os.environ.get("CTRL_KAGGLE_SMOKE", "0") == "1"
    root = Path(tempfile.mkdtemp(prefix="controls_kaggle_")) if smoke else Path("/kaggle")
    return {
        "smoke": smoke,
        "group": os.environ.get("CTRL_KAGGLE_GROUP", "controls96_kaggle_fp16"),
        "run_target": os.environ.get("CTRL_KAGGLE_RUN_TARGET", ""),
        "output_root": os.environ.get("CTRL_KAGGLE_OUTPUT_ROOT", str(root / "working" / "controls96")),
        "temp_root": os.environ.get("CTRL_KAGGLE_TEMP_ROOT", str(root / "temp" / "controls96")),
        "raw_root": os.environ.get("CTRL_KAGGLE_RAW_ROOT", ""),
        "bilateral_root": os.environ.get("CTRL_KAGGLE_BILATERAL_ROOT", ""),
        "resume_root": os.environ.get("CTRL_KAGGLE_RESUME_ROOT", ""),
        "artifact_refs": {},
        "allow_initialized_recovery": False,
        "gpu_mode": "auto",
        "max_runs_per_session": None if smoke else 2,
        "session_hours": 10.0,
        "grace_seconds": 600,
        "vram_fraction": 0.80,
        "output_limit_gib": 20,
        "job_reserve_gib": 6,
        "fp16_init_scale": 1024,
        "num_workers": 0 if smoke or os.name == "nt" else 2,
        "epochs": 1 if smoke else 20,
        "patience": 2 if smoke else 21,
        "seeds": [42] if smoke else [42, 43, 44],
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "effective_batch": 16,
        "checkpoint_steps": 10,
        "store_res": 8 if smoke else 200,
        "res3d": 8 if smoke else 96,
        "res2d": 8 if smoke else 224,
        "run_xai": False,
        "run_info": False,
    }


def validate_config(cfg):
    pending = [cfg]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if {str(k).upper() for k in value} & {"HF_TOKEN", "WANDB_API_KEY", "API_KEY", "TOKEN"}:
                raise ValueError("Credentials belong in the inherited environment, never job configuration")
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
    frozen = dict(lr=1e-4, weight_decay=1e-4, effective_batch=16, checkpoint_steps=10, run_xai=False, run_info=False)
    frozen.update(
        dict(epochs=1, patience=2, seeds=[42], store_res=8, res3d=8, res2d=8)
        if cfg["smoke"]
        else dict(epochs=20, patience=21, seeds=[42, 43, 44], store_res=200, res3d=96, res2d=224)
    )
    if any(cfg[k] != v for k, v in frozen.items()):
        raise ValueError("Frozen study changed; this adapter does not authorize another protocol")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cfg["group"]):
        raise ValueError("Use a simple group name")
    cap = cfg["max_runs_per_session"]
    if cap is not None and (not isinstance(cap, int) or isinstance(cap, bool) or cap < 1):
        raise ValueError("max_runs_per_session must be positive or None")
    if cfg["gpu_mode"] not in ("auto", "single"):
        raise ValueError("gpu_mode must be auto or single; DDP is not supported")
    for key in ("session_hours", "grace_seconds", "output_limit_gib", "job_reserve_gib", "fp16_init_scale"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not 0 < cfg["vram_fraction"] <= 0.8 or cfg["num_workers"] < 0:
        raise ValueError("Keep at least 20% VRAM headroom and nonnegative workers")
    if cfg["session_hours"] * 3600 <= cfg["grace_seconds"]:
        raise ValueError("Session budget must exceed checkpoint grace")
    if not cfg["smoke"]:
        if not Path(cfg["output_root"]).resolve().is_relative_to("/kaggle/working"):
            raise ValueError("Real outputs must live under /kaggle/working")
        if not Path(cfg["temp_root"]).resolve().is_relative_to("/kaggle/temp"):
            raise ValueError("Real downloads/caches must live under /kaggle/temp")
    jobs(cfg)
    if not isinstance(cfg["allow_initialized_recovery"], bool):
        raise ValueError("allow_initialized_recovery must be boolean")
    valid_tags = {job[2] for job in jobs({**cfg, "run_target": ""})}
    if set(cfg["artifact_refs"]) - valid_tags - {"_summary"}:
        raise ValueError("Unknown artifact recovery tag")


def load_credentials(*, smoke=False):
    ft.load_env_file()
    if smoke:
        return
    for key in ("HF_TOKEN", "WANDB_API_KEY"):
        if not os.environ.get(key):
            try:
                from kaggle_secrets import UserSecretsClient

                os.environ[key] = UserSecretsClient().get_secret(key)
            except Exception as exc:
                raise RuntimeError(f"Enable Kaggle Internet and attach Kaggle Secret {key}") from exc
        if not os.environ.get(key):
            raise RuntimeError(f"Missing {key}")
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"


def configure_runtime(cfg):
    root = Path(cfg["temp_root"])
    root.mkdir(parents=True, exist_ok=True)
    Path(cfg["output_root"]).mkdir(parents=True, exist_ok=True)
    for key, relative in {
        "HF_HOME": "hf",
        "HF_HUB_CACHE": "hf/hub",
        "HF_XET_CACHE": "hf/xet",
        "TORCH_HOME": "torch",
        "WANDB_CACHE_DIR": "wandb-cache",
        "WANDB_DATA_DIR": "wandb-staging",
    }.items():
        os.environ[key] = str(root / relative)
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"


def jobs(cfg):
    plan = [(spec, seed, f"{spec['code']}_s{seed}") for spec in SPECS for seed in cfg["seeds"]]
    if cfg["run_target"]:
        plan = [job for job in plan if job[2] == cfg["run_target"]]
        if not plan:
            raise ValueError("RUN_TARGET must name one study job, e.g. B4_s42")
    return plan


def detect_gpus(mode="auto", *, listing=None):
    if listing is None:
        listing = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"], text=True
        )
    devices = []
    for row in csv.reader(listing.strip().splitlines()):
        physical, name, memory = (value.strip() for value in row)
        if "P100" not in name and "T4" not in name:
            raise RuntimeError(f"Unvalidated accelerator: {name}; require Kaggle P100 or T4")
        devices.append(dict(physical=physical, name=name, memory_mib=int(memory)))
    if not devices:
        raise RuntimeError("No GPU detected; real training requires P100/T4")
    if mode == "single" or len(devices) == 1:
        return devices[:1]
    if len(devices) == 2 and all("T4" in device["name"] for device in devices):
        return devices
    raise RuntimeError("Auto accepts one P100/T4 or two independent T4s, not pooled VRAM")


class KaggleArtifacts:
    def __init__(self, local, *, smoke=False):
        self.local, self.remote, self.smoke = Path(local), None, smoke
        self.local.mkdir(parents=True, exist_ok=True)
        self.run = None

    def save(self, value, name):
        if name.startswith("emergency_weights_"):
            name = "emergency_weights.pt"
        path = self.local / name
        ft.atomic_save(value, path)
        return path

    def fetch(self, name):
        return self.local / name

    def sync(self, path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    def publish(self, stage):
        if self.smoke:
            return
        import wandb

        (self.local / "archived.json").unlink(missing_ok=True)
        if self.run is None or self.run.offline or self.run.disabled:
            raise RuntimeError("Remote checkpoint requires active online W&B")
        artifact = wandb.Artifact(f"controls-{self.run.id}", type="checkpoint", metadata={"stage": stage})
        write_json(self.local / "recovery.json", recovery_descriptor(self.local, stage))
        names = sorted(RECOVERY_FILES | {"recovery.json"})
        for name in names:
            if (self.local / name).is_file():
                artifact.add_file(str(self.local / name), name=name)
        try:
            receipt = self.run.log_artifact(artifact, aliases=["latest"])
            receipt.wait()
        except Exception as exc:
            raise RuntimeError("W&B checkpoint FAILED; local checkpoint retained; stop this session") from exc
        write_json(self.local / "archived.json", {"stage": stage, "artifact": receipt.qualified_name})
        print(f"[checkpoint] W&B acknowledged {stage}", flush=True)


class KaggleTrainer(ct.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not kwargs.get("resume", False):
            self.scaler = torch.amp.GradScaler(
                "cuda", enabled=self.device.type == "cuda", init_scale=self.config["fp16_init_scale"]
            )

    def commit(self, *, best=False):
        super().commit(best=False)
        if best and self.best_state is not None:
            self.artifacts.save(self.best_state, "best_weights.pt")


def settling_updates(update, *, max_skips=8, required=2):
    successes = skips = 0
    while successes < required:
        if update():
            successes += 1
        else:
            skips += 1
            if skips > max_skips:
                raise FloatingPointError("FP16 probe did not settle within eight skipped updates")
    return skips


def probe_batch(model, dataset, config, *, fraction=0.8, initial_scale=1024):
    device = next(model.parameters()).device
    if device.type == "cpu":
        return {"batch_size": 2, "fp16_init_scale": initial_scale, "peak_gib": 0, "status": "synthetic-cpu"}
    original, rng = ft.cpu_state(model), ft.rng_state()
    modes = {name: module.training for name, module in model.named_modules()}
    budget = torch.cuda.get_device_properties(device).total_memory * fraction
    trials = []
    try:
        for batch in (16, 8, 4, 2, 1):
            optimizer = scaler = None
            try:
                model.load_state_dict(original)
                ft.restore_rng(rng)
                model.train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
                scaler = torch.amp.GradScaler("cuda", init_scale=initial_scale)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                weights = torch.tensor(config["class_weights"], device=device, dtype=torch.float32)

                def update():
                    optimizer.zero_grad(set_to_none=True)
                    denominator = 0.0
                    for start in range(0, 16, batch):
                        samples = [dataset[(start + i) % len(dataset)] for i in range(batch)]
                        x, views, labels = (torch.stack(items).to(device) for items in zip(*samples))
                        with torch.autocast("cuda", dtype=torch.float16):
                            loss = torch.nn.functional.cross_entropy(
                                model(x, views), labels, weight=weights, reduction="sum"
                            )
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite probe loss")
                        scaler.scale(loss).backward()
                        denominator += weights[labels].sum().item()
                    scaler.unscale_(optimizer)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(denominator)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
                    scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    return scaler.get_scale() >= scale

                skips = settling_updates(update)
                torch.cuda.synchronize(device)
                peak = torch.cuda.max_memory_reserved(device)
                trials.append({"batch_size": batch, "peak_gib": peak / 1024**3, "skips": skips})
                if peak <= budget:
                    return {
                        **trials[-1],
                        "fp16_init_scale": scaler.get_scale(),
                        "trials": trials,
                        "successful_updates": 2,
                    }
            except torch.cuda.OutOfMemoryError:
                trials.append({"batch_size": batch, "oom": True})
            finally:
                model.zero_grad(set_to_none=True)
                optimizer = scaler = None
                torch.cuda.empty_cache()
        raise RuntimeError(f"No safe microbatch including batch 1: {trials}")
    finally:
        model.load_state_dict(original)
        for name, module in model.named_modules():
            module.training = modes[name]
        ft.restore_rng(rng)


class SmokeModel(ft.SmokeModel):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec

    def forward(self, x, views):
        if not self.spec["use_3d"]:
            x = torch.zeros_like(x)
        masked = torch.zeros_like(views)
        masked[:, self.spec["view_indices"]] = views[:, self.spec["view_indices"]]
        return super().forward(x, masked)


def make_model(spec, *, smoke, resume=False):
    return (
        SmokeModel(spec)
        if smoke
        else cm.ControlsModel(
            **{key: value for key, value in spec.items() if key != "code"},
            D=256,
            enc2d="maxvit_tiny_rw_224",
            enc3d_features=(32, 64, 128, 192),
            enc2d_pretrained=not resume,
        )
    )


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def study_contract(cfg):
    code_hashes = {}
    for name in (
        "controls_model",
        "controls_training",
        "final_model",
        "final_training",
        "controls_kaggle_data",
        "controls_kaggle",
    ):
        code_hashes[name] = sha256(Path(__file__).with_name(name + ".py"))
    return {
        **{
            key: cfg[key]
            for key in (
                "epochs",
                "patience",
                "lr",
                "weight_decay",
                "checkpoint_steps",
                "store_res",
                "res3d",
                "res2d",
                "group",
                "effective_batch",
            )
        },
        "amp_dtype": "float16",
        "synthetic": cfg["smoke"],
        "backend": {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "timm": "synthetic" if cfg["smoke"] else importlib.metadata.version("timm"),
        },
        "code_hashes": code_hashes,
        "step_metrics": True,
        "telemetry": not cfg["smoke"],
        "pin_memory": not cfg["smoke"],
        "non_blocking": not cfg["smoke"],
        "fused_adamw": False,
    }


def training_config(cfg, prepared, spec, seed, dataset):
    weights = len(dataset) / (2 * np.bincount(dataset.labels, minlength=2).astype(float))
    kind = "bilateral" if spec["code"] == "B4" else "raw"
    return {
        **study_contract(cfg),
        "seed": seed,
        "spec": spec,
        "class_weights": weights.tolist(),
        "data": {s: e["identity"] for s, e in prepared[kind]["entries"].items()},
        "dataset": kind,
    }


def validate_identity(cfg, tag, identity, prepared=None):
    if not isinstance(identity, dict) or not isinstance(identity.get("config"), dict):
        raise ValueError("Malformed run identity")
    spec, seed, _ = jobs({**cfg, "run_target": tag})[0]
    config = identity.get("config", {})
    kind = "bilateral" if spec["code"] == "B4" else "raw"
    expected = {**study_contract(cfg), "spec": spec, "seed": seed, "dataset": kind}
    if (
        not isinstance(identity.get("id"), str)
        or not identity["id"]
        or identity.get("warm_start", "")
        or any(config.get(k) != v for k, v in expected.items())
    ):
        raise ValueError(f"Run identity/study/code mismatch: {tag}")
    batch, accum = config.get("batch_size"), config.get("grad_accum")
    if (
        not isinstance(batch, int)
        or isinstance(batch, bool)
        or batch not in (1, 2, 4, 8, 16)
        or not isinstance(accum, int)
        or isinstance(accum, bool)
        or batch * accum != 16
        or not math.isfinite(config.get("fp16_init_scale", 0))
        or config.get("fp16_init_scale", 0) <= 0
    ):
        raise ValueError(f"Invalid saved batch/scaler identity: {tag}")
    weights = config.get("class_weights", [])
    if len(weights) != 2 or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("Invalid saved class weights")
    data = config.get("data", {})
    if set(data) != set(kd.SPLITS):
        raise ValueError("Missing saved split identity")
    for split, entry in data.items():
        source = entry.get("identity", {})
        recipe = kd.projection_signature(cfg["res2d"])
        recipe.update(
            repo=kd.cd.DEFAULT_REPO if kind == "bilateral" else kd.RAW_REPO,
            revision=kd.cd.DEFAULT_REVISION if kind == "bilateral" else kd.RAW_REVISION,
        )
        hashes = [entry.get("views"), entry.get("dz"), source.get("volumes"), source.get("labels")]
        if kind == "bilateral" and not cfg["smoke"]:
            hashes.append(source.get("manifest"))
        shape = source.get("shape", [])
        if (
            any(source.get(k) != v for k, v in recipe.items())
            or any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes)
            or len(shape) not in (4, 5)
            or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in shape)
            or len(shape) == 5
            and shape[1] != 1
            or shape[-3:] != [cfg["store_res"]] * 3
            or shape[0] < 1
        ):
            raise ValueError(f"Saved data/projection identity mismatch: {tag}/{split}")
    if prepared is not None and kind in prepared:
        entries = prepared[kind]["entries"]
        if data != {s: e["identity"] for s, e in entries.items()}:
            raise ValueError(f"Prepared data identity differs: {tag}")
        labels = np.load(entries["Training"]["labels"], mmap_mode="r")
        actual_weights = len(labels) / (2 * np.bincount(labels, minlength=2).astype(float))
        if weights != actual_weights.tolist():
            raise ValueError("Prepared class weights differ")
    return config


def checkpoint_header(state):
    return {
        "run_id": state["run_id"],
        "config_digest": digest(state["config"]),
        "epoch": state["epoch"],
        "cursor": state["cursor"],
        "step": state["step"],
    }


def validate_header(identity, header, *, completed=False):
    config = identity["config"]
    if (
        header.get("run_id") != identity["id"]
        or header.get("config_digest") != digest(config)
        or any(
            not isinstance(header.get(k), int) or isinstance(header[k], bool) or header[k] < 0
            for k in ("epoch", "cursor", "step")
        )
        or header["epoch"] > config["epochs"]
        or completed
        and (header["epoch"] != config["epochs"] or header["cursor"] != 0 or header["step"] < 1)
    ):
        raise ValueError("Checkpoint/completion run ID, config or progress mismatch")


def validate_completion(cfg, tag, record, prepared=None):
    if not isinstance(record, dict):
        raise ValueError("Malformed completion provenance")
    if (
        record.get("schema") != 2
        or record.get("tag") != tag
        or record.get("protocol_digest") != digest(study_contract(cfg))
    ):
        raise ValueError(f"Completion provenance mismatch: {tag}")
    if record.get("record_digest") != digest({k: v for k, v in record.items() if k != "record_digest"}):
        raise ValueError("Completion record digest mismatch")
    identity = record.get("identity", {})
    validate_identity(cfg, tag, identity, prepared)
    if record.get("identity_digest") != digest(identity):
        raise ValueError("Completion identity digest mismatch")
    validate_header(identity, record.get("checkpoint", {}), completed=True)
    required = {"best_weights.pt", "logits.npz", "logits_receipt.json", "report.json"}
    if not required <= record.get("files", {}).keys() or not isinstance(record.get("metrics"), dict):
        raise ValueError("Incomplete completion report provenance")
    if not record["files"].keys() <= RECOVERY_FILES or any(
        not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in record["files"].values()
    ):
        raise ValueError("Invalid completion file hashes")
    return record


def validate_run_dir(cfg, tag, local, prepared=None):
    local = Path(local)
    if not (local / "run_identity.json").is_file() or not (local / "run_identity.pt").is_file():
        raise ValueError(f"Incomplete recovery identity: {tag}")
    identity = json.loads((local / "run_identity.json").read_text())
    validate_identity(cfg, tag, identity, prepared)
    if torch.load(local / "run_identity.pt", weights_only=False, map_location="cpu") != identity:
        raise ValueError("JSON/checkpoint identity mismatch")
    completed = None
    if (local / "last.pt").is_file():
        state = torch.load(local / "last.pt", weights_only=False, map_location="cpu", mmap=True)
        if state.get("format") != 1 or state.get("config") != identity["config"]:
            raise ValueError("Invalid checkpoint format/config")
        header = checkpoint_header(state)
        validate_header(identity, header)
        if not {"model", "optimizer", "scheduler", "scaler", "rng"} <= state.keys():
            raise ValueError("Incomplete resumable state")
        if (local / "completed.json").exists():
            completed = validate_completion(cfg, tag, json.loads((local / "completed.json").read_text()), prepared)
            if completed["identity"] != identity or completed["checkpoint"] != header:
                raise ValueError("Completed marker differs from checkpoint")
            for name, expected in completed["files"].items():
                if name not in RECOVERY_FILES or not (local / name).is_file() or sha256(local / name) != expected:
                    raise ValueError(f"Completed artifact checksum mismatch: {name}")
            if json.loads((local / "report.json").read_text())["test"] != completed["metrics"]:
                raise ValueError("Completion metrics differ from report")
    else:
        marker = local / "initial_stage.json"
        if (
            not cfg["allow_initialized_recovery"]
            or not marker.is_file()
            or json.loads(marker.read_text()) != {"stage": "identity", "tag": tag, "identity_digest": digest(identity)}
            or (local / "completed.json").exists()
        ):
            raise ValueError("Missing last.pt; initialized recovery requires explicit authorization and stage metadata")
    return identity, completed


def recovery_descriptor(local, stage):
    identity = json.loads((local / "run_identity.json").read_text())
    files = {p.name: sha256(p) for p in local.iterdir() if p.name in RECOVERY_FILES and p.is_file()}
    header = None
    if (local / "last.pt").exists():
        header = checkpoint_header(torch.load(local / "last.pt", weights_only=False, map_location="cpu", mmap=True))
        validate_header(identity, header)
    return {"schema": 2, "stage": stage, "identity_digest": digest(identity), "files": files, "checkpoint": header}


def inspect_cloud(cfg, tag, destination, *, full=False):
    import wandb

    artifact = wandb.Api().artifact(cfg["artifact_refs"][tag], type="checkpoint")
    entries = artifact.manifest.entries

    def download(name):
        entry = entries.get(name)
        if entry is None or entry.size is None or entry.size <= 0:
            raise ValueError(f"Missing/empty requested artifact file: {tag}/{name}")
        if name.endswith(".json") and entry.size > 1024**2:
            raise ValueError("Oversized recovery metadata")
        path = Path(artifact.get_path(name).download(root=str(destination)))
        if path.is_symlink() or path.resolve().parent != destination.resolve() or path.name != name:
            raise ValueError("Artifact download escaped owned staging directory")
        return path

    descriptor = json.loads(download("recovery.json").read_text())
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("files"), dict):
        raise ValueError("Malformed recovery artifact manifest")
    files = descriptor.get("files", {})
    required = {"run_identity.json", "run_identity.pt"}
    initialized = descriptor.get("stage") == "identity" and descriptor.get("checkpoint") is None
    if initialized and "last.pt" in files:
        raise ValueError("Contradictory initialized/checkpoint recovery metadata")
    required.add("initial_stage.json" if initialized else "last.pt")
    if (
        descriptor.get("schema") != 2
        or descriptor.get("stage") != artifact.metadata.get("stage")
        or not required <= files.keys()
        or not files.keys() <= RECOVERY_FILES
        or any(name not in entries or not entries[name].size for name in files)
    ):
        raise ValueError("Incomplete requested recovery artifact")
    if initialized and not cfg["allow_initialized_recovery"]:
        raise ValueError("Initialized recovery is not authorized; no last.pt")
    selected = {"run_identity.json"} | ({"completed.json"} if "completed.json" in files else set())
    if initialized:
        selected.add("initial_stage.json")
    for name in selected:
        path = download(name)
        if sha256(path) != files[name]:
            raise ValueError(f"Recovery artifact checksum mismatch: {name}")
    identity = json.loads((destination / "run_identity.json").read_text())
    validate_identity(cfg, tag, identity)
    if digest(identity) != descriptor.get("identity_digest"):
        raise ValueError("Recovery artifact identity digest mismatch")
    if initialized and json.loads((destination / "initial_stage.json").read_text()) != {
        "stage": "identity",
        "tag": tag,
        "identity_digest": digest(identity),
    }:
        raise ValueError("Invalid initialized recovery stage metadata")
    if not initialized:
        validate_header(identity, descriptor.get("checkpoint", {}))
    completed = None
    if "completed.json" in files:
        completed = validate_completion(cfg, tag, json.loads((destination / "completed.json").read_text()))
        if completed["identity"] != identity or completed["checkpoint"] != descriptor["checkpoint"]:
            raise ValueError("Cloud completion/checkpoint mismatch")
        if any(files.get(name) != value for name, value in completed["files"].items()):
            raise ValueError("Cloud completion file provenance mismatch")
    if full:
        require_disk(destination, sum(entries[name].size for name in files) + 1024**3)
        for name in sorted(set(files) - selected):
            if sha256(download(name)) != files[name]:
                raise ValueError(f"Recovery artifact checksum mismatch: {name}")
    return identity, completed


def hydrate(cfg, tag, prepared=None):
    validate_config(cfg)
    root = Path(cfg["output_root"]) / cfg["group"]
    with run_lock(root / f".{tag}.job.lock"):
        return _hydrate(cfg, tag, prepared)


def _hydrate(cfg, tag, prepared=None):
    local = Path(cfg["output_root"]) / cfg["group"] / tag
    local.parent.mkdir(parents=True, exist_ok=True)
    reference = cfg["artifact_refs"].get(tag)
    source = Path(cfg["resume_root"]) / tag if cfg["resume_root"] else None
    selection = {"reference": reference, "attached": str(source.resolve()) if source and not reference else None}
    requested = bool(reference or source)
    if source and (
        source.resolve().is_relative_to(local.resolve()) or local.resolve().is_relative_to(source.resolve())
    ):
        raise ValueError("Attached recovery source overlaps destination")
    receipt = local / "hydrated.json"
    if requested and receipt.is_file() and json.loads(receipt.read_text())["selection"] == selection:
        try:
            identity, _ = validate_run_dir(cfg, tag, local, prepared)
            if json.loads(receipt.read_text())["identity_digest"] == digest(identity):
                return local
        except (ValueError, RuntimeError, EOFError, pickle.UnpicklingError):
            pass
    if not requested:
        if local.exists() and any(p.name in RECOVERY_FILES for p in local.iterdir()):
            validate_run_dir(cfg, tag, local, prepared)
        return local
    if not reference and not source.is_dir():
        raise ValueError(f"Requested attached recovery source missing or aliases destination: {source}")
    stage = Path(tempfile.mkdtemp(prefix=f".{tag}-hydrate-", dir=local.parent))
    owned_download = None
    try:
        if reference:
            temp_root = Path(cfg["temp_root"])
            temp_root.mkdir(parents=True, exist_ok=True)
            owned_download = Path(tempfile.mkdtemp(prefix=f"restore-{tag}-", dir=temp_root))
            inspect_cloud(cfg, tag, owned_download, full=True)
            source = owned_download
        needed = sum(p.stat().st_size for p in source.iterdir() if p.name in RECOVERY_FILES and p.is_file())
        require_disk(stage, needed + 1024**3)
        for name in RECOVERY_FILES:
            path = source / name
            if path.is_file():
                expected = sha256(path)
                shutil.copyfile(path, stage / name)
                if sha256(stage / name) != expected:
                    raise ValueError("Recovery source changed while staging")
        identity, _ = validate_run_dir(cfg, tag, stage, prepared)
        if (local / "run_identity.pt").exists():
            try:
                previous = torch.load(local / "run_identity.pt", weights_only=False, map_location="cpu")
            except (ValueError, RuntimeError, EOFError, pickle.UnpicklingError):
                previous = (
                    json.loads((local / "run_identity.json").read_text())
                    if (local / "run_identity.json").is_file()
                    else None
                )
            if previous is not None and previous != identity:
                raise ValueError("Requested recovery conflicts with local run identity")
            if (local / "last.pt").exists():
                try:
                    validate_run_dir(cfg, tag, local, prepared)
                    current = torch.load(local / "last.pt", weights_only=False, map_location="cpu", mmap=True)
                    restored = (
                        torch.load(stage / "last.pt", weights_only=False, map_location="cpu", mmap=True)
                        if (stage / "last.pt").exists()
                        else {"epoch": 0, "cursor": 0, "step": 0}
                    )
                    keep_local = (current["epoch"], current["cursor"], current["step"]) >= (
                        restored["epoch"],
                        restored["cursor"],
                        restored["step"],
                    )
                    del current, restored
                    if keep_local:
                        write_json(receipt, {"selection": selection, "identity_digest": digest(identity)})
                        return local
                except (ValueError, RuntimeError, EOFError, pickle.UnpicklingError):
                    pass
        write_json(
            stage / "hydrated.json",
            {
                "selection": selection,
                "identity_digest": digest(identity),
                "files": {p.name: sha256(p) for p in stage.iterdir() if p.is_file()},
            },
        )
        backup = local.with_name(f".{tag}-previous-{uuid.uuid4().hex}")
        if local.exists():
            os.replace(local, backup)
        try:
            os.replace(stage, local)
        except BaseException:
            if backup.exists():
                os.replace(backup, local)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return local
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if owned_download is not None:
            shutil.rmtree(owned_download)


def evaluate_callback(dataset, batch_size, num_workers=0, split="val"):
    def evaluate(model):
        probs, labels, logits = ft.predict(model, dataset, batch_size, num_workers=num_workers, amp_dtype="float16")
        metrics = fm.full_metrics(probs, labels)
        metrics["loss"] = torch.nn.functional.cross_entropy(torch.tensor(logits), torch.tensor(labels)).item()
        print(f"[{split}] {metrics}", flush=True)
        return metrics

    return evaluate


def report_best(model, datasets, artifacts, *, batch_size, smoke, num_workers=0, report_root=None):
    train, val, test = datasets
    train.train = False
    saved = artifacts.local / "logits.npz"
    predictions = {}
    names = {id(train): "train", id(val): "val", id(test): "test"}
    predict = ft.predict
    identity = json.loads((artifacts.local / "run_identity.json").read_text())
    provenance = {
        "schema": 2,
        "identity_digest": digest(identity),
        "best_weights": sha256(artifacts.local / "best_weights.pt"),
        "data": identity["config"]["data"],
    }
    receipt_path = artifacts.local / "logits_receipt.json"
    if saved.exists():
        if not receipt_path.is_file():
            raise ValueError("Saved logits have no provenance receipt")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("provenance") != provenance or receipt.get("sha256") != sha256(saved):
            raise ValueError("Stale/corrupt logits or changed best weights/data identity")
        with np.load(saved) as arrays:
            predictions = {key: arrays[key] for key in arrays.files}
        required = {f"{split}_{kind}" for split in ("train", "val", "test") for kind in ("logits", "labels")}
        if set(predictions) != required:
            raise ValueError("Incomplete saved logits")
        for split, dataset, storage_split in zip(("train", "val", "test"), datasets, kd.SPLITS):
            labels, logits = predictions[f"{split}_labels"], predictions[f"{split}_logits"]
            expected_n = provenance["data"][storage_split]["identity"]["shape"][0]
            if (
                labels.shape != (expected_n,)
                or logits.shape != (expected_n, 2)
                or not np.isfinite(logits).all()
                or digest(labels.tolist()) != receipt["labels"][split]
                or hasattr(dataset, "labels")
                and not np.array_equal(labels, dataset.labels)
            ):
                raise ValueError("Saved logits split labels/shape mismatch")
    else:
        ft.load_weights(model, artifacts.local / "best_weights.pt")

    def capture(current, dataset, *args, **kwargs):
        name = names[id(dataset)]
        if f"{name}_logits" not in predictions:
            _, labels, logits = predict(current, dataset, *args, **kwargs)
            predictions[f"{name}_logits"], predictions[f"{name}_labels"] = logits, labels
        logits, labels = predictions[f"{name}_logits"], predictions[f"{name}_labels"]
        return torch.tensor(logits).softmax(1)[:, 1].numpy(), labels, logits

    with patch.object(ft, "predict", capture):
        result, _, _ = ct.calibrated_report(
            model, val, test, batch_size, train=train, smoke=smoke, num_workers=num_workers, amp_dtype="float16"
        )
    if not saved.exists():
        temporary = saved.with_name(saved.name + ".tmp-" + uuid.uuid4().hex)
        try:
            with temporary.open("xb") as stream:
                np.savez(stream, **predictions)
                stream.flush()
                os.fsync(stream.fileno())
            write_json(
                receipt_path,
                {
                    "provenance": provenance,
                    "sha256": sha256(temporary),
                    "labels": {s: digest(predictions[f"{s}_labels"].tolist()) for s in ("train", "val", "test")},
                },
            )
            os.replace(temporary, saved)
        finally:
            temporary.unlink(missing_ok=True)
    output = Path(report_root) if report_root is not None else artifacts.local
    write_json(output / "report.json", finite_json(result))
    with (output / "report.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", *result["test"].keys()])
        writer.writeheader()
        for split in ("train", "val", "test"):
            writer.writerow({"split": split, **result[split]})
    if artifacts.run is not None:
        ft.log_report(artifacts.run, result)
    return result


def rerender_reports(cfg):
    validate_config(cfg)
    load_credentials(smoke=cfg["smoke"])
    root = Path(cfg["output_root"]) / cfg["group"]
    for _, _, tag in jobs(cfg):
        local = root / tag
        if (local / "logits.npz").exists():
            identity, _ = validate_run_dir(cfg, tag, local)
            datasets = [SimpleNamespace(train=False) for _ in range(3)]
            linked = KaggleArtifacts(root / "_reports" / tag, smoke=cfg["smoke"])
            run = ft.init_wandb(
                f"{cfg['group']}-{tag}-report-replay",
                {
                    "scope": "report-replay",
                    "parent_run_id": identity["id"],
                    "identity_digest": digest(identity),
                },
                linked,
                resume=(linked.local / "run_identity.pt").exists(),
                smoke=cfg["smoke"],
            )
            artifacts = KaggleArtifacts(local, smoke=cfg["smoke"])
            artifacts.run = run
            exit_code = 1
            try:
                report_best(None, datasets, artifacts, batch_size=1, smoke=cfg["smoke"], report_root=linked.local)
                if not cfg["smoke"]:
                    import wandb

                    report = wandb.Artifact(
                        f"controls-report-{identity['id']}", type="report", metadata={"parent_run_id": identity["id"]}
                    )
                    for name in ("report.json", "report.csv"):
                        report.add_file(str(linked.local / name), name=name)
                    report.add_file(str(local / "logits_receipt.json"), name="logits_receipt.json")
                    run.log_artifact(report, aliases=["latest"]).wait()
                exit_code = 0
            finally:
                run.finish(exit_code=exit_code)
            print(f"[report] {tag}: rerendered from saved logits, no model or GPU")


def completion_record(cfg, tag, local):
    identity = json.loads((local / "run_identity.json").read_text())
    state = torch.load(local / "last.pt", weights_only=False, map_location="cpu", mmap=True)
    record = {
        "schema": 2,
        "tag": tag,
        "protocol_digest": digest(study_contract(cfg)),
        "identity": identity,
        "identity_digest": digest(identity),
        "checkpoint": checkpoint_header(state),
        "files": {
            name: sha256(local / name)
            for name in ("best_weights.pt", "logits.npz", "logits_receipt.json", "report.json")
        },
        "metrics": json.loads((local / "report.json").read_text())["test"],
    }
    record["record_digest"] = digest(record)
    return validate_completion(cfg, tag, record)


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def check_cuda_runtime(*, smoke, run):
    metadata = {"torch": str(torch.__version__), "cuda": torch.version.cuda, "status": "synthetic-cpu-skip"}
    try:
        if smoke:
            return metadata
        metadata["status"] = "checking"
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the assigned worker")
        metadata["visible_devices"] = torch.cuda.device_count()
        if metadata["visible_devices"] != 1:
            raise RuntimeError("Worker must see exactly one assigned GPU")
        metadata.update(
            device=torch.cuda.get_device_name(0),
            capability=list(torch.cuda.get_device_capability(0)),
            supported_arches=torch.cuda.get_arch_list(),
            cudnn=torch.backends.cudnn.version(),
        )
        expected = next((cap for name, cap in (("P100", [6, 0]), ("T4", [7, 5])) if name in metadata["device"]), None)
        if expected is None or metadata["capability"] != expected:
            raise RuntimeError("Assigned device must be a P100 (sm_60) or T4 (sm_75) with matching capability")
        tiny = torch.ones(4, dtype=torch.float32, device="cuda:0")
        try:
            tiny.add_(1)
            torch.cuda.synchronize("cuda:0")
        finally:
            del tiny
        metadata["status"] = "passed"
        return metadata
    except (RuntimeError, OSError, AssertionError) as exc:
        metadata.update(status="failed", error=str(exc))
        raise RuntimeError(
            f"Assigned-worker CUDA compatibility check failed: {exc}. Runtime: {metadata}. "
            "Select a compatible Kaggle image/PyTorch CUDA build for the assigned GPU. "
            "P100 requires Pascal sm_60 support, which newer CUDA 13 builds may omit; "
            "alternatively select T4 with a compatible image. Leave Kaggle's installed Torch untouched; "
            "no automatic reinstall was attempted. Model construction, pretrained downloads and probing were not started."
        ) from exc
    finally:
        run.summary.update({"runtime/device_check": metadata})
        run.log({"runtime/phase": "device-check", "runtime/device_check/status": metadata["status"]})


def worker(cfg, prepared, tag, *, deadline, stop_path):
    validate_config(cfg)
    configure_runtime(cfg)
    load_credentials(smoke=cfg["smoke"])
    spec, seed, _ = next(job for job in jobs({**cfg, "run_target": tag}) if job[2] == tag)
    local = hydrate(cfg, tag, prepared)
    with run_lock(local.parent / f".{tag}.job.lock"):
        saved = None
        if any((local / name).exists() for name in ("run_identity.pt", "last.pt", "completed.json")):
            saved, completed = validate_run_dir(cfg, tag, local, prepared)
            if completed:
                return
        datasets = make_datasets(prepared, spec, seed, smoke=cfg["smoke"])
        config = training_config(cfg, prepared, spec, seed, datasets[0])
        artifacts = KaggleArtifacts(local, smoke=cfg["smoke"])
        resume = (local / "last.pt").exists()
        if saved:
            old = saved["config"]
            if any(old.get(key) != value for key, value in config.items()):
                raise ValueError("Saved study/data/code identity differs; refusing resume")
            config = old
        runtime_artifacts = KaggleArtifacts(local.parent / "_runtime" / tag, smoke=cfg["smoke"])
        runtime = ft.init_wandb(
            f"{cfg['group']}-{tag}-runtime",
            {
                "group": cfg["group"],
                "tag": tag,
                "scope": "runtime-probe",
                "protocol_digest": digest(study_contract(cfg)),
            },
            runtime_artifacts,
            resume=(runtime_artifacts.local / "run_identity.pt").exists(),
            smoke=cfg["smoke"],
        )
        runtime_exit = 1
        try:
            check_cuda_runtime(smoke=cfg["smoke"], run=runtime)
            if cfg["smoke"]:
                torch.set_num_threads(1)
            else:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            device = torch.device("cpu" if cfg["smoke"] else "cuda:0")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            runtime.log({"runtime/phase": "model-build", "runtime/resume": resume})
            model = make_model(spec, smoke=cfg["smoke"], resume=resume).to(device)
            if not saved:
                probe_key = {
                    **config,
                    "device": "cpu" if cfg["smoke"] else torch.cuda.get_device_name(0),
                    "memory": 0 if cfg["smoke"] else torch.cuda.get_device_properties(0).total_memory,
                    "physical": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "fraction": cfg["vram_fraction"],
                    "initial_scale": cfg["fp16_init_scale"],
                }
                probe_key.pop("seed")
                probe_path = Path(cfg["temp_root"]) / "probes" / f"{digest(probe_key)}.json"
                runtime.log({"runtime/phase": "probe"})
                if probe_path.exists():
                    cached = json.loads(probe_path.read_text())
                    if cached.get("key") != probe_key:
                        raise ValueError("Probe cache identity mismatch")
                    probe = cached["result"]
                else:
                    probe = probe_batch(
                        model, datasets[0], config, fraction=cfg["vram_fraction"], initial_scale=cfg["fp16_init_scale"]
                    )
                    write_json(probe_path, {"key": probe_key, "result": probe})
                if (
                    probe.get("batch_size") not in (1, 2, 4, 8, 16)
                    or not math.isfinite(probe.get("fp16_init_scale", 0))
                    or probe["fp16_init_scale"] <= 0
                    or not cfg["smoke"]
                    and (
                        probe.get("successful_updates") != 2
                        or probe.get("peak_gib", math.inf) * 1024**3 > probe_key["memory"] * cfg["vram_fraction"]
                    )
                ):
                    raise ValueError("Invalid or unvalidated probe result")
                config.update(
                    batch_size=probe["batch_size"],
                    grad_accum=16 // probe["batch_size"],
                    fp16_init_scale=probe["fp16_init_scale"],
                    probe=probe,
                    probe_device=probe_key["device"],
                )
                runtime.log({"batch/size": probe["batch_size"], "batch/initial_scale": probe["fp16_init_scale"]})
            runtime_exit = 0
        finally:
            runtime.finish(exit_code=runtime_exit)
        run = ft.init_wandb(f"{cfg['group']}-{tag}", config, artifacts, resume=saved is not None, smoke=cfg["smoke"])
        artifacts.run = run
        exit_code = 1
        try:
            identity = torch.load(local / "run_identity.pt", weights_only=False, map_location="cpu")
            write_json(local / "run_identity.json", identity)
            if not resume:
                write_json(
                    local / "initial_stage.json", {"stage": "identity", "tag": tag, "identity_digest": digest(identity)}
                )
            trainer = KaggleTrainer(
                model, datasets[0], config, artifacts, run, resume=resume, num_workers=cfg["num_workers"]
            )
            if not resume:
                artifacts.publish("identity")
            run.log(
                {
                    "recovery/mode": "checkpoint" if resume else "initialized" if saved else "new",
                    "recovery/initialized_authorized": bool(saved and not resume and cfg["allow_initialized_recovery"]),
                }
            )
            run.log(
                {
                    "batch/size": config["batch_size"],
                    "batch/accum": config["grad_accum"],
                    "batch/peak_gib": config["probe"].get("peak_gib", 0),
                    "batch/fp16_init_scale": config["fp16_init_scale"],
                }
            )
            print(f"[batch] {tag}: microbatch={config['batch_size']} accum={config['grad_accum']} FP16", flush=True)
            if not cfg["smoke"]:
                torch.cuda.reset_peak_memory_stats(device)
            timer = time.monotonic()

            def boundary(current):
                if Path(stop_path).exists() or time.time() >= deadline:
                    current.stop_requested = True

            def epoch_hook(current):
                nonlocal timer
                if current.best_state is not None:
                    artifacts.save(current.best_state, "best_weights.pt")
                write_json(local / "history.json", finite_json(current.history))
                artifacts.publish(f"epoch-{current.epoch}")
                elapsed = time.monotonic() - timer
                peak = 0 if cfg["smoke"] else torch.cuda.max_memory_reserved(device) / 1024**3
                run.log({"train/epoch_seconds": elapsed, "train/epoch": current.epoch, "train/peak_reserved_gib": peak})
                print(f"[train] epoch {current.epoch} done in {elapsed:.1f}s", flush=True)
                timer = time.monotonic()
                if not cfg["smoke"]:
                    torch.cuda.reset_peak_memory_stats(device)
                boundary(current)

            if Path(stop_path).exists() or time.time() >= deadline:
                trainer.commit()
                completed = False
            else:
                completed = trainer.fit(
                    evaluate_callback(datasets[1], config["batch_size"], cfg["num_workers"], "val"),
                    test_evaluate=evaluate_callback(datasets[2], config["batch_size"], cfg["num_workers"], "test"),
                    boundary_hook=boundary,
                    epoch_hook=epoch_hook,
                )
            if not completed:
                request_stop(stop_path)
                artifacts.publish("stop")
                run.summary["status"] = "stopped"
            else:
                report_best(
                    model,
                    datasets,
                    artifacts,
                    batch_size=config["batch_size"],
                    smoke=cfg["smoke"],
                    num_workers=cfg["num_workers"],
                )
                write_json(local / "completed.json", completion_record(cfg, tag, local))
                try:
                    artifacts.publish("completed")
                except BaseException:
                    (local / "completed.json").unlink(missing_ok=True)
                    raise
                run.summary["status"] = "completed"
            exit_code = 0
        finally:
            run.finish(exit_code=exit_code)


def output_preflight(cfg, active=0):
    root = Path(cfg["output_root"])
    used = sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.exists() else 0
    reserve = cfg["job_reserve_gib"] * 1024**3 * (active + 1)
    if not cfg["smoke"] and used + reserve > cfg["output_limit_gib"] * 1024**3:
        raise OSError("Output quota would be exceeded; Save Version/download and start a fresh session")
    require_disk(root, 16 * 1024**2 if cfg["smoke"] else reserve)


def child_env(physical):
    return {**os.environ, "CUDA_VISIBLE_DEVICES": str(physical), "PYTHONUNBUFFERED": "1"}


def request_stop(stop_path):
    stop_path = Path(stop_path)
    if stop_path.exists():
        return
    temporary = stop_path.with_name(stop_path.name + ".stop-" + uuid.uuid4().hex)
    try:
        write_json(temporary, {"requested_at": time.time(), "controller_pid": os.getpid()})
        try:
            os.link(temporary, stop_path)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def wait_owned(stop_path, grace_seconds):
    request_stop(stop_path)
    until = time.monotonic() + grace_seconds
    while OWNED_CHILDREN and time.monotonic() < until:
        for pid, child in list(OWNED_CHILDREN.items()):
            if child.poll() is not None:
                child.wait()
                OWNED_CHILDREN.pop(pid)
        if OWNED_CHILDREN:
            time.sleep(0.2)
    if OWNED_CHILDREN:
        raise RuntimeError(
            f"Checkpoint grace expired; owned children STILL ALIVE {list(OWNED_CHILDREN)}. "
            "Stop marker remains; call wait_owned again. No unrelated process was signalled."
        )


def validate_summary(cfg, value, prepared=None):
    validate_config(cfg)
    if not isinstance(value, dict):
        raise ValueError("Malformed summary provenance")
    if (
        value.get("schema") != 2
        or value.get("group") != cfg["group"]
        or value.get("protocol_digest") != digest(study_contract(cfg))
        or not isinstance(value.get("runs"), dict)
    ):
        raise ValueError("Summary group/protocol provenance mismatch")
    valid = {job[2] for job in jobs({**cfg, "run_target": ""})}
    data_by_kind = {}
    for tag, record in value["runs"].items():
        if tag not in valid:
            raise ValueError(f"Unknown summary job: {tag}")
        validate_completion(cfg, tag, record, prepared)
        config = record["identity"]["config"]
        previous = data_by_kind.setdefault(config["dataset"], config["data"])
        if previous != config["data"]:
            raise ValueError("Summary contains inconsistent source/view identities across seeds/specs")
    return value


def restore_summary(cfg, prepared=None):
    root = Path(cfg["output_root"]) / cfg["group"]
    reference = cfg["artifact_refs"].get("_summary")
    if reference:
        import wandb

        artifact = wandb.Api().artifact(reference, type="report")
        name = "controls_96_summary.json"
        temp_root = Path(cfg["temp_root"])
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="summary-", dir=temp_root) as owned:
            entry = artifact.manifest.entries.get(name)
            if entry is None or not entry.size or entry.size > 10 * 1024**2:
                raise ValueError("Invalid group summary artifact")
            downloaded = Path(artifact.get_path(name).download(root=owned))
            if downloaded.is_symlink() or downloaded.resolve().parent != Path(owned).resolve():
                raise ValueError("Summary download escaped owned staging")
            value = validate_summary(cfg, json.loads(downloaded.read_text()), prepared)
            if (root / name).exists():
                old = validate_summary(cfg, json.loads((root / name).read_text()), prepared)
                value["runs"].update(old["runs"])
            validate_summary(cfg, value, prepared)
            write_json(root / name, value)
    value = aggregate(cfg, prepared)
    temp_root = Path(cfg["temp_root"])
    temp_root.mkdir(parents=True, exist_ok=True)
    for tag in cfg["artifact_refs"]:
        if tag == "_summary":
            continue
        with tempfile.TemporaryDirectory(prefix=f"metadata-{tag}-", dir=temp_root) as owned:
            _, completed = inspect_cloud(cfg, tag, Path(owned))
        if completed:
            if tag in value["runs"] and value["runs"][tag] != completed:
                raise ValueError("Conflicting cloud/local completion provenance")
            value["runs"][tag] = completed
    validate_summary(cfg, value, prepared)
    write_json(root / "controls_96_summary.json", value)
    return aggregate(cfg, prepared)


def session_jobs(cfg, prepared=None):
    validate_config(cfg)
    path = Path(cfg["temp_root"]) / f"session-{cfg['group']}.json"
    selection = {"run_target": cfg["run_target"], "cap": cfg["max_runs_per_session"]}
    if path.exists():
        state = json.loads(path.read_text())
        if state["selection"] != selection:
            raise ValueError("Session selection changed; start a fresh session/temp directory explicitly")
    else:
        prior = restore_summary(cfg, prepared)["runs"]
        if cfg["resume_root"] and cfg["run_target"] and cfg["run_target"] not in cfg["artifact_refs"]:
            validate_run_dir(cfg, cfg["run_target"], Path(cfg["resume_root"]) / cfg["run_target"], prepared)
        remaining = []
        for job in jobs(cfg):
            tag = job[2]
            if tag not in prior:
                remaining.append(tag)
        if cfg["max_runs_per_session"] is not None:
            remaining = remaining[: cfg["max_runs_per_session"]]
        state = {
            "selection": selection,
            "tags": remaining,
            "deadline": time.time() + cfg["session_hours"] * 3600 - cfg["grace_seconds"],
        }
        write_json(path, state)
    return [job for job in jobs(cfg) if job[2] in state["tags"]], state["deadline"]


def dispatch(cfg, prepared):
    validate_config(cfg)
    configure_runtime(cfg)
    if OWNED_CHILDREN:
        raise RuntimeError("Previous owned children still tracked; wait_owned before dispatch")
    root = Path(cfg["output_root"]) / cfg["group"]
    root.mkdir(parents=True, exist_ok=True)
    stop = root / "STOP.json"
    with run_lock(root / "controller.lock"):
        stop.unlink(missing_ok=True)
        selected, deadline = session_jobs(cfg, prepared)
        prior = restore_summary(cfg, prepared)["runs"]
        write_json(root / "plan.json", {"jobs": [job[2] for job in jobs(cfg)], "max_runs": cfg["max_runs_per_session"]})
        devices = [{"physical": "", "name": "synthetic-cpu"}] if cfg["smoke"] else detect_gpus(cfg["gpu_mode"])
        pending, dispatched, active = list(selected), [], {}
        previous = {}

        def interrupt(signum, frame):
            request_stop(stop)

        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, interrupt)
        stop_waited = False
        try:
            while pending or active:
                if time.time() >= deadline:
                    request_stop(stop)
                if stop.exists():
                    stop_waited = True
                    wait_owned(stop, cfg["grace_seconds"])
                    break
                for device in devices:
                    physical = device["physical"]
                    if physical in active or not pending:
                        continue
                    if cfg["max_runs_per_session"] is not None and len(dispatched) >= cfg["max_runs_per_session"]:
                        pending.clear()
                        break
                    _, _, tag = pending.pop(0)
                    if tag in prior:
                        continue
                    output_preflight(cfg, len(active))
                    job_cfg = dict(cfg)
                    if cfg["resume_root"] and not cfg["run_target"] and not (Path(cfg["resume_root"]) / tag).exists():
                        job_cfg["resume_root"] = ""
                    local = hydrate(job_cfg, tag, prepared)
                    if (local / "run_identity.pt").exists():
                        _, completed = validate_run_dir(job_cfg, tag, local, prepared)
                        if completed:
                            continue
                    payload = {
                        "cfg": job_cfg,
                        "prepared": prepared,
                        "tag": tag,
                        "deadline": deadline,
                        "stop_path": str(stop),
                    }
                    job_path = root / f"{tag}.job.json"
                    write_json(job_path, payload)
                    dispatched.append(tag)
                    print(f"[train] dispatch {tag} -> physical {physical or 'CPU'} ({device['name']})", flush=True)
                    if cfg["smoke"]:
                        worker(**payload)
                    else:
                        child = subprocess.Popen(
                            [sys.executable, "-m", "scripts.controls_kaggle", "--worker", str(job_path)],
                            env=child_env(physical),
                            cwd=Path(__file__).resolve().parents[1],
                        )
                        active[physical] = child
                        OWNED_CHILDREN[child.pid] = child
                for physical, child in list(active.items()):
                    status = child.poll()
                    if status is not None:
                        child.wait()
                        OWNED_CHILDREN.pop(child.pid)
                        active.pop(physical)
                        if status:
                            raise RuntimeError(f"Owned worker {child.pid} failed with exit {status}")
                if active:
                    time.sleep(0.2)
            return dispatched
        except BaseException:
            if not stop_waited:
                wait_owned(stop, cfg["grace_seconds"])
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def aggregate(cfg, prepared=None):
    validate_config(cfg)
    root = Path(cfg["output_root"]) / cfg["group"]
    records = {}
    roots = ([Path(cfg["resume_root"])] if cfg["resume_root"] else []) + [root]
    for source in roots:
        attached = bool(cfg["resume_root"]) and source == Path(cfg["resume_root"])
        if attached and not source.is_dir():
            raise ValueError("Missing requested resume_root")
        recognized = False
        summary = source / "controls_96_summary.json"
        if summary.exists():
            saved = validate_summary(cfg, json.loads(summary.read_text()), prepared)
            records.update(saved["runs"])
            recognized = True
        for _, _, tag in jobs({**cfg, "run_target": ""}):
            local = source / tag
            if attached and local.exists() or (local / "completed.json").exists():
                _, completed = validate_run_dir(cfg, tag, local, prepared)
                recognized = True
                if completed:
                    if tag in records and records[tag] != completed:
                        raise ValueError("Conflicting persisted completion provenance")
                    records[tag] = completed
        if attached and not recognized:
            raise ValueError("Empty/incomplete requested resume_root")
    value = {"schema": 2, "group": cfg["group"], "protocol_digest": digest(study_contract(cfg)), "runs": records}
    validate_summary(cfg, value, prepared)
    result = {}
    for spec in SPECS:
        tags = [f"{spec['code']}_s{seed}" for seed in cfg["seeds"] if f"{spec['code']}_s{seed}" in records]
        metrics = {}
        for key in sorted({key for tag in tags for key in records[tag]["metrics"]}):
            values = [records[tag]["metrics"][key] for tag in tags if records[tag]["metrics"].get(key) is not None]
            metrics[key] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "n": len(values),
            }
        result[spec["code"]] = {"n_seeds": len(tags), "partial": len(tags) < 3, "metrics": metrics}
    value["summary"] = result
    write_json(root / "controls_96_summary.json", value)
    with (root / "controls_96_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["spec", "n_seeds", "partial", "metric", "mean", "std", "n"])
        for code, item in result.items():
            for key, metric in item["metrics"].items():
                writer.writerow([code, item["n_seeds"], item["partial"], key, *metric.values()])
    return value


def persist_summary(cfg, value):
    validate_summary(cfg, value)
    load_credentials(smoke=cfg["smoke"])
    root = Path(cfg["output_root"]) / cfg["group"]
    artifacts = KaggleArtifacts(root / "aggregate", smoke=cfg["smoke"])
    run = ft.init_wandb(
        f"{cfg['group']}-summary",
        {"group": cfg["group"], "scope": "aggregate"},
        artifacts,
        resume=(artifacts.local / "run_identity.pt").exists(),
        smoke=cfg["smoke"],
    )
    try:
        import wandb

        rows = [[code, item["n_seeds"], item["partial"]] for code, item in value["summary"].items()]
        run.log({"report/seed_counts": wandb.Table(columns=["spec", "n_seeds", "partial"], data=rows)})
        metric_rows = [
            [code, item["n_seeds"], item["partial"], key, metric["mean"], metric["std"], metric["n"]]
            for code, item in value["summary"].items()
            for key, metric in item["metrics"].items()
        ]
        run.log(
            {
                "report/controls_summary": wandb.Table(
                    columns=["spec", "n_seeds", "partial", "metric", "mean", "std", "n"], data=metric_rows
                )
            }
        )
        run.summary.update({"completed_runs": len(value["runs"]), "study_runs": 21})
        if not cfg["smoke"]:
            artifact = wandb.Artifact(f"controls-summary-{cfg['group']}", type="report")
            for name in ("controls_96_summary.json", "controls_96_summary.csv"):
                artifact.add_file(str(root / name), name=name)
            run.log_artifact(artifact, aliases=["latest"]).wait()
    finally:
        run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args()
    worker(**json.loads(args.worker.read_text()))


if __name__ == "__main__":
    main()
