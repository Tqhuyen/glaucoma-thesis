"""Disk-backed stages for the final Bilateral five-epoch experiment."""

import gc
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts import final_data as fd
from scripts import final_model as fm
from scripts import final_reporting as fr
from scripts import final_training as ft

PARENT_SUFFIX = "raw_s42_recovered_20260910/raw_s42/best_weights.pt"
SPLITS = ("Training", "Validation", "Test")
ACTIVE_MODEL = ACTIVE_TRAINER = None
RAW_REPO = "tqhuyen/harvard-oct-glaucoma-200"
CPU_PARAMS = dict(
    sigma_color=0.10, sigma_spatial=4.0, bins=10000, mode="constant", cval=0, channel_axis=None, win_size=None
)
CPU_IMPLEMENTATION = "skimage.restoration.denoise_bilateral"
BUNDLE_FILES = (
    "final_model.py",
    "final_training.py",
    "final_data.py",
    "final_reporting.py",
    "final_execution.py",
    "resolution_study.py",
    "compare_denoise_methods.py",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


class CheckedArtifacts(ft.Artifacts):
    def sync(self, path):
        path = Path(path)
        if self.remote is None:
            if not self.smoke:
                raise RuntimeError("Drive is mandatory")
            return
        pending = path.with_name(path.name + ".sync-pending.pt")
        try:
            fd.sync_verified(path, self.remote / path.relative_to(self.local))
            pending.unlink(missing_ok=True)
        except BaseException as exc:
            ft.atomic_save({"source": str(path), "error": str(exc)}, pending)
            raise RuntimeError(f"Drive verification failed; local artifact retained: {path}") from exc

    def restore(self, name):
        path = self.local / name
        if not path.exists() and self.remote is not None:
            fd.sync_verified(self.remote / name, path)
        return path


def workspace(local, remote=None, *, smoke=False, context=None):
    artifacts = CheckedArtifacts(local, remote, smoke=smoke)
    if context is not None:
        path = artifacts.local / "context.pt"
        if path.exists() or (artifacts.remote and (artifacts.remote / path.name).exists()):
            saved = load(artifacts.restore(path.name))
            if saved != context:
                raise ValueError("Workspace context changed; use a new run group")
        else:
            artifacts.save(context, path.name)
    return artifacts


def open_stage(root, stage=None):
    root = Path(root)
    context = load(root / "context.pt")
    remote = Path(context["remote"]) if context.get("remote") else None
    artifacts = CheckedArtifacts(root, remote, smoke=context["smoke"])
    if stage:
        artifacts = CheckedArtifacts(root / stage, remote / stage if remote else None, smoke=context["smoke"])
    return context, artifacts


def code_identity():
    directory = Path(__file__).resolve().parent
    result = {name: normalized_hash(directory / name) for name in BUNDLE_FILES}
    result["bundle_sha256"] = digest(result)
    result["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return result


def normalized_hash(path):
    return hashlib.sha256(
        Path(path).read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n").encode()
    ).hexdigest()


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_parent(weights_path, *, smoke=False):
    path = Path(weights_path)
    if not smoke and not path.as_posix().endswith(PARENT_SUFFIX):
        raise ValueError("Only the pinned latest raw parent is permitted")
    identity = load(path.parent / "run_identity.pt")
    state = load(path.parent / "last.pt")
    metrics = json.loads((path.parent / "metrics.json").read_text(encoding="utf-8"))
    config = identity["config"]
    required = dict(
        dataset="raw",
        seed=42,
        res3d=8 if smoke else 200,
        res2d=8 if smoke else 224,
        store_res=8 if smoke else 200,
        n2d=2,
        latent=256,
        enc2d="maxvit_tiny_rw_224",
    )
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError("Parent raw seed/dimensions/architecture mismatch")
    sources = config.get("data")
    if (
        not isinstance(sources, list)
        or len(sources) != 6
        or any(
            not isinstance(item, dict)
            or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))
            or not isinstance(item.get("size"), int)
            or item["size"] <= 0
            for item in sources
        )
    ):
        raise ValueError("Parent requires six raw-volume/label content identities in split order")
    if state.get("config") != config or state.get("run_id") != identity["id"]:
        raise ValueError("Parent checkpoint identity/config mismatch")
    if metrics.get("tag") != "raw_s42" or metrics.get("seed") != 42:
        raise ValueError("Parent metrics identity mismatch")
    weights = torch.load(path, map_location="cpu", weights_only=True)
    best = state.get("best_state")
    if (
        not isinstance(weights, dict)
        or not weights
        or not all(isinstance(v, torch.Tensor) for v in weights.values())
        or not isinstance(best, dict)
        or weights.keys() != best.keys()
        or any(not torch.equal(weights[k], best[k]) for k in weights)
    ):
        raise ValueError("Parent best_weights differs from last.best_state")
    history = state["history"]
    if not history or metrics.get("hist", metrics.get("history")) != history:
        raise ValueError("Parent metrics/history mismatch")
    candidates = [h for h in history if np.isfinite(h["val_auc"])]
    if not candidates:
        raise ValueError("Parent has no finite validation AUC")
    best_row = max(candidates, key=lambda h: h["val_auc"])
    if best_row["val_auc"] != state["best_auc"]:
        raise ValueError("Parent best AUC/history mismatch")
    if state.get("best_epoch") is not None and state["best_epoch"] != best_row["epoch"]:
        raise ValueError("Parent best epoch mismatch")
    if state["epoch"] < config["epochs"] and state.get("bad", 0) < config["patience"]:
        raise ValueError("Parent training phase is incomplete")
    marker = path.parent / "completed.pt"
    if marker.exists():
        complete = load(marker)
        if complete["run_id"] != identity["id"] or complete["config"] != config:
            raise ValueError("Parent completion identity mismatch")
    return {
        "path": str(path),
        "weights": ft.data_identity(path),
        "wandb_id": identity["id"],
        "source_epoch": best_row["epoch"],
        "source_val_auc": best_row["val_auc"],
        "phase_status": "completed_marker_present"
        if marker.exists()
        else "training_finished_reports_present_no_completion_marker",
        "identity": ft.data_identity(path.parent / "run_identity.pt"),
        "last": ft.data_identity(path.parent / "last.pt"),
        "metrics": ft.data_identity(path.parent / "metrics.json"),
        "data": sources,
    }


def parent_stage(root):
    context, artifacts = open_stage(root)
    parent = verify_parent(context["parent"], smoke=context["smoke"])
    artifacts.save(parent, "parent.pt")
    return parent


def smoke_parent(path):
    path = Path(path)
    config = dict(
        dataset="raw",
        seed=42,
        res3d=8,
        res2d=8,
        store_res=8,
        n2d=2,
        latent=256,
        enc2d="maxvit_tiny_rw_224",
        epochs=1,
        patience=5,
    )
    config["data"] = []
    for _, volumes, labels in smoke_arrays():
        for value in (volumes, labels):
            stream = io.BytesIO()
            np.save(stream, value)
            content = stream.getvalue()
            config["data"].append({"sha256": hashlib.sha256(content).hexdigest(), "size": len(content)})
    history = [{"epoch": 1, "val_auc": 0.5}]
    weights = ft.cpu_state(ft.SmokeModel())
    ft.atomic_save(weights, path)
    ft.atomic_save({"id": "synthetic-parent", "config": config}, path.parent / "run_identity.pt")
    ft.atomic_save(
        dict(
            config=config, run_id="synthetic-parent", best_state=weights, best_auc=0.5, history=history, epoch=1, bad=0
        ),
        path.parent / "last.pt",
    )
    (path.parent / "metrics.json").write_text(json.dumps(dict(tag="raw_s42", seed=42, hist=history)))


def smoke_arrays():
    rng = np.random.default_rng(42)
    for split, n in zip(SPLITS, (10, 6, 6)):
        yield split, rng.integers(0, 256, (n, 1, 8, 8, 8), dtype=np.uint8), np.arange(n, dtype=np.int64) % 2


def verify_raw_sources(root, *, allow_missing=False):
    context, artifacts = open_stage(root)
    parent = load(artifacts.restore("parent.pt"))
    sources = {}
    for index, split in enumerate(SPLITS):
        sources[split] = {}
        for offset, kind in enumerate(("volumes", "labels")):
            path = Path(context["data_root"]) / f"{split}_{kind}.npy"
            expected = parent["data"][2 * index + offset]
            if path.exists():
                if ft.data_identity(path) != {k: expected[k] for k in ("sha256", "size")}:
                    raise ValueError(f"Parent raw data mismatch: {path}; refusing replacement or denoising")
            elif not allow_missing:
                raise FileNotFoundError(path)
            sources[split]["sha256" if offset == 0 else "labels_sha256"] = expected["sha256"]
    return sources


def raw_stage(root, *, revision=None, cpu_export_root=None):
    context, artifacts = open_stage(root)
    data = Path(context["data_root"])
    data.mkdir(parents=True, exist_ok=True)
    sources = verify_raw_sources(root, allow_missing=True)
    manifest_path = data / "source_manifest.json"
    saved = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if saved is not None and (
        saved.get("source_repo") != RAW_REPO
        or not re.fullmatch(r"[0-9a-f]{40}", str(saved.get("source_revision", "")))
        or any(
            saved.get("sources", {}).get(split, {}).get(key) != value
            for split, entry in sources.items()
            for key, value in entry.items()
        )
    ):
        raise ValueError("Trusted raw source_manifest.json does not match pinned parent hashes/repo/revision")
    if cpu_export_root:
        exported = json.loads((Path(cpu_export_root) / "manifest.json").read_text())
        exported_revision = exported["identity"]["source_revision"]
        if exported["identity"]["source_repo"] != RAW_REPO or not re.fullmatch(r"[0-9a-f]{40}", exported_revision):
            raise ValueError("CPU export requires the expected repo and immutable source revision")
        if revision and revision != exported_revision:
            raise ValueError("Requested HF revision differs from CPU export")
        revision = exported_revision
    if revision and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("HF_RAW_REVISION must be an immutable 40-character commit SHA")
    if saved and revision and saved["source_revision"] != revision:
        raise ValueError("Existing raw provenance differs from requested revision")
    revision = revision or (saved["source_revision"] if saved else None)
    names = [f"{s}_{k}.npy" for s in SPLITS for k in ("volumes", "labels")]
    missing = [name for name in names if not (data / name).exists()]
    if context["smoke"]:
        for split, volumes, labels in smoke_arrays():
            if not (data / f"{split}_volumes.npy").exists():
                np.save(data / f"{split}_volumes.npy", volumes)
            if not (data / f"{split}_labels.npy").exists():
                np.save(data / f"{split}_labels.npy", labels)
        provenance = {
            "source_repo": None,
            "source_revision": None,
            "sources": sources,
            "verification": "synthetic_smoke",
        }
    elif missing or (revision and not saved):
        from huggingface_hub import HfApi, snapshot_download

        plan_path = artifacts.local / "raw_download_plan.pt"
        if not plan_path.exists() and artifacts.remote and (artifacts.remote / plan_path.name).exists():
            artifacts.restore(plan_path.name)
        if plan_path.exists():
            plan = load(plan_path)
            if plan["source_repo"] != RAW_REPO or (revision and plan["source_revision"] != revision):
                raise ValueError("Existing download plan revision mismatch; inspect provenance before changing it")
        else:
            info = HfApi(token=os.environ.get("HF_TOKEN")).dataset_info(
                RAW_REPO, revision=revision or (saved["source_revision"] if saved else None), files_metadata=True
            )
            if not re.fullmatch(r"[0-9a-f]{40}", info.sha) or (revision and info.sha != revision):
                raise ValueError("HF API did not resolve the requested immutable revision")
            files = {f.rfilename: f for f in info.siblings}
            metadata = {}
            for name in names:
                remote = files[name]
                lfs = remote.lfs
                sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
                if not sha and (name.endswith("_volumes.npy") or not remote.blob_id):
                    raise ValueError(
                        "HF lacks verifiable LFS/source metadata; provide trusted raw provenance, never refilter"
                    )
                metadata[name] = {"sha256": sha, "blob_id": remote.blob_id, "size": remote.size}
            plan = {"source_repo": RAW_REPO, "source_revision": info.sha, "files": metadata}
        parent = load(artifacts.restore("parent.pt"))
        for name, expected in zip(names, parent["data"]):
            remote = plan["files"][name]
            if remote["size"] != expected["size"] or (remote["sha256"] and remote["sha256"] != expected["sha256"]):
                raise ValueError(f"Pinned HF revision differs from parent data: {name}; no download/denoise started")
        artifacts.save(plan, plan_path.name)
        if missing:
            snapshot_download(
                repo_id=RAW_REPO,
                repo_type="dataset",
                revision=plan["source_revision"],
                local_dir=str(data),
                token=os.environ.get("HF_TOKEN"),
                allow_patterns=missing,
            )
        verify_raw_sources(root)
        for name, remote in plan["files"].items():
            if not remote["sha256"]:
                value = (data / name).read_bytes()
                blob = hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest()
                if blob != remote["blob_id"]:
                    raise ValueError(f"HF pinned Git blob mismatch: {name}")
        provenance = {
            "source_repo": RAW_REPO,
            "source_revision": plan["source_revision"],
            "sources": sources,
            "verification": "HF immutable LFS SHA256 / Git blob plus parent SHA256",
        }
        save_json(provenance, manifest_path)
    else:
        provenance = saved or {
            "source_repo": None,
            "source_revision": None,
            "sources": sources,
            "verification": "parent_hashes_only; HF revision unknown",
        }
    verify_raw_sources(root)
    artifacts.save(provenance, "raw_provenance.pt")
    if manifest_path.exists():
        fd.sync_verified(manifest_path, artifacts.local / manifest_path.name)
        artifacts.sync(artifacts.local / manifest_path.name)
    return provenance


def denoise_bilateral(volume):
    from scripts import compare_denoise_methods as cdm

    return cdm.denoise_volume(volume, "bilateral", workers=2, cache_path=None)[0]


def prepare_stage(root, *, allow_build=False, limit=0, manifest_path=None, cpu_export_root=None, publish_cache=False):
    import scipy
    import skimage

    from scripts import compare_denoise_methods as cdm

    context, artifacts = open_stage(root)
    verify_raw_sources(root)
    if manifest_path and cpu_export_root:
        raise ValueError("Select CACHE_MANIFEST or CPU_EXPORT_ROOT, not both")
    print(
        "Shared-cache bulk publication ENABLED."
        if publish_cache
        else "Bulk publication OFF: local preparation is pending until all derived arrays are verified on shared Drive cache."
    )
    params = {"sigma_color": 0.10, "sigma_spatial": 4.0}
    cdm.METHODS["bilateral"]["params"] = dict(params)
    implementation = {
        "module_sha256": normalized_hash(cdm.__file__),
        "skimage": skimage.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
    }
    if cpu_export_root:
        source_path = Path(context["data_root"]) / "source_manifest.json"
        if not source_path.exists():
            raise ValueError(
                "CPU import needs trusted raw source_manifest.json. Run section 4 with CPU_EXPORT_ROOT "
                "to verify parent bytes against the export's pinned HF LFS metadata; do not rerun the filter."
            )
        manifest = fd.import_cpu_export(
            context["data_root"],
            cpu_export_root,
            context["cache_root"],
            SPLITS,
            res2d=context["config"]["res2d"],
            method="bilateral",
            params=CPU_PARAMS,
            implementation=CPU_IMPLEMENTATION,
            smoke=context["smoke"],
            allow_publish=publish_cache,
            source_manifest_path=source_path,
        )
    else:
        paths = (
            [Path(manifest_path)]
            if manifest_path
            else [
                *sorted((Path(context["data_root"]) / ".final_data").glob("*.json")),
                Path(context["cache_root"]) / "manifest.json",
                *sorted(Path(context["cache_root"]).glob("*/manifest.json")),
            ]
        )
        profiles = {}
        for path in paths:
            if not path.exists():
                continue
            saved = json.loads(path.read_text(encoding="utf-8"))
            cfg = saved.get("config", {})
            if cfg.get("method") != "bilateral" or cfg.get("res2d") != context["config"]["res2d"]:
                continue
            if cfg.get("implementation") == CPU_IMPLEMENTATION and cfg.get("params") == CPU_PARAMS:
                profiles[digest(cfg.get("compatibility", cfg))] = cfg
            elif (
                cfg.get("params") == params
                and isinstance(cfg.get("implementation"), dict)
                and cfg["implementation"].get("module_sha256") == implementation["module_sha256"]
            ):
                profiles[digest(cfg.get("compatibility", cfg))] = cfg
        if len(profiles) > 1:
            raise ValueError("Multiple filter producers available; select CACHE_MANIFEST explicitly")
        if manifest_path and not profiles:
            raise ValueError("CACHE_MANIFEST has unsupported Bilateral semantics; re-import the verified CPU export")
        selected = next(iter(profiles.values()), None)
        cpu = selected and selected["implementation"] == CPU_IMPLEMENTATION
        if selected and not cpu and allow_build and selected["implementation"] != implementation:
            raise ValueError(
                "Producer filter environment differs; restore with ALLOW_BUILD_DENOISED=False. "
                "Do not label a new filter build with another producer's versions."
            )
        if selected:
            params, implementation = selected["params"], selected["implementation"]
        if cpu and allow_build:
            raise ValueError("CPU-export restore never runs a denoiser; set ALLOW_BUILD_DENOISED=False")
        manifest = fd.prepare_data(
            context["data_root"],
            context["cache_root"],
            SPLITS,
            res2d=context["config"]["res2d"],
            denoise_fn=None if cpu else denoise_bilateral,
            method="bilateral",
            params=params,
            implementation=implementation,
            allow_build=allow_build,
            limit=limit,
            smoke=context["smoke"],
            manifest_path=manifest_path,
            publish_cache=publish_cache,
            filter_code_sha256=selected.get("compatibility", {}).get("filter_code_sha256") if cpu else None,
        )
    artifacts.save(manifest, "data_manifest.pt")
    published = cache_published(context, manifest)
    artifacts.save({"published": published, "manifest_digest": digest(manifest)}, "preparation.pt")
    if not published:
        raise RuntimeError(
            "Preparation retained locally but Drive publication is pending. Set PUBLISH_DATA_CACHE=True "
            "and rerun section 5 to verify/upload existing arrays without refiltering; training is blocked."
        )
    return manifest


def cache_published(context, manifest):
    key = hashlib.sha256(json.dumps(manifest["config"], sort_keys=True).encode()).hexdigest()
    for root in (Path(context["cache_root"]) / key, Path(context["cache_root"])):
        path = root / "manifest.json"
        if not path.exists() or json.loads(path.read_text(encoding="utf-8")) != manifest:
            continue
        for split in SPLITS:
            names = dict(
                denoised=f"{split}_volumes_dn.npy",
                labels=f"{split}_labels.npy",
                views=f"{split}_volumes_dn_views_{manifest['config']['res2d']}.npy",
                dzs=f"{split}_volumes_dn_dzs_{manifest['config']['res2d']}.npy",
            )
            for kind, expected in manifest["splits"][split]["files"].items():
                path = root / names[kind]
                if not path.exists() or ft.data_identity(path) != {k: expected[k] for k in ("sha256", "size")}:
                    return False
        return True
    return False


def datasets(root, config, *, splits=SPLITS):
    context, artifacts = open_stage(root)
    manifest = load(artifacts.restore("data_manifest.pt"))
    preparation = load(artifacts.restore("preparation.pt"))
    if preparation != {"published": True, "manifest_digest": digest(manifest)} or not cache_published(
        context, manifest
    ):
        raise ValueError("Verified shared-cache publication is required before dataset use")
    verify_raw_sources(root)
    data = Path(context["data_root"])
    result = []
    for split in splits:
        entry = manifest["splits"][split]
        names = {
            "raw": f"{split}_volumes.npy",
            "labels": f"{split}_labels.npy",
            "denoised": f"{split}_volumes_dn.npy",
            "views": f"{split}_volumes_dn_views_{config['res2d']}.npy",
            "dzs": f"{split}_volumes_dn_dzs_{config['res2d']}.npy",
        }
        for kind, expected in {**entry["sources"], **entry["files"]}.items():
            if ft.data_identity(data / names[kind]) != {k: expected[k] for k in ("sha256", "size")}:
                raise ValueError(f"Data manifest mismatch: {split}/{kind}")
        dataset = ft.FinalDataset(
            data / names["denoised"],
            data / names["labels"],
            res3d=config["res3d"],
            res2d=config["res2d"],
            seed=42,
            train=split == "Training",
        )
        if not context["smoke"] and tuple(dataset.volumes.shape[-3:]) != (200, 200, 200):
            raise ValueError("Real data must remain raw 200 cubed")
        if set(np.unique(dataset.labels)) != {0, 1}:
            raise ValueError(f"Both binary classes required: {split}")
        for kind in ("views", "dzs"):
            expected = entry["files"][kind]
            if ft.data_identity(data / names[kind]) != {k: expected[k] for k in ("sha256", "size")}:
                raise ValueError("Dataset construction changed verified projections")
        result.append(dataset)
    return result


def audit_data_stage(root):
    context, artifacts = open_stage(root)
    config = dict(context["config"])
    validate_profile(config)
    tr, _, _ = datasets(root, config)
    config["class_weights"] = [len(tr) / (2 * int((tr.labels == c).sum())) for c in (0, 1)]
    config["data"] = load(artifacts.restore("data_manifest.pt"))
    config["parent"] = load(artifacts.restore("parent.pt"))
    config["raw_provenance"] = load(artifacts.restore("raw_provenance.pt"))
    artifacts.save(config, "execution.pt")
    return config


def validate_profile(config):
    required = dict(
        dataset="bilateral",
        seed=42,
        n2d=2,
        latent=256,
        enc2d="maxvit_tiny_rw_224",
        lr=5e-5,
        weight_decay=1e-4,
        patience=5,
        checkpoint_steps=10,
        compile=False,
        activation_checkpointing=False,
    )
    if not config["smoke"]:
        required.update(
            epochs=5,
            batch_size=2,
            grad_accum=8,
            store_res=200,
            res3d=200,
            res2d=224,
            pin_memory=True,
            non_blocking=True,
        )
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError("Final 80GB experiment contract mismatch")


def hardware(smoke=False):
    if smoke:
        return {"device": "cpu", "smoke": True, "torch": str(torch.__version__)}
    if not torch.cuda.is_available():
        raise RuntimeError("Real training/preflight requires CUDA with at least 70 GiB; CPU fallback forbidden")
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    if props.total_memory < 70 * 1024**3:
        raise RuntimeError("80GB profile requires at least 70 GiB total GPU memory")
    return {
        "device": f"cuda:{index}",
        "name": props.name,
        "uuid": str(getattr(props, "uuid", "unavailable")),
        "total_memory": props.total_memory,
        "capability": list(torch.cuda.get_device_capability(index)),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        "python": platform.python_version(),
        "timm": importlib.metadata.version("timm"),
        "torchvision": importlib.metadata.version("torchvision"),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }


def ensure_idle():
    if ACTIVE_MODEL is not None or ACTIVE_TRAINER is not None:
        raise RuntimeError("An ACTIVE_MODEL/ACTIVE_TRAINER still exists; inspect failure then release_active()")


def release_active():
    global ACTIVE_MODEL, ACTIVE_TRAINER
    ACTIVE_TRAINER = ACTIVE_MODEL = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_model(config, device):
    if not config["smoke"] and importlib.metadata.version("timm") != "1.0.29":
        raise ValueError("Reviewed MaxViT construction requires timm==1.0.29; rerun setup in a fresh runtime")
    model = (
        ft.SmokeModel()
        if config["smoke"]
        else fm.FinalModel(n_2d=config["n2d"], D=config["latent"], enc2d=config["enc2d"], enc2d_pretrained=False)
    )
    return model.to(device)


def prediction(model, dataset, config):
    return ft.predict(
        model, dataset, config["batch_size"], **{k: config[k] for k in ("amp_dtype", "pin_memory", "non_blocking")}
    )


def binding(config, environment):
    return digest({"config": config, "environment": environment, "code": code_identity()})


def preflight_stage(root, *, enabled=False):
    if not enabled:
        return {"status": "disabled"}
    ensure_idle()
    context, artifacts = open_stage(root)
    config = load(artifacts.restore("execution.pt"))
    validate_profile(config)
    environment = hardware(context["smoke"])
    config["amp_dtype"] = "bfloat16" if environment.get("bf16_supported") else "float16"
    config["amp_reason"] = "bf16 supported" if environment.get("bf16_supported") else "float16 fallback (or CPU smoke)"
    artifacts.save(config, "execution.pt")
    if verify_parent(context["parent"], smoke=context["smoke"]) != config["parent"]:
        raise ValueError("Parent changed since CPU verification")
    train, _, _ = datasets(root, config)
    device = torch.device(environment["device"])
    fraction = config["preflight_fraction"]
    if not 0 < fraction <= 0.85 or config["preflight_windows"] != 2:
        raise ValueError("Preflight requires two optimizer windows and fraction <= 0.85")
    free_before = 0
    if device.type == "cuda":
        free_before = torch.cuda.mem_get_info(device)[0]
        if free_before < environment["total_memory"] * fraction:
            raise RuntimeError("Insufficient free GPU memory for the full-model budget")
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        torch.cuda.reset_peak_memory_stats(device)
    snapshot = ft.rng_state()
    model = optimizer = scaler = loader = iterator = batch = x = views = y = logits = loss = parameter = weights = None
    started = time.monotonic()
    report = {"status": "failed", "environment": environment, "free_before": free_before}
    probe_artifacts, run = stage_run(root, "preflight", config, "not-started")
    try:
        model = make_model(config, device)
        model.load_state_dict(torch.load(context["parent"], map_location="cpu", weights_only=True), strict=True)
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            **({"fused": True} if config["fused_adamw"] else {}),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config["amp_dtype"] == "float16")
        loader = DataLoader(train, batch_size=config["batch_size"], num_workers=0, pin_memory=config["pin_memory"])
        iterator = iter(loader)
        weights = torch.tensor(config["class_weights"], device=device)
        for _ in range(config["preflight_windows"]):
            denominator = 0.0
            optimizer.zero_grad(set_to_none=True)
            for _ in range(config["grad_accum"]):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                x, views, y = (t.to(device, non_blocking=config["non_blocking"]) for t in batch)
                with torch.autocast(
                    device.type, dtype=getattr(torch, config["amp_dtype"]), enabled=device.type == "cuda"
                ):
                    logits = model(x, views)
                    loss = F.cross_entropy(logits, y, weight=weights, reduction="sum")
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite preflight loss")
                scaler.scale(loss).backward()
                denominator += weights[y].sum().item()
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(denominator)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale or not optimizer.state:
                raise RuntimeError("Preflight must allocate optimizer state and complete updates")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            report.update(
                allocated=torch.cuda.memory_allocated(device),
                reserved=torch.cuda.memory_reserved(device),
                peak_allocated=torch.cuda.max_memory_allocated(device),
                peak_reserved=torch.cuda.max_memory_reserved(device),
                free_after=torch.cuda.mem_get_info(device)[0],
            )
            if report["peak_reserved"] > environment["total_memory"] * fraction or report["free_after"] < environment[
                "total_memory"
            ] * (1 - fraction):
                raise RuntimeError("Preflight headroom gate failed")
        report.update(status="passed", digest=binding(config, environment), micro_steps=2 * config["grad_accum"])
    except BaseException:
        run.finish(exit_code=1)
        raise
    finally:
        model = optimizer = scaler = loader = iterator = batch = x = views = y = logits = loss = parameter = weights = (
            None
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        ft.restore_rng(snapshot)
        report["wall_seconds"] = time.monotonic() - started
        report["estimated_train_seconds"] = (
            report["wall_seconds"]
            * config["epochs"]
            * np.ceil(len(train) / config["batch_size"])
            / (2 * config["grad_accum"])
        )
        try:
            artifacts.save(report, "preflight.pt")
            probe_artifacts.save(report, "preflight.pt")
        except BaseException:
            run.finish(exit_code=1)
            raise
    try:
        run.log({f"preflight/{k}": v for k, v in report.items() if isinstance(v, (int, float, str))})
        finish_stage(probe_artifacts, run, {"parent_id": config["parent"]["wandb_id"], "digest": report["digest"]})
    except BaseException:
        report["status"] = "publication_failed"
        artifacts.save(report, "preflight.pt")
        run.finish(exit_code=1)
        raise
    return report


def start_run(artifacts, config, stage, *, resume=False, warm_start=""):
    return ft.init_wandb(
        "final_bilateral_s42_" + stage, config, artifacts, resume=resume, smoke=config["smoke"], warm_start=warm_start
    )


def stage_run(root, stage, config, train_id):
    context, artifacts = open_stage(root, stage)
    session = "sessions/" + uuid.uuid4().hex
    if artifacts.remote:
        (artifacts.remote / "sessions").mkdir(exist_ok=True)
    run_artifacts = CheckedArtifacts(
        artifacts.local / session, artifacts.remote / session if artifacts.remote else None, smoke=context["smoke"]
    )
    run = start_run(run_artifacts, {**config, "stage": stage, "train_id": train_id}, stage)
    try:
        (artifacts.local / "completed.pt").unlink(missing_ok=True)
        if artifacts.remote:
            (artifacts.remote / "completed.pt").unlink(missing_ok=True)
    except BaseException:
        run.finish(exit_code=1)
        raise
    return artifacts, run


def audit_files(artifacts):
    files = {}
    for path in sorted(artifacts.local.rglob("*")):
        if not path.is_file() or "wandb" in path.relative_to(artifacts.local).parts:
            continue
        if path.name.endswith((".sha256.pt", ".sync-pending.pt")) or path.name in ("completed.pt", "audit.pt"):
            continue
        artifacts.sync(path)
        files[path.relative_to(artifacts.local).as_posix()] = ft.data_identity(path)
    if list(artifacts.local.rglob("*.sync-pending.pt")):
        raise RuntimeError("Pending Drive artifacts prevent completion")
    return files


def finish_stage(artifacts, run, provenance):
    files = audit_files(artifacts)
    marker = {"status": "complete", "files": files, "run_id": run.id, "provenance": provenance}
    artifacts.save(marker, "audit.pt")
    run.summary.update({"stage/artifacts_verified": True, "stage/provenance": provenance})
    run.finish(exit_code=0)
    artifacts.save(marker, "completed.pt")
    return marker


def validated_completion(artifacts):
    marker = load(artifacts.restore("completed.pt"))
    if marker.get("status") != "complete" or not marker.get("files"):
        raise ValueError("Invalid completion marker")
    for name, expected in marker["files"].items():
        path = artifacts.restore(name)
        if ft.data_identity(path) != expected:
            raise ValueError(f"Completed artifact changed: {name}")
        artifacts.sync(path)
    artifacts.sync(artifacts.local / "completed.pt")
    return marker


def training_stage(root, *, enabled=False, resume=False):
    global ACTIVE_MODEL, ACTIVE_TRAINER
    if not enabled:
        return {"status": "disabled"}
    ensure_idle()
    context, base = open_stage(root)
    config = load(base.restore("execution.pt"))
    validate_profile(config)
    environment = hardware(context["smoke"])
    preflight = load(base.restore("preflight.pt"))
    if preflight.get("status") != "passed" or preflight.get("digest") != binding(config, environment):
        raise ValueError("Successful preflight for exact config/data/weights/GPU stack required")
    _, probe_artifacts = open_stage(root, "preflight")
    if validated_completion(probe_artifacts)["provenance"]["digest"] != preflight["digest"]:
        raise ValueError("Preflight publication digest mismatch")
    if verify_parent(context["parent"], smoke=context["smoke"]) != config["parent"]:
        raise ValueError("Pinned parent changed")
    tr, va, _ = datasets(root, config)
    if not context["smoke"] and torch.cuda.mem_get_info()[0] < max(
        preflight["peak_reserved"], environment["total_memory"] * config["preflight_fraction"]
    ):
        raise RuntimeError("Insufficient free GPU memory since preflight")
    if not context["smoke"]:
        torch.cuda.set_per_process_memory_fraction(config["preflight_fraction"])
    _, artifacts = open_stage(root, "train")
    checkpoint_resume = resume and (
        (artifacts.local / "last.pt").exists() or bool(artifacts.remote and (artifacts.remote / "last.pt").exists())
    )
    if resume:
        if not (artifacts.local / "run_identity.pt").exists() and not (
            artifacts.remote and (artifacts.remote / "run_identity.pt").exists()
        ):
            raise ValueError(
                "RESUME requires an existing safe last.pt or verified stranded run identity; entirely new run rejected"
            )
        identity = load(artifacts.restore("run_identity.pt"))
        if identity.get("config") != config or not identity.get("id"):
            raise ValueError("Persisted resume identity/config mismatch")
        if not checkpoint_resume and identity.get("warm_start") != context["parent"]:
            raise ValueError("Stranded identity must reference the exact verified pinned parent")
    if (artifacts.local / "completed.pt").exists() or (
        artifacts.remote and (artifacts.remote / "completed.pt").exists()
    ):
        if not resume:
            raise FileExistsError("Training already complete; use independent evaluation")
        completed = validated_completion(artifacts)
        if completed["provenance"]["config"] != config:
            raise ValueError("Completed training config mismatch")
        return completed
    run = start_run(
        artifacts, config, "train", resume=resume, warm_start="" if checkpoint_resume else context["parent"]
    )
    try:
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        import random

        random.seed(config["seed"])
        ACTIVE_MODEL = make_model(config, environment["device"])
        ACTIVE_TRAINER = ft.Trainer(
            ACTIVE_MODEL,
            tr,
            config,
            artifacts,
            run,
            resume=checkpoint_resume,
            warm_start="" if checkpoint_resume else context["parent"],
        )

        def evaluate(model):
            p, y, logits = prediction(model, va, config)
            return {
                **fr.metrics(p, y, ranking_scores=logits[:, 1] - logits[:, 0]),
                "loss": F.cross_entropy(torch.tensor(logits), torch.tensor(y)).item(),
            }

        if not ACTIVE_TRAINER.fit(evaluate):
            audit_files(artifacts)
            run.summary.update({"stopped_safely": True})
            run.finish(exit_code=0)
            release_active()
            return {"status": "stopped_safely"}
        artifacts.save(ACTIVE_TRAINER.best_state, "best_weights.pt")
        handoff = {
            "config": config,
            "history": ACTIVE_TRAINER.history,
            "train_id": run.id,
            "best_epoch": ACTIVE_TRAINER.best_epoch,
            "best_step": ACTIVE_TRAINER.best_step,
            "best_auc": ACTIVE_TRAINER.best_auc,
            "parent": config["parent"],
            "code": code_identity(),
            "weights": ft.data_identity(artifacts.local / "best_weights.pt"),
            "environment": environment,
        }
        artifacts.save(handoff, "handoff.pt")
        result = finish_stage(artifacts, run, handoff)
    except BaseException:
        run.finish(exit_code=1)
        raise
    release_active()
    return result


def training_handoff(root):
    _, artifacts = open_stage(root, "train")
    handoff = load(artifacts.restore("handoff.pt"))
    weights = artifacts.restore("best_weights.pt")
    state = load(artifacts.restore("last.pt"))
    identity = load(artifacts.restore("run_identity.pt"))
    if (
        handoff["weights"] != ft.data_identity(weights)
        or identity["id"] != handoff["train_id"]
        or state["run_id"] != identity["id"]
        or state["config"] != handoff["config"]
        or identity["config"] != handoff["config"]
    ):
        raise ValueError("Training handoff identity mismatch")
    if state["epoch"] < state["config"]["epochs"] and state["bad"] < state["config"]["patience"]:
        raise ValueError("Training is not finished")
    best = torch.load(weights, map_location="cpu", weights_only=True)
    if (
        best.keys() != state["best_state"].keys()
        or any(not torch.equal(best[k], state["best_state"][k]) for k in best)
        or handoff["history"] != state["history"]
        or handoff["best_epoch"] != state["best_epoch"]
    ):
        raise ValueError("Training best weights/history mismatch")
    handoff["training_phase_status"] = (
        "completed_marker_present"
        if (artifacts.local / "completed.pt").exists()
        or (artifacts.remote and (artifacts.remote / "completed.pt").exists())
        else "training_finished_handoff_present_no_completion_marker"
    )
    return handoff, weights


def evaluation_stage(root, *, enabled=False):
    if not enabled:
        return {"status": "disabled"}
    ensure_idle()
    handoff, weights = training_handoff(root)
    config = handoff["config"]
    identity = digest(
        {"weights": handoff["weights"], "data": config["data"], "config": config, "inference_code": code_identity()}
    )
    artifacts, run = stage_run(root, "evaluation", config, handoff["train_id"])
    model = None
    try:
        for split, source in (("val", "Validation"), ("test", "Test")):
            name = f"{split}_logits.pt"
            path = artifacts.local / name
            receipt_name = f"{split}_logits_receipt.pt"
            receipt_path = artifacts.local / receipt_name
            if not path.exists() and artifacts.remote and (artifacts.remote / name).exists():
                path = artifacts.restore(name)
            if not receipt_path.exists() and artifacts.remote and (artifacts.remote / receipt_name).exists():
                receipt_path = artifacts.restore(receipt_name)
            if path.exists() and receipt_path.exists():
                receipt = load(receipt_path)
                if receipt != {"identity": identity, "content": ft.data_identity(path)}:
                    raise ValueError("Cached prediction checksum/identity mismatch")
                cached = load(path)
                if cached["identity"] != identity:
                    raise ValueError("Cached prediction identity mismatch; use a new analysis directory")
            else:
                (dataset,) = datasets(root, config, splits=(source,))
                if model is None:
                    env = hardware(config["smoke"])
                    model = make_model(config, env["device"])
                    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
                _, labels, logits = prediction(model, dataset, config)
                cached = {
                    "identity": identity,
                    "labels": labels,
                    "logits": logits,
                    "split": split,
                    "weights": handoff["weights"],
                    "indices": np.arange(len(labels)),
                }
                artifacts.save(cached, name)
                artifacts.save({"identity": identity, "content": ft.data_identity(path)}, receipt_name)
            values = fr.metrics(
                torch.tensor(cached["logits"]).softmax(1)[:, 1].numpy(),
                cached["labels"],
                ranking_scores=cached["logits"][:, 1] - cached["logits"][:, 0],
            )
            run.log({f"{split}/{k}": v for k, v in values.items()})
        artifacts.save(handoff, "handoff.pt")
        return finish_stage(artifacts, run, {"train_id": handoff["train_id"], "identity": identity})
    except BaseException:
        run.finish(exit_code=1)
        raise
    finally:
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def report_stage(root):
    _, evaluation = open_stage(root, "evaluation")
    validated_completion(evaluation)
    handoff = load(evaluation.restore("handoff.pt"))
    val, test = (load(evaluation.restore(f"{split}_logits.pt")) for split in ("val", "test"))
    if val["identity"] != test["identity"]:
        raise ValueError("Validation/test inference identity mismatch")
    report, predictions = fr.calibrate_and_score(
        val["logits"],
        val["labels"],
        test["logits"],
        test["labels"],
        n_boot=20 if handoff["config"]["smoke"] else 1000,
        seed=42,
    )
    artifacts, run = stage_run(root, "analysis", handoff["config"], handoff["train_id"])
    try:
        provenance = {**handoff, "prediction_identity": val["identity"], "report_code": code_identity()}
        fr.write_analysis(report, predictions, handoff["history"], artifacts, run, provenance)
        finish_stage(artifacts, run, provenance)
        return report
    except BaseException:
        run.finish(exit_code=1)
        raise


def xai_stage(root, *, enabled=False):
    if not enabled:
        return {"status": "disabled"}
    ensure_idle()
    handoff, weights = training_handoff(root)
    config = handoff["config"]
    artifacts, run = stage_run(root, "xai", config, handoff["train_id"])
    model = None
    try:
        (dataset,) = datasets(root, config, splits=("Validation",))
        model = make_model(config, hardware(config["smoke"])["device"])
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        ft.save_xai(model, dataset, artifacts, run, smoke=config["smoke"])
        return finish_stage(
            artifacts,
            run,
            {
                "train_id": handoff["train_id"],
                "weights": handoff["weights"],
                "sample": "Validation row 0",
                "branch_drop_first": "zero all 2D inputs, NOT 3D",
            },
        )
    except BaseException:
        run.finish(exit_code=1)
        raise
    finally:
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def completion_stage(root):
    _, base = open_stage(root)
    _, analysis = open_stage(root, "analysis")
    marker = validated_completion(analysis)
    report = json.loads(analysis.restore("metrics.json").read_text(encoding="utf-8"))
    base.save({"analysis": marker, "report": report}, "final_summary.pt")
    audit_files(base)
    return {"status": "analysis_complete", "train_id": marker["provenance"]["train_id"]}
