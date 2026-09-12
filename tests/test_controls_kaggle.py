import concurrent.futures
import copy
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import controls_kaggle as kg
from scripts import controls_kaggle_data as kd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "3d_glaucoma_controls_96_kaggle.ipynb"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("CTRL_KAGGLE_SMOKE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with monkeypatch.context() as context:
        context.setattr(kg.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
        value = kg.default_config()
    return value


def prepared(cfg):
    sources = {kind: kd.prepare_source(cfg, kind) for kind in ("raw", "bilateral")}
    kd.verify_bilateral(sources["raw"], sources["bilateral"], smoke=True)
    return {kind: kd.prepare_views(cfg, source) for kind, source in sources.items()}


def bundle(cfg, data, root, tag="P_s42", *, completed=False, initialized=False):
    root.mkdir(parents=True, exist_ok=True)
    spec, seed, _ = kg.jobs({**cfg, "run_target": tag})[0]
    dataset = kd.make_datasets(data, spec, seed, smoke=cfg["smoke"])[0]
    config = kg.training_config(cfg, data, spec, seed, dataset)
    config.update(batch_size=2, grad_accum=8, fp16_init_scale=1024, probe={"batch_size": 2}, probe_device="cpu")
    identity = {"id": f"run-{tag}", "config": config, "warm_start": ""}
    kg.ft.atomic_save(identity, root / "run_identity.pt")
    kg.write_json(root / "run_identity.json", identity)
    kg.write_json(
        root / "initial_stage.json", {"stage": "identity", "tag": tag, "identity_digest": kg.digest(identity)}
    )
    if not initialized:
        state = {
            "format": 1,
            "config": config,
            "run_id": identity["id"],
            "epoch": cfg["epochs"] if completed else 0,
            "cursor": 0,
            "step": 1 if completed else 0,
            "model": {},
            "optimizer": {},
            "scheduler": {},
            "scaler": {},
            "rng": {},
        }
        kg.ft.atomic_save(state, root / "last.pt")
    if completed:
        kg.ft.atomic_save({}, root / "best_weights.pt")
        labels = np.array([0, 1, 0, 1], dtype=np.int64)
        predictions = {f"{s}_labels": labels for s in ("train", "val", "test")}
        predictions.update(
            {
                f"{s}_logits": np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8]], dtype=np.float32)
                for s in ("train", "val", "test")
            }
        )
        np.savez(root / "logits.npz", **predictions)
        kg.write_json(
            root / "logits_receipt.json",
            {
                "provenance": {
                    "schema": 2,
                    "identity_digest": kg.digest(identity),
                    "best_weights": kd.sha256(root / "best_weights.pt"),
                    "data": config["data"],
                },
                "sha256": kd.sha256(root / "logits.npz"),
                "labels": {s: kg.digest(labels.tolist()) for s in ("train", "val", "test")},
            },
        )
        kg.write_json(root / "report.json", {"test": {"auc_roc": 0.75}})
        kg.write_json(root / "completed.json", kg.completion_record(cfg, tag, root))
    stage = "identity" if initialized else "completed" if completed else "epoch-0"
    kg.write_json(root / "recovery.json", kg.recovery_descriptor(root, stage))
    return identity


def fake_cloud(monkeypatch, sources, calls):
    import wandb

    def artifact(reference, type):
        assert type == "checkpoint"
        root = sources[reference]
        descriptor = root / "recovery.json"
        metadata = {"stage": json.loads(descriptor.read_text())["stage"]} if descriptor.exists() else {}

        def get_path(name):
            def download(root):
                calls.append((reference, name))
                destination = Path(root) / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(sources[reference] / name, destination)
                return str(destination)

            return SimpleNamespace(download=download)

        return SimpleNamespace(
            metadata=metadata,
            get_path=get_path,
            manifest=SimpleNamespace(
                entries={p.name: SimpleNamespace(size=p.stat().st_size) for p in root.iterdir() if p.is_file()}
            ),
        )

    monkeypatch.setattr(wandb, "Api", lambda: SimpleNamespace(artifact=artifact))


def test_plan_and_frozen_defaults(monkeypatch):
    monkeypatch.delenv("CTRL_KAGGLE_SMOKE", raising=False)
    cfg = kg.default_config()
    assert len(kg.jobs(cfg)) == len({job[2] for job in kg.jobs(cfg)}) == 21
    assert (cfg["epochs"], cfg["patience"], cfg["max_runs_per_session"], cfg["session_hours"]) == (20, 21, 2, 10)
    assert cfg["seeds"] == [42, 43, 44]
    assert cfg["effective_batch"] == 16
    assert cfg["store_res"] == 200 and cfg["res3d"] == 96 and cfg["res2d"] == 224
    assert [(s["code"], s["use_3d"], s["n_2d"], s["view_indices"], s["fusion"], s["gate_fixed"]) for s in kg.SPECS] == [
        ("P", True, 2, [0, 1], "crossgate", False),
        ("B1", True, 1, [0], "crossgate", False),
        ("B2", True, 1, [1], "crossgate", False),
        ("B3", False, 2, [0, 1], "concat", False),
        ("C1", True, 2, [0, 1], "concat", False),
        ("C2", True, 2, [0, 1], "crossgate", True),
        ("B4", True, 2, [0, 1], "crossgate", False),
    ]
    assert kg.jobs({**cfg, "run_target": "B4_s44"})[0][2] == "B4_s44"
    with pytest.raises(ValueError, match="RUN_TARGET"):
        kg.jobs({**cfg, "run_target": "B4"})


def test_gpu_detection_and_env(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "not-persisted")
    assert len(kg.detect_gpus(listing="0, Tesla P100-PCIE-16GB, 16384")) == 1
    dual = "0, Tesla T4, 15360\n1, Tesla T4, 15360"
    devices = kg.detect_gpus(listing=dual)
    assert [item["physical"] for item in devices] == ["0", "1"]
    assert len(kg.detect_gpus("single", listing=dual)) == 1
    envs = [kg.child_env(item["physical"]) for item in devices]
    assert envs[0]["CUDA_VISIBLE_DEVICES"] != envs[1]["CUDA_VISIBLE_DEVICES"]
    assert all(env["WANDB_API_KEY"] == "not-persisted" for env in envs)
    with pytest.raises(RuntimeError, match="Unvalidated"):
        kg.detect_gpus(listing="0, A100, 81920")


@pytest.fixture
def cuda_runtime(monkeypatch):
    calls, logs, finishes = [], [], []
    run = SimpleNamespace(summary={}, config={}, log=logs.append, finish=lambda **kwargs: finishes.append(kwargs))
    monkeypatch.setattr(torch.version, "cuda", "12.6")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Tesla T4")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_75", "sm_80", "compute_90"])
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 90500)

    class TinyTensor:
        def add_(self, value):
            assert value == 1
            calls.append("add")
            return self

    def ones(size, *, dtype, device):
        assert size == 4 and dtype == torch.float32 and device == "cuda:0"
        calls.append("ones")
        return TinyTensor()

    def synchronize(device):
        assert device == "cuda:0"
        calls.append("synchronize")

    monkeypatch.setattr(torch, "ones", ones)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    return SimpleNamespace(run=run, calls=calls, logs=logs, finishes=finishes)


def test_cuda_runtime_supported_t4(cuda_runtime):
    metadata = kg.check_cuda_runtime(smoke=False, run=cuda_runtime.run)
    assert cuda_runtime.calls == ["ones", "add", "synchronize"]
    assert metadata["status"] == "passed" and metadata["device"] == "Tesla T4"
    assert metadata["capability"] == [7, 5] and metadata["cuda"] == "12.6"
    assert metadata["supported_arches"] == ["sm_75", "sm_80", "compute_90"]
    assert metadata["cudnn"] == 90500
    assert cuda_runtime.run.summary["runtime/device_check"] == metadata
    assert cuda_runtime.run.config == {}


def test_p100_cuda13_fails_before_model_download_or_probe(cfg, cuda_runtime, monkeypatch):
    data = prepared(cfg)
    cfg["smoke"] = False
    monkeypatch.setattr(kg, "validate_config", lambda *a: None)
    monkeypatch.setattr(kg, "load_credentials", lambda **kwargs: None)
    monkeypatch.setattr(kg.ft, "init_wandb", lambda *a, **kwargs: cuda_runtime.run)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Tesla P100-PCIE-16GB")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (6, 0))

    def no_kernel(device):
        cuda_runtime.calls.append("synchronize")
        raise RuntimeError("CUDA error: no kernel image is available for execution on the device")

    monkeypatch.setattr(torch.cuda, "synchronize", no_kernel)
    monkeypatch.setattr(kg, "make_model", lambda *a, **kwargs: pytest.fail("Model/pretrained download before check"))
    monkeypatch.setattr(kg, "probe_batch", lambda *a, **kwargs: pytest.fail("Probe before check"))
    with pytest.raises(RuntimeError, match="no kernel image") as error:
        kg.worker(cfg, data, "P_s42", deadline=float("inf"), stop_path=str(Path(cfg["output_root"]) / "stop"))
    assert "compatible Kaggle image" in str(error.value) and "sm_60" in str(error.value)
    assert "no automatic reinstall" in str(error.value)
    assert cuda_runtime.calls == ["ones", "add", "synchronize"]
    metadata = cuda_runtime.run.summary["runtime/device_check"]
    assert metadata["status"] == "failed" and metadata["capability"] == [6, 0] and metadata["cuda"] == "13.0"
    assert cuda_runtime.finishes == [{"exit_code": 1}]
    assert not (Path(cfg["output_root"]) / cfg["group"] / "P_s42" / "run_identity.pt").exists()


@pytest.mark.parametrize("problem", ["unavailable", "multiple", "wrong-hardware"])
def test_cuda_runtime_rejects_invalid_assignment(cuda_runtime, monkeypatch, problem):
    if problem == "unavailable":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    elif problem == "multiple":
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    else:
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "H100")
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    with pytest.raises(RuntimeError, match="compatibility check failed"):
        kg.check_cuda_runtime(smoke=False, run=cuda_runtime.run)
    assert not cuda_runtime.calls
    assert cuda_runtime.run.summary["runtime/device_check"]["status"] == "failed"


def test_cuda_runtime_cpu_smoke_never_touches_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU smoke must not query or initialize CUDA")

    for name in (
        "is_available",
        "device_count",
        "get_device_name",
        "get_device_capability",
        "get_arch_list",
        "synchronize",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch, "ones", forbidden)
    run = SimpleNamespace(summary={}, log=lambda values: None)
    assert kg.check_cuda_runtime(smoke=True, run=run)["status"] == "synthetic-cpu-skip"


def test_readonly_views_identity_and_augmentation(cfg, monkeypatch):
    sources = prepared(cfg)
    raw = sources["raw"]
    before_cache = {split: entry["views"] for split, entry in raw["entries"].items()}
    kd.prepare_views(cfg, raw)
    assert {split: entry["views"] for split, entry in raw["entries"].items()} == before_cache
    files = {p: p.read_bytes() for p in Path(cfg["temp_root"]).joinpath("raw").iterdir()}
    for path in files:
        path.chmod(0o444)
    try:

        def forbidden(*args, **kwargs):
            pytest.fail("Original adjacent-source builder must never be invoked")

        monkeypatch.setattr(kg.ft, "build_views", forbidden)
        dataset = kd.make_datasets(sources, kg.SPECS[0], 42, smoke=True)[0]
        entry = raw["entries"]["Training"]
        volume = kg.ft.volume_sample(dataset.volumes, 0)
        volume = kg.fm.to_depth_last(volume, int(dataset.dzs[0]))
        views = np.asarray(dataset.views[0])
        rng = np.random.default_rng(np.random.SeedSequence([42, 0, 0]))
        volume, views = kg.fm.aug_pair(volume, views, rng)
        actual, actual_views, _ = dataset[0]
        np.testing.assert_array_equal(actual.numpy()[0], volume.astype(np.float32) / 255)
        np.testing.assert_array_equal(actual_views.numpy()[:, 0], views.astype(np.float32) / 255)
        b3 = kd.make_datasets(sources, kg.SPECS[3], 42, smoke=True)[0]
        assert b3.volumes is None
        assert b3[0][0].shape == (1, 1, 1, 1)
        identity = json.dumps(entry["identity"])
        assert cfg["temp_root"] not in identity
        assert entry["identity"]["identity"]["half"] == 16
        for path, content in files.items():
            assert path.read_bytes() == content
        cache = Path(entry["views"])
        with cache.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"\x00")
        with pytest.raises(ValueError, match="cache"):
            kd.prepare_views(cfg, kd.prepare_source(cfg, "raw"))
    finally:
        for path in files:
            path.chmod(0o666)


def test_pinned_corruption_blocked(tmp_path):
    source = tmp_path / "source.npy"
    source.write_bytes(b"trusted")
    remote = SimpleNamespace(size=7, lfs={"sha256": hashlib.sha256(b"trusted").hexdigest()}, blob_id=None)
    assert kd.cd._verify_file(source, remote) == remote.lfs["sha256"]
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        kd.cd._verify_file(source, remote)


def test_real_attached_pins_and_manifest(cfg, monkeypatch, tmp_path):
    import huggingface_hub

    roots = {kind: tmp_path / kind for kind in ("raw", "bilateral")}
    for kind, root in roots.items():
        root.mkdir()
        suffix = "volumes" if kind == "raw" else "volumes_dn"
        shape = (2, 1, 200, 200, 200) if kind == "raw" else (2, 200, 200, 200)
        for split in kd.SPLITS:
            np.save(root / f"{split}_{suffix}.npy", np.zeros(shape, dtype=np.uint8))
            np.save(root / f"{split}_labels.npy", np.array([0, 1], dtype=np.int64))
    identity = {
        "source_repo": kd.RAW_REPO,
        "source_revision": kd.RAW_REVISION,
        "resolution": 200,
        "sources": {
            s: {
                "repo_revision": kd.RAW_REVISION,
                "shape": [2, 1, 200, 200, 200],
                "labels_sha256": kd.sha256(roots["raw"] / f"{s}_labels.npy"),
            }
            for s in kd.SPLITS
        },
    }
    markers = {
        s: {"complete": True, "identity": identity, "sha256": kd.sha256(roots["bilateral"] / f"{s}_volumes_dn.npy")}
        for s in kd.SPLITS
    }
    kd.write_json(roots["bilateral"] / "manifest.json", {"complete": True, "identity": identity, "splits": markers})
    for split, marker in markers.items():
        kd.write_json(roots["bilateral"] / f"{split}_complete.json", marker)
    infos = {}
    for kind, root in roots.items():
        repo, revision = (
            (kd.RAW_REPO, kd.RAW_REVISION) if kind == "raw" else (kd.cd.DEFAULT_REPO, kd.cd.DEFAULT_REVISION)
        )
        siblings = [
            SimpleNamespace(rfilename=p.name, size=p.stat().st_size, lfs={"sha256": kd.sha256(p)}, blob_id=None)
            for p in root.iterdir()
        ]
        infos[repo] = SimpleNamespace(sha=revision, siblings=siblings)

    def info(repo, revision, files_metadata):
        assert files_metadata and infos[repo].sha == revision
        return infos[repo]

    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token: SimpleNamespace(dataset_info=info))
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kwargs: pytest.fail("Attached inputs must not download")
    )
    cfg.update(smoke=False, raw_root=str(roots["raw"]), bilateral_root=str(roots["bilateral"]))
    before = {p: p.stat().st_mtime_ns for root in roots.values() for p in root.iterdir()}
    raw, bilateral = kd.prepare_source(cfg, "raw"), kd.prepare_source(cfg, "bilateral")
    kd.verify_bilateral(raw, bilateral)
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in before.items())
    bilateral["manifest"]["identity"]["source_revision"] = "0" * 40
    with pytest.raises(ValueError, match="manifest"):
        kd.verify_bilateral(raw, bilateral)


def test_disk_failure_before_download(cfg, monkeypatch):
    import huggingface_hub

    names = [f"{s}_{part}.npy" for s in kd.SPLITS for part in ("volumes", "labels")]
    info = SimpleNamespace(
        sha=kd.RAW_REVISION, siblings=[SimpleNamespace(rfilename=name, size=10**10) for name in names]
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kwargs: SimpleNamespace(dataset_info=lambda *a, **kw: info))
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kwargs: pytest.fail("Download before disk check")
    )
    monkeypatch.setattr(kd.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    cfg["smoke"] = False
    with pytest.raises(OSError, match="Insufficient disk"):
        kd.prepare_source(cfg, "raw")


def test_checkpoint_failure_retains_local(cfg, tmp_path, monkeypatch):
    import wandb

    monkeypatch.setattr(wandb, "Artifact", lambda *a, **k: SimpleNamespace(add_file=lambda *a, **k: None))
    root = tmp_path / "artifacts"
    bundle(cfg, prepared(cfg), root)
    storage = kg.KaggleArtifacts(root)
    kg.write_json(root / "archived.json", {"stage": "old"})

    def fail(*args, **kwargs):
        raise RuntimeError("quota")

    storage.run = SimpleNamespace(id="test", offline=False, disabled=False, log_artifact=fail)
    with pytest.raises(RuntimeError, match="local checkpoint retained"):
        storage.publish("epoch-1")
    assert torch.load(root / "last.pt", weights_only=False)["format"] == 1
    assert not (root / "archived.json").exists()
    waited = []
    receipt = SimpleNamespace(wait=lambda: waited.append(True), qualified_name="entity/project/checkpoint:v0")
    storage.run.log_artifact = lambda *a, **k: receipt
    storage.publish("epoch-1")
    assert waited == [True]
    assert (root / "archived.json").exists()


def test_scaler_settling():
    sequence = iter([False] * 8 + [True, True])
    assert kg.settling_updates(lambda: next(sequence)) == 8
    with pytest.raises(FloatingPointError, match="eight"):
        kg.settling_updates(lambda: False)


def test_batch_one_stop_resume_without_probe(cfg, monkeypatch):
    data = prepared(cfg)
    cfg["run_target"] = "P_s42"
    runs = []
    monkeypatch.setattr(kg.ft, "load_env_file", lambda: None)
    original_datasets = kg.make_datasets

    class Repeated(kd.KaggleDataset):
        def __getitem__(self, index):
            return super().__getitem__(index % 4)

    class DropoutSmoke(kg.SmokeModel):
        def forward(self, x, views):
            return torch.nn.functional.dropout(super().forward(x, views), p=0.1, training=self.training)

    def datasets(*args, **kwargs):
        values = original_datasets(*args, **kwargs)
        values[0].__class__ = Repeated
        values[0].labels = np.tile(values[0].labels, 8)
        return values

    monkeypatch.setattr(kg, "make_datasets", datasets)
    monkeypatch.setattr(kg, "make_model", lambda spec, **kwargs: DropoutSmoke(spec))

    def init(name, config, artifacts, *, resume, smoke):
        assert smoke
        if resume:
            state = torch.load(artifacts.local / "run_identity.pt", weights_only=False)
            assert state["config"] == config
        else:
            state = {"id": "cpu-resume", "config": config}
            artifacts.save(state, "run_identity.pt")
        run = SimpleNamespace(
            id=state["id"], summary={}, log=lambda *a, **kw: None, finish=lambda **kwargs: runs.append(kwargs)
        )
        return run

    monkeypatch.setattr(kg.ft, "init_wandb", init)
    monkeypatch.setattr(kg, "probe_batch", lambda *a, **kw: {"batch_size": 1, "fp16_init_scale": 1024})
    original_fit = kg.ct.Trainer.fit

    def interrupted(trainer, evaluate, **kwargs):
        original_boundary = kwargs["boundary_hook"]

        def boundary(current):
            original_boundary(current)
            current.stop_requested = True

        return original_fit(trainer, evaluate, **{**kwargs, "boundary_hook": boundary})

    monkeypatch.setattr(kg.ct.Trainer, "fit", interrupted)
    stop = str(Path(cfg["output_root"]) / "STOP")
    kg.worker(cfg, data, "P_s42", deadline=float("inf"), stop_path=stop)
    local = Path(cfg["output_root"]) / cfg["group"] / "P_s42"
    checkpoint = torch.load(local / "last.pt", weights_only=False)
    assert checkpoint["epoch"] == 0 and checkpoint["cursor"] == 16
    assert checkpoint["config"]["batch_size"] == 1 and checkpoint["config"]["grad_accum"] == 16
    assert not (local / "completed.json").exists()
    Path(stop).unlink()
    monkeypatch.setattr(kg.ct.Trainer, "fit", original_fit)
    monkeypatch.setattr(kg, "probe_batch", lambda *a, **kw: pytest.fail("Resume must not probe"))
    kg.worker(cfg, data, "P_s42", deadline=float("inf"), stop_path=stop)
    resumed = torch.load(local / "last.pt", weights_only=False)
    assert resumed["epoch"] == 1 and resumed["step"] == checkpoint["step"] + 1
    assert resumed["scaler"] == checkpoint["scaler"]
    assert (local / "completed.json").exists()
    assert runs == [{"exit_code": 0}] * 4
    baseline_cfg = {**cfg, "output_root": str(Path(cfg["output_root"]) / "baseline")}
    kg.worker(baseline_cfg, data, "P_s42", deadline=float("inf"), stop_path=stop)
    baseline_path = Path(baseline_cfg["output_root"]) / cfg["group"] / "P_s42" / "last.pt"
    baseline = torch.load(baseline_path, weights_only=False)
    for key, value in resumed["model"].items():
        torch.testing.assert_close(value, baseline["model"][key], rtol=0, atol=0)


def test_lock_duplicate_and_stale(tmp_path):
    path = tmp_path / "job.lock"
    with kg.run_lock(path):
        saved = json.loads(path.read_text())
        assert saved["pid"] == os.getpid() and saved["create_time"] > 0
        with pytest.raises(RuntimeError, match="Duplicate"):
            with kg.run_lock(path):
                pass
    kg.write_json(path, {"pid": os.getpid(), "create_time": 0})
    with kg.run_lock(path):
        assert json.loads(path.read_text())["create_time"] > 0
    assert not path.exists()


def test_hydrate_only_target_no_input_writes(cfg, tmp_path):
    source = tmp_path / "attached"
    data = prepared(cfg)
    bundle(cfg, data, source / "P_s42")
    (source / "B1_s42").mkdir()
    (source / "P_s42" / "Training_volumes.npy").write_bytes(b"never-copy")
    (source / "B1_s42" / "last.pt").write_bytes(b"other-run")
    cfg["resume_root"] = str(source)
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    local = kg.hydrate(cfg, "P_s42")
    assert (local / "last.pt").is_file() and (local / "hydrated.json").is_file()
    assert not (local / "Training_volumes.npy").exists()
    assert not (local.parent / "B1_s42").exists()
    assert all(path.read_bytes() == value for path, value in before.items())


def test_output_preflight(cfg, monkeypatch):
    cfg.update(smoke=False, output_limit_gib=10, job_reserve_gib=6)
    monkeypatch.setattr(kg, "require_disk", lambda *args: None)
    kg.output_preflight(cfg, 0)
    with pytest.raises(OSError, match="quota"):
        kg.output_preflight(cfg, 1)


def test_dispatch_two_distinct_workers_budget_and_owned_stop(cfg, monkeypatch):
    cfg.update(smoke=False, max_runs_per_session=2)
    monkeypatch.setattr(kg, "validate_config", lambda cfg: None)
    monkeypatch.setattr(kg, "output_preflight", lambda *args: None)
    detect = kg.detect_gpus
    monkeypatch.setattr(kg, "detect_gpus", lambda *args: detect(listing="0, T4, 15360\n1, T4, 15360"))
    calls = []

    class Child:
        def __init__(self, args, **kwargs):
            self.pid = 90000 + len(calls)
            self.polls = 0
            calls.append((args, kwargs))

        def poll(self):
            self.polls += 1
            return 0 if self.polls > 1 else None

        def wait(self):
            return 0

    monkeypatch.setattr(kg.subprocess, "Popen", Child)
    dispatched = kg.dispatch(cfg, {})
    assert dispatched == ["P_s42", "B1_s42"]
    assert len(calls) == 2
    assert {kwargs["env"]["CUDA_VISIBLE_DEVICES"] for _, kwargs in calls} == {"0", "1"}
    assert not kg.OWNED_CHILDREN
    for args, _ in calls:
        job = json.loads(Path(args[-1]).read_text())
        assert "WANDB_API_KEY" not in json.dumps(job) and "HF_TOKEN" not in json.dumps(job)


def test_stop_tracks_alive_children(cfg, monkeypatch):
    alive = SimpleNamespace(poll=lambda: None)
    kg.OWNED_CHILDREN[12345] = alive
    stop = Path(cfg["output_root"]) / "STOP.json"
    try:
        with pytest.raises(RuntimeError, match="STILL ALIVE"):
            kg.wait_owned(stop, 0)
        assert kg.OWNED_CHILDREN == {12345: alive} and stop.is_file()
        alive.poll = lambda: 0
        alive.wait = lambda: None
        kg.wait_owned(stop, 1)
        assert not kg.OWNED_CHILDREN
    finally:
        kg.OWNED_CHILDREN.clear()


def test_partial_summary(cfg):
    root = Path(cfg["output_root"]) / cfg["group"]
    bundle(cfg, prepared(cfg), root / "P_s42", completed=True)
    summary = kg.aggregate(cfg)["summary"]["P"]
    assert summary["n_seeds"] == 1 and summary["partial"]
    assert summary["metrics"]["auc_roc"] == {"mean": 0.75, "std": None, "n": 1}


@pytest.mark.parametrize("damage", ["missing", "empty", "identity", "last", "truncated"])
def test_requested_attached_recovery_never_starts_fresh(cfg, tmp_path, damage, monkeypatch):
    source = tmp_path / "attached" / "P_s42"
    if damage == "empty":
        source.mkdir(parents=True)
    elif damage not in ("missing", "empty"):
        bundle(cfg, prepared(cfg), source)
        if damage == "truncated":
            (source / "last.pt").write_bytes(b"partial")
        else:
            (source / ("run_identity.pt" if damage == "identity" else "last.pt")).unlink()
    cfg["resume_root"] = str(source.parent)
    monkeypatch.setattr(kg, "make_model", lambda *a, **k: pytest.fail("Fresh model after broken recovery"))
    with pytest.raises((ValueError, RuntimeError, EOFError, kg.pickle.UnpicklingError)):
        kg.hydrate(cfg, "P_s42")
    assert not (Path(cfg["output_root"]) / cfg["group"] / "P_s42" / "hydrated.json").exists()


@pytest.mark.parametrize("damage", ["empty", "missing-last", "wrong-seed", "wrong-group", "wrong-data", "wrong-run-id"])
def test_cloud_restore_rejects_bad_provenance(cfg, tmp_path, monkeypatch, damage):
    source = tmp_path / "cloud"
    if damage == "empty":
        source.mkdir()
    else:
        bundle(cfg, prepared(cfg), source)
        if damage == "missing-last":
            (source / "last.pt").unlink()
        elif damage in ("wrong-seed", "wrong-group", "wrong-data"):
            identity = json.loads((source / "run_identity.json").read_text())
            if damage == "wrong-seed":
                identity["config"]["seed"] = 43
            elif damage == "wrong-group":
                identity["config"]["group"] = "other-study"
            else:
                identity["config"]["data"]["Training"]["identity"]["revision"] = "0" * 40
            kg.write_json(source / "run_identity.json", identity)
            descriptor = json.loads((source / "recovery.json").read_text())
            descriptor["files"]["run_identity.json"] = kd.sha256(source / "run_identity.json")
            descriptor["identity_digest"] = kg.digest(identity)
            kg.write_json(source / "recovery.json", descriptor)
        else:
            state = torch.load(source / "last.pt", weights_only=False)
            state["run_id"] = "different-run"
            kg.ft.atomic_save(state, source / "last.pt")
            descriptor = json.loads((source / "recovery.json").read_text())
            descriptor["files"]["last.pt"] = kd.sha256(source / "last.pt")
            descriptor["checkpoint"] = kg.checkpoint_header(state)
            kg.write_json(source / "recovery.json", descriptor)
    calls = []
    fake_cloud(monkeypatch, {"ref": source}, calls)
    cfg["artifact_refs"] = {"P_s42": "ref"}
    with pytest.raises(ValueError):
        kg.hydrate(cfg, "P_s42")
    assert all(name != "last.pt" for _, name in calls)
    assert not list(Path(cfg["temp_root"]).glob("restore-*"))


def test_cloud_hydrate_retry_is_transactional_and_never_deletes_attachment(cfg, tmp_path, monkeypatch):
    data = prepared(cfg)
    cloud, attached = tmp_path / "cloud", tmp_path / "attached"
    identity = bundle(cfg, data, cloud)
    bundle(cfg, data, attached / "P_s42")
    sentinel = attached / "P_s42" / "user-file.txt"
    sentinel.write_text("must survive")
    source_hashes = {p: kd.sha256(p) for p in attached.rglob("*") if p.is_file()}
    cfg.update(artifact_refs={"P_s42": "ref"}, resume_root=str(attached))
    calls = []
    fake_cloud(monkeypatch, {"ref": cloud}, calls)
    local = Path(cfg["output_root"]) / cfg["group"] / "P_s42"
    kg.ft.atomic_save(identity, local / "run_identity.pt")
    (local / "last.pt").write_bytes(b"partial checkpoint")
    original_copy = shutil.copyfile
    copies = []

    def interrupted(src, dst, *args, **kwargs):
        copies.append(str(dst))
        if "-hydrate-" in str(dst):
            raise OSError("simulated copy interruption")
        return original_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", interrupted)
    with pytest.raises(OSError, match="interruption"):
        kg.hydrate(cfg, "P_s42", data)
    assert (local / "last.pt").read_bytes() == b"partial checkpoint"
    assert not (local / "hydrated.json").exists()
    monkeypatch.setattr(shutil, "copyfile", original_copy)
    kg.hydrate(cfg, "P_s42", data)
    assert kg.validate_run_dir(cfg, "P_s42", local, data)[0] == identity
    before = len(calls)
    kg.hydrate(cfg, "P_s42", data)
    assert len(calls) == before
    assert all(p.exists() and kd.sha256(p) == h for p, h in source_hashes.items())
    assert not list(Path(cfg["temp_root"]).glob("restore-*"))
    assert not list(local.parent.glob(".P_s42-hydrate-*"))


def test_initialized_recovery_requires_stage_and_authorization(cfg, tmp_path, monkeypatch):
    cloud = tmp_path / "cloud"
    bundle(cfg, prepared(cfg), cloud, initialized=True)
    fake_cloud(monkeypatch, {"ref": cloud}, [])
    cfg["artifact_refs"] = {"P_s42": "ref"}
    with pytest.raises(ValueError, match="not authorized"):
        kg.hydrate(cfg, "P_s42")
    cfg["allow_initialized_recovery"] = True
    local = kg.hydrate(cfg, "P_s42")
    assert not (local / "last.pt").exists()
    assert kg.validate_run_dir(cfg, "P_s42", local)[0]["config"]["seed"] == 42
    (local / "initial_stage.json").unlink()
    with pytest.raises(ValueError, match="stage metadata"):
        kg.validate_run_dir(cfg, "P_s42", local)


def test_hydration_promotion_failure_rolls_back_partial_destination(cfg, tmp_path, monkeypatch):
    data = prepared(cfg)
    source = tmp_path / "attached" / "P_s42"
    identity = bundle(cfg, data, source)
    cfg["resume_root"] = str(source.parent)
    local = Path(cfg["output_root"]) / cfg["group"] / "P_s42"
    kg.ft.atomic_save(identity, local / "run_identity.pt")
    before = (local / "run_identity.pt").read_bytes()
    original_replace = os.replace

    def fail_promotion(src, dst):
        if Path(src).is_dir() and "-hydrate-" in Path(src).name and Path(dst) == local:
            raise OSError("promotion failed")
        return original_replace(src, dst)

    monkeypatch.setattr(kg.os, "replace", fail_promotion)
    with pytest.raises(OSError, match="promotion failed"):
        kg.hydrate(cfg, "P_s42", data)
    assert (local / "run_identity.pt").read_bytes() == before
    assert not (local / "last.pt").exists() and not (local / "hydrated.json").exists()
    assert not list(local.parent.glob(".P_s42-previous-*"))
    assert not list(local.parent.glob(".P_s42-hydrate-*"))
    assert (source / "last.pt").is_file()


@pytest.mark.parametrize(
    "change",
    [
        {"max_runs_per_session": True},
        {"max_runs_per_session": 1.5},
        {"extra": {"WANDB_API_KEY": "must-not-be-written"}},
    ],
)
def test_bad_cap_and_credentials_are_rejected_before_job_serialization(cfg, change):
    cfg.update(change)
    with pytest.raises(ValueError):
        kg.validate_config(cfg)
    assert not list(Path(cfg["output_root"]).glob("**/*.job.json"))


@pytest.mark.parametrize("damage", ["tag", "seed", "run-id", "protocol", "weights", "metrics"])
def test_completed_skip_and_aggregate_require_provenance(cfg, tmp_path, monkeypatch, damage):
    data = prepared(cfg)
    local = Path(cfg["output_root"]) / cfg["group"] / "P_s42"
    bundle(cfg, data, local, completed=True)
    record = json.loads((local / "completed.json").read_text())
    if damage == "tag":
        record["tag"] = "B1_s42"
    elif damage == "seed":
        record["identity"]["config"]["seed"] = 43
    elif damage == "run-id":
        record["checkpoint"]["run_id"] = "wrong-id"
    elif damage == "protocol":
        record["protocol_digest"] = "0" * 64
    elif damage == "weights":
        (local / "best_weights.pt").write_bytes(b"different")
    else:
        record["metrics"]["auc_roc"] = 1.0
    kg.write_json(local / "completed.json", record)
    monkeypatch.setattr(kg, "make_model", lambda *a, **kw: pytest.fail("Bad completion reached GPU/model"))
    with pytest.raises(ValueError):
        kg.worker(cfg, data, "P_s42", deadline=float("inf"), stop_path=str(tmp_path / "stop"))
    with pytest.raises(ValueError):
        kg.aggregate(cfg, data)


def test_cloud_completed_refs_are_resolved_before_cap(cfg, tmp_path, monkeypatch):
    cfg["max_runs_per_session"] = 2
    data = prepared(cfg)
    sources = {}
    for tag in ("P_s42", "B1_s42"):
        source = tmp_path / f"cloud-{tag}"
        bundle(cfg, data, source, tag, completed=True)
        sources[tag] = source
    cfg["artifact_refs"] = {tag: tag for tag in sources}
    calls = []
    fake_cloud(monkeypatch, sources, calls)
    selected, _ = kg.session_jobs(cfg, data)
    assert [tag for _, _, tag in selected] == ["B2_s42", "B3_s42"]
    assert all(name in {"recovery.json", "run_identity.json", "completed.json"} for _, name in calls)
    assert not list((Path(cfg["output_root"]) / cfg["group"]).glob("*/last.pt"))
    summary = kg.aggregate(cfg, data)
    assert set(summary["runs"]) == {"P_s42", "B1_s42"}
    assert summary["summary"]["P"]["n_seeds"] == 1


def test_two_gpu_dispatch_refills_after_cloud_completions(cfg, tmp_path, monkeypatch):
    data = prepared(cfg)
    cfg.update(smoke=False, max_runs_per_session=2)
    sources = {}
    for tag in ("P_s42", "B1_s42"):
        sources[tag] = tmp_path / tag
        bundle(cfg, data, sources[tag], tag, completed=True)
    cfg["artifact_refs"] = {tag: tag for tag in sources}
    downloads = []
    fake_cloud(monkeypatch, sources, downloads)
    monkeypatch.setattr(kg, "validate_config", lambda *a: None)
    monkeypatch.setattr(kg, "output_preflight", lambda *a: None)
    devices = kg.detect_gpus(listing="0, T4, 15360\n1, T4, 15360")
    monkeypatch.setattr(kg, "detect_gpus", lambda *a: devices)
    launches = []

    class Child:
        def __init__(self, args, **kwargs):
            self.pid = 123000 + len(launches)
            self.polls = 0
            launches.append((json.loads(Path(args[-1]).read_text()), kwargs["env"]))

        def poll(self):
            self.polls += 1
            return 0 if self.polls > 1 else None

        def wait(self):
            return 0

    monkeypatch.setattr(kg.subprocess, "Popen", Child)
    assert kg.dispatch(cfg, data) == ["B2_s42", "B3_s42"]
    assert {env["CUDA_VISIBLE_DEVICES"] for _, env in launches} == {"0", "1"}
    assert len(launches) == 2 and not kg.OWNED_CHILDREN
    assert all(name != "last.pt" for _, name in downloads)


def test_metadata_only_summary_rejects_missing_or_changed_provenance(cfg):
    data = prepared(cfg)
    local = Path(cfg["output_root"]) / cfg["group"] / "P_s42"
    bundle(cfg, data, local, completed=True)
    summary = kg.aggregate(cfg)
    shutil.rmtree(local)
    assert kg.aggregate(cfg)["runs"] == summary["runs"]
    broken = copy.deepcopy(summary)
    broken["runs"]["P_s42"] = {"auc_roc": 0.75}
    with pytest.raises(ValueError, match="provenance"):
        kg.validate_summary(cfg, broken)
    broken = copy.deepcopy(summary)
    broken["runs"]["P_s42"]["identity"]["config"]["data"]["Training"]["views"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        kg.validate_summary(cfg, broken)


def test_parallel_json_and_stop_writers_are_atomic_and_idempotent(tmp_path):
    target, stop = tmp_path / "shared.json", tmp_path / "STOP.json"

    def write(index):
        for _ in range(20):
            kg.write_json(target, {"index": index, "payload": "x" * 10000})
            kg.request_stop(stop)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(8)))
    assert json.loads(target.read_text())["payload"] == "x" * 10000
    original = stop.read_bytes()
    kg.request_stop(stop)
    assert stop.read_bytes() == original
    assert {p.name for p in tmp_path.iterdir()} == {"shared.json", "STOP.json"}


@pytest.mark.parametrize("damage", ["weights", "data", "labels", "truncated"])
def test_report_replay_rejects_stale_logits(cfg, tmp_path, damage):
    local = tmp_path / "report"
    bundle(cfg, prepared(cfg), local, completed=True)
    if damage == "weights":
        (local / "best_weights.pt").write_bytes(b"other weights")
    elif damage == "data":
        identity = json.loads((local / "run_identity.json").read_text())
        identity["config"]["data"]["Validation"]["identity"]["labels"] = "0" * 64
        kg.write_json(local / "run_identity.json", identity)
    elif damage == "labels":
        receipt = json.loads((local / "logits_receipt.json").read_text())
        receipt["labels"]["val"] = "0" * 64
        kg.write_json(local / "logits_receipt.json", receipt)
    else:
        (local / "logits.npz").write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="Stale|corrupt|mismatch"):
        kg.report_best(
            None,
            [SimpleNamespace(train=False) for _ in range(3)],
            kg.KaggleArtifacts(local, smoke=True),
            batch_size=1,
            smoke=True,
        )


def test_runtime_wandb_precedes_model_and_probe(cfg, monkeypatch):
    data = prepared(cfg)
    events = []
    original_init = kg.ft.init_wandb

    def init(name, config, *args, **kwargs):
        events.append(config.get("scope", "scientific"))
        return original_init(name, config, *args, **kwargs)

    monkeypatch.setattr(kg.ft, "load_env_file", lambda: None)
    monkeypatch.setattr(kg.ft, "init_wandb", init)

    def fail(*args, **kwargs):
        events.append("model")
        raise RuntimeError("model construction failed")

    monkeypatch.setattr(kg, "make_model", fail)
    with pytest.raises(RuntimeError, match="construction"):
        kg.worker(cfg, data, "P_s42", deadline=float("inf"), stop_path=str(Path(cfg["output_root"]) / "stop"))
    assert events == ["runtime-probe", "model"]
    assert not (Path(cfg["output_root"]) / cfg["group"] / "P_s42" / "run_identity.pt").exists()


def test_report_replay_logs_linked_run_and_preserves_completion(cfg, monkeypatch):
    root = Path(cfg["output_root"]) / cfg["group"]
    identity = bundle(cfg, prepared(cfg), root / "P_s42", completed=True)
    original = (root / "P_s42" / "completed.json").read_bytes()
    calls, logs, finishes = [], [], []
    monkeypatch.setattr(kg, "load_credentials", lambda **kwargs: None)

    def init(name, config, artifacts, **kwargs):
        calls.append(config)
        return SimpleNamespace(
            id="analysis",
            summary={},
            log=lambda values: logs.append(values),
            finish=lambda **kwargs: finishes.append(kwargs),
        )

    monkeypatch.setattr(kg.ft, "init_wandb", init)
    monkeypatch.setattr(kg.ft, "predict", lambda *a, **kw: pytest.fail("Replay must not predict"))
    kg.rerender_reports(cfg)
    assert calls[0]["scope"] == "report-replay" and calls[0]["parent_run_id"] == identity["id"]
    assert any("report/split_table" in values for values in logs)
    assert finishes == [{"exit_code": 0}]
    assert (root / "_reports" / "P_s42" / "report.json").is_file()
    assert (root / "P_s42" / "completed.json").read_bytes() == original


def test_interrupted_npz_write_does_not_publish_partial_archive(cfg, tmp_path, monkeypatch):
    local = tmp_path / "report"
    bundle(cfg, prepared(cfg), local, completed=True)
    (local / "logits.npz").unlink()
    (local / "logits_receipt.json").unlink()
    monkeypatch.setattr(kg.ft, "load_weights", lambda *a: None)
    monkeypatch.setattr(
        kg.ft,
        "predict",
        lambda *a, **kw: (
            np.array([0.1, 0.9, 0.2, 0.8]),
            np.array([0, 1, 0, 1]),
            np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8]]),
        ),
    )

    def fail(stream, **kwargs):
        stream.write(b"partial ZIP")
        raise OSError("write interrupted")

    monkeypatch.setattr(kg.np, "savez", fail)
    with pytest.raises(OSError, match="interrupted"):
        kg.report_best(
            None,
            [SimpleNamespace(train=False) for _ in range(3)],
            kg.KaggleArtifacts(local, smoke=True),
            batch_size=1,
            smoke=True,
        )
    assert not (local / "logits.npz").exists()
    assert not (local / "logits_receipt.json").exists()
    assert not list(local.glob("logits.npz.tmp-*"))


def test_full_notebook_cpu_smoke(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith(("CTRL_", "WANDB_", "HF_"))}
    env.update(
        CTRL_KAGGLE_SMOKE="1",
        CTRL_KAGGLE_OUTPUT_ROOT=str(tmp_path / "working"),
        CTRL_KAGGLE_TEMP_ROOT=str(tmp_path / "temp"),
        CUDA_VISIBLE_DEVICES="",
        WANDB_SILENT="true",
        PYTHONIOENCODING="utf-8",
        PYTHONPATH=str(ROOT),
    )
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())], env=env, cwd=ROOT, text=True, capture_output=True, timeout=240
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "KAGGLE SEVEN-SPEC SMOKE OK" in result.stdout


def offline_smoke():
    kg.ft.load_env_file = lambda: None
    socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(AssertionError("Network forbidden"))
    import huggingface_hub

    def forbidden(*args, **kwargs):
        raise AssertionError("Network/GPU forbidden in smoke")

    huggingface_hub.snapshot_download = huggingface_hub.hf_hub_download = forbidden
    torch.cuda.init = forbidden
    namespace = {"__name__": "kaggle_smoke"}
    notebook = json.loads(NOTEBOOK.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "<kaggle-cell>", "exec"), namespace)
    cfg = namespace["CFG"]
    assert len(namespace["DISPATCHED"]) == 7
    root = Path(cfg["output_root"]) / cfg["group"]
    for spec in kg.SPECS:
        local = root / f"{spec['code']}_s42"
        assert (local / "completed.json").is_file()
        state = torch.load(local / "last.pt", map_location="cpu", weights_only=False)
        assert state["epoch"] == 1
        assert state["config"]["batch_size"] * state["config"]["grad_accum"] == 16
        assert state["config"]["fp16_init_scale"] == 1024
        assert "test" in state["history"][0]
        assert (local / "best_weights.pt").is_file() and (local / "logits.npz").is_file()
        assert not list(local.glob("*.png"))
    kg.probe_batch = forbidden
    assert kg.dispatch(cfg, namespace["PREPARED"]) == []
    assert all(item["n_seeds"] == 1 for item in namespace["SUMMARY"]["summary"].values())
    print("KAGGLE SEVEN-SPEC SMOKE OK")


if __name__ == "__main__":
    offline_smoke()
