import ast
import copy
import hashlib
import inspect
import json
import shutil
import signal
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import final_execution as fx
from scripts import final_training as ft


@pytest.fixture
def prepared(monkeypatch, request):
    tmp_path = Path(tempfile.mkdtemp(prefix="fx_"))
    request.addfinalizer(lambda: shutil.rmtree(tmp_path))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    fx.release_active()
    config = dict(
        dataset="bilateral",
        seed=42,
        epochs=1,
        batch_size=2,
        grad_accum=2,
        lr=5e-5,
        weight_decay=1e-4,
        patience=5,
        checkpoint_steps=10,
        store_res=8,
        res3d=8,
        res2d=8,
        n2d=2,
        latent=256,
        enc2d="maxvit_tiny_rw_224",
        smoke=True,
        amp_dtype="float16",
        pin_memory=False,
        non_blocking=False,
        fused_adamw=False,
        telemetry=True,
        max_scaler_skips=3,
        preflight_windows=2,
        preflight_fraction=0.85,
        compile=False,
        activation_checkpointing=False,
    )
    parent = tmp_path / "parent/best_weights.pt"
    fx.smoke_parent(parent)
    context = dict(
        config=config,
        smoke=True,
        parent=str(parent),
        remote=str(tmp_path / "drive/run"),
        data_root=str(tmp_path / "data"),
        cache_root=str(tmp_path / "drive/data_cache/bilateral"),
    )
    artifacts = fx.workspace(tmp_path / "run", context["remote"], smoke=True, context=context)
    runs = []

    class Run:
        def __init__(self, identity):
            self.id = identity
            self.summary, self.logs, self.finishes = {}, [], []

        def log(self, values):
            self.logs.append(values)

        def finish(self, exit_code=0):
            self.finishes.append(exit_code)

    def init(name, config, artifacts, *, resume=False, smoke=False, warm_start=""):
        if resume:
            identity = fx.load(artifacts.restore("run_identity.pt"))
            assert identity["config"] == config
        else:
            identity = dict(id=f"run-{len(runs)}", config=config, warm_start=warm_start)
            artifacts.save(identity, "run_identity.pt")
        run = Run(identity["id"])
        runs.append(run)
        return run

    monkeypatch.setattr(ft, "init_wandb", init)
    fx.parent_stage(artifacts.local)
    fx.raw_stage(artifacts.local)
    fx.prepare_stage(artifacts.local, allow_build=True, publish_cache=True)
    fx.audit_data_stage(artifacts.local)
    yield SimpleNamespace(root=artifacts.local, context=context, artifacts=artifacts, runs=runs)
    fx.release_active()
    torch.set_num_threads(old_threads)


def test_numbered_sections_and_no_fit_in_analysis():
    path = Path(__file__).resolve().parents[1] / "notebooks/3d_glaucoma_final_2x2d_3d_crossgate.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    headings = ["".join(c["source"]).splitlines()[0] for c in notebook["cells"] if c["cell_type"] == "markdown"]
    assert len(headings) == 14
    for number, heading in enumerate(headings):
        assert heading.lstrip("# ").startswith(f"{number}.")
    for fn in (fx.evaluation_stage, fx.report_stage, fx.xai_stage, fx.preflight_stage):
        source = inspect.getsource(fn)
        assert "Trainer(" not in source and ".fit(" not in source
    assert "make_model" not in inspect.getsource(fx.report_stage)


def test_parent_missing_old_xai_marker_is_not_claimed_complete(prepared):
    parent = fx.verify_parent(prepared.context["parent"], smoke=True)
    assert parent["phase_status"] == "training_finished_reports_present_no_completion_marker"
    assert parent["source_epoch"] == 1 and parent["source_val_auc"] == 0.5
    assert parent["wandb_id"] == "synthetic-parent"
    assert len(parent["weights"]["sha256"]) == 64


@pytest.mark.parametrize("mutation", ["weights", "identity", "history", "epoch", "completed", "data"])
def test_parent_rejects_mismatches(prepared, mutation):
    path = Path(prepared.context["parent"])
    if mutation == "weights":
        state = torch.load(path, weights_only=True)
        state["head.bias"].add_(1)
        ft.atomic_save(state, path)
    elif mutation == "identity":
        state = fx.load(path.parent / "run_identity.pt")
        state["config"]["seed"] = 43
        ft.atomic_save(state, path.parent / "run_identity.pt")
    elif mutation == "data":
        state = fx.load(path.parent / "run_identity.pt")
        state["config"]["data"] = []
        ft.atomic_save(state, path.parent / "run_identity.pt")
    elif mutation == "history":
        (path.parent / "metrics.json").write_text(json.dumps(dict(tag="raw_s42", seed=42, hist=[])))
    elif mutation == "epoch":
        state = fx.load(path.parent / "last.pt")
        state["epoch"] = 0
        ft.atomic_save(state, path.parent / "last.pt")
    else:
        ft.atomic_save(dict(run_id="wrong", config={}), path.parent / "completed.pt")
    with pytest.raises(ValueError, match="Parent"):
        fx.verify_parent(path, smoke=True)


def test_parent_real_pinned_path_enforced(tmp_path):
    with pytest.raises(ValueError, match="pinned"):
        fx.verify_parent(tmp_path / "emergency_weights.pt")


def test_disabled_stages_do_not_touch_hardware_or_disk(monkeypatch, tmp_path):
    monkeypatch.setattr(fx, "hardware", lambda *a: pytest.fail("Disabled stage touched CUDA"))
    for fn in (fx.preflight_stage, fx.training_stage, fx.evaluation_stage, fx.xai_stage):
        assert fn(tmp_path)["status"] == "disabled"
    assert not list(tmp_path.iterdir())


def test_real_hardware_never_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback forbidden"):
        fx.hardware()
    assert fx.hardware(smoke=True)["device"] == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: SimpleNamespace(total_memory=40 * 1024**3))
    with pytest.raises(RuntimeError, match="70 GiB"):
        fx.hardware()


@pytest.mark.parametrize("bf16", [False, True])
def test_gpu_free_memory_gate_and_recorded_amp_selection(prepared, monkeypatch, bf16):
    context = copy.deepcopy(prepared.context)
    context["smoke"] = False
    prepared.artifacts.save(context, "context.pt")
    config = fx.load(prepared.root / "execution.pt")
    config.update(
        smoke=False,
        epochs=5,
        batch_size=2,
        grad_accum=8,
        store_res=200,
        res3d=200,
        res2d=224,
        pin_memory=True,
        non_blocking=True,
    )
    prepared.artifacts.save(config, "execution.pt")
    monkeypatch.setattr(
        fx, "hardware", lambda smoke: dict(device="cuda:0", total_memory=80 * 1024**3, bf16_supported=bf16)
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (40 * 1024**3, 80 * 1024**3))
    monkeypatch.setattr(fx, "verify_parent", lambda *a, **k: config["parent"])
    monkeypatch.setattr(fx, "datasets", lambda *a, **k: [object(), object(), object()])
    monkeypatch.setattr(fx, "make_model", lambda *a, **k: pytest.fail("Free-memory gate must precede allocation"))
    with pytest.raises(RuntimeError, match="Insufficient free GPU"):
        fx.preflight_stage(prepared.root, enabled=True)
    assert fx.load(prepared.root / "execution.pt")["amp_dtype"] == ("bfloat16" if bf16 else "float16")


def test_preflight_restores_rng_and_never_writes_training_checkpoint(prepared):
    snapshot = ft.rng_state()
    result = fx.preflight_stage(prepared.root, enabled=True)
    assert result["status"] == "passed" and result["micro_steps"] == 4
    torch.testing.assert_close(torch.get_rng_state(), snapshot["torch"], rtol=0, atol=0)
    np.testing.assert_equal(np.random.get_state(), snapshot["numpy"])
    assert not (prepared.root / "train/last.pt").exists()
    assert fx.ACTIVE_MODEL is fx.ACTIVE_TRAINER is None
    assert prepared.runs[-1].finishes == [0]
    assert (prepared.root / "preflight/completed.pt").exists()


def test_failed_preflight_has_no_success_and_restores_rng(prepared, monkeypatch):
    snapshot = ft.rng_state()

    def fail(*args):
        torch.rand(3)
        raise RuntimeError("allocation failed")

    monkeypatch.setattr(fx, "make_model", fail)
    with pytest.raises(RuntimeError, match="allocation failed"):
        fx.preflight_stage(prepared.root, enabled=True)
    assert fx.load(prepared.root / "preflight.pt")["status"] == "failed"
    assert not (prepared.root / "train/last.pt").exists()
    assert not (prepared.root / "preflight/completed.pt").exists()
    torch.testing.assert_close(torch.get_rng_state(), snapshot["torch"], rtol=0, atol=0)
    assert prepared.runs[-1].finishes == [1]


def test_training_requires_exact_successful_preflight_and_existing_resume(prepared, monkeypatch):
    monkeypatch.setattr(ft.Trainer, "fit", lambda *a, **k: pytest.fail("Gate must prevent fit"))
    with pytest.raises(FileNotFoundError):
        fx.training_stage(prepared.root, enabled=True)
    fx.preflight_stage(prepared.root, enabled=True)
    with pytest.raises(ValueError, match="existing safe last"):
        fx.training_stage(prepared.root, enabled=True, resume=True)
    original = fx.load(prepared.root / "execution.pt")
    changed = copy.deepcopy(original)
    changed["max_scaler_skips"] = 2
    prepared.artifacts.save(changed, "execution.pt")
    with pytest.raises(ValueError, match="exact config"):
        fx.training_stage(prepared.root, enabled=True)
    assert fx.binding(original, {"device": "cuda:0", "uuid": "a"}) != fx.binding(
        original, {"device": "cuda:0", "uuid": "b"}
    )


def test_full_source_audit_includes_view_and_depth_hashes(prepared):
    config = fx.load(prepared.root / "execution.pt")
    for entry in config["data"]["splits"].values():
        assert set(entry["files"]) == {"denoised", "views", "dzs", "labels"}
        assert all(len(value["sha256"]) == 64 for value in entry["files"].values())
    path = Path(prepared.context["data_root"]) / "Validation_volumes_dn_dzs_8.npy"
    values = np.load(path)
    values[0] = (values[0] + 1) % 3
    np.save(path, values)
    with pytest.raises(ValueError, match="manifest mismatch"):
        fx.datasets(prepared.root, config)


def test_training_failure_retains_active_and_blocks_duplicate(prepared, monkeypatch):
    fx.preflight_stage(prepared.root, enabled=True)
    original = ft.Trainer.fit

    def fail(self, evaluate):
        self.commit()
        raise RuntimeError("training failure")

    monkeypatch.setattr(ft.Trainer, "fit", fail)
    with pytest.raises(RuntimeError, match="training failure"):
        fx.training_stage(prepared.root, enabled=True)
    assert fx.ACTIVE_MODEL is not None and fx.ACTIVE_TRAINER is not None
    with pytest.raises(RuntimeError, match="ACTIVE_MODEL"):
        fx.preflight_stage(prepared.root, enabled=True)
    safe = (prepared.root / "train/last.pt").read_bytes()
    fx.release_active()
    assert (prepared.root / "train/last.pt").read_bytes() == safe
    monkeypatch.setattr(ft.Trainer, "fit", original)
    fx.training_stage(prepared.root, enabled=True, resume=True)
    assert fx.ACTIVE_MODEL is fx.ACTIVE_TRAINER is None


def test_evaluation_resumes_each_split_and_report_fresh_runtime_without_training(prepared, monkeypatch):
    fx.preflight_stage(prepared.root, enabled=True)
    fx.training_stage(prepared.root, enabled=True)
    predict = fx.prediction
    calls = []

    def fail_test(model, dataset, config):
        calls.append(dataset.source.name)
        if dataset.source.name.startswith("Test"):
            raise RuntimeError("test inference interrupted")
        return predict(model, dataset, config)

    monkeypatch.setattr(fx, "prediction", fail_test)
    with pytest.raises(RuntimeError, match="inference interrupted"):
        fx.evaluation_stage(prepared.root, enabled=True)
    assert (prepared.root / "evaluation/val_logits.pt").exists()
    assert not (prepared.root / "evaluation/completed.pt").exists()
    with pytest.raises(FileNotFoundError):
        fx.report_stage(prepared.root)
    monkeypatch.setattr(ft.Trainer, "fit", lambda *a, **k: pytest.fail("Evaluation retrained"))

    def remaining(model, dataset, config):
        assert dataset.source.name.startswith("Test")
        return predict(model, dataset, config)

    monkeypatch.setattr(fx, "prediction", remaining)
    fx.evaluation_stage(prepared.root, enabled=True)
    fresh = prepared.root.parent / "fresh"
    fx.workspace(fresh, prepared.context["remote"], smoke=True, context=prepared.context)
    monkeypatch.setattr(fx, "make_model", lambda *a, **k: pytest.fail("CPU report allocated a model"))
    monkeypatch.setattr(fx, "datasets", lambda *a, **k: pytest.fail("CPU report loaded data"))
    monkeypatch.setattr(fx, "hardware", lambda *a, **k: pytest.fail("CPU report touched hardware"))
    report = fx.report_stage(fresh)
    assert report["calibration_protocol"]["selection_split"] == "val"
    assert fx.completion_stage(fresh)["status"] == "analysis_complete"
    assert not (fresh / "train/last.pt").exists()
    _, analysis = fx.open_stage(fresh, "analysis")
    before = fx.validated_completion(analysis)
    with pytest.raises(pytest.fail.Exception, match="loaded data"):
        fx.xai_stage(prepared.root, enabled=True)
    assert fx.validated_completion(analysis) == before


def test_completion_requires_hashes_and_verifies_before_wandb_success(prepared, monkeypatch):
    artifacts, run = fx.stage_run(prepared.root, "analysis", prepared.context["config"], "train-linked")
    artifacts.save({"value": 1}, "scientific.pt")
    original = run.finish

    def finish(exit_code=0):
        assert (artifacts.remote / "scientific.pt").exists()
        assert (artifacts.remote / "audit.pt").exists()
        assert not (artifacts.local / "completed.pt").exists()
        original(exit_code)

    monkeypatch.setattr(run, "finish", finish)
    fx.finish_stage(artifacts, run, {"train_id": "train-linked"})
    assert fx.validated_completion(artifacts)["status"] == "complete"
    ft.atomic_save({"changed": True}, artifacts.local / "scientific.pt")
    with pytest.raises(ValueError, match="artifact changed"):
        fx.validated_completion(artifacts)


def test_partial_metrics_never_publish_summary(prepared):
    _, artifacts = fx.open_stage(prepared.root, "analysis")
    (artifacts.local / "metrics.json").write_text('{"test": {"acc": 1}}')
    with pytest.raises(FileNotFoundError):
        fx.completion_stage(prepared.root)
    assert not (prepared.root / "final_summary.pt").exists()


def test_prediction_checksum_rejects_tampering_before_reuse(prepared):
    fx.preflight_stage(prepared.root, enabled=True)
    fx.training_stage(prepared.root, enabled=True)
    fx.evaluation_stage(prepared.root, enabled=True)
    path = prepared.root / "evaluation/val_logits.pt"
    saved = fx.load(path)
    saved["logits"][0, 0] += 0.1
    ft.atomic_save(saved, path)
    with pytest.raises(ValueError, match="checksum"):
        fx.evaluation_stage(prepared.root, enabled=True)
    assert prepared.runs[-1].finishes == [1]
    assert not (prepared.root / "evaluation/completed.pt").exists()


def test_preflight_wandb_failure_prevents_training_publication(prepared, monkeypatch):
    stage_run = fx.stage_run

    def fail_run(*args, **kwargs):
        artifacts, run = stage_run(*args, **kwargs)

        def fail(values):
            raise RuntimeError("W&B logging failed")

        run.log = fail
        return artifacts, run

    monkeypatch.setattr(fx, "stage_run", fail_run)
    with pytest.raises(RuntimeError, match="W&B logging failed"):
        fx.preflight_stage(prepared.root, enabled=True)
    assert fx.load(prepared.root / "preflight.pt")["status"] == "publication_failed"
    assert not (prepared.root / "preflight/completed.pt").exists()
    with pytest.raises(ValueError, match="Successful preflight"):
        fx.training_stage(prepared.root, enabled=True)


def test_checked_drive_copy_failure_never_finishes_success(prepared, monkeypatch):
    artifacts, run = fx.stage_run(prepared.root, "analysis", prepared.context["config"], "train-linked")
    ft.atomic_save({"scientific": True}, artifacts.local / "result.pt")

    def fail(*args):
        raise OSError("Drive disconnected")

    monkeypatch.setattr(fx.fd, "sync_verified", fail)
    with pytest.raises(RuntimeError, match="local artifact retained"):
        fx.finish_stage(artifacts, run, {})
    assert run.finishes == []
    assert (artifacts.local / "result.pt").exists()
    assert not (artifacts.local / "completed.pt").exists()
    assert list(artifacts.local.rglob("*.sync-pending.pt"))


def test_completion_marker_copy_failure_recovers_without_retraining(prepared, monkeypatch):
    artifacts, run = fx.stage_run(prepared.root, "analysis", prepared.context["config"], "train-linked")
    artifacts.save({"scientific": True}, "result.pt")
    sync = fx.fd.sync_verified

    def fail_marker(source, destination):
        if Path(source).name == "completed.pt":
            raise OSError("Drive disconnected after W&B finish")
        return sync(source, destination)

    monkeypatch.setattr(fx.fd, "sync_verified", fail_marker)
    with pytest.raises(RuntimeError, match="local artifact retained"):
        fx.finish_stage(artifacts, run, {"train_id": "train-linked"})
    assert run.finishes == [0]
    assert not (artifacts.remote / "completed.pt").exists()
    assert (artifacts.local / "completed.pt.sync-pending.pt").exists()
    monkeypatch.setattr(fx.fd, "sync_verified", sync)
    fx.validated_completion(artifacts)
    assert (artifacts.remote / "completed.pt").exists()
    assert not (artifacts.local / "completed.pt.sync-pending.pt").exists()


def test_bfloat16_scaler_disabled_nonfinite_loss_aborts_before_backward_and_sigterm(prepared):
    config = fx.load(prepared.root / "execution.pt")
    config["amp_dtype"] = "bfloat16"
    (dataset,) = fx.datasets(prepared.root, config, splits=("Training",))
    _, artifacts = fx.open_stage(prepared.root, "train")
    run = fx.start_run(artifacts, config, "train")
    model = ft.SmokeModel()
    trainer = ft.Trainer(model, dataset, config, artifacts, run)
    assert trainer.amp_dtype == torch.bfloat16 and not trainer.scaler.is_enabled()
    previous = signal.getsignal(signal.SIGTERM)
    with trainer.interrupts():
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert trainer.stop_requested
    assert signal.getsignal(signal.SIGTERM) == previous
    trainer.stop_requested = False
    with torch.no_grad():
        model.head.bias.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="before backward"):
        trainer.fit(lambda model: pytest.fail("Nonfinite loss reached evaluation"))
    assert all(p.grad is None for p in model.parameters())
    assert trainer.step == 0


def test_real_cpu_stage_creates_verified_session_parent(prepared):
    context = {**prepared.context, "smoke": False}
    prepared.artifacts.save(context, "context.pt")
    config = {**context["config"], "smoke": False}
    artifacts, run = fx.stage_run(prepared.root, "real_cpu_check", config, "parent-linked")
    assert (artifacts.remote / "sessions").is_dir()
    assert len(list((artifacts.remote / "sessions").glob("*/run_identity.pt"))) == 1
    fx.finish_stage(artifacts, run, {"mode": "CPU mock W&B, real storage guards"})
    assert run.finishes == [0]


@pytest.mark.parametrize("remote_only", [False, True])
def test_stranded_identity_resumes_same_wandb_with_fresh_parent_optimizer(prepared, monkeypatch, remote_only):
    fx.preflight_stage(prepared.root, enabled=True)
    config = fx.load(prepared.root / "execution.pt")
    _, artifacts = fx.open_stage(prepared.root, "train")
    run = fx.start_run(artifacts, config, "train", warm_start=prepared.context["parent"])
    if remote_only:
        (artifacts.local / "run_identity.pt").unlink()
    parent = torch.load(prepared.context["parent"], weights_only=True)
    original = ft.Trainer.fit

    def inspect_fresh(trainer, evaluate):
        assert trainer.run.id == run.id
        assert trainer.epoch == trainer.step == 0 and not trainer.optimizer.state
        assert trainer.history == []
        for name, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, parent[name], rtol=0, atol=0)
        return original(trainer, evaluate)

    monkeypatch.setattr(ft.Trainer, "fit", inspect_fresh)
    fx.training_stage(prepared.root, enabled=True, resume=True)
    assert fx.load(artifacts.local / "last.pt")["run_id"] == run.id


def test_stranded_identity_rejects_changed_parent_path(prepared):
    fx.preflight_stage(prepared.root, enabled=True)
    config = fx.load(prepared.root / "execution.pt")
    _, artifacts = fx.open_stage(prepared.root, "train")
    fx.start_run(artifacts, config, "train", warm_start="emergency_weights.pt")
    with pytest.raises(ValueError, match="exact verified pinned parent"):
        fx.training_stage(prepared.root, enabled=True, resume=True)


def test_parent_raw_identities_survive_and_changed_raw_blocks_before_filter(prepared, monkeypatch):
    parent = fx.load(prepared.root / "parent.pt")
    assert parent["data"] == fx.load(Path(prepared.context["parent"]).parent / "run_identity.pt")["config"]["data"]
    assert len(parent["data"]) == 6
    path = Path(prepared.context["data_root"]) / "Test_labels.npy"
    np.save(path, 1 - np.load(path))
    monkeypatch.setattr(fx.fd, "prepare_data", lambda *a, **k: pytest.fail("Different source reached denoising"))
    with pytest.raises(ValueError, match="Parent raw data mismatch"):
        fx.prepare_stage(prepared.root)
    with pytest.raises(ValueError, match="Parent raw data mismatch"):
        fx.raw_stage(prepared.root)


def hf_fixture(prepared, monkeypatch, *, bad_sha=False):
    import huggingface_hub

    monkeypatch.setenv("HF_TOKEN", "mock-token-no-network")

    context = {**prepared.context, "smoke": False}
    prepared.artifacts.save(context, "context.pt")
    data = Path(context["data_root"])
    payload = {f"{s}_{k}.npy": (data / f"{s}_{k}.npy").read_bytes() for s in fx.SPLITS for k in ("volumes", "labels")}
    calls = []

    class Api:
        def __init__(self, **kwargs):
            pass

        def dataset_info(self, repo, *, revision, files_metadata):
            calls.append((repo, revision, files_metadata))
            siblings = [
                SimpleNamespace(
                    rfilename=name,
                    size=len(value),
                    lfs=SimpleNamespace(sha256="0" * 64 if bad_sha else hashlib.sha256(value).hexdigest())
                    if name.endswith("_volumes.npy")
                    else None,
                    blob_id=hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest(),
                )
                for name, value in payload.items()
            ]
            return SimpleNamespace(sha="a" * 40, siblings=siblings)

    def download(**kwargs):
        assert kwargs["revision"] == "a" * 40
        assert len(calls) == 1
        for name in kwargs["allow_patterns"]:
            (data / name).write_bytes(payload[name])

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    return data, calls


def test_existing_raw_adoption_never_invents_revision(prepared, monkeypatch):
    data, calls = hf_fixture(prepared, monkeypatch)
    result = fx.raw_stage(prepared.root)
    assert result["source_repo"] is result["source_revision"] is None
    assert not calls and not (data / "source_manifest.json").exists()
    assert "parent_hashes_only" in result["verification"]


def test_pinned_download_resolves_once_and_parent_hash_checks(prepared, monkeypatch):
    data, calls = hf_fixture(prepared, monkeypatch)
    (data / "Training_volumes.npy").unlink()
    result = fx.raw_stage(prepared.root)
    assert result["source_revision"] == "a" * 40
    assert len(calls) == 1
    assert fx.load(prepared.root / "raw_download_plan.pt")["source_revision"] == "a" * 40
    assert json.loads((data / "source_manifest.json").read_text())["sources"] == fx.verify_raw_sources(prepared.root)
    (data / "Test_volumes.npy").unlink()
    fx.raw_stage(prepared.root)
    assert len(calls) == 1


def test_existing_bytes_verified_at_export_revision_without_downloading(prepared, monkeypatch):
    import huggingface_hub

    data, calls = hf_fixture(prepared, monkeypatch)
    export = prepared.root.parent / "export_revision"
    export.mkdir()
    fx.save_json({"identity": {"source_repo": fx.RAW_REPO, "source_revision": "a" * 40}}, export / "manifest.json")
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **k: pytest.fail("Matching raw bytes need no download")
    )
    result = fx.raw_stage(prepared.root, cpu_export_root=export)
    assert calls == [(fx.RAW_REPO, "a" * 40, True)]
    assert result["source_revision"] == "a" * 40 and (data / "source_manifest.json").exists()


def test_hf_revision_mismatch_rejected_before_download(prepared, monkeypatch):
    import huggingface_hub

    data, _ = hf_fixture(prepared, monkeypatch, bad_sha=True)
    (data / "Training_volumes.npy").unlink()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **k: pytest.fail("Wrong source was downloaded"))
    with pytest.raises(ValueError, match="differs from parent data"):
        fx.raw_stage(prepared.root)
    assert not (data / "source_manifest.json").exists()


def test_local_cache_publication_requires_explicit_consent_without_refilter(prepared, monkeypatch):
    cache = Path(prepared.context["cache_root"])
    shutil.rmtree(cache)
    monkeypatch.setattr(ft, "build_denoised", lambda *a, **k: pytest.fail("Verified cache should not be refiltered"))
    with pytest.raises(RuntimeError, match="publication is pending"):
        fx.prepare_stage(prepared.root)
    assert not cache.exists()
    with pytest.raises(ValueError, match="publication"):
        fx.audit_data_stage(prepared.root)
    fx.prepare_stage(prepared.root, publish_cache=True)
    assert cache.is_dir()
    assert fx.load(prepared.root / "preparation.pt")["published"]


def test_cpu_export_import_and_portable_restore_preserve_producer(prepared, monkeypatch):
    data = prepared.root.parent / "cpu_data"
    export = prepared.root.parent / "cpu_export"
    cache = prepared.root.parent / "cpu_cache"
    data.mkdir()
    export.mkdir()
    sources = {}
    for split in fx.SPLITS:
        for kind in ("volumes", "labels"):
            shutil.copyfile(Path(prepared.context["data_root"]) / f"{split}_{kind}.npy", data / f"{split}_{kind}.npy")
        raw = np.load(data / f"{split}_volumes.npy")
        labels = np.load(data / f"{split}_labels.npy")
        raw_id = ft.data_identity(data / f"{split}_volumes.npy")
        np.save(export / f"{split}_volumes_dn.npy", raw // 2)
        shutil.copyfile(data / f"{split}_labels.npy", export / f"{split}_labels.npy")
        sources[split] = dict(
            shape=list(raw.shape),
            dtype="uint8",
            offset=raw_id["size"] - raw.size,
            sample_bytes=8**3,
            size=raw_id["size"],
            filename=f"{split}_volumes.npy",
            repo_revision="a" * 40,
            labels_sha256=ft.data_identity(data / f"{split}_labels.npy")["sha256"],
            class_counts={str(i): int((labels == i).sum()) for i in (0, 1)},
            url=f"https://huggingface.co/datasets/{fx.RAW_REPO}/resolve/{'a' * 40}/{split}_volumes.npy",
        )
    identity = dict(
        source_repo=fx.RAW_REPO,
        source_revision="a" * 40,
        sources=sources,
        method="bilateral",
        params=fx.CPU_PARAMS,
        axis=0,
        implementation=fx.CPU_IMPLEMENTATION,
        filter_code_sha256=fx.fd._CPU_FILTER_HASH,
        skimage="producer-skimage",
        numpy="producer-numpy",
        output_dtype="uint8",
        resolution=200,
        quantization=fx.fd._CPU_QUANTIZATION,
        format=1,
    )
    markers = {}
    for split in fx.SPLITS:
        marker = dict(
            identity=identity,
            shape=sources[split]["shape"],
            dtype="uint8",
            file=f"{split}_volumes_dn.npy",
            complete=True,
            sha256=ft.data_identity(export / f"{split}_volumes_dn.npy")["sha256"],
        )
        fx.save_json(marker, export / f"{split}_complete.json")
        markers[split] = marker
    fx.save_json(dict(identity=identity, complete=True, total_volumes=22, splits=markers), export / "manifest.json")
    context = {**prepared.context, "data_root": str(data), "cache_root": str(cache)}
    prepared.artifacts.save(context, "context.pt")
    with pytest.raises(ValueError, match="source_manifest.json"):
        fx.prepare_stage(prepared.root, cpu_export_root=export)
    fx.save_json(
        dict(source_repo=fx.RAW_REPO, source_revision="a" * 40, sources=fx.verify_raw_sources(prepared.root)),
        data / "source_manifest.json",
    )
    monkeypatch.setattr(fx, "denoise_bilateral", lambda *a: pytest.fail("CPU export must never use cdm filter"))
    first = fx.prepare_stage(prepared.root, cpu_export_root=export, publish_cache=True)
    assert first["splits"]["Training"]["provenance"]["cpu_export"] == identity
    manifest_path = next(cache.glob("*/manifest.json"))
    monkeypatch.setattr(fx.fd, "_VERSIONS", {**fx.fd._VERSIONS, "torch": "different-consumer-torch"})
    restored = fx.prepare_stage(prepared.root, manifest_path=manifest_path)
    assert restored == first
    assert fx.audit_data_stage(prepared.root)["data"] == first


def test_reviewed_helper_hashes_are_exact_and_normalized():
    notebook = Path(__file__).resolve().parents[1] / "notebooks/3d_glaucoma_final_2x2d_3d_crossgate.ipynb"
    source = next(
        "".join(c["source"])
        for c in json.loads(notebook.read_text())["cells"]
        if c["cell_type"] == "code" and "EXPECTED_HELPER_SHA256 =" in "".join(c["source"])
    )
    values = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ("EXPECTED_HELPER_SHA256", "EXPECTED_BUNDLE_SHA256")
    }
    actual = fx.code_identity()
    assert values["EXPECTED_HELPER_SHA256"] == {name: actual[name] for name in fx.BUNDLE_FILES}
    assert values["EXPECTED_BUNDLE_SHA256"] == actual["bundle_sha256"]


def test_notebook_rejects_stale_imported_helpers_and_dirty_disk(tmp_path, monkeypatch):
    import sys

    notebook = Path(__file__).resolve().parents[1] / "notebooks/3d_glaucoma_final_2x2d_3d_crossgate.ipynb"
    source = next(
        "".join(c["source"])
        for c in json.loads(notebook.read_text())["cells"]
        if c["cell_type"] == "code" and "EXPECTED_HELPER_SHA256 =" in "".join(c["source"])
    )
    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "verify_reviewed_helpers"
    )
    helper = ast.get_source_segment(source, node)
    path = tmp_path / "scripts/final_training.py"
    path.parent.mkdir()
    path.write_bytes(b"VALUE = 1\r\n")
    sha = hashlib.sha256(b"VALUE = 1\n").hexdigest()
    expected = {"final_training.py": sha}
    scope = dict(
        Path=Path,
        hashlib=hashlib,
        json=json,
        sys=sys,
        REPO_ROOT=tmp_path,
        EXPECTED_HELPER_SHA256=expected,
        EXPECTED_BUNDLE_SHA256=fx.digest(expected),
    )
    exec(helper, scope)
    monkeypatch.setitem(sys.modules, "scripts", SimpleNamespace(__path__=[str(tmp_path / "scripts")]))
    for name in list(sys.modules):
        if name.rsplit(".", 1)[-1] == "final_training":
            monkeypatch.delitem(sys.modules, name)
    module = SimpleNamespace(__file__=str(path), _FINAL_REVIEWED_SOURCE_SHA256=sha)
    monkeypatch.setitem(sys.modules, "scripts.final_training", module)
    scope["verify_reviewed_helpers"]()
    module._FINAL_REVIEWED_SOURCE_SHA256 = "old"
    with pytest.raises(RuntimeError, match="restart the runtime"):
        scope["verify_reviewed_helpers"]()
    module._FINAL_REVIEWED_SOURCE_SHA256 = sha
    path.write_text("VALUE = 2\n")
    with pytest.raises(RuntimeError, match="missing/dirty/stale"):
        scope["verify_reviewed_helpers"]()


def test_pinned_timm_maxvit_cpu_construction_without_pretrained_download(monkeypatch):
    import timm

    create = timm.create_model
    calls = []

    def offline(name, **kwargs):
        assert kwargs["pretrained"] is False
        calls.append(name)
        return create(name, **kwargs)

    monkeypatch.setattr(timm, "create_model", offline)
    model = fx.make_model(dict(smoke=False, n2d=2, latent=256, enc2d="maxvit_tiny_rw_224"), "cpu")
    assert calls == ["maxvit_tiny_rw_224", "maxvit_tiny_rw_224"]
    assert len(model.enc2ds) == 2 and model.head.in_features == 256 and model.head.out_features == 2
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_information_phases_restore_independently_without_training_or_test_use(prepared, monkeypatch):
    fx.preflight_stage(prepared.root, enabled=True)
    fx.training_stage(prepared.root, enabled=True)
    fx.evaluation_stage(prepared.root, enabled=True)
    fx.report_stage(prepared.root)
    first = fx.information_stage(prepared.root, enabled=True, phase="collect")
    assert first["test_used"] is False
    _, analysis = fx.open_stage(prepared.root, "analysis")
    completed = fx.validated_completion(analysis)
    monkeypatch.setattr(fx, "make_model", lambda *a, **k: pytest.fail("CPU information phase allocated model"))
    monkeypatch.setattr(fx, "datasets", lambda *a, **k: pytest.fail("CPU information phase reread data"))
    monkeypatch.setattr(ft.Trainer, "fit", lambda *a, **k: pytest.fail("Information phase retrained"))
    for phase in ("probes", "estimators", "surrogates", "interactions", "plane", "report"):
        fx.information_stage(prepared.root, enabled=True, phase=phase)
    assert fx.validated_completion(analysis) == completed
    report = json.loads((prepared.root / "information/report/info_theory.json").read_text())
    assert report["test_used"] is False
    assert report["plane"]["best_checkpoint_only"]["epoch"] == 1
    assert not list((prepared.root / "information").rglob("*.png"))
    assert all(run.finishes == [0] for run in prepared.runs)
    assert fx.information_stage(prepared.root, enabled=True, phase="report")["options"] == report["options"]


def test_information_failure_does_not_invalidate_analysis(prepared, monkeypatch):
    from scripts import information_theory as it

    fx.preflight_stage(prepared.root, enabled=True)
    fx.training_stage(prepared.root, enabled=True)
    fx.evaluation_stage(prepared.root, enabled=True)
    fx.report_stage(prepared.root)
    fx.information_stage(prepared.root, enabled=True, phase="collect")
    monkeypatch.setattr(it, "estimate_mi", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("estimator failed")))
    with pytest.raises(RuntimeError, match="estimator failed"):
        fx.information_stage(prepared.root, enabled=True, phase="estimators")
    assert prepared.runs[-1].finishes == [1]
    ft.atomic_save(
        {"optional": "publication pending"}, prepared.root / "information/estimators/result.pt.sync-pending.pt"
    )
    assert fx.completion_stage(prepared.root)["status"] == "analysis_complete"
    assert fx.information_stage(prepared.root / "not_a_run")["status"] == "disabled"


def test_final_profile_rejects_epoch_extension_and_worker_drift(prepared):
    config = fx.load(prepared.root / "execution.pt")
    for key in ("extend_epochs", "num_workers"):
        with pytest.raises(ValueError, match="five-epoch profile"):
            fx.validate_profile({**config, key: 1})
