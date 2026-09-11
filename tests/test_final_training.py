import copy
import json
import signal
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import final_training as ft


class Run:
    id = "test-run"

    def __init__(self):
        self.logs = []

    def log(self, values):
        self.logs.append(values)


class Data(torch.utils.data.Dataset):
    deterministic_by_index = True

    def __init__(self, n=10):
        self.n, self.epoch = n, 0

    def __len__(self):
        return self.n

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        rng = np.random.default_rng([self.epoch, index])
        return (
            torch.tensor(rng.random((1, 3, 3, 3)), dtype=torch.float32),
            torch.tensor(rng.random((2, 1, 3, 3)), dtype=torch.float32),
            torch.tensor(index % 2),
        )


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(**overrides):
    return dict(
        epochs=2,
        batch_size=2,
        grad_accum=2,
        lr=0.001,
        weight_decay=0.01,
        patience=5,
        checkpoint_steps=1,
        seed=42,
        class_weights=[1.0, 2.0],
        **overrides,
    )


def evaluate(model):
    return {"auc_roc": 0.7, "f1": 0.5, "balanced_acc": 0.5, "mcc": 0.0}


def trainer(path, *, cfg=None, resume=False, model=None, dataset=None, warm_start=""):
    return ft.Trainer(
        model if model is not None else ft.SmokeModel(),
        dataset if dataset is not None else Data(),
        cfg or config(),
        ft.Artifacts(path, smoke=True),
        Run(),
        resume=resume,
        warm_start=warm_start,
    )


def test_resume_matches_uninterrupted_with_order_rng_and_remainder(tmp_path):
    torch.manual_seed(3)
    initial = ft.cpu_state(ft.SmokeModel())
    reference = trainer(tmp_path / "reference")
    reference.model.load_state_dict(initial)
    reference.fit(evaluate)
    interrupted = trainer(tmp_path / "interrupted")
    interrupted.model.load_state_dict(initial)

    def stop(current):
        current.stop_requested = True

    assert not interrupted.fit(evaluate, boundary_hook=stop)
    committed = torch.load(tmp_path / "interrupted" / "last.pt", weights_only=False)
    assert committed["cursor"] == 4
    resumed = trainer(tmp_path / "interrupted", resume=True)
    assert resumed.fit(evaluate)
    assert resumed.step == reference.step == 6
    assert resumed.scheduler.last_epoch == 6
    assert resumed.history == reference.history
    for key, value in reference.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[key], value, rtol=0, atol=0)
    assert (tmp_path / "interrupted" / "best.pt").exists()


def test_interrupted_partial_optimizer_never_overwrites_safe_checkpoint(tmp_path, monkeypatch):
    current = trainer(tmp_path)
    current.commit()
    safe = (tmp_path / "last.pt").read_bytes()

    def partial_step(*args, **kwargs):
        with torch.no_grad():
            next(current.model.parameters()).add_(100)
        raise KeyboardInterrupt

    monkeypatch.setattr(current.optimizer, "step", partial_step)
    with pytest.raises(KeyboardInterrupt):
        current.fit(evaluate)
    assert (tmp_path / "last.pt").read_bytes() == safe
    emergency = torch.load(next(tmp_path.glob("emergency_weights_*.pt")), weights_only=False)
    assert emergency["resumable"] is False
    with pytest.raises(RuntimeError, match="Failed trainer"):
        current.fit(evaluate)
    restored = trainer(tmp_path, resume=True)
    assert restored.cursor == restored.step == 0


def test_checkpoint_rejects_pending_gradients(tmp_path):
    current = trainer(tmp_path)
    next(current.model.parameters()).grad = torch.ones_like(next(current.model.parameters()))
    with pytest.raises(RuntimeError, match="pending gradients"):
        current.commit()
    assert not (tmp_path / "last.pt").exists()


def test_warm_start_is_weights_only_and_mismatch_rejected(tmp_path):
    original = ft.SmokeModel()
    path = tmp_path / "rescued.pt"
    ft.atomic_save(original.state_dict(), path)
    current = trainer(tmp_path / "new", warm_start=str(path))
    assert current.epoch == current.step == 0
    assert not current.optimizer.state
    assert current.history == []
    for key, value in original.state_dict().items():
        torch.testing.assert_close(current.model.state_dict()[key], value)
    current.commit()
    changed = config()
    changed["grad_accum"] = 3
    with pytest.raises(ValueError, match="mismatch"):
        trainer(tmp_path / "new", cfg=changed, resume=True)
    with pytest.raises(ValueError, match="choose one"):
        trainer(tmp_path / "new", resume=True, warm_start=str(path))
    with pytest.raises(FileExistsError, match="overwrite"):
        trainer(tmp_path / "new", warm_start=str(path))


def test_accumulation_matches_large_batch_including_short_final_batch(tmp_path):
    torch.manual_seed(4)
    model = ft.SmokeModel()
    small = config()
    small.update(epochs=1, batch_size=2, grad_accum=3)
    large = copy.deepcopy(small)
    large.update(batch_size=6, grad_accum=1)
    a = trainer(tmp_path / "small", cfg=small, model=copy.deepcopy(model), dataset=Data(9))
    b = trainer(tmp_path / "large", cfg=large, model=copy.deepcopy(model), dataset=Data(9))
    a.fit(evaluate)
    b.fit(evaluate)
    assert a.step == b.step == 2
    assert a.history[0]["loss"] == pytest.approx(b.history[0]["loss"], rel=1e-6)
    for key, value in a.model.state_dict().items():
        torch.testing.assert_close(value, b.model.state_dict()[key], atol=1e-7, rtol=1e-6)


def test_amp_skipped_step_does_not_advance_scheduler(tmp_path):
    current = trainer(tmp_path)

    class SkipScaler:
        value = 8.0

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def get_scale(self):
            return self.value

        def step(self, optimizer):
            pass

        def update(self):
            self.value /= 2

        def state_dict(self):
            return {}

    current.scaler = SkipScaler()
    current.fit(evaluate)
    assert current.step == current.scheduler.last_epoch == 0


def test_sigint_flag_second_interrupt_and_handler_restored(tmp_path):
    current = trainer(tmp_path)
    previous = signal.getsignal(signal.SIGINT)
    with pytest.raises(KeyboardInterrupt), current.interrupts():
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)
        assert current.stop_requested
        handler(signal.SIGINT, None)
    assert signal.getsignal(signal.SIGINT) == previous


def test_cloud_failure_retains_local_commit_and_pending_marker(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    artifacts = ft.Artifacts(tmp_path / "local", remote)

    def fail(*args):
        raise OSError("Drive disconnected")

    monkeypatch.setattr(ft.shutil, "copyfile", fail)
    with pytest.raises(RuntimeError, match="local artifact retained"):
        artifacts.save({"safe": True}, "last.pt")
    assert torch.load(artifacts.local / "last.pt", weights_only=False)["safe"]
    assert (artifacts.local / "last.pt.sync-pending.pt").exists()


def test_uncommitted_batches_replay_and_restore_dropout_rng(tmp_path):
    class DropoutModel(ft.SmokeModel):
        def forward(self, x, views):
            return super().forward(torch.nn.functional.dropout(x, 0.2, self.training), views)

    cfg = config()
    cfg["checkpoint_steps"] = 2
    torch.manual_seed(8)
    initial = DropoutModel()
    rng = ft.rng_state()
    baseline = trainer(tmp_path / "baseline", cfg=cfg, model=copy.deepcopy(initial))
    baseline.fit(evaluate)
    ft.restore_rng(rng)
    broken = trainer(tmp_path / "broken", cfg=cfg, model=copy.deepcopy(initial))

    def interrupt(current):
        if current.step == 1:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        broken.fit(evaluate, boundary_hook=interrupt)
    assert torch.load(tmp_path / "broken/last.pt", weights_only=False)["cursor"] == 0
    resumed = trainer(tmp_path / "broken", cfg=cfg, resume=True, model=DropoutModel())
    resumed.fit(evaluate)
    assert resumed.history == baseline.history
    for key, value in baseline.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[key], value, rtol=0, atol=0)


def test_data_identity_survives_copy_but_detects_changes(tmp_path):
    a, b = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(a, [1, 2, 3])
    ft.shutil.copyfile(a, b)
    assert ft.data_identity(a) == ft.data_identity(b)
    np.save(b, [1, 2, 4])
    assert ft.data_identity(a) != ft.data_identity(b)


def test_calibration_threshold_and_bootstrap_use_same_scale(monkeypatch):
    from scripts import final_model as fm

    logits = np.array([[4.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    labels = np.array([0, 1])
    monkeypatch.setattr(ft, "predict", lambda *args: (None, labels, logits))
    monkeypatch.setattr(fm, "temperature_scale", lambda *args, **kwargs: 2.0)
    expected = torch.softmax(torch.tensor(logits) / 2.0, 1)[:, 1].numpy()

    def threshold(probs, y):
        np.testing.assert_allclose(probs, expected)
        return 0.7

    def bootstrap(probs, y, **kwargs):
        np.testing.assert_allclose(probs, expected)
        assert kwargs["threshold"] == 0.7
        return {}

    monkeypatch.setattr(fm, "tune_threshold", threshold)
    monkeypatch.setattr(fm, "bootstrap_ci", bootstrap)
    report, _, _ = ft.calibrated_report(None, None, None, 2)
    assert report["threshold"] == 0.7


def test_wandb_requires_online_and_reuses_persisted_identity(tmp_path, monkeypatch):
    import wandb

    events, calls = [], []
    monkeypatch.setattr(ft, "load_env_file", lambda: events.append("env"))
    monkeypatch.setenv("WANDB_API_KEY", "test-only-not-a-real-key")

    class FakeRun(Run):
        disabled = offline = False

        def define_metric(self, *args, **kwargs):
            pass

    def init(**kwargs):
        assert events
        calls.append(kwargs)
        return FakeRun()

    monkeypatch.setattr(wandb, "init", init)
    artifacts = ft.Artifacts(tmp_path, smoke=True)
    ft.init_wandb("run", config(), artifacts)
    ft.init_wandb("run", config(), artifacts, resume=True)
    assert calls[0]["id"] == calls[1]["id"]
    assert calls[1]["mode"] == "online" and calls[1]["resume"] == "allow"
    monkeypatch.setattr(wandb, "init", lambda **kwargs: None)
    with pytest.raises(RuntimeError, match="Mandatory W&B"):
        ft.init_wandb("run", config(), artifacts, resume=True)


@pytest.mark.parametrize("channel", [False, True])
def test_partial_denoise_and_per_source_views(tmp_path, channel):
    raw = np.arange(3 * 8**3, dtype=np.uint16).reshape(3, 8, 8, 8).astype(np.uint8)
    if channel:
        raw = raw[:, None]
    source, target = tmp_path / "Training_volumes.npy", tmp_path / "Training_volumes_dn.npy"
    np.save(source, raw)
    np.save(tmp_path / "labels.npy", [0, 1, 0])
    np.save(target, np.zeros_like(raw))

    def denoise(volume):
        return np.full_like(volume, 42)

    assert not ft.build_denoised(source, target, denoise, method="test", params={}, implementation="test-v1", limit=1)
    with pytest.raises(ValueError, match="completion marker"):
        ft.FinalDataset(target, tmp_path / "labels.npy", res3d=8, res2d=8, seed=0)
    assert ft.build_denoised(source, target, denoise, method="test", params={}, implementation="test-v1")
    assert (np.load(target) == 42).all()
    raw_views, _ = ft.build_views(source, res2d=8)
    dn_views, _ = ft.build_views(target, res2d=8)
    assert raw_views != dn_views
    assert (np.load(dn_views) == 42).all()
    ds = ft.FinalDataset(target, tmp_path / "labels.npy", res3d=8, res2d=8, seed=0, train=True)
    assert ds[0][0].shape == (1, 8, 8, 8)
    torch.testing.assert_close(ds[0][0], ds[0][0], rtol=0, atol=0)


def test_notebook_bilateral_finetune_preset():
    notebook = Path(__file__).resolve().parents[1] / "notebooks/3d_glaucoma_final_2x2d_3d_crossgate.ipynb"
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    source = next("".join(c["source"]) for c in cells if "RUN_GROUP = " in "".join(c["source"]))
    namespace = {"SMOKE": False}
    exec(source.split("SMOKE_ROOT =")[0], namespace)
    assert namespace["DATASETS"] == ["bilateral"]
    assert namespace["RUN_TARGET"] == namespace["WARM_START_TARGET"] == "bilateral_s42"
    assert namespace["EPOCHS"] == 5
    assert namespace["PATIENCE"] >= namespace["EPOCHS"]
    assert namespace["BUILD_DENOISED"]
    assert namespace["WARM_START_WEIGHTS"].endswith("raw_s42_recovered_20260910/raw_s42/best_weights.pt")
    assert not namespace["RESUME"]
    assert not namespace["RUN_XAI"]


def test_notebook_smoke_offline_end_to_end(monkeypatch):
    monkeypatch.setenv("FINAL_SMOKE", "1")
    notebook = Path(__file__).resolve().parents[1] / "notebooks/3d_glaucoma_final_2x2d_3d_crossgate.ipynb"
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(json.loads(notebook.read_text(encoding="utf-8"))["cells"]):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"<notebook-cell-{index}>", "exec"), namespace)
    assert list(namespace["RESULTS"]) == ["bilateral_s42"]
    local = namespace["LOCAL_ROOT"] / "bilateral_s42"
    assert namespace["config"]["denoise_method"] == "bilateral"
    for dataset, split in zip((namespace["tr"], namespace["va"], namespace["te"]), namespace["SPLITS"]):
        assert dataset.source.name == f"{split}_volumes_dn.npy"
        assert "_volumes_dn_views_" in str(dataset.views.filename)
        raw = np.load(namespace["DATA_ROOT"] / f"{split}_volumes.npy")
        assert not np.array_equal(raw, dataset.volumes)
    assert (local / "last.pt").exists()
    assert (local / "metrics.json").exists()
    assert (local / "gradcam3d.png").exists()
    assert list((local / "wandb").glob("offline-run-*"))
    assert (local / "completed.pt").exists()
    namespace["RESUME"] = True
    train_cell = next(
        cell
        for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        if cell["cell_type"] == "code" and "RESULTS = {}" in "".join(cell["source"])
    )
    monkeypatch.setattr(ft, "init_wandb", lambda *a, **k: pytest.fail("Completed run should be skipped"))
    exec(compile("".join(train_cell["source"]), "<resume-completed>", "exec"), namespace)
    assert list(namespace["RESULTS"]) == ["bilateral_s42"]


def test_sweep_status_new_initialized_resume_and_completed_remote(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    artifacts = ft.Artifacts(tmp_path / "local", remote)
    cfg = config()
    assert ft.run_status(artifacts, cfg, resume=True)["status"] == "new"
    artifacts.save({"id": Run.id, "config": cfg, "warm_start": "rescued.pt"}, "run_identity.pt")
    status = ft.run_status(artifacts, cfg, resume=True)
    assert status == {"status": "initialized", "warm_start": "rescued.pt"}
    current = ft.Trainer(ft.SmokeModel(), Data(), cfg, artifacts, Run())
    assert current.epoch == current.step == 0
    current.fit(evaluate)
    assert ft.run_status(artifacts, cfg, resume=True)["status"] == "resume"
    result = {"tag": "raw_s42", "seed": 42}
    (artifacts.local / "metrics.json").write_text(json.dumps(result))
    ft.complete_run(artifacts, cfg, Run.id)
    fresh = ft.Artifacts(tmp_path / "fresh", remote)
    assert ft.run_status(fresh, cfg, resume=True) == {"status": "complete", "result": result, "warm_start": ""}
    (remote / "metrics.json").unlink()
    assert ft.run_status(fresh, cfg, resume=True)["status"] == "resume"
    (remote / "run_identity.pt").unlink()
    with pytest.raises(RuntimeError, match="without identity"):
        ft.run_status(fresh, cfg, resume=True)


def test_group_summary_preserves_remote_and_unselected_rows(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    artifacts = ft.Artifacts(tmp_path / "local", remote)

    def result(tag, value):
        return {
            "tag": tag,
            "seed": 42,
            "threshold": 0.5,
            "temperature": 1.0,
            "val": {"acc": value},
            "test": {"acc": value},
        }

    for root, tag, value in ((remote, "raw_s42", 0.6), (artifacts.local, "bilateral_s42", 0.7)):
        (root / tag).mkdir()
        (root / tag / "metrics.json").write_text(json.dumps(result(tag, value)))
    (remote / "final_metrics.json").write_text(json.dumps([result("raw_s43", 0.8)]))
    rows = ft.publish_summary(artifacts)
    assert {r["tag"] for r in rows} == {"raw_s42", "raw_s43", "bilateral_s42"}
    (artifacts.local / "bilateral_s42/metrics.json").write_text(json.dumps(result("bilateral_s42", 0.9)))
    rows = ft.publish_summary(artifacts)
    assert len(rows) == 3
    assert next(r for r in rows if r["tag"] == "bilateral_s42")["test_acc"] == 0.9
    assert json.loads((remote / "final_metrics.json").read_text()) == json.loads(
        (artifacts.local / "final_metrics.json").read_text()
    )


def test_emergency_weights_roundtrip_and_validation(tmp_path):
    broken = trainer(tmp_path / "broken")

    def interrupt(current):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        broken.fit(evaluate, boundary_hook=interrupt)
    emergency = next((tmp_path / "broken").glob("emergency_weights_*.pt"))
    restarted = trainer(tmp_path / "warm", warm_start=str(emergency))
    assert restarted.step == restarted.epoch == 0
    assert not restarted.optimizer.state
    for key, value in broken.model.state_dict().items():
        torch.testing.assert_close(restarted.model.state_dict()[key], value, rtol=0, atol=0)
    ft.atomic_save({"weights": ft.cpu_state(broken.model), "resumable": True}, tmp_path / "invalid.pt")
    with pytest.raises(ValueError, match="resumable=False"):
        trainer(tmp_path / "bad", warm_start=str(tmp_path / "invalid.pt"))
    ft.atomic_save({"weights": {"x": "not a tensor"}, "resumable": False}, tmp_path / "invalid.pt")
    with pytest.raises(ValueError, match="tensor state_dict"):
        trainer(tmp_path / "bad", warm_start=str(tmp_path / "invalid.pt"))


@pytest.mark.parametrize("complete", [False, True])
def test_denoise_rejects_parameter_and_implementation_changes(tmp_path, complete):
    source, target = tmp_path / "raw.npy", tmp_path / "dn.npy"
    np.save(source, np.zeros((2, 4, 4, 4), dtype=np.uint8))
    kwargs = dict(method="bilateral", params={"sigma_color": 0.1}, implementation={"sha": "v1"})
    ft.build_denoised(source, target, lambda v: v, limit=0 if complete else 1, **kwargs)
    for changed in (dict(params={"sigma_color": 0.2}), dict(implementation={"sha": "v2"})):
        with pytest.raises(ValueError, match="identity mismatch"):
            ft.build_denoised(source, target, lambda v: v, **(kwargs | changed))


def test_actual_sigint_during_accumulation_commits_and_resumes(tmp_path):
    torch.manual_seed(23)
    model = ft.SmokeModel()
    baseline = trainer(tmp_path / "baseline", model=copy.deepcopy(model))
    baseline.fit(evaluate)
    current = trainer(tmp_path / "interrupted", model=copy.deepcopy(model))
    calls = []

    def interrupt(module, inputs):
        calls.append(1)
        if len(calls) == 2:
            assert any(p.grad is not None for p in module.parameters())
            signal.raise_signal(signal.SIGINT)
            assert current.stop_requested

    previous = signal.getsignal(signal.SIGINT)
    hook = current.model.register_forward_pre_hook(interrupt)
    try:
        assert not current.fit(evaluate)
    finally:
        hook.remove()
    assert signal.getsignal(signal.SIGINT) == previous
    committed = torch.load(tmp_path / "interrupted/last.pt", weights_only=False)
    assert committed["cursor"] == 4 and committed["step"] == 1
    assert all(p.grad is None for p in current.model.parameters())
    resumed = trainer(tmp_path / "interrupted", resume=True)
    resumed.fit(evaluate)
    assert resumed.history == baseline.history
    for key, value in baseline.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[key], value, rtol=0, atol=0)
