import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import controls_training as ct
from scripts import final_model as fm
from scripts import final_training as ft


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("use_3d", [None, True, False])
@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("seed", [0, 42])
def test_controls_dataset_matches_baseline_without_unused_volume_reads(tmp_path, monkeypatch, use_3d, train, seed):
    source, labels = tmp_path / "volumes.npy", tmp_path / "labels.npy"
    raw = np.random.default_rng(7).integers(0, 256, (3, 6, 6, 6), dtype=np.uint8)
    np.save(source, raw)
    np.save(labels, [0, 1, 0])
    ft.build_views(source, res2d=4)
    kwargs = dict(res3d=3, res2d=4, seed=seed, train=train)
    baseline = ft.FinalDataset(source, labels, **kwargs)
    expected = {}
    for epoch in (0, 1, 7):
        baseline.set_epoch(epoch)
        for index in range(len(baseline)):
            expected[epoch, index] = baseline[index]

    class UnreadableVolumes:
        def __getitem__(self, index):
            pytest.fail("Views-only dataset must not read raw volume payload")

    def no_interpolate(*args, **kwargs):
        pytest.fail("Views-only dataset must not interpolate 3D")

    def no_volume_sample(*args, **kwargs):
        pytest.fail("Prebuilt view cache must not read raw samples during initialization")

    if use_3d is False:
        monkeypatch.setattr(ft, "volume_sample", no_volume_sample)
        monkeypatch.setattr(torch.nn.functional, "interpolate", no_interpolate)
    dataset = ct.ControlsDataset(source, labels, **kwargs, **({} if use_3d is None else {"use_3d": use_3d}))
    assert dataset.use_3d is (use_3d is not False)
    assert dataset.deterministic_by_index
    if use_3d is False:
        dataset.volumes = UnreadableVolumes()
    for epoch in (0, 1, 7):
        dataset.set_epoch(epoch)
        for index in range(len(dataset)):
            x, views, label = dataset[index]
            baseline_x, baseline_views, baseline_label = expected[epoch, index]
            torch.testing.assert_close(views, baseline_views, rtol=0, atol=0)
            torch.testing.assert_close(label, baseline_label, rtol=0, atol=0)
            if use_3d is False:
                assert x.shape == (1, 1, 1, 1) and x.dtype == torch.float32
                assert torch.count_nonzero(x) == 0
            else:
                torch.testing.assert_close(x, baseline_x, rtol=0, atol=0)


@pytest.mark.parametrize("amp_dtype", [None, "float16", "bfloat16"])
@pytest.mark.parametrize("with_train", [False, True])
def test_calibrated_report_precision_and_protocol(monkeypatch, amp_dtype, with_train):
    logits = np.array([[4.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    labels = np.array([0, 1])
    expected = torch.softmax(torch.tensor(logits) / 2.0, 1)[:, 1].numpy()
    calls, bootstraps = [], []

    def predict(model, dataset, batch_size, *, num_workers, amp_dtype="float16"):
        calls.append((dataset, batch_size, num_workers, amp_dtype))
        return None, labels, logits

    def temperature(values, targets, *, max_iter):
        np.testing.assert_array_equal(values, logits)
        np.testing.assert_array_equal(targets, labels)
        assert len(calls) == 1 and calls[0][0] == "val"
        assert max_iter == 10
        return 2.0

    def threshold(probs, targets):
        np.testing.assert_allclose(probs, expected)
        return 0.7

    def bootstrap(probs, targets, **kwargs):
        np.testing.assert_allclose(probs, expected)
        assert kwargs == {"n_boot": 20, "threshold": 0.7}
        bootstraps.append(1)
        return {}

    monkeypatch.setattr(ft, "predict", predict)
    monkeypatch.setattr(fm, "temperature_scale", temperature)
    monkeypatch.setattr(fm, "tune_threshold", threshold)
    monkeypatch.setattr(fm, "bootstrap_ci", bootstrap)
    kwargs = {} if amp_dtype is None else {"amp_dtype": amp_dtype}
    report, probs, targets = ct.calibrated_report(
        None, "val", "test", 2, train="train" if with_train else None, num_workers=3, smoke=True, **kwargs
    )
    splits = ["val", "test", "train"] if with_train else ["val", "test"]
    assert calls == [(split, 2, 3, amp_dtype or "float16") for split in splits]
    assert len(bootstraps) == len(splits)
    assert ("train" in report) == ("train_ci" in report) == with_train
    np.testing.assert_allclose(probs, expected)
    np.testing.assert_array_equal(targets, labels)
    calls.clear()
    original, _, _ = ft.calibrated_report(
        None, "val", "test", 2, train="train" if with_train else None, num_workers=3, smoke=True
    )
    assert report == original


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_calibrated_report_rejects_invalid_temperature(monkeypatch, temperature):
    monkeypatch.setattr(ft, "predict", lambda *a, **k: (None, np.array([0, 1]), np.zeros((2, 2))))
    monkeypatch.setattr(fm, "temperature_scale", lambda *a, **k: temperature)
    with pytest.raises(RuntimeError, match="finite positive temperature"):
        ct.calibrated_report(None, None, None, 2)


def make_trainer(path, *, resume=False):
    class Data(torch.utils.data.TensorDataset):
        deterministic_by_index = True

        def set_epoch(self, epoch):
            pass

    data = Data(torch.ones(4, 1, 3, 3, 3), torch.ones(4, 2, 1, 3, 3), torch.tensor([0, 1, 0, 1]))
    config = dict(
        epochs=5,
        batch_size=2,
        grad_accum=1,
        lr=0.001,
        weight_decay=0.01,
        patience=1,
        checkpoint_steps=1,
        seed=42,
        class_weights=[1.0, 1.0],
    )
    logs = []
    run = SimpleNamespace(id="controls-test", logs=logs, log=logs.append)
    return ct.Trainer(ft.SmokeModel(), data, config, ft.Artifacts(path, smoke=True), run, resume=resume)


def evaluate(model):
    return {"auc_roc": 0.7, "f1": 0.5, "balanced_acc": 0.5, "mcc": 0.0}


def test_epoch_hook_order_early_stop_and_completed_resume(tmp_path):
    current = make_trainer(tmp_path)
    events = []

    def test_evaluate(model):
        events.append(("test", current.epoch))
        return evaluate(model)

    def hook(state):
        assert state is current
        assert events[-1] == ("test", state.epoch)
        events.append(("hook", state.epoch))
        assert len(state.history) == state.epoch
        assert state.history[-1]["val"] == evaluate(None)
        assert state.history[-1]["test"] == evaluate(None)
        assert state.run.logs[-1]["test/epoch"] == state.epoch
        saved = torch.load(tmp_path / "last.pt", weights_only=False)
        assert saved["epoch"] == state.epoch
        assert saved["history"] == state.history
        if state.epoch == state.best_epoch:
            assert torch.load(tmp_path / "best.pt", weights_only=False)["epoch"] == state.epoch
        state.commit()
        state.log({"train/epoch_seconds": 1.0, "progress/step": state.step})

    assert current.fit(evaluate, test_evaluate=test_evaluate, epoch_hook=hook)
    assert events == [("test", 1), ("hook", 1), ("test", 2), ("hook", 2)]
    assert all(row["test"] == evaluate(None) for row in current.history)
    assert "seconds" not in json.dumps(current.history)
    assert current._epoch_hook is None
    resumed = make_trainer(tmp_path, resume=True)
    assert resumed.fit(evaluate, test_evaluate=evaluate, epoch_hook=lambda state: pytest.fail("Already complete"))
    assert resumed.history == current.history


def test_epoch_hook_skips_partial_epoch_and_runs_on_resume(tmp_path):
    current = make_trainer(tmp_path)
    epochs = []

    def stop(state):
        state.stop_requested = True

    def hook(state):
        epochs.append(state.epoch)

    assert not current.fit(evaluate, boundary_hook=stop, test_evaluate=evaluate, epoch_hook=hook)
    assert epochs == []
    resumed = make_trainer(tmp_path, resume=True)
    assert resumed.fit(evaluate, test_evaluate=evaluate, epoch_hook=hook)
    assert epochs == [1, 2]


def test_epoch_hook_skips_failed_evaluation(tmp_path):
    current = make_trainer(tmp_path)

    def hook(state):
        pytest.fail("No completed test evaluation")

    def fail(model):
        raise RuntimeError("test evaluation failed")

    with pytest.raises(RuntimeError, match="test evaluation failed"):
        current.fit(evaluate, test_evaluate=fail, epoch_hook=hook)
    assert current._epoch_hook is None


def test_epoch_hook_without_test_and_stop_after_durable_epoch(tmp_path):
    current = make_trainer(tmp_path)
    current.commit()
    epochs = []

    def stop(state):
        epochs.append(state.epoch)
        assert "test" not in state.history[-1]
        assert state.run.logs[-1]["val/epoch"] == state.epoch
        assert torch.load(tmp_path / "last.pt", weights_only=False)["history"] == state.history
        state.stop_requested = True

    assert not current.fit(evaluate, epoch_hook=stop)
    assert epochs == [1]
    assert current._epoch_hook is None
    resumed = make_trainer(tmp_path, resume=True)
    assert resumed.fit(evaluate, epoch_hook=lambda state: epochs.append(state.epoch))
    assert epochs == [1, 2]


def test_epoch_hook_failure_preserves_checkpoint_and_clears_callback(tmp_path):
    current = make_trainer(tmp_path)
    epochs = []

    def fail(state):
        epochs.append(state.epoch)
        assert state._last_notified_epoch == state.epoch
        raise RuntimeError("hook failed")

    with pytest.raises(RuntimeError, match="hook failed"):
        current.fit(evaluate, epoch_hook=fail)
    assert current._epoch_hook is None
    assert torch.load(tmp_path / "last.pt", weights_only=False)["epoch"] == 1
    resumed = make_trainer(tmp_path, resume=True)
    assert resumed.fit(evaluate, epoch_hook=lambda state: epochs.append(state.epoch))
    assert epochs == [1, 2]


def test_trainer_without_hook_preserves_original_api(tmp_path):
    assert make_trainer(tmp_path).fit(evaluate)


def test_frozen_final_training_sha_matches_approved_notebook():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/3d_glaucoma_final_2x2d_3d_crossgate.ipynb").read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "EXPECTED_HELPER_SHA256 =" in "".join(cell["source"])
    )
    expected = next(
        ast.literal_eval(node.value)["final_training.py"]
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "EXPECTED_HELPER_SHA256"
    )
    normalized = (
        (root / "scripts/final_training.py").read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    )
    assert hashlib.sha256(normalized.encode()).hexdigest() == expected
