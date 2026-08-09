import tempfile
from pathlib import Path

import torch

from pipeline.utils import (
    atomic_save,
    classification_metrics,
    load_config,
    prune_checkpoints,
    sample_system_metrics,
)


def test_atomic_save_roundtrip():
    state = {"a": torch.tensor([1.0, 2.0]), "b": 42}
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.pt"
        atomic_save(state, p)
        assert p.exists()
        loaded = torch.load(p, map_location="cpu")
        assert loaded["b"] == 42
        assert torch.equal(loaded["a"], torch.tensor([1.0, 2.0]))


def test_prune_checkpoints():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp)
        for i in [1, 2, 3, 4, 5]:
            p = ckpt_dir / f"step_{i * 100}.pt"
            p.write_text("dummy")
        prune_checkpoints(ckpt_dir, keep=2)
        remaining = sorted(ckpt_dir.glob("step_*.pt"))
        assert len(remaining) == 2
        assert remaining[0].name == "step_400.pt"
        assert remaining[1].name == "step_500.pt"


def test_config_override():
    cfg = load_config("configs/glaucoma.yaml", overrides=["train.lr=0.0005", "run_name=test-override"])
    assert cfg["train"]["lr"] == 0.0005
    assert cfg["run_name"] == "test-override"


def test_config_override_typo():
    try:
        load_config("configs/glaucoma.yaml", overrides=["train.laerning_rate=0.001"])
    except KeyError:
        return
    assert False, "Should have raised KeyError on typo"


def test_classification_metrics_perfect():
    preds = torch.tensor([1, 1, 0, 0, 1, 0])
    targets = torch.tensor([1, 1, 0, 0, 1, 0])
    m = classification_metrics(preds, targets, num_classes=2)
    assert m["acc"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["precision_pos"] == 1.0
    assert m["recall_pos"] == 1.0


def test_classification_metrics_handcrafted():
    # 6 samples, class1 = glaucoma. Predict all-positive.
    preds = torch.tensor([1, 1, 1, 1, 1, 1])
    targets = torch.tensor([1, 1, 0, 0, 1, 1])
    m = classification_metrics(preds, targets, num_classes=2)
    assert m["acc"] == 4 / 6
    # positive class (1): tp=4 (targets 1 predicted 1), fp=2 (targets 0 pred 1)
    assert abs(m["precision_pos"] - 4 / 6) < 1e-6
    assert m["recall_pos"] == 1.0


def test_classification_metrics_empty_safe():
    preds = torch.tensor([], dtype=torch.long)
    targets = torch.tensor([], dtype=torch.long)
    m = classification_metrics(preds, targets, num_classes=2)
    assert m["acc"] == 0.0
    assert 0.0 <= m["f1"] <= 1.0


def test_sample_system_metrics_keys():
    m = sample_system_metrics()
    assert "sys/cpu_percent" in m
    assert "sys/ram_percent" in m
    assert "sys/ram_used_gb" in m
    assert "sys/disk_free_gb" in m
