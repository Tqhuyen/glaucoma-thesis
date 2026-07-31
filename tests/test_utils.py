import tempfile
from pathlib import Path

import torch

from pipeline.utils import atomic_save, load_config, prune_checkpoints


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
    cfg = load_config("configs/glaucoma_96.yaml", overrides=["train.lr=0.0005", "run_name=test-override"])
    assert cfg["train"]["lr"] == 0.0005
    assert cfg["run_name"] == "test-override"


def test_config_override_typo():
    try:
        load_config("configs/glaucoma_96.yaml", overrides=["train.laerning_rate=0.001"])
    except KeyError:
        return
    assert False, "Should have raised KeyError on typo"
