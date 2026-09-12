import json

import numpy as np
import pytest
import torch

from scripts import final_model as fm
from scripts import xai_identity as xi


@pytest.mark.parametrize(
    "split, index, stem",
    [
        ("Training", 0, "data_0001"),
        ("Training", 2099, "data_2100"),
        ("Validation", 0, "data_2101"),
        ("Validation", 299, "data_2400"),
        ("Test", 0, "data_2401"),
        ("Test", 899, "data_3300"),
    ],
)
def test_resolve_stem_boundaries(split, index, stem):
    assert xi.resolve_stem(split, index) == stem


def test_resolve_stem_rejects_bad_input():
    with pytest.raises(ValueError, match="Unknown split"):
        xi.resolve_stem("Nope", 0)
    with pytest.raises(ValueError, match="nonnegative"):
        xi.resolve_stem("Validation", -1)


def test_sample_identity_and_prefix():
    identity = xi.sample_identity("Validation", 0, 1)
    assert identity == {
        "split": "Validation",
        "index": 0,
        "stem": "data_2101",
        "label": 1,
        "source_repo": "tqhuyen/harvard-oct-glaucoma-200",
        "source_revision": "939a38876b7b9313162842ef2d44b7edc2b57020",
    }
    assert xi.identity_prefix(identity) == "Validation_idx0_data_2101_label1"


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 1, 1)

    def gcam3d_module(self):
        return self.conv

    def gcam2d_module(self, index):
        return self.conv


class FakeDataset:
    def __getitem__(self, index):
        x = torch.zeros(1, 4, 8, 8)
        v = torch.zeros(2, 1, 8, 8)
        return x, v, 1


class FakeArtifacts:
    def __init__(self, local):
        self.local = local
        self.saved = []

    def sync(self, path):
        return path

    def save(self, value, name):
        path = self.local / name
        torch.save(value, path)
        self.saved.append(name)
        return path


class FakeRun:
    def __init__(self):
        self.logs = []
        self.summary = {}

    def log(self, values):
        self.logs.append(values)


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    monkeypatch.setattr(fm, "grad_cam", lambda *a, **k: (np.ones((4, 4, 4)), None, None))
    monkeypatch.setattr(fm, "occlusion_sensitivity", lambda *a, **k: (np.ones((4, 4, 4)), None))
    monkeypatch.setattr(fm, "integrated_gradients", lambda *a, **k: np.ones((8, 8)))
    monkeypatch.setattr(fm, "crossgate_attention", lambda *a, **k: (np.array([[0.6, 0.4]]), 0.61))
    monkeypatch.setattr(
        fm, "branch_drop_importance", lambda *a, **k: (["3D", "2D-0", "2D-1"], np.array([0.01, 0.1, 0.0]))
    )
    import wandb

    monkeypatch.setattr(wandb, "Image", lambda path: path)
    return FakeArtifacts(tmp_path), FakeRun()


def test_save_xai_identified_embeds_identity(stubbed):
    artifacts, run = stubbed
    meta = xi.save_xai_identified(FakeModel(), FakeDataset(), artifacts, run, split="Validation", index=0, smoke=False)
    assert meta["identity"]["stem"] == "data_2101"
    assert meta["identity"]["label"] == 1
    prefix = "Validation_idx0_data_2101_label1"
    for name in ("gradcam3d", "gradcam2d_0", "gradcam2d_1", "occlusion", "integrated_gradients"):
        assert (artifacts.local / f"{prefix}_{name}.png").is_file()
    assert (artifacts.local / f"{prefix}_gradcam2d_0_overlay.png").is_file()
    assert f"{prefix}_saliency_tensors.pt" in artifacts.saved
    saved_meta = json.loads((artifacts.local / f"{prefix}_xai_meta.json").read_text())
    assert saved_meta["identity"]["stem"] == "data_2101"
    assert saved_meta["tensors"]["gradcam3d"] == [4, 4, 4]
    assert saved_meta["fusion"]["gate"] == pytest.approx(0.61)
    assert saved_meta["fusion"]["attention"] == {"2D-0": 0.6, "2D-1": 0.4}
