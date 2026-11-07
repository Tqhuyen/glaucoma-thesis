import pytest
import torch

from models.glaucoma.model import Pyramid3DCNN, PyramidResNeXt3D
from pipeline.config import validate_config

HIDDEN = (8, 16, 32, 64)


def _model(**kwargs):
    kwargs.setdefault("hidden", HIDDEN)
    kwargs.setdefault("dropout", 0.0)
    return Pyramid3DCNN(**kwargs)


def _config(**model_overrides):
    model = {
        "type": "glaucoma",
        "architecture": "pyramid3dcnn",
        "input_channels": 1,
        "num_classes": 2,
        "hidden": list(HIDDEN),
        "fusion_dim": 64,
        "fuse_levels": 3,
    }
    model.update(model_overrides)
    return {
        "run_name": "test-pyramid",
        "seed": 42,
        "model": model,
        "hf": {"use_auth": False, "fast_transfer": False},
        "train": {
            "epochs": 1,
            "batch_size": 2,
            "grad_accum_steps": 1,
            "lr": 1e-4,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "max_grad_norm": 1.0,
            "amp": False,
            "compile": False,
            "log_every_steps": 1,
            "eval_every_steps": 1,
            "save_every_steps": 1,
            "keep_last_checkpoints": 1,
            "early_stop_patience": 1,
        },
        "logging": {"tensorboard": False, "jsonl": False, "wandb": False},
        "output_dir": "/tmp/test-pyramid",
    }


@pytest.mark.parametrize("fuse_levels", [0, 2, 3, 9])
def test_forward_backward(fuse_levels):
    model = _model(fuse_levels=fuse_levels)
    x = torch.randn(2, 1, 24, 24, 24)
    logits = model(x)["logits"]
    assert logits.shape == (2, 2)

    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_head_sees_every_level():
    full = _model(fuse_levels=3)
    assert full.classifier[1].in_features == 8 + 64 * 3
    unfused = _model(fuse_levels=0)
    assert unfused.classifier[1].in_features == sum(HIDDEN)
    assert len(unfused.laterals) == 0


def test_groupnorm_is_batch_size_independent():
    model = _model(fuse_levels=3).train()
    x = torch.randn(3, 1, 24, 24, 24)
    solo = model(x[:1])["logits"]
    batched = model(x)["logits"][:1]
    assert torch.allclose(solo, batched, atol=1e-5)


def test_pyramid_config_validates():
    validate_config(_config())
    validate_config(_config(fuse_levels=0))


@pytest.mark.parametrize("overrides", [{"fusion_dim": 4}, {"fusion_dim": 1024}, {"fuse_levels": -1}])
def test_pyramid_config_rejects_bad_keys(overrides):
    with pytest.raises(ValueError, match="out of allowed range"):
        validate_config(_config(**overrides))


def _resnext(**kwargs):
    kwargs.setdefault("features", (8, 16, 32, 64))
    kwargs.setdefault("groups", 4)
    kwargs.setdefault("base_width", 8)
    kwargs.setdefault("fusion_dim", 32)
    kwargs.setdefault("fuse_levels", 3)
    kwargs.setdefault("dropout", 0.0)
    return PyramidResNeXt3D(**kwargs)


@pytest.mark.parametrize("fuse_levels", [0, 2, 4])
def test_resnext_forward_backward(fuse_levels):
    model = _resnext(fuse_levels=fuse_levels)
    x = torch.randn(2, 1, 32, 32, 32)
    logits = model(x)["logits"]
    assert logits.shape == (2, 2)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_resnext_keeps_depth_at_first_stage():
    model = _resnext()
    feats = []
    for _, refine in model.stages:
        refine.register_forward_hook(lambda _m, _i, out: feats.append(out))
    model(torch.randn(1, 1, 32, 32, 32))
    # stride (2,2,1) halves the first two axes and keeps the last (depth) at 32.
    assert feats[0].shape == (1, 8, 16, 16, 32)
    assert [f.shape[2] for f in feats] == [16, 8, 4, 2]
    assert [f.shape[4] for f in feats] == [32, 16, 8, 4]


def test_resnext_config_validates():
    validate_config(_config(architecture="pyramid_resnext3d", groups=8, base_width=8))
