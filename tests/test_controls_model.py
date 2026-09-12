import pytest
import torch

from scripts import controls_model as cm

ENC3D = (8, 16, 16, 16)
ENC2D = "resnet18"


def build(**kwargs):
    return cm.ControlsModel(n_2d=2, D=32, enc2d=ENC2D, enc2d_pretrained=False, enc3d_features=ENC3D, **kwargs)


def test_default_model_shapes_and_fuse_contract():
    model = build()
    x = torch.rand(2, 1, 16, 16, 16)
    views = torch.rand(2, 2, 1, 64, 64)
    assert model(x, views).shape == (2, 2)
    z, tokens = model.fuse(x, views)
    assert z.shape == (2, 32)
    assert tokens.shape == (2, 3, 32)


def test_view_indices_select_the_requested_view():
    x = torch.rand(1, 1, 16, 16, 16)
    a, b, c = torch.rand(1, 1, 64, 64), torch.rand(1, 1, 64, 64), torch.rand(1, 1, 64, 64)
    first = torch.stack([a, c], dim=1)
    second = torch.stack([b, c], dim=1)

    def logits(view_indices):
        model = cm.ControlsModel(
            n_2d=1,
            view_indices=view_indices,
            D=32,
            enc2d=ENC2D,
            enc2d_pretrained=False,
            enc3d_features=ENC3D,
        )
        model.eval()
        with torch.no_grad():
            return model(x, first), model(x, second)

    view_one_first, view_one_second = logits((1,))
    assert torch.allclose(view_one_first, view_one_second)
    view_zero_first, view_zero_second = logits((0,))
    assert not torch.allclose(view_zero_first, view_zero_second)


def test_two_dimensional_only_concat_variant():
    model = cm.ControlsModel(
        n_2d=2,
        use_3d=False,
        fusion="concat",
        D=32,
        enc2d=ENC2D,
        enc2d_pretrained=False,
        enc3d_features=ENC3D,
    )
    x = torch.rand(2, 1, 16, 16, 16)
    views = torch.rand(2, 2, 1, 64, 64)
    assert model(x, views).shape == (2, 2)
    assert model.enc3d is None
    assert model.head.in_features == 64


def test_three_branch_concat_variant():
    model = build(fusion="concat")
    x = torch.rand(2, 1, 16, 16, 16)
    views = torch.rand(2, 2, 1, 64, 64)
    assert model(x, views).shape == (2, 2)
    assert model.head.in_features == 96


def test_fixed_gate_has_no_parameter_and_unit_value():
    gate = cm.FixedCrossGate(32, 2)
    assert not any(name.endswith("gate") for name, _ in gate.named_parameters())
    out = gate(torch.rand(4, 32), torch.rand(4, 2, 32))
    assert out.shape == (4, 32)
    assert gate.last_gate == 1.0


def test_crossgate_rejects_2d_only_configuration():
    with pytest.raises(ValueError, match="CrossGate"):
        cm.ControlsModel(n_2d=2, use_3d=False, fusion="crossgate")
