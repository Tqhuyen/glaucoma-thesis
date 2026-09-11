import numpy as np
import pytest
import torch

from scripts import final_training as ft
from scripts import information_theory as it


class TinyDataset(torch.utils.data.Dataset):
    deterministic_by_index = True

    def __init__(self, n=8):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        rng = np.random.default_rng(index)
        return (
            torch.tensor(rng.random((1, 4, 4, 4)), dtype=torch.float32),
            torch.tensor(rng.random((2, 1, 4, 4)), dtype=torch.float32),
            torch.tensor(index % 2),
        )


def test_validate_estimators_detects_xor_and_independence():
    report = it.validate_estimators(n=600, steps=400, seed=0)
    assert report["xor_mi"] > 0.3
    assert report["independent_mi"] < 0.25
    assert report["ksg_dependent_ok"]
    assert report["ksg_independent_ok"]


def test_linear_probe_recovers_signal_and_beats_permutation_null():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 200)
    features = rng.normal(size=(200, 32)) + 2.0 * labels[:, None]
    probe = it.linear_probe(features[:120], labels[:120], features[120:], labels[120:], steps=300, seed=0)
    assert probe["auc"] > 0.8
    shuffled = it.linear_probe(
        features[:120], rng.permutation(labels[:120]), features[120:], labels[120:], steps=300, seed=0
    )
    assert shuffled["auc"] < 0.75


def test_mic_and_surrogate_test_flag_dependence():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 1))
    y = (x[:, 0] > 0).astype(np.int64)
    assert it.mic_approx(x[:, 0], y)["mic"] > 0.5
    result = it.surrogate_test(lambda a, b: it.ksg_mi(a, b), x, y, n=30, seed=0)
    assert result["real"] > result["null_mean"]
    assert result["p_value"] <= 0.1


def test_collect_embeddings_contract_for_smoke_model():
    model = ft.SmokeModel()
    result = it.collect_embeddings(model, TinyDataset(8), 4, input_res=4)
    assert result["z"].shape == (8, 4)
    assert result["e3d"].shape == (8, 2)
    assert result["e2d"].shape == (8, 1, 2)
    assert result["x"].shape == (8, 64)
    assert result["y"].shape == (8,)
    assert np.all((result["probs"] >= 0) & (result["probs"] <= 1))


def test_info_nce_and_multi_seed_estimate_ordering():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 1))
    y = x + 0.3 * rng.normal(size=(300, 1))
    independent = it.info_nce_mi(x, rng.normal(size=(300, 1)), steps=200, seed=0)["mi"]
    dependent = it.info_nce_mi(x, y, steps=200, seed=0)["mi"]
    assert dependent > independent
    stats = it.estimate_mi(x, y, methods=("dv", "infonce"), seeds=(0, 1), steps=100)
    assert set(stats) == {"dv", "infonce"}
    for method in stats.values():
        assert method["n_seeds"] == 2
        assert 0.0 <= method["negative_rate"] <= 1.0
        assert method["min"] <= method["median"] <= method["max"]
    assert it.label_entropy(np.array([0, 0, 1, 1])) == pytest.approx(0.6931, abs=1e-3)


def test_interaction_information_flags_redundancy():
    rng = np.random.default_rng(1)
    labels = (rng.normal(size=400) > 0).astype(np.int64)
    z1 = labels[:, None] + 0.4 * rng.normal(size=(400, 1))
    z2 = labels[:, None] + 0.4 * rng.normal(size=(400, 1))
    redundant = it.interaction_information(z1, z2, labels, steps=100, seeds=(0, 1))
    independent = it.interaction_information(
        rng.normal(size=(400, 2)), rng.normal(size=(400, 2)), labels, steps=100, seeds=(0, 1)
    )
    assert redundant["interaction"] > independent["interaction"]


def test_mi_seed_is_reproducible_and_infonce_has_log_two_normalization(monkeypatch):
    x = np.random.default_rng(3).normal(size=(12, 2))
    first = it.mine_mi(x, x[:, :1], steps=2, seed=7)["mi"]
    torch.rand(10)
    assert it.mine_mi(x, x[:, :1], steps=2, seed=7)["mi"] == pytest.approx(first)

    class Constant(it.MINEStats):
        def forward(self, x, y):
            return super().forward(x, y) * 0

    monkeypatch.setattr(it, "MINEStats", Constant)
    assert it.info_nce_mi(x, x[:, :1], steps=1)["mi"] == pytest.approx(0, abs=1e-6)


def test_mic_ties_are_not_split_by_row_order():
    values = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    bins, _ = it._equal_count_bins(values, 2)
    assert len(np.unique(bins[values == 0])) == len(np.unique(bins[values == 1])) == 1


def test_final_model_fuse_extraction_preserves_legacy_forward_and_state_keys(monkeypatch):
    from scripts import final_model as fm

    class Encoder(torch.nn.Module):
        out_dim = 2

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))

        def forward(self, x):
            return x.flatten(1).mean(1)[:, None] * self.weight

    monkeypatch.setattr(fm, "Timm2D", Encoder)
    monkeypatch.setattr(fm, "Enc3DResNeXt", Encoder)
    model = fm.FinalModel(n_2d=2, D=8, enc2d_pretrained=False).eval()
    keys = list(model.state_dict())
    x, views = torch.rand(2, 1, 4, 4, 4), torch.rand(2, 2, 1, 4, 4)
    tokens = torch.stack([p(e) for p, e in zip(model.projs, model.embed(x, views))], 1)
    expected = model.head(model.fusion(tokens[:, 0], tokens[:, 1:]))
    torch.testing.assert_close(model(x, views), expected, rtol=0, atol=0)
    z, actual_tokens = model.fuse(x, views)
    torch.testing.assert_close(model.head(z), expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_tokens, tokens, rtol=0, atol=0)
    assert list(model.state_dict()) == keys
