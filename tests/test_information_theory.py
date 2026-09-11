import numpy as np
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
