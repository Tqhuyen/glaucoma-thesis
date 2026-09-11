import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn import metrics as skm

from scripts import final_reporting as fr


def fixture_data():
    margin = np.array([-3, -1, 0, 0, 1, 4.0])
    return np.column_stack([np.zeros(6), margin]), np.array([0, 1, 0, 1, 1, 0])


def test_tied_rankings_are_permutation_invariant():
    probs = np.array([0.2, 0.2, 0.8, 0.8, 0.8, 0.9])
    labels = np.array([0, 1, 0, 1, 1, 0])
    for seed in range(10):
        idx = np.random.default_rng(seed).permutation(len(labels))
        result = fr.metrics(probs[idx], labels[idx])
        assert result["auc_pr"] == pytest.approx(skm.average_precision_score(labels, probs))
        assert result["auc_roc"] == pytest.approx(skm.roc_auc_score(labels, probs))
    assert fr.metrics([1, 1], [0, 1], ranking_scores=[40, 50])["auc_roc"] == 1


def test_ece_endpoint_bins():
    probs = np.linspace(0, 1, 11)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    counts, means, rates = fr._calibration_bins(probs, labels)
    np.testing.assert_array_equal(counts, [1] * 9 + [2])
    assert means[-1] == pytest.approx(0.95)
    assert rates[-1] == 0.5
    expected = (np.abs(probs[:9] - labels[:9]).sum() + 2 * abs(0.95 - 0.5)) / 11
    assert fr.metrics(probs, labels)["ece"] == pytest.approx(expected)


def test_calibration_independent_of_test_labels_and_recomputable():
    logits, labels = fixture_data()
    report, predictions = fr.calibrate_and_score(logits, labels, logits * 2, labels, n_boot=20)
    other, _ = fr.calibrate_and_score(logits, labels, logits * 2, 1 - labels, n_boot=0)
    assert report["temperature"] == other["temperature"]
    assert report["threshold"] == other["threshold"]
    assert report["val"] == other["val"]
    assert report["val"]["loss"] <= report["uncalibrated"]["val"]["loss"]
    for split, payload in predictions.items():
        recomputed = fr.metrics(
            payload["calibrated_probs"], payload["labels"], payload["threshold"], payload["ranking_scores"]
        )
        assert recomputed == {k: v for k, v in report[split].items() if k != "loss"}
        assert payload["logits"].dtype == np.float64
    json.dumps(report, allow_nan=False)


def test_bootstrap_keeps_threshold_and_ranking(monkeypatch):
    logits, labels = fixture_data()
    original = fr.metrics
    calls = []

    def record(probs, labels, threshold=0.5, ranking_scores=None):
        calls.append((threshold, np.asarray(ranking_scores).copy()))
        return original(probs, labels, threshold, ranking_scores)

    monkeypatch.setattr(fr, "metrics", record)
    report, _ = fr.calibrate_and_score(logits, labels, logits, labels, n_boot=20, seed=8)
    rng = np.random.default_rng(8)
    expected = []
    for _ in range(20):
        idx = rng.integers(0, len(labels), len(labels))
        if np.unique(labels[idx]).size == 2:
            expected.append(logits[idx, 1])
    assert len(calls[6:]) == len(expected) == report["bootstrap"]["valid"]
    for (threshold, scores), margin in zip(calls[6:], expected):
        assert threshold == report["threshold"]
        np.testing.assert_array_equal(scores, margin)


def test_threshold_exact_candidates_and_ties():
    assert fr._threshold(np.array([0.1, 0.34567]), np.array([0, 1])) == 0.34567
    assert fr._threshold(np.array([0.5, 0.5]), np.array([0, 1])) == 0.5
    p = np.array([0.0, 0.1, 0.6, 0.7])
    y = np.array([1, 1, 0, 0])
    threshold = fr._threshold(p, y)
    assert threshold > p.max()
    assert not (p >= threshold).any()


@pytest.mark.parametrize(
    "probs,labels,scores",
    [
        ([np.nan, 0], [0, 1], None),
        ([np.inf, 0], [0, 1], None),
        ([0, 1], [0, 0], None),
        ([0, 1], [0, np.nan], None),
        ([0, 1], [0, 1], [0, np.inf]),
        ([0, 1.1], [0, 1], None),
    ],
)
def test_invalid_metrics(probs, labels, scores):
    with pytest.raises(ValueError):
        fr.metrics(probs, labels, ranking_scores=scores)


def test_nonfinite_logits_and_fallback(monkeypatch):
    logits, labels = fixture_data()
    bad = logits.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        fr.calibrate_and_score(bad, labels, logits, labels)
    monkeypatch.setattr(fr, "minimize_scalar", lambda *a, **k: SimpleNamespace(x=0.0, success=False))
    report, _ = fr.calibrate_and_score(logits, labels, logits, labels, n_boot=0)
    assert report["temperature"] == 1
    assert report["calibration_protocol"]["temperature_status"] == "fallback_T1"
    assert all(ci == [None, None] for ci in report["test_ci"].values())
    json.dumps(report, allow_nan=False)


def test_single_class_bootstrap_draws_are_skipped():
    logits = np.array([[0.0, -1.0], [0.0, 1.0]])
    report, _ = fr.calibrate_and_score(logits, [0, 1], logits, [0, 1], n_boot=20, seed=42)
    info = report["bootstrap"]
    assert 0 < info["valid"] < 20
    assert info["valid"] + info["skipped_single_class"] == 20
    assert report["test_ci"]["auc_roc"] == [1.0, 1.0]


def test_worse_temperature_falls_back(monkeypatch):
    logits = np.array([[0.0, -2.0], [0.0, 2.0]])
    monkeypatch.setattr(fr, "minimize_scalar", lambda *a, **k: SimpleNamespace(x=5.0, success=True))
    report, _ = fr.calibrate_and_score(logits, [0, 1], logits, [0, 1], n_boot=0)
    assert report["temperature"] == 1.0
    assert report["calibration_protocol"]["temperature_status"] == "fallback_T1"


def test_persistence_and_rerender(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    images, logs, synced, events = [], [], [], []

    def image(path):
        images.append(path)
        return {"image": path}

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=image))

    class Artifacts:
        local = tmp_path

        def sync(self, path):
            assert path.exists()
            synced.append(path.name)
            events.append(path.name)

        def save(self, value, name):
            path = self.local / name
            temporary = path.with_suffix(".tmp")
            torch.save(value, temporary)
            temporary.replace(path)
            self.sync(path)
            return path

    logits, labels = fixture_data()
    report, predictions = fr.calibrate_and_score(logits, labels, logits, labels, n_boot=20)
    run = SimpleNamespace(log=logs.append, summary={})
    provenance = {"checkpoint": "best.pt:sha256-example", "data": "split-manifest-example"}
    history = [{"epoch": 1, "train": {"loss": 0.8, "acc": 0.5}, "val": report["val"], "lr": float("nan")}]
    paths = fr.write_analysis(report, predictions, history, Artifacts(), run, provenance)
    assert events[:2] == ["val_predictions.pt", "test_predictions.pt"]
    assert len(images) == 4
    assert run.summary["final/report"] == report
    assert run.summary["final/provenance"] == provenance
    assert set(synced) == {path.name for path in paths.values()}
    document = json.loads(paths["metrics"].read_text())
    assert document["history"][0]["lr"] is None
    assert document["history"][0]["val"] == report["val"]
    assert "train/acc" in paths["history_csv"].read_text()
    assert "not patient-level" in paths["report"].read_text()
    for split in ("val", "test"):
        payload = torch.load(paths[f"{split}_predictions"], weights_only=False)
        assert payload["provenance"] == provenance
        np.testing.assert_array_equal(payload["logits"], predictions[split]["logits"])
    fr.write_analysis(report, {s: paths[f"{s}_predictions"] for s in ("val", "test")}, [], Artifacts(), run, provenance)
    assert len(images) == 8
    with pytest.raises(ValueError, match="W&B"):
        fr.write_analysis(report, predictions, [], Artifacts(), None, provenance)
