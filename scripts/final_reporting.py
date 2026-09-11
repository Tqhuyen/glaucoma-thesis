"""CPU-only final evaluation; requires numpy, scipy, sklearn, torch and matplotlib.

ECE measures positive-class probability, not predicted-class confidence, in ten
equal-width bins [0,.1), ... [.9,1]. Undefined precision/F1/MCC are zero (sklearn
convention); missing intervals and nonfinite historical values serialize as null.
Final evaluation requires both classes. Saved .pt files must be trusted locally.
"""

import csv
import io
import json
import os
import uuid
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn import metrics as skm

METRIC_VERSION = "final-binary-1.0.0"


def _array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _labels(labels, n):
    labels = _array(labels)
    if labels.shape != (n,) or not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be a finite binary vector matching predictions")
    if np.unique(labels).size != 2:
        raise ValueError("final evaluation requires both classes")
    return labels.astype(np.int64)


def _calibration_bins(probs, labels):
    bins = np.searchsorted(np.linspace(0, 1, 11), probs, side="right") - 1
    bins = np.clip(bins, 0, 9)
    counts = np.bincount(bins, minlength=10)
    means = np.divide(np.bincount(bins, weights=probs, minlength=10), counts, out=np.zeros(10), where=counts > 0)
    rates = np.divide(np.bincount(bins, weights=labels, minlength=10), counts, out=np.zeros(10), where=counts > 0)
    return counts, means, rates


def metrics(probs, labels, threshold=0.5, ranking_scores=None):
    """Binary metrics with >= threshold; auc_pr is sklearn average precision.

    Threshold may exceed one to represent all-negative predictions. Raw logit
    margins should be supplied for ranking to avoid probability saturation ties.
    """
    probs = _array(probs)
    if probs.ndim != 1 or not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError("probs must be a finite vector in [0, 1]")
    labels = _labels(labels, len(probs))
    if not np.isscalar(threshold) or not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    scores = probs if ranking_scores is None else _array(ranking_scores)
    if scores.shape != probs.shape or not np.isfinite(scores).all():
        raise ValueError("ranking_scores must be finite and match probs")
    pred = probs >= threshold
    tn, fp, fn, tp = (int(v) for v in skm.confusion_matrix(labels, pred, labels=[0, 1]).ravel())
    counts, means, rates = _calibration_bins(probs, labels)
    recall = float(skm.recall_score(labels, pred, zero_division=0))
    return {
        "acc": float(skm.accuracy_score(labels, pred)),
        "balanced_acc": float(skm.balanced_accuracy_score(labels, pred)),
        "precision": float(skm.precision_score(labels, pred, zero_division=0)),
        "recall": recall,
        "sensitivity": recall,
        "specificity": tn / (tn + fp),
        "f1": float(skm.f1_score(labels, pred, zero_division=0)),
        "mcc": float(skm.matthews_corrcoef(labels, pred)),
        "auc_roc": float(skm.roc_auc_score(labels, scores)),
        "auc_pr": float(skm.average_precision_score(labels, scores)),
        "ece": float(np.sum(counts * np.abs(means - rates)) / len(labels)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": len(labels),
    }


def _logits(logits, labels):
    logits = _array(logits)
    if logits.ndim != 2 or logits.shape[1] != 2 or not np.isfinite(logits).all():
        raise ValueError("logits must be finite with shape (n, 2)")
    labels = _labels(labels, len(logits))
    with np.errstate(over="ignore", invalid="ignore"):
        margin = logits[:, 1] - logits[:, 0]
    if not np.isfinite(margin).all():
        raise ValueError("logit margin overflow")
    return logits, labels, margin


def _nll(margin, labels, temperature):
    with np.errstate(over="ignore", invalid="ignore"):
        losses = np.logaddexp(0, (1 - 2 * labels) * (margin / temperature))
        return float(np.sum(losses / len(labels)))


def _threshold(probs, labels):
    candidates = np.unique(np.r_[0.0, 0.5, 1.0, probs, np.nextafter(probs.max(), np.inf)])
    order = np.argsort(probs)
    sorted_labels = labels[order]
    positives = np.r_[0, np.cumsum(sorted_labels)]
    negatives = np.r_[0, np.cumsum(1 - sorted_labels)]
    positions = np.searchsorted(probs[order], candidates, side="left")
    objective = negatives[positions] * positives[-1] - positives[positions] * negatives[-1]
    best = candidates[objective == objective.max()]
    return float(min(best, key=lambda t: (abs(t - 0.5), t)))


def calibrate_and_score(val_logits, val_labels, test_logits, test_labels, n_boot=1000, seed=42):
    """Fit on validation only; percentile CIs condition on fixed fitted choices.

    Indices are split-local row offsets, not patient identifiers. n_boot is the
    number of attempted ordinary scan resamples; single-class draws are skipped.
    """
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 0:
        raise ValueError("n_boot must be a nonnegative integer")
    vl, vy, vm = _logits(val_logits, val_labels)
    tl, ty, tm = _logits(test_logits, test_labels)
    baseline = _nll(vm, vy, 1.0)
    if not np.isfinite(baseline):
        raise ValueError("validation NLL is not finite")
    fit = minimize_scalar(lambda log_t: _nll(vm, vy, np.exp(log_t)), bounds=(-20, 20), method="bounded")
    candidate = float(np.exp(fit.x))
    fitted_loss = _nll(vm, vy, candidate)
    accepted = fit.success and np.isfinite(candidate) and candidate > 0 and np.isfinite(fitted_loss)
    accepted = accepted and fitted_loss <= baseline
    temperature = candidate if accepted else 1.0
    vp = expit(vm / temperature)
    threshold = _threshold(vp, vy)
    report = {
        "temperature": temperature,
        "threshold": threshold,
        "metric_version": METRIC_VERSION,
        "uncalibrated": {},
        "operating_point_05": {},
        "calibration_protocol": {
            "selection_split": "val",
            "temperature_status": "fitted" if accepted else "fallback_T1",
            "optimizer_success": bool(fit.success),
            "temperature_method": "float64 CPU bounded logT [-20,20] NLL",
            "val_nll_before": baseline,
            "val_nll_after": _nll(vm, vy, temperature),
            "threshold_objective": "maximum validation Youden J (equivalently balanced accuracy)",
            "threshold_candidates": "unique calibrated validation scores, 0, 0.5, 1, nextafter(max,+inf)",
            "threshold_ties": "closest to 0.5, then smallest; prediction uses >=",
            "ranking": "raw logit margin; sklearn ROC AUC and average precision (not trapezoidal PR AUC)",
            "ece": "positive probability; 10 equal-width bins [0,.1), ... [.9,1]; weighted absolute gap",
            "undefined_policy": "precision/F1/MCC zero; unavailable CI and nonfinite history null; invalid inputs rejected",
            "validation_caveat": "validation metrics are selection-set estimates, not held-out estimates",
        },
    }
    predictions = {}
    for split, logits, labels, margin in (("val", vl, vy, vm), ("test", tl, ty, tm)):
        raw, calibrated = expit(margin), expit(margin / temperature)
        loss = _nll(margin, labels, temperature)
        raw_loss = _nll(margin, labels, 1.0)
        if not np.isfinite([loss, raw_loss]).all():
            raise ValueError(f"{split} NLL is not finite")
        report[split] = {**metrics(calibrated, labels, threshold, margin), "loss": loss}
        report["uncalibrated"][split] = {**metrics(raw, labels, 0.5, margin), "loss": raw_loss}
        report["operating_point_05"][split] = {**metrics(calibrated, labels, 0.5, margin), "loss": loss}
        predictions[split] = {
            "logits": logits,
            "labels": labels,
            "indices": np.arange(len(labels)),
            "raw_probs": raw,
            "calibrated_probs": calibrated,
            "ranking_scores": margin,
            "temperature": temperature,
            "threshold": threshold,
            "metric_version": METRIC_VERSION,
            "index_scope": "split-local row offsets; not patient IDs",
        }
    keys = [k for k in report["test"] if k not in {"tp", "tn", "fp", "fn", "n"}]
    samples = {k: [] for k in keys}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        idx = rng.integers(0, len(ty), len(ty))
        if np.unique(ty[idx]).size != 2:
            continue
        result = metrics(predictions["test"]["calibrated_probs"][idx], ty[idx], threshold, tm[idx])
        result["loss"] = _nll(tm[idx], ty[idx], temperature)
        for key in keys:
            samples[key].append(result[key])
    valid = len(samples[keys[0]])
    report["test_ci"] = {
        key: np.percentile(values, [2.5, 97.5]).tolist() if values else [None, None] for key, values in samples.items()
    }
    report["bootstrap"] = {
        "requested": int(n_boot),
        "valid": valid,
        "skipped_single_class": int(n_boot) - valid,
        "seed": int(seed),
        "confidence": 0.95,
        "method": "ordinary scan-level percentile bootstrap; single-class draws skipped",
        "conditional": "fixed checkpoint, validation temperature and threshold; no refitting on test resamples",
        "limitations": "not patient-level or cluster-aware; excludes training/selection uncertainty; no patient claims",
    }
    return report, predictions


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_text(artifacts, name, text):
    path = Path(artifacts.local) / name
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    artifacts.sync(path)
    return path


def _flatten(row, prefix=""):
    result = {}
    for key, value in row.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(_flatten(value, name))
        else:
            result[name] = value
    return result


def write_analysis(report, predictions, history, artifacts, run, provenance):
    """Persist and rerender without inference. Return all output paths.

    artifacts provides local, atomic save(value,name) with sync, and sync(path).
    run is mandatory and already initialized by the caller. provenance should
    identify checkpoint and dataset/split order. Predictions may map split names
    to trusted .pt paths for rerendering; numpy payloads require weights_only=False.
    """
    if run is None:
        raise ValueError("an initialized W&B run is required")
    import matplotlib.pyplot as plt
    import wandb

    paths, saved = {}, {}
    for split in ("val", "test"):
        payload = predictions[split]
        if isinstance(payload, (str, Path)):
            payload = torch.load(payload, map_location="cpu", weights_only=False)
        payload = {**payload, "provenance": _json_safe(provenance), "split": split}
        paths[f"{split}_predictions"] = artifacts.save(payload, f"{split}_predictions.pt")
        saved[split] = torch.load(paths[f"{split}_predictions"], map_location="cpu", weights_only=False)
    document = _json_safe({**report, "history": list(history), "provenance": provenance})
    paths["metrics"] = _write_text(artifacts, "metrics.json", json.dumps(document, indent=2, allow_nan=False))
    rows = []
    for group in ("selected", "uncalibrated", "operating_point_05"):
        for split in ("val", "test"):
            values = report[split] if group == "selected" else report[group][split]
            for key, value in values.items():
                ci = (
                    report["test_ci"].get(key, [None, None])
                    if group == "selected" and split == "test"
                    else [None, None]
                )
                rows.append(
                    {"group": group, "split": split, "metric": key, "value": value, "ci_low": ci[0], "ci_high": ci[1]}
                )
    for name, records in (("metrics", rows), ("history", [_flatten(h) for h in document["history"]])):
        stream = io.StringIO(newline="")
        columns = list(dict.fromkeys(k for row in records for k in row)) or ["epoch"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(_json_safe(records))
        paths[f"{name}_csv"] = _write_text(artifacts, f"{name}.csv", stream.getvalue())
    lines = [
        "# Final Scientific Report",
        "",
        f"Metric version: {report['metric_version']}",
        f"Temperature: {report['temperature']:.8g}; selected threshold: {report['threshold']:.8g}",
        "",
        "| Metric | Validation (selection set) | Test | Test 95% conditional CI |",
        "| --- | ---: | ---: | --- |",
    ]
    for key, value in report["test"].items():
        lines.append(f"| {key} | {report['val'][key]:.6g} | {value:.6g} | {report['test_ci'].get(key, 'N/A')} |")
    lines += [
        "",
        "## Protocol and Limitations",
        "",
        *[f"- {k}: {v}" for k, v in report["calibration_protocol"].items()],
        *[f"- Bootstrap {k}: {v}" for k, v in report["bootstrap"].items()],
        "",
        "Loss is mean binary softmax NLL, separate from threshold metrics.",
        "Uncalibrated and calibrated 0.5 operating points are in metrics.json and metrics.csv.",
        "All available history is retained; absent training metrics are not reconstructed.",
        "",
        "## Provenance",
        "",
        "```json",
        json.dumps(document["provenance"], indent=2),
        "```",
    ]
    paths["report"] = _write_text(artifacts, "report.md", "\n".join(lines) + "\n")

    def figure(name, fig):
        path = Path(artifacts.local) / f"{name}.png"
        try:
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            artifacts.sync(path)
            run.log({f"figures/{name}": wandb.Image(str(path))})
            paths[name] = path
        finally:
            plt.close(fig)

    hist = [_flatten(h) for h in document["history"]]
    numeric = sorted(
        {k for h in hist for k, v in h.items() if k not in {"epoch", "step"} and isinstance(v, (int, float))}
    )
    fig, axes = plt.subplots(
        max(1, (len(numeric) + 2) // 3), 3, figsize=(12, 3 * max(1, (len(numeric) + 2) // 3)), squeeze=False
    )
    for ax, key in zip(axes.flat, numeric):
        ax.plot([h.get("epoch", i + 1) for i, h in enumerate(hist)], [h.get(key, np.nan) for h in hist])
        ax.set(title=key, xlabel="Epoch / history row")
    for ax in list(axes.flat)[len(numeric) :]:
        ax.set_axis_off()
    if not numeric:
        fig.text(0.2, 0.5, "No numeric training history supplied")
    figure("curves", fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split, data in saved.items():
        labels, scores = data["labels"], data["ranking_scores"]
        fpr, tpr, _ = skm.roc_curve(labels, scores)
        precision, recall, _ = skm.precision_recall_curve(labels, scores)
        axes[0].plot(fpr, tpr, label=split)
        axes[1].step(recall, precision, where="post", label=split)
    for ax, title, x, y in zip(
        axes, ("ROC (raw margins)", "Precision-Recall (raw margins)"), ("FPR", "Recall"), ("TPR", "Precision")
    ):
        ax.set(title=title, xlabel=x, ylabel=y, xlim=(0, 1), ylim=(0, 1.02))
        ax.legend()
    figure("roc_pr", fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (split, data) in zip(axes, saved.items()):
        for key in ("raw_probs", "calibrated_probs"):
            counts, means, rates = _calibration_bins(data[key], data["labels"])
            mask = counts > 0
            ax.plot(means[mask], rates[mask], "o-", label=key)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(
            title=f"{split}: 10-bin positive calibration",
            xlabel="Mean P(y=1)",
            ylabel="Positive fraction",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend()
    figure("calibration", fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, (split, data) in zip(axes, saved.items()):
        matrix = skm.confusion_matrix(data["labels"], data["calibrated_probs"] >= data["threshold"], labels=[0, 1])
        skm.ConfusionMatrixDisplay(matrix, display_labels=[0, 1]).plot(ax=ax, colorbar=False)
        ax.set_title(f"{split}: threshold {data['threshold']:.5g}")
    figure("confusion", fig)
    run.log({f"final/{split}/{k}": v for split in ("val", "test") for k, v in report[split].items()})
    run.summary.update(
        {
            "final/metric_version": METRIC_VERSION,
            "final/temperature": report["temperature"],
            "final/threshold": report["threshold"],
            "final/report": _json_safe(report),
            "final/provenance": document["provenance"],
        }
    )
    return paths
