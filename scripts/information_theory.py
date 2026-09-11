import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _as_2d(value):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("Expected a 1D or 2D array")
    return array


def _standardize(array):
    array = _as_2d(array)
    mean = array.mean(0, keepdims=True)
    std = array.std(0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (array - mean) / std, mean, std


class PCA:
    def __init__(self, dim):
        self.dim = int(dim)
        self.mean = None
        self.components = None

    def fit(self, x):
        array = _as_2d(x)
        self.mean = array.mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(array - self.mean, full_matrices=False)
        self.components = vt[: max(1, min(self.dim, len(vt)))]
        return self

    def transform(self, x):
        if self.mean is None:
            raise RuntimeError("PCA must be fit before transform")
        return (_as_2d(x) - self.mean) @ self.components.T

    def fit_transform(self, x):
        return self.fit(x).transform(x)


def roc_auc_score(labels, scores):
    labels = np.asarray(labels, dtype=np.int64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    positive, negative = int((labels == 1).sum()), int((labels == 0).sum())
    if positive == 0 or negative == 0 or len(labels) != len(scores):
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ordered = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and ordered[j + 1] == ordered[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[labels == 1].sum() - positive * (positive + 1) / 2.0) / (positive * negative))


def binary_metrics(labels, probs, threshold=0.5):
    labels = np.asarray(labels, dtype=np.int64).ravel()
    probs = np.asarray(probs, dtype=np.float64).ravel()
    pred = (probs >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "acc": float((tp + tn) / max(len(labels), 1)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "auc": roc_auc_score(labels, probs),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": int(len(labels)),
    }


class MINEStats(nn.Module):
    def __init__(self, dx, dy, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(dx) + int(dy), int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1)).squeeze(1)


def _mi_estimate(net, x, y, bound):
    with torch.no_grad():
        permutation = torch.randperm(len(x), device=x.device)
        joint = net(x, y)
        marginal = net(x, y[permutation])
        if bound == "nwj":
            return float(joint.mean() - torch.exp(marginal - 1.0).mean())
        return float(joint.mean() - (torch.logsumexp(marginal, dim=0) - math.log(len(x))))


def mine_mi(
    x,
    y,
    *,
    bound="dv",
    steps=600,
    batch=64,
    hidden=64,
    lr=1e-3,
    seed=0,
    device="cpu",
    val_frac=0.2,
    log_every=0,
):
    if bound not in ("dv", "nwj"):
        raise ValueError("bound must be 'dv' or 'nwj'")
    x, y = _as_2d(x), _as_2d(y)
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    if len(x) < 4:
        raise ValueError("At least four samples are required")
    x, _, _ = _standardize(x)
    y, _, _ = _standardize(y)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    n_val = min(max(2, int(round(len(x) * val_frac))), len(x) - 2)
    val_idx, train_idx = order[:n_val], order[n_val:]
    xt = torch.as_tensor(x[train_idx], dtype=torch.float32, device=device)
    yt = torch.as_tensor(y[train_idx], dtype=torch.float32, device=device)
    xv = torch.as_tensor(x[val_idx], dtype=torch.float32, device=device)
    yv = torch.as_tensor(y[val_idx], dtype=torch.float32, device=device)
    net = MINEStats(xt.shape[1], yt.shape[1], hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    size = min(int(batch), len(xt))
    curve = []
    for step in range(max(1, int(steps))):
        index = torch.as_tensor(rng.integers(0, len(xt), size), device=device)
        xb, yb = xt[index], yt[index]
        permutation = torch.randperm(size, device=device)
        joint, marginal = net(xb, yb), net(xb, yb[permutation])
        if bound == "nwj":
            objective = joint.mean() - torch.exp(marginal - 1.0).mean()
        else:
            objective = joint.mean() - (torch.logsumexp(marginal, dim=0) - math.log(size))
        optimizer.zero_grad(set_to_none=True)
        (-objective).backward()
        optimizer.step()
        if log_every and (step + 1) % int(log_every) == 0:
            curve.append({"step": step + 1, "mi_val": _mi_estimate(net, xv, yv, bound)})
    return {
        "mi": _mi_estimate(net, xv, yv, bound),
        "mi_train": _mi_estimate(net, xt, yt, bound),
        "bound": bound,
        "steps": int(steps),
        "n": int(len(x)),
        "curve": curve,
    }


def ksg_mi(x, y, *, k=5, max_n=500, seed=0):
    x, y = _as_2d(x), _as_2d(y)
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    if len(x) < 3:
        raise ValueError("At least three samples are required")
    if len(x) > int(max_n):
        selection = np.random.default_rng(seed).choice(len(x), int(max_n), replace=False)
        x, y = x[selection], y[selection]
    x, _, _ = _standardize(x)
    y, _, _ = _standardize(y)
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    dx = torch.cdist(xt, xt, p=float("inf"))
    dy = torch.cdist(yt, yt, p=float("inf"))
    joint = torch.maximum(dx, dy).fill_diagonal_(float("inf"))
    neighbours = int(max(1, min(int(k), len(xt) - 2)))
    eps = torch.kthvalue(joint, neighbours, dim=1).values
    nx = (dx <= eps[:, None]).sum(1) - 1
    ny = (dy <= eps[:, None]).sum(1) - 1
    n = len(xt)
    mi = (
        torch.digamma(torch.tensor(float(neighbours)))
        + torch.digamma(torch.tensor(float(n)))
        - torch.digamma(nx + 1).mean()
        - torch.digamma(ny + 1).mean()
    )
    return float(mi)


def _equal_count_bins(values, bins):
    values = np.asarray(values, dtype=np.float64).ravel()
    unique = len(np.unique(values))
    if unique < 2:
        return None
    bins = int(min(max(2, int(bins)), unique))
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    binned = np.floor(ranks * bins / len(values)).astype(np.int64)
    return np.clip(binned, 0, bins - 1), bins


def _binned_mi(bx, by):
    joint = np.zeros((int(bx.max()) + 1, int(by.max()) + 1), dtype=np.float64)
    np.add.at(joint, (bx, by), 1.0)
    joint /= joint.sum()
    px = joint.sum(1, keepdims=True)
    py = joint.sum(0, keepdims=True)
    mask = joint > 0
    return float((joint[mask] * np.log(joint[mask] / (px @ py)[mask])).sum())


def mic_approx(x, y, *, max_bins=None, min_bins=2):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    cap = int(max_bins) if max_bins else max(int(min_bins) + 1, int(round(len(x) ** 0.6)))
    best = {"mic": 0.0, "mi": 0.0, "grid": None}
    for bx in range(int(min_bins), cap + 1):
        x_bins = _equal_count_bins(x, bx)
        if x_bins is None:
            continue
        for by in range(int(min_bins), cap // bx + 1):
            y_bins = _equal_count_bins(y, by)
            if y_bins is None:
                continue
            mi = _binned_mi(x_bins[0], y_bins[0])
            score = mi / math.log(min(x_bins[1], y_bins[1]))
            if score > best["mic"]:
                best = {"mic": float(score), "mi": float(mi), "grid": [x_bins[1], y_bins[1]]}
    return best


def max_mic_features(embedding, labels, *, pcs=8, max_bins=None):
    x = _as_2d(embedding)
    y = np.asarray(labels, dtype=np.float64).ravel()
    dim = max(1, min(int(pcs), x.shape[1], max(1, len(x) - 1)))
    projected = PCA(dim).fit_transform(x)
    scores = [mic_approx(projected[:, i], y, max_bins=max_bins)["mic"] for i in range(projected.shape[1])]
    return {"max_mic": float(max(scores)), "per_pc": scores, "pcs": dim}


def linear_probe(
    xtr,
    ytr,
    xte,
    yte,
    *,
    kind="logistic",
    hidden=64,
    steps=400,
    lr=1e-2,
    weight_decay=1e-4,
    seed=0,
    device="cpu",
):
    if kind not in ("logistic", "mlp"):
        raise ValueError("kind must be 'logistic' or 'mlp'")
    train, test = _as_2d(xtr), _as_2d(xte)
    ytr = np.asarray(ytr, dtype=np.int64).ravel()
    yte = np.asarray(yte, dtype=np.int64).ravel()
    mean = train.mean(0, keepdims=True)
    std = np.where(train.std(0, keepdims=True) < 1e-8, 1.0, train.std(0, keepdims=True))
    train = (train - mean) / std
    test = (test - mean) / std
    torch.manual_seed(seed)
    if kind == "mlp":
        model = nn.Sequential(nn.Linear(train.shape[1], int(hidden)), nn.ReLU(), nn.Linear(int(hidden), 2))
    else:
        model = nn.Linear(train.shape[1], 2)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    inputs = torch.as_tensor(train, dtype=torch.float32, device=device)
    targets = torch.as_tensor(ytr, device=device)
    for _ in range(max(1, int(steps))):
        loss = F.cross_entropy(model(inputs), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(test, dtype=torch.float32, device=device))
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    metrics = binary_metrics(yte, probs)
    metrics.update({"kind": kind, "steps": int(steps), "probs": probs.tolist()})
    return metrics


def probe_auc_cv(embedding, labels, *, folds=3, seed=0, **kwargs):
    x = _as_2d(embedding)
    y = np.asarray(labels, dtype=np.int64).ravel()
    order = np.random.default_rng(seed).permutation(len(x))
    folds = max(2, min(int(folds), len(x)))
    chunks = np.array_split(order, folds)
    rows = []
    for index in range(folds):
        test_idx = chunks[index]
        train_idx = np.concatenate([chunks[j] for j in range(folds) if j != index])
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        result = linear_probe(x[train_idx], y[train_idx], x[test_idx], y[test_idx], seed=seed, **kwargs)
        rows.append({"acc": result["acc"], "auc": result["auc"], "f1": result["f1"]})
    if not rows:
        return {"acc": float("nan"), "auc": float("nan"), "f1": float("nan"), "folds": 0}
    metrics = {key: float(np.nanmean([row[key] for row in rows])) for key in ("acc", "auc", "f1")}
    metrics["folds"] = len(rows)
    return metrics


def conditional_mi(x, y, z, *, steps=600, seed=0, device="cpu", hidden=64):
    x, y, z = _as_2d(x), _as_2d(y), _as_2d(z)
    joint_input = np.concatenate([x, z], axis=1)
    joint = mine_mi(joint_input, y, steps=steps, seed=seed, device=device, hidden=hidden)["mi"]
    conditioning = mine_mi(z, y, steps=steps, seed=seed, device=device, hidden=hidden)["mi"]
    return {
        "mi_xy_given_z": float(joint - conditioning),
        "mi_joint_xz_y": float(joint),
        "mi_z_y": float(conditioning),
    }


def collect_embeddings(model, dataset, batch_size, *, input_res=None, device=None):
    device = device or next(model.parameters()).device
    model.eval()
    fused, e3d, e2d, inputs, labels, logits = [], [], [], [], [], []
    pin = device.type == "cuda"
    with torch.no_grad():
        for x, views, target in DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=pin):
            x = x.to(device, non_blocking=True)
            views = views.to(device, non_blocking=True)
            z, tokens = model.fuse(x, views)
            fused.append(z.float().cpu().numpy())
            e3d.append(tokens[:, 0].float().cpu().numpy())
            e2d.append(tokens[:, 1:].float().cpu().numpy())
            if input_res:
                inputs.append(F.adaptive_avg_pool3d(x, int(input_res)).flatten(1).float().cpu().numpy())
            labels.append(np.asarray(target))
            logits.append(model.head(z).float().cpu().numpy())
    result = {
        "z": np.concatenate(fused),
        "e3d": np.concatenate(e3d),
        "e2d": np.concatenate(e2d),
        "y": np.concatenate(labels),
        "logits": np.concatenate(logits),
    }
    result["probs"] = torch.softmax(torch.as_tensor(result["logits"]), dim=1)[:, 1].numpy()
    if inputs:
        result["x"] = np.concatenate(inputs)
    return result


def surrogate_test(estimate, x, y, *, n=100, seed=0):
    y = np.asarray(y)
    real = float(estimate(x, y))
    rng = np.random.default_rng(seed)
    null = np.asarray([float(estimate(x, rng.permutation(y))) for _ in range(int(n))], dtype=np.float64)
    deviation = float(null.std(ddof=1)) if len(null) > 1 else 0.0
    return {
        "real": real,
        "null_mean": float(null.mean()),
        "null_std": deviation,
        "p_value": float((1 + int((null >= real).sum())) / (len(null) + 1)),
        "z_score": float((real - null.mean()) / (deviation + 1e-12)) if len(null) > 1 else float("nan"),
        "n": int(n),
        "null": null.tolist(),
    }


def information_plane(
    inputs,
    embeddings,
    labels,
    *,
    folds=3,
    pca_dim=32,
    probe_kind="logistic",
    probe_steps=400,
    mine_steps=300,
    seed=0,
    device="cpu",
    hidden=64,
):
    x = _as_2d(inputs)
    y = np.asarray(labels, dtype=np.int64).ravel()
    zs = [_as_2d(z) for z in embeddings]
    if not zs:
        return []
    dim = max(1, min(int(pca_dim), x.shape[1], max(1, len(x) - 1)))
    reduced = PCA(dim).fit_transform(x)
    n = min([len(reduced), len(y)] + [len(z) for z in zs])
    rows = []
    for epoch, z in enumerate(zs):
        zz, yy, xx = z[:n], y[:n], reduced[:n]
        probe = probe_auc_cv(zz, yy, folds=folds, kind=probe_kind, steps=probe_steps, seed=seed, device=device)
        rows.append(
            {
                "epoch": epoch,
                "mi_zy": mine_mi(zz, yy[:, None], steps=mine_steps, seed=seed, device=device, hidden=hidden)["mi"],
                "mi_xz": mine_mi(xx, zz, steps=mine_steps, seed=seed, device=device, hidden=hidden)["mi"],
                "probe_auc": probe["auc"],
                "probe_acc": probe["acc"],
            }
        )
    return rows


def validate_estimators(*, n=800, steps=500, seed=0, device="cpu"):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, (int(n), 2)).astype(np.float64)
    xor = np.logical_xor(bits[:, 0], bits[:, 1]).astype(np.float64)[:, None]
    independent = rng.integers(0, 2, (int(n), 1)).astype(np.float64)
    continuous = rng.normal(size=(int(n), 1))
    dependent = 0.9 * continuous + 0.3 * rng.normal(size=(int(n), 1))
    xor_mi = mine_mi(bits, xor, steps=steps, seed=seed, device=device)["mi"]
    independent_mi = mine_mi(bits, independent, steps=steps, seed=seed, device=device)["mi"]
    dependent_ksg = ksg_mi(continuous, dependent, seed=seed)
    independent_ksg = ksg_mi(continuous, rng.normal(size=(int(n), 1)), seed=seed)
    return {
        "xor_mi": float(xor_mi),
        "xor_target": float(math.log(2.0)),
        "independent_mi": float(independent_mi),
        "dependent_ksg": float(dependent_ksg),
        "independent_ksg": float(independent_ksg),
        "xor_ok": bool(xor_mi > 0.3),
        "independent_ok": bool(independent_mi < 0.25),
        "ksg_dependent_ok": bool(dependent_ksg > 0.1),
        "ksg_independent_ok": bool(abs(independent_ksg) < 0.1),
    }


def plot_information_plane(rows, path, title="Information plane"):
    import matplotlib.pyplot as plt

    xs = [row["mi_xz"] for row in rows]
    ys = [row["mi_zy"] for row in rows]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(xs, ys, "o-")
    for row in rows:
        ax.annotate(str(row["epoch"]), (row["mi_xz"], row["mi_zy"]), textcoords="offset points", xytext=(4, 3))
    ax.set(xlabel="I(X;Z) MINE", ylabel="I(Z;Y) MINE", title=title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_surrogate(results, path, title="Surrogate data testing"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, result in results.items():
        ax.hist(
            result["null"],
            bins=int(min(20, max(5, result["n"]))),
            alpha=0.4,
            label=f"{name} null p={result['p_value']:.3f}",
        )
        ax.axvline(result["real"], linestyle="--")
    ax.set(xlabel="MI estimate", ylabel="surrogate count", title=title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_probes(rows, path, title="Linear probes: train to test"):
    import matplotlib.pyplot as plt

    positions = np.arange(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(positions - width / 2, [row["acc"] for row in rows], width, label="acc")
    ax.bar(positions + width / 2, [row["auc"] for row in rows], width, label="AUC")
    ax.set_xticks(positions)
    ax.set_xticklabels([row["name"] for row in rows], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set(title=title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)
