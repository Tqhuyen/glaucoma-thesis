import numpy as np


def _interp_axis(a, axis, m):
    n = a.shape[axis]
    if n == m:
        return a
    xi = np.linspace(0.0, n - 1.0, m)
    x0 = np.floor(xi).astype(np.int64)
    x1 = np.minimum(x0 + 1, n - 1)
    w = (xi - x0).astype(a.dtype)
    a0 = np.take(a, x0, axis=axis)
    a1 = np.take(a, x1, axis=axis)
    shape = [1] * a.ndim
    shape[axis] = m
    w = w.reshape(shape)
    return a0 * (1.0 - w) + a1 * w


def resize_volume(vol, size):
    size = tuple(int(s) for s in size)
    out = np.asarray(vol, dtype=np.float32)
    for ax, m in enumerate(size):
        out = _interp_axis(out, ax, m)
    return out


def _box1d(a, axis, k):
    if k <= 1:
        return a
    pad = k // 2
    a = np.moveaxis(a, axis, 0)
    ap = np.concatenate([np.repeat(a[:1], pad, 0), a, np.repeat(a[-1:], pad, 0)], 0)
    cs = np.cumsum(ap, axis=0, dtype=np.float64)
    cs = np.concatenate([np.zeros_like(cs[:1]), cs], 0)
    out = (cs[k:] - cs[:-k]) / float(k)
    return np.moveaxis(out.astype(a.dtype), 0, axis)


def smooth_volume(vol, sigma):
    k = max(3, 2 * int(round(float(sigma))) + 1)
    out = np.asarray(vol, dtype=np.float32)
    for _ in range(2):
        for ax in range(3):
            out = _box1d(out, ax, k)
    return out


def downsample_volume(vol, size, mode="gaussian_trilinear"):
    vol = np.asarray(vol)
    size = tuple(int(s) for s in size)
    if mode == "maxpool":
        factors = [vol.shape[i] / size[i] for i in range(3)]
        if all(abs(f - round(f)) < 1e-9 and round(f) >= 1 for f in factors):
            f = tuple(int(round(x)) for x in factors)
            s = vol.shape
            v = vol[: s[0] - s[0] % f[0], : s[1] - s[1] % f[1], : s[2] - s[2] % f[2]]
            v = v.reshape(v.shape[0] // f[0], f[0], v.shape[1] // f[1], f[1], v.shape[2] // f[2], f[2])
            return v.max(axis=(1, 3, 5))
    if mode == "gaussian_trilinear":
        factors = [vol.shape[i] / size[i] for i in range(3)]
        sigma = float(np.mean(factors)) / 2.0
        vol = smooth_volume(vol, sigma)
    return resize_volume(vol, size)


def psnr(a, b, data_range=255.0):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def _box3d(vol, k):
    out = vol
    for ax in range(3):
        out = _box1d(out, ax, k)
    return out


def ssim3d(a, b, data_range=255.0, win=7):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ux = _box3d(a, win)
    uy = _box3d(b, win)
    uxx = _box3d(a * a, win)
    uyy = _box3d(b * b, win)
    uxy = _box3d(a * b, win)
    vx = uxx - ux * ux
    vy = uyy - uy * uy
    vxy = uxy - ux * uy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    return float(np.mean(s))


def linear_cka(X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xty = X.T @ Y
    xtx = X.T @ X
    yty = Y.T @ Y
    hsic = float(np.sum(xty * xty))
    denom = float(np.sqrt(np.sum(xtx * xtx)) * np.sqrt(np.sum(yty * yty)))
    return hsic / denom if denom > 0 else 0.0


def expected_calibration_error(probs, labels, n_bins=15):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    conf = np.where(probs >= 0.5, probs, 1.0 - probs)
    pred = (probs >= 0.5).astype(np.int64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(m):
            continue
        acc = float(np.mean(pred[m] == labels[m]))
        avg_conf = float(np.mean(conf[m]))
        ece += (m.sum() / n) * abs(acc - avg_conf)
    return float(ece)


def _rank_avg(scores):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    s = np.asarray(scores)[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def roc_auc_score_np(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rank_avg(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision_np(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / tp[-1]
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_r)
        prev_r = r
    return float(ap)


def classification_metrics(probs, labels, threshold=0.5):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pred = (probs >= threshold).astype(np.int64)
    tp = int(np.sum((pred == 1) & (labels == 1)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    tn = int(np.sum((pred == 0) & (labels == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    sens = rec
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "auc_roc": roc_auc_score_np(labels, probs),
        "auc_pr": average_precision_np(labels, probs),
        "f1": float(f1),
        "balanced_acc": float((sens + spec) / 2.0),
        "precision": float(prec),
        "recall": float(rec),
        "ece": expected_calibration_error(probs, labels),
    }


def silhouette_np(X, labels):
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    n = len(X)
    uniq = np.unique(labels)
    if len(uniq) < 2 or n < 3:
        return float("nan")
    d = np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0.0))
    out = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        a = d[i, same].mean() if same.sum() > 0 else 0.0
        b = min(d[i, labels == c].mean() for c in uniq if c != labels[i])
        out[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(out.mean())
