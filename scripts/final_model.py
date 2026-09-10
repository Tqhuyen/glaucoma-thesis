import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resolution_study as rs  # noqa: E402

VIEWS = ("slab_mip", "aip_full")


def gnorm(c, groups=None):
    g = min(c, 8) if groups is None else groups
    while c % g != 0:
        g -= 1
    return nn.GroupNorm(g, c)


class ResXBlock3D(nn.Module):
    def __init__(self, cin, cout, stride=(1, 1, 1), groups=8, base_width=8):
        super().__init__()
        width = int(cout * base_width / 64.0) * groups
        self.c1 = nn.Conv3d(cin, width, 1, bias=False)
        self.g1 = gnorm(width)
        self.c2 = nn.Conv3d(width, width, 3, stride=stride, padding=1, groups=groups, bias=False)
        self.g2 = gnorm(width)
        self.c3 = nn.Conv3d(width, cout, 1, bias=False)
        self.g3 = gnorm(cout)
        self.short = (stride != (1, 1, 1) or cin != cout)
        if self.short:
            self.sc = nn.Conv3d(cin, cout, 1, stride=stride, bias=False)
            self.sg = gnorm(cout)

    def forward(self, x):
        r = self.c3(F.relu(self.g2(self.c2(F.relu(self.g1(self.c1(x)))))))
        if self.short:
            x = self.sg(self.sc(x))
        return F.relu(x + r)


class Enc3DResNeXt(nn.Module):
    def __init__(self, in_ch=1, features=(32, 64, 128, 192),
                 strides=((2, 2, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2))):
        super().__init__()
        self.out_dim = int(features[-1])
        self.stem = nn.Sequential(nn.Conv3d(in_ch, features[0], 3, padding=1, bias=False),
                                  gnorm(features[0]), nn.ReLU())
        self.stages = nn.ModuleList()
        cin = features[0]
        for i, (c, st) in enumerate(zip(features, strides)):
            if i > 0:
                self.stages.append(ResXBlock3D(cin, c, tuple(int(s) for s in st)))
                cin = c
            self.stages.append(ResXBlock3D(cin, c, (1, 1, 1)))
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.stem(x)
        for st in self.stages:
            x = st(x)
        return self.pool(x).flatten(1)

    def gcam_module(self):
        return find_last_conv(self.stages[-1])


class Timm2D(nn.Module):
    def __init__(self, name="maxvit_tiny_rw_224", pretrained=True, in_ch=1):
        super().__init__()
        import timm
        try:
            self.net = timm.create_model(name, pretrained=pretrained, num_classes=0,
                                         global_pool="avg", in_chans=in_ch)
        except Exception:
            self.net = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
            self._expand = True
        self.out_dim = int(self.net.num_features)

    def forward(self, x):
        if getattr(self, "_expand", False) and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.net(x)

    def gcam_module(self):
        m = find_last_conv(self.net, nn.Conv2d)
        return m


def find_last_conv(m, conv_cls=nn.Conv3d):
    last = None
    for _, mod in m.named_modules():
        if isinstance(mod, conv_cls):
            last = mod
    return last


class Proj(nn.Module):
    def __init__(self, d, D):
        super().__init__()
        self.fc = nn.Linear(d, D)

    def forward(self, x):
        return F.relu(self.fc(x))


class CrossGate(nn.Module):
    def __init__(self, D, n_2d, heads=None):
        super().__init__()
        h = heads or next(x for x in (8, 4, 2, 1) if D % x == 0)
        self.norm = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, h, dropout=0.1, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.last_w = None
        self.last_gate = None

    def forward(self, c3, toks2d):
        q = c3.unsqueeze(1)
        h = self.norm(toks2d)
        out, w = self.attn(q, h, h, need_weights=True)
        self.last_w = w[..., 0, :].detach()
        self.last_gate = float(torch.sigmoid(self.gate).detach())
        z = c3 + torch.sigmoid(self.gate) * out.squeeze(1)
        return z


class FinalModel(nn.Module):
    def __init__(self, n_2d=2, D=256, num_classes=2, enc2d="maxvit_tiny_rw_224",
                 enc2d_pretrained=True, enc3d_features=(32, 64, 128, 192)):
        super().__init__()
        assert 1 <= n_2d <= len(VIEWS), "n_2d must be 1 or 2"
        self.n_2d = int(n_2d)
        self.enc3d = Enc3DResNeXt(features=tuple(enc3d_features))
        self.enc2ds = nn.ModuleList([Timm2D(enc2d, pretrained=enc2d_pretrained) for _ in range(self.n_2d)])
        dims = [self.enc3d.out_dim] + [e.out_dim for e in self.enc2ds]
        self.projs = nn.ModuleList([Proj(d, D) for d in dims])
        self.fusion = CrossGate(D, self.n_2d)
        self.head = nn.Linear(D, num_classes)

    def embed(self, x3d, views):
        es = [self.enc3d(x3d)]
        for i, enc in enumerate(self.enc2ds):
            es.append(enc(views[:, i]))
        return es

    def forward(self, x3d, views):
        es = self.embed(x3d, views)
        toks = torch.stack([p(e) for p, e in zip(self.projs, es)], dim=1)
        z = self.fusion(toks[:, 0], toks[:, 1:])
        return self.head(z)

    def gcam3d_module(self):
        return self.enc3d.gcam_module()

    def gcam2d_module(self, i):
        return self.enc2ds[i].gcam_module()


# ---------------------------------------------------------------- preprocessing
def depth_axis(vol):
    f = vol.astype(np.float32)
    stds = [float(f.mean(axis=tuple(i for i in range(3) if i != ax)).std()) for ax in range(3)]
    return int(np.argmax(stds))


def to_depth_last(vol, dz):
    if dz == 2:
        return vol
    others = [i for i in range(3) if i != dz]
    return np.transpose(vol, tuple(others) + (dz,))


def project_views(dvol, half=16):
    S = dvol.shape[2]
    half = min(int(half), max(2, S // 8))
    prof = dvol.mean(axis=(0, 1)).astype(np.float32)
    peak = int(prof.argmax())
    lo, hi = max(0, peak - half), min(S, peak + half + 1)
    slab_mip = dvol[:, :, lo:hi].max(axis=2)
    aip_full = dvol.mean(axis=2)
    return np.stack([np.round(slab_mip).clip(0, 255).astype(np.uint8),
                     np.round(aip_full).clip(0, 255).astype(np.uint8)], axis=0)


# ---------------------------------------------------------------- augmentation
def aug_volume(vol, rng, intensity=(0.9, 1.1), shift=8.0):
    x = vol.astype(np.float32)
    if rng.random() < 0.5:
        x = x[::-1]
    if rng.random() < 0.5:
        x = x[:, ::-1]
    k = rng.integers(0, 4)
    if k:
        x = np.rot90(x, k, axes=(0, 1))
    x = x * rng.uniform(*intensity) + rng.uniform(-shift, shift)
    return np.clip(x, 0, 255).astype(np.uint8)


def aug_views(views, rng, intensity=(0.9, 1.1), shift=8.0):
    out = np.empty_like(views)
    for i in range(views.shape[0]):
        v = views[i].astype(np.float32)
        if rng.random() < 0.5:
            v = v[::-1]
        if rng.random() < 0.5:
            v = v[:, ::-1]
        k = rng.integers(0, 4)
        if k:
            v = np.rot90(v, k)
        v = v * rng.uniform(*intensity) + rng.uniform(-shift, shift)
        out[i] = np.clip(v, 0, 255).astype(np.uint8)
    return out


def aug_pair(vol3d, views, rng, intensity=(0.9, 1.1), shift=8.0):
    v = vol3d.astype(np.float32)
    w = views.astype(np.float32)
    if rng.random() < 0.5:
        v = v[::-1]
        w = w[:, ::-1]
    if rng.random() < 0.5:
        v = v[:, ::-1]
        w = w[:, :, ::-1]
    k = int(rng.integers(0, 4))
    if k:
        v = np.rot90(v, k, axes=(0, 1))
        w = np.rot90(w, k, axes=(1, 2))
    s = float(rng.uniform(*intensity))
    b = float(rng.uniform(-shift, shift))
    v = np.clip(v * s + b, 0, 255)
    w = np.clip(w * s + b, 0, 255)
    return v.astype(np.uint8), w.astype(np.uint8)


# ---------------------------------------------------------------- metrics
def _mcc(tp, tn, fp, fn):
    num = tp * tn - fp * fn
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return float(num / den) if den > 0 else 0.0


def full_metrics(probs, labels, threshold=0.5):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pred = (probs >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    acc = (tp + tn) / max(len(labels), 1)
    return {
        "acc": float(acc), "balanced_acc": float((sens + spec) / 2),
        "precision": float(prec), "recall": float(sens), "sensitivity": float(sens),
        "specificity": float(spec), "f1": float(f1), "mcc": _mcc(tp, tn, fp, fn),
        "auc_roc": rs.roc_auc_score_np(labels, probs),
        "auc_pr": rs.average_precision_np(labels, probs),
        "ece": rs.expected_calibration_error(probs, labels),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": int(len(labels)),
    }


def bootstrap_ci(probs, labels, n_boot=1000, seed=0, threshold=0.5):
    rng = np.random.default_rng(seed)
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    n = len(labels)
    keys = ["auc_roc", "auc_pr", "f1", "balanced_acc", "mcc"]
    out = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        m = full_metrics(probs[idx], labels[idx], threshold)
        for k in keys:
            out[k].append(m[k])
    ci = {}
    for k in keys:
        v = np.asarray(out[k], dtype=float)
        ci[k] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else [float("nan")] * 2
    return ci


def tune_threshold(probs, labels):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    best_t, best_j = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (probs >= t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def temperature_scale(logits, labels, max_iter=200):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / torch.exp(logT), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(logT).item())


# ---------------------------------------------------------------- X-AI
class _ActHook:
    def __init__(self, module):
        self.act = None
        self.grad = None
        self.fh = module.register_forward_hook(self._fwd)
        self.bh = module.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.act = o

    def _bwd(self, m, gi, go):
        self.grad = go[0]

    def close(self):
        self.fh.remove()
        self.bh.remove()


def grad_cam(module, model, x3d, views, target=None, is_3d=True):
    model.eval()
    hook = _ActHook(module)
    x3d = x3d.clone().requires_grad_(False)
    views = views.clone().requires_grad_(False)
    logits = model(x3d, views)
    cls = int(logits.argmax(1).item()) if target is None else int(target)
    model.zero_grad(set_to_none=True)
    logits[0, cls].backward()
    act, grad = hook.act, hook.grad
    hook.close()
    w = grad.mean(dim=tuple(range(2, grad.ndim)), keepdim=True)
    cam = F.relu((w * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x3d.shape[2:] if is_3d else views.shape[-2:],
                        mode="trilinear" if is_3d else "bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, cls, float(F.softmax(logits, 1)[0, cls].item())


def occlusion_sensitivity(model, x3d, views, target, n=6, fill=0.0):
    model.eval()
    base = F.softmax(model(x3d, views), 1)[0, target].item()
    D = x3d.shape[2]
    step = D // n
    drop = np.zeros((n, n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                xo = x3d.clone()
                xo[:, :, i * step:(i + 1) * step, j * step:(j + 1) * step, k * step:(k + 1) * step] = fill
                with torch.no_grad():
                    p = F.softmax(model(xo, views), 1)[0, target].item()
                drop[i, j, k] = max(0.0, base - p)
    return drop, base


def integrated_gradients(model, x3d, views, target, steps=16):
    model.eval()
    baseline = torch.zeros_like(x3d)
    total = torch.zeros_like(x3d)
    for a in torch.linspace(0, 1, steps):
        xi = (baseline + a * (x3d - baseline)).clone().requires_grad_(True)
        logits = model(xi, views)
        model.zero_grad(set_to_none=True)
        logits[0, target].backward()
        total = total + xi.grad
    attr = ((x3d - baseline) * total / steps)[0, 0].detach().cpu().numpy()
    return np.abs(attr)


def crossgate_attention(model, x3d, views):
    model.eval()
    with torch.no_grad():
        model(x3d, views)
    w = model.fusion.last_w
    return w[0].detach().cpu().numpy() if w is not None else None, model.fusion.last_gate


def branch_drop_importance(model, x3d, views, target):
    model.eval()
    with torch.no_grad():
        base = F.softmax(model(x3d, views), 1)[0, target].item()
    names, drops = ["3D"], []
    with torch.no_grad():
        drops.append(max(0.0, base - F.softmax(model(x3d, torch.zeros_like(views)), 1)[0, target].item()))
    for i in range(views.shape[1]):
        v = views.clone()
        v[:, i] = 0
        with torch.no_grad():
            names.append(f"2D-{i}")
            drops.append(max(0.0, base - F.softmax(model(x3d, v), 1)[0, target].item()))
    return names, drops
