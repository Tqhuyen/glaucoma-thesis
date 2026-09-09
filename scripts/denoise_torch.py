import importlib.util
import json
import os
import tempfile
import time
import urllib.request

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_OK = True
except Exception:
    torch = None
    nn = None
    TORCH_OK = False

WEIGHTS_DIR = os.environ.get("GF_DENOISE_WEIGHTS") or os.path.join(tempfile.gettempdir(), "gf_denoise_weights")

DNCNN_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_gray_blind.pth"
DNCNN_NAME = "dncnn_gray_blind.pth"
SWINIR_URL = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth"
SWINIR_NAME = "004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth"
SWINIR_SRC_URL = "https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/models/network_swinir.py"
SWINIR_SRC_NAME = "network_swinir.py"

BATCH = {"dncnn": 32, "swinir": 4}


def available():
    return TORCH_OK


def _download(url, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return dst
    print(f"[torch denoiser] downloading {url.split('/')[-1]} -> {dst}", flush=True)
    tmp = dst + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst)
    return dst


def _load_state(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "params" in ckpt:
        ckpt = ckpt["params"]
    return ckpt


def _ensure_timm_shim():
    try:
        import timm.models.layers  # noqa: F401

        return
    except Exception:
        pass
    import sys
    import types

    timm = types.ModuleType("timm")
    timm_models = types.ModuleType("timm.models")
    timm_layers = types.ModuleType("timm.models.layers")
    timm.__path__ = []
    timm_models.__path__ = []

    def _to_2tuple(x):
        return (x, x) if not isinstance(x, (tuple, list)) else tuple(x)

    def _trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        def _no_grad_trunc_normal_(t, m, s, lo, hi):
            with torch.no_grad():
                t.normal_(mean=m, std=s)
                lo_v = t < lo
                hi_v = t > hi
                while lo_v.any() or hi_v.any():
                    t = torch.where(lo_v, t + torch.normal(0, s, t.shape, device=t.device), t)
                    t = torch.where(hi_v, t - torch.normal(0, s, t.shape, device=t.device), t)
                    lo_v = t < lo
                    hi_v = t > hi
                return t

        if std <= 0:
            return tensor.fill_(mean)
        return _no_grad_trunc_normal_(tensor, mean, std, a, b)

    def _drop_path(x, drop_prob=0.0, training=False):
        if drop_prob <= 0.0 or not training:
            return x
        keep_prob = 1.0 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            mask = mask.div_(keep_prob)
        return x * mask

    class _DropPath(nn.Module):
        def __init__(self, drop_prob=None):
            super().__init__()
            self.drop_prob = float(drop_prob or 0.0)

        def forward(self, x):
            return _drop_path(x, self.drop_prob, self.training)

    timm_layers.to_2tuple = _to_2tuple
    timm_layers.trunc_normal_ = _trunc_normal_
    timm_layers.DropPath = _DropPath
    sys.modules["timm"] = timm
    sys.modules["timm.models"] = timm_models
    sys.modules["timm.models.layers"] = timm_layers


class _DnCNN(nn.Module):
    def __init__(self, nb):
        super().__init__()
        layers = []
        for i in range(nb - 1):
            c = nn.Conv2d(1, 64, 3, 1, 1, bias=True) if i == 0 else nn.Conv2d(64, 64, 3, 1, 1, bias=True)
            layers += [c, nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(64, 1, 3, 1, 1, bias=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.model(x)


def _build_dncnn(device):
    ckpt = _load_state(_download(DNCNN_URL, os.path.join(WEIGHTS_DIR, DNCNN_NAME)))
    nb = sum(1 for t in ckpt.values() if t.ndim == 4)
    model = _DnCNN(nb).to(device)
    model.load_state_dict(ckpt, strict=True)
    return model.eval()


def _build_swinir(device):
    _ensure_timm_shim()
    src = _download(SWINIR_SRC_URL, os.path.join(WEIGHTS_DIR, SWINIR_SRC_NAME))
    spec = importlib.util.spec_from_file_location("network_swinir", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.SwinIR(
        upscale=1,
        in_chans=1,
        img_size=128,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="",
        resi_connection="1conv",
    )
    ckpt = _load_state(_download(SWINIR_URL, os.path.join(WEIGHTS_DIR, SWINIR_NAME)))
    model.load_state_dict(ckpt, strict=True)
    return model.eval().to(device)


def _build(name, device):
    if name == "dncnn":
        return _build_dncnn(device)
    if name == "swinir":
        return _build_swinir(device)
    raise ValueError(name)


_MODEL_CACHE = {}


def get_model(name, device="auto"):
    if not TORCH_OK:
        raise RuntimeError("torch not importable")
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key = (name, str(device))
    model = _MODEL_CACHE.get(key)
    if model is None:
        print(f"[torch denoiser] building {name} on {device} ...", flush=True)
        model = _build(name, device)
        _MODEL_CACHE[key] = model
    return model


def _denoise_volume(model, device, name, vol):
    n = vol.shape[0]
    h, w = vol.shape[1:]
    win = 8 if name == "swinir" else 1
    hp = 0 if h % win == 0 else win - h % win
    wp = 0 if w % win == 0 else win - w % win
    out = np.empty_like(vol)
    bs = BATCH.get(name, 8)
    i = 0
    while i < n:
        b = min(bs, n - i)
        try:
            x = torch.from_numpy(vol[i : i + b].astype(np.float32) / 255.0)[:, None].to(device)
            if hp or wp:
                x = torch.nn.functional.pad(x, (0, wp, 0, hp), mode="reflect")
            with torch.no_grad():
                y = model(x)
            if hp or wp:
                y = y[:, :, :h, :w]
            y = y[:, 0].clamp_(0, 1).cpu().numpy()
            out[i : i + b] = np.clip(y * 255.0, 0, 255).astype(np.uint8)
            i += b
        except RuntimeError as e:
            if "out of memory" not in str(e) or bs <= 1:
                raise
            bs = max(1, bs // 2)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return out


def denoise_volume(name, vol, device="auto", cache_path=None):
    side = cache_path.replace(".npy", ".json") if cache_path else None
    meta = json.load(open(side)) if (side and os.path.exists(side)) else {}
    if cache_path and os.path.exists(cache_path):
        out = np.load(cache_path)
        if out.shape == vol.shape:
            return out, float(meta.get("time_s", 0.0))
    if not TORCH_OK:
        raise RuntimeError("torch not importable")
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(name, device)
    t0 = time.time()
    out = _denoise_volume(model, device, name, vol)
    dt = time.time() - t0
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, out)
        if side:
            with open(side, "w") as fh:
                json.dump({"time_s": round(dt, 2), "device": str(device)}, fh)
    return out, dt
