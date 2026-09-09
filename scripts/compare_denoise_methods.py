import argparse
import csv
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian, median
from skimage.restoration import (
    denoise_bilateral,
    denoise_nl_means,
    denoise_tv_chambolle,
    denoise_wavelet,
    estimate_sigma,
)

try:
    import bm3d as _bm3d

    HAVE_BM3D = True
except Exception:
    _bm3d = None
    HAVE_BM3D = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures", "denoise")
CACHE_DIR = os.path.join(tempfile.gettempdir(), "gf_vol_cache")
DENOISE_CACHE = os.path.join(FIG_DIR, "cache")
AXIS_IDX = {"x": 0, "y": 1, "z": 2}

DEFAULT_PLANES = [("x", 0, 132), ("x", 0, 148),
                  ("y", 1, 22), ("y", 1, 38),
                  ("z", 2, 114), ("z", 2, 130)]

METHODS = {
    "gaussian": {"label": "Gaussian", "params": {"sigma": 1.5}},
    "median": {"label": "Median", "params": {"footprint": 3}},
    "bilateral": {"label": "Bilateral", "params": {"sigma_color": 0.10, "sigma_spatial": 4.0}},
    "tv": {"label": "Total Variation (Chambolle)", "params": {"weight": 0.10}},
    "wavelet": {"label": "Wavelet (BayesShrink)", "params": {"wavelet": "db4", "mode": "soft", "method": "BayesShrink"}},
    "nlm": {"label": "Non-Local Means", "params": {"h": 0.10, "patch_size": 5, "patch_distance": 3}},
    "bm3d": {"label": "BM3D", "params": {"sigma_psd": "auto"}},
}
ORDER = ["gaussian", "median", "bilateral", "tv", "wavelet", "nlm", "bm3d"]


def load_volume(stem):
    p = os.path.join(CACHE_DIR, f"raw_{stem}.npy")
    if os.path.exists(p):
        return np.load(p)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from render_report_views import load_volume as _lv
    return _lv(stem, CACHE_DIR)


def old_meta_planes(stem):
    p = os.path.join(FIG_DIR, f"denoise_compare_{stem}_meta.json")
    if not os.path.exists(p):
        return None
    ax = json.load(open(p)).get("axes")
    if not ax:
        return None
    planes = []
    for n in ("x", "y", "z"):
        sl = ax.get(n, {}).get("slices") or []
        for i in sl:
            planes.append((n, AXIS_IDX[n], int(i)))
    return planes or None


def depth_axis(vol):
    f = vol.astype(np.float32)
    return int(np.argmax([f.mean(axis=tuple(i for i in range(3) if i != ax)).std()
                          for ax in range(3)]))


def crop_lateral(vol, dz, pad=40):
    idx = [slice(None)] * 3
    for a in (i for i in range(3) if i != dz):
        idx[a] = slice(pad, vol.shape[a] - pad)
    return vol[tuple(idx)]


def band_indexes(volc, dz, half=14, above=24):
    prof = volc.mean(axis=tuple(i for i in range(3) if i != dz)).astype(np.float32)
    peak = int(prof.argmax())
    lo = max(0, peak - half)
    hi = min(volc.shape[dz], peak + half)
    bl = max(0, lo - above)
    return lo, hi, bl, lo


def stack_band(volc, dz, i0, i1):
    return np.take(volc, range(i0, i1), axis=dz)


def depth_edges(volc, dz):
    lo, hi, _, _ = band_indexes(volc, dz)
    f = volc.astype(np.float32)
    gz = np.gradient(f, axis=dz)
    glat = np.gradient(f, axis=1 if dz != 1 else 2)
    gm = np.sqrt(gz ** 2 + glat ** 2)
    slc = [slice(None)] * 3
    slc[dz] = slice(lo, hi)
    return gm[tuple(slc)].ravel()


def volume_metrics(orig, img, dz, thresh):
    o = crop_lateral(orig, dz)
    v = crop_lateral(img, dz)
    lo, hi, bl, _ = band_indexes(v, dz)
    sig_v = stack_band(v, dz, lo, hi).ravel()
    bg_v = stack_band(v, dz, bl, lo).ravel()
    sig = sig_v[sig_v > thresh]
    bg = bg_v[bg_v <= thresh]
    mu_s, sd_s = (float(sig.mean()), float(sig.std())) if sig.size else (0.0, 0.0)
    mu_b, sd_b = (float(bg.mean()), float(bg.std())) if bg.size else (0.0, 0.0)
    snr = mu_s / sd_s if sd_s > 0 else 0.0
    enl = snr * snr
    cnr = (mu_s - mu_b) / np.sqrt(sd_s ** 2 + sd_b ** 2) if (sd_s + sd_b) > 0 else 0.0

    ge_o = depth_edges(o, dz)
    ge_v = depth_edges(v, dz)
    mask = ge_o > np.percentile(ge_o, 85)
    if mask.sum() < 3:
        beta = 0.0
    else:
        eo, ed = ge_o[mask], ge_v[mask]
        beta = float(np.corrcoef(eo, ed)[0, 1]) if eo.std() > 0 and ed.std() > 0 else 0.0
    bg_std = float(stack_band(v, dz, bl, lo).std()) if bl < lo else 0.0
    return {"snr": snr, "enl": enl, "cnr": cnr, "beta": beta, "bg_noise_std": bg_std}


def noise_sigma_estimate(vol, nsamp=40, workers=8):
    n = min(nsamp, vol.shape[0])
    idx = np.random.default_rng(0).choice(vol.shape[0], size=n, replace=False)

    def _est(i):
        im = vol[i].astype(np.float32) / 255.0
        if im.max() - im.min() < 1e-4:
            return float("nan")
        return float(estimate_sigma(im, average_sigmas=True, channel_axis=None))

    vals = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for v in ex.map(_est, idx):
            vals.append(v)
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.median(vals)) if vals else 0.05


def make_slice_fn(name, params):
    if name == "gaussian":
        return lambda im: gaussian(im, sigma=params["sigma"], channel_axis=None)
    if name == "median":
        return lambda im: median(im, footprint=np.ones((params["footprint"],) * 2))
    if name == "bilateral":
        return lambda im: denoise_bilateral(im, sigma_color=params["sigma_color"],
                                            sigma_spatial=params["sigma_spatial"], channel_axis=None)
    if name == "tv":
        return lambda im: denoise_tv_chambolle(im, weight=params["weight"], channel_axis=None)
    if name == "wavelet":
        return lambda im: denoise_wavelet(im, wavelet=params["wavelet"], mode=params["mode"],
                                          method=params["method"], rescale_sigma=True, channel_axis=None)
    if name == "nlm":
        return lambda im: denoise_nl_means(im, h=params["h"], patch_size=params["patch_size"],
                                           patch_distance=params["patch_distance"], fast_mode=True)
    if name == "bm3d":
        return lambda im: _bm3d.bm3d(im, sigma_psd=params["sigma_psd"])
    raise ValueError(name)


def denoise_volume(vol, name, workers=8, cache_path=None):
    side = cache_path.replace(".npy", ".json") if cache_path else None
    meta = json.load(open(side)) if (side and os.path.exists(side)) else {}
    if cache_path and os.path.exists(cache_path):
        out = np.load(cache_path)
        if out.shape == vol.shape:
            if "sigma_psd" in meta:
                METHODS[name]["params"]["sigma_psd"] = meta["sigma_psd"]
            return out, float(meta.get("time_s", 0.0))
    if name == "bm3d":
        METHODS[name]["params"]["sigma_psd"] = noise_sigma_estimate(vol, workers=workers)
    fn = make_slice_fn(name, METHODS[name]["params"])
    out = np.empty_like(vol)

    def _one(d):
        im = vol[d].astype(np.float32) / 255.0
        res = fn(im)
        return np.clip(np.nan_to_num(res, nan=im) * 255.0, 0, 255).astype(np.uint8)

    t0 = time.time()
    if name == "bm3d":
        for d in range(vol.shape[0]):
            out[d] = _one(d)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for d, o in zip(range(vol.shape[0]), ex.map(_one, range(vol.shape[0]))):
                out[d] = o
    dt = time.time() - t0
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, out)
        if side:
            with open(side, "w") as fh:
                json.dump({"time_s": round(dt, 2),
                           "sigma_psd": METHODS[name]["params"].get("sigma_psd")}, fh)
    return out, dt


def oriented_plane(vol, dz, axis, idx):
    im = np.take(vol, idx, axis=axis)
    axes = [i for i in range(3) if i != axis]
    b, c = axes
    if dz == c:
        im = im.T
    return im


def row_label(name):
    p = METHODS[name]["params"]
    if name == "gaussian":
        return f"Gaussian (sigma={p['sigma']})"
    if name == "median":
        return f"Median ({p['footprint']}x{p['footprint']})"
    if name == "bilateral":
        return f"Bilateral (sc={p['sigma_color']}, ss={p['sigma_spatial']})"
    if name == "tv":
        return f"TV Chambolle (lambda={p['weight']})"
    if name == "wavelet":
        return f"Wavelet BayesShrink ({p['wavelet']})"
    if name == "nlm":
        return f"NLM (h={p['h']}, patch={p['patch_size']}, d={p['patch_distance']})"
    if name == "bm3d":
        return f"BM3D (sigma_psd={p['sigma_psd']:.3f})" if isinstance(p["sigma_psd"], float) else "BM3D"
    return name


def draw_grid(rows, planes, out, dpi=170, suptitle=None):
    ncol = len(planes)
    nrow = len(rows)
    base = rows[0][1]
    dz = depth_axis(base)
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 2.55, nrow * 2.2))
    axs = np.atleast_2d(axs)
    lims = []
    for (name, axis, idx) in planes:
        im = oriented_plane(base, dz, axis, idx)
        lims.append((np.percentile(im, 1), np.percentile(im, 99)))
    for r, (lab, vol) in enumerate(rows):
        for c, (name, axis, idx) in enumerate(planes):
            im = oriented_plane(vol, dz, axis, idx)
            p1, p99 = lims[c]
            axs[r, c].imshow(im, cmap="gray", vmin=p1, vmax=p99)
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])
            if r == 0:
                axs[r, c].set_title(f"{name} = {idx}", fontsize=9)
        axs[r, 0].set_ylabel(lab, fontsize=8.5, rotation=0, labelpad=58, va="center")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0.06, 0, 1, 0.97 if suptitle else 1))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def draw_metric_table(rows, out, title=None):
    headers = ["Method", "SNR", "ENL", "CNR", "beta", "time (s)"]
    data = [[r[0]] + [f"{v:.3f}" if isinstance(v, float) else str(v) for v in r[1:]]
            for r in rows]
    fig, ax = plt.subplots(figsize=(9.5, 0.45 * len(rows) + 1.2))
    ax.axis("off")
    tbl = ax.table(cellText=data, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", default="2404")
    ap.add_argument("--methods", default=",".join(ORDER))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--planes-json", default=None)
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    names = [m for m in args.methods.split(",") if m in METHODS]
    if not HAVE_BM3D and "bm3d" in names:
        names.remove("bm3d")
        print("[warn] bm3d not importable -> dropped")

    vol = load_volume(args.volume)
    dz = depth_axis(vol)
    volc = crop_lateral(vol, dz)
    lo, hi, bl, _ = band_indexes(volc, dz)
    band = stack_band(volc, dz, lo, hi).astype(np.float32).ravel()
    thresh = float(np.percentile(band, 60))
    print(f"[vol {args.volume}] dz={dz} rnfl_band=[{lo},{hi}) bg=[{bl},{lo}) "
          f"signal_thresh={thresh:.1f}")

    if args.planes_json:
        pm = json.load(open(args.planes_json))
        planes = [(n, AXIS_IDX[n], int(i)) for n, s in pm.items() for i in s]
    else:
        planes = old_meta_planes(args.volume) or DEFAULT_PLANES

    volumes = {"original": vol}
    elapsed = {"original": 0.0}
    for name in names:
        print(f"[denoise] {name} ...", flush=True)
        cache_path = os.path.join(DENOISE_CACHE, f"{args.volume}_{name}.npy")
        den, sec = denoise_volume(vol, name, workers=args.workers, cache_path=cache_path)
        volumes[name] = den
        elapsed[name] = sec
        print(f"[denoise] {name} done in {sec:.1f}s")

    met = {}
    for key, v in volumes.items():
        m = volume_metrics(vol, v, dz, thresh)
        met[key] = m
        print(f"[metrics] {key:9s} SNR={m['snr']:.3f} ENL={m['enl']:.3f} "
              f"CNR={m['cnr']:.3f} beta={m['beta']:.3f}")

    rows_img = [("Original (raw 200^3)", vol)] + [(row_label(n), volumes[n]) for n in names]
    suptitle = (f"Harvard-GF volume data_{args.volume} (200^3) | speckle denoising comparison | "
                f"identical planes & contrast for every row")
    draw_grid(rows_img, planes, os.path.join(FIG_DIR, f"denoise_compare_{args.volume}_all.png"),
              suptitle=suptitle)
    for name in names:
        draw_grid([("Original (raw 200^3)", vol), (row_label(name), volumes[name])],
                  planes,
                  os.path.join(FIG_DIR, f"denoise_compare_{args.volume}_{name}.png"),
                  suptitle=f"data_{args.volume} | {row_label(name)}")

    table_rows = [("Original", met["original"]["snr"], met["original"]["enl"],
                   met["original"]["cnr"], met["original"]["beta"], 0.0)]
    for name in names:
        m = met[name]
        table_rows.append((name, m["snr"], m["enl"], m["cnr"], m["beta"], elapsed[name]))

    md = ["| Method | SNR | ENL | CNR | beta | time (s) |", "|---|---|---|---|---|---|"]
    for r in table_rows:
        md.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    print("\n".join(md))

    csv_path = os.path.join(FIG_DIR, f"denoise_metrics_{args.volume}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "snr", "enl", "cnr", "beta", "time_s"])
        for r in table_rows:
            w.writerow(r)
    print("wrote", csv_path)

    draw_metric_table(table_rows,
                      os.path.join(FIG_DIR, f"denoise_metrics_{args.volume}.png"),
                      title=f"Speckle denoising metrics - Harvard-GF volume data_{args.volume}")

    meta = {
        "volume": args.volume,
        "glaucoma": True,
        "resolution": 200,
        "depth_axis": int(dz),
        "metrics_regions": {"rnfl_band": [lo, hi], "background": [bl, lo],
                            "signal_threshold_p60_of_band": thresh,
                            "crop_lateral_pad": 40},
        "planes": {},
        "methods": {n: {"label": METHODS[n]["label"],
                        "params": {k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in METHODS[n]["params"].items()},
                        "time_s": round(elapsed[n], 2)}
                    for n in names},
        "metrics": met,
        "figures": {"all": f"denoise_compare_{args.volume}_all.png",
                    "per_method": [f"denoise_compare_{args.volume}_{n}.png" for n in names],
                    "metrics_table": f"denoise_metrics_{args.volume}.png",
                    "csv": f"denoise_metrics_{args.volume}.csv"},
    }
    for (name, axis, idx) in planes:
        d = meta["planes"].setdefault(name, {"axis": axis, "slices": []})
        d["slices"].append(int(idx))
    for d in meta["planes"].values():
        d["slices"] = sorted(set(d["slices"]))
    meta_path = os.path.join(FIG_DIR, f"denoise_compare_{args.volume}_all_meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print("wrote", meta_path)


if __name__ == "__main__":
    main()
