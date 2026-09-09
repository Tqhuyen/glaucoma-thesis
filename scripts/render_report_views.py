import io
import json
import os
import struct
import tempfile
import zlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

REPO_URL = ("https://huggingface.co/datasets/harvardairobotics/Harvard-GF/"
            "resolve/main/Dataset/dataset.zip")
CSV_URL = ("https://huggingface.co/datasets/harvardairobotics/Harvard-GF/"
           "resolve/main/ReadMe/data_summary.csv")
ZIP_SIZE = 20521840001
TAIL = 8 * 2 ** 20
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
N_EDGE, HALF = 30, 8


def fetch(url, start, end):
    r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=180)
    r.raise_for_status()
    return r.content


def load_cd():
    data = fetch(REPO_URL, ZIP_SIZE - TAIL, ZIP_SIZE - 1)
    base = ZIP_SIZE - TAIL
    o = data.rfind(b"PK\x06\x06")
    cd_size, cd_off = struct.unpack_from("<QQ", data, o + 40)
    cd = {}
    p = cd_off - base
    end = cd_off + cd_size - base
    while p < end:
        if data[p:p + 4] != b"PK\x01\x02":
            raise RuntimeError("bad central directory")
        flags, method = struct.unpack_from("<HH", data, p + 8)
        csize, usize = struct.unpack_from("<II", data, p + 20)
        nl, el, cl = struct.unpack_from("<HHH", data, p + 28)
        lho = struct.unpack_from("<I", data, p + 42)[0]
        name = data[p + 46:p + 46 + nl].decode("utf-8", "replace")
        extra = data[p + 46 + nl:p + 46 + nl + el]
        pos = 0
        while pos + 4 <= len(extra):
            hid, sz = struct.unpack_from("<HH", extra, pos)
            if hid == 0x0001:
                body = extra[pos + 4:pos + 4 + sz]
                q = 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack_from("<Q", body, q)[0]
                    q += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack_from("<Q", body, q)[0]
                    q += 8
                if lho == 0xFFFFFFFF:
                    lho = struct.unpack_from("<Q", body, q)[0]
            pos += 4 + sz
        cd[name] = (flags, method, csize, usize, lho)
        p += 46 + nl + el + cl
    return cd


CD = load_cd()


def extract(stem):
    key = f"Test/data_{stem.zfill(4)}.npz"
    if key not in CD:
        raise KeyError(stem)
    _, method, csize, usize, lho = CD[key]
    head = fetch(REPO_URL, lho, lho + 30 + 256)
    nl, el = struct.unpack_from("<HH", head, 26)
    start = lho + 30 + nl + el
    comp = fetch(REPO_URL, start, start + csize - 1)
    if method == 0:
        raw = comp
    else:
        raw = zlib.decompressobj(-15).decompress(comp)
    return np.load(io.BytesIO(raw))["oct_bscans"]


def depth_axis(vol):
    f = vol.astype(np.float32)
    return int(np.argmax([f.mean(axis=tuple(i for i in range(3) if i != ax)).std()
                          for ax in range(3)]))


def to_depth_last(vol, dz):
    if dz == 2:
        return vol
    others = [i for i in range(3) if i != dz]
    return np.transpose(vol, tuple(others) + (dz,))


def slab_of(volc, half=16):
    S = volc.shape[2]
    half = min(half, max(2, S // 8))
    prof = volc.mean(axis=(0, 1)).astype(np.float32)
    peak = int(prof.argmax())
    return max(0, peak - half), min(S, peak + half + 1)


def slice_metric(im):
    lo, hi = np.percentile(im, 1), np.percentile(im, 99)
    return float(np.clip(im, lo, hi).std())


def best_slice_axis(volc, axis):
    lo, hi = N_EDGE, volc.shape[axis] - N_EDGE
    idxs = range(lo, hi)
    best, score = lo, -1.0
    for i in idxs:
        im = np.take(volc, i, axis=axis)
        s = slice_metric(im)
        if s > score:
            score, best = s, i
    return best


def quality(vol):
    dz = depth_axis(vol)
    volc = to_depth_last(vol, dz)
    cols = []
    for ax in (0, 1, 2):
        i = best_slice_axis(volc, ax)
        cols.append(slice_metric(np.take(volc, i, axis=ax)))
    return min(cols)


def load_volume(stem, cache):
    p = os.path.join(cache, f"raw_{stem}.npy")
    if os.path.exists(p):
        return np.load(p)
    vol = extract(stem)
    os.makedirs(cache, exist_ok=True)
    np.save(p, vol)
    return vol


def enface(volc, kind, lo, hi):
    if kind == "aip_full":
        return volc.mean(axis=2).astype(np.float32)
    if kind == "mip_full":
        return volc.max(axis=2).astype(np.float32)
    if kind == "aip_slab":
        return volc[:, :, lo:hi].mean(axis=2).astype(np.float32)
    if kind == "mip_slab":
        return volc[:, :, lo:hi].max(axis=2).astype(np.float32)
    raise ValueError(kind)


def render_volume(stem, label, out):
    vol = load_volume(stem, CACHE)
    dz = depth_axis(vol)
    volc = to_depth_last(vol, dz)
    lo, hi = slab_of(volc, 16)
    bscan_i = best_slice_axis(volc, 1)
    xsect_i = best_slice_axis(volc, 0)
    bscan = np.take(volc, bscan_i, axis=1).T
    xsect = np.take(volc, xsect_i, axis=0).T
    enfs = {"aip_full": "En-face AIP (full depth)", "mip_full": "En-face MIP (full depth)",
            "aip_slab": "En-face AIP (RNFL band)", "mip_slab": "En-face MIP (RNFL band)"}
    panels = [(enface(volc, k, lo, hi), t) for k, t in enfs.items()] + \
             [(bscan, "Central B-scan"), (xsect, "Orthogonal cross-section")]
    fig, axs = plt.subplots(2, 3, figsize=(12.6, 8.2))
    for ax, (im, t) in zip(axs.ravel(), panels):
        p1, p99 = np.percentile(im, 1), np.percentile(im, 99)
        ax.imshow(im, cmap="gray", vmin=p1, vmax=p99)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    gl = "glaucoma positive" if label == 1 else "healthy"
    fig.suptitle(f"Harvard-GF volume data_{stem} ({gl}) | 200^3 raw | "
                 f"2D views derived by projection (depth axis = {dz})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def render_2d_inputs(rows, out):
    names = [("aip_full", "En-face AIP (full)"), ("aip_slab", "En-face AIP (RNFL band)"),
             ("mip_slab", "En-face MIP (RNFL band)")]
    fig, axs = plt.subplots(len(rows), len(names), figsize=(11, 9.6))
    for r, (stem, lab) in enumerate(rows):
        volc, lo, hi = CACHED[stem]
        for c, (kind, t) in enumerate(names):
            im = enface(volc, kind, lo, hi)
            p1, p99 = np.percentile(im, 1), np.percentile(im, 99)
            axs[r, c].imshow(im, cmap="gray", vmin=p1, vmax=p99)
            if r == 0:
                axs[r, c].set_title(t, fontsize=10)
            if c == 0:
                axs[r, c].set_ylabel(f"data_{stem} ({'glc+' if lab == 1 else 'normal'})", fontsize=10)
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])
    fig.suptitle("2D branch inputs derived from each raw 3D volume", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


CACHE = os.path.join(tempfile.gettempdir(), "gf_vol_cache")
CACHED = {}


def main(glc_plus, glc_minus):
    os.makedirs(FIG_DIR, exist_ok=True)
    rows = []
    meta = {}
    for stem, lab in glc_plus + glc_minus:
        vol = load_volume(stem, CACHE)
        dz = depth_axis(vol)
        volc = to_depth_last(vol, dz)
        lo, hi = slab_of(volc, 16)
        CACHED[stem] = (volc, lo, hi)
        rows.append((stem, lab))
        render_volume(stem, lab, os.path.join(FIG_DIR, f"report_views_{stem}_"
                                              + ("glcplus" if lab == 1 else "normal") + ".png"))
        meta[stem] = {"glaucoma": bool(lab), "depth_axis": int(dz), "slab": [int(lo), int(hi)]}
    render_2d_inputs(rows, os.path.join(FIG_DIR, "report_2d_branch_inputs.png"))
    with open(os.path.join(FIG_DIR, "report_views_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


if __name__ == "__main__":
    df = pd.read_csv(io.StringIO(requests.get(CSV_URL, timeout=60).text))
    df["id"] = df["filename"].str.extract(r"(\d+)", expand=False).astype(int)
    test = df[df["use"] == "test"]
    pos = test[test["glaucoma"] == "yes"]
    neg = test[test["glaucoma"] == "no"]
    cand_pos = ["2404", "3294"]
    step = max(1, len(neg) // 12)
    cand_neg = [str(int(i)) for i in neg.iloc[::step]["id"].tolist()[:12]]
    scored = []
    for stem in set(cand_pos + cand_neg):
        lab = 1 if stem in cand_pos else 0
        vol = load_volume(stem, CACHE)
        scored.append((quality(vol), stem, lab))
    scored.sort(key=lambda t: -t[0])
    top_pos = [t for t in scored if t[2] == 1][:2]
    top_neg = [t for t in scored if t[2] == 0][:1]
    print("scored top:", [t[:3] for t in scored[:6]])
    main([(s, l) for _, s, l in top_pos], [(s, l) for _, s, l in top_neg])
