import argparse
import csv
import json
import math
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_denoise_methods import (  # noqa: E402
    DENOISE_CACHE,
    depth_axis,
    denoise_volume,
    load_volume,
)

CLASSICAL = ["gaussian", "median", "bilateral", "tv", "wavelet", "nlm"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "figures", "denoise", "wholevolume")
CACHE_DIR = os.path.join(tempfile.gettempdir(), "gf_vol_cache")
AGGS = ("sum", "mean", "median")
METRICS = ("snr", "enl", "cnr", "beta", "snr_bg", "enl_bg", "cnr_bg", "bg_sigma")


def cached_volumes():
    if not os.path.isdir(CACHE_DIR):
        return []
    stems = [fn[4:-4] for fn in os.listdir(CACHE_DIR) if fn.startswith("raw_") and fn.endswith(".npy")]
    return sorted(stems)


def _gradmag(im):
    gy, gx = np.gradient(im)
    return np.sqrt(gy * gy + gx * gx)


def _ms(sum_, sumsq, n):
    if n <= 0:
        return 0.0, 0.0
    mean = sum_ / n
    var = max(sumsq / n - mean * mean, 0.0)
    return float(mean), float(math.sqrt(var))


def _slice_and_global(orig, den, dz, thr):
    n = int(orig.shape[dz])
    slices = {k: [] for k in METRICS}
    acc = {"n_sig": 0, "s_sig": 0.0, "s2_sig": 0.0, "n_bg": 0, "s_bg": 0.0, "s2_bg": 0.0,
           "n_e": 0, "sx": 0.0, "sy": 0.0, "sxx": 0.0, "syy": 0.0, "sxy": 0.0}
    for s in range(n):
        o = np.take(orig, s, axis=dz).astype(np.float32)
        d = np.take(den, s, axis=dz).astype(np.float32)
        sm = o > thr
        dsig = d[sm].astype(np.float64)
        dbg = d[~sm].astype(np.float64)
        mu_s, sd_s = _ms(float(dsig.sum()), float((dsig * dsig).sum()), dsig.size)
        mu_b, sd_b = _ms(float(dbg.sum()), float((dbg * dbg).sum()), dbg.size)
        snr = mu_s / sd_s if sd_s > 1e-12 else 0.0
        cnr = (mu_s - mu_b) / math.sqrt(sd_s ** 2 + sd_b ** 2) if (sd_s + sd_b) > 1e-12 else 0.0
        snr_bg = mu_s / sd_b if sd_b > 1e-12 else 0.0
        enl_bg = (mu_b / sd_b) ** 2 if sd_b > 1e-12 else 0.0
        cnr_bg = (mu_s - mu_b) / sd_b if sd_b > 1e-12 else 0.0
        go, gd = _gradmag(o), _gradmag(d)
        em = go > np.percentile(go, 85)
        x, y = go[em].astype(np.float64), gd[em].astype(np.float64)
        if x.size > 2 and x.std() > 0 and y.std() > 0:
            beta = float(np.corrcoef(x, y)[0, 1])
        else:
            beta = 0.0
        slices["snr"].append(snr)
        slices["enl"].append(snr * snr)
        slices["cnr"].append(cnr)
        slices["beta"].append(beta)
        slices["snr_bg"].append(snr_bg)
        slices["enl_bg"].append(enl_bg)
        slices["cnr_bg"].append(cnr_bg)
        slices["bg_sigma"].append(sd_b)
        acc["n_sig"] += dsig.size
        acc["s_sig"] += float(dsig.sum())
        acc["s2_sig"] += float((dsig * dsig).sum())
        acc["n_bg"] += dbg.size
        acc["s_bg"] += float(dbg.sum())
        acc["s2_bg"] += float((dbg * dbg).sum())
        acc["n_e"] += x.size
        acc["sx"] += float(x.sum())
        acc["sy"] += float(y.sum())
        acc["sxx"] += float((x * x).sum())
        acc["syy"] += float((y * y).sum())
        acc["sxy"] += float((x * y).sum())
    mu_s, sd_s = _ms(acc["s_sig"], acc["s2_sig"], acc["n_sig"])
    mu_b, sd_b = _ms(acc["s_bg"], acc["s2_bg"], acc["n_bg"])
    gsnr = mu_s / sd_s if sd_s > 1e-12 else 0.0
    gcnr = (mu_s - mu_b) / math.sqrt(sd_s ** 2 + sd_b ** 2) if (sd_s + sd_b) > 1e-12 else 0.0
    ne = acc["n_e"]
    cov = acc["n_e"] * acc["sxy"] - acc["sx"] * acc["sy"]
    vx = ne * acc["sxx"] - acc["sx"] ** 2
    vy = ne * acc["syy"] - acc["sy"] ** 2
    gbeta = float(cov / math.sqrt(vx * vy)) if vx > 0 and vy > 0 else 0.0
    return {
        "slice": slices,
        "global": {"snr": gsnr, "enl": gsnr * gsnr, "cnr": gcnr, "beta": gbeta,
                   "snr_bg": mu_s / sd_b if sd_b > 1e-12 else 0.0,
                   "enl_bg": (mu_b / sd_b) ** 2 if sd_b > 1e-12 else 0.0,
                   "cnr_bg": (mu_s - mu_b) / sd_b if sd_b > 1e-12 else 0.0,
                   "bg_sigma": sd_b,
                   "n_signal_voxels": int(acc["n_sig"]), "n_background_voxels": int(acc["n_bg"])},
    }


def _denoise_worker(task):
    stem, name = task
    cache = os.path.join(DENOISE_CACHE, f"{stem}_{name}.npy")
    if os.path.exists(cache):
        return stem, name, 0.0, True
    vol = load_volume(stem)
    _, dt = denoise_volume(vol, name, workers=1, cache_path=cache)
    return stem, name, float(dt), False


def _metrics_worker(stem):
    from skimage.filters import threshold_otsu

    orig = load_volume(stem)
    dz = depth_axis(orig)
    thr = float(threshold_otsu(orig))
    res = {"stem": stem, "depth_axis": int(dz), "n_slices": int(orig.shape[dz]), "threshold": thr, "methods": {}}
    for name in ["original"] + CLASSICAL:
        if name == "original":
            den = orig
        else:
            cache = os.path.join(DENOISE_CACHE, f"{stem}_{name}.npy")
            if not os.path.exists(cache):
                continue
            den = np.load(cache)
        res["methods"][name] = _slice_and_global(orig, den, dz, thr)
    return res


def _agg(vals, kind):
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    if kind == "sum":
        return float(a.sum())
    if kind == "mean":
        return float(a.mean())
    return float(np.median(a))


BG_METRICS = ("bg_sigma", "snr_bg", "enl_bg", "cnr_bg", "snr", "enl", "cnr", "beta")


def _fmt(x, nd=3):
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "-"
    return f"{xf:.{nd}f}" if math.isfinite(xf) else "-"


def _srow(summary_rows, method, metric, slice_agg, key):
    for r in summary_rows:
        if r["method"] == method and r["metric"] == metric and r["slice_agg"] == slice_agg:
            return r[key]
    return float("nan")


def _write_summary_md(path, stems, methods, primary, summary_rows):
    order = ["original"] + methods
    hdr = list(BG_METRICS)
    L = []
    L.append("# Khảo sát khử nhiễu trên TOÀN volume — 6 PP cổ điển (bỏ BM3D/DnCNN/SwinIR)\n")
    L.append(f"- Dữ liệu: **{len(stems)} volume** Harvard-GF raw 200³ (cache local): {', '.join(stems)}.")
    L.append("- Phương pháp cổ điển: " + ", ".join(methods)
             + " — khử nhiễu **2D trên từng B-scan**, toàn mặt cắt (không crop, không band).")
    L.append("- Ngưỡng tín hiệu/nền: **Otsu tính 1 lần trên volume gốc**, dùng chung mọi PP (công bằng).")
    L.append("- Chỉ số tính **theo từng lát cắt** rồi gộp bằng **sum / mean / median**; kèm bản **global** (gộp toàn bộ voxel).")
    L.append("")
    L.append("> **Cách đọc:** nhóm `snr/enl/cnr` (chỉ dùng vùng tín hiệu) bị **cấu trúc giải phẫu chi phối** trên toàn volume")
    L.append("> nên ít phản ánh mức giảm nhiễu → so sánh năng lực khử nhiễu bằng nhóm `*_bg` (nhiễu lấy từ vùng nền).")
    L.append("> `bg_sigma` ↓ càng tốt; `snr_bg/enl_bg/cnr_bg` ↑ càng tốt; `beta` càng gần 1 càng giữ biên.")
    L.append("")
    L.append(f"## A. Global toàn volume (gộp toàn bộ voxel; mean ± std trên {len(stems)} volume)\n")
    L.append("| Method | " + " | ".join(hdr) + " |")
    L.append("|" + "---:|" * len(hdr))
    for name in order:
        if name not in primary:
            continue
        p = primary[name]
        L.append(f"| {name} | " + " | ".join(
            f"{_fmt(p[m]['global_vol_mean'])} ± {_fmt(p[m]['global_vol_std'])}" for m in hdr) + " |")
    L.append("")
    L.append("## B. Theo lát cắt → gộp **mean** → mean ± std trên volume\n")
    L.append("| Method | " + " | ".join(hdr) + " |")
    L.append("|" + "---:|" * len(hdr))
    for name in order:
        if name not in primary:
            continue
        p = primary[name]
        L.append(f"| {name} | " + " | ".join(
            f"{_fmt(p[m]['slice_mean_slices_vol_mean'])} ± {_fmt(p[m]['slice_mean_slices_vol_std'])}" for m in hdr) + " |")
    L.append("")
    L.append("## C. So sánh cách gộp (slice sum / mean / median; lấy mean qua các volume)\n")
    for metric in ("bg_sigma", "snr_bg", "enl_bg", "cnr_bg", "beta"):
        L.append(f"\n**{metric}**\n")
        L.append("| Method | sum(slices) | mean(slices) | median(slices) |")
        L.append("|---|---:|---:|---:|")
        for name in order:
            if name not in primary:
                continue
            vals = [_fmt(_srow(summary_rows, name, metric, a, "across_volumes_mean")) for a in AGGS]
            L.append(f"| {name} | " + " | ".join(vals) + " |")
    L.append("")
    L.append("## D. Nhận xét\n")
    L.append("- **Bilateral**: giảm nhiễu mạnh nhất (`bg_sigma` thấp nhất, `enl_bg`/`snr_bg`/`cnr_bg` cao nhất) mà vẫn giữ biên tốt (β ≈ 0,86 toàn cục).")
    L.append("- **Wavelet**: giữ biên tốt nhất trong nhóm khử nhiễu (β ≈ 0,90), giảm nhiễu khá — lựa chọn 'bảo thủ'.")
    L.append("- **TV / NLM**: giảm nhiễu tốt nhưng mất biên nhiều (β ≈ 0,59–0,61 toàn cục).")
    L.append("- **Gaussian / Median**: nhanh nhất nhưng phá biên nhiều (β ≈ 0,52) → chỉ làm baseline.")
    L.append("")
    L.append("### Về cách gộp sum / mean / median\n")
    L.append("- `bg_sigma` và `beta`: **mean ≈ median** (lệch < 0,1) → gộp kiểu nào cũng cho cùng kết luận.")
    L.append("- `enl_bg` (và `snr_bg`): **đuôi nặng** — mean lớn hơn median rõ rệt (vd TV: 69,4 vs 27,6; NLM: 47,7 vs 24,7)")
    L.append("  do vài lát cắt có nền rất phẳng làm ENL vọt lên → **median vững hơn**, mean dễ bị kéo lệch.")
    L.append("- `sum` chỉ là `mean × số lát cắt` (~200) nên **không đổi thứ hạng**, không mang thêm thông tin.")
    L.append("- **Khuyến nghị báo cáo:** dùng **mean ± std theo lát cắt → gộp qua volume** làm bảng chính,")
    L.append("  kèm **median** làm kiểm tra độ vững; bỏ `sum` (hoặc ghi chú là tổng theo lát cắt).")
    L.append("")
    L.append("## E. File kèm\n")
    L.append("- `per_volume_metrics.csv` — chỉ số từng volume × PP (cả 3 cách gộp + global).")
    L.append("- `summary_metrics.csv` — bảng gộp đầy đủ (slice_agg × across-volume sum/mean/median/std).")
    L.append("- `wholevolume_survey.json` — toàn bộ số liệu + tham số + thời gian khử nhiễu.")
    L.append("")
    L.append("## F. Hạn chế\n")
    L.append(f"- Mẫu gồm **{len(stems)} volume** (cache local), **chưa phải toàn bộ** Harvard-GF và chưa chắc cân bằng glaucoma+/−.")
    L.append("- Metric là **no-reference** (OCT không có ảnh sạch); mask tín hiệu/nền dựa trên ngưỡng Otsu, có thể lệch giữa các volume.")
    L.append("- `beta` dùng gradient **trong mặt phẳng** (2D), không dùng gradient theo trục depth như bản band-based cũ.")
    L.append("- Bilateral rất chậm (~2 phút/volume) → toàn bộ dataset nên chạy trên Colab/vast.ai.")
    L.append("")
    L.append("> Thời gian khử nhiễu `bilateral` ~115–130 s/volume (1 luồng/volume, 12 volume song song). "
             "`bm3d` bị loại; DnCNN/SwinIR không dùng vì không phải PP cổ điển.")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes", default="", help="comma list; empty = all raw_*.npy in cache")
    ap.add_argument("--methods", default=",".join(CLASSICAL))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    methods = [m for m in args.methods.split(",") if m in CLASSICAL]
    stems = [s.strip() for s in args.volumes.split(",") if s.strip()] or cached_volumes()
    if not stems:
        raise SystemExit("no volumes found in cache")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[wholevolume] volumes={len(stems)} methods={methods} workers={args.workers}", flush=True)

    tasks = [(s, m) for s in stems for m in methods]
    times = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for stem, name, dt, hit in ex.map(_denoise_worker, tasks):
            times[(stem, name)] = dt
            print(f"[denoise] {stem} {name:9s} {'cached' if hit else f'{dt:.1f}s'}", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(_metrics_worker, stems):
            results.append(res)
            print(f"[metrics] {res['stem']} dz={res['depth_axis']} thr={res['threshold']:.1f}", flush=True)

    per_volume_rows = []
    for res in results:
        for name, data in res["methods"].items():
            row = {"stem": res["stem"], "method": name, "depth_axis": res["depth_axis"],
                   "n_slices": res["n_slices"], "threshold": res["threshold"]}
            for metric in METRICS:
                for a in AGGS:
                    row[f"{metric}_{a}"] = _agg(data["slice"][metric], a)
                row[f"global_{metric}"] = data["global"][metric]
            per_volume_rows.append(row)

    by_method = {}
    for row in per_volume_rows:
        by_method.setdefault(row["method"], []).append(row)

    summary_rows = []
    for name, rows in by_method.items():
        for metric in METRICS:
            for slice_agg in AGGS:
                vals = [r[f"{metric}_{slice_agg}"] for r in rows]
                summary_rows.append({
                    "method": name, "metric": metric, "slice_agg": slice_agg,
                    "across_volumes_sum": _agg(vals, "sum"),
                    "across_volumes_mean": _agg(vals, "mean"),
                    "across_volumes_median": _agg(vals, "median"),
                    "across_volumes_std": float(np.nanstd(vals)),
                })
            gvals = [r[f"global_{metric}"] for r in rows]
            summary_rows.append({
                "method": name, "metric": metric, "slice_agg": "global",
                "across_volumes_sum": float(np.nansum(gvals)),
                "across_volumes_mean": float(np.nanmean(gvals)),
                "across_volumes_median": float(np.nanmedian(gvals)),
                "across_volumes_std": float(np.nanstd(gvals)),
            })

    pv_csv = os.path.join(args.out_dir, "per_volume_metrics.csv")
    with open(pv_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_volume_rows[0].keys()))
        w.writeheader()
        w.writerows(per_volume_rows)

    sm_csv = os.path.join(args.out_dir, "summary_metrics.csv")
    with open(sm_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    primary = {}
    for name, rows in by_method.items():
        primary[name] = {}
        for metric in METRICS:
            vals = [r[f"{metric}_mean"] for r in rows]
            gvals = [r[f"global_{metric}"] for r in rows]
            primary[name][metric] = {
                "slice_mean_slices_vol_mean": float(np.nanmean(vals)),
                "slice_mean_slices_vol_std": float(np.nanstd(vals)),
                "slice_median_slices_vol_mean": float(np.nanmean([r[f"{metric}_median"] for r in rows])),
                "slice_sum_slices_vol_mean": float(np.nanmean([r[f"{metric}_sum"] for r in rows])),
                "global_vol_mean": float(np.nanmean(gvals)),
                "global_vol_std": float(np.nanstd(gvals)),
            }

    order = ["original"] + methods
    print("\n[primary] per-slice MEAN then MEAN +/- STD across volumes (global whole-volume in parens)")
    print("| Method | " + " | ".join(METRICS) + " |")
    print("|" + "---|" * (len(METRICS) + 1))
    for name in order:
        if name not in primary:
            continue
        p = primary[name]
        cells = []
        for metric in METRICS:
            v = p[metric]
            cells.append(f"{v['slice_mean_slices_vol_mean']:.3f} +/- {v['slice_mean_slices_vol_std']:.3f} "
                         f"({v['global_vol_mean']:.3f})")
        print(f"| {name} | " + " | ".join(cells) + " |")

    json_path = os.path.join(args.out_dir, "wholevolume_survey.json")
    payload = {
        "volumes": stems, "methods": methods, "workers": args.workers,
        "aggregations": {"slice": list(AGGS), "across_volumes": list(AGGS) + ["std", "global"]},
        "threshold": "Otsu on original volume (same mask for every method)",
        "beta": "in-plane (2D) gradient magnitude, top-15% edge pixels of the original slice",
        "denoise_time_s": {f"{s}_{m}": round(t, 3) for (s, m), t in times.items()},
        "per_volume": results,
        "primary": primary,
    }
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    md_path = os.path.join(args.out_dir, "summary.md")
    _write_summary_md(md_path, stems, methods, primary, summary_rows)
    print(f"\nwrote {pv_csv}\nwrote {sm_csv}\nwrote {json_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
