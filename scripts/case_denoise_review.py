"""Per-case denoising review for the report's example OCT cases.

Fetches only the needed test volumes from the raw 200^3 HF dataset (byte-range reads), applies the six classical
filters with the thesis survey parameters (``compare_denoise_methods.METHODS``) and scores them with the survey's
no-reference metrics (``denoise_wholevolume_survey._slice_and_global``). Also writes middle B-scan crops per method.

    python scripts/case_denoise_review.py --cases artifacts/report_src/glaucoma/post/report-assets/glaucoma/cases.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_denoise_methods import METHODS, denoise_volume, depth_axis  # noqa: E402
from denoise_wholevolume_survey import CLASSICAL, _slice_and_global  # noqa: E402

REPO = "tqhuyen/harvard-oct-glaucoma-200"


def fetch_volumes(indices, token):
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=token)
    out = {}
    with fs.open(f"datasets/{REPO}/Test_volumes.npy", "rb", block_size=8 * 1024 * 1024) as fh:
        version = np.lib.format.read_magic(fh)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, _, dtype = reader(fh)
        start = fh.tell()
        per = int(np.prod(shape[1:])) * np.dtype(dtype).itemsize
        for i in indices:
            fh.seek(start + i * per)
            out[i] = np.frombuffer(fh.read(per), dtype=dtype).reshape(shape[-3:]).copy()
    return out


def to_png(im, path, low, high):
    a = np.clip((im.astype(np.float32) - low) / max(high - low, 1e-6), 0, 1)
    Image.fromarray((a * 255).astype(np.uint8)).save(path)


def score_case(index, vol, methods, workers, out):
    """Denoise one volume with ``methods`` and return its survey metrics; writes middle B-scan crops."""
    from skimage.filters import threshold_otsu

    dz = depth_axis(vol)
    thr = float(threshold_otsu(vol))
    mid = vol.shape[0] // 2
    low, high = np.percentile(vol[mid], [1, 99])
    to_png(vol[mid], out / f"case{index}_original.png", low, high)
    entry = {"index": index, "depth_axis": dz, "threshold": thr, "methods": {}}
    entry["methods"]["original"] = _slice_and_global(vol, vol, dz, thr)["global"]
    for name in methods:
        den, seconds = denoise_volume(vol, name, workers=workers)
        scores = {**_slice_and_global(vol, den, dz, thr)["global"], "seconds": round(seconds, 1)}
        if name == "bm3d":
            scores["sigma_psd"] = METHODS["bm3d"]["params"]["sigma_psd"]
        entry["methods"][name] = scores
        to_png(den[mid], out / f"case{index}_{name}.png", low, high)
    print(f"[case {index}] cnr_bg { ({m: round(v['cnr_bg'], 3) for m, v in entry['methods'].items()}) }", flush=True)
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/case_denoise"))
    parser.add_argument("--methods", default=",".join(CLASSICAL), help="comma list")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8, help="threads per volume (non-bm3d)")
    parser.add_argument("--procs", type=int, default=2, help="volumes in parallel when bm3d is requested")
    args = parser.parse_args()

    from concurrent.futures import ProcessPoolExecutor, as_completed

    from pipeline.utils import load_env_file, secret_env

    load_env_file()
    methods = [m for m in args.methods.split(",") if m]
    cases = json.loads(args.cases.read_text())
    order = [c["index"] for c in cases]
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "metrics.json"

    def merge(entry):
        merged = json.loads(path.read_text()) if path.exists() else {"params": {}, "cases": []}
        by_index = {c["index"]: c for c in merged["cases"]}
        slot = by_index.setdefault(entry["index"], {"index": entry["index"], "methods": {}})
        slot.update({k: v for k, v in entry.items() if k != "methods"})
        slot["methods"].update(entry["methods"])
        merged["params"].update({m: METHODS[m]["params"] for m in methods if m != "bm3d"})
        if "bm3d" in methods:
            merged["params"]["bm3d"] = {"profile": "np", "sigma_psd": "estimate_sigma per volume (see per-case value)"}
        merged["cases"] = [by_index[i] for i in order if i in by_index]
        path.write_text(json.dumps(merged, indent=1))

    volumes = fetch_volumes(order, secret_env("HF_TOKEN").reveal())
    if "bm3d" in methods:
        with ProcessPoolExecutor(max_workers=max(1, args.procs)) as pool:
            futures = [pool.submit(score_case, i, volumes[i], methods, 1, args.out) for i in order]
            for future in as_completed(futures):
                merge(future.result())
    else:
        for i in order:
            merge(score_case(i, volumes[i], methods, args.workers, args.out))


if __name__ == "__main__":
    main()
