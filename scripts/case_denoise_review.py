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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/case_denoise"))
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    args = parser.parse_args()

    from skimage.filters import threshold_otsu

    from pipeline.utils import load_env_file, secret_env

    load_env_file()
    cases = json.loads(args.cases.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    volumes = fetch_volumes([c["index"] for c in cases], secret_env("HF_TOKEN").reveal())
    results = []
    for c in cases:
        i = c["index"]
        vol = volumes[i]
        dz = depth_axis(vol)
        thr = float(threshold_otsu(vol))
        mid = vol.shape[0] // 2
        low, high = np.percentile(vol[mid], [1, 99])
        to_png(vol[mid], args.out / f"case{i}_original.png", low, high)
        entry = {"index": i, "depth_axis": dz, "threshold": thr, "methods": {}}
        entry["methods"]["original"] = _slice_and_global(vol, vol, dz, thr)["global"]
        for name in CLASSICAL:
            den, seconds = denoise_volume(vol, name, workers=args.workers)
            entry["methods"][name] = {**_slice_and_global(vol, den, dz, thr)["global"], "seconds": round(seconds, 1)}
            to_png(den[mid], args.out / f"case{i}_{name}.png", low, high)
        results.append(entry)
        best = {m: round(v["cnr_bg"], 3) for m, v in entry["methods"].items()}
        print(f"[case {i}] cnr_bg {best}", flush=True)
    (args.out / "metrics.json").write_text(
        json.dumps({"params": {m: METHODS[m]["params"] for m in CLASSICAL}, "cases": results}, indent=1)
    )


if __name__ == "__main__":
    main()
