"""Pilot runner for VapourSynth-BM3DCUDA (R2.15) on cached 200^3 OCT volumes.

Feeds each volume's 200 B-scans as GRAYS float32 frames, runs two-step spatial BM3D
(basic + empirical Wiener) at radius=0, and reports wall time plus live GPU usage.

Requires: `pip install vapoursynth` and the plugin DLL from
https://github.com/WolframRhodium/VapourSynth-BM3DCUDA/releases (see docs/bm3d-cuda-local.md).
"""

import argparse
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import vapoursynth as vs

DEFAULT_DLL = os.environ.get("BM3D_DLL", "")
DEFAULT_VOLUME_CACHE = os.environ.get("GF_VOL_CACHE", str(Path(tempfile.gettempdir()) / "gf_vol_cache"))


class GpuSampler:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=10,
                ).strip()
                util, mem = (int(value) for value in out.split(","))
                self.samples.append((util, mem))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)

    def summary(self):
        if not self.samples:
            return {"samples": 0}
        util = [s[0] for s in self.samples]
        mem = [s[1] for s in self.samples]
        return {
            "samples": len(self.samples),
            "gpu_util_max": max(util),
            "gpu_util_mean": round(sum(util) / len(util), 1),
            "mem_used_max_mib": max(mem),
        }


def load_plugin(dll):
    if not dll:
        raise RuntimeError("Set --dll or BM3D_DLL to the bm3dcuda.dll path")
    if not Path(dll).is_file():
        raise FileNotFoundError(dll)
    vs.core.std.LoadPlugin(path=str(dll))


def build_clip(core, volume, repeat):
    stack = np.tile(volume, (repeat, 1, 1))
    data = np.ascontiguousarray(stack.astype(np.float32) / 255.0)
    clip = core.std.BlankClip(
        width=data.shape[2], height=data.shape[1], format=vs.GRAYS, length=data.shape[0], color=[0.0]
    )

    def fill(n, f):
        writable = f.copy()
        np.asarray(writable[0])[:] = data[n]
        return writable

    return core.std.ModifyFrame(clip, clip, fill), data.shape[0]


def run(core, volume, repeat, sigma, two_step, warm=True):
    src, frames = build_clip(core, volume, repeat)
    basic = core.bm3dcuda.BM3D(src, sigma=sigma, radius=0, fast=False)
    out = core.bm3dcuda.BM3D(src, ref=basic, sigma=sigma, radius=0, fast=False) if two_step else basic
    if warm:
        np.asarray(out.get_frame(0)[0])
    with GpuSampler() as gpu:
        started = time.perf_counter()
        result = np.empty((frames, src.height, src.width), dtype=np.float32)
        for n in range(frames):
            result[n] = np.asarray(out.get_frame(n)[0])
        elapsed = time.perf_counter() - started
    return result, elapsed, frames, gpu.summary()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", default=DEFAULT_DLL)
    parser.add_argument("--volume-cache", default=DEFAULT_VOLUME_CACHE)
    parser.add_argument("--volumes", default="2404")
    parser.add_argument("--repeat", type=int, default=1, help="tile the volume N times to lengthen the workload")
    parser.add_argument("--sigma", type=float, default=10.0, help="8-bit-scale strength; not tuned")
    parser.add_argument("--one-step-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/bm3d_pilot"))
    args = parser.parse_args()
    load_plugin(args.dll)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for vid in [value.strip() for value in args.volumes.split(",") if value.strip()]:
        volume = np.load(Path(args.volume_cache) / f"raw_{vid}.npy")
        result, elapsed, frames, gpu = run(vs.core, volume, args.repeat, args.sigma, not args.one_step_only)
        first = np.asarray(result[0])
        print(
            f"volume {vid}: repeat={args.repeat} frames={frames} sigma={args.sigma} two_step={not args.one_step_only}"
        )
        print(f"  eval_s={elapsed:.3f} per_frame_ms={1000 * elapsed / frames:.2f} fps={frames / elapsed:.1f}")
        print(
            f"  shape={result.shape} min={result.min():.4f} max={result.max():.4f} finite={np.isfinite(result).all()}"
        )
        print(f"  cuda {gpu}")
        print(
            f"  output_frame0_mean={first.mean():.4f} raw_frame0_mean={volume[0].astype(np.float32).mean() / 255:.4f}"
        )
        np.save(args.out_dir / f"bm3d_pilot_{vid}.npy", result)


if __name__ == "__main__":
    main()
