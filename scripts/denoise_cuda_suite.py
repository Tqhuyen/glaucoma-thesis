"""Lazy CUDA filters and raw-fixed, full-field OCT descriptive metrics.

This module does not schedule experiments, download data, or publish artifacts.
Callers must inspect validate_backend()['passed'] before running a dataset.
"""

import hashlib
import inspect
import os
import tempfile
import time
from pathlib import Path

import numpy as np

BM3D_SHA256 = "a1cce5fc1d16ddb7200107c66c9edc68ea35568ef62af2b24528cdb144465076"
BM3D_RELATIVE_DLL = "bm3d/extracted/VapourSynth-BM3DCUDA-R2.15/bm3dcuda.dll"
_BM3D_CORE = None
_BM3D_PATH = None

_GAUSSIAN_KERNEL = r"""
extern "C" __global__ void gaussian_symmetric_nearest(
    const float* image, float* output, const double* weights,
    const long long size, const int stride, const int length, const int radius) {
    const long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= size) return;
    const int coordinate = (index / stride) % length;
    const long long base = index - (long long)coordinate * stride;
    double value = (double)image[index] * weights[radius];
    for (int offset = radius; offset > 0; --offset) {
        const int left = max(0, coordinate - offset);
        const int right = min(length - 1, coordinate + offset);
        const double pair = (double)image[base + (long long)left * stride]
                          + (double)image[base + (long long)right * stride];
        value = value + pair * weights[radius - offset];
    }
    output[index] = (float)value;
}
"""

metric_protocol = {
    "version": "raw-fixed-full-field-bscan-v2",
    "intensity_units": "uint8 saved output, 0..255; pixel arithmetic float32, metric reductions float64",
    "mask": "one global raw uint8 Otsu threshold; signal raw > threshold, background raw <= threshold",
    "otsu": "256-bin histogram, first maximum between-class variance; constant input returns its value",
    "region": "full field, no filtered-dependent bands, cropping, segmentation or RNFL claim",
    "slice_axis": 0,
    "depth_axis": "argmax std(raw mean profile), float32; metadata only, does not rotate B-scans",
    "std": "population standard deviation (ddof=0)",
    "SNR": "mu_s / sigma_s",
    "ENL": "SNR ** 2",
    "CNR": "(mu_s - mu_b) / sqrt(sigma_s**2 + sigma_b**2)",
    "bg_sigma": "sigma_b on the raw-fixed background mask",
    "SNR_bg": "mu_s / sigma_b",
    "ENL_bg": "(mu_b / sigma_b) ** 2",
    "CNR_bg": "(mu_s - mu_b) / sigma_b",
    "beta": "Pearson correlation of 2D gradient magnitudes on raw > per-B-scan 85th percentile",
    "gradient": "numpy.gradient equivalent, edge_order=1, source axes (1,2); no through-slice gradient",
    "beta_global": "pool selected gradient pairs across B-scans, not mean of slice correlations",
    "reductions": "float64 per-slice sufficient statistics; centered beta covariance merge; one batched CPU transfer",
    "gradient_magnitude_ratio": "sum(filtered selected magnitudes) / sum(raw selected magnitudes)",
    "residual": "raw - saved output; RMS, population std and mean over all pixels",
    "saturation": "fractions of saved output equal to 0 and 255, not pre-clipping frequency",
    "invalid": "None with invalid_reasons entry; empty masks, zero denominators, beta count < 3 or zero variance",
}


def _cupy():
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA device")
        return cp
    except Exception as exc:
        raise RuntimeError("CuPy with a working CUDA device is required; no CPU fallback") from exc


def _check_volume(volume):
    if volume.ndim != 3 or volume.dtype != np.uint8 or min(volume.shape) < 1:
        raise ValueError("Expected a nonempty 3D uint8 volume with source slice axis 0")


def _require_memory(cp):
    free, _ = cp.cuda.runtime.memGetInfo()
    if free < 1024**3:
        raise RuntimeError("CUDA suite requires at least 1 GiB free device memory")


def release_memory():
    """Release unused CuPy pool blocks after callers drop volume/context references.

    Live arrays (including another consumer's arrays) are never freed. This affects
    the process-wide default pools, not another process or a CUDA context reset.
    """
    cp = _cupy()
    cp.cuda.get_current_stream().synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


class _CudaGaussian:
    """SciPy NI_Correlate1D symmetric summation, float64 weights/accumulator.

    Each separable pass is rounded to float32, including before the second axis.
    Only the 13 coefficients are constructed on CPU; all pixel work is CUDA.
    """

    def __init__(self, cp):
        import scipy
        from scipy.ndimage._filters import _gaussian_kernel1d

        self.cp = cp
        weights = _gaussian_kernel1d(1.5, 0, 6).astype(np.float64)
        self.weights = cp.asarray(weights)
        self.coefficients_sha256 = hashlib.sha256(weights.tobytes()).hexdigest()
        self.scipy_version = scipy.__version__
        self.kernel = cp.RawKernel(
            _GAUSSIAN_KERNEL,
            "gaussian_symmetric_nearest",
            options=("--fmad=false", "--ftz=false"),
        )
        self.kernel.compile()

    def __call__(self, image):
        cp = self.cp
        current = cp.ascontiguousarray(image, dtype=cp.float32)
        for stride, length in ((image.shape[2], image.shape[1]), (1, image.shape[2])):
            output = cp.empty_like(current)
            self.kernel(
                ((image.size + 127) // 128,),
                (128,),
                (current, output, self.weights, np.int64(image.size), np.int32(stride), np.int32(length), np.int32(6)),
            )
            current = output
        return current


def _tv_chambolle(image, cp):
    result = cp.empty_like(image)
    for start in range(0, len(image), 32):
        x = image[start : start + 32]
        p = cp.zeros((2,) + x.shape, dtype=cp.float32)
        g = cp.zeros_like(p)
        active = cp.ones(len(x), dtype=cp.bool_)
        frozen = x.copy()
        previous = initial = None
        for iteration in range(200):
            d = -p.sum(axis=0)
            d[:, 1:, :] += p[0, :, :-1, :]
            d[:, :, 1:] += p[1, :, :, :-1]
            out = x + d if iteration else x
            g[0, :, :-1, :] = cp.diff(out, axis=1)
            g[1, :, :, :-1] = cp.diff(out, axis=2)
            norm = cp.sqrt((g * g).sum(axis=0))
            energy = (d * d).sum(axis=(1, 2))
            energy += 0.1 * norm.sum(axis=(1, 2))
            energy /= float(x.shape[1] * x.shape[2])
            frozen = cp.where(active[:, None, None], out, frozen)
            if iteration == 0:
                initial = energy
            else:
                active &= cp.abs(previous - energy) >= 0.0002 * initial
            previous = energy
            norm *= 0.25 / 0.1
            norm += 1.0
            p -= 0.25 * g
            p /= norm[None]
        result[start : start + len(x)] = frozen
    return result


def _load_bm3d(dll):
    global _BM3D_CORE, _BM3D_PATH
    configured = dll or os.environ.get("BM3D_DLL")
    if configured:
        path = Path(configured).resolve()
    else:
        root = Path(tempfile.gettempdir())
        candidates = (root / BM3D_RELATIVE_DLL, root / "opencode" / BM3D_RELATIVE_DLL)
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0]).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != BM3D_SHA256:
        raise RuntimeError(f"BM3DCUDA plugin SHA256 mismatch: {digest}")
    import vapoursynth as vs

    if vs.__api_version__.api_major != 4 or vs.__version__.release_major != 79:
        raise RuntimeError("BM3D adapter requires VapourSynth R79 API 4")
    if _BM3D_CORE is None:
        vs.core.std.LoadPlugin(path=str(path))
        _BM3D_CORE, _BM3D_PATH = vs.core, path
    elif path != _BM3D_PATH:
        raise RuntimeError("BM3DCUDA is already loaded from a different path")
    return _BM3D_CORE


class Backend:
    def __init__(self, name, dll=None):
        if name in {"nlm", "wavelet"}:
            raise NotImplementedError(f"{name}: CUDA port pending independent parity validation")
        parameters = {
            "original": {},
            "gaussian": {"sigma": (0, 1.5, 1.5), "mode": "nearest", "truncate": 4.0},
            "median": {"size": (1, 3, 3), "mode": "nearest"},
            "tv": {"weight": 0.1, "eps": 0.0002, "max_num_iter": 200, "batch_planes": 32},
            "bilateral": {"sigma_color": 0.1, "sigma_spatial": 4.0, "bins": 10000, "mode": "constant"},
            "bm3d": {"sigma": 10.0, "radius": 0, "fast": False, "stages": 2},
        }
        if name not in parameters:
            raise ValueError(f"Unknown CUDA method: {name}")
        self.name, self.cp = name, _cupy()
        _require_memory(self.cp)
        self.engine = None
        implementation = inspect.getsource(type(self)) + inspect.getsource(_tv_chambolle)
        if name == "gaussian":
            self.engine = _CudaGaussian(self.cp)
            implementation += inspect.getsource(_CudaGaussian) + _GAUSSIAN_KERNEL + self.engine.coefficients_sha256
        elif name == "bilateral":
            from scripts import bilateral_cuda

            self.engine = bilateral_cuda.CudaBilateral()
            implementation += inspect.getsource(bilateral_cuda)
        elif name == "bm3d":
            self.engine = _load_bm3d(dll)
            implementation += inspect.getsource(_load_bm3d)
        props = self.cp.cuda.runtime.getDeviceProperties(self.cp.cuda.Device().id)
        device_name = props["name"]
        self.metadata = {
            "method": name,
            "parameters": parameters[name],
            "implementation": "scripts.denoise_cuda_suite.Backend",
            "implementation_sha256": hashlib.sha256(implementation.encode()).hexdigest(),
            "backend": "VapourSynth-BM3DCUDA-R2.15" if name == "bm3d" else "cupy",
            "cupy_version": self.cp.__version__,
            "device": device_name.decode() if isinstance(device_name, bytes) else str(device_name),
            "device_id": self.cp.cuda.Device().id,
            "quantization": "float32 /255, clip(filtered*255,0,255), uint8 truncation",
            "variant": "new-fixed-sigma-GPU-BM3D-not-CPU-profile-equivalent" if name == "bm3d" else "historical-2d",
            "validated": False,
        }
        if name == "bm3d":
            self.metadata.update(plugin_sha256=BM3D_SHA256, plugin_path=str(_BM3D_PATH), vapoursynth_release=79)
        elif name == "gaussian":
            self.metadata.update(
                implementation="scripts.denoise_cuda_suite._CudaGaussian",
                arithmetic="SciPy symmetric center-first outer-to-inner float64 sum; float32 per-axis output; no FMA",
                coefficients_sha256=self.engine.coefficients_sha256,
                scipy_version=self.engine.scipy_version,
            )
        elif name == "original":
            self.metadata.update(quantization="none; unchanged uint8 bytes", variant="raw-passthrough")
        self.last_seconds = None

    def __call__(self, raw):
        if not isinstance(raw, np.ndarray):
            raise TypeError("Backend input must be a NumPy uint8 volume")
        _check_volume(raw)
        cp = self.cp
        _require_memory(cp)
        cp.cuda.runtime.deviceSynchronize()
        started = time.perf_counter()
        if self.name == "original":
            out = cp.asarray(raw)
        elif self.name == "bilateral":
            out = cp.asarray(self.engine(raw))
        elif self.name == "bm3d":
            import vapoursynth as vs

            data = np.ascontiguousarray(raw, dtype=np.float32) / 255.0
            core = self.engine
            clip = core.std.BlankClip(width=raw.shape[2], height=raw.shape[1], format=vs.GRAYS, length=len(raw))

            def fill(n, f):
                frame = f.copy()
                np.asarray(frame[0])[:] = data[n]
                return frame

            src = core.std.ModifyFrame(clip, clip, fill)
            basic = core.bm3dcuda.BM3D(src, sigma=10.0, radius=0, fast=False)
            filtered = core.bm3dcuda.BM3D(src, ref=basic, sigma=10.0, radius=0, fast=False)
            host = np.empty_like(raw)
            for i in range(len(raw)):
                with filtered.get_frame(i) as frame:
                    pixels = np.asarray(frame[0])
                    if not np.isfinite(pixels).all():
                        raise RuntimeError("BM3DCUDA returned nonfinite pixels")
                    host[i] = np.clip(pixels * 255.0, 0, 255).astype(np.uint8)
            out = cp.asarray(host)
        else:
            x = cp.asarray(raw, dtype=cp.float32) / cp.float32(255)
            if self.name == "gaussian":
                filtered = self.engine(x)
            elif self.name == "median":
                from cupyx.scipy import ndimage

                filtered = ndimage.median_filter(x, size=(1, 3, 3), mode="nearest")
            else:
                filtered = _tv_chambolle(x, cp)
            if not bool(cp.isfinite(filtered).all()):
                raise RuntimeError(f"{self.name} returned nonfinite pixels")
            out = cp.clip(filtered * cp.float32(255), 0, 255).astype(cp.uint8)
        cp.cuda.runtime.deviceSynchronize()
        self.last_seconds = time.perf_counter() - started
        return out


def get_backend(name, dll=None):
    return Backend(name, dll=dll)


def get_method(name, dll=None):
    return get_backend(name, dll=dll)


def validate_backend(backend, oct_sample=None):
    """Bounded synthetic CPU-reference parity, with no data access or CPU fallback.

    BM3D checks execution only: it cannot establish equivalence to the historical
    CPU profile. A false passed value is a mandatory dataset-run pause signal.
    """
    backend.metadata["validated"] = False
    backend.metadata.pop("validation", None)
    rng = np.random.default_rng(42)
    edge = np.zeros((2, 17, 19), dtype=np.uint8)
    edge[:, :, 9:] = 255
    cases = {
        "constant": np.stack([np.full((17, 19), v, np.uint8) for v in (0, 13, 127, 255)]),
        "edge": edge,
        "random": rng.integers(0, 256, (3, 17, 19), dtype=np.uint8),
    }
    if oct_sample is not None:
        _check_volume(oct_sample)
        cases["oct_sample"] = np.ascontiguousarray(oct_sample[:3, :32, :32])
    import skimage
    from scipy.ndimage import gaussian_filter, median_filter
    from skimage.restoration import denoise_bilateral, denoise_tv_chambolle

    results = {}
    for label, raw in cases.items():
        actual = backend.cp.asnumpy(backend(raw))
        stats = {"seconds": backend.last_seconds, "shape": list(actual.shape)}
        if backend.name == "bm3d":
            stats.update(passed=actual.shape == raw.shape and actual.dtype == np.uint8, reference="execution-only")
        else:
            x = raw.astype(np.float32) / 255.0
            if backend.name == "original":
                expected = raw
            else:
                if backend.name == "gaussian":
                    filtered = gaussian_filter(x, (0, 1.5, 1.5), mode="nearest", truncate=4.0)
                elif backend.name == "median":
                    filtered = median_filter(x, (1, 3, 3), mode="nearest")
                elif backend.name == "tv":
                    filtered = np.stack([denoise_tv_chambolle(p, weight=0.1, eps=0.0002, max_num_iter=200) for p in x])
                else:
                    filtered = np.stack([denoise_bilateral(p, sigma_color=0.1, sigma_spatial=4.0) for p in x])
                expected = np.clip(filtered * 255.0, 0, 255).astype(np.uint8)
            error = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
            stats.update(
                passed=bool(np.array_equal(actual, expected)),
                max_abs_error=int(error.max()),
                mean_abs_error=float(error.mean()),
                mismatched_pixels=int(np.count_nonzero(error)),
            )
        results[label] = stats
    passed = all(case["passed"] for case in results.values())
    if backend.name == "tv" and skimage.__version__ != "0.26.0":
        passed = False
    validation = {
        "passed": passed,
        "kind": "execution-only-new-variant" if backend.name == "bm3d" else "exact-uint8-CPU-parity",
        "cpu_equivalent": passed and backend.name != "bm3d",
        "reference_skimage": skimage.__version__,
        "cases": results,
    }
    backend.metadata["validated"] = passed
    backend.metadata["validation"] = validation
    return validation


def _otsu(raw, cp):
    hist = cp.bincount(raw.ravel(), minlength=256).astype(cp.float32)
    occupied = cp.flatnonzero(hist)
    if len(occupied) == 1:
        return int(occupied[0])
    lo, hi = int(occupied[0]), int(occupied[-1])
    counts = hist[lo : hi + 1]
    bins = cp.arange(lo, hi + 1)
    w1 = cp.cumsum(counts)
    w2 = cp.cumsum(counts[::-1])[::-1]
    m1 = cp.cumsum(counts * bins) / w1
    m2 = (cp.cumsum((counts * bins)[::-1]) / w2[::-1])[::-1]
    variance = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    return lo + int(cp.argmax(variance))


def _gradient(image, cp):
    dy, dx = cp.gradient(image, axis=(1, 2))
    return cp.sqrt(dy * dy + dx * dx)


class MetricContext:
    def __init__(self, raw_np, *, slice_axis=0, depth_axis=None):
        _check_volume(raw_np)
        if slice_axis != 0:
            raise ValueError("The metric protocol requires source slice_axis=0")
        if min(raw_np.shape[1:]) < 2:
            raise ValueError("B-scans need at least two pixels per in-plane axis for gradients")
        if depth_axis is not None and depth_axis not in (0, 1, 2):
            raise ValueError("depth_axis must be 0, 1, 2 or None")
        self.cp = cp = _cupy()
        _require_memory(cp)
        raw = cp.asarray(raw_np)
        self.threshold = _otsu(raw, cp)
        self.raw = raw.astype(cp.float32)
        self.signal = raw > self.threshold
        self.depth_axis = depth_axis
        if depth_axis is None:
            profile_std = cp.stack([self.raw.mean(axis=tuple(i for i in range(3) if i != a)).std() for a in range(3)])
            self.depth_axis = int(cp.argmax(profile_std))
        self.gradient = _gradient(self.raw, cp)
        thresholds = cp.percentile(self.gradient, 85, axis=(1, 2))
        self.edges = self.gradient > thresholds[:, None, None]
        self.protocol = dict(metric_protocol, depth_axis_resolved=self.depth_axis, otsu_threshold=self.threshold)

    def _scores(self, raw, den, signal, raw_g, den_g, edges):
        cp = self.cp
        invalid = {}
        record = {"pixel_count": int(raw.size), "invalid_reasons": invalid}
        for suffix, mask in (("s", signal), ("b", ~signal)):
            values = den[mask].astype(cp.float64)
            record[f"n_{suffix}"] = int(values.size)
            for prefix in ("mu", "sigma"):
                key = f"{prefix}_{suffix}"
                if values.size:
                    record[key] = float(values.mean() if prefix == "mu" else values.std())
                else:
                    record[key] = None
                    invalid[key] = "empty_raw_mask"
        ms, ss, mb, sb = (record[k] for k in ("mu_s", "sigma_s", "mu_b", "sigma_b"))

        def ratio(key, numerator, denominator):
            if numerator is None or denominator is None:
                record[key], invalid[key] = None, "empty_raw_mask"
            elif denominator == 0:
                record[key], invalid[key] = None, "zero_denominator"
            else:
                record[key] = numerator / denominator

        ratio("SNR", ms, ss)
        ratio("SNR_bg", ms, sb)
        contrast = None if ms is None or mb is None else ms - mb
        ratio("CNR", contrast, None if ss is None or sb is None else float(np.hypot(ss, sb)))
        ratio("CNR_bg", contrast, sb)
        ratio("ENL_bg", mb, sb)
        if record["ENL_bg"] is not None:
            record["ENL_bg"] **= 2
        record["ENL"] = None if record["SNR"] is None else record["SNR"] ** 2
        if record["ENL"] is None:
            invalid["ENL"] = invalid["SNR"]
        record["bg_sigma"] = sb
        if sb is None:
            invalid["bg_sigma"] = "empty_raw_mask"
        a, b = raw_g[edges].astype(cp.float64), den_g[edges].astype(cp.float64)
        record["beta_count"] = int(a.size)
        record["beta"] = None
        if a.size < 3:
            invalid["beta"] = "fewer_than_3_raw_edges"
        else:
            ac, bc = a - a.mean(), b - b.mean()
            denominator = float(cp.sqrt(cp.sum(ac * ac) * cp.sum(bc * bc)))
            if denominator == 0:
                invalid["beta"] = "zero_gradient_variance"
            else:
                record["beta"] = float(cp.clip(cp.sum(ac * bc) / denominator, -1, 1))
        ratio("gradient_magnitude_ratio", float(b.sum()), float(a.sum()))
        residual = (raw - den).astype(cp.float64)
        record.update(
            residual_rms=float(cp.sqrt(cp.mean(residual * residual))),
            residual_std=float(residual.std()),
            residual_mean=float(residual.mean()),
            saturation_zero=float(cp.mean(den == 0, dtype=cp.float64)),
            saturation_255=float(cp.mean(den == 255, dtype=cp.float64)),
        )
        return record

    def evaluate(self, out):
        """Reduce all B-scans on device, then transfer one small statistics table.

        Integer intensity moments are exact in float64 at OCT volume sizes. Beta
        uses centered per-slice moments and the parallel covariance merge rule,
        avoiding cancellation from subtracting nearly equal gradient moments.
        """
        _check_volume(out)
        if out.shape != self.raw.shape:
            raise ValueError("Raw and denoised volumes must have identical shapes")
        cp = self.cp
        den = cp.asarray(out, dtype=cp.float32)
        den_g = _gradient(den, cp)
        stats = {}
        for suffix, mask in (("s", self.signal), ("b", ~self.signal)):
            stats[f"n_{suffix}"] = mask.sum(axis=(1, 2), dtype=cp.float64)
            values = cp.where(mask, den, 0)
            stats[f"sum_{suffix}"] = values.sum(axis=(1, 2), dtype=cp.float64)
            stats[f"sq_{suffix}"] = (values * values).sum(axis=(1, 2), dtype=cp.float64)
        counts = stats["beta_count"] = self.edges.sum(axis=(1, 2), dtype=cp.float64)
        stats["gradient_raw_sum"] = cp.where(self.edges, self.gradient, 0).sum(axis=(1, 2), dtype=cp.float64)
        stats["gradient_den_sum"] = cp.where(self.edges, den_g, 0).sum(axis=(1, 2), dtype=cp.float64)
        mean_a = stats["gradient_raw_sum"] / cp.maximum(counts, 1)
        mean_b = stats["gradient_den_sum"] / cp.maximum(counts, 1)
        a = cp.where(self.edges, self.gradient - mean_a[:, None, None], 0)
        b = cp.where(self.edges, den_g - mean_b[:, None, None], 0)
        stats["gradient_m2_a"] = (a * a).sum(axis=(1, 2), dtype=cp.float64)
        stats["gradient_m2_b"] = (b * b).sum(axis=(1, 2), dtype=cp.float64)
        stats["gradient_cov"] = (a * b).sum(axis=(1, 2), dtype=cp.float64)
        del a, b, values
        residual = self.raw - den
        stats["residual_sum"] = residual.sum(axis=(1, 2), dtype=cp.float64)
        stats["residual_sq"] = (residual * residual).sum(axis=(1, 2), dtype=cp.float64)
        stats["zero_count"] = (den == 0).sum(axis=(1, 2), dtype=cp.float64)
        stats["full_count"] = (den == 255).sum(axis=(1, 2), dtype=cp.float64)
        totals = {key: value.sum() for key, value in stats.items()}
        total_n = cp.maximum(totals["beta_count"], 1)
        offset_a = mean_a - totals["gradient_raw_sum"] / total_n
        offset_b = mean_b - totals["gradient_den_sum"] / total_n
        totals["gradient_m2_a"] += (counts * offset_a * offset_a).sum()
        totals["gradient_m2_b"] += (counts * offset_b * offset_b).sum()
        totals["gradient_cov"] += (counts * offset_a * offset_b).sum()
        table = cp.stack([cp.concatenate((totals[key][None], value)) for key, value in stats.items()], axis=1)
        host = np.asarray(table) if cp is np else cp.asnumpy(table)
        records = []
        area = self.raw.shape[1] * self.raw.shape[2]
        for index, values in enumerate(host):
            row = dict(zip(stats, map(float, values), strict=True))
            invalid = {}
            record = {
                "pixel_count": int(self.raw.size if index == 0 else area),
                "invalid_reasons": invalid,
                "otsu_threshold": self.threshold,
                "depth_axis": self.depth_axis,
                "slice_axis": 0,
            }
            for suffix in ("s", "b"):
                count = record[f"n_{suffix}"] = int(row[f"n_{suffix}"])
                mean_key, std_key = f"mu_{suffix}", f"sigma_{suffix}"
                if count:
                    mean = record[mean_key] = row[f"sum_{suffix}"] / count
                    record[std_key] = float(np.sqrt(max(0.0, row[f"sq_{suffix}"] / count - mean * mean)))
                else:
                    record[mean_key] = record[std_key] = None
                    invalid[mean_key] = invalid[std_key] = "empty_raw_mask"
            ms, ss, mb, sb = (record[key] for key in ("mu_s", "sigma_s", "mu_b", "sigma_b"))
            contrast = None if ms is None or mb is None else ms - mb
            cnr_denom = None if ss is None or sb is None else float(np.hypot(ss, sb))
            for key, numerator, denominator in (
                ("SNR", ms, ss),
                ("ENL", ms, ss),
                ("SNR_bg", ms, sb),
                ("ENL_bg", mb, sb),
                ("CNR", contrast, cnr_denom),
                ("CNR_bg", contrast, sb),
                ("gradient_magnitude_ratio", row["gradient_den_sum"], row["gradient_raw_sum"]),
            ):
                if numerator is None or denominator is None:
                    record[key], invalid[key] = None, "empty_raw_mask"
                elif denominator == 0:
                    record[key], invalid[key] = None, "zero_denominator"
                else:
                    record[key] = (numerator / denominator) ** (2 if key in ("ENL", "ENL_bg") else 1)
            record["bg_sigma"] = sb
            if sb is None:
                invalid["bg_sigma"] = "empty_raw_mask"
            record["beta_count"] = int(row["beta_count"])
            record["beta"] = None
            if record["beta_count"] < 3:
                invalid["beta"] = "fewer_than_3_raw_edges"
            elif row["gradient_m2_a"] == 0 or row["gradient_m2_b"] == 0:
                invalid["beta"] = "zero_gradient_variance"
            else:
                denom = np.sqrt(row["gradient_m2_a"] * row["gradient_m2_b"])
                record["beta"] = float(np.clip(row["gradient_cov"] / denom, -1, 1))
            count = record["pixel_count"]
            residual_mean = row["residual_sum"] / count
            residual_sq = row["residual_sq"] / count
            record.update(
                residual_rms=float(np.sqrt(residual_sq)),
                residual_std=float(np.sqrt(max(0.0, residual_sq - residual_mean * residual_mean))),
                residual_mean=residual_mean,
                saturation_zero=row["zero_count"] / count,
                saturation_255=row["full_count"] / count,
            )
            if index:
                record["slice_index"] = index - 1
            records.append(record)
        return records[0], records[1:]


def volume_scores(raw_u8, den_u8, *, slice_axis=0, depth_axis=None):
    return MetricContext(raw_u8, slice_axis=slice_axis, depth_axis=depth_axis).evaluate(den_u8)
