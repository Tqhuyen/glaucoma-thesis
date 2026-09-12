"""Standalone CUDA port of scikit-image 0.26 fast 2D NLM, source slice axis 0.

No scheduling, dataset access, publishing, or CPU production fallback. Validation
is an explicit caller gate; synthetic success alone does not approve a dataset run.
"""

import hashlib
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

PARAMETERS = {"h": 0.1, "patch_size": 5, "patch_distance": 3, "fast_mode": True, "sigma": 0, "channel_axis": None}
REFERENCE = "https://github.com/scikit-image/scikit-image/blob/v0.26.0/src/skimage/restoration/"
OPTIONS = ("--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")

CUDA_SOURCE = r"""
__device__ int reflect_index(int i, int n) {
    if (n == 1) return 0;
    int period = 2 * (n - 1);
    i = ((i % period) + period) % period;
    return i < n ? i : period - i;
}
extern "C" __global__ void prepare(const unsigned char* raw, double* padded,
                                     int h, int w, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int H = h + 12, W = w + 12;
    if (i >= count * H * W) return;
    int p = i / (H * W), r = (i / W) % H, c = i % W;
    float value = __fdiv_rn((float)raw[(p*h + reflect_index(r-6,h))*w + reflect_index(c-6,w)], 255.0f);
    padded[i] = (double)value;
}
extern "C" __global__ void integrals(const double* padded, double* table, int H, int W) {
    int shift = blockIdx.x % 28, plane = blockIdx.x / 28;
    int dr = shift / 4 - 3, dc = shift % 4;
    int begin = max(1, -dr), end = min(H, H-dr);
    double* I = table + (long long)blockIdx.x * H * W;
    const double* P = padded + plane * H * W;
    for (int diagonal = begin + 1; diagonal < end + W-dc-1; ++diagonal) {
        for (int c = 1 + threadIdx.x; c < W-dc; c += blockDim.x) {
            int r = diagonal - c;
            if (r >= begin && r < end) {
                int k = r*W+c;
                double t = P[k] - P[k+dr*W+dc];
                I[k] = ((t*t + I[k-W]) + I[k-1]) - I[k-W-1];
            }
        }
        __syncthreads();
    }
}
__device__ double pair_weight(const double* I, int r, int c, int W, int dc, double scale) {
    double distance = ((I[(r+2)*W+c+2] + I[(r-2)*W+c-2])
                      - I[(r-2)*W+c+2]) - I[(r+2)*W+c-2];
    distance = fmax(distance, 0.0) / scale;
    if (distance > 5.0) return 0.0;
    int high = (int)(1512775.3951951856938 * (-distance)) + 1072632447;
    return (dc == 0 ? 0.5 : 1.0) * __hiloint2double(high, 0);
}
extern "C" __global__ void collect(const double* padded, const double* table,
                                    float* output, int h, int w, int count, double scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count*h*w) return;
    int p = i/(h*w), r = (i/w)%h+6, c = i%w+6, H=h+12, W=w+12;
    const double* P = padded + p*H*W;
    double value = 0.0, total = 0.0;
    for (int shift=0; shift<28; ++shift) {
        int dr=shift/4-3, dc=shift%4;
        const double* I = table + ((long long)p*28+shift)*H*W;
        double outgoing = pair_weight(I,r,c,W,dc,scale);
        double incoming = pair_weight(I,r-dr,c-dc,W,dc,scale);
        bool incoming_first = dr > 0 || (dr == 0 && dc > 0);
        double first = incoming_first ? incoming : outgoing;
        double second = incoming_first ? outgoing : incoming;
        int delta = incoming_first ? -dr*W-dc : dr*W+dc;
        total += first;
        value += first * P[r*W+c+delta];
        total += second;
        value += second * P[r*W+c-delta];
    }
    output[i] = (float)(value / total);
}
extern "C" __global__ void quantize(const float* input, unsigned char* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = (unsigned char)fminf(255.0f, fmaxf(0.0f, input[i]*255.0f));
}
"""


def _check_volume(raw):
    if not isinstance(raw, np.ndarray):
        raise TypeError("Expected a NumPy uint8 volume")
    if raw.ndim != 3 or raw.dtype != np.uint8 or min(raw.shape) < 1:
        raise ValueError("Expected a nonempty 3D uint8 volume, source slice axis 0")


class NLMBackend:
    """All-pixel CUDA NLM with bounded per-batch float64 integral images.

    Diagonal wavefronts preserve the Cython recurrence's operation order. Each
    output thread gathers both pair contributions in original row-major order,
    avoiding atomic races. Only validation imports the CPU reference filter.
    """

    name = "nlm"

    def __init__(self, batch_planes=8):
        if isinstance(batch_planes, bool) or not isinstance(batch_planes, int) or not 1 <= batch_planes <= 8:
            raise ValueError("batch_planes must be an integer in [1, 8]")
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("No CUDA device")
        except Exception as exc:
            raise RuntimeError("CuPy with CUDA is required; no CPU fallback") from exc
        self.cp = cp
        self.batch_planes = batch_planes
        self.pool = cp.cuda.MemoryPool()
        self.pool.set_limit(size=768 * 1024**2)
        self.kernels = {
            name: cp.RawKernel(CUDA_SOURCE, name, options=OPTIONS)
            for name in ("prepare", "integrals", "collect", "quantize")
        }
        for kernel in self.kernels.values():
            kernel.compile()
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        device = props["name"]
        self.metadata = {
            "method": self.name,
            "parameters": dict(PARAMETERS),
            "slice_axis": 0,
            "padding": "numpy reflect, patch radius + search radius + 1 = 6",
            "implementation": "scripts.denoise_nlm_cuda.NLMBackend",
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cuda_source_sha256": hashlib.sha256(CUDA_SOURCE.encode()).hexdigest(),
            "reference": REFERENCE + "_nl_means_denoising.pyx",
            "reference_fast_exp": REFERENCE.replace("restoration/", "_shared/") + "fast_exp.h",
            "backend": "cupy.RawKernel",
            "compiler_options": list(OPTIONS),
            "versions": {name: version(name) for name in ("numpy", "scipy", "scikit-image")},
            "cupy_version": cp.__version__,
            "python_version": platform.python_version(),
            "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
            "cuda_driver_version": cp.cuda.runtime.driverGetVersion(),
            "device": device.decode() if isinstance(device, bytes) else str(device),
            "device_id": cp.cuda.Device().id,
            "batch_planes": batch_planes,
            "memory_limit_bytes": 768 * 1024**2,
            "quantization": "float32 /255, float64 filtering, float32 output, clip(*255,0,255), uint8 truncation",
            "arithmetic": "Cython float64 recurrence, four-corner distance, Schraudolph exp, ordered symmetric pairs",
            "variant": "historical-2d-exact-uint8-target",
            "validated": False,
        }
        self.last_seconds = None
        self.last_peak_pool_bytes = 0

    def __call__(self, raw, *, filtered_float=False):
        _check_volume(raw)
        cp = self.cp
        count, h, w = raw.shape
        batch = min(count, self.batch_planes)
        required = raw.size * (4 if filtered_float else 1) + batch * ((h + 12) * (w + 12) * 232 + h * w * 5)
        if required > 700 * 1024**2:
            raise MemoryError("NLM volume exceeds the bounded 700 MiB allocation budget")
        cp.cuda.get_current_stream().synchronize()
        started = time.perf_counter()
        with cp.cuda.using_allocator(self.pool.malloc):
            output = cp.empty(raw.shape, dtype=cp.float32 if filtered_float else cp.uint8)
            for start in range(0, count, batch):
                n = min(batch, count - start)
                data = cp.asarray(np.ascontiguousarray(raw[start : start + n]))
                padded = cp.empty((n, h + 12, w + 12), dtype=cp.float64)
                table = cp.zeros((n, 28, h + 12, w + 12), dtype=cp.float64)
                floats = cp.empty((n, h, w), dtype=cp.float32)
                dims = tuple(np.int32(v) for v in (h, w, n))
                self.kernels["prepare"](((padded.size + 127) // 128,), (128,), (data, padded, *dims))
                self.kernels["integrals"]((n * 28,), (256,), (padded, table, np.int32(h + 12), np.int32(w + 12)))
                self.kernels["collect"](
                    ((floats.size + 127) // 128,),
                    (128,),
                    (padded, table, floats, *dims, np.float64(1 * 0.1 * 0.1 * 5 * 5)),
                )
                if filtered_float:
                    output[start : start + n] = floats
                else:
                    self.kernels["quantize"](
                        ((floats.size + 127) // 128,),
                        (128,),
                        (floats, output[start : start + n], np.int32(floats.size)),
                    )
                self.last_peak_pool_bytes = max(self.last_peak_pool_bytes, self.pool.total_bytes())
                del data, padded, table, floats
        cp.cuda.get_current_stream().synchronize()
        self.last_seconds = time.perf_counter() - started
        self.pool.free_all_blocks()
        return output

    def filtered_float(self, raw):
        return self(raw, filtered_float=True)

    def validate(self, oct_sample=None):
        """Compare exact saved bytes; optional OCT input is never cropped in-plane.

        For a 200-cubed Training volume, validate eight evenly spaced full
        B-scans and then every pixel of the volume. No dataset is fetched here.
        """
        from skimage.restoration import denoise_nl_means

        self.metadata["validated"] = False
        self.metadata.pop("validation", None)
        rng = np.random.default_rng(42)
        edge = np.zeros((2, 31, 37), np.uint8)
        edge[0, :, 18:] = 255
        edge[1, 15:, :] = 255
        cases = {
            "scalar": np.arange(256, dtype=np.uint8).reshape(256, 1, 1),
            "constant": np.broadcast_to(np.arange(256, dtype=np.uint8)[:, None, None], (256, 7, 9)).copy(),
            "edges": edge,
            "random": rng.integers(0, 256, (8, 31, 37), dtype=np.uint8),
            "high_gray_low_range": rng.integers(239, 256, (8, 200, 200), dtype=np.uint8),
            "low_gray_low_range": rng.integers(0, 17, (8, 200, 200), dtype=np.uint8),
            "random_200": rng.integers(0, 256, (8, 200, 200), dtype=np.uint8),
        }
        if oct_sample is not None:
            _check_volume(oct_sample)
            indices = np.linspace(0, len(oct_sample) - 1, min(8, len(oct_sample)), dtype=int)
            cases["oct_8_slices"] = oct_sample[indices]
            cases["oct_full_volume"] = oct_sample
        results = {}
        for label, raw in cases.items():
            actual_float = self.cp.asnumpy(self.filtered_float(raw))
            gpu_seconds = self.last_seconds
            actual = self.cp.asnumpy(self(raw))
            started = time.perf_counter()
            x = raw.astype(np.float32) / np.float32(255)
            expected_float = np.stack([np.asarray(denoise_nl_means(p, **PARAMETERS)).reshape(p.shape) for p in x])
            expected = np.clip(expected_float * np.float32(255), 0, 255).astype(np.uint8)
            error = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
            results[label] = {
                "passed": bool(np.array_equal(actual, expected)),
                "shape": list(raw.shape),
                "input_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                "mismatched_pixels": int(np.count_nonzero(error)),
                "max_abs_error": int(error.max()),
                "mean_abs_error": float(error.mean()),
                "float_max_abs_error": float(np.max(np.abs(actual_float - expected_float))),
                "float_mismatched_pixels": int(np.count_nonzero(actual_float != expected_float)),
                "gpu_float_seconds": gpu_seconds,
                "gpu_uint8_seconds": self.last_seconds,
                "cpu_reference_seconds": time.perf_counter() - started,
            }
        passed = all(case["passed"] for case in results.values()) and version("scikit-image") == "0.26.0"
        validation = {
            "passed": passed,
            "kind": "exact-uint8-CPU-parity",
            "reference_skimage": version("scikit-image"),
            "cases": results,
            "oct_full_volume_checked": oct_sample is not None and oct_sample.shape == (200, 200, 200),
            "dataset_run_ready": passed and oct_sample is not None and oct_sample.shape == (200, 200, 200),
            "peak_pool_bytes": self.last_peak_pool_bytes,
        }
        self.metadata.update(validated=passed, validation=validation)
        return validation
