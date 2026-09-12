"""GPU-only per-B-scan db4 BayesShrink; CPU references are validation-only.

DWT/IDWT follow PyWavelets 1.8's convolution order, including reversed right
overhang terms. Subband energy follows NumPy 2.1's 8192-element reduction chunks,
128-element pairwise leaves and eight float32 accumulators. MAD uses the finest
diagonal band, excluding exact zeros; an empty band deliberately propagates NaN.
"""

import hashlib
import math
import time
import warnings
from pathlib import Path

import numpy as np

_SOURCE = r"""
extern "C" __global__ void normalize(const unsigned char* x, float* y, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) y[k] = __fdiv_rn((float)x[k], 255.0f);
}
__device__ int reflect(int i, int n) {
    i = ((i % (2*n)) + 2*n) % (2*n);
    return i < n ? i : 2*n-1-i;
}
extern "C" __global__ void dwt(const float* x, float* a, float* d,
    const float* taps, int n, int stride, int outn, int size) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= size) return;
    int c = (k / stride) % outn, i = 2*c+1;
    int base = (k / (stride*outn)) * stride*n + k % stride;
    float lo = 0, hi = 0;
    int over = max(0, i-n+1);
    for (int j=over-1; j>=0; --j) {
        float v = x[base + reflect(i-j,n)*stride];
        lo += taps[j]*v; hi += taps[8+j]*v;
    }
    for (int j=over; j<8; ++j) {
        float v = x[base + reflect(i-j,n)*stride];
        lo += taps[j]*v; hi += taps[8+j]*v;
    }
    a[k] = lo; d[k] = hi;
}
extern "C" __global__ void idwt(const double* a, const double* d, double* y,
    const double* taps, int n, int stride, int outn, int size) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= size) return;
    int c = (k / stride) % outn, i = c/2+3;
    int base = (k / (stride*outn)) * stride*n + k % stride;
    double lo = 0, hi = 0;
    for (int j=0; j<4; ++j) {
        lo += taps[2*j+c%2]*a[base+(i-j)*stride];
        hi += taps[8+2*j+c%2]*d[base+(i-j)*stride];
    }
    y[k] = lo + hi;
}
extern "C" __global__ void mean_blocks(const float* x, float* sums, int size) {
    __shared__ float tree[128];
    int t = threadIdx.x, start = blockIdx.x*8192;
    int n = min(8192, size-start), leaves = 1, largest = n;
    while (largest > 128) { largest -= (largest/2)/8*8; leaves *= 2; }
    if (t < leaves) {
        bool active = true;
        for (int bit=leaves/2; bit; bit/=2) {
            if (n <= 128) { active = (t % (2*bit) == 0); break; }
            int half = (n/2)/8*8;
            if (t & bit) { start += half; n -= half; }
            else n = half;
        }
        float sum = -0.0f;
        if (!active) {
            sum = 0;
        } else if (n < 8) {
            for (int j=0; j<n; ++j) sum += x[start+j]*x[start+j];
        } else {
            float r[8];
            for (int j=0; j<8; ++j) r[j] = x[start+j]*x[start+j];
            int j=8;
            for (; j<n-n%8; j+=8)
                for (int q=0; q<8; ++q) r[q] += x[start+j+q]*x[start+j+q];
            sum = ((r[0]+r[1])+(r[2]+r[3])) + ((r[4]+r[5])+(r[6]+r[7]));
            for (; j<n; ++j) sum += x[start+j]*x[start+j];
        }
        tree[t] = sum;
    }
    __syncthreads();
    for (int width=1; width<leaves; width*=2) {
        if (t % (2*width) == 0 && t+width < leaves) tree[t] += tree[t+width];
        __syncthreads();
    }
    if (t == 0) sums[blockIdx.x] = tree[0];
}
extern "C" __global__ void mean_finish(const float* sums, double* out, int chunks, int size) {
    float sum = 0;
    for (int i=0; i<chunks; ++i) sum += sums[i];
    out[0] = (double)__fdiv_rn(sum, (float)size);
}
"""


class WaveletBackend:
    name = "wavelet"

    def __init__(self):
        import pywt
        import scipy
        import skimage

        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("No CUDA device")
        except Exception as exc:
            raise RuntimeError("Wavelet requires working CuPy/CUDA; no CPU fallback") from exc
        self.cp = cp
        wavelet = pywt.Wavelet("db4")
        self.dec = cp.asarray(np.array(wavelet.filter_bank[:2], dtype=np.float32).ravel())
        self.rec = cp.asarray(np.array(wavelet.filter_bank[2:], dtype=np.float64).ravel())
        self.kernels = {
            name: cp.RawKernel(_SOURCE, name, options=("--fmad=false",))
            for name in ("normalize", "dwt", "idwt", "mean_blocks", "mean_finish")
        }
        device = cp.cuda.Device().id
        device_name = cp.cuda.runtime.getDeviceProperties(device)["name"]
        self.metadata = {
            "method": self.name,
            "parameters": {
                "wavelet": "db4",
                "mode": "soft",
                "method": "BayesShrink",
                "sigma": None,
                "wavelet_levels": None,
                "level_rule": "max(dwtn_max_level(plane_shape, db4)-3, 1)",
                "boundary": "symmetric",
                "slice_axis": 0,
                "channel_axis": None,
                "rescale_sigma": True,
            },
            "implementation": "scripts.denoise_wavelet_cuda.WaveletBackend",
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "backend": "cupy",
            "cupy_version": cp.__version__,
            "numpy_version": np.__version__,
            "pywt_version": pywt.__version__,
            "skimage_version": skimage.__version__,
            "scipy_version": scipy.__version__,
            "reference_versions": {"scikit-image": "0.26.0", "PyWavelets": "1.8.0", "numpy": "2.1.1"},
            "device": device_name.decode() if isinstance(device_name, bytes) else str(device_name),
            "device_id": device,
            "validated": False,
            "quantization": "float32 /255, clip(filtered*255,0,255), uint8 truncation; NaN to zero",
            "precision": "DWT float32; MAD/threshold/IDWT float64; output float32; FMA disabled",
            "variant": "historical-2d",
        }
        self.last_seconds = None

    def _transform(self, a, axis, d=None):
        cp = self.cp
        inverse = d is not None
        a = cp.ascontiguousarray(a, dtype=cp.float64 if inverse else cp.float32)
        shape = list(a.shape)
        n = shape[axis]
        shape[axis] = 2 * n - 6 if inverse else (n + 7) // 2
        out = cp.empty(shape, a.dtype)
        other = cp.ascontiguousarray(d, dtype=cp.float64) if inverse else cp.empty_like(out)
        stride = math.prod(a.shape[axis + 1 :])
        args = (a, other, out, self.rec) if inverse else (a, out, other, self.dec)
        self.kernels["idwt" if inverse else "dwt"](
            ((out.size + 127) // 128,),
            (128,),
            (*args, np.int32(n), np.int32(stride), np.int32(shape[axis]), np.int32(out.size)),
        )
        return out if inverse else (out, other)

    def filtered_float(self, raw):
        """Return unclipped float32 reconstruction, as denoise_wavelet(float32) does."""
        if not isinstance(raw, np.ndarray) or raw.dtype != np.uint8:
            raise TypeError("Expected a NumPy uint8 volume")
        if raw.ndim != 3 or min(raw.shape) < 1:
            raise ValueError("Expected nonempty 3D volume with B-scan axis 0")
        cp = self.cp
        result = cp.empty(raw.shape, cp.float32)
        levels = max(max(0, int(math.log2(min(raw.shape[1:]) / 7))) - 3, 1)
        for index in range(len(raw)):
            src = cp.asarray(np.ascontiguousarray(raw[index]))
            a = cp.empty(src.shape, cp.float32)
            self.kernels["normalize"](((src.size + 127) // 128,), (128,), (src, a, np.int32(src.size)))
            details = []
            for _ in range(levels):
                low, high = self._transform(a, 0)
                a, ad = self._transform(low, 1)
                da, dd = self._transform(high, 1)
                details.append((ad, da, dd))
            dd = details[0][-1]
            nonzero = cp.abs(dd[dd != 0])
            median = cp.median(nonzero) if nonzero.size else cp.asarray(cp.nan, dtype=cp.float32)
            sigma = median.astype(cp.float64) / np.float64(0.6744897501960817)
            var = sigma * sigma
            for bands in reversed(details):
                shrunk = []
                for band in bands:
                    chunks = (band.size + 8191) // 8192
                    sums = cp.empty(chunks, cp.float32)
                    dvar = cp.empty((), cp.float64)
                    self.kernels["mean_blocks"]((chunks,), (128,), (band, sums, np.int32(band.size)))
                    self.kernels["mean_finish"]((1,), (1,), (sums, dvar, np.int32(chunks), np.int32(band.size)))
                    threshold = var / cp.sqrt(cp.maximum(dvar - var, np.finfo(np.float32).eps))
                    factor = cp.maximum(1.0 - threshold / cp.abs(band).astype(cp.float64), 0.0)
                    shrunk.append(band * factor)
                ad, da, dd = shrunk
                a = a[: ad.shape[0], : ad.shape[1]]
                low = self._transform(a, 1, ad)
                high = self._transform(da, 1, dd)
                a = self._transform(low, 0, high)
            result[index] = a[: raw.shape[1], : raw.shape[2]].astype(cp.float32)
        return result

    def __call__(self, raw):
        start = time.perf_counter()
        cp = self.cp
        filtered = self.filtered_float(raw)
        out = cp.nan_to_num(cp.clip(filtered * cp.float32(255), 0, 255), nan=0).astype(cp.uint8)
        cp.cuda.get_current_stream().synchronize()
        self.last_seconds = time.perf_counter() - start
        return out

    def validate(self, oct_sample=None):
        """Strict byte parity gate; CPU filtering is confined to this explicit check."""
        from skimage.restoration import denoise_wavelet

        self.metadata["validated"] = False
        self.metadata.pop("validation", None)
        rng = np.random.default_rng(42)
        edge = np.zeros((2, 199, 201), np.uint8)
        edge[:, :, 100:] = 255
        cases = {
            "constants": np.stack([np.full((200, 200), v, np.uint8) for v in (0, 13, 127, 255)]),
            "edges_odd": edge,
            "noise": rng.integers(0, 256, (2, 200, 200), dtype=np.uint8),
            "tiny": rng.integers(0, 256, (2, 3, 5), dtype=np.uint8),
        }
        if oct_sample is not None:
            cases["oct_sample"] = oct_sample[: min(8, len(oct_sample))]
        results = {}
        for name, raw in cases.items():
            actual_float = self.cp.asnumpy(self.filtered_float(raw))
            actual = self.cp.asnumpy(self(raw))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", UserWarning)
                expected_float = np.stack(
                    [
                        denoise_wavelet(p, wavelet="db4", mode="soft", method="BayesShrink", rescale_sigma=True)
                        for p in raw.astype(np.float32) / 255.0
                    ]
                )
                expected = np.clip(expected_float * 255.0, 0, 255).astype(np.uint8)
            error = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
            finite = np.isfinite(expected_float) & np.isfinite(actual_float)
            float_error = np.abs(actual_float[finite].astype(np.float64) - expected_float[finite])
            nan_match = bool(np.array_equal(np.isnan(actual_float), np.isnan(expected_float)))
            results[name] = {
                "passed": bool(not np.any(error) and nan_match),
                "shape": list(raw.shape),
                "max_abs_error": int(error.max()),
                "mismatched_pixels": int(np.count_nonzero(error)),
                "float_max_abs_error": float(float_error.max()) if float_error.size else 0.0,
                "float_mean_abs_error": float(float_error.mean()) if float_error.size else 0.0,
                "nan_mask_equal": nan_match,
                "seconds": self.last_seconds,
            }
        versions_match = (
            self.metadata["numpy_version"] == "2.1.1"
            and self.metadata["pywt_version"] == "1.8.0"
            and self.metadata["skimage_version"] == "0.26.0"
        )
        passed = versions_match and all(case["passed"] for case in results.values())
        report = {
            "passed": passed,
            "kind": "exact-uint8-CPU-parity",
            "cpu_equivalent": passed,
            "versions_match": versions_match,
            "cases": results,
        }
        self.metadata.update(validated=passed, validation=report)
        return report
