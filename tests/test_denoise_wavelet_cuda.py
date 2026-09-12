"""Bounded real-CUDA parity checks; no uploads, training, or saved results."""

import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

from scripts import denoise_wavelet_cuda as wavelet_cuda


@pytest.fixture(scope="module")
def backend():
    if os.environ.get("RUN_DENOISE_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_DENOISE_CUDA_TESTS=1 to authorize bounded CUDA tests")
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA unavailable: {exc}")
    if not count:
        pytest.skip("CUDA unavailable")
    return wavelet_cuda.WaveletBackend()


def reference(raw):
    from skimage.restoration import denoise_wavelet

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        floats = np.stack(
            [
                denoise_wavelet(p, wavelet="db4", mode="soft", method="BayesShrink", rescale_sigma=True)
                for p in raw.astype(np.float32) / 255.0
            ]
        )
        output = np.clip(floats * 255.0, 0, 255).astype(np.uint8)
    return floats, output


def test_import_is_cuda_lazy():
    subprocess.run(
        [sys.executable, "-c", "import sys; import scripts.denoise_wavelet_cuda; assert 'cupy' not in sys.modules"],
        check=True,
        capture_output=True,
        text=True,
    )


def test_no_cpu_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        wavelet_cuda.WaveletBackend()


@pytest.mark.parametrize("shape", [(2, 1, 1), (2, 3, 5), (2, 17, 19), (2, 199, 201), (2, 200, 200), (2, 257, 259)])
def test_shape_boundary_and_multilevel_parity(backend, shape):
    raw = np.random.default_rng(19).integers(0, 256, shape, dtype=np.uint8)
    before = raw.copy()
    floats, expected = reference(raw)
    actual_float = backend.cp.asnumpy(backend.filtered_float(raw))
    output = backend(raw)
    assert isinstance(output, backend.cp.ndarray)
    assert output.dtype == backend.cp.uint8
    assert actual_float.dtype == np.float32
    np.testing.assert_array_equal(backend.cp.asnumpy(output), expected)
    np.testing.assert_allclose(actual_float, floats, rtol=0, atol=1e-7, equal_nan=True)
    np.testing.assert_array_equal(raw, before)


def test_validation_and_metadata(backend):
    report = backend.validate()
    assert report["passed"], json.dumps(report, indent=2)
    assert backend.metadata["validated"] is True
    assert backend.metadata["validation"] == report
    assert (
        backend.metadata["implementation_sha256"]
        == hashlib.sha256(Path(wavelet_cuda.__file__).read_bytes()).hexdigest()
    )
    assert backend.metadata["parameters"]["wavelet_levels"] is None
    assert backend.name == "wavelet"
    assert all(k.options == ("--fmad=false",) for k in backend.kernels.values())
    json.dumps(backend.metadata)


def test_validation_never_accepts_one_lsb(backend, monkeypatch):
    original = type(backend).__call__

    def altered(self, raw):
        result = original(self, raw)
        result[0, 0, 0] ^= self.cp.uint8(1)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(type(backend), "__call__", altered)
        report = backend.validate()
        assert not report["passed"]
        assert not backend.metadata["validated"]
        assert report["cases"]["noise"]["max_abs_error"] == 1


def test_gpu_only_filtering(backend, monkeypatch):
    import pywt
    import skimage.restoration

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU transform/filter or device-to-host pixel transfer")

    with monkeypatch.context() as patch:
        for name in ("dwt", "idwt", "dwtn", "idwtn", "wavedecn", "waverecn"):
            patch.setattr(pywt, name, forbidden)
        patch.setattr(skimage.restoration, "denoise_wavelet", forbidden)
        patch.setattr(backend.cp, "asnumpy", forbidden)
        raw = np.random.default_rng(7).integers(0, 256, (2, 200, 200), dtype=np.uint8)
        result = backend(raw)
        assert result.shape == raw.shape
    assert np.any(backend.cp.asnumpy(result) != raw)


def test_impulses_noncontiguous_source(backend):
    raw = np.zeros((2, 200, 200), np.uint8)
    raw[0, 0, 0] = 255
    raw[0, -1, -1] = 127
    raw[1, 100, 100] = 255
    raw = raw[:, ::-1, :].transpose(0, 2, 1)
    assert not raw.flags.c_contiguous
    floats, expected = reference(raw)
    np.testing.assert_array_equal(backend.cp.asnumpy(backend(raw)), expected)
    np.testing.assert_allclose(
        backend.cp.asnumpy(backend.filtered_float(raw)), floats, rtol=0, atol=1e-7, equal_nan=True
    )


@pytest.mark.parametrize(
    "raw,error",
    [
        (np.zeros((2, 3, 4), np.float32), TypeError),
        (np.zeros((2, 3), np.uint8), ValueError),
        (np.zeros((0, 3, 4), np.uint8), ValueError),
    ],
)
def test_input_contract(backend, raw, error):
    with pytest.raises(error):
        backend(raw)


@pytest.mark.parametrize("size", [7, 128, 129, 255, 256, 257, 8191, 8192, 10609])
def test_numpy_float32_mean_order(backend, size):
    cp = backend.cp
    values = np.random.default_rng(42).standard_normal(size).astype(np.float32)
    x = cp.asarray(values)
    chunks = (size + 8191) // 8192
    sums = cp.empty(chunks, cp.float32)
    out = cp.empty((), cp.float64)
    backend.kernels["mean_blocks"]((chunks,), (128,), (x, sums, np.int32(size)))
    backend.kernels["mean_finish"]((1,), (1,), (sums, out, np.int32(chunks), np.int32(size)))
    assert cp.asnumpy(out) == np.mean(values * values)


def test_full_training_volume(backend):
    path = Path(__file__).resolve().parents[1] / "data/denoise_benchmark_raw/Training_volumes.npy"
    if not path.exists():
        pytest.skip("Local benchmark source not present; never download in tests")
    source = np.load(path, mmap_mode="r")
    raw = source[0]
    if raw.shape == (1, 200, 200, 200):
        raw = raw[0]
    assert raw.shape == (200, 200, 200)
    report = backend.validate(raw)
    assert report["passed"], json.dumps(report)
    start = time.perf_counter()
    actual = backend.cp.asnumpy(backend(raw))
    gpu_seconds = time.perf_counter() - start
    start = time.perf_counter()
    floats, expected = reference(raw)
    cpu_seconds = time.perf_counter() - start
    actual_float = backend.cp.asnumpy(backend.filtered_float(raw))
    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    float_error = np.abs(actual_float - floats)
    print(
        json.dumps(
            {
                "source": "Training[0]",
                "shape": list(raw.shape),
                "gpu_seconds_including_transfer": gpu_seconds,
                "cpu_seconds": cpu_seconds,
                "maxdiff": int(diff.max()),
                "mismatched_pixels": int(np.count_nonzero(diff)),
                "float_max_abs_error": float(np.nanmax(float_error)),
                "cupy_pool_bytes": backend.cp.get_default_memory_pool().total_bytes(),
            }
        )
    )
    assert backend.cp.get_default_memory_pool().total_bytes() < 1024**3
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(actual_float, floats, rtol=0, atol=1e-7, equal_nan=True)
