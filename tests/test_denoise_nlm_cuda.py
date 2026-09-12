"""Standalone NLM contract and opt-in real CUDA parity checks; no downloads."""

import builtins
import hashlib
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts import denoise_nlm_cuda as nlm


@pytest.fixture(scope="module")
def backend():
    if os.environ.get("RUN_DENOISE_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_DENOISE_CUDA_TESTS=1 to authorize bounded CUDA tests")
    return nlm.NLMBackend()


@pytest.mark.parametrize("raw", [np.zeros((2, 3), np.uint8), np.zeros((1, 2, 3)), np.zeros((0, 2, 3), np.uint8)])
def test_input_shape_dtype(raw):
    with pytest.raises(ValueError):
        nlm._check_volume(raw)


def test_input_numpy_only():
    with pytest.raises(TypeError):
        nlm._check_volume([[[1]]])


def test_lazy_import_and_no_cpu_fallback(monkeypatch):
    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("CUDA unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    importlib.reload(nlm)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        nlm.NLMBackend()


@pytest.mark.parametrize("batch", [0, 9, True, 1.5])
def test_invalid_batch(batch):
    with pytest.raises(ValueError):
        nlm.NLMBackend(batch_planes=batch)


def test_validation(backend):
    report = backend.validate()
    print(json.dumps(report, indent=2))
    assert report["passed"]
    assert not report["dataset_run_ready"]
    assert report["peak_pool_bytes"] < 1024**3
    assert backend.metadata["implementation_sha256"] == hashlib.sha256(Path(nlm.__file__).read_bytes()).hexdigest()
    json.dumps(backend.metadata, allow_nan=False)


def test_batch_order_and_quantization(backend):
    raw = np.random.default_rng(10).integers(0, 256, (9, 17, 21), dtype=np.uint8)
    original = raw.copy()
    result = backend(raw)
    assert isinstance(result, backend.cp.ndarray)
    assert result.shape == raw.shape and result.dtype == np.uint8
    combined = backend.cp.asnumpy(result)
    separate = np.concatenate([backend.cp.asnumpy(backend(raw[i : i + 1])) for i in range(len(raw))])
    np.testing.assert_array_equal(combined, separate)
    floats = backend.cp.asnumpy(backend.filtered_float(raw))
    np.testing.assert_array_equal(combined, np.clip(floats * 255, 0, 255).astype(np.uint8))
    np.testing.assert_array_equal(raw, original)
    from skimage.restoration import denoise_nl_means

    reversed_raw = raw[:, ::-1, ::-1]
    expected = np.stack([denoise_nl_means(p.astype(np.float32) / 255, **nlm.PARAMETERS) for p in reversed_raw])
    np.testing.assert_array_equal(
        backend.cp.asnumpy(backend(reversed_raw)), np.clip(expected * 255, 0, 255).astype(np.uint8)
    )


def test_production_does_not_call_cpu_reference(backend, monkeypatch):
    import skimage.restoration

    def forbidden(*args, **kwargs):
        pytest.fail("CPU filter called in production")

    monkeypatch.setattr(skimage.restoration, "denoise_nl_means", forbidden)
    result = backend(np.full((2, 9, 11), 127, np.uint8))
    np.testing.assert_array_equal(backend.cp.asnumpy(result), 127)


def test_memory_budget_rejects_before_allocation(backend):
    raw = np.broadcast_to(np.uint8(0), (1, 10000, 10000))
    with pytest.raises(MemoryError, match="budget"):
        backend(raw)


def test_validation_rejects_one_byte_difference(backend, monkeypatch):
    original = nlm.NLMBackend.__call__

    def corrupted(self, raw, *, filtered_float=False):
        output = original(self, raw, filtered_float=filtered_float)
        if not filtered_float:
            output.flat[0] ^= self.cp.uint8(1)
        return output

    monkeypatch.setattr(nlm.NLMBackend, "__call__", corrupted)
    report = backend.validate()
    assert not report["passed"]
    assert not report["dataset_run_ready"]
    assert not backend.metadata["validated"]
    assert all(case["mismatched_pixels"] == 1 for case in report["cases"].values())


def test_training_volume(backend):
    path = os.environ.get("NLM_OCT_SAMPLE")
    if not path:
        pytest.skip("Set NLM_OCT_SAMPLE to a local raw Training .npy for full-volume validation")
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    raw = data if data.shape == (200, 200, 200) else data[0].reshape(200, 200, 200)
    assert raw.shape == (200, 200, 200)
    report = backend.validate(raw)
    print(json.dumps(report, indent=2))
    assert report["passed"] and report["dataset_run_ready"]
