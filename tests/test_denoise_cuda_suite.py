"""CPU contract tests; opt-in CUDA tests with RUN_DENOISE_CUDA_TESTS=1.

The opt-in gate lets the experiment orchestrator avoid competing GPU jobs.
No tests access credentials, datasets, networks, or artifact stores.
"""

import builtins
import importlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import denoise_cuda_suite as suite


@pytest.fixture
def cp():
    if os.environ.get("RUN_DENOISE_CUDA_TESTS") != "1":
        pytest.skip("CUDA tests require orchestrator opt-in: RUN_DENOISE_CUDA_TESTS=1")
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")
        cupy.zeros(1)
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA unavailable: {exc}")
    return cupy


@pytest.fixture(params=["numpy", "cuda"])
def metric_engine(request, monkeypatch):
    if request.param == "cuda":
        return request.getfixturevalue("cp")
    monkeypatch.setattr(suite, "_cupy", lambda: np)
    monkeypatch.setattr(suite, "_require_memory", lambda cp: None)
    return np


def test_lazy_import(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"cupy", "cupyx", "torch", "vapoursynth"}:
            raise AssertionError(f"Eager GPU import: {name}")
        return original(name, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", guarded)
        importlib.reload(suite)
    json.dumps(suite.metric_protocol, allow_nan=False)


@pytest.mark.parametrize("name", ["nlm", "wavelet"])
def test_pending_backends(name, monkeypatch):
    monkeypatch.setattr(suite, "_cupy", lambda: pytest.fail("Pending methods must not initialize CUDA"))
    with pytest.raises(NotImplementedError, match="pending"):
        suite.get_backend(name)


def test_unknown_backend():
    with pytest.raises(ValueError, match="Unknown"):
        suite.get_method("not-a-method")


def test_no_cpu_fallback(monkeypatch):
    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        suite.get_backend("gaussian")


def test_memory_guard():
    cp = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=lambda: (1024**3 - 1, 4 * 1024**3))))
    with pytest.raises(RuntimeError, match="1 GiB"):
        suite._require_memory(cp)


def test_plugin_hash_rejected_before_import(tmp_path, monkeypatch):
    dll = tmp_path / "bm3dcuda.dll"
    dll.write_bytes(b"not the authorized plugin")
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "vapoursynth":
            pytest.fail("Hash must be verified before plugin import/load")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        suite._load_bm3d(dll)


@pytest.mark.parametrize("shape,dtype", [((2, 3), np.uint8), ((0, 3, 3), np.uint8), ((2, 3, 3), np.float32)])
def test_invalid_volume(shape, dtype):
    with pytest.raises(ValueError, match="uint8"):
        suite.MetricContext(np.zeros(shape, dtype=dtype))


def test_metric_axes_checked_without_cuda():
    raw = np.zeros((2, 3, 4), np.uint8)
    with pytest.raises(ValueError, match="slice_axis"):
        suite.MetricContext(raw, slice_axis=1)
    with pytest.raises(ValueError, match="depth_axis"):
        suite.MetricContext(raw, depth_axis=3)
    with pytest.raises(ValueError, match="two pixels"):
        suite.MetricContext(raw[:, :1])


def test_failed_validation_does_not_advertise_parity():
    class IncorrectBackend:
        name = "original"
        cp = SimpleNamespace(asnumpy=np.asarray)
        metadata = {"validated": True}
        last_seconds = 0.001

        def __call__(self, raw):
            return np.zeros_like(raw)

    backend = IncorrectBackend()
    result = suite.validate_backend(backend)
    assert result["passed"] is False
    assert result["cpu_equivalent"] is False
    assert backend.metadata["validated"] is False
    assert result["cases"]["random"]["mismatched_pixels"] > 0


def test_tv_stopping_cpu_array_algebra():
    from skimage.restoration import denoise_tv_chambolle

    rng = np.random.default_rng(13)
    x = rng.random((33, 7, 9), dtype=np.float32)
    x[0].fill(0)
    x[1] *= 0.005
    expected = np.stack([denoise_tv_chambolle(p) for p in x])
    actual = suite._tv_chambolle(x, np)
    np.testing.assert_array_equal(actual, expected)


def _reference_scores(raw, den, mask, raw_gradient, den_gradient, edge_mask):
    record = {"pixel_count": raw.size}
    for suffix, selection in (("s", mask), ("b", ~mask)):
        selected = den[selection].astype(np.float64)
        record[f"n_{suffix}"] = selected.size
        record[f"mu_{suffix}"] = float(selected.mean()) if selected.size else None
        record[f"sigma_{suffix}"] = float(selected.std()) if selected.size else None
    ms, ss, mb, sb = (record[k] for k in ("mu_s", "sigma_s", "mu_b", "sigma_b"))

    def divide(a, b):
        return None if a is None or b is None or b == 0 else a / b

    contrast = None if ms is None or mb is None else ms - mb
    record.update(
        SNR=divide(ms, ss),
        SNR_bg=divide(ms, sb),
        CNR=divide(contrast, None if ss is None or sb is None else np.hypot(ss, sb)),
        CNR_bg=divide(contrast, sb),
        bg_sigma=sb,
    )
    record["ENL"] = None if ss is None or ss == 0 else (ms / ss) ** 2
    record["ENL_bg"] = None if sb is None or sb == 0 else (mb / sb) ** 2
    a = raw_gradient[edge_mask].astype(np.float64)
    b = den_gradient[edge_mask].astype(np.float64)
    record["beta_count"] = a.size
    record["beta"] = float(np.corrcoef(a, b)[0, 1]) if a.size >= 3 and a.std() > 0 and b.std() > 0 else None
    record["gradient_magnitude_ratio"] = divide(float(b.sum()), float(a.sum()))
    residual = raw.astype(np.float64) - den
    record.update(
        residual_rms=float(np.sqrt(np.mean(residual**2))),
        residual_mean=float(residual.mean()),
        residual_std=float(residual.std()),
        saturation_zero=float(np.mean(den == 0)),
        saturation_255=float(np.mean(den == 255)),
    )
    return record


def _reference(raw, den):
    from skimage.filters import threshold_otsu

    threshold = int(threshold_otsu(raw))
    mask = raw > threshold
    gradients = []
    for image in (raw, den):
        dy, dx = np.gradient(image.astype(np.float32), axis=(1, 2))
        gradients.append(np.sqrt(dy * dy + dx * dx))
    raw_g, den_g = gradients
    edges = raw_g > np.percentile(raw_g, 85, axis=(1, 2))[:, None, None]
    global_metrics = _reference_scores(raw, den, mask, raw_g, den_g, edges)
    global_metrics["otsu_threshold"] = threshold
    rows = [_reference_scores(raw[i], den[i], mask[i], raw_g[i], den_g[i], edges[i]) for i in range(len(raw))]
    return global_metrics, rows


def _assert_metrics(actual, expected):
    json.dumps(actual, allow_nan=False)
    for key, value in expected.items():
        if value is None:
            assert actual[key] is None, key
            assert key in actual["invalid_reasons"]
        else:
            assert actual[key] == pytest.approx(value, rel=2e-6, abs=2e-6), key


@pytest.mark.parametrize("case", ["random", "zero", "constant", "edge", "original", "flattened", "mixed"])
def test_metrics_against_independent_numpy(metric_engine, case):
    cp = metric_engine
    rng = np.random.default_rng(917)
    raw = rng.integers(0, 256, (3, 19, 23), dtype=np.uint8)
    den = np.clip(raw.astype(np.int16) // 2 + 19, 0, 255).astype(np.uint8)
    if case == "zero":
        raw.fill(0)
        den.fill(0)
    elif case == "constant":
        raw.fill(127)
        den.fill(126)
    elif case == "edge":
        raw.fill(0)
        raw[:, :, 11:] = 255
        den = raw.copy()
    elif case == "original":
        den = raw.copy()
    elif case == "flattened":
        den.fill(127)
    elif case == "mixed":
        raw[0].fill(0)
        raw[1].fill(255)
        den = raw.copy()
    context = suite.MetricContext(raw)
    actual, rows = context.evaluate(cp.asarray(den))
    expected, expected_rows = _reference(raw, den)
    _assert_metrics(actual, expected)
    slow = context._scores(
        context.raw,
        cp.asarray(den, dtype=cp.float32),
        context.signal,
        context.gradient,
        suite._gradient(cp.asarray(den, dtype=cp.float32), cp),
        context.edges,
    )
    _assert_metrics(slow, {key: value for key, value in expected.items() if key != "otsu_threshold"})
    for i, (row, expected_row) in enumerate(zip(rows, expected_rows, strict=True)):
        _assert_metrics(row, expected_row)
        assert row["slice_index"] == i
    expected_depth = int(
        np.argmax([raw.astype(np.float32).mean(axis=tuple(i for i in range(3) if i != a)).std() for a in range(3)])
    )
    assert actual["depth_axis"] == expected_depth
    if case == "original":
        assert actual["beta"] == pytest.approx(1.0)
        assert actual["residual_rms"] == 0
    cached_mask = np.asarray(context.signal) if cp is np else cp.asnumpy(context.signal)
    second, _ = context.evaluate(cp.zeros_like(cp.asarray(den)))
    assert second["n_s"] == actual["n_s"]
    current_mask = np.asarray(context.signal) if cp is np else cp.asnumpy(context.signal)
    np.testing.assert_array_equal(current_mask, cached_mask)


def test_background_enl_uses_background_mean(metric_engine):
    cp = metric_engine
    raw = np.tile(np.array([10, 20, 200, 220], np.uint8), (3, 5, 1))
    context = suite.MetricContext(raw)
    actual, rows = context.evaluate(cp.asarray(raw))
    assert suite.metric_protocol["ENL_bg"] == "(mu_b / sigma_b) ** 2"
    for row in [actual, *rows]:
        assert row["ENL_bg"] == pytest.approx(9.0)
        assert row["SNR_bg"] ** 2 == pytest.approx(1764.0)


def test_evaluate_transfers_statistics_once(cp, monkeypatch):
    raw = np.random.default_rng(55).integers(0, 256, (9, 17, 19), dtype=np.uint8)
    context = suite.MetricContext(raw)
    original = cp.asnumpy
    transfers = []

    def counted(array, *args, **kwargs):
        transfers.append(array.shape)
        return original(array, *args, **kwargs)

    monkeypatch.setattr(cp, "asnumpy", counted)
    monkeypatch.setattr(context, "_scores", lambda *args: pytest.fail("No per-slice GPU scalar path"))
    context.evaluate(cp.asarray(raw))
    assert transfers == [(10, 16)]


def test_bscan_gradient_not_through_slice(metric_engine):
    cp = metric_engine
    raw = np.stack([np.full((7, 9), v, np.uint8) for v in (0, 100, 255)])
    metrics, _ = suite.volume_scores(raw, cp.asarray(raw), depth_axis=1)
    assert metrics["beta_count"] == 0
    assert metrics["beta"] is None
    assert metrics["depth_axis"] == 1


@pytest.mark.parametrize("name", ["original", "gaussian", "median", "tv", "bilateral"])
def test_backend_parity(cp, name):
    backend = suite.get_backend(name)
    result = suite.validate_backend(backend)
    assert result["passed"], result
    assert result["cpu_equivalent"]
    assert backend.metadata["validated"]
    assert len(backend.metadata["implementation_sha256"]) == 64
    raw = np.full((2, 7, 9), 127, np.uint8)
    out = backend(raw)
    assert isinstance(out, cp.ndarray)
    assert out.dtype == cp.uint8
    assert backend.last_seconds > 0


def test_tv_batched_independent_stopping(cp):
    from skimage.restoration import denoise_tv_chambolle

    rng = np.random.default_rng(117)
    x = rng.random((5, 17, 19), dtype=np.float32)
    x[0].fill(0)
    x[1] *= 0.005
    x[2, :, :9] = 0
    expected = np.stack([denoise_tv_chambolle(plane) for plane in x])
    actual = cp.asnumpy(suite._tv_chambolle(cp.asarray(x), cp))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)
    alone = cp.asnumpy(suite._tv_chambolle(cp.asarray(x[1:2]), cp))
    np.testing.assert_array_equal(actual[1:2], alone)


def test_bm3d_explicit_opt_in(cp):
    if os.environ.get("RUN_BM3D_CUDA_TESTS") != "1":
        pytest.skip("Plugin execution requires RUN_BM3D_CUDA_TESTS=1 and verified R79/R2.15 installation")
    backend = suite.get_backend("bm3d")
    validation = suite.validate_backend(backend)
    assert validation["passed"], validation
    assert not validation["cpu_equivalent"]
    assert validation["kind"] == "execution-only-new-variant"
    assert backend.metadata["plugin_sha256"] == suite.BM3D_SHA256
