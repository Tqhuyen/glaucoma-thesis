import numpy as np

from scripts import denoise_torch as dt


def test_bilateral_volume_cpu_preserves_shape_and_smooths():
    rng = np.random.default_rng(0)
    clean = np.full((2, 32, 32), 128, dtype=np.uint8)
    noisy = np.clip(clean.astype(np.int16) + rng.integers(-40, 41, clean.shape), 0, 255).astype(np.uint8)
    out, seconds = dt.bilateral_volume(noisy, sigma_color=0.1, sigma_spatial=2.0, device="cpu", batch=2)
    assert out.shape == noisy.shape
    assert out.dtype == np.uint8
    assert out.std() < noisy.std()
    assert seconds >= 0.0


def test_bilateral_volume_constant_image_is_stable():
    image = np.full((1, 16, 16), 77, dtype=np.uint8)
    out, _ = dt.bilateral_volume(image, sigma_color=0.1, sigma_spatial=1.0, device="cpu", batch=1)
    assert np.all(np.abs(out.astype(np.int16) - 77) <= 1)
