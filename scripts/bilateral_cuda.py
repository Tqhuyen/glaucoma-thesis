"""CUDA port of the grayscale scikit-image 0.26 bilateral calculation.

The canonical LUTs are constructed by scikit-image on the CPU. Accumulation
order, constant padding, float32 rounding, and uint8 truncation are retained.
Reference: scikit-image v0.26.0 src/skimage/restoration/_denoise_cy.pyx.
"""

import numpy as np

KERNEL = r'''
extern "C" __global__ void bilateral_u8(
    const unsigned char* input, unsigned char* output,
    const float* normal, const float* colors, const float* spatial,
    const unsigned char* maximum, const unsigned char* constant,
    const int planes, const int height, const int width) {
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    const int area = height * width;
    if (index >= planes * area) return;
    const int plane = index / area;
    const int position = index - plane * area;
    const int row = position / width;
    const int col = position - row * width;
    const int center = input[index];
    if (constant[plane]) {
        output[index] = (unsigned char)(normal[center] * 255.0f);
        return;
    }
    const float* lut = colors + maximum[plane] * 65536 + center * 256;
    float weighted = 0.0f;
    float weights = 0.0f;
    int k = 0;
    for (int dy = -12; dy <= 12; ++dy) {
        const int rr = row + dy;
        for (int dx = -12; dx <= 12; ++dx, ++k) {
            const int cc = col + dx;
            const int value = (rr < 0 || rr >= height || cc < 0 || cc >= width)
                ? 0 : input[plane * area + rr * width + cc];
            const float weight = spatial[k] * lut[value];
            weighted = weighted + normal[value] * weight;
            weights = weights + weight;
        }
    }
    const float result = (weighted / weights) * 255.0f;
    output[index] = (unsigned char)fminf(255.0f, fmaxf(0.0f, result));
}
'''


class CudaBilateral:
    def __init__(self):
        import cupy as cp
        import skimage
        from skimage.restoration._denoise import _compute_color_lut, _compute_spatial_lut

        if skimage.__version__ != "0.26.0":
            raise RuntimeError("CUDA compatibility must be revalidated for this scikit-image version")
        self.cp = cp
        normal = np.arange(256, dtype=np.float32) / 255.0
        difference = normal[:, None] - normal[None, :]
        distance = np.sqrt((difference * difference).astype(np.float64)).astype(np.float32)
        colors = np.empty((256, 256, 256), dtype=np.float32)
        colors[0].fill(0)
        for value in range(1, 256):
            lut = _compute_color_lut(10000, 0.1, normal[value], dtype=np.float32)
            scale = np.float32(10000.0 / float(normal[value]))
            bins = np.minimum((distance * scale).astype(np.int64), 9999)
            colors[value] = lut[bins]
        self.normal = cp.asarray(normal)
        self.colors = cp.asarray(colors)
        self.spatial = cp.asarray(_compute_spatial_lut(25, 4.0, dtype=np.float32))
        self.kernel = cp.RawKernel(KERNEL, "bilateral_u8", options=("--fmad=false", "--prec-div=true", "--ftz=false"))
        self.kernel.compile()

    def __call__(self, volume):
        if volume.dtype != np.uint8 or volume.ndim != 3:
            raise ValueError("Expected a uint8 grayscale volume, with slice axis 0")
        cp = self.cp
        volume = np.ascontiguousarray(volume)
        maxima = volume.max(axis=(1, 2))
        constant = (volume.min(axis=(1, 2)) == maxima).astype(np.uint8)
        inputs = cp.asarray(volume)
        outputs = cp.empty_like(inputs)
        self.kernel(((volume.size + 127) // 128,), (128,),
                    (inputs, outputs, self.normal, self.colors, self.spatial,
                     cp.asarray(maxima), cp.asarray(constant), *map(np.int32, volume.shape)))
        return cp.asnumpy(outputs)


def validate():
    import time
    from scripts.prepare_bilateral_200 import bilateral

    backend = CudaBilateral()
    rng = np.random.default_rng(42)
    cases = [np.full((1, 17, 19), value, dtype=np.uint8) for value in (0, 1, 13, 127, 255)]
    cases += [rng.integers(0, high, (3, 17, 19), dtype=np.uint8) for high in (2, 17, 64, 128, 256)]
    cases += [np.tile(np.arange(200, dtype=np.uint8), (2, 200, 1))]
    for index, case in enumerate(cases):
        expected = bilateral(case)
        began = time.perf_counter()
        result = backend(case)
        mismatch = np.count_nonzero(expected != result)
        print(f"case={index} shape={case.shape} mismatched_uint8={mismatch} gpu_s={time.perf_counter()-began:.4f}", flush=True)
        np.testing.assert_array_equal(result, expected)
    print("All CUDA/CPU uint8 equivalence checks passed.")


if __name__ == "__main__":
    validate()
