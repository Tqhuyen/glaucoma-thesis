import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


def main():
    source = Path(tempfile.gettempdir()) / "gf_vol_cache/raw_2404.npy"
    output = Path("figures/oct_sample_2404/solid")
    raw = np.load(source, mmap_mode="r", allow_pickle=False)
    data = raw.transpose(0, 2, 1)
    grid = pv.ImageData(dimensions=np.array(data.shape) + 1)
    grid.cell_data["intensity"] = data.ravel(order="F")
    surface = grid.extract_surface(algorithm="dataset_surface")
    centers = np.floor(surface.cell_centers().points).astype(int)
    centers = np.clip(centers, 0, np.array(data.shape) - 1)
    expected = data[tuple(centers.T)]
    np.testing.assert_array_equal(surface.cell_data["intensity"], expected)
    left = data[-1, :, :].T
    right = data[:, 0, :].T
    np.testing.assert_array_equal(left[:, 0], right[:, -1])
    width = 32
    strip = np.concatenate([left[:, :width][:, ::-1], right[:, -width:-1][:, ::-1]], axis=1)
    fig, axs = plt.subplots(1, 3, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1, 0.6]}, layout="constrained")
    for ax, pixels, title in zip(
        axs,
        [left[:, ::-1], right[:, ::-1], strip],
        [
            "Left face: X = 199\nY decreases toward the shared edge",
            "Right face: Y = 0\nX decreases away from the shared edge",
            "Unfolded edge\n32 columns per face",
        ],
    ):
        ax.imshow(pixels, cmap="gray", vmin=0, vmax=255, interpolation="nearest", aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Z: depth index")
        ax.set_xlabel("Display column")
    axs[0].axvline(left.shape[1] - 1, color="#c58112", linewidth=1)
    axs[1].axvline(0, color="#c58112", linewidth=1)
    axs[2].axvline(width - 1, color="#c58112", linewidth=0.7, linestyle="--")
    fig.suptitle("Shared-edge check: original OCT voxel intensities", fontsize=15)
    fig.savefig(output / "edge_diagnostic.png", dpi=220)
    plt.close(fig)
    report = {
        "source": str(source),
        "visible_faces_xyz": ["X=max", "Y=0"],
        "surface_cells_verified": int(surface.n_cells),
        "surface_mapping_exact": True,
        "shared_edge_equal": True,
        "shared_edge_max_difference": int(np.abs(left[:, 0].astype(int) - right[:, -1]).max()),
        "unfolded_edge_column": width - 1,
        "adjacent_y_mean_absolute_difference": float(np.abs(left[:, 0].astype(float) - left[:, 1]).mean()),
        "adjacent_x_mean_absolute_difference": float(np.abs(right[:, -1].astype(float) - right[:, -2]).mean()),
    }
    (output / "edge_diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
