import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.patches import Arc, Circle, ConnectionPatch, Ellipse, Polygon, Rectangle


def main():
    parser = argparse.ArgumentParser(description="Opaque OCT voxel block with original boundary intensities.")
    parser.add_argument("--volume", type=Path, default=Path(tempfile.gettempdir()) / "gf_vol_cache/raw_2404.npy")
    parser.add_argument("--output", type=Path, default=Path("figures/oct_sample_2404/solid"))
    args = parser.parse_args()
    raw = np.load(args.volume, mmap_mode="r", allow_pickle=False)
    if raw.ndim != 3 or raw.dtype != np.uint8:
        parser.error("Expected a single 3D uint8 .npy volume")
    data = raw.transpose(0, 2, 1)
    args.output.mkdir(parents=True, exist_ok=True)
    grid = pv.ImageData(dimensions=np.array(data.shape) + 1, spacing=(1, 1, 1))
    grid.cell_data["intensity"] = data.ravel(order="F")
    np.testing.assert_array_equal(grid["intensity"].reshape(data.shape, order="F"), data)
    plotter = pv.Plotter(off_screen=True, window_size=(2000, 1800))
    plotter.set_background("white")
    plotter.add_mesh(
        grid.extract_surface(algorithm="dataset_surface"),
        scalars="intensity",
        preference="cell",
        cmap="gray",
        clim=(0, 255),
        opacity=1,
        lighting=False,
        show_edges=False,
        show_scalar_bar=False,
    )
    plotter.add_mesh(grid.outline(), color="#525252", line_width=1)
    size = np.array(data.shape, dtype=float)
    center = size / 2
    plotter.camera_position = [
        (center + np.array([2.6, -3.0, -1.8]) * max(size)).tolist(),
        center.tolist(),
        [0, 0, -1],
    ]
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = max(size) * 0.86
    plotter.add_axes(viewport=(0.80, 0.01, 0.99, 0.20), color="#303030", xlabel="X", ylabel="Y", zlabel="Z")
    plotter.show(auto_close=False, interactive=False)
    image = plotter.screenshot(args.output / "oct_solid_block.png")
    if np.mean(image[:, :, :3].mean(axis=2) < 200) < 0.1:
        raise RuntimeError("Expected an opaque visible voxel block")
    camera = {
        "position": list(plotter.camera.position),
        "focal_point": list(plotter.camera.focal_point),
        "view_up": list(plotter.camera.up),
        "parallel_scale": plotter.camera.parallel_scale,
    }
    positions = [data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 6]
    x, y, z = np.array(positions, dtype=float) + 0.5
    sx, sy, sz = size
    colors = ["#1678ae", "#ba463b", "#a67608"]
    traces = [
        [[x, 0, sz], [x, 0, 0], [x, sy, 0]],
        [[sx, y, sz], [sx, y, 0], [0, y, 0]],
        [[0, 0, z], [sx, 0, z], [sx, sy, z]],
    ]
    targets = [[x, 0, sz * 0.25], [sx, y, sz * 0.65], [sx * 0.9, 0, z]]
    for trace, color in zip(traces, colors):
        plotter.add_lines(np.asarray(trace), color=color, width=5, connected=True)
    plotter.render()
    marked = plotter.screenshot(args.output / "oct_solid_slice_locations.png")
    projected = []
    for target in targets:
        plotter.renderer.SetWorldPoint(*target, 1)
        plotter.renderer.WorldToDisplay()
        px, py, _ = plotter.renderer.GetDisplayPoint()
        if not (0 <= px < marked.shape[1] and 0 <= py < marked.shape[0]):
            raise RuntimeError("Slice location lies outside the rendered image")
        projected.append((px, marked.shape[0] - 1 - py))
    plotter.close()
    fig, ax = plt.subplots(figsize=(10, 9))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(image)
    ax.set_axis_off()
    fig.savefig(args.output / "oct_solid_block.pdf", dpi=300)
    plt.close(fig)
    fig = plt.figure(figsize=(15, 9), facecolor="white")
    fig.text(0.04, 0.95, "From the retinal scan region to OCT slices", fontsize=21)
    eye = fig.add_axes((0.025, 0.35, 0.33, 0.53))
    eye.set(xlim=(-1.7, 1.8), ylim=(-1.45, 1.5), aspect="equal")
    eye.set_axis_off()
    eye.add_patch(
        Polygon(
            [[0.86, -0.05], [1.57, -0.39], [1.49, -0.69], [0.81, -0.35]],
            facecolor="#eccbad",
            edgecolor="#9f8067",
            linewidth=1.4,
        )
    )
    eye.add_patch(Circle((0, 0), 1.05, facecolor="#fffaf1", edgecolor="#9b8170", linewidth=2))
    eye.add_patch(Arc((0, 0), 1.96, 1.96, theta1=-105, theta2=105, color="#c57b64", linewidth=5))
    eye.add_patch(Ellipse((-0.93, 0), 0.51, 1.05, facecolor="#e9f4f7", edgecolor="#6b98a9", linewidth=1.5))
    eye.add_patch(Ellipse((-0.60, 0), 0.30, 0.68, facecolor="#c8e4ec", edgecolor="#6b98a9", linewidth=1.2))
    eye.plot([-0.80, -0.80], [0.17, 0.42], color="#628498", linewidth=4)
    eye.plot([-0.80, -0.80], [-0.17, -0.42], color="#628498", linewidth=4)
    eye.add_patch(
        Polygon(
            [[-1.47, 0.02], [-0.65, 0.02], [0.96, 0.08], [0.85, -0.49], [-0.65, 0.02]],
            facecolor="#41aa9b",
            alpha=0.10,
            edgecolor="none",
        )
    )
    eye.plot([-1.48, -0.65, 0.96], [0.02, 0.02, -0.19], color="#369082", linestyle="--", linewidth=1.2)
    eye.add_patch(
        Rectangle((0.80, -0.48), 0.29, 0.57, facecolor="#51b1a0", alpha=0.22, edgecolor="#187f70", linewidth=2)
    )
    eye.annotate(
        "Cornea", xy=(-1.17, 0.1), xytext=(-1.58, 0.91), fontsize=11, arrowprops={"arrowstyle": "-", "color": "#596269"}
    )
    eye.annotate(
        "Lens", xy=(-0.60, 0.16), xytext=(-0.68, 1.19), fontsize=11, arrowprops={"arrowstyle": "-", "color": "#596269"}
    )
    eye.annotate(
        "Retina", xy=(0.61, 0.76), xytext=(0.58, 1.18), fontsize=11, arrowprops={"arrowstyle": "-", "color": "#596269"}
    )
    eye.annotate(
        "Optic nerve",
        xy=(1.29, -0.41),
        xytext=(0.70, -1.19),
        fontsize=11,
        arrowprops={"arrowstyle": "-", "color": "#596269"},
    )
    eye.text(-1.57, -0.24, "OCT beam", fontsize=10, color="#23766a")
    eye.text(0.19, -0.79, "Illustrative scan region\nnear the optic nerve head", fontsize=10, color="#187f70")
    fig.text(0.04, 0.87, "(a) Location in the eye", fontsize=14)
    block = fig.add_axes((0.34, 0.29, 0.40, 0.58))
    block.imshow(marked)
    block.set_axis_off()
    fig.text(0.39, 0.87, "(b) Acquired OCT voxel block", fontsize=14)
    fig.add_artist(
        ConnectionPatch(
            xyA=(1.09, -0.19),
            coordsA=eye.transData,
            xyB=(0.10, 0.56),
            coordsB=block.transAxes,
            color="#187f70",
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.5,
        )
    )
    fig.text(0.39, 0.27, "X, Y: lateral scan coordinates\nZ: depth into the retinal tissue", fontsize=11)
    fig.text(0.77, 0.91, "(c) Sections of the same block", fontsize=13)
    slice_arrays = [data[positions[0], :, :].T, data[:, positions[1], :].T, data[:, :, positions[2]]]
    np.testing.assert_array_equal(slice_arrays[0], raw[positions[0], :, :])
    np.testing.assert_array_equal(slice_arrays[1], raw[:, :, positions[1]].T)
    np.testing.assert_array_equal(slice_arrays[2], raw[:, positions[2], :])
    plane_names = ["YZ", "XZ", "XY"]
    axis_labels = [("Y", "Z"), ("X", "Z"), ("Y", "X")]
    for index, (pixels, color) in enumerate(zip(slice_arrays, colors)):
        section = fig.add_axes((0.79, 0.65 - index * 0.28, 0.18, 0.23))
        section.imshow(pixels, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        section.set_title(f"{plane_names[index]} | {'XYZ'[index]} = {positions[index]}", fontsize=11, color=color)
        section.set(xlabel=axis_labels[index][0], ylabel=axis_labels[index][1], xticks=[], yticks=[])
        for spine in section.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.7)
        fig.add_artist(
            ConnectionPatch(
                xyA=(0, 0.5),
                coordsA=section.transAxes,
                xyB=projected[index],
                coordsB=block.transData,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.3,
                color=color,
                shrinkA=6,
            )
        )
    fig.text(0.04, 0.17, "The OCT block covers a small retinal region, not the whole eye.", fontsize=12)
    fig.text(
        0.04,
        0.105,
        "Eye drawing: anatomical schematic, not patient-specific registration.\n"
        "Block and slices: original Harvard-GF voxel intensities; voxel units, not millimeters.",
        fontsize=10,
        color="#555555",
    )
    for extension in ("png", "pdf"):
        fig.savefig(args.output / f"oct_eye_location_and_slices.{extension}", dpi=300)
    plt.close(fig)
    metadata = {
        "source": str(args.volume.resolve()),
        "source_sha256": hashlib.sha256(args.volume.read_bytes()).hexdigest(),
        "source_shape": list(raw.shape),
        "display_axis_source_order_xyz": [0, 2, 1],
        "depth_axis_assumed": 1,
        "spacing": [1, 1, 1],
        "units": "voxel",
        "opacity": 1,
        "representation": "Opaque boundary of voxel cells, original intensity per cell",
        "clim": [0, 255],
        "smoothing": None,
        "downsampling": None,
        "lighting": False,
        "camera": camera,
        "slice_indices_xyz": positions,
        "slice_plane_world_coordinates": [float(x), float(y), float(z)],
        "eye_location": "Illustrative optic-nerve-head region; no patient-specific registration",
        "pyvista_version": pv.__version__,
        "drive_sync": "deferred by user; local only",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.output / "caption.txt").write_text(
        "Harvard-GF OCT sample displayed as an opaque voxel block. The visible boundary cells retain their "
        "original grayscale intensities (0-255); internal voxels are occluded. No transparency, smoothing, "
        "segmentation or downsampling is applied. X, Y and Z correspond to source axes 0, 2 and 1, respectively. "
        "Z is assumed depth. Unit voxel spacing is used; physical spacing and laterality are unverified.\n\n"
        "Eye-location figure: A schematic cross-section of the eye indicates an illustrative retinal scan region "
        "near the optic nerve head. The enlarged voxel block and its three orthogonal slices use actual Harvard-GF "
        "data. The eye schematic is not patient-specific registration and does not establish the exact scan location "
        "or laterality of this sample. Colored traces show slice-plane intersections with the block exterior. "
        "X and Y are local lateral coordinates; Z is assumed depth. No nasal/temporal or superior/inferior "
        "orientation is asserted.\n",
        encoding="utf-8",
    )
    print(f"Figure: {(args.output / 'oct_solid_block.png').resolve()}")


if __name__ == "__main__":
    main()
