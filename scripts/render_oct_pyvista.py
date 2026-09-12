import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.patches import ConnectionPatch


def main():
    parser = argparse.ArgumentParser(description="True OCT volume rendering and linked raw orthogonal slices.")
    parser.add_argument("--volume", type=Path, default=Path(tempfile.gettempdir()) / "gf_vol_cache/raw_2404.npy")
    parser.add_argument("--output", type=Path, default=Path("figures/oct_sample_2404/pyvista"))
    parser.add_argument("--depth-axis", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--drive-dir", type=Path, default=os.environ.get("DRIVE_SYNC_DIR"))
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    if args.drive_dir and (not args.drive_dir.is_absolute() or not args.drive_dir.is_dir()):
        parser.error("--drive-dir must be an existing absolute Google Drive-backed directory")
    raw = np.load(args.volume, mmap_mode="r", allow_pickle=False)
    if raw.ndim != 3 or raw.dtype != np.uint8 or min(raw.shape) < 2:
        parser.error("Expected a single 3D uint8 .npy volume")
    order = [axis for axis in range(3) if axis != args.depth_axis] + [args.depth_axis]
    data = np.ascontiguousarray(raw.transpose(order))
    limits = np.array(data.shape) - 1
    positions = [int(limits[0] // 2), int(limits[1] // 2), int(limits[2] // 6)]
    args.output.mkdir(parents=True, exist_ok=True)
    grid = pv.ImageData(dimensions=data.shape, spacing=(1, 1, 1))
    grid.point_data["intensity"] = data.ravel(order="F")
    np.testing.assert_array_equal(grid["intensity"].reshape(data.shape, order="F"), data)
    knots = [0, 50, 70, 100, 150, 255]
    alpha = [0, 0, 0.01, 0.10, 0.35, 0.60]
    opacity = np.round(255 * np.interp(np.arange(256), knots, alpha)).astype(np.uint8)
    plotter = pv.Plotter(off_screen=not args.interactive, window_size=(1800, 1500))
    plotter.set_background("white")
    actor = plotter.add_volume(
        grid,
        scalars="intensity",
        cmap="gray",
        clim=(0, 255),
        opacity=opacity,
        opacity_unit_distance=2.0,
        shade=True,
        ambient=0.45,
        diffuse=0.55,
        specular=0.1,
        blending="composite",
        mapper="fixed_point",
        show_scalar_bar=False,
    )
    actor.mapper.SetSampleDistance(0.5)
    plotter.add_mesh(grid.outline(), color="#aeb5bb", line_width=1, opacity=0.45)
    center = limits / 2
    eye = center + np.array([2.6, -3.0, -2.1]) * max(limits)
    plotter.camera_position = [eye.tolist(), center.tolist(), [0, 0, -1]]
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = max(limits) * 0.83
    plotter.show(auto_close=False, interactive=False)
    volume_image = plotter.screenshot(args.output / "oct_volume_only.png")
    if np.mean(volume_image[:, :, :3].mean(axis=2) < 200) < 0.01:
        raise RuntimeError("Volume rendering is unexpectedly empty; check the opacity transfer function")
    colors = ["#1678ae", "#ba463b", "#a67608"]
    x, y, z = positions
    lx, ly, lz = limits
    lines = [
        [[x, 0, lz], [x, 0, 0], [x, ly, 0]],
        [[lx, y, lz], [lx, y, 0], [0, y, 0]],
        [[0, 0, z], [lx, 0, z], [lx, ly, z]],
    ]
    targets = [[x, 0, lz * 0.32], [lx, y, lz * 0.32], [lx * 0.85, 0, z]]
    for line, color in zip(lines, colors):
        plotter.add_lines(np.array(line, dtype=float), color=color, width=3, connected=True)
    axis_origin = np.array([0, 0, lz], dtype=float)
    for index, name in enumerate("XYZ"):
        endpoint = axis_origin.copy()
        endpoint[index] += (-1 if index == 2 else 1) * limits[index] * 0.24
        plotter.add_lines(np.array([axis_origin, endpoint]), color="#303840", width=2)
        plotter.add_point_labels(
            [endpoint],
            ["-Z" if index == 2 else name],
            font_size=24,
            text_color="#303840",
            show_points=False,
            shape=None,
            always_visible=True,
        )
    plotter.render()
    image = plotter.screenshot(args.output / "oct_volume_marked.png")
    image_points = []
    for target in targets:
        plotter.renderer.SetWorldPoint(*target, 1)
        plotter.renderer.WorldToDisplay()
        px, py, _ = plotter.renderer.GetDisplayPoint()
        if not (0 <= px < image.shape[1] and 0 <= py < image.shape[0]):
            raise RuntimeError("A slice marker projects outside the rendered image")
        image_points.append((px, image.shape[0] - 1 - py))
    camera = {
        "position": list(plotter.camera.position),
        "focal_point": list(plotter.camera.focal_point),
        "view_up": list(plotter.camera.up),
        "parallel_scale": plotter.camera.parallel_scale,
    }
    if args.interactive:
        plotter.show(auto_close=False, interactive=True)
    plotter.close()
    fig = plt.figure(figsize=(14, 10), facecolor="white")
    volume_ax = fig.add_axes((0.255, 0.33, 0.49, 0.59))
    volume_ax.imshow(image)
    volume_ax.set_axis_off()
    fig.text(0.5, 0.95, "OCT volume and its orthogonal slices", ha="center", fontsize=20)
    fig.text(0.5, 0.91, " x ".join(map(str, data.shape)) + " voxels", ha="center", fontsize=12)
    boxes = [(0.765, 0.46, 0.22, 0.33), (0.045, 0.46, 0.22, 0.33), (0.40, 0.075, 0.20, 0.28)]
    slices = [data[x, :, :].T, data[:, y, :].T, data[:, :, z]]
    for index, pixels in enumerate(slices):
        source = np.take(raw, positions[index], axis=order[index])
        remaining = [axis for axis in range(3) if axis != order[index]]
        expected_row = order[2] if index < 2 else order[0]
        np.testing.assert_array_equal(pixels, np.moveaxis(source, remaining.index(expected_row), 0))
    labels = [("YZ", "X", "Y", "Z"), ("XZ", "Y", "X", "Z"), ("XY", "Z", "Y", "X")]
    for index, (box, pixels, label, color) in enumerate(zip(boxes, slices, labels, colors)):
        section = fig.add_axes(box)
        section.imshow(pixels, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        section.set_title(f"{label[0]} slice | {label[1]} = {positions[index]}", color=color, fontsize=13)
        section.set(xlabel=label[2], ylabel=label[3], xticks=[0, pixels.shape[1] - 1], yticks=[0, pixels.shape[0] - 1])
        for spine in section.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.8)
        start = [(0, 0.5), (1, 0.5), (1, 0.8)][index]
        fig.add_artist(
            ConnectionPatch(
                xyA=start,
                coordsA=section.transAxes,
                xyB=image_points[index],
                coordsB=volume_ax.transData,
                arrowstyle="-|>",
                mutation_scale=17,
                linewidth=1.6,
                color=color,
                shrinkA=8,
                shrinkB=1,
            )
        )
    fig.text(
        0.5,
        0.02,
        "3D: intensity-based transparency. 2D: unmodified slices. Coordinates: voxel indices.",
        ha="center",
        fontsize=10,
        color="#454b50",
    )
    for extension in ("png", "pdf"):
        fig.savefig(args.output / f"thesis_oct_volume_slices.{extension}", dpi=300)
    plt.close(fig)
    caption = (
        f"Three-dimensional OCT volume and orthogonal sections from Harvard-GF sample {args.volume.stem}. "
        "The center shows composite volume rendering of the full intensity array with intensity-dependent "
        "opacity, rather than textured exterior faces. Low intensities are transparent to reveal internal signal. "
        "Colored traces mark the intersections of the selected section planes with the volume bounding box; "
        "arrows connect the raw 2D sections to these traces. "
        f"The YZ, XZ and XY slices are at X={x}, Y={y} and Z={z}, respectively. "
        f"Display X, Y and Z correspond to source axes {order}; Z is assumed depth. "
        "The negative Z direction points upward on the page; Z indices increase with depth. "
        "All original voxels are supplied to the renderer without smoothing or downsampling. "
        "The opacity transfer function changes visibility and is not anatomical segmentation. "
        "Physical spacing and laterality are unverified; the geometry uses unit voxel spacing, not millimeters. "
        "No retinal layer boundaries or diagnoses are inferred from this rendering.\n"
    )
    (args.output / "caption.txt").write_text(caption, encoding="utf-8")
    metadata = {
        "source": str(args.volume.resolve()),
        "source_sha256": hashlib.sha256(args.volume.read_bytes()).hexdigest(),
        "source_shape": list(raw.shape),
        "display_axis_source_order_xyz": order,
        "slice_positions_xyz": positions,
        "spacing": [1, 1, 1],
        "units": "voxel",
        "opacity_intensities": knots,
        "opacity_values": alpha,
        "opacity_unit_distance": 2,
        "clim": [0, 255],
        "cmap": "gray",
        "mapper": "fixed_point",
        "sample_distance": 0.5,
        "shade": True,
        "ambient": 0.45,
        "diffuse": 0.55,
        "specular": 0.1,
        "camera": camera,
        "pyvista_version": pv.__version__,
        "vtk_version": pv.vtk_version_info,
        "smoothing": None,
        "downsampling": None,
        "renderer_window": list(image.shape[:2][::-1]),
        "arrow_world_targets": np.asarray(targets, dtype=float).tolist(),
        "opacity_lookup_uint8": opacity.tolist(),
        "arrow_image_targets": image_points,
        "drive_sync": "pending (user requested local delivery)" if not args.drive_dir else str(args.drive_dir),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.drive_dir:
        destination = args.drive_dir / "oct_sample_2404" / args.output.name
        destination.mkdir(parents=True, exist_ok=True)
        for source in args.output.iterdir():
            target = destination / source.name
            if source.is_file() and source.resolve() != target.resolve():
                shutil.copy2(source, target)
                if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(target.read_bytes()).digest():
                    raise OSError(f"Drive copy verification failed: {target}")
    print(f"Thesis figure: {(args.output / 'thesis_oct_volume_slices.png').resolve()}")


if __name__ == "__main__":
    main()
