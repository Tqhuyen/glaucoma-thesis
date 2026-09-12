import argparse
import base64
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
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.mplot3d import proj3d
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Render one OCT volume and an offline, all-slice HTML viewer.")
    parser.add_argument("--volume", type=Path, default=Path(tempfile.gettempdir()) / "gf_vol_cache/raw_2404.npy")
    parser.add_argument("--output", type=Path, default=Path("figures/oct_sample_2404"))
    parser.add_argument("--depth-axis", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--drive-dir", type=Path, default=os.environ.get("DRIVE_SYNC_DIR"))
    args = parser.parse_args()
    if args.drive_dir and (not args.drive_dir.is_absolute() or not args.drive_dir.is_dir()):
        parser.error("--drive-dir must be an existing absolute Google Drive-backed directory")
    volume = np.load(args.volume, mmap_mode="r", allow_pickle=False)
    if volume.ndim != 3 or volume.dtype != np.uint8 or min(volume.shape) < 2:
        parser.error("Expected one 3D uint8 OCT .npy volume, with at least two voxels per axis")
    args.output.mkdir(parents=True, exist_ok=True)
    lateral = [axis for axis in range(3) if axis != args.depth_axis]
    order = [*lateral, args.depth_axis]
    cube = volume.transpose(order)
    names = {lateral[0]: "X", lateral[1]: "Y", args.depth_axis: "Z"}
    labels = [f"{names[axis]} (voxel index)" for axis in order]
    labels[2] = "Z: depth (voxel index)"
    centers = [size // 2 for size in cube.shape]
    fig = plt.figure(figsize=(10, 8), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    for fixed in range(3):
        other = [axis for axis in range(3) if axis != fixed]
        grids = np.meshgrid(*(np.arange(cube.shape[axis]) for axis in other), indexing="ij")
        coords = [None] * 3
        coords[fixed] = np.full(grids[0].shape, centers[fixed])
        for axis, grid in zip(other, grids):
            coords[axis] = grid
        pixels = np.take(cube, centers[fixed], axis=fixed)
        ax.plot_surface(*coords, facecolors=plt.cm.gray(pixels / 255), rstride=2, cstride=2, shade=False)
    ax.set(xlabel=labels[0], ylabel=labels[1], zlabel=labels[2])
    ax.set_xlim(0, cube.shape[0] - 1)
    ax.set_ylim(0, cube.shape[1] - 1)
    ax.set_zlim(cube.shape[2] - 1, 0)
    ax.set_box_aspect(cube.shape)
    ax.view_init(elev=24, azim=-48)
    ax.set_title("OCT volume: three orthogonal cross-sections", fontsize=15, pad=18)
    path_3d = args.output / "oct_3d_axes.png"
    fig.savefig(path_3d, dpi=160)
    plt.close(fig)
    panels = []
    fig, axs = plt.subplots(1, 3, figsize=(14, 5), layout="constrained")
    for fixed, ax in enumerate(axs):
        remaining = [axis for axis in range(3) if axis != fixed]
        row = args.depth_axis if fixed != args.depth_axis else remaining[0]
        column = next(axis for axis in remaining if axis != row)
        stack = volume.transpose(fixed, row, column)
        count, height, width = stack.shape
        atlas = Image.new("L", (width * 10, height * ((count + 9) // 10)))
        for index, image in enumerate(stack):
            atlas.paste(Image.fromarray(image), ((index % 10) * width, (index // 10) * height))
        atlas_path = args.output / f"axis_{fixed}_all_slices.png"
        atlas.save(atlas_path)
        encoded = base64.b64encode(atlas_path.read_bytes()).decode("ascii")
        panels.append(
            {
                "axis": fixed,
                "row": row,
                "column": column,
                "count": count,
                "width": width,
                "height": height,
                "image": encoded,
            }
        )
        ax.imshow(stack[count // 2], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set(
            title=f"Source axis {fixed}: slice {count // 2}/{count - 1}",
            xlabel=f"Source axis {column} (voxels)",
            ylabel=f"Source axis {row} (voxels)",
        )
    fig.suptitle(f"{args.volume.stem} | Central slices | Raw intensity 0-255")
    fig.savefig(args.output / "oct_2d_central_slices.png", dpi=160)
    plt.close(fig)
    fig = plt.figure(figsize=(14, 10), facecolor="white")
    ax = fig.add_axes((0.28, 0.32, 0.44, 0.61), projection="3d", computed_zorder=False)
    limits = np.array(cube.shape) - 1
    for fixed, location in ((0, limits[0]), (1, 0), (2, 0)):
        other = [axis for axis in range(3) if axis != fixed]
        grids = np.meshgrid(*(np.arange(cube.shape[axis]) for axis in other), indexing="ij")
        coords = [None] * 3
        coords[fixed] = np.full(grids[0].shape, location)
        for axis, values in zip(other, grids):
            coords[axis] = values
        ax.plot_surface(
            *coords,
            facecolors=plt.cm.gray(np.take(cube, location, axis=fixed) / 255),
            rstride=1,
            cstride=1,
            shade=False,
            antialiased=False,
            rasterized=True,
            zorder=1,
        )
    for direction in range(3):
        other = [axis for axis in range(3) if axis != direction]
        for first, second in ((0, 0), (0, 1), (1, 0), (1, 1)):
            visible = {0: 1, 1: 0, 2: 0}
            if first != visible[other[0]] and second != visible[other[1]]:
                continue
            line = np.zeros((3, 2))
            line[direction] = (0, limits[direction])
            line[other[0]] = first * limits[other[0]]
            line[other[1]] = second * limits[other[1]]
            ax.plot(*line, color="#555555", linewidth=0.65, zorder=2)
    ax.set(xlim=(0, limits[0]), ylim=(0, limits[1]), zlim=(limits[2], 0), xlabel="X", ylabel="Y", zlabel="Z (depth)")
    ax.set_box_aspect(cube.shape, zoom=0.82)
    ax.view_init(elev=25, azim=-55)
    for axis, limit in zip((ax.xaxis, ax.yaxis, ax.zaxis), limits):
        axis.set_ticks([0, int(limit // 2), int(limit)])
        axis.pane.fill = False
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(limits[0] * 0.45, -12, limits[2] + 10, "X", fontsize=12)
    ax.text(limits[0] + 12, limits[1] * 0.6, limits[2] + 10, "Y", fontsize=12)
    fig.text(0.5, 0.91, " x ".join(str(size) for size in cube.shape) + " voxels", ha="center", fontsize=14)
    plane_axes = [lateral[0], lateral[1], args.depth_axis]
    boxes = [(0.05, 0.47, 0.21, 0.31), (0.77, 0.47, 0.21, 0.31), (0.40, 0.085, 0.20, 0.28)]
    colors = ["#2074b5", "#b6473b", "#a87812"]
    connections = []
    overview_indices = [centers[0], centers[1], max(1, cube.shape[2] // 6)]
    for index, fixed in enumerate(plane_axes):
        panel = panels[fixed]
        stack = volume.transpose(fixed, panel["row"], panel["column"])
        section = fig.add_axes(boxes[index])
        location = overview_indices[index]
        section.imshow(stack[location], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        plane = "".join(name for axis, name in names.items() if axis != fixed)
        section.set_title(f"{plane} slice | {names[fixed]} = {location}", color=colors[index], fontsize=14, pad=10)
        section.set_xlabel(names[panel["column"]])
        section.set_ylabel(names[panel["row"]])
        section.set_xticks([0, stack.shape[2] - 1])
        section.set_yticks([0, stack.shape[1] - 1])
        for spine in section.spines.values():
            spine.set_color(colors[index])
            spine.set_linewidth(2)
        if index == 0:
            line = np.array([[location, location, location], [0, 0, limits[1]], [limits[2], 0, 0]])
            target = (location, 0, limits[2] * 0.5)
            start = (1, 0.5)
        elif index == 1:
            line = np.array([[limits[0], limits[0], 0], [location, location, location], [limits[2], 0, 0]])
            target = (limits[0], location, limits[2] * 0.5)
            start = (0, 0.5)
        else:
            line = np.array([[0, limits[0], limits[0]], [0, 0, limits[1]], [location, location, location]])
            target = (limits[0], 0, location)
            start = (1, 0.8)
        ax.plot(*line, color=colors[index], linewidth=2.5, zorder=5)
        connections.append((section, start, target, colors[index]))
    fig.text(0.5, 0.96, "From a 3D OCT volume to 2D slices", ha="center", fontsize=21)
    fig.text(
        0.5,
        0.015,
        "Colored lines locate each slice in the volume. Coordinates are voxel indices.",
        ha="center",
        fontsize=11,
        color="#444444",
    )
    fig.canvas.draw()
    for section, start, target, color in connections:
        x, y, _ = proj3d.proj_transform(*target, ax.get_proj())
        fig.add_artist(
            ConnectionPatch(
                xyA=start,
                coordsA=section.transAxes,
                xyB=(x, y),
                coordsB=ax.transData,
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=1.8,
                color=color,
                shrinkA=8,
                shrinkB=2,
                clip_on=False,
            )
        )
    fig.savefig(args.output / "thesis_oct_data_overview.png", dpi=300)
    fig.savefig(args.output / "thesis_oct_data_overview.pdf", dpi=300)
    plt.close(fig)
    fig, axs = plt.subplots(3, 5, figsize=(13, 8), layout="constrained")
    selected = {}
    for row_index, fixed in enumerate(plane_axes):
        panel = panels[fixed]
        stack = volume.transpose(fixed, panel["row"], panel["column"])
        indices = np.linspace(0, len(stack) - 1, 7, dtype=int)[1:-1]
        selected[str(fixed)] = indices.tolist()
        for ax, index in zip(axs[row_index], indices):
            ax.imshow(stack[index], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(f"{names[fixed]} = {index}")
            ax.set_xlabel(names[panel["column"]])
            ax.set_ylabel(names[panel["row"]])
            ax.set_xticks([0, stack.shape[2] - 1])
            ax.set_yticks([0, stack.shape[1] - 1])
    fig.suptitle("Sections through one OCT volume", fontsize=17)
    fig.savefig(args.output / "thesis_oct_slice_sequence.png", dpi=300)
    fig.savefig(args.output / "thesis_oct_slice_sequence.pdf", dpi=300)
    plt.close(fig)
    caption = (
        "Overview: Three-dimensional optical coherence tomography (OCT) data from one Harvard-GF sample "
        f"(cached volume {args.volume.stem}; array shape {tuple(volume.shape)}). "
        "The central cuboid displays the original exterior image slices on its three visible faces. "
        "Separate orthogonal sections surround the volume; colored lines and arrows identify their locations. "
        f"The YZ, XZ and XY sections use X, Y and Z indices {overview_indices}, respectively. "
        "The XY image is an en-face section, "
        "not a depth-averaged projection. Grayscale represents stored OCT intensity (0-255). "
        f"X, Y and Z correspond to source array axes {order}, respectively; Z is the assumed depth axis. "
        "Coordinates are voxel indices, not physical distances; acquisition spacing and anatomical "
        "laterality have not been verified. The cuboid represents the full volume extent using exterior textures; "
        "it is not a segmented retinal surface or a transparent rendering of internal voxels.\n\n"
        "Slice sequence: Five evenly spaced interior sections along each coordinate axis of the same OCT volume. "
        "Rows vary X, Y and Z, respectively; all panels share a fixed grayscale range of 0-255. "
        "The sequence illustrates changes in cross-sectional appearance throughout the volume. "
        "The supplementary HTML viewer includes every slice along all three axes.\n"
    )
    (args.output / "thesis_captions.txt").write_text(caption, encoding="utf-8")
    preview = base64.b64encode(path_3d.read_bytes()).decode("ascii")
    html = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OCT volume explorer</title>
<style>
body{font:16px system-ui;background:#f4f3ef;color:#202b32;margin:0;padding:24px}
main{max-width:1300px;margin:auto}h1{font-size:32px}p{line-height:1.5}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:24px}
section{background:white;padding:20px;border:1px solid #ccd1d0;border-radius:8px}
canvas{width:100%;height:auto;image-rendering:pixelated;background:black}input{width:100%}
.preview{display:block;max-width:780px;width:100%;margin:24px auto}label{display:block;margin:16px 0}
button{padding:8px 16px;margin-right:8px}small{display:block;margin-bottom:12px}
</style><main><h1>One OCT, three section planes</h1>
<p id="description"></p>
<p>Move each slider to inspect every slice. Indices are zero-based; raw grayscale is fixed at 0-255.
Rows increase downward, columns to the right. No filtering, resizing, or per-slice contrast adjustment.</p>
<div class="grid" id="panels"></div>
<h2>3D spatial reference</h2>
<p>Three central orthogonal image planes, not a segmented retinal surface or volumetric ray rendering.
The 3D illustration samples every second voxel for rendering. 2D slices retain all pixels.</p>
<img class="preview" alt="Three OCT midplanes with source-axis labels" src="data:image/png;base64,__PREVIEW__">
<p>Physical spacing and anatomical laterality are unverified; equal voxel units do not imply equal physical distances.</p>
</main><script>
const metadata = __METADATA__;
document.getElementById('description').textContent = metadata.description;
const panels = __PANELS__;
for (const p of panels) {
  const section = document.createElement('section');
  section.innerHTML = `<h2>Slice axis ${p.axis}</h2><small>Horizontal: source axis ${p.column}; vertical: source axis ${p.row} (voxels)</small>
    <canvas width="${p.width}" height="${p.height}"></canvas>
    <label>Slice <output></output><input aria-label="Source axis ${p.axis} slice" type="range" min="0" max="${p.count-1}" value="${Math.floor(p.count/2)}"></label>
    <button type="button">Previous</button><button type="button">Next</button>`;
  document.getElementById('panels').appendChild(section);
  const slider = section.querySelector('input'), canvas = section.querySelector('canvas');
  const ctx = canvas.getContext('2d'), image = new Image();
  function draw() {
    const i = Number(slider.value);
    ctx.drawImage(image, (i%10)*p.width, Math.floor(i/10)*p.height, p.width, p.height, 0, 0, p.width, p.height);
    section.querySelector('output').textContent = `${i} / ${p.count-1}`;
  }
  image.onload = draw; image.src = 'data:image/png;base64,' + p.image;
  slider.addEventListener('input', draw);
  section.querySelectorAll('button').forEach((button, index) => button.onclick = () => {
    slider.value = Math.max(0, Math.min(p.count-1, Number(slider.value) + (index ? 1 : -1))); draw();
  });
}
</script></html>"""
    metadata = {
        "source": str(args.volume.resolve()),
        "source_sha256": hashlib.sha256(args.volume.read_bytes()).hexdigest(),
        "shape": list(volume.shape),
        "dtype": str(volume.dtype),
        "depth_axis_assumed": args.depth_axis,
        "description": f"{args.volume.stem}: {volume.shape}, uint8. Assumed depth: source axis {args.depth_axis}.",
        "atlas_layout": "10 columns, row-major, slice 0 at top left; no padding between slices",
        "display_axis_source_order_xyz": order,
        "sequence_source_indices": selected,
        "overview_slice_indices_xyz": overview_indices,
        "drive_sync": "pending" if not args.drive_dir else str(args.drive_dir / args.output.name),
    }
    html = html.replace("__PREVIEW__", preview).replace("__PANELS__", json.dumps(panels))
    html = html.replace("__METADATA__", json.dumps(metadata).replace("<", "\\u003c"))
    (args.output / "oct_slice_viewer.html").write_text(html, encoding="utf-8")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.drive_dir:
        destination = args.drive_dir / args.output.name
        destination.mkdir(parents=True, exist_ok=True)
        for source in args.output.iterdir():
            if source.is_file():
                target = destination / source.name
                if source.resolve() != target.resolve():
                    shutil.copy2(source, target)
                if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(target.read_bytes()).digest():
                    raise OSError(f"Drive copy verification failed: {target}")
    print(f"Viewer: {(args.output / 'oct_slice_viewer.html').resolve()}")
    print(f"Drive sync: {metadata['drive_sync']}")


if __name__ == "__main__":
    main()
