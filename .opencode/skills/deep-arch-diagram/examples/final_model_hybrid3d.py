import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import final_model_publication as pub
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from matplotlib.transforms import Affine2D

ad = pub.ad
BLUE = "#b9d4e6"
GREEN = "#c6ddbd"
LILAC = "#d5c8e1"
SAND = "#eed29e"
COLORS = (BLUE, GREEN, LILAC, SAND)


def shadow(ax, vertices):
    vertices = np.asarray(vertices, dtype=float) + [0.55, -0.65]
    ax.add_patch(Polygon(vertices, facecolor="#263847", alpha=0.08, edgecolor="none", zorder=-1))


def register(ax, bounds, label):
    pub.BOXES.append((ax, bounds, label))


def feature_volume(ax, x, yc, w, h, depth, color, shape, label, grid=False):
    y, rise = yc - h / 2, depth * 0.6
    shadow(ax, [(x, y), (x + w + depth, y), (x + w + depth, y + h + rise), (x, y + h)])
    ad.volume(ax, x, y, w, h, depth, rise, color, lw=0.75)
    if grid:
        for ratio in (0.25, 0.5, 0.75):
            ax.plot([x + w * ratio] * 2, [y, y + h], color=pub.INK, alpha=0.2, lw=0.45)
            ax.plot([x, x + w], [y + h * ratio] * 2, color=pub.INK, alpha=0.2, lw=0.45)
    cx = x + (w + depth) / 2
    pub.text(ax, cx, yc + 13, shape, fs=10.5)
    pub.text(ax, cx, yc - 12, label, fs=10, color=pub.MUTED)
    register(ax, (x, y, x + w + depth, y + h + rise), shape)
    return x + w + depth


def contrast(image):
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, (1, 99))
    return np.clip((image - low) / max(high - low, 1e-6), 0, 1)


def image_face(ax, image, transform, vertices):
    patch = Polygon(vertices, facecolor="none", edgecolor=pub.INK, lw=0.75, zorder=4)
    ax.add_patch(patch)
    artist = ax.imshow(
        contrast(image),
        extent=(0, 1, 0, 1),
        origin="upper",
        cmap="gray",
        interpolation="bilinear",
        transform=transform + ax.transData,
        aspect="auto",
        zorder=3,
    )
    artist.set_clip_path(patch)


def oct_cube(ax, volume, views):
    x, y, w, h, dx, dy = 3, 82, 11, 12, 3, 3
    shadow(ax, [(x, y), (x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + dx, y + h + dy), (x, y + h)])
    image_face(
        ax,
        volume[:, 100, :].T,
        Affine2D().from_values(w, 0, 0, h, x, y),
        [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
    )
    image_face(
        ax,
        volume[100].T,
        Affine2D().from_values(dx, dy, 0, h, x + w, y),
        [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
    )
    image_face(
        ax,
        views[1],
        Affine2D().from_values(w, 0, dx, dy, x, y + h),
        [(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)],
    )
    register(ax, (x, y, x + w + dx, y + h + dy), "OCT cube")
    pub.text(ax, 10, 102, "Raw OCT", fs=12)
    pub.text(ax, 10, 77, r"uint8, $200^3$", fs=11)


def view_tile(ax, image, yc, name):
    x, y, w, h = 54, yc - 6, 10, 12
    shadow(ax, [(x, y), (x + w, y), (x + w + 1.4, y + h + 1.2), (x, y + h)])
    ad.volume(ax, x, y, w, h, 1.4, 1.2, "#d5e0e5", lw=0.7)
    image_face(ax, image, Affine2D().from_values(w, 0, 0, h, x, y), [(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
    register(ax, (x, y, x + w + 1.4, y + h + 1.2), name)
    pub.text(ax, x + 5.7, yc + 13, r"$1\times224^2$", fs=10.5)
    pub.text(ax, x + 5.7, yc - 12, name, fs=10, color=pub.MUTED)


def embedding(ax, x, yc):
    y = yc - 6.5
    for i in range(8):
        ax.add_patch(
            Rectangle(
                (x, y + i * 1.625), 2.5, 1.625, facecolor=BLUE if i % 2 else "#d8e5ef", edgecolor="#627989", lw=0.4
            )
        )
    register(ax, (x, y, x + 2.5, y + 13), "256-D vector")
    pub.text(ax, x + 1.25, yc + 10, "256", fs=10.5)


def render_hybrid(ax, volume, views):
    ax.set_xlim(0, 207)
    ax.set_ylim(0, 153)
    ax.axis("off")
    pub.text(ax, 3, 149, "OCT glaucoma classification", fs=19, weight="bold", ha="left")
    pub.text(ax, 203, 149, "2D / 3D features  +  gated cross-attention", fs=12, color=pub.MUTED, ha="right")
    for yc, color in ((125, "#edf3f8"), (88, "#f0f4eb"), (51, "#f0f4eb")):
        pub.region(ax, 52, yc - 16, 129, 33, color)
    pub.region(ax, 1, 5, 203, 23, "#faf2e6")
    pub.text(ax, 54, 144, "(a) ResNeXt3D  |  seven residual blocks", fs=12, color="#2e5a78", ha="left")
    pub.text(ax, 54, 107, "(b) MaxViT-Tiny  |  Slab MIP", fs=12, color="#45623c", ha="left")
    pub.text(ax, 54, 70, "(c) MaxViT-Tiny  |  Full AIP", fs=12, color="#45623c", ha="left")
    pub.text(ax, 3, 32, "(d) Gated fusion and test-time readout: right to left", fs=12, color="#73562f", ha="left")

    oct_cube(ax, volume, views)
    pub.route(ax, (17, 88), (20, 88), head=False)
    pub.junction(ax, 20, 88)
    pub.route(ax, (20, 88), (20, 125), (24, 125))
    pub.route(ax, (20, 88), (24, 88))
    pub.route(ax, (20, 88), (20, 51), (24, 51))
    pub.block(ax, 24, 117, 25, 16, "Raw OR bilateral\nDepth-last\nfloat32 / 255", pub.SAND, fs=11)
    pub.block(ax, 24, 80, 25, 16, "Raw depth-last\nSlab MIP + resize\nfloat32 / 255", pub.SAND, fs=11)
    pub.block(ax, 24, 43, 25, 16, "Raw depth-last\nFull AIP + resize\nfloat32 / 255", pub.SAND, fs=11)
    pub.text(ax, 36.5, 110, "3D source", fs=10, color=pub.MUTED)

    end = 49
    for x, w, h, depth, shape, label, color in (
        (55, 10, 14, 3, r"$32\times200^3$", "Stem + block\nstride 1", COLORS[0]),
        (77, 8.5, 12, 3.5, r"$64\times100^3$", "2 blocks\nstrides 2, 1", COLORS[1]),
        (99, 7, 10, 4, r"$128\times50^3$", "2 blocks\nstrides 2, 1", COLORS[2]),
        (119, 5.5, 8, 4, r"$192\times25^3$", "2 blocks\nstrides 2, 1", COLORS[3]),
    ):
        pub.route(ax, (end + (0.3 if end != 49 else 0), 125), (x - 0.3, 125))
        end = feature_volume(ax, x, 125, w, h, depth, color, shape, label)
    pub.route(ax, (end + 0.3, 125), (132, 125))
    pub.block(ax, 132, 118, 14, 14, "GAP\n192-D", pub.BLUE, fs=11)

    for yc, image, title in ((88, views[0], "Slab MIP"), (51, views[1], "Full AIP")):
        pub.route(ax, (49, yc), (53.7, yc))
        view_tile(ax, image, yc, title)
        pub.text(ax, 67.5, yc + 3.5, "stem", fs=8.5, color=pub.MUTED)
        end = 65.4
        for x, w, h, depth, shape, count, color in (
            (70, 8, 10, 1.8, r"$64\times56^2$", 2, COLORS[0]),
            (85, 7, 9, 2, r"$128\times28^2$", 2, COLORS[1]),
            (100, 6, 8, 2.4, r"$256\times14^2$", 5, COLORS[2]),
            (115, 5, 6, 2.8, r"$512\times7^2$", 2, COLORS[3]),
        ):
            pub.route(ax, (end + 0.3, yc), (x - 0.3, yc))
            end = feature_volume(ax, x, yc, w, h, depth, color, shape, f"{count} MaxViT\nblocks", grid=True)
        pub.route(ax, (end + 0.3, yc), (132, yc))
        pub.block(ax, 132, yc - 7, 14, 14, "LN2d\nGAP\n512-D", pub.BLUE, fs=10.5)

    for yc, dim in ((125, 192), (88, 512), (51, 512)):
        pub.route(ax, (146, yc), (153, yc))
        pub.block(ax, 153, yc - 7, 20, 14, f"Linear\n{dim} to 256\nReLU", pub.LILAC, fs=11)
        pub.route(ax, (173, yc), (175.5, yc))
        embedding(ax, 176, yc)

    pub.block(ax, 176, 32, 20, 12, "Stack 2D tokens\n(B, 2, 256)", pub.BLUE, fs=10.5)
    pub.route(ax, (178.5, 88), (193, 88), (193, 44))
    pub.route(ax, (178.5, 51), (183, 51), (183, 44))
    pub.route(ax, (178.5, 125), (202, 125), (202, 24), (196, 24))
    pub.text(ax, 199, 112, r"$p_3$", fs=12)

    pub.block(
        ax,
        145,
        8,
        51,
        20,
        "CrossGate: Q = 3D; K/V = LN(2D)\n8-head attention; scalar sigmoid gate\n"
        r"$z=p_3+\sigma(\alpha)\,o$",
        pub.SAND,
        fs=11,
    )
    pub.route(ax, (186, 32), (186, 28))
    pub.block(ax, 112, 8, 26, 20, "Linear 256 to 2\nRaw logits\n(B, 2)", pub.GREEN, fs=11)
    pub.route(ax, (145, 18), (138, 18))
    pub.block(ax, 80, 8, 25, 20, "Divide logits by T\nSoftmax", pub.LILAC, fs=11)
    pub.route(ax, (112, 18), (105, 18), dashed=True)
    pub.block(ax, 40, 8, 33, 20, r"$P(y=0),\ P(y=1)$" + "\n0: non-glaucoma\n1: glaucoma", pub.BLUE, fs=11)
    pub.route(ax, (80, 18), (73, 18), dashed=True)
    pub.block(ax, 5, 8, 28, 20, "Reported label\n" + r"$\hat{y}=\mathbf{1}[P(y=1)\geq\tau]$", pub.ROSE, fs=11)
    pub.route(ax, (40, 18), (33, 18), dashed=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", type=Path, default=Path(tempfile.gettempdir()) / "gf_vol_cache/raw_2404.npy")
    parser.add_argument("--drive-dir", type=Path)
    args = parser.parse_args()
    if not args.volume.is_file():
        raise FileNotFoundError(f"A real cached OCT volume is required: {args.volume}")
    import final_model as fm

    raw = np.load(args.volume)
    if raw.shape != (200, 200, 200) or raw.dtype != np.uint8:
        raise ValueError("Expected a raw uint8 volume of shape (200,200,200).")
    depth = fm.depth_axis(raw)
    volume = fm.to_depth_last(raw, depth)
    views = fm.project_views(volume)
    verification = pub.verify()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    pub.OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(20, 13.5))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.965, bottom=0.105)
    render_hybrid(ax, volume, views)
    fig.text(
        0.025,
        0.014,
        "Feature blocks / grids are schematic, not measured activations or a linear size scale. "
        "LN2d = LayerNorm2d; GAP = global average pooling.\n"
        f"Texture source: {args.volume.name}; cube = two central B-scans + Full AIP, contrast-adjusted for display. "
        "MaxViT stems are omitted visually.\n"
        "MaxViT weights are independent. Paired training augmentation precedes /255; bilateral affects only 3D. "
        "Dashed arrows: notebook post-processing.\n"
        "T is fitted on validation logits; tau is tuned before validation calibration but applied after test calibration "
        "(existing protocol mismatch).",
        fontsize=12.5,
        color=pub.MUTED,
        linespacing=1.4,
    )
    qa = pub.layout_check(fig)
    print("Layout:", qa)
    files = []
    for suffix in ("png", "svg", "pdf"):
        path = pub.OUT / f"10_end_to_end_hybrid3d.{suffix}"
        fig.savefig(path, dpi=300, facecolor="white")
        files.append(path)
        print("Wrote", path)
    plt.close(fig)
    manifest = pub.OUT / "10_hybrid3d_verification.json"
    manifest.write_text(
        json.dumps(
            {
                "code_verification": verification,
                "layout": qa,
                "texture_source": str(args.volume),
                "depth_axis": depth,
                "texture_sha256": hashlib.sha256(args.volume.read_bytes()).hexdigest(),
                "feature_map_textures": "schematic grids; not activations",
                "projection_previews": "raw 200-square views displayed as thumbnails; model resizes to 224",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    files.append(manifest)
    drive = args.drive_dir or (Path(os.environ["DRIVE_SYNC_DIR"]) if os.environ.get("DRIVE_SYNC_DIR") else None)
    if drive is None and os.name != "nt":
        candidate = Path("/content/drive/MyDrive/MasterBKDN/Thesis/architecture_figures")
        if candidate.parent.is_dir():
            drive = candidate
    if drive:
        if not drive.parent.is_dir():
            raise FileNotFoundError(f"Drive parent must exist: {drive.parent}")
        drive.mkdir(exist_ok=True)
        for path in files:
            shutil.copy2(path, drive / path.name)
        print("Synced to configured Drive destination:", drive)
    else:
        print("Drive sync pending: no verified mounted Drive destination.")


if __name__ == "__main__":
    main()
