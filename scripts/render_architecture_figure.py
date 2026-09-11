import os
import shutil
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
sys.path.insert(0, os.path.join(ROOT, "scripts"))

COL = {
    "blue": ("#dae8fc", "#6c8ebf"),
    "green": ("#d5e8d4", "#82b366"),
    "purple": ("#e1d5e7", "#9673a6"),
    "yellow": ("#ffe6cc", "#d79b00"),
    "gray": ("#f5f5f5", "#666666"),
    "red": ("#f8cecc", "#b85450"),
    "white": ("#ffffff", "#333333"),
}
INK = "#1a1a1a"
ARROW = "#333333"


def norm_img(im):
    im = im.astype(np.float32)
    lo, hi = np.percentile(im, [1, 99])
    return np.clip((im - lo) / (hi - lo + 1e-6), 0, 1)


def load_oct():
    p = os.path.join(tempfile.gettempdir(), "gf_vol_cache", "raw_2404.npy")
    if not os.path.exists(p):
        return None, None
    import final_model as fm

    vol = np.load(p)
    dz = fm.depth_axis(vol)
    dvol = fm.to_depth_last(vol, dz)
    views = fm.project_views(dvol)
    idx = [80, 100, 120, 140]
    bscans = np.stack([dvol[:, j, :].T for j in idx])
    return views, bscans


def plates(ax, x, y, w, h, n, color, ox=0.9, oy=0.9, lw=1.0, z=10, round=0.6):
    fc, ec = COL[color]
    for i in range(n - 1, -1, -1):
        ax.add_patch(FancyBboxPatch((x + i * ox, y + i * oy), w, h,
                                    boxstyle=f"round,pad=0,rounding_size={round}",
                                    fc=fc, ec=ec, lw=lw, zorder=z + (n - 1 - i)))


def image_plates(ax, x, y, w, h, imgs, ox, oy, z=10, ec="#333333"):
    n = len(imgs)
    for j, im in enumerate(imgs):
        xj, yj = x + j * ox, y + j * oy
        zz = z + (n - 1 - j)
        ax.imshow(norm_img(im), extent=[xj, xj + w, yj, yj + h], cmap="gray",
                  origin="upper", zorder=zz, interpolation="bilinear", aspect="auto")
        ax.add_patch(Rectangle((xj, yj), w, h, fill=False, ec=ec, lw=1.0, zorder=zz + 0.1))


def arrow(ax, p1, p2, color=ARROW, lw=1.7, ms=15, z=60, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0, zorder=z))


def ortho(ax, pts, color=ARROW, lw=1.7, ms=15, z=60, ls="-"):
    for i in range(len(pts) - 1):
        last = i == len(pts) - 2
        ax.add_patch(FancyArrowPatch(pts[i], pts[i + 1], arrowstyle="-|>" if last else "-",
                                     mutation_scale=ms if last else 1, color=color, lw=lw,
                                     ls=ls, shrinkA=0, shrinkB=0, zorder=z))


def top_label(ax, x, y, text, fs=10.5):
    ax.text(x, y, text, ha="center", va="bottom", fontsize=fs, color=INK, weight="bold", zorder=70)


def bottom_label(ax, x, y, text, fs=9.2):
    ax.text(x, y, text, ha="center", va="top", fontsize=fs, color=INK, zorder=70, linespacing=1.3)


def circle_col(ax, x, yc, span, color, r=0.95, n=7, z=30):
    fc, ec = COL[color]
    ys = np.linspace(yc + span / 2, yc - span / 2, n)
    out = []
    for k, yy in enumerate(ys):
        if k == n // 2:
            ax.text(x, yy, r"$\vdots$", ha="center", va="center", fontsize=10, color=ec, zorder=z + 1)
            continue
        ax.add_patch(Circle((x, yy), r, fc=fc, ec=ec, lw=1.1, zorder=z))
        out.append(yy)
    return out


def mesh(ax, x1, ys1, x2, ys2, color="#444444", lw=0.55, alpha=0.5, z=20):
    for a in ys1:
        for b in ys2:
            ax.plot([x1, x2], [a, b], color=color, lw=lw, alpha=alpha, zorder=z)


def box(ax, x, y, w, h, text, color, fs=9.5, weight="normal", round=1.2, z=30, ec=None):
    fc, ec0 = COL[color]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.15,rounding_size={round}",
                                fc=fc, ec=ec or ec0, lw=1.3, zorder=z))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=INK, weight=weight, zorder=z + 1, linespacing=1.4)


def panel(ax, x, y, w, h, fc, ec, round=2.0, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={round}",
                                fc=fc, ec=ec, lw=1.2, zorder=z))


def op_node(ax, x, y, sym, r=2.2, z=65):
    ax.add_patch(Circle((x, y), r, fc="white", ec="#333333", lw=1.3, zorder=z))
    ax.text(x, y, sym, ha="center", va="center", fontsize=11.5, color=INK, zorder=z + 1)


def main():
    views, bscans = load_oct()
    fig, ax = plt.subplots(figsize=(27.0, 10.0))
    plt.rcParams["svg.fonttype"] = "none"
    ax.set_xlim(0, 270)
    ax.set_ylim(-8, 100)
    ax.axis("off")

    ax.text(1, 95.5, "Main model:  2\u00d72D (MaxViT-Tiny) + 1\u00d73D (ResNeXt3D) + CrossGate"
                     "   \u2014   \u2248 58.30 M parameters",
            fontsize=16, weight="bold", color=INK)

    y3d = 74.0
    yA, yB = 44.0, 16.0

    # ================= 3D branch =================
    image_plates(ax, 3, y3d - 8.5, 13, 17, bscans, 0.95, 0.95)
    top_label(ax, 3 + 13 / 2 + 1.9, y3d + 11.0, "1\u00d7200\u00d7200\u00d7200", fs=10.0)
    bottom_label(ax, 3 + 13 / 2 + 1.9, y3d - 10.2, "Input 3D OCT\n200\u00d7200\u00d7200")

    stacks = [
        (12.0, 17.0, "blue", "32\u00d7200\u00b3", "Stem: Conv + GN + ReLU\n+ residual block, s1"),
        (10.5, 14.0, "green", "64\u00d7100\u00b3", "2 residual blocks\nstrides: 2, 1"),
        (9.0, 11.5, "purple", "128\u00d750\u00b3", "2 residual blocks\nstrides: 2, 1"),
        (7.5, 9.0, "yellow", "192\u00d725\u00b3", "2 residual blocks\nstrides: 2, 1"),
    ]
    xcur = 26.0
    prev_right = 19.2
    for (w, h, color, top, bot) in stacks:
        y = y3d - h / 2
        plates(ax, xcur, y, w, h, 5, color)
        cx = xcur + w / 2 + 1.8
        top_label(ax, cx, y + h + 4.6, top, fs=10.0)
        bottom_label(ax, cx, y - 1.8, bot)
        ortho(ax, [(prev_right, y3d), (xcur, y3d)])
        prev_right = xcur + w + 4 * 0.9 + 0.6
        xcur += w + 6.4

    ortho(ax, [(prev_right, y3d), (93.4, y3d)])
    ys3 = circle_col(ax, 96, y3d, 14, "gray", n=7)
    top_label(ax, 96, y3d + 8.2, "192", fs=10.5)
    bottom_label(ax, 96, y3d - 8.2, "GAP + Flatten\n192-d")

    ys3p = circle_col(ax, 110, y3d, 14, "blue", n=7)
    top_label(ax, 110, y3d + 8.2, "Linear 192\u2192256 + ReLU", fs=9.6)
    bottom_label(ax, 110, y3d - 8.2, "p3D\n(256)")
    mesh(ax, 96.95, ys3, 109.05, ys3p, alpha=0.4, lw=0.45)

    # ================= 2D branches =================
    def branch_2d(y, view, title):
        image_plates(ax, 3, y - 7, 13, 14, [view], 0, 0)
        top_label(ax, 9.5, y + 7.6, "1\u00d7224\u00d7224", fs=10.0)
        bottom_label(ax, 9.5, y - 8.4, title)
        ortho(ax, [(16.2, y), (20.0, y)])
        box(ax, 20, y - 7, 20, 14, "", "purple")
        ax.text(30, y + 2.6, "MaxViT-Tiny", ha="center", va="center", fontsize=9.6, weight="bold",
                color=INK, zorder=70)
        ax.text(30, y - 1.8, "ImageNet \u2022 independent\nMBConv + multi-axis attn", ha="center", va="center",
                fontsize=8.2, color=INK, zorder=70, linespacing=1.3)
        ortho(ax, [(40.2, y), (44.0, y)])
        plates(ax, 44, y - 5.2, 8.5, 10.5, 5, "green", ox=0.8, oy=0.8, round=0.5)
        top_label(ax, 48.8, y + 9.5, "512\u00d77\u00d77", fs=10.0)
        bottom_label(ax, 48.8, y - 7.0, "feature map\n512\u00d77\u00d77")
        ortho(ax, [(55.9, y), (57.5, y)])
        ys = circle_col(ax, 60, y, 11, "gray", n=7)
        top_label(ax, 60, y + 7.0, "512", fs=10.5)
        bottom_label(ax, 60, y - 7.0, "GAP + Flatten")
        ysp = circle_col(ax, 84, y, 11, "blue", n=7)
        top_label(ax, 84, y + 7.0, "Linear 512\u2192256 + ReLU", fs=9.6)
        mesh(ax, 60.95, ys, 83.05, ysp, alpha=0.4, lw=0.45)
        return ys, ysp

    _, yspA = branch_2d(yA, views[0], "View: Slab MIP\n1\u00d7224\u00d7224")
    _, yspB = branch_2d(yB, views[1], "View: Full AIP\n1\u00d7224\u00d7224")
    bottom_label(ax, 84, yA - 7.0, "p2D\u00b9\n(256)")
    bottom_label(ax, 84, yB - 7.0, "p2D\u00b2\n(256)")

    ortho(ax, [(86, yA), (98, yA), (98, 34), (105, 34)])
    ortho(ax, [(86, yB), (98, yB), (98, 26), (105, 26)])
    box(ax, 105, 23, 18, 14, "Stack tokens\n(B, 2, 256)", "blue", fs=11)

    # ================= CrossGate (MAXIM-style CGB) =================
    panel(ax, 128, 21, 88, 68, "#fbe7e7", "#d98c8c")
    ax.text(171, 86.5, "CrossGate fusion", ha="center", va="center", fontsize=15, weight="bold",
            color=INK, zorder=70)
    ortho(ax, [(112, y3d), (134, y3d)])
    ax.add_patch(Circle((134, y3d), 0.55, fc=ARROW, ec=ARROW, zorder=66))
    ax.text(140, 76.2, "Query: p3D", ha="left", va="bottom", fontsize=11,
            color=INK, zorder=70)
    ortho(ax, [(123, 30), (132, 30)])

    ortho(ax, [(134, y3d), (134, 84), (204, 84), (204, 62.3)], ls="--")
    ortho(ax, [(134, y3d), (140, y3d), (140, 62), (145.8, 62)])
    box(ax, 132, 25, 22, 10, "LayerNorm\nD = 256", "white", fs=11, round=0.9)
    ortho(ax, [(154, 30), (166, 30), (166, 49.8)])
    box(ax, 146, 50, 40, 20, "", "white", round=1.4)
    ax.text(166, 64.0, "Multi-Head\nCross-Attention", ha="center", va="center", fontsize=13,
            weight="bold", color=INK, zorder=70, linespacing=1.3)
    ax.text(166, 54.5, "Q input: p3D; K/V inputs: LN(T2D)\n8 heads  |  dropout 0.1", ha="center",
            va="center", fontsize=10, color=INK, zorder=70, linespacing=1.3)
    ax.text(167.5, 40, "LN(T2D)\n(B, 2, 256)", ha="left", va="center",
            fontsize=10, color=INK, zorder=70)

    ortho(ax, [(186, 60), (192.6, 60)])
    op_node(ax, 195, 60, "\u00d7")
    box(ax, 187, 39, 17, 11, "Sigmoid(alpha)\nscalar gate", "yellow", fs=10)
    ortho(ax, [(195, 50), (195, 57.7)])
    ax.text(195.5, 36.5, "alpha: learnable\ninit 0.5; gate = 0.622", ha="center", va="top",
            fontsize=10, color=INK, zorder=70)
    ortho(ax, [(197.2, 60), (201.6, 60)])
    op_node(ax, 204, 60, "+")
    ax.text(208, 63, "z\n(B, 256)", ha="left", va="bottom", fontsize=10, color=INK, zorder=70)
    ortho(ax, [(206.2, 60), (226, 60)])

    # ================= Classification head =================
    panel(ax, 222, 12, 44, 67, "#eef4fb", "#9bb7d4")
    ax.text(244, 75, "Classification head", ha="center", va="center", fontsize=15, weight="bold",
            color=INK, zorder=70)
    box(ax, 226, 53, 34, 14, "Linear (256 to 2)", "green", fs=13)
    ortho(ax, [(243, 53), (243, 49.2)])
    ax.text(261.5, 51.0, "logits (B, 2)", ha="right", va="center", fontsize=10, color=INK, zorder=70)
    box(ax, 226, 39, 34, 10, "Softmax (inference)", "blue", fs=12)
    ortho(ax, [(243, 39), (243, 34.2)])
    box(ax, 226, 16, 34, 18, "", "white", round=1.2)
    rows = [("0: Non-glaucoma", "P(y=0)", "#f8cecc"),
            ("1: Glaucoma", "P(y=1)", "#d5e8d4")]
    for k, (name, p, c) in enumerate(rows):
        yy = 28.0 - k * 7.0
        ax.add_patch(Rectangle((228.6, yy - 1.8), 3.6, 3.6, fc=c, ec="#666666", lw=0.9, zorder=71))
        ax.text(233.4, yy, name, ha="left", va="center", fontsize=10, color=INK, zorder=71)
        ax.text(259, yy, p, ha="right", va="center", fontsize=10, color=INK, zorder=71)

    ax.text(2, 88.5, "3D branch", fontsize=11, weight="bold", color="#1f4e79", zorder=70)
    ax.text(2, 57.0, "2D branches", fontsize=11, weight="bold", color="#38761d", zorder=70)
    ax.text(1, -5, "Shapes: channels first; B = batch. s2 = (2,2,2), s1 = (1,1,1). Seven residual blocks; stride-1 blocks retain identity shortcuts.\n"
                    "Measured with timm 1.0.29: total 58,304,467 parameters; each MaxViT 28,543,736. Inputs: Harvard-GF data_2404; probabilities are symbolic.",
            fontsize=11, color="#555555", zorder=70, linespacing=1.5)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds_list = []
    for label in ax.texts:
        bounds = label.get_window_extent(renderer)
        assert bounds.x0 >= 0 and bounds.y0 >= 0, label.get_text()
        assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height, label.get_text()
        for previous, previous_bounds in bounds_list:
            assert not bounds.overlaps(previous_bounds), (label.get_text(), previous.get_text())
        bounds_list.append((label, bounds))
    print("Layout QA: all text inside canvas; no text-text overlaps.")
    outputs = []
    for extension in ("png", "svg", "pdf"):
        out = os.path.join(FIG_DIR, f"arch_main_nnsvg_style.{extension}")
        fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
        outputs.append(out)
        print("wrote", out)
    plt.close(fig)
    drive = os.environ.get("DRIVE_SYNC_DIR", "/content/drive/MyDrive/MasterBKDN/Thesis/architecture_figures")
    if os.path.isdir(drive):
        for out in outputs:
            shutil.copy2(out, drive)
        print("Drive sync:", drive)
    else:
        print("Drive sync pending: set DRIVE_SYNC_DIR to an existing mounted Drive folder.")


if __name__ == "__main__":
    main()
