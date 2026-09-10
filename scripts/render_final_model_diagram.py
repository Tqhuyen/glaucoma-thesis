import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
CB = "#555555"
ECO = "#111111"


def box(ax, x, y, w, h, text, fc, fs=8.2, ec=CB, lw=1.2, weight="normal", tc=ECO):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.8",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight=weight, linespacing=1.35)


def arrow(ax, p1, p2, color="#444444", lw=1.4):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0))


def render(path):
    fig, ax = plt.subplots(figsize=(16.5, 9.2))
    ax.set_xlim(0, 200)
    ax.set_ylim(-6, 104)
    ax.axis("off")

    box(ax, 2, 80, 34, 15, "Raw 3D OCT\n1\u00d7200\u00d7200\u00d7200\n(uint8 \u2192 /255)", "#ffe599", fs=8.4)
    box(ax, 42, 80, 54, 15,
        "3D branch \u2014 ResNeXt3D\nGroupNorm residual; anisotropic stride\n"
        "(2,2,1),(2,2,2)\u00d73 ; ch 32\u219264\u2192128\u2192192\n"
        "0.64 M params", "#dbe9f6", fs=8.0)
    box(ax, 102, 80, 30, 15, "Proj\nLinear+ReLU\n192 \u2192 256", "#cfe2f3", fs=8.2)

    box(ax, 2, 57, 34, 16,
        "2 en-face views\n\u2022 Slab MIP\n\u2022 AIP full\n224\u00d7224", "#fff2cc", fs=8.0)
    box(ax, 42, 53, 54, 24,
        "2D branches \u2014 MaxViT-Tiny \u00d72\nImageNet pretrained, INDEPENDENT weights\n"
        "512-d feature per view\n28.54 M params each", "#d9ead3", fs=8.0)
    box(ax, 102, 53, 30, 24, "Proj \u00d72\nLinear+ReLU\n512 \u2192 256", "#cfe2f3", fs=8.2)

    box(ax, 138, 40, 44, 44, "", "#f4cccc", fs=8.2)
    ax.text(160, 79, "CrossGate fusion", ha="center", va="center", fontsize=9.6, weight="bold")
    ax.text(160, 71, "q = p\u2083\u1d39  (1 token)", ha="center", va="center", fontsize=8.2)
    ax.text(160, 64, "K = V = LayerNorm(p\u2082\u1d37\u00b9..\u00b2)\n(2 tokens)", ha="center", va="center", fontsize=8.2)
    ax.text(160, 54, "Multi-Head Cross-Attention\n8 heads, dropout 0.1\nw = softmax(qK\u1d40/\u221ad)",
            ha="center", va="center", fontsize=8.2)
    ax.text(160, 44, "z = p\u2083\u1d39 + \u03c3(\u03b1)\u00b7o\n(\u03b1 learnable, \u03c3(0.5)=0.62)",
            ha="center", va="center", fontsize=8.2)
    ax.text(160, 88, "0.26 M params", ha="center", va="center", fontsize=8.0, color="#333333")

    box(ax, 186, 48, 13, 22, "Head\nLinear\n256 \u2192 2\n\u2192 logits", "#fff2cc", fs=8.0)
    box(ax, 186, 24, 13, 14, "softmax\n\u2192 class", "#f6b26b", fs=8.0)

    arrow(ax, (19, 80), (19, 73))
    arrow(ax, (19, 57), (19, 50))
    arrow(ax, (36, 87.5), (42, 87.5))
    arrow(ax, (36, 65), (42, 65))
    arrow(ax, (96, 87.5), (102, 87.5))
    arrow(ax, (96, 65), (102, 65))
    arrow(ax, (132, 87.5), (160, 87.5))
    arrow(ax, (160, 87.5), (160, 84))
    arrow(ax, (132, 65), (160, 65))
    arrow(ax, (160, 65), (160, 66))
    arrow(ax, (182, 62), (186, 60))
    arrow(ax, (192, 48), (192, 38))

    ax.text(1, 101, "Main model: 2\u00d72D (MaxViT-Tiny) + 1\u00d73D (ResNeXt3D) + CrossGate  \u2014  58.30 M params",
            fontsize=12.5, weight="bold", color=ECO)
    ax.text(2, 75, "3D branch", fontsize=9.5, color="#1f4e79", weight="bold")
    ax.text(2, 49, "2D branches", fontsize=9.5, color="#38761d", weight="bold")

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    render(os.path.join(FIG, "model_final_2x2d_3d_crossgate.png"))
