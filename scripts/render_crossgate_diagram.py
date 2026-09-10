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
            fontsize=fs, color=tc, weight=weight, linespacing=1.3)


def arrow(ax, p1, p2, color="#444444", lw=1.4):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0))


def render(path):
    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    ax.set_xlim(0, 168)
    ax.set_ylim(-6, 86)
    ax.axis("off")

    box(ax, 2, 68, 30, 12, "Raw 3D OCT\n1\u00d7200\u00d7200\u00d7200", "#ffe599", fs=8.4)
    box(ax, 2, 50, 30, 12, "Enc3D\n(ConvNeXt3D)\n\u2192 e\u2083\u1d39", "#dbe9f6", fs=8.4)
    box(ax, 2, 34, 30, 11, "Proj: Linear+ReLU\np\u2083\u1d39 \u2208 \u211d^D", "#cfe2f3", fs=8.2)

    box(ax, 2, 20, 30, 10, "3 en-face views\nAIP / SlabAIP / SlabMIP", "#fff2cc", fs=7.8)
    box(ax, 2, 8, 30, 9, "Enc2D \u00d73\n(one per view) \u2192 e\u2082\u1d37\u2071", "#d9ead3", fs=7.6)
    box(ax, 2, -3, 30, 8, "Proj \u00d73\np\u2082\u1d37\u2071 \u2208 \u211d^D  (i=1..3)", "#cfe2f3", fs=7.4)

    box(ax, 40, 33, 34, 13, "q = p\u2083\u1d39\n(1 token, dim D)", "#d5a6bd", fs=8.4)
    box(ax, 40, 15, 34, 15, "K = V = LayerNorm(p\u2082\u1d37\u00b9..\u00b3)\n(3 tokens, dim D)", "#d5a6bd", fs=8.2)

    box(ax, 82, 26, 40, 30,
        "Multi-Head Cross-Attention\nheads h, dropout 0.1\n\n"
        "w = softmax(q K\u1d40 / \u221ad)\no = \u03a3\u2c7c w\u2c7c V\u2c7c  \u2208 \u211d^D",
        "#f4cccc", fs=8.2)
    box(ax, 82, 8, 40, 13,
        "gate \u03b1 (learnable scalar)\nz = p\u2083\u1d39 + \u03c3(\u03b1) \u00b7 o",
        "#ead1dc", fs=8.2)

    box(ax, 130, 27, 20, 18, "Head\nLinear(D \u2192 2)\n\u2192 logits", "#fff2cc", fs=8.2)
    box(ax, 154, 31, 12, 10, "softmax\n\u2192 class", "#f6b26b", fs=8.0)

    arrow(ax, (17, 68), (17, 62))
    arrow(ax, (17, 50), (17, 45))
    arrow(ax, (17, 34), (17, 30))
    arrow(ax, (17, 20), (17, 17))
    arrow(ax, (17, 8), (17, 5))
    arrow(ax, (32, 39.5), (40, 39.5))
    arrow(ax, (32, 1), (40, 1))
    arrow(ax, (40, 1), (57, 1))
    arrow(ax, (57, 1), (57, 15))
    arrow(ax, (57, 39.5), (82, 39.5))
    arrow(ax, (57, 22.5), (82, 22.5))
    arrow(ax, (102, 41), (102, 21))
    arrow(ax, (102, 14.5), (130, 36))
    arrow(ax, (150, 36), (154, 36))

    ax.text(84, 78, "CrossGate fusion (as implemented in FusionCrossGate)",
            fontsize=12, weight="bold", color=ECO)
    ax.text(2, 82, "3D branch", fontsize=9.5, color="#1f4e79", weight="bold")
    ax.text(2, 32, "2D branches", fontsize=9.5, color="#38761d", weight="bold")

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    render(os.path.join(FIG, "model_crossgate.png"))
