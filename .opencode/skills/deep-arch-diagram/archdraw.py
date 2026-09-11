import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

TEXT = "#111111"
MUTED = "#555555"
EDGE = "#444444"

PALETTE = {
    "input": "#ffe599",
    "proc": "#f6b26b",
    "conv": "#dbe9f6",
    "conv_deep": "#cfe2f3",
    "token": "#cfe2f3",
    "bottleneck": "#d5a6bd",
    "head": "#ead1dc",
    "note": "#f3f3f3",
    "removed": "#f3f3f3",
}


def _shade(color, f):
    r, g, b = mcolors.to_rgb(color)
    if f >= 0:
        return r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f
    return r * (1 + f), g * (1 + f), b * (1 + f)


def new_figure(xlim, ylim, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def volume(ax, x, y, w, h, dpx, dpy, fill, label=None, fs=7.0, lw=0.9, tc=TEXT, label_pos="in"):
    top = _shade(fill, 0.10)
    right = _shade(fill, -0.16)
    ax.add_patch(Rectangle((x + dpx, y), w, h, facecolor=right, edgecolor=EDGE, lw=lw))
    ax.add_patch(
        Polygon(
            [(x, y + h), (x + w, y + h), (x + w + dpx, y + h + dpy), (x + dpx, y + h + dpy)],
            closed=True,
            facecolor=top,
            edgecolor=EDGE,
            lw=lw,
        )
    )
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=EDGE, lw=lw))
    if label:
        cx, cy = x + w / 2, y + h / 2
        if label_pos == "below":
            ax.text(x + w / 2 + dpx / 2, y - 0.4, label, ha="center", va="top", fontsize=fs, color=tc, linespacing=1.25)
        else:
            ax.text(cx, cy, label, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.3)


def rbox(ax, x, y, w, h, text, fc, fs=7.2, lw=1.0, ec=MUTED, tc=TEXT, weight="normal", dashed=False, rounding=0.9):
    style = "round,pad=0.05,rounding_size=0.35"
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=style,
            facecolor=fc,
            edgecolor=ec,
            lw=lw,
            linestyle=(0, (3, 1.5)) if dashed else "solid",
        )
    )
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, weight=weight, linespacing=1.35)


def arrow(ax, p, q, color=MUTED, lw=1.5, style="-|>", ls="solid", ms=13, shrink=1.0):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle=style,
            mutation_scale=ms,
            color=color,
            lw=lw,
            linestyle=ls,
            shrinkA=shrink,
            shrinkB=shrink,
        )
    )


def text(ax, x, y, s, fs=7.2, color=MUTED, weight="normal", ha="center", va="center", style="normal"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color, weight=weight, style=style, linespacing=1.35)


def save(fig, path, dpi=250):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def line(ax, p, q, color=MUTED, lw=1.0, ls=(0, (3, 2))):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, linestyle=ls)
