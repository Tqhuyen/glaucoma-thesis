import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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


def overview(path):
    fig, ax = plt.subplots(figsize=(17.5, 10))
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 78)
    ax.axis("off")

    cube = (2, 52, 15, 16)
    proj = (2, 27, 40, 12)
    views = [(5, 13, 9, 9), (17, 13, 9, 9), (29, 13, 9, 9)]
    e3d_box = (55, 50, 40, 16)
    e2d_box = (55, 10, 40, 18)
    ch3 = (104, 53, 10, 8)
    ch2 = (104, 15, 10, 8)
    fus = (124, 24, 33, 34)

    box(ax, *cube, "Raw 3D OCT volume\n1\u00d7200\u00d7200\u00d7200\n(uint8 \u2192 /255)", "#ffe599")
    box(ax, *proj, "Group A \u2014 report-style 2D projection\n"
                   "1) auto-detect depth axis   2) RNFL-band slab (peak\u00b116)\n"
                   "3) en-face reduce over depth \u2192 three 200\u00d7200 views", "#f6b26b", fs=7.6)
    vn = ["aip_full\n(mean, full depth)", "slab_aip\n(mean, RNFL band)", "slab_mip\n(max, RNFL band)"]
    for (x, y, w, h), n in zip(views, vn):
        box(ax, x, y, w, h, n, "#fff2cc", fs=6.4)

    box(ax, *e3d_box, "3D branch \u2014 Enc3D\nGroupNorm residual CNN\nfeatures (24,48,96,192)\n"
                      "anisotropic stride: later/less on depth\n(RNFL kept), GroupNorm (no BN)",
        "#dbe9f6", fs=7.8)
    box(ax, *e2d_box, "2D branches \u2014 Enc2D \u00d7 3  (one per view)\n"
                      "same ResNet-18-style architecture, INDEPENDENT weights\n"
                      "1-ch input 200\u00d7200, features (64,128,256,512)\n"
                      "GroupNorm residual blocks; no ImageNet pretrain",
        "#d9ead3", fs=7.8)
    box(ax, *ch3, "e\u2083\u1d39\n192", "#d5a6bd", fs=7.6)
    box(ax, *ch2, "e\u2082\u1d37\u00b9..\u00b3\n512", "#d5a6bd", fs=7.6)

    box(ax, *fus, "Fusion + classifier\n"
                  "project each embedding to latent 256 (Linear+ReLU)\n"
                  "then ONE of:\n"
                  "\u2022  concat: z=[proj(e\u2081);\u2026;proj(e\u2084)]  (4\u00b7256)\n"
                  "\u2022  add:    z = \u03a3\u1d62 proj(e\u1d62)\n"
                  "\u2022  mul:    z = \u220f\u1d62 proj(e\u1d62)\n"
                  "\u2022  attn:   CLS + self-attention over {proj(e\u1d62)}\n"
                  "                 (Transformer encoder, 1 layer, 8 heads)\n"
                  "head: Linear \u2192 2 logits  \u2192 softmax\n"
                  "CE loss (glaucoma / no-glaucoma)", "#ead1dc", fs=7.7)

    arrow(ax, (17, 60), (55, 58))
    arrow(ax, (9, 52), (9, 39.5))
    for (x, y, w, h), xc in zip(views, (9.5, 21.5, 33.5)):
        arrow(ax, (xc, 27), (xc, y + h + 0.2))
        arrow(ax, (x + w + 0.2, y + h / 2), (55, y + h / 2 + 0.0))
    arrow(ax, (95, 58), (104, 57))
    arrow(ax, (95, 19), (104, 19))
    arrow(ax, (114, 57), (124, 44))
    arrow(ax, (114, 19), (124, 34))
    box(ax, 2, 0, 120, 6.5,
        "Ablation baselines \u2014 single3d: head straight on e\u2083\u1d39 (2D branches skipped); "
        "single2d\u1d62: head on one e\u2082\u1d37\u1d62 (3D + other views skipped) \u2192 isolates each view and the 3D stream",
        "#f3f3f3", fs=7.0)
    ax.text(2, 76, "(a) Group-A multi-branch architecture: raw volume \u2192 3D CNN  +  "
                   "depth-projections \u2192 3 independent 2D CNNs \u2192 fusion \u2192 glaucoma head",
            fontsize=11.5, weight="bold")
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def fusion_panel(ax, title, eq, note, tokens, opstr):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 48)
    ax.axis("off")
    ax.text(2, 44, title, fontsize=13, weight="bold", color=ECO)
    n = len(tokens)
    step = 88 / n
    for i, t in enumerate(tokens):
        x = 4 + i * step
        box(ax, x, 30, 9, 7, t, "#d5a6bd", fs=7.5)
        arrow(ax, (x + 4.5, 30), (x + 4.5, 24.5))
    box(ax, 2, 18, 96, 6, opstr, "#ead1dc", fs=8.6)
    ax.text(2, 10.5, eq, fontsize=10.5, family="DejaVu Sans")
    ax.text(2, 3.5, note, fontsize=8.6, color="#555555", style="italic")


def fusion(path):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10.5))
    specs = [
        ("(a) Concatenation",
         "z = [ proj(e\u2083\u1d39); proj(e\u2082\u1d37\u00b9); proj(e\u2082\u1d37\u00b2); proj(e\u2082\u1d37\u00b3) ]\nz \u2208 R^{4\u00d7256} \u2192 Linear(4\u00b7256 \u2192 2)",
         "Keeps every branch feature; dimension grows 4\u00d7 \u2014 needs data / regularisation.",
         ["e\u2083\u1d39", "e\u2082\u1d37\u00b9", "e\u2082\u1d37\u00b2", "e\u2082\u1d37\u00b3"],
         "proj = Linear + ReLU, then concatenate \u2192 4\u00b7256 \u2192 head Linear(1024\u21922)"),
        ("(b) Element-wise addition",
         "z = \u03a3\u1d62 proj(e\u1d62) ,  z \u2208 R^{256} \u2192 Linear(256 \u2192 2)",
         "Fixed latent size; implicit assumption branches share a common semantic space.",
         ["e\u2083\u1d39", "e\u2082\u1d37\u00b9", "e\u2082\u1d37\u00b2", "e\u2082\u1d37\u00b3"],
         "proj = Linear + ReLU, then sum over branches \u2192 256 \u2192 head Linear(256\u21922)"),
        ("(c) Element-wise multiplication",
         "z = \u220f\u1d62 proj(e\u1d62) ,  z \u2208 R^{256} \u2192 Linear(256 \u2192 2)",
         "Acts like feature gating / co-occurrence; may be harder to optimise.",
         ["e\u2083\u1d39", "e\u2082\u1d37\u00b9", "e\u2082\u1d37\u00b2", "e\u2082\u1d37\u00b3"],
         "proj = Linear + ReLU, then product over branches \u2192 256 \u2192 head Linear(256\u21922)"),
        ("(d) Self-attention fusion",
         "tokens = [CLS ; proj(e\u2083\u1d39) ; proj(e\u2082\u1d37\u00b9) ; \u2026]\nTransformer encoder (1 layer, 8 heads, d=256)\ny = Linear(CLS \u2192 2)",
         "Branches can attend to each other (dynamic weighting); most flexible.",
         ["CLS", "e\u2083\u1d39", "e\u2082\u1d37\u00b9", "\u2026"],
         "proj = Linear + ReLU \u2192 prepend CLS token \u2192 Transformer encoder (d=256) \u2192 head Linear(256\u21922)"),
    ]
    for ax, sp in zip(axs.ravel(), specs):
        fusion_panel(ax, *sp)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    overview(os.path.join(FIG, "model_multiview_overview.png"))
    fusion(os.path.join(FIG, "model_fusion_variants.png"))
