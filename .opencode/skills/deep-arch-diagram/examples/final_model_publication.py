import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox

SKILL = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SKILL))
sys.path.insert(0, str(ROOT / "scripts"))
import archdraw as ad

OUT = ROOT / "figures" / "architecture_publication"
BLUE = "#d4e3ed"
GREEN = "#dce8d5"
LILAC = "#e4deed"
SAND = "#f3dfbf"
ROSE = "#efd9d7"
INK = "#20272c"
MUTED = "#52606b"
BOXES = []
INSIDE = []
SEGMENTS = []


def text(ax, x, y, label, fs=11, color=INK, **kwargs):
    ad.text(ax, x, y, label, fs=fs, color=color, **kwargs)
    ax.texts[-1].set_zorder(8)


def block(ax, x, y, w, h, label, color, fs=11):
    ad.rbox(ax, x, y, w, h, label, color, fs=fs, lw=0.8, ec=INK, tc=INK)
    BOXES.append((ax, (x, y, x + w, y + h), label))
    INSIDE.append((ax, ax.texts[-1], (x, y, x + w, y + h)))


def region(ax, x, y, w, h, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.5,rounding_size=4", facecolor=color, edgecolor="none", zorder=-5
        )
    )


def route(ax, *points, dashed=False, head=True, color=INK):
    for index, (p, q) in enumerate(zip(points, points[1:])):
        last = index == len(points) - 2
        ad.arrow(
            ax,
            p,
            q,
            color=color,
            lw=0.85,
            ms=9,
            shrink=0,
            style="->" if last and head else "-",
            ls=(0, (4, 3)) if dashed else "solid",
        )
        SEGMENTS.append((ax, p, q))


def junction(ax, x, y):
    ax.plot(x, y, "o", ms=3, color=INK, zorder=6)


def operator(ax, x, y, symbol):
    ax.plot(x, y, "o", ms=18, markerfacecolor="white", markeredgecolor=INK, markeredgewidth=0.85, zorder=6)
    text(ax, x, y, symbol, fs=13)


def setup(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(-12, 112)
    ax.axis("off")


def caption(ax, value):
    text(ax, 50, -7, value, fs=13, color="#244c67")


def overview(ax):
    setup(ax)
    region(ax, 1, 0, 98, 106, "#f6f7f7")
    columns = (17, 50, 83)
    inputs = ("OCT volume", "Slab MIP", "Full AIP")
    shapes = (r"$1\times200^3$", r"$1\times224^2$", r"$1\times224^2$")
    backbones = ("ResNeXt3D\n+ global pooling", "MaxViT-Tiny\n+ norm / pooling", "MaxViT-Tiny\n+ norm / pooling")
    dims = (192, 512, 512)
    for x, label, shape, encoder, dim in zip(columns, inputs, shapes, backbones, dims):
        block(ax, x - 13, 9, 26, 9, label, SAND)
        text(ax, x, 4, shape, fs=11)
        block(ax, x - 13, 31, 26, 17, encoder, BLUE if dim == 192 else GREEN, fs=10.5)
        route(ax, (x, 18), (x, 31))
        block(ax, x - 13, 59, 26, 12, f"Linear {dim} to 256\nReLU", LILAC, fs=10)
        route(ax, (x, 48), (x, 59))
        text(ax, x + 7, 53.5, str(dim), fs=10, color=MUTED)
    block(ax, 40, 78, 56, 8, r"Stack 2D tokens: $T_{2D}\in\mathbb{R}^{B\times2\times256}$", BLUE, fs=10)
    route(ax, (50, 71), (50, 78))
    route(ax, (83, 71), (83, 78))
    block(ax, 36, 92, 28, 10, "CrossGate", SAND, fs=12)
    route(ax, (17, 71), (17, 97), (36, 97))
    text(ax, 11, 83, r"$p_3$", fs=12)
    route(ax, (68, 86), (68, 97), (64, 97))
    route(ax, (50, 102), (50, 107))
    text(ax, 68, 108, r"$z\in\mathbb{R}^{B\times256}$", fs=11)
    caption(ax, "(a) Three-branch feature extraction")


def crossgate(ax):
    setup(ax)
    region(ax, 2, 1, 96, 106, "#f8eddf")
    text(ax, 18, 6, r"$p_3$", fs=14)
    text(ax, 82, 6, r"$T_{2D}$", fs=14)
    route(ax, (18, 11), (18, 50), (24, 50))
    junction(ax, 18, 16)
    route(ax, (18, 16), (7, 16), (7, 93), (46.5, 93))
    text(ax, 13, 37, r"$Q$", fs=12)
    block(ax, 66, 23, 31, 10, "LayerNorm", BLUE, fs=11)
    route(ax, (82, 11), (82, 23))
    route(ax, (82, 33), (82, 50), (76, 50))
    text(ax, 89, 39, r"$K,V$", fs=11)
    block(ax, 24, 43, 52, 20, "Multi-head cross-attention\n8 heads; dropout 0.1", SAND, fs=10.5)
    route(ax, (50, 63), (50, 70.5))
    operator(ax, 50, 74, r"$\times$")
    block(ax, 68, 69, 29, 10, r"$\sigma(\alpha)$", LILAC, fs=14)
    route(ax, (68, 74), (53.5, 74))
    route(ax, (50, 77.5), (50, 89.5))
    operator(ax, 50, 93, "+")
    route(ax, (50, 96.5), (50, 101))
    text(ax, 50, 105, r"$z=p_3+\sigma(\alpha)\,o$", fs=12)
    text(ax, 84, 84, "Scalar gate", fs=10, color=MUTED)
    caption(ax, "(b) CrossGate module")


def head(ax):
    setup(ax)
    region(ax, 5, 1, 90, 106, "#eaf1f5")
    text(ax, 50, 7, r"$z\;(B,256)$", fs=13)
    block(ax, 12, 26, 76, 12, "Linear 256 to 2", GREEN, fs=12)
    route(ax, (50, 11), (50, 26))
    block(ax, 12, 49, 76, 12, "Raw logits (B, 2)", BLUE, fs=11)
    route(ax, (50, 38), (50, 49))
    route(ax, (50, 61), (50, 74), dashed=True)
    block(ax, 12, 74, 76, 13, "Divide by T\nSoftmax", LILAC, fs=12)
    route(ax, (50, 87), (50, 95), dashed=True)
    text(ax, 50, 102, r"$P(y=0),\;P(y=1)$", fs=12)
    caption(ax, "(c) Classification head")


def encoder3d(ax):
    setup(ax)
    region(ax, 2, 0, 96, 107, "#edf3f7")
    text(ax, 50, 105, r"$x_3:\;1\times200^3$", fs=12)
    labels = [
        ("Stem: 3D Conv + GN + ReLU", r"$32\times200^3$", BLUE),
        ("1 residual block, stride 1", r"$32\times200^3$", GREEN),
        ("2 residual blocks; strides 2, 1", r"$64\times100^3$", GREEN),
        ("2 residual blocks; strides 2, 1", r"$128\times50^3$", GREEN),
        ("2 residual blocks; strides 2, 1", r"$192\times25^3$", GREEN),
        ("Adaptive average pool + flatten", "192-D embedding", LILAC),
    ]
    previous = 101
    for y, (name, shape, color) in zip((85, 69, 53, 37, 21, 5), labels):
        block(ax, 9, y, 82, 11, name + "\n" + shape, color, fs=10.5)
        route(ax, (50, previous), (50, y + 11))
        previous = y
    caption(ax, "(a) ResNeXt3D encoder")


def residual(ax):
    setup(ax)
    region(ax, 1, 0, 98, 107, "#f8eddf")
    text(ax, 61, 105, r"$x$", fs=14)
    route(ax, (61, 101), (61, 96))
    junction(ax, 61, 99)
    route(ax, (61, 99), (12, 99), (12, 70), head=True)
    block(ax, 1, 43, 23, 27, "Identity\nor\nConv 1x1x1\n(stride s)\n+ GN", BLUE, fs=9.5)
    route(ax, (12, 43), (12, 18), (57, 18))
    labels = [
        (85, "Conv 1x1x1", GREEN),
        (71, "GroupNorm + ReLU", BLUE),
        (57, "Conv 3x3x3\n8 groups; stride s", GREEN),
        (43, "GroupNorm + ReLU", BLUE),
        (29, "Conv 1x1x1", GREEN),
    ]
    previous = None
    for y, name, color in labels:
        block(ax, 32, y, 58, 11, name, color, fs=10.5)
        if previous is not None:
            route(ax, (61, previous), (61, y + 11))
        previous = y
    route(ax, (61, 29), (61, 21.5))
    operator(ax, 61, 18, "+")
    block(ax, 32, 2, 58, 9, "ReLU", LILAC, fs=11)
    route(ax, (61, 14.5), (61, 11))
    caption(ax, "(b) Executed residual block")


def maxvit(ax):
    setup(ax)
    region(ax, 2, 0, 96, 107, "#eef3e9")
    text(ax, 50, 105, r"$v_i:\;1\times224^2$", fs=12)
    labels = [
        ("Stem: Conv s2 + norm / act + Conv s1", r"$64\times112^2$", BLUE),
        ("MaxViT stage 0: 2 blocks", r"$64\times56^2$", GREEN),
        ("MaxViT stage 1: 2 blocks", r"$128\times28^2$", GREEN),
        ("MaxViT stage 2: 5 blocks", r"$256\times14^2$", GREEN),
        ("MaxViT stage 3: 2 blocks", r"$512\times7^2$", GREEN),
        ("LayerNorm2d + average pool / flatten", "512-D embedding", LILAC),
    ]
    previous = 101
    for y, (name, shape, color) in zip((85, 69, 53, 37, 21, 5), labels):
        block(ax, 6, y, 88, 11, name + "\n" + shape, color, fs=10.5)
        route(ax, (50, previous), (50, y + 11))
        previous = y
    caption(ax, "(c) MaxViT-Tiny encoder (each view)")


def end_to_end(ax):
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 204)
    ax.axis("off")
    region(ax, 2, 103, 116, 92, "#f7f3ed")
    region(ax, 2, 50, 116, 51, "#edf3f7")
    text(ax, 60, 200, "End-to-end OCT glaucoma classification", fs=16)

    block(ax, 45, 183, 30, 9, r"Raw OCT: uint8, $200^3$", SAND, fs=11.5)
    route(ax, (60, 183), (60, 179), head=False)
    junction(ax, 60, 179)
    route(ax, (60, 179), (23, 179), (23, 174))
    route(ax, (60, 179), (87, 179), (87, 174))
    block(ax, 7, 162, 32, 12, "3D source selection\nRaw OR bilateral", SAND, fs=12)
    block(ax, 67, 162, 40, 12, "Raw-derived view cache\nMove depth axis last", BLUE, fs=12)

    block(ax, 7, 142, 32, 12, "Move depth axis last\nusing raw-derived axis", BLUE, fs=11.5)
    route(ax, (23, 162), (23, 154))
    route(ax, (87, 162), (87, 158), head=False)
    junction(ax, 87, 158)
    route(ax, (87, 158), (63, 158), (63, 154))
    route(ax, (87, 158), (99, 158), (99, 154))
    block(ax, 47, 142, 32, 12, "Slab MIP\nPeak-centered depth slab", SAND, fs=11.5)
    block(ax, 83, 142, 32, 12, "Full AIP\nMean over all depth", SAND, fs=12)
    for x, label in (
        (23, r"Keep full resolution: $200^3$"),
        (63, r"Resize to $224\times224$"),
        (99, r"Resize to $224\times224$"),
    ):
        block(ax, x - 16, 124, 32, 10, label, BLUE, fs=11.5)
        route(ax, (x, 142), (x, 134))
        route(ax, (x, 124), (x, 117))

    block(
        ax,
        7,
        106,
        108,
        11,
        "Paired augmentation (training only), then float32 / 255\n"
        r"$x_3:(B,1,200,200,200)$; views: $(B,2,1,224,224)$",
        LILAC,
        fs=12,
    )
    for x, label, dim, color in (
        (23, "ResNeXt3D\nGAP + flatten: 192-D", 192, BLUE),
        (63, "MaxViT-Tiny (view 0)\nNorm + pool: 512-D", 512, GREEN),
        (99, "MaxViT-Tiny (view 1)\nNorm + pool: 512-D", 512, GREEN),
    ):
        block(ax, x - 16, 85, 32, 12, label, color, fs=12)
        route(ax, (x, 106), (x, 97))
        block(ax, x - 16, 67, 32, 10, f"Linear {dim} to 256\nReLU", LILAC, fs=12)
        route(ax, (x, 85), (x, 77))

    block(ax, 55, 52, 60, 10, r"Stack 2D tokens: $T_{2D}$" + "\n(B, 2, 256)", BLUE, fs=12)
    route(ax, (63, 67), (63, 62))
    route(ax, (99, 67), (99, 62))
    block(ax, 7, 32, 38, 13, "CrossGate: 3D query + 2D K/V\n" + r"$z=p_3+\sigma(\alpha)\,o$", SAND, fs=11.5)
    route(ax, (23, 67), (23, 45))
    text(ax, 30, 57, r"$p_3:(B,256)$", fs=11.5)
    route(ax, (85, 52), (85, 49), (36, 49), (36, 45))

    block(ax, 55, 32, 28, 13, "Linear 256 to 2\nRaw logits (B, 2)", GREEN, fs=11.5)
    route(ax, (45, 38.5), (55, 38.5))
    block(ax, 93, 32, 22, 13, "Divide by T\nSoftmax", LILAC, fs=12)
    route(ax, (83, 38.5), (93, 38.5), dashed=True)
    block(ax, 68, 7, 47, 13, r"$P(y=0),\;P(y=1)$" + "\n0: non-glaucoma; 1: glaucoma", BLUE, fs=12)
    route(ax, (104, 32), (104, 20), dashed=True)
    block(
        ax,
        7,
        7,
        48,
        13,
        "Reported decision\n" + r"$\hat{y}=1\ \mathrm{if}\ P(y=1)\geq\tau;\ \mathrm{else}\ 0$",
        ROSE,
        fs=12,
    )
    route(ax, (68, 13.5), (55, 13.5), dashed=True)


def end_to_end_thesis(ax):
    ax.set_xlim(0, 120)
    ax.set_ylim(-26, 156)
    ax.axis("off")
    region(ax, 2, 72, 116, 77, "#f7f3ed")
    region(ax, 2, 18, 116, 51, "#edf3f7")
    region(ax, 2, -19, 116, 35, "#f7f1e9")
    text(ax, 60, 152, "End-to-end OCT classification", fs=13)

    block(ax, 42, 136, 36, 11, "Raw OCT: uint8\n" + r"$200\times200\times200$", SAND, fs=10.5)
    route(ax, (60, 136), (60, 132), head=False)
    junction(ax, 60, 132)
    route(ax, (60, 132), (22, 132), (22, 128))
    route(ax, (60, 132), (81.5, 132), (81.5, 128))
    block(ax, 5, 113, 34, 15, "Raw / bilateral\nDepth-last\n" + r"$1\times200^3$", SAND, fs=10.5)
    block(ax, 48, 113, 67, 15, "Raw-derived views\nDepth-last orientation", BLUE, fs=10.5)
    route(ax, (81.5, 113), (81.5, 110), head=False)
    junction(ax, 81.5, 110)
    route(ax, (81.5, 110), (60, 110), (60, 105))
    route(ax, (81.5, 110), (98, 110), (98, 105))
    block(ax, 43, 92, 34, 13, "Slab MIP\n" + r"Resize: $224^2$", SAND, fs=10.5)
    block(ax, 81, 92, 34, 13, "Full AIP\n" + r"Resize: $224^2$", SAND, fs=10.5)

    block(ax, 5, 74, 110, 12, "Paired augmentation (training only)\nfloat32 / 255", LILAC, fs=10.5)
    for x, previous in ((22, 113), (60, 92), (98, 92)):
        route(ax, (x, previous), (x, 86))
    for x, dim, encoder, color in (
        (22, 192, "ResNeXt3D\nGAP / flatten\n192-D", BLUE),
        (60, 512, "MaxViT-Tiny\nLN2d + GAP\n512-D", GREEN),
        (98, 512, "MaxViT-Tiny\nLN2d + GAP\n512-D", GREEN),
    ):
        block(ax, x - 17, 52, 34, 14, encoder, color, fs=10.5)
        route(ax, (x, 74), (x, 66))
        block(ax, x - 17, 34, 34, 11, f"Linear {dim} to 256\nReLU", LILAC, fs=10.5)
        route(ax, (x, 52), (x, 45))
    block(ax, 50, 20, 65, 9, r"Stack $T_{2D}$: (B, 2, 256)", BLUE, fs=10.5)
    route(ax, (60, 34), (60, 29))
    route(ax, (98, 34), (98, 29))
    block(ax, 5, 3, 45, 12, "CrossGate\n" + r"$z=p_3+\sigma(\alpha)o$", SAND, fs=11)
    route(ax, (22, 34), (22, 15))
    text(ax, 32, 24, r"$p_3$", fs=11)
    route(ax, (82.5, 20), (82.5, 17), (40, 17), (40, 15))
    block(ax, 62, 3, 53, 12, "Linear 256 to 2\nRaw logits (B, 2)", GREEN, fs=10.5)
    route(ax, (50, 9), (62, 9))
    block(ax, 62, -17, 53, 13, "Test: logits / T, softmax\n" + r"$P(y=0),\;P(y=1)$", LILAC, fs=10.5)
    route(ax, (88.5, 3), (88.5, -4), dashed=True)
    block(ax, 5, -17, 45, 13, r"1 if $P(y=1)\geq\tau$" + "\notherwise 0", ROSE, fs=10.5)
    route(ax, (62, -10.5), (50, -10.5), dashed=True)
    text(ax, 60, -23, "Dashed: notebook test-time processing.", fs=9.5, color=MUTED)


def verify():
    import final_model as fm
    import timm
    import torch

    torch.set_num_threads(min(6, os.cpu_count() or 1))
    model = fm.FinalModel(enc2d_pretrained=False).eval()
    captured = {}
    hooks = []
    for i, stage in enumerate(model.enc2ds[0].net.stages):
        hooks.append(stage.register_forward_hook(lambda m, args, out, i=i: captured.update({str(i): list(out.shape)})))
    norm = model.enc2ds[0].net.norm
    called_g3 = []
    for b in model.enc3d.stages:
        hooks.append(b.g3.register_forward_hook(lambda *args: called_g3.append(True)))
    with torch.inference_mode():
        features = model.enc2ds[0].net.forward_features(torch.zeros(1, 1, 224, 224))
        logits = model(torch.zeros(1, 1, 16, 16, 16), torch.zeros(1, 2, 1, 224, 224))
    for h in hooks:
        h.remove()
    assert not called_g3, "Residual g3 is now executed: update the block diagram."
    assert tuple(features.shape) == (1, 512, 7, 7)
    assert tuple(logits.shape) == (1, 2)
    assert [len(s.blocks) for s in model.enc2ds[0].net.stages] == [2, 2, 5, 2]
    assert captured == {"0": [1, 64, 56, 56], "1": [1, 128, 28, 28], "2": [1, 256, 14, 14], "3": [1, 512, 7, 7]}
    assert type(norm).__name__ == "LayerNorm2d"
    assert model.fusion.attn.num_heads == 8 and model.fusion.attn.dropout == 0.1
    assert model.fusion.gate.item() == 0.5
    assert [(p.fc.in_features, p.fc.out_features) for p in model.projs] == [(192, 256), (512, 256), (512, 256)]
    with torch.inference_mode():
        p3, tokens = torch.randn(2, 256), torch.randn(2, 2, 256)
        expected = (
            p3
            + model.fusion.gate.sigmoid()
            * model.fusion.attn(p3[:, None], model.fusion.norm(tokens), model.fusion.norm(tokens))[0][:, 0]
        )
        torch.testing.assert_close(model.fusion(p3, tokens), expected)
    total = sum(p.numel() for p in model.parameters())
    per2d = sum(p.numel() for p in model.enc2ds[0].parameters())
    unused = sum(p.numel() for b in model.enc3d.stages for p in b.g3.parameters())
    assert total == 58304467 and per2d == 28543736
    encoder = model.enc3d.to("meta")
    sizes = []
    x = encoder.stem(torch.empty(1, 1, 200, 200, 200, device="meta"))
    for block3d in encoder.stages:
        x = block3d(x)
        sizes.append(list(x.shape))
    assert [s[-1] for s in sizes] == [200, 100, 100, 50, 50, 25, 25]
    assert [s[1] for s in sizes] == [32, 64, 64, 128, 128, 192, 192]
    paths = [ROOT / "scripts/final_model.py", ROOT / "notebooks/3d_glaucoma_final_2x2d_3d_crossgate.ipynb"]
    return {
        "torch": torch.__version__,
        "timm": timm.__version__,
        "parameters": total,
        "parameters_per_maxvit": per2d,
        "unused_g3_parameters": unused,
        "3d_stage_shapes_meta": sizes,
        "maxvit_stage_shapes_cpu": captured,
        "maxvit_final_norm": type(norm).__name__,
        "crossgate_equation_test": "passed",
        "full_model_test_input": "synthetic 16^3 volume + two 224^2 images; pretrained=False",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }


def layout_check(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    issues = []
    labels = []
    for label in [*fig.texts, *(label for ax in fig.axes for label in ax.texts)]:
        bounds = label.get_window_extent(renderer)
        if not fig.bbox.contains(bounds.x0, bounds.y0) or not fig.bbox.contains(bounds.x1, bounds.y1):
            issues.append("outside canvas: " + label.get_text())
        for other, other_bounds in labels:
            if bounds.overlaps(other_bounds):
                issues.append("text overlap: " + label.get_text() + " / " + other.get_text())
        labels.append((label, bounds))
    for ax, label, rect in INSIDE:
        if ax.figure is not fig:
            continue
        window = Bbox.from_extents(*rect).transformed(ax.transData)
        extent = label.get_window_extent(renderer)
        if extent.x0 < window.x0 or extent.x1 > window.x1 or extent.y0 < window.y0 or extent.y1 > window.y1:
            issues.append("text outside block: " + label.get_text())
    boxes = [(a, r, s) for a, r, s in BOXES if a.figure is fig]
    for i, (ax, rect, label) in enumerate(boxes):
        for other_ax, other_rect, other_label in boxes[i + 1 :]:
            if ax is other_ax and Bbox.from_extents(*rect).overlaps(Bbox.from_extents(*other_rect)):
                issues.append("block overlap: " + label + " / " + other_label)
    for ax, p, q in SEGMENTS:
        if ax.figure is not fig:
            continue
        p, q = ax.transData.transform(p), ax.transData.transform(q)
        for label, bounds in labels:
            if label.axes is not ax:
                continue
            if bounds.expanded(0.98, 0.98).contains(*p) or bounds.expanded(0.98, 0.98).contains(*q):
                issues.append("arrow endpoint intersects label: " + label.get_text())
            if abs(p[0] - q[0]) < 0.1 and bounds.x0 < p[0] < bounds.x1:
                if max(min(p[1], q[1]), bounds.y0) < min(max(p[1], q[1]), bounds.y1):
                    issues.append("vertical arrow crosses label: " + label.get_text())
            if abs(p[1] - q[1]) < 0.1 and bounds.y0 < p[1] < bounds.y1:
                if max(min(p[0], q[0]), bounds.x0) < min(max(p[0], q[0]), bounds.x1):
                    issues.append("horizontal arrow crosses label: " + label.get_text())
    if issues:
        raise AssertionError("\n".join(issues))
    return {
        "texts_checked": len(labels),
        "blocks_checked": len(boxes),
        "overlaps": 0,
        "page_inches": fig.get_size_inches().tolist(),
        "minimum_font_pt": min(label.get_fontsize() for label, _ in labels),
    }


def render(name, functions, widths, size, footnote=None):
    fig, axes = plt.subplots(1, len(functions), figsize=size, gridspec_kw={"width_ratios": widths})
    fig.subplots_adjust(left=0.02, right=0.985, top=0.975, bottom=0.075 if not footnote else 0.13, wspace=0.065)
    if len(functions) == 1:
        axes = [axes]
    for ax, draw in zip(axes, functions):
        draw(ax)
    if footnote:
        fig.text(0.025, 0.025, footnote, fontsize=10.5, color=MUTED, linespacing=1.5)
    qa = layout_check(fig)
    files = []
    for extension in ("png", "svg", "pdf"):
        path = OUT / f"{name}.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        files.append(path)
        print("Wrote", path)
    plt.close(fig)
    print(name, qa)
    return files, qa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-dir", type=Path, default=None)
    args = parser.parse_args()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)
    verified = verify()
    files = []
    qa = {}
    configs = [
        (
            "01_model_plate",
            [overview, crossgate, head],
            [1.6, 1, 0.8],
            (15, 7.4),
            "B denotes batch size. The two MaxViT encoders have independent weights. Labels: 0 = non-glaucoma, 1 = glaucoma.\n"
            "Dashed arrows: post-hoc test calibration; T is fitted on validation logits. Training uses class-weighted cross-entropy on raw logits.",
        ),
        ("02_crossgate", [crossgate], [1], (6.5, 7.5), None),
        (
            "03_encoder_details",
            [encoder3d, residual, maxvit],
            [1.05, 0.9, 1.15],
            (15, 8.5),
            "Each MaxViT block: MBConv, window attention + FFN, grid attention + FFN. The two encoders have independent weights.\n"
            "Executed graph: no GroupNorm after the final 1x1x1 convolution in the residual branch (g3 is declared but unused).",
        ),
        ("04_resnext3d", [encoder3d], [1], (6.5, 8), None),
        ("05_residual_block", [residual], [1], (6.5, 8), None),
        ("06_maxvit", [maxvit], [1], (6.5, 8), None),
        ("07_overview", [overview], [1], (8, 8), None),
        (
            "08_output_head",
            [head],
            [1],
            (5.5, 7.5),
            "Dashed path: post-hoc calibration.\nT is fitted on validation logits; it is not a model layer.",
        ),
        (
            "09_end_to_end",
            [end_to_end],
            [1],
            (12.5, 14.5),
            "Bilateral affects only the 3D input; both 2D views remain raw-derived. Two MaxViT encoders have independent weights.\n"
            "Dashed arrows: notebook test-time processing. T is fitted on validation logits; tau is selected on uncalibrated validation probabilities.\n"
            "The current notebook applies tau after test calibration (a protocol mismatch). Training uses weighted cross-entropy on raw logits.",
        ),
        ("11_end_to_end_thesis", [end_to_end_thesis], [1], (6.3, 8.0), None),
    ]
    for name, functions, widths, size, footnote in configs:
        rendered, checked = render(name, functions, widths, size, footnote)
        files.extend(rendered)
        qa[name] = checked
    manifest = OUT / "verification.json"
    manifest.write_text(json.dumps({"code_verification": verified, "layout_checks": qa}, indent=2), encoding="utf-8")
    files.append(manifest)
    if (OUT / "README.md").is_file():
        files.append(OUT / "README.md")
    drive = args.drive_dir or (Path(os.environ["DRIVE_SYNC_DIR"]) if os.environ.get("DRIVE_SYNC_DIR") else None)
    if drive is None and os.name != "nt":
        candidate = Path("/content/drive/MyDrive/MasterBKDN/Thesis/architecture_figures")
        if candidate.parent.is_dir():
            drive = candidate
    if drive is not None:
        if not drive.parent.is_dir():
            raise FileNotFoundError(f"Drive parent must exist: {drive.parent}")
        drive.mkdir(exist_ok=True)
        for path in files:
            shutil.copy2(path, drive / path.name)
        print("Synced to configured Drive directory:", drive)
    else:
        print(
            "Drive sync pending: no verified Drive mount configured. Windows C:/content is not treated as Google Drive."
        )


if __name__ == "__main__":
    main()
