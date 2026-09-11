import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from archdraw import arrow, new_figure, rbox, save, text, volume

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "figures" / "arch_swinunetr_classifier.png"

PAL = {
    "input": "#ffe599",
    "proc": "#f6b26b",
    "head": "#ead1dc",
    "note": "#f5f5f5",
    "out": "#d9ead3",
}

FUNNEL = [
    (8.2, 6.4, "48\u00b3 \u00b7 48", "#cfe2f3", "patch embed (stride 2)\n96\u00b3 \u2192 48\u00b3 tokens"),
    (6.8, 5.4, "24\u00b3 \u00b7 96", "#a4c2f4", None),
    (5.4, 4.4, "12\u00b3 \u00b7 192", "#8e7cc3", None),
    (4.2, 3.5, "6\u00b3 \u00b7 384", "#8e7cc3", None),
    (3.3, 2.7, "3\u00b3 \u00b7 768", "#674ea7", "bottleneck (spatial = input/32)"),
]

STAGES = [
    "Stage 1\n2\u00d7 Swin block\nW-MSA / SW-MSA\nwindow 7\u00b3 \u00b7 LN \u00b7 MLP\n48\u00b3\u00b748 \u2192 24\u00b3\u00b796",
    "Stage 2\n2\u00d7 Swin block\nW-MSA / SW-MSA\nwindow 7\u00b3 \u00b7 LN \u00b7 MLP\n24\u00b3\u00b796 \u2192 12\u00b3\u00b7192",
    "Stage 3\n2\u00d7 Swin block\nW-MSA / SW-MSA\nwindow 7\u00b3 \u00b7 LN \u00b7 MLP\n12\u00b3\u00b7192 \u2192 6\u00b3\u00b7384",
    "Stage 4\n2\u00d7 Swin block\nW-MSA / SW-MSA\nwindow 7\u00b3 \u00b7 LN \u00b7 MLP\n6\u00b3\u00b7384 \u2192 3\u00b3\u00b7768",
]


def main():
    fig, ax = new_figure((0, 150), (0, 64), (17.0, 8.6))
    ax.text(75, 63.3, "SwinUNETR (encoder only) chuy\u1ec3n sang ph\u00e2n lo\u1ea1i glaucoma \u2014 MONAI backbone + MLP head",
            ha="center", va="center", fontsize=11.5, weight="bold", color="#111111")

    cy = 37.0
    volume(ax, 1.2, cy - 3.75, 7.5, 7.5, 4.0, 1.4, PAL["input"], "OCT volume\n200\u00b3 \u00b7 1ch\n(/255)", fs=5.2)
    arrow(ax, (9.2, cy), (12.8, cy))
    rbox(ax, 13.0, cy - 3.0, 9.0, 6.0, "resize on-the-fly\ntrilinear\n200\u00b3 \u2192 96\u00b3", PAL["proc"], fs=5.2)
    arrow(ax, (22.7, cy), (27.2, cy))

    xs = []
    x0 = 28.0
    ymid = cy
    for (w, h, lab, col, extra) in FUNNEL:
        xs.append(x0)
        volume(ax, x0, ymid - h / 2, w, h, 3.0, 1.1, col, lab, fs=5.1, label_pos="below")
        if extra:
            text(ax, x0 + w / 2, ymid + h / 2 + 1.7, extra, fs=5.1, color="#674ea7", style="italic")
        x0 += w + 6.0

    for i in range(len(FUNNEL) - 1):
        xa = xs[i] + FUNNEL[i][0] + 0.5
        xb = xs[i + 1] - 0.5
        arrow(ax, (xa, cy), (xb, cy))
        text(ax, (xa + xb) / 2, cy + 5.6, "\u00f72 space \u00b7 \u00d72 ch\n(PatchMerging)", fs=4.9, color="#555555")

    last_right = xs[-1] + FUNNEL[-1][0]
    arrow(ax, (last_right + 0.6, cy), (last_right + 3.2, cy))
    hx = last_right + 3.4
    rbox(ax, hx, cy - 3.6, 13.0, 7.2, "AdaptiveAvgPool3d\n3\u00b3\u00b7768\n\u2192 768-d vector", PAL["head"], fs=5.4, ec="#b4869e")
    hx += 13.0 + 2.2
    arrow(ax, (hx - 0.3, cy), (hx + 0.2, cy))
    rbox(ax, hx + 0.4, cy - 5.2, 20.5, 10.4,
         "MLP head (fine-tune)\nLinear 768\u2192192 \u00b7 ReLU\nDropout 0.3\nLinear 192\u21922\n\u2192 raw logits {B,2}", PAL["head"], fs=5.4, ec="#b4869e")
    hx += 20.5 + 2.0
    arrow(ax, (hx - 0.3, cy), (hx + 0.2, cy))
    rbox(ax, hx + 0.4, cy - 3.6, 13.0, 7.2, "CrossEntropyLoss\n(normal / glaucoma)", PAL["out"], fs=5.4, ec="#6aa84f")

    sx = 28.0
    for t in STAGES:
        rbox(ax, sx, 18.5, 12.6, 9.6, t, "#fbfbfb", fs=5.0, ec="#8e7cc3", lw=1.0)
        sx += 12.6 + 1.2

    text(ax, 28.0, 15.4, "(b) Encoder gi\u1eef l\u1ea1i: swinViT 4 stage (SSL-pretrained) \u2014 d\u00f9ng hidden_states_out[4] s\u00e2u nh\u1ea5t",
         ha="left", fs=7.0, weight="bold")
    text(ax, 1.2, 47.0, "(a) D\u1eef li\u1ec7u & ti\u1ec1n x\u1eed l\u00fd", ha="left", fs=7.0, weight="bold")

    rbox(ax, 6.0, 5.0, 66.0, 8.0,
         "\u2717 Decoder c\u1ee7a SwinUNETR b\u1ecb xo\u00e1: encoder1\u20134/10, decoder5\u20131, out\n"
         "(UnetrBasic/UpBlock) kh\u00f4ng c\u00f2n trong forward \u2192 gi\u1ea3m tham s\u1ed1 & VRAM",
         PAL["note"], fs=5.4, ec="#cc0000", dashed=True, tc="#8b0000")
    rbox(ax, 76.0, 5.0, 68.0, 8.0,
         "SSL pretrain: masked volumetric inpainting tr\u00ean 5050 CT\n(Tang et al., CVPR 2022) \u2192 fine-tune glaucoma; "
         "freeze_encoder=True ch\u1ec9 train MLP head",
         PAL["note"], fs=5.4, ec="#666666", dashed=True)

    save(fig, OUT)


if __name__ == "__main__":
    main()
