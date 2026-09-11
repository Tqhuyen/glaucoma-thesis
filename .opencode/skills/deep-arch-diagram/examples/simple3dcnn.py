import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from archdraw import arrow, new_figure, rbox, save, text, volume

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "figures" / "arch_simple3dcnn.png"

PAL = {
    "input": "#ffe599",
    "conv": "#dbe9f6",
    "gap": "#ead1dc",
    "head": "#ead1dc",
    "out": "#d9ead3",
    "note": "#f5f5f5",
}


def main():
    fig, ax = new_figure((0, 214), (0, 58), (19.5, 7.6))
    ax.text(107, 56.4, "Simple3DCNN \u2014 baseline 3D CNN glaucoma (GroupNorm + pool b\u1ea5t \u0111\u1eb3ng h\u01b0\u1edbng (2,2,1))",
            ha="center", va="center", fontsize=11.5, weight="bold", color="#111111")

    cy = 41.0
    dpx = 4.5
    gap = 5.4
    x = 1.0
    boxes = []

    volume(ax, x, cy - 4.0, 8.0, 8.0, dpx, 1.5, PAL["input"], "OCT\n200\u00b3\u00b71ch", fs=5.0)
    w0 = 8.0
    boxes.append([x, x + w0, cy])
    x += w0 + gap

    for (c, plane, glyph) in [(16, 100, 6.9), (32, 50, 5.7), (64, 25, 4.7), (128, 12, 3.9)]:
        card = (
            f"Conv3\u00b3 \u00b7 {c}\nGroupNorm(8)+ReLU\n+ residual skip\nMaxPool (2,2,1)"
        )
        rbox(ax, x, cy - 5.0, 12.5, 10.0, card, PAL["conv"], fs=5.0, ec="#6fa8dc")
        boxes.append([x, x + 12.5, cy])
        x += 12.5 + gap
        volume(ax, x, cy - glyph / 2, glyph, glyph, dpx, 1.5, PAL["conv"], "", fs=5.0)
        boxes.append([x, x + glyph, cy])
        text(ax, x + glyph / 2 + dpx / 2, cy - glyph / 2 - 0.6, f"{plane}\u00d7{plane}\u00d7200 \u00b7 {c}", fs=5.0)
        x += glyph + gap

    rbox(ax, x, cy - 3.6, 13.5, 7.2, "AdaptiveAvgPool3d(1)\n128-d vector", PAL["gap"], fs=5.3, ec="#b4869e")
    boxes.append([x, x + 13.5, cy])
    x += 13.5 + gap
    rbox(ax, x, cy - 5.0, 20.0, 10.0, "MLP head\nLinear 128\u219264 \u00b7 ReLU\nDropout 0.3\nLinear 64\u21922\n\u2192 raw logits", PAL["head"], fs=5.4, ec="#b4869e")
    boxes.append([x, x + 20.0, cy])
    x += 20.0 + gap
    rbox(ax, x, cy - 3.6, 13.5, 7.2, "CrossEntropyLoss\n(normal/glaucoma)", PAL["out"], fs=5.3, ec="#6aa84f")

    for i in range(len(boxes) - 1):
        arrow(ax, (boxes[i][1] + 0.4, cy), (boxes[i + 1][0] - 0.4, cy))

    text(ax, 1.0, 30.5, "Cu\u1ed1i m\u1ed7i stage: MaxPool stride (2,2,1) \u2014 gi\u1ea3m m\u1eb7t ph\u1eb3ng B-scan (H,W), "
         "GI\u1eee NGUY\u00caN chi\u1ec1u s\u00e2u D=200 \u2192 RNFL m\u1ecfng kh\u00f4ng b\u1ecb \u0111\u00e8 n\u00e1t (c\u00e1c kh\u1ed1i d\u00e0i = D gi\u1eef nguy\u00ean)",
         ha="left", fs=6.5, color="#555555")

    rbox(ax, 1.0, 6.0, 212.0, 15.0,
         "V\u00ec sao c\u1ea5u tr\u00fac n\u00e0y h\u1ed9i t\u1ee5 \u1ed5n \u0111\u1ecbnh v\u1edbi d\u1eef li\u1ec7u nh\u1ecf / batch 2\u20134:\n"
         "\u2022 GroupNorm (8 nh\u00f3m) thay BatchNorm \u2192 kh\u00f4ng ph\u1ee5 thu\u1ed9c batch size\n"
         "\u2022 residual skip sau stage \u0111\u1ea7u \u2192 gradient m\u1ecbn h\u01a1n cho m\u1ea1ng s\u00e2u\n"
         "\u2022 AdaptiveAvgPool3d ch\u1ec9 \u1edf cu\u1ed1i \u2192 nh\u1eadn input k\u00edch th\u01b0\u1edbc b\u1ea5t k\u1ef3 (100\u2013200\u00b3)\n"
         "\u2022 logits th\u00f4 + CrossEntropyLoss \u2014 kh\u1edbp contract {\"logits\": ...}",
         PAL["note"], fs=5.9, ec="#8e7cc3", dashed=True)

    save(fig, OUT)


if __name__ == "__main__":
    main()
