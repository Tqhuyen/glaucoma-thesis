import math
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parent
sys.path.insert(0, str(SKILL))

import archdraw

events = []


def wrap(fn, kind):
    def w(*a, **k):
        events.append((kind, a, k))
        return fn(*a, **k)

    return w


archdraw.volume = wrap(archdraw.volume, "vol")
archdraw.rbox = wrap(archdraw.rbox, "rbox")
archdraw.text = wrap(archdraw.text, "text")

from examples import simple3dcnn, swinunetr_classifier  # noqa: E402

SCALES = {}


def data_scale(ax):
    if id(ax) not in SCALES:
        fig = ax.figure
        xl, yl = ax.get_xlim(), ax.get_ylim()
        w_in, h_in = fig.get_size_inches()
        SCALES[id(ax)] = (xl[1] - xl[0]) / w_in / 1.0, (yl[1] - yl[0]) / h_in / 1.0
    return SCALES[id(ax)]


def char_du(ax, fs):
    sx, sy = data_scale(ax)
    return 0.62 * fs / 72.0 * sx, 1.3 * fs / 72.0 * sy


def text_extent(ax, x, y, s, fs, ha):
    cw, lh = char_du(ax, fs)
    lines = s.split("\n")
    w = max(len(t) for t in lines) * cw
    h = len(lines) * lh
    x0 = x - w / 2 if ha == "center" else x
    return x0, y - h / 2, x0 + w, y + h / 2


def footprints(kind, args, kwargs, ax):
    if kind == "vol":
        x, y, w, h, dpx, dpy = args[1:7]
        return (x, y, x + w + dpx, y + h + dpy)
    if kind == "rbox":
        x, y, w, h = args[1:5]
        return (x, y, x + w, y + h)
    return None


def run(mod):
    events.clear()
    SCALES.clear()
    mod.main()
    boxes = []
    texts = []
    for kind, a, k in events:
        ax = a[0]
        if kind in ("vol", "rbox"):
            boxes.append((kind, a[1], a[2], a[3], a[4], a[5] if kind == "vol" else 0.0, footprints(kind, a, k, ax)))
        else:
            texts.append((kind, a[1], a[2], a[3], k.get("fs", 7.2), k.get("ha", "center")))

    print(f"### {mod.__name__}: {len(boxes)} boxes, {len(texts)} texts")

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ax = boxes[i][6]
            b1 = boxes[i][6]
            b2 = boxes[j][6]
            if b1[0] < b2[2] - 0.2 and b2[0] < b1[2] - 0.2 and b1[1] < b2[3] - 0.2 and b2[1] < b1[3] - 0.2:
                print(f"  OVERLAP {boxes[i][0]}@({boxes[i][1]:.1f},{boxes[i][2]:.1f})  <->  "
                      f"{boxes[j][0]}@({boxes[j][1]:.1f},{boxes[j][2]:.1f})")
    for t in texts:
        pass
    print()


for m in (swinunetr_classifier, simple3dcnn):
    run(m)
print("done")
