---
name: deep-arch-diagram
description: Use when the user asks to draw, redraw, or update deep-learning / neural-network architecture diagrams or model-pipeline figures (vẽ sơ đồ kiến trúc mô hình học sâu / architecture diagram / model figure / network schematic) for thesis reports. Renders matplotlib-only block & 3D-volume diagrams from the bundled library — no LaTeX needed. Trigger on keywords like "vẽ kiến trúc", "architecture diagram", "model figure", "sơ đồ mô hình", "block diagram".
---

# deep-arch-diagram

Draw publication-style deep-learning architecture figures for the glaucoma thesis
(and similar CNN / transformer models) with a small matplotlib-only library. No
LaTeX/TikZ, no external tools — works on this Windows CPU machine with `matplotlib`.

## Layout of this skill

```
.opencode/skills/deep-arch-diagram/
  SKILL.md                 <- this file
  archdraw.py              <- the drawing library (all primitives + theme)
  check_layout.py          <- geometric overlap QA (run after edits)
  examples/
    swinunetr_classifier.py  <- sample: SwinUNETR encoder -> MLP classifier
    simple3dcnn.py           <- sample: baseline 3D CNN
    sample_*.png             <- rendered outputs of the two samples
```

## How to produce a figure (the only correct workflow)

1. Copy the closest sample script from `examples/` (or write a new one) into the
   `examples/` folder. NEVER hand-edit the PNG.
2. Build the layout with the primitives below using explicit 2D data coordinates.
3. Render from the repo root:
   ```bash
   python ".opencode/skills/deep-arch-diagram/examples/<script>.py"
   ```
   Scripts write to `<repo>/figures/` (see the `ROOT = Path(__file__).resolve().parents[4]`
   line — do not change the depth of that path without updating `parents[N]`).
4. Verify no overlaps, then let the user eyeball the PNG:
   ```bash
   python ".opencode/skills/deep-arch-diagram/check_layout.py"
   ```
   It reports box-box overlaps (volumes include their 3D depth shadow, so keep a
   gap of `>= dpx` before the next element).
5. IMPORTANT: the coding model may not see images. After rendering, state the
   output path to the user and ask them to open it; use `check_layout.py` plus
   coordinate reasoning to self-check instead of relying on vision.

## Library API (`archdraw.py`)

All drawing happens on one axes with fixed `xlim/ylim`; helpers:

- `new_figure(xlim, ylim, figsize)` -> `(fig, ax)`; draw nothing outside xlim/ylim.
- `volume(ax, x, y, w, h, dpx, dpy, fill, label=None, fs=7.0, label_pos="in"|"below")`
  Pseudo-3D cuboid: front face at `(x,y,w,h)`, top face skewed by `(dpx,dpy)`.
  The dark right face protrudes `dpx` to the right — leave `dpx`+ margin of gap.
- `rbox(ax, x, y, w, h, text, fc, fs, ec, lw, dashed=..., tc=...)` — rounded block.
- `arrow(ax, (x1,y1), (x2,y2), ...)` — flow arrow.
- `text(ax, x, y, s, ...)` — annotation/caption.
- `save(fig, path, dpi=250)`.

### Style rules (match repo figure style)
- Use the light pastel palette (`#ffe599` input, `#f6b26b` preprocess/pool,
  `#dbe9f6`-`#cfe2f3` conv/Swin, `#d5a6bd`/`#ead1dc` head, `#d9ead3` output,
  `#f5f5f5` notes; dark dashed red for "removed" parts, purple borders for groups).
- Font sizes 5–7 for inside/under labels; 11 for the bold figure title.
- Always add data-flow arrows left→right (or explicit elbow) and label shapes.
- Add caption markers `(a)`, `(b)`, `(c)` above each logical zone.
- Vietnamese or English text is fine depending on the thesis language.

## When asked about "which tool/library"
- Do NOT recommend PlotNeuralNet-style LaTeX pipelines unless the user explicitly
  wants `.tex`/PDF output — it needs a full TeX install (not present here).
- Do NOT fall back to Mermaid/draw.io/excalidraw JSON for publication figures;
  they auto-layout poorly for dense model diagrams.
- Use `archdraw.py` (this skill) — matplotlib is already a repo dependency.

## Verify
- `check_layout.py` exits with no `OVERLAP` lines.
- The script wrote `<repo>/figures/<name>.png` (check `Length > 0`).
- Ask the user to visually confirm before declaring done.
