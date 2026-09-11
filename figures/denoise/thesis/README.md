# Thesis Denoise Figures

These plates re-render the existing volume 2404 comparison without rerunning
denoising or changing reported metrics. Original figures are unchanged.

- `denoise_2404_print.pdf`: all 12 plates in a single review document.
- `denoise_2404_<plane>_full.pdf`: original plus seven denoisers for one slice;
  the gold box marks the region shown in the matching detail plate.
- `denoise_2404_<plane>_detail.pdf`: the same 80 x 80 pixel crop for every method.
- Matching PNGs: 1920 x 2820 pixels, 300 DPI.
- `denoise_2404_render.json`: input hashes, original method metadata, slices,
  display contrast and exact crop coordinates.

Planes: `x132`, `x148`, `y22`, `y38`, `z114`, `z130`.
For the main discussion, start with `x132_full` and `x132_detail`.
Use the remaining planes in an appendix if needed, rather than shrinking all
six slices into one figure.

## Insertion

Prefer individual PDF plates for LaTeX: text remains vector-based. Each plate
is designed at 6.4 x 9.4 inches with 11 pt panel labels. Use a full-page figure,
not a small subfigure. Allow room for the caption:

```latex
\begin{figure}[p]
  \centering
  \includegraphics[width=\textwidth,height=0.82\textheight,keepaspectratio]
    {figures/denoise/thesis/denoise_2404_x132_detail.pdf}
  \caption{Comparison of denoising methods on the same enlarged region of
  Harvard-GF volume 2404, slice $x=132$. All panels use identical intensity
  limits derived from the original full slice.}
  \label{fig:denoise-detail}
\end{figure}
```

For Word, use the PNG at page text width and disable image compression.
The crop magnifies existing pixels by 2.5 relative to the full-slice panel;
it does not add resolution. Nearest-neighbor display avoids adding smoothing.
The crop is a visualization region, not an anatomical segmentation. Full-slice
1st/99th percentile limits from the raw image are shared by all methods and
both full/detail plates. Do not normalize each method independently.

## Regeneration And Drive

```powershell
python scripts/render_denoise_thesis.py
python scripts/render_denoise_thesis.py --drive-sync-dir "G:\My Drive\MasterBKDN\Thesis\denoise_figures"
```

The second command is an example: supply an actual existing Google Drive
directory, not an arbitrary local folder. `DRIVE_SYNC_DIR` is also supported.
The script uses only local raw/denoised caches and does not launch an experiment.
No training or new numerical evaluation is performed.

Drive synchronization is currently PENDING: no mounted or configured Drive
destination was available in this Windows session. The Colab destination is
`/content/drive/MyDrive/MasterBKDN/Thesis/denoise_figures`.
The compiled thesis/source is not present here, so no thesis PDF was rebuilt.
