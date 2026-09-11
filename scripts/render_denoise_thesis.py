import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Render cached denoise comparisons at thesis page width.")
    parser.add_argument("--volume", default="2404")
    parser.add_argument("--raw-cache", type=Path, default=Path(tempfile.gettempdir()) / "gf_vol_cache")
    parser.add_argument("--output-dir", type=Path, default=root / "figures/denoise/thesis")
    parser.add_argument("--drive-sync-dir", type=Path, default=os.environ.get("DRIVE_SYNC_DIR"))
    args = parser.parse_args()
    source = root / "figures/denoise"
    meta_path = source / f"denoise_compare_{args.volume}_all_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    methods = ["original", *meta["methods"]]
    if len(methods) != 8:
        raise ValueError("This publication layout expects the original and seven cached denoisers.")
    paths = [args.raw_cache / f"raw_{args.volume}.npy"] + [
        source / "cache" / f"{args.volume}_{method}.npy" for method in methods[1:]
    ]
    volumes = [np.load(path, mmap_mode="r") for path in paths]
    if any(vol.shape != (200, 200, 200) or vol.dtype != np.uint8 for vol in volumes):
        raise ValueError("All inputs must be raw-resolution 200^3 uint8 volumes.")
    if args.drive_sync_dir and not args.drive_sync_dir.is_dir():
        raise ValueError("Drive destination must be an existing mounted/synced directory.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    labels = ["Original", "Gaussian", "Median", "Bilateral", "TV (Chambolle)", "Wavelet", "NLM", "BM3D"]
    label_map = dict(zip(["original", "gaussian", "median", "bilateral", "tv", "wavelet", "nlm", "bm3d"], labels))
    dz = meta["depth_axis"]
    outputs = []
    records = []
    bundle = args.output_dir / f"denoise_{args.volume}_print.pdf"
    with PdfPages(bundle) as pdf:
        for name, spec in meta["planes"].items():
            axis = spec["axis"]
            remaining = [i for i in range(3) if i != axis]
            for index in spec["slices"]:
                images = [np.take(vol, index, axis=axis) for vol in volumes]
                if dz == remaining[1]:
                    images = [image.T for image in images]
                low, high = np.percentile(images[0], [1, 99])
                size = 80
                center_y = sum(meta["metrics_regions"]["rnfl_band"]) // 2 if dz != axis else 100
                y0 = int(np.clip(center_y - size // 2, 0, 200 - size))
                x0 = (200 - size) // 2
                records.append(
                    {
                        "axis": name,
                        "slice": index,
                        "vmin": float(low),
                        "vmax": float(high),
                        "roi_xywh": [x0, y0, size, size],
                    }
                )
                for detail in (False, True):
                    fig, axes = plt.subplots(4, 2, figsize=(6.4, 9.4))
                    fig.subplots_adjust(left=0.025, right=0.975, top=0.925, bottom=0.04, hspace=0.22, wspace=0.04)
                    view = "Detail: same 80 x 80 pixel region" if detail else "Full slice: box marks detail region"
                    fig.suptitle(f"Denoising comparison | {name} = {index}\n{view}", fontsize=12, y=0.985)
                    for panel, (ax, image, method) in enumerate(zip(axes.flat, images, methods)):
                        displayed = image[y0 : y0 + size, x0 : x0 + size] if detail else image
                        ax.imshow(displayed, cmap="gray", vmin=low, vmax=high, interpolation="nearest")
                        if not detail:
                            ax.add_patch(
                                Rectangle(
                                    (x0 - 0.5, y0 - 0.5), size, size, fill=False, edgecolor="#f0b429", linewidth=1.2
                                )
                            )
                        ax.set_title(f"({chr(97 + panel)}) {label_map[method]}", fontsize=11, pad=4)
                        ax.set_axis_off()
                    fig.text(
                        0.5,
                        0.012,
                        f"Harvard-GF data_{args.volume} | Identical contrast across methods",
                        ha="center",
                        fontsize=9,
                    )
                    stem = f"denoise_{args.volume}_{name}{index}_{'detail' if detail else 'full'}"
                    for suffix in ("png", "pdf"):
                        path = args.output_dir / f"{stem}.{suffix}"
                        fig.savefig(path, dpi=300)
                        outputs.append(path)
                    pdf.savefig(fig, dpi=300)
                    plt.close(fig)
                    print(f"Rendered {stem}")
    outputs.append(bundle)
    manifest = args.output_dir / f"denoise_{args.volume}_render.json"
    manifest.write_text(
        json.dumps(
            {
                "source_metadata": str(meta_path),
                "source_metadata_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                "inputs": [
                    {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths
                ],
                "methods": methods,
                "method_parameters_from_source": meta["methods"],
                "figure_inches": [6.4, 9.4],
                "dpi": 300,
                "interpolation": "nearest",
                "planes": records,
                "notes": "Cached data only; no denoising or metric recomputation. ROI is a display crop, not a segmentation.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    outputs.append(manifest)
    if args.drive_sync_dir:
        for path in outputs:
            shutil.copy2(path, args.drive_sync_dir / path.name)
        print(f"Copied {len(outputs)} artifacts to {args.drive_sync_dir}")
    else:
        print("Drive sync PENDING: provide --drive-sync-dir pointing to an existing Google Drive folder.")
    print(f"Print bundle: {bundle}")


if __name__ == "__main__":
    main()
