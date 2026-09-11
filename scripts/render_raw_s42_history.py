"""Render the supplied, complete 22-epoch validation history without training."""

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import xml.etree.ElementTree as ET
from decimal import Decimal
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from PIL import Image


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "figures/raw_s42_validation_log.csv"
    stem = source.with_name("raw_s42_validation")
    raw = source.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    fields = ["epoch", "logged_loss", "val_auc", "val_balanced_accuracy", "val_mcc"]
    if reader.fieldnames != fields:
        raise ValueError(f"Unexpected CSV schema: {reader.fieldnames}")
    rows = list(reader)
    if len(rows) != 22 or any(set(row) != set(fields) for row in rows):
        raise ValueError("Expected exactly 22 complete rows")
    original_epochs = [int(row["epoch"]) for row in rows]
    rows.sort(key=lambda row: int(row["epoch"]))
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, 23)):
        raise ValueError("Expected unique epochs 1 through 22, without gaps")
    for row in rows:
        for key in fields[1:]:
            value = Decimal(row[key])
            if not value.is_finite():
                raise ValueError(f"Nonfinite value: {row}")
            if key != "logged_loss" and not (Decimal(-1 if key == "val_mcc" else 0) <= value <= 1):
                raise ValueError(f"Metric outside its valid range: {row}")
    metrics = fields[2:]
    expected_selected = dict(zip(metrics, ["0.8603", "0.7437", "0.5307"], strict=True))
    expected_maxima = dict(zip(metrics, [(19, "0.8603"), (17, "0.7860"), (17, "0.5666")], strict=True))
    maxima = {}
    for metric, (epoch, value) in expected_maxima.items():
        maximum = max(Decimal(row[metric]) for row in rows)
        best_epochs = [int(row["epoch"]) for row in rows if Decimal(row[metric]) == maximum]
        if best_epochs != [epoch] or maximum != Decimal(value):
            raise ValueError(f"Unexpected maximum for {metric}: {best_epochs}, {maximum}")
        maxima[metric] = {"epoch": epoch, "value": float(maximum)}
    if any(Decimal(rows[18][metric]) != Decimal(value) for metric, value in expected_selected.items()):
        raise ValueError("Selected epoch 19 does not match supplied values")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 4.5), dpi=400, sharex=True)
    fig.subplots_adjust(left=0.12, right=0.975, bottom=0.115, top=0.925, hspace=0.40)
    styles = {
        "val_auc": ("#005A8D", "o", "-", "AUROC"),
        "val_balanced_accuracy": ("#005A8D", "s", "-", "Balanced accuracy"),
        "val_mcc": ("#A74700", "^", "--", "MCC"),
    }
    curves = {}
    for metric, (color, marker, linestyle, label) in styles.items():
        ax = axes[0] if metric == "val_auc" else axes[1]
        values = [float(row[metric]) for row in rows]
        (line,) = ax.plot(
            epochs,
            values,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.3,
            markersize=3.3,
            markeredgewidth=0.6,
            label=label,
        )
        if list(line.get_xdata()) != epochs or list(line.get_ydata()) != values:
            raise ValueError("Plotted data differ from the CSV")
        curves[metric] = values
        peak = maxima[metric]
        ax.plot(
            peak["epoch"],
            peak["value"],
            marker=marker,
            markersize=7,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.5,
            zorder=5,
        )
    for ax in axes:
        ax.set_xlim(1, 22)
        ax.set_xticks([1, 4, 7, 10, 13, 16, 19, 22])
        ax.axvline(19, color="0.42", linestyle=(0, (3, 3)), linewidth=0.7, zorder=0)
        ax.grid(axis="y", color="0.88", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=0.7)
    axes[0].set_ylim(0.70, 0.925)
    axes[0].set_yticks([0.70, 0.75, 0.80, 0.85, 0.90])
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("(a) Validation AUROC", loc="left", pad=9)
    axes[1].set_ylim(0.25, 0.925)
    axes[1].set_yticks([0.30, 0.45, 0.60, 0.75, 0.90])
    axes[1].set_ylabel("Score")
    axes[1].set_xlabel("Epoch", labelpad=5)
    axes[1].set_title("(b) Threshold-based validation metrics", loc="left", pad=9)
    legend = axes[1].legend(loc="upper left", frameon=False, borderaxespad=0.3, handlelength=2)
    annotations = []
    for ax, text, xy, xytext, color in [
        (axes[0], "Max AUROC: 0.8603\nSelected epoch 19", (19, 0.8603), (10.0, 0.918), "#005A8D"),
        (axes[1], "Max BA: 0.7860 (epoch 17)", (17, 0.7860), (11.0, 0.900), "#005A8D"),
        (axes[1], "Max MCC: 0.5666 (epoch 17)", (17, 0.5666), (10.5, 0.665), "#A74700"),
    ]:
        annotations.append(
            ax.annotate(
                text,
                xy=xy,
                xytext=xytext,
                ha="left",
                va="top",
                fontsize=10,
                color=color,
                arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.7, "shrinkA": 3, "shrinkB": 5},
            )
        )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(annotations) + legend.get_texts()
    for ax in axes:
        texts.extend([ax._left_title, ax.xaxis.label, ax.yaxis.label])
        texts.extend(ax.get_xticklabels() + ax.get_yticklabels())
    boxes = [
        (text.get_text(), Text.get_window_extent(text, renderer))
        for text in texts
        if text.get_visible() and text.get_text()
    ]
    outside = [
        label for label, box in boxes if not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(box.x1, box.y1)
    ]
    overlaps = [(a, b) for (a, box_a), (b, box_b) in combinations(boxes, 2) if box_a.overlaps(box_b)]
    if outside or overlaps:
        raise ValueError(f"Text layout failed: outside={outside}, overlaps={overlaps}")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=400)
    plt.close(fig)
    with Image.open(stem.with_suffix(".png")) as image:
        png_size = list(image.size)
        png_dpi = image.info.get("dpi")
        image.verify()
    if png_size != [2520, 1800] or not png_dpi or any(abs(dpi - 400) > 0.01 for dpi in png_dpi):
        raise ValueError(f"Incorrect PNG dimensions or DPI: {png_size}, {png_dpi}")
    pdf = stem.with_suffix(".pdf").read_bytes()
    match = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]", pdf)
    if match is None:
        raise ValueError("PDF MediaBox missing")
    x0, y0, x1, y1 = map(float, match.groups())
    pdf_size = [x1 - x0, y1 - y0]
    svg = ET.parse(stem.with_suffix(".svg")).getroot()
    svg_size = [float(svg.attrib[key].removesuffix("pt")) for key in ("width", "height")]
    for size in (pdf_size, svg_size):
        if not all(
            math.isclose(actual, expected, abs_tol=0.01) for actual, expected in zip(size, [453.6, 324.0], strict=True)
        ):
            raise ValueError(f"Incorrect vector dimensions: {size}")
    if b"/Subtype /Image" in pdf or any(node.tag.endswith("}image") for node in svg.iter()):
        raise ValueError("Unexpected raster content in vector output")
    if source.read_bytes() != raw:
        raise ValueError("Source CSV changed during rendering")
    drive_env = os.environ.get("RAW_S42_DRIVE_DIR")
    drive = Path(drive_env).expanduser() if drive_env else None
    drive_status = {
        "environment_variable": "RAW_S42_DRIVE_DIR",
        "status": "pending",
        "reason": "No explicit existing Drive destination configured",
    }
    if drive is not None and drive.is_dir() and drive.resolve() != stem.parent.resolve():
        for suffix in ("png", "pdf", "svg"):
            artifact = stem.with_suffix(f".{suffix}")
            target = drive / artifact.name
            shutil.copy2(artifact, target)
            if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(artifact.read_bytes()).digest():
                raise ValueError(f"Drive copy verification failed: {target}")
        drive_status = {
            "environment_variable": "RAW_S42_DRIVE_DIR",
            "status": "verified_copy",
            "destination": str(drive.resolve()),
            "verification": "SHA-256 byte equality",
        }
    else:
        drive = None
    metadata = {
        "source": str(source.relative_to(root)),
        "source_csv_sha256": sha,
        "scope": "Retrospective validation only; complete supplied log, epochs 1-22",
        "row_count": len(rows),
        "epochs": epochs,
        "source_was_sorted": original_epochs == epochs,
        "sorted_before_plotting": True,
        "extrapolation": False,
        "smoothing": False,
        "test_data_used": False,
        "weights_used": False,
        "training_performed": False,
        "wandb_experiment_created": False,
        "logged_loss": "Preserved verbatim in source CSV; not plotted or interpreted as ordinary cross-entropy",
        "selected_epoch": {
            "epoch": 19,
            "criterion": "maximum validation AUROC",
            **{key: float(value) for key, value in expected_selected.items()},
        },
        "metric_maxima": maxima,
        "other_metric_best_epoch": {
            "epoch": 17,
            "val_balanced_accuracy": 0.7860,
            "val_mcc": 0.5666,
            "separate_from_selected_epoch": True,
        },
        "plotted_values": curves,
        "layout": {
            "size_inches": [6.3, 4.5],
            "size_cm": [16.002, 11.43],
            "font_points": [10, 11],
            "panel_captions": ["(a) Validation AUROC", "(b) Threshold-based validation metrics"],
            "selected_epoch_guides": [19, 19],
            "best_other_metric_markers_epoch": 17,
        },
        "checks": {
            "all_rows_validated": True,
            "unique_contiguous_epochs": True,
            "expected_maxima_validated": True,
            "curves_equal_csv": True,
            "source_unchanged": True,
            "png_pixels": png_size,
            "png_dpi": png_dpi,
            "pdf_points": pdf_size,
            "svg_points": svg_size,
            "vector_outputs_have_no_raster_images": True,
            "text_layout_renderer": "Matplotlib Agg at 400 dpi; text rectangles excluding leader lines",
            "text_items_checked": len(boxes),
            "text_outside_canvas": outside,
            "text_rectangle_overlaps": overlaps,
        },
        "drive_sync": drive_status,
        "outputs": [str(stem.with_suffix(f".{suffix}").relative_to(root)) for suffix in ("png", "pdf", "svg", "json")],
    }
    metadata_path = stem.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if drive is not None:
        target = drive / metadata_path.name
        shutil.copy2(metadata_path, target)
        if target.read_bytes() != metadata_path.read_bytes():
            raise ValueError("Drive metadata verification failed")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
