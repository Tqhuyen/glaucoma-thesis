"""Offline thesis plates; run from any directory with the three cached raw volumes."""

import ast
import hashlib
import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "projection_thesis"
SIZE = (6.3, 5.2)
DPI = 400


def pure_helpers(path, names):
    """Load only named, local NumPy functions, never module-level imports or I/O."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise RuntimeError(f"Missing reference functions: {path}")
    scope = {"np": np, "N_EDGE": 30}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_layout(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for ax in fig.axes:
        texts.extend([ax.title, *ax.texts])
        if ax.axison:
            texts.extend([ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()])
    texts = [t for t in texts if t.get_visible() and t.get_text()]
    boxes = [t.get_window_extent(renderer) for t in texts]
    page = fig.bbox
    for text, box in zip(texts, boxes):
        if not (page.x0 <= box.x0 < box.x1 <= page.x1 and page.y0 <= box.y0 < box.y1 <= page.y1):
            raise AssertionError(f"Text outside page: {text.get_text()}")
    for i, box in enumerate(boxes):
        for j in range(i):
            if box.overlaps(boxes[j]):
                raise AssertionError(f"Overlapping text: {texts[i].get_text()} / {texts[j].get_text()}")
        for ax in fig.axes:
            if box.overlaps(ax.get_window_extent(renderer)):
                raise AssertionError(f"Text overlaps image: {texts[i].get_text()}")
    axes_boxes = [ax.get_window_extent(renderer) for ax in fig.axes]
    for box in axes_boxes:
        np.testing.assert_allclose(box.width, box.height, atol=1e-6)
        np.testing.assert_allclose(box.width, axes_boxes[0].width, atol=1e-6)
        if not (page.x0 <= box.x0 < box.x1 <= page.x1 and page.y0 <= box.y0 < box.y1 <= page.y1):
            raise AssertionError("Image outside page")
    for i, box in enumerate(axes_boxes):
        if any(box.overlaps(other) for other in axes_boxes[:i]):
            raise AssertionError("Overlapping images")
    assert axes_boxes[0].width / fig.dpi >= 1.7
    assert all(text.get_fontsize() >= 10 for text in texts)
    baselines = []
    for start in (0, 3):
        row = fig.axes[start : start + 3]
        assert all("\n" not in ax.title.get_text() and ax.title.get_va() == "baseline" for ax in row)
        positions = [ax.title.get_transform().transform(ax.title.get_position())[1] for ax in row]
        np.testing.assert_allclose(positions, positions[0], atol=1e-6)
        baselines.append(positions[0] / fig.dpi)
    return {
        "passed": True,
        "visible_text_count": len(texts),
        "equal_square_panels": True,
        "panel_width_inches": axes_boxes[0].width / fig.dpi,
        "panel_height_inches": axes_boxes[0].height / fig.dpi,
        "single_line_titles": True,
        "row_title_baselines_inches_from_bottom": baselines,
        "minimum_font_pt": min(text.get_fontsize() for text in texts),
        "text_and_image_bounds": "inside fixed page; no text/text, text/image or image/image overlap",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old_path = ROOT / "scripts" / "render_report_views.py"
    model_path = ROOT / "scripts" / "final_model.py"
    old = pure_helpers(
        old_path, ["depth_axis", "to_depth_last", "slab_of", "slice_metric", "best_slice_axis", "enface"]
    )
    model = pure_helpers(model_path, ["depth_axis", "to_depth_last", "project_views"])
    label_path = ROOT / "figures" / "report_views_meta.json"
    labels = json.loads(label_path.read_text(encoding="utf-8"))
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 10,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    metadata = {
        "page_inches": SIZE,
        "page_cm": [round(x * 2.54, 3) for x in SIZE],
        "png_dpi": DPI,
        "font": "DejaVu Sans",
        "font_pt": {"panel_titles": 10, "header": 10.5, "footer": 10},
        "reference_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (old_path, model_path, label_path)},
        "renderer_sha256": sha256(Path(__file__)),
        "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "label_provenance": {
            "file": "figures/report_views_meta.json",
            "source_csv": "https://huggingface.co/datasets/harvardairobotics/Harvard-GF/resolve/main/ReadMe/data_summary.csv",
            "verification": "Verified against source CSV during the preceding review; not re-fetched in this run",
            "source_split": "Test",
            "source_npz_key": "oct_bscans",
        },
        "processing": "No denoising, retouching, spatial cropping or resampling of arrays. Per-panel linear P1/P99 display clipping only.",
        "axis_convention": "Source-array axes and indices are zero-based; canonical axes put detected depth last, other axes in source order.",
        "reference_loading": "AST-selected pure NumPy functions only; neither reference module imported/executed at top level.",
        "means": "NumPy uint8 mean (float64 accumulation), then float32, exactly as old renderer; never rounded to uint8.",
        "selection": "First maximum of std(clip(section, P1, P99)), indices 30 through 169 inclusive on canonical axes 1 and 0.",
        "volumes": {},
    }
    captions = [
        "# Thesis Projection Plates",
        "",
        "Insert at 6.3 in (16.002 cm) wide without further reduction. PDF/SVG keep text as vector text; OCT images are raster.",
        f"Page size: {SIZE[0]} x {SIZE[1]} in ({SIZE[0] * 2.54:.3f} x {SIZE[1] * 2.54:.3f} cm).",
        "All axes and indices below refer to the source array and are zero-based. No physical pixel spacing is assumed.",
        "",
    ]
    for stem, glaucoma in (("2404", True), ("3294", True), ("2615", False)):
        source = Path(tempfile.gettempdir()) / "gf_vol_cache" / f"raw_{stem}.npy"
        vol = np.load(source, allow_pickle=False)
        assert vol.shape == (200, 200, 200) and vol.dtype == np.uint8
        assert labels[stem]["glaucoma"] is glaucoma
        dz = model["depth_axis"](vol)
        canonical = model["to_depth_last"](vol, dz)
        permutation = [i for i in range(3) if i != dz] + [dz]
        np.testing.assert_array_equal(canonical, old["to_depth_last"](vol, old["depth_axis"](vol)))
        profile = canonical.mean(axis=(0, 1)).astype(np.float32)
        peak = int(profile.argmax())
        lo, hi = max(0, peak - 16), min(200, peak + 17)
        assert (lo, hi) == old["slab_of"](canonical, 16)
        assert dz == labels[stem]["depth_axis"] and [lo, hi] == labels[stem]["slab"]
        selection = []
        sections = []
        for axis in (1, 0):
            scores = []
            for index in range(30, 170):
                section = np.take(canonical, index, axis=axis)
                p1, p99 = np.percentile(section, [1, 99])
                scores.append(float(np.clip(section, p1, p99).std()))
            index = 30 + int(np.argmax(scores))
            assert index == old["best_slice_axis"](canonical, axis)
            section = np.take(canonical, index, axis=axis).T
            sections.append(section)
            source_section = np.take(vol, index, axis=permutation[axis])
            remaining_axes = [i for i in range(3) if i != permutation[axis]]
            source_section = np.moveaxis(source_section, remaining_axes.index(dz), 0)
            np.testing.assert_array_equal(section, source_section)
            selection.append(
                {
                    "canonical_axis": axis,
                    "source_axis": permutation[axis],
                    "index": index,
                    "candidate_range_inclusive": [30, 169],
                    "scores": scores,
                    "selected_score": scores[index - 30],
                    "tie_break": "first maximum",
                }
            )
        arrays = [
            canonical.mean(axis=2).astype(np.float32),
            canonical.max(axis=2).astype(np.float32),
            canonical[:, :, lo:hi].mean(axis=2).astype(np.float32),
            canonical[:, :, lo:hi].max(axis=2).astype(np.float32),
            *sections,
        ]
        for array, kind in zip(arrays, ("aip_full", "mip_full", "aip_slab", "mip_slab")):
            np.testing.assert_array_equal(array, old["enface"](canonical, kind, lo, hi))
        np.testing.assert_array_equal(arrays[3], model["project_views"](canonical, 16)[0])
        assert all(np.any(a != np.round(a)) for a in (arrays[0], arrays[2]))
        assert all(a.shape == (200, 200) for a in arrays)
        label = "glaucoma" if glaucoma else "non-glaucoma"
        titles = [
            "(a) Full-depth AIP",
            "(b) Full-depth MIP",
            "(c) Peak-slab AIP",
            "(d) Peak-slab MIP",
            f"(e) B-scan (i={selection[0]['index']})",
            f"(f) Section (i={selection[1]['index']})",
        ]
        fig, axs = plt.subplots(2, 3, figsize=SIZE, dpi=DPI)
        fig.subplots_adjust(left=0.025, right=0.975, bottom=0.12, top=0.875, wspace=0.09, hspace=0.16)
        fig.text(0.025, 0.975, f"Harvard-GF | data_{stem} | {label}", va="top", fontsize=10.5)
        fig.text(0.025, 0.07, "Display: per-panel P1-P99", va="top")
        fig.text(0.025, 0.035, "AIP: mean; MIP: maximum. Each panel: 200 x 200 pixels.", va="top")
        panels = []
        for number, (ax, array, title) in enumerate(zip(axs.flat, arrays, titles)):
            p1, p99 = np.percentile(array, [1, 99])
            ax.imshow(
                array,
                cmap="gray",
                vmin=p1,
                vmax=p99,
                origin="upper",
                aspect="equal",
                interpolation="nearest",
                extent=(-0.5, 199.5, 199.5, -0.5),
            )
            ax.set_title(title, pad=5, verticalalignment="baseline")
            ax.set_axis_off()
            panels.append(
                {
                    "panel": chr(97 + number),
                    "title": title,
                    "kind": "projection" if number < 4 else "section",
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "array_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                    "display_p1": float(p1),
                    "display_p99": float(p99),
                    "display_percentile_method": "linear",
                    "origin": "upper",
                    "interpolation": "nearest",
                    "row_source_axis": permutation[0] if number < 4 else dz,
                    "column_source_axis": permutation[1] if number < 4 else permutation[number - 4],
                    "reduction": ("mean" if number % 2 == 0 else "max") if number < 4 else None,
                    "depth_range_half_open": ([0, 200] if number < 2 else [lo, hi]) if number < 4 else None,
                    "selection": selection[number - 4] if number >= 4 else None,
                }
            )
        layout = check_layout(fig)
        base = OUT / f"report_views_{stem}_thesis"
        for extension in ("png", "svg", "pdf"):
            fig.savefig(base.with_suffix(f".{extension}"), dpi=DPI, facecolor="white")
        plt.close(fig)
        with Image.open(base.with_suffix(".png")) as image:
            assert image.size == tuple(round(x * DPI) for x in SIZE)
            assert min(image.info["dpi"]) >= 300
            png_size = list(image.size)
        svg = ET.parse(base.with_suffix(".svg")).getroot()
        assert len(svg.findall(".//{http://www.w3.org/2000/svg}text")) >= 9
        assert "Display: per-panel P1-P99" in " ".join(svg.itertext())
        assert svg.attrib["width"] == "453.6pt"
        np.testing.assert_allclose(float(svg.attrib["height"].removesuffix("pt")), SIZE[1] * 72)
        pdf_data = base.with_suffix(".pdf").read_bytes()
        assert b"/FontFile2" in pdf_data and b"/Subtype /Type3" not in pdf_data
        metadata["volumes"][stem] = {
            "source": str(source),
            "source_sha256": sha256(source),
            "shape": list(vol.shape),
            "dtype": str(vol.dtype),
            "glaucoma": glaucoma,
            "label": label,
            "depth_axis": dz,
            "canonical_axis_to_source": permutation,
            "depth_axis_rule": "argmax of float32 mean-profile standard deviation over source axes; first tie wins",
            "depth_profile": profile.tolist(),
            "peak": peak,
            "half_requested": 16,
            "half_effective": 16,
            "slab_half_open": [lo, hi],
            "slab_inclusive": [lo, hi - 1],
            "slab_width": hi - lo,
            "slab_rule": "argmax of float32-cast mean intensity profile, +/-16 pixels, clipped to volume bounds; not RNFL segmentation",
            "panels": panels,
            "layout": layout,
            "validation": {
                "old_renderer_exact_arrays": True,
                "old_renderer_exact_indices": True,
                "original_metadata_matches": True,
                "model_slab_mip_exact": True,
                "floating_means_preserved": True,
                "png_pixels": png_size,
                "svg_text": True,
                "pdf_embedded_truetype": True,
            },
            "outputs": {
                ext: {"file": f"{base.name}.{ext}", "sha256": sha256(base.with_suffix(f".{ext}"))}
                for ext in ("png", "svg", "pdf")
            },
        }
        english = (
            f"Views of raw Harvard-GF Test volume data_{stem} ({label}; 200 x 200 x 200 uint8). "
            "(a,b) Full-depth average-intensity (AIP) and maximum-intensity (MIP) en-face projections. "
            f"(c,d) Corresponding projections of the peak-centered slab, depth pixels {lo}-{hi - 1} inclusive "
            f"on source axis {dz}, centered at pixel {peak}. "
            f"(e) Selected B-scan at source axis {selection[0]['source_axis']}, index i={selection[0]['index']}. "
            f"(f) Selected orthogonal section at source axis {selection[1]['source_axis']}, index i={selection[1]['index']}. "
            "Panels (a-d) combine depth samples, whereas (e,f) show individual sections. "
            "The footer identifies the display convention; shared processing and selection details appear in Methods Notes."
        )
        vietnamese = (
            f"Thể tích OCT thô Harvard-GF data_{stem} ({'có glaucoma' if glaucoma else 'không glaucoma'}). "
            "(a,b) Ảnh chiếu en-face trung bình (AIP) và cực đại (MIP) toàn chiều sâu. "
            f"(c,d) Các phép chiếu tương ứng trong dải pixel {lo}-{hi - 1}, gồm hai đầu, "
            f"trên trục nguồn {dz}, tâm tại đỉnh {peak}. "
            f"(e) B-scan được chọn: trục nguồn {selection[0]['source_axis']}, chỉ số i={selection[0]['index']}. "
            f"(f) Lát cắt trực giao: trục nguồn {selection[1]['source_axis']}, chỉ số i={selection[1]['index']}. "
            "(a-d) gộp chiều sâu; (e,f) là lát cắt riêng. Quy ước hiển thị nằm dưới hình; "
            "phương pháp chung xem Methods Notes."
        )
        word_counts = {"en": len(english.split()), "vi": len(vietnamese.split())}
        assert all(80 <= count <= 110 for count in word_counts.values()), word_counts
        metadata["volumes"][stem]["validation"]["caption_whitespace_word_counts"] = word_counts
        captions.extend(
            [
                f"## data_{stem} ({label})",
                "",
                f"**English.** {english}",
                "",
                f"**Tiếng Việt (gợi ý).** {vietnamese}",
                "",
            ]
        )
        print(
            f"{stem}: depth axis={dz}, peak={peak}, slab=[{lo},{hi}), "
            f"sections={[(s['source_axis'], s['index']) for s in selection]}, "
            f"panel={layout['panel_width_inches']:.6f} in square, page={SIZE} in, checks PASS"
        )
    captions.extend(
        [
            "## Methods Notes",
            "",
            "**English.** All plates use the same processing. Depth is the source axis with the largest standard "
            "deviation of its mean-intensity profile. Other axes retain source order. The peak-centered slab spans "
            "the global mean-profile maximum plus/minus 16 pixels, clipped to volume bounds (33 pixels in these "
            "volumes); it is not an anatomically segmented RNFL band. AIP retains floating means; MIP takes maxima. "
            "Sections maximize standard deviation after P1/P99 clipping over candidate indices 30-169 on each "
            "non-depth axis; ties retain the first index. They are not necessarily central or clinically representative. "
            "Source axes/indices are zero-based; section depth increases downward.",
            "",
            "Each complete 200 x 200 panel uses its own linear P1/P99 display bounds, recorded in metadata; "
            "brightness is not directly comparable across panels. No denoising, retouching or spatial cropping "
            "was applied. Pixel aspect is 1:1 in array coordinates, not calibrated physical aspect. Labels come "
            "from figures/report_views_meta.json, previously checked against the Harvard-GF source CSV as supplied "
            "by the user; the CSV was not re-fetched. Non-glaucoma does not imply absence of other pathology. "
            "Source hashes, per-panel display bounds and full selection scores are in report_views_thesis_meta.json.",
            "",
            "**Tiếng Việt.** Các hình dùng chung quy trình. Trục chiều sâu có độ lệch chuẩn lớn nhất của hồ sơ "
            "cường độ trung bình; các trục còn lại giữ thứ tự nguồn. Dải chiếu lấy đỉnh hồ sơ trung bình cộng/trừ "
            "16 pixel, giới hạn trong thể tích (33 pixel ở đây), không phải lớp RNFL được phân đoạn giải phẫu. "
            "AIP giữ trung bình số thực; MIP lấy cực đại. Chọn lát cắt có độ lệch chuẩn lớn nhất sau giới hạn "
            "P1/P99 trong khoảng chỉ số 30-169 trên từng trục không phải chiều sâu; khi bằng nhau, lấy chỉ số đầu. "
            "Lát cắt không nhất thiết ở trung tâm hoặc đại diện lâm sàng. Trục/chỉ số nguồn bắt đầu từ 0; "
            "chiều sâu lát cắt tăng từ trên xuống.",
            "",
            "Mỗi ô giữ đủ 200 x 200 pixel, dùng giới hạn hiển thị tuyến tính P1/P99 riêng ghi trong metadata; "
            "không so sánh trực tiếp độ sáng giữa các ô. Không khử nhiễu, chỉnh sửa hay cắt xén. Tỷ lệ pixel "
            "1:1 theo mảng không phải tỷ lệ vật lý đã hiệu chuẩn. Nhãn từ figures/report_views_meta.json đã "
            "được đối chiếu CSV nguồn Harvard-GF trước đó theo thông tin người dùng; lần này không tải lại CSV. "
            "Không glaucoma không đồng nghĩa với không có bệnh lý khác. Hash nguồn, giới hạn hiển thị và điểm "
            "chọn lát cắt được lưu trong report_views_thesis_meta.json.",
            "",
        ]
    )
    explicit = os.environ.get("DRIVE_SYNC_DIR")
    mounted = Path("/content/drive")
    destination = None
    if explicit:
        candidate = Path(explicit).expanduser()
        if not candidate.is_absolute() or not candidate.is_dir():
            raise RuntimeError("DRIVE_SYNC_DIR must be an explicitly provided, existing absolute directory")
        if os.name == "nt" and candidate.as_posix().lower().split(":")[-1].startswith("/content"):
            raise RuntimeError("Refusing a fake Windows /content Drive destination")
        destination = candidate / "projection_thesis"
    elif os.name == "posix" and os.path.ismount(mounted) and (mounted / "MyDrive").is_dir():
        destination = mounted / "MyDrive/MasterBKDN/Thesis/projection_thesis"
    metadata["drive_sync"] = {
        "status": "pending" if destination is None else "verified-copy",
        "destination": str(destination) if destination else None,
        "reason": "No verified explicit DRIVE_SYNC_DIR or mounted POSIX Colab Drive"
        if destination is None
        else "Explicit existing destination or detected POSIX mount; copied bytes verified by SHA-256",
    }
    captions.extend(["## Delivery", "", f"Drive sync: {metadata['drive_sync']['status']}.", ""])
    (OUT / "captions.md").write_text("\n".join(captions), encoding="utf-8")
    (OUT / "report_views_thesis_meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if destination is not None:
        if destination.resolve() == OUT.resolve():
            raise RuntimeError("Drive destination must differ from local output directory")
        destination.mkdir(parents=True, exist_ok=True)
        for source in OUT.iterdir():
            if source.is_file():
                target = destination / source.name
                shutil.copy2(source, target)
                assert sha256(source) == sha256(target)
    print(f"Output: {OUT}\nDrive sync: {metadata['drive_sync']['status']}")


if __name__ == "__main__":
    main()
