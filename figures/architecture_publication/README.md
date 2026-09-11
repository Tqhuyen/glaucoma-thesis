# Publication Architecture Figures

These figures replace the visual design of the earlier `arch_main_nnsvg_style` draft. They borrow the references' serif typography, compact two-dimensional blocks, soft panel backgrounds, and orthogonal connections, **not MAXIM's network architecture**. The old drafts are retained unchanged.

## Files

Every figure is available as PNG (300 dpi), SVG (editable text), and PDF (vector shapes and embedded fonts).

| File stem | Purpose |
|---|---|
| `11_end_to_end_thesis` | **Recommended for the thesis body:** full connected workflow at native 6.3 in (16 cm) width; 10.5 pt block labels, 13 pt heading. Detailed stage diagrams are intentionally separate. |
| `10_end_to_end_hybrid3d` | **Hybrid 3D version:** a textured OCT cube, real en-face previews, shrinking feature volumes / grids, and a connected flat fusion / readout strip |
| `10_hybrid3d_verification.json` | Separate code/layout verification and SHA-256 provenance of the real input texture |
| `09_end_to_end` | **Complete connected pipeline:** raw OCT, 3D source selection, raw-derived views, preprocessing, three encoders, fusion, logits, calibrated probabilities, and the notebook's reported decision |
| `01_model_plate` | Three-panel model overview: feature extraction, CrossGate, and output head |
| `02_crossgate` | Large standalone CrossGate diagram |
| `03_encoder_details` | Three-panel encoder detail: ResNeXt3D, its executed residual block, and MaxViT |
| `04_resnext3d` | Standalone 3D stage diagram for portrait-page use |
| `05_residual_block` | Standalone executed residual block |
| `06_maxvit` | Standalone MaxViT stage diagram |
| `07_overview` | Standalone three-branch overview |
| `08_output_head` | Standalone classifier and test-time probability calculation |
| `verification.json` | Dependency versions, source hashes, shape/equation checks, and layout QA |

Use the combined plates at full landscape width. Use the standalone panels on a normal portrait thesis page instead of shrinking all three panels into a narrow column.

For a regular portrait thesis page, insert **`11_end_to_end_thesis.pdf` at 16 cm wide** (native size 16.00 x 20.32 cm). Do not reduce it to half-page or column width. Labels are physically larger at this placement than those of the 20-inch-wide detailed hybrid diagram scaled down to 16 cm. The minimum label is the 9.5 pt dashed-arrow legend; block text is 10.5-11 pt. Keep the caption below in the document rather than shrinking it into the image. `09_end_to_end` and `10_end_to_end_hybrid3d` remain available for larger-format/detail views.

### Caption for the thesis workflow

**English.** End-to-end workflow for OCT-based glaucoma classification. The 3D branch uses raw or bilateral-filtered volumes, while both en-face views remain raw-derived. Paired augmentation is applied only during training; all inputs are converted to float32 and divided by 255. Independent MaxViT encoders and ResNeXt3D feed 256-D projections and CrossGate fusion. Training uses class-weighted cross-entropy on raw logits. Dashed arrows show notebook test processing: temperature T is fitted on validation logits. The current notebook selects threshold tau on uncalibrated validation probabilities but applies it after test calibration, an existing evaluation inconsistency. Labels 0/1 denote non-glaucoma/glaucoma. GAP denotes global average pooling; LN2d denotes LayerNorm2d.

**Tiếng Việt.** Quy trình phân loại glaucoma từ thể tích OCT. Nhánh 3D sử dụng dữ liệu raw hoặc lọc bilateral; hai ảnh chiếu en-face vẫn được tạo từ dữ liệu raw. Tăng cường đồng bộ chỉ áp dụng khi huấn luyện, sau đó đầu vào được chuyển sang float32 và chia cho 255. Ba nhánh mã hóa tạo đặc trưng, chiếu về 256 chiều và hợp nhất bằng CrossGate. Nét đứt biểu diễn hậu xử lý kiểm thử. T được khớp trên logits validation; notebook hiện chọn ngưỡng tau trước calibration nhưng áp dụng sau calibration trên test, là một bất nhất đánh giá cần lưu ý. Nhãn 0/1 tương ứng không glaucoma/glaucoma; GAP là pooling trung bình toàn cục, LN2d là LayerNorm2d.

`10_end_to_end_hybrid3d` preserves the scope of `09_end_to_end` in a landscape design. Follow the three feature-extraction rows left to right, then the bottom readout strip **right to left**, following its arrows. Use the SVG/PDF at landscape or large-format size; the full stage-level figure is not intended for a narrow paper column.

The input cube is a **composite illustration**, not a literal reconstructed retinal surface: its front and side contain two central B-scans, while its top contains Full AIP. The default texture file is the existing local `raw_2404.npy` cache from Harvard-GF (`harvardairobotics/Harvard-GF`). Display contrast is percentile-adjusted; this is not model-input normalization. The two en-face thumbnails use the code's raw-derived Slab MIP and Full AIP. Feature-map colors and grids are schematic, not measured activation values. Depth/size effects are not drawn to a linear scale. MaxViT stems execute before the first displayed stage but are omitted visually.

## Code Fidelity

Source: `scripts/final_model.py` and `notebooks/3d_glaucoma_final_2x2d_3d_crossgate.ipynb`. Source hashes are recorded on each render. Dependency-specific checks use the locally installed **timm 1.0.29** and **PyTorch 2.13.0+cpu**; they do not establish which timm version a remote training session installed.

- ResNeXt3D: stem followed by seven residual blocks. Stage strides are `1`, then `(2,1)`, `(2,1)`, `(2,1)`; stride 2 means `(2,2,2)`. The constructor's first `(2,2,1)` entry is not executed.
- The residual branch is `Conv1 -> GN -> ReLU -> grouped Conv3 -> GN -> ReLU -> Conv1`. Its output is added to the shortcut, then passed through ReLU. `g3` is declared but is **not called**; no post-Conv1 GroupNorm is drawn. The shortcut is identity only when channels and spatial shape match, otherwise `Conv1(stride=s) -> GN`.
- The two MaxViT encoders have independent weights. Each has stage depths `2/2/5/2`, final `LayerNorm2d`, global average pooling and a 512-D output. The removed ImageNet classifier is not drawn. Each stage box abstracts MBConv, window attention with FFN, and grid attention with FFN.
- Each repository projection is a separate Linear followed by ReLU. `p3` is `(B,256)`; the 2D tokens are stacked, not added, into `T2D` of shape `(B,2,256)`.
- CrossGate is asymmetric: `p3` supplies the query and residual; LayerNorm is applied only to `T2D` before it supplies key/value inputs. The MHA box includes PyTorch's internal learned query/key/value and output projections.
- Eight attention heads, 32 dimensions per head, attention dropout 0.1. One learned scalar `alpha` gates the complete attention output, not individual channels or views. `alpha` starts at 0.5, giving `sigmoid(alpha) = 0.622459...` initially.
- The classifier is exactly `Linear(256,2)` returning raw logits. Dashed arrows show notebook post-processing, not network layers. `T` is fitted on validation logits. No extra GELU, Dense, reciprocal gate, or post-fusion LayerNorm has been invented to imitate MAXIM.
- Registered parameter counts: **58,304,467 total**, **28,543,736 per MaxViT branch**. The total includes **1,600 unused g3 parameters**, so it is not a count of parameters active in the forward graph.

The panels share tensor names rather than connecting arrows across panel boundaries. Main plate panel (b) expands panel (a)'s CrossGate box; panel (c) consumes its `z`. Channel-first shapes omit batch unless `B` is shown. In the notebook, anatomical depth is moved to the last spatial axis; do not interpret PyTorch's first spatial dimension as anatomical depth.

## Preprocessing and Evaluation Caveats

`01_model_plate` begins **after preprocessing** and does not claim that the three inputs are independent acquisitions. **Use `09_end_to_end` for the complete connected raw-OCT-to-prediction pipeline.** The left input path selects raw or bilateral for a given experiment, not two simultaneous 3D branches. Its anatomical axis is inferred from the raw data even when the selected volume is denoised.

- Both en-face views come from the raw volume cache. The bilateral experiment replaces **only the 3D volume**, not the 2D cached views. Augmentation is paired and inputs are converted to float and divided by 255.
- Slab MIP uses a slab centered on the peak of the mean depth-intensity profile, up to 33 depth samples at 200-cube resolution. It is not a segmented RNFL slab. Full AIP averages all depth samples. Views are resized to 224 square.
- The notebook tunes its decision threshold on **uncalibrated validation probabilities**, then applies that threshold to **calibrated test probabilities**. Its bootstrap call uses the default threshold 0.5 instead of the tuned threshold. These are existing evaluation inconsistencies, not fixed by the renderer.
- The original output panel ends at symbolic class probabilities. `09_end_to_end` additionally shows the notebook's thresholded decision with an explicit warning about the threshold/calibration mismatch. Neither figure invents test results or depicts a corrected threshold protocol as if it were the implementation.

## Reproduce

From the repository root:

```powershell
python ".opencode/skills/deep-arch-diagram/examples/final_model_publication.py"
```

The renderer verifies CPU MaxViT feature maps and a synthetic end-to-end forward pass, checks the CrossGate equation numerically, checks 200-cube 3D shapes on the meta device, and rejects text/block overlaps, text outside blocks, and orthogonal arrows passing through labels. These are structural checks, not training or convergence experiments. No pretrained weights or clinical data are downloaded for verification.

Render the hybrid 3D version:

```powershell
python ".opencode/skills/deep-arch-diagram/examples/final_model_hybrid3d.py"
```

This renderer **requires an existing real volume** for its textures, defaulting to `%TEMP%/gf_vol_cache/raw_2404.npy`. Supply `--volume "path/to/raw_volume.npy"` to use another uint8 `(200,200,200)` volume. It fails explicitly if the file is unavailable rather than substituting fabricated scans. It does not download data or weights, and records the texture filename and hash. Structural verification still uses synthetic tensors and untrained model weights. The training code and the earlier flat diagrams are unchanged.

To synchronize to a mounted Google Drive folder:

```powershell
python ".opencode/skills/deep-arch-diagram/examples/final_model_publication.py" --drive-dir "G:\My Drive\MasterBKDN\Thesis\architecture_figures"
```

The parent folder must exist. `DRIVE_SYNC_DIR` is also supported. On Windows, a local `C:\content\drive\...` directory is **not** treated as proof of a Google Drive mount. Drive synchronization remains pending on this machine until an actual destination is configured.
