# SwinUNETR (Encoder) cho phân loại Glaucoma trên OCT 3D — Lý thuyết & Quyết định thiết kế

> Tài liệu ghi lại toàn bộ phân tích lý thuyết, so sánh mô hình và cách triển khai
> việc chuyển **SwinUNETR (MONAI)** từ tác vụ phân đoạn sang tác vụ **phân loại nhị phân**
> glaucoma/normal trên dữ liệu **Harvard-GF** (volume OCT 200³ uint8).

---

## 1. Bối cảnh bài toán

| Mục | Giá trị |
|---|---|
| Nhiệm vụ | Phân loại nhị phân (glaucoma / normal) trên toàn bộ volume |
| Dữ liệu | Harvard-GF: volume OCT 200³ uint8, **dense** (mọi voxel có tín hiệu) |
| Kích thước tập | Nhỏ (vài trăm scan) → cần transfer learning, không nên train from scratch mô hình lớn |
| Ràng buộc hạ tầng | Batch size 2–4 (VRAM), pipeline model-agnostic, chạy Colab / vast.ai / SLURM |
| Lưu trữ | Luôn giữ raw 200³ trên disk; model input có thể resize on-the-fly |

**Các "bẫy" đã được tài liệu hóa trong AGENTS.md khi huấn luyện 3D trên OCT:**
1. Pooling không được đè nát lớp RNFL mỏng → pool bất đẳng hướng `(2,2,1)`, chỉ `AdaptiveAvgPool3d(1)` ở cuối.
2. BatchNorm hỏng khi batch 2–4 → dùng GroupNorm / InstanceNorm.
3. Dữ liệu uint8 0–255 → `/255.0` (minmax) hoặc robust clip 1/99 percentile.
4. Logits thô phải đi với `CrossEntropyLoss`.

---

## 2. Vì sao các backbone "voxel" phổ biến (point cloud) KHÔNG áp dụng được

Một danh sách backbone thường được gợi ý cho "dữ liệu voxel" thực chất thuộc miền
**point cloud / LiDAR** — đã kiểm chứng trực tiếp trên README từng repo:

| Mô hình | README nói gì (trích) | Miền thực | Áp cho OCT 200³ đặc? |
|---|---|---|---|
| **Swin3D / Swin3D++** (Microsoft) | *"pretrained transformer backbone for 3D **indoor scene understanding**"; "self-attention on **sparse voxels**"*; pretrain trên **Structured3D** (point cloud, cRSE: XYZ/RGB/NORM); fine-tune ScanNet/S3DIS | Point cloud → sparse voxel occupancy | ❌ |
| **VoTr** (PointsCoder) | *"Voxel Transformer for **3D object detection**"*, dựa trên **OpenPCDet**, chạy **KITTI & Waymo**; *"checkpoints will not be released"*; cần GPU 32GB, 60+ epochs | LiDAR point cloud, sparse conv | ❌ |
| **DVST** | Deformable attention cho 3D detection (Waymo) | Như VoTr | ❌ |
| **MinkowskiEngine** (NVIDIA) | *"auto-diff library for **sparse tensors**"*; ví dụ classification = **ModelNet40** (CAD point cloud) | Point cloud → `sparse_quantize` | ❌ |
| **SpConv / OpenShape / mmdetection3d** | sparse conv; CLIP text-image cho shape; detection framework | Point cloud / CAD | ❌ |

### Lý do cốt lõi (domain mismatch)
- Chữ **"voxel"** trong các repo này = *lượng tử hóa point cloud thành ô occupancy để tạo
  **sparse tensor***, không phải volume y khoa đặc.
- OCT 200³ là **dense**: gần như mọi voxel có tín hiệu → **không có "ô trống"** để sparse
  tiết kiệm. Đưa 200³ vào sparse-conv = 8 triệu coordinate/mẫu, mất sạch lợi thế + thêm overhead.
- Pretrained weights train trên indoor/LiDAR/CAD → không mang tri thức giải phẫu võng mạc.
- Trên dữ liệu nhỏ, backbone **pretrained đúng domain** (SwinUNETR/SSL-CT) thắng mô hình
  point-cloud dù "mới hơn".
- Rào cản triển khai: VoTr **không release weights** + khóa PyTorch 1.5/spconv 1.2;
  MinkowskiEngine phải **compile CUDA trên Linux** (không hỗ trợ Windows); cả hai dùng
  BatchNorm → dính "bẫy #2".

---

## 3. Lựa chọn: SwinUNETR encoder + MONAI

SwinUNETR là bản "Swin cho dense 3D medical" hợp lệ — cùng triết lý shifted-window
nhưng được thiết kế và pretrained cho **volume y khoa đặc** (không phải point cloud).

### 3.1 Lý thuyết: vì sao transformer 3D (Swin) vượt trội cho phân loại volume

1. **Ngữ cảnh toàn cục (global context)**
   - CNN truyền thống dùng kernel nhỏ (3×3×3) → trường thụ cảm (receptive field) hẹp,
     muốn nhìn xa phải chồng nhiều lớp (thông tin càng xa càng bị "loãng").
   - **Self-attention** tính tương quan giữa *mọi cặp token* trong cửa sổ ngay từ layer đầu.
   - **Shifted Window Attention** (Liu et al., *Swin Transformer*, ICCV 2021):
     - Chia feature map 3D thành các window kích thước `W³` (vd 7³).
     - Mỗi block: cửa sổ bình thường → rồi **dịch cửa sổ** (`shift_size = W//2`) để block kế
       tiếp nhìn qua biên giới cửa sổ → thông tin lan truyền giữa các window.
     - Độ phức tạp **tuyến tính** theo số token (thay vì bình phương như global attention),
       nên khả thi trên volume 3D lớn.
   - Với OCT, dấu hiệu glaucoma (độ mỏng RNFL, lõm đĩa thị) phân bố rải rác trong volume →
     mô hình cần tương quan tầm xa giữa các B-scan.

2. **Pretrained self-supervised khổng lồ (điểm "ăn tiền")**
   - Tang et al., *Self-supervised pre-training of Swin transformers for 3D medical image
     analysis* (CVPR 2022): pretrain encoder Swin trên **5.050 volume CT không nhãn** bằng
     **masked volumetric inpainting** (che ngẫu nhiên các khối, mô hình tự điền lại).
   - Khi fine-tune trên tập nhỏ (Harvard-GF), mô hình đã có "tri thức nền" về giải phẫu →
     đạt độ chính xác cao hơn hẳn train from scratch.

3. **Đặc trưng đa quy mô (hierarchical)**
   - Giống ResNet nhưng bằng transformer: qua mỗi stage, `PatchMerging` giảm độ phân giải
     không gian ½ và tăng gấp đôi số kênh.
   - Stage thấp giữ chi tiết cục bộ; stage cao mang ngữ nghĩa vĩ mô → vector đặc trưng đầy đủ
     cho phân loại toàn volume.

### 3.2 Sơ đồ chuyển Segmentation → Classification

```
Input x (B,1,96,96,96)   ← resize on-the-fly 200³, yêu cầu chia hết cho 32
   │
   ▼
SwinUNETR (MONAI) chỉ dùng ENCODER:
   ├── swinViT  (SwinTransformer): patch embed (stride 2) + 4 stage Swin + 4× PatchMerging
   │        → hidden_states_out[0..4]   (4 feature maps đa tỉ lệ)
   │           hidden[4] = token sâu nhất: 16×feature_size kênh, spatial = input/32
   └── (decoder5..1, encoder1..4/10, out) → ❌ XÓA BỎ (không bao giờ được gọi)
   │
   ▼
AdaptiveAvgPool3d(1)   → (B, 16×feature_size)
   │
   ▼
MLP head: Linear(16f → max(64, 16f/4)) → ReLU → Dropout → Linear(→ num_classes)
   │
   ▼
{"logits": (B, 2)}   ← contract pipeline: logits thô + CrossEntropyLoss
```

**Xác minh từ source MONAI** (`monai/networks/nets/swin_unetr.py`, đã đọc trực tiếp):
- `SwinUNETR.forward` gọi `hidden_states_out = self.swinViT(x, self.normalize)` rồi dùng
  `hidden_states_out[0..4]`; `encoder10` có `in_channels=16*feature_size`.
- Ràng buộc không gian: input phải chia hết cho `patch_size**5` = **32** (patch_size mặc định 2).
  → **112 KHÔNG hợp lệ cho SwinUNETR**; dùng **96 / 128** (weight SSL chuẩn train ở 96³).
- `SwinTransformer` và các khối của nó nằm trong `__all__` của module MONAI.
- Checkpoint SSL chính thức (`model_swinvit.pt`) có key dạng `module.<name>` và chỉ phủ
  **encoder** → map được thẳng vào `backbone.swinViT.*` (bỏ tiền tố `module.`).

---

## 4. Danh sách sweep các model cần thử (đã chốt)

| # | `architecture` | Họ | Pretrained? | Params ≈ | VRAM bs2 fp16 | Ghi chú |
|---|---|---|---|---|---|---|
| 0 | `simple3dcnn` | CNN | — | ~5M | rất thấp | **Baseline** (có sẵn), chạy ở 200³ |
| 1 | **`swinunetr`** | Swin 3D transformer | ✅ SSL 5050 CT | 40–60M | ~10–14GB @96³ | **Primary** — encoder + head, xoá decoder |
| 2 | `unetr` (ViT-B 3D) | ViT | ❌ | ~86M | ~12–16GB @112³ | ViT from scratch → dễ underfit; chỉ thử nếu có self-sup riêng |
| 3 | `mednext` | ConvNeXt-3D | ❌ | ~45M | ~14–18GB | Kiến trúc mới họ CNN nhưng from scratch → rủi ro |
| 4 | `densenet121_3d` | DenseNet (InstanceNorm) | ❌ | ~15M | thấp | Đối chứng cũ (đã có kết quả 96³) |
| 5 | `medicalnet_resnet50` | ResNet3D | ✅ CT/MRI y khoa | ~46M | ~8–10GB | Rẻ & bền; nhưng BatchNorm → cần batch 4–8 hoặc cố định BN |
| 6 | (notebook) `3dino-vit` | ViT self-sup | ✅ tự SSL trên OCT | — | — | Hướng đang chạy ở notebook — giữ riêng |

**Logic sắp xếp:** dữ liệu nhỏ + batch 2–4 → *pretrained beats from-scratch*; ViT random-init
(#2, #3) gần như chắc chắn thua pretrained conv/Swin → xếp phụ. SwinUNETR thắng cả 3 tiêu chí:
SOTA gần đây, pretrained cho 3D medical, nằm trong chính MONAI.

---

## 5. Triển khai trong repository

### 5.1 `models/glaucoma/model.py` — `SwinUNETRClassifier`

- **Lazy import** `monai.networks.nets.SwinUNETR` bên trong `__init__` (repo không bắt buộc có
  MONAI khi import module).
- Build `SwinUNETR(...)` rồi **xóa** các thuộc tính `encoder1..4`, `encoder10`, `decoder5..1`,
  `out` → chỉ còn `swinViT` (giảm tham số, đúng yêu cầu "xoá decoder").
- `forward`: `hidden = backbone.swinViT(x, backbone.normalize)` → `pool(hidden[4])` →
  `classifier` → `{"logits": ...}`.
- `_load_pretrained(path)`: đọc `ckpt["state_dict"]` (hoặc raw), bỏ tiền tố `module.`,
  match key có trong `backbone.swinViT` cùng shape → `load_state_dict(..., strict=False)`.
  Không có key khớp → `ValueError`.
- `freeze_encoder=True`: gán `requires_grad=False` toàn bộ `backbone` (chỉ head trainable).
- `build_model(cfg)` dispatch theo `model.architecture`:
  - `simple3dcnn` → giữ nguyên hành vi cũ.
  - `swinunetr` → kiểm tra `data.model_input_shape % 32 == 0`, thiếu/sai thì `ValueError` rõ ràng.

### 5.2 `models/glaucoma/data.py` — resize on-the-fly

- `GlaucomaNpyDataset.__init__` nhận thêm `model_input_shape` (int).
- `__getitem__`: cast float32 → normalize → `F.interpolate(mode="trilinear",
  align_corners=False)` về `(s,s,s)³`. Raw 200³ vẫn giữ trên disk.
- `build_dataloaders` đọc `data.model_input_shape` và truyền xuống 3 split.

### 5.3 `pipeline/config.py` — keys optional (old configs vẫn hợp lệ)

| Key | Type | Ràng buộc |
|---|---|---|
| `data.model_input_shape` | int | `>= 32` (Swin cần chia hết 32; kiểm tra sớm ở builder) |
| `model.feature_size` | int | `12..192`, bội của 12 |
| `model.freeze_encoder` | bool | — |
| `model.use_checkpoint` | bool | — |
| `model.pretrained_ckpt` | str | đường dẫn file `.pt` |

### 5.4 `pyproject.toml`

- `glaucoma = ["monai>=1.2"]` → `make setup` tự cài MONAI trên máy chạy thật.

### 5.5 Ví dụ config sweep (chạy trên GPU)

```yaml
run_name: "swinunetr-ssl-finetune"
seed: 42

model:
  type: "glaucoma"
  architecture: "swinunetr"
  input_channels: 1
  num_classes: 2
  feature_size: 48
  dropout: 0.3
  freeze_encoder: true
  use_checkpoint: false
  pretrained_ckpt: "model_swinvit.pt"

data:
  data_dir: "glaucoma_all"
  cache_in_ram: false
  num_workers: auto
  normalize: "minmax"
  model_input_shape: 96

train:
  epochs: 30
  batch_size: 2
  grad_accum_steps: 4
  lr: 1.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.05
  amp: true
  ...
```

---

## 6. Kết quả kiểm chứng (MONAI 1.6.0, CPU, torch 2.13)

- Build encoder-only: `backbone` chỉ còn child `swinViT`; forward `32³ → logits (1,2)` ✓
- `freeze_encoder=True`: toàn bộ param trainable nằm trong `classifier`; backward + optimizer step chạy ✓
- Validator chặn `model_input_shape=100` với thông báo rõ ✓
- Dataset resize 32→16³ đúng chuẩn minmax (0–1, float32) ✓
- `ruff check`: không phát sinh lỗi mới (7 lỗi còn lại thuộc notebook archive, được chấp nhận theo AGENTS.md)
- `pytest tests`: **13 passed** ✓

---

## 7. Các bước tiếp theo (khuyến nghị)

1. Chạy **overfit sanity gate** trên GPU với config Swin (`make sanity` — 10 mẫu, 100 epoch,
   dropout/wd/warmup = 0) trước khi chạy thật: loss → ~0, train acc → 1.0 tức code/arch đúng.
2. So sánh: `swinunetr` (SSL, `freeze_encoder=true` → un-freeze sau 1–2 epoch) vs `simple3dcnn`
   baseline vs `3dino-vit` notebook.
3. Nếu VRAM < 16GB: giảm `model_input_shape` về 96, bật `use_checkpoint: true`, hoặc dùng
   `medicalnet_resnet50` (đã pretrain y khoa) làm phương án dự phòng nhẹ.
4. Mọi run phải log lên **wandb** (project `glaucoma-thesis`) — pipeline đã tự làm khi
   `logging.wandb: true` và `.env` có `WANDB_API_KEY`.

---

## 8. Tài liệu tham khảo

- MONAI Network architectures (docs): `https://monai-dev.readthedocs.io/en/fixes-sphinx/networks.html`
- Source `swin_unetr.py` (MONAI, đã đọc để xác minh cấu trúc): `monai/networks/nets/swin_unetr.py`
- Swin UNETR: *Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI*
  (Hatamizadeh et al., arXiv:2201.01266)
- SSL SwinUNETR: *Self-supervised pre-training of Swin transformers for 3D medical image analysis*
  (Tang et al., CVPR 2022) — checkpoint SSL `model_swinvit.pt`
- Swin Transformer: *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*
  (Liu et al., ICCV 2021)
- Các repo point-cloud đã kiểm chứng (không áp dụng cho OCT đặc): microsoft/Swin3D,
  PointsCoder/VOTR, NVIDIA/MinkowskiEngine, open-mmlab/mmdetection3d
