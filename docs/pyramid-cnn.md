# Pyramid 3D CNN cho phân loại glaucoma (OCT)

Thí nghiệm kiến trúc **pyramid/FPN 3D** cho bài toán glaucoma trên Harvard-GF, chạy trong
cùng pipeline với `Simple3DCNN` và các backbone 3D khác. Mã nguồn:
`models/glaucoma/model.py` (`Pyramid3DCNN`, `architecture: pyramid3dcnn`).

## 1. Động lực

`Simple3DCNN` hiện tại lấy đặc trưng bằng global average pooling sau 4 khối conv. Cách đó giữ
được tín hiệu toàn cục nhưng **mất bố cục không gian**: độ dày RNFL thay đổi cục bộ, còn hình
dạng globe/đĩa thị chỉ hiện ở thang thô. Pyramid 3D giải quyết cả hai:

| Thang | Thông tin giữ lại |
|---|---|
| Level 0-1 (H/W lớn) | chi tiết lớp võng mạc, ranh giới RNFL mỏng |
| Level 2-3 (H/W nhỏ, depth đầy đủ) | hình dạng toàn cục, phân bố lớp theo trục depth |

Top-down fusion đưa ngữ nghĩa thô xuống level mịn (FPN); head lấy **mọi** level nên classifier
thấy đồng thời chi tiết và ngữ cảnh.

## 2. Kiến trúc (`Pyramid3DCNN`)

Giữ nguyên 3 fix bắt buộc của repo:

1. **GroupNorm** (`norm: group`, `norm_groups: 8`) — ổn định với batch 2-4.
2. **Pooling dị hướng `(2,2,1)`** — H/W giảm, depth giữ nguyên 200, RNFL không bị nghiền.
3. **Raw logits + CrossEntropyLoss**, `dropout` trước classifier.

Thêm phần pyramid:

- Stem `Conv3d(1→16, 3³) + GroupNorm + ReLU` tại độ phân giải gốc.
- 4 level, channels `(16, 32, 64, 128)`; mỗi level: conv 3³ + GN + ReLU + residual 3³, rồi pool.
- **Lateral 1×1** cho `fuse_levels` level thô nhất (mặc định 3) → `fusion_dim` (64).
- **Top-down**: upsample trilinear level sâu hơn về đúng H/W của level trên, cộng lateral,
  smooth bằng conv 3³ + GN + ReLU.
- **Head đa thang**: global average pool từng level (level không fuse giữ kênh gốc, level fuse
  dùng `fusion_dim`), concat → `Linear(→64) → ReLU → Dropout(0.3) → Linear(→2)`.

Shape với input `1×200×200×200`, `fuse_levels=3`:

| Level | Feature map (C × D × H × W) | Vai trò |
|---|---|---|
| stem | 16 × 200 × 200 × 200 | conv đầu, chưa pool |
| L0 | 16 × 200 × 100 × 100 | raw → head |
| L1 | 32 × 200 × 50 × 50 | lateral → 64 |
| L2 | 64 × 200 × 25 × 25 | lateral → 64 |
| L3 | 128 × 200 × 12 × 12 | lateral → 64, đỉnh top-down |
| head | `16 + 3×64 = 208` | concat sau GAP |

Tham số: **1,129,298** (pyramid, mặc định) vs **881,122** (`simple3dcnn`, cùng hidden) —
+28%, chủ yếu ở lateral/smooth và head rộng hơn.

## 2b. Biến thể `PyramidResNeXt3D` (ResNeXt + FPN)

`PyramidResNeXt3D` giữ nguyên **block/stage của encoder thắng sweep** (`single3d-resxt3d`:
grouped bottleneck, `features=(32,64,128,192)`, `groups=8`, `base_width=8`, GroupNorm,
stride `(2,2,1)` rồi `(2,2,2)×3`) nhưng mở 4 level và gắn pyramid:

- stem conv 3³ → 32 kênh tại độ phân giải gốc;
- mỗi level: 1 block giảm mẫu + 1 block giữ mẫu (đúng chuỗi của `Enc3DResNeXt`);
- lateral 1×1 → `fusion_dim` (mặc định 128) cho `fuse_levels` level cuối, top-down
  trilinear + add + smooth 3³;
- head GAP đa thang: level không fuse giữ kênh gốc, level fuse dùng `fusion_dim`;
- `fuse_levels=0` = ResNeXt + multi-scale head (ablation).

Tham số: **1,608M** (mặc định) vs 1,129M của `Pyramid3DCNN`. Vì block grouped và các level
sau nhỏ, compute thấp hơn pyramid CNN thuần ở cùng 96³.

Chạy: kernel `trnquanghuyn/glaucoma-pyramid-resnext-bilateral96`, 15 epoch, batch 4 × accum 2,
lr 1e-4, seed 42; so trực tiếp với `single3d-resxt3d` (77,11% test) và run pyramid 96³ (74,00%).

## 3. Cấu hình

`configs/glaucoma.yaml` là config glaucoma duy nhất (raw 200³). Hai khóa mới có mặt với giá trị
mặc định, không ảnh hưởng `simple3dcnn`:

```yaml
model:
  fusion_dim: 64
  fuse_levels: 3   # 0 = tắt top-down, chỉ còn multi-scale head
```

Chạy pyramid bằng override (không tạo config resolution khác):

```bash
python -m pipeline.train --cfg configs/glaucoma.yaml \
  --set model.architecture=pyramid3dcnn model.fusion_dim=64 model.fuse_levels=3 \
        run_name=pyramid-bilateral96-s42
```

Thực nghiệm chính chạy trên **Kaggle GPU**: `notebooks/3d_glaucoma_pyramid_kaggle.ipynb`
(Bilateral-96 từ HF, batch 4 × grad-accum 2, 15 epoch, AMP fp16, log W&B + artifact).
Notebook chỉ dùng GPU; không chạy train trên máy local.

**Vì sao 96³ chứ không 200³:** đo thực tế trên T4 với raw 200³, batch 2 × accum 4 là
**33 s/optimizer-step** (W&B `pyramid-raw200-s42`, 30 step đầu) ⇒ 263 step/epoch ≈ **2,4 h/epoch**,
15 epoch ≈ **36 h** — vượt giới hạn session Kaggle (và gần gấp đôi quota GPU tuần). 96³ giảm
compute ~(96/200)³ ≈ 0,11 ⇒ ~4 h cho 15 epoch, cùng nhóm so sánh 96³ của `3d-backbone-sweep.md`.
Muốn thử 200³ chỉ nên chạy 2-3 epoch như cost/accuracy probe, không đủ để kết luận.

Đo chi phí step/VRAM trên GPU:

```bash
python scripts/bench_pyramid.py --res 96 --batch-sizes 1,2,4 --steps 3
```

## 4. Kiểm chứng đã chạy

- `ruff check` + `ruff format --check` trên các file thay đổi: sạch.
- `pytest tests/test_pyramid.py -q`: forward/backward, gradient hữu hạn, `fuse_levels` 0/2/3/9,
  head nhận đủ level, GroupNorm bất biến theo batch size, validate config (kể cả giá trị sai).
- End-to-end qua `pipeline.train --smoke` trên dữ liệu tổng hợp 32³ (32 train / 8 val / 8 test,
  tín hiệu phân biệt được): chạy hết train → eval → test → lưu checkpoint/curves, val/test acc 1.0.
  Đây chỉ là kiểm tra wiring, không phải kết quả khoa học.
- Toàn bộ `pytest tests` trên Windows có 38 test fail sẵn ở các module `test_final_*`
  (`OSError: could not get source code`, lỗi môi trường), không liên quan thay đổi này.

## 5. Kết quả run 96³ (seed 42)

Run: W&B `glaucoma-thesis/pyramid-bilateral96-s42` (kernel private
`trnquanghuyn/glaucoma-pyramid-bilateral96`). Bilateral-96, batch 4 × grad-accum 2, lr 1e-4,
tối đa 15 epoch, early stop patience 6. GPU T4, ~3,9 s/step, dừng ở step 2300 (≈8,7 epoch).

- **Val tốt nhất:** step 1700 (≈6,5 epoch), loss **0,5222**, acc **76,67%**, F1 **76,05%**.
- **Test (best model, 900 mẫu):**

| Metric | Giá trị |
|---|---:|
| Loss | 0,5160 |
| Accuracy | **74,00%** |
| Balanced accuracy | **74,37%** |
| MCC | **0,486** |
| F1 (macro) | **73,99%** |
| F1 (glaucoma +) | 74,57% |
| Precision / Recall (glaucoma +) | 79,58% / 70,14% |

Confusion (suy ra từ precision/recall + 489 dương / 411 âm): TP 343, FP 88, FN 146, TN 323.

![Training curves](../figures/pyramid_bilateral96_s42_curves.png)

**So sánh tham chiếu** (cùng nhóm 96³ trong `3d-backbone-sweep.md`, không cùng recipe):

| Model | Test acc | Balanced acc | Test F1 | MCC |
|---|---:|---:|---:|---:|
| `Pyramid3DCNN` (run này) | **74,00%** | **74,37%** | **73,99%** | **0,486** |
| `Simple3DCNN` (matched, cùng recipe) | 72,00% | 72,58% | 72,00% | 0,452 |
| `single3d-cnn3d` (tham chiếu sweep) | 74,44% | — | 77,41% | 0,483 |
| `single3d-resxt3d` (tham chiếu sweep) | **77,11%** | 77,19% | **78,36%** | **0,542** |

Baseline matched: kernel `trnquanghuyn/glaucoma-simple3dcnn-bilateral96` (cùng 96³, bs 4 × accum 2,
15 epoch, early stop step 2390; test 72,00%; TP 322, FP 85, FN 167, TN 326).

**Đọc kết quả:** với **cùng recipe**, pyramid hơn `Simple3DCNN` **+2,0 điểm acc, +1,8 điểm balanced
acc, +0,034 MCC** — hướng nhất quán nhưng 1 seed và test 900 mẫu (SE của hiệu ≈ 2,1 điểm) nên chưa
đủ để khẳng định; cần thêm seed. So với `single3d-resxt3d` vẫn kém 3,1 điểm acc. Với 1 seed và
early stop ở 8,7 epoch, **chưa có bằng chứng top-down fusion tạo lợi ích**; cần baseline cùng recipe
+ ablation `fuse_levels=0` để tách đóng góp FPN so với multi-scale head (ablation đã bị hoãn để
dành quota cho resolution/ResNeXt).

Test đạt 70,9% ở step 600, 73,8% ở step 1200 rồi dao động 72–74% đến step 2300 ⇒ **plateau từ
~4,5 epoch**; kéo dài schedule không phải đòn bẩy chính. Lưu ý val chỉ 300 mẫu (SE ≈ 2,6 điểm),
nên "best val" ở step 1700 là chọn theo nhiễu; test 900 mẫu có CI95 ≈ ±2,9 điểm — chênh lệch
0,44 điểm so với `single3d-cnn3d` nằm trong nhiễu.

## 6. Kế hoạch đọc kết quả

1. So với `simple3dcnn` **cùng 96³, cùng batch/lr/epoch, seed 42** (đã đẩy kernel
   `trnquanghuyn/glaucoma-simple3dcnn-bilateral96`; chờ slot GPU).
2. Ablation tách đóng góp:
   - `fuse_levels=0`: chỉ multi-scale head (không top-down) → đo giá trị của FPN.
   - `fuse_levels=2`: bỏ lateral 24×24 (giảm VRAM) → kiểm tra level mịn có cần fuse không.
   - `fusion_dim=32`: kiểm tra head rộng có phải nguồn lợi ích.
3. **Anisotropic 96×96×200** (đang chờ slot): giữ nguyên depth 200, chỉ downsample H/W về 96
   từ raw 200³ trên Kaggle, rồi train cùng recipe (10 epoch, batch 4 × accum 2, seed 42).
   Storage ~6,1 GB thay vì 26,4 GB, nhưng lúc build vẫn phải tải raw 200³ (đỉnh scratch ~21 GB).
   Đây là phép so sánh sạch với run 96³: cùng H/W, chỉ khác mẫu trục depth — tách đúng giả thuyết
   "mất chi tiết lớp mỏng khi resize". So test tại cùng epoch (96³: 73,8% ở epoch 10) và dùng
   CI95 ±2,9 điểm để kết luận.
4. Báo cáo **balanced accuracy, MCC, AUC, F1** cùng accuracy; test lệch lớp 411/489 nên
   accuracy đơn lẻ dễ gây kết luận sai. Loại model nếu dự đoán lệch một lớp (balanced acc ≈ 0.5).
5. Ghi `sec/epoch` và peak VRAM từ W&B cạnh baseline để so chi phí.

## 7. Rủi ro

- **Chi phí 200³**: đo 33 s/step (T4) ⇒ không khả thi trong 12 h; nếu cần 200³ thật phải giảm
  `fuse_levels`/`fusion_dim` hoặc chấp nhận 2-3 epoch. 96³ là cấu hình thực nghiệm chính.
- **Overfit**: 2100 mẫu train, model 1.1M tham số; early stop patience 6, dropout 0.3, wd 0.01.
- **Không so sánh trực tiếp** với các run 200³ khử nhiễu trong `3d-backbone-sweep.md` khi khác
  resolution/tiền xử lý; chỉ dùng làm tham chiếu.
