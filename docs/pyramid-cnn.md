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

## 5. Kế hoạch đọc kết quả

1. So với `simple3dcnn` **cùng 96³, cùng batch/lr/epoch, seed 42** (chạy lại baseline bằng
   `--set model.architecture=simple3dcnn`).
2. Ablation tách đóng góp:
   - `fuse_levels=0`: chỉ multi-scale head (không top-down) → đo giá trị của FPN.
   - `fuse_levels=2`: bỏ lateral 24×24 (giảm VRAM) → kiểm tra level mịn có cần fuse không.
   - `fusion_dim=32`: kiểm tra head rộng có phải nguồn lợi ích.
3. Báo cáo **balanced accuracy, MCC, AUC, F1** cùng accuracy; test lệch lớp 411/489 nên
   accuracy đơn lẻ dễ gây kết luận sai. Loại model nếu dự đoán lệch một lớp (balanced acc ≈ 0.5).
4. Ghi `sec/epoch` và peak VRAM từ W&B cạnh baseline để so chi phí.

## 6. Rủi ro

- **Chi phí 200³**: đo 33 s/step (T4) ⇒ không khả thi trong 12 h; nếu cần 200³ thật phải giảm
  `fuse_levels`/`fusion_dim` hoặc chấp nhận 2-3 epoch. 96³ là cấu hình thực nghiệm chính.
- **Overfit**: 2100 mẫu train, model 1.1M tham số; early stop patience 6, dropout 0.3, wd 0.01.
- **Không so sánh trực tiếp** với các run 200³ khử nhiễu trong `3d-backbone-sweep.md` khi khác
  resolution/tiền xử lý; chỉ dùng làm tham chiếu.
