# Kết quả ƯỚC TÍNH — mô hình chính 2×2D + 1×3D + CrossGate

> ⚠️ **ĐÂY KHÔNG PHẢI KẾT QUẢ THẬT.** Toàn bộ số trong tài liệu này là **giả sử / ước tính** được
> ngoại suy từ các kết quả đã có của các sweep độc lập. **Tuyệt đối không trích dẫn như kết quả thực nghiệm.**
> Sau khi notebook [`3d_glaucoma_final_2x2d_3d_crossgate.ipynb`](../notebooks/3d_glaucoma_final_2x2d_3d_crossgate.ipynb)
> train xong, **thay bảng mục 4 bằng số thật** (bảng mục 6) rồi mới đưa vào báo cáo/luận văn.

Mô hình đang train: **2 nhánh 2D (MaxViT-Tiny, view `slab_mip` + `aip_full`) + 1 nhánh 3D (ResNeXt3D @200³),
hợp nhất bằng CrossGate** (`scripts/final_model.py` → `FinalModel`, 58,30 M params).
Kiến trúc chi tiết: [`model-architecture.md`](model-architecture.md).

---

## 1. Cấu hình huấn luyện thực tế (đang chạy)

| Thành phần | Giá trị |
|---|---|
| Input 3D | raw 200³ (uint8 → /255) |
| Input 2D | 2 view en-face 224×224 (`slab_mip`, `aip_full`) |
| Dataset | `raw` + `bilateral` (khử nhiễu bilateral, lưu local/Drive, **không** push HF) |
| Batch / effective | bs 2 + grad-accum 8 (effective 16), AMP bf16 |
| Optimizer | AdamW, lr 2e-4, wd 1e-4, warmup 5% + cosine |
| Epochs / early-stop | 30, patience 6 |
| Seeds | 42, 43, 44 |
| Chọn checkpoint | best **validation AUC** (không dùng epoch cuối) |
| Calibration | temperature scaling trên val; threshold chọn trên val |
| Loss | CrossEntropyLoss + class weights |
| Đo test | held-out test, có bootstrap CI (1000 lần) |

---

## 2. Baseline THẬT dùng làm mốc (nguồn đã có)

Các số dưới đây là **kết quả thật** từ sweep, dùng làm cơ sở để ước tính:

| Nguồn | Test acc | Test AUC | PR-AUC | F1 | Bal.Acc | MCC | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3D đơn — `single3d-resxt3d` ([3d-backbone-sweep](3d-backbone-sweep.md)) | 0,7711 | **0,8538** | **0,8798** | 0,7836 | 0,7719 | 0,542 | 0,0482 |
| 2D đơn — MaxViT-Tiny, `slab_aip` ([2d sweep](2d-feature-extraction-sweep.md)) | 0,6889 | 0,7638 | 0,7925 | 0,7065 | 0,6889 | 0,3765 | 0,1265 |
| 2D đơn — Tiny2D, `slab_mip` ([2d sweep](2d-feature-extraction-sweep.md)) | 0,6656 | 0,7242 | 0,7509 | 0,7195 | 0,6538 | 0,3210 | 0,0549 |
| Fusion cũ (collapse, `convnextv2_tiny`) ([fusion ablation](fusion-ablation-sweep.md)) | 0,5433 | 0,5685 | 0,5981 | 0,7041 | 0,5000 | 0,0000 | 0,0104 |

**Kỳ vọng thiết kế** ([final-4branch-design](final-4branch-design.md) §4): model cuối phải
**≥ 3D đơn tốt nhất** (AUC ≥ 0,854), **cải thiện calibration** (ECE ≤ ~0,05) và **không collapse**
(bal-acc > 0,75, MCC > 0,5).

---

## 3. Giả định khi ước tính

1. Nhánh 3D giữ nguyên hành vi như `single3d-resxt3d` ⇒ **trần ~0,854 AUC** nếu fusion không bổ trợ.
2. Fusion có thể **+1 đến +3 điểm phần trăm AUC** so với nhánh 3D tốt nhất (kinh nghiệm ensemble/fusion);
   nếu vượt trần ít/không ⇒ 2D **không bổ trợ** cho 3D (một kết luận hợp lệ).
3. Nhánh 2D MaxViT đưa trực tiếp vào fusion nên **AUC 2D ~0,76**; fusion kéo calibration lên nhờ temperature scaling.
4. `aip_full` thường yếu hơn `slab_mip` (xem [2d sweep](2d-feature-extraction-sweep.md) §5) ⇒ 2 nhánh 2D có thể
   **không cộng hưởng nhiều**; cổng CrossGate có thể học **α nhỏ** (2D chỉ điều chỉnh nhẹ).
5. Mỗi số là **trung bình 3 seed**; độ lệch chuẩn giữa seed giả định **~0,01–0,02** (AUC/MCC).
6. Threshold tối ưu trên val, temperature scaling chỉ áp cho calibration ⇒ **ECE giảm mạnh** so với MaxViT gốc.

---

## 4. BẢNG ƯỚC TÍNH (GIẢ SỬ — chờ kết quả thật)

Giá trị ngoài ngoặc là **trung bình kỳ vọng**; trong ngoặc là **khoảng hợp lý** (không phải CI thống kê).

### 4.1. Test (held-out)

| Metric | Ước tính (kỳ vọng) | Khoảng hợp lý | So với 3D đơn (0,854 AUC) |
|---|---:|---:|---|
| Accuracy | 0,79 | 0,76 – 0,81 | +~2 điểm phần trăm |
| Balanced accuracy | 0,79 | 0,76 – 0,82 | +~2 |
| AUC (ROC) | **0,87** | 0,84 – 0,89 | +1 đến +3 |
| PR-AUC | **0,89** | 0,87 – 0,91 | +~1 |
| F1 | 0,80 | 0,78 – 0,82 | +~1,5 |
| Sensitivity (recall) | 0,79 | 0,75 – 0,83 | — |
| Specificity | 0,80 | 0,76 – 0,84 | — |
| MCC | 0,58 | 0,50 – 0,64 | +~0,04 |
| ECE ↓ (sau temp. scaling) | **0,05** | 0,03 – 0,07 | ≈ hoặc tốt hơn (0,048) |
| Params | 58,30 M | — | 2D chiếm 97 % |

### 4.2. Validation (dùng để chọn checkpoint)

| Metric | Ước tính | Khoảng hợp lý |
|---|---:|---:|
| Loss | 0,50 | 0,45 – 0,58 |
| AUC | 0,88 | 0,85 – 0,90 |
| PR-AUC | 0,89 | 0,87 – 0,91 |
| Balanced accuracy | 0,80 | 0,77 – 0,83 |
| F1 | 0,81 | 0,79 – 0,83 |
| MCC | 0,59 | 0,52 – 0,65 |

### 4.3. Theo seed (kỳ vọng, dùng để dự phóng mean ± std)

| Seed | AUC | Bal.Acc | MCC | F1 | ECE↓ |
|---|---:|---:|---:|---:|---:|
| 42 | 0,871 | 0,790 | 0,578 | 0,801 | 0,048 |
| 43 | 0,865 | 0,783 | 0,565 | 0,795 | 0,053 |
| 44 | 0,874 | 0,796 | 0,590 | 0,806 | 0,045 |
| **Mean ± std** | **0,870 ± 0,005** | **0,790 ± 0,007** | **0,578 ± 0,013** | **0,801 ± 0,006** | **0,049 ± 0,004** |

### 4.4. Ablation ước tính (để đối chiếu sau này)

| Cấu hình | Test AUC (ước tính) | Nhận xét kỳ vọng |
|---|---:|---|
| Chỉ 3D (ResNeXt3D) | 0,854 | mốc thật từ sweep, không phải ước tính |
| Chỉ 2D (MaxViT ×2) | 0,77 – 0,80 | 2 view có thể nhích hơn 1 view (0,764) |
| 3D + 2D (CrossGate, model chính) | **0,87** | kỳ vọng +1..+3 điểm so với chỉ 3D |
| 3D + 2D (Attention đối chứng) | 0,85 – 0,87 | nếu tương đương CrossGate ⇒ fusion nhạy cảm thấp |

### 4.5. X-AI ước tính (định tính)

| Tín hiệu | Kỳ vọng |
|---|---|
| Cổng CrossGate σ(α) | 0,45 – 0,70 (2D có đóng góp nhưng nhỏ); nếu → ~0 ⇒ 2D không bổ trợ |
| CrossGate attention | dồn về 1 trong 2 view 2D (nhiều khả năng `slab_mip`) |
| Branch-drop (giảm xác suất khi bỏ nhánh) | 3D: 0,05 – 0,12; mỗi nhánh 2D: 0,00 – 0,04 |
| Grad-CAM 3D | tập trung dải RNFL / đĩa thị (định tính) |

---

## 5. Cách dùng bảng ước tính này

- Dùng để **chuẩn bị khung bảng cho luận văn** và đối chiếu nhanh khi có kết quả.
- Dùng để đặt **tiêu chí chấp nhận**: nếu số thật **thấp hơn hẳn khoảng hợp lý** (ví dụ AUC < 0,84,
  bal-acc < 0,75, MCC < 0,5) ⇒ nghi ngờ **bug/collapse**, kiểm tra lại thay vì kết luận kiến trúc kém.
- **Không** dùng bảng này làm kết quả. Chỉ bảng mục 6 (số thật) mới được trích dẫn.

---

## 6. BẢNG CHỜ KẾT QUẢ THẬT (điền sau khi train)

> Thay các ô `—` bằng số thật từ `metrics.jsonl` / W&B (project `glaucoma-thesis`) sau run
> `final_raw_s*` và `final_bilateral_s*`. Ghi kèm mean ± std trên 3 seed.

| Metric | Val (mean ± std) | Test (mean ± std) | Test 95 % bootstrap CI |
|---|---:|---:|---:|
| Accuracy | — | — | — |
| Balanced accuracy | — | — | — |
| AUC (ROC) | — | — | — |
| PR-AUC | — | — | — |
| F1 | — | — | — |
| Sensitivity | — | — | — |
| Specificity | — | — | — |
| MCC | — | — | — |
| ECE ↓ (trước scaling) | — | — | — |
| ECE ↓ (sau scaling) | — | — | — |
| Temperature | — | — | — |
| Threshold (val) | — | — | — |
| Gate σ(α) | — | — | — |

**So sánh raw vs bilateral (chờ điền):**

| Dataset | Test AUC | Test Bal.Acc | Test MCC | ECE↓ |
|---|---:|---:|---:|---:|
| raw | — | — | — | — |
| bilateral | — | — | — | — |

---

## 7. Liên kết liên quan

- [`model-architecture.md`](model-architecture.md) — kiến trúc + params model chính.
- [`3d-backbone-sweep.md`](3d-backbone-sweep.md) — kết quả thật nhánh 3D.
- [`2d-feature-extraction-sweep.md`](2d-feature-extraction-sweep.md) — kết quả thật nhánh 2D.
- [`fusion-ablation-sweep.md`](fusion-ablation-sweep.md) — ablation fusion (cảnh báo collapse).
- [`final-4branch-design.md`](final-4branch-design.md) — thiết kế & kỳ vọng mô hình fusion.
