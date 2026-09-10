# Sweep trích xuất đặc trưng 2D (Harvard-GF) — Nhận xét & bảng cho báo cáo

Nguồn: **W&B** (project `glaucoma-thesis`), nhóm run "2D feature extraction", **6 cấu hình, mỗi cấu hình 1 run**.
Dữ liệu: Harvard-GF, các view en-face 2D; đầu vào 224×224; batch size 2.
Trường **không được log** ghi `—` (không suy đoán).

> Lưu ý phương pháp: (1) mỗi cấu hình chỉ 1 seed → **chưa có trung bình ± độ lệch chuẩn**;
> (2) giá trị validation dưới đây là **epoch cuối được ghi nhận**, *không phải* giá trị tốt nhất;
> (3) **precision / recall / specificity trên test không được log** → để `—`;
> (4) kiểu view của các backbone timm **suy ra từ tên run** (chưa lưu thành trường config riêng);
> (5) số tham số / FLOPs / tốc độ suy luận **chưa được lưu**.

---

## 1. Chuẩn hóa tên backbone & kiểu ảnh

| Ký hiệu trong báo cáo | Tên run (W&B) | Backbone (chuẩn) | Kiểu ảnh | Pretrained |
|---|---|---|---|---|
| Tiny2D-AIP | `single2d-tiny2d-0-aip_full` | Tiny2D (CNN from-scratch) | AIP toàn ảnh (`aip_full`) | Không |
| Tiny2D-SlabAIP | `single2d-tiny2d-1-slab_aip` | Tiny2D (CNN from-scratch) | Slab AIP (`slab_aip`) | Không |
| Tiny2D-SlabMIP | `single2d-tiny2d-2-slab_mip` | Tiny2D (CNN from-scratch) | Slab MIP (`slab_mip`) | Không |
| ConvNeXtV2 | `single2d-convnextv2_tiny-slabaip` | ConvNeXtV2-Tiny | Slab AIP (`slab_aip`) | ImageNet |
| DeiT3 | `single2d-deit3_small_patch16_224-slabaip` | DeiT3-Small (patch16, 224) | Slab AIP (`slab_aip`) | ImageNet |
| MaxViT | `single2d-maxvit_tiny_rw_224-slabaip` | MaxViT-Tiny (RW, 224) | Slab AIP (`slab_aip`) | ImageNet |

---

## 2. Cấu hình huấn luyện

| Ký hiệu | Bộ mã hóa 2D | Biểu diễn ảnh | Kích thước | Batch | Epoch hoàn thành | Thời gian |
|---|---|---|---:|---:|---:|---:|
| Tiny2D-AIP | Tiny2D | AIP toàn ảnh | 224×224 | 2 | 15 | 4.45 phút |
| Tiny2D-SlabAIP | Tiny2D | Slab AIP | 224×224 | 2 | 15 | 4.48 phút |
| **Tiny2D-SlabMIP** | Tiny2D | Slab MIP | 224×224 | 2 | 14 | **4.12 phút** |
| ConvNeXtV2 | ConvNeXtV2-Tiny | Slab AIP | 224×224 | 2 | 8 | 8.55 phút |
| DeiT3 | DeiT3-Small Patch16 | Slab AIP | 224×224 | 2 | 14 | 12.48 phút |
| MaxViT | MaxViT-Tiny | Slab AIP | 224×224 | 2 | 15 | 37.95 phút |

---

## 3. Validation (epoch cuối được ghi nhận)

| Mô hình | Loss ↓ | Acc ↑ | AUROC ↑ | PR-AUC ↑ | Bal.Acc ↑ | Prec ↑ | Recall ↑ | Spec ↑ | F1 ↑ | MCC ↑ | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tiny2D-AIP | 0.6257 | 0.6433 | 0.7002 | 0.7706 | 0.6519 | 0.7413 | 0.6023 | **0.7016** | 0.6646 | 0.2996 | 0.0873 |
| Tiny2D-SlabAIP | 0.6493 | 0.6600 | 0.6307 | 0.6849 | 0.6280 | 0.6745 | **0.8125** | 0.4435 | 0.7371 | 0.2769 | **0.0528** |
| **Tiny2D-SlabMIP** | **0.5941** | 0.6733 | **0.7323** | **0.7886** | 0.6596 | 0.7143 | 0.7386 | 0.5806 | 0.7263 | 0.3219 | 0.0872 |
| ConvNeXtV2 | 0.7122 | 0.4133 | 0.5023 | 0.5895 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1305 |
| DeiT3 | 0.6474 | 0.6233 | 0.6572 | 0.7231 | 0.6051 | 0.6684 | 0.7102 | 0.5000 | 0.6887 | 0.2136 | 0.0737 |
| **MaxViT** | 2.3924 | **0.7033** | 0.7137 | 0.7265 | **0.6924** | **0.7430** | 0.7557 | 0.6290 | **0.7493** | **0.3862** | 0.2869 |

**Nhận xét validation**
- **AUROC cao nhất:** Tiny2D-SlabMIP (0.7323); kế đó MaxViT (0.7137), Tiny2D-AIP (0.7002).
- **MaxViT** dẫn đầu Acc/Bal.Acc/F1/MCC nhưng **loss cuối rất cao (2.3924)** và **ECE 0.2869** → dấu hiệu **quá khớp + quá tự tin**; theo ghi nhận AUROC đỉnh ~0.7898 quanh epoch 8 rồi giảm.
- **ConvNeXtV2 suy biến:** Acc 0.4133, Recall 0, Spec 1.0, F1 0, MCC 0 → gần như chỉ dự đoán một lớp; các chỉ số "đẹp" (Spec=1, ECE thấp) **không** phản ánh năng lực.
- **Tiny2D-SlabAIP** thiên về dương tính (Recall 0.8125, Spec 0.4435) — phù hợp nếu ưu tiên bắt bệnh, nhưng nhiều dương tính giả.
- **Calibration tốt nhất (val):** Tiny2D-SlabAIP 0.0528; DeiT3 0.0737.

---

## 4. Test

| Mô hình | Acc ↑ | AUROC ↑ | PR-AUC ↑ | Bal.Acc ↑ | Prec | Recall | Spec | F1 ↑ | MCC ↑ | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tiny2D-AIP | 0.5911 | 0.6605 | 0.6949 | 0.5614 | — | — | — | 0.7061 | 0.1704 | 0.0884 |
| Tiny2D-SlabAIP | 0.6478 | 0.6526 | 0.6563 | 0.6349 | — | — | — | 0.7073 | 0.2838 | 0.0550 |
| **Tiny2D-SlabMIP** | 0.6656 | 0.7242 | 0.7509 | 0.6538 | — | — | — | **0.7195** | 0.3210 | **0.0549**² |
| ConvNeXtV2 | 0.5433 | 0.5152 | 0.5547 | 0.5000 | — | — | — | 0.7041³ | 0.0000 | 0.0021³ |
| DeiT3 | 0.6567 | 0.6988 | 0.7205 | 0.6503 | — | — | — | 0.6962 | 0.3040 | 0.0568 |
| **MaxViT** | **0.6889** | **0.7638** | **0.7925** | **0.6889** | — | — | — | 0.7065 | **0.3765** | 0.1265 |

² ECE tốt nhất trong số các mô hình **không suy biến**.
³ ConvNeXtV2 có Bal.Acc = 0.5 và MCC = 0 ⇒ suy biến một lớp; F1/ECE thấp **không** nên diễn giải là tốt.
Precision/Recall/Specificity trên test **không được log** nên để `—`.

**Nhận xét test**
- **MaxViT mạnh nhất về phân biệt:** Acc 0.6889, AUROC 0.7638, PR-AUC 0.7925, MCC 0.3765 — nhưng **ECE 0.1265** (quá tự tin) và chi phí cao.
- **Tiny2D-SlabMIP cân bằng nhất:** AUROC 0.7242, F1 **0.7195 (cao nhất)**, ECE **0.0549**, chỉ 4.12 phút.
- **DeiT3** bám sát (AUROC 0.6988, MCC 0.3040, ECE 0.0568).
- **Tiny2D-AIP yếu nhất** trong nhóm không suy biến (AUROC 0.6605, MCC 0.1704).

---

## 5. Ablation kiểu ảnh (cùng backbone Tiny2D)

| View | Test AUROC | Test MCC | Val Recall | Val Spec | Nhận xét |
|---|---:|---:|---:|---:|---|
| AIP toàn ảnh | 0.6605 | 0.1704 | 0.6023 | 0.7016 | Nền chung; kém phân biệt nhất |
| Slab AIP | 0.6526 | 0.2838 | **0.8125** | 0.4435 | Nhạy dương tính, nhiều FP |
| **Slab MIP** | **0.7242** | **0.3210** | 0.7386 | 0.5806 | **Tốt nhất** — giữ chi tiết sợi/mạch trong dải RNFL |

→ Với Tiny2D, **Slab MIP > AIP toàn ảnh ≈ Slab AIP** về AUROC/MCC. Điều này ủng hộ giả thuyết dải RNFL (đặc biệt MIP) mang tín hiệu glaucoma tốt hơn.

---

## 6. Calibration & chi phí

| Mô hình | Val ECE ↓ | Test ECE ↓ | Thời gian (phút) | AUROC test / phút |
|---|---:|---:|---:|---:|
| Tiny2D-AIP | 0.0873 | 0.0884 | 4.45 | 0.148 |
| Tiny2D-SlabAIP | **0.0528** | 0.0550 | 4.48 | 0.146 |
| **Tiny2D-SlabMIP** | 0.0872 | **0.0549** | **4.12** | **0.176** |
| ConvNeXtV2 | 0.1305 | 0.0021³ | 8.55 | 0.060 (suy biến) |
| DeiT3 | 0.0737 | 0.0568 | 12.48 | 0.056 |
| MaxViT | 0.2869 | 0.1265 | 37.95 | 0.020 |

**Nhận xét:** Tiny2D-SlabMIP cho **hiệu năng/chi phí tốt nhất**; MaxViT đắt (~9× Tiny2D) và **calibration kém**; ConvNeXtV2 bỏ qua do suy biến.

---

## 7. Tổng hợp & khuyến nghị kết hợp nguồn

**Xếp hạng theo mục tiêu:**

| Mục tiêu | Lựa chọn | Lý do |
|---|---|---|
| Phân biệt tối đa (AUROC/MCC test) | **MaxViT** | AUROC 0.7638, MCC 0.3765 — cần **best checkpoint + hiệu chỉnh** |
| Cân bằng hiệu năng/ổn định/calibration/chi phí | **Tiny2D-SlabMIP** | AUROC 0.7242, F1 0.7195, ECE 0.0549, 4.12 phút |
| Nhạy bắt dương tính (screening) | Tiny2D-SlabAIP | Val Recall 0.8125 (đánh đổi Spec thấp) |
| Loại bỏ | ConvNeXtV2 (run hiện tại) | Suy biến một lớp (MCC=0) |

**Cách "củng cố" bằng kết hợp nguồn (định hướng cho mô hình cuối):**
1. **Ensemble xác suất** (trung bình có trọng số hoặc temperature scaling): `MaxViT + DeiT3 + Tiny2D-SlabMIP`.
   MaxViT cho discrimination, Tiny2D-SlabMIP/DeiT3 kéo **calibration** xuống (ECE ~0.05).
2. **Đưa các nhánh 2D mạnh vào fusion đa luồng (Tier C)** cùng nhánh 3D:
   thay 2D branch mặc định (`convnextv2_tiny` đang suy biến) bằng **MaxViT** hoặc **Tiny2D-SlabMIP**,
   ghép với 3D branch (`convnext3d` @ 96³) và/hoặc **3DINO** (Tier A) qua các phép `attn`/`crossgate`/`film`.
3. **View tốt nhất cho 2D branch:** dùng **Slab MIP** (không phải Slab AIP) làm view chính.
4. **MaxViT bắt buộc dùng best checkpoint** (val AUROC đỉnh ~0.7898 quanh epoch 8), **không** lấy epoch cuối; kèm **calibration** (temperature/isotonic) trước khi báo cáo.

---

## 8. Hạn chế & việc cần làm

- **1 seed/cấu hình** → chạy thêm ≥3 seed để có mean ± std trước khi kết luận.
- **Validation dùng epoch cuối**, chưa lưu best-epoch theo từng run → cần log `best_ep` + metric tại best.
- **Thiếu trên test:** Precision, Recall, Specificity → `—`; nếu cần cho báo cáo, bổ sung log các trường này.
- **Thiếu:** số tham số, FLOPs, tốc độ suy luận, số epoch tới best.
- **Kiểu view** của backbone timm mới suy từ tên run → nên lưu thành trường config (`view`) khi log.
- **ConvNeXtV2** cần kiểm tra lại (LR, nhãn, khởi tạo) trước khi dùng.

---

## 9. Đoạn mô tả dùng trực tiếp trong báo cáo

> Trong sáu cấu hình trích xuất đặc trưng 2D, **MaxViT-Tiny** đạt kết quả phân biệt cao nhất trên tập kiểm thử
> (accuracy 0.6889, AUROC 0.7638, PR-AUC 0.7925, balanced accuracy 0.6889, MCC 0.3765) nhưng có dấu hiệu
> **quá khớp** (validation loss tăng từ 0.5798 lên 2.3924) và **kém hiệu chỉnh** (test ECE 0.1265). Ngược lại,
> **Tiny2D với biểu diễn slab-MIP** cho kết quả **ổn định và cân bằng nhất** (AUROC 0.7242, F1 0.7195 — cao nhất,
> ECE 0.0549) và chỉ tốn **4.12 phút** huấn luyện. Như vậy, MaxViT-Tiny thể hiện tiềm năng phân biệt cao nhất,
> còn Tiny2D-SlabMIP là lựa chọn cân bằng giữa hiệu năng, độ ổn định, khả năng hiệu chỉnh và chi phí tính toán.
> Về mặt biểu diễn ảnh, **slab-MIP vượt trội** so với AIP toàn ảnh và slab-AIP trong cùng backbone Tiny2D.
> Để tận dụng cả hai, khuyến nghị **ensemble MaxViT + DeiT3 + Tiny2D-SlabMIP** (có temperature scaling) hoặc
> đưa chúng làm **nhánh 2D trong mô hình fusion đa luồng** cùng nhánh 3D.
