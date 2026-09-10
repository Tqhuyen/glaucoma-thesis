# Sweep backbone 3D (Harvard-GF) — Nhận xét & bảng cho báo cáo

Nguồn: **W&B** (project `glaucoma-thesis`), nhóm run "3D single-branch", **17 run hiển thị**.
Đã **chuẩn hóa tên metric** bị log dưới nhiều dạng (`test/acc` ↔ `test.acc`, …).
Trường **không được log** ghi `—` (không suy đoán). Mỗi kiến trúc **1 run** → chưa có mean ± std.

> **Nguyên tắc chọn:** không chọn theo accuracy nếu có dấu hiệu **dự đoán lệch một lớp**
> (balanced accuracy ≈ 0.5, MCC ≈ 0). Các model như vậy bị loại dù F1 trông cao.

---

## 1. Kết luận chính

**Mô hình tốt nhất tổng thể: `single3d-resxt3d` (ResNeXt3D).**

`denoise-enc-32-d5` có test accuracy cao nhất (77,44%) nhưng:
- chỉ hơn ResNeXt3D **0,33 điểm phần trăm**;
- **không log** AUC / PR-AUC / balanced accuracy / MCC;
- **khác quy trình** (200³ + tiền xử lý NLM, 20 epoch) so với `single3d` (96³, 15 epoch) → **confound**.

Trong nhóm được đánh giá **đầy đủ và cùng pipeline**, **ResNeXt3D đứng đầu đồng thời về AUC, PR-AUC, F1 và balanced accuracy** ⇒ lựa chọn đáng tin cậy nhất.

---

## 2. Mô hình chiến thắng — `single3d-resxt3d`

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 77,33% | **77,11%** |
| AUC | 84,72% | **85,38%** |
| PR-AUC | — | **87,98%** |
| Balanced accuracy | — | **77,19%** |
| F1 | 79,76% | **78,36%** |
| MCC | — | **0,542** |
| ECE ↓ | — | 4,82% |
| Validation loss | 0,489 | — |

**Điểm mạnh:** test AUC/PR-AUC/F1/balanced-acc cao nhất; MCC cao nhất (ngang VNet); khoảng cách val–test rất nhỏ
(accuracy lệch **0,22 điểm phần trăm**) ⇒ tổng quát hóa ổn định; **không collapse**.

---

## 3. Bảng xếp hạng đầy đủ (17 run)

| Hạng (acc) | Model | Test acc | Test AUC | PR-AUC | Test F1 | MCC | Ghi chú |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `denoise-enc-32-d5` | **77,44%** | — | — | 77,72% | — | 200³ + NLM, 20 ep; thiếu metric |
| 2 | **`single3d-resxt3d`** | 77,11% | **85,38%** | **87,98%** | **78,36%** | **0,542** | **Tốt nhất tổng thể** |
| 3 | `single3d-vnet` | 76,89% | 83,91% | 87,14% | 77,63% | 0,542 | ECE tốt nhất (3,46%) |
| 4 | `single3d-segresnet` | 76,67% | 83,45% | 86,90% | 77,37% | 0,538 | Val mạnh (78,00%) |
| 5 | `denoise-enc-32` | 75,33% | — | — | 76,68% | — | 200³ + NLM |
| 6 | `3dino-112` | 75,22% | — | — | — | — | 3DINO @112³ (thiếu metric) |
| 7 | `single3d-cnn3d` | 74,44% | 83,50% | 86,67% | 77,41% | 0,483 | baseline CNN |
| 8 | `single3d-mednet10` | 74,33% | 82,10% | 86,14% | 75,35% | 0,489 | MedicalNet nhẹ |
| 9 | `denoise-enc-24` | 74,33% | — | — | 75,91% | — | 200³ + NLM |
| 10 | `single3d-3dino` | 74,11% | 82,32% | 85,65% | 75,50% | 0,482 | 3DINO frozen @96³ |
| 11 | `single3d-mednet50` | 74,11% | 82,21% | 85,15% | 76,95% | 0,476 | sâu hơn, không lợi hơn |
| 12 | `single3d-dynunet16` | 72,89% | 79,19% | 83,38% | 75,05% | 0,454 | MONAI DynUNet |
| 13 | `single3d-unetr` | 69,11% | 76,19% | 81,10% | 72,20% | 0,375 | ViT encoder |
| 14 | `single3d-convnext3d` | 54,33% | 53,39% | 56,44% | 70,41% | **0** | **collapse** |
| 15 | `single3d-vit3d` | 54,33% | 52,70% | 56,50% | 70,41% | **0** | **collapse** |
| 16 | `single3d-swinunetr` | 54,33% | 48,18% | 52,65% | 70,41% | **0** | **collapse** |

`denoise-gpu-2404` là **benchmark khử nhiễu**, không phải run phân loại → không xếp hạng.

---

## 4. Ứng viên mạnh nhất

### 1) ResNeXt3D — tốt nhất tổng thể
Đứng đầu gần như mọi metric có ý nghĩa với dữ liệu mất cân bằng ⇒ **3D encoder chính** cho fusion.

### 2) VNet — á quân rất sát
Acc 76,89%; balanced acc 77,18%; MCC 0,5417 (≈ ResNeXt3D); **ECE 3,46%** (tốt hơn ResNeXt3D).
Nhược: val acc chỉ 74% (thấp hơn test 2,89 điểm) → cần kiểm tra nhiều seed.

### 3) SegResNet — ổn định val–test
Val acc **78,00%**, val F1 **81,14%** (cao trong nhóm `single3d`); test acc 76,67%, MCC 0,538.
AUC/PR-AUC vẫn thấp hơn ResNeXt3D.

### 4) `denoise-enc-32-d5` — accuracy cao nhất nhưng thiếu metric
Test acc **77,44%**, F1 77,72%, precision 83,89%, recall 72,39%, best val acc 78,67%.
Giữ làm **baseline**, nhưng chưa thể kết luận tốt hơn ResNeXt3D (chênh 0,33 điểm; thiếu AUC/MCC/bal-acc; khác 200³+NLM).

---

## 5. Model KHÔNG nên tiếp tục (collapse)

**ConvNeXt3D, ViT3D, SwinUNETR:** test acc 54,33%, **balanced acc 50%, MCC 0** ⇒ dự đoán toàn bộ về một lớp.
F1 70,41% **gây hiểu nhầm**. Nguyên nhân khả dĩ: LR/schedule chưa phù hợp, batch nhỏ, backbone quá lớn so với dữ liệu,
train-from-scratch thay vì freeze/unfreeze, class weighting/threshold chưa đúng.
⇒ **Không đưa vào fusion** trước khi sửa collapse.

---

## 6. Đánh giá theo nhóm kiến trúc

- **CNN 3D:** ResNeXt3D, VNet, SegResNet mạnh nhất — acc 76,7–77,1%, AUC 83,4–85,4%, MCC ≈ 0,54.
- **Transformer/pretrained:** `single3d-3dino` AUC 82,32% nhưng acc 74,11%; `3dino-112` acc 75,22% (thiếu metric);
  UNETR 69,11%; ViT3D/SwinUNETR collapse ⇒ **Transformer 3D chưa vượt CNN 3D** trên bộ này.
- **MedicalNet:** MedNet10 ≈ MedNet50 về AUC (~82%); MedNet10 val tốt hơn, PR-AUC cao hơn ⇒ nếu cần baseline nhẹ, **MedNet10 hợp lý hơn**.
- **Denoising encoder:** `enc-32-d5` tốt nhất nhóm, cần chạy lại cùng protocol để tách ảnh hưởng encoder vs NLM vs 200³.

---

## 7. Hạn chế

1. Mỗi kiến trúc **1 run** → chưa có mean ± std.
2. **Hai pipeline khác nhau** (`single3d` 96³ vs denoise 200³+NLM).
3. **Logging chưa đồng nhất** (`test.acc` ↔ `test/acc`).
4. Nhiều run thiếu AUC/PR-AUC/MCC/specificity/calibration.
5. Không có khoảng tin cậy ⇒ chênh lệch nhỏ đầu bảng **chưa chắc có ý nghĩa thống kê**.
6. Kết quả chỉ đúng **trong các run hiện tại**.

---

## 8. Khuyến nghị

- **Encoder 3D chính:** `single3d-resxt3d`.
- **Đối chứng mạnh:** `single3d-vnet` (calibration tốt), `single3d-segresnet` (val mạnh).
- **Baseline pipeline khác:** `denoise-enc-32-d5`.
- **Loại:** ConvNeXt3D / ViT3D / SwinUNETR ở cấu hình hiện tại.

**Thí nghiệm tiếp theo (4 model trên):** cùng split, cùng resolution, cùng preprocessing, cùng epoch/early-stop,
**3–5 seed**, log thống nhất (acc, balanced acc, AUC, PR-AUC, F1, MCC, sensitivity, specificity, ECE),
**chọn checkpoint theo val AUC/PR-AUC** (không theo test acc), giữ test độc lập.

---

## 9. Đoạn mô tả dùng trực tiếp trong báo cáo

> Trong 17 run khảo sát backbone 3D, `denoise-enc-32-d5` đạt accuracy kiểm thử cao nhất (77,44%) nhưng thiếu
> AUC/PR-AUC/MCC/balanced-accuracy và chạy ở protocol khác (200³ + NLM), nên chưa đủ cơ sở kết luận.
> Trong nhóm được đánh giá đầy đủ và cùng pipeline, **ResNeXt3D** dẫn đầu đồng thời về AUC (85,38%),
> PR-AUC (87,98%), F1 (78,36%) và balanced accuracy (77,19%), với khoảng cách validation–test chỉ 0,22 điểm phần trăm.
> VNet và SegResNet bám sát; ConvNeXt3D, ViT3D và SwinUNETR bị **collapse một lớp** (balanced accuracy 50%, MCC 0).
> Do đó **ResNeXt3D** được chọn làm nhánh đặc trưng 3D; cần xác nhận bằng nhiều seed và protocol thống nhất.
