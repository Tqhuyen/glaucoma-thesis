# Nhận xét X-AI — `final_bilateral_s42` (3D + 2×2D CrossGate)

## 1. Nguồn dữ liệu và danh tính mẫu

| Hạng mục | Giá trị |
|---|---|
| Run X-AI | `final_bilateral_s42_xai` (W&B id `iow9ispv`) |
| Run overlay | `final_bilateral_s42_xai_overlay_reconstruction` (W&B id `l4yahq6c`) |
| Run huấn luyện nguồn | `rkdhqgvh` (bilateral, seed 42) |
| Mẫu | **Validation row 0**, `label = 1` (glaucoma positive) |
| Danh tính dataset | `Validation_idx0_data_2101_label1` → scan gốc **`data_2101`** |
| Dataset revision | `939a38876b7b9313162842ef2d44b7edc2b57020` |
| Depth axis (raw & bilateral) | 1; peak slice 23; slab MIP dải 7–39 |
| Nguồn heatmap | **Phục hồi từ PNG đã log**, không phải tensor saliency gốc |

Heatmap hiển thị là ảnh CAM-only (không có nền OCT) trong run X-AI; vị trí overlay chỉ **xấp xỉ** vì hàm phục hồi từ PNG. `reconstruction_exact_preprocess = True` nghĩa là ảnh đầu vào raw/bilateral được tái tạo đúng, còn bản thân saliency thì không.

Ảnh thật đã tải về `figures/xai/final_bilateral_s42/` (11 PNG + `manifest.json`). Drive không mount trên máy local nên lấy trực tiếp từ W&B.

## 2. Bối cảnh định lượng

- Test: AUC-ROC 0.8472, AUC-PR 0.8818, balanced accuracy 0.7731, sensitivity 0.6728, specificity 0.8735, ECE 0.0674.
- CrossGate attention 2D-0/2D-1 = **0.595/0.405**; gate 0.616.
- Branch drop: bỏ 2D-0 giảm xác suất **0.105**, bỏ 2D-1 ~**0.000**, bỏ toàn bộ 2D **0.0497**.
- Linear probe AUC: 3D **0.8381**, 2D-1 **0.8381**, 2D-0 **0.5188**, fused **0.8640**, concat 0.8372.

## 3. Quan sát trên ảnh

**Grad-CAM 3D.** Vùng nóng gồm hai cụm gọn, chạy dọc theo dải lớp võng mạc (cung trên-phải và cung dưới-trái). Đây là kết quả hợp lý nhất về giải phẫu: mô hình tập trung vào dải lớp trong — nơi liên quan lớp sợi thần kinh/ganglion. Tuy nhiên cần xác nhận bằng toạ độ và contour định lượng.

**Grad-CAM 2D-0 (slab MIP).** Vùng nóng là một khối lớn, khuếch tán, bám theo vùng sáng rộng của slab chứ không khu trú vào một cấu trúc nhỏ. Đi kèm với probe AUC 0.5188 (gần mức ngẫu nhiên), nhánh 2D-0 khó được xem là nguồn thông tin chính.

**Grad-CAM 2D-1 (AIP full).** Vùng nóng lan rộng, có một "lỗ" tối ở giữa đúng vùng đĩa thị/đầu thần kinh thị giác. Nghĩa là mô hình **không** dựa vào tâm đĩa thị ở view này mà phân bố saliency quanh vùng võng mạc. Đây là giả thuyết cần kiểm chứng, không phải kết luận giải phẫu, vì probe 2D-1 mạnh (0.8381) nhưng CAM lại khuếch tán.

**Integrated Gradients.** Vùng nóng là **một đường thẳng đứng ở rìa trái** — dấu hiệu artefact biên (padding/biên ảnh), không phải cấu trúc. IG trong lần này không dùng được để diễn giải.

**Occlusion 4×4×4.** Lưới quá thô (4×4×4), vùng sáng nằm ở góc dưới-trái; độ phân giải quá thấp để nói về cấu trúc. Giá trị diễn giải thấp.

## 4. Kết luận nhận xét

1. **Mẫu là xác định và duy nhất:** `Validation row 0` = `data_2101`, nhãn dương tính. Mọi heatmap trong report/run X-AI đều thuộc cùng một mẫu; không có biến thiên theo mẫu.
2. **Grad-CAM 3D là bằng chứng thuyết phục nhất**, bám dải lớp võng mạc. 2D-1 khuếch tán quanh đĩa thị; 2D-0 khuếch tán theo slab và probe yếu. IG bị artefact biên, occlusion quá thô.
3. **Attention ≠ causal importance:** attention 2D-0 cao hơn 2D-1, nhưng branch-drop cho thấy bỏ 2D-1 gần như không đổi xác suất, còn bỏ 2D-0 giảm 0.105. Cần dùng branch-drop/ablation, không chỉ attention, khi kết luận nhánh nào quyết định.
4. **Chưa thể kết luận lâm sàng:** chỉ một mẫu dương tính, không có TP/TN/FP/FN, không có kiểm định định lượng cho X-AI. Vùng nóng có thể là cấu trúc, cũng có thể là biên/feature-map/artefact upsampling.

## 5. Đề xuất bước tiếp theo

- Dùng helper mới `scripts/xai_identity.py` để lưu **tensor saliency gốc + index/stem/label** vào tên file và `*_xai_meta.json`, tránh phải phục hồi từ PNG.
- Mở rộng cohort tối thiểu 12–20 ca gồm TP/TN/FP/FN, cả hai lớp; với 3D lấy thêm lát cắt có saliency cực đại; với 2D dùng đúng hai view mô hình nhận.
- Thêm kiểm định định lượng: deletion/insertion curve, randomization test, stability khi nhiễu/flip, tỉ lệ saliency ở viền ảnh, mức đồng thuận Grad-CAM ↔ IG ↔ occlusion.
- Chuẩn hoá color scale và overlay alpha giữa các ca; vẽ contour top 10–20% để so sánh.
- Kiểm tra riêng nhánh 2D-1: saliency có thực sự tránh đĩa thị không, hay chỉ là hiệu ứng của phép chiếu AIP.
