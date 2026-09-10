# Ablation fusion 2D–3D (Harvard-GF) — Nhận xét & bảng cho báo cáo

Nguồn: **W&B** (project `glaucoma-thesis`), nhóm run "fusion 2D–3D", **6 phương pháp, mỗi phương pháp 1 run**.
Trường **không được log** ghi `—` (không suy đoán).

> **Cảnh báo quan trọng (đọc trước khi trích dẫn):** tất cả 6 run có **Balanced Accuracy = 0.5, MCC = 0**
> ⇒ bộ phân loại **dự đoán gần như toàn bộ về một lớp** (collapse). Vì vậy **Accuracy 0.5433, F1 0.7041,
> ECE thấp và AUC xếp hạng không phản ánh năng lực phân biệt thực sự**. Kết quả dưới đây là **sơ bộ**.
> Ngoài ra mỗi phương pháp chỉ **1 seed** → chưa có mean ± std.

---

## 1. Cấu hình thí nghiệm chung

| Thành phần | Cấu hình |
|---|---|
| Số nhánh | **3 nhánh 2D + 1 nhánh 3D** |
| Encoder 2D | ConvNeXt V2 Tiny (`convnextv2_tiny`) |
| Encoder 3D | ConvNeXt3D (`convnext3d`) |
| Kích thước ảnh 2D | 224×224 |
| Kích thước ảnh 3D | 200³ |
| Batch size | 2 |
| Số epoch tối đa | 10 |
| Phép fusion | Concat, Add, Attention, CrossGate, Mamba, FiLM |
| Số lần chạy | 1 run/phương pháp |

---

## 2. Mô tả ngắn các phương pháp fusion

| Phương pháp | Mô tả (theo tên chuẩn) |
|---|---|
| Concat | Nối vector đặc trưng các nhánh trước tầng phân loại. |
| Add | Cộng theo phần tử các đặc trưng đã chiếu về cùng số chiều. |
| Attention | Học trọng số tương tác giữa đặc trưng 2D và 3D (self/cross-attention). |
| CrossGate | Cổng cho đặc trưng nhánh này điều khiển thông tin từ nhánh kia. |
| Mamba | Mô hình hoá chuỗi/token đặc trưng bằng state-space có chọn lọc. |
| FiLM | Điều chỉnh đặc trưng bằng tham số scale/shift sinh từ đặc trưng điều kiện. |

*Mô tả dựa theo tên phương pháp chuẩn; cần đối chiếu phần cài đặt trong source code để mô tả chính xác kiến trúc của luận văn.*

---

## 3. Validation — lựa chọn checkpoint

**Bảng X. So sánh hiệu năng validation của các phương pháp hợp nhất 2D–3D**

| Phương pháp | Epoch đạt Val AUC cao nhất | Best Val AUC ↑ | Val PR-AUC tại epoch đó ↑ | Min Val Loss ↓ | Final Val AUC | Mức giảm sau đỉnh ↓ |
|---|---:|---:|---:|---:|---:|---:|
| **Mamba** | 6 | **0.6056** | **0.6448** | 0.6789 | 0.5066 | 0.0990 |
| FiLM | 4 | 0.5718 | 0.6290 | 0.6788 | 0.4934 | 0.0783 |
| CrossGate | 1 | 0.5690 | 0.6441 | 0.6839 | 0.4658 | 0.1032 |
| Attention | 4 | 0.5664 | 0.6292 | 0.6796 | **0.5411** | **0.0252** |
| Add | 6 | 0.5389 | 0.6090 | **0.6781** | 0.4484 | 0.0906 |
| Concat | 2 | 0.5302 | 0.5967 | 0.6830 | 0.5036 | 0.0267 |

**Nhận xét validation**
- **Mamba** đạt Val AUC cao nhất (0.6056) nhưng **sụt mạnh** ở epoch cuối (−0.0990).
- **Attention** **ổn định nhất**: chỉ giảm 0.0252 sau đỉnh và có Final Val AUC cao nhất (0.5411).
- **CrossGate** đạt đỉnh ngay **epoch 1** rồi suy giảm mạnh → bất ổn/overfit sớm.
- **Min Val Loss** của mọi phương pháp đều ~**0.678–0.684**, sát mức **log(2) ≈ 0.693** (loss khi đoán theo tỉ lệ lớp)
  → dấu hiệu mô hình chưa học được tín hiệu phân biệt.
- **Chọn checkpoint:** dùng **best-epoch theo Val AUC** (Mamba ep6, FiLM ep4, CrossGate ep1, Attention ep4, Add ep6, Concat ep2),
  **không** dùng epoch cuối.

---

## 4. Test

**Bảng Y. Hiệu năng phân loại trên tập kiểm thử**

| Phương pháp | Test AUC ↑ | Test PR-AUC ↑ | Accuracy | F1 | Bal.Acc | MCC | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **CrossGate** | **0.5685** | **0.5981** | 0.5433 | 0.7041 | 0.5000 | 0.0000 | **0.0104** |
| Attention | 0.5205 | 0.5567 | 0.5433 | 0.7041 | 0.5000 | 0.0000 | 0.0323 |
| Concat | 0.5031 | 0.5449 | 0.5433 | 0.7041 | 0.5000 | 0.0000 | 0.0394 |
| FiLM | 0.5028 | 0.5452 | 0.5433 | 0.7041 | 0.5000 | 0.0000 | 0.0622 |
| Add | 0.4914 | 0.5376 | 0.5433 | 0.7041 | 0.5000 | 0.0000 | 0.1481 |
| Mamba | 0.4715 | 0.5299 | 0.5433 | 0.7041 | 0.5000 | 0.0000 | 0.0402 |

Precision / Recall / Specificity trên test **không được log** → `—`.

**Xếp hạng theo Test AUC:** CrossGate 0.5685 → Attention 0.5205 → Concat 0.5031 → FiLM 0.5028 → Add 0.4914 → Mamba 0.4715.

**Nhận xét test**
- **CrossGate** đứng đầu test (AUC 0.5685, PR-AUC 0.5981) nhưng vẫn **rất thấp** và **không tương ứng** với việc bị chọn làm best ở epoch 1.
- **Mamba** tốt nhất trên validation nhưng **xếp cuối trên test** (AUC 0.4715 < 0.5) → **không nên kết luận từ validation AUC**.
- **Toàn bộ** có Acc = 0.5433, F1 = 0.7041, **Bal.Acc = 0.5, MCC = 0** → **collapse một lớp**; các giá trị F1/ECE/Accuracy
  **không dùng để so sánh năng lực**.

---

## 5. Thời gian & epoch kết thúc

| Phương pháp | Epoch cuối | Thời gian chạy |
|---|---:|---:|
| Attention | 7 | 2.01 giờ |
| Add | 7 | 2.03 giờ |
| CrossGate | 7 | 2.04 giờ |
| Mamba | 7 | 2.04 giờ |
| FiLM | 9 | 2.64 giờ |
| Concat | 10 | 2.88 giờ |

Các run dừng trước epoch 10 nhiều khả năng do **early stopping**.

---

## 6. Phân tích nguyên nhân collapse & hướng khắc phục

1. **Nhánh 2D yếu/suy biến:** cả 6 run dùng `convnextv2_tiny`, và trong **sweep 2D độc lập** (xem
   [`2d-feature-extraction-sweep.md`](2d-feature-extraction-sweep.md)) backbone này **suy biến (MCC = 0, Recall = 0)**.
   Đưa một nhánh 2D suy biến vào fusion ⇒ fusion **không thể tốt hơn** nhánh tốt nhất.
2. **Mất cân bằng/độ khó:** loss ~log(2) và MCC = 0 ⇒ mô hình chưa tách được hai lớp (LR/khởi tạo/early-stop/chuẩn hoá).
3. **Overfit sớm:** nhiều fusion đạt đỉnh ở epoch 1–2 rồi giảm (CrossGate ep1, Concat ep2).

**Khắc phục đề xuất (theo thứ tự ưu tiên):**
1. **Thay nhánh 2D** bằng backbone mạnh đã kiểm chứng: **MaxViT** (AUROC test 0.7638) hoặc **Tiny2D-SlabMIP**
   (AUROC 0.7242, ECE 0.0549), và dùng **view Slab MIP** thay vì Slab AIP.
2. **Kiểm tra collapse**: log balanced accuracy/MCC mỗi epoch; thêm class weighting hoặc focal loss; kiểm tra LR/khởi tạo.
3. **Chọn checkpoint theo best Val AUC**, không dùng epoch cuối; cân nhắc **warmup + cosine** dài hơn 10 epoch.
4. **Chạy ≥3 seed/phương pháp** và báo cáo **mean ± std**; giữ nguyên split để so sánh công bằng.
5. (Tuỳ chọn) ghép thêm nhánh **3D mạnh** (ConvNeXt3D @96³ / **3DINO**) — xem [`multiview-model.md`](multiview-model.md).

---

## 7. Đoạn mô tả dùng trực tiếp trong báo cáo

> Trong sáu cơ chế hợp nhất được khảo sát, **Mamba** đạt AUC cao nhất trên tập validation (0.6056) nhưng
> **không duy trì được** trên tập kiểm thử (Test AUC 0.4715). **CrossGate** cho kết quả kiểm thử tốt nhất
> (Test AUC 0.5685, PR-AUC 0.5981), còn **Attention** có đường validation **ổn định nhất** (chỉ giảm 0.0252 sau đỉnh).
> Tuy nhiên, **tất cả** phương pháp đều có **balanced accuracy = 0.5 và MCC = 0**, cho thấy bộ phân loại
> **dự đoán gần như toàn bộ về một lớp**. Do đó, các kết quả hiện tại **chưa chứng minh được lợi ích của hợp nhất 2D–3D**
> và cần được coi là **kết quả sơ bộ**; cần sửa hiện tượng collapse và chạy nhiều seed trước khi kết luận.

---

## 8. Lưu ý bắt buộc khi báo cáo

- **1 run/phương pháp** → chưa có **mean ± standard deviation**.
- **Không** kết luận "Mamba tốt nhất" chỉ từ validation AUC.
- Nếu buộc chọn theo test hiện tại: **CrossGate đứng đầu** (Test AUC 0.5685) — nhưng vẫn **thấp**.
- **Accuracy 0.5433 / F1 0.7041 dễ gây hiểu nhầm**: Bal.Acc = 0.5, MCC = 0, val recall = 1, val specificity = 0 → dự đoán một lớp.
- Trường **thiếu**: precision/recall/specificity (val & test), params, FLOPs, tốc độ suy luận → `—`.
- Trước khi công bố: **chạy ≥3 seed**, **sửa collapse**, rồi trình bày **mean ± std**.

---

## 9. Liên kết liên quan

- [`2d-feature-extraction-sweep.md`](2d-feature-extraction-sweep.md) — sweep 2D độc lập (chứng minh `convnextv2_tiny` suy biến).
- [`multiview-model.md`](multiview-model.md) — kiến trúc đa luồng 2D+3D & các phép fusion.
- [`datasets.md`](datasets.md) — thống kê Harvard-GF (splits, nhãn).
