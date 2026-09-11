# Kết quả mô hình ResNeXt3D + 2×MaxViT + CrossGate trên dữ liệu raw

## 1. Thông tin kết quả

| Thuộc tính | Giá trị |
|---|---|
| Mô hình | Một nhánh ResNeXt3D, hai nhánh MaxViT-Tiny, hợp nhất CrossGate |
| Backbone 2D | `maxvit_tiny_rw_224` |
| Ảnh chiếu | Slab MIP và Full AIP |
| Dữ liệu | Harvard-GF, raw; không gán kết quả này cho Bilateral |
| Run tag | `raw_s42` |
| Seed | 42 |
| Giai đoạn | Warm-start từ trọng số đã cứu, theo cấu hình run raw đã sử dụng |
| Độ phân giải đầu vào theo cấu hình | Thể tích 200 × 200 × 200; ảnh chiếu 224 × 224 |
| Số mẫu validation | 300 |
| Số mẫu test | 900 |
| Ngưỡng phân loại | 0.430000 |
| Temperature | 1.1326569318771362 |
| Số run được báo cáo | 1 |
| Nguồn số liệu | Dictionary kết quả mới nhất do người dùng cung cấp trong phiên trao đổi |

Ngưỡng gốc được xuất dưới dạng `0.42999999999999994`, tương đương 0.43 khi trình bày. Cấu hình giai đoạn raw cho phép tối đa 10 epoch bổ sung, nhưng dictionary kết quả không có số epoch thực chạy hoặc epoch của best checkpoint. Vì vậy không khẳng định run đã chạy đủ 10 epoch. Bảng này không chứa kết quả fine-tune Bilateral 5 epoch được cấu hình sau đó.

## 2. Bảng kết quả đầy đủ

Các chỉ số tỷ lệ được trình bày trên thang 0–1 và làm tròn sáu chữ số thập phân; số đếm giữ nguyên. F1 là F1 của lớp glaucoma dương, không phải macro-F1. Cột `auc_pr` trong cài đặt tương ứng Average Precision (AP).

| Chỉ số | Validation | Test |
|---|---:|---:|
| Accuracy | 0.790000 | 0.766667 |
| Balanced accuracy | 0.785282 | 0.769170 |
| Precision, lớp glaucoma | 0.826590 | 0.813483 |
| Recall, lớp glaucoma | 0.812500 | 0.740286 |
| Sensitivity | 0.812500 | 0.740286 |
| Specificity | 0.758065 | 0.798054 |
| F1-score, lớp glaucoma | 0.819484 | 0.775161 |
| Matthews correlation coefficient (MCC) | 0.568653 | 0.536347 |
| ROC-AUC | 0.852662 | 0.847626 |
| Average Precision (`auc_pr`) | 0.899907 | 0.878161 |
| Expected calibration error (ECE) | 0.078183 | 0.022868 |
| Loss | 0.469017 | 0.487398 |
| True positive (TP) | 143 | 362 |
| True negative (TN) | 94 | 328 |
| False positive (FP) | 30 | 83 |
| False negative (FN) | 33 | 127 |
| Tổng số mẫu | 300 | 900 |

Recall và sensitivity là hai tên của cùng đại lượng trong bài toán này; giữ cả hai hàng để phản ánh đầy đủ các trường kết quả đã cung cấp, không xem chúng như hai bằng chứng độc lập.

Theo quy trình notebook, temperature và threshold được xác định trên validation. Các chỉ số validation sau hiệu chỉnh vì vậy không phải đánh giá độc lập của bước hiệu chỉnh; test được đánh giá với các giá trị đã chọn. Không lựa chọn lại threshold bằng nhãn test.

## 3. Bảng rút gọn để đưa vào luận văn

Accuracy, balanced accuracy, precision, sensitivity và specificity trong bảng này dùng đơn vị phần trăm. Các chỉ số còn lại dùng thang giá trị gốc.

| Tập dữ liệu | N | Acc (%) | Bal. Acc (%) | Precision (%) | Sensitivity (%) | Specificity (%) | F1 | MCC | ROC-AUC | AP | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation raw | 300 | 79.00 | 78.53 | 82.66 | 81.25 | 75.81 | 0.8195 | 0.5687 | 0.8527 | 0.8999 | 0.0782 |
| Test raw | 900 | 76.67 | 76.92 | 81.35 | 74.03 | 79.81 | 0.7752 | 0.5363 | 0.8476 | 0.8782 | 0.0229 |

## 4. Ma trận nhầm lẫn

Hàng biểu diễn nhãn thực tế; cột biểu diễn nhãn dự đoán. Lớp dương là glaucoma.

### Validation

| Nhãn thực tế | Dự đoán không glaucoma | Dự đoán glaucoma | Tổng |
|---|---:|---:|---:|
| Không glaucoma | 94 (TN) | 30 (FP) | 124 |
| Glaucoma | 33 (FN) | 143 (TP) | 176 |
| Tổng | 127 | 173 | 300 |

### Test

| Nhãn thực tế | Dự đoán không glaucoma | Dự đoán glaucoma | Tổng |
|---|---:|---:|---:|
| Không glaucoma | 328 (TN) | 83 (FP) | 411 |
| Glaucoma | 127 (FN) | 362 (TP) | 489 |
| Tổng | 455 | 445 | 900 |

Các số đếm nhất quán với tổng số mẫu và các chỉ số accuracy, precision, sensitivity, specificity, F1, balanced accuracy và MCC được cung cấp. ROC-AUC, AP, ECE và loss không thể tính lại chỉ từ ma trận nhầm lẫn; các giá trị này được giữ theo output của người dùng.

## 5. Đoạn mô tả kết quả

> Trên tập kiểm thử raw gồm 900 thể tích OCT Harvard-GF, mô hình kết hợp ResNeXt3D, hai nhánh MaxViT-Tiny và CrossGate đạt accuracy 76.67%, balanced accuracy 76.92%, ROC-AUC 0.8476 và Average Precision 0.8782. Với ngưỡng phân loại 0.43 được lựa chọn trên tập validation, mô hình đạt precision 81.35%, sensitivity 74.03%, specificity 79.81% và F1-score của lớp glaucoma bằng 0.7752. MCC đạt 0.5363, trong khi ECE sau hiệu chỉnh temperature đạt 0.0229. Ma trận nhầm lẫn ghi nhận 362 mẫu glaucoma được nhận diện đúng, 328 mẫu không glaucoma được nhận diện đúng, 83 mẫu dương tính giả và 127 mẫu âm tính giả. Tại ngưỡng đang sử dụng, sensitivity thấp hơn specificity, cho thấy mô hình vẫn bỏ sót một phần các trường hợp glaucoma. Đây là kết quả của một run với seed 42 trên dữ liệu raw; chưa có cơ sở từ riêng kết quả này để kết luận tác động của khử nhiễu hoặc độ ổn định qua nhiều seed.

## 6. Giới hạn báo cáo

- Không cung cấp mean ± std vì chỉ có một run.
- Không bổ sung khoảng tin cậy: output được cung cấp không chứa các giá trị CI.
- Chưa xác nhận thời gian chạy, epoch thực chạy, best epoch hoặc W&B run ID từ dictionary này.
- Số mẫu là số thể tích; không tự diễn giải thành số bệnh nhân độc lập.
- Không suy ra lợi ích riêng của CrossGate, MaxViT hoặc denoise khi chưa có ablation cùng điều kiện.
- Không diễn giải ECE thấp hoặc AUC hiện có như bằng chứng đủ cho ứng dụng chẩn đoán lâm sàng.

Thí nghiệm fine-tune Bilateral 5 epoch chưa hoàn thành do giới hạn tài nguyên, theo cập nhật của người dùng. Không dùng bộ số liệu raw này để thay thế kết quả Bilateral. Xem [bảng trạng thái raw–Bilateral và phần thảo luận cho luận văn](thesis-results-discussion.md).

## 7. Số liệu gốc

```json
{
  "tag": "raw_s42",
  "seed": 42,
  "threshold": 0.42999999999999994,
  "temperature": 1.1326569318771362,
  "val_acc": 0.79,
  "val_balanced_acc": 0.7852822580645161,
  "val_precision": 0.8265895953757225,
  "val_recall": 0.8125,
  "val_sensitivity": 0.8125,
  "val_specificity": 0.7580645161290323,
  "val_f1": 0.8194842406876791,
  "val_mcc": 0.5686525925457464,
  "val_auc_roc": 0.852662206744868,
  "val_auc_pr": 0.8999068669007121,
  "val_ece": 0.07818281367421152,
  "val_tp": 143,
  "val_tn": 94,
  "val_fp": 30,
  "val_fn": 33,
  "val_n": 300,
  "val_loss": 0.4690170884132385,
  "test_acc": 0.7666666666666667,
  "test_balanced_acc": 0.7691699132745212,
  "test_precision": 0.8134831460674158,
  "test_recall": 0.7402862985685071,
  "test_sensitivity": 0.7402862985685071,
  "test_specificity": 0.7980535279805353,
  "test_f1": 0.7751605995717344,
  "test_mcc": 0.5363473595556816,
  "test_auc_roc": 0.8476258713596942,
  "test_auc_pr": 0.8781611967077808,
  "test_ece": 0.022868436012003186,
  "test_tp": 362,
  "test_tn": 328,
  "test_fp": 83,
  "test_fn": 127,
  "test_n": 900,
  "test_loss": 0.48739755153656006
}
```
