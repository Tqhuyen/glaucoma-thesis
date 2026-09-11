# So sánh mô hình kết hợp 3D–2D với các mô hình đơn nhánh trên dữ liệu raw

## 1. Mục đích và kết luận chính

Tài liệu này đối chiếu kết quả raw mới nhất của mô hình **ResNeXt3D + hai MaxViT-Tiny + CrossGate** với kết quả các mô hình 3D và 2D đơn nhánh đã được báo cáo trong repository. Mục tiêu là xác định vị trí của mô hình kết hợp so với các baseline, đồng thời chỉ rõ những giới hạn khi diễn giải sự khác biệt.

**Kết luận chính:** trong các số liệu hiện có, mô hình kết hợp đạt hiệu năng cao hơn các cấu hình 2D đơn nhánh được liệt kê, nhưng **chưa vượt ResNeXt3D đơn nhánh** trên những chỉ số phân loại chính. Kết quả này chưa chứng minh rằng việc thêm hai nhánh MaxViT và CrossGate cải thiện hiệu năng so với một baseline 3D mạnh.

Đây là so sánh giữa các cấu hình đã được chạy ở những điều kiện không hoàn toàn tương đương, **không phải ablation có kiểm soát** để tách riêng đóng góp của fusion. Tài liệu không sử dụng kết quả Bilateral hoặc coi run raw là run đã khử nhiễu.

## 2. Nguồn và mức độ xác minh

| Nhóm kết quả | Nguồn | Mức độ xác minh trong tài liệu này |
|---|---|---|
| Mô hình kết hợp trên raw | Dictionary kết quả `raw_s42` mới nhất do người dùng cung cấp; được lưu trong `docs/final-model-raw-results.md` | Giữ nguyên số liệu; đối chiếu được tính nhất quán của các chỉ số theo ngưỡng với confusion matrix. Chưa có file xác suất để tính lại ROC-AUC, AP, ECE và loss. |
| Mô hình 3D đơn nhánh | `docs/3d-backbone-sweep.md`, ghi nguồn W&B | Dùng số liệu tổng hợp trong tài liệu; chưa xác minh lại export W&B, checkpoint hoặc dự đoán gốc. |
| Mô hình 2D đơn nhánh | `docs/2d-feature-extraction-sweep.md`, ghi nguồn W&B | Dùng số liệu tổng hợp trong tài liệu; chưa xác minh lại export W&B, checkpoint hoặc dự đoán gốc. |

Các snapshot output trong notebook sweep từng có những run mang trạng thái lỗi, trong khi tài liệu tổng hợp báo cáo số liệu thành công. Chưa có đủ run ID và artifact để xác nhận các nguồn đó thuộc cùng một lần chạy. Vì vậy, baseline trong tài liệu này được gọi là **kết quả đã được báo cáo**, không phải kết quả đã tái lập và xác minh độc lập.

Không bổ sung số liệu thiếu, số chữ số vượt độ chính xác nguồn, khoảng tin cậy hoặc mean ± std chưa được cung cấp. Dấu “Chưa có” trong bảng chỉ sự thiếu số liệu trong nguồn được sử dụng, không có nghĩa metric không thể tính hoặc có giá trị bằng 0.

## 3. Mô hình kết hợp và kết quả raw mới nhất

### 3.1. Cấu hình

Mô hình kết hợp gồm một nhánh ResNeXt3D xử lý thể tích OCT, hai nhánh MaxViT-Tiny độc lập xử lý ảnh chiếu Slab MIP và Full AIP, cùng khối hợp nhất CrossGate. Backbone 2D là `maxvit_tiny_rw_224` của thư viện `timm`. Đặc trưng các nhánh được chiếu về 256 chiều; CrossGate sử dụng đặc trưng 3D làm query, hai vector 2D làm key/value và bổ sung đầu ra attention vào đặc trưng 3D qua một cổng scalar học được.

Run `raw_s42` dùng seed 42. Theo cấu hình giai đoạn đã thực hiện, mô hình warm-start từ bộ trọng số được cứu, với đầu vào thể tích 200³ và ảnh chiếu 224². Ngân sách cấu hình là tối đa 10 epoch bổ sung; output được cung cấp **không chứa số epoch thực chạy hoặc epoch của best checkpoint**, nên không khẳng định run đã thực hiện đủ 10 epoch.

### 3.2. Kết quả validation và test

| Chỉ số | Validation | Test |
|---|---:|---:|
| Số thể tích | 300 | 900 |
| Accuracy (%) | 79.00 | 76.67 |
| Balanced accuracy (%) | 78.53 | 76.92 |
| Precision (%) | 82.66 | 81.35 |
| Sensitivity / Recall (%) | 81.25 | 74.03 |
| Specificity (%) | 75.81 | 79.81 |
| F1 lớp glaucoma | 0.8195 | 0.7752 |
| MCC | 0.5687 | 0.5363 |
| ROC-AUC | 0.8527 | 0.8476 |
| Average Precision | 0.8999 | 0.8782 |
| ECE | 0.0782 | 0.0229 |
| Loss | 0.469017 | 0.487398 |

Ngưỡng phân loại là **0.43**, temperature là **1.1326569318771362**. Theo quy trình notebook, temperature và threshold được lựa chọn bằng validation, rồi áp dụng lên test. Các chỉ số validation sau bước hiệu chỉnh không phải đánh giá độc lập của chính bước hiệu chỉnh đó.

Ma trận nhầm lẫn test có TP = 362, TN = 328, FP = 83 và FN = 127. Mô hình đưa ra dự đoán ở cả hai lớp, không biểu hiện dự đoán toàn bộ về một lớp như một số run fusion cũ. Tuy vậy, ở ngưỡng hiện tại, mô hình bỏ sót 127 trong 489 mẫu glaucoma, tương ứng khoảng 25.97%. Đây là hạn chế cần nêu khi thảo luận khả năng phát hiện bệnh; không được chỉ nhấn mạnh accuracy hoặc AUC.

ROC-AUC validation và test chênh khoảng 0.0050; tuy nhiên, chênh lệch nhỏ trong một run không đủ chứng minh mô hình tổng quát hóa ổn định trên nhiều lần chia dữ liệu hoặc quần thể khác.

## 4. Bảng so sánh test với các baseline

Accuracy và balanced accuracy dùng đơn vị phần trăm. F1 là chỉ số lớp glaucoma dương. AP là Average Precision, tương ứng tên `PR-AUC`/`auc_pr` trong các bảng nguồn và cài đặt dùng `average_precision_score`. MCC và ECE dùng thang giá trị gốc.

| Nhóm | Mô hình | Acc (%) | Bal. Acc (%) | ROC-AUC | AP | F1 | MCC | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Kết hợp 3D–2D** | **ResNeXt3D + 2×MaxViT + CrossGate, raw** | **76.67** | **76.92** | **0.8476** | **0.8782** | **0.7752** | **0.5363** | **0.0229** |
| 3D đơn nhánh | ResNeXt3D | 77.11 | 77.19 | 0.8538 | 0.8798 | 0.7836 | 0.5420 | 0.0482 |
| 3D đơn nhánh | VNet | 76.89 | 77.18 | 0.8391 | 0.8714 | 0.7763 | 0.5417 | 0.0346 |
| 3D đơn nhánh | SegResNet | 76.67 | Chưa có | 0.8345 | 0.8690 | 0.7737 | 0.5380 | Chưa có |
| 3D đơn nhánh | CNN3D | 74.44 | Chưa có | 0.8350 | 0.8667 | 0.7741 | 0.4830 | Chưa có |
| 2D đơn nhánh | MaxViT-Tiny, Slab AIP | 68.89 | 68.89 | 0.7638 | 0.7925 | 0.7065 | 0.3765 | 0.1265 |
| 2D đơn nhánh | Tiny2D, Slab MIP | 66.56 | 65.38 | 0.7242 | 0.7509 | 0.7195 | 0.3210 | 0.0549 |
| 2D đơn nhánh | DeiT3, Slab AIP | 65.67 | 65.03 | 0.6988 | 0.7205 | 0.6962 | 0.3040 | 0.0568 |
| 2D đơn nhánh | Tiny2D, Slab AIP | 64.78 | 63.49 | 0.6526 | 0.6563 | 0.7073 | 0.2838 | 0.0550 |
| 2D đơn nhánh | Tiny2D, Full AIP | 59.11 | 56.14 | 0.6605 | 0.6949 | 0.7061 | 0.1704 | 0.0884 |

Hàng in đậm đánh dấu mô hình đang nghiên cứu, không hàm ý mô hình đó dẫn đầu. Đây là tập baseline chọn lọc: gồm các mô hình 3D liên quan trực tiếp và các cấu hình 2D không suy biến trong báo cáo. Toàn bộ sweep, bao gồm các run suy biến và những run thiếu metric, vẫn được lưu trong các tài liệu nguồn.

Precision, sensitivity và specificity test chưa có đầy đủ cho baseline nên không lập bảng so sánh những chỉ số này. Riêng baseline NLM `denoise-enc-32-d5` không đưa vào bảng raw vì dùng tiền xử lý khác, đồng thời thiếu ROC-AUC/AP/MCC trong nguồn.

## 5. Chênh lệch với hai baseline chính

Chênh lệch được tính theo **mô hình kết hợp trừ baseline**, dùng số liệu đầy đủ của mô hình kết hợp và số liệu đã làm tròn của baseline. Vì vậy, chênh lệch cũng là giá trị xấp xỉ.

| Chỉ số | So với ResNeXt3D đơn nhánh | So với MaxViT-Tiny đơn nhánh |
|---|---:|---:|
| Accuracy | −0.44 điểm phần trăm | +7.78 điểm phần trăm |
| Balanced accuracy | −0.27 điểm phần trăm | +8.03 điểm phần trăm |
| ROC-AUC | −0.0062 | +0.0838 |
| AP | −0.0016 | +0.0857 |
| F1 lớp glaucoma | −0.0084 | +0.0687 |
| MCC | −0.0057 | +0.1598 |
| ECE | −0.0253 | −0.1036 |

Đơn vị “điểm phần trăm” không phải phần trăm cải thiện tương đối. Ví dụ, accuracy tăng từ 68.89% lên khoảng 76.67% là tăng khoảng 7.78 điểm phần trăm, không phải tăng tương đối 7.78%.

### 5.1. So với MaxViT-Tiny và các mô hình 2D

So với MaxViT-Tiny đơn nhánh, mô hình kết hợp có ROC-AUC cao hơn khoảng 0.0838 và AP cao hơn khoảng 0.0857. Các chỉ số accuracy, balanced accuracy, F1 và MCC cũng cao hơn. Các giá trị phân loại trong bảng của mô hình kết hợp đều cao hơn năm cấu hình 2D không suy biến được liệt kê.

Quan sát này cho phép nhận xét rằng **cấu hình kết hợp đạt hiệu năng cao hơn các cấu hình 2D đã báo cáo**, nhưng chưa cho phép quy mức tăng cho riêng CrossGate. Mô hình kết hợp đồng thời có thêm một encoder 3D, thêm một encoder 2D và thay đổi loại ảnh chiếu: hai nhánh dùng Slab MIP và Full AIP, trong khi MaxViT đơn nhánh dùng Slab AIP. Số tham số và ngân sách huấn luyện cũng khác.

Vì vậy, không nên viết “CrossGate giúp tăng AUC 0.0838” hoặc “thêm một nhánh 3D chắc chắn mang lại mức tăng này”. Đây là chênh lệch giữa hai cấu hình hoàn chỉnh.

### 5.2. So với ResNeXt3D đơn nhánh

ResNeXt3D đơn nhánh có accuracy 77.11%, ROC-AUC 0.8538 và AP 0.8798; mô hình kết hợp tương ứng đạt khoảng 76.67%, 0.8476 và 0.8782. F1, balanced accuracy và MCC cũng thấp hơn nhẹ trong kết quả của mô hình kết hợp.

Do đó, **chưa quan sát được lợi ích vượt baseline ResNeXt3D về các chỉ số phân loại chính**. Tuy nhiên, chênh lệch nhỏ không đủ để kết luận hai mô hình tương đương thống kê, cũng không đủ để kết luận mô hình kết hợp kém hơn một cách có ý nghĩa thống kê. Cần nhiều seed và dự đoán ghép cặp để đánh giá sự không chắc chắn.

Kết quả này cũng không chứng minh thông tin 2D hoàn toàn không hữu ích. Có thể tồn tại cấu hình encoder, view hoặc quy trình tối ưu khác khai thác được thông tin bổ trợ, nhưng dữ liệu hiện tại chưa kiểm chứng giả thuyết đó.

### 5.3. So với VNet, SegResNet và CNN3D

Mô hình kết hợp có ROC-AUC và AP cao hơn ba baseline này trong số liệu được báo cáo. Tuy nhiên, ưu thế không đồng nhất trên mọi chỉ số. VNet vẫn có accuracy, balanced accuracy, F1 và MCC nhỉnh hơn; SegResNet có MCC nhỉnh hơn và accuracy gần như bằng nhau ở độ chính xác hiển thị. Vì vậy, không mô tả mô hình kết hợp là tốt hơn mọi mô hình 3D.

### 5.4. Về hiệu chỉnh xác suất

ECE của mô hình kết hợp là 0.0229, thấp hơn các baseline không suy biến có ECE trong bảng. Tuy nhiên, mô hình kết hợp đã được temperature scaling trên validation. Quy trình hiệu chỉnh của baseline chưa được xác nhận tương đương trong các tài liệu được sử dụng.

Nhận xét phù hợp là **“ECE sau hiệu chỉnh thấp hơn trong các kết quả được báo cáo”**, không phải “CrossGate tự tạo ra xác suất được hiệu chỉnh tốt hơn”. Để so sánh calibration công bằng, cần áp dụng cùng quy trình và cùng định nghĩa ECE cho tất cả mô hình, đồng thời báo cáo kết quả trước và sau hiệu chỉnh.

ECE thấp cũng không chứng minh mô hình có khả năng chẩn đoán tốt: một dự báo gần tỷ lệ lớp có thể có ECE thấp nhưng ít khả năng phân biệt bệnh.

## 6. Những khác biệt làm hạn chế kết luận nhân quả

| Yếu tố | Mô hình kết hợp hiện tại | Baseline trong báo cáo | Hệ quả |
|---|---|---|---|
| Độ phân giải 3D | 200³ | Nhóm single3D chủ yếu 96³ | Không tách được ảnh hưởng của fusion khỏi độ phân giải. |
| Lịch sử huấn luyện | Warm-start từ trọng số đã học | Các run sweep có lịch sử/ngân sách khác | Không xem là cùng ngân sách tối ưu. |
| Biểu diễn 2D | Slab MIP + Full AIP | MaxViT đơn nhánh dùng Slab AIP | Thay đổi đồng thời số nhánh và loại view. |
| Ngưỡng phân loại | 0.43, chọn bằng validation | Cần đối chiếu từng run; code sweep dùng ngưỡng 0.5 | Accuracy/F1/sensitivity/specificity không hoàn toàn cùng chính sách quyết định. |
| Calibration | Temperature scaling trên validation | Chưa xác nhận tương đương | Không quy chênh lệch ECE cho kiến trúc. |
| Checkpoint | Best validation AUC theo notebook mới | Quy trình cũ có khác biệt về tiêu chí và nhãn trường | Cần xác minh checkpoint thật sự đã sinh từng kết quả test. |
| Số lần lặp | Một run seed 42 | Mỗi cấu hình một run theo tài liệu | Chưa có phân bố kết quả hoặc kiểm định qua seed. |
| Độ phức tạp | Ba encoder và fusion | Một encoder | Chưa có benchmark inference/FLOPs thống nhất để kết luận hiệu quả tài nguyên. |

Các dữ liệu đều được mô tả là Harvard-GF, nhưng trước khi thực hiện kiểm định ghép cặp cần xác nhận manifest split, thứ tự mẫu và tính độc lập theo bệnh nhân. Không tự coi nhiều scan của một bệnh nhân là các mẫu độc lập.

Vì test metrics đã được quan sát trong quá trình phát triển, những điều chỉnh tiếp theo cần được mô tả minh bạch. Không dùng test để chọn threshold hoặc chọn checkpoint, và không coi việc tiếp tục điều chỉnh sau khi xem test là một quy trình hoàn toàn không có nguy cơ thiên lệch lựa chọn.

## 7. Những kết luận được và chưa được hỗ trợ

### Có thể kết luận trong phạm vi số liệu hiện tại

- Mô hình kết hợp raw đạt ROC-AUC test 0.8476, AP 0.8782 và accuracy 76.67% trong run được cung cấp.
- Các chỉ số phân loại của cấu hình kết hợp cao hơn các baseline 2D không suy biến được liệt kê trong bảng nguồn.
- ResNeXt3D đơn nhánh có các chỉ số phân loại chính nhỉnh hơn mô hình kết hợp trong những run được báo cáo.
- Mô hình kết hợp có ECE sau hiệu chỉnh thấp hơn các baseline không suy biến có số liệu ECE trong bảng, với giới hạn về khác biệt quy trình calibration.
- Chưa có bằng chứng thực nghiệm được kiểm soát xác nhận lợi ích của việc thêm hai nhánh 2D vào ResNeXt3D.

### Không nên kết luận

- Mô hình đề xuất tốt nhất trong tất cả kiến trúc hoặc vượt mọi baseline.
- CrossGate riêng lẻ gây ra toàn bộ mức tăng so với 2D.
- Mô hình kết hợp và ResNeXt3D tương đương thống kê, hoặc một mô hình vượt trội có ý nghĩa thống kê.
- ECE thấp chứng minh mô hình sẵn sàng triển khai lâm sàng.
- Kết quả raw này chứng minh lợi ích của Bilateral hoặc bất kỳ phương pháp khử nhiễu nào.

## 8. Đoạn đề xuất đưa vào luận văn

> Trên tập kiểm thử raw gồm 900 thể tích OCT Harvard-GF, mô hình kết hợp ResNeXt3D, hai nhánh MaxViT-Tiny và CrossGate đạt ROC-AUC 0.8476, Average Precision 0.8782 và accuracy 76.67%. So với MaxViT-Tiny đơn nhánh trong bảng kết quả đã báo cáo, các chỉ số này cao hơn lần lượt khoảng 0.0838, 0.0857 và 7.78 điểm phần trăm. Tuy nhiên, mô hình kết hợp chưa vượt ResNeXt3D đơn nhánh, vốn đạt ROC-AUC 0.8538, Average Precision 0.8798 và accuracy 77.11%. Các chỉ số F1, balanced accuracy và MCC của mô hình kết hợp cũng thấp hơn nhẹ so với baseline ResNeXt3D. Do đó, kết quả hiện tại cho thấy cấu hình kết hợp có hiệu năng cao hơn các cấu hình 2D đơn nhánh được so sánh, nhưng chưa xác nhận lợi ích bổ sung so với một encoder 3D mạnh. ECE sau hiệu chỉnh temperature của mô hình kết hợp đạt 0.0229, nhưng chưa thể quy ưu thế này cho riêng kiến trúc do quy trình calibration giữa các thí nghiệm chưa được đồng nhất. Các khác biệt về độ phân giải, loại ảnh chiếu, lịch sử huấn luyện và lựa chọn ngưỡng khiến phép đối chiếu này chỉ mang tính tham khảo giữa các cấu hình, chưa phải ablation có kiểm soát. Ngoài ra, mỗi cấu hình mới có một run và chưa có khoảng tin cậy hoặc kiểm định khác biệt; do đó không kết luận ưu thế hay tương đương thống kê từ các chênh lệch quan sát được.

### Phần Bilateral chưa hoàn thành

Thí nghiệm fine-tune 5 epoch trên Bilateral đã được cấu hình nhưng chưa hoàn thành do giới hạn tài nguyên tính toán, theo trạng thái người dùng xác nhận. Vì chưa có số liệu, không đưa Bilateral vào bảng xếp hạng hiệu năng và không kết luận khử nhiễu giúp cải thiện phân loại. Bảng raw–Bilateral với các ô chưa có kết quả, phần giới hạn và kết luận tổng hợp được trình bày tại [kết quả thực nghiệm và thảo luận](thesis-results-discussion.md).

## 9. Nguồn truy vết

1. [Kết quả raw mới nhất, gồm dictionary gốc và confusion matrix](final-model-raw-results.md).
2. [Báo cáo các backbone 3D đơn nhánh](3d-backbone-sweep.md).
3. [Báo cáo các backbone và ảnh chiếu 2D đơn nhánh](2d-feature-extraction-sweep.md).
4. [Rà soát phương pháp fusion và giới hạn bằng chứng cũ](fusion-thesis-review.md).
5. [Cài đặt mô hình kết hợp](../scripts/final_model.py).

Tài liệu này chỉ đối chiếu các số liệu đã có; không chạy thêm huấn luyện hoặc đánh giá và không bổ sung kết quả giả định. Các liên kết trên là nguồn nội bộ để truy vết, không thay thế trích dẫn học thuật về kiến trúc.
