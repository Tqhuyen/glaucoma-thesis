# Kết quả thực nghiệm, thảo luận và giới hạn nghiên cứu

## 1. Phạm vi kết quả được báo cáo

Nghiên cứu hiện có hai nhóm kết quả: khảo sát chất lượng ảnh sau khử nhiễu và đánh giá mô hình phân loại glaucoma trên dữ liệu OCT raw. Hai nhóm này trả lời hai câu hỏi khác nhau. Các chỉ số ảnh có thể hỗ trợ lựa chọn phương pháp tiền xử lý, nhưng không tự chứng minh phương pháp đó làm tăng hiệu năng phân loại.

Thí nghiệm fine-tune mô hình trên dữ liệu Bilateral trong 5 epoch đã được cấu hình, nhưng **chưa hoàn thành và chưa có kết quả để báo cáo do giới hạn tài nguyên tính toán**, theo trạng thái người dùng xác nhận. Sự tồn tại của notebook, checkpoint đầu vào hoặc smoke test không được tính là bằng chứng thí nghiệm Bilateral đã hoàn thành.

| Nội dung | Trạng thái và bằng chứng | Phạm vi kết luận |
|---|---|---|
| Khảo sát khử nhiễu ảnh | Có bảng chỉ số của 14 thể tích trong báo cáo nội bộ | Đối chiếu các chỉ số chất lượng ảnh gián tiếp, lựa chọn ứng viên để kiểm chứng tiếp. |
| Phân loại trên raw | Có output run `raw_s42`, validation 300 mẫu, test 900 mẫu | Báo cáo hiệu năng của cấu hình model trên raw trong một run. |
| So sánh với đơn nhánh | Có các bảng sweep 2D/3D nội bộ | So sánh tham khảo; chưa phải ablation đồng nhất điều kiện. |
| Fine-tune Bilateral 5 epoch | Chưa hoàn thành do giới hạn tài nguyên; chưa có output hợp lệ được cung cấp | Không báo cáo chỉ số và không kết luận về thay đổi hiệu năng phân loại. |

Số liệu khử nhiễu và baseline được lấy từ các tài liệu nội bộ đã có, chưa được tái lập trong lần tổng hợp này. Số liệu raw mới nhất do người dùng cung cấp; bản gốc được giữ tại [bảng kết quả raw](final-model-raw-results.md).

## 2. Khảo sát chất lượng ảnh sau khử nhiễu

Khảo sát trên 14 thể tích Harvard-GF so sánh sáu phương pháp cổ điển: Gaussian, Median, Bilateral, TV Chambolle, Wavelet BayesShrink và NLM. Dữ liệu được lọc theo từng B-scan 2D. Bảng dưới đây trình bày một số chỉ số tính ở mức toàn thể tích, sau đó lấy trung bình qua 14 thể tích; độ lệch chuẩn và các cách tổng hợp khác được trình bày trong tài liệu nguồn.

| Phương pháp | Độ lệch chuẩn nền | SNR_bg | CNR_bg | Tương quan gradient β |
|---|---:|---:|---:|---:|
| Ảnh gốc | 11.941 | 7.242 | 4.997 | 1.000 |
| Bilateral | 8.817 | 8.737 | 5.746 | 0.858 |
| Wavelet BayesShrink | 9.993 | 8.316 | 5.621 | 0.901 |
| TV Chambolle | 9.945 | 7.617 | 4.756 | 0.598 |
| NLM | 9.917 | 7.679 | 4.827 | 0.606 |

Nguồn: [khảo sát toàn thể tích](../figures/denoise/wholevolume/summary.md). Bảng chọn lọc này không thay thế bảng đầy đủ sáu phương pháp và không được dùng như bảng hiệu năng phân loại.

Bilateral có độ biến thiên nền thấp và CNR_bg cao trong bảng tổng hợp, trong khi Wavelet có tương quan gradient với ảnh gốc cao nhất trong các phương pháp lọc được khảo sát. TV là ứng viên làm mịn mạnh, nhưng vị trí so với NLM phụ thuộc cách tổng hợp chỉ số. Vì vậy, Bilateral, Wavelet và TV được ưu tiên cho các thí nghiệm tiếp theo, với NLM là đối chứng quan trọng; đây không phải thứ hạng tốt nhất tuyệt đối.

Các chỉ số không có ảnh sạch tham chiếu và chưa có phân đoạn RNFL được xác nhận. β cao không đồng nghĩa bảo toàn từng ấy phần trăm cấu trúc giải phẫu; ảnh gốc còn chứa speckle và có β bằng 1 khi tự so sánh. Độ biến thiên nền thấp có thể phản ánh làm mịn, nhưng không chứng minh thông tin phục vụ chẩn đoán được giữ nguyên. Vì vậy, **chưa chuyển kết luận về chất lượng ảnh thành kết luận cải thiện phân loại**.

Phân tích chi tiết công thức, vùng đo và giới hạn: [đánh giá các phương pháp khử nhiễu](denoise-top3-thesis-review.md).

## 3. Kết quả phân loại trên raw và trạng thái Bilateral

Mô hình gồm một nhánh ResNeXt3D, hai nhánh MaxViT-Tiny nhận Slab MIP và Full AIP, cùng khối hợp nhất CrossGate. Run raw được báo cáo có seed 42 và được warm-start từ trọng số đã cứu. File kết quả không có epoch thực chạy hoặc epoch của best checkpoint, nên không tự điền những thông tin đó.

| Chỉ số | Raw: validation | Raw: test | Bilateral sau fine-tune 5 epoch: validation | Bilateral sau fine-tune 5 epoch: test |
|---|---:|---:|---|---|
| Accuracy (%) | 79.00 | 76.67 | Chưa có | Chưa có |
| Balanced accuracy (%) | 78.53 | 76.92 | Chưa có | Chưa có |
| Precision (%) | 82.66 | 81.35 | Chưa có | Chưa có |
| Recall / Sensitivity (%) | 81.25 | 74.03 | Chưa có | Chưa có |
| Specificity (%) | 75.81 | 79.81 | Chưa có | Chưa có |
| F1 lớp glaucoma | 0.8195 | 0.7752 | Chưa có | Chưa có |
| MCC | 0.5687 | 0.5363 | Chưa có | Chưa có |
| ROC-AUC | 0.8527 | 0.8476 | Chưa có | Chưa có |
| Average Precision | 0.8999 | 0.8782 | Chưa có | Chưa có |
| ECE | 0.0782 | 0.0229 | Chưa có | Chưa có |
| Loss | 0.469017 | 0.487398 | Chưa có | Chưa có |
| TP | 143 | 362 | Chưa có | Chưa có |
| TN | 94 | 328 | Chưa có | Chưa có |
| FP | 30 | 83 | Chưa có | Chưa có |
| FN | 33 | 127 | Chưa có | Chưa có |
| Số mẫu đã đánh giá | 300 | 900 | Chưa xác nhận | Chưa xác nhận |

“Chưa có” biểu thị thí nghiệm chưa hoàn thành, không phải giá trị 0 hoặc kết quả thất bại về chất lượng. Không tính chênh lệch raw–Bilateral khi chưa có số liệu Bilateral. Không dùng các giá trị kỳ vọng để điền vào các ô này.

Ngưỡng phân loại của run raw là 0.43 và temperature là 1.1326569318771362. Theo quy trình notebook, các giá trị này được xác định trên validation rồi áp dụng lên test. Chúng không được tự chuyển thành tham số đánh giá tối ưu cho Bilateral.

Trên test raw, mô hình nhận diện đúng 362 trong 489 mẫu glaucoma và 328 trong 411 mẫu không glaucoma. Mô hình không suy biến thành dự đoán một lớp, nhưng còn bỏ sót 127 mẫu glaucoma, tương ứng khoảng 25.97% số mẫu dương ở ngưỡng đang sử dụng. Kết quả này cần được báo cáo cùng accuracy và AUC để thể hiện đầy đủ hạn chế.

## 4. So sánh tham khảo với mô hình đơn nhánh

| Cấu hình | Test accuracy (%) | Test ROC-AUC | Test AP | Test F1 | Test MCC |
|---|---:|---:|---:|---:|---:|
| ResNeXt3D + 2×MaxViT + CrossGate, raw | 76.67 | 0.8476 | 0.8782 | 0.7752 | 0.5363 |
| ResNeXt3D đơn nhánh | 77.11 | 0.8538 | 0.8798 | 0.7836 | 0.5420 |
| MaxViT-Tiny đơn nhánh, Slab AIP | 68.89 | 0.7638 | 0.7925 | 0.7065 | 0.3765 |

Mô hình kết hợp có kết quả cao hơn MaxViT đơn nhánh được báo cáo, nhưng chưa vượt ResNeXt3D đơn nhánh trên các chỉ số phân loại chính. Điều này chưa xác nhận lợi ích của việc thêm hai nhánh 2D so với một encoder 3D mạnh.

Các cấu hình khác nhau về độ phân giải, ảnh chiếu, lịch sử huấn luyện, cách chọn checkpoint, threshold và calibration. Do đó, không quy toàn bộ chênh lệch cho CrossGate và không kết luận khác biệt có ý nghĩa thống kê khi chưa có kiểm định. Chi tiết: [so sánh với các mô hình đơn nhánh](final-model-raw-vs-single-branch.md).

## 5. Giới hạn nghiên cứu

Nghiên cứu chưa hoàn thành thí nghiệm fine-tune Bilateral do giới hạn tài nguyên tính toán. Vì vậy, giả thuyết khử nhiễu Bilateral giúp cải thiện phân loại glaucoma vẫn chưa được kiểm chứng trong cấu hình mô hình cuối. Đây là phần thực nghiệm còn thiếu, không được thay thế bằng đánh giá chất lượng ảnh hoặc bằng kết quả raw.

Ngoài ra, kết quả phân loại mới nhất chỉ gồm một run seed 42. Chưa có mean ± std qua nhiều seed, khoảng tin cậy được cung cấp cho bảng này hoặc kiểm định ghép cặp với baseline. Số mẫu được báo cáo là số thể tích; tính độc lập theo bệnh nhân cần được xác minh từ manifest trước khi thực hiện suy luận thống kê.

Temperature và threshold được chọn trên validation, nên chỉ số validation sau hiệu chỉnh không phải phép kiểm tra độc lập của bước hiệu chỉnh. Test đã được quan sát trong quá trình phát triển; việc tiếp tục điều chỉnh mô hình cần được công bố rõ và tránh chọn cấu hình hoặc ngưỡng dựa trực tiếp vào test.

Các kết quả hiện có chưa đủ để khẳng định tính sẵn sàng triển khai lâm sàng, tính vượt trội của fusion hoặc hiệu quả của denoise đối với chẩn đoán.

## 6. Hướng phát triển

Khi có tài nguyên phù hợp, thí nghiệm tiếp theo là fine-tune 5 epoch trên Bilateral từ checkpoint của model raw, giữ cố định split và seed, tạo cả hai ảnh chiếu từ dữ liệu Bilateral. Checkpoint được chọn trên validation Bilateral; temperature và threshold cũng được xác định trên validation đó trước khi đánh giá test.

Kết quả thí nghiệm này, nếu hoàn thành, cần được mô tả là **fine-tune trên Bilateral từ model đã học trên raw**, không phải huấn luyện hoàn toàn trên Bilateral từ đầu. Để tách ảnh hưởng của denoise khỏi lợi ích của việc huấn luyện thêm, cần thêm đối chứng tiếp tục fine-tune 5 epoch trên raw với ngân sách tương đương. Chỉ so model raw trước fine-tune với model Bilateral sau fine-tune chưa tách được hai tác động này.

Các bước mở rộng khác gồm nhiều seed, đánh giá cấu trúc trên ROI được xác nhận, và ablation cùng điều kiện giữa ResNeXt3D đơn nhánh, Concat và CrossGate. Đây là đề xuất tương lai, không phải thí nghiệm đã thực hiện. Tài liệu này không khởi chạy thêm tác vụ tính toán.

## 7. Kết luận đề xuất đưa vào luận văn

> Nghiên cứu đã khảo sát các phương pháp khử nhiễu trên ảnh OCT và đánh giá mô hình kết hợp ResNeXt3D, hai nhánh MaxViT-Tiny và CrossGate trên dữ liệu raw. Khảo sát chỉ số ảnh cho thấy Bilateral là ứng viên có sự cân bằng thuận lợi giữa giảm biến thiên nền và duy trì tương quan gradient với ảnh gốc, nhưng chưa chứng minh hiệu quả cải thiện phân loại. Trên tập kiểm thử raw gồm 900 thể tích, mô hình kết hợp đạt ROC-AUC 0.8476, Average Precision 0.8782, accuracy 76.67% và F1 lớp glaucoma 0.7752. Kết quả cao hơn cấu hình MaxViT đơn nhánh được báo cáo, nhưng chưa vượt baseline ResNeXt3D đơn nhánh; các khác biệt về quy trình thí nghiệm chưa cho phép quy chênh lệch cho riêng cơ chế fusion. Thí nghiệm fine-tune Bilateral chưa hoàn thành do giới hạn tài nguyên tính toán, nên nghiên cứu chưa đưa ra kết luận về tác động của khử nhiễu đối với hiệu năng phân loại glaucoma. Kiểm chứng tác động này bằng các đối chứng cùng ngân sách và nhiều seed là hướng phát triển tiếp theo.

## 8. Nguồn truy vết

1. [Kết quả phân loại raw và số liệu gốc](final-model-raw-results.md).
2. [So sánh với các mô hình 3D/2D đơn nhánh](final-model-raw-vs-single-branch.md).
3. [Rà soát chỉ số và lựa chọn phương pháp khử nhiễu](denoise-top3-thesis-review.md).
4. [Bảng khảo sát khử nhiễu trên 14 thể tích](../figures/denoise/wholevolume/summary.md).
5. [Cấu hình fine-tune và cơ chế checkpoint; không phải bằng chứng đã hoàn thành](final-training-recovery.md).

Không lấy số liệu từ tài liệu `final-model-expected-results.md` để thay cho kết quả thực nghiệm. Các giá trị giả định trong tài liệu đó không thuộc bảng kết quả của luận văn.
