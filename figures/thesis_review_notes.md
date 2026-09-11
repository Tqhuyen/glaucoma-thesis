# Rà soát bản thảo sau cập nhật kết quả raw

## Các tệp cần giữ cùng nhau

- `main.tex`: nội dung chính, các phụ lục kết quả lịch sử và toàn bộ 22 epoch raw được cung cấp.
- `thesis_supporting_methods.tex`: phụ lục bổ trợ được `main.tex` gọi bằng `\input`; chứa đối chiếu bộ dữ liệu, định nghĩa chỉ số và quy trình khảo sát khử nhiễu dài đã chuyển khỏi Chương 4.
- `raw_s42_validation_log.csv`: bản chép chính xác nhật ký người dùng cung cấp, không phải bản xuất W&B.
- `raw_s42_validation.pdf` (hoặc PNG/SVG): biểu đồ các chỉ số validation, không có test và không ngoại suy epoch 23–30.

Khi chuyển bản thảo sang môi trường soạn thảo khác, cần chuyển cả tệp phụ lục và các ảnh được tham chiếu. Hai đường dẫn `prism-uploads` đã được đổi sang các PDF tương ứng có thật trong kho.

## Cấu trúc hiện tại

| Chương | Các mục cấp section |
|---|---|
| 1. Mở đầu | Đặt vấn đề; mục tiêu; câu hỏi; phạm vi; đóng góp; cấu trúc luận văn |
| 2. Tổng quan tài liệu và cơ sở lý thuyết | Tổng quan bài toán; kỹ thuật nền; nghiên cứu liên quan; khoảng trống; tóm tắt |
| 3. Phương pháp đề xuất | Tổng quan kiến trúc; các thành phần; cơ chế tích hợp; huấn luyện; độ phức tạp; tóm tắt |
| 4. Thực nghiệm và đánh giá | Thiết lập; kết quả; giới hạn đối chứng thành phần; định tính và XAI; chi phí/tái lập; thảo luận; tóm tắt |
| 5. Kết luận và hướng phát triển | Kết quả; đóng góp; hạn chế; hướng phát triển ngoài phạm vi |

Không bổ sung nội dung chỉ để đạt một số trang quy định trong dàn ý tham khảo. Quy cách phần đầu, tên đề tài chính thức và phần lý thuyết tạm ẩn của bản gốc được giữ lại.

## Bằng chứng mới được sử dụng

| Mốc | Epoch | Val AUROC | Val BA | Val MCC | Loss nhật ký |
|---|---:|---:|---:|---:|---:|
| Cao nhất theo AUROC trong log đã có | 19 | 0.8603 | 0.7437 | 0.5307 | 0.0656 |
| Cực đại BA và MCC riêng | 17 | 0.8520 | 0.7860 | 0.5666 | 0.0666 |
| Mốc cuối được cung cấp | 22 | 0.8502 | 0.7609 | 0.5156 | 0.0615 |

Đây là một đoạn nhật ký gồm 22 epoch trên lịch danh nghĩa 30 epoch. Không xác nhận đã hoàn tất huấn luyện, đã dừng sớm hay đã lưu checkpoint epoch 19. Chưa có kết quả test của raw hoặc kết quả lượt denoised được cung cấp. Các cực đại ở hai epoch khác nhau không được ghép thành một bộ chỉ số chung.

## Phạm vi cuối đã chốt

Chỉ còn **một thí nghiệm tinh chỉnh 10 epoch trên dữ liệu khử nhiễu**, khởi tạo từ trọng số raw có nguồn gốc đã xác minh. Sau đó cố định trọng số mạng, chỉ thực hiện đánh giá, hiệu chuẩn, lựa chọn ngưỡng và XAI. Không cam kết thêm huấn luyện raw control, seed, mô hình đơn nhánh hoặc ablation. Đối chứng huấn luyện chưa có là hạn chế, không phải công việc bắt buộc bổ sung.

Chỉnh sửa bản thảo không thay đổi notebook hay kernel đang chạy. Notebook trong checkout có các giá trị mặc định khác với giao thức cuối (30 epoch, hai điều kiện đầu vào, nhiều seed); phải chốt cấu hình một lượt trước khi thực thi, không chạy mặc định toàn bộ vòng lặp. Nếu chỉ có trọng số, lần chạy tiếp theo là warm-start, không phải khôi phục nguyên trạng optimizer/scheduler của lần chạy cũ.

## Những điểm phải công khai khi diễn giải

1. Mô hình hiện tại có một nhánh 3D và hai nhánh 2D. Khảo sát ConvNeXt fusion lịch sử vẫn có ba nhánh 2D; không đổi lịch sử để khớp kiến trúc mới.
2. Log raw có định dạng phù hợp với notebook legacy. Trong mã đó, loss được in sau khi chia hệ số tích lũy gradient; mặc định là 8 nhưng cấu hình phiên chạy thực tế chưa được xác nhận. Không coi giá trị này là CE thông thường hoặc ghép trực tiếp với loss Trainer mới.
3. Trainer hiện tại thay đổi cả tổng hợp loss, xử lý cập nhật, cache theo nguồn và thứ tự hiệu chuẩn. Các sửa đổi mới không được gán ngược cho lần chạy raw lịch sử.
4. So sánh raw với denoised sau 10 epoch là so sánh trước–sau mang tính thăm dò. Không tách được khử nhiễu khỏi tối ưu hóa bổ sung và khác biệt triển khai.
5. Lựa chọn một số thành phần đã tham khảo kết quả Test của khảo sát lịch sử. Tập Test đó không được mô tả như hoàn toàn chưa từng tham gia lựa chọn thiết kế. Chọn checkpoint của lượt cuối chỉ bằng validation không xóa được hạn chế này.
6. Grad-CAM, integrated gradients, occlusion và che đầu vào được thực hiện với trọng số cố định. Chúng không thay thế ablation có huấn luyện hoặc xác nhận giải phẫu. Cần kiểm tra đúng nhánh thực sự bị che thay vì chỉ tin tên nhãn XAI.

## Kiểm tra và phần chưa xác minh

- Đã đối chiếu đủ 22 hàng raw trong LaTeX với CSV, xác định đúng các cực đại theo epoch.
- Đã kiểm tra cấu trúc 6/5/6/7/4 mục, dẫn chiếu và nhãn sau khi chuyển phụ lục, cân bằng ngoặc và môi trường LaTeX.
- Đã giữ các giá trị số trong 25 bảng lịch sử, bibliography và các khối comment/tạm ẩn; một số phương trình phương pháp được sửa để phản ánh hai nhánh 2D và phân biệt logits với nhãn.
- Chưa biên dịch PDF tại máy này: không có trình biên dịch LaTeX phù hợp và còn thiếu các hình trích bài báo/các ảnh khử nhiễu đã cắt ở đúng đường dẫn được tham chiếu. Chưa xác minh số trang, ngắt bảng/hình hoặc trang tóm tắt song ngữ sau thay đổi.
- Chưa xác minh tệp trọng số raw hoặc tình trạng phiên huấn luyện từ xa; không tải hoặc chạy huấn luyện thêm trong công việc biên tập này.
- Tên đề tài tiếng Anh và một số thông tin hành chính vẫn là chỗ chờ xác nhận; không tự điền thông tin không được cung cấp.
