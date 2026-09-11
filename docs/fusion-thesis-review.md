# Đánh giá các phương pháp hợp nhất đặc trưng 2D–3D và lựa chọn cấu hình kiểm chứng

## 1. Mục tiêu và mức độ xác minh kết quả

Hợp nhất đặc trưng (fusion) nhằm kết hợp thông tin không gian từ thể tích OCT với thông tin từ các ảnh chiếu 2D. Mục tiêu là khai thác thông tin bổ trợ giữa các nhánh, không đơn thuần tăng số lượng encoder hoặc độ phức tạp mô hình. Vì vậy, lợi ích của một cơ chế fusion cần được xác nhận bằng thí nghiệm giữ cố định các thành phần còn lại và đối chiếu với baseline đơn giản.

Tài liệu này rà soát sáu cơ chế được báo cáo trong khảo sát fusion của repository: Concat, Add, Self-attention, CrossGate, khối SSM có gating mang tên `FusionMamba`, và gating theo nhánh kiểu SE mang tên `FusionFiLM`.

**Kết luận quan trọng:** hiện chưa đủ bằng chứng để xác lập ba phương pháp fusion tốt nhất. Bảng nội bộ cho thấy CrossGate có Test ROC-AUC cao nhất, nhưng cả sáu phương pháp đều có balanced accuracy bằng 0.5 và MCC bằng 0. Ngoài ra, nguồn số liệu chưa được đối chiếu đầy đủ với artifact thực nghiệm gốc.

| Nguồn | Nội dung hiện có | Mức độ sử dụng |
|---|---|---|
| `docs/fusion-ablation-sweep.md` | Bảng validation/test của sáu phương pháp, ghi nguồn W&B; mỗi phương pháp một run | Kết quả được tài liệu nội bộ báo cáo, chưa xác minh lại bằng export gốc. |
| Notebook multiview SOTA | Có cài đặt fusion; bảng output lưu sáu dòng fusion ở trạng thái `ERR`, không có test metrics hợp lệ | Chưa khớp với bảng số trong tài liệu; không suy đoán nguyên nhân lỗi. |
| Notebook mô hình cuối | Có code cho ResNeXt3D + hai MaxViT + CrossGate; các cell huấn luyện chưa có output lưu | Bằng chứng cài đặt và kế hoạch, không phải kết quả huấn luyện. |
| Tài liệu kết quả kỳ vọng | Giá trị ước tính và bảng chờ điền kết quả | Không sử dụng như kết quả thực nghiệm. |

Không tìm thấy export CSV/JSON/JSONL fusion độc lập trong các artifact được rà soát để giải quyết mâu thuẫn giữa bảng báo cáo và notebook. Có thể các nguồn thuộc những lần chạy hoặc phiên bản khác nhau; chưa có căn cứ kết luận nguồn nào phản ánh phiên chạy cuối. Lần rà soát này không truy cập lịch sử W&B hoặc Drive bên ngoài repository.

**Các bảng dưới đây phải được xem là sơ bộ cho đến khi xác nhận run ID, cấu hình, checkpoint và dự đoán tương ứng.**

## 2. Thiết kế khảo sát

| Thành phần | Cấu hình trong sweep fusion |
|---|---|
| Số nhánh | Một nhánh 3D và ba nhánh 2D độc lập |
| Encoder 3D | ConvNeXt3D |
| Encoder 2D | ConvNeXtV2-Tiny, có tiền huấn luyện |
| Ảnh chiếu 2D | Full AIP, Slab AIP và Slab MIP |
| Đầu vào 3D | 200 × 200 × 200 |
| Đầu vào 2D | 224 × 224 |
| Chiều đặc trưng sau projection | 256 mỗi nhánh |
| Batch size / gradient accumulation | 2 / 8; effective batch danh nghĩa 16 |
| Optimizer | AdamW, weight decay 0.0001 |
| Learning rate | Giá trị cấu hình 0.0002; công thức scale trong code cho 0.0004 với effective batch 16 |
| Loss | Cross-entropy không có trọng số lớp trong hàm train được rà soát |
| Ngân sách | Tối đa 10 epoch, patience 6 |
| Số lần chạy được báo cáo | Một run cho mỗi phương pháp; chưa có tổng hợp nhiều seed |
| Chọn checkpoint trong code | Validation accuracy, không phải validation ROC-AUC |
| Ngưỡng phân loại khi tính chỉ số | 0.5 |

Đây là thiết kế ablation tương đối có kiểm soát về backbone, view và độ phân giải. Tuy nhiên, sự tồn tại của cấu hình trong code không thay thế việc xác nhận cấu hình thực sự của từng run trong bảng báo cáo.

Theo tài liệu dataset, tập Training/Validation/Test có lần lượt 2,100/300/900 thể tích. Test gồm 489 mẫu glaucoma dương và 411 mẫu âm. Đây là thống kê tập dữ liệu, chưa phải manifest đã xác nhận cho từng run fusion. Không quy đổi số thể tích thành số bệnh nhân độc lập nếu chưa kiểm tra định danh bệnh nhân.

## 3. Cơ chế fusion theo cài đặt thực tế

Đặt $e_i$ là đặc trưng đầu ra encoder của nhánh $i$. Mỗi nhánh có projection riêng:

$$
p_i=\mathrm{ReLU}(W_i e_i+b_i)\in\mathbb{R}^{256},
$$

trong đó $p_0$ thuộc nhánh 3D và $p_1,p_2,p_3$ thuộc ba nhánh 2D.

### 3.1. Concat và Add

Concat nối bốn vector trước tầng phân loại tuyến tính:

$$
z=[p_0;p_1;p_2;p_3]\in\mathbb{R}^{1024}.
$$

Add cộng các vector đã chiếu về cùng số chiều:

$$
z=\sum_{i=0}^{3}p_i\in\mathbb{R}^{256}.
$$

Concat là baseline dễ diễn giải, giữ các thành phần nhánh tách biệt trước classifier. Trong cài đặt này, sau concat chỉ có linear head, không có MLP tương tác phi tuyến bổ sung. Add đơn giản hơn về biểu diễn đầu ra nhưng không có cơ chế trọng số nhánh tường minh.

Phép nhân từng phần tử (`mul`) cũng có trong code, nhưng không được kích hoạt trong danh sách sáu cấu hình Tier C và không có bảng kết quả tương ứng; vì vậy không xếp hạng phương pháp này.

### 3.2. Self-attention

Các vector nhánh cùng một token CLS học được tạo thành chuỗi đầu vào. Một khối multi-head self-attention và feed-forward có pre-LayerNorm, residual được sử dụng; token CLS cuối khối được đưa vào classifier. Ở chiều 256, cài đặt sử dụng tám attention head.

Phương pháp này cho phép tương tác giữa tất cả token nhánh. Nó khác CrossGate ở chỗ không cố định nhánh 3D làm query duy nhất.

### 3.3. CrossGate

Nhánh 3D đóng vai trò query; các nhánh 2D cung cấp key/value sau LayerNorm:

$$
H=\mathrm{LN}([p_1;p_2;p_3]),\qquad
o=\mathrm{MHA}(p_0,H,H),
$$

$$
z=p_0+\sigma(\alpha)o.
$$

Ở đây MHA bao gồm các phép chiếu query/key/value và output nội bộ. Tham số $\alpha$ được khởi tạo bằng 0.5, nên hệ số cổng ban đầu là $\sigma(0.5)\approx0.6225$.

**Cổng là một scalar học được dùng chung**, không phải cổng riêng theo mẫu, theo kênh hoặc theo view. Trọng số attention mới phụ thuộc nội dung đầu vào. Đường dư giữ đặc trưng 3D làm thành phần nền và bổ sung thông tin 2D qua attention; đây là động cơ thiết kế, không phải bảo đảm hiệu năng sẽ tốt hơn Concat.

### 3.4. Khối SSM có gating (`FusionMamba`)

Cài đặt sử dụng projection, nhánh gating, một state-space model đường chéo, residual, LayerNorm và mean pooling qua token. Các tham số trạng thái và bước thời gian là tham số học chung, không được sinh phụ thuộc đầu vào ở từng token.

Vì vậy, tài liệu này gọi phương pháp là **khối SSM có gating, tên code `FusionMamba`**. Không mô tả nó như một triển khai đầy đủ của selective Mamba chuẩn chỉ dựa trên tên class.

### 3.5. Gating theo nhánh kiểu SE (`FusionFiLM`)

Cài đặt lấy trung bình theo chiều đặc trưng của mỗi nhánh, qua MLP và sigmoid để tạo trọng số:

$$
m_i=\frac{1}{256}\sum_d p_{i,d},\qquad
w=\sigma(\mathrm{MLP}(m)),\qquad
z=\sum_iw_i p_i.
$$

Mỗi mẫu có một trọng số scalar cho mỗi nhánh. Các trọng số sigmoid không bắt buộc cộng bằng 1. Cài đặt không tạo cặp scale/shift theo feature như FiLM affine chuẩn; tên chính xác khi mô tả là **gating theo nhánh kiểu SE, tên code `FusionFiLM`**.

## 4. Kết quả validation được báo cáo

| Phương pháp | Epoch đạt Val AUC cao nhất | Best Val ROC-AUC | AP tại epoch đó | Min Val Loss | Final Val ROC-AUC |
|---|---:|---:|---:|---:|---:|
| SSM có gating (`Mamba`) | 6 | 0.6056 | 0.6448 | 0.6789 | 0.5066 |
| Gating kiểu SE (`FiLM`) | 4 | 0.5718 | 0.6290 | 0.6788 | 0.4934 |
| CrossGate | 1 | 0.5690 | 0.6441 | 0.6839 | 0.4658 |
| Self-attention | 4 | 0.5664 | 0.6292 | 0.6796 | 0.5411 |
| Add | 6 | 0.5389 | 0.6090 | 0.6781 | 0.4484 |
| Concat | 2 | 0.5302 | 0.5967 | 0.6830 | 0.5036 |

Nguồn: bảng validation trong [báo cáo fusion ban đầu](fusion-ablation-sweep.md). Giữ nguyên độ chính xác của số liệu được báo cáo; không bổ sung độ lệch chuẩn hoặc khoảng tin cậy chưa có.

Trong code, chỉ số mang tên PR-AUC được tính bằng `average_precision_score`; vì vậy tài liệu này gọi chính xác là **Average Precision (AP)**, không đồng nhất với diện tích PR tính bằng tích phân hình thang.

Khối SSM đạt đỉnh validation cao nhất nhưng không giữ được mức đó ở epoch cuối. Self-attention có Final Val AUC cao nhất trong bảng, nhưng chỉ một run và chênh lệch đỉnh–cuối chưa đủ để kết luận phương pháp ổn định nhất.

**Min Val Loss không nhất thiết thuộc epoch đạt Best Val AUC.** Hơn nữa, hàm train chọn checkpoint bằng validation accuracy. Vì vậy, không được khẳng định test metrics sau đây được đo tại epoch đạt validation AUC cao nhất nếu chưa kiểm tra checkpoint gốc.

## 5. Kết quả test được báo cáo

| Phương pháp | ROC-AUC | AP | Accuracy | F1 lớp dương | Balanced accuracy | MCC | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| CrossGate | 0.5685 | 0.5981 | 0.5433 | 0.7041 | 0.5000 | 0.0000 | 0.0104 |
| Self-attention | 0.5205 | 0.5567 | 0.5433 | 0.7041 | 0.5000 | 0.0000 | 0.0323 |
| Concat | 0.5031 | 0.5449 | 0.5433 | 0.7041 | 0.5000 | 0.0000 | 0.0394 |
| Gating kiểu SE (`FiLM`) | 0.5028 | 0.5452 | 0.5433 | 0.7041 | 0.5000 | 0.0000 | 0.0622 |
| Add | 0.4914 | 0.5376 | 0.5433 | 0.7041 | 0.5000 | 0.0000 | 0.1481 |
| SSM có gating (`Mamba`) | 0.4715 | 0.5299 | 0.5433 | 0.7041 | 0.5000 | 0.0000 | 0.0402 |

Nguồn: bảng test trong [báo cáo fusion ban đầu](fusion-ablation-sweep.md). Đây là số liệu một run/phương pháp được tài liệu nội bộ báo cáo, chưa đối chiếu với artifact gốc. Precision, recall và specificity không có giá trị trong bảng nguồn nên không tự bổ sung như số đo thực nghiệm.

F1 là F1 của lớp glaucoma dương, không phải macro-F1. ECE trong code dùng 10 bin trên xác suất lớp dương, so sánh xác suất trung bình với tỷ lệ dương thực tế trong bin; không nên mô tả thành một biến thể ECE khác mà không giải thích.

### 5.1. Vì sao F1 bằng 0.7041 không chứng minh chất lượng tốt?

Với 489 mẫu dương trong tổng số 900 mẫu test, giả sử bộ phân loại dự đoán tất cả mẫu là glaucoma, ta có:

$$
TP=489,\quad FP=411,\quad TN=0,\quad FN=0,
$$

$$
\mathrm{Accuracy}=\frac{489}{900}\approx0.5433,
$$

$$
F1_+=\frac{2\times489}{2\times489+411}\approx0.7041.
$$

Balanced accuracy khi đó bằng 0.5 và MCC được quy về 0. Các giá trị này khớp với cả sáu hàng sau làm tròn. Do đó, các chỉ số được báo cáo **phù hợp với hành vi dự đoán toàn bộ về lớp dương**, không thể dùng F1 0.7041 để khẳng định mô hình hoạt động tốt.

Đây là kiểm tra tính nhất quán toán học của giả thuyết dự đoán một lớp, **không phải confusion matrix đã được khôi phục từ file dự đoán thực tế**.

### 5.2. Phân biệt suy biến nhãn và suy biến điểm số

Mọi xác suất có thể lớn hơn 0.5, dẫn đến dự đoán chỉ một lớp, nhưng xác suất giữa các mẫu vẫn khác nhau. Khi đó ROC-AUC vẫn đo khả năng sắp thứ tự điểm số. Vì vậy, không chính xác nếu nói AUC hoàn toàn mất ý nghĩa chỉ vì nhãn dự đoán bị suy biến.

Tuy nhiên, AUC 0.5685 của CrossGate chỉ là tín hiệu phân biệt yếu trong bảng sơ bộ, chưa có khoảng tin cậy hoặc nhiều seed để chứng minh ưu thế đáng tin cậy. ECE thấp cũng không đảm bảo phân biệt tốt: một mô hình dự báo gần tỷ lệ lớp có thể được hiệu chỉnh tương đối tốt nhưng ít hữu ích cho phân loại.

## 6. Có thể lựa chọn ba phương pháp nào?

### 6.1. Thứ tự số học không phải thứ hạng đã xác nhận

| Tiêu chí | Thứ tự trong bảng sơ bộ |
|---|---|
| Best Val ROC-AUC | SSM (`Mamba`) → gating SE (`FiLM`) → CrossGate → Self-attention → Add → Concat |
| Test ROC-AUC | CrossGate → Self-attention → Concat → gating SE (`FiLM`) → Add → SSM (`Mamba`) |
| Test Accuracy, F1, balanced accuracy, MCC | Không phân biệt được sáu phương pháp |
| Ưu thế được xác nhận qua nhiều seed hoặc kiểm định | Chưa có |

Concat chỉ hơn gating SE 0.0003 Test AUC theo các số đã làm tròn. Không có căn cứ coi chênh lệch này là khác biệt chất lượng có ý nghĩa. Việc SSM đứng đầu validation nhưng cuối test cũng chưa cho phép kết luận thuật toán kém nói chung hoặc loại bỏ nguyên tắc lựa chọn bằng validation; cần kiểm tra độ biến thiên và quy trình checkpoint.

Thứ tự test chỉ được mô tả hậu nghiệm. Không dùng chính tập test để chọn kiến trúc rồi xem nó như đánh giá cuối hoàn toàn độc lập.

### 6.2. Bộ ba ưu tiên cho thí nghiệm kiểm chứng

| Phương pháp | Vai trò | Lý do lựa chọn | Giới hạn kết luận |
|---|---|---|---|
| **CrossGate** | Ứng viên chính | Có đường dư 3D và cross-attention định hướng; Test AUC cao nhất trong bảng sơ bộ | Chưa chứng minh vượt các fusion khác hoặc cải thiện chẩn đoán. |
| **Self-attention** | Đối chứng attention | Kiểm tra tương tác giữa tất cả nhánh so với thiết kế lấy 3D làm query | Chưa được xác nhận là phương pháp tốt thứ hai. |
| **Concat** | Baseline bắt buộc | Đơn giản, dễ diễn giải; kiểm tra lợi ích thực sự của việc tăng độ phức tạp fusion | Không được gọi là phương pháp tốt thứ ba chỉ vì hơn FiLM 0.0003 AUC. |

**Đây là bộ ba phục vụ thiết kế ablation, không phải top 3 hiệu năng đã được xác nhận.** Nếu ngân sách cho phép, nên giữ Add làm baseline chi phí thấp và khảo sát các khối SSM/gating dưới tên mô tả đúng cài đặt.

## 7. Không chuyển kết luận sang mô hình cuối một cách trực tiếp

Sweep trong các bảng trên dùng ConvNeXt3D và ba ConvNeXtV2. Mô hình cuối trong `scripts/final_model.py` dùng **ResNeXt3D + hai MaxViT-Tiny + CrossGate**, với hai view Slab MIP và Full AIP. Đây là cấu hình khác về encoder và số view; quy trình huấn luyện trong notebook cuối còn khác về loss và cách đánh giá.

Vì vậy, nếu mô hình cuối đạt kết quả tốt hơn, không được quy toàn bộ cải thiện cho riêng CrossGate. Cần so sánh CrossGate với Concat và Self-attention trên chính cùng bộ encoder, view và quy trình huấn luyện đó.

Notebook cuối có kế hoạch so sánh dữ liệu raw/bilateral với ba seed, nhưng chưa có output huấn luyện thật lưu trong các cell được rà soát. Các giá trị trong tài liệu “expected results” là ước tính, không đưa vào bảng kết quả thực nghiệm. Kiểm tra forward hoặc số tham số chỉ xác nhận kiến trúc/phép tính, không xác nhận hội tụ hoặc hiệu năng trên Harvard-GF.

Tương tự, không quy chênh lệch giữa fusion và single-3D cũ hoàn toàn cho fusion khi baseline single-3D dùng độ phân giải 96³ còn fusion dùng 200³. Sự khác biệt về backbone, độ phân giải và ngân sách huấn luyện phải được loại trừ trước kết luận nhân quả.

## 8. Các vấn đề cần xử lý trước khi công bố kết quả chính thức

- **Nguồn gốc artifact:** thu hồi run ID, history, summary, config, checkpoint và dự đoán validation/test từ W&B/Drive; đối chiếu bảng số với sáu dòng lỗi trong notebook.
- **Chính sách checkpoint:** xác nhận test được đo tại checkpoint nào. Code hiện chọn theo validation accuracy, khác với đề xuất chọn best validation AUC trong tài liệu cũ.
- **Nhãn trường tổng hợp:** trường `best_ep` có chỗ nhận epoch cuối thay vì epoch tốt nhất; một định nghĩa `train_one` cũ ghi `val_roc_auc` từ test. Cần kiểm tra phiên bản đã sinh artifact, không tự áp lỗi của một phiên bản cho mọi run.
- **Gradient accumulation:** vòng train sweep chỉ cập nhật sau nhóm đủ tám minibatch, chưa thấy cập nhật phần dư cuối epoch. Cần kiểm tra/sửa khi tái lập.
- **Tính lặp lại:** DataLoader được cache và có generator shuffle riêng; gọi lại seed toàn cục chưa chắc reset thứ tự minibatch giữa các cấu hình.
- **Hiệu chỉnh xác suất ở notebook cuối:** threshold được chọn trên xác suất validation chưa temperature scaling nhưng áp lên xác suất test đã scaling. Cần chọn threshold trên cùng thang xác suất đã hiệu chỉnh; đây là phát hiện từ code, không phải chứng minh các số liệu sweep đã bị ảnh hưởng.
- **Độc lập tập test:** lựa chọn kiến trúc và ngưỡng trên validation; chỉ đánh giá test sau khi khóa cấu hình. Kiểm tra phân chia theo bệnh nhân và tránh trùng bệnh nhân giữa các split.
- **Nguyên nhân chưa hội tụ:** chưa có bằng chứng quy hiện tượng dự đoán một lớp cho duy nhất encoder 2D, learning rate hoặc cơ chế fusion. Cần kiểm tra loss/output, dữ liệu, gradient, chuẩn hóa và chạy sanity overfit trước khi kết luận nguyên nhân.
- **Thống kê:** chạy nhiều seed, báo mean ± std và khoảng tin cậy phù hợp. Khi so sánh trên cùng bệnh nhân, ưu tiên phân tích ghép cặp và xử lý phụ thuộc giữa các scan cùng bệnh nhân.

Các mục trên là yêu cầu xác minh hoặc vấn đề phát hiện khi đọc code. Tài liệu này không báo cáo rằng chúng đã được sửa, cũng không coi chúng là nguyên nhân suy biến đã được thực nghiệm chứng minh.

## 9. Đoạn kết luận đề xuất đưa vào luận văn

> Sáu cơ chế hợp nhất đặc trưng 2D–3D đã được khảo sát trong tài liệu thực nghiệm sơ bộ, gồm nối đặc trưng, cộng đặc trưng, self-attention, CrossGate, khối SSM có gating và gating theo nhánh kiểu SE. Bảng tổng hợp nội bộ ghi nhận CrossGate đạt ROC-AUC kiểm thử cao nhất là 0.5685, trong khi khối SSM đạt ROC-AUC validation cao nhất là 0.6056 nhưng chỉ đạt 0.4715 trên tập kiểm thử. Tuy nhiên, cả sáu phương pháp đều có balanced accuracy bằng 0.5 và MCC bằng 0; accuracy và F1 phù hợp với hành vi dự đoán toàn bộ mẫu về lớp glaucoma. Các kết quả này chưa chứng minh lợi ích phân loại của fusion và chưa đủ cơ sở xác lập ba phương pháp tốt nhất. Hơn nữa, bảng số liệu và trạng thái lỗi lưu trong notebook cần được đối chiếu bằng artifact gốc trước khi sử dụng như kết quả chính thức. Trên cơ sở động cơ thiết kế và bằng chứng sơ bộ, nghiên cứu lựa chọn CrossGate, self-attention và Concat làm ba cấu hình ưu tiên cho thí nghiệm kiểm chứng có kiểm soát. Đây là quyết định thiết kế ablation, không phải thứ hạng hiệu năng đã được xác nhận. Kết quả của mô hình cuối sử dụng ResNeXt3D và hai nhánh MaxViT phải được báo cáo riêng sau khi hoàn thành huấn luyện và đánh giá.

## 10. Nguồn truy vết

Các liên kết sau phục vụ truy vết nội bộ, không thay thế trích dẫn học thuật cho attention, state-space model hoặc các backbone.

1. [Bảng fusion được báo cáo ban đầu](fusion-ablation-sweep.md).
2. [Notebook SOTA chứa cài đặt và output sweep](../notebooks/3d_glaucoma_multiview_sota_sweep_xai.ipynb).
3. [Notebook multiview ban đầu](../notebooks/3d_glaucoma_multiview_2d3d.ipynb).
4. [Mô tả thiết kế multiview](multiview-model.md).
5. [Cài đặt mô hình cuối](../scripts/final_model.py).
6. [Notebook mô hình cuối](../notebooks/3d_glaucoma_final_2x2d_3d_crossgate.ipynb).
7. [Tài liệu kết quả kỳ vọng, không phải kết quả thực nghiệm](final-model-expected-results.md).
8. [Thống kê dataset](datasets.md).
9. [Khảo sát backbone 3D](3d-backbone-sweep.md).
10. [Khảo sát backbone 2D](2d-feature-extraction-sweep.md).

Tài liệu này tổng hợp và rà soát bằng chứng đã có trong repository; không báo cáo một lần huấn luyện mới, không bổ sung số liệu giả định và không thay đổi kết quả gốc.
