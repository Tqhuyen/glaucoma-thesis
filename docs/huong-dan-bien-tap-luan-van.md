# Hướng dẫn chuyển bản thảo từ nhật ký nghiên cứu sang luận văn học thuật

Áp dụng cho luận văn phân loại glaucoma từ khối OCT và hai ảnh chiếu en-face, sử dụng ResNeXt3D, MaxViT-Tiny và CrossGate.

> Không tổ chức luận văn theo trình tự “đã làm gì tiếp theo”, mà theo trình tự “vấn đề nghiên cứu là gì, giải pháp được thiết kế ra sao, bằng chứng nào đã có và bằng chứng đó cho phép kết luận đến đâu”.

Luận văn cần mô tả thực nghiệm và khả năng tái lập, nhưng không cần kể lại toàn bộ quá trình thử, lỗi, đổi cấu hình và xử lý sự cố. Hướng dẫn này không yêu cầu huấn luyện thêm ngoài một lượt tinh chỉnh cuối cùng với ngân sách 10 epoch trên ảnh khử nhiễu đã được chốt.

## 1. Phân biệt nhật ký nghiên cứu và luận văn

| Nhật ký nghiên cứu | Luận văn |
|---|---|
| Sắp xếp theo thời gian thực hiện | Sắp xếp theo câu hỏi và lập luận |
| Ghi mọi lần thử, lỗi và thay đổi | Chọn bằng chứng liên quan đến mục tiêu |
| “Sau đó thử mô hình B…” | “Mô hình B được khảo sát để đánh giá…” |
| “Chạy bị hết bộ nhớ GPU nên đổi…” | “Giới hạn bộ nhớ đặt ra yêu cầu…” |
| Liệt kê từng epoch | Tổng hợp xu hướng, mốc lựa chọn và kết quả |
| Lặp “cần bổ sung”, “cần kiểm tra” ở nhiều nơi | Xác định rõ giao thức, kết quả hiện có và hạn chế |
| Nhấn mạnh các công việc đã thực hiện | Nhấn mạnh ý nghĩa của thiết kế và kết quả |

**Nguyên tắc chọn nội dung:** một đoạn nên nằm trong nội dung chính nếu nó giúp người đọc hiểu bài toán, phương pháp, bằng chứng hoặc kết luận. Chi tiết phục vụ truy vết nhưng không giúp lập luận nên chuyển sang phụ lục.

Không loại bỏ thông tin bất lợi chỉ để bản thảo trông thuyết phục hơn. Những thay đổi phiên bản, sai khác quy trình hoặc hạn chế dữ liệu ảnh hưởng đến kết luận vẫn phải được công khai.

## 2. Chọn một mạch lập luận xuyên suốt

Mạch chính phù hợp với đề tài này là:

> OCT chứa thông tin cấu trúc ba chiều, nhưng việc khai thác khối ảnh có chi phí tính toán cao và chịu ảnh hưởng của nhiễu. Nghiên cứu lựa chọn cách kết hợp đặc trưng thể tích với đặc trưng từ hai ảnh chiếu, xây dựng mô hình ba nhánh và đánh giá mô hình trên dữ liệu hiện có. Một giai đoạn tinh chỉnh trên ảnh khử nhiễu được dùng để khảo sát sự thay đổi hiệu năng; các phân tích XAI bổ sung góc nhìn về hành vi dự đoán.

Từ mạch đó, mỗi chương thực hiện một nhiệm vụ riêng. Không để cả năm chương cùng lặp danh sách ResNeXt3D, MaxViT, Bilateral và các điểm số khảo sát.

| Chương | Câu hỏi cần trả lời |
|---|---|
| 1. Mở đầu | Vấn đề là gì, tại sao cần nghiên cứu, mục tiêu và phạm vi ra sao? |
| 2. Tổng quan tài liệu và cơ sở lý thuyết | Các hướng tiếp cận và nguyên lý nào liên quan đến thiết kế? |
| 3. Phương pháp đề xuất | Hệ thống được xây dựng và các thành phần được kết nối như thế nào? |
| 4. Thực nghiệm và đánh giá | Bằng chứng đã có là gì và cho phép kết luận đến đâu? |
| 5. Kết luận và hướng phát triển | Đã đạt được gì, còn hạn chế gì và có thể mở rộng theo hướng nào? |

## 3. Chuyển đổi nội dung theo từng chương

### 3.1. Chương 1: nêu vấn đề và mục tiêu

Chương này cần làm rõ ý nghĩa của bài toán phân loại glaucoma từ OCT, khó khăn kỹ thuật, hướng giải quyết, phạm vi và đóng góp. Không kể quá trình thử nghiệm theo thời gian.

**Không nên viết:**

> Ban đầu thử ResNet nhưng kết quả chưa tốt, sau đó thử DenseNet, rồi chuyển sang MaxViT và ResNeXt3D.

**Nên viết:**

> Nghiên cứu xem xét kiến trúc kết hợp đặc trưng thể tích và ảnh chiếu nhằm khai thác các biểu diễn khác nhau của cùng khối OCT. Các khảo sát thành phần được sử dụng làm cơ sở lựa chọn bộ trích xuất đặc trưng và cơ chế hợp nhất.

Giới hạn tài nguyên chỉ cần nêu ngắn gọn trong phạm vi nghiên cứu. Không đưa câu chuyện credit, thuê máy hoặc gián đoạn phiên chạy vào phần đặt vấn đề, trừ khi chúng trực tiếp xác định thiết kế thực nghiệm và cần được giải thích.

Sáu mục chính nên gồm:

1. Đặt vấn đề.
2. Mục tiêu nghiên cứu.
3. Câu hỏi nghiên cứu.
4. Phạm vi và đối tượng nghiên cứu.
5. Đóng góp của luận văn.
6. Cấu trúc luận văn.

Mục tiêu không nên được viết như một kết quả đã bảo đảm, chẳng hạn “xây dựng mô hình có độ chính xác cao hơn mọi mô hình đơn lẻ”. Thay vào đó, nêu mục tiêu xây dựng và đánh giá, rồi để thực nghiệm xác định mức hiệu quả.

### 3.2. Chương 2: trình bày tri thức nền và nghiên cứu liên quan

Chương này trả lời “cơ sở nào dẫn đến thiết kế?”, không phải “mô hình của nghiên cứu đạt bao nhiêu điểm?”.

Nội dung nên gồm đặc điểm OCT và các ảnh chiếu, nguyên lý liên quan của Bilateral, CNN/ResNeXt, MaxViT và cross-attention, các hướng tiếp cận trong tài liệu, cùng những vấn đề mà thiết kế cần xem xét.

Các con số từ khảo sát của chính luận văn nên tập trung ở Chương 4. Chương 2 dẫn chiếu đến đó khi cần, tránh lặp bảng xếp hạng của nghiên cứu trong phần tổng quan tài liệu.

Không dùng câu:

> Chưa có nghiên cứu nào kết hợp các mô hình này.

trừ khi có khảo sát tài liệu đủ mạnh để chứng minh. Cách viết thận trọng hơn là:

> Trong phạm vi tài liệu được khảo sát, lợi ích của việc kết hợp các biểu diễn này đối với thiết lập dữ liệu đang xét chưa được xác định đầy đủ.

Năm mục chính nên gồm:

1. Tổng quan về bài toán.
2. Các kỹ thuật nền tảng.
3. Các nghiên cứu liên quan, được nhóm theo hướng tiếp cận.
4. Khoảng trống nghiên cứu.
5. Tóm tắt chương và dẫn sang phương pháp đề xuất.

Phân biệt khoảng trống trong tài liệu với việc bản thân nghiên cứu chưa thực hiện một phép kiểm chứng. Thiếu một thí nghiệm trong luận văn không tự chứng minh rằng lĩnh vực chưa từng nghiên cứu vấn đề đó.

### 3.3. Chương 3: đặc tả phương pháp, không viết thành hướng dẫn chạy notebook

Mỗi thành phần nên có cùng khung trình bày:

> Mục đích → đầu vào → phép xử lý → đầu ra → lý do thiết kế.

**Ví dụ cho phép chiếu đặc trưng:**

> Các bộ trích xuất 2D và 3D tạo ra vector có số chiều khác nhau. Vì vậy, mỗi vector được ánh xạ về không gian 256 chiều bằng một lớp tuyến tính kết hợp ReLU. Các vector sau phép chiếu là đầu vào của khối CrossGate.

Đây là mô tả phương pháp. Tên cell, lệnh cài thư viện, đường dẫn Drive và thao tác khôi phục phiên chạy nên nằm trong phụ lục tái lập.

Phân biệt rõ bốn nhóm nội dung:

- **Kiến trúc mạng:** các phép toán từ đầu vào đến logits.
- **Chiến lược huấn luyện:** dữ liệu, hàm mất mát, bộ tối ưu và lựa chọn checkpoint.
- **Hậu xử lý:** temperature scaling và ngưỡng quyết định.
- **Giải thích mô hình:** Grad-CAM, integrated gradients, occlusion và các phương pháp liên quan.

Không gom tất cả thành một danh sách thao tác liên tiếp. Khi mô tả tensor và công thức, phân biệt logits, xác suất và nhãn dự đoán; không dùng cùng một ký hiệu cho cả ba.

Sáu mục chính nên gồm:

1. Tổng quan kiến trúc đề xuất.
2. Mô tả chi tiết các thành phần.
3. Cơ chế tích hợp và kết nối.
4. Chiến lược huấn luyện.
5. Độ phức tạp và yêu cầu tính toán.
6. Tóm tắt chương.

Các thông số đã kiểm tra trong mã hiện tại phải được phân biệt với cấu hình của lần chạy lịch sử. Không gán ngược các sửa lỗi hoặc cải tiến mới cho một nhật ký cũ khi chưa xác minh phiên bản thực thi.

### 3.4. Chương 4: trình bày bằng chứng theo câu hỏi

Đây là phần dễ trở thành nhật ký nhất.

**Không nên viết:**

> Epoch 14 tăng lên 0,8353, epoch 16 tăng tiếp, đến epoch 19 đạt 0,8603, sau đó giảm nên chúng tôi…

**Nên viết:**

> Trong 22 epoch được ghi nhận, AUROC trên tập xác thực đạt cực đại 0,8603 tại epoch 19. Balanced accuracy và MCC ở cùng epoch lần lượt là 0,7437 và 0,5307. Các cực đại của hai chỉ số phụ xuất hiện tại epoch 17, cho thấy mốc lựa chọn phụ thuộc tiêu chí đánh giá. Luận văn sử dụng AUROC xác thực làm tiêu chí lựa chọn đã xác định.

Sử dụng một bảng ngắn cho các mốc quan trọng, một biểu đồ cho diễn biến và phụ lục để lưu đủ 22 dòng nhật ký. Không ghép AUROC tốt nhất của epoch 19 với BA/MCC tốt nhất của epoch 17 thành một bộ chỉ số chung.

Mỗi bảng hoặc hình nên được phân tích theo bốn bước:

1. **Quan sát:** kết quả nào xuất hiện?
2. **Diễn giải:** kết quả gợi ý điều gì?
3. **Giới hạn:** chưa thể kết luận điều gì?
4. **Liên hệ:** kết quả trả lời câu hỏi nghiên cứu nào?

Không diễn giải lại mọi ô trong bảng bằng văn xuôi. Tập trung vào xu hướng, khác biệt đáng chú ý, trường hợp thất bại và điều kiện so sánh.

Bảy mục chính nên gồm:

1. Thiết lập thực nghiệm: dữ liệu, chỉ số, môi trường và phương pháp so sánh.
2. Kết quả chính.
3. Đánh giá đóng góp thành phần và giới hạn đối chứng.
4. Phân tích định tính và XAI.
5. Phân tích chi phí và khả năng tái lập.
6. Thảo luận, đối chiếu với câu hỏi nghiên cứu.
7. Tóm tắt chương.

Nếu không có ablation được huấn luyện trong cùng điều kiện, mục về đóng góp thành phần phải trình bày đúng giới hạn đó. Không cần tạo một bảng ablation giả định để đáp ứng hình thức của dàn ý mẫu.

### 3.5. Chương 5: tổng kết điều đã biết

Chương kết luận không nên chỉ liệt kê:

> Đã chạy mô hình A, đã tạo ảnh B, đã lưu lên Drive và sẽ làm XAI.

Nên tập trung vào hệ thống đã xây dựng, bằng chứng đã có, kết luận được hỗ trợ và những hạn chế chưa giải quyết.

Đóng góp về tích hợp hệ thống và tổ chức thực nghiệm có giá trị, nhưng không đồng nghĩa với việc đã chứng minh mô hình tốt hơn mọi baseline.

Bốn mục chính nên gồm:

1. Tóm tắt kết quả nghiên cứu.
2. Đóng góp chính.
3. Hạn chế.
4. Hướng phát triển ngoài phạm vi luận văn.

Không giới thiệu kết quả mới ở chương kết luận. Hướng phát triển phải được phân biệt với các công việc còn cần hoàn thành trong phạm vi hiện tại.

## 4. Trình bày giai đoạn tinh chỉnh cuối cùng 10 epoch

**Không nên viết:**

> Do còn ít credit nên quyết định train thêm 10 epoch cho kết quả đẹp hơn.

**Cách viết khi đây vẫn là kế hoạch:**

> Trong phạm vi nguồn lực thực nghiệm, nghiên cứu dự kiến thực hiện một giai đoạn tinh chỉnh bổ sung với ngân sách 10 epoch trên dữ liệu khử nhiễu, khởi tạo từ trọng số của mô hình đã huấn luyện trên dữ liệu raw. Sau giai đoạn này, trọng số mạng được cố định để đánh giá và phân tích XAI.

Chỉ đổi “dự kiến thực hiện” thành “đã thực hiện” sau khi có bằng chứng thực tế. Số epoch thực hiện, checkpoint sử dụng và kết quả phải được ghi đúng theo nhật ký; không mặc định kế hoạch đã hoàn thành.

Giới hạn so sánh nên được trình bày đầy đủ tại phần thiết lập hoặc thảo luận:

> So sánh trước–sau phản ánh đồng thời thay đổi đầu vào và tối ưu hóa bổ sung; do không có lượt tiếp tục huấn luyện raw tương ứng, kết quả không xác định riêng tác động của khử nhiễu.

Không cần lặp nguyên cảnh báo này ở mọi tiểu mục. Tại các phần khác, dùng một câu ngắn hoặc dẫn chiếu đến phần thảo luận.

Những điểm cần giữ chính xác:

- Chỉ có một thí nghiệm tinh chỉnh cuối cùng; không cam kết thêm raw control, seed hoặc huấn luyện ablation.
- Nếu không xác minh được trọng số epoch 19, không gọi checkpoint đang có là “checkpoint tốt nhất tại epoch 19”. Ghi đúng nguồn trọng số thực sự sử dụng.
- Nạp trọng số rồi khởi tạo lại optimizer và lịch học là warm-start, không phải khôi phục nguyên trạng toàn bộ lần chạy cũ.
- Phân biệt kết quả validation với test; không so sánh trực tiếp AUROC validation của mô hình mới với AUROC test của mô hình lịch sử để tuyên bố mức cải thiện.
- Ghi nhận thay đổi phiên bản triển khai nếu chúng ảnh hưởng định nghĩa loss, cache hoặc đánh giá.

## 5. Viết XAI như phân tích, không như bộ sưu tập ảnh

Không chỉ chèn Grad-CAM rồi viết:

> Mô hình tập trung đúng vùng bệnh.

Cách trình bày phù hợp hơn:

> Bản đồ Grad-CAM cho thấy các vùng có ảnh hưởng lớn đến điểm dự đoán của lớp đang xét. Việc đối chiếu bản đồ với ảnh đầu vào giúp khảo sát hành vi mô hình; tuy nhiên, chưa có nhãn vùng tổn thương để xác nhận sự phù hợp về giải phẫu.

Mỗi hình XAI nên làm rõ:

- Mẫu và lớp đang được giải thích.
- Dự đoán đúng hay sai theo nhãn tham chiếu.
- Vùng nào có ảnh hưởng nổi bật.
- Quan sát có nhất quán giữa các mẫu hay không.
- Giới hạn của phương pháp giải thích.

Sau lượt tinh chỉnh cuối, các phân tích này sử dụng trọng số mạng cố định. Việc tính gradient để giải thích không đồng nghĩa với cập nhật trọng số bằng optimizer.

Che một nhánh khi suy luận nên gọi là **phân tích độ nhạy khi loại bỏ đầu vào**, không gọi là ablation có huấn luyện. Kết quả che đầu vào cũng không tự chứng minh lợi ích của nhánh đó nếu được huấn luyện hoặc loại bỏ theo một thiết kế khác.

## 6. Phân chia nội dung chính và phụ lục

| Giữ trong nội dung chính | Chuyển sang phụ lục |
|---|---|
| Sơ đồ end-to-end và CrossGate | Cấu hình đầy đủ, lệnh chạy và phiên bản |
| Thống kê dữ liệu chính | Đối chiếu dài giữa nhiều bộ dữ liệu |
| Bảng kết quả quan trọng | Toàn bộ bảng khảo sát lịch sử |
| Biểu đồ diễn biến validation | Nhật ký từng epoch |
| Công thức trực tiếp giải thích mô hình | Suy dẫn dài và định nghĩa chỉ số thông dụng |
| Một số trường hợp XAI tiêu biểu | Bộ ảnh XAI bổ sung |
| Giới hạn ảnh hưởng đến kết luận | Chi tiết lỗi kỹ thuật, cache và khôi phục |

Chi tiết kỹ thuật ảnh hưởng trực tiếp đến cách hiểu kết quả vẫn phải được nhắc trong nội dung chính. Ví dụ, loss legacy có cách ghi khác loss hiện tại: giải thích ngắn ở Chương 4 và dẫn sang phụ lục, không giấu hoàn toàn.

Khi di chuyển nội dung trong LaTeX, di chuyển cả môi trường và nhãn liên quan, không sao chép tạo nhãn trùng. Cập nhật các câu dẫn để người đọc biết thông tin chi tiết nằm ở đâu.

## 7. Văn phong và thuật ngữ

Ưu tiên cách diễn đạt trung tính, có chủ thể và có phạm vi rõ ràng:

| Tránh | Nên dùng |
|---|---|
| “Kết quả rất đẹp/tốt” | Nêu chỉ số, điều kiện đo và ý nghĩa của kết quả |
| “Mô hình này thắng” | “Mô hình có chỉ số cao nhất trong các lần chạy được báo cáo” |
| “Cần kiểm tra lại nhiều thứ” | Chỉ rõ thông tin chưa xác minh và ảnh hưởng đến kết luận |
| “Sau đó thử tiếp…” | Nêu mục đích của phép khảo sát hoặc đối chứng |
| “Ảnh mượt hơn nên phân loại tốt hơn” | Phân biệt chất lượng ảnh với hiệu năng phân loại |
| “Chứng minh hiệu quả khử nhiễu” khi chỉ có so sánh trước–sau | “Ghi nhận thay đổi hiệu năng sau tinh chỉnh trên ảnh khử nhiễu” |

Thống nhất các cặp thuật ngữ: bộ trích xuất đặc trưng/encoder, hợp nhất đặc trưng/fusion, quy trình/pipeline, ảnh chiếu/view, lần chạy/run. Có thể giới thiệu tên tiếng Anh ở lần xuất hiện đầu, sau đó dùng nhất quán một cách gọi trong văn xuôi. Giữ nguyên tên kiến trúc, định danh mã nguồn và tên lần chạy.

Dùng “tập huấn luyện”, “tập xác thực” và “tập kiểm tra” nhất quán. Không dịch cả accuracy và precision thành một cụm “độ chính xác” mà không phân biệt.

Trong nội dung chính, dùng “nghiên cứu”, “luận văn” hoặc “chúng tôi” nhất quán; tránh giọng trao đổi như “em”, “mình”, “cho đẹp”, “chạy thử xem”. Lời cam đoan và lời cảm ơn có thể theo ngôi và mẫu do trường quy định.

## 8. Format đúng gồm ba lớp

1. **Hình thức:** tuân theo mẫu trường về bìa, lề, font, giãn dòng, đánh số, danh mục và tài liệu tham khảo. Dàn ý chung không thay thế quy định của trường.
2. **Cấu trúc:** năm chương có nhiệm vụ riêng, các phần được đặt đúng vị trí và không lặp không cần thiết.
3. **Lập luận:** kết luận xuất phát từ thiết kế và bằng chứng, không từ mong muốn có kết quả đẹp.

Không có một số trang cố định bảo đảm luận văn đạt yêu cầu. Một bản ngắn hơn nhưng lập luận rõ, số liệu có nguồn và giới hạn được xác định chính xác tốt hơn một bản dài với nhiều nội dung trùng lặp.

Đối với hình và bảng:

- Có câu giới thiệu trước khi xuất hiện và đoạn phân tích sau đó.
- Caption đủ giải thích ký hiệu, điều kiện đo và nguồn mà không chứa toàn bộ phần thảo luận.
- Cỡ chữ phải đọc được ở kích thước chèn thực tế; tăng DPI không khắc phục được chữ quá nhỏ.
- Không bôi đậm “kết quả tốt nhất” giữa các phép đo không cùng tập đánh giá hoặc không cùng điều kiện.
- Chú thích rõ ảnh minh họa, xác suất giả định, giá trị ước tính hoặc dữ liệu chưa có; không trình bày chúng như kết quả thực nghiệm.

## 9. Quy trình biên tập thực hành

1. Chốt một đoạn mô tả vấn đề, giải pháp và phạm vi bằng chứng của luận văn.
2. Gán cho mỗi mục một câu hỏi cụ thể cần trả lời.
3. Phân loại các đoạn hiện có thành nền tảng, phương pháp, kết quả, thảo luận hoặc chi tiết tái lập.
4. Di chuyển nội dung về đúng chương; chuyển chi tiết dài sang phụ lục.
5. Viết lại từng đoạn theo cấu trúc câu chủ đề, thông tin hỗ trợ và câu kết nối.
6. Gom các cảnh báo trùng lặp về phần phương pháp hoặc hạn chế phù hợp, nhưng giữ những điều kiện cần để hiểu từng kết quả.
7. Rà soát thì và mức độ khẳng định: đã quan sát, dự kiến thực hiện, chưa có dữ liệu, chưa được kiểm chứng.
8. Đối chiếu lại tất cả số liệu, bảng, hình, công thức, trích dẫn và tham chiếu nội bộ.
9. Biên dịch và đọc PDF ở kích thước trang thực tế để kiểm tra ngắt dòng, bảng, hình và độ rõ của chữ.

## 10. Câu hỏi tự kiểm tra trước khi hoàn thiện

- Đoạn này giúp người đọc hiểu hoặc đánh giá nghiên cứu, hay chỉ kể một công việc đã làm?
- Nội dung thuộc đúng chương chưa?
- Nhận xét có bằng chứng đi kèm không?
- Kết quả có đúng tập dữ liệu, epoch, checkpoint và quy trình được nêu không?
- Có đang biến một kế hoạch thành kết quả đã đạt hay không?
- Có suy ra tác động nhân quả hoặc ý nghĩa thống kê vượt quá thiết kế không?
- Có vô tình tạo thêm cam kết huấn luyện ngoài lượt cuối đã chốt không?
- Chi tiết này có thể chuyển sang phụ lục mà không làm đứt mạch lập luận không?
- Caption và đoạn thảo luận có bổ sung ý nghĩa thay vì chỉ lặp số trong bảng không?
- Một người không tham gia quá trình thực hiện có hiểu được lý do thiết kế và giới hạn kết luận không?

> Câu tự kiểm tra cốt lõi: **“Đoạn này giúp người đọc hiểu hoặc đánh giá nghiên cứu, hay chỉ cho biết một công việc đã được thực hiện?”** Nếu chỉ thuộc vế sau, hãy rút gọn, chuyển thành mô tả phương pháp hoặc đưa vào phụ lục.
