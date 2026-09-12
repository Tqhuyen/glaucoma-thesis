# 4.X. So sánh ảnh thô, ảnh khử nhiễu bilateral và ảnh thô ở độ phân giải 3D thấp

> **Ghi chú về nguồn số liệu:** Nội dung được biên tập từ bảng kết quả do tác giả cung cấp, chưa đối chiếu trực tiếp với tệp dự đoán hoặc nhật ký thí nghiệm gốc. Các giá trị trong bảng được giữ theo số liệu cung cấp; chênh lệch được tính từ các giá trị đã làm tròn. Mục này không bổ sung kết quả huấn luyện mới. Ký hiệu `4.X` cần được thay bằng số mục tương ứng khi tích hợp vào luận văn.

## 4.X.1. Mục tiêu và thiết lập so sánh

Mục tiêu của phép so sánh là khảo sát hiệu quả phân loại bệnh tăng nhãn áp từ ảnh cắt lớp quang học kết hợp (OCT) khi thay đổi phương án tiền xử lý và độ phân giải đầu vào của nhánh ba chiều. Hai hướng khảo sát gồm khử nhiễu bằng bộ lọc bilateral và giảm kích thước đầu vào 3D từ 200 × 200 × 200 xuống 96 × 96 × 96 điểm ảnh thể tích. Ba cấu hình được trình bày trong Bảng 4.X.1.

**Bảng 4.X.1. Các cấu hình được đưa vào so sánh**

| Cấu hình | Dữ liệu đầu vào | Kích thước đầu vào nhánh 3D | Đặc điểm huấn luyện |
|---|---|---|---|
| Raw 200 | Ảnh thô, không khử nhiễu | 200 × 200 × 200 | Cấu hình tham chiếu |
| Bilateral 200 | Ảnh đã khử nhiễu bằng bộ lọc bilateral | 200 × 200 × 200 | Khởi tạo từ trọng số Raw 200 và tiếp tục tinh chỉnh trên ảnh khử nhiễu |
| Raw 96 | Ảnh thô, không khử nhiễu | 96 × 96 × 96 | Cấu hình đầu vào 3D độ phân giải thấp; thiết lập huấn luyện khác Raw 200 |

Các tên Raw 200, Bilateral 200 và Raw 96 được giữ thống nhất để đối chiếu với kết quả thí nghiệm. Giá trị 200 hoặc 96 biểu thị kích thước đầu vào của nhánh 3D, không mặc nhiên biểu thị độ phân giải lưu trữ hoặc kích thước đầu vào của các nhánh 2D. Vì vậy, không nên diễn giải Raw 96 là cấu hình giảm đồng thời độ phân giải của mọi nhánh mô hình.

Cả ba cấu hình sử dụng hạt giống ngẫu nhiên bằng 42. Tập kiểm thử gồm 900 mẫu, trong đó có 489 mẫu dương tính và 411 mẫu âm tính. Đơn vị được báo cáo là mẫu ảnh OCT; số mẫu không được diễn giải thành số bệnh nhân nếu chưa xác nhận quan hệ giữa ảnh, mắt và người tham gia. Các giai đoạn kiểm tra dữ liệu, phân tích, đánh giá và trực quan hóa của cùng một mô hình chỉ là những bước xử lý kết quả, không được tính thành các lần huấn luyện độc lập.

Đây là phép so sánh giữa ba cấu hình thực nghiệm, chưa phải thiết kế đối chứng tách biệt hoàn toàn từng yếu tố. Đặc biệt, Bilateral 200 kế thừa trọng số Raw 200 và có thêm giai đoạn tinh chỉnh; Raw 96 khác Raw 200 về tốc độ học, số vòng huấn luyện tối đa và cơ chế dừng sớm. Do đó, kết quả cho phép mô tả khác biệt hiệu năng quan sát được, nhưng chưa đủ để xác lập tác động nhân quả riêng của khử nhiễu hoặc giảm độ phân giải.

## 4.X.2. Kết quả phân loại trên tập kiểm thử

**Bảng 4.X.2. Hiệu quả phân loại của ba cấu hình trên 900 mẫu kiểm thử**

| Chỉ số | Raw 200 | Bilateral 200 | Raw 96 |
|---|---:|---:|---:|
| Accuracy (%) | **76,67** | 76,44 | 72,56 |
| Balanced Accuracy (%) | 76,92 | **77,31** | 73,56 |
| AUC-ROC (%) | **84,76** | 84,72 | 79,78 |
| AUC-PR (%) | 87,82 | **88,18** | 84,85 |
| Precision (%) | 81,35 | **86,35** | 83,24 |
| Sensitivity/Recall (%) | **74,03** | 67,28 | 61,96 |
| Specificity (%) | 79,81 | **87,35** | 85,16 |
| F1-score (%) | **77,52** | 75,63 | 71,04 |
| MCC | 0,5363 | **0,5507** | 0,4783 |
| ECE (%), thấp hơn tốt hơn | **2,29** | 6,74 | 9,91 |
| Test loss, thấp hơn tốt hơn | **0,4874** | 0,4910* | 0,5770 |

*Test loss của Bilateral 200 được lấy từ giai đoạn phân tích của cùng mô hình vì giai đoạn đánh giá độc lập không ghi lại chỉ số này. Khả năng so sánh trực tiếp giá trị đó phụ thuộc vào việc hai giai đoạn sử dụng cùng trọng số, tập mẫu, định nghĩa hàm mất mát, trọng số lớp và trạng thái hiệu chỉnh xác suất. Khi chưa xác nhận các điều kiện này, giá trị được xem là thông tin tham khảo.*

MCC được trình bày dưới dạng hệ số trên thang từ −1 đến 1 thay vì phần trăm; việc đổi cách biểu diễn không làm thay đổi số liệu. Các giá trị in đậm là giá trị tốt nhất quan sát được trong từng hàng, không hàm ý khác biệt có ý nghĩa thống kê.

Raw 200 và Bilateral 200 có AUC-ROC lần lượt là 84,76% và 84,72%, chênh lệch 0,04 điểm phần trăm. AUC-PR của Bilateral 200 cao hơn Raw 200 0,36 điểm phần trăm. Như vậy, hai cấu hình có giá trị ước lượng về khả năng phân biệt hai lớp rất gần nhau trong lần đánh giá này. Tuy nhiên, sự gần nhau của các ước lượng không chứng minh tính tương đương thống kê, cũng không đủ để khẳng định bộ lọc bilateral cải thiện hoặc không cải thiện khả năng biểu diễn của mô hình.

Raw 96 đạt AUC-ROC 79,78% và AUC-PR 84,85%, thấp hơn Raw 200 lần lượt 4,98 và 2,97 điểm phần trăm. Sự suy giảm xuất hiện ở cả hai thước đo không phụ thuộc vào một ngưỡng phân loại duy nhất, cho thấy khác biệt giữa hai cấu hình không chỉ thể hiện ở điểm vận hành đang được báo cáo.

## 4.X.3. Phân tích ma trận nhầm lẫn

**Bảng 4.X.3. Ma trận nhầm lẫn tại ngưỡng phân loại được báo cáo của từng cấu hình**

| Cấu hình | TP | TN | FP | FN |
|---|---:|---:|---:|---:|
| Raw 200 | **362** | 328 | 83 | **127** |
| Bilateral 200 | 329 | **359** | **52** | 160 |
| Raw 96 | 303 | 350 | 61 | 186 |

Các hàng đều thỏa mãn TP + FN = 489 và TN + FP = 411. Tổng số mẫu được phân loại đúng lần lượt là 690, 688 và 653, tương ứng với Accuracy 76,67%, 76,44% và 72,56% trong Bảng 4.X.2.

Raw 200 nhận diện đúng 362 mẫu dương tính, nhiều hơn Bilateral 200 33 mẫu và nhiều hơn Raw 96 59 mẫu. Đồng thời, FN của Raw 200 là 127, thấp nhất trong ba cấu hình. Tại các ngưỡng đang xét, đây là ưu điểm nếu mục tiêu đánh giá ưu tiên hạn chế bỏ sót mẫu có bệnh.

Bilateral 200 giảm FP từ 83 xuống 52 so với Raw 200, nhưng FN tăng từ 127 lên 160. Tổng số dự đoán dương tính giảm từ 445 xuống 381. Do đó, cấu hình này thể hiện điểm vận hành thiên về Specificity: giảm cảnh báo sai trên nhóm âm tính nhưng đồng thời nhận diện ít mẫu dương tính hơn. Đây là mô tả về hành vi dự đoán tại ngưỡng đã dùng, không phải bằng chứng rằng riêng bộ lọc bilateral làm mô hình trở nên “thận trọng” hơn.

Raw 96 có FP là 61 và FN là 186. Mặc dù Specificity đạt 85,16%, cấu hình chỉ phát hiện đúng 303/489 mẫu dương tính, tương ứng Sensitivity 61,96%. So với hai cấu hình còn lại, Raw 96 có bất lợi rõ về số mẫu dương tính được nhận diện tại điểm vận hành hiện tại.

## 4.X.4. So sánh Raw 200 và Bilateral 200

**Bảng 4.X.4. Chênh lệch của Bilateral 200 so với Raw 200**

| Chỉ số | Chênh lệch |
|---|---:|
| Accuracy | −0,23 điểm phần trăm |
| Balanced Accuracy | +0,39 điểm phần trăm |
| AUC-ROC | −0,04 điểm phần trăm |
| AUC-PR | +0,36 điểm phần trăm |
| Precision | +5,00 điểm phần trăm |
| Sensitivity/Recall | −6,75 điểm phần trăm |
| Specificity | +7,54 điểm phần trăm |
| F1-score | −1,89 điểm phần trăm |
| MCC | +0,0144 |
| ECE | +4,45 điểm phần trăm |

Kết quả cho thấy đánh đổi giữa Sensitivity và Specificity nổi bật hơn thay đổi về AUC. Bilateral 200 có Precision và Specificity cao hơn, nhưng Sensitivity và F1-score thấp hơn Raw 200. Balanced Accuracy tăng nhẹ vì mức tăng Specificity lớn hơn mức giảm Sensitivity khi hai đại lượng được lấy trung bình với trọng số bằng nhau. Trong khi đó, Accuracy chung giảm nhẹ, tương ứng với hai mẫu phân loại đúng ít hơn.

Việc AUC-ROC gần nhau nhưng Sensitivity và Specificity khác nhau gợi ý cần xem xét ngưỡng phân loại, phân bố điểm dự đoán và cách hiệu chỉnh xác suất. Tuy nhiên, chỉ từ một giá trị AUC tổng hợp không thể kết luận ngưỡng là nguyên nhân chính: hai đường cong ROC có thể có cùng diện tích nhưng khác hình dạng tại vùng vận hành quan tâm. Để so sánh công bằng hơn, có thể đánh giá Specificity tại cùng một mức Sensitivity hoặc Sensitivity tại cùng một mức Specificity, với quy tắc lựa chọn ngưỡng được xác định trên tập xác thực.

Về cơ chế, bộ lọc bilateral có thể làm giảm nhiễu cục bộ đồng thời bảo toàn một phần biên cấu trúc. Ngược lại, tùy tham số lọc, một số chi tiết nhỏ cũng có thể bị làm trơn. Đây chỉ là các giả thuyết giải thích; số liệu phân loại hiện tại không xác định được liệu cấu trúc liên quan đến bệnh đã được bảo toàn hay suy giảm. Hơn nữa, Bilateral 200 có thêm giai đoạn tinh chỉnh từ Raw 200, nên ảnh hưởng của tiền xử lý chưa được tách khỏi ảnh hưởng của quá trình cập nhật trọng số bổ sung.

ECE tăng từ 2,29% lên 6,74%, cho thấy sai lệch lớn hơn giữa xác suất dự đoán và tần suất đúng quan sát được, nếu hai giá trị được tính bằng cùng quy trình. Precision cao hơn không đồng nghĩa với xác suất dự đoán được hiệu chỉnh tốt hơn. Trước khi đề xuất hiệu chỉnh bổ sung bằng phương pháp điều chỉnh nhiệt độ, cần xác nhận các chỉ số đang được báo cáo trước hay sau hiệu chỉnh và có cùng cách chia khoảng xác suất hay không. Nếu thực hiện hiệu chỉnh, tham số chỉ được ước lượng trên tập xác thực; tập kiểm thử không được dùng để lựa chọn tham số.

## 4.X.5. So sánh Raw 200 và Raw 96

**Bảng 4.X.5. Chênh lệch của Raw 96 so với Raw 200**

| Chỉ số | Chênh lệch |
|---|---:|
| Accuracy | −4,11 điểm phần trăm |
| Balanced Accuracy | −3,36 điểm phần trăm |
| AUC-ROC | −4,98 điểm phần trăm |
| AUC-PR | −2,97 điểm phần trăm |
| Precision | +1,89 điểm phần trăm |
| Sensitivity/Recall | −12,07 điểm phần trăm |
| Specificity | +5,35 điểm phần trăm |
| F1-score | −6,48 điểm phần trăm |
| MCC | −0,0580 |
| ECE | +7,62 điểm phần trăm |

So với Raw 200, Raw 96 suy giảm đồng thời về Accuracy, AUC-ROC, AUC-PR, F1-score và MCC. Mức giảm Sensitivity 12,07 điểm phần trăm tương ứng với 59 mẫu dương tính được nhận diện ít hơn. ECE tăng lên 9,91%, cao nhất trong ba cấu hình, cho thấy chất lượng hiệu chỉnh xác suất cũng kém hơn theo cách tính được báo cáo.

Giảm độ phân giải có thể làm giảm khả năng biểu diễn những cấu trúc nhỏ hoặc chi tiết không gian ba chiều có liên quan đến bệnh. Tuy nhiên, đây chưa phải kết luận nhân quả từ thiết kế hiện tại. Raw 96 sử dụng tốc độ học 10⁻⁴, trong khi Raw 200 sử dụng 5 × 10⁻⁵; số vòng huấn luyện tối đa và cơ chế dừng sớm cũng khác nhau. Những yếu tố này có thể ảnh hưởng đến quá trình hội tụ và điểm vận hành cuối cùng.

Ngoài ra, chưa có số liệu thời gian huấn luyện, thời gian suy luận hoặc bộ nhớ GPU trong bảng so sánh này. Vì vậy, dù đầu vào 3D nhỏ hơn có tiềm năng giảm chi phí tính toán, chưa thể định lượng mức đánh đổi giữa hiệu năng và tài nguyên, cũng chưa đủ cơ sở để loại bỏ Raw 96 trong mọi bối cảnh sử dụng.

## 4.X.6. Thảo luận về ý nghĩa của kết quả

Trong phạm vi một lần huấn luyện cho mỗi cấu hình và tại các ngưỡng đang được báo cáo, Raw 200 có ưu thế về Accuracy, AUC-ROC, F1-score, Sensitivity và ECE. Bilateral 200 có ưu thế về Precision, Specificity, AUC-PR, Balanced Accuracy và MCC. Do đó, không có một cấu hình tốt nhất trên mọi tiêu chí; việc lựa chọn cần gắn với mục tiêu đánh giá cụ thể.

Nếu ưu tiên giảm bỏ sót mẫu có bệnh, Raw 200 là cấu hình tham chiếu hợp lý trong ba cấu hình được khảo sát. Tuy vậy, Sensitivity 74,03% vẫn tương ứng với 127/489 mẫu dương tính bị bỏ sót. Kết quả này chỉ hỗ trợ lựa chọn cấu hình cho nghiên cứu hiện tại, không đủ để khẳng định mô hình đáp ứng yêu cầu sàng lọc lâm sàng hoặc có thể triển khai độc lập.

Bilateral 200 có thể đáng quan tâm khi tiêu chí đánh giá nhấn mạnh hạn chế FP. Tuy nhiên, không thể từ đó kết luận cấu hình phù hợp cho giai đoạn xác nhận chẩn đoán. Nhận định về vai trò lâm sàng cần thêm bằng chứng về quần thể đích, tỷ lệ hiện mắc bệnh, chi phí sai phân loại và khả năng khái quát hóa trên dữ liệu độc lập. Đặc biệt, Precision phụ thuộc vào tỷ lệ dương tính; giá trị đo trên tập kiểm thử có 54,33% mẫu dương tính không tự động chuyển sang quần thể sàng lọc có tỷ lệ bệnh khác.

Vì AUC-ROC của Raw 200 và Bilateral 200 gần nhau, một hướng phân tích trên các mô hình đã cố định trọng số là so sánh tại cùng yêu cầu Sensitivity hoặc Specificity. Điều này không bảo đảm Raw 200 sẽ đạt đúng điểm vận hành của Bilateral 200, nhưng giúp đánh giá liệu lợi ích quan sát được có còn tồn tại khi chuẩn hóa tiêu chí lựa chọn ngưỡng. Ngưỡng phải được chọn trên tập xác thực trước khi đánh giá trên tập kiểm thử; không tối ưu ngưỡng theo chính kết quả kiểm thử đang dùng để báo cáo.

## 4.X.7. Hạn chế và điều kiện diễn giải

1. **Một hạt giống ngẫu nhiên.** Mỗi cấu hình mới có kết quả với hạt giống 42, nên chưa đánh giá được biến thiên giữa các lần huấn luyện. Sử dụng cùng hạt giống không bảo đảm ba mô hình có cùng quá trình tối ưu hoặc loại bỏ ảnh hưởng ngẫu nhiên.
2. **Khác biệt về lịch sử huấn luyện.** Bilateral 200 được tinh chỉnh từ Raw 200, không phải mô hình huấn luyện độc lập chỉ khác tiền xử lý. Chưa có cấu hình tiếp tục huấn luyện trên ảnh thô với ngân sách tương ứng để tách riêng tác động của tinh chỉnh bổ sung.
3. **Khác biệt siêu tham số.** Raw 96 khác Raw 200 về tốc độ học, số vòng huấn luyện và dừng sớm, nên chưa thể quy toàn bộ mức suy giảm cho độ phân giải 3D.
4. **Thiếu thông tin điểm vận hành.** Nội dung cung cấp chưa nêu cụ thể ngưỡng của từng cấu hình, quy tắc chọn ngưỡng và trạng thái hiệu chỉnh xác suất. Các yếu tố này cần được xác nhận khi đối chiếu số liệu gốc.
5. **Chưa có độ bất định của chênh lệch.** Bảng hiện tại chưa cung cấp khoảng tin cậy hoặc kiểm định thống kê cho khác biệt giữa các mô hình. Khi có dự đoán theo từng mẫu, có thể ước lượng khoảng tin cậy ghép cặp; nếu nhiều ảnh thuộc cùng người hoặc cùng mắt, cách lấy mẫu lại cần bảo toàn cấu trúc phụ thuộc này.
6. **Giới hạn diễn giải ECE và Test loss.** ECE phụ thuộc cách chia khoảng xác suất và trạng thái hiệu chỉnh. Test loss của Bilateral 200 được lấy từ một giai đoạn xử lý khác, nên phải kiểm tra tính đồng nhất trước khi dùng làm bằng chứng so sánh trực tiếp.
7. **Chưa đánh giá đầy đủ tính ứng dụng.** Chưa có đối chiếu chi phí tính toán, đánh giá trên quần thể độc lập hoặc tiêu chí chấp nhận lâm sàng. Hiệu năng trên 900 mẫu kiểm thử không thay thế được xác nhận khả năng triển khai thực tế.

Các giới hạn này không phủ nhận giá trị mô tả của kết quả, nhưng giới hạn mức độ khái quát hóa và độ chắc chắn của kết luận. Những nghiên cứu nhiều hạt giống hoặc đối chứng đồng nhất siêu tham số chỉ là hướng kiểm chứng trong tương lai nếu được phê duyệt, không phải các thí nghiệm đã thực hiện hay yêu cầu mở rộng phạm vi hiện tại.

## 4.X.8. Kết luận

Trong phép so sánh hiện tại, Raw 200 đạt AUC-ROC 84,76%, F1-score 77,52%, Sensitivity 74,03% và ECE 2,29%. Đây là cấu hình có lợi thế khi ưu tiên khả năng phát hiện mẫu dương tính và chất lượng hiệu chỉnh xác suất tại các điểm vận hành được báo cáo. Trên cơ sở đó, Raw 200 được chọn làm cấu hình chính cho các phân tích tiếp theo trong phạm vi luận văn, thay vì xem đây là bằng chứng về tính ưu việt trong mọi điều kiện hoặc sự sẵn sàng triển khai lâm sàng.

Bilateral 200 có AUC-ROC gần Raw 200, đồng thời tăng Specificity và Precision nhưng giảm Sensitivity. Kết quả chưa cung cấp bằng chứng đủ mạnh để khẳng định khử nhiễu bilateral cải thiện hiệu quả phân loại tổng thể, cũng chưa tách được tác động của bộ lọc khỏi giai đoạn tinh chỉnh bổ sung. Raw 96 có AUC-ROC, F1-score, MCC và Sensitivity thấp hơn, nhưng chưa thể quy toàn bộ suy giảm này cho việc giảm độ phân giải.

Ưu tiên phân tích tiếp theo là đối chiếu nguồn số liệu, xác nhận ngưỡng và trạng thái hiệu chỉnh, đồng thời đánh giá độ bất định của chênh lệch từ các dự đoán đã lưu nếu có. Các bước này có thể được tiến hành với trọng số mô hình đã cố định, không cần tự động bổ sung các lần huấn luyện mới.
