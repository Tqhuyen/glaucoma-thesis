# Đánh giá và lựa chọn phương pháp khử nhiễu cho ảnh OCT

## 1. Mục tiêu và phạm vi đánh giá

Khử nhiễu ảnh OCT cần cân bằng giữa giảm speckle và duy trì các đặc trưng có ích cho phân tích cấu trúc võng mạc và phân loại glaucoma. Một ảnh có độ trơn cao hoặc chỉ số SNR lớn không nhất thiết cung cấp thông tin chẩn đoán tốt hơn. Vì vậy, việc lựa chọn phương pháp trong nghiên cứu này dựa trên nhiều chỉ số bổ trợ, thay vì tối ưu riêng một chỉ số.

Cơ sở đánh giá chính là khảo sát trên **14 thể tích OCT Harvard-GF**, mỗi thể tích có kích thước **200 × 200 × 200 voxel**, lưu ở dạng uint8. Các mã thể tích gồm: 2401, 2404, 2469, 2541, 2615, 2683, 2760, 2863, 2967, 3037, 3100, 3162, 3234 và 3294. Sáu phương pháp được đối chiếu gồm Gaussian, Median, Bilateral, Total Variation theo thuật toán Chambolle, Wavelet BayesShrink và Non-Local Means (NLM). Các bộ lọc được áp dụng theo từng B-scan 2D; đây không phải khảo sát các bộ lọc thể tích 3D.

Khảo sát ban đầu trên một thể tích, `data_2404`, được sử dụng làm bằng chứng bổ sung. Kết quả một thể tích không được xem là đủ để suy rộng cho toàn bộ tập dữ liệu. BM3D, DnCNN và SwinIR chưa có kết quả tương ứng trong khảo sát 14 thể tích nên không được xếp hạng trực tiếp cùng sáu phương pháp nói trên.

Các thể tích khảo sát được lấy từ dữ liệu có sẵn trong bộ nhớ đệm cục bộ. Chưa có bằng chứng xác nhận tập con này cân bằng nhãn hoặc đại diện cho toàn bộ quần thể nghiên cứu. Số thể tích không được diễn giải thành số bệnh nhân độc lập khi chưa kiểm tra định danh bệnh nhân.

## 2. Chỉ số và cách diễn giải

Gọi $\mu_s$, $\sigma_s$ lần lượt là trung bình và độ lệch chuẩn cường độ của vùng tín hiệu; $\mu_b$, $\sigma_b$ là các đại lượng tương ứng của vùng nền. Trong khảo sát nhiều thể tích, ngưỡng Otsu được xác định từ thể tích gốc và mask được giữ cố định khi đánh giá các phương pháp trên cùng thể tích.

| Chỉ số | Công thức sử dụng | Ý nghĩa và lưu ý |
|---|---|---|
| Độ lệch chuẩn nền | $\sigma_b$ | Giá trị thấp biểu thị nền ít biến thiên hơn; không tự chứng minh cấu trúc được giữ nguyên. |
| SNR nền (`SNR_bg`) | $\mu_s/\sigma_b$ | So sánh mức tín hiệu với biến thiên nền. |
| ENL nền (`ENL_bg`) | $(\mu_b/\sigma_b)^2$ | Chỉ số độ đồng nhất tương đối của nền; có thể tăng do thay đổi trung bình nền. |
| CNR nền (`CNR_bg`) | $(\mu_s-\mu_b)/\sigma_b$ | Độ phân biệt tín hiệu và nền, chuẩn hóa bởi biến thiên nền. |
| SNR | $\mu_s/\sigma_s$ | Có thể tăng khi làm phẳng cả texture giải phẫu trong vùng tín hiệu. |
| ENL | $(\mu_s/\sigma_s)^2$ | Bằng bình phương SNR, không phải bằng chứng độc lập với SNR. |
| CNR | $(\mu_s-\mu_b)/\sqrt{\sigma_s^2+\sigma_b^2}$ | Sử dụng cả vùng tín hiệu và vùng nền. |
| Tương quan gradient $\beta$ | $\mathrm{corr}(|\nabla I_{\mathrm{gốc}}|,|\nabla I_{\mathrm{lọc}}|)$ trên các vị trí gradient mạnh của ảnh gốc | Đo mức tương đồng gradient với ảnh còn nhiễu, không đo trực tiếp tỷ lệ bảo toàn cấu trúc giải phẫu. |

Các tỷ số SNR và CNR trong báo cáo là tỷ số tuyến tính, **không phải dB**. ENL không được diễn giải như số lần quan sát độc lập thực sự khi chưa kiểm chứng giả định vùng đồng nhất và mô hình nhiễu phù hợp. Mask Otsu chỉ chia vùng sáng/tối; không đồng nhất với phân đoạn lớp sợi thần kinh võng mạc (RNFL) hoặc nền không khí thuần nhất.

Đối với $\beta$, các vị trí đánh giá được chọn bằng ngưỡng percentile 85 của độ lớn gradient ảnh gốc. Giá trị cao có thể phản ánh bảo toàn cấu trúc, nhưng cũng có thể phản ánh giữ lại speckle. Tương quan còn có thể cao khi biên độ gradient đã giảm. Do đó, $\beta=0.90$ **không có nghĩa bảo toàn 90% RNFL hoặc 90% biên giải phẫu**. Ảnh gốc có $\beta=1$ do tự so sánh với chính nó, không phải vì ảnh gốc có chất lượng tối ưu.

## 3. Kết quả trên 14 thể tích

### 3.1. Chỉ số ở mức toàn thể tích

Mỗi chỉ số được tính ở mức toàn thể tích, sau đó lấy trung bình và độ lệch chuẩn qua 14 thể tích. Bảng trình bày **trung bình ± độ lệch chuẩn**. Độ lệch chuẩn phản ánh độ phân tán mô tả, không phải khoảng tin cậy hoặc kết quả kiểm định thống kê.

| Phương pháp | Độ lệch chuẩn nền | SNR_bg | ENL_bg | CNR_bg | β |
|---|---:|---:|---:|---:|---:|
| Ảnh gốc | 11.941 ± 0.515 | 7.242 ± 0.233 | 5.063 ± 0.767 | 4.997 ± 0.298 | 1.000 ± 0.000 |
| Gaussian | 10.458 ± 0.831 | 7.245 ± 0.252 | 7.484 ± 1.854 | 4.526 ± 0.349 | 0.515 ± 0.036 |
| Median | 11.055 ± 0.748 | 7.116 ± 0.267 | 6.544 ± 1.396 | 4.570 ± 0.391 | 0.520 ± 0.044 |
| **Bilateral** | **8.817 ± 0.696** | **8.737 ± 0.433** | **9.036 ± 1.997** | **5.746 ± 0.518** | **0.858 ± 0.026** |
| TV Chambolle | 9.945 ± 0.804 | 7.617 ± 0.292 | 8.293 ± 2.104 | 4.756 ± 0.389 | 0.598 ± 0.033 |
| **Wavelet BayesShrink** | **9.993 ± 0.621** | **8.316 ± 0.304** | **7.311 ± 1.346** | **5.621 ± 0.402** | **0.901 ± 0.022** |
| NLM | 9.917 ± 0.783 | 7.679 ± 0.294 | 8.234 ± 2.041 | 4.827 ± 0.399 | 0.606 ± 0.033 |

Nguồn: [bảng khảo sát toàn thể tích](../figures/denoise/wholevolume/summary.md), đối chiếu với [CSV tổng hợp](../figures/denoise/wholevolume/summary_metrics.csv). Các hàng in đậm nhấn mạnh hai ứng viên ưu tiên, không hàm ý mọi giá trị trong hàng đều tối ưu.

### 3.2. Đối chiếu bằng trung bình theo lát cắt

Để kiểm tra ảnh hưởng của cách tổng hợp, các chỉ số còn được tính theo lát cắt, lấy trung bình trong mỗi thể tích, rồi lấy trung bình và độ lệch chuẩn qua 14 thể tích. Bảng sau tập trung vào các ứng viên lựa chọn và NLM để làm rõ sự cạnh tranh ở vị trí thứ ba.

| Phương pháp | Độ lệch chuẩn nền | SNR_bg | ENL_bg | CNR_bg | β |
|---|---:|---:|---:|---:|---:|
| Ảnh gốc | 11.433 ± 0.439 | 6.753 ± 0.277 | 5.956 ± 0.754 | 4.333 ± 0.362 | 1.000 ± 0.000 |
| Bilateral | 6.677 ± 0.835 | 9.934 ± 0.766 | 27.831 ± 7.307 | 5.017 ± 0.918 | 0.695 ± 0.028 |
| Wavelet BayesShrink | 9.107 ± 0.557 | 7.779 ± 0.416 | 10.091 ± 1.447 | 4.644 ± 0.595 | 0.749 ± 0.047 |
| TV Chambolle | 7.321 ± 1.167 | 11.033 ± 1.600 | 69.352 ± 32.419 | 4.337 ± 0.989 | 0.368 ± 0.062 |
| NLM | 7.534 ± 1.062 | 10.045 ± 1.157 | 47.667 ± 19.011 | 4.198 ± 0.927 | 0.363 ± 0.061 |

TV có SNR_bg và ENL_bg trung bình theo lát cắt cao nhất trong sáu phương pháp khảo sát, trong khi Bilateral có độ lệch chuẩn nền thấp nhất và CNR_bg cao nhất. Wavelet duy trì $\beta$ cao nhất. Điều này cho thấy không tồn tại một phương pháp đồng thời dẫn đầu tất cả tiêu chí.

Thứ hạng còn nhạy với việc dùng trung bình hay trung vị: trung vị CNR_bg theo lát cắt của Wavelet là 4.287, cao hơn Bilateral là 4.084. ENL_bg của TV có trung bình theo lát cắt là 69.352 nhưng trung vị là 27.616, cho thấy không nên diễn giải chỉ từ giá trị trung bình. Các lát cắt trong cùng thể tích không được xem là các mẫu thống kê độc lập.

## 4. Ba phương pháp được ưu tiên

Thứ tự dưới đây là **lựa chọn đa tiêu chí phục vụ thiết kế thí nghiệm tiếp theo**, không phải kết quả của một hàm điểm tổng hợp đã được định nghĩa trước hoặc một kiểm định xác nhận thứ hạng.

### 4.1. Ưu tiên thứ nhất: Bilateral

Bilateral cho sự cân bằng thuận lợi nhất giữa giảm biến thiên nền, tăng tương phản và duy trì tương quan gradient với ảnh gốc. Ở mức toàn thể tích, độ lệch chuẩn nền giảm từ 11.941 xuống 8.817, tương ứng khoảng **26.2%**, trong khi CNR_bg tăng từ 4.997 lên 5.746. Giá trị $\beta=0.858$ thấp hơn Wavelet nhưng cao hơn rõ rệt các phương pháp làm mịn mạnh như TV, NLM, Gaussian và Median trong bảng số liệu.

Kết quả theo trung bình lát cắt cũng ủng hộ Bilateral: phương pháp có độ lệch chuẩn nền thấp nhất và CNR_bg cao nhất. Tuy nhiên, Bilateral không dẫn đầu mọi chỉ số và chưa được chứng minh có ưu thế có ý nghĩa thống kê so với Wavelet. Chi phí tính toán cũng cao hơn một số bộ lọc đơn giản.

Vì vậy, **Bilateral được chọn làm ứng viên tiền xử lý chính**, với điều kiện lợi ích cuối cùng cần được kiểm tra bằng thí nghiệm phân loại có kiểm soát.

### 4.2. Ưu tiên thứ hai: Wavelet BayesShrink

Wavelet có tương quan gradient cao nhất trong sáu phương pháp được khảo sát, với $\beta=0.901$ ở mức toàn thể tích và $\beta=0.749$ theo trung bình lát cắt. CNR_bg toàn thể tích đạt 5.621, đứng sau Bilateral. Kết quả này cho thấy Wavelet là phương án đáng quan tâm khi cần hạn chế can thiệp mạnh lên các biến thiên cường độ của ảnh gốc.

Đổi lại, mức giảm biến thiên nền nhỏ hơn Bilateral và ảnh có thể còn nhiều speckle hơn. Không thể chỉ từ $\beta$ cao mà kết luận Wavelet bảo toàn RNFL tốt nhất, vì chỉ số này cũng nhạy với nhiễu còn giữ lại.

**Wavelet được chọn làm ứng viên bảo thủ và chi phí thấp**, phù hợp để đối chiếu với Bilateral trong thí nghiệm downstream.

### 4.3. Ưu tiên thứ ba có điều kiện: TV Chambolle

TV có khả năng làm mịn mạnh. Theo trung bình lát cắt, TV đạt SNR_bg 11.033 và ENL_bg 69.352; độ lệch chuẩn nền, SNR_bg và CNR_bg đều thuận lợi hơn NLM, trong khi $\beta$ của hai phương pháp gần tương đương.

Tuy nhiên, **ưu thế của TV so với NLM phụ thuộc cách tổng hợp**. Ở mức toàn thể tích, NLM lại có độ lệch chuẩn nền thấp hơn, SNR_bg và CNR_bg cao hơn, đồng thời $\beta$ nhỉnh hơn TV. CNR_bg toàn thể tích của cả hai phương pháp còn thấp hơn ảnh gốc, và tương quan gradient thấp hơn đáng kể Bilateral và Wavelet.

Do đó, **TV được chọn làm đối chứng làm mịn mạnh**, không phải phương pháp đứng thứ ba một cách chắc chắn về chất lượng chẩn đoán. Nếu chỉ ưu tiên các chỉ số toàn thể tích, việc lựa chọn NLM thay TV ở vị trí thứ ba cũng có cơ sở. NLM cần được giữ làm đối chứng khi kiểm chứng lựa chọn này.

### 4.4. Tóm tắt quyết định

| Mức ưu tiên | Phương pháp | Vai trò đề xuất | Giới hạn chính |
|---|---|---|---|
| 1 | Bilateral | Ứng viên cân bằng giữa giảm nhiễu và duy trì cấu trúc biểu hiện qua các chỉ số gián tiếp | Chưa chứng minh cải thiện phân loại; chi phí cao hơn bộ lọc đơn giản. |
| 2 | Wavelet BayesShrink | Ứng viên bảo thủ, nhanh, có tương quan gradient cao | Có thể giữ lại speckle; β không xác nhận bảo toàn RNFL. |
| 3, có điều kiện | TV Chambolle | Đối chứng làm mịn mạnh | Cạnh tranh sát với NLM; thứ hạng thay đổi theo cách gộp chỉ số. |

## 5. Đối chiếu với khảo sát ban đầu và các phương pháp còn lại

Trong bảng một thể tích `2404`, TV có SNR và ENL cao nhất, nhưng **NLM mới có CNR cao nhất**: 4.130 so với 4.048 của TV. Vì vậy, nhận định TV dẫn đầu đồng thời SNR, ENL và CNR trong tài liệu cũ cần được điều chỉnh. SNR và ENL cũng không được tính như hai bằng chứng độc lập do quan hệ ENL = SNR².

Một hạn chế khác của phép đo ban đầu là band quanh đỉnh sáng và các pixel thuộc vùng tín hiệu/nền được xác định lại trên ảnh sau lọc. Chỉ ngưỡng cường độ được lấy từ ảnh gốc và giữ chung. Do đó, không nên mô tả quy trình cũ là sử dụng hoàn toàn cùng vùng đo giữa mọi phương pháp. Band quanh đỉnh sáng cũng chưa phải vùng RNFL được xác nhận giải phẫu.

BM3D có tương quan gradient cao trên `2404`, nhưng chưa có khảo sát 14 thể tích tương ứng. Thời gian được ghi nhận ở phép đo cũ khoảng 686.8 giây/thể tích, lớn hơn nhiều các phương pháp đơn giản. DnCNN và SwinIR có kết quả thử nghiệm một thể tích trong notebook GPU, nhưng chưa đủ bằng chứng nhiều thể tích theo cùng quy trình để đưa vào danh sách ưu tiên trên. Điều này không chứng minh các phương pháp đó kém trong mọi cấu hình hoặc sau khi huấn luyện phù hợp với OCT.

Gaussian và Median vẫn có giá trị làm baseline đơn giản. Tuy nhiên, trong khảo sát hiện có, sự cân bằng giữa tương phản nền và tương quan gradient của hai phương pháp này kém thuận lợi hơn Bilateral và Wavelet.

Các số liệu thời gian giữa những lần chạy khác nhau không phải benchmark thống nhất do khác mức song song và điều kiện thực thi. Trường thời gian bằng 0 khi sử dụng cache không có nghĩa thuật toán khử nhiễu tức thời. Vì vậy, thời gian không được dùng làm căn cứ quyết định thứ hạng chất lượng trong báo cáo này.

## 6. Giới hạn và yêu cầu kiểm chứng

- **Không có ảnh sạch tham chiếu:** các chỉ số đang dùng là chỉ số gián tiếp; ảnh gốc còn nhiễu không phải chuẩn chất lượng tuyệt đối.
- **Vùng đo chưa được xác nhận giải phẫu:** mask Otsu có thể chứa mô tối hoặc bóng đổ trong vùng được gọi là nền; band quanh đỉnh sáng không đồng nhất với RNFL.
- **β giữa hai quy trình không đồng nhất:** khảo sát ban đầu đo gradient trong band chứa chiều depth; khảo sát toàn thể tích lấy gradient trên các mặt phẳng vuông góc với trục depth, không trực tiếp đo thành phần gradient theo depth. Không so trực tiếp trị số β giữa hai bảng như cùng một phép đo.
- **Tổng hợp có thể làm đổi thứ hạng:** β toàn thể tích không phải trung bình β theo lát; các tỷ số cũng nhạy với các lát có độ lệch chuẩn nhỏ. Code có xử lý trường hợp suy biến bằng giá trị 0, cần rà soát ảnh hưởng trước phân tích thống kê chính thức.
- **Chưa có kiểm định khác biệt:** trung bình và độ lệch chuẩn không chứng minh khác biệt có ý nghĩa thống kê. Độ lệch chuẩn tổng hợp hiện dùng quy ước `ddof=0`.
- **Phạm vi cấu hình còn hạn chế:** kết luận áp dụng cho các tham số và dữ liệu đã khảo sát, không phải mọi cách điều chỉnh từng thuật toán. Cache cần được xác nhận nguồn gốc và tham số khi tái lập thí nghiệm chính thức.
- **Chưa có bằng chứng downstream có kiểm soát:** không thể kết luận Bilateral, Wavelet hoặc TV cải thiện AUROC, độ nhạy hoặc độ chính xác phân loại chỉ từ bảng khử nhiễu.

Để xác nhận lựa chọn, cần so sánh ảnh gốc, Bilateral, Wavelet và TV, đồng thời giữ NLM làm đối chứng, trên cùng cách chia dữ liệu theo bệnh nhân, cùng kiến trúc, độ phân giải, ngân sách huấn luyện và nhiều seed. Tham số tiền xử lý phải được lựa chọn trên tập huấn luyện/validation; tập test được giữ độc lập cho đánh giá cuối cùng. Có thể bổ sung đánh giá mù của chuyên gia hoặc chỉ số cấu trúc trên ROI/segmentation đã xác nhận, nếu có dữ liệu phù hợp.

## 7. Đoạn kết luận đề xuất đưa vào luận văn

> Trên khảo sát 14 thể tích OCT thuộc bộ dữ liệu Harvard-GF, Bilateral cho sự cân bằng thuận lợi nhất giữa giảm biến thiên nền, tăng tương phản và duy trì tương quan gradient với ảnh gốc. Wavelet BayesShrink có tương quan gradient cao nhất trong sáu phương pháp cổ điển được khảo sát và là phương án đối chiếu ít can thiệp mạnh lên ảnh. TV Chambolle được lựa chọn làm phương pháp đối chứng làm mịn mạnh nhờ các chỉ số trung bình theo lát cắt thuận lợi; tuy nhiên, ưu thế so với Non-Local Means phụ thuộc cách tổng hợp chỉ số. Trên cơ sở đó, nghiên cứu ưu tiên Bilateral, Wavelet BayesShrink và TV Chambolle cho các thí nghiệm tiếp theo, đồng thời giữ Non-Local Means làm đối chứng. Thứ tự này là lựa chọn đa tiêu chí trong phạm vi cấu hình đã khảo sát, chưa phải bằng chứng về bảo toàn RNFL hoặc cải thiện hiệu quả chẩn đoán glaucoma. Các kết luận cần được xác nhận bằng đánh giá cấu trúc giải phẫu và thí nghiệm phân loại có kiểm soát.

## 8. Nguồn số liệu và mã đánh giá

Các liên kết dưới đây phục vụ truy vết kết quả nội bộ, không thay thế tài liệu tham khảo học thuật về từng thuật toán.

1. [Báo cáo khảo sát 14 thể tích](../figures/denoise/wholevolume/summary.md).
2. [CSV chỉ số tổng hợp](../figures/denoise/wholevolume/summary_metrics.csv).
3. [CSV chỉ số từng thể tích](../figures/denoise/wholevolume/per_volume_metrics.csv).
4. [Mã đánh giá toàn thể tích](../scripts/denoise_wholevolume_survey.py).
5. [CSV khảo sát ban đầu trên thể tích 2404](../figures/denoise/denoise_metrics_2404.csv).
6. [Mã khử nhiễu và đánh giá band-based ban đầu](../scripts/compare_denoise_methods.py).
7. [Notebook so sánh bổ sung trên GPU](../notebooks/3d_glaucoma_denoise_gpu_compare.ipynb).
8. [Bộ hình denoise dành cho luận văn](../figures/denoise/thesis/README.md).

Tài liệu này tổng hợp và rà soát các kết quả đã có; không báo cáo một lần chạy khử nhiễu, đánh giá số hoặc huấn luyện mới.
