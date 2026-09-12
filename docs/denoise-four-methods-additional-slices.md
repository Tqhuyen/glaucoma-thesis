# So sánh Bilateral, Wavelet, TV Chambolle và BM3D trên các lát cắt bổ sung

## 1. Phạm vi

Tài liệu bổ sung BM3D vào nhóm Bilateral, Wavelet BayesShrink và TV Chambolle. So sánh trực quan dùng **cùng một thể tích Harvard-GF `2404`**, vì cache của thể tích này có đủ bốn phương pháp. Sáu lát cắt bổ sung không phải sáu bệnh nhân hay sáu thể tích độc lập.

Các hình được render lại từ dữ liệu raw và denoised đã lưu, không chạy lại bộ lọc, không sử dụng GPU và không huấn luyện thêm. Các số liệu định lượng bên dưới được trích nguyên từ kết quả cũ; không phải metric mới tính riêng trên sáu lát cắt vừa xuất.

## 2. Bảng định lượng trên volume 2404

| Phương pháp | SNR | ENL | CNR | β | Thời gian lọc đã ghi nhận (giây/volume) |
|---|---:|---:|---:|---:|---:|
| Original | 5.5524 | 30.8294 | 3.1334 | 1.0000 | Không áp dụng |
| Bilateral | 7.0742 | 50.0445 | 3.7338 | 0.7969 | 34.33 |
| Wavelet BayesShrink | 5.9414 | 35.2998 | 3.3758 | 0.8824 | 0.55 |
| TV Chambolle | 8.4114 | 70.7509 | 4.0481 | 0.4756 | 2.24 |
| BM3D | 5.8463 | 34.1788 | 3.5197 | 0.8999 | 686.80 |

Nguồn: [CSV gốc](../figures/denoise/denoise_metrics_2404.csv) và [metadata phép đo](../figures/denoise/denoise_compare_2404_all_meta.json). Thời gian là số cũ của lần lọc, không phải thời gian render hình lần này; không coi là benchmark tốc độ thống nhất trên mọi phần cứng.

Tham số được metadata ghi nhận: Bilateral `sigma_color=0.10`, `sigma_spatial=4.0`; Wavelet `db4`, BayesShrink, soft thresholding; TV `weight=0.10`; BM3D `sigma_psd` khoảng `0.0416` trên thang 0–1. Cache BM3D ghi sigma chi tiết `0.04164508007606316`.

## 3. Nhận xét theo phương pháp

### Bilateral

Bilateral tăng SNR/CNR so với raw và giữ tương quan gradient cao hơn TV trong bảng `2404`. Trong hình phóng to `x=60`, nền tối ít hạt hơn raw/Wavelet, trong khi các dải sáng vẫn phân biệt được. Đây là nhận xét trực quan trên hình đã xem, không phải xác nhận bảo toàn RNFL bằng segmentation.

Bằng chứng bổ sung trên [14 volume](../figures/denoise/wholevolume/summary.md) cho Bilateral độ lệch chuẩn nền global 8.817 và CNR_bg 5.746, so với raw 11.941 và 4.997. Vì vậy, Bilateral vẫn là lựa chọn cân bằng có cơ sở nhiều volume hơn trong dữ liệu hiện có. Không diễn giải rằng Bilateral đứng đầu mọi chỉ số hoặc mọi cách tổng hợp.

### Wavelet BayesShrink

Wavelet có β cao và còn giữ nhiều cấu trúc hạt tương tự ảnh gốc. Trên các hình `x=60` và `z=160`, nền còn nhiều hạt hơn Bilateral/BM3D, phù hợp với mức làm mịn tương đối nhẹ trong những vùng này. Thời gian lọc được ghi nhận thấp nhất trong bốn phương pháp của bảng trên.

Wavelet là phương án bảo thủ về mức thay đổi ảnh, nhưng β cao có thể một phần đến từ nhiễu còn giữ lại. Không kết luận chỉ từ β rằng nó giữ cấu trúc giải phẫu tốt nhất.

### TV Chambolle

TV có SNR, ENL và CNR cao nhất **trong bốn bộ lọc ở bảng `2404` này**, đồng thời β thấp nhất. Trên `x=60`, texture trong các dải sáng bị làm phẳng rõ hơn; trên lát tái tạo `z=160`, hình có các vệt theo chiều dọc và độ trơn khác đáng kể so với raw. Không tự quy các vệt này cho một nguyên nhân giải phẫu hoặc lỗi cụ thể khi chưa phân tích thêm.

TV phù hợp làm đối chứng làm mịn mạnh, nhưng ảnh trơn hơn không tự động tốt hơn cho phân loại glaucoma. Không bỏ qua mất texture khi lựa chọn chỉ theo SNR/ENL.

### BM3D

BM3D có β cao nhất trong bốn bộ lọc trên `2404` (0.8999), nhỉnh hơn Wavelet (0.8824), nhưng SNR thấp hơn Wavelet và CNR thấp hơn Bilateral/TV. Trên `x=60`, BM3D làm vùng nền tối sạch hơn trong khi vẫn còn texture trong các dải sáng. Kết quả trực quan đáng quan tâm, nhưng chưa đủ để khẳng định giữ cấu trúc bệnh học tốt nhất.

Đổi lại, thời gian cũ khoảng 686.8 giây/volume cao hơn đáng kể các phương pháp khác trong phép đo này. BM3D chưa có bảng 14 volume tương ứng, nên không được xếp hạng như đã được kiểm chứng trên cùng phạm vi với Bilateral/Wavelet/TV.

## 4. Kết luận lựa chọn

| Ưu tiên đánh giá | Nhận xét có thể bảo vệ bằng số liệu hiện tại |
|---|---|
| Cân bằng chất lượng ảnh và bằng chứng nhiều volume | Bilateral là ứng viên ưu tiên; chưa phải tối ưu mọi tiêu chí. |
| Thay đổi ảnh tương đối nhẹ, chi phí thấp | Wavelet đáng giữ làm đối chứng. |
| Làm mịn mạnh | TV, nhưng cần kiểm tra mất texture và ảnh hưởng downstream. |
| Tương quan gradient cao trên mẫu 2404 | BM3D đáng khảo sát thêm, nhưng tốn thời gian và thiếu đối chiếu nhiều volume. |

Không có cơ sở xếp một thứ tự tốt nhất tuyệt đối cho cả bốn phương pháp chỉ từ bảng này. Đặc biệt, không suy ra phương pháp nào tăng accuracy/AUC của model phân loại từ các chỉ số ảnh hoặc các hình minh họa.

## 5. Các hình bổ sung

[PDF tổng hợp 12 trang](../figures/denoise/top4_new_slices/denoise_2404_print.pdf) gồm sáu lát cắt và hai cách hiển thị cho mỗi lát. Các chỉ số lát cắt là zero-based trên trục mảng gốc, không tự đồng nhất với tên mặt phẳng giải phẫu.

| Lát cắt | Toàn cảnh (PDF) | Phóng to (PDF) | Phóng to (PNG) |
|---|---|---|---|
| x = 60 | [Full](../figures/denoise/top4_new_slices/denoise_2404_x60_full.pdf) | [Detail](../figures/denoise/top4_new_slices/denoise_2404_x60_detail.pdf) | [PNG](../figures/denoise/top4_new_slices/denoise_2404_x60_detail.png) |
| x = 100 | [Full](../figures/denoise/top4_new_slices/denoise_2404_x100_full.pdf) | [Detail](../figures/denoise/top4_new_slices/denoise_2404_x100_detail.pdf) | [PNG](../figures/denoise/top4_new_slices/denoise_2404_x100_detail.png) |
| x = 160 | [Full](../figures/denoise/top4_new_slices/denoise_2404_x160_full.pdf) | [Detail](../figures/denoise/top4_new_slices/denoise_2404_x160_detail.pdf) | [PNG](../figures/denoise/top4_new_slices/denoise_2404_x160_detail.png) |
| z = 60 | [Full](../figures/denoise/top4_new_slices/denoise_2404_z60_full.pdf) | [Detail](../figures/denoise/top4_new_slices/denoise_2404_z60_detail.pdf) | [PNG](../figures/denoise/top4_new_slices/denoise_2404_z60_detail.png) |
| z = 100 | [Full](../figures/denoise/top4_new_slices/denoise_2404_z100_full.pdf) | [Detail](../figures/denoise/top4_new_slices/denoise_2404_z100_detail.pdf) | [PNG](../figures/denoise/top4_new_slices/denoise_2404_z100_detail.png) |
| z = 160 | [Full](../figures/denoise/top4_new_slices/denoise_2404_z160_full.pdf) | [Detail](../figures/denoise/top4_new_slices/denoise_2404_z160_detail.pdf) | [PNG](../figures/denoise/top4_new_slices/denoise_2404_z160_detail.png) |

Các lát được chọn trước khi xem kết quả lọc, ở các chỉ số 60, 100 và 160 trên hai trục, thay vì chọn riêng lát đẹp nhất của một phương pháp. Các bản full PNG tương ứng cũng nằm trong cùng thư mục.

## 6. Quy tắc hiển thị và giới hạn

- Mỗi hình gồm Original, Bilateral, Wavelet, TV và BM3D; không ghép kết quả từ các volume khác nhau.
- Trong một lát cắt, giới hạn cường độ là percentile 1/99 của ảnh raw toàn lát, dùng chung cho mọi phương pháp và cả bản full/detail. Không chỉnh sáng riêng từng phương pháp.
- Vùng phóng to 80 × 80 pixel có cùng tọa độ giữa mọi phương pháp và được đánh dấu trên bản toàn cảnh. Đây là vùng hiển thị, không phải ROI giải phẫu đã được xác nhận.
- Nội suy hiển thị nearest-neighbor; không thêm làm mịn khi xuất hình. File PNG 300 DPI và PDF giữ chữ dạng vector, không tạo thêm độ phân giải mô học.
- Metadata xuất hình lưu hash các file đầu vào, tham số được báo cáo, lát cắt, giới hạn sáng và tọa độ crop. Hash xác định dữ liệu đã render, không tự chứng minh nguồn gốc của cache cũ.
- Bảng metric cũ dùng band/mask có thể thay đổi theo output lọc; việc giữ cùng ROI khi vẽ lần này không sửa hồi tố hạn chế của phép đo cũ.
- ENL trong bảng cũ bằng SNR², không phải bằng chứng độc lập; β là tương quan gradient với ảnh còn nhiễu, không phải tỷ lệ phần trăm biên giải phẫu được giữ.
- Các bộ lọc đã được áp dụng slice-wise theo axis 0. Lát `z` là lát tái tạo từ các ảnh đã lọc, không phải kết quả một thuật toán lọc thể tích 3D.

## 7. Tái tạo và trạng thái lưu trữ

```powershell
python scripts/render_denoise_thesis.py --volume 2404 --methods bilateral,wavelet,tv,bm3d --planes x:60 x:100 x:160 z:60 z:100 z:160 --output-dir figures/denoise/top4_new_slices
```

Hình cũ không bị ghi đè. Đồng bộ Google Drive còn **pending** vì phiên Windows này chưa có đích Drive được xác thực. Script hỗ trợ `--drive-sync-dir` trỏ tới một thư mục Drive đã đồng bộ thực tế; không coi một thư mục local có tên Drive là đã tải lên cloud. Chưa có bản PDF luận văn được dựng lại trong lần xuất hình này.
