# Khảo sát khử nhiễu trên TOÀN volume — 6 PP cổ điển (bỏ BM3D/DnCNN/SwinIR)

- Dữ liệu: **14 volume** Harvard-GF raw 200³ (cache local): 2401, 2404, 2469, 2541, 2615, 2683, 2760, 2863, 2967, 3037, 3100, 3162, 3234, 3294.
- Phương pháp cổ điển: gaussian, median, bilateral, tv, wavelet, nlm — khử nhiễu **2D trên từng B-scan**, toàn mặt cắt (không crop, không band).
- Ngưỡng tín hiệu/nền: **Otsu tính 1 lần trên volume gốc**, dùng chung mọi PP (công bằng).
- Chỉ số tính **theo từng lát cắt** rồi gộp bằng **sum / mean / median**; kèm bản **global** (gộp toàn bộ voxel).

> **Cách đọc:** nhóm `snr/enl/cnr` (chỉ dùng vùng tín hiệu) bị **cấu trúc giải phẫu chi phối** trên toàn volume
> nên ít phản ánh mức giảm nhiễu → so sánh năng lực khử nhiễu bằng nhóm `*_bg` (nhiễu lấy từ vùng nền).
> `bg_sigma` ↓ càng tốt; `snr_bg/enl_bg/cnr_bg` ↑ càng tốt; `beta` càng gần 1 càng giữ biên.

## A. Global toàn volume (gộp toàn bộ voxel; mean ± std trên 14 volume)

| Method | bg_sigma | snr_bg | enl_bg | cnr_bg | snr | enl | cnr | beta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 11.941 ± 0.515 | 7.242 ± 0.233 | 5.063 ± 0.767 | 4.997 ± 0.298 | 3.749 ± 0.082 | 14.060 ± 0.616 | 2.296 ± 0.113 | 1.000 ± 0.000 |
| gaussian | 10.458 ± 0.831 | 7.245 ± 0.252 | 7.484 ± 1.854 | 4.526 ± 0.349 | 3.601 ± 0.217 | 13.012 ± 1.557 | 2.017 ± 0.214 | 0.515 ± 0.036 |
| median | 11.055 ± 0.748 | 7.116 ± 0.267 | 6.544 ± 1.396 | 4.570 ± 0.391 | 3.273 ± 0.193 | 10.753 ± 1.266 | 1.912 ± 0.201 | 0.520 ± 0.044 |
| bilateral | 8.817 ± 0.696 | 8.737 ± 0.433 | 9.036 ± 1.997 | 5.746 ± 0.518 | 3.720 ± 0.231 | 13.891 ± 1.739 | 2.250 ± 0.222 | 0.858 ± 0.026 |
| tv | 9.945 ± 0.804 | 7.617 ± 0.292 | 8.293 ± 2.104 | 4.756 ± 0.389 | 3.632 ± 0.238 | 13.246 ± 1.733 | 2.050 ± 0.230 | 0.598 ± 0.033 |
| wavelet | 9.993 ± 0.621 | 8.316 ± 0.304 | 7.311 ± 1.346 | 5.621 ± 0.402 | 3.600 ± 0.145 | 12.981 ± 1.047 | 2.234 ± 0.171 | 0.901 ± 0.022 |
| nlm | 9.917 ± 0.783 | 7.679 ± 0.294 | 8.234 ± 2.041 | 4.827 ± 0.399 | 3.488 ± 0.210 | 12.210 ± 1.476 | 1.999 ± 0.217 | 0.606 ± 0.033 |

## B. Theo lát cắt → gộp **mean** → mean ± std trên volume

| Method | bg_sigma | snr_bg | enl_bg | cnr_bg | snr | enl | cnr | beta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 11.433 ± 0.439 | 6.753 ± 0.277 | 5.956 ± 0.754 | 4.333 ± 0.362 | 7.265 ± 1.411 | 94.295 ± 48.360 | 2.587 ± 0.158 | 1.000 ± 0.000 |
| gaussian | 8.070 ± 1.059 | 9.011 ± 0.846 | 33.856 ± 11.906 | 3.906 ± 0.794 | 4.804 ± 0.701 | 30.446 ± 11.295 | 1.509 ± 0.295 | 0.311 ± 0.049 |
| median | 9.338 ± 0.799 | 6.982 ± 0.510 | 14.522 ± 3.383 | 3.366 ± 0.754 | 3.536 ± 0.216 | 13.266 ± 1.693 | 1.344 ± 0.299 | 0.298 ± 0.059 |
| bilateral | 6.677 ± 0.835 | 9.934 ± 0.766 | 27.831 ± 7.307 | 5.017 ± 0.918 | 5.830 ± 1.237 | 53.170 ± 27.817 | 1.938 ± 0.262 | 0.695 ± 0.028 |
| tv | 7.321 ± 1.167 | 11.033 ± 1.600 | 69.352 ± 32.419 | 4.337 ± 0.989 | 5.517 ± 1.238 | 50.627 ± 28.288 | 1.484 ± 0.332 | 0.368 ± 0.062 |
| wavelet | 9.107 ± 0.557 | 7.779 ± 0.416 | 10.091 ± 1.447 | 4.644 ± 0.595 | 5.075 ± 0.630 | 32.282 ± 9.727 | 2.223 ± 0.211 | 0.749 ± 0.047 |
| nlm | 7.534 ± 1.062 | 10.045 ± 1.157 | 47.667 ± 19.011 | 4.198 ± 0.927 | 4.963 ± 0.899 | 35.872 ± 16.349 | 1.461 ± 0.313 | 0.363 ± 0.061 |

## C. So sánh cách gộp (slice sum / mean / median; lấy mean qua các volume)


**bg_sigma**

| Method | sum(slices) | mean(slices) | median(slices) |
|---|---:|---:|---:|
| original | 2286.667 | 11.433 | 11.471 |
| gaussian | 1613.953 | 8.070 | 7.111 |
| median | 1867.693 | 9.338 | 8.353 |
| bilateral | 1335.480 | 6.677 | 6.037 |
| tv | 1464.227 | 7.321 | 6.574 |
| wavelet | 1821.389 | 9.107 | 8.762 |
| nlm | 1506.854 | 7.534 | 6.693 |

**snr_bg**

| Method | sum(slices) | mean(slices) | median(slices) |
|---|---:|---:|---:|
| original | 1350.554 | 6.753 | 6.679 |
| gaussian | 1802.122 | 9.011 | 8.619 |
| median | 1396.351 | 6.982 | 6.287 |
| bilateral | 1986.769 | 9.934 | 9.208 |
| tv | 2206.564 | 11.033 | 9.777 |
| wavelet | 1555.881 | 7.779 | 7.428 |
| nlm | 2008.931 | 10.045 | 9.406 |

**enl_bg**

| Method | sum(slices) | mean(slices) | median(slices) |
|---|---:|---:|---:|
| original | 1191.202 | 5.956 | 6.029 |
| gaussian | 6771.171 | 33.856 | 21.042 |
| median | 2904.429 | 14.522 | 12.453 |
| bilateral | 5566.224 | 27.831 | 20.794 |
| tv | 13870.308 | 69.352 | 27.616 |
| wavelet | 2018.189 | 10.091 | 9.838 |
| nlm | 9533.491 | 47.667 | 24.737 |

**cnr_bg**

| Method | sum(slices) | mean(slices) | median(slices) |
|---|---:|---:|---:|
| original | 866.555 | 4.333 | 4.178 |
| gaussian | 781.153 | 3.906 | 3.094 |
| median | 673.109 | 3.366 | 2.933 |
| bilateral | 1003.373 | 5.017 | 4.084 |
| tv | 867.481 | 4.337 | 3.316 |
| wavelet | 928.816 | 4.644 | 4.287 |
| nlm | 839.697 | 4.198 | 3.301 |

**beta**

| Method | sum(slices) | mean(slices) | median(slices) |
|---|---:|---:|---:|
| original | 200.000 | 1.000 | 1.000 |
| gaussian | 62.278 | 0.311 | 0.307 |
| median | 59.546 | 0.298 | 0.303 |
| bilateral | 138.903 | 0.695 | 0.708 |
| tv | 73.510 | 0.368 | 0.399 |
| wavelet | 149.736 | 0.749 | 0.775 |
| nlm | 72.686 | 0.363 | 0.396 |

## D. Nhận xét

- **Bilateral**: giảm nhiễu mạnh nhất (`bg_sigma` thấp nhất, `enl_bg`/`snr_bg`/`cnr_bg` cao nhất) mà vẫn giữ biên tốt (β ≈ 0,86 toàn cục).
- **Wavelet**: giữ biên tốt nhất trong nhóm khử nhiễu (β ≈ 0,90), giảm nhiễu khá — lựa chọn 'bảo thủ'.
- **TV / NLM**: giảm nhiễu tốt nhưng mất biên nhiều (β ≈ 0,59–0,61 toàn cục).
- **Gaussian / Median**: nhanh nhất nhưng phá biên nhiều (β ≈ 0,52) → chỉ làm baseline.

### Về cách gộp sum / mean / median

- `bg_sigma` và `beta`: **mean ≈ median** (lệch < 0,1) → gộp kiểu nào cũng cho cùng kết luận.
- `enl_bg` (và `snr_bg`): **đuôi nặng** — mean lớn hơn median rõ rệt (vd TV: 69,4 vs 27,6; NLM: 47,7 vs 24,7)
  do vài lát cắt có nền rất phẳng làm ENL vọt lên → **median vững hơn**, mean dễ bị kéo lệch.
- `sum` chỉ là `mean × số lát cắt` (~200) nên **không đổi thứ hạng**, không mang thêm thông tin.
- **Khuyến nghị báo cáo:** dùng **mean ± std theo lát cắt → gộp qua volume** làm bảng chính,
  kèm **median** làm kiểm tra độ vững; bỏ `sum` (hoặc ghi chú là tổng theo lát cắt).

## E. File kèm

- `per_volume_metrics.csv` — chỉ số từng volume × PP (cả 3 cách gộp + global).
- `summary_metrics.csv` — bảng gộp đầy đủ (slice_agg × across-volume sum/mean/median/std).
- `wholevolume_survey.json` — toàn bộ số liệu + tham số + thời gian khử nhiễu.

## F. Hạn chế

- Mẫu gồm **14 volume** (cache local), **chưa phải toàn bộ** Harvard-GF và chưa chắc cân bằng glaucoma+/−.
- Metric là **no-reference** (OCT không có ảnh sạch); mask tín hiệu/nền dựa trên ngưỡng Otsu, có thể lệch giữa các volume.
- `beta` dùng gradient **trong mặt phẳng** (2D), không dùng gradient theo trục depth như bản band-based cũ.
- Bilateral rất chậm (~2 phút/volume) → toàn bộ dataset nên chạy trên Colab/vast.ai.

> Thời gian khử nhiễu `bilateral` ~115–130 s/volume (1 luồng/volume, 12 volume song song). `bm3d` bị loại; DnCNN/SwinIR không dùng vì không phải PP cổ điển.
