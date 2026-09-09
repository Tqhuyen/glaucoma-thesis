# So sánh các phương pháp khử nhiễu speckle trên OCT 3D (Harvard-GF)

**Mẫu thí nghiệm:** volume `data_2404` (glaucoma+), 200³ uint8, trích trực tiếp từ
Harvard-GF. Không dùng ground-truth sạch; mọi ảnh khử nhiễu được ánh xạ ngược về
thang [0,255] của ảnh gốc để so sánh cùng thang đo.

**Script:** `scripts/compare_denoise_methods.py` — output tại `figures/denoise/`.

---

## 1. Danh sách phương pháp (từ đơn giản → phức tạp)

| # | Phương pháp | Loại | Tham số thí nghiệm | Nguồn |
|---|---|---|---|---|
| 0 | **Original** | — | ảnh thô 200³ | Harvard-GF |
| 1 | **Gaussian** | Tuyến tính (làm mịn) | `sigma = 1.5` (px) | `skimage.filters.gaussian` |
| 2 | **Median** | Phi tuyến (thống kê cửa sổ) | cửa sổ `3×3` | `skimage.filters.median` |
| 3 | **Bilateral** | Giữ biên (lọc miền giá trị) | `sigma_color = 0.10`, `sigma_spatial = 4.0` (trên [0,1]) | Tomasi & Manduchi 1998; `skimage.restoration.denoise_bilateral` |
| 4 | **Total Variation (Chambolle)** | Khử nhiễu biến phân | `weight (λ) = 0.10` | Rudin-Osher-Fatemi 1992; `skimage.restoration.denoise_tv_chambolle` |
| 5 | **Wavelet (BayesShrink)** | Khử nhiễu miền wavelet | `db4`, soft, `BayesShrink`, `rescale_sigma=True` | Chang et al. 2000; `skimage.restoration.denoise_wavelet` |
| 6 | **Non-Local Means (NLM)** | Phi cục bộ (khớp patch) | `h = 0.10`, `patch_size = 5`, `patch_distance = 3`, `fast_mode` | Buades et al. 2005; `skimage.restoration.denoise_nl_means` |
| 7 | **BM3D** | Cộng tác khối (miền biến đổi 3D) | profile `np`, `sigma_psd ≈ 0.042` (ước lượng tự động) | Dabov et al. 2007; gói `bm3d` |
| 8 | **DnCNN** (deep learning, GPU) | CNN học phần dư (residual learning) | gray **blind** (σ tự đo), 64 ch, ~20 lớp | Zhang et al. 2017; weights KAIR `dncnn_gray_blind.pth` |
| 9 | **SwinIR** (deep learning, GPU) | Swin-Transformer khử nhiễu | gray denoising **σ=25**, SwinIR-M, `window=8` | Liang et al. 2021; `004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth` |

Mọi phương pháp được áp dụng **2D trên từng B-scan** (slice theo trục 0), đúng quy
trình notebook NLM trước đó. Với BM3D ước lượng `sigma_psd` bằng
`skimage.restoration.estimate_sigma` (median trên 40 slice, thang [0,1]).

**DnCNN & SwinIR** chạy qua PyTorch (`scripts/denoise_torch.py`): tự chọn GPU nếu có
(`cuda`), fallback CPU; batch qua từng slice rồi ánh xạ về [0,255] giống các phương
pháp CPU. Weights tải một lần về `%TEMP%/gf_denoise_weights` (DnCNN ~2.7 MB,
SwinIR ~123 MB). Yêu cầu `pip install torch` (+ bản CUDA trên Colab).

---

## 2. Kết quả số liệu (volume `data_2404`)

Định nghĩa vùng đo: band RNFL `[29,57)` theo trục depth (dz=1); nền đồng nhất
`[5,29)` (phía trên RNFL); ngưỡng tín hiệu = percentile 60 của band ảnh gốc
(=84/255); crop 40 px biên lateral. **Cùng vùng & cùng ngưỡng cho mọi phương pháp.**

| Method | SNR | ENL | CNR | β (edge) | time/vol |
|---|---|---|---|---|---|
| Original | 5.552 | 30.83 | 3.133 | 1.000 | – |
| Gaussian | 8.217 | 67.52 | 3.834 | 0.345 | 0.10 s |
| Median | 6.849 | 46.91 | 3.803 | 0.197 | 0.21 s |
| Bilateral | 7.074 | 50.04 | 3.734 | 0.797 | 34.3 s |
| **Total Variation** | **8.411** | **70.75** | 4.048 | 0.476 | 2.24 s |
| Wavelet | 5.941 | 35.30 | 3.376 | 0.882 | 0.55 s |
| NLM | 7.911 | 62.59 | **4.130** | 0.275 | 0.99 s |
| BM3D | 5.846 | 34.18 | 3.520 | **0.900** | 687 s (~11 min) |

> Ghi chú: `time/vol` đo trên máy cá nhân (12 luồng CPU, Windows). `β` càng gần 1
> càng giữ biên; `β=1` của Original là do ảnh tự so với chính nó.

**Cách đọc nhanh:**
- **TV Chambolle**: giảm nhiễu mạnh nhất theo SNR/ENL/CNR, rất nhanh → **cân bằng tốt nhất**.
- **BM3D / Wavelet / Bilateral**: giữ cấu trúc + biên tốt (β cao), nhưng BM3D quá chậm cho cả dataset.
- **NLM** (baseline cũ): CNR cao nhất nhưng β thấp (dễ làm mờ lớp RNFL mỏng).
- **Gaussian / Median**: nhanh nhưng phá biên nhiều (β thấp) — chỉ dùng làm baseline.

---

## 3. Chi tiết từng phương pháp

### 3.1 Gaussian (làm mịn Gaussian)
- **Nguyên lý:** chập ảnh với nhân Gaussian đối xứng → lọc thông thấp, loại nhiễu tần số cao.
- **Tham số:** bán kính `sigma=1.5` pixel (2D, từng B-scan).
- **Đặc điểm:** đơn giản nhất, rất nhanh; nhưng xoá luôn chi tiết mảnh → biên RNFL bị mờ, β thấp (0.345).
- **Thời gian:** ~0.1 s/volume.

### 3.2 Median (lọc trung vị)
- **Nguyên lý:** thay mỗi điểm ảnh bằng trung vị trong cửa sổ → khử nhiễu "đốm" mà vẫn giữ bước nhảy cường độ.
- **Tham số:** cửa sổ `3×3`.
- **Đặc điểm:** tốt cho nhiễu xung; với speckle dày đặc, cửa sổ nhỏ nên giảm nhiễu vừa phải (SNR 6.85); β thấp nhất (0.197) trong thí nghiệm này → không lý tưởng cho chi tiết mỏng.
- **Thời gian:** ~0.2 s/volume.

### 3.3 Bilateral (lọc song phương)
- **Nguyên lý:** trung bình có trọng số theo *khoảng cách không gian* và *khác biệt cường độ* — chỉ gộp các pixel "giống" nên giữ biên.
- **Tham số:** `sigma_color=0.10` (trên [0,1]), `sigma_spatial=4.0` px.
- **Đặc điểm:** cân bằng khá tốt: SNR 7.07, β 0.797. Nhược: chậm hơn Gaussian/Median (34 s/vol) và có thể tạo hiệu ứng "step" nếu sigma_color nhỏ.
- **Thời gian:** ~34 s/volume.

### 3.4 Total Variation – Chambolle (TV)
- **Nguyên lý:** cực tiểu hoá `‖∇u‖₁` với ràng buộc dữ liệu (λ) → khử nhiễu nhưng giữ biên sắc (mô hình piecewise-constant).
- **Tham số:** `weight (λ)=0.10`.
- **Đặc điểm:** trong thí nghiệm đạt SNR/ENL/CNR **cao nhất** (8.41 / 70.8 / 4.05), thời gian rất ngắn (2.2 s). β trung bình (0.476) — có thể làm phẳng vân mịn, hiện tượng "staircase" ở vùng gradient thoải.
- **Thời gian:** ~2.2 s/volume → **khuyến nghị cho pipeline xử lý hàng loạt**.

### 3.5 Wavelet – BayesShrink (db4)
- **Nguyên lý:** biến đổi wavelet rời rạc, ngưỡng mềm hệ số chi tiết theo ngưỡng Bayes thích ứng từng subband rồi tái tạo.
- **Tham số:** `wavelet='db4'`, `mode='soft'`, `method='BayesShrink'`, `rescale_sigma=True`.
- **Đặc điểm:** khử nhiễu **bảo thủ** (SNR 5.94, gần nguyên bản) nhưng giữ biên rất tốt (β 0.882), rất nhanh. Hợp khi cần "làm sạch nhẹ" mà không đụng cấu trúc.
- **Thời gian:** ~0.55 s/volume.

### 3.6 Non-Local Means (NLM) — baseline của luận văn
- **Nguyên lý:** với mỗi patch, trung bình các patch giống nó trên *toàn ảnh* (tự tương tự) → khử speckle hiệu quả, bảo toàn kết cấu lặp lại.
- **Tham số:** `h=0.10` (trên [0,1]), `patch_size=5`, `patch_distance=3`, `fast_mode=True`.
- **Đặc điểm:** CNR cao nhất (4.13), SNR 7.91; nhưng β 0.275 cho thấy dễ làm trơn chi tiết mỏng/biên. Đây đúng tham số đã dùng trong notebook NLM trước.
- **Thời gian:** ~1 s/volume.

### 3.7 BM3D
- **Nguyên lý:** gom các khối tương tự thành nhóm → biến đổi 3D (2D DCT + 1D) → lọc ngưỡng/shrinkage cộng tác → tái tạo; hiện "state of the art" nhóm phương pháp cổ điển.
- **Tham số:** profile `np` (mặc định), `sigma_psd≈0.0416` ước lượng tự động từ ảnh.
- **Đặc điểm:** giữ cấu trúc/biên **tốt nhất** (β 0.900), ảnh nhìn sạch & tự nhiên nhất. Nhược: **rất chậm** — ~11 phút/volume trên CPU 12 luồng ⇒ ~2,5 ngày cho full Harvard-GF (3.300 volume) nếu chạy 10 process song song.
- **Thời gian:** ~687 s/volume (profile `np`); profile nhẹ `lc` nhanh gấp ~3× (≈0.9 s/slice) với chất lượng giảm chút, có thể dùng khi cần chạy hàng loạt.

---

## 4. Định nghĩa chỉ số (không cần ground-truth)

| Chỉ số | Công thức | Ý nghĩa |
|---|---|---|
| **SNR** | `μ_s / σ_s` trên vùng tín hiệu (band RNFL) | Tỉ số tín hiệu/nhiễu; cao = sạch hơn |
| **ENL** | `(μ_s/σ_s)²` | Số "looks" tương đương; cao = speckle bị nén mạnh |
| **CNR** | `(μ_s − μ_b)/√(σ_s² + σ_b²)` (b: nền) | Tương phản mô vs nền tính trên nhiễu |
| **β (edge)** | Tương quan gradient-magnitude giữa ảnh gốc & ảnh khử nhiễu tại pixel biên mạnh (top 15%) | 1 = giữ nguyên biên; <1 = mất chi tiết biên |

> Cảnh báo khoa học: OCT không có ảnh sạch thật nên các số này là **no-reference**.
> `β` được đo tại biên phát hiện trên ảnh gốc (vốn chứa speckle), vì vậy cần xem
> kèm ảnh trực quan khi kết luận.

---

## 5. Ảnh so sánh (trong `figures/denoise/`)

6 mặt cắt hiển thị: `x=132, x=148, y=22, y=38, z=114, z=130` (2 vị trí/hướng cắt);
cùng thang xám (percentile 1–99 của ảnh gốc) cho mọi hàng.

| File | Nội dung |
|---|---|
| `denoise_compare_2404_all.png` | **1 ảnh lớn**: Original + 7 pp × 6 mặt cắt (so mắt nhanh) |
| `denoise_compare_2404_{method}.png` | Ảnh riêng từng pp: hàng trên = Original, hàng dưới = pp đó |
| `denoise_compare_2404.png` | Bản NLM cũ (trước thí nghiệm này) |
| `denoise_metrics_2404.png` | Bảng số liệu dạng ảnh để dán vào báo cáo |
| `denoise_metrics_2404.csv` | Bảng số liệu dạng CSV |
| `denoise_compare_2404_all_meta.json` | Tham số + vùng đo + metric đầy đủ (dạng JSON) |
| `cache/` | Volume 200³ đã khử nhiễu (uint8) để tái sử dụng, không phải ảnh báo cáo |

---

## 6. Tái tạo

```bash
# 7 phương pháp trên volume 2404 (cache giúp không tính lại BM3D)
python scripts/compare_denoise_methods.py --volume 2404 --workers 8

# Chỉ vài phương pháp / volume khác
python scripts/compare_denoise_methods.py --volume 2404 --methods nlm,tv
python scripts/compare_denoise_methods.py --volume 3294 --methods bm3d

# Thêm 2 phương pháp deep-learning dùng GPU (Colab: pip install torch bm3d scikit-image)
# Chạy lại từng method để tải weights (~2.7 MB DnCNN + ~123 MB SwinIR) rồi so ảnh:
python scripts/compare_denoise_methods.py --volume 2404 --methods dncnn
python scripts/compare_denoise_methods.py --volume 2404 --methods swinir
python scripts/compare_denoise_methods.py --volume 2404 --methods dncnn,swinir
```

Tham số từng pp sửa trực tiếp trong dict `METHODS` đầu script; output luôn ghi vào
`figures/denoise/` (cache: `figures/denoise/cache/`). Volume thô tự tải qua
HTTP-range từ Harvard-GF nếu chưa có trong cache `%TEMP%/gf_vol_cache`.

## 7. Tài liệu tham khảo chính

- Tomasi & Manduchi, *Bilateral filtering for gray and color images*, ICCV 1998.
- Rudin, Osher & Fatemi, *Nonlinear total variation based noise removal*, Physica D 1992.
- Chambolle, *An algorithm for total variation minimization and applications*, JMIV 2004.
- Chang, Yu & Vetterli, *Adaptive wavelet thresholding for image denoising and compression*, IEEE TIP 2000.
- Buades, Coll & Morel, *A review of image denoising algorithms, with a new one*, SIAM MMS 2005.
- Dabov, Foi, Katkovnik & Egiazarian, *Image denoising by sparse 3-D transform-domain collaborative filtering*, IEEE TIP 2007.
- Sklearn-image: `skimage.restoration.*`, `skimage.filters.gaussian/median`.
- Gói `bm3d` (bao bọc OpenCV): https://pypi.org/project/bm3d/
