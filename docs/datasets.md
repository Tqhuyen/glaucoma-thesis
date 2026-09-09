# Thống kê bộ dữ liệu & phân tích lựa chọn dataset cho luận văn

**Luận văn:** 3D CNN phát hiện glaucoma từ volume OCT (Harvard-GF), kèm bước tiền xử lý
khử nhiễu speckle (BM3D/DnCNN/SwinIR/Noise2Void).

**Phạm vi tài liệu:** thống kê tối đa số liệu có thể xác minh của dataset chính (Harvard-GF)
và các bộ dữ liệu OCT công khai ứng viên, kèm phân tích lý do nên chọn/từ chối từng bộ.

> Số liệu Harvard-GF được **tự đếm trực tiếp** từ `data_summary.csv` của dataset
> (ngày truy cập 09/09/2026). Số liệu các bộ khác đối chiếu từ paper/trang chính thức
> (kèm URL); ô ghi `≈` = theo mô tả paper, `?` = chưa xác minh được.

---

## 1. Dataset chính: Harvard-GF (3.300 volume 200³)

| Mục | Giá trị |
|---|---|
| Nguồn | Harvard Ophthalmology AI Lab (MIT); IEEE TMI 2024 |
| Đường dẫn | HF `harvardairobotics/Harvard-GF` |
| Số mẫu | **3.300** mẫu (mỗi mẫu = 1 bệnh nhân/1 mắt) |
| Modality | **Volume OCT B-scan 200³ uint8** (`oct_bscans`) + RNFLT map + VF |
| Vùng quét | Đầu thị thần kinh / lớp RNFL (phù hợp dò glaucoma) |
| Dung lượng | ~200³×1 byte ≈ 8 MB/volume → ~26 GB toàn bộ (raw) |
| Độ phân giải lưu trữ | 200 × 200 × 200 (xử lý/chuẩn hoá về 200³) |

### 1.1. Phân bố theo split × nhãn glaucoma (đếm trực tiếp từ CSV)

| Split | Glaucoma+ | Glaucoma− | Tổng |
|---|---|---|---|
| Training | 1.083 | 1.017 | **2.100** |
| Validation | 176 | 124 | **300** |
| Test | 489 | 411 | **900** |
| **Tổng** | **1.748 (53.0%)** | **1.552 (47.0%)** | **3.300** |

### 1.2. Demographics (fairness — điểm mạnh của bộ này)

| Thuộc tính | Phân bố |
|---|---|
| Tuổi (n=3.300) | min 10,1 · median 61,4 · mean 59,1 · max 98,0 |
| Giới tính | female 1.812 (54,9%) · male 1.488 (45,1%) |
| Chủng tộc | asian 1.100 · black 1.100 · white 1.100 (**cân bằng 1/3 đều**) |
| Ethnicity | non-hispanic 3.025 · hispanic 87 · unknown 188 |
| Ngôn ngữ | english 2.877 · other 354 · spanish 38 · unknown 31 |
| Tình trạng hôn nhân | married/partnered 1.884 · single 947 · widowed 181 · divorced 168 · legally separated 39 · unknown 81 |

### 1.3. Đặc điểm kỹ thuật & truy cập

| Mục | Giá trị |
|---|---|
| Nhãn | Glaucoma binary (yes/no); + demographics (race/gender/age/ethnicity/language/marital) |
| Fairness | Có — thiết kế để nghiên cứu công bằng giữa các nhóm chủng tộc/giới |
| Format | Dataset zip: `Test|Training|Validation/data_XXXX.npz` (nén per-scan, `oct_bscans`) |
| Split file | `ReadMe/data_summary.csv` (cột `use` = training/validation/test) |
| Truy cập | Public (HF, không gated) |
| Giấy phép | Repo MIT (dataset theo điều khoản Harvard GF, dùng nghiên cứu) |
| Liên hệ trích dẫn | Luo et al., *Harvard Glaucoma Fairness (Harvard-GF)*, IEEE TMI 2024 |

---

## 2. Các bộ dữ liệu OCT công khai ứng viên (so sánh)

| Dataset | Modality | Vùng quét | Số mẫu | Nhãn glaucoma | Fairness | Truy cập / License |
|---|---|---|---|---|---|---|
| **Harvard-GF** | **3D OCT volume (B-scan 200³)** | **ONH / RNFL** | **3.300** | ✅ | ✅ race/gender/age | HF public; MIT (repo) |
| **GAMMA** (MICCAI'21/MedIA) | 3D OCT volume (256 B-scan × 992×512) + fundus | **Macula** | 300 (276 BN) | ✅ grading 3-4 mức (non-glc 150; glc 150: early 78/inter 43/adv 29) | ⚠️ chỉ sex/age | cần đăng ký grand-challenge; CC BY-NC-ND |
| **OCTDL** (Sci. Data 2024) | **2D B-scan** (không volume) | Macula (canh fovea) | 2.064 ảnh / 821 BN | ❌ (7 nhóm: AMD, DME, ERM, Normal, RAO, RVO, VID) | ⚠️ age/sex | Mendeley `sncdhf53xc`, CC BY 4.0 |
| **OCT2017 / Kermany** (Cell 2018) | **2D B-scan** | Macula | **84.484 ảnh** (train 83.484 + test 1.000) | ❌ (CNV/DME/Drusen/Normal) | ❌ | Mendeley `rscbjbr9sj`, CC BY 4.0; mirror Kaggle |
| **OCTID** (2020) | 2D B-scan | Macula | >500 ảnh (25 normal kèm GT segment) | ❌ (NO/MH/AMD/CSR/DR) | ❌ | Borealis dataverse, open |
| **Duke OCT / Srinivasan 2014** | 3D volume | Macula | 45 volume (15 Normal/15 AMD/15 DME) | ❌ | ❌ | trang Duke; không license rõ |
| **Harvard-GDP** (ICCV'23) | **RNFLT map 2D (225×225) + VF** — **không B-scan** | ONH (suy từ OCT) | 1.000 | ✅ glaucoma + progression (500 mẫu) | ✅ race/sex | HF; CC BY-NC-ND |

---

## 3. Ma trận căn chỉnh với yêu cầu luận văn

| Tiêu chí | Harvard-GF | GAMMA | OCTDL | OCT2017 | OCTID | Duke | Harvard-GDP |
|---|---|---|---|---|---|---|---|
| Có volume OCT **3D** (B-scan stack) | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Vùng quét **ONH/RNFL** (khớp glaucoma) | ✅ | ❌ (macula) | ❌ | ❌ | ❌ | ❌ | (map RNFL) |
| Nhãn **glaucoma** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Đủ lớn train 3D-CNN | ✅ 3.300 | ~300 | ❌ | ❌ | ❌ | 45 | ❌ |
| Dữ liệu demographics/fairness | ✅ | một phần | một phần | ❌ | ❌ | ❌ | ✅ |
| Public không gated | ✅ | đăng ký | ✅ | ✅ | ✅ | ✅ | ✅ |
| Phù hợp khử nhiễu speckle (nhiều B-scan nhiễu) | ✅ 660k B-scan | ~77k | 2.064 | 84.484 | ~500 | ~2k | — |

*Số B-scan ước lượng: Harvard-GF = 3.300 × 200 = 660.000; GAMMA ≈ 300 × 256 ≈ 76.800.*

---

## 4. Phân tích: tại sao chọn / từ chối từng bộ

### 4.1. Harvard-GF — CHỌN LÀM DỮ LIỆU CHÍNH (và dùng TRỌN 3.300 volume)
- **Khớp bài toán nhất:** duy nhất trong danh sách là volume B-scan **vùng ONH/RNFL** —
  đúng cấu trúc tổn thương glaucoma (bèo RNFL) mà luận văn cần.
- **Đủ lớn & cân bằng:** 3.300 volume ≈ 660.000 B-scan; class glaucoma+ 53% / − 47%;
  chia sẵn train/val/test chuẩn 2.100/300/900, bệnh nhân không trùng giữa các split.
- **Fairness trọn gói:** cân bằng race 1/3 (Asian/Black/White), đủ gender/age/ethnicity —
  phục vụ trực tiếp phân tích công bằng trong luận văn.
- **Phù hợp denoise self-learn:** dùng **toàn bộ 3.300 volume** (không chỉ test split) cho
  Noise2Void = nguồn B-scan cùng máy/cùng vùng lớn nhất, không domain-shift.

### 4.2. GAMMA — chỉ dùng như *external evaluation* (KHÔNG train chính)
- Điểm mạnh: dataset glaucoma có **3D OCT volume** thứ hai + nhãn grading (non/early/
  intermediate/advanced) để kiểm chứng ngoài.
- Điểm loại: chỉ 300 mẫu; **quét macula** (không phải ONH) → khác cấu trúc vùng khảo sát;
  license CC BY-NC-ND + phải đăng ký → chỉ phù hợp đánh giá ngoài, không dùng làm dữ liệu train.

### 4.3. OCTDL & OCT2017 — chỉ cho *pretrain denoiser 2D*, không dùng cho mô hình 3D
- Lợi ích giới hạn: là **2D B-scan** (đúng đơn vị mà U-Net N2V xử lý từng slice), nhiều ảnh
  (OCT2017 ~84k), speckle cùng bản chất → có thể tăng đa dạng khi train denoiser.
- Hạn chế quyết định: không có glaucoma, quét **macula** khác vùng ONH, khác máy/độ phân giải,
  **không phải volume** → không dùng được cho 3D-CNN cũng như không đánh giá được RNFL.
  ⇒ Chỉ thêm khi thực sự cần đa dạng speckle; lợi ích mơ hồ hơn so với dùng full Harvard-GF.

### 4.4. OCTID & Duke — KHÔNG chọn (đánh giá nhanh)
- OCTID: >500 ảnh 2D macula, không glaucoma, không split → chỉ tham khảo.
- Duke 2014: 45 volume, không glaucoma, kích thước volume **không đồng nhất**
  (31–97 B-scan, 512–1024 A-scan) → khó chuẩn hoá về pipeline 200³, quá nhỏ.

### 4.5. Harvard-GDP — KHÔNG chọn cho phần OCT B-scan
- Chỉ phát hành **RNFLT map 2D (225×225) + VF**, không có volume B-scan gốc → không tương
  thích 3D-CNN trên raw B-scan. (Có thể tham chiếu ở phần phát hiện/progression dùng RNFLT,
  nhưng nằm ngoài phạm vi mô hình này.)

---

## 5. Kết luận & khuyến nghị

1. **Dữ liệu chính:** toàn bộ **Harvard-GF 3.300 volume** (train/val/test theo CSV chuẩn).
2. **Bước denoise self-learn:** dùng full Harvard-GF (≈660.000 B-scan) — đủ lớn, cùng
   domain; chỉ thêm OCT2017/OCTDL nếu muốn thử nghiệm tăng cường đa dạng speckle (đánh dấu
   rõ là ngoài phân phối).
3. **Đánh giá ngoài (tuỳ chọn):** GAMMA nếu cần minh chứng glaucoma-OCT thứ hai (chú thích
   khác biệt macula vs ONH).
4. **Loại khỏi pipeline raw-B-scan:** Harvard-GDP (không B-scan), OCTID/Duke (quá nhỏ/
   không glaucoma), OCTDL/OCT2017 (2D macula — chỉ denoiser).

---

## 6. Nguồn tham khảo (truy cập 09/09/2026)

- Harvard-GF: https://github.com/Harvard-Ophthalmology-AI-Lab/Harvard-GF · HF `harvardairobotics/Harvard-GF` · `data_summary.csv`
- GAMMA: arXiv:2202.06511 · https://gamma.grand-challenge.org/
- OCTDL: Nature Scientific Data (2024), doi:10.1038/s41597-024-03182-7 · Mendeley 10.17632/sncdhf53xc
- OCT2017/Kermany: Cell 172:1122–1131 (2018) · Mendeley 10.17632/rscbjbr9sj.2
- OCTID: arXiv:1812.07056 · Borealis dataverse/OCTID
- Duke OCT 2014: people.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.htm
- Harvard-GDP: arXiv:2308.13411 · HF `harvardairobotics/Harvard-GDP`
