# data/ — dữ liệu cục bộ (KHÔNG BAO GIỜ PUSH LÊN GIT)

Thư mục này bị `.gitignore` (`data/*`, chỉ cho phép `data/README.md`).
Mọi file dữ liệu (`.npy`, `.npz`, `.tar.zst`, ...) đặt ở đây sẽ **không** được commit/push.

## Cách bỏ dữ liệu vào

Notebook `notebooks/3d_glaucoma_resolution_96_vs_128.ipynb` đọc theo thứ tự:

1. Biến môi trường `GF_DATA_DIR` (nếu đặt), hoặc
2. `data/glaucoma_all/` (mặc định), hoặc
3. `glaucoma_all/` ở gốc repo.

### Layout mong đợi (khớp `models/glaucoma/data.py`)

```
data/glaucoma_all/
  Training_volumes.npy     # uint8, shape (N, 1, 200, 200, 200) hoặc (N, 200, 200, 200)
  Training_labels.npy      # int64, shape (N,)
  Validation_volumes.npy
  Validation_labels.npy
  Test_volumes.npy
  Test_labels.npy
```

- Raw **200³ uint8**, không lưu bản hạ mẫu. Notebook tự resize 200→128/96 khi chạy.
- Có thể chỉ cần `Training_*` + `Validation_*` cho thí nghiệm resolution.

### Nguồn tải

- HF `Tqhuyen/harvard-oct-glaucoma-200` (consolidated .npy) — khuyến nghị.
- Hoặc archive Drive `MyDrive/MasterBKDN/Thesis/glaucoma_all_96.tar.zst` (bản 96³ cũ).

> Nếu thiếu file, notebook sẽ báo rõ và dừng (không tự tải/không ghi đè dữ liệu của bạn).
