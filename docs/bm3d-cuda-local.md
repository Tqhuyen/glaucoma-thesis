# Chạy BM3D trên CUDA (local, Windows) — hướng dẫn và kết quả pilot

Tài liệu này ghi lại **cách chạy BM3D bằng CUDA trên máy local** (Windows + RTX 3050 Laptop 4 GB),
kèm **kết quả pilot thật** trên 2 volume OCT 200³. Đây là khảo sát **tiền xử lý độc lập**, không phải
một phần của lượt huấn luyện cuối 10 epoch và không thay đổi thí nghiệm Bilateral đã chốt.

Công cụ dùng: **VapourSynth 79** + plugin **BM3DCUDA R2.15** (bản CUDA dựng sẵn cho Windows).
Không cần `nvcc`, không cần CUDA Toolkit, và **không cần bản PyTorch CUDA** (plugin tự dùng CUDA runtime).

---

## 1. Môi trường đã kiểm chứng

| Thành phần | Giá trị |
|---|---|
| OS | Windows 11 (10.0.26340) |
| Python | 3.12.0 |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4096 MiB, compute capability 8.6 |
| Driver | 610.60 |
| PyTorch | 2.13.0+**cpu**, `torch.cuda.is_available() = False` (không ảnh hưởng BM3D CUDA) |
| VapourSynth | **R79**, core API R4.2, 12 luồng, cache 8056 MB |
| Plugin | BM3DCUDA **R2.15** (CUDA variant), nạp qua `core.std.LoadPlugin` |
| API plugin | VapourSynth **API3** (có cảnh báo deprecated, vẫn chạy) |

**Kiểm chứng bằng file:**

| Tệp | SHA256 | Kích thước |
|---|---|---:|
| `VapourSynth-BM3DCUDA-R2.15.7z` | `a96dd2c3766debe20bac26c5e6134e1859f699e8b97d0ba14189614b675b39bb` | 2 535 698 B |
| `bm3dcuda.dll` | `a1cce5fc1d16ddb7200107c66c9edc68ea35568ef62af2b24528cdb144465076` | 16 223 232 B |

SHA256 của file `.7z` **khớp** digest công bố trong GitHub Release R2.15.

---

## 2. Cài đặt (một lần)

### 2.1. Cài VapourSynth (có bindings Python)

```powershell
python -m pip install --user vapoursynth
python -c "import vapoursynth as vs; print(vs.__version__)"
```

### 2.2. Tải plugin BM3DCUDA R2.15

```powershell
$dir = "$env:TEMP\bm3d"; New-Item -ItemType Directory -Force -Path $dir | Out-Null
$url = "https://github.com/WolframRhodium/VapourSynth-BM3DCUDA/releases/download/R2.15/VapourSynth-BM3DCUDA-R2.15.7z"
python -c "import requests,sys;open(sys.argv[2],'wb').write(requests.get(sys.argv[1],timeout=180).content)" $url "$dir\bm3dcuda-R2.15.7z"
```

Có thể tải bằng `Invoke-WebRequest $url -OutFile "$dir\bm3dcuda-R2.15.7z"` nếu không có `requests`.

### 2.3. Giải nén

`py7zr` **không** giải nén được archive này (lỗi `BCJ2 filter is not supported`).
Dùng `7z.exe`:

```powershell
& "C:\Program Files (x86)\Intel\Platform Flash Tool Lite\7z.exe" x "$dir\bm3dcuda-R2.15.7z" -o"$dir\extracted" -y
```

Kết quả:

```text
bm3d/extracted/VapourSynth-BM3DCUDA-R2.15/bm3dcuda.dll
```

### 2.4. Các bản release khác (nếu cần)

| File | Dùng khi |
|---|---|
| `VapourSynth-BM3DCUDA-R2.15.7z` | Bản CUDA chuẩn — **đang dùng** |
| `VapourSynth-BM3DCUDA_RTC-R2.15.7z` | Biên dịch kernel lúc chạy (NVRTC); có tham số thực nghiệm |
| `VapourSynth-BM3DCPU-R2.15.7z` | Bản CPU AVX2 để đối chiếu |

---

## 3. Nạp plugin và xác nhận CUDA

```python
import vapoursynth as vs
vs.core.std.LoadPlugin(path=r"...\bm3dcuda.dll")
print(vs.core.bm3dcuda.namespace)   # bm3dcuda
print([f for f in dir(vs.core.bm3dcuda) if not f.startswith("_")])  # BM3D, BM3Dv2, VAggregate
```

Xác nhận **thực thi trên GPU** bằng cách lấy mẫu `nvidia-smi` trong lúc chạy: VRAM tăng từ
~1,22 GB (nền) lên ~1,49 GB và `utilization.gpu > 0`. Plugin là bản CUDA nên **không có fallback CPU**;
nếu CUDA không dùng được, lệnh sẽ báo lỗi thay vì âm thầm chạy CPU.

Trong pilot, một bộ lấy mẫu chạy nền 0,05 giây/lần:
`nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits`.

---

## 4. Đưa dữ liệu OCT vào VapourSynth

BM3D CUDA cần clip **GRAYS float32**. Với volume OCT 200³, mỗi **B-scan = 1 frame 200×200**,
chuẩn hóa `uint8 / 255` → `[0,1]`. Adapter dùng `BlankClip` + `ModifyFrame`:

```python
import numpy as np, vapoursynth as vs

volume = np.load("raw_2404.npy")                      # (200,200,200) uint8
data = np.ascontiguousarray(volume.astype(np.float32) / 255.0)
clip = vs.core.std.BlankClip(width=200, height=200, format=vs.GRAYS, length=200, color=[0.0])

def fill(n, frame):
    writable = frame.copy()                           # frame trong callback là read-only
    np.asarray(writable[0])[:] = data[n]              # frame[0] là memoryview của plane
    return writable

src = vs.core.std.ModifyFrame(clip, clip, fill)
```

Lưu ý: frame truyền vào callback **read-only**, phải `frame.copy()` rồi ghi vào bản sao;
nếu không sẽ gặp `ValueError: assignment destination is read-only`.

---

## 5. Gọi BM3D hai bước (basic + Wiener)

```python
basic = vs.core.bm3dcuda.BM3D(src, sigma=10.0, radius=0, fast=False)
final = vs.core.bm3dcuda.BM3D(src, ref=basic, sigma=10.0, radius=0, fast=False)

result = np.stack([np.asarray(final.get_frame(n)[0]) for n in range(200)])  # (200,200,200) float32
```

- **Hai bước**: `basic` (hard-thresholding) và `final` (empirical Wiener, dùng `ref=basic`).
  Gọi một lần chỉ là bước basic.
- **`radius=0`**: lọc từng B-scan độc lập (đúng phạm vi khảo sát này). `radius>0` sẽ thêm frame lân cận
  và trở thành một biến thể khác.
- Lần gọi thứ hai vẫn truyền `src` gốc; `basic` chỉ là ảnh tham chiếu.
- `fast=False`: chế độ bộ nhớ thấp. `fast=True` tăng mức dùng bộ nhớ ~4× theo tài liệu.
- **200×200 không cần padding** lên 224/256.

### Cảnh báo đơn vị `sigma`

Đầu vào là float `[0,1]` nhưng **`sigma` dùng thang kiểu 8-bit**. Nếu có độ lệch chuẩn nhiễu
`σ_norm` trên `[0,1]`, giá trị API tương ứng là `255 × σ_norm`; **không** chia thêm cho 255.
Repo nói rõ cường độ lọc **không tương đương tuyệt đối** với các triển khai BM3D khác.

---

## 6. Kết quả pilot (đo thật)

Volume nguồn: `raw_2404.npy` và `raw_3294.npy` (200 B-scan mỗi volume), `sigma=10`, hai bước, `radius=0`.

### 6.1. Thời gian

| Volume | Frames | Thời gian | ms/frame | fps |
|---|---:|---:|---:|---:|
| 2404 | 200 | 0,186 s | 0,93 | 1074 |
| 3294 | 200 | 0,196 s | 0,98 | 1020 |
| 2404 lặp ×10 | 2000 | 2,075 s | **1,04** | 964 |

Tốc độ ổn định ≈ **1,0 ms/frame** ⇒ khoảng **0,21 giây cho một volume 200³** (200 B-scan, hai bước).

### 6.2. Mức dùng GPU

| Workload | GPU util mean | GPU util max | VRAM max |
|---|---:|---:|---:|
| 200 frame | 14–27 % | 27 % | ~1488 MiB |
| 2000 frame | **31,2 %** | **43 %** | ~1486 MiB |

Util không đạt 100 % vì mỗi frame 200×200 quá nhỏ, chi phí phát kernel chiếm ưu thế; **thông lượng**
(frame/giây) mới là chỉ số có ý nghĩa.

### 6.3. Kiểm tra khử nhiễu (chỉ số năng lượng tần số cao)

`hp_std` = trung bình độ lệch chuẩn của ảnh trừ làm mờ 3×3, tính cách 4 lát một lần.

| Ảnh | hp_std | Tỷ lệ / raw |
|---|---:|---:|
| raw | 13,26 | 1,00 |
| **BM3D CUDA** (`sigma=10`) | **9,53** | **0,72** |
| bilateral (`sigma_color=0,10`) | 6,29 | 0,47 |

BM3D giảm nhiễu **nhẹ hơn** cấu hình bilateral đã dùng. Đây là kết quả ở `sigma=10` **chưa tinh chỉnh**,
không phải kết luận chất lượng hay khuyến nghị loại bỏ bilateral.

---

## 7. Tái lập nhanh

```powershell
# 1) plugin
python scripts/bm3dcuda_pilot.py --dll "$env:TEMP\bm3d\extracted\VapourSynth-BM3DCUDA-R2.15\bm3dcuda.dll" `
  --volumes 2404,3294 --sigma 10

# 2) workload dài để đo thông lượng ổn định
python scripts/bm3dcuda_pilot.py --dll "$env:TEMP\...\bm3dcuda.dll" --volumes 2404 --repeat 10
```

Script: [`scripts/bm3dcuda_pilot.py`](../scripts/bm3dcuda_pilot.py).
Đầu ra ghi vào `outputs/bm3d_pilot/bm3d_pilot_<volume>.npy`.

---

## 8. Lưu ý bắt buộc

1. **Đầu ra vượt [0,1]**: pilot cho `min=-0,034`, `max=0,818` (tức −8,6 … 208,7 trên thang 0–255).
   Khi tích hợp phải **clip về [0,1]** rồi mới cast `uint8`, giống các phương pháp CPU khác.
2. **`sigma` chưa tinh chỉnh**: `sigma=10` chỉ là mốc thử; muốn so sánh công bằng phải chọn tham số trên
   tập phát triển, không theo tập kiểm tra.
3. **Giấy phép GPL-2.0-or-later**: dùng nội bộ cho luận văn thì ổn; nếu phân phối binary kèm ứng dụng thì
   cần rà soát nghĩa vụ GPL.
4. **API3 deprecated**: VapourSynth hiện vẫn hỗ trợ nạp plugin API3 nhưng cảnh báo sẽ bỏ trong tương lai;
   cần ghim phiên bản VapourSynth khi tái lập.
5. **Không phải 3D/temporal**: `radius=0` xử lý từng B-scan; không phải BM3D 3D trên cả volume.
6. **Chỉ là khảo sát tiền xử lý**: không thay đổi lượt huấn luyện cuối, không tự động đưa vào dataset.

---

## 9. Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân / cách sửa |
|---|---|
| `ModuleNotFoundError: No module named 'vapoursynth'` | `python -m pip install --user vapoursynth` |
| `py7zr ... BCJ2 filter is not supported` | Dùng `7z.exe x` thay vì `py7zr` |
| `Plugin ... already loaded` | Chỉ `LoadPlugin` **một lần** cho mỗi tiến trình |
| `ValueError: assignment destination is read-only` | `frame.copy()` trước khi ghi vào plane |
| `AttributeError: 'VideoFrame' object has no attribute 'planes'` | Dùng `frame[i]` (memoryview), không dùng `.planes` |
| Plugin API3 deprecated warning | Vô hại; ghim VapourSynth khi tái lập |
| Nghi ngờ chạy CPU | Kiểm tra VRAM tăng trong `nvidia-smi`; bản CUDA không có fallback CPU |
