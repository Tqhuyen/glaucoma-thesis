# Kế hoạch huấn luyện mô hình đa luồng 2D+3D — nền tảng & phương pháp tối ưu chi phí/thời gian

> Tài liệu liệt kê chi tiết các lựa chọn hạ tầng (máy local, Google Colab, vast.ai)
> và phương pháp tổ chức huấn luyện để tối ưu **chi phí** và **thời gian** cho thí nghiệm
> trong `notebooks/3d_glaucoma_multiview_2d3d.ipynb`.

---

## 1. Bối cảnh thí nghiệm cần huấn luyện

- Input chính: **khối OCT 3D 200×200×200** (raw, uint8 → /255) → nhánh 3D `Enc3D`.
- Kèm theo: **3 view 2D en-face 200×200** (aip_full, slab_aip, slab_mip) → 3 nhánh 2D `Enc2D`.
- Tổng tham số: Enc3D ≈ **2.0M**; mỗi Enc2D ≈ **11.2M**; toàn mô hình fusion ≈ **36.5M**.
- Yêu cầu GPU để chạy ở resolution đầy đủ; **activation tầng đầu của nhánh 3D rất tốn VRAM**.

Các cấu hình chạy (8 run, resume-safe, mỗi run log lên wandb `glaucoma-thesis`):

| # | Spec | Mô tả |
|---|---|---|
| 1 | `single3d` | baseline: chỉ nhánh 3D |
| 2–4 | `single2d-{aip,slabaip,slabmip}` | baseline: từng view 2D đơn lẻ |
| 5 | `fusion-concat` | concat embeddings → head |
| 6 | `fusion-add` | cộng phần tử theo phần tử |
| 7 | `fusion-mul` | nhân phần tử theo phần tử |
| 8 | `fusion-attn` | CLS + self-attention |

Tham số mặc định: `batch_size=2`, `grad_accum=8` (effective bs = 16), `epochs=15`, early-stop `patience=8`, AMP.

---

## 2. So sánh các nền tảng huấn luyện

### 2.1. Máy local hiện tại — KHÔNG NÊN train

- Trạng thái máy: **CPU-only** (torch `+cpu`, không CUDA, không GPU).
- Hậu quả: 3D CNN 200³ trên CPU chạy lâu gấp hàng chục lần GPU → ước tính **nhiều ngày đến nhiều tuần** cho 1 run.
- Kết luận: chỉ dùng local để **viết code, sinh hình cho báo cáo, chạy smoke test** (đã hoàn tất, chi phí 0 đồng, không đốt GPU).

### 2.2. Google Colab

| Ưu điểm | Nhược điểm |
|---|---|
| Không cần quản lý máy chủ | Giá mỗi giờ GPU **cao hơn** vast (tính theo "compute unit") |
| Workflow Drive quen thuộc | Session giới hạn ~12–24h, đôi khi bị ngắt/đứng hàng đợi |
| Môi trường Python sẵn, gắn liền notebook | Nếu dữ liệu chưa cache trong Drive → **mỗi session tải lại ~20–26 GB** |

- `T4` (free hoặc rẻ): **16 GB VRAM — KHÔNG đủ** 200³ full-resolution.
- `A100` (pay-as-you-go): đủ VRAM nhưng **tổng chi phí cao hơn** và rủi ro phải lặp lại bước tải/giải nén dữ liệu.

### 2.3. vast.ai (thuê GPU theo giờ) — KHUYẾN NGHỊ

| Loại instance (tham khảo, giá biến động) | VRAM | Ước giá spot | Ghi chú |
|---|---|---|---|
| RTX 4090 | 24 GB | ~$0.2–0.5/h | Rẻ nhưng 200³ dễ OOM → cần hạ batch/channel |
| **A100 40GB** | 40 GB | **~$0.4–0.9/h** | **Cân bằng tối ưu** cho config mặc định bs=2 |
| A100 80GB / H100 | 80 GB+ | ~$1–2/h | Chỉ cần khi muốn batch lớn hơn |

Ưu điểm quyết định:
- **Giá $/GPU-giờ thấp nhất** cho A100-class.
- Spot: rẻ hơn nữa; nếu bị trả máy thì **resume** chạy tiếp (pipeline đã resume-safe).
- **Giữ 1 instance suốt cả cụm sweep** → chỉ **tải dữ liệu 1 lần**, tiết kiệm rất nhiều thời gian so với việc tải lại mỗi phiên.

### 2.4. Bảng quyết định nhanh

| Nếu bạn… | Chọn |
|---|---|
| Muốn rẻ + chủ động thời gian | **vast.ai spot A100-40GB** |
| Đã có sẵn raw/denoised npy cache trong Drive, chấp nhận đắt hơn | Colab A100 pay-as-you-go |
| Chỉ test nhỏ / chỉnh code | Local CPU (smoke) — không tốn tiền |
| Muốn batch lớn / chạy song song nhiều run | A100-80GB / H100 (hoặc 2 instance) |

---

## 3. Yêu cầu bộ nhớ GPU (điểm nghẽn chính)

- Nhánh 3D giữ resolution cao ở các tầng đầu (200³) → activation rất lớn dù model chỉ ~2M tham số.
- Khuyến nghị an toàn: **VRAM ≥ 40 GB** với `batch_size=2` + AMP.
- Nếu chỉ có **24 GB**: bắt buộc `batch_size=1`, `grad_accum=16`, giảm kênh các stage đầu (vd `(16,32,64,128)`) — **chấp nhận đánh đổi chất lượng/hiệu năng**.

> Quy tắc: trước khi quét cả 8 run, luôn chạy **probe 1 run × 2 epoch ở 200³** để đo `s/ep` và VRAM thực tế.

---

## 4. Các phương pháp tổ chức huấn luyện (giảm chi phí & thời gian)

| Phương pháp | Mô tả | Lợi ích |
|---|---|---|
| **Smoke test trước trên CPU** | Chạy code path với volume tổng hợp, model mini (`MV_SMOKE=1`) | Đảm bảo không bug trước khi thuê GPU → không đốt tiền vào lỗi |
| **VRAM/time probe** | 1 run thật 2 epoch trên máy thuê | Xác định OOM & tốc độ để chọn instance đúng |
| **Spot + resume** | Thuê spot, checkpoint/resume | Giá rẻ; mất máy không mất công sức |
| **Một instance cho cả sweep** | Không tạo instance mới giữa các run | Dữ liệu tải 1 lần (~20–26 GB), tiết kiệm nhiều GPU-giờ |
| **AMP (mixed precision)** | bf16/fp16 trên GPU | Giảm VRAM + tăng tốc |
| **Early-stop & epochs hợp lý** | `patience=8`, theo dõi val | Tránh train thừa khi đã hội tụ |
| **Chạy theo thứ tự ưu tiên** | `single3d` → `fusion-concat` → `fusion-attn` trước | Có kết quả chính sớm; `add`/`mul` chạy sau nếu còn hạn mức |
| **Log wandb từng run** | project `glaucoma-thesis` | Theo dõi từ xa, dừng sớm khi thấy bất thường |
| **tmux + log tail** | Chạy nền, theo dõi bằng `tail -f` | Không phụ thuộc SSH mở liên tục |
| **Copy kết quả rồi hủy máy** | `best_*.pt` + JSON → Drive, destroy instance | Không trả tiền cho thời gian máy chạy không |

### Chi phí ước lượng (kinh nghiệm tham khảo)

| Hạng mục | GPU-giờ | Ước $ (A100-40 spot) |
|---|---|---|
| Chuẩn bị dữ liệu (tải/giải nén raw ~26 GB) | 1–2 | ~$1–2 |
| Probe VRAM | 0.5–1 | < $1 |
| Sweep 8 run (nếu ~60–150 s/ep × 15 ep) | 6–12 | ~$4–12 |
| **Tổng cả cụm** | ~8–15 | **~$5–15** |

> Colab cùng khối lượng ước tính đắt gấp ~1.5–2× và có rủi ro phải tải lại dữ liệu khi session reset.

---

## 5. Các bước thực hiện cụ thể

1. **Pha 0 — Local (0$):** chạy smoke đã hoàn tất; giữ local cho figure/viết code.
2. **Pha 1 — Thuê vast spot A100-40GB** (ubuntu + python ≥ 3.10):
   ```
   git clone <repo> && cd <repo>
   pip install torch numpy matplotlib wandb requests pandas
   export HF_TOKEN=hf_xxx WANDB_API_KEY=wandb_xxx
   ```
   Tải dữ liệu 1 lần về `/workspace/data` (raw npy caches: `Training|Validation|Test_volumes.npy` + `_labels.npy`); nếu tự dựng từ zip thì dùng processor headless.
3. **Pha 2 — Probe:** chạy 1 run 200³, 2 epoch → ghi nhận `s/ep`, VRAM. OOM → hạ `batch_size=1`/channel.
4. **Pha 3 — Full sweep:** chạy cell experiment (resume-safe), tmux `tail -f`, theo dõi wandb; chạy theo thứ tự ưu tiên ở mục 4.
5. **Pha 4 — Thu hoạch:** copy `best_*.pt` + JSON về Drive → **destroy instance ngay**.

---

## 6. Kết luận

- **Không train trên local** (CPU-only, không khả thi với 200³).
- **Tối ưu nhất: vast.ai spot A100-40GB**, giữ 1 instance suốt cụm thí nghiệm,
  kết hợp probe → smoke → resume + W&B → thu hoạch rồi hủy máy.
- Tổng chi phí kỳ vọng cho cả cụm thí nghiệm nhóm A: **khoảng $5–15**.
