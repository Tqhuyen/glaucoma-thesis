# 3 view en-face có trùng lặp đến mức chỉ cần 1 nhánh 2D không?

**Câu hỏi:** ba biểu diễn 2D (`aip_full`, `slab_aip`, `slab_mip`) sinh từ volume OCT 3D có khác nhau đủ để
biện minh cho **3 nhánh 2D** không? Nếu chúng gần như trùng nhau thì có thể rút còn **1 nhánh 2D**.

**Dữ liệu phân tích:** 150 volume **raw 200³** (split Validation, Harvard-GF), dùng đúng code chiếu của notebook
sweep (`depth_axis` → `to_depth_last` → dải RNFL `peak ± 16` → `mean`/`max`).
Script: `scripts/analyze_views.py` → `figures/view_analysis/`.

---

## 1. Chỉ số tương đồng giữa các cặp view (trung bình trên 150 volume)

| Cặp view | Pearson r | Spearman | SSIM(2D) | NMI | MAD (0–255) |
|---|---:|---:|---:|---:|---:|
| `aip_full` \| `slab_aip` | 0.463 | 0.438 | **0.763** | 0.091 | 19.5 |
| `aip_full` \| `slab_mip` | 0.390 | 0.372 | **0.229** | 0.061 | 73.5 |
| `slab_aip` \| `slab_mip` | **0.838** | 0.791 | 0.383 | 0.224 | 56.1 |

`aip_full` = mean toàn chiều sâu; `slab_aip` = mean trong dải RNFL; `slab_mip` = max trong dải RNFL.

## 2. Diễn giải

- **Không có cặp nào là bản sao gần đúng.** Không cặp nào đạt đồng thời r Pearson cao **và** SSIM cao (> 0.9);
  SSIM cao nhất chỉ **0.763** (aip_full vs slab_aip).
- **`aip_full` và `slab_aip` tương đối giống nhau** (SSIM 0.763, MAD 19.5) vì cùng là phép **mean**, chỉ khác dải chiếu
  ⇒ phần lớn thông tin tần số thấp **dùng chung**, có thể coi là **bán trùng lặp**.
- **`slab_mip` khác biệt rõ** với cả hai (SSIM 0.229–0.383; MAD 56–74) — vì là phép **max**, giữ cấu trúc sợi/mạch
  và đỉnh cường độ khác hẳn mean ⇒ **mang thông tin bổ trợ thực sự**.
- `slab_aip` vs `slab_mip` có Pearson cao (0.838) nhưng **SSIM thấp (0.383)** và **MAD lớn (56)** ⇒ tương quan điểm ảnh
  cao do cùng nền sáng, nhưng **cấu trúc cục bộ và phân bố cường độ khác nhau**.

## 3. Đối chiếu với kết quả task (sweep 2D)

Từ [`2d-feature-extraction-sweep.md`](2d-feature-extraction-sweep.md) (cùng backbone Tiny2D, test AUROC):

| View | Test AUROC | Test MCC |
|---|---:|---:|
| AIP toàn ảnh | 0.6605 | 0.1704 |
| Slab AIP | 0.6526 | 0.2838 |
| **Slab MIP** | **0.7242** | **0.3210** |

→ **AIP và Slab AIP cho kết quả gần như nhau** (0.660 vs 0.653) — **khớp** với việc chúng bán trùng lặp (SSIM 0.763).
→ **Slab MIP vượt trội và khác biệt** — khớp với việc nó có cấu trúc khác (SSIM thấp).

## 4. Kết luận & khuyến nghị

**Không nên rút xuống 1 nhánh 2D một cách mù quáng** — nhưng **3 nhánh là thừa**:

- **1 nhánh 2D (chỉ `slab_mip`)** là phương án **tối thiểu** hợp lý (view mạnh nhất, khác biệt nhất), nhưng sẽ mất
  thông tin tần số thấp của `aip_full`/`slab_aip`.
- **2 nhánh 2D (`slab_mip` + `aip_full`)** là **điểm cân bằng tốt nhất**: một view "max/RNFL" khác biệt + một view
  "mean toàn ảnh" bổ trợ; bỏ bớt `slab_aip` vì bán trùng lặp với `aip_full` (SSIM 0.763) và không cho lợi ích task.
- **3 nhánh** chỉ nên giữ nếu ablation chứng minh fusion 3 view **vượt 2 view**; dữ liệu hiện tại **chưa** ủng hộ điều đó.

**Đề xuất cho mô hình chính:**
1. **Cấu hình khuyến nghị:** `1×3D (ResNeXt3D) + 2×2D (SlabMIP + AIP full, MaxViT-Tiny)`, fusion CrossGate.
2. Chạy kèm **ablation 1 nhánh 2D** (chỉ SlabMIP) và **3 nhánh** để chứng minh số nhánh tối ưu bằng số liệu.
3. Chỉ số quyết định: **val/test AUROC + PR-AUC + balanced accuracy + MCC** (không dùng accuracy đơn thuần).

## 5. Hạn chế

- Phân tích trên **ảnh raw**; view trên dữ liệu **đã khử nhiễu** có thể khác (nên chạy lại `scripts/analyze_views.py`
  trên tập denoised).
- Chỉ số điểm ảnh (SSIM/NMI) nhạy cảm với nhiễu speckle; cần đọc kèm kết quả task.
- NMI ước lượng từ histogram 32 bin có thể bị chệch thấp trên dữ liệu liên tục.
- Kết luận dựa trên **1 bộ view/1 dải RNFL**; thay đổi `half` (bề dày dải) có thể ảnh hưởng.

## 6. Đoạn mô tả dùng trong báo cáo

> Ba biểu diễn en-face không trùng lặp hoàn toàn nhưng mức độ khác biệt không đồng đều: `aip_full` và `slab_aip`
> tương đối giống nhau (SSIM 0,763) và cho hiệu năng phân loại gần như nhau (AUROC 0,660 vs 0,653), trong khi
> `slab_mip` khác biệt rõ (SSIM 0,229–0,383) và đạt AUROC cao nhất (0,724). Điều này cho thấy **hai nhánh 2D**
> (`slab_mip` + `aip_full`) là đủ để bao phủ thông tin bổ trợ, còn nhánh thứ ba (`slab_aip`) phần lớn dư thừa.
> Số nhánh tối ưu sẽ được xác nhận bằng ablation 1/2/3 nhánh trên cùng pipeline.
