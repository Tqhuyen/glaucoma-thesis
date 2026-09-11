# Cấu trúc & quy tắc chung khi tạo notebook train model

Tài liệu này chuẩn hoá **cấu trúc cell** và **quy tắc bắt buộc** khi tạo notebook train/sweep/nghiên cứu
mô hình trong repo. Nguồn tham chiếu chính là notebook sweep 2D–3D fusion
[`notebooks/3d_glaucoma_multiview_sota_sweep_xai.ipynb`](../notebooks/3d_glaucoma_multiview_sota_sweep_xai.ipynb),
đối chiếu với [`notebooks/3d_glaucoma_final_2x2d_3d_crossgate.ipynb`](../notebooks/3d_glaucoma_final_2x2d_3d_crossgate.ipynb)
(recovery/W&B) và các quy tắc trong `AGENTS.md`. Khi tạo notebook mới, coi đây là checklist bắt buộc.

## 1. Cấu trúc cell chuẩn

Thứ tự dưới đây lấy từ sweep notebook; notebook train đơn lẻ có thể bỏ các phần sweep/X-AI nhưng
**không được bỏ** phần 2–4, 9, 11–12.

| # | Cell | Nội dung | Ví dụ trong sweep |
|---|---|---|---|
| 1 | Title/overview (markdown) | Bài toán, dataset, pipeline, runtime budget, cách chạy, cách bật smoke | Cell 1: dataset + "Honest runtime expectation" |
| 2 | Setup | `IN_COLAB`/`SMOKE` guard, clone repo, `pip install` (timm/monai/wandb/sklearn/skimage/wandb…), import, `DEVICE`, seed | Cell 2–4 |
| 3 | Config | **Một nguồn chân lý duy nhất** (`CFG`/hằng số): data paths, `STORE_RES`, `MODEL_RES3D`/`RES2D`, epochs, bs, grad-accum, lr, tier/spec, `SAVE_DIR`/`DRIVE_DIR`, env override | Cell 5–6 |
| 4 | Data | Tải từ HF **có xác thực token**, chỉ tải split/pattern dùng đến; build storage ở `STORE_RES` theo config (200/128/96/…, idempotent, cache) + cache view/depth-axis một lần per split; `manifest.json` | Cell 7–10 |
| 5 | Dataset + loaders | memmap đọc per-sample, view đã cache, augmentation deterministic theo `(seed, epoch, index)`, loader lazy/cache theo `(res3d, res2d, bs, …)`; nhánh không dùng thì **không đọc** modality đó | Cell 11 |
| 6 | Branch encoder registry (nếu sweep) | Mỗi encoder trả embedding pooled `out_dim`, input `(B,1,D,H,W)` trong `[0,1]`; 2D `(B,1,H,W)` | Cell 12–18 |
| 7 | Fusion + model | Các phép fusion (concat/add/mul/attn/crossgate/mamba/film) + model đa nhánh | Cell 19–20 |
| 8 | Metrics | Bộ metric lâm sàng đầy đủ + calibration + threshold + bootstrap CI (chi tiết mục 2.8) | Cell 21–23 |
| 9 | Probe + train loop | `probe()` kiểm tra 1 forward/backward, params, VRAM; `train_one()` resume-safe, lưu JSON per-run | Cell 24–27 |
| 10 | Sweep driver (nếu sweep) | Tier/spec, mỗi run xong ghi `results.json` (local + Drive) ngay, figure per-run xuất ngay | Cell 28–31 |
| 11 | X-AI (nếu có) | Grad-CAM 3D/2D, occlusion, integrated gradients, branch importance, LIME; lưu PNG + markdown | Cell 32–38 |
| 12 | Chạy + reporting | Chạy sweep/train, render bảng CSV/PNG, markdown report, **Drive sync + `run.finish()`** | Cell 39–49 |
| 13 | Notes/limitations (markdown) | Cảnh báo single-seed, metric thiếu ghi `—`, confound, cách resume | Cell 50 |

## 2. Chi tiết từng mục

### 2.1 Title/overview (markdown)
Mục tiêu + kiến trúc tóm tắt, dataset, chính sách resolution (`STORE_RES` theo config, model input có thể resize
on-the-fly), runtime budget thật theo backbone (phút/run), cách chạy top-to-bottom, cách bật `*_SMOKE`, nơi lưu
artifact (local + Drive + W&B). Ghi rõ phạm vi: train đơn lẻ, fine-tune hay sweep nhiều tier.

### 2.2 Setup
`IN_COLAB`/`SMOKE` guard; clone/pull repo; pip theo nhóm (`timm`, `monai`, `scikit-learn`, `scikit-image`, `wandb`,
`huggingface_hub`, `python-dotenv`); mount Drive; đọc `HF_TOKEN`/`WANDB_API_KEY` từ `.env` hoặc Colab Secrets;
import core script; `DEVICE`; seed; AMP dtype (bf16 nếu có, ngược lại fp16 + `GradScaler`).

### 2.3 Configuration
Một nguồn chân lý duy nhất (`CFG`/hằng số): data paths, `STORE_RES` (config-driven, ví dụ 200/128/96 — không
hardcode 200), `MODEL_RES3D`/`RES2D`, epochs, batch_size, grad_accum, lr, weight_decay, patience, `log_every`,
`SAVE_DIR`/`DRIVE_DIR`/`RUN_GROUP`, spec/tier, cờ `RUN_XAI`, env override. Validate sớm (fail fast) trước khi
build data/model; đổi `RUN_GROUP` khi đổi cấu hình train.

### 2.4 Data
- **Resolution linh hoạt theo config**: storage ở `STORE_RES` có thể là 200, 128, 96, … tùy bài toán; model input
  (`RES3D`/`RES2D`) khai báo riêng và resize on-the-fly. Không hardcode 200; kiểm tra shape theo `STORE_RES`.
- **Luôn tải HF bằng authenticated credentials**: `HF_TOKEN` từ `.env`/Colab Secret, truyền `token=` vào
  `snapshot_download`/`hf_hub_download`; bật `hf_transfer` nếu có để đạt tốc độ tối đa. Real run thiếu token → fail sớm.
- **Chỉ tải phần dùng đến**: dùng `allow_patterns` (hoặc tải từng file) cho đúng split/loại file cần; không
  `snapshot_download` cả repo.
- **Tái sử dụng & cache**: build storage idempotent (reuse qua nhiều notebook/run), ghi `manifest.json`; cache view
  en-face (`aip_full`, `slab_aip`, `slab_mip`) + depth-axis một lần per split.
- **Dữ liệu đã xử lý (khử nhiễu, view, augmentation nặng)**: xem có nên upload lên HF dataset repo (versioned theo
  method + params + implementation hash) để lần sau khỏi tính lại; chỉ upload khi thực sự tiết kiệm thời gian và
  được người dùng đồng ý, kèm identity để tái lập. Việc rẻ/thay đổi liên tục thì giữ local/Drive, không upload.

### 2.5 Dataset + loaders
Đọc volume bằng memmap theo từng sample; view đọc từ cache; augmentation deterministic theo `(seed, epoch, index)`;
loader lazy + cache theo `(res3d, res2d, bs, load3d, loadviews)`; nhánh không dùng thì không đọc modality đó;
Windows `num_workers=0`; item trả `(x, views, y)` và 3D là `(1,D,H,W)` (không dùng `raw[None,None]`).

### 2.6 Branch encoders (registry)
Mỗi 3D encoder `(B,1,D,H,W) -> out_dim` pooled; mỗi 2D encoder `(B,1,H,W) -> out_dim`; ghi rõ yêu cầu resolution
(chia hết patch/window 16/32), norm phù hợp batch nhỏ (GroupNorm/LayerNorm), pooling giữ depth `(2,2,1)`.

### 2.7 Fusion + model
Các phép fusion tham chiếu (concat/add/mul/attn/crossgate/mamba/film); mặc định CrossGate (3D làm query, 2D làm
key/value, gate học được). `forward` trả logits thô để ghép `CrossEntropyLoss`; expose `fuse()`/`embed()` khi cần
X-AI/phân tích embedding.

### 2.8 Metrics (đủ bộ theo sweep)
Bộ metric đầy đủ theo `metrics_full` của sweep (mở rộng trong `fm.full_metrics`):
`acc`, `balanced_acc`, `precision/PPV`, `recall/sensitivity`, `specificity`, `npv`, `f1`, `f1_macro`, `mcc`,
`kappa`, `youden`, `auc_roc`, `auc_pr`, `ece`, `logloss`, `brier`, `tn/fp/fn/tp`, `n`.

Nhịp log (train từng bước, val/test từng epoch):
- **Train mỗi optimizer step** trên cửa sổ grad-accum: `train/<metric>` cho toàn bộ bộ metric + `train/loss`,
  `train/acc` (running epoch), `train/lr`, `train/epoch`, `progress/step`.
- **Val mỗi epoch**: `val/<metric>` cho toàn bộ bộ metric; dùng val AUC/PR-AUC để chọn best + early stop.
- **Test mỗi epoch**: `test/<metric>` + `test/epoch` với cùng bộ metric.
- **Chốt cuối** (best state): `calibrated_report` fit temperature + threshold (Youden-J) trên val rồi tính
  calibrated `train`/`val`/`test` + bootstrap CI; `ft.log_report` ghi bảng so sánh `report/split_table` và summary
  `train|val|test/<metric>` (+ `_lo`/`_hi` cho CI).
- AUC/PR-AUC trên cửa sổ nhỏ có thể NaN khi cửa sổ chỉ có một lớp — đọc xu hướng theo epoch, đừng chọn theo bước.

### 2.9 Probe + train loop
`probe()` chạy 1 forward/backward (params, peak VRAM, fallback batch 1 khi OOM). Train: AdamW + cosine + warmup +
AMP + grad-accum + grad-clip + early stop; lr theo effective batch; best theo val AUC; checkpoint atomic; `history`
lưu đầy đủ `val`/`test` metrics mỗi epoch để vẽ và đối chiếu sau này.

### 2.10 Sweep driver (nếu sweep)
Mỗi spec override `res3d`/`res2d`/`batch_size`/`epochs`; mỗi run xong ghi ngay `results.json` (local + Drive) và
figure per-run; interrupt giữ các run đã xong; chọn tier qua env (`*_TIER_*`).

### 2.11 X-AI (chạy sau khi train xong)
Trên best model + mẫu test cân bằng lớp:
- **Grad-CAM 3D** (conv cuối/patch-embed) và **Grad-CAM 2D** từng view.
- **Occlusion sensitivity** (3D) và **integrated gradients** (2D view, tùy chọn 3D 64³).
- **Fusion attention** + **branch drop** (leave-one-branch-out) trả lời nhánh nào quyết định.
- **LIME** mức view (tùy chọn).

Lưu local + Drive: PNG, `XAI_REPORT.md`, `fusion_xai.pt` (weights/gate/branch_drop/attention), sync **ngay khi sinh ra**.
W&B: ảnh `wandb.Image`, bảng **`xai/fusion_table`** (`branch`, `crossgate_attention`, `drop_probability`) và summary
`xai/gate`, `xai/drop_*`, `xai/attention_*` để so sánh X-AI giữa các model.

### 2.12 Reporting
Learning curves, ROC, PR, calibration, confusion; `metrics.json` + CSV + bảng split; log W&B; sync Drive;
`run.finish(exit_code=...)`; nếu có X-AI thì bảng/giá trị ở mục 2.11.

### 2.13 Notes/limitations (markdown)
Single-seed, metric thiếu ghi `—`, ảnh hưởng resolution/denoise, cách resume, ngoại lệ preset (ví dụ preset
fine-tune có thể để `RUN_XAI=False` để tiết kiệm GPU nhưng phải ghi rõ lý do).

## 3. Quy tắc bắt buộc (hard rules)

### 3.1 W&B — bắt buộc cho mọi run
- `load_env_file()` (hoặc Colab Secret `WANDB_API_KEY`) trước `wandb.init()`; project `glaucoma-thesis`.
- Helper `init_wandb(run_name, config)` idempotent (`WANDB_RUN is None`), gọi **trước khi train**.
- Log **live** đúng prefix: `train/...` cho train, `val/...` cho val, `test/...` cho test; không trộn history.
- Log figure bằng `wandb.Image`; kết thúc bằng `run.summary.update(...)` + `run.finish(exit_code=...)`.
- Không bao giờ tắt/bỏ wandb để "tiết kiệm thời gian". Chỉ smoke nội bộ được phép `mode="offline"`.

### 3.2 Drive — mọi artifact đều phải lên Drive
- Mount Drive có guard (`os.path.ismount`) để chạy headless không lỗi.
- Root: `/content/drive/MyDrive/MasterBKDN/Thesis/<experiment>[_figures]`; giữ đúng path các notebook cũ
  (`sota_200`, `multiview`, `denoise_sweep`, `3dino_ft`, `final_2x2d_3d_crossgate`, …).
- Figure/ROC/PR/calibration/confusion/history, CSV/JSON/report, checkpoint: copy **ngay khi sinh ra**
  (không đợi cuối run) để interrupt vẫn giữ kết quả.
- "Chỉ lưu local hoặc git" là **chưa xong**.

### 3.3 Dữ liệu
- Storage resolution do config quyết định (`STORE_RES`: 200/128/96/…), không hardcode; model input khai báo riêng
  và resize on-the-fly (cast float trước `F.interpolate`).
- **Luôn tải HF có xác thực** (`HF_TOKEN` + `token=`); chỉ tải split/pattern thực sự dùng (`allow_patterns`),
  không tải cả repo; ưu tiên `hf_transfer` để tăng tốc.
- Cache/tái sử dụng dữ liệu đã xử lý; quyết định có nên upload lên HF (versioned, kèm identity) khi xử lý đắt và
  sẽ dùng lại; chỉ upload khi được yêu cầu/đồng ý.
- `data/` bị gitignore — không commit/push dữ liệu vào git.
- 2D view (aip/slab) chiếu từ khối storage và cache một lần per split (cache dùng được cho mọi `RES3D`/`RES2D`).

### 3.4 Resolution naming (bẫy đã từng gặp)
- Đặt tên **khác nhau** cho storage và model input: `STORE_RES` (200/128/96/…), `MODEL_RES3D`/`RES3D`, `RES2D`.
- Resize on-the-fly; cast sang float **trước** `F.interpolate` (triliear lỗi với uint8).
- 3DINO-ViT cần input chia hết cho patch 16 (112³); transformer khác (Swin/UNETR) chia hết 16/32.

### 3.5 Model & loss
- `forward()` trả logits; loss `CrossEntropyLoss` trên raw logits (2 lớp, có class weights).
- Batch nhỏ (2–4) → **GroupNorm/LayerNorm**, tránh BatchNorm (grad-accum không cứu BN).
- Pooling 3D giữ depth: anisotropic `(2,2,1)` cho stage đầu, `AdaptiveAvgPool3d(1)` chỉ ở cuối.
- AMP: bf16 khi có, ngược lại fp16 + `GradScaler`; grad-accum; grad-clip; cosine + warmup.

### 3.6 Chọn checkpoint & metric
- Chọn best theo **val AUC/PR-AUC**, không bao giờ theo test.
- Test chỉ chấm ở epoch tốt nhất; threshold chọn bằng Youden-J trên val; temperature scaling fit trên val.
- Bộ metric tối thiểu: acc, balanced acc, precision/PPV, recall/sensitivity, specificity, NPV, F1, MCC,
  AUC-ROC, PR-AUC, ECE (+ bootstrap CI); metric thiếu ghi `—`.

### 3.7 Resume-safe & smoke
- Mỗi run ghi kết quả ngay vào `results.json` (local + Drive) và checkpoint atomic; interrupt giữ nguyên
  các run đã hoàn thành, chạy lại thì resume chứ không chạy lại từ đầu.
- Trước khi đốt GPU: `*_SMOKE=1` chạy CPU với dữ liệu synthetic nhỏ, **không download**; toàn bộ cell
  phải pass. Với pipeline đầy đủ, `make sanity` phải overfit 10 mẫu trước khi nghi ngờ dữ liệu.

### 3.8 Code style & kỹ thuật
- Line length 120; `ruff`; không thêm comment khi không được yêu cầu; theo pattern registry/config-driven.
- Không dùng `vols[:,0]` trên memmap 5D (load 26 GB) — index `vols[i]` rồi squeeze per sample.
- Windows: `num_workers=0` (mmap + multiprocessing dễ segfault).
- Augmentation per sample phải deterministic theo `(seed, epoch, index)`.
- Biến thể mới: logic tái sử dụng đặt trong `scripts/*.py`, notebook chỉ gọi; unit-test core mới.

### 3.9 Colab
- Env override cho mọi chế độ: `*_SMOKE`, `*_SWEEP`, `*_XAI`, `*_TIER_*`, `*_RESUME`.
- Guard `IN_COLAB`/`SMOKE` cho clone/pip/mount; token HF/W&B lấy từ `.env` hoặc Colab Secrets.
- Khi sửa xong notebook/config: nhắc người dùng **restart opencode** nếu đổi config, và push repo để Colab clone được.

### 3.10 Hiệu năng & tối ưu tốc độ (mặc định)
- **Đo trước, tối ưu sau**: dùng `probe()` (params, peak VRAM) + `sec/epoch`; chỉ tối ưu hot path thực sự chậm.
- **Tính trên GPU, hạn chế đồng bộ host**: tránh `.item()`/`print` mỗi micro-batch; gom log theo optimizer step;
  chỉ chuyển tensor sang CPU khi thật cần (metrics/log/artifact).
- **Data pipeline**: loader lazy + cache theo key; không đọc modality không dùng; CUDA dùng `pin_memory=True` +
  `.to(device, non_blocking=True)`; Windows `num_workers=0`; cache view/denoise/augmentation thay vì tính lại.
- **Tải dữ liệu**: `allow_patterns` chỉ tải phần dùng; token + `hf_transfer` để tối đa băng thông.
- **AMP/TF32**: bf16 nếu có, ngược lại fp16 + GradScaler; trên CUDA bật `cudnn.benchmark=True` và TF32 cho
  matmul/conv khi phần cứng hỗ trợ; `torch.compile` chỉ dùng sau khi kiến trúc đã ổn định.
- **Batch theo VRAM**: tăng batch/`grad_accum` tới ngưỡng an toàn; model input ở mức đủ dùng, không cao hơn
  mức có ích; tránh OOM giữa run.
- **Phân tích nặng** (MINE/surrogate/probe/IB, X-AI): giới hạn subset/steps/PCA-dim; cache embedding thay vì
  forward lại; log bảng/giá trị thay vì lưu tensor lớn.
- **Không tính lại cái đã có**: tái sử dụng cache (local/Drive) và cân nhắc upload HF cho preprocessing đắt.

## 4. Checklist trước khi giao notebook

- [ ] Đúng thứ tự cell mục 1; config một nguồn duy nhất, có env override.
- [ ] `*_SMOKE=1` chạy hết cell trên CPU, synthetic, không download — pass.
- [ ] W&B init + log live đúng prefix + figures as `wandb.Image` + `summary.update` + `finish`.
- [ ] **Train log đủ bộ metric mỗi optimizer step; val + test log đủ bộ metric mỗi epoch.**
- [ ] **Chốt cuối có calibrated `train`/`val`/`test` + bootstrap CI và bảng `report/split_table` trên W&B.**
- [ ] **Sau train chạy X-AI trên best model, log `xai/fusion_table` + giá trị `xai/*` lên W&B (hoặc ghi rõ ngoại lệ).**
- [ ] Mọi artifact (figure/model/CSV/report/X-AI) sync Drive ngay khi sinh ra.
- [ ] HF download có `HF_TOKEN` + `allow_patterns` chỉ tải phần dùng; real run thiếu token fail sớm.
- [ ] `STORE_RES` lấy từ config (200/128/96/…), không hardcode; check shape theo `STORE_RES`; tách khỏi `RES3D`/`RES2D`.
- [ ] Quyết định cache/tái sử dụng/upload dữ liệu đã xử lý (khử nhiễu/view/augmentation) được ghi rõ.
- [ ] AMP bật khi CUDA + `pin_memory`/`non_blocking`; hạn chế `.item()`/đồng bộ host trong vòng train.
- [ ] Loader lazy/cache, chỉ đọc dữ liệu dùng đến; preprocessing đắt được cache/tái sử dụng.
- [ ] Đo `sec/epoch` + peak VRAM; phân tích nặng (MI/X-AI) có giới hạn subset/steps.
- [ ] Best checkpoint theo val AUC; threshold/calibration fit trên val, không dùng test.
- [ ] Resume-safe: `results.json` append + checkpoint atomic.
- [ ] `ruff check` sạch cho file Python mới; unit-test core mới pass.
- [ ] Notes/limitations được ghi rõ.

## 5. Notebook tham chiếu

| Notebook | Dùng để tham khảo |
|---|---|
| [`3d_glaucoma_multiview_sota_sweep_xai.ipynb`](../notebooks/3d_glaucoma_multiview_sota_sweep_xai.ipynb) | Cấu trúc sweep đầy đủ: registry backbone, fusion ablation, X-AI, resume-safe, reporting |
| [`3d_glaucoma_train_3branch_crossgate.ipynb`](../notebooks/3d_glaucoma_train_3branch_crossgate.ipynb) | Train mô hình chính 3 nhánh (1×3D ResNeXt + 2×2D MaxViT + CrossGate): 3D chạy **96³** resize on-the-fly, **view 2D chiếu từ raw 200³**; full metrics 3 tập, X-AI + bảng W&B, resume, smoke |
| [`3d_glaucoma_final_2x2d_3d_crossgate.ipynb`](../notebooks/3d_glaucoma_final_2x2d_3d_crossgate.ipynb) | Train 1 model với recovery contract (`last.pt`/`best.pt`/emergency weights), warm-start, calibrated report |
| [`3d_glaucoma_crossgate_resnext96_infotheory.ipynb`](../notebooks/3d_glaucoma_crossgate_resnext96_infotheory.ipynb) | Biến thể resolution + phân tích lý thuyết thông tin (probe/MINE/surrogate/IB) |
| [`3d_glaucoma_resolution_96_128_200.ipynb`](../notebooks/3d_glaucoma_resolution_96_128_200.ipynb) | So sánh resolution trên cùng backbone |
| [`3d_glaucoma_denoise_gpu_compare.ipynb`](../notebooks/3d_glaucoma_denoise_gpu_compare.ipynb) | So sánh phương pháp khử nhiễu + metric ảnh |

Core tái sử dụng: `scripts/final_model.py` (ResNeXt3D/Timm2D/CrossGate/metrics/X-AI), `scripts/final_training.py`
(Trainer recovery/W&B), `scripts/information_theory.py` (probe/MINE/MIC/surrogate/IB),
`scripts/resolution_study.py` (metric numpy), `scripts/compare_denoise_methods.py` (khử nhiễu).
