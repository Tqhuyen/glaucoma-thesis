# Cấu trúc & quy tắc chung khi tạo notebook train model

Tài liệu này chuẩn hoá **cấu trúc cell** và **quy tắc bắt buộc** khi tạo notebook train/sweep/nghiên cứu
mô hình trong repo. Nguồn tham chiếu chính là notebook sweep 2D–3D fusion
[`notebooks/3d_glaucoma_multiview_sota_sweep_xai.ipynb`](../notebooks/3d_glaucoma_multiview_sota_sweep_xai.ipynb),
đối chiếu với [`notebooks/3d_glaucoma_final_2x2d_3d_crossgate.ipynb`](../notebooks/3d_glaucoma_final_2x2d_3d_crossgate.ipynb)
(recovery/W&B) và các quy tắc trong `AGENTS.md`. Khi tạo notebook mới, coi đây là checklist bắt buộc.

## 1. Cấu trúc cell chuẩn

Thứ tự dưới đây lấy từ sweep notebook; notebook train đơn lẻ có thể bỏ các phần sweep/X-AI nhưng
**không được bỏ** phần 2–4, 9, 11–12.

**Nguyên tắc chia cell:** mỗi cell làm **một việc** hoặc một nhóm việc cùng loại; tách cả **data**, **train**,
**eval**, **info/X-AI** và reporting — không chỉ tách riêng phần info. Giữ cell ngắn, ít side effect chéo, biến
trung gian ở global để có thể **chạy lại đúng cell đó** khi debug/sửa lỗi thay vì chạy lại cả notebook. Cell quá
dài (> ~80 dòng) phải tách theo bước và đặt markdown heading rõ ràng.

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
| 11b | Phân tích tùy chọn (nếu có) | Info-theory/embedding analysis chạy **trong cùng notebook** qua cờ `RUN_INFO`; logic ở `scripts/information_theory.py` | `3d_glaucoma_train_3branch_crossgate.ipynb` |
| 12 | Chạy + reporting | Chạy sweep/train, render bảng CSV, W&B table/scalar, markdown report, **Drive sync + `run.finish()`** | Cell 39–49 |
| 13 | Notes/limitations (markdown) | Cảnh báo single-seed, metric thiếu ghi `—`, confound, cách resume | Cell 50 |

### 1.1 Chia nhỏ cell theo từng bước (bắt buộc)

| Nhóm | Cell nhỏ | Nội dung |
|---|---|---|
| Data | D1 tải dữ liệu | HF auth (`HF_TOKEN`) + `allow_patterns`, chỉ tải split/pattern dùng đến |
| | D2 build/denoise | build storage ở `STORE_RES` (idempotent) hoặc khối denoise (partial + completion marker) |
| | D3 view cache | chiếu en-face + depth-axis, cache một lần per split, identity theo source/res |
| | D4 dataset/loader | memmap per-sample, augmentation deterministic, smoke synthetic; loader lazy/cache |
| Train | T1 model | build model **hoặc load weights từ path trước** (`ft.load_weights`), chỉ tạo mới khi chưa có |
| | T2 trainer | `Trainer` + optimizer/scheduler/AMP, probe batch size theo VRAM (`ft.find_batch_size`) |
| | T3 fit | vòng `fit()` + callback eval (val/test định nghĩa ở cell riêng) |
| Eval | E1 val callback | full metrics trên val mỗi epoch |
| | E2 test callback | full metrics trên test mỗi epoch |
| | E3 calibrated report | temperature + threshold fit trên val; calibrated `train/val/test` + bootstrap CI |
| | E4 log bảng | `log_report` + history table + scalar summary (không vẽ đồ thị metrics) |
| | E5 X-AI | heatmap PNG + `xai/fusion_table` + `xai/*` (khi `RUN_XAI`) |
| | E6 info | 7 cell như mục 2.11b (khi `RUN_INFO`) |

Có thể gộp 2–3 cell nhỏ nếu rất ngắn, nhưng **không được gộp data/train/eval chung một cell**. Áp dụng cho
notebook mới và mỗi lần sửa notebook; notebook cũ refactor dần khi có dịp.

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

**Chia nhóm tham số trong cell config bằng dòng `#` (bắt buộc).** Cell config phải chia thành các nhóm rõ ràng,
mỗi nhóm mở đầu bằng một dòng comment `#` mô tả **mục đích** và **mức độ được phép sửa**, để người đọc biết
cần cấu hình gì ở đâu và **không sửa nhầm sang phần đã ổn định**. Mẫu nhóm:

```python
# ===== RUN IDENTITY & RESUME (edit only when starting a new experiment) =====
RUN_GROUP = ...
RUN_TARGET = ...
RESUME = False
# ===== STAGE SWITCHES (enable only the stages you intend to run) =====
ENABLE_TRAIN = ...
ENABLE_EVAL = ...
RUN_XAI = ...
# ===== DATA SOURCES & IMPORT (HF repos, import/publish cache) =====
HF_DN_REPO = ...
PUBLISH_DATA_CACHE = ...
# ===== FROZEN STUDY CONFIG (validated for this study; do not retune) =====
EPOCHS, BS, GRAD_ACCUM = ...
LR, WD, PATIENCE = ...
# ===== WARM-START PARENT (pinned weights; do not substitute) =====
WARM_START_WEIGHTS = ...
# ===== DERIVED PATHS & DRIVE (computed from the above; do not edit) =====
```

- Nhóm **cần cấu hình** (identity, stage switches, data sources) ghi rõ `edit`/`set ...`.
- Nhóm **đã cố định** (hyperparameter đã chốt, parent weights, đường dẫn suy ra) ghi rõ `FROZEN`/`do not edit`.
- Comment `#` không được làm hỏng việc parse/exec cell hay test hiện có; giữ nguyên thứ tự các tham số mà
  test/notebook phụ thuộc. Nếu đổi thứ tự nhóm, cập nhật test tương ứng.

### 2.4 Data
- **Resolution linh hoạt theo config**: storage ở `STORE_RES` có thể là 200, 128, 96, … tùy bài toán; model input
  (`RES3D`/`RES2D`) khai báo riêng và resize on-the-fly. Không hardcode 200; kiểm tra shape theo `STORE_RES`.
- **Luôn tải HF bằng authenticated credentials**: `HF_TOKEN` từ `.env`/Colab Secret, truyền `token=` vào
  `snapshot_download`/`hf_hub_download`; bật `HF_XET_HIGH_PERFORMANCE=1` để đạt tốc độ tối đa (biến cũ
  `HF_HUB_ENABLE_HF_TRANSFER` đã bị deprecated). Real run thiếu token → fail sớm.
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
- **Khởi tạo model theo thứ tự ưu tiên**: (1) resume `last.pt` qua Trainer; (2) load weights từ path
  (`WARM_START_WEIGHTS` hoặc `best_weights.pt` của run) bằng `ft.load_weights(model, path)` — khi đó **bỏ tải
  pretrained backbone**; (3) chỉ khi path chưa tồn tại mới khởi tạo model mới (pretrained). Tách việc build model
  thành cell riêng để dễ chạy lại khi debug.

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
- **Chỉ số không vẽ đồ thị**: log bằng W&B Table/số (`report/split_table`, `report/history_table`, `report/*_table`)
  và summary; không tạo PNG cho metrics (ROC/PR/calibration/confusion/history). Đồ thị chỉ dành cho X-AI/ảnh.
- **Thời gian epoch**: mỗi epoch log `train/epoch_seconds` lên W&B và in `[train] epoch N done in Xs`; dùng để đo
  `sec/epoch`, ước lượng budget và so sánh tốc độ giữa các cấu hình (không đưa wall-clock vào `history` để giữ
  tính tái lập khi resume).
- **In val/test mỗi epoch**: in ra stdout dòng `[val ] epoch N: ...` và `[test ] epoch N: ...` với đầy đủ metric
  chính (loss, acc, balanced_acc, precision, recall, specificity, npv, f1, mcc, kappa, youden, auc_roc, auc_pr,
  ece, logloss, brier) — **không chỉ log W&B**, để theo dõi trực tiếp khi train.
- AUC/PR-AUC trên cửa sổ nhỏ có thể NaN khi cửa sổ chỉ có một lớp — đọc xu hướng theo epoch, đừng chọn theo bước.

### 2.9 Probe + train loop
`probe()` chạy 1 forward/backward (params, peak VRAM, fallback batch 1 khi OOM). Train: AdamW + cosine + warmup +
AMP + grad-accum + grad-clip + early stop; lr theo effective batch; best theo val AUC; checkpoint atomic; `history`
lưu đầy đủ `val`/`test` metrics mỗi epoch để vẽ và đối chiếu sau này.
- **Train thêm epoch**: đặt `RESUME=True` + `EXTEND_EPOCHS=N` rồi chạy lại cell train; hệ thống cộng N epoch vào
  target đã lưu (config chỉ được phép khác `epochs`), cập nhật identity/`last.pt`, mở lại `completed.pt` (nếu có),
  reset patience và LR chạy theo cosine của tổng epoch mới. Không cần tạo `RUN_GROUP` mới.

### 2.10 Sweep driver (nếu sweep)
Mỗi spec override `res3d`/`res2d`/`batch_size`/`epochs`; mỗi run xong ghi ngay `results.json` (local + Drive) và
figure per-run; interrupt giữ các run đã xong; chọn tier qua env (`*_TIER_*`).

### 2.11 X-AI (chạy sau khi train xong)
Trên best model + mẫu test cân bằng lớp:
- **Grad-CAM 3D** (conv cuối/patch-embed) và **Grad-CAM 2D** từng view.
- **Occlusion sensitivity** (3D) và **integrated gradients** (2D view, tùy chọn 3D 64³).
- **Fusion attention** + **branch drop** (leave-one-branch-out) trả lời nhánh nào quyết định.
- **LIME** mức view (tùy chọn).

Lưu local + Drive: PNG heatmap (Grad-CAM 3D/2D, occlusion, IG), `XAI_REPORT.md`, `fusion_xai.pt`
(weights/gate/branch_drop/attention), sync **ngay khi sinh ra**.
W&B: ảnh heatmap `wandb.Image`, bảng **`xai/fusion_table`** (`branch`, `crossgate_attention`, `drop_probability`)
và summary `xai/gate`, `xai/drop_*`, `xai/attention_*`; các giá trị này log bằng bảng/số, **không vẽ chart**.

### 2.11b Phân tích tùy chọn (information theory / embedding)
- Cùng notebook với train, gate bằng cờ config (`RUN_INFO=True/False`); **không** tạo notebook fork chỉ để thêm phân tích.
- `evaluate()` chỉ capture embedding mỗi epoch khi cờ bật; khi tắt thì dùng predict thường (không thêm chi phí).
- Chia thành các cell nhỏ: (a) load model + collect embedding; (b) probe + ablation nhánh; (c) đa-seed MI;
  (d) surrogate; (e) redundancy/synergy; (f) information plane; (g) lưu + `run.finish()`.
- **Probe ablation**: logistic + MLP trên từng nhánh, cặp 3D+view, concat tất cả và `z_fused` (ablation mức
  representation; ablation mức retrain dùng `N_2D=1`/`RUN_GROUP` khác).
- **Đa-seed estimator**: DV, NWJ, InfoNCE × `IT_ESTIMATOR_SEEDS` (≥5 cho luận văn), báo cáo
  median/mean/std/min/max và **negative-rate**; **không clamp giá trị âm** (âm = estimator chưa hội tụ/bias).
  Kèm MIC và normalized MI (`MI/H(Y)`); log entropy `entropy/labels`.
- **Surrogate**: permutation cho MIC (`IT_SURROGATES`, ≥1000 cho p-value resolution) và cho MINE
  (`IT_MINE_SURROGATES`, nhỏ hơn vì đắt); báo p-value, z-score và ngưỡng Bonferroni.
- **Redundancy/synergy**: interaction information `II = I(Z1;Z2) − I(Z1;Z2|Y)` (dương ~ redundancy, âm ~ synergy)
  + conditional MI và joint MI; là proxy, không thay thế PID đầy đủ.
- **Information plane**: I(X;Z) vs I(Z;Y) theo epoch từ embedding đã capture; đọc xu hướng, không kết luận
  bottleneck khi giá trị âm.
- Lưu `info_theory.json`/`info_theory.pt`; log **bảng** (`report/probe_table`, `report/mi_estimators_table`,
  `report/surrogate_table`, `report/interaction_table`, `report/plane_table`) + scalar summary
  (`entropy/*`, `mi/*`, `surrogate/*`, `interaction/*`); **không vẽ figure**.
- Giới hạn subset/steps/PCA-dim; dùng validation để chọn estimator/probe, giữ test cho đánh giá cuối; cell cuối
  chịu trách nhiệm `run.finish()` khi cờ bật.

### 2.12 Reporting
`metrics.json` + history + CSV/bảng split; log W&B bằng table/scalar (`report/split_table`, `report/history_table`),
**không vẽ đồ thị metrics**; sync Drive; `run.finish(exit_code=...)`; X-AI ở mục 2.11.

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
  không tải cả repo; ưu tiên `HF_XET_HIGH_PERFORMANCE=1` để tăng tốc.
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
- **Gia hạn train**: `RESUME=True` + `EXTEND_EPOCHS=N` (mặc định 0) để train thêm N epoch trên cùng run; `epochs`
  là trường duy nhất được phép khác so với config đã lưu; `completed.pt` được mở lại, patience reset, LR đi theo
  cosine của tổng epoch mới. Chạy lại cell train là đủ, không tạo run mới.
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
- **Tải dữ liệu**: `allow_patterns` chỉ tải phần dùng; token + `HF_XET_HIGH_PERFORMANCE=1` để tối đa băng thông.
- **AMP/TF32**: bf16 nếu có, ngược lại fp16 + GradScaler; trên CUDA bật `cudnn.benchmark=True` và TF32 cho
  matmul/conv khi phần cứng hỗ trợ; `torch.compile` chỉ dùng sau khi kiến trúc đã ổn định.
- **Batch theo VRAM**: thay vì đoán, **chọn batch size theo cấu hình train + dữ liệu hiện tại** bằng
  `ft.find_batch_size(build_model, train_dataset, device=DEVICE, start=BS, target_gb=TARGET_VRAM_GB,
  num_workers=NUM_WORKERS)`. Helper chạy forward+backward+optimizer step thật trên chính model/dataset với batch
  tăng gấp đôi, đo peak VRAM và chọn batch lớn nhất trong ngân sách (`min(TARGET_VRAM_GB, VRAM_total*0.9)` để chừa
  headroom cho eval/X-AI). Giữ effective batch không đổi: `GRAD_ACCUM = max(1, round(EFFECTIVE_BATCH / BS))`.
- **Resume an toàn config**: khi `RESUME=True`, lấy lại `batch_size`/`grad_accum` từ `run_identity.pt` thay vì
  probe lại, để identity/config không đổi. Log `batch/size`, `batch/accum`, `batch/peak_gb`, `batch/target_gb` +
  bảng `report/batch_probe_table`. Trên CPU/không CUDA, probe trả `start` với `status="cpu"` và không chạy.
- **num_workers**: Linux/Colab dùng >0 (dataset augment per-sample nên kết quả không đổi), Windows để 0.
- **Phân tích nặng** (MINE/surrogate/probe/IB, X-AI): giới hạn subset/steps/PCA-dim; cache embedding thay vì
  forward lại; log bảng/giá trị thay vì lưu tensor lớn.
- **Không tính lại cái đã có**: tái sử dụng cache (local/Drive) và cân nhắc upload HF cho preprocessing đắt.

### 3.11 Một notebook chuẩn cho mỗi kiến trúc
- Mỗi kiến trúc/preset chỉ có **một notebook train chuẩn**; các giai đoạn tùy chọn (X-AI, information theory,
  phân tích embedding) là **cờ config** (`RUN_XAI`, `RUN_INFO`) trong cùng notebook — không nhân bản/fork notebook.
- Giữ đúng thứ tự cell mục 1; logic tái sử dụng nằm ở `scripts/*.py`, notebook chỉ cấu hình và gọi.
- Notebook mới phải pass `*_SMOKE=1` (CPU, synthetic) và checklist mục 4 trước khi giao.

### 3.12 Chia cell nhỏ theo thành phần
- Mỗi cell một việc/nhóm việc giống nhau; tách tối thiểu theo bảng 1.1: **data** (tải / build-denoise / view /
  dataset-loader), **train** (model / trainer+batch probe / vòng fit), **eval** (val callback / test callback /
  calibrated report / log bảng / X-AI / info).
- Không gộp data + train + eval vào cùng một cell; định nghĩa callback eval ở cell riêng thay vì nhét vào cell train.
- **Bắt buộc thấy rõ trong notebook**: D1–D4, T1–T3, E1–E6 phải là **cell riêng có markdown heading** (ví dụ
  `### D1 - Download`, `### T1 - Model`, `### E1/E2 - Val/test callbacks`, `### T3 - Training loop`), không chỉ liệt
  kê trong tài liệu; data/train/eval gộp chung một cell là **chưa đạt**.
- Cell nên chạy lại độc lập khi debug; biến trung gian giữ ở global; đặt markdown heading cho từng phần.
- Model: load từ path trước (`ft.load_weights`, `WARM_START_WEIGHTS`/`best_weights.pt`); chỉ khởi tạo mới khi path
  chưa có (và khi đó mới tải pretrained backbone).

### 3.13 Log trạng thái từng bước
- Mỗi bước in **một dòng trạng thái** có tiền tố rõ: `[data]`, `[denoise]`, `[views]`, `[dataset]`, `[cache]`,
  `[model]`, `[batch]`, `[wandb]`, `[train]`, `[eval]`, `[checkpoint]`, `[report]`, `[xai]`, `[info]`, `[sync]`.
- Log các mốc: tải/cache data (số pattern, cache hit hay tải mới), build denoise/views (cache hit hay tạo mới, số
  mẫu), dataset (n + số positive mỗi split, resolution), model (load từ path hay khởi tạo mới, params), batch probe
  (BS/peak GB/budget), train start (device, epochs, effective batch, workers), mỗi epoch (in `[val ]` và `[test ]`
  đầy đủ metric chính, history, `train/epoch_seconds`, `[train] epoch N done in Xs`), checkpoint (file đã lưu),
  calibrated report (temperature/threshold + test/val AUC/F1), X-AI/info bắt đầu–kết thúc, sync Drive + finish.
- Log **ngắn gọn, không trùng** với W&B (W&B vẫn là nguồn số liệu chính); không in trong vòng lặp micro-batch.
- Ưu tiên log trong shared helper (`scripts/*.py`) để mọi notebook cùng format; notebook in thêm mốc riêng.

### 3.14 Notebook đối chứng / ablation
- Mỗi biến thể **train lại từ đầu** trên cùng split/seed/protocol; **không** giữ model đầy đủ rồi chỉ che input
  lúc test. Bỏ nhánh 3D/2D phải huấn luyện lại với kiến trúc/fusion tương ứng.
- Spec table nằm trong config (mã, model kwargs: `use_3d`, `n_2d`, `view_indices`, `fusion`, `gate_fixed`) và được
  ghi vào run config để resume/so sánh chính xác.
- Cùng epoch budget và cùng `PATIENCE` cho mọi biến thể; chọn best theo val AUC; chạy **≥3 seed** và báo cáo
  **mean ± std** (kèm per-seed, không chỉ seed đẹp).
- Mỗi cặp `(spec, seed)` là **một run riêng**, resume-safe; có cell aggregation tổng hợp bảng so sánh
  (`*_summary.csv/json` + W&B table) và sync Drive.
- Baseline chính (P) phải được train lại cùng budget nếu protocol khác trước đó.

## 4. Checklist trước khi giao notebook

- [ ] Đúng thứ tự cell mục 1; config một nguồn duy nhất, có env override.
- [ ] **Cell config chia nhóm bằng dòng `#`** (nhóm cần sửa vs. `FROZEN`/derived), để biết cần cấu hình gì ở đâu và không sửa nhầm phần đã ổn.
- [ ] Cell tách theo thành phần (mỗi cell 1 việc/nhóm việc), chạy lại được từng cell khi debug/sửa lỗi.
- [ ] Tách cell data (tải/build-denoise/view/dataset-loader), train (model/trainer-probe/fit) và eval (val callback, test callback, calibrated report, log bảng) theo bảng 1.1.
- [ ] Kiểm tra notebook thực tế có cell riêng + markdown heading cho D1–D4/T1–T3/E1–E6 (không gộp data/train/eval), không chỉ ghi trong tài liệu.
- [ ] Mỗi bước in log trạng thái có tiền tố (`[data]`/`[model]`/`[train]`/`[eval]`/`[xai]`/...) để theo dõi và debug.
- [ ] Mỗi epoch log thời gian hoàn thành (`train/epoch_seconds` + in `[train] epoch N done in Xs`) để đo `sec/epoch`.
- [ ] Mỗi epoch in rõ `[val ]` và `[test ]` (metric chính) ra stdout, không chỉ log W&B.
- [ ] Model load từ path trước (`WARM_START_WEIGHTS`/`best_weights.pt`); chỉ khởi tạo mới khi chưa có weights.
- [ ] `*_SMOKE=1` chạy hết cell trên CPU, synthetic, không download — pass.
- [ ] W&B init + log live đúng prefix + figures as `wandb.Image` + `summary.update` + `finish`.
- [ ] **Train log đủ bộ metric mỗi optimizer step; val + test log đủ bộ metric mỗi epoch.**
- [ ] **Chốt cuối có calibrated `train`/`val`/`test` + bootstrap CI và bảng `report/split_table` trên W&B.**
- [ ] **Sau train chạy X-AI trên best model, log `xai/fusion_table` + giá trị `xai/*` lên W&B (hoặc ghi rõ ngoại lệ).**
- [ ] Chỉ số log bằng W&B table/scalar; **không vẽ đồ thị metrics** (đồ thị chỉ dùng cho X-AI/ảnh).
- [ ] Info-theory (nếu có): đa-seed DV/NWJ/InfoNCE có negative-rate (không clamp); surrogate ≥1000 cho MIC; interaction information cho redundancy/synergy; tất cả log bằng table/scalar.
- [ ] Notebook đối chứng/ablation: mỗi biến thể train lại, cùng budget/patience, ≥3 seed, có bảng mean ± std và mỗi (spec, seed) là run riêng.
- [ ] **Optional stages (X-AI/info-theory) là cờ trong cùng notebook, không tách/fork notebook; logic ở `scripts/`.**
- [ ] Mọi artifact (figure/model/CSV/report/X-AI) sync Drive ngay khi sinh ra.
- [ ] HF download có `HF_TOKEN` + `allow_patterns` chỉ tải phần dùng; real run thiếu token fail sớm.
- [ ] `STORE_RES` lấy từ config (200/128/96/…), không hardcode; check shape theo `STORE_RES`; tách khỏi `RES3D`/`RES2D`.
- [ ] Quyết định cache/tái sử dụng/upload dữ liệu đã xử lý (khử nhiễu/view/augmentation) được ghi rõ.
- [ ] AMP bật khi CUDA + `pin_memory`/`non_blocking`; hạn chế `.item()`/đồng bộ host trong vòng train.
- [ ] Loader lazy/cache, chỉ đọc dữ liệu dùng đến; preprocessing đắt được cache/tái sử dụng.
- [ ] Đo `sec/epoch` + peak VRAM; phân tích nặng (MI/X-AI) có giới hạn subset/steps.
- [ ] Batch size chọn theo cấu hình+dữ liệu (`ft.find_batch_size`) tới ngân sách VRAM; giữ effective batch qua grad-accum; resume dùng lại batch đã lưu.
- [ ] Best checkpoint theo val AUC; threshold/calibration fit trên val, không dùng test.
- [ ] Resume-safe: `results.json` append + checkpoint atomic.
- [ ] Cell train hỗ trợ gia hạn (`RESUME=True` + `EXTEND_EPOCHS=N`) mà không cần `RUN_GROUP` mới.
- [ ] `ruff check` sạch cho file Python mới; unit-test core mới pass.
- [ ] Notes/limitations được ghi rõ.

## 5. Notebook tham chiếu

| Notebook | Dùng để tham khảo |
|---|---|
| [`3d_glaucoma_multiview_sota_sweep_xai.ipynb`](../notebooks/3d_glaucoma_multiview_sota_sweep_xai.ipynb) | Cấu trúc sweep đầy đủ: registry backbone, fusion ablation, X-AI, resume-safe, reporting |
| [`3d_glaucoma_train_3branch_crossgate.ipynb`](../notebooks/3d_glaucoma_train_3branch_crossgate.ipynb) | Notebook train chuẩn 3 nhánh (1×3D ResNeXt + 2×2D MaxViT + CrossGate): 3D chạy **96³** resize on-the-fly, **view 2D chiếu từ raw 200³**; full metrics 3 tập, X-AI, info-theory tùy chọn (`RUN_INFO`), resume, smoke |
| [`3d_glaucoma_final_2x2d_3d_crossgate.ipynb`](../notebooks/3d_glaucoma_final_2x2d_3d_crossgate.ipynb) | Notebook train chuẩn cho tập **Bilateral 200³** (denoise cache + recovery contract): full metrics 3 tập, X-AI + bảng W&B, warm-start tùy chọn, smoke |
| [`3d_glaucoma_controls_96.ipynb`](../notebooks/3d_glaucoma_controls_96.ipynb) | Notebook đối chứng 96³: B1–B3 (bỏ nhánh), C1–C2 (fusion), P (baseline); mỗi spec × 3 seed, 20 epoch early stop, bảng mean ± std |
| [`3d_glaucoma_resolution_96_128_200.ipynb`](../notebooks/3d_glaucoma_resolution_96_128_200.ipynb) | So sánh resolution trên cùng backbone |
| [`3d_glaucoma_denoise_gpu_compare.ipynb`](../notebooks/3d_glaucoma_denoise_gpu_compare.ipynb) | So sánh phương pháp khử nhiễu + metric ảnh |

Core tái sử dụng: `scripts/final_model.py` (ResNeXt3D/Timm2D/CrossGate/metrics/X-AI), `scripts/final_training.py`
(Trainer recovery/W&B), `scripts/information_theory.py` (probe/MINE/MIC/surrogate/IB),
`scripts/resolution_study.py` (metric numpy), `scripts/compare_denoise_methods.py` (khử nhiễu).
