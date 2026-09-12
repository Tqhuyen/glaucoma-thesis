# Standard Structure and Rules for Model Training Notebooks

**Language and authority policy:** Agents must read this English document before doing notebook work.
This is the canonical, authoritative notebook-conventions document for agents; the
[Vietnamese version](notebook-conventions.md) is human-facing. Maintain both versions in sync when conventions
change. These conventions remain subject to higher-priority instructions and the explicit scope of the experiment.
They do not authorize starting or extending training, adding sweeps/seeds/ablations, enabling optional stages,
uploading data, or pushing changes beyond what the user has authorized. Procedures and checklist requirements
below apply within that scope; document explicit scope exceptions rather than silently expanding the experiment.

**Review note:** Both language versions reconcile four inconsistencies in the earlier Vietnamese source: per-epoch test
monitoring is allowed but never used for selection, and the final held-out report uses the best validation
checkpoint; metric figures must not be generated (other figure references mean X-AI/images, while existing
artifacts still require Drive sync); config-grouping comments are a required exception to the no-comments rule;
and view caches can be reused across `RES3D` only when `RES2D`, source, and projection identities match.
These clarifications are reflected in both language versions.

This document standardizes the **cell structure** and **mandatory rules** for creating training, sweep, and
model-research notebooks in this repository. The primary reference is the 2D-3D fusion sweep notebook
[`notebooks/3d_glaucoma_multiview_sota_sweep_xai.ipynb`](../notebooks/3d_glaucoma_multiview_sota_sweep_xai.ipynb),
cross-checked against
[`notebooks/3d_glaucoma_final_2x2d_3d_crossgate.ipynb`](../notebooks/3d_glaucoma_final_2x2d_3d_crossgate.ipynb)
(recovery/W&B) and the rules in `AGENTS.md`. Treat this as a mandatory checklist when creating a new notebook.

## 1. Standard Cell Structure

The order below comes from the sweep notebook. A single-run training notebook may omit sweep sections but
**must not omit** sections 2-4, 9, and 12. Section 11 is required unless the preset or approved scope documents
an X-AI exception; optional analysis remains conditional as marked below.

**Cell separation principle:** Each cell performs **one task** or a group of similar tasks. Separate **data**,
**training**, **evaluation**, **info/X-AI**, and reporting, not just the info section. Keep cells short with few
cross-cell side effects, and retain intermediate variables at global scope so that **that specific cell can be
rerun** during debugging/fixes instead of rerunning the entire notebook. Split overly long cells (> ~80 lines)
by step and provide clear Markdown headings.

| # | Cell | Contents | Sweep example |
|---|---|---|---|
| 1 | Title/overview (Markdown) | Problem, dataset, pipeline, runtime budget, how to run, how to enable smoke | Cell 1: dataset + "Honest runtime expectation" |
| 2 | Setup | `IN_COLAB`/`SMOKE` guard, clone repo, `pip install` (timm/monai/wandb/sklearn/skimage/wandb...), imports, `DEVICE`, seed | Cells 2-4 |
| 3 | Config | **Single source of truth** (`CFG`/constants): data paths, `STORE_RES`, `MODEL_RES3D`/`RES2D`, epochs, bs, grad-accum, lr, tier/spec, `SAVE_DIR`/`DRIVE_DIR`, env overrides | Cells 5-6 |
| 4 | Data | Download from HF **with token authentication**, only required splits/patterns; build storage at configured `STORE_RES` (200/128/96/..., idempotent, cached) + cache views/depth-axis once per split; `manifest.json` | Cells 7-10 |
| 5 | Dataset + loaders | Per-sample memmap reads, cached views, deterministic augmentation by `(seed, epoch, index)`, lazy/cached loaders keyed by `(res3d, res2d, bs, ...)`; **do not read** a modality for an unused branch | Cell 11 |
| 6 | Branch encoder registry (for sweeps) | Each encoder returns a pooled `out_dim` embedding; input `(B,1,D,H,W)` in `[0,1]`; 2D `(B,1,H,W)` | Cells 12-18 |
| 7 | Fusion + model | Fusion operations (concat/add/mul/attn/crossgate/mamba/film) + multibranch model | Cells 19-20 |
| 8 | Metrics | Full clinical metric set + calibration + threshold + bootstrap CI (details in section 2.8) | Cells 21-23 |
| 9 | Probe + train loop | `probe()` checks one forward/backward pass, params, VRAM; resume-safe `train_one()`, save per-run JSON | Cells 24-27 |
| 10 | Sweep driver (for sweeps) | Tier/spec; write `results.json` (local + Drive) immediately after each run; export per-run X-AI/image figures immediately, not metric figures | Cells 28-31 |
| 11 | X-AI (if applicable) | 3D/2D Grad-CAM, occlusion, integrated gradients, branch importance, LIME; save PNG + Markdown | Cells 32-38 |
| 11b | Optional analysis (if applicable) | Info-theory/embedding analysis runs **in the same notebook** behind `RUN_INFO`; logic in `scripts/information_theory.py` | `3d_glaucoma_train_3branch_crossgate.ipynb` |
| 12 | Execution + reporting | Run sweep/training, render CSV tables, W&B tables/scalars, Markdown report, **Drive sync + `run.finish()`** | Cells 39-49 |
| 13 | Notes/limitations (Markdown) | Single-seed warning, mark missing metrics with `\u2014`, confounds, how to resume | Cell 50 |

### 1.1 Split Cells by Step (Mandatory)

| Group | Small cell | Contents |
|---|---|---|
| Data | D1 download data | HF auth (`HF_TOKEN`) + `allow_patterns`; download only required splits/patterns |
| | D2 build/denoise | Build storage at `STORE_RES` (idempotent) or denoising block (partial + completion marker) |
| | D3 view cache | En-face projection + depth-axis; cache once per split, identity keyed by source/resolution/projection |
| | D4 dataset/loader | Per-sample memmap, deterministic augmentation, synthetic smoke; lazy/cached loaders |
| Train | T1 model | Build model **or load weights from a path first** (`ft.load_weights`); initialize a new model only when weights are absent |
| | T2 trainer | `Trainer` + optimizer/scheduler/AMP; probe batch size against VRAM (`ft.find_batch_size`) |
| | T3 fit | `fit()` loop + evaluation callbacks (val/test defined in separate cells) |
| Eval | E1 val callback | Full validation metrics each epoch |
| | E2 test callback | Full test metrics each epoch for monitoring only, never selection |
| | E3 calibrated report | Fit temperature + threshold on val; calibrated `train/val/test` + bootstrap CI from the best validation checkpoint |
| | E4 table logging | `log_report` + history table + scalar summary (no metric plots) |
| | E5 X-AI | Heatmap PNG + `xai/fusion_table` + `xai/*` (when `RUN_XAI`) |
| | E6 info | Seven cells as in section 2.11b (when `RUN_INFO`) |

Two or three small cells may be combined if very short, but **data/training/evaluation must not be combined in
one cell**. Apply this to new notebooks and every notebook edit; refactor older notebooks incrementally when
the opportunity arises. Keep distinct steps visible with Markdown headings as required in section 3.12.

## 2. Details by Section

### 2.1 Title/Overview (Markdown)
State the objective + architecture summary, dataset, resolution policy (config-driven `STORE_RES`, model input
may resize on-the-fly), realistic runtime budget by backbone (minutes/run), top-to-bottom execution instructions,
how to enable `*_SMOKE`, and artifact locations (local + Drive + W&B). Clearly identify the scope: single-run
training, fine-tuning, or a multitier sweep.

### 2.2 Setup
`IN_COLAB`/`SMOKE` guards; clone/pull repo; install packages by group (`timm`, `monai`, `scikit-learn`,
`scikit-image`, `wandb`, `huggingface_hub`, `python-dotenv`); mount Drive; read `HF_TOKEN`/`WANDB_API_KEY` from
`.env` or Colab Secrets; import core scripts; `DEVICE`; seed; AMP dtype (bf16 if available, otherwise fp16 +
`GradScaler`).

### 2.3 Configuration
Use a single source of truth (`CFG`/constants): data paths, `STORE_RES` (config-driven, e.g. 200/128/96, not
hardcoded to 200), `MODEL_RES3D`/`RES2D`, epochs, batch_size, grad_accum, lr, weight_decay, patience, `log_every`,
`SAVE_DIR`/`DRIVE_DIR`/`RUN_GROUP`, spec/tier, `RUN_XAI`, and env overrides. Validate early (fail fast) before
building data/model; change `RUN_GROUP` when changing the training configuration, except for the explicitly
supported epoch-extension procedure in section 2.9.

**Group parameters in the config cell using `#` lines (mandatory).** Divide the config cell into clear groups.
Start each group with a `#` comment describing its **purpose** and **permitted edit scope**, so readers know
what to configure where and **do not accidentally modify stabilized settings**. These required grouping comments
are an explicit exception to the general no-comments rule in section 3.8. Example groups:

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

- Groups **requiring configuration** (identity, stage switches, data sources) must explicitly say `edit`/`set ...`.
- **Fixed** groups (finalized hyperparameters, parent weights, derived paths) must say `FROZEN`/`do not edit`.
- `#` comments must not break cell parsing/execution or existing tests; preserve parameter order where tests or
  the notebook depend on it. If group order changes, update the corresponding tests.

### 2.4 Data
- **Config-driven resolution flexibility**: Storage at `STORE_RES` may be 200, 128, 96, ... depending on the task;
  model input (`RES3D`/`RES2D`) is declared separately and resized on-the-fly. Do not hardcode 200; check shapes
  against `STORE_RES`.
- **Always download from HF with authenticated credentials**: Read `HF_TOKEN` from `.env`/Colab Secret and pass
  `token=` to `snapshot_download`/`hf_hub_download`; enable `HF_XET_HIGH_PERFORMANCE=1` for maximum speed (the old
  `HF_HUB_ENABLE_HF_TRANSFER` variable is deprecated). Fail early if a real run has no token.
- **Download only what is used**: Use `allow_patterns` (or per-file downloads) for the required splits/file types;
  never `snapshot_download` the entire repo.
- **Reuse and cache**: Build storage idempotently (reuse across notebooks/runs), write `manifest.json`; cache
  en-face views (`aip_full`, `slab_aip`, `slab_mip`) + depth-axis once per split. View-cache reuse across `RES3D`
  requires matching `RES2D`, source identity (including split/storage resolution/preprocessing), and projection
  identity (method, parameters, and implementation). Do not reuse incompatible cached views.
- **Processed data (denoising, views, heavy augmentation)**: Consider uploading to an HF dataset repo (versioned
  by method + params + implementation hash) to avoid recomputation. Upload only when it genuinely saves time and
  the user agrees, with identity metadata for reproducibility. Keep cheap/frequently changing work local/on
  Drive rather than uploading it.

### 2.5 Dataset + Loaders
Read volumes per sample using memmap; read views from cache; make augmentation deterministic by
`(seed, epoch, index)`; use lazy loaders + caching keyed by `(res3d, res2d, bs, load3d, loadviews)`; do not read
modalities for unused branches; use `num_workers=0` on Windows. Items return `(x, views, y)` and the 3D tensor
is `(1,D,H,W)` (do not use `raw[None,None]`).

### 2.6 Branch Encoders (Registry)
Each 3D encoder maps `(B,1,D,H,W) -> out_dim` with pooling; each 2D encoder maps `(B,1,H,W) -> out_dim`. Document
resolution requirements (divisibility by patch/window 16/32), use normalization suitable for small batches
(GroupNorm/LayerNorm), and preserve depth with `(2,2,1)` pooling.

### 2.7 Fusion + Model
Reference fusion operations: concat/add/mul/attn/crossgate/mamba/film. Default: CrossGate (3D as query, 2D as
key/value, learned gate). `forward` returns raw logits for `CrossEntropyLoss`; expose `fuse()`/`embed()` when
needed for X-AI/embedding analysis.
- **Model initialization priority**: (1) Resume `last.pt` through Trainer; (2) load weights from a path
  (`WARM_START_WEIGHTS` or the run's `best_weights.pt`) with `ft.load_weights(model, path)`, **skipping pretrained
  backbone downloads** in that case; (3) initialize a new (pretrained) model only when the path does not exist.
  Put model building in a separate cell so it can easily be rerun during debugging.

### 2.8 Metrics (Full Sweep Set)
Use the full metric set from the sweep's `metrics_full` (extended in `fm.full_metrics`):
`acc`, `balanced_acc`, `precision/PPV`, `recall/sensitivity`, `specificity`, `npv`, `f1`, `f1_macro`, `mcc`,
`kappa`, `youden`, `auc_roc`, `auc_pr`, `ece`, `logloss`, `brier`, `tn/fp/fn/tp`, `n`.

Logging cadence (train by step, val/test by epoch):
- **Train at every optimizer step**, over the grad-accum window: `train/<metric>` for the full metric set +
  `train/loss`, `train/acc` (running epoch), `train/lr`, `train/epoch`, `progress/step`.
- **Val every epoch**: `val/<metric>` for the full metric set; use val AUC/PR-AUC for best-checkpoint selection
  and early stopping.
- **Test every epoch**: `test/<metric>` + `test/epoch` with the same metric set, for monitoring only. Never use
  test metrics to select checkpoints, hyperparameters, thresholds, calibration, or stopping decisions.
- **Final report** (best validation state): `calibrated_report` fits temperature + threshold (Youden-J) on val,
  then computes calibrated `train`/`val`/`test` + bootstrap CI. `ft.log_report` writes the `report/split_table`
  comparison table and `train|val|test/<metric>` summaries (+ `_lo`/`_hi` for CI). The final held-out test report
  must use the best validation checkpoint, not a checkpoint chosen from test monitoring.
- **Do not plot metrics**: Log W&B tables/numbers (`report/split_table`, `report/history_table`, `report/*_table`)
  and summaries; do not generate metric PNGs or other metric figures (ROC/PR/calibration/confusion/history).
  Figures are for X-AI/images only. Elsewhere in this document, figure-generation references have this same
  meaning; existing artifacts, including legacy metric figures, must still be synced to Drive.
- **Epoch timing**: Each epoch, log `train/epoch_seconds` to W&B and print `[train] epoch N done in Xs`; use this
  to measure `sec/epoch`, estimate the budget, and compare configurations' speed. Do not put wall-clock timing
  in `history`, to preserve reproducibility on resume.
- **Print val/test every epoch**: Print `[val ] epoch N: ...` and `[test ] epoch N: ...` to stdout with the full
  main metric set (loss, acc, balanced_acc, precision, recall, specificity, npv, f1, mcc, kappa, youden, auc_roc,
  auc_pr, ece, logloss, brier). **W&B logging alone is not sufficient**; allow direct monitoring during training.
- AUC/PR-AUC over small windows may be NaN when only one class is present. Read epoch-level trends; do not select
  by step-level values.

### 2.9 Probe + Train Loop
`probe()` runs one forward/backward pass (params, peak VRAM, fallback to batch 1 on OOM). Training uses AdamW +
cosine + warmup + AMP + grad-accum + grad-clip + early stopping; set lr according to effective batch; select best
by val AUC; write atomic checkpoints. Store the full per-epoch `val`/`test` metrics in `history` for later
tabular reporting and comparison, not metric plots.
- **Additional training epochs**: Set `RESUME=True` + `EXTEND_EPOCHS=N`, then rerun the training cell. The system
  adds N epochs to the saved target (only `epochs` may differ in the config), updates identity/`last.pt`, reopens
  `completed.pt` if present, resets patience, and uses a cosine LR schedule over the new total epoch count.
  No new `RUN_GROUP` is needed. This describes the mechanism, not permission to extend an experiment.

### 2.10 Sweep Driver (For Sweeps)
Each spec overrides `res3d`/`res2d`/`batch_size`/`epochs`. Immediately after each run, write `results.json`
(local + Drive) and export any per-run X-AI/image figures, not metric figures. Preserve completed runs on
interrupt; select tiers through env (`*_TIER_*`).

### 2.11 X-AI (After Training)
Use the best validation model + a class-balanced sample from test:
- **3D Grad-CAM** (last convolution/patch-embed) and **2D Grad-CAM** for each view.
- **Occlusion sensitivity** (3D) and **integrated gradients** (2D views, optionally 3D 64^3).
- **Fusion attention** + **branch drop** (leave-one-branch-out) to identify which branch drives the decision.
- View-level **LIME** (optional).

Save locally + to Drive: PNG heatmaps (3D/2D Grad-CAM, occlusion, IG), `XAI_REPORT.md`, `fusion_xai.pt`
(weights/gate/branch_drop/attention); sync **as soon as each is produced**.
W&B: heatmap images via `wandb.Image`, the **`xai/fusion_table`** table
(`branch`, `crossgate_attention`, `drop_probability`), and `xai/gate`, `xai/drop_*`, `xai/attention_*` summaries.
Log these values as tables/numbers, **not charts**.

### 2.11b Optional Analysis (Information Theory / Embeddings)
- Keep it in the same notebook as training, gated by config (`RUN_INFO=True/False`); **do not** fork a notebook
  just to add analysis.
- `evaluate()` captures embeddings each epoch only when enabled; otherwise use ordinary prediction with no
  additional cost.
- Split into small cells: (a) load model + collect embeddings; (b) probe + branch ablation; (c) multiseed MI;
  (d) surrogate; (e) redundancy/synergy; (f) information plane; (g) save + `run.finish()`.
- **Probe ablation**: Logistic + MLP probes on each branch, 3D+view pairs, all concatenated features, and
  `z_fused` (representation-level ablation; retraining-level ablation uses `N_2D=1`/a different `RUN_GROUP`).
- **Multiseed estimators**: DV, NWJ, InfoNCE x `IT_ESTIMATOR_SEEDS` (>=5 for the thesis); report
  median/mean/std/min/max and **negative-rate**. **Do not clamp negative values** (negative means estimator
  nonconvergence/bias). Include MIC and normalized MI (`MI/H(Y)`); log entropy as `entropy/labels`.
- **Surrogates**: Permutation for MIC (`IT_SURROGATES`, >=1000 for p-value resolution) and MINE
  (`IT_MINE_SURROGATES`, fewer because it is expensive); report p-value, z-score, and Bonferroni threshold.
- **Redundancy/synergy**: Interaction information `II = I(Z1;Z2) - I(Z1;Z2|Y)` (positive ~ redundancy, negative ~
  synergy) + conditional MI and joint MI. This is a proxy, not a replacement for full PID.
- **Information plane**: I(X;Z) vs I(Z;Y) by epoch from captured embeddings; read trends, and do not infer a
  bottleneck from negative values.
- Save `info_theory.json`/`info_theory.pt`; log **tables** (`report/probe_table`, `report/mi_estimators_table`,
  `report/surrogate_table`, `report/interaction_table`, `report/plane_table`) + scalar summaries
  (`entropy/*`, `mi/*`, `surrogate/*`, `interaction/*`); **do not generate figures**.
- Bound subset/steps/PCA-dim; use validation to select estimators/probes and reserve test for final evaluation.
  The final cell is responsible for `run.finish()` when this flag is enabled.

### 2.12 Reporting
`metrics.json` + history + CSV/split tables; log W&B tables/scalars (`report/split_table`, `report/history_table`),
**without metric plots**; sync Drive; `run.finish(exit_code=...)`; see section 2.11 for X-AI.

### 2.13 Notes/Limitations (Markdown)
Document single-seed limitations, missing metrics marked `\u2014`, resolution/denoising effects, how to resume,
and preset exceptions (e.g. a fine-tuning preset may set `RUN_XAI=False` to save GPU time, but must state why).
Here and elsewhere, `\u2014` denotes the source's em-dash missing-value marker (written as an ASCII escape).

## 3. Mandatory Rules (Hard Rules)

### 3.1 W&B: Mandatory for Every Run
- Call `load_env_file()` (or read Colab Secret `WANDB_API_KEY`) before `wandb.init()`; project `glaucoma-thesis`.
- Use idempotent helper `init_wandb(run_name, config)` (`WANDB_RUN is None`); call it **before training**.
- Log **live** with correct prefixes: `train/...` for train, `val/...` for val, `test/...` for test; do not mix
  histories.
- Log X-AI/image figures with `wandb.Image`; finish with `run.summary.update(...)` + `run.finish(exit_code=...)`.
- Never disable/drop wandb to "save time". Only internal smoke runs may use `mode="offline"`.

### 3.2 Drive: Every Artifact Must Reach Drive
- Guard Drive mounting with `os.path.ismount` so headless execution does not fail on mounting.
- Root: `/content/drive/MyDrive/MasterBKDN/Thesis/<experiment>[_figures]`; preserve existing notebook paths
  (`sota_200`, `multiview`, `denoise_sweep`, `3dino_ft`, `final_2x2d_3d_crossgate`, ...).
- Copy X-AI/image figures, CSV/JSON/reports, and checkpoints **as soon as they are produced**, not at the end of
  the run, so interruptions preserve results. Existing artifacts, including legacy
  ROC/PR/calibration/confusion/history figures, must also sync; this does **not** permit creating new metric
  figures (section 2.8).
- "Saved only locally or in git" is **not done**.

### 3.3 Data
- Storage resolution is config-driven (`STORE_RES`: 200/128/96/...), not hardcoded; declare model input separately
  and resize on-the-fly (cast to float before `F.interpolate`).
- **Always authenticate HF downloads** (`HF_TOKEN` + `token=`); download only the actual required splits/patterns
  (`allow_patterns`), never the entire repo; prefer `HF_XET_HIGH_PERFORMANCE=1` for speed.
- Cache/reuse processed data; consider HF upload (versioned, with identity) for expensive processing that will
  be reused; upload only when requested/approved.
- `data/` is gitignored; do not commit/push data into git.
- Project 2D views (aip/slab) from the storage volume and cache once per split. Reuse across `RES3D` is allowed
  only when `RES2D`, source identity (split/storage resolution/preprocessing), and projection identity
  (method/parameters/implementation) match. A different `RES2D` requires a matching cache entry; changing only
  `RES3D` does not require recomputing otherwise identical 2D views.

### 3.4 Resolution Naming (A Previous Pitfall)
- Use **distinct names** for storage and model input: `STORE_RES` (200/128/96/...), `MODEL_RES3D`/`RES3D`, `RES2D`.
- Resize on-the-fly; cast to float **before** `F.interpolate` (trilinear fails on uint8).
- 3DINO-ViT requires input divisible by patch 16 (112^3); other transformers (Swin/UNETR) require divisibility
  by 16/32.

### 3.5 Model & Loss
- `forward()` returns logits; apply `CrossEntropyLoss` to raw logits (2 classes, with class weights).
- Small batches (2-4): use **GroupNorm/LayerNorm**, avoid BatchNorm (grad-accum does not fix BN).
- Preserve depth in 3D pooling: anisotropic `(2,2,1)` in early stages, `AdaptiveAvgPool3d(1)` only at the end.
- AMP: bf16 when available, otherwise fp16 + `GradScaler`; grad-accum; grad-clip; cosine + warmup.

### 3.6 Checkpoint & Metric Selection
- Select best by **val AUC/PR-AUC**, never by test.
- Per-epoch test evaluation is allowed for monitoring as specified in section 2.8, but must never influence
  checkpoint/model/hyperparameter selection, early stopping, threshold selection, or calibration. The final
  held-out test report must evaluate the **best validation checkpoint**. Select the Youden-J threshold on val
  and fit temperature scaling on val only.
- Minimum metric set: acc, balanced acc, precision/PPV, recall/sensitivity, specificity, NPV, F1, MCC,
  AUC-ROC, PR-AUC, ECE (+ bootstrap CI); mark missing metrics with `\u2014`.

### 3.7 Resume Safety & Smoke
- Write each run's results immediately to `results.json` (local + Drive) and use atomic checkpoints. Preserve
  completed runs on interrupt; rerunning must resume rather than restart from scratch.
- **Training extension**: `RESUME=True` + `EXTEND_EPOCHS=N` (default 0) adds N epochs to the same run; `epochs`
  is the only field allowed to differ from the saved config. Reopen `completed.pt`, reset patience, and use a
  cosine LR schedule over the new total epoch count. Rerunning the training cell is sufficient; do not create
  a new run. Extend only within explicit experiment authorization.
- Before spending GPU time, run `*_SMOKE=1` on CPU with small synthetic data and **no downloads**; all cells
  must pass. For the full pipeline, `make sanity` must overfit 10 samples before blaming the data.

### 3.8 Code Style & Technical Rules
- Line length 120; `ruff`; do not add comments unless requested, **except for the mandatory config-grouping
  comments in section 2.3**; follow registry/config-driven patterns.
- Do not use `vols[:,0]` on a 5D memmap (loads 26 GB); index `vols[i]` and squeeze per sample instead.
- Windows: `num_workers=0` (mmap + multiprocessing can segfault).
- Per-sample augmentation must be deterministic by `(seed, epoch, index)`.
- New variants: put reusable logic in `scripts/*.py`; notebooks only call it; unit-test new core logic.

### 3.9 Colab
- Provide env overrides for all modes: `*_SMOKE`, `*_SWEEP`, `*_XAI`, `*_TIER_*`, `*_RESUME`.
- Guard clone/pip/mount with `IN_COLAB`/`SMOKE`; obtain HF/W&B tokens from `.env` or Colab Secrets.
- After notebook/config edits, remind the user to **restart opencode** if config changed, and to push the repo
  so Colab can clone it. This reminder is not authorization for an agent to push without an explicit request.

### 3.10 Performance & Speed Optimization (Default)
- **Measure first, optimize second**: Use `probe()` (params, peak VRAM) + `sec/epoch`; optimize only hot paths
  that are actually slow.
- **Compute on GPU, minimize host synchronization**: Avoid `.item()`/`print` each micro-batch; group logging by
  optimizer step; transfer tensors to CPU only when necessary (metrics/logging/artifacts).
- **Data pipeline**: Lazy loaders + keyed caching; do not read unused modalities; on CUDA use `pin_memory=True`
  + `.to(device, non_blocking=True)`; on Windows use `num_workers=0`; cache views/denoising/augmentation rather
  than recomputing them.
- **Data downloads**: Use `allow_patterns` for only the needed subset; token + `HF_XET_HIGH_PERFORMANCE=1` to
  maximize bandwidth.
- **AMP/TF32**: bf16 when available, otherwise fp16 + GradScaler; enable `cudnn.benchmark=True` and TF32 for
  matmul/conv on CUDA when supported by hardware; use `torch.compile` only after the architecture is stable.
- **Batch size against VRAM**: Instead of guessing, **choose batch size using the current training config and
  data** with
  `ft.find_batch_size(build_model, train_dataset, device=DEVICE, start=BS, target_gb=TARGET_VRAM_GB, num_workers=NUM_WORKERS)`.
  The helper runs real forward+backward+optimizer steps on the actual model/dataset, doubling batch size,
  measuring peak VRAM, and choosing the largest batch within budget (`min(TARGET_VRAM_GB, VRAM_total*0.9)` to
  leave headroom for eval/X-AI). Keep effective batch constant:
  `GRAD_ACCUM = max(1, round(EFFECTIVE_BATCH / BS))`.
- **Config-safe resume**: When `RESUME=True`, restore `batch_size`/`grad_accum` from `run_identity.pt` instead of
  probing again, so identity/config remains unchanged. Log `batch/size`, `batch/accum`, `batch/peak_gb`,
  `batch/target_gb` + `report/batch_probe_table`. On CPU/without CUDA, the probe returns `start` with
  `status="cpu"` and does not execute the probe.
- **num_workers**: Use >0 on Linux/Colab (per-sample dataset augmentation keeps results unchanged), 0 on Windows.
- **Heavy analysis** (MINE/surrogate/probe/IB, X-AI): Bound subset/steps/PCA-dim; cache embeddings instead of
  repeating forward passes; log tables/values instead of saving large tensors.
- **Do not recompute existing results**: Reuse caches (local/Drive) and consider HF upload for expensive
  preprocessing, subject to approval.

### 3.11 One Canonical Notebook per Architecture
- Each architecture/preset has **one canonical training notebook**. Optional stages (X-AI, information theory,
  embedding analysis) are **config flags** (`RUN_XAI`, `RUN_INFO`) in that notebook; do not duplicate/fork it.
- Preserve the cell order in section 1; reusable logic belongs in `scripts/*.py`; notebooks only configure
  and call it.
- New notebooks must pass `*_SMOKE=1` (CPU, synthetic) and the section 4 checklist before delivery.

### 3.12 Small Cells by Component
- One task/group of similar tasks per cell; at minimum separate as in table 1.1: **data** (download /
  build-denoise / views / dataset-loader), **training** (model / trainer+batch probe / fit loop), **evaluation**
  (val callback / test callback / calibrated report / table logging / X-AI / info).
- Do not combine data + training + evaluation in one cell; define evaluation callbacks in separate cells
  instead of embedding them in the training cell.
- **Must be visible in the notebook itself**: D1-D4, T1-T3, E1-E6 must be **separate cells with Markdown
  headings** (e.g. `### D1 - Download`, `### T1 - Model`, `### E1/E2 - Val/test callbacks`,
  `### T3 - Training loop`), not just listed in documentation. Apply the short-cell combination allowance in
  section 1.1 within a component, retaining clear step headings; optional E5/E6 follow their stage flags.
  Combining data/training/evaluation in one cell is **not acceptable**.
- Cells should be independently rerunnable for debugging; keep intermediate variables global; provide a
  Markdown heading for each section.
- Model: load from a path first (`ft.load_weights`, `WARM_START_WEIGHTS`/`best_weights.pt`); initialize a new
  model only if the path is absent (and only then download the pretrained backbone).

### 3.13 Step-by-Step Status Logging
- Print **one status line** for each step with a clear prefix: `[data]`, `[denoise]`, `[views]`, `[dataset]`,
  `[cache]`, `[model]`, `[batch]`, `[wandb]`, `[train]`, `[eval]`, `[checkpoint]`, `[report]`, `[xai]`, `[info]`,
  `[sync]`.
- Log milestones: data download/cache (pattern count, cache hit or new download), denoise/view building (cache
  hit or newly built, sample count), dataset (n + positive count per split, resolution), model (loaded from
  path or newly initialized, params), batch probe (BS/peak GB/budget), training start (device, epochs, effective
  batch, workers), each epoch (print `[val ]` and `[test ]` with the full main metric set, history,
  `train/epoch_seconds`, `[train] epoch N done in Xs`), checkpoint (saved file), calibrated report
  (temperature/threshold + test/val AUC/F1), X-AI/info start/end, Drive sync + finish.
- Keep logging **brief and nonredundant** with W&B (W&B remains the primary metric source); do not print in
  micro-batch loops.
- Prefer logging in shared helpers (`scripts/*.py`) so all notebooks use the same format; notebooks may add
  their own milestones.

### 3.14 Control / Ablation Notebooks
- Each variant **retrains from scratch** on the same split/seed/protocol; **do not** keep the full model and
  merely mask its input at test time. Removing a 3D/2D branch requires retraining the corresponding
  architecture/fusion.
- Put the spec table in config (code, model kwargs: `use_3d`, `n_2d`, `view_indices`, `fusion`, `gate_fixed`) and
  record it in the run config for accurate resumption/comparison.
- Use the same epoch budget and `PATIENCE` for every variant; select best by val AUC; run **>=3 seeds** and
  report **mean +/- std** (including per-seed results, not just the favorable seed).
- Each `(spec, seed)` pair is **a separate run**, resume-safe; include an aggregation cell for comparison tables
  (`*_summary.csv/json` + W&B table) and sync to Drive.
- Retrain the primary baseline (P) with the same budget if the protocol differs from the previous one.

These are requirements for an authorized control/ablation study, not authorization to add variants, retraining,
or seeds to a differently scoped experiment.

### 3.15 Mandatory Pre-Push Real-Run Check

Before committing or pushing any notebook/config intended for a real run, run the smoke step **locally**, then
switch every smoke/validation flag back to real-run mode. This switch is a required step of every push, not an
optional cleanup. A pushed artifact must be immediately runnable as the real experiment.

- Set every `*_SMOKE` flag to real-run: unset, or explicitly `0`/`false`. Never leave `SMOKE=True`,
  `os.environ['<PREFIX>_SMOKE'] = '1'`, `setdefault(..., '1')`, or any smoke default on.
- Remove smoke-only defaults: tiny `EPOCHS`/sample caps, synthetic-data branches, offline W&B, `/tmp` outputs,
  reduced `RES2D`/`RES3D`, and smoke `RUN_GROUP`/`RUN_TARGET` values.
- Set real experiment identity and stage switches deliberately (`RUN_GROUP`, `RUN_TARGET`, `RESUME`,
  `EXTEND_EPOCHS`, `RUN_XAI`, `RUN_INFO`), consistent with the approved scope.
- Keep smoke support in the code (env-gated), but the committed state must be the real-run state.
- Verify before pushing:
  ```bash
  rg -n "_SMOKE|SMOKE\s*=\s*True|setdefault\(['\"][A-Z_]*SMOKE" notebooks configs scripts
  ```
  Every hit must be env-gated with a real-run default; no hit may force smoke on.
- Report the check alongside the commit/push; do not push if any smoke flag is left enabled.

## 4. Pre-Delivery Notebook Checklist

- [ ] Follow section 1 cell order; use a single config source with env overrides.
- [ ] **Group config-cell parameters with `#` lines** (editable vs. `FROZEN`/derived groups), making clear what to configure where without accidentally changing stabilized settings.
- [ ] Separate cells by component (one task/group per cell), allowing individual cells to be rerun during debugging/fixes.
- [ ] Separate data (download/build-denoise/views/dataset-loader), training (model/trainer-probe/fit), and evaluation (val callback, test callback, calibrated report, table logging) cells according to table 1.1.
- [ ] Verify that the actual notebook has separate cells + Markdown headings for D1-D4/T1-T3/E1-E6, subject to the short-cell and optional-stage qualifications above, not combined data/training/evaluation or documentation-only labels.
- [ ] Each step prints prefixed status logs (`[data]`/`[model]`/`[train]`/`[eval]`/`[xai]`/...) for monitoring/debugging.
- [ ] Every epoch logs completion time (`train/epoch_seconds` + `[train] epoch N done in Xs`) to measure `sec/epoch`.
- [ ] Every epoch clearly prints `[val ]` and `[test ]` (main metrics) to stdout, not only W&B.
- [ ] Load model weights from a path first (`WARM_START_WEIGHTS`/`best_weights.pt`); initialize a new model only when weights are absent, following the resume priority.
- [ ] `*_SMOKE=1` passes all cells on CPU with synthetic data and no downloads.
- [ ] **Pre-push real-run check** (section 3.15): run smoke locally, then set every `*_SMOKE` flag back to real run (unset/`0`), remove smoke-only defaults, and set real identity/stage switches before pushing.
- [ ] W&B initialization + live logs with correct prefixes + X-AI/image figures as `wandb.Image` + `summary.update` + `finish`.
- [ ] **Train logs the full metric set every optimizer step; val + test log the full metric set every epoch; test monitoring is never used for selection.**
- [ ] **The final report uses the best validation checkpoint and includes calibrated `train`/`val`/`test` + bootstrap CI and W&B `report/split_table`.**
- [ ] **After training, run X-AI on the best validation model and log `xai/fusion_table` + `xai/*` values to W&B (or explicitly document an exception).**
- [ ] Log metrics as W&B tables/scalars; **do not generate metric plots** (figures are only for X-AI/images).
- [ ] Info-theory (if applicable): multiseed DV/NWJ/InfoNCE with negative-rate (no clamping); >=1000 MIC surrogates; interaction information for redundancy/synergy; all logged as tables/scalars.
- [ ] Control/ablation notebook: retrain each variant, same budget/patience, >=3 seeds, mean +/- std table, and a separate run per (spec, seed), within the authorized study scope.
- [ ] **Optional stages (X-AI/info-theory) are flags in the same notebook, not separate/forked notebooks; logic is in `scripts/`.**
- [ ] Sync every artifact (figure/model/CSV/report/X-AI) to Drive as soon as produced; existing artifacts, including legacy metric figures, must still sync.
- [ ] HF downloads use `HF_TOKEN` + `allow_patterns` for only the required subset; real runs fail early without a token.
- [ ] `STORE_RES` is config-driven (200/128/96/...), not hardcoded; check shapes against `STORE_RES`; keep it separate from `RES3D`/`RES2D`.
- [ ] Document decisions about caching/reusing/uploading processed data (denoising/views/augmentation); view-cache reuse across `RES3D` must match `RES2D`, source, and projection identity.
- [ ] Enable AMP on CUDA + `pin_memory`/`non_blocking`; minimize `.item()`/host synchronization in the training loop.
- [ ] Lazy/cached loaders read only data in use; expensive preprocessing is cached/reused.
- [ ] Measure `sec/epoch` + peak VRAM; bound heavy analysis (MI/X-AI) by subset/steps.
- [ ] Select batch size against config+data (`ft.find_batch_size`) within the VRAM budget; preserve effective batch through grad-accum; reuse the saved batch on resume.
- [ ] Select the best checkpoint by val AUC; fit threshold/calibration on val, not test; use that checkpoint for the final held-out report.
- [ ] Resume-safe: append `results.json` + atomic checkpoints.
- [ ] The training cell supports extension (`RESUME=True` + `EXTEND_EPOCHS=N`) without a new `RUN_GROUP`; do not execute an extension without authorization.
- [ ] `ruff check` is clean for new Python files; unit tests for new core logic pass.
- [ ] Clearly document notes/limitations.

## 5. Reference Notebooks

| Notebook | Reference purpose |
|---|---|
| [`3d_glaucoma_multiview_sota_sweep_xai.ipynb`](../notebooks/3d_glaucoma_multiview_sota_sweep_xai.ipynb) | Full sweep structure: backbone registry, fusion ablation, X-AI, resume safety, reporting |
| [`3d_glaucoma_train_3branch_crossgate.ipynb`](../notebooks/3d_glaucoma_train_3branch_crossgate.ipynb) | Canonical three-branch training notebook (1x3D ResNeXt + 2x2D MaxViT + CrossGate): 3D at **96^3**, resized on-the-fly, **2D views projected from raw 200^3**; full metrics on all three splits, X-AI, optional info-theory (`RUN_INFO`), resume, smoke |
| [`3d_glaucoma_final_2x2d_3d_crossgate.ipynb`](../notebooks/3d_glaucoma_final_2x2d_3d_crossgate.ipynb) | Canonical training notebook for the **Bilateral 200^3** dataset (denoise cache + recovery contract): full metrics on all three splits, X-AI + W&B tables, optional warm-start, smoke |
| [`3d_glaucoma_controls_96.ipynb`](../notebooks/3d_glaucoma_controls_96.ipynb) | 96^3 control notebook: B1-B3 (branch removal), C1-C2 (fusion), P (baseline); each spec x 3 seeds, 20 epochs with early stopping, mean +/- std table |
| [`3d_glaucoma_resolution_96_128_200.ipynb`](../notebooks/3d_glaucoma_resolution_96_128_200.ipynb) | Resolution comparison on the same backbone |
| [`3d_glaucoma_denoise_gpu_compare.ipynb`](../notebooks/3d_glaucoma_denoise_gpu_compare.ipynb) | Denoising-method comparison + image metrics |

Reusable core: `scripts/final_model.py` (ResNeXt3D/Timm2D/CrossGate/metrics/X-AI), `scripts/final_training.py`
(Trainer recovery/W&B), `scripts/information_theory.py` (probe/MINE/MIC/surrogate/IB),
`scripts/resolution_study.py` (numpy metrics), `scripts/compare_denoise_methods.py` (denoising).
