---
name: train-notebook-builder
description: Use ONLY when creating or editing a Jupyter/Colab notebook to train, fine-tune, sweep, or research a model in this repo (glaucoma-thesis). Trigger on requests like "tạo notebook train", "build training notebook", "notebook sweep", "notebook X-AI", "train model trên 2 tập", "resolution study", "smoke test notebook", or when a notebook must log wandb + save to Drive. Do not use for pipeline/script-only changes or non-notebook tasks.
---

# Train / research notebook builder (glaucoma-thesis)

Build standalone Colab notebooks for training and model research that follow this repo's
conventions and the patterns already proven in existing notebooks. The goal is notebooks that
run end-to-end on Colab GPU **without** the bugs previously hit in the sweep notebook.

## When to use
- User asks to create/build a notebook to train, fine-tune, sweep, ablate, or research a model.
- User asks for a notebook with wandb, Drive sync, smoke test, X-AI, metrics, or plots.

## Before writing anything
1. Read `AGENTS.md` (criteria + flows) — it is the contract.
2. Read the closest existing notebook(s) and reuse their structure/idioms:
   - `notebooks/3d_glaucoma_final_2x2d_3d_crossgate.ipynb` — main training (2×2D + 1×3D CrossGate, metrics, X-AI, smoke).
   - `notebooks/3d_glaucoma_multiview_sota_sweep_xai.ipynb` — registry sweep + X-AI (source of many bug fixes).
   - `notebooks/3d_glaucoma_resolution_96_128_200.ipynb`, `..._96_vs_128_hf.ipynb` — resolution comparison.
   - `notebooks/3d_glaucoma_noise2void_train.ipynb`, `..._denoise_gpu_compare.ipynb` — self-supervised / denoise.
   - `notebooks/3d_glaucoma_bm3d_preprocess_upload.ipynb` — streaming build + HF upload (only if explicitly asked).
3. Put reusable logic in `scripts/*.py`, test it locally, and keep the notebook thin. Proven cores:
   - `scripts/final_model.py` — ResNeXt3D, Timm2D (MaxViT), CrossGate, FinalModel, full metrics, X-AI, `aug_pair`.
   - `scripts/resolution_study.py` — numpy metrics (PSNR/SSIM/CKA/ECE/AUC/AP/silhouette).
   - `scripts/denoise_torch.py` — GPU denoisers (DnCNN/SwinIR), model cache.
   - `scripts/compare_denoise_methods.py` — classical denoisers (bilateral/tv/nlm/…), `denoise_volume`.
   - `scripts/denoise_torch.py` + `scripts/n2v.py` — N2V.
   - `scripts/compare_resolutions.py` — resolution comparison runner.

## Hard rules (from AGENTS.md — do not skip)
- **W&B mandatory** every run: `init_wandb(run_name, config)` before training; log **live** per epoch
  with **correct prefixes** (`train/...` for train, `val/...` for val, `test/...` for test); log
  figures as `wandb.Image`; `run.summary.update(...)`; `run.finish()`. Never disable wandb.
- **Drive sync mandatory for EVERY artifact**: figures, CSV/JSON, models → repo Drive root
  `/content/drive/MyDrive/MasterBKDN/Thesis/<experiment>[_figures]` (mirror paths used by other notebooks).
  Guard mount so headless runs don't fail.
- **Data**: raw **200³ only**; never persist downsampled arrays. `data/` is gitignored — **never commit
  or push data**. Do not push datasets to HF unless the user explicitly asks.
- **Config**: `STORE_RES` (storage) and `MODEL_RES`/`RES3D`/`RES2D` (model input) must be **distinct names**
  (shadowing caused a real bug).
- **Style**: no code comments unless asked; line length 120; follow existing registry/config-driven patterns.
- **Forward contract**: return `{"logits": tensor}` or a tensor of logits; pair with `CrossEntropyLoss`.

## Notebook skeleton (cells)
1. **Title/overview** (markdown): pipeline, how to run, smoke instructions.
2. **Setup**: `!git clone ... || git pull`, `%cd`, `!pip install -q ...`, `!nvidia-smi`.
3. **Imports + device + seed**: `sys.path.insert(0,"scripts")`, import core modules; set `DEVICE`, seeds.
4. **Config**: `SMOKE = os.environ.get("..._SMOKE","0")=="1"`; all hyperparams; Drive dirs; HF repos.
5. **Data**: `snapshot_download` HF consolidated repo **or** build from Harvard-GF; cache views/dz once
   (`*_views.npy`, `*_dzs.npy`); denoise build is **resume-safe and limited** (`DENOISE_LIMIT`).
6. **Dataset + loaders**: memmap volumes; return tensors only; augmentation applied **consistently** to
   3D and 2D (`final_model.aug_pair`); `num_workers=0` on Windows.
7. **Model + optimizer + class weights + train loop**: AMP, grad-accum, warmup+cosine, grad-clip,
   early-stop, **checkpoint by val AUC/PR-AUC** (not test acc).
8. **Post-processing**: threshold via Youden-J on val; temperature scaling on val logits.
9. **Metrics**: full set (acc, balanced acc, precision, recall/sensitivity, specificity, F1, MCC,
   AUC-ROC, PR-AUC, ECE) + bootstrap CI; save CSV + JSON.
10. **Plots**: learning curves, ROC, PR, calibration, confusion → save PNG.
11. **X-AI** (when relevant): Grad-CAM 3D/2D, occlusion, integrated gradients, fusion attention,
    branch-drop; log each as `wandb.Image`.
12. **Drive sync + wandb finish**.
13. **Notes/limitations** (markdown): single-seed caveat, missing metrics as `—`, confounds.

## Smoke test (MANDATORY before handing off)
- Provide a `SMOKE=1` mode using synthetic small data (no downloads) that runs on CPU.
- Locally validate by executing the notebook's code cells in order, stripping `!`/`%` magics, with the
  smoke env var set (see the harness pattern below). All cells must pass; then push.
- Also unit-test any new `scripts/*.py` core locally (forward/backward + one X-AI call) before wiring it in.

Harness pattern (run from repo root; Windows: set `PYTHONIOENCODING=utf-8`):
```python
import json, os, sys, traceback
os.environ["<PREFIX>_SMOKE"] = "1"
nb = json.load(open("notebooks/<name>.ipynb", encoding="utf-8"))
g = {"__name__": "__main__"}
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    body = "\n".join(l for l in "".join(cell["source"]).splitlines()
                     if not l.lstrip().startswith(("!", "%")))
    exec(compile(body, f"<cell{i}>", "exec"), g)
print("SMOKE OK")
```

## Known-bug checklist (these all happened — verify every time)
- **Dataset item dims**: 3D item is `(1,D,H,W)`; interpolate via `x.unsqueeze(0)` then `.squeeze(0)`.
  Do **not** use `raw[None, None]` (produces 6D after collate).
- **Collate**: `idx` must be a tensor or handled separately — `torch.stack` of `int` fails.
- **memmap 5D**: never `vols[:,0]` (loads 26 GB); index `vols[i]` and squeeze per sample.
- **matplotlib**: `ax.set_title(...).set_xlabel(...)` fails (`Text` has no `set_xlabel`) — use separate calls.
- **Sort keys**: use `-((x) or 0)`, never `-(x) or 0` (unary minus binds tighter than `or`).
- **wandb prefixes**: log val under `val/`, train under `train/`; never dump a mixed history dict under `train/`.
- **AMP**: `torch.amp.GradScaler("cuda", enabled=...)`; autocast only when CUDA.
- **HF auth**: `load_env_file()` / Colab Secret `HF_TOKEN`; pass token to `hf_hub_download`/`snapshot_download`.
- **Denoise scope**: classical CPU denoisers (bilateral) are slow — always expose a limit and resume;
  never silently denoise the entire dataset.
- **Git**: `git pull --rebase`/fetch before pushing; Colab commits churn notebook JSON — merge with `-X ours`
  to keep local code edits.

## Definition of done
- Notebook compiles (all code cells), smoke passes locally, wandb + Drive wired with correct prefixes and
  figure logging, metrics complete, no data committed, and the user is told to **restart opencode** if any
  config changed.
