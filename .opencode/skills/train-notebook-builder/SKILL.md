---
name: train-notebook-builder
description: Use ONLY when creating or editing a Jupyter/Colab notebook to train, fine-tune, sweep, or research a model in this repo (glaucoma-thesis). Trigger on requests like "tạo notebook train", "build training notebook", "notebook sweep", "notebook X-AI", "train model trên 2 tập", "resolution study", "smoke test notebook", or when a notebook must log wandb + save to Drive. Do not use for pipeline/script-only changes or non-notebook tasks.
---

# Train / research notebook builder (glaucoma-thesis)

Build standalone Colab notebooks for training and model research that follow this repo's
canonical English conventions in `docs/notebook-conventions.en.md` and proven notebook patterns. The goal is notebooks that
run end-to-end on Colab GPU **without** the bugs previously hit in the sweep notebook.

## When to use
- User asks to create/build a notebook to train, fine-tune, sweep, ablate, or research a model.
- User asks for a notebook with wandb, Drive sync, smoke test, X-AI, metrics, or plots.

## Before writing anything
1. Read `AGENTS.md` (criteria + flows) — it is the contract.
2. Always read `docs/notebook-conventions.en.md` before creating, editing, or reviewing a notebook.
   It is the canonical notebook specification for agents; `docs/notebook-conventions.md` is for Vietnamese
   human readers. Keep both versions synchronized when changing conventions. Explicit experiment limits still apply.
3. Read the closest existing notebook(s) and reuse their structure/idioms:
   - `notebooks/3d_glaucoma_final_2x2d_3d_crossgate.ipynb` — main training (2×2D + 1×3D CrossGate, metrics, X-AI, smoke).
   - `notebooks/3d_glaucoma_multiview_sota_sweep_xai.ipynb` — registry sweep + X-AI (source of many bug fixes).
   - `notebooks/3d_glaucoma_resolution_96_128_200.ipynb`, `..._96_vs_128_hf.ipynb` — resolution comparison.
   - `notebooks/3d_glaucoma_noise2void_train.ipynb`, `..._denoise_gpu_compare.ipynb` — self-supervised / denoise.
   - `notebooks/3d_glaucoma_bm3d_preprocess_upload.ipynb` — streaming build + HF upload (only if explicitly asked).
4. Put reusable logic in `scripts/*.py`, test it locally, and keep the notebook thin. Proven cores:
   - `scripts/final_model.py` — ResNeXt3D, Timm2D (MaxViT), CrossGate, FinalModel, full metrics, X-AI, `aug_pair`.
   - `scripts/resolution_study.py` — numpy metrics (PSNR/SSIM/CKA/ECE/AUC/AP/silhouette).
   - `scripts/denoise_torch.py` — GPU denoisers (DnCNN/SwinIR), model cache.
   - `scripts/compare_denoise_methods.py` — classical denoisers (bilateral/tv/nlm/…), `denoise_volume`.
   - `scripts/denoise_torch.py` + `scripts/n2v.py` — N2V.
   - `scripts/compare_resolutions.py` — resolution comparison runner.

## Hard rules (summary of AGENTS.md and the English conventions)
- **W&B mandatory** every run: load credentials before `init_wandb(run_name, config)`; initialize before training;
  log **live** train metrics per optimizer step and val/test metrics per epoch
  with **correct prefixes** (`train/...` for train, `val/...` for val, `test/...` for test); log
  figures as `wandb.Image`; `run.summary.update(...)`; `run.finish()`. Never disable wandb.
- **Drive sync mandatory for EVERY artifact**: figures, CSV/JSON, models → repo Drive root
  `/content/drive/MyDrive/MasterBKDN/Thesis/<experiment>[_figures]` (mirror paths used by other notebooks).
  Guard mount so headless runs don't fail.
- **Data**: config-driven `STORE_RES` (default raw 200 cubed), distinct from model input resized on the fly.
  Authenticated selective HF downloads require `HF_TOKEN`, `token=`, and `allow_patterns` or per-file downloads.
  Use `HF_XET_HIGH_PERFORMANCE=1`; reuse caches matching source/projection/resolution identity.
  `data/` is gitignored; never commit or push data. HF publication requires user approval.
- **Config**: `STORE_RES` (storage) and `MODEL_RES`/`RES3D`/`RES2D` (model input) must be **distinct names**
  (shadowing caused a real bug).
- **Style**: no code comments unless asked, except mandatory config-group headings from section 2.3 of the
  English conventions; line length 120; follow existing registry/config-driven patterns.
- **Forward contract**: return `{"logits": tensor}` or a tensor of logits; pair with `CrossEntropyLoss`.
- **Pre-push real-run**: run smoke locally first, then leave every `*_SMOKE` flag off (unset/`0`) with real
  identity/stage switches before committing or pushing (English conventions section 3.15).

## Notebook skeleton (cells)
Follow sections 1 and 1.1 of `docs/notebook-conventions.en.md`; this is only a summary, not a separate specification.
1. **Title/overview** (markdown): pipeline, how to run, smoke instructions.
2. **Setup**: `!git clone ... || git pull`, `%cd`, `!pip install -q ...`, `!nvidia-smi`.
3. **Imports + device + seed**: `sys.path.insert(0,"scripts")`, import core modules; set `DEVICE`, seeds.
4. **Config**: `SMOKE = os.environ.get("..._SMOKE","0")=="1"`; all hyperparams; Drive dirs; HF repos.
5. **Data (D1-D4)**: authenticated selective HF download **or** build from Harvard-GF; cache views/dz once
   (`*_views.npy`, `*_dzs.npy`); denoise build is **resume-safe and limited** (`DENOISE_LIMIT`).
6. **Dataset + loaders**: memmap volumes; return tensors only; augmentation applied **consistently** to
   3D and 2D (`final_model.aug_pair`); `num_workers=0` on Windows.
7. **Train (T1-T3, separate headed cells)**: model/load weights, trainer + batch probe, then fit.
   Use AMP, grad-accum, warmup+cosine, grad-clip, and **checkpoint by val AUC/PR-AUC** (not test acc).
   Define val/test callbacks in separate evaluation cells; never combine data/train/eval in one cell.
8. **Post-processing**: threshold via Youden-J on val; temperature scaling on val logits.
9. **Metrics**: full set (acc, balanced acc, precision, recall/sensitivity, specificity, F1, MCC,
   AUC-ROC, PR-AUC, ECE) + bootstrap CI; save CSV + JSON.
10. **Reporting**: W&B tables/scalars plus CSV/JSON; no metric PNGs (history/ROC/PR/calibration/confusion).
11. **X-AI** (when relevant): Grad-CAM 3D/2D, occlusion, integrated-gradient heatmaps as `wandb.Image`.
    Log fusion attention and branch-drop values as `xai/fusion_table` and `xai/*` scalar summaries, not charts.
12. **Drive sync + wandb finish**.
13. **Notes/limitations** (markdown): single-seed caveat, missing metrics as `—`, confounds.

## Smoke test (MANDATORY before handing off)
- Provide a `SMOKE=1` mode using synthetic small data (no downloads) that runs on CPU.
- Locally validate by executing the notebook's code cells in order, stripping `!`/`%` magics, with the
   smoke env var set (see the harness pattern below). All cells must pass; push only when requested.
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
- **Git**: fetch and inspect both versions before merging notebook JSON; preserve local and remote changes.
  Do not use blanket conflict overrides. Commit/push only when requested.

## Definition of done
- Notebook compiles (all code cells), smoke passes locally, wandb + Drive wired with correct prefixes and
  figure logging, metrics complete, no data committed, and the user is told to **restart opencode** if any
  config changed.
- **Mandatory pre-push real-run check** (`docs/notebook-conventions.en.md` §3.15): after running smoke locally,
  set every `*_SMOKE` flag back to real run (unset/`0`), remove smoke-only defaults and smoke identity, and
  confirm the pushed notebook/config starts the real experiment. Report the check with the push.
