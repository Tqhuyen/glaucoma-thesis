# AGENTS.md

Working guide for AI agents contributing to this repository. It documents the
**criteria** (what "done" / "correct" means) and the **flows** (how data, training,
logging, and experiments work) so agents don't guess.

## Project in one paragraph

Master thesis: **3D CNN for glaucoma detection from OCT volumes**. One
model-agnostic training pipeline that runs unchanged on Colab, vast.ai, a single
GPU box, or multi-node SLURM. The thesis dataset is **Harvard-GF** (3D OCT
volumes, 200³ uint8), with **raw 200³ data only** — no 96³/128³ downsampled
configs.

## Criteria (what counts as "correct")

### Code style
- Line length 120; `ruff` enforced (see `pyproject.toml`):
  - `make lint` = `ruff check pipeline models tests` + `ruff format --check`
  - **Never edit archived notebooks for lint** — pre-existing notebook lint noise is accepted
    (CI only lints `pipeline models tests`, not `notebooks/`).
- No code comments unless asked; follow existing patterns (registry pattern,
  config-driven, rank-0-only logging).
- `forward()` contract: return `{"logits": tensor}` (or HF-style object with `.loss`).
- DataLoader must return `(x, y)` tuples or dicts with `"labels"`.

### Tests
- `make test` / `pytest tests -q` must pass (9 tests).
- Config schema validates in <1s (`pipeline/config.py`): presence + type + range.
  Optional keys validated only if present (old configs stay valid).

### Config as contract
- `configs/glaucoma.yaml` is the **only** glaucoma config (raw 200³). Never
  re-add `glaucoma_96.yaml` / `glaucoma_128.yaml`.
- Every config change must keep `validate_config(cfg)` green (add optional keys
  to `_OPTIONAL_SCHEMA`, not the required base schema, unless truly required).

### Model convergence (the "4 traps" checklist)
When a 3D CNN on 200³ OCT doesn't converge, check in this order:
1. **Resolution destruction** — pooling must NOT crush the thin RNFL layer.
   Fix: anisotropic pooling stride `(2,2,1)` keeps B-scan depth resolution;
   `AdaptiveAvgPool3d(1)` only at the very end. See `models/glaucoma/model.py`.
2. **BatchNorm with tiny batch** — batch_size 2–4 (VRAM) breaks BatchNorm.
   Fix: **GroupNorm** (`norm: "group"`, `norm_groups: 8`). Gradient accumulation
   does NOT help BN (it only affects effective batch).
3. **Pixel intensity** — data is uint8 0–255; `/255.0` (minmax) is correct.
   For robustness use `normalize: "robust"` (percentile 1/99 clip + z-score).
   Always verify `min/max` before training.
4. **Loss/output mismatch** — raw logits (`nn.Linear(64, num_classes)`) must pair
   with `CrossEntropyLoss` (never sigmoid+BCELoss with raw logits, never MSE).

**Overfit sanity gate (mandatory before blaming data):**
`make sanity` (or `python -m pipeline.train --cfg configs/glaucoma.yaml --sanity`)
overfits 10 samples, 100 epochs, dropout=0/wd=0/warmup=0, no eval/save.
- Loss → ~0 (e.g. 1e-3) and train acc → 1.0 ⇒ code is correct; problem is
  architecture/LR/data size.
- Loss stuck ⇒ bug is in code/arch/data, not the dataset.

### Commands
```bash
make setup    # pip install -e ".[dev,glaucoma,nlp]"
make lint     # ruff check pipeline models tests
make test     # pytest tests -q
make smoke    # quick end-to-end validation (CPU)
make sanity   # overfit-10-samples convergence gate
make train    # full train on raw 200³ (configs/glaucoma.yaml)
make ddp      # single-node multi-GPU
make tb       # TensorBoard live at localhost:6006
make plot     # re-render training_curves.png from metrics.jsonl
```

## Flows

### Data flow
```
harvardairobotics/Harvard-GF (HF, per-scan .npz 'oct_bscans' 200³ uint8)
  → scripts/harvard_oct_processor.py   (streams + uploads consolidated .npy)
  → glaucoma_all/{Training,Validation,Test}_{volumes,labels}.npy
  → models/glaucoma/data.py            (GlaucomaNpyDataset, mmap)
      normalize: minmax (default) | robust | none
  → pipeline/train.py
```
- Store raw **200³** on disk; never persist downsampled arrays.
- mmap (`cache_in_ram: false`) for 200³ (~26 GB); on Windows use `num_workers=0`
  (mmap + multiprocessing can segfault).

### Training flow (pipeline/train.py)
```
config.yaml → validate_config (fast fail) → build loaders → build model
  → DDP wrap → AdamW + warmup/cosine + AMP GradScaler
  → loop: compute_loss → scaler.backward → clip → step (grad_accum)
  → live train/acc every log_every_steps
  → val eval every eval_every_steps → best-model save + early stop
  → test eval every eval_test_every_steps (live) + final held-out test
  → render_and_sync(): PNG curves + metrics.jsonl + best_model.pt → Drive
  → run_manifest.json (reproducibility)
```
- `--smoke` / `--sanity` / `--resume` profiles applied via `apply_profile(cfg, args)`.
- SIGTERM/SIGINT → atomic checkpoint → clean exit; `--resume` continues.
- Rank 0 exclusively: logging, checkpointing, test eval, plotting.

### Logging flow (3 sinks + Drive)
- **TensorBoard**: live, `outputs/<run>/tb` (Colab inline / `make tb`).
- **JSONL**: `outputs/<run>/metrics.jsonl` — per split (`train/`, `val/`, `test/`):
  `loss`, `acc`, `precision`, `recall`, `f1` (+ `_pos` = class-1/glaucoma-positive),
  plus `sys/cpu_percent`, `sys/ram_used_gb`, `sys/ram_percent`, `sys/disk_free_gb`,
  `sys/gpu_*` (when CUDA).
- **W&B** (optional): `logging.wandb: true`, project `glaucoma-thesis`.
  - Same keys as JSONL, logged live every `log_every_steps` + each eval.
  - GPU/CPU/RAM/disk auto-tracked by wandb's System panel; `sys/*` metrics also
    logged explicitly.
  - Key from `.env` (`WANDB_API_KEY`) via `load_env_file()` in
    `pipeline/utils.py` — **always call `load_env_file()` before `wandb.init()`**.
  - `.env` is gitignored; never commit real keys.
- **Drive sync**: `logging.drive_sync_dir: "/content/drive/MyDrive/..."` copies
  `training_curves.png` + `metrics.jsonl` + `best_model.pt` after each run
  (`pipeline/plotting.py`).

### Notebook conventions
- Notebooks are standalone Colab experiments (inline code, `!pip`/`!git` cells).
- **Variable shadowing trap**: define storage resolution (`STORE_RES=200`, raw)
  and model-input resolution (`MODEL_RES=112`) as **distinct names**. Do NOT
  reuse one `RESOLUTION` var for both — cell 6 must not overwrite cell 4's value
  (this caused a real patch-embed assert bug).
- 3DINO-ViT needs input divisible by patch 16 → 112³ (7³ tokens). Resize
  on-the-fly in `preprocess_volume` (cast to float **before** `F.interpolate` —
  trilinear fails on uint8), never persist the downsampled data.
- Wandb in notebooks: `init_wandb(run_name)` helper (idempotent via
  `WANDB_RUN is None` guard), log probe + per-epoch train/val + test, then
  `run.summary.update()` + `run.finish()`.

## Repo layout (key files)
```
pipeline/train.py        training loop (DDP, AMP, preemption, test eval)
pipeline/config.py       schema validation (required + optional keys)
pipeline/metrics.py      TensorBoard + JSONL + wandb (load_env_file first!)
pipeline/plotting.py     render_curves() + sync_to_drive() + render_and_sync()
pipeline/utils.py        seeding, atomic save, load_env_file()
models/glaucoma/model.py Simple3DCNN: GroupNorm + (2,2,1) pool + residual
models/glaucoma/data.py  GlaucomaNpyDataset (minmax/robust/none)
configs/glaucoma.yaml    the ONLY glaucoma config (raw 200³)
notebooks/3d_glaucoma_3dino_experiment.ipynb  3DINO-ViT transfer + wandb
scripts/                 colab_setup, vast_setup, launch_ddp, slurm
```
