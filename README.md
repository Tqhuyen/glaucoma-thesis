# Glaucoma Thesis — 3D CNN Classification with Enterprise Training Pipeline

[![CI](https://github.com/Tqhuyen/glaucoma-thesis/actions/workflows/ci.yml/badge.svg)](https://github.com/Tqhuyen/glaucoma-thesis/actions/workflows/ci.yml)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Tqhuyen/glaucoma-thesis)

Master thesis: **3D CNN for glaucoma detection from OCT scans**. One training pipeline that runs unchanged on Colab, vast.ai, an 8-GPU box, or a multi-node SLURM cluster — disconnect-proof, preemption-safe, CI-gated.

## Scaling Matrix

One `pipeline/train.py`, four launch commands:

| Hardware | Command |
|---|---|
| 1 GPU (Colab, vast.ai, local) | `python -m pipeline.train --cfg configs/glaucoma_96.yaml` |
| N GPUs, one node | `bash scripts/launch_ddp.sh configs/glaucoma_96.yaml` |
| Multi-node, manual | `torchrun --nnodes=$N --node_rank=$R --master_addr=$ADDR --nproc_per_node=8 ...` |
| SLURM cluster | `sbatch scripts/slurm_train.sbatch` |

Effective batch = `batch_size × world_size × grad_accum_steps` — printed at startup.

## Quick Start

### Local

```bash
git clone https://github.com/Tqhuyen/glaucoma-thesis.git
cd glaucoma-thesis
pip install -e ".[glaucoma]"
python -m pipeline.train --cfg configs/glaucoma_96.yaml --smoke --output-dir /tmp/smoke
python -m pipeline.train --cfg configs/glaucoma_96.yaml
```

### Google Colab

```bash
# Cell 1: GPU check
!nvidia-smi

# Cell 2: HF token from Colab Secrets (key icon → HF_TOKEN → hf_xxx)
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

# Cell 3: clone, install, train
!git clone https://github.com/Tqhuyen/glaucoma-thesis.git
%cd glaucoma-thesis
!pip install -q -e ".[glaucoma]"
!python -m pipeline.train --cfg configs/glaucoma_96.yaml

# Cell 4: live TensorBoard
%load_ext tensorboard
%tensorboard --logdir outputs
```

Or one-shot: `bash scripts/colab_setup.sh glaucoma_96`

### vast.ai

```bash
# On instance creation (GPU idles, data downloads at full bandwidth):
git clone https://github.com/Tqhuyen/glaucoma-thesis.git
cd glaucoma-thesis
bash scripts/vast_setup.sh glaucoma_96 download

# Then start training in tmux:
bash scripts/vast_setup.sh glaucoma_96 train

# Watch: tmux attach -t train
# TensorBoard: ssh -L 6006:localhost:6006 ... then http://localhost:6006

# If preempted — resume seamlessly:
bash scripts/vast_setup.sh glaucoma_96 resume
```

### Makefile Shortcuts

```bash
make setup        # install deps + dev tools
make lint         # ruff check
make test         # pytest (9 tests)
make smoke        # CPU smoke train (128 samples, ~2min)
make train-g96    # full train, 96³ resolution
make train-g128   # full train, 128³ resolution
make ddp          # all GPUs on this node
make tb           # TensorBoard at localhost:6006
```

## Architecture

### Model Registry Pattern

This repo uses a **model-agnostic training pipeline** with a registry. Adding a new model requires no changes to `pipeline/train.py` — you only add a module in `models/` and a config YAML.

```
configs/glaucoma_96.yaml
    model.type: "glaucoma" ─────────┐
                                    ▼
                          models/__init__.py     models/glaucoma/
                          _MODEL_REGISTRY   →   ├── __init__.py  (build_model, build_dataloaders)
                          {                     ├── model.py     (Simple3DCNN)
                            "glaucoma": ...     └── data.py      (.npy DataLoader)
                            "nlp": ...
                          }
                                    │
                                    ▼
pipeline/train.py  ←──  same training loop for ALL model types
  DDP · AMP · grad-accum · NaN guard · preemption · early-stop · 3 dashboards
```

### Data Flow

```
 config.yaml ──► pipeline/config.py  (schema validation, fails in <1s)
                      │
 HF / local ──► models/<type>/data.py  (model-specific dataloaders)
                      │
 torchrun ──► pipeline/train.py  ── DDP(model) · AMP · grad-accum · NaN guard
                      │              │
                      │              ├─ SIGTERM (spot reclaim / disconnect)
                      │              │     └─► checkpoint → exit 0 → --resume
                      │              ├─ rank 0: atomic checkpoints + prune
                      │              └─ rank 0: MetricsLogger
                      │                    ├─ TensorBoard (live)
                      │                    ├─ metrics.jsonl
                      │                    └─ wandb (optional)
                      ▼
           final TEST eval → test_results.json
```

## Thesis Model: Glaucoma Detection

### Dataset

3D OCT volumes (retinal scans), binary classification (glaucoma: yes/no).

| Split | Samples | Class 0 | Class 1 |
|---|---|---|---|
| Training | 2,100 | 1,017 | 1,083 |
| Validation | 300 | 124 | 176 |
| Test | 900 | 411 | 489 |

Two resolutions available: **96³** and **128³** (resized from original 200×200×200).

### Model Architecture

`Simple3DCNN`: 4 convolutional layers (16→32→64→128 channels) with BatchNorm, ReLU, and MaxPool3d, followed by AdaptiveAvgPool3d and a classifier head (128→64→2) with dropout.

```
Input: (1, S, S, S)   where S = 96 or 128
  → Conv3d(1→16)  → BN → ReLU → MaxPool3d   (S/2)
  → Conv3d(16→32) → BN → ReLU → MaxPool3d   (S/4)
  → Conv3d(32→64) → BN → ReLU → MaxPool3d   (S/8)
  → Conv3d(64→128)→ BN → ReLU → AdaptiveAvgPool3d(1)
  → Flatten → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→2)
```

### Configs

| Config | Resolution | Batch Size | Use Case |
|---|---|---|---|
| `configs/glaucoma_96.yaml` | 96³ | 4 | Faster iteration, T4 GPU |
| `configs/glaucoma_128.yaml` | 128³ | 2 | Higher resolution, V100/A100 |

### Original Thesis Artifacts

Archived in `models/glaucoma/notebooks/`:
- `thesis_3d_glaucoma.py` — original Colab script
- `3d_glaucoma_optimized.ipynb` — optimized version (AMP, torch.compile, caching)

Training plots in `models/glaucoma/results/`.

## Adding a New Model

Three steps, zero changes to `pipeline/train.py`:

### 1. Create the model module

```
models/resnet/
├── __init__.py       # def build_model(cfg), def build_dataloaders(cfg, smoke)
├── model.py          # Your model class (forward must return {"logits": ...})
└── data.py           # Your dataloader returning (train, val, test)
```

### 2. Register it

In `models/__init__.py`, add one line:
```python
_MODEL_REGISTRY = {
    "glaucoma": "models.glaucoma",
    "nlp": "models.nlp",
    "resnet": "models.resnet",        # ← new model
}
```

### 3. Create a config

```yaml
# configs/resnet.yaml
model:
  type: "resnet"
  # ... model-specific keys

train:
  epochs: 50
  batch_size: 16
  # ...
```

Then train: `python -m pipeline.train --cfg configs/resnet.yaml`

**Contract:** Your `model.forward()` must return `{"logits": tensor_of_logits}` (or an HF-style object with `.loss`). Your dataloader must return `(x_tensor, y_tensor)` tuples or dicts with `"labels"`.

## Repo Layout

```
glaucoma-thesis/
├── pipeline/                        # Model-agnostic training framework
│   ├── train.py                     # DDP loop, preemption, test eval
│   ├── config.py                    # Schema validation (extensible per model)
│   ├── distributed.py               # NCCL init, rank helpers, barriers
│   ├── metrics.py                   # TensorBoard + JSONL + wandb
│   ├── utils.py                     # Seeding, atomic ckpts, core autotune
│   └── download.py                  # HF dataset prefetch
│
├── models/                          # Model implementations
│   ├── __init__.py                  # Registry: map type string → module
│   ├── glaucoma/                    # Thesis: 3D CNN for OCT glaucoma
│   │   ├── __init__.py
│   │   ├── model.py                 # Simple3DCNN
│   │   ├── data.py                  # .npy DataLoader (HF-ready)
│   │   ├── notebooks/               # Archived thesis code
│   │   └── results/                 # Training plots
│   └── nlp/                         # Example: HF text classification
│       ├── __init__.py
│       ├── model.py
│       └── data.py
│
├── configs/
│   ├── base.yaml                    # NLP example config
│   ├── glaucoma_96.yaml             # Thesis: 96³
│   └── glaucoma_128.yaml            # Thesis: 128³
│
├── scripts/
│   ├── colab_setup.sh               # One-shot Colab launcher
│   ├── vast_setup.sh                # vast.ai: download|smoke|train|resume
│   ├── launch_ddp.sh                # Single-node multi-GPU
│   ├── slurm_train.sbatch           # Multi-node SLURM
│   └── tunnel_dashboard.sh          # Cloudflare tunnel for TensorBoard
│
├── tests/
│   ├── test_config.py               # Config schema validation
│   └── test_utils.py                # Atomic save, checkpoint prune, overrides
│
├── .github/workflows/ci.yml         # ruff + pytest + CPU smoke on every PR
├── pyproject.toml                   # Package config, ruff, pytest
├── Makefile                         # setup · lint · test · smoke · train · ddp · tb
├── Dockerfile                       # Containerized (PyTorch 2.4 + CUDA 12.1)
└── .env.example                     # HF_TOKEN, WANDB_API_KEY
```

## Enterprise Guarantees

1. **Elastic checkpoints** — saved from the *unwrapped* model. A checkpoint written on 16 GPUs resumes on 1 GPU and vice versa.

2. **Preemption safety** — SIGTERM/SIGINT triggers checkpoint-then-clean-exit. SLURM walltime, vast.ai spot reclaim, Colab disconnects — `--resume` picks up exactly where it left off.

3. **Rank discipline** — rank 0 exclusively logs, saves, and evaluates. No 8× duplicate downloads (`main_process_first`), no corrupted concurrent writes.

4. **Config as contract** — `pipeline/config.py` validates presence, types (blocks YAML `true`-as-int footgun), and value ranges before anything loads. Extensible per model type.

5. **Reproducibility manifest** — every run writes `run_manifest.json`: git SHA, CUDA version, GPU models, world size. Auditable months later.

6. **CI gate** — GitHub Actions runs ruff + pytest + CPU smoke train on every PR. Pipeline must train end-to-end to pass.

7. **Three dashboards** — TensorBoard (live local), JSONL (grep/pandas), wandb (cloud, optional). All from rank 0 only.

8. **Fail-fast everywhere** — config validated in <1s, dataset validated before GPU, NaN guard during training, atomic checkpoints (never corrupted).

## Remote Dashboard

- **TensorBoard (local/vast.ai)**: `make tb` or `tensorboard --logdir outputs --port 6006`. On vast.ai, tunnel via SSH: `ssh -L 6006:localhost:6006 ...`
- **TensorBoard (Colab)**: `%load_ext tensorboard` + `%tensorboard --logdir outputs` — inline charts
- **TensorBoard (public, no SSH)**: `bash scripts/tunnel_dashboard.sh` — free Cloudflare tunnel URL
- **wandb (cloud, any device)**: `wandb login`, then `--set logging.wandb=true`

## Config Reference

| Key | Type | Description |
|---|---|---|
| `run_name` | str | Unique run identifier |
| `seed` | int | Random seed (+ rank offset for DDP) |
| `model.type` | str | Model registry key (`"glaucoma"`, `"nlp"`) |
| `model.architecture` | str | Model variant name |
| `model.input_channels` | int | Input channel count (1 for grayscale OCT) |
| `model.num_classes` | int | Output classes (2 for binary) |
| `train.epochs` | int | Training epochs |
| `train.batch_size` | int | Per-GPU batch size |
| `train.grad_accum_steps` | int | Gradient accumulation steps |
| `train.lr` | float | Learning rate |
| `train.amp` | bool | Automatic mixed precision (bf16 on Ampere+, fp16 on T4) |
| `train.compile` | bool | `torch.compile` (~10-30% faster on A100/H100) |
| `train.early_stop_patience` | int | Eval intervals without improvement before stopping |
| `logging.tensorboard` | bool | TensorBoard scalar logging |
| `logging.jsonl` | bool | JSONL metrics file |
| `logging.wandb` | bool | wandb cloud dashboard |
| `data.cache_in_ram` | bool | Load .npy fully into RAM (96³ fits; 128³ = ~4GB) |
| `data.num_workers` | auto/int | DataLoader workers (auto = min(cores, 8)) |

All values overridable from CLI: `--set train.lr=0.0005 train.epochs=50`

## Scaling Beyond DDP

DDP replicates the full model per GPU — fine up to ~1-3B params with AMP. Beyond that, swap `DDP(model)` for FSDP (`torch.distributed.fsdp`) or drive with `accelerate`/DeepSpeed. Config, data, checkpointing, logging, and preemption all stay unchanged.

## License

MIT
