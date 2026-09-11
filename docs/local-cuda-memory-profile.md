# Local CUDA memory measurements

## Scope

Measured on NVIDIA GeForce RTX 3050 Laptop GPU (4 GiB), using the existing
`myenv` Python environment: PyTorch 2.1.1+cu118, Python 3.8. The default Python
has CPU-only PyTorch and was used only to orchestrate subprocesses and log.

The benchmark extracts the actual `gnorm`, `ResXBlock3D`, and `Enc3DResNeXt`
definitions from `scripts/final_model.py`. It adds a 192-to-2 linear head.
**It does not execute the two MaxViTs or CrossGate.** Inputs and weights are
synthetic; no existing checkpoint, dataset or scientific result is modified.

Each fresh subprocess performs two forward/backward/AdamW steps with FP16
autocast and GradScaler. AdamW uses `foreach=False`; the real trainer uses
the optimizer default, another reason these are not full-run measurements.
Peak allocated memory includes both steps, including optimizer-state creation.
The allocator is capped at 55% of physical GPU memory to leave room for the
Windows desktop. cuDNN benchmarking is disabled.

Checkpoint mode uses non-reentrant checkpointing over stem + first residual
block + first downsampling block, then over each remaining 3D block. This is
an experimental benchmark path, not a change to the production model.

## Measured peak allocated memory (GiB)

| Input | Batch 2, no checkpoint | Batch 1, no checkpoint | Batch 1, 3D checkpoint |
|---|---:|---:|---:|
| 32 cubed | 0.3052 | 0.2226 | 0.2078 |
| 48 cubed | 0.6855 | 0.4148 | 0.3588 |
| 64 cubed | 1.3957 | 0.7685 | 0.6158 |
| 80 cubed | OOM under allocator cap | 1.3661 | 1.1271 |

The OOM is a failure under the configured approximately 2.2 GiB allocator cap,
not proof that the configuration cannot use all 4 GiB on an otherwise empty GPU.

## Extrapolation, not measurement at 200 cubed

A linear fit of peak allocated GiB against the voxel count (resolution cubed)
gives these rough **3D-only** extrapolations:

| Configuration | Extrapolated 200-cubed peak (GiB) |
|---|---:|
| Batch 2, no checkpoint | 38.1 |
| Batch 1, no checkpoint | 19.2 |
| Batch 1, tested checkpoint grouping | 15.4 |

These estimates extrapolate far beyond measured resolutions. Convolution
algorithm/workspace choices, full-model activation overlap, allocator behavior,
CUDA/PyTorch version and gradient accumulation can change the real peak.
There are no confidence intervals, repeated runs or full-model 200-cubed
measurements. In particular, **15.4 GiB is not a claim that the full model fits
a 16 GiB GPU**. The MaxViT branches, their training states and system headroom
are excluded. Simple batch reduction plus this checkpoint grouping does not
establish a safe 16 GiB configuration.

Further candidates are finer-grained checkpointing around the high-resolution
3D operators or activation offloading. Any modified training protocol and
target-hardware feasibility require separate correctness and peak-memory tests.

## Reproduction and logs

```powershell
python scripts/profile_final_3d_memory.py --cuda-python "C:\Users\huyen\.conda\envs\myenv\python.exe"
```

Machine-readable results: `outputs/memory_profile_local/measurements.json`.
W&B logs were created in explicit offline mode under that directory; they have
not been uploaded. Drive synchronization is pending because no verified local
Drive mount was available. No figures or model checkpoints were generated.

After subprocesses exited, `nvidia-smi` reported approximately 1284 MiB total
device usage and 1% utilization, close to the desktop baseline. No existing
GPU processes were terminated and no packages were installed or replaced.
