# Full CUDA Denoising Benchmark

`scripts/run_denoise_full_cuda.py` is an unattended, resumable launcher, not a
training script. It processes all 3,300 uint8 200-cubed volumes: Training 2,100
(1,017/1,083 classes 0/1), Validation 300 (124/176), Test 900 (411/489).

## Launch

Use the environment already provisioned for `scripts/denoise_cuda_suite.py`:
NumPy, CuPy with working CUDA, SciPy, scikit-image 0.26.0, pandas, matplotlib,
requests, huggingface_hub, psutil, and wandb. BM3D additionally needs the pinned
VapourSynth R79 API 4 and BM3DCUDA DLL specified by that suite. Pass `--dll PATH`
or set `BM3D_DLL`; no CPU fallback or dependency auto-installation is performed.

```powershell
python -m pytest tests/test_denoise_full_cuda.py -q
python scripts/run_denoise_full_cuda.py --execute
```

The orchestrating main agent starts the long-lived subprocess after tests. The
implementation/test task itself does not launch the full GPU job.

```powershell
python scripts/run_denoise_full_cuda.py --pilot-only --methods gaussian
python scripts/run_denoise_full_cuda.py --execute --methods bilateral
python scripts/run_denoise_full_cuda.py --execute --allow-slow
```

`--pilot-only` validates and scores one Training volume per selected method plus
the original baseline; it still authenticates, verifies/downloads the pinned raw
cache, creates/checks the private destination, initializes online W&B, and
uploads small provenance/pilot reports and labels. It does not produce/upload
full volume shards. It is **not** a download-free smoke test.

Set `HF_TOKEN` and `WANDB_API_KEY` in the gitignored root `.env` or environment.
`load_env_file()` runs before W&B initialization. Missing HF credentials abort
before creating directories. `whoami()` checks identity; repository creation
and actual uploads enforce write authorization. An existing public destination
is refused, never silently made private. W&B must initialize online before GPU
work. Tokens and signed URLs are not printed in launcher errors.

## Scope And Gates

- Default sequential order: original baseline, bilateral reuse, BM3D, Gaussian,
  median, TV. All seven classical methods are recorded; wavelet and NLM remain
  explicitly pending independent CUDA parity validation.
- Bilateral **reuses** `data/bilateral_200` arrays. Every split SHA256, labels,
  counts and pinned provenance must pass before use. The existing manifest is
  retained in output provenance. This is not newly performed CUDA filtering.
- New backends run the suite's synthetic and OCT parity/execution validation,
  then 12 full Training volumes with all metrics. In addition to the suite's
  cropped OCT check, the launcher checks eight complete 200-by-200 Training
  planes against the CPU reference (execution-only for BM3D).
  BM3D validation is execution-only for a new fixed-sigma GPU variant, not
  equivalence to the historical CPU BM3D profile.
- Failed backends are recorded and skipped, not silently declared complete.
  DnCNN/SwinIR have pending README entries only; no inference or GPU training.
- `--max-compute-hours 4` is an explicit conservative per-method pilot estimate
  gate, not a user budget or a hard runtime deadline. Override with `--allow-slow`
  or change the bound. Estimate includes filtering and full metrics, excludes
  uploads, remote full-stream verification, startup and figure rendering.
- GPU availability requires at least 1.2 GiB free, polled every 15 seconds for up
  to `--wait-gpu-seconds 1200`. A memory gate is not a promise against later OOM;
  allocation failures pause the method. No competing GPU processes are killed.
- Disk checks retain 10 GiB free before each raw download and generated shard.
  Raw data is about 24.6 GiB and a 128-volume shard about 0.95 GiB, plus reports.

## Data And Verification

Raw consolidated files are downloaded selectively and authenticated from
`tqhuyen/harvard-oct-glaucoma-200`, immutable revision
`939a38876b7b9313162842ef2d44b7edc2b57020`, into the new directory
`data/denoise_benchmark_raw`. Complete caches are reused only after hashing
against pinned LFS SHA256 (or native Git blob hash for small non-LFS files).
Interrupted downloads resume contiguous `.partial` files with strict HTTP
Content-Range checks, 60-second read timeout and bounded exponential retries.
Corrupt existing cache files are preserved and refused, not overwritten.

Destination: **private** `tqhuyen/harvard-gf-denoise-benchmark-v2`.
Upstream Harvard-GF attribution and NC-ND restrictions are retained in its
README. Private backup does not imply permission to publicly redistribute
derived data or change upstream licensing.

```text
source.json
labels/<split>_labels.npy
baseline/original/<configid>/metrics/<split>/shard-00000.jsonl
classical/<method>/<configid>/volumes/<split>/shard-00000.npy
classical/<method>/<configid>/metrics/<split>/shard-00000.jsonl
classical/<method>/<configid>/metrics/<split>/shard-00000.slices.jsonl.gz
classical/<method>/<configid>/figures/Training-0-slice-80.png
classical/<method>/<configid>/aggregate.csv
classical/<method>/<configid>/manifest.json
classical/<method>/<configid>/_COMPLETE.json
deep/DnCNN/README.md
deep/SwinIR/README.md
manifest.json
_COMPLETE_SELECTED.json
```

Each artifact upload is followed by authenticated **full streaming SHA256 and
size verification at the returned immutable commit**. This entails unavoidable
network egress roughly equal to uploaded data, but no second permanent disk
copy. A durable local shard receipt is written only after all its metrics,
figures and generated volume have verified. Only then may the exact generated
`shards/<split>/shard-NNNNN.npy` in that receipt be deleted. No recursive delete,
raw cache cleanup, existing bilateral cleanup, checkpoint, report or log
deletion is performed. Failed uploads/verification leave generated volumes.

## Metrics And Figures

The suite's versioned raw-fixed protocol and invalid-reason dictionaries are
saved in manifests. All volume-global and all 200 per-slice scores run on GPU.
Undefined mathematical results remain null, never replaced with fabricated
zeros. The original is a shared baseline, computed independently of each
denoiser, rather than copied into each method's archive. Pilot evaluations are
bounded extra work. Slice scores stream to compressed JSONL shard by shard;
the launcher never accumulates all 660,000 slice records in RAM.

Per-volume JSONL supports small-dataframe aggregate CSVs grouped by split and
class: valid count, invalid count, volume count, mean, sample standard deviation
(`ddof=1`), median, p25 and p75. Empty CSV fields denote undefined statistics.
The metric protocol's within-image standard deviations remain population
statistics, distinct from this between-volume sample standard deviation.

Figures are preselected Training indices 0, 1, 25 at slices 80 and 120. Each
compares raw/denoised full field and the identical fixed raw-coordinate
`[60:140,60:140]` ROI. Both columns use raw p1/p99 display limits. No Test
beauty-selection, diagnosis, RNFL localization or cross-method figure requiring
all deleted volumes in RAM is performed. A later all-method plate can download
the archived shards selectively.

W&B project `glaucoma-thesis` receives live per-volume scores/timing/progress
with a bounded key namespace and automatic system monitoring. Figures and
reports are saved locally and to HF. For a compact Drive mirror, explicitly set
`DENOISE_DRIVE_DIR` or `--drive-dir` to an **existing, verified local Drive
mount** containing an operator-provisioned `.denoise-drive-approved` sentinel.
Use a subfolder under the usual `MyDrive/MasterBKDN/Thesis` root. The launcher
does not fabricate `/content/drive` on Windows. Without an approved destination,
status explicitly says Drive pending and uses the user-approved HF primary.
The mirror omits volumes, partial files and W&B internals.

## Resume And Status

Local default state root: `outputs/denoise_full_cuda`.

- `status.json`: PID, phase, method/split/index, W&B URL, pending/failed states.
- `<method>/<configid>/metrics/<split>/*.receipt.json`: durable verified shard
  boundaries and immutable remote commit/hash receipts.
- `<method>/<configid>/manifest.json`: full provenance and committed shard list.
- `<method>/<configid>/complete.receipt.json`: verified final report receipts.
- `job.lock`: PID plus creation time prevents duplicate launchers on the same
  work directory; stale PID locks are recovered.

Rerun the same command/work directory to resume. Verified shards are checked
remotely before skipping; an interrupted partial shard can recompute at most
128 volumes. Stable config IDs include source, metric protocol, implementation
and launcher hashes, excluding device identity and validation timing. Keep the
local reports/receipts for partial-method resume. A fresh work directory can
recover a remotely completed method through its pinned completion manifest
after verifying every artifact; partial remote-only recovery is not supported.
Changes to implementation generate a new ID.

SIGINT/SIGTERM or a `STOP` file in the work directory request a soft stop: the
active shard finishes, uploads and verifies before exiting. Remove the STOP
file before a deliberate resume. Network failures can prevent completion of
that boundary; in that case unverified files are retained. Force-killing a
process loses only the uncommitted shard's work, not verified receipts.

A selected-method completion marker does not claim all seven methods ran.
Pending, failed, budget-paused and stopped methods remain explicit. Exit codes:
0 selected methods complete (or pilot-only), 2 incomplete selected scope,
1 fatal launcher/transport error. No paid compute or training is launched.
