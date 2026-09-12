# Controls-96 on Kaggle

## Scope

This is a platform-only exception to the canonical controls notebook, not a new
study. The original notebook and frozen model, trainer and data helpers remain
unchanged. Entry point: `notebooks/3d_glaucoma_controls_96_kaggle.ipynb`.
Implementation: `scripts/controls_kaggle_setup.py` (stdlib bootstrap),
`scripts/controls_kaggle.py`, and `scripts/controls_kaggle_data.py`.

| Setting | Frozen protocol |
| --- | --- |
| Specifications | P, B3 (seed 44 only), C1, C2, B4, BM; B1/B2 disabled |
| Seeds | 42, 43, 44; B3 runs seed 44 only: 16 independent jobs |
| Training | 20 epochs each; patience 21 disables early stopping; no extension |
| AdamW | LR 1e-4, weight decay 1e-4; frozen warmup/cosine schedule |
| Batch | Effective 16; microbatch selected from 16, 8, 4, 2, 1 |
| Resolution | Raw/BM3D/bilateral storage 200 cubed; model 96 cubed; views 224 square from 200 |
| Initialization | Scratch study training with pretrained ImageNet 2D encoders, not warm-start |
| Precision | FP16 + GradScaler; BF16 and TF32 disabled |
| Selection | Validation AUC; per-epoch test monitoring never used for selection |
| Optional stages | X-AI and information theory disabled; no metric PNGs |

P uses all three branches with CrossGate. B3 uses two 2D branches with concatenation
on seed 44 only. C1 concatenates all branches, C2 fixes the attention gate to 1,
B4 uses the full model on bilateral-200, and BM uses the full model on BM3D-200
resampled to the model 96. `cm.ControlsModel` and `ct.Trainer` are reused, not
copied or modified.

## Setup

1. Enable Kaggle Internet and select one P100 or two T4 GPUs. Two T4s are **two
   independent 16 GB-class devices**, never a pooled 32 GB GPU and never DDP.
2. Attach Kaggle Secrets `HF_TOKEN` and `WANDB_API_KEY`. The helper calls
   `load_env_file()` first, then environment values, then `UserSecretsClient`.
   Credentials are inherited through the child environment, never job JSON/config.
3. Use a checkout containing the new files. The setup cell can clone the public
   repository, but uncommitted local additions are not magically in that clone.
   Attach/publish the intended checkout yourself before launching Kaggle. An
   existing checkout is not pulled or overwritten. `CTRL_KAGGLE_REPO_ROOT` can
   select it; an explicit missing root is an error, not an implicit clone elsewhere.
   Setup checks the helper bundle/API, prints checked git HEAD and source paths/hashes,
   and rejects already-imported modules from another checkout or changed source.
   A checked HEAD is provenance, not a promise that an unpulled clone is latest.
4. Optionally attach raw, bilateral and BM3D dataset files read-only. Set
   `CTRL_KAGGLE_RAW_ROOT`, `CTRL_KAGGLE_BILATERAL_ROOT` and `CTRL_KAGGLE_BM3D_ROOT`
   to directories containing the split files. BM3D attached roots must already
   contain pre-consolidated `{split}_volumes.npy` files; they are never written.
   Empty roots select authenticated, selective HF downloads. Only the sources
   required by the session's selected jobs (`raw`, `bilateral`, `bm3d`) are prepared.
5. Run the notebook sections in order. Only T3 dispatches training. D1-D3 prepare
   all required sources and caches in the parent before any worker is launched.

Internet is required even with attached data, for pinned HF metadata verification,
ImageNet initialization when needed, W&B metrics and checkpoint acknowledgment.

### Early GPU Bootstrap

Setup runs before importing the training/data helpers or Torch/HF/W&B consumers.
Cache environment variables are configured first with stdlib, respecting existing
environment overrides and `CTRL_KAGGLE_TEMP_ROOT` / `CTRL_KAGGLE_OUTPUT_ROOT`.
`HF_HOME`, `HF_HUB_CACHE`, `HF_XET_CACHE`, `TORCH_HOME`, `WANDB_CACHE_DIR`,
`WANDB_DATA_DIR` and `WANDB_DIR` therefore take effect at import time.

The bootstrap queries `nvidia-smi` for physical GPU and driver, ignoring any
`CUDA_VISIBLE_DEVICES` value (including an empty one left by a prior CPU smoke run)
so physical devices are always enumerated. `begin_setup()` also removes an empty
`CUDA_VISIBLE_DEVICES` before a real run. If `nvidia-smi` is missing, exits non-zero,
or lists no GPU, setup stops with the accelerators to enable (GPU P100 or GPU T4 x2)
and the instruction to restart the session; it never falls back to CPU training.
Each GPU gets an isolated Python subprocess that imports Torch/torchvision, reports versions,
CUDA runtime, architecture list and capability, executes a tiny CUDA addition,
and verifies native torchvision NMS. The notebook parent never initializes CUDA.
These probes run only when the user actually executes real setup on Kaggle.
Smoke returns before subprocesses, GPU inspection, pip, or downloads, even on a
machine with a `/kaggle` directory.

The P100 wheel profile references
`3d_glaucoma_3branch_bm3d_200_kaggle.ipynb`; that reference is not modified.
Unlike its blanket `--no-deps` support installs, this adapter resolves dependencies
under a temporary constraints file pinning every installed `torch`, `torchvision`,
`torchaudio`, `triton`, and `nvidia-*` distribution to its current version.
Missing/import-incompatible support packages are installed with dependencies:
timm **1.0.29**, wandb, huggingface_hub, python-dotenv, scikit-learn, scikit-image,
scipy, matplotlib, psutil and filelock. Afterwards isolated imports, protected
versions, and native GPU operations must still pass. Constraint conflicts stop
setup with a manual-profile error; support setup never blindly reinstalls Torch.
A restart is requested only when support installation actually changed protected
installed versions or a newly installed support module had already been imported
and could be stale. A resident unrelated module such as numpy does **not** trigger
a restart. Fresh Save-Version kernels continue the same session after support
installation; rerun Setup only if a restart was explicitly requested.

| Situation | Action |
| --- | --- |
| Functional P100 with sm_60 or T4 with sm_75 | Keep installed Torch stack; no downgrade |
| P100 missing sm_60 / no kernel image, otherwise classified CUDA architecture failure and pip installable | One-time coherent cu121 triple repair; continue when the kernel is Torch-free after a passing re-probe |
| Kernel already imported torch/torchvision/torchaudio before repair | Repair installs, then restart is required; loaded binaries are never reloaded |
| Unsupported T4, mismatched torchvision, driver or library import failure | Stop; choose a coherent manual image/profile, not P100 auto-repair |
| Python 3.13+ or older than 3.9 on a P100 requiring repair | Stop before pip; choose a Python 3.9-3.12 Kaggle image |
| Repair already attempted but post-restart validation fails | Stop; no repeated install |

`AUTO_REPAIR_P100=True` defaults on only in real Kaggle execution. Set
`CTRL_KAGGLE_AUTO_REPAIR_P100=0` before Setup to disable it. The repair always
installs the coherent triple `torch==2.5.1`, `torchvision==0.20.1` and
`torchaudio==2.5.1` in one command from
`https://download.pytorch.org/whl/cu121`, even when torchaudio is currently absent,
so a `torchaudio` 2.10 / `torch==2.10` version conflict cannot appear. After pip,
setup re-probes in a fresh subprocess (Torch import, versions, architecture list,
capability, tiny CUDA op and native torchvision op). Pip cache and temporary wheel
work use the configured temp root (default `/kaggle/temp/controls96`), not the output
partition. At least **8 GiB free temp disk** is required; pip can need more. Low disk
means start a fresh session, not automatic deletions. No blanket NVIDIA uninstall,
loader-path modification, or loaded-Torch reload is performed.

**When a repair runs from a Torch-free kernel**, it does not require a restart.
Before pip starts, setup persists `/kaggle/working/controls-restart-required.json`
and an environment guard. The marker and guard are cleared only after the full
verification chain passes: a fresh subprocess re-probe showing sm_60, support
installation, protected-version checks and the final native-operation probe. The
later `import torch` in the notebook then loads the newly installed 2.5.1+cu121
build. If the kernel had already imported any of `torch`, `torchvision` or
`torchaudio` before the repair, setup keeps the marker and guard and requests a
restart; loaded binaries are never reloaded. A pip failure, a still-missing sm_60
after a successful pip, an incompatible torch/torchaudio pair, or a failed
verification stops setup with an error, leaves the marker in place and never repeats
the install; it does not claim success. Data/model/training entry points and
notebook stages reject old globals while this marker exists. A failed setup also
blocks dependent stages until Setup succeeds. Do not manually delete the restart
marker to bypass verification.

Every real Setup attempt invalidates readiness immediately, before checkout
discovery, helper loading, API checks, or source validation. This also applies
after a previously successful Setup: a changed source file or missing checkout
cannot leave old `kg`/`kd` globals usable for D1 or dispatch. The notebook calls
`bootstrap(defer_ready=True)`, so successful hardware/dependency verification
permits helper imports only, not data preparation or training. Readiness is cleared
only by the final `complete_setup()` after helper imports, import-hash recording,
repeat repository validation, and credentials all succeed. Any failure leaves the
pending guard set; correct the issue (restart for stale imports) and rerun the
whole Setup cell. Smoke lifecycle calls neither set nor clear readiness guards.
Direct `bootstrap()` callers retain the default non-deferred completion behavior.

Kaggle Secrets fall back independently per missing key. An existing environment
value is retained; failure to retrieve one secret does not prevent attempting the
other. Errors name missing keys, never their values.

## Sources and Caches

Raw: `tqhuyen/harvard-oct-glaucoma-200` at
`939a38876b7b9313162842ef2d44b7edc2b57020`.

Bilateral: `tqhuyen/harvard-oct-glaucoma-200-bilateral` at
`47632c96b206707fd6423ee5b4da159069f63eaf`.

BM3D: `tqhuyen/harvard-gf-denoise-benchmark-v2`, method prefix
`classical/bm3d/3375a321513938835d2c`, default branch (no pinned full-repo
revision). Per split, shard files
`{prefix}/volumes/{split}/shard-*.npy` are listed through
`HfApi.dataset_info(files_metadata=True)`; the per-method `_COMPLETE.json` must
exist and the concatenated shard count must equal that split's label count. Shards
are downloaded authenticated, verified against their HF LFS SHA256, merged in
ascending `shard-00000...` order into one `{split}_volumes.npy` under temp, and
each shard is deleted after it is merged to bound disk. Labels come from the raw
repo revision above. The BM3D run identity is the ordered shard LFS SHA256 list
plus the labels SHA256, shape and repo/prefix, so 26 GB of volumes is never
re-hashed. If `CTRL_KAGGLE_BM3D_ROOT` is attached, a pre-consolidated
`{split}_volumes.npy` must already exist there; it is verified and never written.

B4 requires the original **bilateral-200** export. The stored bilateral-96 export
is not a substitute: paired augmentation occurs before float resize, and stored
uint8 resize changes quantization. The BM3D source is 200-cubed and is resized to
the 96 model input on the fly, exactly like raw and bilateral. No denoiser is run
and no existing background denoise job is changed or stopped.

Remote commit identity, file size, LFS SHA256/Git blob digest, source shapes,
binary labels, bilateral manifest and completion markers are checked. Raw and
bilateral label/source provenance must agree, and BM3D labels are checked against
the raw revision. Corrupt attached files are rejected, never replaced. Attach the
original names, including bilateral manifest and split completion JSONs. Complete
split files, not partial exports, are required.

By default downloads, source hash sidecars and view caches live under
`/kaggle/temp/controls96`; outputs live under
`/kaggle/working/controls96/<group>`. Expect roughly 25 GB raw, 25 GB bilateral
and 26 GB BM3D on disk. A single session should need at most two of these sources,
because no session mixes all three datasets; raw + bilateral + BM3D cannot coexist
on a typical Kaggle temp disk. None must land in the 20 GB output area. Actual
filesystem free space is checked before each download and cache allocation. These
checks do not assume Kaggle guarantees any particular temp-disk capacity.
Insufficient space fails visibly. No source files are copied beside the view cache.
No data is uploaded to HF.
HF/Torch model caches and W&B artifact staging/cache also use temp, not working.
The redundant Xet chunk cache is disabled; verified downloaded files remain reusable.
Each source checksum prints start/done with the source filename and hash. New
view caches report progress every 200 volumes or approximately 10 seconds, plus
completion (the time bound is checked between volumes, not an interrupt timer).

The new dataset adapter bypasses the inherited constructor that writes adjacent
to source volumes. It accepts explicit source, labels, views and depth-axis paths;
all are opened read-only. B3 never opens the 3D volume mmap. It reuses the frozen
item implementation: paired flips, rotation and intensity, seeded by
`(seed, epoch, index)`, followed by 3D float resize.

View identity includes source and label SHA256, shape, `res2d`, view order,
`half=16`, bilinear resize/uint8 semantics, code and package versions; actual view
and depth-axis hashes are checked before reuse. Projections are `slab_mip` and
`aip_full`, **not anatomical RNFL segmentation**. Run data identity contains
content hashes, not absolute runtime paths, so the source/cache location can change
between sessions. Worker job JSON contains runtime paths separately.

## Runtime and Probe

`GPU_MODE='auto'` selects one process for a single P100/T4, two processes for dual
T4. `GPU_MODE='single'` limits execution to the first listed device. Other layouts
fail instead of guessing. The parent queries `nvidia-smi`, never initializes CUDA.
`subprocess.Popen` starts each interpreter with its physical
`CUDA_VISIBLE_DEVICES` already set. Each worker validates that it sees exactly one
device and uses logical `cuda:0`, with a distinct tag, directory and W&B run ID.
Controller source hashes are included in job payloads and checked in the worker
before data/model construction; a different child checkout cannot quietly train.
Workers write unbuffered stdout/stderr to unique `<group>/<tag>.GPU<id>.*.log`
files under working. The controller relays at most 64 KiB per worker per poll,
prefixing lines with job tag and physical GPU. No pipes can fill and deadlock an
unread child. Logs remain available after exit; stop/grace handling continues
relaying while waiting for owned children only; nonzero exits remain errors.
Known HF/W&B secret values are redacted from Python stdout/stderr before they
reach the worker file, including values split across writes; the relay also
redacts them. No credentials are placed in job payloads. Probe candidate/scale skips,
epoch-numbered validation/test metrics, and full-state upload start plus remote
acknowledgment/local paths are visible in the notebook.

The probe targets at most 80% of actual device memory (approximately 12.8 GiB if
the device really has 16 GiB). It uses actual data/model and weighted accumulation
windows of 16 samples, FP16, AdamW and clipping. It allows at most eight scaler
overflow skips and requires two successful updates. Nonfinite loss is still fatal.
It restores weights, buffers, modes and RNG and discards the probe optimizer.
Fresh training uses the selected settled scaler value (initial default 1024),
not the frozen trainer's default 65536. Exact saved scaler/batch/accum are restored
without probing on resume. Probe cache identity includes spec/config, device,
physical slot, backend version and headroom. Tiny smoke models only check platform
contracts; they are not evidence that MaxViT fits a real 16 GB GPU.

Before model construction (including pretrained downloads) or any expensive probe,
the child opens a separate W&B run with scope `runtime-probe`. Failures finish this
run with a nonzero exit code. Batch results are logged there; only afterwards is
the scientific training identity created with the resolved batch. Runtime runs
are not scientific completions and never contribute seeds to the study summary.
Probe cache reuse checks the complete backend/device/config key and still requires
two successful FP16 updates and a peak within the selected physical device budget.

The frozen trainer computes summed class-weighted loss and divides accumulated
gradients by the float sum of class weights across the entire window. It does not
average per-microbatch means. The Kaggle subclass changes scaler initialization
and checkpoint storage only, not loss, clipping, selection or epoch budget.

## Session Budget

Default `MAX_RUNS_PER_SESSION=2`. A P100 executes those two jobs sequentially;
dual T4 executes them concurrently. This is an explicit session subset of the
16-job study, not reduced training epochs. `RUN_TARGET='B4_s42'` optionally limits
the queue to one pair. `MAX_RUNS_PER_SESSION=None` removes only the count cap.
The assignment and deadline are stored under temp before source preparation.
Rerunning T3 resumes/skips that same assignment rather than silently admitting
another two jobs. A fresh Kaggle session/temp directory selects the next subset.
Only the dataset sources used by the selected jobs are prepared (raw, bilateral
and/or BM3D); a session should stage at most two of them.

Default `SESSION_HOURS=10`, with 600 seconds reserved for checkpoint grace. Do not
assume Kaggle always provides 12 hours. No new jobs are dispatched after the soft
deadline; workers request stopping at accumulation/epoch boundaries. A single
epoch/probe/upload may still exceed the reserve, so stopping before platform hard
termination cannot be guaranteed. Check measured epoch times on the actual GPU.

Output admission checks measured local use plus a conservative 6 GiB reservation
per new/outstanding worker against 20 GiB and actual free disk. Local versions are
flat `last.pt`, `best_weights.pt`, identity, logs, logits and reports, not every
epoch checkpoint. Atomic replacement requires transient disk headroom. W&B staging
also consumes disk: inspect usage, and start a fresh session rather than silently
pruning checkpoints. The admission check is not a hard filesystem quota or a proof
that arbitrary larger checkpoints will fit. With the default two runs, checkpoints
do not accumulate for all 16 jobs in a single session.

Controller and job locks record PID + process creation time. Live duplicates are
rejected; stale locks can be reclaimed. Interrupting the controller writes a stop
marker, stops new dispatch and waits for **owned children only**. Workers commit
at safe boundaries before exit. The adapter never uses global process termination
or signals unrelated jobs. If 600 seconds expires, it raises with still-live PIDs
and retains handles in `kg.OWNED_CHILDREN`. Keep the kernel alive and call
`kg.wait_owned(STOP_PATH, 600)` again. Do not mistake the error for cancellation or
close the kernel and forget the workers. A process outside the fit loop may not
observe the stop marker until model loading/probing finishes.

JSON writes use unique UUID temporary files and atomic replacement. Concurrent
STOP requests publish one complete marker using exclusive hard-link creation;
later requests are idempotent. Windows sharing violations during JSON replacement
receive a bounded retry; permanent permission failures are surfaced, not ignored.

## Persistence and Recovery

**Explicit policy exception: W&B + Kaggle outputs, no Drive API/mount/credentials.**

Local checkpoints are atomic every 10 optimizer steps (and the frozen wall-clock
fallback). Full-state W&B artifacts upload once per completed epoch and on stop,
not every 10 steps. Run identity is uploaded before training. Upload receipt
`.wait()` must succeed before an acknowledgment marker is written. An upload/quota
failure stops the session, retaining local state. Abrupt VM loss can lose up to the
current epoch since the last successfully acknowledged checkpoint; first-epoch
loss requires restarting that run from its initial identity.

Best weights are included alongside latest resumable state and reports in
`controls-<WANDB_RUN_ID>:latest`. Remote versions are not free: full AdamW state
changes each epoch, and content deduplication does not make those changes disappear.
Check your W&B storage/network quotas before the study. Older W&B artifact versions
are **never automatically deleted**. Manage retention yourself only after checking
recovery requirements. Local files and the latest acknowledged artifact are separate
recovery sources; a failed remote upload is never called a successful archive.

**Interactive `/kaggle/working` files are not automatically permanent.** Save Version
with outputs or download the best weights/checkpoints before ending the session.
W&B acknowledgment provides remote checkpoint persistence, not Kaggle Save Version.

For a new session:

1. Attach prior saved outputs read-only and set `CTRL_KAGGLE_RESUME_ROOT` to the
   prior group directory (the directory containing `P_s42`, etc.), keeping the same
   group. Only the selected/admitted job's identity/state/reports are hydrated into
   working storage. Raw data and all other run checkpoints are not duplicated.
2. Alternatively set `ARTIFACT_REFS={'P_s42': 'entity/glaucoma-thesis/controls-ID:latest'}`.
   Add `_summary: 'entity/glaucoma-thesis/controls-summary-GROUP:latest'` to restore
   group results/queue progress without downloading completed-run checkpoints.
3. Rebuild/reuse verified temp view caches in the parent. Resume validates the
   frozen study/data/code identity, reuses saved batch and scaler, and continues to
   the original 20-epoch target. A different dataset/implementation is rejected.
4. Persist group summary each session. Completed jobs in the supplied summary or
   saved outputs are skipped. Without prior outputs/summary/ref information, an
   ephemeral session cannot know which prior runs completed; it must not guess.

Only restore trusted checkpoint artifacts: the frozen trainer uses Python/Torch
checkpoint deserialization. W&B artifact refs must belong to the intended study.
No new seed, epoch extension or arbitrary warm-start is supported here.

### Strict Recovery Format

Recovery and completion metadata now use schema 2; the earlier unshipped
metrics-only summaries/identity-less fixtures are not treated as valid recovery.
No compatibility guesswork or silent fresh start is performed.

**This setup/code update changes recovery identity.** The code hashes remain strict,
and backend identity now also records protected GPU and support-package versions. A prior
checkpoint is not automatically migrated, deleted, or treated as a fresh run.
To continue an incompatible old run, recover the exact source revision and package
environment recorded with its `run_identity.json` and originating run, including
the recorded Torch/CUDA/timm versions and source hashes. If those cannot be
reconstructed, request a separate migration review identifying the actual old
checkpoint/artifact and its metadata. No specific old checkpoint was supplied for
this change, so no migration is authorized or implemented. Do not change the group
merely to evade an incompatible-resume error.

- Required normal recovery: matching `run_identity.json`, `run_identity.pt` and
  complete `last.pt`. CPU validation checks tag/spec/seed/group, the frozen protocol,
  implementation/backend hashes, source/view/label identities, batch/scaler settings,
  checkpoint run ID, config and progress. Prepared data is compared again before
  dispatch/worker execution. Checkpoints are read with CPU mmap, not on a GPU.
- An explicit artifact ref, missing target directory, empty attachment or incomplete
  checkpoint fails clearly. Hydration stages all required files into an owned
  unique directory, verifies identity/checksums, writes an atomic hydration receipt,
  then promotes the directory. An exception during promotion rolls back the prior
  directory. Partial files are never installed one-by-one into a usable run.
- A partial local identity does not suppress a requested cloud restore. Retry
  repairs matching incomplete state; an already validated local checkpoint is kept
  only when it is at least as advanced as the restored state. Conflicting identities
  are rejected. A completed hydration receipt permits the worker to reuse the
  parent's verified local restore without downloading it again.
- Only UUID download/staging directories owned by that call are removed. Attached
  sources are never deleted, even when both `resume_root` and artifact refs are set.
  An artifact ref takes precedence for its tag; overlapping source/destination
  directory trees are rejected. If the process is forcibly killed during the
  backup/promotion interval, its hidden staging/previous directories may remain;
  retain them for recovery rather than assuming cancellation removed them.
- `ALLOW_INITIALIZED_RECOVERY=False` is the default. Setting it true explicitly
  authorizes starting a recorded run from its initial seed **only** when the
  identity-stage metadata matches and the requested artifact was published with
  `stage='identity'`. This is initialized recovery, not continuation of optimizer
  progress. A missing `last.pt` without matching stage metadata remains an error.

For a full queue, `resume_root` must be a nonempty, recognized prior group inventory:
validated run directories and/or a schema-2 group summary. Existing incomplete runs
are validated, not ignored. Jobs absent from that inventory can be explicitly
dispatched as new jobs with no per-job restore request. In contrast, setting a
single `RUN_TARGET` with `resume_root` requests that exact target and requires its
directory to exist. A typo must not result in random initialization.

Each `completed.json` carries the full small scientific identity, protocol and
record digests, checkpoint header, output hashes and test metrics. Local completion
requires matching identity files, last-state header and report/weight/logit hashes.
The same provenance records are retained in the group summary so a saved summary
can skip finished runs without copying their large checkpoints. Cloud scheduling
reads only `recovery.json`, `run_identity.json` and `completed.json` to validate
completion before applying the two-**unfinished**-job cap. Thus two finished cloud
refs cannot consume the session slots and prevent queue progress. These records
describe previously verified trusted artifacts, not a claim that this session
downloaded and rehashed all historical raw volumes or checkpoints.

## Reports and Validation

Each epoch logs full validation/test metrics and prints them; train window metrics
are logged every optimizer step. Test is monitoring only. Best-state reporting uses
`ct.calibrated_report`, fitting temperature and Youden threshold on validation,
then reporting train/val/test metrics and bootstrap CI. The known frozen AP/ECE
semantics are retained for comparability, not silently repaired. Raw logits/labels
are saved so another explicitly named analysis can recompute metrics later.

The worker captures the report's predictions once rather than repeating inference
to save logits. E3/E4 can rerender from `logits.npz` without model/GPU/training.
`logits_receipt.json` binds the NPZ checksum to the best-weight SHA256, scientific
identity, source/view/label config and per-split label digests. Replay rejects stale
weights, changed data/labels, wrong shapes or partial archives before calibration.
NPZ output uses a unique fully written/fsynced temporary file; its receipt is
committed before the final atomic NPZ rename, so an interrupted save cannot expose
a partially written archive as reusable logits. A replay opens a linked W&B
`report-replay` run, logs the split tables and waits for report-artifact upload;
there is no unlogged local-only replay path in the notebook.
Replay files live under `<group>/_reports/<tag>` and never overwrite the canonical
scientific report or completion record. Runtime W&B identities live separately
under `<group>/_runtime/<tag>`. Only validated scientific completions are aggregated.
Reports use JSON, CSV and W&B tables/scalars only, never metric PNGs. Group summaries
merge persisted prior results; per-metric mean, sample std and count are included.
Missing std is null, not zero. `n_seeds < 3` is explicitly partial.

CPU-only verification commands:

```text
python -m pytest tests/test_controls_kaggle_setup.py tests/test_controls_kaggle.py tests/test_controls_training.py tests/test_controls_data.py tests/test_controls_model.py -q
python -m ruff check scripts/controls_kaggle_setup.py scripts/controls_kaggle.py scripts/controls_kaggle_data.py tests/test_controls_kaggle_setup.py tests/test_controls_kaggle.py
python -m ruff format --check scripts/controls_kaggle_setup.py scripts/controls_kaggle.py scripts/controls_kaggle_data.py tests/test_controls_kaggle_setup.py tests/test_controls_kaggle.py
```

The smoke test executes every notebook code cell, all six active specs (B3 on seed
44), one epoch, using tiny synthetic data, `ft.SmokeModel`-derived variants, CPU and
offline W&B. The smoke harness runs outside the repo cwd with bootstrap subprocesses,
downloads and GPU initialization blocked. Bootstrap repair/pip/driver tests use
mocks; the relay test launches only two tiny CPU print processes. Tests also check
resume scheduler/LR sequence equality without changing the frozen trainer.
The original controls notebook is outside this validation scope.
This does not certify actual Kaggle
driver/timm compatibility, real backbone batch-1 behavior, 16 GB VRAM consumption,
throughput, remote upload latency/quota, or platform Save Version behavior. Those
require a separately authorized real Kaggle check; no GPU training, Kaggle launch,
HF/Drive upload, or commit/push is performed by the implementation tests.
