# Wavelet BayesShrink and Fast NLM on CUDA

## Implementations

Both backends are isolated from the completed benchmark's `denoise_cuda_suite.py`:

- `scripts/denoise_wavelet_cuda.py`: db4, symmetric extension, soft BayesShrink, automatic sigma and levels, per-B-scan processing.
- `scripts/denoise_nlm_cuda.py`: scikit-image fast NLM with `h=0.1`, patch size 5, search distance 3, `sigma=0`, reflect padding and ordered symmetric accumulation.

Production pixel filtering is CUDA-only. CPU reference filters are used only by the validation gates. PyWavelets supplies the fixed db4 coefficients, not production image transforms. Input is uint8, normalized by float32 division by 255; reconstruction is clipped and truncated to uint8 for storage and scoring.

For 200x200 B-scans, scikit-image's automatic db4 level rule selects **one decomposition level**, not four. The Wavelet port preserves MAD exclusion of zero coefficients, subband reduction order and reconstruction dtype promotion. The NLM port reproduces the reference's integral-image recurrence and approximate exponential; it is not a generic replacement with different patch weighting.

## Verification

Pinned local references: scikit-image 0.26.0, PyWavelets 1.8.0, NumPy 2.1.1. CUDA was exercised on an RTX 3050 Laptop 4 GB with CuPy 14.2.0.

Both implementations matched the saved uint8 output on one full Training volume (8,000,000 pixels), with **zero byte mismatches**. Tests also cover constants, edges, random data, boundaries and quantization. Runtime gates add Training indices 0/25/100/1000, including a full-volume comparison at index 0. Finite-sample checks do not prove bit-identical output for every possible image or software stack; the version and validation results are persisted per run.

The combined backend and uploader tests initially passed 92 tests with CUDA enabled. A subsequent real multi-shard run exposed a bookkeeping variable-shadowing bug. It was corrected and covered with two-shard failure/resume tests; the uploader test suite now passes 53 tests. The already-computed second Wavelet shard was checked against all stored artifact hashes, raw label order, shape and slice-record counts before correcting its pending identity. No numerical outputs were edited or refiltered for this correction.

The bounded full-volume timings were approximately 0.42 s for NLM and 0.57 s for Wavelet in the initial tests. These are not full-dataset runtime measurements. Wavelet was not faster than the CPU reference in that test; CUDA execution is established, but a speedup is not claimed. The first 12-volume runtime Wavelet pilot estimated about 0.47 hours for filtering plus scoring of 3,300 volumes, excluding upload and additional I/O.

## Run and archival

```powershell
python scripts/run_denoise_full_cuda.py --execute --methods all7
```

The completed Original/Bilateral/BM3D/Gaussian/Median/TV archives keep their existing configuration IDs and are verified/skipped. Wavelet and NLM run sequentially through the same fixed-mask scoring protocol. Each new backend's own source hash is part of its metadata and configuration ID.

Authenticated destination, made public at the owner's explicit request:

https://huggingface.co/datasets/tqhuyen/harvard-gf-denoise-benchmark-v2

New method paths are `classical/wavelet/<configuration-id>/` and `classical/nlm/<configuration-id>/`. A 128-volume shard is uploaded together with its metric records, verified against commit-pinned remote metadata or downloaded contents, then its generated local volume file is deleted. Pending-upload receipts allow retries without repeating completed filtering. Raw inputs, metrics, logs and receipts are retained.

Do not delete unfinished shards manually. Do not claim either method's full dataset is complete before its `_COMPLETE.json` marker and final reports are present. The root manifest reports `all_seven_complete` only after all seven method archives are verified.

Google Drive synchronization remains pending on this Windows host; HF is the approved primary archive. The CPU/GPU comparison does not establish anatomical preservation or improved glaucoma classification.
