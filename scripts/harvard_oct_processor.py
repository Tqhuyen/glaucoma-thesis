#!/usr/bin/env python3
"""
Harvard OCT B-Scan Dataset Processor — Streaming (low-RAM), additive uploads.
Processes three Harvard Ophthalmology AI Lab datasets one at a time, uploading
each incrementally to HF Hub to keep peak disk usage < 60 GB.

Usage:
    export HF_TOKEN=hf_xxx
    python scripts/harvard_oct_processor.py
"""
from __future__ import annotations

import gzip
import io
import json
import os
import random
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, create_repo, hf_hub_download, upload_folder
from tqdm.auto import tqdm

# ── config ──────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set.", flush=True)
    sys.exit(1)

api = HfApi(token=HF_TOKEN)
HF_USER = api.whoami()["name"]
OUTPUT_REPO = f"{HF_USER}/harvard-oct-glaucoma-200"

WORK_DIR = Path(os.environ.get("WORK_DIR", "/content/harvard_oct_work"))
STAGING = WORK_DIR / "staging"
STAGING.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

print(f"HF: {HF_USER}  |  Output: {OUTPUT_REPO}  |  Disk free: {shutil.disk_usage(STAGING).free/1e9:.1f} GB", flush=True)

# ── state ───────────────────────────────────────────────────────────────────
manifest_path = WORK_DIR / "manifest.jsonl"
MANIFEST: list[dict] = []
UPLOAD_COUNT = 0

def flush_manifest():
    with open(manifest_path, "w") as f:
        for e in MANIFEST:
            f.write(json.dumps(e) + "\n")

# ── helpers ─────────────────────────────────────────────────────────────────

def save_scan(sample_id: str, oct_bscans: np.ndarray, glaucoma: int, split: str) -> str:
    rel = f"{split}/{sample_id}.npy.gz"
    fp = STAGING / rel
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_bytes(gzip.compress(oct_bscans.astype(np.uint8).tobytes(), compresslevel=1))
    MANIFEST.append({"file": rel, "glaucoma": glaucoma, "split": split, "shape": list(oct_bscans.shape)})
    return rel


def download_zip(repo_id: str, filepath: str) -> Path:
    name = f"{repo_id.replace('/', '_')}_{Path(filepath).name}"
    local = WORK_DIR / name
    if local.exists():
        print(f"  [cached] {name}", flush=True)
        return local
    print(f"  downloading {repo_id}/{filepath} ...", flush=True)
    t0 = time.time()
    dl = hf_hub_download(repo_id=repo_id, filename=filepath, repo_type="dataset")
    shutil.copy(dl, local)
    gb = local.stat().st_size / 1e9
    print(f"  done in {(time.time()-t0)/60:.1f}m ({gb:.1f} GB)", flush=True)
    return local


def extract_from_zip(zip_path: Path, label_map: dict, split_map: dict, desc: str = ""):
    count, errors, skipped = 0, 0, 0
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".npz"))
        for name in tqdm(names, desc=desc):
            try:
                raw = np.load(io.BytesIO(zf.read(name)), allow_pickle=True)
                scan = np.asarray(raw["oct_bscans"], dtype=np.uint8)
                label = label_map.get(name, -1)
                split = split_map.get(name, "")
                if split not in ("train", "val", "test"):
                    skipped += 1
                    continue
                save_scan(Path(name).stem, scan, label, split)
                count += 1
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  [WARN] {name}: {e}", flush=True)
    print(f"  saved={count}  skipped={skipped}  errors={errors}", flush=True)


def load_csv(repo_id: str, path: str, col: str) -> dict:
    f = hf_hub_download(repo_id=repo_id, filename=path, repo_type="dataset")
    df = pd.read_csv(f)
    if "filename" not in df.columns or col not in df.columns:
        return {}
    return dict(zip(df["filename"].astype(str), df[col].astype(str)))


def label_map(repo_id: str, csv_path: str) -> dict:
    raw = load_csv(repo_id, csv_path, "glaucoma")
    return {f: 1 if str(v).strip().lower() in ("yes","1","true") else 0 for f, v in raw.items()}


def split_map(data: dict) -> dict:
    n = {"training": "train", "validation": "val", "test": "test",
         "valid": "val", "testing": "test"}
    return {k: n.get(v.lower().strip(), v.lower().strip()) for k, v in data.items()}


def split_random(files: list[str]) -> dict:
    s = sorted(files); random.shuffle(s)
    n, nt, nv = len(s), int(len(s)*0.7), int(len(s)*0.1)
    return {f: "train" for f in s[:nt]} | \
           {f: "val"   for f in s[nt:nt+nv]} | \
           {f: "test"  for f in s[nt+nv:]}


def upload_and_clear(dataset_name: str):
    """Upload current STAGING to HF Hub, then clear staging (but not manifest)."""
    global UPLOAD_COUNT
    flush_manifest()
    shutil.copy(manifest_path, STAGING / "manifest.jsonl")
    upload_folder(repo_id=OUTPUT_REPO, folder_path=str(STAGING), repo_type="dataset",
                  commit_message=f"[{dataset_name}] {len(list(STAGING.rglob('*.npy.gz')))} scans")
    UPLOAD_COUNT += 1
    for f in STAGING.rglob("*"):
        if f.is_file():
            f.unlink()
    for d in sorted(STAGING.rglob("*"), reverse=True):
        if d.is_dir() and d != STAGING:
            try: d.rmdir()
            except: pass
    print(f"  uploaded + cleared (commit #{UPLOAD_COUNT})", flush=True)


# ==============================================================================
# STEP 0: Create HF repo
# ==============================================================================
create_repo(repo_id=OUTPUT_REPO, repo_type="dataset", private=False, exist_ok=True)

# ==============================================================================
# STEP 1: Harvard-GF
# ==============================================================================
print("\n" + "=" * 54, flush=True)
print("  [1/3] Harvard-GF  (3,300 samples)", flush=True)
print("=" * 54, flush=True)

gf_labels = label_map("harvardairobotics/Harvard-GF", "ReadMe/data_summary.csv")
gf_splits = split_map(load_csv("harvardairobotics/Harvard-GF", "ReadMe/data_summary.csv", "use"))
gf_zip = download_zip("harvardairobotics/Harvard-GF", "Dataset/dataset.zip")
extract_from_zip(gf_zip, gf_labels, gf_splits, desc="  Harvard-GF")
gf_zip.unlink(missing_ok=True)
print(f"  disk free: {shutil.disk_usage(STAGING).free/1e9:.1f} GB", flush=True)
upload_and_clear("Harvard-GF")

# ==============================================================================
# STEP 2: FairFedMed-Oph
# ==============================================================================
print("\n" + "=" * 54, flush=True)
print("  [2/3] FairFedMed-Oph  (15,165 samples)", flush=True)
print("=" * 54, flush=True)

ffm_labels = label_map("harvardairobotics/FairFedMed", "FairFedMed-Oph/ReadMe/data_summary.csv")
ffm_zip = download_zip("harvardairobotics/FairFedMed", "FairFedMed-Oph/Dataset/dataset.zip")
with zipfile.ZipFile(str(ffm_zip), "r") as zf:
    ffm_files = sorted(n for n in zf.namelist() if n.endswith(".npz"))
ffm_splits = split_random(ffm_files)
extract_from_zip(ffm_zip, ffm_labels, ffm_splits, desc="  FairFedMed")
ffm_zip.unlink(missing_ok=True)
print(f"  disk free: {shutil.disk_usage(STAGING).free/1e9:.1f} GB", flush=True)
upload_and_clear("FairFedMed-Oph")

# ==============================================================================
# STEP 3: FairGenMed
# ==============================================================================
print("\n" + "=" * 54, flush=True)
print("  [3/3] FairGenMed  (10,052 samples)", flush=True)
print("=" * 54, flush=True)

fgm_labels = label_map("harvardairobotics/FairGenMed", "ReadMe/data_summary.csv")
fgm_splits = split_map(load_csv("harvardairobotics/FairGenMed", "ReadMe/data_summary.csv", "use"))

for sd, sn in [("Training", "train"), ("Validation", "val"), ("Test", "test")]:
    zf = download_zip("harvardairobotics/FairGenMed", f"Dataset/{sd}/NPZ.zip")
    extract_from_zip(zf, fgm_labels, fgm_splits, desc=f"  FairGenMed-{sn}")
    zf.unlink(missing_ok=True)
print(f"  disk free: {shutil.disk_usage(STAGING).free/1e9:.1f} GB", flush=True)
upload_and_clear("FairGenMed")

# ==============================================================================
# FINAL: manifest + README
# ==============================================================================
flush_manifest()

sc: dict = {}; sg: dict = {}
for e in MANIFEST:
    s = e["split"]
    sc[s] = sc.get(s, 0) + 1
    if e["glaucoma"] == 1:
        sg[s] = sg.get(s, 0) + 1
total = sum(sc.values())

print("\n" + "=" * 54, flush=True)
print(f"  TOTAL: {total} scans  |  train={sc.get('train',0)}  val={sc.get('val',0)}  test={sc.get('test',0)}", flush=True)
for s in ("train", "val", "test"):
    n = sc.get(s, 0); g = sg.get(s, 0)
    print(f"  {s}: {n} scans, {g} glau+ ({g/n*100:.1f}%)" if n else f"  {s}: 0", flush=True)
print("=" * 54, flush=True)

readme = f"""---
license: cc-by-nc-nd-4.0
task_categories: [image-classification]
tags: [ophthalmology, oct, glaucoma, medical-imaging]
pretty_name: Harvard OCT Glaucoma B-scans (200³)
size_categories: [10K<n<100K]
---

# Harvard OCT Glaucoma B-scans (200³)
{total} samples, {sc.get('train',0)} train / {sc.get('val',0)} val / {sc.get('test',0)} test.
200×200×200 uint8 volumes, individually gzip-compressed.

## Sources
- [Harvard-GF](https://huggingface.co/datasets/harvardairobotics/Harvard-GF) (3,300)
- [FairFedMed-Oph](https://huggingface.co/datasets/harvardairobotics/FairFedMed) (15,165)
- [FairGenMed](https://huggingface.co/datasets/harvardairobotics/FairGenMed) (10,052)

## Usage
```python
from datasets import load_dataset
ds = load_dataset("{HF_USER}/harvard-oct-glaucoma-200")
```
"""
(STAGING / "README.md").write_text(readme)
shutil.copy(manifest_path, STAGING / "manifest.jsonl")

upload_folder(repo_id=OUTPUT_REPO, folder_path=str(STAGING), repo_type="dataset",
              commit_message=f"FINAL: {total} Harvard OCT B-scans, 3 sources")

print(f"\n  Done!  https://huggingface.co/datasets/{OUTPUT_REPO}", flush=True)
shutil.rmtree(WORK_DIR, ignore_errors=True)
