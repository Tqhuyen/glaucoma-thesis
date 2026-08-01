#!/usr/bin/env python3
"""
Harvard OCT B-Scan Dataset Processor — Standalone Script (no browser needed).

Downloads all three Harvard Ophthalmology AI Lab OCT B-scan datasets from
Hugging Face, extracts only `oct_bscans` + `glaucoma` label (stripping
demographics, fundus images, RNFLT maps), and pushes a unified compressed
dataset back to HF Hub as one combined repo.

Usage:
    export HF_TOKEN=hf_xxx
    python scripts/harvard_oct_processor.py

Runtime: ~2-4 hours depending on bandwidth.
Disk:  ~80 GB peak (one dataset at a time).
"""
from __future__ import annotations

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

# ── config ────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    HUGGING_FACE_HUB_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if HUGGING_FACE_HUB_TOKEN:
        HF_TOKEN = HUGGING_FACE_HUB_TOKEN
        os.environ["HF_TOKEN"] = HF_TOKEN
    else:
        print("ERROR: HF_TOKEN environment variable not set.")
        print("       export HF_TOKEN=hf_xxx")
        sys.exit(1)

api = HfApi(token=HF_TOKEN)
HF_USER = api.whoami()["name"]
OUTPUT_REPO = f"{HF_USER}/harvard-oct-glaucoma-200"
WORK_DIR = Path(__file__).resolve().parent.parent / "data" / "harvard_oct_work"
os.makedirs(WORK_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

print(f"HF User:  {HF_USER}")
print(f"Output:   {OUTPUT_REPO}")
print(f"Work dir: {WORK_DIR}")
print(f"HF Hub:   https://huggingface.co/datasets/{OUTPUT_REPO}")
print()

# ── helpers ──────────────────────────────────────────────────────────
def download_zip(repo_id: str, filepath: str) -> Path:
    """Download a zip from HF Hub, cache in WORK_DIR."""
    local = WORK_DIR / f"{repo_id.replace('/', '_')}_{Path(filepath).name}"
    if local.exists():
        print(f"  Using cached: {local}")
        return local
    print(f"  Downloading {repo_id}/{filepath} ...")
    t0 = time.time()
    downloaded = hf_hub_download(repo_id=repo_id, filename=filepath, repo_type="dataset")
    # Copy to work dir for isolation
    shutil.copy(downloaded, local)
    print(f"  Done in {(time.time() - t0) / 60:.1f} min ({local.stat().st_size / 1e9:.1f} GB)")
    return local


def extract_npz_zip(zip_path: Path, label_map: dict, out_dir: Path) -> tuple[int, int]:
    """Open a .zip, for each .npz extract oct_bscans + glaucoma, save as .npz."""
    os.makedirs(out_dir, exist_ok=True)
    count, errors = 0, 0
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        npz_names = sorted(n for n in zf.namelist() if n.endswith(".npz"))
        for name in tqdm(npz_names, desc=f"  {out_dir.name}"):
            try:
                raw = np.load(io.BytesIO(zf.read(name)), allow_pickle=True)
                oct_scan = np.asarray(raw["oct_bscans"], dtype=np.uint8)
                label = label_map.get(name, -1)
                np.savez_compressed(str(out_dir / name), oct_bscans=oct_scan, glaucoma=label)
                count += 1
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  [WARN] {name}: {e}")
    return count, errors


def load_split_csv(repo_id: str, csv_path: str) -> dict:
    """Download CSV and return {filename: split} mapping."""
    local = hf_hub_download(repo_id=repo_id, filename=csv_path, repo_type="dataset")
    df = pd.read_csv(local)
    if "use" in df.columns:
        mapping = dict(zip(df["filename"], df["use"]))
        # Normalize split names
        norm = {"training": "train", "validation": "val", "test": "test"}
        return {k: norm.get(v, "train") for k, v in mapping.items()}
    return {}


def load_label_csv(repo_id: str, csv_path: str) -> dict:
    local = hf_hub_download(repo_id=repo_id, filename=csv_path, repo_type="dataset")
    df = pd.read_csv(local)
    return dict(zip(df["filename"], df["glaucoma"].map({"yes": 1, "no": 0})))


# ======================================================================
# STEP 1: Download + Extract all three datasets
# ======================================================================

all_scans = {"train": [], "val": [], "test": []}
all_labels = {"train": [], "val": [], "test": []}


def add_extracted(source_dir: str, split_map: dict, label_map: dict):
    """Load extracted .npz files into all_scans by split."""
    base = Path(source_dir)
    npz_files = list(base.rglob("*.npz"))
    print(f"  Loading {len(npz_files)} files ...")
    for npz_path in tqdm(npz_files, desc=f"  {Path(source_dir).name}"):
        try:
            data = dict(np.load(npz_path, allow_pickle=True))
            scan = np.asarray(data["oct_bscans"], dtype=np.uint8)
            label = int(data["glaucoma"])
            fname = npz_path.name
            split = split_map.get(fname, "train") if split_map else "pool"
            if split == "pool":
                continue
            all_scans[split].append(scan)
            all_labels[split].append(label)
        except Exception:
            pass


# ── Harvard-GF ──────────────────────────────────────────────────────
print("=" * 60)
print("  Harvard-GF (3,300 samples)")
print("=" * 60)
gf_labels = load_label_csv("harvardairobotics/Harvard-GF", "ReadMe/data_summary.csv")
gf_splits = load_split_csv("harvardairobotics/Harvard-GF", "ReadMe/data_summary.csv")
gf_zip = download_zip("harvardairobotics/Harvard-GF", "Dataset/dataset.zip")
gf_out = WORK_DIR / "harvard_gf_extracted"
gf_n, gf_err = extract_npz_zip(gf_zip, gf_labels, gf_out)
print(f"  Extracted: {gf_n} files, {gf_err} errors")
add_extracted(str(gf_out), gf_splits, gf_labels)

# ── FairFedMed-Oph ──────────────────────────────────────────────────
print()
print("=" * 60)
print("  FairFedMed-Oph (15,165 samples)")
print("=" * 60)
ffm_labels = load_label_csv("harvardairobotics/FairFedMed", "FairFedMed-Oph/ReadMe/data_summary.csv")
ffm_zip = download_zip("harvardairobotics/FairFedMed", "FairFedMed-Oph/Dataset/dataset.zip")
ffm_out = WORK_DIR / "fairfedmed_extracted"
ffm_n, ffm_err = extract_npz_zip(ffm_zip, ffm_labels, ffm_out)
print(f"  Extracted: {ffm_n} files, {ffm_err} errors")

# FairFedMed has no splits → shuffle and split 70/10/20
print("  Splitting 70/10/20 ...")
ffm_scans, ffm_lbls = [], []
for npz_path in tqdm(list(Path(ffm_out).glob("*.npz")), desc="  FairFedMed"):
    try:
        d = dict(np.load(npz_path, allow_pickle=True))
        ffm_scans.append(np.asarray(d["oct_bscans"], dtype=np.uint8))
        ffm_lbls.append(int(d["glaucoma"]))
    except Exception:
        pass

combined = list(zip(ffm_scans, ffm_lbls))
random.shuffle(combined)
n = len(combined)
n_tr, n_vl = int(n * 0.7), int(n * 0.1)
print(f"  Total: {n}, train={n_tr}, val={n_vl}, test={n - n_tr - n_vl}")

for scan, lbl in combined[:n_tr]:
    all_scans["train"].append(scan)
    all_labels["train"].append(lbl)
for scan, lbl in combined[n_tr : n_tr + n_vl]:
    all_scans["val"].append(scan)
    all_labels["val"].append(lbl)
for scan, lbl in combined[n_tr + n_vl :]:
    all_scans["test"].append(scan)
    all_labels["test"].append(lbl)

# ── FairGenMed ──────────────────────────────────────────────────────
print()
print("=" * 60)
print("  FairGenMed (10,052 samples)")
print("=" * 60)
fgm_labels = load_label_csv("harvardairobotics/FairGenMed", "ReadMe/data_summary.csv")
fgm_splits = load_split_csv("harvardairobotics/FairGenMed", "ReadMe/data_summary.csv")

for split_dir, split_name in [("Training", "training"), ("Validation", "validation"), ("Test", "test")]:
    print(f"  [{split_name}]")
    zfile = f"Dataset/{split_dir}/NPZ.zip"
    fgm_zip = download_zip("harvardairobotics/FairGenMed", zfile)
    fgm_out = WORK_DIR / f"fairgenmed_{split_name}"
    n_files, n_err = extract_npz_zip(fgm_zip, fgm_labels, fgm_out)
    print(f"    Extracted: {n_files} files, {n_err} errors")

add_extracted(str(WORK_DIR / "fairgenmed_training"), fgm_splits, fgm_labels)
add_extracted(str(WORK_DIR / "fairgenmed_validation"), fgm_splits, fgm_labels)
add_extracted(str(WORK_DIR / "fairgenmed_test"), fgm_splits, fgm_labels)

# ======================================================================
# STEP 2: Summary
# ======================================================================
print()
print("=" * 60)
print("  Combined Dataset Summary")
print("=" * 60)
print(f"{'Split':<10} {'Samples':<12} {'Glaucoma+':<12} {'%':<8}")
print("-" * 44)
grand = 0
for split in ["train", "val", "test"]:
    n_tot = len(all_scans[split])
    n_gl = sum(1 for l in all_labels[split] if l == 1)
    grand += n_tot
    pct = n_gl / n_tot * 100 if n_tot else 0
    print(f"{split:<10} {n_tot:<12} {n_gl:<12} {pct:.1f}%")
print("-" * 44)
print(f"{'TOTAL':<10} {grand}")

# ======================================================================
# STEP 3: Save unified dataset
# ======================================================================
print()
print("=" * 60)
print("  Saving unified .npz files")
print("=" * 60)

upload_dir = WORK_DIR / "unified"
os.makedirs(upload_dir, exist_ok=True)

for split_name in ["train", "val", "test"]:
    scans = all_scans[split_name]
    labels = all_labels[split_name]
    if not scans:
        continue

    X = np.stack(scans, axis=0)
    y = np.array(labels, dtype=np.int16)

    out_path = upload_dir / f"{split_name}_volumes.npz"
    print(f"  {split_name}: X={X.shape} y={y.shape} ... ", end="", flush=True)
    np.savez_compressed(out_path, oct_bscans=X, glaucoma=y)
    print(f"{out_path.stat().st_size / 1e9:.2f} GB")

manifest = {
    "description": "Unified Harvard OCT B-scan dataset (200x200x200) for glaucoma classification",
    "total_samples": grand,
    "resolution": [200, 200, 200],
    "dtype": "uint8",
    "channels": 1,
    "classes": {"0": "no_glaucoma", "1": "glaucoma"},
    "source_datasets": [
        "harvardairobotics/Harvard-GF",
        "harvardairobotics/FairFedMed (Oph subset)",
        "harvardairobotics/FairGenMed",
    ],
    "license": "cc-by-nc-nd-4.0",
    "splits": {},
}
for s in ["train", "val", "test"]:
    if all_scans[s]:
        manifest["splits"][s] = {
            "samples": len(all_scans[s]),
            "glaucoma_positive": int(sum(1 for l in all_labels[s] if l == 1)),
            "file": f"{s}_volumes.npz",
        }

with open(upload_dir / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

total_upload = sum(f.stat().st_size for f in upload_dir.glob("*.npz"))
print(f"\n  Total upload size: {total_upload / 1e9:.1f} GB")

# ======================================================================
# STEP 4: Push to HF Hub
# ======================================================================
print()
print("=" * 60)
print(f"  Uploading to https://huggingface.co/datasets/{OUTPUT_REPO}")
print("=" * 60)

create_repo(repo_id=OUTPUT_REPO, repo_type="dataset", private=False, exist_ok=True)
upload_folder(
    repo_id=OUTPUT_REPO,
    folder_path=str(upload_dir),
    repo_type="dataset",
    commit_message="Add unified Harvard OCT B-scans dataset (200x200x200, 3 sources combined)",
)

print()
print(f"  Done! Dataset at: https://huggingface.co/datasets/{OUTPUT_REPO}")
print(f"  Load with: datasets.load_dataset('{OUTPUT_REPO}')")

# Cleanup work dir
print(f"\n  Cleaning up work dir ({WORK_DIR}) ...")
shutil.rmtree(WORK_DIR, ignore_errors=True)
print("  Done. All temp files removed.")
