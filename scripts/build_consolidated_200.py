import argparse
import csv
import io
import json
import os
import sys
import time
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RES = 200
SPLITS = ("Training", "Validation", "Test")
SPLIT_ALIAS = {"training": "Training", "validation": "Validation",
               "valid": "Validation", "test": "Test", "testing": "Test"}


def ensure_meta(src_repo, work):
    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    os.makedirs(work, exist_ok=True)
    csv_path = hf_hub_download(repo_id=src_repo, filename="ReadMe/data_summary.csv",
                               repo_type="dataset", token=token)
    zip_path = hf_hub_download(repo_id=src_repo, filename="Dataset/dataset.zip",
                               repo_type="dataset", token=token)
    meta = {}
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            split = SPLIT_ALIAS.get((r["use"] or "").strip().lower())
            if split is None:
                continue
            stem = os.path.splitext(os.path.basename(r["filename"]))[0]
            meta[stem] = (split, 1 if str(r["glaucoma"]).strip().lower() in ("yes", "1", "true") else 0)
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".npz")
                       and os.path.splitext(os.path.basename(n))[0] in meta)
    return zip_path, meta, names


def counts_by_split(meta, names):
    c = {s: 0 for s in SPLITS}
    for n in names:
        c[meta[os.path.splitext(os.path.basename(n))[0]][0]] += 1
    return c


def build(zip_path, meta, names, out_dir, save_every=25):
    os.makedirs(out_dir, exist_ok=True)
    counts = counts_by_split(meta, names)
    print("[build] counts:", counts, flush=True)
    vols, labs = {}, {}
    for s in SPLITS:
        if counts[s] == 0:
            continue
        vp = os.path.join(out_dir, f"{s}_volumes.npy")
        lp = os.path.join(out_dir, f"{s}_labels.npy")
        mode = "r+" if os.path.exists(vp) else "w+"
        vols[s] = np.lib.format.open_memmap(vp, mode=mode, dtype=np.uint8, shape=(counts[s], 1, RES, RES, RES))
        labs[s] = np.lib.format.open_memmap(lp, mode="r+" if os.path.exists(lp) else "w+",
                                            dtype=np.int64, shape=(counts[s],))
    prog_path = os.path.join(out_dir, "progress.json")
    done = set(json.load(open(prog_path)).get("stems", [])) if os.path.exists(prog_path) else set()
    filled = {s: sum(1 for st in done if meta[st][0] == s) for s in SPLITS}
    todo = [n for n in names if os.path.splitext(os.path.basename(n))[0] not in done]
    print(f"[build] resume {len(done)}/{len(names)}; todo {len(todo)}", flush=True)
    t0, w0 = time.time(), len(done)
    with zipfile.ZipFile(zip_path) as zf:
        for i, entry in enumerate(todo):
            stem = os.path.splitext(os.path.basename(entry))[0]
            split, label = meta[stem]
            vol = np.load(io.BytesIO(zf.read(entry)))["oct_bscans"].astype(np.uint8)
            vols[split][filled[split]] = vol[None]
            labs[split][filled[split]] = label
            filled[split] += 1
            done.add(stem)
            if (len(done) - w0) % save_every == 0 or i == len(todo) - 1:
                for s in SPLITS:
                    if s in vols:
                        vols[s].flush()
                        labs[s].flush()
                with open(prog_path, "w") as fh:
                    json.dump({"stems": sorted(done)}, fh)
                el = time.time() - t0
                rate = el / max(len(done) - w0, 1)
                rem = rate * max(len(todo) - i - 1, 0)
                print(f"[build] {len(done)}/{len(names)} | {rate:.2f}s/vol | ETA {rem/60:.1f} min", flush=True)
    for s in SPLITS:
        if s in vols:
            vols[s].flush()
            labs[s].flush()
    with open(prog_path, "w") as fh:
        json.dump({"stems": sorted(done)}, fh)
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump({"size_name": str(RES), "store_shape": [1, RES, RES, RES], "source_shape": [RES, RES, RES],
                   "antialias": False, "source_datasets": ["harvardairobotics/Harvard-GF"],
                   "splits": {s: {"built_n": counts[s]} for s in SPLITS},
                   "files_layout": {s: {"volumes": f"{s}_volumes.npy", "labels": f"{s}_labels.npy"} for s in SPLITS}},
                  fh, indent=2)
    print("[build] DONE", flush=True)


def upload(out_dir, repo, private):
    from huggingface_hub import HfApi
    try:
        from pipeline.utils import load_env_file
        load_env_file()
    except Exception:
        pass
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo = repo or f"{user}/harvard-oct-glaucoma-200"
    api.create_repo(repo_id=repo, repo_type="dataset", private=bool(private), exist_ok=True)
    print(f"[upload] {out_dir} -> {repo}", flush=True)
    api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=out_dir)
    print("uploaded:", "https://huggingface.co/datasets/" + repo, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-repo", default="harvardairobotics/Harvard-GF")
    ap.add_argument("--out-dir", default=os.path.join("data", "glaucoma_all_200"))
    ap.add_argument("--out-repo", default="")
    ap.add_argument("--private", type=int, default=0)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        args.download = args.build = args.upload = True

    try:
        from pipeline.utils import load_env_file
        load_env_file()
    except Exception as e:
        print("[warn] load_env_file failed:", e)
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        print("[warn] HF_TOKEN not set - downloads will be unauthenticated/slower")

    if args.download:
        t = time.time()
        zip_path, meta, names = ensure_meta(args.src_repo, args.out_dir)
        print(f"[download] zip + csv ready in {(time.time()-t)/60:.1f} min; {len(names)} volumes", flush=True)
    else:
        zip_path, meta, names = ensure_meta(args.src_repo, args.out_dir)
    if args.build:
        build(zip_path, meta, names, args.out_dir)
    if args.upload:
        upload(args.out_dir, args.out_repo, args.private)


if __name__ == "__main__":
    main()
