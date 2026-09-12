"""Build and optionally upload a 96-cubed storage view of the Bilateral-200 export.

The 3D volumes are resized exactly like ``FinalDataset`` does on the fly: cast to
float32, divide by 255, ``F.interpolate(mode="trilinear", align_corners=False)``,
then round and clip back to uint8. The 2D en-face views still have to be projected
from the 200-cubed source; this dataset only accelerates the 3D branch input.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SPLITS = ("Training", "Validation", "Test")
SOURCE_REPO = "tqhuyen/harvard-oct-glaucoma-200-bilateral"
SOURCE_REVISION = "47632c96b206707fd6423ee5b4da159069f63eaf"
DEST_REPO = "tqhuyen/harvard-oct-glaucoma-200-bilateral-96"


def sha256_file(path, chunk=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def resize_split(src, dst, split, res, batch):
    volumes = np.load(src / f"{split}_volumes_dn.npy", mmap_mode="r")
    labels = np.load(src / f"{split}_labels.npy")
    count = len(volumes)
    target = dst / f"{split}_volumes_dn{res}.npy"
    out = np.lib.format.open_memmap(target, mode="w+", dtype=np.uint8, shape=(count, 1, res, res, res))
    np.save(dst / f"{split}_labels.npy", labels)
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        chunk = torch.from_numpy(np.asarray(volumes[start:stop], dtype=np.float32) / 255.0)
        small = F.interpolate(chunk, size=(res, res, res), mode="trilinear", align_corners=False)
        out[start:stop] = torch.round(small.clamp(0, 1) * 255.0).to(torch.uint8).numpy()
        out.flush()
        print(f"[resize] {split} {stop}/{count}", flush=True)
    del out
    (dst / f"{split}_resize.complete.json").write_text(
        json.dumps({"split": split, "count": count, "res": res}), encoding="utf-8"
    )
    return {"count": count, "labels": labels}


def build_manifest(src, dst, res):
    source = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    raw = source.get("identity", {})
    manifest = {
        "complete": True,
        "total_volumes": source.get("total_volumes"),
        "identity": {
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "raw_source_repo": raw.get("source_repo"),
            "raw_source_revision": raw.get("source_revision"),
            "storage_resolution": res,
            "dtype": "uint8",
            "resize": {
                "method": "trilinear",
                "align_corners": False,
                "cast": "float32 and /255 before interpolate",
                "quantize": "round after *255 and clip to [0,255]",
            },
            "note": "Only the 3D branch input is accelerated; 2D views must still be projected from the 200-cubed source.",
        },
        "splits": {},
    }
    for split in SPLITS:
        volumes = dst / f"{split}_volumes_dn{res}.npy"
        labels = dst / f"{split}_labels.npy"
        values = np.load(labels)
        manifest["splits"][split] = {
            "volume_file": volumes.name,
            "labels_file": labels.name,
            "shape": list(np.load(volumes, mmap_mode="r").shape),
            "volume_sha256": sha256_file(volumes),
            "labels_sha256": sha256_file(labels),
            "class_counts": {"0": int((values == 0).sum()), "1": int((values == 1).sum())},
        }
        json.dump(manifest["splits"][split], open(dst / f"{split}_complete.json", "w"), indent=2)
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (dst / "README.md").write_text(
        "\n".join(
            [
                f"# Bilateral 96-cubed storage ({res}^3)",
                "",
                f"Derived from `{SOURCE_REPO}` (revision `{SOURCE_REVISION}`) by resizing only the 3D volumes to {res}^3.",
                "Resize: float32 /255, trilinear `align_corners=False`, round and clip back to uint8.",
                "2D en-face views are NOT included and must be projected from the 200-cubed source.",
                "",
                "Private derivative; respect the upstream dataset terms.",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def upload(dst, repo, token, revision=None):
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(dst),
        commit_message="Add Bilateral 96-cubed storage (uint8, trilinear from 200)",
        revision=revision,
    )
    return commit


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build (and optionally upload) the Bilateral 96 storage")
    parser.add_argument("--src", type=Path, default=Path("data/bilateral_200"))
    parser.add_argument("--dst", type=Path, default=Path("data/bilateral_96"))
    parser.add_argument("--res", type=int, default=96)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--repo", default=DEST_REPO)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args(argv)
    args.dst.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        if (args.dst / f"{split}_resize.complete.json").is_file():
            print(f"[resize] {split} cache hit")
        else:
            resize_split(args.src, args.dst, split, args.res, args.batch)
    manifest = build_manifest(args.src, args.dst, args.res)
    print("[manifest]", json.dumps({s: manifest["splits"][s]["shape"] for s in SPLITS}))
    if args.upload:
        import os

        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN required for upload")
        commit = upload(args.dst, args.repo, token)
        print("[upload] repo", args.repo, "commit", getattr(commit, "oid", commit))
    return args.dst


if __name__ == "__main__":
    main()
