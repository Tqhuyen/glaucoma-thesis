"""Upload the completed bilateral dataset to an authenticated HF dataset repo."""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOW = ["*_volumes_dn.npy", "*_labels.npy", "manifest.json", "README.md"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=ROOT / "data/bilateral_200")
    parser.add_argument("--repo", default="tqhuyen/harvard-oct-glaucoma-200-bilateral")
    parser.add_argument("--private", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    directory = args.dir.resolve()
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError("Preprocessing manifest is missing; refusing to upload an unverified dataset")
    missing = [p for p in ("Training", "Validation", "Test") if not (directory / f"{p}_complete.json").is_file()]
    if missing:
        raise RuntimeError(f"Incomplete splits: {missing}; refusing to upload")
    sys.path.insert(0, str(ROOT))
    from pipeline.utils import load_env_file

    load_env_file()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set; authenticated upload is required")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    user = api.whoami()["name"]
    print(f"[upload] authenticated as {user} -> {args.repo} (private={bool(args.private)})", flush=True)
    api.create_repo(repo_id=args.repo, repo_type="dataset", private=bool(args.private), exist_ok=True)
    began = time.time()
    api.upload_large_folder(
        repo_id=args.repo,
        folder_path=directory,
        repo_type="dataset",
        private=bool(args.private),
        allow_patterns=ALLOW,
        num_workers=args.workers,
        print_report=True,
        print_report_every=30,
    )
    url = f"https://huggingface.co/datasets/{args.repo}"
    print(f"[upload] done in {(time.time() - began) / 60:.1f} min -> {url}", flush=True)


if __name__ == "__main__":
    main()
