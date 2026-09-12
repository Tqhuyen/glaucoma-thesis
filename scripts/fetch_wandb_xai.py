"""Download logged W&B media (images) for runs identified by display name."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path, chunk=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(entity, project, names, out, prefix="media/images/"):
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    import wandb

    api = wandb.Api()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in names:
        matches = list(api.runs(f"{entity}/{project}", filters={"display_name": name}))
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one run named {name}, found {len(matches)}")
        run = matches[0]
        dest = out / name
        summary = {
            key: value
            for key, value in run.summary.items()
            if isinstance(value, (int, float, str)) and key.startswith(("xai", "stage"))
        }
        entry = {
            "id": run.id,
            "state": run.state,
            "url": run.url,
            "config": dict(run.config),
            "summary": summary,
            "files": {},
        }
        for remote in run.files():
            if not remote.name.startswith(prefix):
                continue
            remote.download(root=str(dest), replace=True)
            downloaded = dest / remote.name
            flat = dest / Path(remote.name).name
            if downloaded != flat:
                flat.parent.mkdir(parents=True, exist_ok=True)
                downloaded.replace(flat)
            entry["files"][flat.name] = {"size": flat.stat().st_size, "sha256": sha256_file(flat)}
        manifest[name] = entry
        print(f"[fetch] {name} ({run.id}): {len(entry['files'])} files -> {dest}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download W&B media for named runs")
    parser.add_argument("--entity", default="quang-huyen")
    parser.add_argument("--project", default="glaucoma-thesis")
    parser.add_argument("--run", dest="runs", action="append", required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "figures/xai")
    parser.add_argument("--prefix", default="media/images/")
    args = parser.parse_args(argv)
    fetch(args.entity, args.project, args.runs, args.out, args.prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
