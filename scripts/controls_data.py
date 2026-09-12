"""Authenticated, cache-only bilateral preparation for control notebooks."""

import hashlib
import json
import re
from pathlib import Path

from scripts import final_data as fd

DEFAULT_REPO = "tqhuyen/harvard-oct-glaucoma-200-bilateral"
DEFAULT_REVISION = "47632c96b206707fd6423ee5b4da159069f63eaf"


def _verify_file(path, remote):
    lfs = remote.lfs
    expected = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    blob = remote.blob_id
    if remote.size != path.stat().st_size or not (
        isinstance(expected, str)
        and re.fullmatch(r"[0-9a-f]{64}", expected)
        or not lfs
        and isinstance(blob, str)
        and re.fullmatch(r"[0-9a-f]{40}", blob)
    ):
        raise ValueError(f"Missing or mismatched pinned HF metadata: {path.name}")
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1(f"blob {remote.size}\0".encode()) if not expected else None
    before = path.stat()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            if sha1 is not None:
                sha1.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source changed during verification: {path.name}")
    if (expected and sha256.hexdigest() != expected) or (sha1 is not None and sha1.hexdigest() != blob):
        raise ValueError(f"Pinned HF checksum mismatch: {path.name}; refusing replacement")
    return sha256.hexdigest()


def prepare_bilateral(
    data_root,
    export_root,
    *,
    token,
    repo=DEFAULT_REPO,
    revision=DEFAULT_REVISION,
    splits=("Training", "Validation", "Test"),
    res2d=224,
    store_res=200,
):
    """Verify pinned sources, import bilateral volumes and build two cached views.

    Existing raw bytes are never replaced. Only missing raw files are downloaded,
    directly into data_root. The CPU export contract requires raw 200-cubed
    storage; res2d controls only the imported bilateral view cache. The returned
    JSON-safe identity contains no credentials and can be logged in run config.
    """
    if not isinstance(token, str) or not token.strip():
        raise ValueError("An explicit HF token is required")
    if not isinstance(repo, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("Require an explicit HF dataset repo")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Require a pinned 40-SHA export revision")
    if type(store_res) is not int or store_res != 200:
        raise ValueError("CPU bilateral export requires store_res=200")
    if type(res2d) is not int or res2d <= 0:
        raise ValueError("Require positive integer res2d")
    if isinstance(splits, str):
        raise ValueError("Require a sequence of split names")
    splits = fd._splits(splits)
    data_root, export_root = Path(data_root), Path(export_root)
    if data_root.resolve() == export_root.resolve():
        raise ValueError("Raw and export directories must be distinct")

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=token)
    export_info = api.dataset_info(repo, revision=revision, files_metadata=True)
    if export_info.sha != revision:
        raise ValueError("HF export revision mismatch")
    export_files = {f.rfilename: f for f in export_info.siblings}
    if "manifest.json" not in export_files:
        raise ValueError("Pinned export is missing manifest.json")
    print("[data] Downloading pinned bilateral manifest")
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        token=token,
        local_dir=str(export_root),
        allow_patterns=["manifest.json"],
    )
    _verify_file(export_root / "manifest.json", export_files["manifest.json"])
    exported = fd._json_read(export_root / "manifest.json")
    if not isinstance(exported, dict) or exported.get("complete") is not True:
        raise ValueError("Bilateral export requires a complete manifest")
    identity = exported.get("identity", {})
    source_repo, source_revision = identity.get("source_repo"), identity.get("source_revision")
    if (
        not isinstance(source_repo, str)
        or not re.fullmatch(r"[\w.-]+/[\w.-]+", source_repo)
        or not isinstance(source_revision, str)
        or not re.fullmatch(r"[0-9a-f]{40}", source_revision)
        or identity.get("resolution") != store_res
    ):
        raise ValueError("Export source repo/revision or storage resolution mismatch")
    patterns = []
    for split in splits:
        source = identity.get("sources", {}).get(split, {})
        marker = exported.get("splits", {}).get(split, {})
        if (
            source.get("repo_revision") != source_revision
            or source.get("shape", [])[-3:] != [store_res] * 3
            or marker.get("complete") is not True
            or marker.get("identity") != identity
        ):
            raise ValueError(f"Incomplete export or source revision/shape mismatch: {split}")
        marker_path = export_root / f"{split}_complete.json"
        if marker_path.exists() and fd._json_read(marker_path) != marker:
            raise ValueError(f"Corrupt completion marker: {split}")
        for suffix in ("volumes_dn.npy", "labels.npy"):
            name = f"{split}_{suffix}"
            if name not in export_files:
                raise ValueError(f"Incomplete pinned export: {name}")
            patterns.append(name)
        if marker_path.name in export_files:
            patterns.append(marker_path.name)

    info = api.dataset_info(source_repo, revision=source_revision, files_metadata=True)
    if info.sha != source_revision:
        raise ValueError("HF raw source revision mismatch")
    files = {f.rfilename: f for f in info.siblings}
    names = [f"{s}_{kind}.npy" for s in splits for kind in ("volumes", "labels")]
    for name in names:
        if name not in files:
            raise ValueError(f"Pinned raw source metadata missing: {name}")
        if name.endswith("_volumes.npy") and not files[name].lfs:
            raise ValueError(f"Pinned raw volumes require LFS SHA256: {name}")
    hashes = {name: _verify_file(data_root / name, files[name]) for name in names if (data_root / name).exists()}
    missing = [name for name in names if name not in hashes]
    if missing:
        snapshot_download(
            repo_id=source_repo,
            repo_type="dataset",
            revision=source_revision,
            token=token,
            local_dir=str(data_root),
            allow_patterns=missing,
        )
        hashes.update({name: _verify_file(data_root / name, files[name]) for name in missing})
    provenance = {"source_repo": source_repo, "source_revision": source_revision, "sources": {}}
    for split in splits:
        source = identity["sources"][split]
        raw_hash, labels_hash = hashes[f"{split}_volumes.npy"], hashes[f"{split}_labels.npy"]
        if source.get("labels_sha256") != labels_hash or source.get("sha256", raw_hash) != raw_hash:
            raise ValueError(f"Export raw/labels checksum mismatch: {split}")
        provenance["sources"][split] = {"sha256": raw_hash, "labels_sha256": labels_hash}
    saved_path = data_root / "source_manifest.json"
    if saved_path.exists():
        saved = fd._json_read(saved_path)
        if not isinstance(saved, dict) or any(
            saved.get(k) != provenance[k] for k in ("source_repo", "source_revision")
        ):
            raise ValueError("Existing raw source repo/revision provenance mismatch")
        if any(saved.get("sources", {}).get(s) != provenance["sources"][s] for s in splits):
            raise ValueError("Existing raw source hashes mismatch")
        provenance["sources"] = {**saved["sources"], **provenance["sources"]}

    print("[data] Raw sources verified; downloading selected bilateral splits")
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        token=token,
        local_dir=str(export_root),
        allow_patterns=patterns,
    )
    for split in splits:
        marker_path = export_root / f"{split}_complete.json"
        marker = exported["splits"][split]
        if marker_path.exists():
            if fd._json_read(marker_path) != marker:
                raise ValueError(f"Corrupt completion marker: {split}")
        else:
            fd._json_write(marker_path, marker)
    fd._json_write(saved_path, provenance)
    print("[cache] Importing verified bilateral volumes and preparing views")
    manifest = fd.import_cpu_export(
        data_root,
        export_root,
        data_root / ".controls_cache",
        splits,
        res2d=res2d,
        method="bilateral",
        params=fd._CPU_PARAMS.copy(),
        implementation=fd._CPU_IMPLEMENTATION,
        smoke=False,
        allow_publish=False,
        source_manifest_path=saved_path,
    )
    return json.loads(
        json.dumps(
            {
                "dataset": "bilateral",
                "repo": repo,
                "revision": revision,
                "store_res": store_res,
                "res2d": res2d,
                "splits": splits,
                "raw_download": provenance,
                "cpu_export": identity,
                "manifest": manifest,
            },
            allow_nan=False,
        )
    )
