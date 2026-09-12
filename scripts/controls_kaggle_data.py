"""Read-only pinned sources and writable, content-addressed Kaggle view caches."""

import contextlib
import hashlib
import inspect
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path

from scripts import controls_kaggle_setup as ks

ks.require_ready(smoke=os.environ.get("CTRL_KAGGLE_SMOKE", "0") == "1", importing=True)

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from scripts import controls_data as cd
from scripts import controls_training as ct
from scripts import final_model as fm
from scripts import final_training as ft

ks.record_imports()

RAW_REPO = "tqhuyen/harvard-oct-glaucoma-200"
RAW_REVISION = "939a38876b7b9313162842ef2d44b7edc2b57020"
BM3D_REPO = "tqhuyen/harvard-gf-denoise-benchmark-v2"
BM3D_PREFIX = "classical/bm3d/3375a321513938835d2c"
SPLITS = ("Training", "Validation", "Test")


def source_recipe(kind):
    if kind == "bilateral":
        return {"repo": cd.DEFAULT_REPO, "revision": cd.DEFAULT_REVISION}
    if kind == "bm3d":
        return {"repo": BM3D_REPO, "prefix": BM3D_PREFIX}
    return {"repo": RAW_REPO, "revision": RAW_REVISION}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                break
            except PermissionError as exc:
                if os.name != "nt" or getattr(exc, "winerror", None) not in (5, 32) or attempt == 19:
                    raise
                time.sleep(0.01 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def require_disk(path, needed):
    path = Path(path)
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free
    if free < needed:
        raise OSError(f"Insufficient disk at {path}: need {needed / 1024**3:.2f} GiB, free {free / 1024**3:.2f}")


@contextlib.contextmanager
def run_lock(path):
    from filelock import FileLock, Timeout

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    guard = FileLock(str(path) + ".guard")
    try:
        guard.acquire(timeout=0)
    except Timeout as exc:
        raise RuntimeError(f"Duplicate controller/job/cache writer rejected: {path}") from exc
    try:
        if path.exists():
            old = json.loads(path.read_text())
            try:
                alive = abs(psutil.Process(old["pid"]).create_time() - old["create_time"]) < 0.01
            except psutil.NoSuchProcess:
                alive = False
            if alive:
                raise RuntimeError(f"Duplicate controller/job/cache writer rejected: {path}")
            path.unlink()
        with path.open("x") as stream:
            json.dump({"pid": os.getpid(), "create_time": psutil.Process().create_time()}, stream)
        try:
            yield
        finally:
            path.unlink(missing_ok=True)
    finally:
        guard.release()


def prepare_source(cfg, kind):
    ks.require_ready(smoke=cfg["smoke"])
    with run_lock(Path(cfg["temp_root"]) / f"source-{kind}.lock"):
        return _prepare_source(cfg, kind)


def _prepare_source(cfg, kind):
    if kind == "bm3d":
        return _prepare_bm3d_source(cfg)
    root = Path(cfg[f"{kind}_root"] or Path(cfg["temp_root"]) / kind)
    root_attached = bool(cfg[f"{kind}_root"])
    suffix = "volumes_dn" if kind == "bilateral" else "volumes"
    names = [f"{split}_{part}.npy" for split in SPLITS for part in (suffix, "labels")]
    if cfg["smoke"]:
        if root_attached:
            raise ValueError("Synthetic smoke requires temp roots; attached inputs are never written")
        root.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        for split in SPLITS:
            for part, array in (
                (suffix, rng.integers(0, 256, (4, 8, 8, 8), dtype=np.uint8)),
                ("labels", np.array([0, 1, 0, 1], dtype=np.int64)),
            ):
                path = root / f"{split}_{part}.npy"
                if not path.exists():
                    np.save(path, array)
        hashes = {name: sha256(root / name) for name in names}
        manifest = None
    else:
        from huggingface_hub import HfApi, snapshot_download

        repo, revision = (cd.DEFAULT_REPO, cd.DEFAULT_REVISION) if kind == "bilateral" else (RAW_REPO, RAW_REVISION)
        token = os.environ["HF_TOKEN"]
        info = HfApi(token=token).dataset_info(repo, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise ValueError("Pinned HF revision mismatch")
        files = {item.rfilename: item for item in info.siblings}
        if kind == "bilateral":
            names += ["manifest.json"] + [f"{split}_complete.json" for split in SPLITS]
        if any(name not in files for name in names):
            raise ValueError("Pinned source missing required files")
        missing = [name for name in names if not (root / name).is_file()]
        if missing and root_attached:
            raise FileNotFoundError(f"Attached source is incomplete (never written): {missing}")
        if missing:
            require_disk(root, sum(files[name].size for name in missing) + 2 * 1024**3)
            root.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                revision=revision,
                token=token,
                local_dir=str(root),
                allow_patterns=missing,
            )
        hashes = {}
        for name in names:
            print(f"[data] hash start {kind}/{name}", flush=True)
            hashes[name] = cd._verify_file(root / name, files[name])
            print(f"[data] hash done {kind}/{name}: {hashes[name]}", flush=True)
        manifest = json.loads((root / "manifest.json").read_text()) if kind == "bilateral" else None
        write_json(Path(cfg["temp_root"]) / f"{kind}_source_hashes.json", hashes)
    entries = {}
    for split in SPLITS:
        vp, lp = root / f"{split}_{suffix}.npy", root / f"{split}_labels.npy"
        volumes, labels = np.load(vp, mmap_mode="r"), np.load(lp, mmap_mode="r")
        res = 8 if cfg["smoke"] else 200
        if volumes.dtype != np.uint8 or volumes.shape[-3:] != (res,) * 3:
            raise ValueError("Source must be uint8 storage-200, never stored bilateral-96")
        if volumes.ndim not in (4, 5) or (volumes.ndim == 5 and volumes.shape[1] != 1):
            raise ValueError("Invalid source dimensions")
        if labels.shape != (len(volumes),) or set(np.unique(labels)) != {0, 1}:
            raise ValueError("Invalid binary split labels")
        entries[split] = {
            "source": str(vp),
            "labels": str(lp),
            "identity": {
                "volumes": hashes[vp.name],
                "labels": hashes[lp.name],
                "shape": list(volumes.shape),
                "repo": cd.DEFAULT_REPO if kind == "bilateral" else RAW_REPO,
                "revision": cd.DEFAULT_REVISION if kind == "bilateral" else RAW_REVISION,
                "manifest": hashes.get("manifest.json"),
            },
        }
    print(f"[data] Verified {kind} sources (read-only)", flush=True)
    return {"kind": kind, "entries": entries, "manifest": manifest, "hashes": hashes}


def _bm3d_identity(shard_hashes, labels_hash, shape=None):
    body = {"repo": BM3D_REPO, "prefix": BM3D_PREFIX, "shards": list(shard_hashes), "labels": labels_hash}
    if shape is not None:
        body["shape"] = list(shape)
    return {
        "volumes": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
        "labels": labels_hash,
        "shape": list(shape) if shape is not None else None,
        "repo": BM3D_REPO,
        "prefix": BM3D_PREFIX,
    }


def _validate_bm3d_split(split, volumes, labels, res):
    if volumes.dtype != np.uint8 or volumes.shape[-3:] != (res,) * 3:
        raise ValueError("BM3D source must be uint8 storage-200, never stored bilateral-96")
    if volumes.ndim not in (4, 5) or (volumes.ndim == 5 and volumes.shape[1] != 1):
        raise ValueError("Invalid BM3D source dimensions")
    if labels.shape != (len(volumes),) or set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"BM3D split {split} labels do not match volume count or are not binary")


def _prepare_bm3d_source(cfg):
    root = Path(cfg["bm3d_root"] or Path(cfg["temp_root"]) / "bm3d")
    root_attached = bool(cfg["bm3d_root"])
    names = [f"{split}_volumes.npy" for split in SPLITS] + [f"{split}_labels.npy" for split in SPLITS]
    res = 8 if cfg["smoke"] else 200
    entries, hashes = {}, {}
    if cfg["smoke"]:
        if root_attached:
            raise ValueError("Synthetic smoke requires temp roots; attached inputs are never written")
        root.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        for split in SPLITS:
            vp, lp = root / f"{split}_volumes.npy", root / f"{split}_labels.npy"
            if not vp.exists():
                np.save(vp, rng.integers(0, 256, (4, 8, 8, 8), dtype=np.uint8))
            if not lp.exists():
                np.save(lp, np.array([0, 1, 0, 1], dtype=np.int64))
        for name in names:
            hashes[name] = sha256(root / name)
        for split in SPLITS:
            vp, lp = root / f"{split}_volumes.npy", root / f"{split}_labels.npy"
            volumes, labels = np.load(vp, mmap_mode="r"), np.load(lp, mmap_mode="r")
            _validate_bm3d_split(split, volumes, labels, res)
            entries[split] = {
                "source": str(vp),
                "labels": str(lp),
                "identity": _bm3d_identity([hashes[vp.name]], hashes[lp.name], list(volumes.shape)),
            }
        print("[data] Verified bm3d sources (read-only synthetic)", flush=True)
        return {"kind": "bm3d", "entries": entries, "manifest": None, "hashes": hashes}
    from huggingface_hub import HfApi, snapshot_download

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    benchmark = api.dataset_info(BM3D_REPO, files_metadata=True)
    files = {item.rfilename: item for item in benchmark.siblings}
    complete_name = f"{BM3D_PREFIX}/_COMPLETE.json"
    if complete_name not in files:
        raise ValueError("BM3D benchmark is missing the per-method _COMPLETE.json")
    label_names = [f"{split}_labels.npy" for split in SPLITS]
    raw_info = api.dataset_info(RAW_REPO, revision=RAW_REVISION, files_metadata=True)
    if raw_info.sha != RAW_REVISION:
        raise ValueError("Pinned HF raw revision mismatch")
    raw_files = {item.rfilename: item for item in raw_info.siblings}
    if any(name not in raw_files for name in label_names):
        raise ValueError("Pinned raw source missing labels")
    missing_labels = [name for name in label_names if not (root / name).is_file()]
    if missing_labels and root_attached:
        raise FileNotFoundError(f"Attached BM3D labels are incomplete (never written): {missing_labels}")
    if missing_labels:
        require_disk(root, sum(raw_files[name].size for name in missing_labels) + 2 * 1024**3)
        root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=RAW_REPO,
            repo_type="dataset",
            revision=RAW_REVISION,
            token=token,
            local_dir=str(root),
            allow_patterns=missing_labels,
        )
    for name in label_names:
        print(f"[data] hash start bm3d/{name}", flush=True)
        hashes[name] = cd._verify_file(root / name, raw_files[name])
        print(f"[data] hash done bm3d/{name}: {hashes[name]}", flush=True)
    write_json(Path(cfg["temp_root"]) / "bm3d_source_hashes.json", hashes)
    for split in SPLITS:
        vp, lp = root / f"{split}_volumes.npy", root / f"{split}_labels.npy"
        labels = np.load(lp, mmap_mode="r")
        shard_names = sorted(
            name for name in files if name.startswith(f"{BM3D_PREFIX}/volumes/{split}/shard-") and name.endswith(".npy")
        )
        if not shard_names:
            raise ValueError(f"BM3D benchmark has no shards for {split}")
        shard_hashes = []
        for name in shard_names:
            lfs = files[name].lfs
            digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"BM3D shard lacks LFS SHA256: {name}")
            shard_hashes.append(digest)
        sidecar = root / f"{split}_volumes.identity.json"
        if vp.is_file() and sidecar.is_file():
            shape = list(np.load(vp, mmap_mode="r").shape)
            identity = _bm3d_identity(shard_hashes, hashes[lp.name], shape)
            saved = json.loads(sidecar.read_text())
            if saved.get("identity") == identity:
                _validate_bm3d_split(split, np.load(vp, mmap_mode="r"), labels, res)
                entries[split] = {"source": str(vp), "labels": str(lp), "identity": identity}
                hashes[vp.name] = saved["volumes_sha256"]
                print(f"[data] bm3d/{split}: resumable consolidated file verified", flush=True)
                continue
        if root_attached:
            if not vp.is_file():
                raise FileNotFoundError(f"Attached BM3D root requires pre-consolidated {vp.name}")
            volumes = np.load(vp, mmap_mode="r")
            _validate_bm3d_split(split, volumes, labels, res)
            identity = _bm3d_identity(shard_hashes, hashes[lp.name], list(volumes.shape))
            hashes[vp.name] = identity["volumes"]
            entries[split] = {"source": str(vp), "labels": str(lp), "identity": identity}
            print(f"[data] bm3d/{split}: attached pre-consolidated file verified (never written)", flush=True)
            continue
        require_disk(root, sum(files[name].size for name in shard_names) + 2 * 1024**3)
        root.mkdir(parents=True, exist_ok=True)
        total = int(labels.shape[0])
        temporary = root / f".{split}_volumes.partial.npy"
        writer, trailing, offset = None, None, 0
        for name, digest in zip(shard_names, shard_hashes):
            local = root / name
            if not local.is_file():
                snapshot_download(
                    repo_id=BM3D_REPO,
                    repo_type="dataset",
                    token=token,
                    local_dir=str(root),
                    allow_patterns=[name],
                )
            print(f"[data] hash start bm3d/{split}/{Path(name).name}", flush=True)
            actual = cd._verify_file(local, files[name])
            if actual != digest:
                raise ValueError(f"BM3D shard checksum mismatch: {name}")
            print(f"[data] hash done bm3d/{split}/{Path(name).name}: {actual}", flush=True)
            shard = np.load(local, mmap_mode="r")
            if writer is None:
                trailing = tuple(shard.shape[1:])
                require_disk(root, total * int(np.prod(trailing)) + files[name].size + 2 * 1024**2)
                writer = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=(total, *trailing))
            elif tuple(shard.shape[1:]) != trailing:
                raise ValueError(f"BM3D shards for {split} have inconsistent trailing shapes")
            count = int(shard.shape[0])
            if offset + count > total:
                raise ValueError(f"BM3D shards for {split} exceed label count {total}")
            writer[offset : offset + count] = shard
            offset += count
            del shard
            local.unlink(missing_ok=True)
        if writer is None or offset != total:
            raise ValueError(f"BM3D shards for {split} produced {offset} volumes, labels expect {total}")
        writer.flush()
        del writer
        os.replace(temporary, vp)
        volumes = np.load(vp, mmap_mode="r")
        _validate_bm3d_split(split, volumes, labels, res)
        identity = _bm3d_identity(shard_hashes, hashes[lp.name], list(volumes.shape))
        saved = {"identity": identity, "volumes_sha256": sha256(vp)}
        write_json(sidecar, saved)
        hashes[vp.name] = saved["volumes_sha256"]
        entries[split] = {"source": str(vp), "labels": str(lp), "identity": identity}
        print(f"[data] bm3d/{split}: consolidated {total} volumes from {len(shard_names)} shards", flush=True)
    print("[data] Verified bm3d sources (read-only)", flush=True)
    return {"kind": "bm3d", "entries": entries, "manifest": None, "hashes": hashes}


def verify_bilateral(raw, bilateral, *, smoke=False):
    ks.require_ready(smoke=smoke)
    if not smoke:
        manifest = bilateral["manifest"]
        identity = manifest.get("identity", {})
        if (
            manifest.get("complete") is not True
            or identity.get("source_repo") != RAW_REPO
            or identity.get("source_revision") != RAW_REVISION
            or identity.get("resolution") != 200
        ):
            raise ValueError("Bilateral manifest source identity mismatch")
    for split in SPLITS:
        raw_entry, bilateral_entry = raw["entries"][split], bilateral["entries"][split]
        ri = raw_entry.get("source_identity", raw_entry["identity"])
        bi = bilateral_entry.get("source_identity", bilateral_entry["identity"])
        if ri["labels"] != bi["labels"] or ri["shape"][0] != bi["shape"][0] or ri["shape"][-3:] != bi["shape"][-3:]:
            raise ValueError("Raw/bilateral split or labels mismatch")
        if not smoke:
            source = identity.get("sources", {}).get(split, {})
            marker = manifest.get("splits", {}).get(split, {})
            marker_path = Path(bilateral["entries"][split]["source"]).with_name(f"{split}_complete.json")
            if (
                source.get("repo_revision") != RAW_REVISION
                or source.get("sha256", ri["volumes"]) != ri["volumes"]
                or source.get("labels_sha256") != ri["labels"]
                or source.get("shape") != ri["shape"]
                or marker.get("complete") is not True
                or marker.get("identity") != identity
                or marker.get("sha256", bi["volumes"]) != bi["volumes"]
                or json.loads(marker_path.read_text()) != marker
            ):
                raise ValueError(f"Bilateral split provenance mismatch: {split}")
    print("[data] D2 bilateral provenance and raw labels match; no denoising performed")


def prepare_views(cfg, source):
    ks.require_ready(smoke=cfg["smoke"])
    with run_lock(Path(cfg["temp_root"]) / f"views-{source['kind']}.lock"):
        return _prepare_views(cfg, source)


def projection_signature(res2d):
    code = "\n".join(
        inspect.getsource(fn)
        for fn in (fm.depth_axis, fm.to_depth_last, fm.project_views, ft.volume_sample, _prepare_views)
    )
    return {
        "res2d": res2d,
        "views": list(fm.VIEWS),
        "half": 16,
        "resize": "bilinear-align_corners=False-clip-uint8",
        "version": 1,
        "code": hashlib.sha256(code.replace("\r\n", "\n").encode()).hexdigest(),
        "numpy": np.__version__,
        "torch": str(torch.__version__).split("+")[0],
    }


def _prepare_views(cfg, source):
    res2d = 8 if cfg["smoke"] else 224
    for split, entry in source["entries"].items():
        entry.setdefault("source_identity", entry["identity"])
        identity = {**entry["source_identity"], **projection_signature(res2d)}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache = Path(cfg["temp_root"]) / "views" / key
        vp, dp, mp = cache / "views.npy", cache / "dz.npy", cache / "metadata.json"
        if mp.exists():
            saved = json.loads(mp.read_text())
            if saved["identity"] != identity or saved["views"] != sha256(vp) or saved["dz"] != sha256(dp):
                raise ValueError("Corrupt or incompatible view cache")
        else:
            volumes = np.load(entry["source"], mmap_mode="r")
            require_disk(cache, len(volumes) * 2 * res2d**2 + 256 * 1024**2)
            cache.mkdir(parents=True, exist_ok=True)
            temporary = cache / "views.partial.npy"
            views = np.lib.format.open_memmap(
                temporary, mode="w+", dtype=np.uint8, shape=(len(volumes), 2, res2d, res2d)
            )
            dzs = np.zeros(len(volumes), dtype=np.int8)
            last_progress = time.monotonic()
            print(f"[views] start {source['kind']}/{split}: {len(volumes)} volumes", flush=True)
            for index in range(len(volumes)):
                raw = ft.volume_sample(volumes, index)
                dzs[index] = fm.depth_axis(raw)
                projected = fm.project_views(fm.to_depth_last(raw, int(dzs[index])))
                for view in range(2):
                    tensor = torch.from_numpy(np.asarray(projected[view], dtype=np.float32).copy())[None, None]
                    views[index, view] = (
                        F.interpolate(tensor, (res2d, res2d), mode="bilinear", align_corners=False)[0, 0]
                        .numpy()
                        .clip(0, 255)
                        .astype(np.uint8)
                    )
                if (index + 1) % 200 == 0 or time.monotonic() - last_progress >= 10 or index + 1 == len(volumes):
                    print(f"[views] {source['kind']}/{split}: {index + 1}/{len(volumes)}", flush=True)
                    last_progress = time.monotonic()
            views.flush()
            del views
            os.replace(temporary, vp)
            np.save(dp, dzs)
            saved = {"identity": identity, "views": sha256(vp), "dz": sha256(dp)}
            write_json(mp, saved)
        entry.update(views=str(vp), dz=str(dp), identity=saved)
        print(f"[views] {source['kind']}/{split}: verified cache {key[:12]}")
    return source


class KaggleDataset(ct.ControlsDataset):
    def __init__(self, entry, *, seed, train, use_3d, smoke=False):
        self.source, self.label_path = Path(entry["source"]), Path(entry["labels"])
        self.use_3d = use_3d
        self.volumes = np.load(self.source, mmap_mode="r") if use_3d else None
        self.views = np.load(entry["views"], mmap_mode="r")
        self.dzs = np.load(entry["dz"], mmap_mode="r")
        self.labels = np.load(self.label_path, mmap_mode="r")
        self.res3d, self.seed, self.train, self.epoch = 8 if smoke else 96, seed, train, 0


def make_datasets(prepared, spec, seed, *, smoke=False):
    ks.require_ready(smoke=smoke)
    entries = prepared[spec.get("dataset", "raw")]["entries"]
    return [
        KaggleDataset(entries[s], seed=seed, train=s == "Training", use_3d=spec["use_3d"], smoke=smoke) for s in SPLITS
    ]
