"""Read-only pinned sources and writable, content-addressed Kaggle view caches."""

import contextlib
import hashlib
import inspect
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from scripts import controls_data as cd
from scripts import controls_training as ct
from scripts import final_model as fm
from scripts import final_training as ft

RAW_REPO = "tqhuyen/harvard-oct-glaucoma-200"
RAW_REVISION = "939a38876b7b9313162842ef2d44b7edc2b57020"
SPLITS = ("Training", "Validation", "Test")


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
    with run_lock(Path(cfg["temp_root"]) / f"source-{kind}.lock"):
        return _prepare_source(cfg, kind)


def _prepare_source(cfg, kind):
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
        hashes = {name: cd._verify_file(root / name, files[name]) for name in names}
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


def verify_bilateral(raw, bilateral, *, smoke=False):
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
    entries = prepared["bilateral" if spec["code"] == "B4" else "raw"]["entries"]
    return [
        KaggleDataset(entries[s], seed=seed, train=s == "Training", use_3d=spec["use_3d"], smoke=smoke) for s in SPLITS
    ]
