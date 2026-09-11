"""Portable, verified preparation for final-training denoised inputs.

Pass a caller-verified Drive data_cache/bilateral directory, not a run directory.
No raw volumes are uploaded. Missing caches require an explicit
ALLOW_BUILD_DENOISED -> allow_build=True decision, preferably on a CPU host.
Existing local legacy completion markers may be adopted only when their source
identity and effective filter parameters match; unknown unmarked files are never
adopted or erased. Hash sidecars trust unchanged path/size/mtime, as ft does.
Publication is opt-in. Producer package/build versions are provenance, not
restore compatibility. Version-1 manifests without a filter signature cannot
prove callback identity and must be re-imported from a verified CPU export.
This module does not manage credentials, W&B sessions, or training artifacts.
"""

import hashlib
import inspect
import json
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import torch

from scripts import final_model as fm
from scripts import final_training as ft


def _code_hash(source):
    return hashlib.sha256(source.replace("\r\n", "\n").replace("\r", "\n").encode()).hexdigest()


_VERSIONS = {
    "manifest": 2,
    "denoise_marker": 2,
    "views_marker": 1,
    "code_sha256": _code_hash(
        "\n".join(
            inspect.getsource(fn)
            for fn in (
                ft.build_denoised,
                ft.build_views,
                ft.volume_sample,
                fm.depth_axis,
                fm.to_depth_last,
                fm.project_views,
            )
        )
    ),
    "numpy": np.__version__,
    "torch": str(torch.__version__).split("+")[0],
}

_CPU_PARAMS = {
    "sigma_color": 0.10,
    "sigma_spatial": 4.0,
    "bins": 10000,
    "mode": "constant",
    "cval": 0,
    "channel_axis": None,
    "win_size": None,
}
_CPU_IMPLEMENTATION = "skimage.restoration.denoise_bilateral"
_CPU_QUANTIZATION = "float32 /255 -> filter -> clip *255 -> uint8 truncation"
_CPU_FILTER_SOURCE = """def bilateral(volume):
    import numpy as np
    from skimage.restoration import denoise_bilateral

    if volume.dtype != np.uint8 or volume.ndim != 3:
        raise ValueError("Expected a uint8 3D volume")
    output = np.empty_like(volume)
    for index, plane in enumerate(volume):
        image = plane.astype(np.float32) / 255.0
        filtered = denoise_bilateral(image, **PARAMS)
        output[index] = np.clip(np.nan_to_num(filtered, nan=image) * 255.0, 0, 255).astype(np.uint8)
    return output
"""
_CPU_FILTER_HASH = _code_hash(_CPU_FILTER_SOURCE)
_VIEW_SIGNATURE = {
    "code_sha256": _code_hash(
        "\n".join(inspect.getsource(fn) for fn in (fm.depth_axis, fm.to_depth_last, fm.project_views))
    ),
    "builder_sha256": _code_hash(inspect.getsource(ft.build_views) + inspect.getsource(ft.volume_sample)),
    "resize": "bilinear-align_corners=False-clip-uint8",
    "n2d": 2,
}


def _config(res2d, method, params, implementation, denoise_fn=None, filter_code_sha256=None):
    if not isinstance(res2d, int) or res2d <= 0 or not method or not implementation or not isinstance(params, dict):
        raise ValueError("Require positive res2d and explicit filter parameters/implementation")
    if implementation == _CPU_IMPLEMENTATION:
        if method != "bilateral" or params != _CPU_PARAMS:
            raise ValueError("CPU bilateral requires the exact explicit effective parameters")
        expected = _CPU_FILTER_HASH
        if filter_code_sha256 is not None and filter_code_sha256 != expected:
            raise ValueError("CPU filter code identity mismatch")
    else:
        expected = filter_code_sha256
    if denoise_fn is not None:
        actual = _code_hash(inspect.getsource(denoise_fn))
        if expected is not None and actual != expected:
            raise ValueError("Caller filter code identity mismatch")
        expected = actual
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Provide denoise_fn or an explicit expected filter_code_sha256 for restore")
    signature = {
        "method": method,
        "params": params,
        "implementation": implementation,
        "filter_code_sha256": expected,
        "res2d": res2d,
        "views": _VIEW_SIGNATURE,
    }
    return json.loads(
        json.dumps(
            {
                "method": method,
                "params": params,
                "implementation": implementation,
                "res2d": res2d,
                "n2d": 2,
                "compatibility": signature,
                "versions": _VERSIONS,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


def _compatible(saved, requested):
    if not isinstance(saved, dict) or not isinstance(saved.get("splits"), dict):
        return False
    config = saved.get("config", {})
    if not isinstance(config, dict):
        return False
    if any(config.get(k) != requested[k] for k in ("method", "params", "implementation", "res2d", "n2d")):
        return False
    if "compatibility" in config:
        return config["compatibility"] == requested["compatibility"]
    return False


def _splits(splits):
    result = list(splits)
    if (
        not result
        or any(not isinstance(s, str) or not s or any(c in s for c in "/\\:") or s in (".", "..") for s in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError("Require safe unique splits")
    return result


def _sources(root, names):
    sources = {kind: _array_identity(root / names[kind]) for kind in ("raw", "labels")}
    shape = sources["raw"]["shape"]
    if (
        len(shape) not in (4, 5)
        or (len(shape) == 5 and shape[1] != 1)
        or any(d <= 0 for d in shape)
        or sources["raw"]["dtype"] != np.dtype("uint8").str
        or sources["labels"]["shape"] != [shape[0]]
        or np.dtype(sources["labels"]["dtype"]).kind not in "iu"
    ):
        raise ValueError(f"Invalid raw volume/label shape or dtype: {names['raw']}")
    return sources


def _candidates(data_root, cache_root, config, manifest_path):
    paths = (
        [Path(manifest_path)]
        if manifest_path is not None
        else [
            *sorted((data_root / ".final_data").glob("*.json")),
            cache_root / "manifest.json",
            *sorted(cache_root.glob("*/manifest.json")),
        ]
    )
    result = []
    for path in paths:
        saved = _json_read(path)
        if _compatible(saved, config):
            root = path.parent.parent if path.parent.name == ".final_data" else path.parent
            result.append((root, saved))
    if manifest_path is not None and not result:
        raise ValueError("Selected manifest is missing, malformed or has incompatible filter/views identity")
    return result


def _recorded(data_root, split):
    for path in (data_root / ".final_data").glob("*.json"):
        saved = _json_read(path)
        if not isinstance(saved, dict) or not isinstance(saved.get("splits"), dict) or split in saved["splits"]:
            return True
    return False


def _json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_read(path):
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


def sync_verified(source, destination):
    """Atomically copy bytes, verifying SHA256; return True only when copied.

    Both identities use ft's cached path/size/mtime SHA256 sidecars. Destination
    metadata changes trigger rehashing, not an upload when the bytes still match.
    A failed copy leaves the previous destination intact and can be retried.
    """
    source, destination = Path(source), Path(destination)
    before = ft.file_identity(source)
    expected = ft.data_identity(source)
    if destination.is_file() and ft.data_identity(destination) == expected:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp-" + uuid.uuid4().hex)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        digest = hashlib.sha256()
        with temporary.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if {"sha256": digest.hexdigest(), "size": temporary.stat().st_size} != expected or ft.file_identity(
            source
        ) != before:
            raise RuntimeError(f"Copy verification failed: {source}")
        os.replace(temporary, destination)
        ft.atomic_save(
            {"file": ft.file_identity(destination), "content": expected}, destination.with_suffix(".sha256.pt")
        )
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _array_identity(path):
    before = ft.file_identity(path)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    result = {**ft.data_identity(path), "shape": list(array.shape), "dtype": array.dtype.str}
    del array
    if before != ft.file_identity(path):
        raise RuntimeError(f"Data changed during verification: {path}")
    return result


def _names(split, res2d):
    return {
        "raw": f"{split}_volumes.npy",
        "labels": f"{split}_labels.npy",
        "denoised": f"{split}_volumes_dn.npy",
        "views": f"{split}_volumes_dn_views_{res2d}.npy",
        "dzs": f"{split}_volumes_dn_dzs_{res2d}.npy",
    }


def _verified(root, names, entry, sources, res2d):
    try:
        if not isinstance(entry, dict) or entry.get("sources") != sources:
            return False
        files = entry["files"]
        if set(files) != {"denoised", "labels", "views", "dzs"}:
            return False
        for kind, expected in files.items():
            if _array_identity(root / names[kind]) != expected:
                return False
        count = sources["raw"]["shape"][0]
        return (
            files["labels"] == sources["labels"]
            and files["denoised"]["shape"] == sources["raw"]["shape"]
            and files["denoised"]["dtype"] == sources["raw"]["dtype"]
            and files["views"]["shape"] == [count, 2, res2d, res2d]
            and files["views"]["dtype"] == np.dtype("uint8").str
            and files["dzs"]["shape"] == [count]
            and files["dzs"]["dtype"] == np.dtype("int8").str
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _markers(root, names, config, shape):
    return {
        (root / names["denoised"]).with_suffix(".complete.pt"): {
            "source": ft.file_identity(root / names["raw"]),
            "method": config["method"],
            "params": config["params"],
            "implementation": config["implementation"],
            "shape": shape,
            "version": 2,
        },
        (root / names["views"]).with_suffix(".complete.pt"): {
            "source": ft.file_identity(root / names["denoised"]),
            "res2d": config["res2d"],
            "n2d": 2,
            "version": 1,
        },
    }


def _entry(root, names, sources):
    return {
        "sources": sources,
        "files": {kind: _array_identity(root / names[kind]) for kind in ("denoised", "labels", "views", "dzs")},
    }


def _space_check(root, needed):
    while not root.exists():
        root = root.parent
    if shutil.disk_usage(root).free < needed:
        raise RuntimeError(f"Insufficient preparation space at {root}: need {needed} bytes including reserve")


def _finish(data_root, cache_root, manifest, split, entry, publish):
    config = manifest["config"]
    names = _names(split, config["res2d"])
    if _sources(data_root, names) != entry["sources"] or not _verified(
        data_root, names, entry, entry["sources"], config["res2d"]
    ):
        raise RuntimeError(f"Prepared data verification failed: {split}")
    for path, identity in _markers(data_root, names, config, entry["sources"]["raw"]["shape"]).items():
        ft.atomic_save(identity, path)
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest["splits"][split] = entry
    _json_write(data_root / ".final_data" / f"{key}.json", manifest)
    if publish:
        remote = cache_root / key
        for kind in entry["files"]:
            sync_verified(data_root / names[kind], remote / names[kind])
        if not _verified(remote, names, entry, entry["sources"], config["res2d"]):
            raise RuntimeError(f"Remote verification failed: {split}; local cache retained")
        _json_write(remote / "manifest.json", manifest)


def prepare_data(
    data_root,
    cache_root,
    splits,
    *,
    res2d,
    denoise_fn,
    method,
    params,
    implementation,
    allow_build=False,
    limit=0,
    smoke=False,
    manifest_path=None,
    publish_cache=False,
    filter_code_sha256=None,
):
    """Return a complete portable manifest, or raise without raw fallback.

    Splits are filename prefixes (e.g. Training, Validation, Test). Raw volumes
    and labels must already exist locally. A producer-specific cache
    subdirectory contains manifest.json and derived arrays only. Local manifests
    live under .final_data. Publication requires publish_cache=True, independently
    of allow_build and smoke. Completed splits are then published incrementally; limited
    denoising raises a 'pending' RuntimeError and retains ft's resumable files.
    Smoke mode only reduces the disk reserve; it never bypasses verification.
    Auto-selection searches local/cache manifests by numerical compatibility,
    not consumer package versions. Different verified outputs are ambiguous;
    select manifest_path explicitly. Producer provenance is retained on restore.
    Arbitrary filters require denoise_fn or its normalized filter_code_sha256;
    implementation must identify/version any transitive filter semantics not
    present in that callable's source. An explicit manifest never bypasses this.
    The known CPU export implementation can be restored with denoise_fn=None.
    """
    data_root, cache_root = Path(data_root), Path(cache_root)
    if not all(isinstance(value, bool) for value in (allow_build, publish_cache, smoke)):
        raise ValueError("allow_build, publish_cache and smoke must be explicit booleans")
    splits = _splits(splits)
    if not isinstance(limit, int) or limit < 0:
        raise ValueError("Require nonnegative limit")
    config = _config(res2d, method, params, implementation, denoise_fn, filter_code_sha256)
    candidates = _candidates(data_root, cache_root, config, manifest_path)
    manifest = {"config": config, "splits": {}}
    for _, saved in reversed(candidates):
        if saved.get("config") == config:
            manifest["splits"].update(saved["splits"])
    plan = []
    estimated = 0
    scratch = 0
    for split in splits:
        names = _names(split, res2d)
        sources = _sources(data_root, names)
        shape = sources["raw"]["shape"]
        chosen, entry = None, None
        matches = {}
        for root, saved in candidates:
            candidate = saved["splits"].get(split)
            if _verified(root, names, candidate, sources, res2d):
                content = json.dumps({k: candidate[k] for k in ("sources", "files")}, sort_keys=True)
                matches.setdefault(content, (root, candidate, saved))
        if len(matches) > 1:
            raise RuntimeError(f"Ambiguous compatible cache for {split}; choose manifest_path explicitly")
        if matches:
            chosen, entry, saved = next(iter(matches.values()))
            if not plan:
                manifest = json.loads(json.dumps(saved))
        if manifest_path is not None and chosen is None:
            raise RuntimeError(f"Selected manifest has no valid complete cache for {split}; refusing fallback/build")
        if chosen is None and not _recorded(data_root, split) and (data_root / names["denoised"]).exists():
            try:
                markers = _markers(data_root, names, config, shape)
                if all(p.exists() and torch.load(p, weights_only=False) == value for p, value in markers.items()):
                    candidate = _entry(data_root, names, sources)
                    if _verified(data_root, names, candidate, sources, res2d):
                        chosen, entry = data_root, candidate
            except (OSError, ValueError, RuntimeError, EOFError):
                pass
        if chosen is None:
            if not allow_build:
                raise RuntimeError(
                    f"No valid complete cache for {split}; prepare on CPU or explicitly set ALLOW_BUILD_DENOISED "
                    "(allow_build=True). Partial data is pending, never a raw fallback."
                )
            if denoise_fn is None:
                raise ValueError("A denoise_fn is required for explicit builds")
            target = data_root / names["denoised"]
            pending = target.with_suffix(".pending.json")
            expected = {"sources": sources, "config": config}
            if pending.exists() and _json_read(pending) != expected:
                raise RuntimeError(f"Partial content/parameter identity mismatch: {target}")
            if target.exists():
                ready = _json_read(target.with_suffix(".denoised.json"))
                if not pending.exists() or ready != {**expected, "denoised": _array_identity(target)}:
                    raise RuntimeError(
                        f"Unverified or mismatched denoised file retained: {target}; use a fresh data directory"
                    )
            if target.with_suffix(".partial.npy").exists() and not pending.exists():
                raise RuntimeError(f"Unknown partial file retained: {target}")
        output_size = sources["raw"]["size"] + sources["labels"]["size"] + shape[0] * (2 * res2d**2 + 1) + 1024
        if chosen != data_root:
            estimated += output_size
        scratch = max(scratch, int(np.prod(shape[1:])) * 16)
        plan.append((split, names, sources, chosen, entry, output_size))
    reserve = 0 if smoke else min(5 * 1024**3, max(1024**2, sum(item[-1] for item in plan) // 4))
    _space_check(data_root, estimated + scratch + reserve)
    _space_check(Path(tempfile.gettempdir()), scratch + reserve)
    if publish_cache:
        remote = cache_root / hashlib.sha256(json.dumps(manifest["config"], sort_keys=True).encode()).hexdigest()
        upload_size = sum(
            size for _, names, sources, _, entry, size in plan if not _verified(remote, names, entry, sources, res2d)
        )
        _space_check(cache_root, upload_size + reserve)
    for split, names, sources, chosen, entry, _ in plan:
        target = data_root / names["denoised"]
        if chosen is None:
            _json_write(target.with_suffix(".pending.json"), {"sources": sources, "config": config})
            if not target.exists():
                complete = ft.build_denoised(
                    data_root / names["raw"],
                    target,
                    denoise_fn,
                    method=method,
                    params=config["params"],
                    implementation=implementation,
                    limit=limit,
                )
                if not complete:
                    raise RuntimeError(
                        f"Denoise pending for {split}; partial files retained; rerun with allow_build=True"
                    )
                _json_write(
                    target.with_suffix(".denoised.json"),
                    {"sources": sources, "config": config, "denoised": _array_identity(target)},
                )
            ft.build_views(target, res2d=res2d, n2d=2)
            entry = _entry(data_root, names, sources)
            entry["provenance"] = {"versions": config["versions"]}
        elif chosen != data_root:
            for kind in entry["files"]:
                sync_verified(chosen / names[kind], data_root / names[kind])
        _finish(data_root, cache_root, manifest, split, entry, publish_cache)
        target.with_suffix(".pending.json").unlink(missing_ok=True)
    return manifest


def import_cpu_export(
    data_root,
    export_root,
    cache_root,
    splits,
    res2d,
    method,
    params,
    implementation,
    smoke=False,
    allow_publish=False,
    *,
    source_manifest_path=None,
):
    """Import prepare_bilateral_200 output using CPU-only view construction.

    Never calls a denoiser or initializes CUDA. All requested exports are verified
    before copies/views. Only missing local views are built. Publication requires
    allow_publish=True, even in smoke mode; no raw volumes are published.

    The exporter has no raw-volume digest. Therefore source_manifest_path (default
    data_root/source_manifest.json) must be a trusted raw-download provenance file:
    {"source_repo": "owner/repo", "source_revision": "<40 hex commit>",
     "sources": {"Training": {"sha256": "<raw NPY SHA256>",
                              "labels_sha256": "<labels NPY SHA256>"}}}.
    Its revision must be the export's pinned revision. No network lookups or
    inferred revisions are used. Smoke permits tiny spatial shapes for tests,
    but does not weaken source, filter, checksum or publication checks.
    """
    data_root, export_root, cache_root = Path(data_root), Path(export_root), Path(cache_root)
    if not isinstance(allow_publish, bool) or not isinstance(smoke, bool):
        raise ValueError("allow_publish and smoke must be explicit booleans")
    splits = _splits(splits)
    config = _config(res2d, method, params, implementation)
    if implementation != _CPU_IMPLEMENTATION:
        raise ValueError("Unsupported CPU export implementation")
    exported = _json_read(export_root / "manifest.json")
    raw_manifest = _json_read(
        Path(source_manifest_path) if source_manifest_path else data_root / "source_manifest.json"
    )
    if not isinstance(exported, dict) or exported.get("complete") is not True:
        raise ValueError("CPU export requires a complete manifest.json")
    identity = exported.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("CPU export identity missing")
    revision = identity.get("source_revision")
    accepted_hashes = {
        _CPU_FILTER_HASH,
        hashlib.sha256(_CPU_FILTER_SOURCE.replace("\n", "\r\n").encode()).hexdigest(),
    }
    if (
        identity.get("format") != 1
        or identity.get("method") != method
        or identity.get("params") != params
        or identity.get("implementation") != implementation
        or identity.get("axis") != 0
        or identity.get("output_dtype") != "uint8"
        or identity.get("resolution") != 200
        or identity.get("quantization") != _CPU_QUANTIZATION
        or identity.get("filter_code_sha256") not in accepted_hashes
        or not isinstance(revision, str)
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
        or not identity.get("skimage")
        or not identity.get("numpy")
        or not isinstance(identity.get("sources"), dict)
        or not isinstance(exported.get("splits"), dict)
        or not isinstance(identity.get("source_repo"), str)
    ):
        raise ValueError("CPU export filter semantics, code, defaults or pinned revision mismatch")
    if (
        not isinstance(raw_manifest, dict)
        or not isinstance(raw_manifest.get("sources"), dict)
        or raw_manifest.get("source_repo") != identity.get("source_repo")
        or raw_manifest.get("source_revision") != revision
    ):
        raise ValueError("Trusted raw source manifest with matching repo/revision and hashes is required")
    try:
        declared_total = sum(source["shape"][0] for source in identity["sources"].values())
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("CPU export source counts are malformed") from exc
    if exported.get("total_volumes") != declared_total or set(exported["splits"]) != set(identity["sources"]):
        raise ValueError("CPU export split/total completion metadata mismatch")
    candidates = _candidates(data_root, cache_root, config, None)
    manifest = {"config": config, "splits": {}}
    for root, saved in candidates:
        if root == data_root:
            manifest = json.loads(json.dumps(saved))
            break
    plan = []
    for split in splits:
        names = _names(split, res2d)
        sources = _sources(data_root, names)
        raw = sources["raw"]
        source = identity["sources"].get(split)
        trusted = raw_manifest["sources"].get(split)
        marker = _json_read(export_root / f"{split}_complete.json")
        if not isinstance(source, dict) or not isinstance(trusted, dict) or not isinstance(marker, dict):
            raise ValueError(f"CPU export missing source/completion metadata: {split}")
        url = urlparse(source.get("url", ""))
        expected_url_path = f"/datasets/{identity['source_repo']}/resolve/{revision}/{names['raw']}"
        if (
            trusted.get("sha256") != raw["sha256"]
            or trusted.get("labels_sha256") != sources["labels"]["sha256"]
            or source.get("labels_sha256") != sources["labels"]["sha256"]
            or source.get("sha256", raw["sha256"]) != raw["sha256"]
            or source.get("repo_revision") != revision
            or source.get("filename") != names["raw"]
            or source.get("shape") != raw["shape"]
            or source.get("dtype") != "uint8"
            or source.get("size") != raw["size"]
            or source.get("sample_bytes") != int(np.prod(raw["shape"][-3:]))
            or source.get("offset") != raw["size"] - int(np.prod(raw["shape"]))
            or url.scheme != "https"
            or url.netloc != "huggingface.co"
            or unquote(url.path) != expected_url_path
            or (not smoke and raw["shape"][-3:] != [200, 200, 200])
        ):
            raise ValueError(f"CPU export raw source hashes/revision/header mismatch: {split}")
        labels = np.load(data_root / names["labels"], mmap_mode="r", allow_pickle=False)
        counts = {str(i): int(np.count_nonzero(labels == i)) for i in (0, 1)}
        del labels
        if sum(counts.values()) != raw["shape"][0] or source.get("class_counts") != counts:
            raise ValueError(f"CPU export label counts mismatch: {split}")
        if (
            marker != exported["splits"].get(split)
            or marker.get("identity") != identity
            or marker.get("complete") is not True
            or marker.get("file") != names["denoised"]
            or marker.get("shape") != raw["shape"]
            or marker.get("dtype") != "uint8"
        ):
            raise ValueError(f"CPU export completion identity mismatch: {split}")
        output = _array_identity(export_root / names["denoised"])
        if (
            output["sha256"] != marker.get("sha256")
            or output["shape"] != raw["shape"]
            or output["dtype"] != raw["dtype"]
            or _array_identity(export_root / names["labels"]) != sources["labels"]
        ):
            raise ValueError(f"CPU export output/labels checksum or shape mismatch: {split}")
        target = data_root / names["denoised"]
        if target.exists() and _array_identity(target) != output:
            raise ValueError(f"Different local denoised data retained: {split}")
        existing = None
        for root, saved in candidates:
            candidate = saved["splits"].get(split)
            if root == data_root and _verified(root, names, candidate, sources, res2d):
                if candidate["files"]["denoised"] == output:
                    existing = candidate
                    if not plan:
                        manifest = json.loads(json.dumps(saved))
                    break
        plan.append((split, names, sources, output, existing))
    total = sum(s["raw"]["size"] + s["raw"]["shape"][0] * (2 * res2d**2 + 1) for _, _, s, _, _ in plan)
    needed = sum(
        (output["size"] if not (data_root / names["denoised"]).exists() else 0)
        + (s["raw"]["shape"][0] * (2 * res2d**2 + 1) + 512 if existing is None else 0)
        for _, names, s, output, existing in plan
    )
    reserve = 0 if smoke else min(5 * 1024**3, max(1024**2, total // 4))
    _space_check(data_root, needed + reserve)
    _space_check(
        Path(tempfile.gettempdir()), max(int(np.prod(s["raw"]["shape"][1:])) * 16 for _, _, s, _, _ in plan) + reserve
    )
    if allow_publish:
        remote = cache_root / hashlib.sha256(json.dumps(manifest["config"], sort_keys=True).encode()).hexdigest()
        upload_size = sum(
            s["raw"]["size"] + s["labels"]["size"] + s["raw"]["shape"][0] * (2 * res2d**2 + 1) + 512
            for _, names, s, _, existing in plan
            if not _verified(remote, names, existing, s, res2d)
        )
        _space_check(cache_root, upload_size + reserve)
    for split, names, sources, output, existing in plan:
        target = data_root / names["denoised"]
        sync_verified(export_root / names["denoised"], target)
        if _array_identity(target) != output:
            raise RuntimeError(f"CPU export changed during import: {split}")
        if existing is None:
            views, dzs = data_root / names["views"], data_root / names["dzs"]
            if views.exists() and dzs.exists():
                marker = views.with_suffix(".complete.pt")
                expected = _markers(data_root, names, config, sources["raw"]["shape"])[marker]
                if (
                    _recorded(data_root, split)
                    or not marker.exists()
                    or torch.load(marker, weights_only=False) != expected
                ):
                    raise ValueError(f"Unverified existing views retained: {split}")
            else:
                ft.build_views(target, res2d=res2d, n2d=2)
        entry = _entry(data_root, names, sources)
        views_versions = config["versions"]
        if existing is not None:
            provenance = existing.get("provenance", {})
            views_versions = provenance.get(
                "views_versions", provenance.get("versions", manifest["config"]["versions"])
            )
        entry["provenance"] = {
            "cpu_export": identity,
            "raw_download": {
                "source_repo": raw_manifest["source_repo"],
                "source_revision": revision,
                "sources": {split: {"sha256": sources["raw"]["sha256"], "labels_sha256": sources["labels"]["sha256"]}},
            },
            "views_versions": views_versions,
        }
        _finish(data_root, cache_root, manifest, split, entry, allow_publish)
    return manifest
