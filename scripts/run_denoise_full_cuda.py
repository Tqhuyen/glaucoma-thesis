"""Resumable full-resolution CUDA benchmark; public release explicitly authorized by the owner."""

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "tqhuyen/harvard-oct-glaucoma-200"
REVISION = "939a38876b7b9313162842ef2d44b7edc2b57020"
DESTINATION = "tqhuyen/harvard-gf-denoise-benchmark-v2"
SPLITS = {"Training": (2100, 1017, 1083), "Validation": (300, 124, 176), "Test": (900, 411, 489)}
METHODS = ("bilateral", "bm3d", "gaussian", "median", "tv", "wavelet", "nlm")
CASES = (0, 1, 25)
SHARD_SIZE = 128
RESERVE = 10 * 1024**3


class FatalJobError(RuntimeError):
    pass


class StorageQuotaError(FatalJobError):
    pass


def storage_quota_error(exc):
    response = getattr(exc, "response", None)
    message = str(exc)
    if response is not None:
        try:
            body = response.json()
            if isinstance(body, dict):
                message += " " + str(body.get("error", ""))
        except ValueError:
            pass
    return "storage limit reached" in message.lower() or "storage quota exceeded" in message.lower()


class GPUBusyError(RuntimeError):
    pass


def log_online(run, values):
    try:
        run.log(values)
    except Exception:
        raise FatalJobError("Mandatory W&B logging failed") from None


def replace_retry(source, target, attempts=30, delay=0.1):
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(delay)


def atomic_json(path, value, *, retries=3):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    for attempt in range(retries):
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            replace_retry(temporary, path)
            return
        except PermissionError:
            if attempt + 1 == retries:
                raise
            time.sleep(0.5)
        finally:
            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path, algorithm="sha256", git=False):
    result = hashlib.new(algorithm)
    if git:
        result.update(f"blob {Path(path).stat().st_size}\0".encode())
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def disk_gate(path, required=0):
    if shutil.disk_usage(path).free < RESERVE + required:
        raise OSError("Insufficient disk: must retain 10 GiB reserve")


def download_verified(session, url, token, target, size, expected, *, git=False, attempts=6, progress=None):
    target = Path(target)
    algorithm = "sha1" if git else "sha256"
    if target.exists():
        if target.stat().st_size == size and digest(target, algorithm, git) == expected:
            return
        raise ValueError("Existing raw cache failed pinned source hash; preserved, refusing overwrite")
    partial = target.with_name(target.name + ".partial")
    for attempt in range(attempts):
        start = partial.stat().st_size if partial.exists() else 0
        if progress:
            progress(start, size)
        if start > size:
            raise ValueError("Oversized partial raw cache; preserved for inspection")
        disk_gate(target.parent, size - start)
        try:
            if start < size:
                headers = {"Authorization": f"Bearer {token}", "Accept-Encoding": "identity"}
                headers["Range"] = f"bytes={start}-{size - 1}"
                with session.get(url, headers=headers, stream=True, timeout=(30, 60)) as response:
                    if not (response.status_code == 200 and start == 0):
                        if response.status_code != 206 or response.headers.get("Content-Range") != (
                            f"bytes {start}-{size - 1}/{size}"
                        ):
                            raise ValueError("Strict Content-Range validation failed")
                    with partial.open("ab") as stream:
                        received = start
                        last_report = time.monotonic()
                        for block in response.iter_content(8 * 1024**2):
                            received += len(block)
                            if received > size:
                                raise ValueError("Oversized HTTP body")
                            stream.write(block)
                            if progress and time.monotonic() - last_report >= 5:
                                progress(received, size)
                                last_report = time.monotonic()
                        stream.flush()
                        os.fsync(stream.fileno())
            if partial.stat().st_size != size:
                raise OSError("Incomplete HTTP body")
            if digest(partial, algorithm, git) != expected:
                raise ValueError("Pinned raw source hash mismatch")
            os.replace(partial, target)
            if progress:
                progress(size, size)
            return
        except Exception:
            if attempt + 1 == attempts:
                raise RuntimeError("Authenticated source download/verification failed; partial preserved") from None
            time.sleep(min(60, 2**attempt))


class Archive:
    def __init__(self, token):
        import requests
        from huggingface_hub import HfApi
        from huggingface_hub.errors import RepositoryNotFoundError

        self.token, self.session = token, requests.Session()
        self.api = HfApi(token=token)
        self._index = None
        self._visibility_at = 0.0
        self.api.whoami()
        try:
            self.api.dataset_info(DESTINATION)
        except RepositoryNotFoundError:
            self.api.create_repo(DESTINATION, repo_type="dataset", private=False, exist_ok=False)
        self.ensure_visibility()

    def ensure_visibility(self, ttl=60):
        if time.monotonic() - self._visibility_at < ttl:
            return
        if self.privacy():
            self.api.update_repo_settings(DESTINATION, repo_type="dataset", private=False)
            time.sleep(5)
        if self.privacy():
            raise RuntimeError("Destination is still private; owner authorized public benchmark uploads")
        self._visibility_at = time.monotonic()

    def privacy(self):
        response = self.session.get(
            f"https://huggingface.co/api/datasets/{DESTINATION}",
            headers={"Authorization": f"Bearer {self.token}"},
            params={"t": time.time()},
            timeout=60,
        )
        response.raise_for_status()
        private = response.json().get("private")
        if not isinstance(private, bool):
            raise RuntimeError("HF visibility response is missing the private boolean")
        return private

    def remote_completed(self, prefix):
        from huggingface_hub import hf_hub_url

        def read_remote(remote, commit):
            url = hf_hub_url(DESTINATION, remote, repo_type="dataset", revision=commit)
            with self.session.get(
                url, headers={"Authorization": f"Bearer {self.token}"}, stream=True, timeout=(30, 60)
            ) as response:
                response.raise_for_status()
                body = bytearray()
                for block in response.iter_content(1024**2):
                    body.extend(block)
                    if len(body) > 8 * 1024**2:
                        raise ValueError("Oversized remote JSON report")
            receipt = {
                "remote": remote,
                "commit": commit,
                "size": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
            return json.loads(body), receipt

        for attempt in range(6):
            try:
                revision = self.api.dataset_info(DESTINATION).sha
                marker_path = f"{prefix}/_COMPLETE.json"
                if not self.api.file_exists(DESTINATION, marker_path, repo_type="dataset", revision=revision):
                    return None
                marker, marker_receipt = read_remote(marker_path, revision)
                if marker.get("complete") is not True or marker.get("volumes") != sum(s[0] for s in SPLITS.values()):
                    raise ValueError("Invalid remote completion marker")
                manifest_receipt = marker["manifest"]
                self.verify(manifest_receipt)
                manifest, actual = read_remote(manifest_receipt["remote"], manifest_receipt["commit"])
                if any(actual[k] != manifest_receipt[k] for k in actual) or manifest.get("complete") is not True:
                    raise ValueError("Remote completion manifest mismatch")
                verify_coverage(manifest["shards"])
                receipts = [*marker["reports"], marker_receipt]
                for receipt in receipts:
                    self.verify(receipt)
                for shard in manifest["shards"]:
                    for receipt in shard["artifacts"]:
                        self.verify(receipt)
                return manifest, {"artifacts": receipts, "remote_only": True}
            except Exception as exc:
                if attempt == 5:
                    raise FatalJobError("Remote completion recovery failed; refusing to recompute blindly") from None
                time.sleep(self.retry_delay(exc, attempt))

    def remote_index(self, revision, force=False):
        if self._index is None:
            self._index = {}
        if revision not in self._index or force:
            for attempt in range(10):
                try:
                    info = self.api.repo_info(DESTINATION, repo_type="dataset", revision=revision, files_metadata=True)
                    if info.sha != revision:
                        raise FatalJobError("Remote metadata revision mismatch")
                    break
                except FatalJobError:
                    raise
                except Exception as exc:
                    if attempt == 9:
                        raise FatalJobError("Pinned remote metadata unavailable; artifacts preserved") from None
                    time.sleep(self.retry_delay(exc, attempt))
            index = {}
            for sibling in info.siblings:
                expected = None
                if sibling.lfs is not None:
                    expected = {"kind": "sha256", "value": sibling.lfs.sha256}
                elif sibling.blob_id is not None:
                    expected = {"kind": "git", "value": sibling.blob_id}
                index[sibling.rfilename] = {"size": sibling.size, "expected": expected}
            self._index[revision] = index
        return self._index[revision]

    def verify(self, receipt):
        entry = self.remote_index(receipt["commit"]).get(receipt["remote"])
        if entry is not None and entry["size"] == receipt["size"] and entry["expected"] is not None:
            expected = entry["expected"]
            if expected["kind"] == "sha256":
                if expected["value"] == receipt["sha256"]:
                    return
                raise FatalJobError(f"Remote LFS SHA256 mismatch for {receipt['remote']}; no cleanup permitted")
            local = Path(receipt.get("local", ""))
            if receipt.get("git_sha1") == expected["value"]:
                return
            if (
                local.is_file()
                and local.stat().st_size == receipt["size"]
                and digest(local) == receipt["sha256"]
                and digest(local, "sha1", git=True) == expected["value"]
            ):
                return
        self.verify_stream(receipt)

    def verify_stream(self, receipt):
        from huggingface_hub import hf_hub_url

        url = hf_hub_url(DESTINATION, receipt["remote"], repo_type="dataset", revision=receipt["commit"])
        last = None
        for attempt in range(10):
            try:
                sha, count = hashlib.sha256(), 0
                with self.session.get(
                    url,
                    headers={"Authorization": f"Bearer {self.token}", "Accept-Encoding": "identity"},
                    stream=True,
                    timeout=(30, 120),
                ) as response:
                    response.raise_for_status()
                    for block in response.iter_content(8 * 1024**2):
                        sha.update(block)
                        count += len(block)
                if count != receipt["size"] or sha.hexdigest() != receipt["sha256"]:
                    raise ValueError("Remote size/SHA256 mismatch")
                return
            except Exception as exc:
                last = f"{type(exc).__name__}: {str(exc)[:200]}"
                if attempt == 9:
                    raise FatalJobError(
                        f"Pinned remote full-stream verification failed after {attempt + 1} tries "
                        f"({receipt['remote']}); no cleanup permitted. Last: {last}"
                    ) from None
                time.sleep(self.retry_delay(exc, attempt))

    @staticmethod
    def retry_delay(exc, attempt, cap=300):
        from email.utils import parsedate_to_datetime

        response = getattr(exc, "response", None)
        header = getattr(response, "headers", {}).get("Retry-After")
        if header:
            try:
                return max(0, float(header))
            except ValueError:
                try:
                    return max(0, parsedate_to_datetime(header).timestamp() - time.time())
                except (TypeError, ValueError):
                    pass
        text = str(exc)
        match = re.search(r"Retry after (\d+)", text)
        if match:
            return int(match.group(1)) + 10
        return min(cap, 2**attempt)

    def upload_many(self, items, commit_message):
        from huggingface_hub import CommitOperationAdd

        self.ensure_visibility()
        items = list(items)
        hashes = [
            {"sha256": digest(path), "git_sha1": digest(path, "sha1", git=True), "size": Path(path).stat().st_size}
            for path, _ in items
        ]
        last = None
        for attempt in range(10):
            try:
                operations = [
                    CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(path)) for path, remote in items
                ]
                info = self.api.create_commit(
                    repo_id=DESTINATION,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=commit_message,
                )
                receipts = [
                    {
                        "remote": remote,
                        **hashed,
                        "commit": info.oid,
                        "local": str(Path(path).resolve()),
                    }
                    for (path, remote), hashed in zip(items, hashes, strict=True)
                ]
                for receipt in receipts:
                    self.verify(receipt)
                return receipts
            except FatalJobError:
                raise
            except Exception as exc:
                if storage_quota_error(exc):
                    raise StorageQuotaError(
                        "Hugging Face private repository storage limit reached. "
                        "Automatic retries stopped; pending local shards are preserved."
                    ) from None
                last = f"{type(exc).__name__}: {str(exc)[:200]}"
                if attempt == 9:
                    raise FatalJobError(
                        f"Authenticated batch upload failed after {attempt + 1} tries "
                        f"({commit_message}); local artifacts preserved. Last: {last}"
                    ) from None
                time.sleep(self.retry_delay(exc, attempt))
        raise AssertionError("unreachable")

    def upload(self, path, remote):
        return self.upload_many([(path, remote)], f"upload {remote}")[0]


def cleanup_generated(path, shard_root, receipt):
    path, shard_root = Path(path).resolve(), Path(shard_root).resolve()
    if not receipt.get("verified") or str(path) != receipt.get("generated"):
        raise ValueError("Cleanup requires a durable verified generated-file receipt")
    if not path.is_relative_to(shard_root) or not re.fullmatch(r"shard-\d{5}\.npy", path.name):
        raise ValueError("Cleanup outside generated shard allowlist")
    matching = [r for r in receipt["artifacts"] if r["remote"].endswith("/" + path.name)]
    if len(matching) != 1:
        raise ValueError("Missing unique volume verification receipt")
    if path.exists():
        if path.stat().st_size != matching[0]["size"] or digest(path) != matching[0]["sha256"]:
            raise ValueError("Generated volume changed after verification; preserved")
        path.unlink()


def verify_coverage(shards):
    if {r["split"] for r in shards} != set(SPLITS):
        raise ValueError("Incomplete split coverage")
    for split, (count, _, _) in SPLITS.items():
        cursor = 0
        for shard in sorted((r for r in shards if r["split"] == split), key=lambda r: r["start"]):
            if not shard.get("verified") or shard["start"] != cursor or not cursor < shard["end"] <= count:
                raise ValueError("Invalid shard coverage")
            cursor = shard["end"]
        if cursor != count:
            raise ValueError("Incomplete volume count")
    return sum(s[0] for s in SPLITS.values())


def upload_pending(receipt_path, pending_path, archive, identity):
    pending = read_json(pending_path)
    if pending["identity"] != identity:
        raise ValueError("Pending shard identity mismatch; files preserved")
    items = []
    for artifact in pending["artifacts"]:
        path = Path(artifact["local"])
        if not path.is_file() or path.stat().st_size != artifact["size"] or digest(path) != artifact["sha256"]:
            raise ValueError("Pending shard hash mismatch; files preserved")
        items.append((path, artifact["remote"]))
    receipts = archive.upload_many(items, f"{identity['prefix']} {pending['split']} {receipt_path.stem}")
    receipt = {k: pending[k] for k in ("split", "start", "end", "generated")}
    receipt.update(verified=True, artifacts=receipts)
    atomic_json(receipt_path, receipt)
    return receipt


def resume_shard(receipt_path, archive, shard_root):
    if not receipt_path.exists():
        return False
    receipt = read_json(receipt_path)
    if not receipt.get("verified"):
        return False
    for artifact in receipt["artifacts"]:
        archive.verify(artifact)
    if receipt.get("generated"):
        cleanup_generated(receipt["generated"], shard_root, receipt)
    return True


def validate_arrays(raw, labels, split):
    count, zeros, ones = SPLITS[split]
    if raw.shape not in ((count, 200, 200, 200), (count, 1, 200, 200, 200)) or raw.dtype != np.uint8:
        raise ValueError(f"Invalid source volumes: {split}")
    if labels.shape != (count,) or not np.isin(labels, (0, 1)).all():
        raise ValueError(f"Invalid source labels: {split}")
    if (int((labels == 0).sum()), int((labels == 1).sum())) != (zeros, ones):
        raise ValueError(f"Source class counts disagree: {split}")


def load_sources(archive, directory, progress=None):
    from huggingface_hub import hf_hub_url

    directory.mkdir(parents=True, exist_ok=True)
    info = archive.api.dataset_info(SOURCE, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError("Source revision mismatch")
    remote = {f.rfilename: f for f in info.siblings}
    arrays, provenance = {}, {}
    for split in SPLITS:
        for kind in ("volumes", "labels"):
            name = f"{split}_{kind}.npy"
            file = remote[name]
            sha = file.lfs.sha256 if file.lfs else file.blob_id
            download_verified(
                archive.session,
                hf_hub_url(SOURCE, name, repo_type="dataset", revision=REVISION),
                archive.token,
                directory / name,
                file.size,
                sha,
                git=not bool(file.lfs),
                progress=(lambda done, total, name=name: progress(name, done, total)) if progress else None,
            )
            provenance[name] = {"size": file.size, "source_hash": sha, "hash_kind": "sha256" if file.lfs else "git"}
        raw = np.load(directory / f"{split}_volumes.npy", mmap_mode="r", allow_pickle=False)
        labels = np.load(directory / f"{split}_labels.npy", allow_pickle=False)
        validate_arrays(raw, labels, split)
        arrays[split] = (raw, labels)
    return arrays, provenance


def load_bilateral(directory, arrays):
    manifest = read_json(directory / "manifest.json")
    identity = manifest["identity"]
    if not manifest.get("complete") or identity["source_repo"] != SOURCE or identity["source_revision"] != REVISION:
        raise ValueError("Bilateral source provenance mismatch")
    outputs, absent = {}, []
    for split in SPLITS:
        record = manifest["splits"][split]
        path = directory / f"{split}_volumes_dn.npy"
        if path.is_file() and (not record.get("complete") or digest(path) != record["sha256"]):
            raise ValueError(f"Bilateral SHA256 mismatch: {split}; source preserved")
        if record.get("identity") != identity:
            raise ValueError("Bilateral split identity mismatch")
        if digest(directory / f"{split}_labels.npy") != identity["sources"][split]["labels_sha256"]:
            raise ValueError("Bilateral label SHA256 mismatch")
        labels = np.load(directory / f"{split}_labels.npy", allow_pickle=False)
        if not np.array_equal(labels, arrays[split][1]):
            raise ValueError("Bilateral labels differ from pinned raw labels")
        if path.is_file():
            outputs[split] = np.load(path, mmap_mode="r", allow_pickle=False)
            validate_arrays(outputs[split], labels, split)
        else:
            absent.append(split)
    reused = outputs if not absent else None
    return reused, {"mode": "reuse-existing-not-new-CUDA-filtering", "manifest": manifest}


def config_id(metadata, source, protocol):
    stable = {k: v for k, v in metadata.items() if k not in {"device", "device_id", "validation", "validated"}}
    payload = {
        "backend": stable,
        "source": source,
        "metric": protocol,
        "suite_sha256": digest(Path(__file__).with_name("denoise_cuda_suite.py")),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def _dedupe_ftz(options):
    seen = False
    deduped = []
    for option in options:
        if option in ("-ftz", "--ftz") or option.startswith(("-ftz=", "--ftz=")):
            if seen:
                continue
            seen = True
        deduped.append(option)
    return tuple(deduped)


def patch_cupy_gaussian():
    import cupy.cuda.compiler as compiler

    original_nvrtc = compiler._compile_using_nvrtc_no_warning
    if getattr(original_nvrtc, "_denoise_ftz_patched", False):
        return

    def patched_nvrtc(source, options=(), *args, **kwargs):
        return original_nvrtc(source, _dedupe_ftz(options), *args, **kwargs)

    patched_nvrtc._denoise_ftz_patched = True
    compiler._compile_using_nvrtc_no_warning = patched_nvrtc

    original_preprocess = compiler._preprocess

    def patched_preprocess(source, options, arch, backend):
        return original_preprocess(source, _dedupe_ftz(options), arch, backend)

    patched_preprocess._denoise_ftz_patched = True
    compiler._preprocess = patched_preprocess


def wait_gpu(cp, seconds):
    deadline = time.monotonic() + seconds
    while cp.cuda.runtime.memGetInfo()[0] < 1.2 * 1024**3:
        if time.monotonic() >= deadline:
            raise GPUBusyError("GPU free-memory gate timed out (requires 1.2 GiB); no CPU fallback")
        time.sleep(min(15, max(0, deadline - time.monotonic())))


def get_extra_backend(name):
    if name == "wavelet":
        from scripts.denoise_wavelet_cuda import WaveletBackend

        return WaveletBackend()
    if name == "nlm":
        from scripts.denoise_nlm_cuda import NLMBackend

        return NLMBackend()
    raise ValueError(f"Not an extra backend: {name}")


def validate_oct_planes(backend, volume, planes=8):
    from scipy.ndimage import gaussian_filter, median_filter
    from skimage.restoration import denoise_nl_means, denoise_tv_chambolle, denoise_wavelet

    raw = np.ascontiguousarray(volume.reshape(200, 200, 200)[:planes])
    began = time.monotonic()
    actual = backend.cp.asnumpy(backend(raw))
    gpu_seconds = time.monotonic() - began
    began = time.monotonic()
    valid = actual.shape == raw.shape and actual.dtype == np.uint8
    if backend.name == "bm3d":
        return {"passed": valid, "planes": 8, "kind": "full-plane-execution-only-new-variant"}
    if backend.name == "original":
        expected = raw
    else:
        x = raw.astype(np.float32) / 255.0
        if backend.name == "gaussian":
            filtered = gaussian_filter(x, (0, 1.5, 1.5), mode="nearest", truncate=4.0)
        elif backend.name == "median":
            filtered = median_filter(x, size=(1, 3, 3), mode="nearest")
        elif backend.name == "tv":
            filtered = np.stack([denoise_tv_chambolle(p, weight=0.1, eps=0.0002, max_num_iter=200) for p in x])
        elif backend.name == "wavelet":
            filtered = np.stack(
                [denoise_wavelet(p, wavelet="db4", mode="soft", method="BayesShrink", rescale_sigma=True) for p in x]
            )
        elif backend.name == "nlm":
            filtered = np.stack(
                [
                    denoise_nl_means(
                        p, h=0.1, patch_size=5, patch_distance=3, fast_mode=True, sigma=0, channel_axis=None
                    )
                    for p in x
                ]
            )
        else:
            raise ValueError("No approved full-plane reference for this backend")
        expected = np.clip(filtered * 255.0, 0, 255).astype(np.uint8)
    return {
        "passed": valid and bool(np.array_equal(actual, expected)),
        "planes": planes,
        "kind": "full-plane-exact-uint8-CPU-parity",
        "input_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "gpu_seconds": gpu_seconds,
        "cpu_reference_seconds": time.monotonic() - began,
    }


def validate_extra_backend(backend, training, path):
    began = time.monotonic()
    report = {"passed": False, "cases": {}}
    try:
        report["backend_validation"] = backend.validate(np.asarray(training[0]).reshape(200, 200, 200))
        for index in (0, 25, 100, 1000):
            report["cases"][f"Training_{index}"] = validate_oct_planes(
                backend, np.asarray(training[index]), planes=200 if index == 0 else 8
            )
        report["passed"] = report["backend_validation"]["passed"] and all(
            case["passed"] for case in report["cases"].values()
        )
        report["oct_full_volume_checked"] = True
        return report
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:300]}
        raise
    finally:
        report["seconds"] = time.monotonic() - began
        report["metadata"] = backend.metadata.copy()
        atomic_json(path, report)


def aggregate(directory, baseline=None):
    import pandas as pd

    rows = []
    for path in sorted(directory.glob("metrics/*/shard-*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    frame = pd.DataFrame(rows)
    numeric = [
        c
        for c in frame
        if c not in {"index", "class", "split", "invalid_reasons", "depth_axis", "slice_axis", "otsu_threshold"}
    ]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    result = []
    groups = [("split_class", split, label, group) for (split, label), group in frame.groupby(["split", "class"])]
    groups += [("split", split, None, group) for split, group in frame.groupby("split")]
    groups.append(("whole_dataset", "all", None, frame))
    for scope, split, label, group in groups:
        for key in numeric:
            values = group[key].dropna()
            result.append(
                {
                    "split": split,
                    "scope": scope,
                    "class": label,
                    "metric": key,
                    "count": len(values),
                    "invalid_count": len(group) - len(values),
                    "volume_count": len(group),
                    "mean": values.mean(),
                    "std_ddof1": values.std(ddof=1),
                    "median": values.median(),
                    "p25": values.quantile(0.25),
                    "p75": values.quantile(0.75),
                }
            )
    target = directory / "aggregate.csv"
    pd.DataFrame(result).to_csv(target, index=False, na_rep="")
    frame.to_csv(directory / "per_volume.csv", index=False, na_rep="")
    if baseline is not None:
        base = pd.read_csv(Path(baseline) / "per_volume.csv")
        paired = frame.merge(base, on=["split", "index", "class"], suffixes=("", "_raw"), validate="one_to_one")
        if len(paired) != len(frame) or len(base) != len(frame):
            raise ValueError("Raw baseline and method do not cover the same samples")
        delta = paired[["split", "index", "class"]].copy()
        for key in numeric:
            delta[key + "_minus_raw"] = paired[key] - paired[key + "_raw"]
        delta.to_csv(directory / "paired_vs_raw.csv", index=False, na_rep="")
    return target


def figure(raw, den, index, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    paths = []
    for plane in (80, 120):
        lo, hi = np.percentile(raw[plane], [1, 99])
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        for col, (image, title) in enumerate(((raw[plane], "Raw"), (den[plane], "Denoised"))):
            axes[0, col].imshow(image, cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
            axes[0, col].add_patch(Rectangle((60, 60), 80, 80, fill=False, edgecolor="orange"))
            axes[1, col].imshow(image[60:140, 60:140], cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
            axes[0, col].set_title(title)
        for ax in axes.flat:
            ax.set_axis_off()
        fig.suptitle(f"Training {index}, slice {plane}; fixed raw ROI, descriptive only, no diagnosis")
        fig.tight_layout()
        path = directory / f"Training-{index}-slice-{plane}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def run_method(
    args, name, backend, metadata, arrays, reused, archive, run, state, status, stopped, cp, suite, resume_only=False
):
    source = {"repo": SOURCE, "revision": REVISION}
    if set(arrays) != set(SPLITS):
        raise ValueError("Source splits incomplete")
    for split, (raws, labels) in arrays.items():
        validate_arrays(raws, labels, split)
    cid = config_id(metadata, source, suite.metric_protocol)
    directory = args.work_dir / name / cid
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"classical/{name}/{cid}" if name != "original" else f"baseline/original/{cid}"
    manifest_path = directory / "manifest.json"
    final_receipt = directory / "complete.receipt.json"
    identity = {"dataset": DESTINATION, "prefix": prefix, "config_id": cid, "source": source}
    if final_receipt.exists():
        completed = read_json(final_receipt)
        saved = read_json(manifest_path)
        if saved.get("complete") is not True or saved["config_id"] != cid or saved["method"] != name:
            raise ValueError("Local completion manifest mismatch")
        verify_coverage(saved["shards"])
        for receipt in completed["artifacts"]:
            archive.verify(receipt)
        for receipt in read_json(manifest_path)["shards"]:
            for artifact in receipt["artifacts"]:
                archive.verify(artifact)
            if receipt.get("generated") and not completed.get("remote_only"):
                cleanup_generated(receipt["generated"], directory / "shards", receipt)
        return {"status": "complete", "config_id": cid, "resumed_remote": True}
    recovered = archive.remote_completed(prefix)
    if recovered is not None:
        saved_manifest, saved_receipts = recovered
        if saved_manifest["config_id"] != cid or saved_manifest["method"] != name:
            raise ValueError("Remote configuration mismatch")
        atomic_json(manifest_path, saved_manifest)
        atomic_json(final_receipt, saved_receipts)
        from huggingface_hub import hf_hub_url

        for receipt in saved_receipts["artifacts"]:
            if receipt["remote"] == f"{prefix}/per_volume.csv":
                download_verified(
                    archive.session,
                    hf_hub_url(DESTINATION, receipt["remote"], repo_type="dataset", revision=receipt["commit"]),
                    archive.token,
                    directory / "per_volume.csv",
                    receipt["size"],
                    receipt["sha256"],
                )
        return {"status": "complete", "config_id": cid, "prefix": prefix, "resumed_remote": True}
    if resume_only:
        return None
    manifest = {
        "method": name,
        "config_id": cid,
        "source": source,
        "metadata": metadata,
        "metric_protocol": suite.metric_protocol,
        "figure_cases": {"Training": CASES, "slices": [80, 120]},
        "baseline": state.get("methods", {}).get("original"),
        "shards": [],
        "complete": False,
    }
    shard_root = directory / "shards"
    shard_root.mkdir(exist_ok=True)
    figure_dir = directory / "figures"
    figure_dir.mkdir(exist_ok=True)
    for split, (raws, labels) in arrays.items():
        metrics = directory / "metrics" / split
        metrics.mkdir(parents=True, exist_ok=True)
        volumes = shard_root / split
        volumes.mkdir(exist_ok=True)
        for start in range(0, len(raws), SHARD_SIZE):
            if stopped():
                return {"status": "stopped", "config_id": cid}
            number, end = start // SHARD_SIZE, min(start + SHARD_SIZE, len(raws))
            stem = f"shard-{number:05d}"
            receipt_path = metrics / f"{stem}.receipt.json"
            shard_identity = dict(identity, split=split, start=start, end=end)
            if resume_shard(receipt_path, archive, shard_root):
                manifest["shards"].append(read_json(receipt_path))
                continue
            pending_path = metrics / f"{stem}.pending.json"
            if pending_path.exists():
                receipt = upload_pending(receipt_path, pending_path, archive, shard_identity)
                manifest["shards"].append(receipt)
                atomic_json(manifest_path, manifest)
                mirror(args, directory)
                if receipt.get("generated"):
                    cleanup_generated(receipt["generated"], shard_root, receipt)
                continue
            disk_gate(directory, (end - start) * 200**3 + 128 * 1024**2)
            target = volumes / f"{stem}.npy"
            partial = volumes / f"{stem}.npy.partial"
            if (
                target.exists()
                or partial.exists()
                or any(metrics.glob(f"{stem}*.partial"))
                or (metrics / f"{stem}.jsonl").exists()
                or (metrics / f"{stem}.slices.jsonl.gz").exists()
            ):
                raise FatalJobError("Unreceipted incomplete shard exists; preserved for explicit recovery")
            output = None
            if name != "original":
                output = np.lib.format.open_memmap(
                    partial, mode="w+", dtype=np.uint8, shape=(end - start, 200, 200, 200)
                )
            global_path, slice_path = metrics / f"{stem}.jsonl", metrics / f"{stem}.slices.jsonl.gz"
            global_tmp, slice_tmp = global_path.with_suffix(".partial"), slice_path.with_suffix(".partial")
            figures = []
            try:
                with global_tmp.open("w", encoding="utf-8") as gs, gzip.open(slice_tmp, "wt", encoding="utf-8") as ss:
                    for index in range(start, end):
                        wait_gpu(cp, args.wait_gpu_seconds)
                        began = time.monotonic()
                        raw = np.asarray(raws[index]).reshape(200, 200, 200)
                        den = cp.asarray(reused[split][index].reshape(raw.shape)) if reused else backend(raw)
                        context = suite.MetricContext(raw)
                        global_scores, slices = context.evaluate(den)
                        sample_identity = {"split": split, "index": index, "class": int(labels[index])}
                        gs.write(json.dumps({**sample_identity, **global_scores}, allow_nan=False) + "\n")
                        for row in slices:
                            ss.write(json.dumps({**sample_identity, **row}, allow_nan=False) + "\n")
                        if output is not None:
                            host = cp.asnumpy(den)
                            output[index - start] = host
                            if split == "Training" and index in CASES:
                                images = figure(raw, host, index, figure_dir)
                                figures.extend(images)
                                import wandb

                                log_online(run, {"figures/comparison": [wandb.Image(str(p)) for p in images]})
                            del host
                        del context, den, slices
                        suite.release_memory()
                        duration = time.monotonic() - began
                        verified_volumes = sum(r["end"] - r["start"] for r in manifest["shards"])
                        status(
                            phase="computing",
                            method=name,
                            split=split,
                            index=index,
                            shard_start=start,
                            verified_volumes=verified_volumes,
                            processed_volumes=verified_volumes + index - start + 1,
                        )
                        log_online(
                            run,
                            {
                                "progress/index": index,
                                "progress/method": name,
                                "progress/split": split,
                                "timing/volume_seconds": duration,
                                "sys/disk_free_gb": shutil.disk_usage(directory).free / 1e9,
                                **{f"metrics/{k}": v for k, v in global_scores.items() if isinstance(v, (int, float))},
                            },
                        )
                        print(f"{name} {split} {index + 1}/{len(raws)} {duration:.2f}s", flush=True)
                    gs.flush()
                    os.fsync(gs.fileno())
                if output is not None:
                    output.flush()
                    output._mmap.close()
                    output = None
                    with partial.open("rb+") as stream:
                        os.fsync(stream.fileno())
                    os.replace(partial, target)
                os.replace(global_tmp, global_path)
                os.replace(slice_tmp, slice_path)
                with slice_path.open("rb+") as stream:
                    os.fsync(stream.fileno())
            finally:
                if output is not None:
                    output._mmap.close()
            status(phase="uploading_and_verifying", method=name, split=split, shard_start=start)
            artifacts = [
                (global_path, f"{prefix}/metrics/{split}/{global_path.name}"),
                (slice_path, f"{prefix}/metrics/{split}/{slice_path.name}"),
            ]
            artifacts.extend((p, f"{prefix}/figures/{p.name}") for p in figures)
            if name != "original":
                artifacts.append((target, f"{prefix}/volumes/{split}/{target.name}"))
            for path, _ in artifacts:
                with path.open("rb+") as stream:
                    os.fsync(stream.fileno())
            pending = {
                "identity": shard_identity,
                "metadata": metadata,
                "metric_protocol": suite.metric_protocol,
                "verified": False,
                "split": split,
                "start": start,
                "end": end,
                "artifacts": [
                    {
                        "local": str(p.resolve()),
                        "remote": remote,
                        "size": p.stat().st_size,
                        "sha256": digest(p),
                        "git_sha1": digest(p, "sha1", git=True),
                    }
                    for p, remote in artifacts
                ],
                "generated": str(target.resolve()) if name != "original" else None,
            }
            atomic_json(pending_path, pending)
            receipt = upload_pending(receipt_path, pending_path, archive, shard_identity)
            manifest["shards"].append(receipt)
            atomic_json(manifest_path, manifest)
            mirror(args, directory)
            if name != "original":
                cleanup_generated(target, shard_root, receipt)
    baseline = None
    if name != "original":
        baseline = args.work_dir / "original" / state["methods"]["original"]["config_id"]
        if not (baseline / "per_volume.csv").is_file():
            raise FatalJobError("Raw baseline per-volume report missing locally; do not publish unpaired comparison")
    count = verify_coverage(manifest["shards"])
    aggregate_path = aggregate(directory, baseline)
    import pandas as pd

    rows = pd.read_csv(directory / "per_volume.csv")
    if len(rows) != count or set(rows["split"]) != set(SPLITS):
        raise ValueError("Metric volume count mismatch")
    for split, (_, labels) in arrays.items():
        part = rows[rows["split"] == split].sort_values("index")
        if not np.array_equal(part["index"], np.arange(len(labels))) or not np.array_equal(part["class"], labels):
            raise ValueError("Metric sample identities disagree with labels")
    manifest["complete"] = True
    atomic_json(manifest_path, manifest)
    end_items = [
        (aggregate_path, f"{prefix}/aggregate.csv"),
        (manifest_path, f"{prefix}/manifest.json"),
    ]
    for report_path in (directory / "per_volume.csv", directory / "paired_vs_raw.csv"):
        if report_path.exists():
            end_items.append((report_path, f"{prefix}/{report_path.name}"))
    receipts = archive.upload_many(end_items, f"{prefix} final reports")
    manifest_receipt = next(receipt for receipt in receipts if receipt["remote"] == f"{prefix}/manifest.json")
    marker = directory / "_COMPLETE.json"
    atomic_json(marker, {"complete": True, "volumes": count, "manifest": manifest_receipt, "reports": receipts.copy()})
    receipts.extend(archive.upload_many([(marker, f"{prefix}/_COMPLETE.json")], f"{prefix} complete marker"))
    atomic_json(final_receipt, {"artifacts": receipts})
    import wandb

    report = wandb.Artifact(f"denoise-{name}-{cid}", type="benchmark-report")
    report.add_file(str(aggregate_path))
    report.add_file(str(manifest_path))
    try:
        run.log_artifact(report)
    except Exception:
        raise FatalJobError("Mandatory W&B report logging failed") from None
    mirror(args, directory)
    return {"status": "complete", "config_id": cid, "prefix": prefix, "volumes": count}


def mirror(args, directory):
    if args.drive_dir is None:
        return
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix not in {".npy", ".partial"} and "wandb" not in path.parts:
            target = args.drive_dir / path.relative_to(args.work_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--execute", action="store_true", help="Authorize public benchmark uploads and the full job")
    result.add_argument(
        "--pilot-only", action="store_true", help="One Training volume per method; no full output upload"
    )
    result.add_argument("--methods", nargs="+", choices=(*METHODS, "all7"), default=list(METHODS))
    result.add_argument(
        "--max-compute-hours", type=float, default=4, help="Per-method pilot extrapolation gate, excluding network"
    )
    result.add_argument("--allow-slow", action="store_true", help="Override extrapolated compute-time gate")
    result.add_argument("--wait-gpu-seconds", type=float, default=1200)
    result.add_argument("--dll", default=None)
    result.add_argument("--work-dir", type=Path, default=ROOT / "outputs/denoise_full_cuda")
    result.add_argument("--raw-dir", type=Path, default=ROOT / "data/denoise_benchmark_raw")
    result.add_argument("--bilateral-dir", type=Path, default=ROOT / "data/bilateral_200")
    result.add_argument("--drive-dir", type=Path, default=os.getenv("DENOISE_DRIVE_DIR"))
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if "all7" in args.methods:
        args.methods = list(METHODS)
    if any((args.work_dir / name).exists() for name in ("STOP", "BLOCKED", "BLOCKED.json")):
        raise SystemExit("STOP/BLOCKED sentinel present; no authentication, download or GPU work started")
    if not args.execute and not args.pilot_only:
        raise SystemExit("Use --execute for full processing or --pilot-only for the bounded pilot")
    if args.max_compute_hours <= 0 or args.wait_gpu_seconds < 0:
        raise SystemExit("Invalid time budget")
    sys.path.insert(0, str(ROOT))
    from pipeline.utils import load_env_file

    load_env_file()
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN required; no directories or remote repositories created")
    if not os.getenv("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY required for unattended online logging; no interactive login")
    import psutil

    import wandb

    args.work_dir = args.work_dir.resolve()
    if args.drive_dir is not None:
        args.drive_dir = args.drive_dir.resolve()
        if not args.drive_dir.is_dir() or not (args.drive_dir / ".denoise-drive-approved").is_file():
            raise SystemExit("Drive requires an existing mounted directory and .denoise-drive-approved sentinel")
        if args.drive_dir.is_relative_to(args.work_dir) or args.work_dir.is_relative_to(args.drive_dir):
            raise SystemExit("Drive mirror and staging directories must not overlap")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    lock = args.work_dir / "job.lock"
    if lock.exists():
        old = read_json(lock)
        if psutil.pid_exists(old["pid"]) and abs(psutil.Process(old["pid"]).create_time() - old["created"]) < 1:
            raise SystemExit("A benchmark launcher already owns this work directory")
        lock.unlink()
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "created": psutil.Process().create_time()}, stream)
    state = {
        "pid": os.getpid(),
        "methods": {name: {"status": "queued" if name in args.methods else "not_selected"} for name in METHODS},
        "deep": {name: {"status": "pending_speed_gate_no_training"} for name in ("DnCNN", "SwinIR")},
        "drive": "verified_local_destination" if args.drive_dir else "pending; HF primary approved",
    }
    stop = False
    run = None

    def request_stop(*unused):
        nonlocal stop
        stop = True

    def stopped():
        return stop or any((args.work_dir / name).exists() for name in ("STOP", "BLOCKED", "BLOCKED.json"))

    status_clock = {"time": 0.0, "phase": None}

    def status(force=False, **updates):
        state.update(updates)
        phase_changed = "phase" in updates and updates["phase"] != status_clock["phase"]
        status_clock["phase"] = state.get("phase")
        now = time.monotonic()
        if not force and not phase_changed and now - status_clock["time"] < 5.0:
            return
        status_clock["time"] = now
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        atomic_json(args.work_dir / "status.json", state)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        status(phase="authenticating")
        archive = Archive(token)
        run = wandb.init(
            project="glaucoma-thesis",
            name="denoise-full-cuda",
            job_type="benchmark",
            mode="online",
            dir=str(args.work_dir),
            config={
                "source": SOURCE,
                "revision": REVISION,
                "methods": args.methods,
                "pilot_only": args.pilot_only,
                "max_compute_hours": args.max_compute_hours,
            },
            settings=wandb.Settings(init_timeout=60),
        )
        if run is None or run.settings.mode != "online":
            raise RuntimeError("Online W&B is mandatory")
        status(phase="downloading_raw", wandb_url=run.url)
        arrays, provenance = load_sources(
            archive,
            args.raw_dir,
            progress=lambda name, done, total: status(
                phase="downloading_raw", file=name, downloaded_bytes=done, expected_bytes=total
            ),
        )
        atomic_json(args.work_dir / "source.json", {"repo": SOURCE, "revision": REVISION, "files": provenance})
        archive.upload(args.work_dir / "source.json", "source.json")
        for split in SPLITS:
            archive.upload(args.raw_dir / f"{split}_labels.npy", f"labels/{split}_labels.npy")
        readme = args.work_dir / "README.md"
        readme.write_text(
            "---\nlicense: cc-by-nc-nd-4.0\n---\n# Harvard-GF denoising research benchmark\n\n"
            "Source: harvardairobotics/Harvard-GF; consolidated source " + SOURCE + " at " + REVISION + ".\n"
            "Public release was explicitly authorized by the repository owner, who confirmed redistribution permission. "
            "Preserve upstream attribution and terms; public access is not a claim that NC-ND generally permits derivatives.\n"
            "Raw-fixed descriptive metrics are not clinical validation. Bilateral is reused, not newly filtered.\n"
            "Wavelet: CUDA db4 soft BayesShrink; NLM: CUDA fast 2D, h=0.1, patch=5, distance=3. "
            "Both require strict uint8 CPU-reference validation on full Training 0 and eight planes each of "
            "Training 25/100/1000 before processing. See validations/, pilots/ and manifest.json for actual status. "
            "The frozen raw-fixed metric-v2 masks and scores are unchanged; historical scores remain comparable "
            "only under the same metric protocol, source and method parameters. "
            "DnCNN/SwinIR await a speed gate. No training performed.\n",
            encoding="utf-8",
        )
        archive.upload(readme, "README.md")
        for name in ("DnCNN", "SwinIR"):
            pending = args.work_dir / f"{name}-README.md"
            pending.write_text(
                f"# {name}\nPending bounded speed gate. No inference or training performed.\n", encoding="utf-8"
            )
            archive.upload(pending, f"deep_learning/{name.lower()}/README.md")
        import cupy as cp

        patch_cupy_gaussian()
        from scripts import denoise_cuda_suite as suite

        status(phase="waiting_gpu")
        wait_gpu(cp, args.wait_gpu_seconds)
        for name in ["original", *dict.fromkeys([*args.methods, *METHODS])]:
            if stopped():
                break
            disk_gate(args.work_dir, SHARD_SIZE * 200**3)
            status(phase="validating", method=name)
            reused, backend = None, None
            validation_started = False
            try:
                if name == "bilateral":
                    reused, metadata = load_bilateral(args.bilateral_dir, arrays)
                    validation = {"passed": True, "kind": "all-split-SHA256-provenance-reuse-not-new-parity"}
                else:
                    backend = (
                        get_extra_backend(name) if name in ("wavelet", "nlm") else suite.get_backend(name, dll=args.dll)
                    )
                    metadata = backend.metadata.copy()
                if not args.pilot_only:
                    completed = run_method(
                        args,
                        name,
                        backend,
                        metadata,
                        arrays,
                        reused,
                        archive,
                        run,
                        state,
                        status,
                        stopped,
                        cp,
                        suite,
                        resume_only=True,
                    )
                    if completed is not None:
                        state["methods"][name] = completed
                        status()
                        continue
                if name != "original" and name not in args.methods:
                    continue
                if name != "bilateral":
                    validation_path = args.work_dir / f"{name}-validation.json"
                    validation_started = True
                    atomic_json(validation_path, {"passed": False, "status": "validating", "metadata": metadata})
                    if name in ("wavelet", "nlm"):
                        validation = validate_extra_backend(backend, arrays["Training"][0], validation_path)
                    else:
                        validation = suite.validate_backend(
                            backend, np.asarray(arrays["Training"][0][0]).reshape(200, 200, 200)[:8]
                        )
                    if validation["passed"] and name not in ("wavelet", "nlm"):
                        validation["full_oct_planes"] = validate_oct_planes(
                            backend, np.asarray(arrays["Training"][0][0])
                        )
                        validation["passed"] = validation["full_oct_planes"]["passed"]
                    metadata = backend.metadata.copy()
                    atomic_json(validation_path, validation)
                    archive.upload(validation_path, f"validations/{name}.json")
                if not validation["passed"]:
                    state["methods"][name] = {"status": "failed_backend", "validation": validation}
                    status()
                    if name == "original":
                        break
                    continue
                began = time.monotonic()
                count = 1 if args.pilot_only else 12
                durations = []
                for index in range(count):
                    wait_gpu(cp, args.wait_gpu_seconds)
                    tick = time.monotonic()
                    raw = np.asarray(arrays["Training"][0][index]).reshape(200, 200, 200)
                    den = cp.asarray(reused["Training"][index].reshape(raw.shape)) if reused else backend(raw)
                    context = suite.MetricContext(raw)
                    scores, rows = context.evaluate(den)
                    cp.cuda.get_current_stream().synchronize()
                    del context, den, rows
                    suite.release_memory()
                    durations.append(time.monotonic() - tick)
                    log_online(run, {"pilot/method": name, "pilot/index": index, "pilot/seconds": durations[-1]})
                estimate = (time.monotonic() - began) / count * 3300 / 3600
                pilot = {
                    "metadata": metadata,
                    "validation": validation,
                    "volumes": count,
                    "seconds": durations,
                    "estimated_compute_hours": estimate,
                    "network_time": "excluded; full verification egress required",
                }
                pilot_path = args.work_dir / f"{name}-pilot.json"
                atomic_json(pilot_path, pilot)
                archive.upload(pilot_path, f"pilots/{name}.json")
                print(f"{name}: projected compute+scoring {estimate:.2f}h; uploads/verification additional", flush=True)
            except FatalJobError:
                raise
            except Exception as exc:
                if validation_started:
                    failure = read_json(validation_path)
                    failure.update(error_type=type(exc).__name__, error=str(exc)[:300])
                    atomic_json(validation_path, failure)
                state["methods"][name] = {
                    "status": "failed_backend",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:300],
                    "trace": traceback.format_exc()[-1200:],
                }
                status()
                if name == "original":
                    break
                continue
            if args.pilot_only or (estimate > args.max_compute_hours and not args.allow_slow):
                state["methods"][name] = {
                    "status": "pilot_only" if args.pilot_only else "paused_compute_budget",
                    "pilot": pilot,
                }
                status()
                if name == "original" and not args.pilot_only:
                    break
                continue
            try:
                state["methods"][name] = run_method(
                    args, name, backend, metadata, arrays, reused, archive, run, state, status, stopped, cp, suite
                )
            except (cp.cuda.memory.OutOfMemoryError, GPUBusyError):
                state["methods"][name] = {"status": "paused_gpu_memory"}
                suite.release_memory()
            status()
            if name == "original" and state["methods"][name]["status"] != "complete":
                break
        complete = all(state["methods"].get(n, {}).get("status") == "complete" for n in ["original", *args.methods])
        all_seven = all(state["methods"].get(n, {}).get("status") == "complete" for n in ["original", *METHODS])
        status(
            phase="stopped" if stopped() else "finished",
            selected_methods_complete=complete,
            all_seven_complete=all_seven,
            force=True,
        )
        manifest_receipt = archive.upload(args.work_dir / "status.json", "manifest.json")
        if complete:
            marker = args.work_dir / "_COMPLETE_SELECTED.json"
            atomic_json(
                marker,
                {
                    "methods": args.methods,
                    "all_seven_complete": all_seven,
                    "volumes_per_method": 3300,
                    "manifest": manifest_receipt,
                },
            )
            archive.upload(marker, "_COMPLETE_SELECTED.json")
        mirror(args, args.work_dir)
        run.summary.update({"selected_methods_complete": complete, "all_seven_complete": all_seven})
        return 0 if complete or args.pilot_only else 2
    except StorageQuotaError as exc:
        status(phase="blocked_hf_storage", force=True, error_type=type(exc).__name__, error=str(exc))
        atomic_json(args.work_dir / "BLOCKED.json", {"reason": "hf_private_storage_limit", "message": str(exc)})
        print(str(exc), flush=True)
        return 78
    except Exception as exc:
        message = str(exc)[:300]
        status(phase="failed", force=True, error_type=type(exc).__name__, error=message)
        print(f"Launcher failed: {type(exc).__name__}: {message}; artifacts preserved. See status.json.", flush=True)
        return 1
    finally:
        try:
            if run is not None:
                run.finish(exit_code=1 if state.get("phase") in {"failed", "blocked_hf_storage"} else 0)
        finally:
            lock.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
