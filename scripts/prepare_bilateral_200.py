"""Stream pinned HF volumes into a resumable CPU bilateral dataset; never train or publish."""

import argparse
import hashlib
import importlib.util
import inspect
import io
import json
import os
import shutil
import signal
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("Training", "Validation", "Test")
PARAMS = {
    "sigma_color": 0.10,
    "sigma_spatial": 4.0,
    "bins": 10000,
    "mode": "constant",
    "cval": 0,
    "channel_axis": None,
    "win_size": None,
}
SESSION = None
HEADERS = None
SOURCES = None
ENGINE = None
BACKEND = "cpu"


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_range(session, url, start, end, size, headers=None):
    expected = end - start + 1
    for attempt in range(6):
        try:
            with session.get(
                url,
                headers={**(headers or {}), "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                stream=True,
                timeout=(30, 180),
            ) as response:
                if response.status_code == 200 and start == 0 and end == size - 1:
                    pass
                elif (
                    response.status_code != 206
                    or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}"
                ):
                    raise ValueError(
                        f"Invalid range response: HTTP {response.status_code}, {response.headers.get('Content-Range')}"
                    )
                payload = response.raw.read(expected + 1, decode_content=True)
                if len(payload) != expected:
                    raise ValueError(f"Truncated/oversized range: wanted {expected}, received {len(payload)}")
                return payload
        except Exception as exc:
            if attempt == 5:
                raise RuntimeError(f"Range download failed after 6 attempts ({type(exc).__name__})") from None
            time.sleep(min(30, 2**attempt))
    raise AssertionError("unreachable")


def parse_volume_header(payload, file_size):
    import numpy as np

    stream = io.BytesIO(payload)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"Unsupported NPY header version: {version}")
    if fortran or dtype != np.dtype("uint8") or len(shape) not in (4, 5):
        raise ValueError("Expected C-order uint8 4D/5D NPY volumes")
    if shape[1:] not in ((200, 200, 200), (1, 200, 200, 200)) or shape[0] < 1:
        raise ValueError(f"Raw resolution must be 200 cubed: {shape}")
    offset = stream.tell()
    if offset + int(np.prod(shape)) != file_size:
        raise ValueError("NPY shape and remote file size disagree")
    return {"shape": list(shape), "dtype": "uint8", "offset": offset, "sample_bytes": 200**3}


def bilateral(volume):
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


def initialize_worker(sources, token, backend="cpu"):
    global SESSION, HEADERS, SOURCES, ENGINE, BACKEND
    import requests

    SESSION = requests.Session()
    HEADERS = {"Authorization": f"Bearer {token}"} if token else {}
    SOURCES = sources
    BACKEND = backend
    if backend == "gpu":
        sys.path.insert(0, str(ROOT))
        from scripts.bilateral_cuda import CudaBilateral

        ENGINE = CudaBilateral()


def process_volume(task):
    import numpy as np

    split, index = task
    source = SOURCES[split]
    start = source["offset"] + index * source["sample_bytes"]
    began = time.monotonic()
    payload = read_range(SESSION, source["url"], start, start + source["sample_bytes"] - 1, source["size"], HEADERS)
    volume = np.frombuffer(payload, dtype=np.uint8).reshape(200, 200, 200)
    result = ENGINE(volume) if BACKEND == "gpu" else bilateral(volume)
    return index, result, time.monotonic() - began


class SplitWriter:
    def __init__(self, directory, split, identity, shape):
        import numpy as np

        self.directory, self.split, self.identity = Path(directory), split, identity
        self.target = self.directory / f"{split}_volumes_dn.npy"
        self.partial = self.directory / f"{split}_volumes_dn.partial.npy"
        self.progress = self.directory / f"{split}_progress.json"
        self.marker = self.directory / f"{split}_complete.json"
        self.shape = tuple(shape)
        self.output = None
        self.done = set()
        self.complete = False
        if self.marker.exists():
            saved = json.loads(self.marker.read_text(encoding="utf-8"))
            if saved["identity"] != identity or not self.target.is_file() or saved["sha256"] != digest(self.target):
                raise ValueError(f"Completed output identity/hash mismatch: {split}")
            if saved.get("shape") != list(self.shape) or saved.get("dtype") != "uint8":
                raise ValueError("Completed output shape/dtype mismatch")
            array = np.load(self.target, mmap_mode="r")
            valid = array.shape == self.shape and array.dtype == np.uint8
            array._mmap.close()
            if not valid:
                raise ValueError("Completed array shape/dtype mismatch")
            self.done = set(range(shape[0]))
            self.complete = True
            return
        if self.progress.exists():
            saved = json.loads(self.progress.read_text(encoding="utf-8"))
            if saved["identity"] != identity:
                raise ValueError(f"Partial preprocessing identity mismatch: {split}")
            self.done = set(saved["done"])
            if not self.done.issubset(range(shape[0])):
                raise ValueError("Invalid completed sample indices")
            if self.partial.exists() and self.target.exists():
                raise FileExistsError("Both partial and unverified final output exist; refusing to overwrite")
            if not self.partial.exists() and self.target.exists() and len(self.done) == shape[0]:
                self.publish()
                return
            if not self.partial.exists():
                raise FileNotFoundError("Progress exists without its partial array")
            self.output = np.load(self.partial, mmap_mode="r+")
            if self.output.shape != self.shape or self.output.dtype != np.uint8:
                raise ValueError("Partial array shape/dtype mismatch")
        else:
            if self.target.exists() or self.partial.exists():
                raise FileExistsError(f"Unverified output exists: {split}; refusing to overwrite")
            self.output = np.lib.format.open_memmap(self.partial, mode="w+", dtype=np.uint8, shape=self.shape)
            self.output.flush()
            atomic_json(self.progress, {"identity": identity, "done": []})

    def write(self, index, result):
        import numpy as np

        expected = self.shape[-3:]
        if self.output is None or index in self.done or not 0 <= index < self.shape[0]:
            raise ValueError("Invalid or already committed sample")
        if result.shape != expected or result.dtype != np.uint8:
            raise ValueError("Denoising changed shape or dtype")
        self.output[index] = result[None] if len(self.shape) == 5 else result
        self.output.flush()
        with self.partial.open("rb+") as stream:
            os.fsync(stream.fileno())
        self.done.add(index)
        atomic_json(self.progress, {"identity": self.identity, "done": sorted(self.done)})

    def publish(self):
        if len(self.done) != self.shape[0]:
            raise ValueError("Cannot publish an incomplete split")
        if self.output is not None:
            self.output.flush()
            self.output._mmap.close()
            self.output = None
        if self.partial.exists():
            if self.target.exists():
                raise FileExistsError("Refusing to overwrite an unverified final output")
            os.replace(self.partial, self.target)
        atomic_json(
            self.marker,
            {
                "identity": self.identity,
                "shape": list(self.shape),
                "dtype": "uint8",
                "file": self.target.name,
                "sha256": digest(self.target),
                "complete": True,
            },
        )
        self.complete = True

    def close(self):
        if self.output is not None:
            self.output._mmap.close()
            self.output = None


def prepare_sources(repo, revision, token, directory):
    import numpy as np
    import requests
    from huggingface_hub import HfApi, hf_hub_url

    info = HfApi(token=token).dataset_info(repo, revision=revision, files_metadata=True)
    remote = {f.rfilename: f for f in info.siblings}
    sources = {}
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with requests.Session() as session:
        for split in SPLITS:
            filename = f"{split}_volumes.npy"
            file = remote[filename]
            url = hf_hub_url(repo, filename, repo_type="dataset", revision=info.sha)
            header = read_range(session, url, 0, 511, file.size, headers)
            source = {
                **parse_volume_header(header, file.size),
                "url": url,
                "size": file.size,
                "filename": filename,
                "repo_revision": info.sha,
            }
            labels_name = f"{split}_labels.npy"
            labels_file = remote[labels_name]
            labels_url = hf_hub_url(repo, labels_name, repo_type="dataset", revision=info.sha)
            labels_bytes = read_range(session, labels_url, 0, labels_file.size - 1, labels_file.size, headers)
            labels = np.load(io.BytesIO(labels_bytes), allow_pickle=False)
            if labels.shape != (source["shape"][0],) or set(np.unique(labels)) != {0, 1}:
                raise ValueError(f"Unexpected binary labels/count: {split}")
            label_path = directory / labels_name
            if label_path.exists():
                if digest(label_path) != hashlib.sha256(labels_bytes).hexdigest():
                    raise ValueError(f"Existing labels differ from the pinned source: {split}")
            temporary = label_path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                stream.write(labels_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, label_path)
            source["labels_sha256"] = hashlib.sha256(labels_bytes).hexdigest()
            source["class_counts"] = {str(i): int((labels == i).sum()) for i in (0, 1)}
            sources[split] = source
    return info.sha, sources


def select_backend(requested):
    if requested == "cpu":
        return "cpu"
    if importlib.util.find_spec("cupy") is None:
        if requested == "gpu":
            raise RuntimeError("GPU backend requested but cupy is not installed")
        return "cpu"
    try:
        import cupy

        cupy.zeros(1).sum()
    except Exception:
        if requested == "gpu":
            raise
        print("[warn] cupy present but CUDA unusable; falling back to CPU", flush=True)
        return "cpu"
    return "gpu"


def validate_gpu(sources, token, directory):
    import numpy as np
    import requests

    sys.path.insert(0, str(ROOT))
    from scripts.bilateral_cuda import CudaBilateral

    missing = [s for s in SPLITS if not (directory / f"{s}_complete.json").exists()]
    if not missing:
        return {"status": "skipped-all-complete"}
    split = missing[0]
    done = set()
    progress = directory / f"{split}_progress.json"
    if progress.exists():
        done = set(json.loads(progress.read_text(encoding="utf-8"))["done"])
    index = next(i for i in range(sources[split]["shape"][0]) if i not in done)
    source = sources[split]
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with requests.Session() as session:
        start = source["offset"] + index * source["sample_bytes"]
        payload = read_range(session, source["url"], start, start + source["sample_bytes"] - 1, source["size"], headers)
    volume = np.frombuffer(payload, dtype=np.uint8).reshape(200, 200, 200)
    expected = bilateral(volume)
    result = CudaBilateral()(volume)
    mismatch = int(np.count_nonzero(expected != result))
    if mismatch:
        raise RuntimeError(f"CUDA bilateral mismatch on validation volume: {mismatch} bytes differ")
    print(f"[gpu-validate] {split}[{index}] matches CPU on all {volume.size} bytes", flush=True)
    return {"status": "passed", "split": split, "index": index, "mismatch_bytes": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="tqhuyen/harvard-oct-glaucoma-200")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/bilateral_200")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    directory = args.out_dir.resolve()
    status_path = directory / "status.json"
    if args.status:
        print(status_path.read_text(encoding="utf-8") if status_path.exists() else "No job status yet.")
        return
    if args.stop:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        (directory / "STOP").touch()
        print("Stop requested; active volumes will finish and commit before exit.")
        return
    if args.workers < 1:
        raise ValueError("workers must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    import psutil

    lock = directory / "job.lock"
    if lock.exists():
        previous = json.loads(lock.read_text(encoding="utf-8"))
        if psutil.pid_exists(previous["pid"]):
            process = psutil.Process(previous["pid"])
            if abs(process.create_time() - previous["created"]) < 1:
                raise RuntimeError(f"Preprocessing already running as PID {previous['pid']}")
        lock.unlink()
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "created": psutil.Process().create_time()}, stream)
    run = writer = None
    began = time.monotonic()
    state = {
        "pid": os.getpid(),
        "phase": "initializing",
        "completed_total": 0,
        "planned_total": None,
        "source_repo": args.repo,
        "output_dir": str(directory),
        "drive_sync": "pending: no mounted destination",
    }
    stop_requested = False

    def request_stop(*unused):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def status(**changes):
        state.update(changes)
        state.update(
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), elapsed_seconds=round(time.monotonic() - began, 1)
        )
        atomic_json(status_path, state)

    try:
        status()
        sys.path.insert(0, str(ROOT))
        from pipeline.utils import load_env_file

        load_env_file()
        import numpy as np
        import skimage
        import wandb

        job_path = directory / "job.json"
        saved = json.loads(job_path.read_text(encoding="utf-8")) if job_path.exists() else None
        revision = args.revision or (saved["identity"]["source_revision"] if saved else None)
        status(phase="reading_remote_headers")
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        revision, sources = prepare_sources(args.repo, revision, token, directory)
        identity = {
            "source_repo": args.repo,
            "source_revision": revision,
            "sources": sources,
            "method": "bilateral",
            "params": PARAMS,
            "axis": 0,
            "implementation": "skimage.restoration.denoise_bilateral",
            "filter_code_sha256": hashlib.sha256(inspect.getsource(bilateral).encode("utf-8")).hexdigest(),
            "skimage": skimage.__version__,
            "numpy": np.__version__,
            "output_dtype": "uint8",
            "resolution": 200,
            "quantization": "float32 /255 -> filter -> clip *255 -> uint8 truncation",
            "format": 1,
        }
        if saved and saved["identity"] != identity:
            raise ValueError("Job identity changed; use a different output directory")
        if not saved:
            saved = {"identity": identity, "wandb_id": wandb.util.generate_id()}
            atomic_json(job_path, saved)
        planned = sum(v["shape"][0] for v in sources.values())
        remaining_bytes = sum(
            v["size"]
            for s, v in sources.items()
            if not (directory / f"{s}_volumes_dn.npy").exists()
            and not (directory / f"{s}_volumes_dn.partial.npy").exists()
        )
        if shutil.disk_usage(directory).free < remaining_bytes + 4 * 1024**3:
            raise OSError("Insufficient free disk for remaining outputs plus a 4 GiB reserve")
        backend = select_backend(args.backend)
        per_worker = 400 * 1024**2 if backend == "gpu" else 150 * 1024**2
        memory_workers = max(1, int((psutil.virtual_memory().available - 768 * 1024**2) / per_worker))
        ceiling = 4 if backend == "gpu" else (os.cpu_count() or 1)
        workers = min(args.workers, ceiling, memory_workers)
        validation = {"status": "not-required"}
        if backend == "gpu":
            status(phase="validating_gpu", backend=backend)
            validation = validate_gpu(sources, token, directory)
        status(backend=backend, gpu_validation=validation, workers=min(args.workers, ceiling))
        run = wandb.init(
            project="glaucoma-thesis",
            job_type="preprocess",
            name="bilateral_200_full",
            id=saved["wandb_id"],
            resume="allow",
            config={**identity, "workers": workers, "backend": backend, "gpu_validation": validation},
            dir=str(directory),
            settings=wandb.Settings(init_timeout=60),
        )
        (directory / "STOP").unlink(missing_ok=True)
        status(
            phase="starting_workers",
            planned_total=planned,
            workers=workers,
            backend=backend,
            gpu_validation=validation,
            source_revision=revision,
            wandb_url=run.url,
        )
        completed = 0
        newly_completed = 0
        started_processing = time.monotonic()
        with ProcessPoolExecutor(
            max_workers=workers, initializer=initialize_worker, initargs=(sources, token, backend)
        ) as pool:
            for split in SPLITS:
                source = sources[split]
                writer = SplitWriter(directory, split, identity, source["shape"])
                completed += len(writer.done)
                if writer.complete:
                    writer.close()
                    continue
                todo = iter(i for i in range(source["shape"][0]) if i not in writer.done)
                futures = {}

                def submit():
                    index = next(todo, None)
                    if index is not None:
                        futures[pool.submit(process_volume, (split, index))] = index

                for _ in range(workers):
                    submit()
                last_heartbeat = 0.0
                while futures:
                    stop_requested = stop_requested or (directory / "STOP").exists()
                    ready, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
                    for future in ready:
                        expected_index = futures.pop(future)
                        index, result, duration = future.result()
                        if index != expected_index:
                            raise RuntimeError("Worker/sample index mismatch")
                        writer.write(index, result)
                        del result
                        completed += 1
                        newly_completed += 1
                        elapsed = time.monotonic() - started_processing
                        rate = newly_completed / max(elapsed, 1)
                        eta = (planned - completed) / rate / 3600
                        status(
                            phase="stopping" if stop_requested else "processing",
                            active_split=split,
                            completed_total=completed,
                            split_completed=len(writer.done),
                            split_total=source["shape"][0],
                            volumes_per_hour=round(rate * 3600, 2),
                            eta_hours=round(eta, 2),
                        )
                        print(
                            f"[{split}] {len(writer.done)}/{source['shape'][0]} | total {completed}/{planned} | "
                            f"task {duration:.1f}s | {rate * 3600:.1f} vol/h | ETA {eta:.2f}h",
                            flush=True,
                        )
                        run.log(
                            {
                                "preprocess/completed": completed,
                                "preprocess/task_seconds": duration,
                                "preprocess/volumes_per_hour": rate * 3600,
                                "preprocess/eta_hours": eta,
                            }
                        )
                        if not stop_requested:
                            submit()
                    if time.monotonic() - last_heartbeat > 15:
                        status(
                            phase="stopping" if stop_requested else "processing",
                            active_split=split,
                            completed_total=completed,
                            split_completed=len(writer.done),
                            split_total=source["shape"][0],
                            cpu_percent=psutil.cpu_percent(),
                            available_ram_gib=round(psutil.virtual_memory().available / 2**30, 2),
                        )
                        run.log(
                            {
                                "sys/cpu_percent": state["cpu_percent"],
                                "sys/ram_available_gb": state["available_ram_gib"],
                                "sys/disk_free_gb": shutil.disk_usage(directory).free / 1e9,
                            }
                        )
                        last_heartbeat = time.monotonic()
                if len(writer.done) == source["shape"][0]:
                    status(phase="verifying_split", active_split=split)
                    writer.publish()
                writer.close()
                writer = None
                if stop_requested:
                    status(phase="stopped", completed_total=completed)
                    return
        manifest = {
            "identity": identity,
            "complete": True,
            "total_volumes": planned,
            "backend": backend,
            "gpu_validation": validation,
            "splits": {s: json.loads((directory / f"{s}_complete.json").read_text()) for s in SPLITS},
            "training_compatibility": "Portable export; validate/import into the training cache before use. Local .complete.pt identities are not portable.",
            "redistribution": "Check the upstream dataset terms before publishing. No HF upload was performed.",
        }
        atomic_json(directory / "manifest.json", manifest)
        run.summary.update({"preprocess/complete": True, "preprocess/total_volumes": planned})
        status(phase="complete", completed_total=planned, eta_hours=0)
        print("Completed and verified all splits. No dataset was uploaded.", flush=True)
    except BaseException as exc:
        status(phase="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if writer is not None:
            writer.close()
        if run is not None:
            run.finish(exit_code=1 if state["phase"] == "failed" else 0)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
