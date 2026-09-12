"""Stdlib-only bootstrap; CUDA checks run exclusively in disposable interpreters."""

import ast
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BOOTSTRAP_API_VERSION = 1
MODULES = (
    "controls_kaggle_setup",
    "controls_kaggle",
    "controls_kaggle_data",
    "controls_data",
    "controls_model",
    "controls_training",
    "final_model",
    "final_training",
)
SUPPORT = {
    "timm": "timm==1.0.29",
    "wandb": "wandb",
    "huggingface_hub": "huggingface_hub",
    "dotenv": "python-dotenv",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "psutil": "psutil",
    "filelock": "filelock",
}
MARKER = Path("/kaggle/working/controls-restart-required.json")
GUARD = "CTRL_KAGGLE_RESTART_REQUIRED"
PENDING = "CTRL_KAGGLE_SETUP_PENDING"
_IMPORT_HASHES = {}
_BOOTSTRAP_VERIFIED = False


class RestartRequired(SystemExit):
    pass


class RedactedOutput:
    def __init__(self, stream):
        self.stream = stream
        self.pending = ""
        self.secrets = [os.environ[key] for key in ("HF_TOKEN", "WANDB_API_KEY") if os.environ.get(key)]

    def write(self, text):
        self.pending += text
        for secret in self.secrets:
            self.pending = self.pending.replace(secret, "[REDACTED]")
        boundary = self.pending.rfind("\n") + 1
        if not boundary and len(self.pending) > 65536:
            boundary = len(self.pending) - max([len(secret) for secret in self.secrets] + [1])
        if boundary:
            self.stream.write(self.pending[:boundary])
            self.pending = self.pending[boundary:]
        return len(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def process_identity():
    stat = Path(f"/proc/{os.getpid()}/stat")
    start = stat.read_text().rsplit(")", 1)[1].split()[19] if stat.exists() else "unknown"
    return f"{os.getpid()}:{start}"


NVML_LIBRARY_DIRS = (
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
    "/usr/local/nvidia/lib64",
    "/usr/local/nvidia/lib",
    "/usr/local/cuda/lib64",
    "/opt/nvidia/lib64",
)
GPU_MEMORY_MIB = {"P100": 16384, "T4": 15360}


def nvidia_smi_candidates():
    candidates, seen = [], set()
    for candidate in (
        shutil.which("nvidia-smi"),
        "/usr/bin/nvidia-smi",
        "/usr/local/nvidia/bin/nvidia-smi",
        "/opt/bin/nvidia-smi",
    ):
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def driver_environment():
    environment = {key: value for key, value in os.environ.items() if key != "CUDA_VISIBLE_DEVICES"}
    parts = [part for part in (environment.get("LD_LIBRARY_PATH") or "").split(":") if part]
    for directory in NVML_LIBRARY_DIRS:
        if directory not in parts:
            parts.append(directory)
    environment["LD_LIBRARY_PATH"] = ":".join(parts)
    return environment


def driver_version():
    path = Path("/proc/driver/nvidia/version")
    if path.is_file():
        match = re.search(r"Kernel Module\s+([0-9][0-9.]*)", path.read_text(errors="ignore"))
        if match:
            return match.group(1)
    return "unknown"


def proc_gpu_devices():
    """Fallback GPU list from /proc/driver/nvidia, used when NVML/nvidia-smi is broken."""
    root = Path("/proc/driver/nvidia/gpus")
    if not root.is_dir():
        return []
    devices = []
    for entry in sorted(root.iterdir()):
        information = entry / "information"
        if not information.is_file():
            continue
        model = ""
        for line in information.read_text(errors="ignore").splitlines():
            if line.lower().startswith("model:"):
                model = line.split(":", 1)[1].strip()
        memory = next((value for key, value in GPU_MEMORY_MIB.items() if key in model), 0)
        devices.append({"physical": str(len(devices)), "name": model, "memory_mib": memory, "driver": driver_version()})
    return devices


def query_nvidia_smi(binary, fields, *, nounits=False):
    fmt = "--format=csv,noheader" + (",nounits" if nounits else "")
    try:
        result = subprocess.run(
            [binary, f"--query-gpu={fields}", fmt],
            capture_output=True,
            text=True,
            timeout=60,
            env=driver_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"nvidia-smi timed out at {binary}") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().replace("\n", " ")[:200]
        raise RuntimeError(f"nvidia-smi rc={result.returncode} at {binary}: {detail or 'no output'}")
    if not result.stdout.strip():
        raise RuntimeError(f"nvidia-smi at {binary} reported no GPUs")
    return result.stdout


def gpu_query(fields, *, nounits=False):
    candidates = nvidia_smi_candidates()
    if not candidates:
        raise RuntimeError(
            "nvidia-smi was not found. Enable the Kaggle GPU accelerator (P100 or T4 x2), restart the session, then "
            "rerun Setup. CPU-only sessions cannot run real training."
        )
    return query_nvidia_smi(candidates[0], fields, nounits=nounits)


def gpu_devices():
    """Enumerate physical GPUs, tolerating a broken nvidia-smi binary on PATH."""
    failures = []
    for binary in nvidia_smi_candidates():
        try:
            listing = query_nvidia_smi(binary, "index,name,memory.total,driver_version", nounits=True)
        except RuntimeError as exc:
            failures.append(str(exc))
            continue
        parsed = []
        valid = True
        for row in csv.reader(listing.strip().splitlines()):
            values = [item.strip() for item in row]
            if len(values) != 4:
                valid = False
                break
            index, name, memory, driver = values
            parsed.append({"physical": index, "name": name, "memory_mib": int(float(memory)), "driver": driver})
        if valid and parsed:
            return parsed
    fallback = proc_gpu_devices()
    if fallback:
        print(
            f"[setup] nvidia-smi unusable; using /proc driver data: {[device['name'] for device in fallback]}",
            flush=True,
        )
        return fallback
    nodes = sorted(path.name for path in Path("/dev").glob("nvidia*")) if Path("/dev").is_dir() else []
    raise RuntimeError(
        "No working NVIDIA GPU detected; "
        + " | ".join(failures[-2:] or ["nvidia-smi was not found"])
        + f"; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}; /dev nodes={nodes or 'none'}. "
        "Enable the Kaggle GPU accelerator (GPU P100 or GPU T4 x2) and restart the session, then rerun Setup."
    )


def begin_setup(*, smoke=False):
    if smoke:
        return
    global _BOOTSTRAP_VERIFIED
    _BOOTSTRAP_VERIFIED = False
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        print("[setup] cleared empty CUDA_VISIBLE_DEVICES left by a prior CPU smoke run", flush=True)
    os.environ[PENDING] = "1"


def complete_setup(*, smoke=False):
    if smoke:
        return
    if not _BOOTSTRAP_VERIFIED:
        raise RuntimeError("Setup bootstrap has not passed verification")
    require_ready(importing=True)
    os.environ.pop(PENDING, None)


def require_ready(*, smoke=False, importing=False):
    if smoke:
        return
    if MARKER.exists() or os.environ.get(GUARD):
        raise RestartRequired(
            "Restart the Kaggle kernel, then rerun Setup to verify it before D1/training. Do not reload Torch."
        )
    if os.environ.get(PENDING) and not (importing and _BOOTSTRAP_VERIFIED):
        raise RuntimeError("Setup has not passed verification. Rerun Setup successfully before D1/training.")


def configure_caches(cfg=None):
    root = Path((cfg or {}).get("temp_root", os.environ.get("CTRL_KAGGLE_TEMP_ROOT", "/kaggle/temp/controls96")))
    output = Path(
        (cfg or {}).get("output_root", os.environ.get("CTRL_KAGGLE_OUTPUT_ROOT", "/kaggle/working/controls96"))
    )
    for key, path in {
        "HF_HOME": root / "hf",
        "HF_HUB_CACHE": root / "hf/hub",
        "HF_XET_CACHE": root / "hf/xet",
        "TORCH_HOME": root / "torch",
        "WANDB_CACHE_DIR": root / "wandb-cache",
        "WANDB_DATA_DIR": root / "wandb-staging",
        "WANDB_DIR": output / "wandb",
    }.items():
        os.environ.setdefault(key, str(path))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def discover_repo():
    explicit = os.environ.get("CTRL_KAGGLE_REPO_ROOT")
    if explicit:
        return Path(explicit).resolve()
    for root in (Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[1]):
        if (root / "scripts/controls_kaggle_setup.py").is_file():
            return root
    raise RuntimeError("Cannot locate controls checkout; set CTRL_KAGGLE_REPO_ROOT")


def record_imports():
    for name in MODULES:
        module = sys.modules.get(f"scripts.{name}")
        if module is not None and getattr(module, "__file__", None):
            _IMPORT_HASHES.setdefault(name, (str(Path(module.__file__).resolve()), source_hash(module.__file__)))


def validate_repo(root, *, display=True):
    root = Path(root).resolve()
    for name in MODULES:
        path = root / "scripts" / f"{name}.py"
        if not path.is_file():
            raise RuntimeError(
                f"Incomplete/stale checkout: missing {path}. Attach the intended revision; no automatic pull."
            )
        required = {
            "controls_kaggle": {"default_config", "worker", "dispatch", "validate_config"},
            "controls_kaggle_data": {"prepare_source", "prepare_views", "make_datasets"},
        }.get(name, set())
        symbols = {
            node.name
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        if required - symbols:
            raise RuntimeError(
                f"Stale helper API in {path}: missing {sorted(required - symbols)}; restart with matching checkout"
            )
        module = sys.modules.get(f"scripts.{name}")
        expected = (str(path), source_hash(path))
        if module is not None and (Path(module.__file__).resolve() != path or _IMPORT_HASHES.get(name) != expected):
            raise RuntimeError(
                f"Stale imported scripts.{name}: checkout/source changed or untracked import. Restart kernel."
            )
        if display:
            print(f"[setup] {path} sha256={expected[1]}", flush=True)
    package = sys.modules.get("scripts")
    if package is not None and str(root / "scripts") not in [str(Path(p).resolve()) for p in package.__path__]:
        raise RuntimeError("Stale scripts package from another checkout; restart kernel")
    if display:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        print(
            f"[setup] checked HEAD={result.stdout.strip() if result.returncode == 0 else 'unavailable (attached source)'}",
            flush=True,
        )


def protected_versions():
    result = {}
    for dist in importlib.metadata.distributions():
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        if name in ("torch", "torchvision", "torchaudio", "triton") or name.startswith("nvidia-"):
            result[name] = dist.version
    return result


def isolated_json(code, *, env=None):
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError("Isolated setup check crashed; check the Kaggle driver/library image. No automatic repair.")
    try:
        return json.loads(result.stdout.splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError("Isolated setup check returned invalid diagnostics") from exc


def probe_device(physical):
    return isolated_json(
        """import json
r = {"stage": "torch-import", "ok": False}
try:
 import torch
 r.update(torch=str(torch.__version__), runtime=torch.version.cuda, torch_import=True)
 r["stage"] = "torchvision-import"
 import torchvision
 r.update(torchvision=str(torchvision.__version__), torchvision_import=True)
 tv = torchvision.__version__.split("+")[0].split(".")
 pt = torch.__version__.split("+")[0].split(".")
 if pt[0] == "2" and (tv[0] != "0" or int(tv[1]) != int(pt[1]) + 15):
  raise RuntimeError("Mismatched Torch/torchvision release pair")
 r["stage"] = "cuda"
 r.update(arches=torch.cuda.get_arch_list(), capability=list(torch.cuda.get_device_capability(0)))
 x = torch.ones(4, device="cuda")
 x.add_(1)
 torch.cuda.synchronize()
 r["stage"] = "torchvision-ops"
 torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]], device="cuda"), torch.ones(1, device="cuda"), 0.5)
 torch.cuda.synchronize()
 r.update(ok=True, stage="passed")
except Exception as e:
 r["error"] = str(e)
print(json.dumps(r))
""",
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(physical)},
    )


def support_imports():
    return isolated_json(f"""import importlib, json
r = {{}}
for name in {list(SUPPORT)!r}:
 try:
  module = importlib.import_module(name)
  if name == "timm" and module.__version__ != "1.0.29":
   raise RuntimeError("requires timm 1.0.29")
 except Exception as e:
  r[name] = type(e).__name__
print(json.dumps(r))
""")


def install_support(root, *, restart_loaded=False):
    before = protected_versions()
    if not {"torch", "torchvision"} <= before.keys():
        raise RuntimeError("Missing Torch/torchvision: select a coherent manual GPU profile first")
    missing = support_imports()
    stale = [name for name in missing if name in sys.modules]
    restart = bool(missing) and restart_loaded and bool(stale)
    if missing:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        if restart:
            MARKER.parent.mkdir(parents=True, exist_ok=True)
            MARKER.write_text(json.dumps({"process": process_identity(), "packages": [], "reason": "loaded-support"}))
            os.environ[GUARD] = process_identity()
        with tempfile.TemporaryDirectory(prefix="constraints-", dir=root) as owned:
            constraints = Path(owned) / "protected.txt"
            constraints.write_text("".join(f"{name}=={version}\n" for name, version in sorted(before.items())))
            print(
                f"[setup] resolving support dependencies under protected-stack constraints: {list(missing)}", flush=True
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "--constraint",
                    str(constraints),
                    *[SUPPORT[name] for name in missing],
                ],
                env={**os.environ, "PIP_CACHE_DIR": str(root / "pip"), "TMPDIR": str(root)},
            )
        if protected_versions() != before:
            raise RuntimeError("Protected GPU packages changed unexpectedly; stop and restart with a coherent image")
        if result.returncode:
            raise RuntimeError(
                "Support dependency resolution failed under Torch constraints; no blind Torch reinstall. Select a coherent manual profile."
            )
    if protected_versions() != before or support_imports():
        raise RuntimeError("Support imports/protected versions failed verification; select a coherent manual profile")
    if restart:
        raise RestartRequired(
            "Support packages changed with consumers already imported. RESTART kernel; no modules were reloaded."
        )


def bootstrap(*, smoke=False, auto_repair_p100=True, defer_ready=False):
    if smoke:
        return {"status": "synthetic-cpu-skip"}
    global _BOOTSTRAP_VERIFIED
    begin_setup()
    if not Path("/kaggle").is_dir():
        raise RuntimeError("Real bootstrap requires Kaggle; use CPU smoke locally")
    configure_caches()
    previous = json.loads(MARKER.read_text()) if MARKER.exists() else None
    if previous and previous["process"] == process_identity() or os.environ.get(GUARD) == process_identity():
        raise RestartRequired(
            "Package installation attempted in this kernel. Restart kernel, then run Setup; never reload Torch."
        )
    devices = gpu_devices()
    if not devices or any(not any(gpu in device["name"] for gpu in ("P100", "T4")) for device in devices):
        raise RuntimeError(f"Require Kaggle P100 or T4 hardware; found {[device['name'] for device in devices]}")
    reports = []
    for device in devices:
        physical, name, driver = device["physical"], device["name"], device["driver"]
        report = probe_device(physical)
        reports.append(report)
        print(f"[setup] GPU {physical} {name} driver={driver} {json.dumps(report)}", flush=True)
        p100 = "P100" in name
        arch = "sm_60" if p100 else "sm_75"
        supported = arch in report.get("arches", []) or arch.replace("sm_", "compute_") in report.get("arches", [])
        capable = report.get("capability") == ([6, 0] if p100 else [7, 5])
        if report["ok"] and supported and capable:
            continue
        error = report.get("error", "").lower()
        repairable = (
            p100
            and capable
            and report.get("stage") in ("cuda", "passed")
            and ("no kernel image" in error or (not supported and report["ok"]))
        )
        if previous or not auto_repair_p100 or not repairable:
            raise RuntimeError(
                f"GPU {physical} incompatible at {report['stage']}; no automatic repair. "
                "Select a coherent manual Torch/torchvision CUDA profile for this GPU and driver, then restart. "
                "P100 needs sm_60; T4 needs sm_75. A previous repair is never repeated."
            )
        if not (3, 9) <= sys.version_info[:2] <= (3, 12):
            raise RuntimeError(
                "P100 cu121 torch 2.5.1 wheels require Python 3.9-3.12; Python 3.13 needs another Kaggle image"
            )
        root = Path(os.environ.get("CTRL_KAGGLE_TEMP_ROOT", "/kaggle/temp/controls96"))
        root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root).free < 8 * 1024**3:
            raise OSError(
                "P100 repair needs at least 8 GiB free temp disk (may need more). Start a fresh session; no uninstall/cleanup performed."
            )
        loaded_before = [module for module in ("torch", "torchvision", "torchaudio") if module in sys.modules]
        packages = ["torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1"]
        installed = protected_versions()
        MARKER.parent.mkdir(parents=True, exist_ok=True)
        MARKER.write_text(json.dumps({"process": process_identity(), "packages": packages}))
        os.environ[GUARD] = process_identity()
        print(
            "[setup] P100 repair: installing coherent cu121 triple; no restart when this kernel never imported Torch",
            flush=True,
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                *packages,
                *(["--force-reinstall"] if installed.get("torch", "").split("+")[0] == "2.5.1" else []),
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
            ],
            env={**os.environ, "PIP_CACHE_DIR": str(root / "pip"), "TMPDIR": str(root)},
        )
        if result.returncode:
            raise RuntimeError(
                "P100 repair pip install failed; blocked and not repeated. Restart the kernel or select a coherent "
                "Python 3.9-3.12 cu121 Kaggle image; do not run D1 or training."
            )
        repaired = probe_device(physical)
        reports[-1] = repaired
        print(f"[setup] GPU {physical} re-probe after repair: {json.dumps(repaired)}", flush=True)
        repaired_ok = arch in repaired.get("arches", []) or arch.replace("sm_", "compute_") in repaired.get(
            "arches", []
        )
        if not repaired_ok:
            detail = repaired.get("error", "")
            lowered = detail.lower()
            if "torchaudio" in lowered or "torch 2.10" in lowered or "2.10.0" in lowered:
                raise RuntimeError(
                    "P100 repair left an incompatible torch/torchaudio pair; use a Kaggle image with Python 3.9-3.12 "
                    f"and cu121 wheels. Reprope: {repaired}"
                )
            raise RuntimeError(
                "P100 repair did not produce sm_60 support; use a Kaggle image with Python 3.9-3.12 and cu121 wheels. "
                f"Reprobe: {repaired}"
            )
        if loaded_before:
            raise RestartRequired(
                "P100 repair succeeded but this kernel already imported "
                + ", ".join(loaded_before)
                + ". RESTART the Kaggle kernel and rerun Setup; never reload loaded binaries."
            )
    if previous:
        versions = protected_versions()
        if any(versions.get(p.split("==")[0], "").split("+")[0] != p.split("==")[1] for p in previous["packages"]):
            raise RuntimeError(
                "P100 repair incomplete after restart; choose a fresh coherent image. No repeated install."
            )
    install_support(os.environ.get("CTRL_KAGGLE_TEMP_ROOT", "/kaggle/temp/controls96"), restart_loaded=True)
    for device in devices:
        if not probe_device(device["physical"])["ok"]:
            raise RuntimeError(
                "Native torchvision/CUDA verification failed after support setup; select a coherent manual profile"
            )
    result = {"status": "passed", "devices": reports, "protected": protected_versions()}
    MARKER.unlink(missing_ok=True)
    os.environ.pop(GUARD, None)
    _BOOTSTRAP_VERIFIED = True
    if not defer_ready:
        complete_setup()
    return result


_IMPORT_HASHES["controls_kaggle_setup"] = (str(Path(__file__).resolve()), source_hash(__file__))
