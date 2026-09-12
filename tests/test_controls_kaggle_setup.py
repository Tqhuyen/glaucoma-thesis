import builtins
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import controls_kaggle_setup as ks

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(ks, "MARKER", tmp_path / "restart.json")
    monkeypatch.setenv(ks.GUARD, "")
    monkeypatch.setenv(ks.PENDING, "")
    monkeypatch.setattr(ks, "_BOOTSTRAP_VERIFIED", False)
    monkeypatch.setenv("CTRL_KAGGLE_TEMP_ROOT", str(tmp_path))
    monkeypatch.setattr(ks, "configure_caches", lambda *a: None)
    is_dir = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda p: p == Path("/kaggle") or is_dir(p))
    monkeypatch.setattr(ks, "process_identity", lambda: "first-kernel")
    monkeypatch.setattr(ks.sys, "version_info", (3, 12, 0))
    versions = {"torch": "2.9.0", "torchvision": "0.24.0", "torchaudio": "2.9.0", "nvidia-cublas-cu12": "12.4"}
    monkeypatch.setattr(ks, "protected_versions", lambda: dict(versions))
    monkeypatch.setattr(ks.shutil, "disk_usage", lambda path: SimpleNamespace(free=20 * 1024**3))
    calls = []
    state = SimpleNamespace(name="Tesla P100", calls=calls, versions=versions, support=[])

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == "nvidia-smi":
            return SimpleNamespace(returncode=0, stdout=f"0, {state.name}, 550.54\n")
        assert command[:4] == [sys.executable, "-m", "pip", "install"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ks.subprocess, "run", run)
    monkeypatch.setattr(ks, "install_support", lambda root, **kwargs: state.support.append(root))
    return state


def report(*, p100=True, supported=True, ok=True, stage="passed", error=""):
    return dict(
        ok=ok,
        stage=stage,
        error=error,
        torch="2.5.1+cu121",
        torchvision="0.20.1+cu121",
        runtime="12.1",
        capability=[6, 0] if p100 else [7, 5],
        arches=["sm_60" if p100 else "sm_75"] if supported else ["sm_80"],
    )


def _clear_loaded_runtime(monkeypatch):
    for module in ("torch", "torchvision", "torchaudio"):
        monkeypatch.delitem(sys.modules, module, raising=False)


def _repair_then_supported(probes):
    def probe(_):
        probes.append(1)
        return report() if len(probes) > 1 else report(supported=False, ok=True, stage="passed")

    return probe


@pytest.mark.parametrize("kernel_loaded", [False, True])
def test_p100_repair_fresh_continues_loaded_restarts(setup_runtime, monkeypatch, kernel_loaded):
    _clear_loaded_runtime(monkeypatch)
    if kernel_loaded:
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    probes = []
    monkeypatch.setattr(ks, "probe_device", _repair_then_supported(probes))
    if kernel_loaded:
        with pytest.raises(ks.RestartRequired, match="RESTART"):
            ks.bootstrap()
        assert ks.MARKER.exists() and os.environ[ks.GUARD] == "first-kernel"
    else:
        assert ks.bootstrap()["status"] == "passed"
        assert not ks.MARKER.exists() and not os.environ.get(ks.GUARD)
    pip, options = setup_runtime.calls[-1]
    assert pip[4:] == [
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        "--index-url",
        "https://download.pytorch.org/whl/cu121",
    ]
    assert "pip" in options["env"]["PIP_CACHE_DIR"]
    assert "LD_LIBRARY_PATH" not in {k for k in options["env"] if k not in os.environ}


def test_p100_repair_missing_sm60_after_pip_is_runtime_error(setup_runtime, monkeypatch):
    _clear_loaded_runtime(monkeypatch)
    monkeypatch.setattr(ks, "probe_device", lambda _: report(supported=False, ok=True, stage="passed"))
    with pytest.raises(RuntimeError, match="sm_60"):
        ks.bootstrap()
    assert ks.MARKER.exists()


def test_p100_repair_pip_failure_blocks_and_never_repeats(setup_runtime, monkeypatch):
    _clear_loaded_runtime(monkeypatch)
    monkeypatch.setattr(ks, "probe_device", lambda _: report(supported=False, ok=True, stage="passed"))
    run = ks.subprocess.run

    def fail(command, **kwargs):
        result = run(command, **kwargs)
        if "pip" in command:
            return SimpleNamespace(returncode=1)
        return result

    monkeypatch.setattr(ks.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="pip install failed"):
        ks.bootstrap()
    assert ks.MARKER.exists()
    assert len([command for command, _ in setup_runtime.calls if "pip" in command]) == 1
    with pytest.raises(ks.RestartRequired):
        ks.bootstrap()
    assert len([command for command, _ in setup_runtime.calls if "pip" in command]) == 1


def test_p100_repair_replaces_torchaudio_210_with_triple(setup_runtime, monkeypatch):
    _clear_loaded_runtime(monkeypatch)
    setup_runtime.versions.update(torch="2.10.0", torchvision="0.25.0", torchaudio="2.10.0")
    monkeypatch.setattr(ks, "probe_device", _repair_then_supported([]))
    assert ks.bootstrap()["status"] == "passed"
    pip = next(command for command, _ in setup_runtime.calls if "pip" in command)
    assert pip[4:] == [
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        "--index-url",
        "https://download.pytorch.org/whl/cu121",
    ]
    assert not any("2.10" in token for token in pip)


def test_helper_import_is_torch_free():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from scripts import controls_kaggle_setup; assert 'torch' not in sys.modules; assert 'wandb' not in sys.modules; assert 'huggingface_hub' not in sys.modules",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("gpu", ["P100", "T4"])
def test_working_stack_never_replaced(setup_runtime, monkeypatch, gpu):
    setup_runtime.name = gpu
    probes = []
    monkeypatch.setattr(ks, "probe_device", lambda p: probes.append(p) or report(p100=gpu == "P100"))
    assert ks.bootstrap()["status"] == "passed"
    assert len(probes) == 2
    assert all(command[0] == "nvidia-smi" for command, _ in setup_runtime.calls)
    assert setup_runtime.support


@pytest.mark.parametrize(
    "gpu,stage,error",
    [
        ("T4", "cuda", "no kernel image"),
        ("P100", "torchvision-import", "operator torchvision::nms does not exist"),
        ("P100", "torch-import", "undefined symbol"),
        ("P100", "cuda", "driver version is insufficient"),
    ],
)
def test_errors_not_blindly_repaired(setup_runtime, monkeypatch, gpu, stage, error):
    setup_runtime.name = gpu
    monkeypatch.setattr(
        ks, "probe_device", lambda _: report(p100=gpu == "P100", supported=False, ok=False, stage=stage, error=error)
    )
    with pytest.raises(RuntimeError, match="coherent manual"):
        ks.bootstrap()
    assert len(setup_runtime.calls) == 1 and not ks.MARKER.exists()


@pytest.mark.parametrize("problem", ["python", "disk", "disabled"])
def test_repair_preconditions(setup_runtime, monkeypatch, problem):
    monkeypatch.setattr(ks, "probe_device", lambda _: report(supported=False))
    if problem == "python":
        monkeypatch.setattr(ks.sys, "version_info", (3, 13, 0))
    if problem == "disk":
        monkeypatch.setattr(ks.shutil, "disk_usage", lambda p: SimpleNamespace(free=6 * 1024**3))
    with pytest.raises((RuntimeError, OSError)):
        ks.bootstrap(auto_repair_p100=problem != "disabled")
    assert not ks.MARKER.exists() and len(setup_runtime.calls) == 1


def test_smoke_no_subprocess_or_environment_work(monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("Smoke must return before subprocess, paths, CUDA, pip or cache changes")

    monkeypatch.setattr(ks.subprocess, "run", forbidden)
    monkeypatch.setattr(ks, "configure_caches", forbidden)
    monkeypatch.setattr(ks, "probe_device", forbidden)
    monkeypatch.setattr(Path, "is_dir", forbidden)
    monkeypatch.setenv(ks.GUARD, "pending-restart")
    assert ks.bootstrap(smoke=True) == {"status": "synthetic-cpu-skip"}


def test_probe_isolated_not_parent(monkeypatch):
    calls = []

    def inspect(code, *, env):
        compile(code, "<isolated-check>", "exec")
        calls.append((code, env))
        return {"ok": True}

    monkeypatch.setattr(ks, "isolated_json", inspect)
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert ks.probe_device("1") == {"ok": True}
    code, env = calls[0]
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert "torchvision.ops.nms" in code and "torch.cuda.synchronize" in code
    assert "get_arch_list" in code and "get_device_capability" in code
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == original


def test_cache_environment_overrides(tmp_path, monkeypatch):
    keys = ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "TORCH_HOME", "WANDB_CACHE_DIR", "WANDB_DATA_DIR", "WANDB_DIR")
    for key in keys:
        monkeypatch.setenv(key, str(tmp_path / key))
    ks.configure_caches({"temp_root": str(tmp_path / "temp"), "output_root": str(tmp_path / "working")})
    assert all(os.environ[key] == str(tmp_path / key) and Path(os.environ[key]).is_dir() for key in keys)


def test_worker_output_redacts_before_persisting(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-private-value")
    monkeypatch.setenv("WANDB_API_KEY", "wandb-private-value")
    stream = io.StringIO()
    output = ks.RedactedOutput(stream)
    output.write("token=hf-private-")
    output.flush()
    assert stream.getvalue() == ""
    output.write("value key=wandb-private-value\n")
    assert stream.getvalue() == "token=[REDACTED] key=[REDACTED]\n"


def test_failed_setup_blocks_dependent_stages(setup_runtime, monkeypatch):
    monkeypatch.setattr(ks, "probe_device", lambda _: report(ok=False, stage="torch-import", error="library error"))
    with pytest.raises(RuntimeError, match="coherent manual"):
        ks.bootstrap()
    with pytest.raises(RuntimeError, match="Setup has not passed"):
        ks.require_ready()


@pytest.fixture
def notebook_setup(setup_runtime, monkeypatch):
    from scripts import controls_kaggle as kg
    from scripts import controls_kaggle_data as kd

    monkeypatch.setenv("CTRL_KAGGLE_SMOKE", "0")
    monkeypatch.setenv("CTRL_KAGGLE_REPO_ROOT", str(ROOT))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(ks, "probe_device", lambda _: report())
    validate = ks.validate_repo
    monkeypatch.setattr(ks, "validate_repo", lambda root, **kwargs: validate(root, display=False))

    def credentials(**kwargs):
        assert os.environ[ks.PENDING] == "1"

    monkeypatch.setattr(kg, "load_credentials", credentials)
    notebook = json.loads((ROOT / "notebooks/3d_glaucoma_controls_96_kaggle.ipynb").read_text())
    cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    code = compile("".join(cell["source"]), "<real-setup-mocked>", "exec")
    return SimpleNamespace(code=code, namespace={"kg": kg, "kd": kd}, runtime=setup_runtime)


def assert_pending_blocks_old_globals(namespace):
    for operation in (
        ks.require_ready,
        lambda: namespace["kd"].prepare_source({"smoke": False}, "raw"),
        lambda: namespace["kg"].dispatch({"smoke": False}, {}),
    ):
        with pytest.raises(RuntimeError, match="Setup has not passed"):
            operation()


def test_success_then_source_change_failed_setup_blocks_old_globals(notebook_setup, monkeypatch):
    setup = notebook_setup
    exec(setup.code, setup.namespace)
    ks.require_ready()
    source_hash = ks.source_hash
    calls = len(setup.runtime.calls)
    with monkeypatch.context() as changed:
        changed.setattr(
            ks, "source_hash", lambda path: "0" * 64 if Path(path).name == "controls_kaggle.py" else source_hash(path)
        )
        with pytest.raises(RuntimeError, match="Stale imported"):
            exec(setup.code, setup.namespace)
        assert len(setup.runtime.calls) == calls
        assert_pending_blocks_old_globals(setup.namespace)
    exec(setup.code, setup.namespace)
    ks.require_ready()
    assert ks.PENDING not in os.environ


@pytest.mark.parametrize(
    "failure",
    ["discovery", "missing-helper", "bootstrap-import", "api", "helper-import", "credentials", "post-import-repo"],
)
def test_notebook_setup_any_failure_keeps_old_globals_blocked(notebook_setup, monkeypatch, failure):
    setup = notebook_setup
    exec(setup.code, setup.namespace)
    ks.require_ready()
    calls = len(setup.runtime.calls)

    def fail(*args, **kwargs):
        assert os.environ[ks.PENDING] == "1"
        raise RuntimeError("injected setup failure")

    with monkeypatch.context() as broken:
        if failure == "discovery":
            broken.setattr(Path, "resolve", fail)
        elif failure == "missing-helper":
            is_file = Path.is_file
            broken.setattr(
                Path, "is_file", lambda path: False if path.name == "controls_kaggle_setup.py" else is_file(path)
            )
        elif failure == "bootstrap-import":
            broken.delitem(sys.modules, "scripts.controls_kaggle_setup")
            broken.setattr(
                importlib.util,
                "spec_from_file_location",
                lambda *a: SimpleNamespace(
                    name="scripts.controls_kaggle_setup", loader=SimpleNamespace(exec_module=fail)
                ),
            )
            broken.setattr(importlib.util, "module_from_spec", lambda spec: SimpleNamespace())
        elif failure == "api":
            broken.setattr(ks, "BOOTSTRAP_API_VERSION", -1)
        elif failure == "helper-import":
            original = builtins.__import__

            def importing(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "scripts" and "controls_kaggle" in fromlist:
                    fail()
                return original(name, globals, locals, fromlist, level)

            broken.setattr(builtins, "__import__", importing)
        elif failure == "credentials":
            broken.setattr(setup.namespace["kg"], "load_credentials", fail)
        else:
            validate = ks.validate_repo
            validations = []

            def validate_twice(*args, **kwargs):
                validations.append(1)
                if len(validations) == 2:
                    fail()
                return validate(*args, **kwargs)

            broken.setattr(ks, "validate_repo", validate_twice)
        with pytest.raises(RuntimeError):
            exec(setup.code, setup.namespace)
        assert_pending_blocks_old_globals(setup.namespace)
        if failure in ("helper-import", "credentials", "post-import-repo"):
            assert len(setup.runtime.calls) > calls and ks._BOOTSTRAP_VERIFIED
        else:
            assert len(setup.runtime.calls) == calls
    exec(setup.code, setup.namespace)
    ks.require_ready()


def test_deferred_bootstrap_only_complete_setup_enables_stages(setup_runtime, monkeypatch):
    monkeypatch.setattr(ks, "probe_device", lambda _: report())
    ks.begin_setup()
    with pytest.raises(RuntimeError, match="bootstrap has not passed"):
        ks.complete_setup()
    with pytest.raises(RuntimeError, match="Setup has not passed"):
        ks.require_ready(importing=True)
    assert ks.bootstrap(defer_ready=True)["status"] == "passed"
    ks.require_ready(importing=True)
    with pytest.raises(RuntimeError, match="Setup has not passed"):
        ks.require_ready()
    ks.complete_setup()
    ks.require_ready()
    ks.begin_setup()
    with pytest.raises(RuntimeError, match="bootstrap has not passed"):
        ks.complete_setup()


def test_smoke_setup_lifecycle_does_not_touch_pending_state(monkeypatch):
    monkeypatch.setenv(ks.PENDING, "keep-pending")
    monkeypatch.setattr(ks, "_BOOTSTRAP_VERIFIED", True)
    ks.begin_setup(smoke=True)
    ks.bootstrap(smoke=True, defer_ready=True)
    ks.complete_setup(smoke=True)
    assert os.environ[ks.PENDING] == "keep-pending" and ks._BOOTSTRAP_VERIFIED


@pytest.mark.parametrize("result", ["success", "conflict", "mutation", "bad-import"])
def test_support_constraints_transitive_and_verification(tmp_path, monkeypatch, result):
    protected = {
        "torch": "2.5.1+cu121",
        "torchvision": "0.20.1+cu121",
        "torchaudio": "2.5.1+cu121",
        "triton": "3.1.0",
        "nvidia-cublas-cu12": "12.1.3.1",
    }
    monkeypatch.setattr(ks, "protected_versions", lambda: dict(protected))
    inspections = iter(
        [{"timm": "wrong-version", "scipy": "missing"}, {"timm": "failed"} if result == "bad-import" else {}]
    )
    monkeypatch.setattr(ks, "support_imports", lambda: next(inspections))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert "--no-deps" not in command
        path = Path(command[command.index("--constraint") + 1])
        assert set(path.read_text().splitlines()) == {f"{key}=={value}" for key, value in protected.items()}
        assert command[-2:] == ["timm==1.0.29", "scipy"]
        assert "torch==" not in " ".join(command)
        if result == "mutation":
            protected["torch"] = "3.0.0"
        return SimpleNamespace(returncode=1 if result == "conflict" else 0)

    monkeypatch.setattr(ks.subprocess, "run", run)
    if result == "success":
        ks.install_support(tmp_path)
    else:
        with pytest.raises(RuntimeError):
            ks.install_support(tmp_path)
    assert len(calls) == 1 and not list(tmp_path.glob("constraints-*"))


def test_protected_inventory_all_nvidia_names(monkeypatch):
    monkeypatch.setattr(
        ks.importlib.metadata,
        "distributions",
        lambda: [
            SimpleNamespace(metadata={"Name": name}, version="1.0")
            for name in ("torch", "torchvision", "triton", "nvidia_cudnn_cu12", "NVIDIA-CUBLAS-CU12", "numpy")
        ],
    )
    assert set(ks.protected_versions()) == {"torch", "torchvision", "triton", "nvidia-cudnn-cu12", "nvidia-cublas-cu12"}


def test_support_install_resident_numpy_does_not_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(ks, "MARKER", tmp_path / "restart.json")
    monkeypatch.setenv(ks.GUARD, "")
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace())
    monkeypatch.setattr(ks, "protected_versions", lambda: {"torch": "2.5.1", "torchvision": "0.20.1"})
    inspections = iter([{"timm": "wrong-version"}, {}])
    monkeypatch.setattr(ks, "support_imports", lambda: next(inspections))
    monkeypatch.setattr(ks.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    ks.install_support(tmp_path, restart_loaded=True)
    assert not ks.MARKER.exists() and not os.environ.get(ks.GUARD)


def test_support_install_loaded_support_module_restarts(tmp_path, monkeypatch):
    monkeypatch.setattr(ks, "MARKER", tmp_path / "restart.json")
    monkeypatch.setenv(ks.GUARD, "")
    monkeypatch.setitem(sys.modules, "timm", SimpleNamespace())
    monkeypatch.setattr(ks, "protected_versions", lambda: {"torch": "2.5.1", "torchvision": "0.20.1"})
    inspections = iter([{"timm": "wrong-version"}, {}])
    monkeypatch.setattr(ks, "support_imports", lambda: next(inspections))
    monkeypatch.setattr(ks.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    with pytest.raises(ks.RestartRequired, match="Support packages changed"):
        ks.install_support(tmp_path, restart_loaded=True)
    assert ks.MARKER.exists()


def test_discovery_explicit_and_module_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CTRL_KAGGLE_REPO_ROOT", raising=False)
    assert ks.discover_repo() == ROOT
    monkeypatch.setenv("CTRL_KAGGLE_REPO_ROOT", str(tmp_path / "explicit"))
    assert ks.discover_repo() == tmp_path / "explicit"
    with pytest.raises(RuntimeError, match="missing"):
        ks.validate_repo(ks.discover_repo(), display=False)


@pytest.mark.parametrize("problem", ["checkout", "changed", "untracked"])
def test_stale_import_rejected(monkeypatch, problem):
    path = ROOT / "scripts/controls_kaggle.py"
    module = SimpleNamespace(__file__=str(path if problem != "checkout" else path.parent.parent / "elsewhere.py"))
    monkeypatch.setitem(sys.modules, "scripts.controls_kaggle", module)
    hashes = dict(ks._IMPORT_HASHES)
    hashes["controls_kaggle"] = (str(path), "old-hash")
    if problem == "untracked":
        hashes.pop("controls_kaggle")
    monkeypatch.setattr(ks, "_IMPORT_HASHES", hashes)
    with pytest.raises(RuntimeError, match="Stale imported"):
        ks.validate_repo(ROOT, display=False)


def test_restart_blocks_old_global_factories(tmp_path, monkeypatch):
    from scripts import controls_kaggle as kg
    from scripts import controls_kaggle_data as kd

    monkeypatch.setattr(ks, "MARKER", tmp_path / "restart.json")
    ks.MARKER.write_text("{}")
    for operation in (
        lambda: kd.prepare_source({"smoke": False}, "raw"),
        lambda: kd.prepare_views({"smoke": False}, {}),
        lambda: kg.dispatch({"smoke": False}, {}),
        lambda: kg.worker({"smoke": False}, {}, "P_s42", deadline=0, stop_path="unused"),
        lambda: kg.make_model({}, smoke=False),
        lambda: kg.probe_batch(None, None, {}),
    ):
        with pytest.raises(ks.RestartRequired):
            operation()


def test_two_real_cpu_children_log_relay_bounded_no_pipes(tmp_path, capsys):
    from scripts import controls_kaggle as kg

    children = []
    try:
        for gpu in (0, 1):
            path = tmp_path / f"worker{gpu}.log"
            with path.open("wb", buffering=0) as log:
                child = subprocess.Popen(
                    [sys.executable, "-u", "-c", "for i in range(3000): print('x'*100)\nprint('FINAL')"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            children.append(child)
            kg.OWNED_LOGS[child.pid] = dict(path=path, offset=0, partial="", tag=f"job{gpu}", physical=str(gpu))
        until = time.monotonic() + 30
        while any(child.pid in kg.OWNED_LOGS for child in children) and time.monotonic() < until:
            for child in children:
                old = kg.OWNED_LOGS.get(child.pid, {}).get("offset", 0)
                kg.relay_log(child.pid, finished=child.poll() is not None)
                state = kg.OWNED_LOGS.get(child.pid)
                if state:
                    assert state["offset"] - old <= 65536
            time.sleep(0.01)
        assert all(child.wait(timeout=5) == 0 for child in children)
        assert not any(child.pid in kg.OWNED_LOGS for child in children)
        output = capsys.readouterr().out
        assert "[job0 GPU 0] FINAL" in output and "[job1 GPU 1] FINAL" in output
        assert all((tmp_path / f"worker{gpu}.log").stat().st_size > 300000 for gpu in (0, 1))
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
            kg.OWNED_LOGS.pop(child.pid, None)


def test_stop_retains_worker_failure(tmp_path):
    from scripts import controls_kaggle as kg

    kg.OWNED_CHILDREN[998877] = SimpleNamespace(poll=lambda: 7, wait=lambda: 7)
    try:
        with pytest.raises(RuntimeError, match="998877, 7"):
            kg.wait_owned(tmp_path / "STOP.json", 5)
        assert not kg.OWNED_CHILDREN
    finally:
        kg.OWNED_CHILDREN.pop(998877, None)
