import ast
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import controls_training as ct
from scripts import final_training as ft

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "3d_glaucoma_controls_96.ipynb"


def code_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def cell(marker):
    matches = [source for source in code_cells() if marker in source]
    assert len(matches) == 1, marker
    return matches[0]


@pytest.fixture
def namespace(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("CTRL_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("CTRL_SMOKE", "1")
    monkeypatch.setattr(ft, "load_env_file", lambda: None)
    import tempfile

    original_mkdtemp = tempfile.mkdtemp
    monkeypatch.setattr(
        tempfile,
        "mkdtemp",
        lambda *args, **kwargs: (
            str(tmp_path) if kwargs.get("prefix") == "ctrl_smoke_" else original_mkdtemp(*args, **kwargs)
        ),
    )
    threads = torch.get_num_threads()
    ns = {"__name__": "controls_test"}
    exec(cell("import os, sys, subprocess"), ns)
    exec(cell("# ===== RUN IDENTITY"), ns)
    exec(cell("def validate_controls_config"), ns)
    exec(cell("def resolve_batch"), ns)
    yield ns
    torch.set_num_threads(threads)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"BS": 3}, "divide"),
        ({"GRAD_ACCUM": 1}, "manual"),
        ({"BS": 0}, "positive"),
        ({"NUM_WORKERS": -1}, "workers"),
        ({"EXTEND_EPOCHS": 1}, "RESUME"),
        ({"TARGET_VRAM_GB": float("nan")}, "finite"),
        ({"AMP_DTYPE": "float32"}, "AMP_DTYPE"),
        ({"PATIENCE": 0}, "Frozen study"),
        ({"EFFECTIVE_BATCH": 8}, "new RUN_GROUP"),
        ({"SMOKE": False}, "requires CUDA"),
    ],
)
def test_validation(namespace, changes, message):
    namespace.update(changes)
    with pytest.raises((ValueError, RuntimeError), match=message):
        namespace["validate_controls_config"]()


def test_cuda_defaults_and_validation(namespace, monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=80_000_000_000)
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    namespace.update(SMOKE=False, DEVICE=torch.device("cuda"))
    exec(cell("# ===== RUN IDENTITY"), namespace)
    assert namespace["AMP_DTYPE"] == "bfloat16"
    assert namespace["TARGET_VRAM_GB"] == 72
    assert namespace["EFFECTIVE_BATCH"] == 16
    assert namespace["SEEDS"] == [42, 43, 44]
    namespace["validate_controls_config"]()
    namespace.update(EFFECTIVE_BATCH=64)
    with pytest.raises(ValueError, match="new RUN_GROUP"):
        namespace["validate_controls_config"]()
    namespace.update(RUN_GROUP="controls_explicit_effective64")
    namespace["validate_controls_config"]()
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(ValueError, match="unsupported"):
        namespace["validate_controls_config"]()
    exec(cell("# ===== RUN IDENTITY"), namespace)
    assert namespace["AMP_DTYPE"] == "float16"


def test_probe_uses_exact_training_config(namespace, monkeypatch):
    ns = namespace
    ns.update(SMOKE=False, AUTO_BATCH=True, DEVICE=torch.device("cuda"), EFFECTIVE_BATCH=16, AMP_DTYPE="bfloat16")
    monkeypatch.setattr(ft, "saved_config", lambda *roots: None)
    monkeypatch.setattr(ft, "data_identity", lambda path: str(path))
    factories, calls = [], []
    monkeypatch.setattr(ns["cm"], "ControlsModel", lambda **kwargs: factories.append(kwargs))
    data = SimpleNamespace(labels=np.array([0, 1, 1]), source=Path("volumes"), label_path=Path("labels"))

    def probe(factory, dataset, **kwargs):
        calls.append(kwargs)
        factory()
        candidate = kwargs["candidates"][0]
        trial = {"batch_size": candidate, "peak_gb": 1.0, "ok": candidate == 1}
        if candidate != 1:
            raise RuntimeError(f"No candidate batch fits the VRAM budget; trials={[trial]}")
        return {"batch_size": 1, "peak_gb": 1.0, "trials": [trial]}

    monkeypatch.setattr(ft, "find_batch_size", probe)
    spec = ns["SPECS"][0]
    bs, accum, result = ns["resolve_batch"]("P_s42", data, spec, 42, data, data)
    assert (bs, accum) == (1, 16)
    assert [c["candidates"] for c in calls] == [[16], [8], [4], [2], [1]]
    assert len(result["trials"]) == 5
    for call, factory in zip(calls, factories):
        config = call["config"]
        assert config["grad_accum"] * call["candidates"][0] == 16
        assert config["amp_dtype"] == "bfloat16"
        assert config["lr"] == config["weight_decay"] == 1e-4
        assert config["class_weights"] == [1.5, 0.75]
        assert call["max_batch"] == 16
        assert factory["enc2d_pretrained"] is False
        for key in ("n_2d", "view_indices", "fusion", "use_3d", "gate_fixed"):
            assert factory[key] == spec[key]

    def unsafe(*args, **kwargs):
        raise RuntimeError("No candidate batch fits the VRAM budget; trials=[]")

    monkeypatch.setattr(ft, "find_batch_size", unsafe)
    with pytest.raises(RuntimeError, match="No safe microbatch"):
        ns["resolve_batch"]("P_s42", data, spec, 42, data, data)


def test_saved_identity_never_probes(namespace, monkeypatch):
    ns = namespace
    saved = {"batch_size": 1, "grad_accum": 4, "amp_dtype": "float16"}
    ft.atomic_save({"config": saved}, ns["LOCAL_ROOT"] / "P_s42" / "run_identity.pt")
    monkeypatch.setattr(ft, "find_batch_size", lambda *a, **kw: pytest.fail("Unexpected probe"))
    ns.update(RESUME=False, AUTO_BATCH=True)
    assert ns["resolve_batch"]("P_s42", None, ns["SPECS"][0], 42, None, None) == (1, 4, None)
    ns["AMP_DTYPE"] = "bfloat16"
    with pytest.raises(ValueError, match="CTRL_AMP_DTYPE"):
        ns["resolve_batch"]("P_s42", None, ns["SPECS"][0], 42, None, None)


@pytest.mark.parametrize("failure", [False, True])
def test_stop_failure_cleanup_preserves_checkpoint(namespace, monkeypatch, failure):
    ns = namespace
    for marker in (
        "rng = np.random.default_rng(0)",
        "views_path, depth_path",
        "def make_datasets",
        "def make_model",
        "METRIC_KEYS =",
        "def finish_tag",
    ):
        exec(cell(marker), ns)
    run = SimpleNamespace(id="controls-test", summary={}, logs=[], finishes=[])
    run.log = lambda values: run.logs.append(values)
    run.finish = lambda exit_code: run.finishes.append(exit_code)

    def init(name, config, artifacts, **kwargs):
        artifacts.save({"id": run.id, "config": config}, "run_identity.pt")
        return run

    original_fit = ct.Trainer.fit

    def interrupted(trainer, evaluate, **kwargs):
        def boundary(current):
            if failure:
                raise RuntimeError("injected training failure")
            current.stop_requested = True

        return original_fit(trainer, evaluate, boundary_hook=boundary, **kwargs)

    monkeypatch.setattr(ft, "init_wandb", init)
    monkeypatch.setattr(ct.Trainer, "fit", interrupted)
    if failure:
        with pytest.raises(RuntimeError, match="injected training failure"):
            exec(cell("RESULTS = {}"), ns)
    else:
        exec(cell("RESULTS = {}"), ns)
    assert run.finishes == [1 if failure else 0]
    assert run.summary["status"] == ("failure" if failure else "stopped")
    assert ns["progress"]["trainer"] is ns["ACTIVE_TRAINER"] is ns["ACTIVE_MODEL"] is ns["WANDB_RUN"] is None
    assert (ns["LOCAL_ROOT"] / "P_s42" / "last.pt").is_file()
    assert not (ns["LOCAL_ROOT"] / "P_s42" / "completed.pt").exists()
    assert not (ns["LOCAL_ROOT"] / "B1_s42").exists()


def test_epoch_timer_partial_resume(namespace, capsys):
    ns = namespace
    tree = ast.parse(cell("RESULTS = {}"))
    timer = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "epoch_timer")
    logs = []
    ns.update(timer={"epoch": 3, "started": ns["time"].monotonic() - 2, "offset": 4}, tr=list(range(10)))
    exec(compile(ast.Module(body=[timer], type_ignores=[]), "<epoch_timer>", "exec"), ns)
    trainer = SimpleNamespace(epoch=4, step=12, run=SimpleNamespace(log=lambda values: logs.append(values)))
    ns["epoch_timer"](trainer)
    assert logs[0]["train/epoch_partial_resume"] is True
    assert logs[0]["train/epoch_seconds"] * logs[0]["train/samples_per_second"] == pytest.approx(6)
    assert logs[0]["train/peak_reserved_gib"] == 0
    assert ns["timer"]["offset"] == 0
    assert "epoch 4" in capsys.readouterr().out


def test_full_offline_seven_spec_smoke(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith(("CTRL_", "WANDB_", "HF_"))}
    env.update(CTRL_SMOKE="1", WANDB_MODE="offline", WANDB_SILENT="true", PYTHONIOENCODING="utf-8")
    env["PYTHONPATH"] = str(NOTEBOOK.parents[1])
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
        cwd=NOTEBOOK.parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CONTROLS SEVEN-SPEC SMOKE OK" in result.stdout


def offline_smoke(root):
    import tempfile

    sys.path.insert(0, str(NOTEBOOK.parents[1]))
    original_mkdtemp = tempfile.mkdtemp
    tempfile.mkdtemp = lambda *args, **kwargs: (
        str(root) if kwargs.get("prefix") == "ctrl_smoke_" else original_mkdtemp(*args, **kwargs)
    )
    ft.load_env_file = lambda: None
    socket.create_connection = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Network forbidden"))
    import huggingface_hub

    import wandb

    def forbidden_download(*args, **kwargs):
        raise AssertionError("HF download forbidden in smoke")

    huggingface_hub.snapshot_download = huggingface_hub.hf_hub_download = forbidden_download

    runs = []
    original_init = wandb.init

    def offline_init(*args, **kwargs):
        assert kwargs.get("mode") == "offline"
        run = original_init(*args, **kwargs)
        runs.append(run)
        return run

    wandb.init = offline_init
    ns = {"__name__": "controls_smoke"}
    original_fit = ct.Trainer.fit
    original_predict = ft.predict
    original_report = ct.calibrated_report
    epochs = []

    def fit(trainer, evaluate, **kwargs):
        assert "boundary_hook" not in kwargs
        hook = kwargs["epoch_hook"]

        def checked(current):
            checkpoint = torch.load(current.artifacts.local / "last.pt", weights_only=False)
            assert checkpoint["epoch"] == current.epoch
            assert "test" in current.history[-1]
            assert not any("seconds" in key for key in current.history[-1])
            epochs.append(current.epoch)
            hook(current)

        return original_fit(trainer, evaluate, **{**kwargs, "epoch_hook": checked})

    def predict(*args, **kwargs):
        assert kwargs["amp_dtype"] == ns["AMP_DTYPE"]
        return original_predict(*args, **kwargs)

    def report(*args, **kwargs):
        assert kwargs["amp_dtype"] == ns["AMP_DTYPE"]
        return original_report(*args, **kwargs)

    ct.Trainer.fit, ft.predict, ct.calibrated_report = fit, predict, report
    for source in code_cells():
        exec(compile(source, "<controls-cell>", "exec"), ns)
    assert set(ns["RESULTS"]) == {f"{code}_s42" for code in ("P", "B1", "B2", "B3", "C1", "C2", "B4")}
    assert epochs == [1] * 7
    assert ns["progress"]["trainer"] is ns["ACTIVE_TRAINER"] is ns["ACTIVE_MODEL"] is None
    assert len(runs) == 8 and all(run._is_finished for run in runs)
    summary = json.loads((ns["STORAGE"].local / "controls_96_summary.json").read_text())
    assert len(summary) == 7
    assert all(item["n_seeds"] == 1 for item in summary.values())
    for tag in ns["RESULTS"]:
        assert (ns["LOCAL_ROOT"] / tag / "completed.pt").is_file()
    assert (ns["LOCAL_ROOT"] / "B4_s42" / "run_identity.pt").is_file()
    ns["RESUME"] = True
    ft.find_batch_size = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Completed run re-probed"))
    exec(cell("RESULTS = {}"), ns)
    assert len(runs) == 8
    print("CONTROLS SEVEN-SPEC SMOKE OK")


if __name__ == "__main__":
    offline_smoke(Path(sys.argv[1]))
