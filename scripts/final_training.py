"""Training recovery for the final notebook; only committed optimizer boundaries resume."""

import contextlib
import copy
import csv
import gc
import hashlib
import io
import json
import math
import os
import random
import shutil
import signal
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from pipeline.utils import load_env_file


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Artifacts:
    def __init__(self, local, remote=None, *, smoke=False):
        self.local = Path(local)
        self.remote = Path(remote) if remote else None
        self.local.mkdir(parents=True, exist_ok=True)
        if not smoke:
            if self.remote is None or not self.remote.parent.is_dir():
                raise RuntimeError("Verified Drive parent must exist before training")
            if self.local.resolve() == self.remote.resolve():
                raise ValueError("Local checkpoints must be separate from Drive")
        if self.remote:
            self.remote.mkdir(parents=True, exist_ok=True)
        self.smoke = smoke
        self.save({"smoke": smoke, "remote": str(self.remote)}, "storage_probe.pt")

    def sync(self, path):
        path = Path(path)
        if self.remote is None:
            if not self.smoke:
                raise RuntimeError("Drive sync unavailable")
            return
        target = self.remote / path.relative_to(self.local)
        temporary = target.with_name(target.name + ".tmp-" + uuid.uuid4().hex)
        pending = path.with_name(path.name + ".sync-pending.pt")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, temporary)
            with temporary.open("rb+") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            pending.unlink(missing_ok=True)
        except Exception as exc:
            atomic_save({"source": str(path), "target": str(target), "error": str(exc)}, pending)
            raise RuntimeError(f"Drive sync FAILED; local artifact retained: {path}; pending: {pending}") from exc
        finally:
            temporary.unlink(missing_ok=True)

    def save(self, value, name):
        path = self.local / name
        atomic_save(value, path)
        self.sync(path)
        return path


def _config_without_epochs(config):
    return {key: value for key, value in config.items() if key != "epochs"}


def _config_compatible(old, new):
    return (
        isinstance(old.get("epochs"), int)
        and not isinstance(old.get("epochs"), bool)
        and isinstance(new.get("epochs"), int)
        and not isinstance(new.get("epochs"), bool)
        and _config_without_epochs(old) == _config_without_epochs(new)
        and new["epochs"] >= old["epochs"]
    )


def run_status(artifacts, config, *, resume, extend_epochs=0):
    if not isinstance(config.get("epochs"), int) or isinstance(config.get("epochs"), bool) or config["epochs"] < 1:
        raise ValueError("epochs must be a positive integer")
    if isinstance(extend_epochs, bool) or not isinstance(extend_epochs, int) or extend_epochs < 0:
        raise ValueError("extend_epochs must be a nonnegative integer")
    if extend_epochs and not resume:
        raise ValueError("Epoch extension requires explicit resume")

    def locate(name):
        local = artifacts.local / name
        if local.exists():
            return local
        return artifacts.remote / name if artifacts.remote else local

    identity = locate("run_identity.pt")
    checkpoint = locate("last.pt")
    if not identity.exists():
        if extend_epochs:
            raise ValueError("Epoch extension requires an existing checkpoint and identity")
        if checkpoint.exists() or locate("completed.pt").exists() or locate("metrics.json").exists():
            raise RuntimeError("Run artifacts exist without identity; refusing to initialize over orphaned state")
        return {"status": "new", "warm_start": "", "config": config}
    if not resume:
        raise FileExistsError("Run already exists: enable RESUME or choose a new RUN_GROUP")
    state = torch.load(identity, map_location="cpu", weights_only=False)
    if _config_without_epochs(state["config"]) != _config_without_epochs(config):
        raise ValueError("Run config mismatch; use a new run directory")
    if not extend_epochs and int(config["epochs"]) > int(state["config"]["epochs"]):
        raise ValueError("Increasing epochs requires explicit extend_epochs")
    target = int(state["config"]["epochs"]) + int(extend_epochs)
    if extend_epochs:
        if not checkpoint.exists():
            raise ValueError("Epoch extension requires an existing safe checkpoint")
        previous = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if previous["run_id"] != state["id"] or not _config_compatible(previous["config"], state["config"]):
            raise ValueError("Extension checkpoint identity/config mismatch")
        state["epoch_extension"] = {"from_epochs": previous["config"]["epochs"], "to_epochs": target}
    if state["config"] != config or int(config["epochs"]) != target:
        extended = dict(config)
        extended["epochs"] = target
        state["config"] = extended
        artifacts.save(state, "run_identity.pt")
        config = extended
    else:
        config = state["config"]
    if extend_epochs > 0:
        for path in (artifacts.local / "completed.pt", artifacts.remote / "completed.pt" if artifacts.remote else None):
            if path is not None:
                path.unlink(missing_ok=True)
    complete = locate("completed.pt")
    if complete.exists():
        marker = torch.load(complete, map_location="cpu", weights_only=False)
        if marker["run_id"] != state["id"] or marker["config"] != config:
            raise ValueError("Completion marker identity/config mismatch")
        if all(locate(name).exists() for name in marker["files"]):
            for name in [*marker["files"], "completed.pt"]:
                local = artifacts.local / name
                if local.with_name(local.name + ".sync-pending.pt").exists():
                    artifacts.sync(local)
            with locate("metrics.json").open() as stream:
                return {"status": "complete", "result": json.load(stream), "warm_start": "", "config": config}
    if not checkpoint.exists() and complete.exists():
        raise RuntimeError("Incomplete reports and no safe checkpoint; cannot recover completed run")
    return {
        "status": "resume" if checkpoint.exists() else "initialized",
        "warm_start": state.get("warm_start", ""),
        "config": config,
    }


def complete_run(artifacts, config, run_id):
    files = sorted(
        p.name
        for p in artifacts.local.iterdir()
        if p.suffix in (".pt", ".png", ".json")
        and not p.name.endswith(".sync-pending.pt")
        and not p.name.startswith("emergency_weights_")
        and p.name != "completed.pt"
    )
    for name in files:
        artifacts.sync(artifacts.local / name)
    artifacts.save({"run_id": run_id, "config": config, "files": files}, "completed.pt")


def publish_summary(artifacts):
    if artifacts.remote is not None and not artifacts.remote.is_dir():
        raise RuntimeError("Drive group unavailable; refusing to publish a potentially incomplete summary")
    results = {}
    roots = [root for root in (artifacts.remote, artifacts.local) if root is not None]
    for root in roots:
        summary = root / "final_metrics.json"
        if summary.exists():
            for result in json.loads(summary.read_text()):
                results[result["tag"]] = result
    for root in roots:
        for path in sorted(root.glob("*/metrics.json")):
            result = json.loads(path.read_text())
            if result["tag"] != path.parent.name:
                raise ValueError(f"Metric tag/directory mismatch: {path}")
            results[result["tag"]] = result
    if not results:
        return []
    ordered = [results[tag] for tag in sorted(results)]
    rows = [
        {
            "tag": r["tag"],
            "seed": r["seed"],
            "threshold": r["threshold"],
            "temperature": r["temperature"],
            **{
                f"{split}_{key}": value
                for split in ("train", "val", "test")
                for key, value in (r.get(split) or {}).items()
            },
        }
        for r in ordered
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
    writer.writeheader()
    writer.writerows(rows)
    for name, text in (("final_metrics.json", json.dumps(ordered, indent=2)), ("final_metrics.csv", stream.getvalue())):
        path = artifacts.local / name
        temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
        try:
            with temporary.open("w", encoding="utf-8", newline="") as output:
                output.write(text)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        artifacts.sync(path)
    return rows


def _generate_run_id(wandb_module):
    generate = getattr(wandb_module.util, "generate_id", None)
    if generate is not None:
        return generate()
    from wandb.sdk.lib.runid import generate_id

    return generate_id()


def init_wandb(name, config, artifacts, *, resume=False, smoke=False, warm_start=""):
    load_env_file()
    if not os.environ.get("WANDB_API_KEY") and not smoke:
        try:
            from google.colab import userdata

            os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
        except Exception as exc:
            raise RuntimeError("W&B requires WANDB_API_KEY in .env/environment or Colab Secrets") from exc
    import wandb

    identity = artifacts.local / "run_identity.pt"
    if resume:
        if not identity.exists() and artifacts.remote:
            source = artifacts.remote / identity.name
            if source.exists():
                atomic_save(torch.load(source, weights_only=False, map_location="cpu"), identity)
        state = torch.load(identity, weights_only=False, map_location="cpu")
        if state["config"] != config:
            raise ValueError("Run config mismatch; use a new run directory for warm-start")
    else:
        if identity.exists() or (artifacts.remote and (artifacts.remote / identity.name).exists()):
            raise FileExistsError("Run already exists: enable RESUME or choose a new RUN_GROUP")
        state = {"id": _generate_run_id(wandb), "config": config, "warm_start": str(warm_start)}
    artifacts.save(state, identity.name)
    run = wandb.init(
        project="glaucoma-thesis",
        name=name,
        id=state["id"],
        config=config,
        resume="allow" if not smoke else None,
        mode="offline" if smoke else "online",
        dir=str(artifacts.local),
        reinit=True,
    )
    if run is None or run.disabled or (not smoke and run.offline):
        if run is not None:
            run.finish(exit_code=1)
        raise RuntimeError("Mandatory W&B initialization did not produce an active online run")
    try:
        run.define_metric("progress/step")
        run.define_metric("train/*", step_metric="progress/step")
        run.define_metric("val/*", step_metric="progress/step")
        run.define_metric("test/*", step_metric="progress/step")
    except BaseException:
        run.finish(exit_code=1)
        raise
    return run


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def read_weights(path):
    weights = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(weights, dict) and "weights" in weights:
        if weights.get("resumable") is not False:
            raise ValueError("Emergency warm-start payload must explicitly have resumable=False")
        weights = weights["weights"]
    if (
        not isinstance(weights, dict)
        or not weights
        or not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in weights.items())
    ):
        raise ValueError("Warm-start must be a tensor state_dict or validated emergency weights payload")
    return weights


def load_weights(model, path):
    path = Path(path)
    if not path.is_file():
        return False
    model.load_state_dict(read_weights(path), strict=True)
    return True


def find_batch_size(
    build_model,
    dataset,
    *,
    device,
    start=2,
    target_gb=0.0,
    max_batch=256,
    num_workers=0,
    candidates=None,
    config=None,
    optimizer_windows=2,
    smoke=False,
):
    device = torch.device(device)
    if start < 1 or max_batch < start or len(dataset) < 1 or optimizer_windows < 2:
        raise ValueError("Positive batch/data and at least two optimizer windows required")
    if os.name == "nt" and num_workers:
        raise ValueError("Windows requires num_workers=0 for mmap safety")
    if device.type != "cuda" and not smoke:
        return {"batch_size": start, "peak_gb": 0.0, "status": "cpu", "trials": []}
    config = config or {}
    dtype = getattr(torch, config.get("amp_dtype", "float16"))
    accum = int(config.get("grad_accum", 2))
    if dtype not in (torch.float16, torch.bfloat16) or accum < 1:
        raise ValueError("Invalid probe AMP dtype/accumulation")
    choices = (
        list(candidates) if candidates is not None else [start * 2**i for i in range(16) if start * 2**i <= max_batch]
    )
    if not choices or any(not isinstance(n, int) or n < 1 or n > max_batch for n in choices):
        raise ValueError("Invalid candidate batch sizes")
    total = torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 0
    budget = min(target_gb * 1e9, total * 0.9) if target_gb else total * 0.85
    if not math.isfinite(target_gb) or target_gb < 0:
        raise ValueError("target_gb must be finite and nonnegative")
    snapshot = rng_state()
    trials, best, peak_used = [], None, 0.0
    for candidate in sorted(set(choices)):
        model = optimizer = scaler = loader = batch = x = views = target = logits = loss = parameter = weights = None
        initial = modes = original_device = None
        began = time.monotonic()
        trial = {
            "batch_size": candidate,
            "ok": False,
            "micro_steps": 0,
            "allocated": 0,
            "reserved": 0,
            "peak_allocated": 0,
            "peak_reserved": 0,
            "free_after": 0,
        }
        try:
            restore_rng(snapshot)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            model = build_model()
            if any(p.grad is not None for p in model.parameters()):
                raise ValueError("Probe factory returned a model with live gradients")
            original_device = next(model.parameters()).device
            initial, modes = cpu_state(model), [(m, m.training) for m in model.modules()]
            model.to(device).train()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.get("lr", 1e-4),
                weight_decay=config.get("weight_decay", 0.01),
                **({"fused": True} if config.get("fused_adamw", False) else {}),
            )
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
            loader = DataLoader(
                Subset(dataset, [i % len(dataset) for i in range(candidate)]),
                batch_size=candidate,
                num_workers=num_workers,
                pin_memory=config.get("pin_memory", device.type == "cuda"),
                generator=torch.Generator().manual_seed(0),
            )
            batch = next(iter(loader))
            x, views, target = (t.to(device, non_blocking=config.get("non_blocking", True)) for t in batch)
            weights = torch.tensor(config.get("class_weights", [1.0, 1.0])).to(device)
            for _ in range(optimizer_windows):
                optimizer.zero_grad(set_to_none=True)
                for _ in range(accum):
                    with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
                        logits = model(x, views)
                        loss = F.cross_entropy(logits, target, weight=weights, reduction="sum")
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite probe loss")
                    scaler.scale(loss).backward()
                    trial["micro_steps"] += 1
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(weights[target].sum() * accum)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < scale or not optimizer.state:
                    raise RuntimeError("Probe requires allocated optimizer state and successful updates")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                trial.update(
                    allocated=torch.cuda.memory_allocated(device),
                    reserved=torch.cuda.memory_reserved(device),
                    peak_allocated=torch.cuda.max_memory_allocated(device),
                    peak_reserved=torch.cuda.max_memory_reserved(device),
                    free_after=torch.cuda.mem_get_info(device)[0],
                )
            trial["ok"] = device.type != "cuda" or trial["peak_reserved"] <= budget
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            trial["error"] = "out of memory"
        finally:
            if model is not None and initial is not None:
                model.load_state_dict(initial)
                model.to(original_device)
                model.zero_grad(set_to_none=True)
                for module, training in modes:
                    module.training = training
                module = None
            model = optimizer = scaler = loader = batch = x = views = target = logits = loss = parameter = weights = (
                None
            )
            initial = modes = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            restore_rng(snapshot)
        trial.update(peak_gb=trial["peak_reserved"] / 1e9, wall_seconds=time.monotonic() - began)
        trials.append(trial)
        if trial["ok"]:
            best, peak_used = candidate, trial["peak_gb"]
        else:
            break
    if best is None:
        raise RuntimeError(f"No candidate batch fits the VRAM budget; trials={trials}")
    return {"batch_size": best, "peak_gb": peak_used, "status": "ok", "trials": trials, "target_gb": budget / 1e9}


class Trainer:
    def __init__(self, model, dataset, config, artifacts, run, *, resume=False, warm_start="", num_workers=0):
        if not isinstance(config.get("epochs"), int) or isinstance(config.get("epochs"), bool):
            raise ValueError("epochs must be a positive integer")
        if resume and warm_start:
            raise ValueError("Warm-start is weights only, not resume; choose one")
        if len(dataset) == 0 or config["batch_size"] < 1 or config["grad_accum"] < 1 or config["epochs"] < 1:
            raise ValueError("Nonempty data and positive batch_size/grad_accum/epochs required")
        if config["checkpoint_steps"] < 1:
            raise ValueError("checkpoint_steps must be positive")
        if len(config["class_weights"]) != 2 or any(not math.isfinite(w) or w <= 0 for w in config["class_weights"]):
            raise ValueError("Two finite positive class weights required")
        if not getattr(dataset, "deterministic_by_index", False):
            raise ValueError("Dataset must implement deterministic per-(seed, epoch, index) augmentation")
        self.model, self.dataset, self.config = model, dataset, copy.deepcopy(config)
        self.artifacts, self.run = artifacts, run
        self.device = next(model.parameters()).device
        self.num_workers = max(0, int(num_workers))
        if os.name == "nt" and self.num_workers:
            raise ValueError("Windows requires num_workers=0 for mmap safety")
        self.amp_dtype = getattr(torch, config.get("amp_dtype", "float16"))
        if self.amp_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("amp_dtype must be float16 or bfloat16")
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            **({"fused": True} if config.get("fused_adamw", False) else {}),
        )
        steps = math.ceil(math.ceil(len(dataset) / config["batch_size"]) / config["grad_accum"]) * config["epochs"]
        warm = int(steps * 0.05)

        def schedule(step):
            if warm and step < warm:
                return 0.01 + 0.99 * step / warm
            return 0.5 * (1 + math.cos(math.pi * min(step - warm, steps - warm) / max(steps - warm, 1)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.device.type == "cuda" and self.amp_dtype == torch.float16
        )
        self.confusion = [0, 0, 0, 0]
        self.scaler_skips = 0
        self.best_epoch = self.best_step = None
        self.epoch, self.cursor, self.step = 0, 0, 0
        self.loss_sum, self.weight_sum, self.correct, self.seen = 0.0, 0.0, 0, 0
        self.history, self.best_state, self.best_auc, self.bad = [], None, -math.inf, 0
        self.stop_requested = False
        self.failed = False
        self.order = None
        self.window_probs, self.window_labels = [], []
        self.warm_start = str(warm_start)
        self.optimizer.zero_grad(set_to_none=True)
        if not resume and (artifacts.local / "last.pt").exists():
            raise FileExistsError("Refusing to overwrite last.pt without resume")
        if resume:
            path = artifacts.local / "last.pt"
            if not path.exists() and artifacts.remote:
                source = artifacts.remote / "last.pt"
                if source.exists():
                    atomic_save(torch.load(source, map_location="cpu", weights_only=False), path)
            state = torch.load(path, map_location="cpu", weights_only=False)
            if (
                state.get("format") != 1
                or state["run_id"] != run.id
                or not _config_compatible(state["config"], self.config)
            ):
                raise ValueError("Checkpoint format/config/run id mismatch")
            extended = int(self.config["epochs"]) > int(state["config"]["epochs"])
            if extended:
                identity_path = artifacts.local / "run_identity.pt"
                if not identity_path.exists() and artifacts.remote:
                    identity_path = artifacts.remote / "run_identity.pt"
                authorization = (
                    torch.load(identity_path, map_location="cpu", weights_only=False) if identity_path.exists() else {}
                )
                if (
                    authorization.get("id") != run.id
                    or authorization.get("config") != self.config
                    or authorization.get("epoch_extension")
                    != {"from_epochs": state["config"]["epochs"], "to_epochs": self.config["epochs"]}
                ):
                    raise ValueError("Epoch extension requires explicit run_status authorization")
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.scaler.load_state_dict(state["scaler"])
            for key in (
                "epoch",
                "cursor",
                "step",
                "loss_sum",
                "weight_sum",
                "correct",
                "seen",
                "history",
                "best_state",
                "best_auc",
                "bad",
                "order",
                "warm_start",
            ):
                setattr(self, key, state[key])
            if extended:
                self.bad = 0
                self.scheduler._last_lr = [base * schedule(self.step) for base in self.scheduler.base_lrs]
                for group, lr in zip(self.optimizer.param_groups, self.scheduler._last_lr):
                    group["lr"] = lr
            restore_rng(state["rng"])
            for key in ("confusion", "scaler_skips", "best_epoch", "best_step"):
                if key in state:
                    setattr(self, key, state[key])
        elif warm_start:
            self.model.load_state_dict(read_weights(warm_start), strict=True)

    def commit(self, *, best=False):
        if any(parameter.grad is not None for parameter in self.model.parameters()):
            raise RuntimeError("Cannot commit with pending gradients")
        state = {
            key: getattr(self, key)
            for key in (
                "epoch",
                "cursor",
                "step",
                "loss_sum",
                "weight_sum",
                "correct",
                "seen",
                "history",
                "best_state",
                "best_auc",
                "bad",
                "order",
                "warm_start",
            )
        }
        state.update(
            confusion=self.confusion,
            scaler_skips=self.scaler_skips,
            best_epoch=self.best_epoch,
            best_step=self.best_step,
            format=1,
            config=self.config,
            run_id=self.run.id,
            model=cpu_state(self.model),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            scaler=self.scaler.state_dict(),
            rng=rng_state(),
        )
        self.artifacts.save(state, "last.pt")
        if (self.artifacts.local / "steps.jsonl").exists():
            self.artifacts.sync(self.artifacts.local / "steps.jsonl")
        if best:
            self.artifacts.save(state, "best.pt")

    @contextlib.contextmanager
    def interrupts(self):
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

        def request_stop(signum, frame):
            if self.stop_requested:
                raise KeyboardInterrupt
            self.stop_requested = True
            print(
                "Stop requested; finishing current accumulation window before checkpoint. Interrupt again for emergency weights.",
                flush=True,
            )

        for sig in previous:
            signal.signal(sig, request_stop)
        try:
            yield
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def window_metrics(self):
        from scripts import final_model as fm

        if not self.window_probs:
            return {}
        return fm.full_metrics(torch.cat(self.window_probs).numpy(), torch.cat(self.window_labels).numpy())

    def log(self, values):
        serializable = {k: v if not isinstance(v, float) or math.isfinite(v) else None for k, v in values.items()}
        with (self.artifacts.local / "steps.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(serializable, allow_nan=False) + "\n")
        self.run.log(values)

    def fit(self, evaluate, *, boundary_hook=None, test_evaluate=None):
        if self.failed:
            raise RuntimeError("Failed trainer may hold partial state; construct a new Trainer with resume=True")
        try:
            started = time.monotonic()
            fit_started = started
            last_sync = started
            initial_seen = self.seen
            with self.interrupts():
                self.commit()
                weights = torch.as_tensor(self.config["class_weights"], dtype=torch.float32, device=self.device)
                while self.epoch < self.config["epochs"] and self.bad < self.config["patience"]:
                    self.dataset.set_epoch(self.epoch)
                    if self.order is None:
                        self.order = (
                            np.random.default_rng(self.config["seed"] + self.epoch)
                            .permutation(len(self.dataset))
                            .tolist()
                        )
                    loader = DataLoader(
                        Subset(self.dataset, self.order[self.cursor :]),
                        batch_size=self.config["batch_size"],
                        num_workers=self.num_workers,
                        pin_memory=self.config.get("pin_memory", self.device.type == "cuda"),
                        generator=torch.Generator().manual_seed(0),
                    )
                    self.model.train()
                    window_weight, micro = 0.0, 0
                    for x, views, labels in loader:
                        x, views, labels = (
                            t.to(self.device, non_blocking=self.config.get("non_blocking", self.device.type == "cuda"))
                            for t in (x, views, labels)
                        )
                        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.device.type == "cuda"):
                            logits = self.model(x, views)
                            loss = F.cross_entropy(logits, labels, weight=weights, reduction="sum")
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite loss; aborting before backward")
                        if self.config.get("step_metrics", True):
                            self.window_probs.append(logits.detach().float().softmax(1)[:, 1].cpu())
                            self.window_labels.append(labels.detach().cpu())
                        self.scaler.scale(loss).backward()
                        denominator = weights[labels].sum().item()
                        window_weight += denominator
                        self.loss_sum += loss.detach().item()
                        self.weight_sum += denominator
                        self.correct += (logits.argmax(1) == labels).sum().item()
                        counts = torch.bincount(labels * 2 + logits.argmax(1), minlength=4).tolist()
                        self.confusion = [a + b for a, b in zip(self.confusion, counts)]
                        self.seen += len(labels)
                        self.cursor += len(labels)
                        micro += 1
                        if micro < self.config["grad_accum"] and self.cursor < len(self.order):
                            continue
                        self.scaler.unscale_(self.optimizer)
                        for parameter in self.model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.div_(window_weight)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            1.0,
                            error_if_nonfinite=not getattr(self.scaler, "is_enabled", lambda: True)(),
                        )
                        old_scale = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        if self.scaler.get_scale() >= old_scale:
                            self.scheduler.step()
                            self.step += 1
                            self.scaler_skips = 0
                        else:
                            self.scaler_skips += 1
                            if self.scaler_skips >= self.config.get("max_scaler_skips", 3):
                                raise FloatingPointError("Excessive AMP scaler skips; no optimizer progress")
                        window_weight, micro = 0.0, 0
                        values = {
                            **{f"train/{key}": value for key, value in self.window_metrics().items() if key != "acc"},
                            "progress/step": self.step,
                            "train/loss": self.loss_sum / self.weight_sum,
                            "train/acc": self.correct / self.seen,
                            "train/lr": self.optimizer.param_groups[0]["lr"],
                            "train/epoch": self.epoch + self.cursor / len(self.order),
                        }
                        self.window_probs, self.window_labels = [], []
                        if self.config.get("telemetry", False):
                            import psutil

                            elapsed = time.monotonic() - started
                            memory = psutil.virtual_memory()
                            values.update(
                                {
                                    "sys/cpu_percent": psutil.cpu_percent(),
                                    "sys/ram_used_gb": memory.used / 1024**3,
                                    "sys/ram_percent": memory.percent,
                                    "sys/disk_free_gb": shutil.disk_usage(self.artifacts.local).free / 1024**3,
                                    "train/session_wall_seconds": time.monotonic() - fit_started,
                                    "train/samples_per_second": (self.seen - initial_seen) / max(elapsed, 1e-9),
                                }
                            )
                            if self.device.type == "cuda":
                                values.update(
                                    {
                                        "sys/gpu_allocated_gb": torch.cuda.memory_allocated(self.device) / 1024**3,
                                        "sys/gpu_reserved_gb": torch.cuda.memory_reserved(self.device) / 1024**3,
                                        "sys/gpu_free_gb": torch.cuda.mem_get_info(self.device)[0] / 1024**3,
                                    }
                                )
                        self.log(values)
                        if (
                            self.step % self.config["checkpoint_steps"] == 0
                            or self.stop_requested
                            or time.monotonic() - last_sync >= self.config.get("checkpoint_seconds", 300)
                        ):
                            self.commit()
                            last_sync = time.monotonic()
                        if boundary_hook:
                            boundary_hook(self)
                        if self.stop_requested:
                            self.commit()
                            return False
                    if self.step == (self.history[-1].get("step", -1) if self.history else 0):
                        raise FloatingPointError("Epoch finished without any optimizer updates")
                    metrics = evaluate(self.model)
                    auc = metrics["auc_roc"]
                    improved = self.best_state is None or (math.isfinite(auc) and auc > self.best_auc)
                    if improved:
                        self.best_auc = auc if math.isfinite(auc) else -math.inf
                        self.best_state, self.bad = cpu_state(self.model), 0
                        self.best_epoch, self.best_step = self.epoch + 1, self.step
                    else:
                        self.bad += 1
                    self.history.append(
                        {
                            "epoch": self.epoch + 1,
                            "loss": self.loss_sum / self.weight_sum,
                            "val_auc": auc,
                            "val_f1": metrics["f1"],
                            "val_bal": metrics["balanced_acc"],
                            "val_mcc": metrics["mcc"],
                            "step": self.step,
                            "train": self.train_metrics(),
                            "val": dict(metrics),
                        }
                    )
                    self.epoch += 1
                    self.cursor, self.order = 0, None
                    self.loss_sum, self.weight_sum, self.correct, self.seen = 0.0, 0.0, 0, 0
                    self.confusion = [0, 0, 0, 0]
                    started, initial_seen = time.monotonic(), 0
                    self.log(
                        {
                            "progress/step": self.step,
                            "val/epoch": self.epoch,
                            **{f"train/{key}": value for key, value in self.history[-1]["train"].items()},
                            **{f"val/{key}": value for key, value in metrics.items()},
                        }
                    )
                    if test_evaluate is not None:
                        test_metrics = test_evaluate(self.model)
                        self.history[-1]["test"] = test_metrics
                        self.log(
                            {
                                "progress/step": self.step,
                                "test/epoch": self.epoch,
                                **{f"test/{key}": value for key, value in test_metrics.items()},
                            }
                        )
                    self.commit(best=improved)
                    print(self.history[-1], flush=True)
                    if self.stop_requested:
                        return False
                return True
        except BaseException:
            self.failed = True
            try:
                self.artifacts.save(
                    {
                        "weights": cpu_state(self.model),
                        "resumable": False,
                        "warning": "May include a partial optimizer operation; warm-start only",
                    },
                    f"emergency_weights_{time.time_ns()}.pt",
                )
            except BaseException as exc:
                print(f"Emergency persistence failed: {exc}; last.pt was not replaced by emergency state", flush=True)
            raise

    def train_metrics(self):
        tn, fp, fn, tp = self.confusion
        return {
            "loss": self.loss_sum / self.weight_sum,
            "acc": self.correct / self.seen,
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }


def volume_sample(volumes, index):
    value = np.asarray(volumes[index])
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3:
        raise ValueError(f"Expected 3D or single-channel 4D sample, got {value.shape}")
    return np.ascontiguousarray(value)


def file_identity(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def data_identity(path):
    path = Path(path)
    identity = file_identity(path)
    cache = path.with_suffix(".sha256.pt")
    if cache.exists():
        saved = torch.load(cache, weights_only=False)
        if saved["file"] == identity:
            return saved["content"]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    content = {"sha256": digest.hexdigest(), "size": identity["size"]}
    atomic_save({"file": identity, "content": content}, cache)
    return content


def build_denoised(source, target, denoise, *, method, params, implementation, limit=0):
    source, target = Path(source), Path(target)
    volumes = np.load(source, mmap_mode="r")
    if not implementation or not isinstance(params, dict):
        raise ValueError("Denoise requires effective parameters and an implementation identifier")
    identity = {
        "source": file_identity(source),
        "method": method,
        "params": copy.deepcopy(params),
        "implementation": implementation,
        "shape": list(volumes.shape),
        "version": 2,
    }
    marker = target.with_suffix(".complete.pt")
    if marker.exists() and target.exists():
        state = torch.load(marker, weights_only=False)
        if state == identity:
            return True
        raise ValueError(f"Denoise cache identity mismatch: {target}")
    partial, progress = target.with_suffix(".partial.npy"), target.with_suffix(".progress.pt")
    cursor = 0
    if progress.exists() and partial.exists():
        state = torch.load(progress, weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Partial denoise cache identity mismatch")
        cursor = state["cursor"]
        out = np.lib.format.open_memmap(partial, mode="r+")
    else:
        out = np.lib.format.open_memmap(partial, mode="w+", dtype=volumes.dtype, shape=volumes.shape)
    stop = len(volumes) if limit <= 0 else min(len(volumes), cursor + limit)
    for index in range(cursor, stop):
        result = denoise(volume_sample(volumes, index))
        out[index] = result[None] if volumes.ndim == 5 else result
        out.flush()
        atomic_save({"identity": identity, "cursor": index + 1}, progress)
    del out
    if stop != len(volumes):
        print(f"Denoise pending {stop}/{len(volumes)}: {partial}; not eligible for training")
        return False
    os.replace(partial, target)
    atomic_save(identity, marker)
    progress.unlink(missing_ok=True)
    return True


def build_views(source, *, res2d, n2d=2):
    from scripts import final_model as fm

    source = Path(source)
    base = source.with_suffix("")
    vp, dp = Path(str(base) + f"_views_{res2d}.npy"), Path(str(base) + f"_dzs_{res2d}.npy")
    marker = Path(str(base) + f"_views_{res2d}.complete.pt")
    identity = {"source": file_identity(source), "res2d": res2d, "n2d": n2d, "version": 1}
    if marker.exists() and vp.exists() and dp.exists() and torch.load(marker, weights_only=False) == identity:
        return vp, dp
    marker.unlink(missing_ok=True)
    volumes = np.load(source, mmap_mode="r")
    temporary = vp.with_suffix(".partial.npy")
    views = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=(len(volumes), n2d, res2d, res2d))
    dzs = np.zeros(len(volumes), dtype=np.int8)
    for index in range(len(volumes)):
        raw = volume_sample(volumes, index)
        dzs[index] = fm.depth_axis(raw)
        projected = fm.project_views(fm.to_depth_last(raw, int(dzs[index])))
        for view in range(n2d):
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
    temporary_dz = dp.with_suffix(".partial.npy")
    np.save(temporary_dz, dzs)
    os.replace(temporary_dz, dp)
    atomic_save(identity, marker)
    return vp, dp


class FinalDataset(Dataset):
    deterministic_by_index = True

    def __init__(self, source, labels, *, res3d, res2d, seed, train=False):
        self.source, self.label_path = Path(source), Path(labels)
        if "_dn" in self.source.stem:
            marker = self.source.with_suffix(".complete.pt")
            if not marker.exists():
                raise ValueError("Denoised cache has no completion marker; rebuild before training")
            identity = torch.load(marker, weights_only=False)
            raw = self.source.with_name(self.source.name.replace("_volumes_dn", "_volumes"))
            if identity["source"] != file_identity(raw):
                raise ValueError("Denoised cache source changed; rebuild before training")
        vp, dp = build_views(source, res2d=res2d)
        self.volumes = np.load(source, mmap_mode="r")
        self.views = np.load(vp, mmap_mode="r")
        self.dzs = np.load(dp, mmap_mode="r")
        self.labels = np.load(labels)
        if len(self.volumes) != len(self.labels):
            raise ValueError("Volume/label count mismatch")
        self.res3d, self.seed, self.train, self.epoch = res3d, seed, train, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        from scripts import final_model as fm

        raw = fm.to_depth_last(volume_sample(self.volumes, index), int(self.dzs[index]))
        views = np.asarray(self.views[index])
        if self.train:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
            raw, views = fm.aug_pair(raw, views, rng)
        x = torch.from_numpy(raw.astype(np.float32) / 255)[None]
        if tuple(x.shape[1:]) != (self.res3d,) * 3:
            x = F.interpolate(x[None], (self.res3d,) * 3, mode="trilinear", align_corners=False)[0]
        return x, torch.from_numpy(views.astype(np.float32) / 255)[:, None], torch.tensor(int(self.labels[index]))


class SmokeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 2, 1)
        self.head = torch.nn.Linear(4, 2)

    def fuse(self, x, views):
        feature = self.conv(x).relu().mean((2, 3, 4))
        summary = views.mean((2, 3, 4))
        return torch.cat([feature, summary], 1), torch.stack([feature, summary], 1)

    def forward(self, x, views):
        z, _ = self.fuse(x, views)
        return self.head(z)


def predict(model, dataset, batch_size, num_workers=0, *, amp_dtype="float16", pin_memory=None, non_blocking=None):
    if os.name == "nt" and num_workers:
        raise ValueError("Windows requires num_workers=0 for mmap safety")
    device = next(model.parameters()).device
    pin_memory = device.type == "cuda" if pin_memory is None else pin_memory
    non_blocking = device.type == "cuda" if non_blocking is None else non_blocking
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for x, views, y in DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=torch.Generator().manual_seed(0),
        ):
            with torch.autocast(device.type, dtype=getattr(torch, amp_dtype), enabled=device.type == "cuda"):
                output = model(x.to(device, non_blocking=non_blocking), views.to(device, non_blocking=non_blocking))
            logits.append(output.float().cpu())
            labels.append(y)
    output = torch.cat(logits)
    return output.softmax(1)[:, 1].numpy(), torch.cat(labels).numpy(), output.numpy()


def calibrated_report(model, val, test, batch_size, *, smoke=False, train=None, num_workers=0):
    from scripts import final_model as fm

    _, vy, vl = predict(model, val, batch_size, num_workers=num_workers)
    temperature = fm.temperature_scale(vl, vy, max_iter=10 if smoke else 200)
    if not math.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("Validation temperature fit did not produce a finite positive temperature")
    vp = torch.softmax(torch.tensor(vl) / temperature, 1)[:, 1].numpy()
    threshold = fm.tune_threshold(vp, vy)
    _, ty, tl = predict(model, test, batch_size, num_workers=num_workers)
    tp = torch.softmax(torch.tensor(tl) / temperature, 1)[:, 1].numpy()
    n_boot = 20 if smoke else 1000
    result = {
        "temperature": temperature,
        "threshold": threshold,
        "val": {
            **fm.full_metrics(vp, vy, threshold),
            "loss": F.cross_entropy(torch.tensor(vl) / temperature, torch.tensor(vy)).item(),
        },
        "test": {
            **fm.full_metrics(tp, ty, threshold),
            "loss": F.cross_entropy(torch.tensor(tl) / temperature, torch.tensor(ty)).item(),
        },
        "val_ci": fm.bootstrap_ci(vp, vy, n_boot=n_boot, threshold=threshold),
        "test_ci": fm.bootstrap_ci(tp, ty, n_boot=n_boot, threshold=threshold),
    }
    if train is not None:
        trp, tr_labels, trl = predict(model, train, batch_size, num_workers=num_workers)
        trp = torch.softmax(torch.tensor(trl) / temperature, 1)[:, 1].numpy()
        result["train"] = {
            **fm.full_metrics(trp, tr_labels, threshold),
            "loss": F.cross_entropy(torch.tensor(trl) / temperature, torch.tensor(tr_labels)).item(),
        }
        result["train_ci"] = fm.bootstrap_ci(trp, tr_labels, n_boot=n_boot, threshold=threshold)
    return result, tp, ty


def log_report(run, result):
    import wandb

    splits = [name for name in ("train", "val", "test") if result.get(name)]
    keys = sorted({key for name in splits for key in result[name]})
    table = wandb.Table(columns=["split", *keys])
    summary = {}
    for name in splits:
        run.log({f"{name}/calibrated_{key}": value for key, value in result[name].items()})
        summary.update({f"{name}/{key}": value for key, value in result[name].items()})
        table.add_data(name, *[result[name].get(key) for key in keys])
    for name in ("train_ci", "val_ci", "test_ci"):
        for key, bounds in (result.get(name) or {}).items():
            if isinstance(bounds, list) and len(bounds) == 2:
                summary[f"{name}/{key}_lo"] = bounds[0]
                summary[f"{name}/{key}_hi"] = bounds[1]
    run.log({"report/split_table": table})
    run.summary.update(summary)
    return table


def save_report(result, probs, labels, artifacts, run):
    import wandb

    path = artifacts.local / "metrics.json"
    with path.open("w") as stream:
        json.dump(result, stream, indent=2)
    artifacts.sync(path)
    artifacts.save({"probs": probs, "labels": labels}, "test_predictions.pt")
    history = result["hist"]
    keys = sorted({key for entry in history for key in entry if key not in ("epoch", "val", "test")})
    table = wandb.Table(columns=["epoch", *keys])
    for entry in history:
        table.add_data(entry.get("epoch"), *[entry.get(key) for key in keys])
    run.log({"report/history_table": table})
    for split in ("train", "val", "test"):
        metrics = result.get(split) or {}
        summary = {f"{split}/{key}": value for key, value in metrics.items()}
        if summary:
            run.summary.update(summary)
    return path


def save_xai(model, dataset, artifacts, run, *, smoke=False):
    import matplotlib.pyplot as plt
    import wandb

    from scripts import final_model as fm

    model.eval()
    device = next(model.parameters()).device
    x, v, y = dataset[0]
    x, v, target = x[None].to(device), v[None].to(device), int(y)

    def save_image(array, name):
        array = np.asarray(array)
        if array.ndim == 3:
            array = array[array.shape[0] // 2]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(np.atleast_2d(array), cmap="hot")
        ax.set_title(name)
        path = artifacts.local / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        artifacts.sync(path)
        run.log({f"xai/{name}": wandb.Image(str(path))})

    module = model.conv if smoke else model.gcam3d_module()
    cam, _, _ = fm.grad_cam(module, model, x, v, target=target, is_3d=True)
    save_image(cam, "gradcam3d")
    if not smoke:
        for index in range(v.shape[1]):
            cam, _, _ = fm.grad_cam(model.gcam2d_module(index), model, x, v, target=target, is_3d=False)
            save_image(cam, f"gradcam2d_{index}")
    occlusion, _ = fm.occlusion_sensitivity(model, x, v, target, n=2 if smoke else 4)
    save_image(occlusion, "occlusion")
    save_image(fm.integrated_gradients(model, x, v, target, steps=2 if smoke else 16), "integrated_gradients")
    if not smoke:
        weights, gate = fm.crossgate_attention(model, x, v)
        names, drops = fm.branch_drop_importance(model, x, v, target)
        names = ["all_2d_inputs", *names[1:]]
        attention = {}
        if weights is not None:
            attention = {f"2D-{index}": float(value) for index, value in enumerate(np.asarray(weights).ravel())}
        artifacts.save(
            {"weights": weights, "gate": gate, "branch_drop": dict(zip(names, drops)), "attention": attention},
            "fusion_xai.pt",
        )
        table = wandb.Table(columns=["branch", "crossgate_attention", "drop_probability"])
        for name, drop in zip(names, drops):
            table.add_data(name, attention.get(name), float(drop))
        run.log({"xai/fusion_table": table})
        run.summary.update(
            {
                "xai/gate": gate,
                **{f"xai/drop_{name}": drop for name, drop in zip(names, drops)},
                **{f"xai/attention_{name}": value for name, value in attention.items()},
            }
        )
