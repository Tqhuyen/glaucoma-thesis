"""Enterprise training loop: model-agnostic. single GPU -> multi-GPU -> multi-node.

Launch matrix:
    1 GPU / CPU:      python -m pipeline.train --cfg configs/glaucoma.yaml
    N GPUs, 1 node:   torchrun --standalone --nproc_per_node=N -m pipeline.train --cfg ...
    Multi-node:       torchrun --nnodes=$N --node_rank=$R --master_addr=$ADDR --master_port=29500 ...
    SLURM cluster:    sbatch scripts/slurm_train.sbatch

Guarantees:
  - Rank 0 exclusively: logging, checkpoints, test eval.
  - SIGTERM/SIGINT (spot reclaim, scancel, Ctrl-C) -> checkpoint saved, clean exit code 0.
  - Effective batch = batch_size x world_size x grad_accum_steps (printed at start).
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .config import validate_config
from .distributed import (
    barrier,
    cleanup_distributed,
    is_distributed,
    is_main,
    reduce_sum,
    setup_distributed,
)
from .metrics import MetricsLogger
from .plotting import render_and_sync
from .utils import (
    atomic_save,
    classification_metrics,
    detect_environment,
    latest_checkpoint,
    load_config,
    load_env_file,
    prune_checkpoints,
    resolve_amp_dtype,
    sample_system_metrics,
    seed_everything,
    setup_hf_env,
    write_run_manifest,
)

_PREEMPTED = False


def _handle_preemption(signum, frame):
    global _PREEMPTED
    _PREEMPTED = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--set", nargs="*", default=[], help="Override config: key.subkey=value")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--sanity", action="store_true", help="Overfit N samples (cfg.sanity) — debug if code/arch can learn at all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def apply_profile(cfg, args):
    """Apply smoke / sanity training profiles to the config.

    smoke  -> tiny run that validates the pipeline end-to-end quickly.
    sanity -> overfit `max_train_samples` (default 10) for `epochs` (default 100)
              with dropout=0, weight_decay=0, warmup=0 and no eval/save.
              Loss must approach ~0; if it doesn't, the bug is in code/arch/data.
    """
    if args.smoke:
        for k in ("epochs", "batch_size", "log_every_steps", "eval_every_steps", "save_every_steps"):
            cfg["train"][k] = cfg["smoke"][k]
        cfg["train"]["eval_test_every_steps"] = cfg["smoke"].get("eval_test_every_steps", 10**9)
    if args.sanity:
        s = cfg.get("sanity", {})
        for k in ("epochs", "batch_size", "log_every_steps", "eval_every_steps", "save_every_steps"):
            cfg["train"][k] = s.get(k, cfg["train"].get(k))
        cfg["train"].update(
            eval_test_every_steps=10**9,
            weight_decay=0.0,
            warmup_ratio=0.0,
            grad_accum_steps=1,
            early_stop_patience=10**9,
        )
        cfg["model"]["dropout"] = 0.0
        print("[sanity] overfit mode: dropout=0, wd=0, warmup=0, eval/save off")


def build_scheduler(optimizer, total_steps, tcfg):
    warmup = int(total_steps * tcfg.get("warmup_ratio", 0.0))
    if warmup > 0:
        warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
        main_sched = CosineAnnealingLR(optimizer, T_max=total_steps - warmup)
        return SequentialLR(optimizer, [warmup_sched, main_sched], milestones=[warmup])
    return CosineAnnealingLR(optimizer, T_max=total_steps)


@torch.no_grad()
def evaluate_sharded(model, loader, device, amp_dtype, num_classes=2):
    """Eval a loader across all ranks. Returns metrics dict (loss/acc/precision/recall/f1)."""
    model.eval()
    loss_sum = torch.zeros(1, device=device)
    correct = torch.zeros(1, device=device)
    n = torch.zeros(1, device=device)
    preds_all = []
    labels_all = []
    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        if isinstance(batch, (tuple, list)):
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(x)
                loss = criterion(out["logits"], y)
            preds = out["logits"].argmax(-1)
        else:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(**batch)
                loss = out.loss if hasattr(out, "loss") else criterion(out["logits"], batch["labels"])
            preds = out["logits"].argmax(-1)
            y = batch["labels"]

        bs = y.size(0)
        loss_sum += loss.detach() * bs
        correct += (preds == y).sum()
        n += bs
        preds_all.append(preds)
        labels_all.append(y)

    loss_sum, correct, n = reduce_sum(loss_sum), reduce_sum(correct), reduce_sum(n)
    model.train()
    if n.item() == 0:
        return {"loss": float("nan"), "acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    loss = (loss_sum / n).item()
    acc = (correct / n).item()
    if is_distributed():
        preds_all = [t.contiguous() for t in preds_all]
        labels_all = [t.contiguous() for t in labels_all]
        preds_cat = torch.cat(preds_all)
        labels_cat = torch.cat(labels_all)
        world = dist.world_size
        gathered_p = [torch.zeros_like(preds_cat) for _ in range(world)]
        gathered_l = [torch.zeros_like(labels_cat) for _ in range(world)]
        dist.all_gather(gathered_p, preds_cat)
        dist.all_gather(gathered_l, labels_cat)
        preds_cat = torch.cat(gathered_p)
        labels_cat = torch.cat(gathered_l)
    else:
        preds_cat = torch.cat(preds_all)
        labels_cat = torch.cat(labels_all)
    cm = classification_metrics(preds_cat, labels_cat, num_classes=num_classes)
    return {"loss": loss, "acc": acc, **cm}


def compute_loss(model, batch, criterion, device, amp_dtype, grad_accum_steps, num_classes=2):
    """Returns (scaled_loss, batch_metrics) where metrics = acc/precision/recall/f1."""
    if isinstance(batch, (tuple, list)):
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(x)
            loss = criterion(out["logits"], y)
        preds = out["logits"].argmax(-1)
        labels = y
    else:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(**batch)
            if hasattr(out, "loss"):
                loss = out.loss
            else:
                loss = criterion(out["logits"], batch["labels"])
        preds = out["logits"].argmax(-1)
        labels = batch["labels"]
    cm = classification_metrics(preds, labels, num_classes=num_classes)
    return loss / grad_accum_steps, cm


def main():
    from models import get_dataloader_builder, get_model_builder

    args = parse_args()
    load_env_file()
    cfg = load_config(args.cfg, args.set)
    validate_config(cfg)
    apply_profile(cfg, args)

    rank, local_rank, world_size = setup_distributed()
    main_proc = is_main(rank)

    signal.signal(signal.SIGTERM, _handle_preemption)
    signal.signal(signal.SIGINT, _handle_preemption)

    seed_everything(cfg["seed"] + rank)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        if main_proc:
            print("WARNING: no GPU — CPU mode (smoke tests only).")
    amp_dtype = resolve_amp_dtype(device) if cfg["train"].get("amp", True) else None
    setup_hf_env(cfg.get("hf", {}))

    out_dir = Path(args.output_dir or cfg["output_dir"]) / cfg["run_name"]
    ckpt_dir = out_dir / "checkpoints"
    if main_proc:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config_used.json").write_text(json.dumps(cfg, indent=2))
        write_run_manifest(out_dir)
    barrier()

    logger = MetricsLogger(cfg, out_dir) if main_proc else None

    tcfg = cfg["train"]
    model_type = cfg["model"]["type"]
    build_model_fn = get_model_builder(model_type)
    build_loaders_fn = get_dataloader_builder(model_type)

    if main_proc:
        eff_batch = tcfg["batch_size"] * world_size * tcfg["grad_accum_steps"]
        print(
            f"[env] {detect_environment()} | model={model_type} | world_size={world_size} | "
            f"device={device} | amp={amp_dtype} | effective_batch={eff_batch} | output={out_dir}"
        )

    train_loader, val_loader, test_loader = build_loaders_fn(cfg, smoke=args.smoke)

    if args.sanity:
        n = cfg.get("sanity", {}).get("max_train_samples", 10)
        train_loader.dataset.labels = train_loader.dataset.labels[:n]
        train_loader.dataset.volumes = train_loader.dataset.volumes[:n]
        val_loader.dataset.labels = val_loader.dataset.labels[:0]
        val_loader.dataset.volumes = val_loader.dataset.volumes[:0]
        if main_proc:
            print(f"[sanity] overfitting {len(train_loader.dataset)} train samples ({len(val_loader.dataset)} val)")

    model = build_model_fn(cfg).to(device)
    if cfg["train"].get("compile", False):
        model = torch.compile(model)
    if is_distributed():
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    raw_model = model.module if isinstance(model, DDP) else model

    criterion = nn.CrossEntropyLoss()
    num_classes = cfg["model"].get("num_classes", 2)

    steps_per_epoch = math.ceil(len(train_loader) / tcfg["grad_accum_steps"])
    total_steps = steps_per_epoch * tcfg["epochs"]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.01)
    )
    scheduler = build_scheduler(optimizer, total_steps, tcfg)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    global_step, start_epoch, best_val_loss, evals_since_best = 0, 0, float("inf"), 0
    if args.resume:
        ckpt_path = latest_checkpoint(ckpt_dir)
        if ckpt_path is None:
            if main_proc:
                print("[resume] no checkpoint found, starting fresh")
        else:
            state = torch.load(ckpt_path, map_location=device)
            raw_model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            global_step = state["global_step"]
            start_epoch = state["epoch"]
            best_val_loss = state["best_val_loss"]
            if main_proc:
                print(f"[resume] restored {ckpt_path.name} (step {global_step}, epoch {start_epoch})")
    barrier()

    def save_ckpt(step, epoch):
        if not main_proc:
            return
        atomic_save(
            {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "global_step": step,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            },
            ckpt_dir / f"step_{step}.pt",
        )
        prune_checkpoints(ckpt_dir, tcfg["keep_last_checkpoints"])

    def save_best(step):
        if not main_proc:
            return
        torch.save(raw_model.state_dict(), out_dir / "best_model.pt")

    def log(metrics, step):
        if logger:
            logger.log(metrics, step)

    model.train()
    t0 = time.time()
    stop = False
    running_acc_sum = 0.0
    running_acc_n = 0
    for epoch in range(start_epoch, tcfg["epochs"]):
        if is_distributed():
            train_loader.sampler.set_epoch(epoch)

        if main_proc:
            from tqdm.auto import tqdm
            iterable = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        else:
            iterable = train_loader

        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(iterable):
            loss, bm = compute_loss(model, batch, criterion, device, amp_dtype, tcfg["grad_accum_steps"], num_classes)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[rank {rank}] Loss is {loss.item()} at step {global_step}. "
                    "Usually: lr too high, bad labels, or fp16 overflow (use bf16 GPUs)."
                )

            scaler.scale(loss).backward()
            running_acc_sum += bm["acc"]
            running_acc_n += 1

            if (i + 1) % tcfg["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if main_proc and global_step % tcfg["log_every_steps"] == 0:
                    lr = scheduler.get_last_lr()[0]
                    step_loss = loss.item() * tcfg["grad_accum_steps"]
                    train_acc = running_acc_sum / max(running_acc_n, 1)
                    running_acc_sum, running_acc_n = 0.0, 0
                    log(
                        {
                            "train/loss": step_loss,
                            "train/acc": train_acc,
                            "train/precision": bm["precision"],
                            "train/recall": bm["recall"],
                            "train/f1": bm["f1"],
                            "train/lr": lr,
                            **sample_system_metrics(),
                        },
                        global_step,
                    )
                    if hasattr(iterable, "set_postfix"):
                        iterable.set_postfix(loss=f"{step_loss:.4f}", acc=f"{train_acc:.3f}", lr=f"{lr:.2e}")

                if global_step % tcfg["eval_every_steps"] == 0:
                    vm = evaluate_sharded(model, val_loader, device, amp_dtype, num_classes)
                    log({f"val/{k}": v for k, v in vm.items()}, global_step)
                    if main_proc:
                        print(f"\n[eval] step {global_step}: val_loss={vm['loss']:.4f} val_acc={vm['acc']:.4f} "
                              f"val_f1={vm['f1']:.4f}")
                    if vm["loss"] < best_val_loss:
                        best_val_loss, evals_since_best = vm["loss"], 0
                        save_best(global_step)
                    else:
                        evals_since_best += 1
                        if evals_since_best >= tcfg["early_stop_patience"]:
                            if main_proc:
                                print("[early-stop] no improvement, stopping.")
                            stop = True

                test_every = tcfg.get("eval_test_every_steps", 0)
                if test_every and test_loader is not None and global_step % test_every == 0:
                    tm = evaluate_sharded(model, test_loader, device, amp_dtype, num_classes)
                    log({f"test/{k}": v for k, v in tm.items()}, global_step)
                    if main_proc:
                        print(f"\n[test ] step {global_step}: test_loss={tm['loss']:.4f} test_acc={tm['acc']:.4f} "
                              f"test_f1={tm['f1']:.4f}")

                if global_step % tcfg["save_every_steps"] == 0:
                    save_ckpt(global_step, epoch)

                if _PREEMPTED:
                    if main_proc:
                        print(f"\n[preempt] signal received — checkpointing at step {global_step}")
                    save_ckpt(global_step, epoch)
                    barrier()
                    cleanup_distributed()
                    return

                if stop:
                    break
        if stop:
            break

        if len(val_loader.dataset) > 0:
            vm = evaluate_sharded(model, val_loader, device, amp_dtype, num_classes)
            log({f"val/{k}": v for k, v in vm.items()}, global_step)
            if main_proc:
                print(f"[epoch {epoch}] val_loss={vm['loss']:.4f} val_acc={vm['acc']:.4f} val_f1={vm['f1']:.4f}")
            if vm["loss"] < best_val_loss:
                best_val_loss = vm["loss"]
                save_best(global_step)
        save_ckpt(global_step, epoch + 1)

    save_ckpt(global_step, tcfg["epochs"])
    barrier()

    if main_proc and test_loader is not None and (out_dir / "best_model.pt").exists():
        print("[test] loading best_model for final held-out evaluation...")
        test_model = build_model_fn(cfg).to(device)
        test_model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
        tm = evaluate_sharded(test_model, test_loader, device, amp_dtype, num_classes)
        log({f"test/{k}": v for k, v in tm.items()}, global_step)
        log(sample_system_metrics(), global_step)
        print(f"[test] loss={tm['loss']:.4f} acc={tm['acc']:.4f} f1={tm['f1']:.4f}")
        (out_dir / "test_results.json").write_text(
            json.dumps({**{f"test_{k}": v for k, v in tm.items()}, "step": global_step}, indent=2)
        )

    if main_proc:
        png = render_and_sync(out_dir, cfg)
        if png:
            print(f"[plot] saved curves -> {png}")
        elif logger:
            print("[plot] nothing to plot yet (no metrics.jsonl) — curves will appear next run.")

    if main_proc:
        mins = (time.time() - t0) / 60
        print(f"\nDone in {mins:.1f} min. Best val_loss={best_val_loss:.4f}")
        print(f"Best model:  {out_dir / 'best_model.pt'}")
        print(f"Metrics:     {out_dir / 'metrics.jsonl'} | TensorBoard: {out_dir / 'tb'}")
        logger.close()
    barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
