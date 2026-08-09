"""One logger, three sinks — train/val/test metrics show up everywhere consistently.

  TensorBoard  -> realtime local dashboard (Colab inline / vast.ai port 6006)
  JSONL        -> outputs/<run>/metrics.jsonl, trivially loadable with pandas
  wandb        -> optional cloud dashboard, watch from any device (enable in config)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from .utils import load_env_file


class MetricsLogger:
    def __init__(self, cfg: dict, out_dir: Path):
        load_env_file()  # ensures WANDB_API_KEY from .env reaches wandb.init()
        lcfg = cfg.get("logging", {})
        self.tb = SummaryWriter(out_dir / "tb") if lcfg.get("tensorboard", True) else None

        self.jsonl = None
        if lcfg.get("jsonl", True):
            self.jsonl = open(out_dir / "metrics.jsonl", "a", buffering=1)

        self.wandb = None
        if lcfg.get("wandb", False):
            try:
                import wandb
                self.wandb = wandb
                wandb.init(
                    project=lcfg.get("wandb_project", "training"),
                    name=cfg["run_name"],
                    config=cfg,
                    resume="allow",
                    id=cfg["run_name"],
                )
            except Exception as e:
                print(f"[logging] wandb disabled ({e}). Run `wandb login` or set WANDB_API_KEY.")
                self.wandb = None

    def log(self, metrics: dict[str, float], step: int):
        if self.tb:
            for k, v in metrics.items():
                self.tb.add_scalar(k, v, step)
        if self.jsonl:
            self.jsonl.write(json.dumps({"step": step, "time": time.time(), **metrics}) + "\n")
        if self.wandb:
            self.wandb.log(metrics, step=step)

    def close(self):
        if self.tb:
            self.tb.close()
        if self.jsonl:
            self.jsonl.close()
        if self.wandb:
            self.wandb.finish()
