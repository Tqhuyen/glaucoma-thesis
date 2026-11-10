"""Replay pipeline metrics.jsonl files to W&B from this machine.

Used when the training host cannot reach W&B (e.g. vast.ai IP blocked with HTTP 403).
Each run folder becomes the W&B run with the same id/name; only lines not yet sent are
logged, so it is safe to re-run or keep following while training continues.

    python scripts/wandb_replay.py artifacts/vast/training_runs            # one pass
    python scripts/wandb_replay.py artifacts/vast/training_runs --follow   # keep syncing
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.utils import load_env_file  # noqa: E402

PROJECT = "glaucoma-thesis"


def replay_run(run_dir: Path, state: dict) -> bool:
    import wandb

    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        return False
    lines = metrics.read_text().splitlines()
    sent = state.get(run_dir.name, 0)
    finished = (run_dir / "test_results.json").exists()
    if sent >= len(lines) and not (finished and not state.get(f"{run_dir.name}:artifact")):
        return finished

    config = json.loads((run_dir / "config_used.json").read_text()) if (run_dir / "config_used.json").exists() else {}
    run = wandb.init(
        project=PROJECT,
        id=run_dir.name,
        name=run_dir.name,
        config=config,
        resume="allow",
        tags=["vast-replay"],
        settings=wandb.Settings(x_disable_stats=True),
    )
    for line in lines[sent:]:
        row = json.loads(line)
        step = int(row.pop("step"))
        row.pop("time", None)
        run.log(row, step=step)
    state[run_dir.name] = len(lines)

    if finished and not state.get(f"{run_dir.name}:artifact"):
        results = json.loads((run_dir / "test_results.json").read_text())
        run.summary.update({f"final/{k}": v for k, v in results.items()})
        art = wandb.Artifact(f"{run_dir.name}-results", type="model")
        for name in ("best_model.pt", "metrics.jsonl", "config_used.json", "test_results.json", "training_curves.png"):
            if (run_dir / name).exists():
                art.add_file(str(run_dir / name))
        run.log_artifact(art)
        state[f"{run_dir.name}:artifact"] = True
    run.finish()
    print(f"[replay] {run_dir.name}: {state[run_dir.name]} rows sent{' + artifact' if finished else ''}", flush=True)
    return finished


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", type=Path)
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--interval", type=int, default=300)
    args = ap.parse_args()
    load_env_file()

    state_file = args.runs_dir / ".wandb_replay.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    while True:
        if args.runs_dir.exists():
            for run_dir in sorted(p for p in args.runs_dir.iterdir() if p.is_dir()):
                replay_run(run_dir, state)
                state_file.write_text(json.dumps(state, indent=1))
        if not args.follow:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
