"""Launch the raw-200^3 suite on the cheapest suitable vast.ai GPU.

    python scripts/vast_launch.py --budget 5              # dry run: show chosen offer + caps
    python scripts/vast_launch.py --budget 5 --launch     # create the instance

Reads VAST_API_KEY, HF_TOKEN, WANDB_API_KEY from .env. The instance clones the repo,
runs scripts/vast_suite.sh, logs to W&B, and destroys itself when finished or when the
wall-clock cap derived from --budget is reached.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.utils import load_env_file  # noqa: E402

IMAGE = "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"
SUITE_URL = "https://raw.githubusercontent.com/Tqhuyen/glaucoma-thesis/master/scripts/vast_suite.sh"
GPUS = ("RTX_3090", "RTX_4090")


def _session_key():
    try:
        from vastai.cli.util import TFAKEY_FILE
    except ImportError:
        return None
    path = Path(TFAKEY_FILE)
    return path.read_text().strip() if path.exists() else None


def vast(args, key):
    exe = shutil.which("vastai")
    if exe is None:
        raise SystemExit("vastai CLI not found: pip install vastai")
    use = key if args[0] == "search" else (_session_key() or key)
    out = subprocess.run([exe, "--api-key", use, *args], capture_output=True, text=True)
    text = (out.stdout + out.stderr).replace(use, "***").replace(key, "***")
    if out.returncode != 0 or '"error": true' in text or "Failed with error" in text:
        raise SystemExit(f"vastai {' '.join(args[:2])} failed: {text.strip()[:400]}")
    return out.stdout if args[0] == "search" else text


def hourly_cost(offer, disk):
    return offer["dph_total"] + offer.get("storage_cost", 0.0) * disk / 730.0


def find_offer(key, disk, max_dph):
    candidates = []
    for gpu in GPUS:
        query = (
            f"reliability>0.98 num_gpus=1 gpu_name={gpu} disk_space>={disk} cpu_ram>=32 "
            "cpu_cores_effective>=8 cuda_vers>=12.1 inet_down>=300 rentable=True verified=True "
            "geolocation notin [CN,VN,RU,IN]"
        )
        offers = json.loads(vast(["search", "offers", query, "-o", "dph+", "--raw"], key) or "[]")
        candidates += [o for o in offers if hourly_cost(o, disk) <= max_dph]
    return min(candidates, key=lambda o: hourly_cost(o, disk), default=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=5.0, help="USD ceiling for the whole suite")
    ap.add_argument("--disk", type=int, default=60)
    ap.add_argument("--max-dph", type=float, default=0.45)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--suite", default="raw200-resnext:42 raw200-simple:42 raw200-pyramid:42 raw200-resnext:43")
    ap.add_argument("--launch", action="store_true")
    args = ap.parse_args()

    load_env_file()
    missing = [k for k in ("VAST_API_KEY", "HF_TOKEN", "WANDB_API_KEY") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"missing in .env: {', '.join(missing)}")
    key = os.environ["VAST_API_KEY"]

    offer = find_offer(key, args.disk, args.max_dph)
    if offer is None:
        raise SystemExit("no verified RTX 3090/4090 offer under the price cap right now")

    dph = offer["dph_total"]
    storage = offer.get("storage_cost", 0.0) * args.disk / 730.0
    hourly = dph + storage
    max_minutes = int(args.budget * 0.9 / hourly * 60)
    print(
        f"offer {offer['id']}: {offer['gpu_name']} x{offer['num_gpus']} {offer['gpu_ram'] / 1024:.0f} GB, "
        f"${dph:.3f}/h (+${storage:.3f}/h storage), reliability {offer['reliability']:.3f}, "
        f"{offer['inet_down']:.0f} Mb/s down, {offer.get('geolocation', '?')}"
    )
    print(f"budget ${args.budget:.2f} -> hard cap {max_minutes} min ({max_minutes / 60:.1f} h), suite: {args.suite}")

    if not args.launch:
        print("dry run only; add --launch to create the instance")
        return

    env = (
        f"-e HF_TOKEN={os.environ['HF_TOKEN']} -e WANDB_API_KEY={os.environ['WANDB_API_KEY']} "
        f"-e MAX_MINUTES={max_minutes} -e EPOCHS={args.epochs} -e SUITE={args.suite.replace(' ', ',')}"
    )
    onstart = (
        "bash -lc 'mkdir -p /workspace && cd /workspace && "
        "(command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl)) && "
        f"curl -fsSL {SUITE_URL} -o suite.sh && nohup bash suite.sh > /workspace/launch.out 2>&1 &'"
    )
    out = vast(
        [
            "create",
            "instance",
            str(offer["id"]),
            "--image",
            IMAGE,
            "--disk",
            str(args.disk),
            "--env",
            env,
            "--onstart-cmd",
            onstart,
            "--raw",
        ],
        key,
    )
    print(out.strip() or "(empty response)")
    if "success" not in out.lower():
        raise SystemExit("create instance did not confirm success; check the vast.ai console before retrying")
    print("launched; follow progress in W&B project glaucoma-thesis (runs raw200-*-s42 / -s43)")


if __name__ == "__main__":
    main()
