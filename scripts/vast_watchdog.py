"""Local cost watchdog for a vast.ai training instance.

Runs on your machine next to the rented GPU and makes sure billing stops:
  * every --interval seconds it pulls run outputs (no checkpoints/tensorboard) over SSH
    into artifacts/vast/;
  * it destroys the instance when the suite has finished but the box is still alive,
    when training stalls (no metrics.jsonl update for --stale-min minutes),
    when the instance is stopped/exited (storage is still billed), or at --deadline.

    python scripts/vast_watchdog.py --instance 60056240 --deadline 2026-11-11T07:50:58Z

SSH key: VAST_SSH_KEY in .env (a path, or the private key text), else ~/.ssh/vast_ed25519.
Account calls use the 2FA session from `vastai tfa login`.
"""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vast_launch as v  # noqa: E402

from pipeline.utils import SecureStr, load_env_file, redact, secret_env  # noqa: E402

DEST = ROOT / "artifacts" / "vast"
REMOTE_PROBE = r"""
cd /workspace 2>/dev/null || exit 3
train=$(pgrep -f "pipeline[.]train|[n]bconvert|hsi[_]mphsir|train[.]py --arm|evaluate[.]py --data" | head -1)
suite=$(pgrep -f "suite[0-9]?[.]sh" | head -1)
newest=$(find training_runs dl_lab_run post_task hsi_run -type f -printf "%T@\n" 2>/dev/null | sort -n | tail -1)
echo "train=${train:-none} suite=${suite:-none} newest=${newest:-0} now=$(date +%s)"
"""


def log(msg):
    print(f"[{dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}] {msg}", flush=True)


def ssh_key_path() -> Path:
    raw = os.environ.get("VAST_SSH_KEY", "").strip()
    if raw and Path(raw).expanduser().is_file():
        return Path(raw).expanduser()
    if "PRIVATE KEY" in raw:
        SecureStr(raw)
        fd, name = tempfile.mkstemp(prefix="vast_key_")
        with os.fdopen(fd, "w", newline="\n") as fh:
            fh.write(raw.replace("\\n", "\n").strip() + "\n")
        os.chmod(name, 0o600)
        return Path(name)
    return Path.home() / ".ssh" / "vast_ed25519"


def instance(key, inst_id):
    raw = v.vast(["show", "instances", "--raw"], key)
    start = raw.find("[")
    items = json.loads(raw[start:]) if start >= 0 else []
    return next((i for i in items if i["id"] == inst_id), None)


def ssh(key_file, info, command, binary=False):
    host, port = info.get("ssh_host"), info.get("ssh_port")
    cmd = [
        "ssh",
        "-i",
        str(key_file),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-p",
        str(port),
        f"root@{host}",
        command,
    ]
    out = subprocess.run(cmd, capture_output=True, timeout=600, text=not binary)
    return out.returncode, out.stdout


def sync(key_file, info):
    rc, data = ssh(
        key_file,
        info,
        'cd /workspace && tar czf - --exclude="*/checkpoints" --exclude="*/tb" $(ls -d training_runs suite.log '
        "dl_lab_run/outputs dl_lab_run/executed.ipynb post_task/outputs post_task/executed.ipynb hsi_run/outputs 2>/dev/null) "
        "2>/dev/null",
        binary=True,
    )
    if rc != 0 or not data:
        return False
    DEST.mkdir(parents=True, exist_ok=True)
    tgz = DEST / "snapshot.tgz"
    tgz.write_bytes(data)
    with tarfile.open(tgz) as tar:
        tar.extractall(DEST, filter="data")
    runs = sorted(p.name for p in (DEST / "training_runs").iterdir()) if (DEST / "training_runs").exists() else []
    log(f"synced {len(data) / 1e6:.1f} MB: {', '.join(runs)}")
    return True


def destroy(key, inst_id, reason):
    log(f"DESTROY {inst_id}: {reason}")
    try:
        log(redact(v.vast(["destroy", "instance", str(inst_id)], key)).strip()[:200])
        return True
    except SystemExit as e:
        log(f"destroy failed: {e} -> run `vastai tfa login` then `vastai destroy instance {inst_id}`")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=int, required=True)
    ap.add_argument("--deadline", required=True, help="UTC ISO time, e.g. 2026-11-11T07:50:58Z")
    ap.add_argument("--stale-min", type=int, default=30)
    ap.add_argument("--interval", type=int, default=600)
    args = ap.parse_args()

    load_env_file()
    key = secret_env("VAST_API_KEY")
    key_file = ssh_key_path()
    deadline = dt.datetime.fromisoformat(args.deadline.replace("Z", "+00:00")).timestamp()
    log(f"watching instance {args.instance}; deadline {args.deadline}; stall limit {args.stale_min} min")

    while True:
        info = instance(key, args.instance)
        if info is None:
            log("instance gone (destroyed); watchdog done")
            return
        status = info.get("actual_status")
        if status in ("exited", "stopped", "offline"):
            destroy(key, args.instance, f"instance {status}; storage is still billed")
            return
        if time.time() >= deadline:
            sync(key_file, info)
            destroy(key, args.instance, "budget deadline reached")
            return
        rc, out = ssh(key_file, info, REMOTE_PROBE)
        if rc == 0 and out:
            fields = dict(part.split("=", 1) for part in out.split())
            age_min = (float(fields["now"]) - float(fields["newest"] or 0)) / 60
            log(f"status={status} train={fields['train']} suite={fields['suite']} metrics age={age_min:.0f} min")
            sync(key_file, info)
            if fields["train"] == "none" and fields["suite"] == "none":
                destroy(key, args.instance, "suite finished but instance still running")
                return
            if fields["train"] != "none" and age_min > args.stale_min:
                destroy(key, args.instance, f"training stalled: no metrics for {age_min:.0f} min")
                return
        else:
            log(f"status={status}; ssh not reachable yet (rc={rc})")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
