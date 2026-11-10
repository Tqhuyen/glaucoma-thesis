---
name: vast-cost-training
description: Use when the user wants to train or run GPU experiments on a rented vast.ai machine at minimum cost (thuê GPU vast.ai, train tiết kiệm chi phí, chạy suite trên vast, tự huỷ máy khi xong, lưu kết quả lên HF). Covers offer selection, budget caps, 2FA, launch, live monitoring, chaining post tasks, result persistence (HF + local + W&B replay), self-destroy, and secret hygiene. Trigger on "vast.ai", "vast", "thuê GPU", "rent GPU", "chi phí tối ưu", "tự huỷ máy", "budget $5".
---

# vast-cost-training

One rented GPU, a fixed USD budget, zero idle billing, nothing lost.
Every rule below comes from a real run in this repo (Nov 2026, RTX 3090, $5 budget).

## Files

```
scripts/vast_launch.py        pick cheapest safe offer, compute hard cap, create instance
scripts/vast_suite.sh         on-instance controller: runs, uploads to HF, chains post task, self-destroys
scripts/vast_raw200.sh        one training run (downloads raw 200^3 once, reuses it)
scripts/vast_dl_lab_denoise.sh  example post task (deep-learning-laboratory notebook via nbconvert)
scripts/vast_watchdog.py      local safety net: sync outputs, destroy on finish/stall/stop/deadline
scripts/vast_status.py        show instances (+ last log lines), secrets redacted
pipeline/utils.py             SecureStr / secret_env / redact (never print secrets)
```

## Cost model (read before choosing anything)

- Cost per hour = `dph_total + storage_cost * disk_GB / 730`. The launcher ranks by this, not by `dph`.
- **Destroy beats stop.** A stopped instance still bills storage; only `destroy` ends billing.
- Hard cap = `0.9 * budget / hourly` minutes. Enforced three times: `timeout` per run, controller
  deadline, local watchdog deadline.
- Measured on RTX 3090 24 GB, raw 200^3, bs1 x accum 8, AMP (native bf16):

| Model | s/sample | min/epoch | 8 epochs |
|---|---:|---:|---:|
| PyramidResNeXt3D | 0.29 | ~13 | ~1.9 h |
| Simple3DCNN | 0.11 | ~4 | ~35 min |
| (T4 on Modal, fp16, same ResNeXt) | 0.78 | ~27 | ~3.6 h |

- At $0.21/h the full suite (4 runs) + a 3-4 h post task costs ~$2-3.

## GPU choice

| Pick | Why |
|---|---|
| **RTX 3090 24 GB** (default) | cheapest card with 24 GB; native bf16 (Ampere); ~$0.14-0.21/h |
| RTX 4090 24 GB | ~1.5-2x faster, ~2x price; only if wall-clock matters |
| A100 40/80 GB | not needed for <2M-param 3D CNNs at bs1-2 |
| <=16 GB cards (4080, 4070 Ti, T4) | ResNeXt+FPN at 200^3 peaks 11.5-16 GB; bs2 OOMs on 15 GB |

Offer filter used by `find_offer`: `reliability>0.98 num_gpus=1 verified=True rentable=True
cuda_vers>=12.1 cpu_ram>=32 cpu_cores_effective>=8 inet_down>=300 disk_space>=60
geolocation notin [CN,VN,RU,IN]` (CN hosts had very slow access to HF/GitHub).

## Prerequisites (one time)

1. `.env` (gitignored) contains `VAST_API_KEY`, `HF_TOKEN` (write scope), `WANDB_API_KEY`,
   optional `VAST_SSH_KEY` (path to a private key or the key text; default `~/.ssh/vast_ed25519`).
2. `pip install vastai`; store the key for the CLI without echoing it:
   `python -c "from pipeline.utils import load_env_file; import os,subprocess; load_env_file(); subprocess.run(['vastai','set','api-key',os.environ['VAST_API_KEY'].strip()])"`
3. **2FA.** With 2FA enabled every account call (create/show/destroy) returns
   `401 ... requires 2FA session` until the **user** runs, on their own machine:
   `vastai tfa login -t totp -c <6-digit code>`. TOTP codes live ~30 s: never relay them through chat.
   The session is stored in `~/.config/vastai/vast_tfa_key`; `vast()` in `vast_launch.py` uses it
   automatically for non-search calls. Offer search works without 2FA.
4. Fund vast.ai with exactly the budget and **disable auto-recharge** (hard money stop).

## Launch workflow

```bash
python scripts/vast_launch.py --budget 5                       # dry run: offer, $/h, hard cap
python scripts/vast_launch.py --budget 5 --launch \
  --suite "raw200-resnext:42 raw200-simple:42" --epochs 8 \
  --post-task scripts/vast_dl_lab_denoise.sh --post-name urban100-s25-seed2
# copy the printed line and start the watchdog (keep it running locally):
python scripts/vast_watchdog.py --instance <ID> --deadline <UTC> --stale-min 45
```

Within the first 10 minutes **verify**, do not assume:
1. `python scripts/vast_status.py 30` -> `actual_status: running`.
2. Attach SSH and probe (the ssh shell does NOT inherit the container env):
   ```bash
   python - <<'EOF'   # attach key once
   import os, sys; sys.path.insert(0, "scripts"); import vast_launch as v
   from pipeline.utils import load_env_file, secret_env; load_env_file()
   print(v.vast(["attach", "ssh", "<ID>", open(os.path.expanduser("~/.ssh/vast_ed25519.pub")).read().strip()], secret_env("VAST_API_KEY")))
   print(v.vast(["ssh-url", "<ID>"], secret_env("VAST_API_KEY")))
   EOF
   ssh -i ~/.ssh/vast_ed25519 -p <PORT> root@<HOST> 'tail -n 20 /workspace/suite.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'
   ```
3. Check the log line `[env] ... amp=...` and the `it/s` figure. On pre-Ampere GPUs
   (T4/P100) AMP must be fp16; bf16 there is software emulation, ~10x slower
   (`resolve_amp_dtype` now requires native bf16).
4. Confirm outputs persist: after the first run `[suite] uploaded <run> -> hf://...` appears.

## What the controller guarantees (`vast_suite.sh`)

- Runs `SUITE` sequentially (`name:seed,...`), each bounded by the remaining cap; skips a run
  if < 30 min remain.
- Probes W&B; if the host is blocked (seen: `403 Forbidden /graphql` from a Polish host) it
  sets `WANDB_ENABLED=false` instead of hanging.
- After each run uploads the **whole run folder** (best_model, checkpoints, metrics.jsonl,
  TensorBoard, config, test_results) to the private HF repo `RESULTS_REPO/runs/<run>`.
- Runs `POST_TASK` with the remaining cap and uploads `outputs/**`, `task.ipynb`,
  `executed.ipynb` to `RESULTS_REPO/post/<POST_NAME>` (never re-uploads public weights/datasets).
- `trap EXIT`: final sync of all runs + `logs/suite.log`, `logs/launch*.out`, then
  `vastai destroy` with the instance-scoped `CONTAINER_API_KEY` (verified: it can read its own
  instance under 2FA; confirm the destroy path on the first finished run);
  if refused -> `stop`; finally `kill -TERM 1` so GPU billing ends even if every API call fails.

## Local watchdog rules (`vast_watchdog.py`)

| Condition | Action |
|---|---|
| suite and training both gone, box still running | sync -> destroy |
| training/nbconvert alive but no file updated for `--stale-min` | sync -> destroy |
| instance `exited` / `stopped` / `offline` (storage still billed) | destroy |
| past `--deadline` | sync -> destroy |
| instance gone | exit |

It pulls `training_runs` (no checkpoints/tb) + post outputs into `artifacts/vast/` every
`--interval` s. It destroys with the user's 2FA session; if that session expired it logs the
exact manual command. Restart it whenever the remote controller script name changes.

## Changing the plan on a live instance (no lost GPU time)

Never edit a running bash script in place. Swap the controller:

```bash
ssh ... 'set -e; cd /workspace
umask 077; tr "\0" "\n" < /proc/<OLD_SUITE_PID>/environ > /root/.suite_env   # once, from the original controller
deadline=$(( $(stat -c %Y /workspace/suite.sh) + $(grep ^MAX_MINUTES= /root/.suite_env | cut -d= -f2) * 60 ))
curl -fsSL https://raw.githubusercontent.com/Tqhuyen/glaucoma-thesis/master/scripts/vast_suite.sh -o suiteN.sh
bash -n suiteN.sh
kill -9 $(pgrep -f "suite[0-9]?[.]sh")        # SIGKILL: old EXIT trap must NOT fire (it would destroy)
nohup env -i bash -c "while IFS= read -r l; do export \"\$l\"; done < /root/.suite_env; \
  export WAIT_PID=<TRAIN_TIMEOUT_PID> WAIT_RUN=<run> DEADLINE_EPOCH=$deadline SUITE=<remaining> \
  POST_TASK=... POST_NAME=...; exec bash /workspace/suiteN.sh" > /workspace/launchN.out 2>&1 &'
```

`WAIT_PID` is the `timeout ... vast_raw200.sh` process of the run in flight; the new controller
adopts it, uploads it when it ends, then continues. Training keeps writing to the old `tee`.
Then restart the local watchdog.

## Secret hygiene

- Read secrets with `secret_env("NAME")` -> `SecureStr`; it prints `***abcd`. Use `.reveal()` only
  at the exact call site (subprocess arg, env var, HTTP header). Pass captured output through
  `redact()` before printing.
- Strip keys: a trailing `\r` from Windows produced an error that echoed the full API key.
- `instance_api_key` in `create instance` responses is redacted by `vast()`.
- Container env (`--env`) is visible in the vast console; the instance self-destroys, but prefer
  scoped HF/W&B tokens.
- Never print `.env`, `/proc/*/environ`, or `kaggle auth print-access-token` output.

## Pitfalls seen in practice

| Symptom | Cause / fix |
|---|---|
| `401 ... Two Factor Authentication` | user runs `vastai tfa login`; launcher then uses the session key |
| `execute only avail on stopped instances` | use SSH (attach key) for running instances |
| ssh shell lacks HF_TOKEN etc. | read `/proc/<controller pid>/environ` into a 600 file |
| `syntax error near $'in\r'` | CRLF in `.sh`; `.gitattributes` has `*.sh text eol=lf` |
| `pgrep -f pattern` always matches | the probe's own cmdline contains it; use `pipeline[.]train` |
| `pkill -f X` / `pgrep -f X` over ssh matches the ssh command's own shell (killed our session mid-swap; made the watchdog see a phantom trainer) | kill by explicit PID; in patterns bracket one char: `[n]bconvert`, `hsi[_]mphsir` |
| W&B 403 on some hosts (even `wandb.ai` homepage: IP-level block, no user whitelist on SaaS) | outputs go to HF; `python scripts/wandb_replay.py artifacts/vast/training_runs --follow` streams synced metrics to W&B from the local machine (~10 min lag) |
| `train.lr=1e-4` rejected as str | YAML override: write `0.0001` |
| outputs under `.../<run>/<run>` | `pipeline.train` already appends `run_name` to `output_dir` |
| Modal Volume 26 GB mmap ~30 s/sample | keep data on local NVMe of the box, never on network volumes |
| Kaggle "Maximum batch GPU session count of 2" | Kaggle limit; vast has none |

## Finish checklist

- [ ] `python scripts/vast_status.py` -> `no instances` (destroyed, not stopped).
- [ ] HF repo `runs/<run>/test_results.json` exists for every run; `post/<name>/outputs/DONE` for post task.
- [ ] If the host was W&B-blocked: `python scripts/wandb_replay.py artifacts/vast/training_runs` (adds artifact + `final/*` summary per finished run).
- [ ] Record results + cost in `docs/` and commit.
