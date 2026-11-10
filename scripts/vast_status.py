import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vast_launch as v  # noqa: E402

from pipeline.utils import load_env_file  # noqa: E402

load_env_file()
key = os.environ["VAST_API_KEY"].strip()
raw = v.vast(["show", "instances", "--raw"], key)
start = raw.find("[")
items = json.loads(raw[start:]) if start >= 0 else []
for i in items:
    print(
        {
            k: i.get(k)
            for k in ("id", "actual_status", "cur_state", "status_msg", "gpu_name", "dph_total", "geolocation")
        }
    )
    if len(sys.argv) > 1:
        logs = v.vast(["logs", str(i["id"]), "--tail", sys.argv[1]], key)
        print(logs[-int(sys.argv[2]) if len(sys.argv) > 2 else -2500 :])
if not items:
    print("no instances")
