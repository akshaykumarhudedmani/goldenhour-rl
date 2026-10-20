import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
if "KAGGLE_API_TOKEN" in os.environ and "KAGGLE_KEY" not in os.environ:
    os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_TOKEN"]

import kaggle

api = kaggle.KaggleApi()
api.authenticate()
kernel_slug = "akshaykumarhudedmani/goldenhour-rl-grpo-training"

seen_lines = set()
log_file = Path("logs/kaggle_live.log")
if log_file.exists():
    for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
        seen_lines.add(line.strip())

print(f"[Watcher] Auto-reconnecting stream monitor for {kernel_slug}...")
print(f"[Watcher] Currently tracking Step 25 landing. Total indexed lines: {len(seen_lines)}")

consecutive_errors = 0
step_25_reached = False

while not step_25_reached and consecutive_errors < 10:
    try:
        for entry in api.kernels_logs_stream(kernel_slug):
            data = entry.get("data", "")
            for raw in data.splitlines():
                r = raw.strip()
                if r and r not in seen_lines:
                    seen_lines.add(r)
                    now = time.strftime("%H:%M:%S")
                    safe = r.encode("ascii", errors="replace").decode("ascii")
                    print(f"[{now}] {safe}", flush=True)
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(r + "\n")
                    if "25/50" in r or "checkpoint-25" in r:
                        step_25_reached = True
        consecutive_errors = 0
    except Exception as e:
        consecutive_errors += 1
        time.sleep(3)

print("[Watcher] Step 25 milestone detected or monitor cycle concluded.", flush=True)
