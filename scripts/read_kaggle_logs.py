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

log_file = Path("logs/kaggle_live.log")
log_file.parent.mkdir(parents=True, exist_ok=True)

lines = []
start_t = time.time()
max_runtime = 25.0  # gather data for at most 25s

try:
    for entry in api.kernels_logs_stream(kernel_slug):
        data = entry.get("data", "")
        for raw in data.splitlines():
            r = raw.strip()
            if r:
                lines.append(r)
        if time.time() - start_t > max_runtime:
            break
except Exception as e:
    pass

if lines:
    with open(log_file, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")

print(f"Total lines captured: {len(lines)}")
print("=== LAST 25 LINES ===")
for l in lines[-25:]:
    safe = l.encode("ascii", errors="replace").decode("ascii")
    print(safe)
