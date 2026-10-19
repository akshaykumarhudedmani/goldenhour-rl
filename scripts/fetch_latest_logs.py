import os
import sys
import time
import queue
import threading
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

q = queue.Queue()
stop_flag = threading.Event()

def worker():
    try:
        for entry in api.kernels_logs_stream(kernel_slug):
            if stop_flag.is_set():
                break
            q.put(entry)
    except Exception:
        pass
    finally:
        q.put(None)

t = threading.Thread(target=worker, daemon=True)
t.start()

lines = []
start_time = time.time()
last_chunk_time = time.time()
max_total_time = 75.0  # At most 75 seconds

while True:
    try:
        entry = q.get(timeout=1.0)
        if entry is None:
            break
        data = entry.get("data", "")
        for raw in data.splitlines():
            r = raw.strip()
            if r:
                lines.append(r)
        last_chunk_time = time.time()
    except queue.Empty:
        # Check silence timeout ONLY if we have already caught up past the backlog (>250 lines)
        if len(lines) > 250 and (time.time() - last_chunk_time > 8.0):
            stop_flag.set()
            break
        if time.time() - start_time > max_total_time:
            stop_flag.set()
            break

with open(log_file, "w", encoding="utf-8") as f:
    for l in lines:
        f.write(l + "\n")

print(f"Total lines captured: {len(lines)}")
print("=== LATEST 30 LOG LINES ===")
for l in lines[-30:]:
    safe = l.encode("ascii", errors="replace").decode("ascii")
    print(safe)
