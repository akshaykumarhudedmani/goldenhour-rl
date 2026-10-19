import os
from dotenv import load_dotenv
load_dotenv()
import wandb

api = wandb.Api()
runs = api.runs("akshaykumarhudedmani-rl/goldenhour-rl")
for r in runs:
    if r.state == "running":
        print(f"=== Active Run: {r.name} ===")
        hist = r.history(pandas=False)
        print(f"History entries logged: {len(hist)}")
        if hist:
            last = hist[-1]
            for k in ["_step", "loss", "reward", "rewards/format_reward_func/mean", "rewards/verifier_reward_func/mean", "completion_length", "completions/clipped_ratio"]:
                if k in last:
                    print(f"  {k}: {last[k]}")
        break
