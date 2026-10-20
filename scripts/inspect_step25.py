import json
import sys
from pathlib import Path

def inspect_state(state_path="checkpoint-25/trainer_state.json"):
    p = Path(state_path)
    if not p.exists():
        print(f"Error: {p} does not exist yet.")
        return

    with open(p, "r", encoding="utf-8") as f:
        state = json.load(f)

    print(f"{'Step':<6} | {'Loss':<8} | {'Format R':<10} | {'Verifier R':<12} | {'KL':<8} | {'Comp Len':<8}")
    print("-" * 64)

    for entry in state.get("log_history", []):
        if "rewards/format_reward_func/mean" in entry or "reward" in entry or "loss" in entry:
            step = entry.get("step", "N/A")
            loss = round(entry.get("loss", 0.0), 4)
            fmt_r = round(entry.get("rewards/format_reward_func/mean", entry.get("format_reward", 0.0)), 3)
            ver_r = round(entry.get("rewards/verifier_reward_func/mean", entry.get("verifier_reward", 0.0)), 3)
            kl = round(entry.get("kl", 0.0), 4)
            comp_len = round(entry.get("completion_length", entry.get("mean_completion_length", 0.0)), 1)
            print(f"{step:<6} | {loss:<8} | {fmt_r:<10} | {ver_r:<12} | {kl:<8} | {comp_len:<8}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoint-25/trainer_state.json"
    inspect_state(path)
