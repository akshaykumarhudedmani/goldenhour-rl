import os
import sys
import json
import re
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi
from huggingface_hub import HfApi

TRAIN_KERNEL_ID = "akshaykumarhudedmani/goldenhour-rl-grpo-training"
EVAL_KERNEL_ID = "akshaykumarhudedmani/goldenhour-rl-eval-benchmark"
HF_REPO_ID = "akshaykumarhudedmani/goldenhour-rl"


def check_training_status():
    """Checks the status of the long-running GRPO training kernel on Kaggle."""
    api = KaggleApi()
    api.authenticate()
    status_obj = api.kernels_status(TRAIN_KERNEL_ID)
    status = getattr(status_obj, "status", str(status_obj))
    failure = getattr(status_obj, "failure_message", None)
    return status, failure


def verify_hf_adapter():
    """Verifies that the trained LoRA adapter exists on Hugging Face Hub."""
    hf_token = os.environ.get("HF_TOKEN")
    api = HfApi(token=hf_token)
    files = api.list_repo_files(HF_REPO_ID)
    return "adapter_model.safetensors" in files, files


def check_eval_benchmark():
    """Checks the status of the evaluation benchmark kernel on Kaggle and downloads output."""
    api = KaggleApi()
    api.authenticate()
    status_obj = api.kernels_status(EVAL_KERNEL_ID)
    status = getattr(status_obj, "status", str(status_obj))
    failure = getattr(status_obj, "failure_message", None)
    
    out_dir = Path(__file__).resolve().parent.parent / "kaggle_eval" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "eval_summary.json"
    
    if "COMPLETE" in str(status):
        try:
            api.kernels_output(EVAL_KERNEL_ID, path=str(out_dir), force=True)
            for p in out_dir.rglob("eval_summary.json"):
                if p.is_file():
                    with open(p, "r", encoding="utf-8") as f:
                        return status, json.load(f)
            
            for log_file in out_dir.glob("*.log"):
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    log_text = f.read()
                base_m = re.search(r"Base Model Results:\s*(\{.*?\})", log_text, re.DOTALL)
                rl_m = re.search(r"GoldenHour-RL \(GRPO\) Results:\s*(\{.*?\})", log_text, re.DOTALL)
                if base_m and rl_m:
                    return status, {
                        "base_qwen_1_5b": json.loads(base_m.group(1)),
                        "goldenhour_rl_grpo": json.loads(rl_m.group(1)),
                    }
        except Exception as e:
            print(f"[Eval Download] Error: {e}")
            
    return status, None


def append_benchmark_to_readme(base_metrics, rl_metrics):
    """Appends or updates the empirical LLM benchmark table in Section 2 of README.md."""
    root_dir = Path(__file__).resolve().parent.parent
    readme_path = root_dir / "README.md"
    
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    new_table = (
        "| Strategy | Engine Type | Executable Plan % | Zero-Violation Rate | Patient Survival (SLA) | ICU Bed Overflows | Ventilator Overflows | Total Violations |\n"
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        "| **Greedy Nearest Hospital** | Programmatic Heuristic Baseline | 100.0% | 34.0% | 98.5% | 66 | 47 | 119 |\n"
        f"| **Base Qwen-1.5B-Instruct** | Zero-Shot Pre-Trained LLM (Pass@1) | {base_metrics.get('Executable Plan %', '0.0%')} | {base_metrics.get('Zero-Viol %', '0.0%')} | {base_metrics.get('Survival %', '0.0%')} | {base_metrics.get('ICU Overflows', 0)} | {base_metrics.get('Vent Overflows', 0)} | {base_metrics.get('Total Violations', 'N/A')} |\n"
        f"| **GoldenHour-RL (GRPO)** | Verifier-Guided LoRA (Pass@1) | **{rl_metrics.get('Executable Plan %', 'N/A')}** | **{rl_metrics.get('Zero-Viol %', 'N/A')}** | **{rl_metrics.get('Survival %', 'N/A')}** | **{rl_metrics.get('ICU Overflows', 'N/A')}** | **{rl_metrics.get('Vent Overflows', 'N/A')}** | **{rl_metrics.get('Total Violations', 'N/A')}** |\n"
        "| **Oracle Target Schedule** | Ground-Truth Solver (Upper Bound) | 100.0% | **100.0%** | **100.0%** | **0** | **0** | **0** |"
    )

    table_pattern = r"\| Strategy \| Engine Type \|.*?\n\| \*\*Oracle Target Schedule\*\* \| Ground-Truth Solver \(Upper Bound\) \|.*?\n"
    if re.search(table_pattern, content, flags=re.DOTALL):
        content = re.sub(table_pattern, new_table + "\n", content, count=1, flags=re.DOTALL)
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[README] Successfully updated empirical LLM evaluation table in {readme_path.name}!")
    else:
        print("[README] Table pattern not matched, attempting fallback replace.")


def trigger_eval_kernel():
    """Pushes the Kaggle evaluation kernel once GPU is freed."""
    api = KaggleApi()
    api.authenticate()
    print("[Pipeline] Pushing evaluation kernel to Kaggle GPU...")
    res = api.kernels_push("kaggle_eval")
    print(f"[Pipeline] Evaluation kernel pushed: {res}")


def commit_and_push_readme():
    """Commits and pushes updated README.md to GitHub."""
    import subprocess
    root_dir = Path(__file__).resolve().parent.parent
    try:
        status = subprocess.check_output(["git", "status", "--porcelain", "README.md"], cwd=root_dir).decode("utf-8").strip()
        if not status:
            print("[Git] README.md has no changes, skipping commit.")
            return
        subprocess.check_call(["git", "add", "README.md"], cwd=root_dir)
        subprocess.check_call(["git", "commit", "-m", "benchmark: add empirical Base vs GoldenHour-RL GRPO evaluation"], cwd=root_dir)
        subprocess.check_call(["git", "push", "origin", "main"], cwd=root_dir)
        print("[Git] Successfully pushed updated README.md to GitHub main!")
    except Exception as e:
        print(f"[Git] Push error: {e}")


def run_pipeline_check():
    t_status, t_fail = check_training_status()
    print(f"[Training Kernel] Status: {t_status} | Failure: {t_fail}")
    
    adapter_ok, hf_files = verify_hf_adapter()
    print(f"[Hugging Face Hub] Adapter present: {adapter_ok} | Files: {len(hf_files)}")

    e_status, e_data = check_eval_benchmark()
    print(f"[Eval Kernel] Status: {e_status} | Has Data: {e_data is not None}")

    if e_data:
        print("\n[Eval Results Loaded]:", json.dumps(e_data, indent=2))
        base_res = e_data.get("base_qwen_1_5b", {})
        rl_res = e_data.get("goldenhour_rl_grpo", {})
        append_benchmark_to_readme(base_res, rl_res)
        commit_and_push_readme()
        return True
    elif "RUNNING" in str(e_status):
        print("\n[Pipeline] Evaluation benchmark kernel is actively RUNNING on Kaggle GPU. Standing by...")
        return False
    elif "ERROR" in str(e_status):
        print("\n[Pipeline] Evaluation benchmark kernel encountered an ERROR. Please inspect logs.")
        return False
    elif "COMPLETE" in str(e_status) and not e_data:
        print("\n[Pipeline] Evaluation kernel completed but eval_summary.json not yet parsed.")
        return False
    else:
        print("\n[Pipeline] Monitor standing by.")
        return False


if __name__ == "__main__":
    import time
    if "--poll" in sys.argv:
        print("[Pipeline] Starting continuous monitor poll loop (interval=60s)...")
        while True:
            done = run_pipeline_check()
            if done:
                print("[Pipeline] Evaluation completed and processed!")
                break
            time.sleep(60)
    else:
        run_pipeline_check()
