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
    
    if status == "COMPLETE":
        try:
            api.kernels_output(EVAL_KERNEL_ID, path=str(out_dir))
            summary_file = out_dir / "eval_summary.json"
            if summary_file.exists():
                with open(summary_file, "r", encoding="utf-8") as f:
                    return status, json.load(f)
        except Exception as e:
            print(f"[Eval Download] Error: {e}")
            
    return status, None


def append_benchmark_to_readme(base_metrics, rl_metrics):
    """Appends the empirical LLM benchmark rows to Section 2 of README.md."""
    root_dir = Path(__file__).resolve().parent.parent
    readme_path = root_dir / "README.md"
    
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    base_row = (
        f"| **Base Qwen-1.5B-Instruct** | Zero-Shot Pre-Trained LLM | "
        f"{base_metrics.get('Zero-Viol %', 'N/A')} | "
        f"{base_metrics.get('Survival %', 'N/A')} | "
        f"{base_metrics.get('ICU Overflows', 'N/A')} | "
        f"{base_metrics.get('Vent Overflows', 'N/A')} | "
        f"{base_metrics.get('Total Violations', 'N/A')} |"
    )

    rl_row = (
        f"| **GoldenHour-RL (Step 120)** | Post-Trained GRPO LoRA | "
        f"**{rl_metrics.get('Zero-Viol %', 'N/A')}** | "
        f"**{rl_metrics.get('Survival %', 'N/A')}** | "
        f"**{rl_metrics.get('ICU Overflows', 'N/A')}** | "
        f"**{rl_metrics.get('Vent Overflows', 'N/A')}** | "
        f"**{rl_metrics.get('Total Violations', 'N/A')}** |"
    )

    # Insert rows right before the Oracle row if not already added
    if "Base Qwen-1.5B-Instruct" not in content:
        target_marker = "| **Oracle Target Schedule**"
        replacement = f"{base_row}\n{rl_row}\n{target_marker}"
        content = content.replace(target_marker, replacement)
        
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[README] Successfully appended LLM evaluation rows to {readme_path.name}!")
    else:
        print("[README] Evaluation rows already present in README.md.")


if __name__ == "__main__":
    t_status, t_fail = check_training_status()
    print(f"[Training Kernel] Status: {t_status} | Failure: {t_fail}")
    
    adapter_ok, hf_files = verify_hf_adapter()
    print(f"[Hugging Face Hub] Adapter present: {adapter_ok} | Files: {len(hf_files)}")

    e_status, e_data = check_eval_benchmark()
    print(f"[Eval Kernel] Status: {e_status} | Has Data: {e_data is not None}")
    if e_data:
        print("[Eval Results]:", json.dumps(e_data, indent=2))
        base_res = e_data.get("base_qwen_1_5b", {})
        rl_res = e_data.get("goldenhour_rl_grpo", {})
        append_benchmark_to_readme(base_res, rl_res)
