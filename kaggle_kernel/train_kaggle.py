import os
import sys
import subprocess
import torch

# 1. Environment Secrets & Config
# Retrieve credentials securely from Kaggle UserSecrets or OS environment
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")
    if not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = user_secrets.get_secret("WANDB_API_KEY")
except Exception:
    pass

os.environ.setdefault("WANDB_PROJECT", "goldenhour-rl")

print("=" * 70)
print("     PROJECT GOLDENHOUR-RL: CLOUD GPU POST-TRAINING PIPELINE")
print("=" * 70)
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

# 2. Install Required Dependencies
print("\n[Step 1/6] Installing dependencies (unsloth, trl, peft, wandb)...")
cmd_install = [
    sys.executable, "-m", "pip", "install", "--quiet",
    "unsloth", "trl>=0.14.0", "peft>=0.10.0", "wandb", "datasets", "pytest"
]
subprocess.check_call(cmd_install)
print("[Step 1/6] Dependencies successfully installed.")

# 3. Clone Repository
print("\n[Step 2/6] Cloning GoldenHour-RL repository...")
repo_dir = "/kaggle/working/goldenhour-rl"
if os.path.exists(repo_dir):
    subprocess.check_call(["rm", "-rf", repo_dir])

subprocess.check_call([
    "git", "clone", "https://github.com/akshaykumarhudedmani/goldenhour-rl.git", repo_dir
])
sys.path.insert(0, repo_dir)
os.chdir(repo_dir)

# 4. Run Pytest Suite on Linux
print("\n[Step 3/6] Running sandbox verification unit tests...")
subprocess.check_call([sys.executable, "-m", "pytest", "tests/", "-v"])

# 5. Generate Training & Evaluation Datasets
print("\n[Step 4/6] Synthesizing verifiable emergency crisis datasets...")
subprocess.check_call([
    sys.executable, "-m", "goldenhour_rl.data.crisis_generator",
    "--output", "data/crisis_train.jsonl", "--num_samples", "300", "--seed", "42"
])
subprocess.check_call([
    sys.executable, "-m", "goldenhour_rl.data.crisis_generator",
    "--output", "data/crisis_eval.jsonl", "--num_samples", "100", "--seed", "999"
])

# 6. Run Baseline Benchmark
print("\n[Step 5/6] Benchmarking baseline dispatch heuristics...")
subprocess.check_call([
    sys.executable, "-m", "goldenhour_rl.eval.benchmark",
    "--eval_data", "data/crisis_eval.jsonl", "--dry_run"
])

# 7. Run GRPO Post-Training Loop
max_steps = int(os.environ.get("MAX_STEPS", "50"))
print(f"\n[Step 6/6] Launching Unsloth 4-bit QLoRA + GRPO training loop ({max_steps} steps)...")
train_cmd = [
    sys.executable, "-m", "goldenhour_rl.training.train_grpo",
    "--model_name", "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "--data_path", "data/crisis_train.jsonl",
    "--output_dir", "/kaggle/working/outputs/goldenhour_grpo",
    "--max_steps", str(max_steps),
]
if os.environ.get("WANDB_API_KEY"):
    train_cmd.append("--use_wandb")

subprocess.check_call(train_cmd)

# 8. Upload Trained Adapter to Hugging Face Hub
output_lora = "/kaggle/working/outputs/goldenhour_grpo/final_lora"
if os.path.exists(output_lora):
    print("\n[Step 7/7] Uploading trained LoRA weights to Hugging Face Hub...")
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    repo_id = "akshaykumarhudedmani/goldenhour-rl"
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=output_lora,
        repo_id=repo_id,
        commit_message=f"Upload GoldenHour-RL GRPO LoRA weights ({max_steps} steps)"
    )
    print(f"[Done] Model successfully pushed to: https://huggingface.co/{repo_id}")

print("\n" + "=" * 70)
print("     GOLDENHOUR-RL POST-TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 70)
