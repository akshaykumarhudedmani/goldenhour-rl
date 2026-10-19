import os
import sys
import subprocess
import torch

# 1. Environment Secrets & Config
# Retrieve credentials securely from local config, Kaggle UserSecrets, or OS environment
try:
    import secrets_config
except Exception:
    pass

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
print(f"Hugging Face Token Configured: {bool(os.environ.get('HF_TOKEN'))}")
print(f"Weights & Biases Key Configured: {bool(os.environ.get('WANDB_API_KEY'))}")

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
print("\n[Step 4/7] Synthesizing verified Cold-Start SFT and Evaluation datasets...")
subprocess.check_call([
    sys.executable, "-m", "goldenhour_rl.data.sft_generator",
    "--output", "data/crisis_sft.jsonl", "--num_samples", "300", "--seed", "42"
])
subprocess.check_call([
    sys.executable, "-m", "goldenhour_rl.data.crisis_generator",
    "--output", "data/crisis_eval.jsonl", "--num_samples", "100", "--seed", "999"
])

# 6. Run Baseline Benchmark
print("\n[Step 5/7] Benchmarking baseline dispatch heuristics...")
subprocess.check_call([
    sys.executable, "-m", "goldenhour_rl.eval.benchmark",
    "--eval_data", "data/crisis_eval.jsonl", "--dry_run"
])

# 7. Stage 1: Cold-Start Supervised Fine-Tuning (SFT)
sft_output = "/kaggle/working/outputs/goldenhour_sft"
print("\n[Step 6/7] Stage 1: Cold-Start SFT (Anchoring syntax & dispatch logic)...")
sft_cmd = [
    sys.executable, "-m", "goldenhour_rl.training.train_sft",
    "--model_name", "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "--data_path", "data/crisis_sft.jsonl",
    "--output_dir", sft_output,
    "--epochs", "2",
]
if os.environ.get("WANDB_API_KEY"):
    sft_cmd.append("--use_wandb")
subprocess.check_call(sft_cmd)

# Upload Stage 1 SFT checkpoint immediately to guarantee preservation
sft_lora_path = f"{sft_output}/final_lora"
if os.path.exists(sft_lora_path) and os.environ.get("HF_TOKEN"):
    try:
        print("\n[Upload Stage 1] Preserving Cold-Start SFT LoRA checkpoint on Hugging Face Hub (path_in_repo='sft_checkpoint')...")
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        repo_id = "akshaykumarhudedmani/goldenhour-rl"
        api.create_repo(repo_id=repo_id, exist_ok=True)
        api.upload_folder(
            folder_path=sft_lora_path,
            repo_id=repo_id,
            path_in_repo="sft_checkpoint",
            commit_message="Upload Stage 1: Cold-Start SFT LoRA checkpoint"
        )
        print(f"[Done] Stage 1 SFT checkpoint pushed to: https://huggingface.co/{repo_id}/tree/main/sft_checkpoint")
    except Exception as e:
        print(f"[Upload Warning] SFT upload notice: {e}")

# 8. Stage 2: GRPO Reinforcement Learning Loop
grpo_output = "/kaggle/working/outputs/goldenhour_grpo"
max_steps = int(os.environ.get("MAX_STEPS", "50"))
save_steps = int(os.environ.get("SAVE_STEPS", "25"))

print(f"\n[Step 7/7] Stage 2: GRPO Policy Optimization ({max_steps} steps, G=8 rollouts, save_steps={save_steps})...")
train_cmd = [
    sys.executable, "-m", "goldenhour_rl.training.train_grpo",
    "--model_name", "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "--init_lora_path", sft_lora_path,
    "--data_path", "data/crisis_sft.jsonl",
    "--output_dir", grpo_output,
    "--max_steps", str(max_steps),
    "--save_steps", str(save_steps),
    "--num_generations", "8",
]
if os.environ.get("WANDB_API_KEY"):
    train_cmd.append("--use_wandb")

subprocess.check_call(train_cmd)

# 9. Upload Final GRPO Adapter to Hugging Face Hub Root
final_lora = f"{grpo_output}/final_lora"
if not os.path.exists(final_lora):
    final_lora = sft_lora_path

if os.path.exists(final_lora) and os.environ.get("HF_TOKEN"):
    print("\n[Upload Stage 2] Uploading GoldenHour-RL GRPO LoRA weights to Hugging Face Hub root...")
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    repo_id = "akshaykumarhudedmani/goldenhour-rl"
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=final_lora,
        repo_id=repo_id,
        commit_message=f"Upload Stage 2: GoldenHour-RL GRPO LoRA weights ({max_steps} steps, G=8)"
    )
    print(f"[Done] Stage 2 GRPO model successfully pushed to: https://huggingface.co/{repo_id}")

print("\n" + "=" * 70)
print("     GOLDENHOUR-RL TWO-STAGE POST-TRAINING PIPELINE COMPLETED!")
print("=" * 70)
