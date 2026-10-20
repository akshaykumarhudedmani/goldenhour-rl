import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()
token = os.environ.get("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN is not set in .env")

api = HfApi(token=token)
repo_id = "akshaykumarhudedmani/goldenhour-rl"

print(f"[Upload] Target repo: https://huggingface.co/{repo_id}")

# 1. Upload Final LoRA to Root
final_path = Path("outputs/goldenhour_grpo/final_lora")
if final_path.exists():
    print("\n[Upload 1/3] Uploading Final GRPO LoRA weights to root...")
    api.upload_folder(
        folder_path=str(final_path),
        repo_id=repo_id,
        commit_message="Stage 2: GoldenHour-RL GRPO LoRA weights (50 steps, G=8, zero-violation jackpot)"
    )
    print("  -> Root upload complete!")

# 2. Upload Checkpoint-25
ckpt25_path = Path("outputs/goldenhour_grpo/checkpoint-25")
if ckpt25_path.exists():
    print("\n[Upload 2/3] Uploading checkpoint-25 to checkpoints/checkpoint-25...")
    api.upload_folder(
        folder_path=str(ckpt25_path),
        repo_id=repo_id,
        path_in_repo="checkpoints/checkpoint-25",
        commit_message="Checkpoints: Stage 2 GRPO checkpoint-25"
    )
    print("  -> checkpoint-25 upload complete!")

# 3. Upload Checkpoint-50
ckpt50_path = Path("outputs/goldenhour_grpo/checkpoint-50")
if ckpt50_path.exists():
    print("\n[Upload 3/3] Uploading checkpoint-50 to checkpoints/checkpoint-50...")
    api.upload_folder(
        folder_path=str(ckpt50_path),
        repo_id=repo_id,
        path_in_repo="checkpoints/checkpoint-50",
        commit_message="Checkpoints: Stage 2 GRPO checkpoint-50"
    )
    print("  -> checkpoint-50 upload complete!")

print("\n[Success] All checkpoints and final weights mirrored on Hugging Face Hub!")
