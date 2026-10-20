import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

hf_token = os.environ.get("HF_TOKEN")
wandb_key = os.environ.get("WANDB_API_KEY")
kaggle_token = os.environ.get("KAGGLE_API_TOKEN")

if not hf_token or not wandb_key:
    raise ValueError("HF_TOKEN and WANDB_API_KEY must be set in .env")

env = dict(os.environ)
if kaggle_token and "KAGGLE_KEY" not in env:
    env["KAGGLE_KEY"] = kaggle_token

script_path = Path("kaggle_kernel/train_kaggle.py")
original_code = script_path.read_text(encoding="utf-8")

# Inject credentials into train_kaggle.py for the Kaggle push
injection = f"""# --- INJECTED CREDENTIALS FOR KAGGLE CONTAINER ---
import os
os.environ["HF_TOKEN"] = {repr(hf_token)}
os.environ["WANDB_API_KEY"] = {repr(wandb_key)}
os.environ["MAX_STEPS"] = "50"
os.environ["SAVE_STEPS"] = "25"
# -------------------------------------------------
"""

modified_code = injection + "\n" + original_code

try:
    script_path.write_text(modified_code, encoding="utf-8")
    print("[Deployer] Injected credentials into train_kaggle.py temporarily.")
    
    print("[Deployer] Pushing kernel version to Kaggle...")
    res = subprocess.run(
        ["kaggle", "kernels", "push", "-p", "kaggle_kernel/"],
        capture_output=True,
        text=True,
        env=env
    )
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"Kaggle push failed with code {res.returncode}")
finally:
    # Always restore original file from git to keep repository perfectly clean
    subprocess.run(["git", "checkout", "--", str(script_path)], check=True)
    print("[Deployer] Restored clean train_kaggle.py from git. Zero secrets exposed locally.")
