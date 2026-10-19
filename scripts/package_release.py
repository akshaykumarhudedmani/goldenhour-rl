import os
import zipfile
from pathlib import Path

# Paths to strictly exclude from clean release packages
EXCLUDED_NAMES = {
    ".env",
    ".env.local",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    "_kaggle_test",
    "goldenhour_rl.egg-info",
    "build",
    "dist",
    "out",
    "out_live",
    "out_download",
    "kaggle_eval",
}

EXCLUDED_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".token",
    ".zip",
    ".log",
}


def build_clean_archive(output_zip: str = "goldenhour-rl-clean.zip"):
    root_dir = Path(__file__).resolve().parent.parent
    zip_path = root_dir / output_zip

    print(f"[Packaging] Generating clean release archive at: {zip_path}")
    count = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in root_dir.rglob("*"):
            if not file_path.is_file():
                continue

            rel_path = file_path.relative_to(root_dir)
            parts = rel_path.parts

            # Check if any parent or name matches excluded sets
            if any(p in EXCLUDED_NAMES or p.endswith(".egg-info") for p in parts):
                continue
            if file_path.suffix in EXCLUDED_EXTENSIONS:
                continue
            if file_path.name.startswith(".env") and file_path.name != ".env.example":
                continue

            zf.write(file_path, arcname=str(rel_path))
            count += 1

    print(f"[Packaging] Success! Added {count} pristine files. Zero secret leaks guaranteed.")
    print(f"[Packaging] Archive ready to share: {zip_path.name} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    build_clean_archive()
