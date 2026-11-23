"""
Download script for FLUX.1-dev (Black Forest Labs)
Repository: https://huggingface.co/black-forest-labs/FLUX.1-dev

Note: FLUX.1-dev is a gated repository.
Before running this script:
1. Visit https://huggingface.co/black-forest-labs/FLUX.1-dev and click "Agree and access repository".
2. Log in using `huggingface-cli login` or set `export HF_TOKEN="your_hf_token"`.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, get_token

MODEL_ID = "black-forest-labs/FLUX.1-dev"
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def download_flux():
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        print("="*60)
        print("ERROR: Hugging Face Token Required!")
        print("="*60)
        print(f"'{MODEL_ID}' is a gated model.")
        print("1. Go to https://huggingface.co/black-forest-labs/FLUX.1-dev and accept the license.")
        print("2. Obtain your access token from https://huggingface.co/settings/tokens")
        print("3. Either run: huggingface-cli login")
        print("   OR run: export HF_TOKEN=\"your_hf_token\"")
        print("="*60)
        sys.exit(1)

    print(f"Token detected. Starting download for {MODEL_ID} (~24 GB)...")
    snapshot_path = snapshot_download(
        repo_id=MODEL_ID,
        token=token,
        resume_download=True
    )
    print(f"Downloaded and cached at: {snapshot_path}")

    # Create local symlink in models/FLUX.1-dev
    local_link = MODELS_DIR / "FLUX.1-dev"
    if local_link.is_symlink() or local_link.exists():
        local_link.unlink()
    try:
        local_link.symlink_to(snapshot_path, target_is_directory=True)
        print(f"Local link created: {local_link} -> {snapshot_path}")
    except Exception as e:
        print(f"Note: Symlink not created ({e}), model is accessible via cache.")

if __name__ == "__main__":
    download_flux()
