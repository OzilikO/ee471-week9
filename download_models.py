"""
Download and verify models for EE471 Week 9:
1. HuggingFaceTB/SmolLM2-360M (Text Generation / Small LLM)
2. Helsinki-NLP/opus-mt-fr-en (Translation French -> English)
3. google/vit-base-patch16-224 (Vision Transformer Image Classification)
4. openai/whisper-large-v3 (Speech-to-Text Transcription)
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS = [
    {
        "id": "HuggingFaceTB/SmolLM2-360M",
        "alias": "SmolLM2-360M",
        "desc": "Small Language Model (Text Generation)"
    },
    {
        "id": "Helsinki-NLP/opus-mt-fr-en",
        "alias": "opus-mt-fr-en",
        "desc": "Translation (French to English)"
    },
    {
        "id": "google/vit-base-patch16-224",
        "alias": "vit-base-patch16-224",
        "desc": "Vision Transformer (Image Classification)"
    },
    {
        "id": "openai/whisper-large-v3",
        "alias": "whisper-large-v3",
        "desc": "Automatic Speech Recognition (ASR)"
    }
]

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def download_and_link():
    print(f"=== Downloading {len(MODELS)} Hugging Face Models ===")
    for idx, model_info in enumerate(MODELS, 1):
        model_id = model_info["id"]
        alias = model_info["alias"]
        desc = model_info["desc"]
        
        print(f"\n[{idx}/{len(MODELS)}] Downloading: {model_id}")
        print(f"    Description: {desc}")
        
        # Download into default HuggingFace hub cache (~/.cache/huggingface/hub)
        snapshot_path = snapshot_download(
            repo_id=model_id,
            resume_download=True
        )
        print(f"    Cached at: {snapshot_path}")
        
        # Create a symbolic link in ./models/<alias> for easy local reference
        local_link = MODELS_DIR / alias
        if local_link.is_symlink() or local_link.exists():
            local_link.unlink()
        try:
            local_link.symlink_to(snapshot_path, target_is_directory=True)
            print(f"    Local link created: {local_link} -> {snapshot_path}")
        except Exception as e:
            print(f"    Note: Could not create symlink ({e}), cache is available at {snapshot_path}")

    print("\n=== All models downloaded and cached successfully! ===")

if __name__ == "__main__":
    download_and_link()
