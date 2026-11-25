"""
Downloads any missing models and populates /mnt/d/EE/ee471week9/ee471-week9/models
with local references for all 10 tasks in Task 1.
"""

import os
import sys
from pathlib import Path

# Enable high-speed Rust-based transfers if available
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Map each task to its model repository ID and local folder name
TASK1_MODELS = {
    "sentiment-analysis": {
        "repo_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "folder": "sentiment-analysis"
    },
    "zero-shot-classification": {
        "repo_id": "facebook/bart-large-mnli",
        "folder": "zero-shot-classification"
    },
    "text-generation": {
        "repo_id": "HuggingFaceTB/SmolLM2-360M",
        "folder": "SmolLM2-360M"
    },
    "fill-mask": {
        "repo_id": "distilbert/distilroberta-base",
        "folder": "fill-mask"
    },
    "ner": {
        "repo_id": "dbmdz/bert-large-cased-finetuned-conll03-english",
        "folder": "ner"
    },
    "question-answering": {
        "repo_id": "distilbert/distilbert-base-cased-distilled-squad",
        "folder": "question-answering"
    },
    "summarization": {
        "repo_id": "sshleifer/distilbart-cnn-12-6",
        "folder": "summarization"
    },
    "translation": {
        "repo_id": "Helsinki-NLP/opus-mt-fr-en",
        "folder": "opus-mt-fr-en"
    },
    "image-classification": {
        "repo_id": "google/vit-base-patch16-224",
        "folder": "vit-base-patch16-224"
    },
    "speech-recognition": {
        "repo_id": "openai/whisper-large-v3",
        "folder": "whisper-large-v3"
    }
}

def main():
    print("=" * 60)
    print("EE471 Week 9 - Downloading & Setting Up All Task 1 Models Locally")
    print("=" * 60)

    for task_name, info in TASK1_MODELS.items():
        repo_id = info["repo_id"]
        folder_name = info["folder"]
        target_dir = MODELS_DIR / folder_name

        print(f"\n[Task: {task_name}]")
        print(f"  Repo: {repo_id}")
        print(f"  Target: {target_dir}")

        # Download / resolve snapshot from HF Hub cache (ignoring non-PyTorch formats like Flax/TF)
        print("  Ensuring snapshot is downloaded...")
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"]
        )
        print(f"  Cached snapshot: {snapshot_path}")

        # Create/update local link in models/ directory
        if target_dir.is_symlink() or target_dir.exists():
            target_dir.unlink()
        try:
            target_dir.symlink_to(snapshot_path, target_is_directory=True)
            print(f"  ✓ Linked {target_dir.name} -> {snapshot_path}")
        except Exception as e:
            print(f"  Note on linking: {e}")

    print("\n" + "=" * 60)
    print("All 10 models are present and linked in models/!")
    print("=" * 60)

if __name__ == "__main__":
    main()
