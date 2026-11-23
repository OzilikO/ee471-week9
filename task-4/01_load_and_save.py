"""
Task 4.1: Load & Save a model
Hugging Face Course Chapter 2.2: Models
"""

import os
import shutil
from pathlib import Path
from transformers import AutoModel

checkpoint = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
save_dir = Path(__file__).resolve().parent / "saved_model"

print("==================================================")
print("Task 4.1: Load & Save a Pretrained Model")
print("==================================================")

# 1. Load pretrained model from Hugging Face Hub
print(f"\n1. Loading pretrained model: '{checkpoint}'...")
model = AutoModel.from_pretrained(checkpoint)
print("   Model loaded successfully!")

# 2. Save model weights and configuration to disk
print(f"\n2. Saving model to local directory: '{save_dir.name}'...")
model.save_pretrained(str(save_dir))
print("   Saved files:")
for f in save_dir.iterdir():
    print(f"     - {f.name} ({f.stat().st_size:,} bytes)")

# 3. Reload model from local directory (completely offline)
print(f"\n3. Reloading model from local directory '{save_dir.name}'...")
reloaded_model = AutoModel.from_pretrained(str(save_dir))
print("   Reloaded model parameters:", sum(p.numel() for p in reloaded_model.parameters()))

print("\n✓ Model successfully loaded, saved, and reloaded!")
