import os
from pathlib import Path
from transformers import pipeline

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
model_path = MODELS_DIR / "vit-base-patch16-224"
model_name = str(model_path) if model_path.exists() else "google/vit-base-patch16-224"

# 9. Vision Transformer Image Classification pipeline
image_path = os.path.join(os.path.dirname(__file__), "sample_data", "cat.jpg")

classifier = pipeline("image-classification", model=model_name)

results = classifier(image_path)

print("=== Vision Transformer (google/vit-base-patch16-224) ===")
print(f"Input Image: {image_path}\n")
print("Top-3 Predictions:")
for pred in results[:3]:
    print(f"  {pred['label']:<25}: {pred['score']:.4f}")
