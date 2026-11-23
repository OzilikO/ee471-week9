import os
from transformers import pipeline

# 9. Vision Transformer Image Classification pipeline
image_path = os.path.join(os.path.dirname(__file__), "sample_data", "cat.jpg")

classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

results = classifier(image_path)

print("=== Vision Transformer (google/vit-base-patch16-224) ===")
print(f"Input Image: {image_path}\n")
print("Top-3 Predictions:")
for pred in results[:3]:
    print(f"  {pred['label']:<25}: {pred['score']:.4f}")
