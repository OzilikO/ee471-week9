from pathlib import Path
from transformers import pipeline

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
model_path = MODELS_DIR / "zero-shot-classification"
model_name = str(model_path) if model_path.exists() else "facebook/bart-large-mnli"

# 2. Zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model=model_name)

sentence = "Berkshire keeps their cash reserves at an extremely high level."
candidate_labels = ["education", "finance", "business", "technology"]

result = classifier(sentence, candidate_labels=candidate_labels)

print("=== Zero-Shot Classification ===")
print(f"Sentence: \"{result['sequence']}\"\n")
print("Candidate Labels & Predictions:")
for label, score in zip(result["labels"], result["scores"]):
    print(f"  {label:<12}: {score:.4f}")
