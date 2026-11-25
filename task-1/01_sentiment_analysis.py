from pathlib import Path
from transformers import pipeline

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
model_path = MODELS_DIR / "sentiment-analysis"
model_name = str(model_path) if model_path.exists() else "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# 1. Sentiment Analysis pipeline
classifier = pipeline("sentiment-analysis", model=model_name)

sentences = [
    "I've been waiting for a modern software practices course my whole life.",
    "I hate software courses"
]

results = classifier(sentences)

print("=== Sentiment Analysis ===")
for sentence, result in zip(sentences, results):
    print(f"\nSentence: \"{sentence}\"")
    print(f"Sentiment: {result['label']} (Score: {result['score']:.4f})")
