from transformers import pipeline

# 1. Sentiment Analysis pipeline
classifier = pipeline("sentiment-analysis")

sentences = [
    "I've been waiting for a modern software practices course my whole life.",
    "I hate software courses"
]

results = classifier(sentences)

print("=== Sentiment Analysis ===")
for sentence, result in zip(sentences, results):
    print(f"\nSentence: \"{sentence}\"")
    print(f"Sentiment: {result['label']} (Score: {result['score']:.4f})")
