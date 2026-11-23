from transformers import pipeline

# 2. Zero-shot classification pipeline from Hugging Face Course Chapter 1.3
classifier = pipeline("zero-shot-classification")

sentence = "Berkshire keeps their cash reserves at an extremely high level."
candidate_labels = ["education", "finance", "business", "technology"]

result = classifier(sentence, candidate_labels=candidate_labels)

print("=== Zero-Shot Classification ===")
print(f"Sentence: \"{result['sequence']}\"\n")
print("Candidate Labels & Predictions:")
for label, score in zip(result["labels"], result["scores"]):
    print(f"  {label:<12}: {score:.4f}")
