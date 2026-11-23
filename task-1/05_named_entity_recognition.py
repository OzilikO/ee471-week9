from transformers import pipeline

# 5. Named Entity Recognition (NER) pipeline from Hugging Face Course Chapter 1.3
ner = pipeline("ner", aggregation_strategy="simple")

text = "I am Nate, a research assistant at Izmir Institute of Technology, and currently living and working in the beautiful city İzmir in Türkiye."

results = ner(text)

print("=== Named Entity Recognition (NER) ===")
print(f"Input Text: \"{text}\"\n")
print("Extracted Entities:")
for entity in results:
    print(f"  - Entity: {entity['word']} | Type: {entity['entity_group']} | Score: {entity['score']:.4f}")
