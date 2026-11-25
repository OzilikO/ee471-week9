from pathlib import Path
from transformers import pipeline

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
model_path = MODELS_DIR / "ner"
model_name = str(model_path) if model_path.exists() else "dbmdz/bert-large-cased-finetuned-conll03-english"

# 5. Named Entity Recognition (NER) pipeline
ner = pipeline("ner", model=model_name, aggregation_strategy="simple")

text = "I am Nate, a research assistant at Izmir Institute of Technology, and currently living and working in the beautiful city İzmir in Türkiye."

results = ner(text)

print("=== Named Entity Recognition (NER) ===")
print(f"Input Text: \"{text}\"\n")
print("Extracted Entities:")
for entity in results:
    print(f"  - Entity: {entity['word']} | Type: {entity['entity_group']} | Score: {entity['score']:.4f}")
