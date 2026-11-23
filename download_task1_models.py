import sys
from huggingface_hub import snapshot_download

models = [
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "facebook/bart-large-mnli",
    "distilbert/distilroberta-base",
    "dbmdz/bert-large-cased-finetuned-conll03-english",
    "distilbert/distilbert-base-cased-distilled-squad",
    "sshleifer/distilbart-cnn-12-6",
    "Helsinki-NLP/opus-mt-en-fr"
]

print(f"Pre-caching {len(models)} default models for Task 1...")
for idx, m in enumerate(models, 1):
    print(f"\n[{idx}/{len(models)}] Downloading {m}...")
    try:
        path = snapshot_download(repo_id=m)
        print(f"✓ Cached: {m}")
    except Exception as e:
        print(f"✗ Failed {m}: {e}")

print("\nAll Task 1 models pre-cached successfully!")
