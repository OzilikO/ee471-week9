from pathlib import Path
from transformers import pipeline

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
model_path = MODELS_DIR / "fill-mask"
model_name = str(model_path) if model_path.exists() else "distilbert/distilroberta-base"

# 4. Mask filling pipeline
unmasker = pipeline("fill-mask", model=model_name)

# Automatically use the tokenizer's mask token (e.g. <mask> for RoBERTa, [MASK] for BERT)
mask_token = unmasker.tokenizer.mask_token
prompt = f"To understand generative AI, one must study {mask_token} well."

results = unmasker(prompt, top_k=2)

print("=== Mask Filling ===")
print(f"Prompt: \"{prompt}\"\n")
for idx, res in enumerate(results, 1):
    print(f"Option {idx}: '{res['token_str']}' (score: {res['score']:.4f})")
    print(f"Filled sentence: {res['sequence']}\n")
