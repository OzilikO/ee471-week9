from transformers import pipeline

# 4. Mask filling pipeline
unmasker = pipeline("fill-mask")

# Automatically use the tokenizer's mask token (e.g. <mask> for RoBERTa, [MASK] for BERT)
mask_token = unmasker.tokenizer.mask_token
prompt = f"To understand generative AI, one must study {mask_token} well."

results = unmasker(prompt, top_k=2)

print("=== Mask Filling ===")
print(f"Prompt: \"{prompt}\"\n")
for idx, res in enumerate(results, 1):
    print(f"Option {idx}: '{res['token_str']}' (score: {res['score']:.4f})")
    print(f"Filled sentence: {res['sequence']}\n")
