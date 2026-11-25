

from transformers import AutoTokenizer

checkpoint = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a modern software practices course my whole life."

print("==================================================")
print("Task 4.5: Adding Special Tokens")
print("==================================================")
print(f"Original Sequence:\n\"{sequence}\"\n")
print(f"Model Special Tokens: CLS = '{tokenizer.cls_token}' (ID: {tokenizer.cls_token_id}), SEP = '{tokenizer.sep_token}' (ID: {tokenizer.sep_token_id})\n")

# 1. Token IDs without special tokens (direct tokenization + convert_tokens_to_ids)
tokens = tokenizer.tokenize(sequence)
ids_without_special = tokenizer.convert_tokens_to_ids(tokens)

print("1. Without Special Tokens:")
print(f"   IDs    : {ids_without_special}")
print(f"   Decoded: \"{tokenizer.decode(ids_without_special)}\"\n")

# 2. Token IDs with special tokens automatically added by tokenizer(...)
model_inputs = tokenizer(sequence)
ids_with_special = model_inputs["input_ids"]

print("2. With Special Tokens Added ([CLS] at start, [SEP] at end):")
print(f"   IDs    : {ids_with_special}")
print(f"   Decoded: \"{tokenizer.decode(ids_with_special)}\"\n")

print("✓ Adding special tokens demonstrated successfully!")
