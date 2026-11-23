"""
Task 4.3: Input Padding
Hugging Face Course Chapter 2.5: Handling Multiple Sequences
"""

from transformers import AutoTokenizer

checkpoint = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sentences = [
    "I've been waiting for a modern software practices course my whole life.",
    "I hate software courses."
]

print("==================================================")
print("Task 4.3: Input Padding for Batching")
print("==================================================")
print("Input Sentences:")
for idx, s in enumerate(sentences, 1):
    print(f"  [{idx}] \"{s}\" (length: {len(tokenizer.tokenize(s))} tokens)")

# Pad sequences to the longest sequence in the batch
batch = tokenizer(sentences, padding=True, return_tensors="pt")

print(f"\nPad Token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})\n")

print("Padded Input IDs Tensor:")
print(batch["input_ids"])

print("\nAttention Mask Tensor (1 = Real Token, 0 = Padded Token):")
print(batch["attention_mask"])

print("\n✓ Input padding demonstrated successfully!")
