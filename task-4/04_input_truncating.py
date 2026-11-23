"""
Task 4.4: Input Truncating
Hugging Face Course Chapter 2.5: Handling Multiple Sequences
"""

from transformers import AutoTokenizer

checkpoint = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

long_text = (
    "The 2008 Global Financial Crisis stands as the most severe economic collapse "
    "of the 21st century, often compared to the Great Depression of the 1930s. "
    "Triggered by the bursting of the United States housing bubble, its effects "
    "rippled across the globe."
)

max_allowed_length = 12

print("==================================================")
print("Task 4.4: Input Truncating")
print("==================================================")
tokens_before = tokenizer.tokenize(long_text)
print(f"Original Text ({len(tokens_before)} tokens):\n\"{long_text}\"\n")

# Truncate text strictly to max_length
batch = tokenizer(
    long_text,
    truncation=True,
    max_length=max_allowed_length,
    return_tensors="pt"
)

print(f"Truncated Input IDs (max_length = {max_allowed_length}):")
print(batch["input_ids"])
print(f"Tensor Shape: {batch['input_ids'].shape} (exactly {batch['input_ids'].shape[1]} tokens)\n")

print("Decoded Truncated Text:")
print(f"\"{tokenizer.decode(batch['input_ids'][0])}\"\n")

print("✓ Input truncating demonstrated successfully!")
