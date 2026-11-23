from transformers import AutoTokenizer

checkpoint = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

long_text = (
    "The 2008 Global Financial Crisis stands as the most severe economic collapse "
    "of the 21st century, often compared to the Great Depression of the 1930s. "
    "Triggered by the bursting of the United States housing bubble, its effects "
    "rippled across the globe."
)



print("==================================================")
print("Task 4.4: Input Truncating")
print("==================================================")
tokens_before = tokenizer.tokenize(long_text)
print(f"Original Text ({len(tokens_before)} tokens):\n\"{long_text}\"\n")

# Truncate text strictly to max_length
batch = tokenizer(
    long_text,
    truncation=True,
    max_length=5,
    return_tensors="pt"
)

print(f"Truncated Input IDs (max_length = 5):")
print(batch["input_ids"])
print(f"Tensor Shape: {batch['input_ids'].shape} (exactly {batch['input_ids'].shape[1]} tokens)\n")

print("Attention Mask Tensor:")
print(batch["attention_mask"])

print("\nToken Type IDs Tensor:")
print(batch["token_type_ids"])

print("\nDecoded Truncated Text:")
print(f"\"{tokenizer.decode(batch['input_ids'][0])}\"\n")

print("✓ Input truncating demonstrated successfully!")
