"""
Task 4.2: Text Encoding and Decoding
Hugging Face Course Chapter 2.3: Tokenizers
"""

from transformers import AutoTokenizer

checkpoint = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

text = "Using a Transformer network is simple"

print("==================================================")
print("Task 4.2: Text Encoding and Decoding")
print("==================================================")
print(f"Original Text:\n\"{text}\"\n")

# Step 1: Tokenize (splitting into wordpieces)
tokens = tokenizer.tokenize(text)
print("1. Tokenize (Text -> Tokens):")
print(f"   {tokens}\n")

# Step 2: Convert tokens to numerical IDs
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print("2. Numerical IDs (Tokens -> IDs):")
print(f"   {input_ids}\n")

# Step 3: Decode (numerical IDs back to human-readable text)
decoded_text = tokenizer.decode(input_ids)
print("3. Decode (IDs -> Text):")
print(f"   \"{decoded_text}\"\n")

print("✓ Text encoding and decoding demonstrated successfully!")
