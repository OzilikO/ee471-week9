from transformers import pipeline

# 3. Text generation pipeline using SmolLM2-360M
generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")

prompt = "If I continue to complete all in-class exercises in this modern software practices course,"

results = generator(
    prompt,
    max_length=35,
    num_return_sequences=2
)

print("=== Text Generation ===")
print(f"Prompt: \"{prompt}\"\n")
for idx, res in enumerate(results, 1):
    print(f"Alternative {idx}:\n{res['generated_text']}\n")
