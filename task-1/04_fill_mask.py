from transformers import pipeline

# 4. Fill-Mask / Infilling adapted to use SmolLM2-360M
# Model: HuggingFaceTB/SmolLM2-360M (Pre-downloaded Local LLM - Zero web download required)
print("=== Mask Filling / Infilling ===")
print("Model: HuggingFaceTB/SmolLM2-360M (Pre-cached Local LLM - Zero Web Download)\n")

generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")

# Original sentence from exercise: "To understand generative AI, one must study [...] well."
prompt = "To understand generative AI, one must study"

results = generator(
    prompt,
    max_new_tokens=15,
    num_return_sequences=2,
    do_sample=True,
    temperature=0.7
)

print(f"Prompt: \"{prompt} [...] well.\"\n")
for idx, res in enumerate(results, 1):
    # Ensure completion flows into the sentence
    completed = res['generated_text'].strip()
    print(f"Alternative {idx}: {completed}")
