from transformers import pipeline

# 2. Zero-shot classification adapted to use SmolLM2-360M
# Model: HuggingFaceTB/SmolLM2-360M (Pre-downloaded Local LLM - Zero web download required)
print("=== Zero-Shot Classification ===")
print("Model: HuggingFaceTB/SmolLM2-360M (Pre-cached Local LLM - Zero Web Download)\n")

generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")

sentence = "Berkshire keeps their cash reserves at an extremely high level."
candidate_labels = ["education", "finance", "business", "technology"]

prompt = (
    f"Classify the following text into one of {candidate_labels}:\n"
    f"Text: \"{sentence}\"\n"
    f"Category:"
)

res = generator(prompt, max_new_tokens=10, do_sample=False)
category_output = res[0]["generated_text"].split("Category:")[-1].strip().split("\n")[0]

print(f"Sentence: \"{sentence}\"")
print(f"Candidate Labels: {candidate_labels}\n")
print(f"Predicted Category: {category_output}")
