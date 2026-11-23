from transformers import pipeline

# 7. Text Summarization adapted to use SmolLM2-360M
# Model: HuggingFaceTB/SmolLM2-360M (Pre-downloaded Local LLM - Zero web download required)
print("=== Text Summarization ===")
print("Model: HuggingFaceTB/SmolLM2-360M (Pre-cached Local LLM - Zero Web Download)\n")

generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")

text = (
    "The 2008 Global Financial Crisis stands as the most severe economic collapse of the "
    "21st century, often compared to the Great Depression of the 1930s. Triggered by the bursting "
    "of the United States housing bubble, its effects rippled across the globe, leading to the collapse "
    "of major financial institutions and a deep international recession. The crisis began with the "
    "subprime mortgage market. In the early 2000s, low interest rates and a push for homeownership led "
    "banks to issue high-risk loans to borrowers with poor credit."
)

prompt = (
    f"Passage: {text}\n\n"
    "Summary in one sentence:"
)

result = generator(prompt, max_new_tokens=40, do_sample=False)
full_output = result[0]["generated_text"]
summary = full_output.split("Summary in one sentence:")[-1].strip().split("\n")[0]

print(f"Original Text:\n{text}\n")
print(f"Summary:\n{summary}")
