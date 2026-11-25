from pathlib import Path
from transformers import pipeline

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
model_path = MODELS_DIR / "SmolLM2-360M"
model_name = str(model_path) if model_path.exists() else "HuggingFaceTB/SmolLM2-360M"

# 3. Text generation pipeline using SmolLM2-360M
generator = pipeline("text-generation", model=model_name)

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
