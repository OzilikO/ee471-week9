import os
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
model_path = MODELS_DIR / "summarization"
model_name = str(model_path) if model_path.exists() else "sshleifer/distilbart-cnn-12-6"

# 7. Summarization pipeline
text = (
    "The 2008 Global Financial Crisis stands as the most severe economic collapse of the "
    "21st century, often compared to the Great Depression of the 1930s. Triggered by the bursting "
    "of the United States housing bubble, its effects rippled across the globe, leading to the collapse "
    "of major financial institutions and a deep international recession. The crisis began with the "
    "subprime mortgage market. In the early 2000s, low interest rates and a push for homeownership led "
    "banks to issue high-risk loans to borrowers with poor credit."
)

print("=== Text Summarization ===")
print(f"Original Text:\n{text}\n")

try:
    summarizer = pipeline("summarization", model=model_name)
    result = summarizer(text, max_length=50, min_length=20, do_sample=False)
    summary = result[0]["summary_text"]
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=50, min_length=20, do_sample=False)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(f"Summary:\n{summary}")
