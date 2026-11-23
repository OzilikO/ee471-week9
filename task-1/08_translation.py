from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# 8. Translation pipeline (French to English)
# Follows Hugging Face Course Chapter 1.3: Ce cours est produit par Hugging Face.
model_name = "Helsinki-NLP/opus-mt-fr-en"
text = "Ce cours est produit par Hugging Face."

print("=== Text Translation (FR -> EN) ===")
print(f"Source (FR):\n{text}\n")

try:
    translator = pipeline("translation", model=model_name)
    translation = translator(text)[0]["translation_text"]
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Translation (EN):\n{translation}")
