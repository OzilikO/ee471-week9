from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# 8. Translation pipeline (English to French)

model_name = "Helsinki-NLP/opus-mt-en-fr"
text = "Ce cours est produit par Hugging Face."

print("=== Text Translation (EN -> FR) ===")
print(f"Source (EN):\n{text}\n")

try:
    translator = pipeline("translation", model=model_name)
    translation = translator(text)[0]["translation_text"]
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Translation (FR):\n{translation}")
