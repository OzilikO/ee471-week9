from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 8. Translation pipeline using Helsinki-NLP/opus-mt-fr-en (Pre-downloaded Week 9 model)
# Translates the 2008 Financial Crisis statement from French to English
model_name = "Helsinki-NLP/opus-mt-fr-en"

# French sentence of the 2008 Financial Crisis:
french_text = (
    "La crise financière mondiale de 2008 constitue le plus grave effondrement "
    "économique du XXIe siècle, souvent comparé à la Grande Dépression."
)

print("=== Text Translation (French -> English) ===")
print(f"Model: {model_name} (Loaded locally from cache)")
print(f"Source (FR):\n{french_text}\n")

# Load tokenizer and model from local cache (runs 100% offline)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

inputs = tokenizer(french_text, return_tensors="pt")
outputs = model.generate(**inputs)
english_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Translation (EN):\n{english_translation}")
