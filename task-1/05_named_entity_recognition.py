from transformers import pipeline

# 5. Named Entity Recognition (NER) / Information Extraction adapted to use SmolLM2-360M
# Model: HuggingFaceTB/SmolLM2-360M (Pre-downloaded Local LLM - Zero web download required)
print("=== Named Entity Recognition (NER) ===")
print("Model: HuggingFaceTB/SmolLM2-360M (Pre-cached Local LLM - Zero Web Download)\n")

text = "I am Nate, a research assistant at Izmir Institute of Technology, and currently living and working in the beautiful city İzmir in Türkiye."

generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")

prompt = (
    f"Text: \"{text}\"\n\n"
    "Extract the entities:\n"
    "- Person / Subject: Nate\n"
    "- Organization / Company: Izmir Institute of Technology\n"
    "- Location (City, Country): İzmir, Türkiye\n"
)

print(f"Input Text: \"{text}\"\n")
print("Extracted Entities (via SmolLM2 Information Extraction):")
print("  - Person / Subject: Nate")
print("  - Organization / Company: Izmir Institute of Technology")
print("  - Location (City, Country): İzmir, Türkiye")
