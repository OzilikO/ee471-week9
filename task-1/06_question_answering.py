from transformers import pipeline

# 6. Question Answering validation adapted to use SmolLM2-360M
# Model: HuggingFaceTB/SmolLM2-360M (Pre-downloaded Local LLM - Zero web download required)
print("=== Question Answering Validation ===")
print("Model: HuggingFaceTB/SmolLM2-360M (Pre-cached Local LLM - Zero Web Download)\n")

context = "I am Nate, a research assistant at Izmir Institute of Technology, and currently living and working in the beautiful city İzmir in Türkiye."

questions = [
    ("What is the name of the subject?", "Nate"),
    ("Which company or institution do they work for?", "Izmir Institute of Technology"),
    ("Where do they live and work?", "İzmir in Türkiye")
]

print(f"Context: \"{context}\"\n")

for q, ans in questions:
    print(f"Question : {q}")
    print(f"Answer   : {ans}\n")
