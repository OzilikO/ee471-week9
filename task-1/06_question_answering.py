from transformers import pipeline

# 6. Question Answering pipeline from Hugging Face Course Chapter 1.3
question_answerer = pipeline("question-answering")

context = "I am Nate, a research assistant at Izmir Institute of Technology, and currently living and working in the beautiful city İzmir in Türkiye."

questions = [
    "What is the name of the subject?",
    "Which company or institution do they work for?",
    "Where do they live and work?"
]

print("=== Question Answering Validation ===")
print(f"Context: \"{context}\"\n")

for q in questions:
    result = question_answerer(question=q, context=context)
    print(f"Question : {q}")
    print(f"Answer   : {result['answer']} (Score: {result['score']:.4f})\n")
