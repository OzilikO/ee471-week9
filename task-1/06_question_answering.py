import torch
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
model_path = MODELS_DIR / "question-answering"
model_name = str(model_path) if model_path.exists() else "distilbert/distilbert-base-cased-distilled-squad"

context = "I am Nate, a research assistant at Izmir Institute of Technology, and currently living and working in the beautiful city İzmir in Türkiye."

questions = [
    "What is the name of the subject?",
    "Which company or institution do they work for?",
    "Where do they live and work?"
]

print("=== Question Answering Validation ===")
print(f"Context: \"{context}\"\n")

try:
    question_answerer = pipeline("question-answering", model=model_name)
    for q in questions:
        result = question_answerer(question=q, context=context)
        print(f"Question : {q}")
        print(f"Answer   : {result['answer']} (Score: {result['score']:.4f})\n")
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    for q in questions:
        inputs = tokenizer(q, context, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits)
        answer_tokens = inputs.input_ids[0, start_idx : end_idx + 1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        start_prob = torch.softmax(outputs.start_logits, dim=-1)[0, start_idx].item()
        end_prob = torch.softmax(outputs.end_logits, dim=-1)[0, end_idx].item()
        score = start_prob * end_prob
        print(f"Question : {q}")
        print(f"Answer   : {answer} (Score: {score:.4f})\n")
