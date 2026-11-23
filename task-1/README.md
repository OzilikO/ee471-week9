# EE471 Week 9 - Task 1: Hugging Face Pipelines

This folder implements Task 1 of the Week 9 In-Class Exercise, directly following the [Hugging Face Course Chapter 1.3: Transformers, what can they do?](https://huggingface.co/learn/llm-course/chapter1/3).

All scripts run **100% offline** with **zero web downloads**, using the models pre-downloaded for Week 9 (`SmolLM2-360M`, `opus-mt-fr-en`, `distilbert-sst2`, `vit-base-patch16-224`, `whisper-large-v3`).

---

## Folder Structure & Model Mapping

```
task-1/
├── 01_sentiment_analysis.py         # Sentiment analysis using distilbert-base-uncased-finetuned-sst-2
├── 02_zero_shot_classification.py   # Zero-shot classification using SmolLM2-360M
├── 03_text_generation.py            # Sentence completion using SmolLM2-360M
├── 04_fill_mask.py                  # Mask filling / infilling using SmolLM2-360M
├── 05_named_entity_recognition.py   # Information extraction / NER using SmolLM2-360M
├── 06_question_answering.py         # QA entity validation using SmolLM2-360M
├── 07_summarization.py              # Text summarization using SmolLM2-360M
├── 08_translation.py                # French-to-English translation using opus-mt-fr-en
├── 09_image_classification.py       # Vision Transformer using google/vit-base-patch16-224
├── 10_speech_recognition.py         # Speech-to-text using openai/whisper-large-v3
├── run_all_task1.py                 # Sequentially runs all 10 subtasks in under 30s
└── sample_data/                     # Sample media files (cat.jpg, sample_speech.wav)
```

---

## 15-Second Screen Recording Guide (Deliverable 2)

Each video should be $\le 15$ seconds showing terminal execution and output. Here are concise 1-sentence explanations to speak while recording:

| Video File Name | Command | What to say in the $\le 15$s recording |
| :--- | :--- | :--- |
| `Task1_sentiment_analysis.avi` | `python3 01_sentiment_analysis.py` | *"Here we run sentiment analysis using DistilBERT on positive and negative sentences about software courses, returning sentiment labels with confidence scores."* |
| `Task1_zero_shot_classification.avi` | `python3 02_zero_shot_classification.py` | *"Here we run zero-shot classification using our local SmolLM2 model on Berkshire's cash reserves, accurately classifying it under the finance category."* |
| `Task1_text_generation.avi` | `python3 03_text_generation.py` | *"Here we use SmolLM2-360M to complete an incomplete sentence about completing all in-class exercises, returning two distinct alternative completions within 35 words."* |
| `Task1_fill_mask.avi` | `python3 04_fill_mask.py` | *"Here we perform masked sentence infilling with SmolLM2, generating realistic completions for 'To understand generative AI, one must study [concept] well'."* |
| `Task1_ner.avi` | `python3 05_named_entity_recognition.py` | *"Here we extract named entities using SmolLM2, identifying the person Nate, the organization Izmir Institute of Technology, and the location İzmir in Türkiye."* |
| `Task1_question_answering.avi` | `python3 06_question_answering.py` | *"Here we validate the extracted information by answering context-based questions about the subject's identity, employer, and living location."* |
| `Task1_summarization.avi` | `python3 07_summarization.py` | *"Here we summarize the 2008 Global Financial Crisis passage into a concise one-sentence takeaway using our local SmolLM2 model."* |
| `Task1_translation.avi` | `python3 08_translation.py` | *"Here we translate the 2008 Financial Crisis statement from French to English using our pre-cached Helsinki-NLP opus-mt-fr-en model."* |
| `Task1_image_classification.avi` | `python3 09_image_classification.py` | *"Here we pass a sample image to Google's Vision Transformer vit-base-patch16-224, accurately classifying the cat image with high confidence."* |
| `Task1_speech_recognition.avi` | `python3 10_speech_recognition.py` | *"Here we transcribe spoken audio using OpenAI's Whisper Large v3 speech recognition model, outputting clear, accurate text."* |

---

## Running All Scripts

To run all 10 scripts in sequence:
```bash
python3 run_all_task1.py
```
