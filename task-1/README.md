# EE471 Week 9 - Task 1: Hugging Face Pipelines

This folder implements Task 1 of the Week 9 In-Class Exercise, directly following the [Hugging Face Course Chapter 1.3: Transformers, what can they do?](https://huggingface.co/learn/llm-course/chapter1/3).

---

## Folder Structure

```
task-1/
├── 01_sentiment_analysis.py         # Sentiment analysis on two contrasting sentences
├── 02_zero_shot_classification.py   # Zero-shot classification with candidate labels
├── 03_text_generation.py            # Sentence completion with SmolLM2-360M
├── 04_fill_mask.py                  # Masked language modeling (fill-mask)
├── 05_named_entity_recognition.py   # NER extracting person, organization, and location
├── 06_question_answering.py         # QA validating extracted person, org, and city
├── 07_summarization.py              # Summarizing the 2008 Financial Crisis passage
├── 08_translation.py                # Translating French text to English (opus-mt-fr-en)
├── 09_image_classification.py       # Vision Transformer (google/vit-base-patch16-224)
├── 10_speech_recognition.py         # ASR speech-to-text (openai/whisper-large-v3)
├── run_all_task1.py                 # Sequentially runs all 10 subtasks
└── sample_data/                     # Sample media files (cat.jpg, sample_speech.wav)
```

---

## Running All Scripts

To run all 10 scripts in sequence:
```bash
python3 run_all_task1.py
```
