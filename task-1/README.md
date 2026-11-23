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
├── 08_translation.py                # Translating English financial text to French
├── 09_image_classification.py       # Vision Transformer (google/vit-base-patch16-224)
├── 10_speech_recognition.py         # ASR speech-to-text (openai/whisper-large-v3)
├── run_all_task1.py                 # Sequentially runs all 10 subtasks
└── sample_data/                     # Sample media files (cat.jpg, sample_speech.wav)
```

---

## 15-Second Screen Recording Guide (Deliverable 2)

Each video should be $\le 15$ seconds showing terminal execution and output. Here are quick 1-sentence explanations to speak while recording:

| Video File Name | Command | What to say in the $\le 15$s recording |
| :--- | :--- | :--- |
| `Task1_sentiment_analysis.avi` | `python3 01_sentiment_analysis.py` | *"Here we use the default sentiment-analysis pipeline to classify positive and negative sentences about software courses, returning positive and negative labels with confidence scores."* |
| `Task1_zero_shot_classification.avi` | `python3 02_zero_shot_classification.py` | *"Here we run zero-shot classification on Berkshire's cash reserves across education, finance, business, and technology without fine-tuning, correctly predicting finance and business."* |
| `Task1_text_generation.avi` | `python3 03_text_generation.py` | *"Here we use SmolLM2-360M to complete an incomplete sentence about completing all in-class exercises, returning two distinct alternative completions within 35 words."* |
| `Task1_fill_mask.avi` | `python3 04_fill_mask.py` | *"Here we use the fill-mask pipeline to predict the missing masked token in 'To understand generative AI, one must study [MASK] well', returning the top predictions."* |
| `Task1_ner.avi` | `python3 05_named_entity_recognition.py` | *"Here we apply Named Entity Recognition to extract the person 'Nate', organization 'Izmir Institute of Technology', and locations 'İzmir' and 'Türkiye'."* |
| `Task1_question_answering.avi` | `python3 06_question_answering.py` | *"Here we validate the extracted information by querying the question-answering pipeline for the subject's name, institution, and living location from context."* |
| `Task1_summarization.avi` | `python3 07_summarization.py` | *"Here we summarize a paragraph describing the 2008 Global Financial Crisis into a concise two-sentence summary using a pretrained summarizer."* |
| `Task1_translation.avi` | `python3 08_translation.py` | *"Here we translate the 2008 financial crisis opening statement from English to French using the MarianMT sequence-to-sequence model."* |
| `Task1_image_classification.avi` | `python3 09_image_classification.py` | *"Here we pass a sample image to Google's Vision Transformer vit-base-patch16-224, accurately classifying the cat image with high probability."* |
| `Task1_speech_recognition.avi` | `python3 10_speech_recognition.py` | *"Here we transcribe spoken audio using OpenAI's Whisper Large v3 speech recognition transformer, outputting accurate text."* |

---

## Running All Scripts

To test or showcase all 10 scripts in sequence:
```bash
python3 run_all_task1.py
```
