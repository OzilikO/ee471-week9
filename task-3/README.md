# EE471 Week 9 - Task 3: OOP Hugging Face Module & Gradio Dashboard

This directory implements Task 3 of the Week 9 In-Class Exercise:
1. An **Object-Oriented Python Module** (`TransformersToolkit` in `pipeline_toolkit.py`) wrapping all 10 Hugging Face tasks.
2. An interactive **Gradio Front-End** (`app.py`) providing a multi-tab web dashboard.

---

## File Structure

```
task-3/
├── pipeline_toolkit.py      # The OOP class encapsulating all 10 Transformer tasks
├── app.py                   # Interactive Gradio dashboard with 10 dedicated tabs
├── sample_data/             # Sample images (cat.jpg) and audio (sample_speech.wav)
└── README.md                # Task documentation
```

---

## Architecture of `TransformersToolkit` (OOP Module)

The module follows clean Python OOP practices and lazy initialization:
- **`__init__(self, device='cpu')`**: Sets up device allocation and on-demand pipeline caching.
- **Methods**:
  1. `sentiment_analysis(text)`: Returns dictionary of class labels and confidence scores.
  2. `question_answering(question, context)`: Returns extracted answer span and score.
  3. `zero_shot_classification(text, candidate_labels)`: Returns probabilities across arbitrary tags.
  4. `text_summarization(text, max_length, min_length)`: Generates abstractive summary.
  5. `text_generation(prompt, max_length, num_return_sequences)`: Generates sentence continuations using `SmolLM2-360M`.
  6. `text_translation(text)`: Translates text from French to English using `opus-mt-fr-en`.
  7. `mask_filling(text_with_mask, top_k)`: Predicts masked tokens.
  8. `image_classification(image)`: Classifies images with `google/vit-base-patch16-224`.
  9. `named_entity_recognition(text)`: Extracts entities (PER, ORG, LOC).
  10. `speech_recognition(audio)`: Transcribes audio using `openai/whisper-large-v3`.

---

## How to Run the Gradio App

Launch the application:
```bash
python3 /mnt/d/EE/ee471week9/ee471-week9/task-3/app.py
```
Open your browser at `http://localhost:7860` (or `http://127.0.0.1:7860`).
