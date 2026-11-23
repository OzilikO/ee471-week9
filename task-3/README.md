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
└── README.md                # Task documentation and 5-minute video guide
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
  6. `text_translation(text)`: Translates text from English to French using `opus-mt-en-fr`.
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

---

## 5-Minute Video Recording Guide (`Task3.avi`)

For Deliverable 2 (screen recording video up to 5 minutes):

1. **Introduction (~30 sec)**:
   - State your name and introduce the assignment: *"In this video, I will showcase Task 3 for EE471 Week 9, where we implemented an Object-Oriented module for Hugging Face Transformers and built an interactive Gradio web front-end."*
   - Show `pipeline_toolkit.py` in your code editor and highlight the `TransformersToolkit` class with lazy loading.

2. **Gradio UI Overview (~30 sec)**:
   - Run `python3 app.py` and open `http://localhost:7860`.
   - Show the 10 tabs at the top corresponding to each task.

3. **Walkthrough of the 10 Tasks (~3 min)**:
   - **Tab 1 (Sentiment Analysis)**: Click *Analyze Sentiment*, show positive/negative labels and confidence score.
   - **Tab 2 (Question Answering)**: Show the context about Nate at IZTECH, ask *"Which company or institution do they work for?"*, click *Find Answer*, show "Izmir Institute of Technology".
   - **Tab 3 (Zero-shot Classification)**: Classify the Berkshire statement across categories, show finance/business ranked highest.
   - **Tab 4 (Summarization)**: Summarize the 2008 Financial Crisis paragraph in two sentences.
   - **Tab 5 (Text Generation)**: Generate 2 completions for the sentence using SmolLM2-360M.
   - **Tab 6 (Translation)**: Translate the financial crisis statement from English to French.
   - **Tab 7 (Mask Filling)**: Predict the masked word in "To understand generative AI, one must study [...] well."
   - **Tab 8 (Image Classification)**: Classify `cat.jpg` with Google's ViT, showing Egyptian cat at ~90%.
   - **Tab 9 (NER)**: Extract entities: Nate (Person), Izmir Institute of Technology (Organization), İzmir and Türkiye (Location).
   - **Tab 10 (Speech Recognition)**: Transcribe `sample_speech.wav` with Whisper Large v3 into text.

4. **Conclusion (~30 sec)**:
   - Summarize that the OOP toolkit cleanly decouples the model inference logic from the Gradio presentation layer.
