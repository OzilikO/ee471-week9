# EE471 Week 9 - Transformers & Hugging Face

This repository contains setup and verification scripts for the Week 9 In-Class Exercise on Hugging Face Transformers.

## Required Models

1. **`HuggingFaceTB/SmolLM2-360M`**
   - **Task**: Text Generation / Causal Language Modeling
   - **Parameters**: 360 Million
   - **Type**: Lightweight Large Language Model (LLM)

2. **`Helsinki-NLP/opus-mt-fr-en`**
   - **Task**: Machine Translation (French $\to$ English)
   - **Architecture**: MarianMT seq2seq transformer

3. **`google/vit-base-patch16-224`**
   - **Task**: Image Classification
   - **Architecture**: Vision Transformer (ViT) with $16\times16$ patches

4. **`openai/whisper-large-v3`**
   - **Task**: Automatic Speech Recognition (ASR) / Transcription
   - **Architecture**: Encoder-Decoder Speech Transformer (~1.5B parameters)

## Project Structure

- `download_models.py`: Downloads all 4 models and caches them in `~/.cache/huggingface/hub` with local symbolic links in `models/`.
- `test_models.py`: Runs a quick offline-capable inference test on all 4 models to verify weights and pipelines.
- `models/`: Directory containing symbolic links to the cached model snapshots.

## Usage

### 1. Download models:
```bash
python3 download_models.py
```

### 2. Verify all models:
```bash
python3 test_models.py
```

### 3. Usage Examples (Hugging Face Course Ch 1.3 & Transformers v5):

#### Text Generation (`SmolLM2-360M`):
```python
from transformers import pipeline

generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
print(generator("The future of artificial intelligence in engineering is"))
```

#### Translation (`opus-mt-fr-en`):
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

text = "Bonjour tout le monde! L'intelligence artificielle est formidable."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Output: Hello everyone! Artificial intelligence is great.
```

#### Vision Transformer (`vit-base-patch16-224`):
```python
from transformers import pipeline

classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
# Pass PIL image or URL/path to image:
# print(classifier("path/to/image.jpg"))
```

#### Speech Recognition (`whisper-large-v3`):
```python
from transformers import pipeline

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
# Pass path to audio file:
# print(transcriber("path/to/audio.mp3"))
```
