import os
from pathlib import Path
import soundfile as sf
from transformers import pipeline

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
model_path = MODELS_DIR / "whisper-large-v3"
model_name = str(model_path) if model_path.exists() else "openai/whisper-large-v3"

audio_path = os.path.join(os.path.dirname(__file__), "sample_data", "sample_speech.wav")

# 10. Automatic Speech Recognition (ASR) pipeline
transcriber = pipeline("automatic-speech-recognition", model=model_name)

# Load audio via soundfile (robust across all Linux/WSL environments without requiring system ffmpeg)
audio_data, sample_rate = sf.read(audio_path)
result = transcriber({"raw": audio_data, "sampling_rate": sample_rate})

print("=== Automatic Speech Recognition (openai/whisper-large-v3) ===")
print(f"Input Audio: {audio_path}\n")
print(f"Transcription:\n\"{result['text'].strip()}\"")
