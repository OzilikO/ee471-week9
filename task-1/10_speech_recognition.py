import os
import soundfile as sf
from transformers import pipeline

audio_path = os.path.join(os.path.dirname(__file__), "sample_data", "sample_speech.wav")

# 10. Automatic Speech Recognition (ASR) pipeline with Whisper Large v3
# Follows Hugging Face Course Chapter 1.3
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")

# Load audio via soundfile (robust across all Linux/WSL environments without requiring system ffmpeg)
audio_data, sample_rate = sf.read(audio_path)
result = transcriber({"raw": audio_data, "sampling_rate": sample_rate})

print("=== Automatic Speech Recognition (openai/whisper-large-v3) ===")
print(f"Input Audio: {audio_path}\n")
print(f"Transcription:\n\"{result['text'].strip()}\"")
