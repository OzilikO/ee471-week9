"""
Runner script to execute all Task 1 demonstrations sequentially.
"""

import subprocess
import sys
from pathlib import Path

TASK1_DIR = Path(__file__).resolve().parent

scripts = [
    "01_sentiment_analysis.py",
    "02_zero_shot_classification.py",
    "03_text_generation.py",
    "04_fill_mask.py",
    "05_named_entity_recognition.py",
    "06_question_answering.py",
    "07_summarization.py",
    "08_translation.py",
    "09_image_classification.py",
    "10_speech_recognition.py"
]

print("==================================================")
print("EE471 Week 9 - Task 1: Complete Pipeline Runner")
print("==================================================")

for idx, script_name in enumerate(scripts, 1):
    script_path = TASK1_DIR / script_name
    print(f"\n[{idx}/10] Running {script_name}...")
    print("-" * 50)
    result = subprocess.run([sys.executable, str(script_path)], capture_output=False)
    if result.returncode != 0:
        print(f"Warning: {script_name} exited with code {result.returncode}")

print("\n" + "=" * 50)
print("All Task 1 subtasks finished!")
print("=" * 50)
