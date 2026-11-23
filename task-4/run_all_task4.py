"""
Runner script to execute all Task 4 demonstrations sequentially.
"""

import subprocess
import sys
from pathlib import Path

TASK4_DIR = Path(__file__).resolve().parent

scripts = [
    "01_load_and_save.py",
    "02_encoding_and_decoding.py",
    "03_input_padding.py",
    "04_input_truncating.py",
    "05_special_tokens.py"
]

print("==================================================")
print("EE471 Week 9 - Task 4: Complete Pipeline Runner")
print("==================================================")

for idx, script_name in enumerate(scripts, 1):
    script_path = TASK4_DIR / script_name
    print(f"\n[{idx}/5] Running {script_name}...")
    print("-" * 50)
    result = subprocess.run([sys.executable, str(script_path)], capture_output=False)
    if result.returncode != 0:
        print(f"Warning: {script_name} exited with code {result.returncode}")

print("\n" + "=" * 50)
print("All Task 4 subtasks finished successfully!")
print("=" * 50)
