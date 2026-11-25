# EE471 Week 9 - Task 4: Models & Tokenizers Fundamentals

This folder implements Task 4 of the Week 9 In-Class Exercise, directly following the [Hugging Face Course Chapter 2: Using Transformers](https://huggingface.co/learn/llm-course/chapter2/1).

---

## File Structure

```
task-4/
├── 01_load_and_save.py         # Loading from Hub, saving locally, and reloading offline
├── 02_encoding_and_decoding.py  # Tokenizing text into IDs and decoding back to text
├── 03_input_padding.py          # Padding unequal length sequences with attention masks
├── 04_input_truncating.py       # Truncating sequences exceeding maximum token length
├── 05_special_tokens.py         # Adding [CLS] and [SEP] special tokens for transformer models
├── run_all_task4.py             # Runs all 5 scripts in sequence
└── README.md                    # Task documentation
```

---

## Running All Scripts

To run all 5 scripts in sequence:
```bash
python3 /mnt/d/EE/ee471week9/ee471-week9/task-4/run_all_task4.py
```
