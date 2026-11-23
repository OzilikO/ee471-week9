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
└── README.md                    # Documentation & 15-second video recording scripts
```

---

## 15-Second Screen Recording Guide (Deliverables)

Per the assignment, record a video ($\le 15$ seconds each) demonstrating each task:

| Video File Name | Command | What to say in the video ($\le 15$s) |
| :--- | :--- | :--- |
| `Task4_load_and_save.avi` | `python3 01_load_and_save.py` | *"Here we load a pretrained model using AutoModel, save its configuration and weights locally to disk, and reload it completely offline."* |
| `Task4_encoding_and_decoding.avi` | `python3 02_encoding_and_decoding.py` | *"Here we tokenize human text into subword tokens, convert them to numerical IDs, and decode the IDs back into the original sentence."* |
| `Task4_input_padding.avi` | `python3 03_input_padding.py` | *"Here we batch two sentences of unequal lengths, adding pad tokens and generating attention masks so the model ignores padding."* |
| `Task4_input_truncating.avi` | `python3 04_input_truncating.py` | *"Here we truncate a long passage down to a fixed maximum length of 12 tokens to fit within the model's sequence length limit."* |
| `Task4_special_tokens.avi` | `python3 05_special_tokens.py` | *"Here we compare token IDs with and without special tokens, showing how the tokenizer prepends CLS for classification and appends SEP for sentence boundary."* |

---

## Running All Scripts

To test or demonstrate all 5 scripts in sequence:
```bash
python3 /mnt/d/EE/ee471week9/ee471-week9/task-4/run_all_task4.py
```
