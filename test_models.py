"""
Verification & Demo script for Week 9 Hugging Face models:
1. SmolLM2-360M (Text Generation)
2. opus-mt-fr-en (French to English Translation)
3. vit-base-patch16-224 (Image Classification)
4. whisper-large-v3 (Automatic Speech Recognition)
"""

import sys
import torch
from transformers import (
    pipeline,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from PIL import Image

def test_smollm():
    print("\n" + "="*50)
    print("Testing 1/4: HuggingFaceTB/SmolLM2-360M (Text Generation)")
    print("="*50)
    try:
        generator = pipeline(
            "text-generation",
            model="HuggingFaceTB/SmolLM2-360M",
            dtype=torch.float32,
            device="cpu"
        )
        prompt = "Artificial Intelligence in engineering enables"
        output = generator(prompt, max_length=40, num_return_sequences=1)
        print(f"Prompt: {prompt}")
        print(f"Generated: {output[0]['generated_text']}")
        print("✓ SmolLM2-360M loaded and generated successfully!")
        return True
    except Exception as e:
        print(f"✗ Failed SmolLM2-360M: {e}")
        return False

def test_translation():
    print("\n" + "="*50)
    print("Testing 2/4: Helsinki-NLP/opus-mt-fr-en (Translation)")
    print("="*50)
    try:
        # Note: In transformers v5+, seq2seq translation uses AutoModelForSeq2SeqLM
        model_name = "Helsinki-NLP/opus-mt-fr-en"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        french_text = "Bonjour tout le monde. L'apprentissage automatique est très puissant."
        inputs = tokenizer(french_text, return_tensors="pt")
        outputs = model.generate(**inputs)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Source (FR): {french_text}")
        print(f"Translation (EN): {translated_text}")
        print("✓ opus-mt-fr-en loaded and translated successfully!")
        return True
    except Exception as e:
        print(f"✗ Failed opus-mt-fr-en: {e}")
        return False

def test_vit():
    print("\n" + "="*50)
    print("Testing 3/4: google/vit-base-patch16-224 (Vision Transformer)")
    print("="*50)
    try:
        classifier = pipeline(
            "image-classification",
            model="google/vit-base-patch16-224",
            device="cpu"
        )
        # Create a simple test image (224x224 RGB image with blue sky / green field)
        img = Image.new("RGB", (224, 224), color=(73, 109, 137))
        preds = classifier(img)
        print("Predictions on sample image:")
        for p in preds[:3]:
            print(f"  - {p['label']}: {p['score']:.4f}")
        print("✓ vit-base-patch16-224 loaded and classified successfully!")
        return True
    except Exception as e:
        print(f"✗ Failed vit-base-patch16-224: {e}")
        return False

def test_whisper():
    print("\n" + "="*50)
    print("Testing 4/4: openai/whisper-large-v3 (Speech Recognition)")
    print("="*50)
    try:
        # Load processor and model weights to verify offline availability
        processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-large-v3",
            dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("✓ openai/whisper-large-v3 processor and model loaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Failed whisper-large-v3: {e}")
        return False

def main():
    print("Running Verification for EE471 Week 9 Models...")
    results = {}
    results["SmolLM2-360M"] = test_smollm()
    results["opus-mt-fr-en"] = test_translation()
    results["vit-base-patch16-224"] = test_vit()
    results["whisper-large-v3"] = test_whisper()

    print("\n" + "="*50)
    print("Summary:")
    all_ok = True
    for model_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {model_name}: {status}")
        if not passed:
            all_ok = False
    print("="*50)
    if all_ok:
        print("All models verified and ready for class!")
    else:
        print("Some models encountered issues.")

if __name__ == "__main__":
    main()
