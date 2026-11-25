"""
EE471 Week 9 - Task 3: OOP Hugging Face Pipeline Toolkit
Encapsulates all 10 Transformer tasks in an object-oriented Python module.
Follows Hugging Face Course Chapters 1 & 2 design principles with on-demand (lazy) pipeline loading.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Union
import torch
from PIL import Image
import soundfile as sf
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

def _resolve_model(folder_name: str, default_repo: str) -> str:
    local_path = MODELS_DIR / folder_name
    return str(local_path) if local_path.exists() else default_repo

class TransformersToolkit:
    """
    Object-Oriented Toolkit wrapping Hugging Face Transformers for 10 core tasks:
    1. Sentiment Analysis
    2. Question Answering
    3. Zero-shot Classification
    4. Text Summarization
    5. Text Generation
    6. Text Translation
    7. Mask Filling
    8. Image Classification
    9. Named Entity Recognition (NER)
    10. Automatic Speech Recognition (ASR)
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._pipelines: Dict[str, Any] = {}
        self._translation_model = None
        self._translation_tokenizer = None
        self._summarization_model = None
        self._summarization_tokenizer = None
        self._qa_model = None
        self._qa_tokenizer = None

    # ------------------------------------------------------------------
    # 1. Sentiment Analysis
    # ------------------------------------------------------------------
    def sentiment_analysis(self, text: str) -> Dict[str, float]:
        """Classify the sentiment of input text (Positive/Negative)."""
        if "sentiment" not in self._pipelines:
            m = _resolve_model("sentiment-analysis", "distilbert/distilbert-base-uncased-finetuned-sst-2-english")
            self._pipelines["sentiment"] = pipeline("sentiment-analysis", model=m, device=self.device)
        results = self._pipelines["sentiment"](text)
        return {res["label"]: float(res["score"]) for res in results}

    # ------------------------------------------------------------------
    # 2. Question Answering
    # ------------------------------------------------------------------
    def question_answering(self, question: str, context: str) -> Dict[str, Any]:
        """Extract an answer to a question given context."""
        m = _resolve_model("question-answering", "distilbert/distilbert-base-cased-distilled-squad")
        try:
            if "qa" not in self._pipelines:
                self._pipelines["qa"] = pipeline("question-answering", model=m, device=self.device)
            res = self._pipelines["qa"](question=question, context=context)
            return {
                "answer": res["answer"],
                "score": round(float(res["score"]), 4),
                "start": res["start"],
                "end": res["end"]
            }
        except Exception:
            if self._qa_model is None or self._qa_tokenizer is None:
                self._qa_tokenizer = AutoTokenizer.from_pretrained(m)
                self._qa_model = AutoModelForQuestionAnswering.from_pretrained(m)
            inputs = self._qa_tokenizer(question, context, return_tensors="pt")
            with torch.no_grad():
                outputs = self._qa_model(**inputs)
            start_idx = torch.argmax(outputs.start_logits)
            end_idx = torch.argmax(outputs.end_logits)
            answer_tokens = inputs.input_ids[0, start_idx : end_idx + 1]
            answer = self._qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)
            start_prob = torch.softmax(outputs.start_logits, dim=-1)[0, start_idx].item()
            end_prob = torch.softmax(outputs.end_logits, dim=-1)[0, end_idx].item()
            return {
                "answer": answer,
                "score": round(start_prob * end_prob, 4),
                "start": int(start_idx),
                "end": int(end_idx)
            }

    # ------------------------------------------------------------------
    # 3. Zero-shot Classification
    # ------------------------------------------------------------------
    def zero_shot_classification(self, text: str, candidate_labels: Union[List[str], str]) -> Dict[str, float]:
        """Classify text into arbitrary categories without domain-specific training."""
        if "zero_shot" not in self._pipelines:
            m = _resolve_model("zero-shot-classification", "facebook/bart-large-mnli")
            self._pipelines["zero_shot"] = pipeline("zero-shot-classification", model=m, device=self.device)
        if isinstance(candidate_labels, str):
            labels = [l.strip() for l in candidate_labels.split(",") if l.strip()]
        else:
            labels = candidate_labels
        res = self._pipelines["zero_shot"](text, candidate_labels=labels)
        return {label: round(float(score), 4) for label, score in zip(res["labels"], res["scores"])}

    # ------------------------------------------------------------------
    # 4. Text Summarization
    # ------------------------------------------------------------------
    def text_summarization(self, text: str, max_length: int = 50, min_length: int = 20) -> str:
        """Condense a long passage into a concise summary."""
        m = _resolve_model("summarization", "sshleifer/distilbart-cnn-12-6")
        try:
            if "summarizer" not in self._pipelines:
                self._pipelines["summarizer"] = pipeline("summarization", model=m, device=self.device)
            res = self._pipelines["summarizer"](text, max_length=max_length, min_length=min_length, do_sample=False)
            return res[0]["summary_text"]
        except Exception:
            if self._summarization_model is None or self._summarization_tokenizer is None:
                self._summarization_tokenizer = AutoTokenizer.from_pretrained(m)
                self._summarization_model = AutoModelForSeq2SeqLM.from_pretrained(m)
            inputs = self._summarization_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = self._summarization_model.generate(inputs["input_ids"], max_length=max_length, min_length=min_length, do_sample=False)
            return self._summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # 5. Text Generation (Incomplete Sentence Completion)
    # ------------------------------------------------------------------
    def text_generation(self, prompt: str, max_length: int = 35, num_return_sequences: int = 2) -> List[str]:
        """Complete an incomplete sentence with generated continuations."""
        if "generator" not in self._pipelines:
            m = _resolve_model("SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M")
            self._pipelines["generator"] = pipeline("text-generation", model=m, device=self.device)
        outputs = self._pipelines["generator"](prompt, max_length=max_length, num_return_sequences=num_return_sequences)
        return [out["generated_text"] for out in outputs]

    # ------------------------------------------------------------------
    # 6. Text Translation (French -> English)
    # ------------------------------------------------------------------
    def text_translation(self, text: str) -> str:
        """Translate text from French to English using Helsinki-NLP/opus-mt-fr-en."""
        model_name = _resolve_model("opus-mt-fr-en", "Helsinki-NLP/opus-mt-fr-en")
        try:
            if "translator" not in self._pipelines:
                self._pipelines["translator"] = pipeline("translation", model=model_name, device=self.device)
            res = self._pipelines["translator"](text)
            return res[0]["translation_text"]
        except Exception:
            if self._translation_model is None or self._translation_tokenizer is None:
                self._translation_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            inputs = self._translation_tokenizer(text, return_tensors="pt")
            outputs = self._translation_model.generate(**inputs)
            return self._translation_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # 7. Mask Filling
    # ------------------------------------------------------------------
    def mask_filling(self, text_with_mask: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Predict the most probable replacement for a masked word."""
        if "unmasker" not in self._pipelines:
            m = _resolve_model("fill-mask", "distilbert/distilroberta-base")
            self._pipelines["unmasker"] = pipeline("fill-mask", model=m, device=self.device)
        
        mask_token = self._pipelines["unmasker"].tokenizer.mask_token
        # Automatically replace placeholder tokens like [...] or <mask> or [MASK]
        formatted_prompt = text_with_mask.replace("[...]", mask_token)
        if mask_token not in formatted_prompt:
            formatted_prompt = text_with_mask.replace("[MASK]", mask_token).replace("<mask>", mask_token)
        
        res = self._pipelines["unmasker"](formatted_prompt, top_k=top_k)
        return [
            {"token": r["token_str"].strip(), "score": round(float(r["score"]), 4), "sequence": r["sequence"]}
            for r in res
        ]

    # ------------------------------------------------------------------
    # 8. Image Classification
    # ------------------------------------------------------------------
    def image_classification(self, image: Union[str, Image.Image]) -> Dict[str, float]:
        """Classify an image using Google's ViT (google/vit-base-patch16-224)."""
        if "image_classifier" not in self._pipelines:
            m = _resolve_model("vit-base-patch16-224", "google/vit-base-patch16-224")
            self._pipelines["image_classifier"] = pipeline("image-classification", model=m, device=self.device)
        res = self._pipelines["image_classifier"](image)
        return {r["label"]: round(float(r["score"]), 4) for r in res[:5]}

    # ------------------------------------------------------------------
    # 9. Named Entity Recognition (NER)
    # ------------------------------------------------------------------
    def named_entity_recognition(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities (People, Organizations, Locations)."""
        if "ner" not in self._pipelines:
            m = _resolve_model("ner", "dbmdz/bert-large-cased-finetuned-conll03-english")
            self._pipelines["ner"] = pipeline("ner", model=m, aggregation_strategy="simple", device=self.device)
        res = self._pipelines["ner"](text)
        return [
            {"entity": r["word"], "type": r["entity_group"], "score": round(float(r["score"]), 4)}
            for r in res
        ]

    # ------------------------------------------------------------------
    # 10. Automatic Speech Recognition (ASR)
    # ------------------------------------------------------------------
    def speech_recognition(self, audio: Union[str, tuple, Any]) -> str:
        """Transcribe audio into text using OpenAI's Whisper Large v3."""
        if "asr" not in self._pipelines:
            m = _resolve_model("whisper-large-v3", "openai/whisper-large-v3")
            self._pipelines["asr"] = pipeline("automatic-speech-recognition", model=m, device=self.device)
        
        # If Gradio audio tuple (sample_rate, numpy_array)
        if isinstance(audio, tuple):
            sr, data = audio
            # Convert int to float if needed
            if data.dtype != "float32":
                data = data.astype("float32") / 32768.0
            res = self._pipelines["asr"]({"raw": data, "sampling_rate": sr})
        elif isinstance(audio, str):
            # Load via soundfile
            data, sr = sf.read(audio)
            res = self._pipelines["asr"]({"raw": data, "sampling_rate": sr})
        else:
            res = self._pipelines["asr"](audio)
            
        return res["text"].strip()
