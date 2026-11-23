"""
EE471 Week 9 - Task 3: Interactive Gradio Front-End
Provides an interactive web interface for all 10 Hugging Face tasks implemented in TransformersToolkit.
"""

import os
import gradio as gr
from pipeline_toolkit import TransformersToolkit

# Initialize the OOP Toolkit
toolkit = TransformersToolkit()

# Paths to sample media
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
SAMPLE_IMG = os.path.join(SAMPLE_DIR, "cat.jpg")
SAMPLE_AUDIO = os.path.join(SAMPLE_DIR, "sample_speech.wav")

custom_css = """
.tab-nav button {
    padding: 6px 10px !important;
    font-size: 13px !important;
}
"""

with gr.Blocks(title="EE471 - Hugging Face Multi-Task AI Dashboard") as demo:
    gr.Markdown("# EE471 Week 9: Hugging Face Transformers Dashboard")

    with gr.Tabs(overflow_behavior="wrap"):

        # Tab 1: Sentiment Analysis
        with gr.TabItem("1. Sentiment"):
            gr.Markdown("### Sentiment Analysis with DistilBERT")
            sentiment_input = gr.Textbox(
                label="Input Sentence",
                value="I've been waiting for a modern software practices course my whole life.",
                lines=2
            )
            sentiment_btn = gr.Button("Analyze Sentiment", variant="primary")
            sentiment_output = gr.Label(label="Sentiment Prediction")
            sentiment_btn.click(toolkit.sentiment_analysis, inputs=sentiment_input, outputs=sentiment_output)

        # Tab 2: Question Answering
        with gr.TabItem("2. Question Answering"):
            gr.Markdown("### Extractive Question Answering")
            qa_context = gr.Textbox(
                label="Context",
                value="I am Nate, a research assistant at Izmir Institute of Technology, and currently living and working in the beautiful city İzmir in Türkiye.",
                lines=3
            )
            qa_question = gr.Textbox(
                label="Question",
                value="Which company or institution do they work for?"
            )
            qa_btn = gr.Button("Find Answer", variant="primary")
            qa_output = gr.JSON(label="Answer Details")
            qa_btn.click(toolkit.question_answering, inputs=[qa_question, qa_context], outputs=qa_output)

        # Tab 3: Zero-shot Classification
        with gr.TabItem("3. Zero-Shot"):
            gr.Markdown("### Zero-Shot Classification with BART-large-MNLI")
            zs_text = gr.Textbox(
                label="Text to Classify",
                value="Berkshire keeps their cash reserves at an extremely high level.",
                lines=2
            )
            zs_labels = gr.Textbox(
                label="Candidate Labels (comma-separated)",
                value="education, finance, business, technology"
            )
            zs_btn = gr.Button("Classify", variant="primary")
            zs_output = gr.Label(label="Prediction Scores")
            zs_btn.click(toolkit.zero_shot_classification, inputs=[zs_text, zs_labels], outputs=zs_output)

        # Tab 4: Text Summarization
        with gr.TabItem("4. Summarization"):
            gr.Markdown("### Abstractive Summarization with DistilBART")
            sum_text = gr.Textbox(
                label="Long Text",
                value=(
                    "The 2008 Global Financial Crisis stands as the most severe economic collapse of the "
                    "21st century, often compared to the Great Depression of the 1930s. Triggered by the bursting "
                    "of the United States housing bubble, its effects rippled across the globe, leading to the collapse "
                    "of major financial institutions and a deep international recession. The crisis began with the "
                    "subprime mortgage market. In the early 2000s, low interest rates and a push for homeownership led "
                    "banks to issue high-risk loans to borrowers with poor credit."
                ),
                lines=5
            )
            sum_btn = gr.Button("Summarize", variant="primary")
            sum_output = gr.Textbox(label="Summary Output", lines=3)
            sum_btn.click(toolkit.text_summarization, inputs=sum_text, outputs=sum_output)

        # Tab 5: Text Generation
        with gr.TabItem("5. Text Generation"):
            gr.Markdown("### Text Generation / Completion with SmolLM2-360M")
            gen_prompt = gr.Textbox(
                label="Incomplete Prompt",
                value="If I continue to complete all in-class exercises in this modern software practices course,",
                lines=2
            )
            gen_len = gr.Slider(minimum=15, maximum=100, value=35, step=5, label="Max Generated Length")
            gen_btn = gr.Button("Generate Alternatives", variant="primary")
            gen_output = gr.JSON(label="Alternative Completions")
            gen_btn.click(toolkit.text_generation, inputs=[gen_prompt, gen_len], outputs=gen_output)

        # Tab 6: Text Translation
        with gr.TabItem("6. Translation"):
            gr.Markdown("### Machine Translation (English -> French) with MarianMT")
            trans_text = gr.Textbox(
                label="English Source Text",
                value="The 2008 Global Financial Crisis stands as the most severe economic collapse of the 21st century, often compared to the Great Depression.",
                lines=3
            )
            trans_btn = gr.Button("Translate to French", variant="primary")
            trans_output = gr.Textbox(label="French Translation", lines=3)
            trans_btn.click(toolkit.text_translation, inputs=trans_text, outputs=trans_output)

        # Tab 7: Mask Filling
        with gr.TabItem("7. Fill-Mask"):
            gr.Markdown("### Masked Language Modeling (Fill-Mask) with RoBERTa")
            mask_text = gr.Textbox(
                label="Prompt with '[...]' or '<mask>'",
                value="To understand generative AI, one must study [...] well."
            )
            mask_btn = gr.Button("Predict Masked Word", variant="primary")
            mask_output = gr.JSON(label="Top Predictions")
            mask_btn.click(toolkit.mask_filling, inputs=mask_text, outputs=mask_output)

        # Tab 8: Image Classification
        with gr.TabItem("8. Image (ViT)"):
            gr.Markdown("### Vision Transformer (ViT) Image Classification")
            img_input = gr.Image(type="pil", label="Upload Image", value=SAMPLE_IMG if os.path.exists(SAMPLE_IMG) else None)
            img_btn = gr.Button("Classify Image", variant="primary")
            img_output = gr.Label(label="Top-5 Class Predictions")
            img_btn.click(toolkit.image_classification, inputs=img_input, outputs=img_output)

        # Tab 9: Named Entity Recognition
        with gr.TabItem("9. NER"):
            gr.Markdown("### Token Classification / NER with BERT-large")
            ner_text = gr.Textbox(
                label="Input Sentence",
                value="I am Nate, a research assistant at Izmir Institute of Technology, and currently living and working in the beautiful city İzmir in Türkiye.",
                lines=2
            )
            ner_btn = gr.Button("Extract Entities", variant="primary")
            ner_output = gr.JSON(label="Extracted Entities (Person, Organization, Location)")
            ner_btn.click(toolkit.named_entity_recognition, inputs=ner_text, outputs=ner_output)

        # Tab 10: Automatic Speech Recognition
        with gr.TabItem("10. Speech (ASR)"):
            gr.Markdown("### Automatic Speech Recognition (ASR) with OpenAI Whisper Large v3")
            audio_input = gr.Audio(type="filepath", label="Upload or Record Audio", value=SAMPLE_AUDIO if os.path.exists(SAMPLE_AUDIO) else None)
            audio_btn = gr.Button("Transcribe Audio", variant="primary")
            audio_output = gr.Textbox(label="Transcribed Text", lines=2)
            audio_btn.click(toolkit.speech_recognition, inputs=audio_input, outputs=audio_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=custom_css)
