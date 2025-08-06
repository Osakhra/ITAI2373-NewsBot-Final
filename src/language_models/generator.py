# src/language_models/generator.py

"""
Content Generator for NewsBot 2.0
Uses Hugging Face transformers for text generation (e.g., GPT-2).
"""

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

class ContentGenerator:
    def __init__(self, model_name="gpt2"):
        if pipeline:
            try:
                self.generator = pipeline("text-generation", model=model_name)
            except Exception as e:
                print(f"Text generation model failed to load: {e}.")
                self.generator = None
        else:
            self.generator = None

    def generate(self, prompt, max_length=100):
        if not self.generator:
            raise RuntimeError("No text generation pipeline available.")
        result = self.generator(prompt, max_length=max_length, num_return_sequences=1)
        return result[0]['generated_text'] if result else ""
