# src/language_models/summarizer.py

"""
Summarization module for NewsBot 2.0
Uses Hugging Face transformers for abstractive summarization; fallback to TextRank if needed.
"""

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

try:
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
except ImportError:
    TextRankSummarizer = None

class Summarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.use_transformer = False
        if pipeline:
            try:
                self.summarizer = pipeline("summarization", model=model_name)
                self.use_transformer = True
            except Exception as e:
                print(f"Transformer model failed to load: {e}. Falling back to TextRank.")
        if not self.use_transformer and TextRankSummarizer:
            self.summarizer = TextRankSummarizer()
        elif not self.use_transformer:
            raise ImportError("No summarizer available. Install transformers or sumy.")

    def summarize(self, text, max_length=120, min_length=30):
        if self.use_transformer:
            result = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return result[0]['summary_text']
        else:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summary_sentences = self.summarizer(parser.document, sentences_count=3)
            return " ".join([str(sentence) for sentence in summary_sentences])
