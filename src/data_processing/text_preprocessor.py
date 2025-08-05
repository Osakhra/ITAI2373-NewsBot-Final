# src/data_processing/text_preprocessor.py

"""
Text Preprocessor module for NewsBot 2.0
Handles text cleaning, tokenization, lemmatization, and stopword removal.
Ready for multilingual support.
"""

import re
import spacy
from nltk.corpus import stopwords
from langdetect import detect, LangDetectException

class TextPreprocessor:
    def __init__(self, language='en'):
        self.language = language
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # If not installed, guide the user
            raise ImportError("spaCy 'en_core_web_sm' model not found. Run: python -m spacy download en_core_web_sm")
        self.stop_words = set(stopwords.words('english'))

    def detect_language(self, text):
        """Detects the language of the given text."""
        try:
            lang = detect(text)
            return lang
        except LangDetectException:
            return "unknown"

    def clean_text(self, text):
        """Removes extra whitespace, special characters, and lowercases."""
        text = re.sub(r'\s+', ' ', str(text)).strip()
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.lower()

    def preprocess(self, text):
        """Full pipeline: clean, tokenize, remove stopwords, lemmatize."""
        text = self.clean_text(text)
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in self.stop_words]
        return ' '.join(tokens)

    def process_batch(self, texts):
        """Preprocesses a list of texts."""
        return [self.preprocess(text) for text in texts]
