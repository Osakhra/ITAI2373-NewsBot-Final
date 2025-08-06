# src/multilingual/cross_lingual_analyzer.py

"""
Cross-Lingual Analyzer for NewsBot 2.0
Supports comparing articles/topics/entities across languages.
"""

from src.multilingual.language_detector import LanguageDetector
from src.multilingual.translator import NewsBotTranslator

class CrossLingualAnalyzer:
    def __init__(self, default_lang='en'):
        self.lang_detector = LanguageDetector()
        self.translator = NewsBotTranslator()
        self.default_lang = default_lang

    def align_texts(self, texts, target_lang=None):
        """
        Translates a list of texts to target_lang (or self.default_lang).
        Returns list of translated texts.
        """
        if target_lang is None:
            target_lang = self.default_lang
        translations = []
        for text in texts:
            lang = self.lang_detector.detect(text)
            if lang != target_lang:
                translations.append(self.translator.translate(text, dest=target_lang))
            else:
                translations.append(text)
        return translations

    # Optional: Add methods for cross-lingual topic/entity comparison!
