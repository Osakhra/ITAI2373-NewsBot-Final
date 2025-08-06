# src/multilingual/language_detector.py

"""
Language Detector for NewsBot 2.0
Uses langdetect to identify language of input text.
"""

from langdetect import detect, DetectorFactory, LangDetectException

# Make language detection deterministic
DetectorFactory.seed = 0

class LanguageDetector:
    def detect(self, text):
        """
        Returns ISO code (e.g. 'en', 'es') for detected language.
        """
        try:
            return detect(text)
        except LangDetectException:
            return "unknown"
