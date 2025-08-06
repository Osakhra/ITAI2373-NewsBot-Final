# src/multilingual/translator.py

"""
Translation module for NewsBot 2.0
Uses googletrans for free translation; easily swapped for pro APIs.
"""

try:
    from googletrans import Translator
except ImportError:
    Translator = None

class NewsBotTranslator:
    def __init__(self):
        if Translator:
            self.translator = Translator()
        else:
            self.translator = None

    def translate(self, text, dest='en'):
        """
        Translates text to the specified language (default: English).
        """
        if not self.translator:
            raise RuntimeError("googletrans not installed.")
        try:
            result = self.translator.translate(text, dest=dest)
            return result.text
        except Exception as e:
            print(f"Translation failed: {e}")
            return text
