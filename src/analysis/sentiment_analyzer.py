# src/analysis/sentiment_analyzer.py

"""
Sentiment Analyzer for NewsBot 2.0
Uses TextBlob for English sentiment, easily extendable for multilingual.
"""

from textblob import TextBlob

class SentimentAnalyzer:
    def __init__(self, language='en'):
        self.language = language

    def analyze(self, text):
        """
        Returns sentiment polarity (-1 to 1) and subjectivity (0 to 1).
        """
        if not text or not isinstance(text, str):
            return {'polarity': 0.0, 'subjectivity': 0.0}
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }

    def label_sentiment(self, polarity):
        """
        Maps polarity score to label (positive, negative, neutral).
        """
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
