# src/conversation/intent_classifier.py

"""
Intent Classifier for NewsBot 2.0
Classifies user queries by intent (e.g., sentiment, topic, summary, etc.).
"""

import re

class IntentClassifier:
    """
    Tiny intent classifier for NewsBot.
    Returns one of: 'category', 'sentiment', 'entities', 'topic', 'summary', 'unknown'
    """
    def __init__(self):
        # You can expand these keyword lists later if you want
        self.intents = {
            "category":  ["category", "classify", "label", "what is this about"],
            "sentiment": ["sentiment", "emotion", "tone", "polarity"],
            "entities":  ["entity", "entities", "person", "organization", "org", "who", "what is mentioned"],
            "topic":     ["topic", "theme", "main subject"],
            "summary":   ["summary", "summarize", "tl;dr"]
        }

    def classify(self, query: str) -> str:
        q = query.lower().strip()
        # quick keyword match
        for intent, keywords in self.intents.items():
            if any(k in q for k in keywords):
                return intent
        return "unknown"

