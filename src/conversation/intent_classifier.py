# src/conversation/intent_classifier.py

"""
Intent Classifier for NewsBot 2.0
Classifies user queries by intent (e.g., sentiment, topic, summary, etc.).
"""

import re

class IntentClassifier:
    def __init__(self):
        # Define keywords for each intent
        self.intent_keywords = {
            "category": ["category", "classify", "label"],
            "sentiment": ["sentiment", "emotion", "feeling", "tone"],
            "entities": ["entity", "entities", "person", "organization", "place", "location", "company"],
            "topic": ["topic", "theme", "subject"],
            "summary": ["summary", "summarize", "abstract", "short version"],
        }

    def classify_intent(self, query):
        query = query.lower()
        for intent, keywords in self.intent_keywords.items():
            if any(re.search(rf'\b{kw}\b', query) for kw in keywords):
                return intent
        return "unknown"
