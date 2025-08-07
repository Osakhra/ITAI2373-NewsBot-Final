# src/conversation/query_processor.py

"""
Query Processor for NewsBot 2.0
Parses user questions and routes to the appropriate module.
"""

import re
from .intent_classifier import IntentClassifier
from data_processing.feature_extractor import FeatureExtractor

class QueryProcessor:
    def __init__(self, classifier, sentiment_analyzer, ner_extractor, topic_modeler, summarizer, feature_extractor):
        self.intent_classifier = IntentClassifier() # Initialize IntentClassifier
        self.classifier = classifier
        self.sentiment_analyzer = sentiment_analyzer
        self.ner_extractor = ner_extractor
        self.topic_modeler = topic_modeler
        self.summarizer = summarizer
        self.feature_extractor = feature_extractor

    def process(self, query, article_text):
        """
        Processes a user query and returns the answer.
        Supported query types: category, sentiment, entities, topics, summary.
        """
        # Use intent classifier to determine the user's intent
        intent = self.intent_classifier.classify(query)
        print(f"Detected Intent: {intent}") # Optional: print detected intent

        if intent == "category" or any(k in query for k in ["category", "classify"]):
            article_features = self.feature_extractor.transform([article_text])
            return f"Predicted Category: {self.classifier.predict(article_features)[0]}"
        elif intent == "sentiment" or "sentiment" in query or "emotion" in query: # Check intent or keywords
            sentiment = self.sentiment_analyzer.analyze(article_text)
            label = self.sentiment_analyzer.label_sentiment(sentiment['polarity'])
            return f"Sentiment: {label} (polarity: {sentiment['polarity']:.2f})"
        elif intent == "entities" or "entity" in query or "person" in query or "organization" in query: # Check intent or keywords
            entities = self.ner_extractor.extract(article_text)
            if entities:
                ents = ', '.join([f"{text} [{label}]" for text, label in entities])
                return f"Entities found: {ents}"
            else:
                return "No entities found."
        elif intent == "topic" or any(k in query for k in ["topic", "theme", "main subject"]):
            # Pass RAW TEXT to TopicModeler; it handles its own vectorization
            topic_id = self.topic_modeler.assign_topic(article_text)
            # If your TopicModeler exposes top words:
        try:
            topic_words = self.topic_modeler.get_topic_words(topic_id, n_words=8)
            return f"Main topic #{topic_id}: {', '.join(topic_words)}"
         except Exception:
            # Fallback if get_topic_words isn't available/throws
            return f"Main topic #{topic_id}"

        elif intent == "summarize" or "summarize" in query or "summary" in query: # Check intent or keywords
            summary = self.summarizer.summarize(article_text)
            return f"Summary: {summary}"
        else:
            return "Sorry, I didn't understand your query. I can help you with category, sentiment, entities, topic, or summary."
