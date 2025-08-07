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

        if intent == "category" or "category" in query or "classify" in query: # Check intent or keywords
            # Transform the article text using the feature extractor before predicting
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
        elif intent == "topic" or "topic" in query or "theme" in query: # Check intent or keywords
            article_features = self.feature_extractor.transform([article_text])
            topic_id = self.topic_modeler.assign_topic(article_features) # Corrected assign_topic call
            return f"Main Topic: {topic_id}" # Assuming assign_topic returns a topic identifier
        elif intent == "summarize" or "summarize" in query or "summary" in query: # Check intent or keywords
            summary = self.summarizer.summarize(article_text)
            return f"Summary: {summary}"
        else:
            return "Sorry, I didn't understand your query. I can help you with category, sentiment, entities, topic, or summary."
