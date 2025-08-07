# src/conversation/query_processor.py

"""
Query Processor for NewsBot 2.0
Parses user questions and routes to the appropriate module.
"""

from analysis.feature_extractor import FeatureExtractor

class QueryProcessor:
    def __init__(self, classifier, sentiment_analyzer, ner_extractor, topic_modeler, summarizer, feature_extractor):
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
        query = query.lower()
        if "category" in query or "classify" in query:
            # Transform the article text using the feature extractor before predicting
            article_features = self.feature_extractor.transform([article_text])
            return f"Predicted Category: {self.classifier.predict(article_features)[0]}"
        elif "sentiment" in query or "emotion" in query:
            sentiment = self.sentiment_analyzer.analyze(article_text)
            label = self.sentiment_analyzer.label_sentiment(sentiment['polarity'])
            return f"Sentiment: {label} (polarity: {sentiment['polarity']:.2f})"
        elif "entity" in query or "person" in query or "organization" in query:
            entities = self.ner_extractor.extract(article_text)
            if entities:
                ents = ', '.join([f"{text} [{label}]" for text, label in entities])
                return f"Entities found: {ents}"
            else:
                return "No entities found."
        elif "topic" in query or "theme" in query:
            article_features = self.feature_extractor.transform([article_text])
            topic_id = self.topic_modeler.assign_topic(article_features)
            topic_words = self.topic_modeler.get_topic_words(topic_id)
            return f"Main topic #{topic_id}: {', '.join(topic_words)}"
        elif "summary" in query or "summarize" in query:
            summary = self.summarizer.summarize(article_text)
            return f"Summary: {summary}"
        else:
            return "Sorry, I didn't understand the query. Try: category, sentiment, entities, topics, or summary."
