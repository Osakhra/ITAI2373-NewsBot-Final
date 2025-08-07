# src/conversation/query_processor.py

"""
Query Processor for NewsBot 2.0
Parses user questions and routes to the appropriate module.
"""

from .intent_classifier import IntentClassifier
# Note: FeatureExtractor is injected from outside; no need to import it here.

class QueryProcessor:
    def __init__(self, classifier, sentiment_analyzer, ner_extractor, topic_modeler, summarizer, feature_extractor):
        self.intent_classifier = IntentClassifier()
        self.classifier = classifier
        self.sentiment_analyzer = sentiment_analyzer
        self.ner_extractor = ner_extractor
        self.topic_modeler = topic_modeler
        self.summarizer = summarizer
        self.feature_extractor = feature_extractor

    def process(self, query, article_text):
        """
        Processes a user query and returns the answer.
        Supported query types: category, sentiment, entities, topic, summary.
        """
        query = (query or "").lower().strip()

        # Use intent classifier to determine the user's intent
        intent = self.intent_classifier.classify(query)
        print(f"Detected Intent: {intent}")

        # CATEGORY
        if intent == "category" or any(k in query for k in ["category", "classify", "classification"]):
            # Use the same vectorizer used during training
            article_features = self.feature_extractor.transform([article_text])
            return f"Predicted Category: {self.classifier.predict(article_features)[0]}"

        # SENTIMENT
        elif intent == "sentiment" or any(k in query for k in ["sentiment", "emotion", "tone", "polarity"]):
            sentiment = self.sentiment_analyzer.analyze(article_text)
            label = self.sentiment_analyzer.label_sentiment(sentiment["polarity"])
            return f"Sentiment: {label} (polarity: {sentiment['polarity']:.2f})"

        # ENTITIES
        elif intent == "entities" or any(k in query for k in ["entity", "entities", "person", "organization", "org", "people", "who is mentioned"]):
            entities = self.ner_extractor.extract(article_text)
            if entities:
                ents = ", ".join([f"{text} [{label}]" for text, label in entities])
                return f"Entities found: {ents}"
            else:
                return "No entities found."

        # TOPIC
        elif intent == "topic" or any(k in query for k in ["topic", "theme", "main subject"]):
            # IMPORTANT: pass RAW TEXT; TopicModeler handles its own vectorizer
            topic_id = self.topic_modeler.assign_topic(article_text)
            try:
                topic_words = self.topic_modeler.get_topic_words(topic_id, n_words=8)
                return f"Main topic #{topic_id}: {', '.join(topic_words)}"
            except Exception:
                # Fallback if get_topic_words isn't available
                return f"Main topic #{topic_id}"

        # SUMMARY
        elif intent in ["summarize", "summary"] or any(k in query for k in ["summarize", "summary", "tl;dr"]):
            summary = self.summarizer.summarize(article_text)
            return f"Summary: {summary}"

        # DEFAULT
        else:
            return "Sorry, I didn't understand your query. I can help with: category, sentiment, entities, topic, or summary."
