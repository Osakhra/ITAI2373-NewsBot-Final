# src/analysis/topic_modeler.py

"""
Topic Modeling for NewsBot 2.0
Implements LDA and NMF for topic discovery, visualization, and clustering.
"""

from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

class TopicModeler:
    def __init__(self, n_topics=10, method='lda', max_features=2000):
        self.n_topics = n_topics
        self.method = method
        self.max_features = max_features
        self.model = None
        self.vectorizer = None

    def fit_transform(self, documents):
        """
        Fits the topic model and transforms documents.
        Returns document-topic matrix.
        """
        if self.method == 'lda':
            self.vectorizer = CountVectorizer(max_features=self.max_features, stop_words='english')
            X = self.vectorizer.fit_transform(documents)
            self.model = LatentDirichletAllocation(n_components=self.n_topics, random_state=42)
        elif self.method == 'nmf':
            self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words='english')
            X = self.vectorizer.fit_transform(documents)
            self.model = NMF(n_components=self.n_topics, random_state=42)
        else:
            raise ValueError("method must be 'lda' or 'nmf'")
        doc_topic_matrix = self.model.fit_transform(X)
        return doc_topic_matrix

    def get_topic_words(self, topic_id, n_words=10):
        """
        Returns the top words for a given topic.
        """
        if not self.model or not self.vectorizer:
            raise RuntimeError("Model not fitted.")
        feature_names = self.vectorizer.get_feature_names_out()
        if self.method == 'lda':
            topic = self.model.components_[topic_id]
        else:  # NMF
            topic = self.model.components_[topic_id]
        top_indices = topic.argsort()[-n_words:][::-1]
        return [feature_names[i] for i in top_indices]

    def get_all_topics(self, n_words=10):
        """
        Returns top words for all topics as a dict.
        """
        topics = {}
        for topic_id in range(self.n_topics):
            topics[topic_id] = self.get_topic_words(topic_id, n_words)
        return topics

    def transform(self, documents):
        """
        Transforms new documents into the topic space.
        """
        X = self.vectorizer.transform(documents)
        return self.model.transform(X)

    def assign_topic(self, document):
        """
        Assigns the main topic to a single document.
        """
        doc_topic = self.transform([document])[0]
        return np.argmax(doc_topic)

    # Optional: visualize topics with pyLDAvis
    def visualize_topics(self, documents):
        try:
            import pyLDAvis
            import pyLDAvis.sklearn
        except ImportError:
            print("pyLDAvis is not installed. Install with: pip install pyldavis")
            return None
        X = self.vectorizer.transform(documents)
        vis_data = pyLDAvis.sklearn.prepare(self.model, X, self.vectorizer)
        return vis_data  # Use pyLDAvis.display(vis_data) in Jupyter/Colab
