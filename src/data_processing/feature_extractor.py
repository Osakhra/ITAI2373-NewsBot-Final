# src/data_processing/feature_extractor.py

"""
Feature Extraction module for NewsBot 2.0
Handles TF-IDF vectorization and can be extended for embeddings.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class FeatureExtractor:
    def __init__(self, max_features=2000, ngram_range=(1,2)):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.fitted = False

    def fit_transform(self, documents):
        """
        Fits the vectorizer to the data and transforms the input documents.
        Returns the TF-IDF matrix.
        """
        X = self.vectorizer.fit_transform(documents)
        self.fitted = True
        return X

    def transform(self, documents):
        """
        Transforms new documents to the existing TF-IDF space.
        """
        if not self.fitted:
            raise RuntimeError("Vectorizer must be fitted before calling transform.")
        return self.vectorizer.transform(documents)

    def get_feature_names(self):
        """
        Returns the feature (word/phrase) names.
        """
        return self.vectorizer.get_feature_names_out()

    def save(self, filepath):
        """Saves the fitted vectorizer to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load(self, filepath):
        """Loads a fitted vectorizer from disk."""
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
            self.fitted = True
