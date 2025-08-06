# src/language_models/embeddings.py

"""
Embeddings module for NewsBot 2.0
Uses Sentence Transformers for semantic similarity, clustering, and search.
"""

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

import numpy as np

class EmbeddingExtractor:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        if SentenceTransformer:
            self.model = SentenceTransformer(model_name)
        else:
            self.model = None

    def encode(self, texts):
        """
        Returns embeddings for a list of texts.
        """
        if not self.model:
            raise RuntimeError("SentenceTransformer not installed.")
        return self.model.encode(texts, convert_to_numpy=True)

    def similarity(self, emb1, emb2):
        """
        Computes cosine similarity between two embeddings.
        """
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
