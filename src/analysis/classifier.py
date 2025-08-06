# src/analysis/classifier.py

"""
Classifier module for NewsBot 2.0
Supports training, saving, loading, and predicting with news classifiers.
"""

import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score

class NewsClassifier:
    def __init__(self, model_type='nb'):
        if model_type == 'nb':
            self.model = MultinomialNB()
        elif model_type == 'logreg':
            self.model = LogisticRegression(max_iter=1000)
        else:
            raise ValueError("Unsupported model_type. Use 'nb' or 'logreg'.")
        self.fitted = False

    def train(self, X, y):
        """Fit the classifier to the data."""
        self.model.fit(X, y)
        self.fitted = True

    def predict(self, X):
        """Predicts labels for new data."""
        if not self.fitted:
            raise RuntimeError("Model not trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Returns confidence scores (probabilities)."""
        if not self.fitted:
            raise RuntimeError("Model not trained yet.")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            return None

    def evaluate(self, X, y_true):
        """Prints a classification report and accuracy."""
        y_pred = self.predict(X)
        print(classification_report(y_true, y_pred))
        print("Accuracy: {:.2f}%".format(accuracy_score(y_true, y_pred) * 100))

    def save(self, filename):
        """Saves the model to disk."""
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, filename):
        """Loads the model from disk."""
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)
        self.fitted = True
