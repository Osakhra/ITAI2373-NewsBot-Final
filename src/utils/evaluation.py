# src/utils/evaluation.py

"""
Evaluation utilities for NewsBot 2.0
Provides metrics for classification, clustering, and topic modeling.
"""

from sklearn.metrics import classification_report, confusion_matrix, silhouette_score

def print_classification_metrics(y_true, y_pred):
    print(classification_report(y_true, y_pred))

def plot_confusion_matrix(y_true, y_pred, labels, figsize=(8,6)):
    import matplotlib.pyplot as plt
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def clustering_score(X, labels):
    return silhouette_score(X, labels)
