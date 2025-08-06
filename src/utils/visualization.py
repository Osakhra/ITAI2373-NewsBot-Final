# src/utils/visualization.py

"""
Visualization tools for NewsBot 2.0
Plots for category counts, sentiment distributions, and topic modeling.
"""

import matplotlib.pyplot as plt
import seaborn as sns

def plot_category_distribution(df, category_col='category'):
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x=category_col, order=df[category_col].value_counts().index)
    plt.title('Category Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_sentiment_distribution(df, sentiment_col='sentiment', category_col='category'):
    plt.figure(figsize=(10,6))
    for cat in df[category_col].unique():
        sns.histplot(df[df[category_col] == cat][sentiment_col], bins=20, kde=True, label=cat, alpha=0.5)
    plt.title('Sentiment Distribution by Category')
    plt.xlabel('Sentiment Polarity')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_top_topic_words(topics_dict, n_words=10):
    plt.figure(figsize=(12,6))
    for topic_id, words in topics_dict.items():
        plt.bar([f'T{topic_id}-{w}' for w in words[:n_words]], [n_words]*n_words, label=f"Topic {topic_id}")
    plt.xticks(rotation=90)
    plt.legend()
    plt.title('Top Words per Topic')
    plt.tight_layout()
    plt.show()
