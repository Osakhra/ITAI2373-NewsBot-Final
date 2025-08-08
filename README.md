# 📰 NewsBot 2.0 — AI-Powered News Intelligence System

## 📌 Overview

NewsBot 2.0 is an **AI-powered, end-to-end news analysis and intelligence platform** I developed as part of my ITAI2373 final project.
It processes, analyzes, and interacts with news articles in real time using advanced **Natural Language Processing (NLP)** and **Machine Learning** techniques.

The system can:

* **Classify** news articles into relevant categories
* **Analyze sentiment** to gauge emotional tone
* **Detect named entities** (people, organizations, locations, dates, etc.)
* **Identify main topics** through topic modeling
* **Summarize** long articles into concise summaries
* **Translate** non-English news into English
* **Answer user questions** via a conversational AI interface (powered by FastAPI + Gradio)

This project was built **module-by-module**, following professional software development practices, with a focus on performance, maintainability, and deployment readiness.

---

## 🚀 Features

* **Text Preprocessing**: Tokenization, stopword removal, lemmatization, language detection
* **Feature Extraction**: TF-IDF vectorization for machine learning models
* **News Classification**: Naive Bayes classifier with \~97% accuracy
* **Sentiment Analysis**: Rule-based polarity detection with labeling
* **Named Entity Recognition**: Extracts people, organizations, locations, and dates
* **Topic Modeling**: LDA-based unsupervised topic detection with top keywords
* **Summarization**: Extractive (TextRank) and transformer-based summaries
* **Multilingual Support**: Detect and translate non-English articles
* **Conversational Interface**: Interactive chatbot using a trained intent classifier and query processor
* **Web Deployment**: FastAPI backend with Gradio front-end for real-time user queries

---

## 📂 Project Structure

```
ITAI2373-NewsBot-Final/
│
├── data/
│   ├── raw/                  # Original datasets
│   ├── processed/            # Cleaned datasets
│   └── models/               # Saved ML models & vectorizers
│
├── src/
│   ├── analysis/             # Classifier, sentiment, topic modeling, NER
│   ├── conversation/         # Intent & query processors
│   ├── data_processing/      # Preprocessing & feature extraction
│   ├── language_models/      # Summarization & translation
│   └── utils/                # Helper functions
│
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Advanced_Classification.ipynb
│   ├── 03_Topic_Modeling.ipynb
│   ├── 04_Sentiment_Analysis.ipynb
│   ├── 05_Text_Summarization.ipynb
│   ├── 06_Multilingual_Analysis.ipynb
│   ├── 07_Conversational_Interface.ipynb
│   └── 08_Model_Saving_and_Deployment.ipynb
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🛠 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Osakhra/ITAI2373-NewsBot-Final.git
cd ITAI2373-NewsBot-Final
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Download NLTK Data

```python
import nltk
nltk.download('stopwords')
```

---

## 📊 Performance

Using the **BBC News dataset**, the Naive Bayes classifier achieved:

```
Accuracy: 96.64%
Precision, Recall, F1-score: ~97% across all categories
```

---

## 💡 How to Use

1. Place your **news dataset** in `data/raw/`
2. Run the preprocessing and training notebooks in order
3. Use the **Conversational Interface** to interact with the system
4. Optionally, run the **FastAPI + Gradio app** for web-based interaction

---

## 🖥 Example Interaction

```
User: What category is this article about?
NewsBot: Predicted Category: business

User: What is the sentiment of this news story?
NewsBot: Sentiment: neutral (polarity: 0.02)

User: Who or what is mentioned in this article?
NewsBot: Entities found: Cynthia Cooper [PERSON], Worldcom [ORG], 2002 [DATE], $11bn [MONEY], New York [GPE]
```

---

## 📚 Technologies Used

* **Python** (Core language)
* **scikit-learn** (Machine learning)
* **NLTK / spaCy** (NLP processing)
* **gensim / pyLDAvis** (Topic modeling & visualization)
* **transformers / torch** (Transformer models)
* **TextBlob / sumy** (Sentiment analysis & summarization)
* **googletrans** (Translation)
* **FastAPI / Gradio** (Web deployment)

---

## 📜 License

This project is released under the MIT License.
