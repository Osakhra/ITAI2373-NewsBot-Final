# NewsBot Intelligence System 2.0

## Overview
NewsBot 2.0 is an advanced, production-ready NLP platform for news analysis, built as a capstone project for ITAI2373.  
It supports everything from text classification and topic modeling to multilingual analysis and conversational search—demonstrating the full range of modern NLP for real business and media use.

## Features
- **Advanced Content Analysis**: Multi-level classification, topic discovery (LDA/NMF), sentiment tracking, entity mapping.
- **Language Understanding & Generation**: Article summarization, text generation (GPT-2), semantic search (embeddings).
- **Multilingual Intelligence**: Automatic language detection, translation, cross-lingual topic/entity analysis.
- **Conversational Interface**: Natural language queries, intent detection, and human-like responses.
- **Production-Grade Code**: Modular, tested, and documented—ready for real projects or job applications.

## Project Structure
```

ITAI2373-NewsBot-Final/
├── README.md
├── requirements.txt
├── config/
├── src/
│   ├── data\_processing/
│   ├── analysis/
│   ├── language\_models/
│   ├── multilingual/
│   ├── conversation/
│   └── utils/
├── notebooks/
├── tests/
├── data/
├── docs/
├── reports/

````

## Getting Started
1. **Clone this repo** and install dependencies:
    ```
    pip install -r requirements.txt
    ```
2. **Configure settings:** Copy `config/api_keys_template.txt` to `api_keys.txt` if needed.
3. **Run demo notebooks:** Open any notebook in `notebooks/` and follow the instructions.
4. **Try the conversational system:** Use the CLI, notebook, or optional web app for live querying and analysis.

## Example Usage
```python
from src.data_processing.text_preprocessor import TextPreprocessor
from src.analysis.classifier import NewsClassifier

pre = TextPreprocessor()
texts = ["Sample news article here..."]
clean = pre.preprocess(texts[0])

clf = NewsClassifier()
# ... train or load a model ...
# category = clf.predict([clean])
````

*(See full demos in `notebooks/`!)*

## Visualizations

* Category and sentiment plots, topic word charts, confusion matrices, and more—see `src/utils/visualization.py` and notebooks.

## Multilingual & Conversational

* Analyze English, Spanish, and more.
* Translate or compare news across languages.
* Use the query processor for “show me positive tech news” and similar requests.

## Documentation

* Technical/user/business docs in `docs/`
* Executive summary and presentation in `reports/`

## Team & Contributions

* \John Castor, \Dylan Castillo, \Milagros Pumasupa, \Ola Bakare
* See `docs/individual_contributions.md` for details

## License & Attribution

* For ITAI2373, \Patricia McManus, Summer 2025
* Powered by: scikit-learn, spaCy, transformers, sentence-transformers, textblob, sumy, googletrans, matplotlib, seaborn, and more.
