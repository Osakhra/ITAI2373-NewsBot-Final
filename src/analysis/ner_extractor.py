# src/analysis/ner_extractor.py

"""
NER Extractor for NewsBot 2.0
Extracts named entities (PERSON, ORG, GPE, etc.) using spaCy.
"""

import spacy

class NERExtractor:
    def __init__(self, language='en'):
        if language == 'en':
            self.nlp = spacy.load("en_core_web_sm")
        else:
            raise NotImplementedError("Only English is supported in this starter. Extend as needed.")
        self.language = language

    def extract(self, text):
        """
        Returns a list of (entity_text, entity_label) tuples.
        """
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def entities_by_type(self, text):
        """
        Returns a dictionary of entity types to lists of entities.
        """
        doc = self.nlp(text)
        ents = {}
        for ent in doc.ents:
            ents.setdefault(ent.label_, []).append(ent.text)
        return ents
