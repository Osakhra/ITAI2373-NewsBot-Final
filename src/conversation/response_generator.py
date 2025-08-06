# src/conversation/response_generator.py

"""
Response Generator for NewsBot 2.0
Creates conversational, human-friendly replies for query results.
"""

class ResponseGenerator:
    def generate_response(self, intent, result):
        if intent == "category":
            return f"The article falls under the category: {result}"
        elif intent == "sentiment":
            return f"The article's sentiment is: {result}"
        elif intent == "entities":
            return f"The following entities were found: {result}"
        elif intent == "topic":
            return f"The main topic is: {result}"
        elif intent == "summary":
            return f"Here's a summary of the article: {result}"
        else:
            return f"I'm not sure how to answer that. Please try a different question."
