"""Banking Customer Support — Multi-Agent AI system.

A multi-agent GenAI system for banking customer support workflows:
a Classifier Agent routes incoming messages to a Feedback Handler Agent
(positive/negative) or a Query Handler Agent (ticket status lookups),
backed by a SQLite support ticket database, with LLMOps tooling
(evaluation, tracing, logging) and a Streamlit UI.
"""
