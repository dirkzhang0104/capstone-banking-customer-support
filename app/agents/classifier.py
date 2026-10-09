"""Classifier Agent: categorize input and decide the downstream route."""
from __future__ import annotations

from typing import Any

from ..llm.client import ALLOWED_LABELS
from .base import BaseAgent

ROUTE_TARGET = {
    "positive_feedback": "Feedback Handler Agent",
    "negative_feedback": "Feedback Handler Agent",
    "query": "Query Handler Agent",
}


class ClassifierAgent(BaseAgent):
    name = "Classifier Agent"

    def _handle(self, payload: dict) -> dict:
        message = payload["message"]
        classification = self.llm.classify(message)
        if classification.label not in ALLOWED_LABELS:  # defensive
            classification.label = "query"
        return {
            "label": classification.label,
            "confidence": classification.confidence,
            "routed_to": ROUTE_TARGET[classification.label],
        }
