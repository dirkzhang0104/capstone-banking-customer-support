"""Feedback Handler Agent: warm thanks for positive, ticket + empathy for negative."""
from __future__ import annotations

from typing import Any

from .base import BaseAgent


class FeedbackHandlerAgent(BaseAgent):
    name = "Feedback Handler Agent"

    def _handle(self, payload: dict) -> dict:
        message = payload["message"]
        label = payload.get("label")

        if label == "positive_feedback":
            response = self.llm.generate_thankyou(message)
            return {
                "response": response,
                "action": "positive_ack",
                "ticket_id": None,
                "db_actions": [],
            }

        if label == "negative_feedback":
            ticket_id = self.db.create_ticket(message, sentiment="negative")
            response = self.llm.generate_empathy(message, ticket_id)
            return {
                "response": response,
                "action": "ticket_created",
                "ticket_id": ticket_id,
                "db_actions": [
                    f"INSERT INTO support_tickets (ticket_id='{ticket_id}', "
                    f"sentiment='negative', status='unresolved')"
                ],
            }

        raise ValueError(f"Feedback Handler Agent cannot handle label '{label}'")
