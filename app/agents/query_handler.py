"""Query Handler Agent: extract ticket number, look it up, report status."""
from __future__ import annotations

from typing import Any

from .base import BaseAgent


class QueryHandlerAgent(BaseAgent):
    name = "Query Handler Agent"

    def _handle(self, payload: dict) -> dict:
        message = payload["message"]
        ticket_id = self.llm.extract_ticket_number(message)

        if not ticket_id:
            return {
                "response": (
                    "I'd be happy to check on your ticket, but I couldn't find a "
                    "6-digit ticket number in your message. Please share your ticket "
                    "number (for example 482913) and I'll look it up right away."
                ),
                "action": "no_ticket_number",
                "ticket_id": None,
                "db_actions": [],
            }

        db_action = f"SELECT * FROM support_tickets WHERE ticket_id='{ticket_id}'"
        ticket = self.db.get_ticket(ticket_id)

        if ticket is None:
            return {
                "response": (
                    f"I couldn't find a ticket with ID {ticket_id} in our records. "
                    "Please double-check the number — it's the 6-digit ID you received "
                    "when we logged your issue. If it's still missing, share a few "
                    "details and we'll create a new ticket for you."
                ),
                "action": "ticket_not_found",
                "ticket_id": ticket_id,
                "db_actions": [db_action],
            }

        status = ticket["status"]
        created = ticket["created_at"][:10]
        if status == "resolved":
            response = (
                f"Good news — ticket {ticket_id} (opened on {created}) has been "
                "marked as resolved. If anything is still not right, just let me "
                "know and we'll reopen it right away."
            )
        else:
            response = (
                f"Here's the latest on ticket {ticket_id}: it is currently "
                f"{status} (opened on {created}). Our team is actively working on "
                "it and we'll keep you posted as soon as there's progress. Thanks "
                "for your patience!"
            )
        return {
            "response": response,
            "action": "status_lookup",
            "ticket_id": ticket_id,
            "ticket": ticket,
            "db_actions": [db_action],
        }
