"""Tests for the Feedback Handler Agent."""
import re

POSITIVE = "I love the new mobile app, it's so fast!"
NEGATIVE = "I'm so frustrated with the hidden fees on my account!"


def test_positive_feedback_returns_warm_thanks_without_ticket(orchestrator):
    out = orchestrator.feedback_handler.handle({"message": POSITIVE, "label": "positive_feedback"})
    assert out["action"] == "positive_ack"
    assert out["ticket_id"] is None
    assert out["db_actions"] == []
    assert len(out["response"].split()) >= 10
    assert "thank" in out["response"].lower()


def test_negative_feedback_creates_unresolved_6_digit_ticket(orchestrator):
    out = orchestrator.feedback_handler.handle({"message": NEGATIVE, "label": "negative_feedback"})
    ticket_id = out["ticket_id"]
    assert re.fullmatch(r"\d{6}", ticket_id)
    assert out["action"] == "ticket_created"
    assert ticket_id in out["response"]
    assert "follow up" in out["response"].lower()

    ticket = orchestrator.db.get_ticket(ticket_id)
    assert ticket is not None
    assert ticket["status"] == "unresolved"
    assert ticket["sentiment"] == "negative"
    assert NEGATIVE in ticket["message"]


def test_ticket_ids_are_unique(orchestrator):
    ids = {
        orchestrator.feedback_handler.handle(
            {"message": f"Bad experience #{i}", "label": "negative_feedback"}
        )["ticket_id"] for i in range(25)
    }
    assert len(ids) == 25


def test_invalid_label_is_caught_and_recorded_as_failure(orchestrator, tracer):
    out = orchestrator.feedback_handler.handle({"message": "hi", "label": "bogus"})
    assert "error" in out
    stats = tracer.agent_stats()
    assert stats["Feedback Handler Agent"]["failure"] >= 1
