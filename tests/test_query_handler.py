"""Tests for the Query Handler Agent."""


def _seed(orchestrator, ticket_id, status):
    orchestrator.db.create_ticket_with_id(ticket_id, "seeded for test", status=status)


def test_status_lookup_for_unresolved_ticket(orchestrator):
    _seed(orchestrator, "123456", "unresolved")
    out = orchestrator.query_handler.handle(
        {"message": "What is the status of my ticket 123456?", "label": "query"}
    )
    assert out["action"] == "status_lookup"
    assert out["ticket_id"] == "123456"
    assert "123456" in out["response"]
    assert "unresolved" in out["response"]
    assert any("SELECT" in a for a in out["db_actions"])


def test_status_lookup_for_resolved_ticket(orchestrator):
    _seed(orchestrator, "654321", "resolved")
    out = orchestrator.query_handler.handle(
        {"message": "Any update on ticket 654321?", "label": "query"}
    )
    assert out["action"] == "status_lookup"
    assert "resolved" in out["response"]


def test_missing_ticket_number_is_handled_gracefully(orchestrator):
    out = orchestrator.query_handler.handle(
        {"message": "Can you help me with my account?", "label": "query"}
    )
    assert out["action"] == "no_ticket_number"
    assert out["ticket_id"] is None
    assert "6-digit" in out["response"]


def test_unknown_ticket_id_is_handled_gracefully(orchestrator):
    out = orchestrator.query_handler.handle(
        {"message": "Status of ticket 999999?", "label": "query"}
    )
    assert out["action"] == "ticket_not_found"
    assert "999999" in out["response"]
