"""End-to-end tests for the Orchestrator (the three spec use-case flows)."""
import json


def test_example1_positive_feedback_flow(orchestrator, tracer):
    trace = orchestrator.handle_message("I love the new mobile app, it's so fast!")
    assert trace["classification"]["label"] == "positive_feedback"
    assert trace["agent_path"] == ["Classifier Agent", "Feedback Handler Agent"]
    assert trace["ticket_id"] is None
    assert "thank" in trace["response"].lower()


def test_example2_negative_feedback_flow(orchestrator, tracer):
    trace = orchestrator.handle_message("I'm so frustrated with the hidden fees on my account!")
    assert trace["classification"]["label"] == "negative_feedback"
    assert trace["agent_path"] == ["Classifier Agent", "Feedback Handler Agent"]
    assert trace["ticket_id"] is not None
    ticket = orchestrator.db.get_ticket(trace["ticket_id"])
    assert ticket and ticket["status"] == "unresolved"


def test_example3_query_flow(orchestrator, tracer):
    orchestrator.db.create_ticket_with_id("482913", "seeded", status="unresolved")
    trace = orchestrator.handle_message("What is the status of my ticket 482913?")
    assert trace["classification"]["label"] == "query"
    assert trace["agent_path"] == ["Classifier Agent", "Query Handler Agent"]
    assert "482913" in trace["response"]
    assert "unresolved" in trace["response"]


def test_trace_is_persisted_to_jsonl(orchestrator, tracer):
    orchestrator.handle_message("Great service, thanks a lot!")
    traces = tracer.load_traces()
    assert len(traces) == 1
    t = traces[0]
    for key in ("trace_id", "message", "classification", "agent_path",
                "response", "db_actions", "prompt_traces"):
        assert key in t
    json.loads(json.dumps(t))  # must be JSON-serializable


def test_agent_stats_recorded(orchestrator, tracer):
    orchestrator.handle_message("Terrible experience today.")
    stats = tracer.agent_stats()
    assert stats["Classifier Agent"]["total"] == 1
    assert stats["Feedback Handler Agent"]["total"] == 1
