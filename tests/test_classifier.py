"""Tests for the Classifier Agent and LLM classification (30-case coverage)."""
import pytest

from app.eval.test_cases import TEST_CASES
from app.llm.client import LLMClient


@pytest.mark.parametrize("case", TEST_CASES, ids=[c.text[:30] for c in TEST_CASES])
def test_classify_labels(case):
    llm = LLMClient(api_key="")
    assert llm.classify(case.text).label == case.expected


def test_classify_confidence_in_range():
    llm = LLMClient(api_key="")
    for case in TEST_CASES:
        c = llm.classify(case.text)
        assert 0.0 <= c.confidence <= 1.0


def test_classifier_agent_returns_route_target():
    llm = LLMClient(api_key="")
    from app.agents.classifier import ClassifierAgent
    from app.db.database import SupportDatabase
    from app.logs.trace import TraceLogger
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    agent = ClassifierAgent(llm, SupportDatabase(tmp / "t.db"), TraceLogger(tmp / "tr.jsonl", tmp / "st.json"))
    out = agent.handle({"message": "What is the status of my ticket 482913?"})
    assert out["label"] == "query"
    assert out["routed_to"] == "Query Handler Agent"
