"""Shared fixtures: isolated temp DB / tracer / forced-heuristic LLM."""
import pytest

from app.db.database import SupportDatabase
from app.llm.client import LLMClient
from app.logs.trace import TraceLogger
from app.orchestrator import Orchestrator


@pytest.fixture
def db(tmp_path):
    database = SupportDatabase(tmp_path / "test.db")
    yield database
    database.close()


@pytest.fixture
def tracer(tmp_path):
    return TraceLogger(tmp_path / "traces.jsonl", tmp_path / "stats.json")


@pytest.fixture
def llm():
    # Empty api key forces the deterministic heuristic engine (offline tests).
    return LLMClient(api_key="")


@pytest.fixture
def orchestrator(llm, db, tracer):
    return Orchestrator(llm=llm, db=db, tracer=tracer)
