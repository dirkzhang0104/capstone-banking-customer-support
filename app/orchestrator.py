"""Agent orchestration: Classifier -> Feedback Handler / Query Handler."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from .agents.classifier import ClassifierAgent
from .agents.feedback_handler import FeedbackHandlerAgent
from .agents.query_handler import QueryHandlerAgent
from .db.database import SupportDatabase
from .llm.client import LLMClient
from .logs.trace import TraceLogger


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Orchestrator:
    """Coordinates the multi-agent pipeline and produces a full message trace."""

    def __init__(self, llm: Optional[LLMClient] = None,
                 db: Optional[SupportDatabase] = None,
                 tracer: Optional[TraceLogger] = None):
        self.llm = llm if llm is not None else LLMClient()
        self.db = db if db is not None else SupportDatabase()
        self.tracer = tracer if tracer is not None else TraceLogger()
        self.classifier = ClassifierAgent(self.llm, self.db, self.tracer)
        self.feedback_handler = FeedbackHandlerAgent(self.llm, self.db, self.tracer)
        self.query_handler = QueryHandlerAgent(self.llm, self.db, self.tracer)
        self._route = {
            "positive_feedback": self.feedback_handler,
            "negative_feedback": self.feedback_handler,
            "query": self.query_handler,
        }

    def handle_message(self, message: str) -> dict:
        """Run the full agent pipeline for one customer message."""
        trace: dict = {
            "trace_id": uuid.uuid4().hex[:12],
            "message": message,
            "ts": _now(),
            "llm_mode": self.llm.mode,
            "prompt_traces": [],
        }

        # 1) Classify
        cls = self.classifier.handle({"message": message})
        label = cls.get("label", "query")
        trace["classification"] = {
            "label": label,
            "confidence": cls.get("confidence"),
            "routed_to": cls.get("routed_to"),
        }
        trace["prompt_traces"].append(self.classifier.last_trace.get("prompt_trace", []))

        # 2) Route to the right handler
        agent = self._route[label]
        out = agent.handle({"message": message, "label": label})

        trace["agent_path"] = [self.classifier.name, agent.name]
        trace["response"] = out.get("response", "")
        trace["ticket_id"] = out.get("ticket_id")
        trace["db_actions"] = out.get("db_actions", [])
        trace["prompt_traces"].append(agent.last_trace.get("prompt_trace", []))
        trace["agent_traces"] = [self.classifier.last_trace, agent.last_trace]

        self.tracer.record_trace(trace)
        return trace


_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """Process-wide singleton used by the Streamlit UI."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
