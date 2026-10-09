"""Base agent: shared tracing, timing, and failure handling."""
from __future__ import annotations

import time
from typing import Any

from ..db.database import SupportDatabase
from ..llm.client import LLMClient
from ..logs.trace import TraceLogger


class BaseAgent:
    name = "Base Agent"

    def __init__(self, llm: LLMClient, db: SupportDatabase, tracer: TraceLogger):
        self.llm = llm
        self.db = db
        self.tracer = tracer
        self.last_trace: dict = {}

    def handle(self, payload: dict) -> dict:
        """Run the agent, record success/failure, and attach its trace."""
        started = time.time()
        success = False
        try:
            result = self._handle(payload)
            success = True
        except Exception as exc:  # keep the pipeline alive; surface a polite fallback
            result = {
                "response": (
                    "I'm sorry, something went wrong while processing your request. "
                    "Our team has been notified and will get back to you shortly."
                ),
                "error": f"{type(exc).__name__}: {exc}",
            }
        duration_ms = round((time.time() - started) * 1000, 1)
        self.tracer.record_agent(
            self.name, success, result.get("action") or result.get("error") or ""
        )
        self.last_trace = {
            "agent": self.name,
            "success": success,
            "duration_ms": duration_ms,
            "prompt_trace": list(getattr(self.llm, "last_prompt_trace", [])),
        }
        result.setdefault("agent_trace", self.last_trace)
        return result

    def _handle(self, payload: dict) -> dict:
        raise NotImplementedError
