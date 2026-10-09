"""Model evaluation (Part 2, LLMOps).

Assesses:
  1. Classification quality — accuracy + per-class precision/recall/F1 over
     the labeled test-case set (QA-based scoring with test-case coverage).
  2. Agent routing success rate — does the orchestrator send each message to
     the correct downstream agent?
  3. Generated-response quality — QA heuristics per agent role:
       positive  -> warmth, clarity, personalization
       negative  -> empathy, ticket reference, follow-up clarity
       query     -> status presence, ticket reference, clarity
"""
from __future__ import annotations

import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..agents.classifier import ROUTE_TARGET
from ..db.database import SupportDatabase
from ..llm.client import LLMClient
from ..logs.trace import TraceLogger
from ..orchestrator import Orchestrator
from .test_cases import TEST_CASES, QUERY_TICKET_IDS

EMPATHY_WORDS = {"sorry", "apologize", "apology", "understand", "frustrating",
                 "patience", "regret", "unfortunate", "upsetting"}
WARM_WORDS = {"thank", "thanks", "glad", "happy", "great", "wonderful",
              "delighted", "appreciate", "love", "delight", "fantastic"}
FOLLOWUP_PHRASES = ("follow up", "follow-up", "our team")
STATUS_WORDS = {"resolved", "unresolved", "open", "progress", "working"}


def _words(text: str) -> set:
    return set(re.findall(r"[a-z']+", text.lower()))


def score_response_qa(kind: str, response: str,
                      ticket_id: Optional[str] = None) -> dict:
    """QA-based scoring of a generated agent response (each metric in 0..1)."""
    words = _words(response)
    n_words = len(response.split())
    scores: dict[str, float] = {}

    if kind == "positive_feedback":
        scores["warmth"] = min(1.0, len(words & WARM_WORDS) / 2)
        scores["clarity"] = 1.0 if 15 <= n_words <= 100 else 0.5
        scores["personalization"] = 1.0 if "you" in words else 0.0
    elif kind == "negative_feedback":
        scores["empathy"] = min(1.0, len(words & EMPATHY_WORDS) / 2)
        scores["ticket_reference"] = 1.0 if (ticket_id and ticket_id in response) else 0.0
        scores["follow_up_clarity"] = 1.0 if any(p in response.lower() for p in FOLLOWUP_PHRASES) else 0.0
    else:  # query
        scores["status_present"] = 1.0 if bool(words & STATUS_WORDS) else 0.0
        scores["ticket_reference"] = 1.0 if (ticket_id and ticket_id in response) else 0.0
        scores["clarity"] = 1.0 if 15 <= n_words <= 120 else 0.5

    scores["overall"] = round(sum(scores.values()) / len(scores), 3)
    return scores


def evaluate_classification(llm: LLMClient) -> dict:
    """Accuracy + per-class precision/recall/F1 over the labeled test cases."""
    tp: dict[str, int] = {}
    fp: dict[str, int] = {}
    fn: dict[str, int] = {}
    misclassified: list[dict] = []
    correct = 0
    labels = ("positive_feedback", "negative_feedback", "query")
    for l in labels:
        tp[l] = fp[l] = fn[l] = 0

    for case in TEST_CASES:
        pred = llm.classify(case.text)
        if pred.label == case.expected:
            correct += 1
            tp[case.expected] += 1
        else:
            fn[case.expected] += 1
            fp[pred.label] += 1
            misclassified.append({
                "text": case.text, "expected": case.expected, "predicted": pred.label,
            })

    per_class = {}
    for l in labels:
        precision = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) else 0.0
        recall = tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[l] = {
            "precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "support": tp[l] + fn[l],
        }
    return {
        "total": len(TEST_CASES),
        "correct": correct,
        "accuracy": round(correct / len(TEST_CASES), 3),
        "per_class": per_class,
        "misclassified": misclassified,
    }


def evaluate_routing(orchestrator: Orchestrator) -> dict:
    """Share of messages routed to the correct downstream agent."""
    successes = 0
    details: list[dict] = []
    for case in TEST_CASES:
        trace = orchestrator.handle_message(case.text)
        expected_agent = ROUTE_TARGET[case.expected]
        ok = trace["agent_path"][-1] == expected_agent
        successes += int(ok)
        details.append({
            "text": case.text, "expected_agent": expected_agent,
            "routed_to": trace["agent_path"][-1], "ok": ok,
        })
    return {
        "total": len(TEST_CASES),
        "successes": successes,
        "success_rate": round(successes / len(TEST_CASES), 3),
        "details": details,
    }


def evaluate_responses(orchestrator: Orchestrator) -> dict:
    """QA-based scoring of the responses generated across all agent roles."""
    by_kind: dict[str, list[dict]] = {}
    for case in TEST_CASES:
        trace = orchestrator.handle_message(case.text)
        kind = case.expected
        ticket_id = trace.get("ticket_id")
        # For query cases the ticket id comes from the message, not the trace.
        if kind == "query" and not ticket_id:
            m = re.search(r"\b(\d{6})\b", case.text)
            ticket_id = m.group(1) if m else None
        by_kind.setdefault(kind, []).append(
            score_response_qa(kind, trace["response"], ticket_id)
        )

    summary = {}
    for kind, rows in by_kind.items():
        metrics = [k for k in rows[0] if k != "overall"]
        summary[kind] = {
            "n": len(rows),
            **{m: round(sum(r[m] for r in rows) / len(rows), 3) for m in metrics},
            "overall": round(sum(r["overall"] for r in rows) / len(rows), 3),
        }
    return summary


def _seed_eval_db(db: SupportDatabase) -> None:
    """Seed the eval DB with every ticket id referenced by the query cases."""
    for i, tid in enumerate(QUERY_TICKET_IDS):
        status = "resolved" if i % 3 == 0 else "unresolved"
        db.create_ticket_with_id(tid, f"Evaluation seed for ticket {tid}",
                                 sentiment="negative", status=status)


def run_full_evaluation(db: Optional[SupportDatabase] = None,
                        llm: Optional[LLMClient] = None) -> dict:
    """Run the whole evaluation suite on an isolated (temporary) database.

    A fresh temp DB is seeded with the ticket ids referenced by the query
    test cases so status lookups succeed; the production database is never
    touched by evaluation.
    """
    llm = llm if llm is not None else LLMClient()
    if db is None:
        tmp = Path(tempfile.mkdtemp(prefix="capstone_eval_"))
        db = SupportDatabase(tmp / "eval.db")
        _seed_eval_db(db)

    tmp = Path(tempfile.mkdtemp(prefix="capstone_eval_"))
    tracer = TraceLogger(tmp / "traces.jsonl", tmp / "stats.json")
    orchestrator = Orchestrator(llm=llm, db=db, tracer=tracer)

    classification = evaluate_classification(llm)
    routing = evaluate_routing(orchestrator)
    responses = evaluate_responses(orchestrator)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "llm_mode": llm.mode,
        "classification": classification,
        "routing": routing,
        "response_quality": responses,
    }


def render_report_markdown(report: dict) -> str:
    """Human-readable markdown rendering of an evaluation report."""
    c = report["classification"]
    r = report["routing"]
    lines = [
        "# Evaluation Report",
        f"- **Generated:** {report['generated_at']}  ",
        f"- **LLM mode:** {report['llm_mode']}  ",
        f"- **Classification accuracy:** {c['accuracy']:.1%} ({c['correct']}/{c['total']} test cases)  ",
        f"- **Routing success rate:** {r['success_rate']:.1%} ({r['successes']}/{r['total']})",
        "",
        "## Per-class classification",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for label, m in c["per_class"].items():
        lines.append(f"| {label} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['support']} |")

    lines += [
        "",
        "## Response quality (QA-based scoring, 0-1 scale)",
    ]
    for kind, m in report["response_quality"].items():
        metrics = [k for k in m if k != "n"]
        lines.append(f"### {kind} (n={m['n']})")
        lines.append("| " + " | ".join(metrics) + " |")
        lines.append("| " + " | ".join("---" for _ in metrics) + " |")
        lines.append("| " + " | ".join(f"{m[k]:.2f}" for k in metrics) + " |")

    if c["misclassified"]:
        lines += ["", "## Misclassified cases"]
        lines += [
            f"- \"{m['text']}\" — expected {m['expected']}, got {m['predicted']}"
            for m in c["misclassified"]
        ]
    return "\n".join(lines)
