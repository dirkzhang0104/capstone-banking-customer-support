"""Streamlit UI for the multi-agent banking customer support system.

Run from the repo root:
    streamlit run app/main.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the repo root importable no matter how the script is launched.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.db.database import SupportDatabase  # noqa: E402
from app.eval.evaluator import render_report_markdown, run_full_evaluation  # noqa: E402
from app.logs.trace import TraceLogger  # noqa: E402
from app.orchestrator import get_orchestrator  # noqa: E402

st.set_page_config(
    page_title="Banking Support — Multi-Agent AI",
    page_icon="🏦",
    layout="wide",
)

LABEL_STYLE = {
    "positive_feedback": ("👍 Positive Feedback", "green"),
    "negative_feedback": ("😞 Negative Feedback", "orange"),
    "query": ("🔎 Query", "blue"),
}

SAMPLE_MESSAGES = [
    "I love the new mobile app, it's so fast and helpful!",
    "I'm so frustrated with the hidden fees on my account!",
    "What is the status of my ticket 482913?",
]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "input_text" not in st.session_state:
    st.session_state.input_text = ""


def render_trace_card(trace: dict, idx: int) -> None:
    """Render one full pipeline trace as a chat bubble + agent card."""
    label = trace.get("classification", {}).get("label", "query")
    label_text, color = LABEL_STYLE.get(label, (label, "gray"))

    st.markdown("#### 👤 Customer\n> " + trace["message"])

    with st.container(border=True):
        col1, col2, col3 = st.columns([1.2, 1.2, 1])
        col1.markdown(
            f"**Classification**\n:#[{color}]{label_text}[/]\n"
            f"confidence {trace.get('classification', {}).get('confidence', 0):.2f}"
        )
        path = "  →  ".join(trace.get("agent_path", []))
        col2.markdown(f"**Agent path**\n{path}")
        if trace.get("ticket_id"):
            col3.markdown(f"**Ticket ID**\n🎫 {trace['ticket_id']}")
        else:
            col3.markdown("**Ticket ID**\n—")
        st.markdown(f"**🤖 Agent response**\n\n{trace.get('response', '')}")
        if trace.get("db_actions"):
            with st.expander("🗄 Database interactions"):
                for action in trace["db_actions"]:
                    st.code(action, language="sql")

        c1, c2, c3 = st.columns(3)
        helpful = c1.button("👍 Helpful", key=f"fb_{trace['trace_id']}_up")
        c2.button("👎 Not helpful", key=f"fb_{trace['trace_id']}_down")
        if helpful:
            get_orchestrator().tracer.record_user_feedback(
                trace["trace_id"], True, "ui_thumbs_up")
            st.session_state.history[idx]["user_helpful"] = True
            st.success("Feedback recorded — thanks for helping us improve!")
    st.divider()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
orchestrator = get_orchestrator()
tracer = orchestrator.tracer

with st.sidebar:
    st.title("🏦 Multi-Agent Support")
    st.caption("Banking customer support AI — classifier, feedback handler, query handler")
    mode = "🟢 Live LLM" if orchestrator.llm.is_live else "🔵 Heuristic engine (offline)"
    st.markdown(f"**LLM mode:** {mode}")
    st.caption(f"model: {orchestrator.llm.model}")
    st.divider()
    if st.button("Seed demo tickets", width="stretch"):
        n = orchestrator.db.seed_demo()
        st.session_state.seed_msg = f"Seeded {n} demo tickets."
        st.rerun()
    if getattr(st.session_state, "seed_msg", None):
        st.success(st.session_state.seed_msg)
    st.divider()
    if st.button("Clear chat history", width="stretch"):
        st.session_state.history = []
        st.rerun()
    st.caption(f"Database: {orchestrator.db.path.name} ({orchestrator.db.count()} tickets)")

st.title("🏦 Banking Customer Support — Multi-Agent AI")
st.caption(
    "Every message flows through the **Classifier Agent**, which routes it to the "
    "**Feedback Handler Agent** (positive/negative) or the **Query Handler Agent** "
    "(ticket status). All prompts, actions and results are traced."
)

tabs = st.tabs(["💬 Chat & Agent Routing", "🗄 Support Tickets", "📜 Logs & Debug", "📊 Evaluation"])

# ---------------------------------------------------------------------------
# Tab 1 — Chat
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Agent routing simulation")

    st.markdown("**Try a sample scenario:**")
    s1, s2, s3 = st.columns(3)
    if s1.button("😊 Positive feedback"):
        st.session_state.input_text = SAMPLE_MESSAGES[0]
    if s2.button("😠 Negative feedback"):
        st.session_state.input_text = SAMPLE_MESSAGES[1]
    if s3.button("🔎 Ticket status query"):
        st.session_state.input_text = SAMPLE_MESSAGES[2]

    if st.session_state.input_text.strip():
        if st.button("Send to agents ➡️", type="primary"):
            trace = orchestrator.handle_message(st.session_state.input_text)
            st.session_state.history.append(trace)
            st.session_state.input_text = ""
            st.rerun()

    st.info("Type a customer message below, or pick a sample above.")
    typed = st.chat_input(
        "e.g. 'Any update on ticket 555666?' or 'Terrible service today!'"
    )
    if typed:
        trace = orchestrator.handle_message(typed)
        st.session_state.history.append(trace)
        st.session_state.input_text = ""
        st.rerun()

    st.subheader(f"Conversation history ({len(st.session_state.history)})")
    for i, trace in enumerate(st.session_state.history):
        render_trace_card(trace, i)

# ---------------------------------------------------------------------------
# Tab 2 — Database
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("support_tickets table")
    tickets = orchestrator.db.list_tickets(limit=200)
    if tickets:
        df = pd.DataFrame(tickets)
        st.dataframe(
            df[["ticket_id", "status", "sentiment", "message", "created_at", "updated_at"]],
            width="stretch", hide_index=True,
            column_config={
                "ticket_id": "Ticket ID", "status": "Status",
                "sentiment": "Sentiment", "message": "Customer message",
                "created_at": "Created (UTC)", "updated_at": "Updated (UTC)",
            },
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Total tickets", len(tickets))
        c2.metric("Unresolved", sum(1 for t in tickets if t["status"] == "unresolved"))
        c3.metric("Resolved", sum(1 for t in tickets if t["status"] == "resolved"))
    else:
        st.info("No tickets yet. Send negative feedback in the chat tab or click 'Seed demo tickets' in the sidebar.")

# ---------------------------------------------------------------------------
# Tab 3 — Logs & Debug
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Agent success / failure rates")
    stats = tracer.agent_stats()
    if stats:
        cols = st.columns(min(len(stats), 4))
        for i, (name, e) in enumerate(stats.items()):
            cols[i % 4].metric(
                name, f"{e['success_rate']:.0%}",
                delta=f"{e['success']} ok / {e['failure']} failed", delta_color="normal",
            )
    else:
        st.info("No agent runs recorded yet.")

    st.subheader("Prompt traces, classification outputs & ticket actions")
    traces = tracer.load_traces(limit=50)
    if not traces:
        st.info("No traces recorded yet. Send a message in the Chat tab.")
    for trace in reversed(traces):
        label = trace.get("classification", {}).get("label", "?")
        msg_preview = trace["message"][:60]
        with st.expander(
            f"{trace.get('ts', '')}  {label}  →  {' / '.join(trace.get('agent_path', []))}  —  {msg_preview}"
        ):
            st.json({
                "classification": trace.get("classification"),
                "ticket_id": trace.get("ticket_id"),
                "db_actions": trace.get("db_actions"),
                "agent_traces": trace.get("agent_traces"),
                "prompt_traces": trace.get("prompt_traces"),
            })

# ---------------------------------------------------------------------------
# Tab 4 — Evaluation
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Model evaluation (LLMOps)")
    st.caption(
        "Runs the full suite on an isolated temporary database: classification "
        "accuracy over 30 labeled test cases, agent routing success rate, and "
        "QA-based response quality scoring."
    )
    if st.button("Run full evaluation ▶️", type="primary"):
        with st.spinner("Running evaluation suite..."):
            report = run_full_evaluation()
        st.session_state.report = report
    report = st.session_state.get("report")
    if report:
        st.markdown(render_report_markdown(report))
        st.expander("Raw report (JSON)")
        st.json(report)
    else:
        st.info("Click 'Run full evaluation' to generate the latest report.")
