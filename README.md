# 🏦 Banking Customer Support AI Agent — Multi-Agent Architecture

A **multi-agent GenAI system** for banking customer support workflows. Incoming
customer messages are classified, routed to the right specialist agent, and
answered with personalized responses, automatic ticket creation, and live
ticket-status lookups — all with LLMOps tooling (evaluation, tracing, logging)
and an interactive Streamlit dashboard.

## Capstone requirements → implementation

| Requirement (PDF) | Where it lives |
|---|---|
| **Part 1 · 1. Classifier Agent** — categorize into positive feedback / negative feedback / query and route | `app/agents/classifier.py`, `app/orchestrator.py` |
| **Part 1 · 2. Feedback Handler Agent** — warm personalized thank-you (positive); unique 6-digit ticket + insert unresolved `support_tickets` row + empathetic message with ticket number (negative) | `app/agents/feedback_handler.py` |
| **Part 1 · 3. Query Handler Agent** — extract ticket number, query `support_tickets`, return status | `app/agents/query_handler.py` |
| **Part 1 · 4. Sample use-case flows** (3 coordination examples) | `tests/test_orchestrator.py` + sample buttons in the UI |
| **Part 2 · 7. Model evaluation** — QA-based scoring, test-case coverage for classification, routing success rate | `app/eval/evaluator.py`, `app/eval/test_cases.py` (30 labeled cases) |
| **Part 2 · 8. Streamlit UI** — input + simulated routing, classification/response/DB display, history & logs, per-role scenario testing | `app/main.py` (4 tabs) |
| **Part 2 · 9. Logs & debugging view** — prompt traces, classification outputs, ticket actions, agent success/failure rates, user feedback loop | `app/logs/trace.py` + UI tabs; user 👍/👎 recorded to `data/logs/user_feedback.jsonl` |

## Architecture

```
                 ┌────────────────────────┐
 Customer msg ──▶│   Classifier Agent     │  positive / negative / query
                 └───────────┬────────────┘
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ Feedback Handler│   │ Feedback Handler│   │  Query Handler  │
│  (positive)     │   │   (negative)    │   │  (status)       │
│ warm thank-you  │   │ 6-digit ticket  │   │ extract 6-digit │
│                 │   │ INSERT unresolved│   │ SELECT status   │
└─────────────────┘   └────────┬────────┘   └────────┬────────┘
                               ▼                     ▼
                     ┌──────────────────────────────────────┐
                     │  SQLite  support_tickets             │
                     │  (ticket_id, message, sentiment,     │
                     │   status, created/updated/resolved)  │
                     └──────────────────────────────────────┘
        Everything is traced (JSONL) + agent success/failure counters  ──▶  UI & evaluation
```

**LLM strategy:** the `LLMClient` uses the OpenAI API when `OPENAI_API_KEY`
is set (model configurable via `LLM_MODEL`). Without a key it transparently
falls back to a deterministic **heuristic engine** (lexicon-based
classification + templated generation), so the entire system — pipeline,
evaluation, tests, UI — runs fully offline.

## Quickstart

```bash
./run.sh                 # creates .venv, installs deps, starts Streamlit on :8501
```

or manually:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app/main.py
.venv/bin/pytest -q      # run the test suite
```

Optional — enable a live LLM:

```bash
cp .env.example .env     # then set OPENAI_API_KEY (and LLM_MODEL if desired)
```

## UI tour

* **💬 Chat & Agent Routing** — send messages (or click the three sample
  scenario buttons); each reply shows classification + confidence, the agent
  path, ticket ID, the generated response, database interactions, and a
  👍/👎 feedback button.
* **🗄 Support Tickets** — live view of the `support_tickets` table with
  unresolved/resolved metrics.
* **📜 Logs & Debug** — per-agent success/failure rates and full prompt
  traces, classification outputs and ticket actions per message.
* **📊 Evaluation** — one click runs the full LLMOps suite on an isolated
  temporary database and renders the report.

## Project layout

```
app/
├── main.py               # Streamlit UI (4 tabs)
├── config.py             # paths, LLM settings, .env loader
├── orchestrator.py       # Classifier → handler routing + full trace
├── agents/
│   ├── base.py           # tracing, timing, failure handling
│   ├── classifier.py
│   ├── feedback_handler.py
│   └── query_handler.py
├── llm/client.py         # OpenAI facade + heuristic fallback engine
├── db/database.py        # SQLite support_tickets store
├── logs/trace.py         # JSONL traces + agent success/failure stats
└── eval/
    ├── test_cases.py     # 30 labeled classification cases
    └── evaluator.py      # accuracy, P/R/F1, routing rate, QA scoring
tests/                    # pytest suite (runs offline, temp DBs)
data/                     # runtime artifacts (gitignored): support.db, logs/
```

## Tests

```bash
.venv/bin/pytest -q
```

Covers: all 30 classification cases, ticket creation/uniqueness (6-digit),
graceful handling of missing/unknown ticket numbers, the three spec use-case
flows end-to-end, trace persistence, agent success/failure stats, and the
evaluation suite (accuracy ≥ 80%, routing ≥ 80%, QA score bounds).
