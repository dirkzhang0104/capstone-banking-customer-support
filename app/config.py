"""Central configuration: paths, LLM settings, and a tiny .env loader."""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = DATA_DIR / "logs"
DB_PATH = DATA_DIR / "support.db"
TRACES_PATH = LOG_DIR / "traces.jsonl"
STATS_PATH = LOG_DIR / "agent_stats.json"
USER_FEEDBACK_PATH = LOG_DIR / "user_feedback.jsonl"

for _p in (DATA_DIR, LOG_DIR):
    _p.mkdir(parents=True, exist_ok=True)


def _load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader so python-dotenv is not a hard dependency."""
    env_file = path or (BASE_DIR / ".env")
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

# ---------------------------------------------------------------------------
# LLM settings (re-read after dotenv so .env values apply)
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
