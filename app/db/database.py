"""SQLite-backed support ticket store (table: support_tickets)."""
from __future__ import annotations

import random
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .. import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS support_tickets (
    ticket_id   TEXT PRIMARY KEY,          -- unique 6-digit ticket number
    message     TEXT NOT NULL,             -- original customer message
    sentiment   TEXT NOT NULL,             -- positive | negative
    status      TEXT NOT NULL DEFAULT 'unresolved',  -- unresolved | resolved
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    resolved_at TEXT
);
"""

DEMO_ROWS = [
    ("482913", "My card was declined three times at the supermarket.", "negative", "unresolved"),
    ("111222", "Hidden fees on my savings account, please explain.", "negative", "resolved"),
    ("998877", "Duplicate charge for a streaming subscription.", "negative", "unresolved"),
    ("555666", "Question about my loan EMI schedule.", "negative", "resolved"),
    ("123456", "ATM swallowed my card this morning.", "negative", "unresolved"),
    ("234567", "App keeps crashing when I open the transfers tab.", "negative", "resolved"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SupportDatabase:
    """Thread-safe wrapper around the support_tickets SQLite table."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else config.DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute(SCHEMA)
            self._conn.commit()

    # -- tickets -------------------------------------------------------------
    def create_ticket(self, message: str, sentiment: str = "negative") -> str:
        """Insert a new *unresolved* ticket with a unique 6-digit id."""
        with self._lock:
            while True:
                ticket_id = str(random.randint(100000, 999999))
                exists = self._conn.execute(
                    "SELECT 1 FROM support_tickets WHERE ticket_id = ?", (ticket_id,)
                ).fetchone()
                if exists is None:
                    break
            now = _now()
            self._conn.execute(
                "INSERT INTO support_tickets "
                "(ticket_id, message, sentiment, status, created_at, updated_at, resolved_at) "
                "VALUES (?, ?, ?, 'unresolved', ?, ?, NULL)",
                (ticket_id, message, sentiment, now, now),
            )
            self._conn.commit()
            return ticket_id

    def create_ticket_with_id(self, ticket_id: str, message: str,
                               sentiment: str = "negative",
                               status: str = "unresolved") -> str:
        """Insert a ticket with an explicit id (used by demo/eval seeding)."""
        if status not in ("unresolved", "resolved"):
            raise ValueError(f"invalid status: {status}")
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM support_tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
            if exists:
                return ticket_id
            now = _now()
            resolved_at = now if status == "resolved" else None
            self._conn.execute(
                "INSERT INTO support_tickets "
                "(ticket_id, message, sentiment, status, created_at, updated_at, resolved_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ticket_id, message, sentiment, status, now, now, resolved_at),
            )
            self._conn.commit()
            return ticket_id

    def get_ticket(self, ticket_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM support_tickets WHERE ticket_id = ?", (str(ticket_id),)
            ).fetchone()
        return dict(row) if row else None

    def update_status(self, ticket_id: str, status: str) -> bool:
        if status not in ("unresolved", "resolved"):
            raise ValueError(f"invalid status: {status}")
        with self._lock:
            resolved_at = _now() if status == "resolved" else None
            cur = self._conn.execute(
                "UPDATE support_tickets SET status = ?, updated_at = ?, resolved_at = ? "
                "WHERE ticket_id = ?",
                (status, _now(), resolved_at, str(ticket_id)),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def list_tickets(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM support_tickets ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM support_tickets").fetchone()[0]

    def seed_demo(self) -> int:
        """Insert sample tickets (idempotent) so the UI has data to show."""
        inserted = 0
        with self._lock:
            for ticket_id, message, sentiment, status in DEMO_ROWS:
                exists = self._conn.execute(
                    "SELECT 1 FROM support_tickets WHERE ticket_id = ?", (ticket_id,)
                ).fetchone()
                if exists:
                    continue
                now = _now()
                resolved_at = now if status == "resolved" else None
                self._conn.execute(
                    "INSERT INTO support_tickets "
                    "(ticket_id, message, sentiment, status, created_at, updated_at, resolved_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (ticket_id, message, sentiment, status, now, now, resolved_at),
                )
                inserted += 1
            self._conn.commit()
        return inserted

    def close(self) -> None:
        with self._lock:
            self._conn.close()
