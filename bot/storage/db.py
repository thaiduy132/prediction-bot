"""SQLite helpers shared by the bot and (later) the read-only dashboard.

WAL mode lets a reader (dashboard, report) read while the bot writes.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from bot.timeutil import now_ms

_STATUS_SCHEMA = """
CREATE TABLE IF NOT EXISTS component_status (
    component   TEXT PRIMARY KEY,   -- e.g. 'ws', 'rest', 'jev'
    state       TEXT NOT NULL,      -- e.g. 'connected', 'reconnecting', 'error'
    updated_at  INTEGER NOT NULL,   -- epoch ms
    details     TEXT                -- JSON
);
"""


def connect(path: str | Path, read_only: bool = False) -> sqlite3.Connection:
    path = Path(path)
    if read_only:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_STATUS_SCHEMA)
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def write_status(conn: sqlite3.Connection, component: str, state: str, ts_ms: int | None = None, **details: Any) -> None:
    conn.execute(
        "INSERT INTO component_status(component, state, updated_at, details) VALUES (?,?,?,?) "
        "ON CONFLICT(component) DO UPDATE SET state=excluded.state, updated_at=excluded.updated_at, details=excluded.details",
        (component, state, ts_ms if ts_ms is not None else now_ms(), json.dumps(details, default=str)),
    )
    conn.commit()


def read_status(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute("SELECT component, state, updated_at, details FROM component_status").fetchall()
    return {c: {"state": s, "updated_at": u, "details": json.loads(d or "{}")} for c, s, u, d in rows}
