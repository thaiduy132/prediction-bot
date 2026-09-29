"""Paper-trading history in SQLite, so the dashboard keeps its trades and PnL across restarts."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from bot.storage.db import connect

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_trades (
    label TEXT NOT NULL, round_open INTEGER NOT NULL, side TEXT NOT NULL, entry_price REAL,
    move_bps REAL NOT NULL, opened_at INTEGER NOT NULL, outcome TEXT NOT NULL, pnl REAL NOT NULL,
    settled_at INTEGER NOT NULL,
    PRIMARY KEY (label, round_open)
) WITHOUT ROWID;
"""


@dataclass(frozen=True, slots=True)
class PaperTradeRow:
    round_open: int
    side: str
    entry_price: float | None
    move_bps: float
    opened_at: int
    outcome: str
    pnl: float
    settled_at: int


class PaperStore:
    def __init__(self, path: str | Path) -> None:
        self.conn: sqlite3.Connection = connect(path)
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def save(self, label: str, t: PaperTradeRow) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO paper_trades VALUES (?,?,?,?,?,?,?,?,?)",
            (label, t.round_open, t.side, t.entry_price, t.move_bps, t.opened_at, t.outcome, t.pnl, t.settled_at))
        self.conn.commit()

    def load(self, label: str) -> list[PaperTradeRow]:
        rows = self.conn.execute(
            "SELECT round_open, side, entry_price, move_bps, opened_at, outcome, pnl, settled_at "
            "FROM paper_trades WHERE label=? ORDER BY round_open", (label,)).fetchall()
        return [PaperTradeRow(*r) for r in rows]
