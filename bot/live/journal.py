"""Every shadow/live action is written here before and after it happens (SQLite).

The risk limits are computed from this table, so they survive restarts: a crash cannot reset
today's loss counter.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bot.storage.db import connect
from bot.timeutil import now_ms

_SCHEMA = """
CREATE TABLE IF NOT EXISTS live_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_open INTEGER NOT NULL, mode TEXT NOT NULL, side TEXT NOT NULL, token_id TEXT,
    stake_usd REAL NOT NULL, decided_price REAL,
    quote_id TEXT, quote_price REAL, quote_fee REAL, quote_json TEXT,
    order_id TEXT, order_json TEXT,
    status TEXT NOT NULL,  -- skipped | quoted | submitted | filled | failed | won | lost | void | sold
    reason TEXT, pnl_usd REAL, pnl_source TEXT,  -- pnl_source: estimate | binance
    sold_shares REAL, sold_usd REAL,  -- shares sold back before the round ended, and what they brought
    created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS live_orders_round ON live_orders(round_open);
"""

# submitted: sent, Binance has not confirmed the fill yet; filled: real position, round not settled.
# Both count as money at risk. Only Binance's order status turns "submitted" into filled/failed.
OPEN_STATUSES = ("submitted", "filled")
BOOKED_STATUSES = ("submitted", "filled", "won", "lost", "void")


@dataclass(frozen=True, slots=True)
class DayStats:
    bets: int  # submitted real orders today (any outcome)
    realized_pnl_usd: float  # booked results of today's settled orders
    open_count: int
    open_stake_usd: float


class LiveJournal:
    def __init__(self, path: str | Path, clock: Callable[[], int] = now_ms) -> None:
        self._clock = clock  # same clock as the RiskManager, so "today" means the same thing
        self.conn: sqlite3.Connection = connect(path)
        self.conn.executescript(_SCHEMA)
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(live_orders)")}
        for col, typ in (("pnl_source", "TEXT"), ("sold_shares", "REAL"), ("sold_usd", "REAL")):
            if col not in cols:  # journals created before this column existed
                self.conn.execute(f"ALTER TABLE live_orders ADD COLUMN {col} {typ}")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def add(self, round_open: int, mode: str, side: str, stake_usd: float, status: str, *,
            token_id: str | None = None, decided_price: float | None = None, reason: str | None = None) -> int:
        ts = self._clock()
        cur = self.conn.execute(
            "INSERT INTO live_orders(round_open, mode, side, token_id, stake_usd, decided_price, status, reason, "
            "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (round_open, mode, side, token_id, stake_usd, decided_price, status, reason, ts, ts))
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def update(self, row_id: int, **fields: Any) -> None:
        for k in ("quote_json", "order_json"):
            if k in fields and not isinstance(fields[k], str) and fields[k] is not None:
                fields[k] = json.dumps(fields[k], default=str)
        fields["updated_at"] = self._clock()
        cols = ", ".join(f"{k}=?" for k in fields)
        self.conn.execute(f"UPDATE live_orders SET {cols} WHERE id=?", (*fields.values(), row_id))
        self.conn.commit()

    def latest_filled(self) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM live_orders WHERE mode='live' AND status='filled' ORDER BY id DESC LIMIT 1", ())
        return rows[0] if rows else None

    def filled_for_round(self, round_open: int) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM live_orders WHERE round_open=? AND status='filled'", (round_open,))

    def to_reconcile(self) -> list[dict[str, Any]]:
        """Real orders whose fill or final PnL has not been confirmed by Binance yet."""
        return self._rows("SELECT * FROM live_orders WHERE mode='live' AND order_id IS NOT NULL AND "
                          "status IN ('submitted','filled','won','lost','void') AND COALESCE(pnl_source,'')!='binance'",
                          ())

    def day_stats(self, day_start_ms: int) -> DayStats:
        bets, pnl = self.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(pnl_usd), 0) FROM live_orders WHERE mode='live' AND created_at>=? "
            "AND status IN ('submitted','filled','won','lost','void','sold')", (day_start_ms,)).fetchone()
        n_open, stake_open = self.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(stake_usd), 0) FROM live_orders WHERE mode='live' "
            "AND status IN ('submitted','filled')").fetchone()
        return DayStats(int(bets), float(pnl), int(n_open), float(stake_open))

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._rows("SELECT id, round_open, mode, side, stake_usd, decided_price, quote_price, quote_fee, "
                          "order_id, status, reason, pnl_usd, pnl_source, sold_shares, sold_usd, created_at FROM live_orders ORDER BY id DESC LIMIT ?",
                          (limit,))

    def row(self, row_id: int) -> dict[str, Any]:
        return self._rows("SELECT * FROM live_orders WHERE id=?", (row_id,))[0]

    def _rows(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        cur = self.conn.execute(sql, params)
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r, strict=True)) for r in cur.fetchall()]
