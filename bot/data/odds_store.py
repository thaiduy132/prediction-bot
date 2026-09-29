"""Recorded odds of Predict.fun Up/Down rounds: one sample per poll, plus round metadata.

`up_bid`/`up_ask` are probabilities (0..1) of the UP outcome. Buying UP costs `up_ask`,
buying DOWN costs `1 - up_bid`; a winning share pays 1.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from bot.storage.db import connect

_SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_samples (
    symbol TEXT NOT NULL, round_start INTEGER NOT NULL, ts INTEGER NOT NULL, market_id INTEGER NOT NULL,
    up_bid REAL, up_ask REAL, bid_qty REAL NOT NULL, ask_qty REAL NOT NULL,
    bid_depth5 REAL NOT NULL, ask_depth5 REAL NOT NULL,
    PRIMARY KEY (symbol, round_start, ts)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS odds_rounds (
    symbol TEXT NOT NULL, round_start INTEGER NOT NULL, market_id INTEGER NOT NULL, slug TEXT NOT NULL,
    price_provider TEXT, start_price REAL, end_price REAL, status TEXT, resolution TEXT,
    PRIMARY KEY (symbol, round_start)
) WITHOUT ROWID;
"""


@dataclass(frozen=True, slots=True)
class OddsSample:
    round_start: int  # ms
    ts: int  # ms, when we read the book
    market_id: int
    up_bid: float | None
    up_ask: float | None
    bid_qty: float
    ask_qty: float
    bid_depth5: float
    ask_depth5: float

    @property
    def has_quote(self) -> bool:
        return self.up_bid is not None or self.up_ask is not None


class OddsStore:
    def __init__(self, path: str | Path, symbol: str) -> None:
        self.symbol = symbol.upper()
        self.conn: sqlite3.Connection = connect(path)
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def add_sample(self, s: OddsSample) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO odds_samples VALUES (?,?,?,?,?,?,?,?,?,?)",
            (self.symbol, s.round_start, s.ts, s.market_id, s.up_bid, s.up_ask, s.bid_qty, s.ask_qty,
             s.bid_depth5, s.ask_depth5))
        self.conn.commit()

    def upsert_round(self, round_start: int, market_id: int, slug: str, provider: str | None,
                     start_price: float | None, end_price: float | None, status: str | None,
                     resolution: str | None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO odds_rounds VALUES (?,?,?,?,?,?,?,?,?)",
            (self.symbol, round_start, market_id, slug, provider, start_price, end_price, status, resolution))
        self.conn.commit()

    def unresolved_rounds(self, before_ms: int) -> list[tuple[int, int]]:
        """(round_start, market_id) of rounds that ended before `before_ms` and are not RESOLVED yet."""
        rows = self.conn.execute(
            "SELECT round_start, market_id FROM odds_rounds WHERE symbol=? AND round_start<? "
            "AND COALESCE(status,'')!='RESOLVED'", (self.symbol, before_ms)).fetchall()
        return [(r[0], r[1]) for r in rows]

    def samples_for_round(self, round_start: int) -> list[OddsSample]:
        rows = self.conn.execute(
            "SELECT round_start, ts, market_id, up_bid, up_ask, bid_qty, ask_qty, bid_depth5, ask_depth5 "
            "FROM odds_samples WHERE symbol=? AND round_start=? ORDER BY ts", (self.symbol, round_start)).fetchall()
        return [OddsSample(*r) for r in rows]

    def get_range(self, start_ms: int, end_ms: int) -> list[OddsSample]:
        """Samples with ts in [start_ms, end_ms), ascending."""
        rows = self.conn.execute(
            "SELECT round_start, ts, market_id, up_bid, up_ask, bid_qty, ask_qty, bid_depth5, ask_depth5 "
            "FROM odds_samples WHERE symbol=? AND ts>=? AND ts<? ORDER BY ts", (self.symbol, start_ms, end_ms)).fetchall()
        return [OddsSample(*r) for r in rows]

    def coverage(self) -> tuple[int, int, int] | None:
        """(first ts, last ts, count) of samples that HAVE a quote, or None."""
        first, last, n = self.conn.execute(
            "SELECT MIN(ts), MAX(ts), COUNT(*) FROM odds_samples WHERE symbol=? "
            "AND (up_bid IS NOT NULL OR up_ask IS NOT NULL)", (self.symbol,)).fetchone()
        return None if not n else (first, last, n)

    def summary(self) -> dict[str, object]:
        rounds, = self.conn.execute("SELECT COUNT(*) FROM odds_rounds WHERE symbol=?", (self.symbol,)).fetchone()
        n, quoted = self.conn.execute(
            "SELECT COUNT(*), SUM(up_bid IS NOT NULL OR up_ask IS NOT NULL) FROM odds_samples WHERE symbol=?",
            (self.symbol,)).fetchone()
        return {"rounds": rounds, "samples": n, "samples_with_quote": quoted or 0}
