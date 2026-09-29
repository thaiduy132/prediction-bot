"""Order-book samples: one row per second, recorded live, replayed by the backtester.

Binance offers no historical spot order book, so this data only exists from the moment
we start recording. Each row is the LAST book state seen during second `ts`
(so it is only knowable at ts + 1000; consumers must respect that to avoid look-ahead).
Depth is stored as cumulative quantity over the top 1/5/10/20 levels, not raw levels.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from bot.data.binance_ws import DepthEvent, StreamEvent
from bot.models import DepthSnapshot
from bot.storage.db import connect

LEVELS = (1, 5, 10, 20)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS book_samples (
    symbol TEXT NOT NULL, ts INTEGER NOT NULL,
    bid REAL NOT NULL, ask REAL NOT NULL,
    bq1 REAL NOT NULL, aq1 REAL NOT NULL, bq5 REAL NOT NULL, aq5 REAL NOT NULL,
    bq10 REAL NOT NULL, aq10 REAL NOT NULL, bq20 REAL NOT NULL, aq20 REAL NOT NULL,
    PRIMARY KEY (symbol, ts)
) WITHOUT ROWID;
"""
_COLS = "symbol, ts, bid, ask, bq1, aq1, bq5, aq5, bq10, aq10, bq20, aq20"


@dataclass(frozen=True, slots=True)
class BookSample:
    ts: int  # ms, start of the second this sample summarises
    bid: float
    ask: float
    bq1: float
    aq1: float
    bq5: float
    aq5: float
    bq10: float
    aq10: float
    bq20: float
    aq20: float

    def depth(self, levels: int) -> tuple[float, float]:
        """(bid qty, ask qty) summed over the top `levels` levels."""
        if levels not in LEVELS:
            raise ValueError(f"levels must be one of {LEVELS}")
        return getattr(self, f"bq{levels}"), getattr(self, f"aq{levels}")


def sample_from_depth(d: DepthSnapshot, ts: int) -> BookSample | None:
    if not d.bids or not d.asks:
        return None

    def cum(side: tuple[tuple[float, float], ...], n: int) -> float:
        return sum(q for _, q in side[:n])

    return BookSample(ts, d.bids[0][0], d.asks[0][0],
                      *(cum(side, n) for n in LEVELS for side in (d.bids, d.asks)))


class BookSampler:
    """Reduces a fast depth stream to one sample per second (the last state of that second)."""

    def __init__(self) -> None:
        self._sec: int | None = None
        self._latest: DepthSnapshot | None = None

    def offer(self, d: DepthSnapshot) -> BookSample | None:
        """Feed a snapshot; returns the finished sample of the previous second when a new one starts."""
        sec = d.recv_time // 1000 * 1000
        out = None
        if self._sec is not None and sec > self._sec and self._latest is not None:
            out = sample_from_depth(self._latest, self._sec)
        if self._sec is None or sec >= self._sec:
            self._sec, self._latest = sec, d
        return out


class BookStore:
    FLUSH_EVERY = 10

    def __init__(self, path: str | Path, symbol: str) -> None:
        self.symbol = symbol.upper()
        self.conn: sqlite3.Connection = connect(path)
        self.conn.executescript(_SCHEMA)
        self._buf: list[tuple[object, ...]] = []

    def add(self, s: BookSample) -> None:
        self._buf.append((self.symbol, s.ts, s.bid, s.ask, s.bq1, s.aq1, s.bq5, s.aq5, s.bq10, s.aq10, s.bq20, s.aq20))
        if len(self._buf) >= self.FLUSH_EVERY:
            self.flush()

    def flush(self) -> None:
        if self._buf:
            self.conn.executemany(f"INSERT OR REPLACE INTO book_samples({_COLS}) VALUES ({','.join('?' * 12)})", self._buf)
            self.conn.commit()
            self._buf.clear()

    def close(self) -> None:
        self.flush()
        self.conn.close()

    def get_range(self, start_ms: int, end_ms: int) -> list[BookSample]:
        """Samples with ts in [start_ms, end_ms), ascending."""
        rows = self.conn.execute(
            f"SELECT {_COLS} FROM book_samples WHERE symbol=? AND ts>=? AND ts<? ORDER BY ts",
            (self.symbol, start_ms, end_ms)).fetchall()
        return [BookSample(*r[1:]) for r in rows]

    def coverage(self) -> tuple[int, int, int] | None:
        """(first ts, last ts, count) or None when nothing is recorded."""
        first, last, n = self.conn.execute(
            "SELECT MIN(ts), MAX(ts), COUNT(*) FROM book_samples WHERE symbol=?", (self.symbol,)).fetchone()
        return None if not n else (first, last, n)


class BookRecorder:
    """Glue: give it stream events, it stores one book sample per second."""

    def __init__(self, store: BookStore | None, on_sample: Callable[[BookSample], None] | None = None) -> None:
        self.store = store  # None: only forward samples (e.g. to the paper trader), do not persist
        self.on_sample = on_sample
        self.sampler = BookSampler()
        self.recorded = 0

    def handle(self, ev: StreamEvent) -> None:
        if isinstance(ev, DepthEvent):
            sample = self.sampler.offer(ev.depth)
            if sample is not None:
                if self.store is not None:
                    self.store.add(sample)
                if self.on_sample is not None:
                    self.on_sample(sample)
                self.recorded += 1
