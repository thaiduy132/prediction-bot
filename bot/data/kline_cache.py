"""On-disk cache of closed klines (SQLite) + loader that only fetches what is missing.

Only CLOSED candles are stored, so the cache never contains a value that could
still change. Candles that Binance itself never produced (exchange outages) are
remembered in `kline_gaps` so we do not re-request them on every run.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from bot.data.binance_rest import BinanceRestClient
from bot.logging_setup import log_event
from bot.models import Kline
from bot.storage.db import connect
from bot.timeutil import ceil_to, floor_to, interval_ms, ms_to_iso, now_ms

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS klines (
    symbol TEXT NOT NULL, interval TEXT NOT NULL, open_time INTEGER NOT NULL,
    close_time INTEGER NOT NULL, open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
    close REAL NOT NULL, volume REAL NOT NULL, quote_volume REAL NOT NULL, trades INTEGER NOT NULL,
    taker_buy_base REAL NOT NULL, taker_buy_quote REAL NOT NULL,
    PRIMARY KEY (symbol, interval, open_time)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS kline_gaps (
    symbol TEXT NOT NULL, interval TEXT NOT NULL, open_time INTEGER NOT NULL,
    PRIMARY KEY (symbol, interval, open_time)
) WITHOUT ROWID;
"""

GAP_GRACE_MS = 10 * 60_000

_COLS = "symbol, interval, open_time, close_time, open, high, low, close, volume, quote_volume, trades, taker_buy_base, taker_buy_quote"


class KlineCache:
    def __init__(self, path: str | Path) -> None:
        self.conn: sqlite3.Connection = connect(path)
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def upsert(self, klines: list[Kline]) -> int:
        rows = [
            (k.symbol, k.interval, k.open_time, k.close_time, k.open, k.high, k.low, k.close, k.volume,
             k.quote_volume, k.trades, k.taker_buy_base, k.taker_buy_quote)
            for k in klines if k.is_closed
        ]
        self.conn.executemany(f"INSERT OR REPLACE INTO klines({_COLS}) VALUES ({','.join('?' * 13)})", rows)
        self.conn.commit()
        return len(rows)

    def get_range(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> list[Kline]:
        """Closed klines with open_time in [start_ms, end_ms), ascending."""
        cur = self.conn.execute(
            f"SELECT {_COLS} FROM klines WHERE symbol=? AND interval=? AND open_time>=? AND open_time<? ORDER BY open_time",
            (symbol.upper(), interval, start_ms, end_ms),
        )
        return [Kline(*row, is_closed=True) for row in cur]

    def missing_open_times(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> list[int]:
        """Expected open_times in [start, end) absent from the cache and not known exchange gaps."""
        step = interval_ms(interval)
        sym = symbol.upper()
        have = {
            r[0] for r in self.conn.execute(
                "SELECT open_time FROM klines WHERE symbol=? AND interval=? AND open_time>=? AND open_time<? "
                "UNION SELECT open_time FROM kline_gaps WHERE symbol=? AND interval=? AND open_time>=? AND open_time<?",
                (sym, interval, start_ms, end_ms, sym, interval, start_ms, end_ms),
            )
        }
        return [t for t in range(ceil_to(start_ms, step), end_ms, step) if t not in have]

    def mark_gaps(self, symbol: str, interval: str, open_times: list[int]) -> None:
        self.conn.executemany(
            "INSERT OR IGNORE INTO kline_gaps(symbol, interval, open_time) VALUES (?,?,?)",
            [(symbol.upper(), interval, t) for t in open_times],
        )
        self.conn.commit()

    def known_gaps(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> list[int]:
        cur = self.conn.execute(
            "SELECT open_time FROM kline_gaps WHERE symbol=? AND interval=? AND open_time>=? AND open_time<? ORDER BY open_time",
            (symbol.upper(), interval, start_ms, end_ms),
        )
        return [r[0] for r in cur]


def contiguous_runs(times: list[int], step: int) -> list[tuple[int, int]]:
    """[t0, t0+step, t0+2step, t5, ...] -> [(t0, t0+3step), (t5, t5+step)] as half-open ranges."""
    runs: list[tuple[int, int]] = []
    for t in sorted(times):
        if runs and runs[-1][1] == t:
            runs[-1] = (runs[-1][0], t + step)
        else:
            runs.append((t, t + step))
    return runs


async def load_klines(
    rest: BinanceRestClient,
    cache: KlineCache,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    now: int | None = None,
) -> list[Kline]:
    """Closed klines with open_time in [start_ms, end_ms), fetching only missing ones.

    `end_ms` is clamped so that only candles already closed at `now` are requested;
    candles that are still missing after fetching are recorded as exchange gaps.
    """
    step = interval_ms(interval)
    now = now_ms() if now is None else now
    last_closed_open = floor_to(now, step) - step
    end_ms = min(end_ms, last_closed_open + step)
    if end_ms <= start_ms:
        return []

    missing = cache.missing_open_times(symbol, interval, start_ms, end_ms)
    for run_start, run_end in contiguous_runs(missing, step):
        fetched = await rest.get_klines(symbol, interval, run_start, run_end)
        cache.upsert(fetched)
        log_event(log, "cache.fetched", symbol=symbol, interval=interval,
                  start=ms_to_iso(run_start), end=ms_to_iso(run_end), rows=len(fetched))

    still_missing = cache.missing_open_times(symbol, interval, start_ms, end_ms)
    if still_missing:
        # Candle long closed yet Binance returned nothing: genuine exchange gap. Very
        # recent candles may just not be published yet, so they are not marked.
        settled = [t for t in still_missing if t + step + GAP_GRACE_MS <= now]
        if settled:
            cache.mark_gaps(symbol, interval, settled)
        log_event(log, "cache.missing", logging.WARNING, symbol=symbol, interval=interval,
                  count=len(still_missing), marked_as_exchange_gap=len(settled),
                  first=ms_to_iso(still_missing[0]), last=ms_to_iso(still_missing[-1]))
    return cache.get_range(symbol, interval, start_ms, end_ms)
