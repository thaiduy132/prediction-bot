"""Live market data with integrity guarantees on top of BinanceStream.

- Closed klines are delivered in order, without duplicates.
- Missing closed candles (gap between two closed candles, or anything that closed
  while we were disconnected) are recovered from REST and emitted with
  source="backfill" before newer candles.
- Unrecoverable holes are reported as GapEvent(filled < expected); downstream
  must treat those rounds as "missing data" and skip them.

NOTE for consumers: iterate quickly. Long work (e.g. calling Jev) must run in a
separate task, otherwise socket reads stall behind it.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import AsyncIterator, Callable

from bot.data.binance_rest import BinanceRestClient, BinanceRestError
from bot.data.binance_ws import BinanceStream, BookTickerEvent, ConnectionEvent, GapEvent, KlineEvent, StreamEvent
from bot.data.gaps import GapTracker, ObsKind
from bot.data.kline_cache import KlineCache
from bot.logging_setup import log_event
from bot.models import BookTicker, Kline
from bot.storage.db import write_status
from bot.timeutil import floor_to, interval_ms, ms_to_iso, now_ms

log = logging.getLogger(__name__)


class LiveFeed:
    def __init__(
        self,
        stream: BinanceStream,
        rest: BinanceRestClient,
        intervals: list[str],
        cache: KlineCache | None = None,
        status_conn: sqlite3.Connection | None = None,
        clock_ms: Callable[[], int] = now_ms,
    ) -> None:
        self.stream = stream
        self.rest = rest
        self.symbol = stream.symbol
        self.cache = cache
        self.status_conn = status_conn
        self._clock = clock_ms
        self.trackers = {i: GapTracker(interval_ms(i)) for i in intervals}
        self.last_book: BookTicker | None = None
        self.last_msg_ms: int | None = None
        self.gaps_detected = 0

    def _status(self, state: str, **details: object) -> None:
        if self.status_conn is not None:
            write_status(self.status_conn, "ws", state, self._clock(), **details)

    async def _fetch(self, interval: str, start: int, end: int) -> list[Kline]:
        try:
            ks = await self.rest.get_klines(self.symbol, interval, start, end)
        except BinanceRestError as e:
            log_event(log, "feed.backfill_failed", logging.ERROR, interval=interval,
                      start=ms_to_iso(start), end=ms_to_iso(end), error=str(e))
            return []
        return [k for k in ks if k.is_closed]

    def _accept(self, k: Kline, source: str) -> KlineEvent | None:
        """Run a closed kline through its tracker; returns the event to emit or None (dup)."""
        obs = self.trackers[k.interval].observe(k.open_time)
        if obs.kind is ObsKind.DUPLICATE:
            return None
        if self.cache is not None:
            self.cache.upsert([k])
        return KlineEvent(k, source)

    async def _fill(self, interval: str, start: int, end: int) -> AsyncIterator[StreamEvent]:
        """Recover [start, end) from REST, emit recovered candles then a GapEvent."""
        step = interval_ms(interval)
        expected = (end - start) // step
        if expected <= 0:
            return
        self.gaps_detected += 1
        got = [k for k in await self._fetch(interval, start, end) if start <= k.open_time < end]
        filled = 0
        for k in got:
            ev = self._accept(k, "backfill")
            if ev is not None:
                filled += 1
                yield ev
        log_event(log, "feed.gap", logging.WARNING if filled == expected else logging.ERROR,
                  interval=interval, start=ms_to_iso(start), end=ms_to_iso(end), expected=expected, filled=filled)
        yield GapEvent(interval, start, end, filled, expected)

    async def _on_closed_kline(self, k: Kline) -> AsyncIterator[StreamEvent]:
        tracker = self.trackers.get(k.interval)
        if tracker is None:
            return
        if tracker.last is not None and k.open_time > tracker.last + tracker.step:
            # Fill the hole first so candles stay in order; tracker must not move yet.
            async for ev in self._fill(k.interval, tracker.last + tracker.step, k.open_time):
                yield ev
            # Anything REST could not recover is skipped; realign tracker so k is "next".
            tracker.last = max(tracker.last, k.open_time - tracker.step)
        ev = self._accept(k, "ws")
        if ev is not None:
            yield ev

    async def _catch_up_after_reconnect(self) -> AsyncIterator[StreamEvent]:
        now = self._clock()
        for interval, tracker in self.trackers.items():
            if tracker.last is None:
                continue
            last_closed_open = floor_to(now, tracker.step) - tracker.step
            start = tracker.last + tracker.step
            if start <= last_closed_open:
                async for ev in self._fill(interval, start, last_closed_open + tracker.step):
                    yield ev
                tracker.last = max(tracker.last, last_closed_open)

    async def events(self) -> AsyncIterator[StreamEvent]:
        async for ev in self.stream.events():
            self.last_msg_ms = self._clock()
            match ev:
                case BookTickerEvent(ticker=t):
                    self.last_book = t
                    yield ev
                case KlineEvent(kline=k) if k.is_closed:
                    async for out in self._on_closed_kline(k):
                        yield out
                    self._status("connected", connection_no=self.stream.connections,
                                  last_closed={i: t.last for i, t in self.trackers.items()},
                                  gaps_detected=self.gaps_detected)
                case KlineEvent():
                    yield ev  # forming candle update; never used for decisions
                case ConnectionEvent(state="connected", connection_no=n):
                    self._status("connected", connection_no=n, gaps_detected=self.gaps_detected)
                    yield ev
                    if n > 1:
                        async for out in self._catch_up_after_reconnect():
                            yield out
                case ConnectionEvent(state="disconnected"):
                    self._status("reconnecting", reason=ev.reason, reconnect_in_s=ev.reconnect_in_s)
                    yield ev
                case _:
                    yield ev

