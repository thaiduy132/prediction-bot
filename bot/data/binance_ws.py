"""Binance combined-stream WebSocket client: transport only.

Responsibilities: connect, parse, reconnect with backoff, detect dead sockets
(no message for `stale_timeout_s`) and rotate the connection before Binance's
24h hard limit. Data integrity (gaps, backfill, dedup) lives in bot.data.feed.
"""

from __future__ import annotations

import asyncio
import json 
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Protocol

from bot.config import WebSocketConfig
from bot.data.gaps import Backoff
from bot.logging_setup import log_event
from bot.models import BookTicker, DepthSnapshot, Kline, book_ticker_from_ws, depth_from_ws, kline_from_ws
from bot.timeutil import now_ms

log = logging.getLogger(__name__)


# ---- events -----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class KlineEvent:
    kline: Kline
    source: str = "ws"  # "ws" | "backfill"


@dataclass(frozen=True, slots=True)
class BookTickerEvent:
    ticker: BookTicker


@dataclass(frozen=True, slots=True)
class DepthEvent:
    depth: DepthSnapshot


@dataclass(frozen=True, slots=True)
class ConnectionEvent:
    state: str  # "connected" | "disconnected"
    connection_no: int  # 1 for the first successful connection
    reason: str | None = None
    reconnect_in_s: float | None = None


@dataclass(frozen=True, slots=True)
class GapEvent:
    interval: str
    missing_start: int  # half-open [start, end) of open_times
    missing_end: int
    filled: int  # how many candles were recovered via REST
    expected: int


StreamEvent = KlineEvent | BookTickerEvent | DepthEvent | ConnectionEvent | GapEvent


# ---- transport ----------------------------------------------------------------

class WsLike(Protocol):
    async def recv(self) -> str | bytes: ...


ConnectFn = Callable[[str], AbstractAsyncContextManager[WsLike]]


def _default_connect(url: str) -> AbstractAsyncContextManager[WsLike]:
    from websockets.asyncio.client import connect

    # websockets answers Binance's server pings automatically.
    return connect(url, open_timeout=10, ping_interval=20, ping_timeout=20, max_size=2**20)


class _Stale(Exception):
    pass


class _Rotate(Exception):
    pass


def build_stream_url(base_url: str, symbol: str, kline_intervals: list[str], book_ticker: bool,
                     depth_levels: int = 0, depth_speed_ms: int = 100) -> str:
    s = symbol.lower()
    streams = [f"{s}@kline_{i}" for i in kline_intervals]
    if book_ticker:
        streams.append(f"{s}@bookTicker")
    if depth_levels:
        streams.append(f"{s}@depth{depth_levels}@{depth_speed_ms}ms")
    return f"{base_url.rstrip('/')}/stream?streams={'/'.join(streams)}"


class BinanceStream:
    def __init__(
        self,
        cfg: WebSocketConfig,
        symbol: str,
        connect_fn: ConnectFn = _default_connect,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        clock_ms: Callable[[], int] = now_ms,
        backoff: Backoff | None = None,
    ) -> None:
        self.cfg = cfg
        self.symbol = symbol.upper()
        self.url = build_stream_url(cfg.base_url, symbol, cfg.kline_intervals, cfg.book_ticker,
                                      cfg.depth_levels, cfg.depth_speed_ms)
        self._connect = connect_fn
        self._sleep = sleep
        self._mono = monotonic
        self._clock_ms = clock_ms
        self._backoff = backoff or Backoff(cfg.backoff_initial_s, cfg.backoff_max_s, cfg.backoff_factor, cfg.backoff_jitter)
        self.connections = 0

    def parse(self, raw: str | bytes) -> StreamEvent | None:
        msg: dict[str, Any] = json.loads(raw)
        stream, data = msg.get("stream", ""), msg.get("data")
        if data is None:
            return None
        if "@kline_" in stream:
            return KlineEvent(kline_from_ws(data))
        if stream.endswith("@bookTicker"):
            return BookTickerEvent(book_ticker_from_ws(data, self._clock_ms()))
        if "@depth" in stream:
            return DepthEvent(depth_from_ws(data, self._clock_ms()))
        return None

    async def events(self) -> AsyncIterator[StreamEvent]:
        """Infinite event stream; reconnects forever. Stop by closing the generator / cancelling."""
        while True:
            reason: str
            delay: float
            try:
                async with self._connect(self.url) as ws:
                    self.connections += 1
                    connected_at = self._mono()
                    log_event(log, "ws.connected", url=self.url, connection_no=self.connections)
                    yield ConnectionEvent("connected", self.connections)
                    while True:
                        age = self._mono() - connected_at
                        if age >= self.cfg.max_connection_age_s:
                            raise _Rotate()
                        timeout = min(self.cfg.stale_timeout_s, self.cfg.max_connection_age_s - age)
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout)
                        except TimeoutError:
                            if self._mono() - connected_at >= self.cfg.max_connection_age_s:
                                raise _Rotate() from None
                            raise _Stale() from None
                        if self._mono() - connected_at >= self.cfg.stable_after_s:
                            self._backoff.reset()
                        try:
                            ev = self.parse(raw)
                        except (ValueError, KeyError, TypeError) as e:
                            log_event(log, "ws.bad_message", logging.WARNING, error=repr(e), raw=str(raw)[:300])
                            continue
                        if ev is not None:
                            yield ev
            except _Rotate:
                reason, delay = "max_connection_age", 0.0
                self._backoff.reset()
            except _Stale:
                reason, delay = f"no message for {self.cfg.stale_timeout_s}s", self._backoff.next()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # network errors, ConnectionClosed, handshake failures...
                reason, delay = repr(e), self._backoff.next()

            log_event(log, "ws.disconnected", logging.WARNING, reason=reason, reconnect_in_s=round(delay, 2))
            yield ConnectionEvent("disconnected", self.connections, reason=reason, reconnect_in_s=delay)
            await self._sleep(delay)
