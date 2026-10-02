"""Local web dashboard: live candles + book ticker pushed to the browser over WebSocket.

Serves one HTML page (bot/ui/index.html) and a /ws endpoint on the same port. History is
loaded from REST at startup, then kept current from LiveFeed. With a PaperTrader it also shows
simulated bets and profit/loss. It never places real orders.
"""

from __future__ import annotations

import asyncio
import json
import logging
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any

from websockets.asyncio.server import ServerConnection, broadcast, serve
from websockets.http11 import Request, Response

from bot.backtest.features import imbalance
from bot.clock_sync import describe_offset, keep_clock_synced, sync_clock
from bot.config import AppConfig
from bot.data.binance_rest import BinanceRestClient, BinanceRestError
from bot.data.binance_ws import BinanceStream, BookTickerEvent, ConnectionEvent, DepthEvent, GapEvent, KlineEvent
from bot.data.book_store import LEVELS, BookRecorder, BookStore, sample_from_depth
from bot.data.feed import LiveFeed
from bot.data.kline_cache import KlineCache
from bot.data.odds_recorder import OddsPoller
from bot.data.odds_store import OddsSample
from bot.data.predict_client import PredictClient
from bot.logging_setup import log_event
from bot.models import Kline
from bot.paper.store import PaperStore
from bot.paper.trader import PaperTrader
from bot.settlement import settle_kline

if TYPE_CHECKING:
    from bot.live.executor import LiveExecutor
from bot.timeutil import floor_to, interval_ms, now_ms

log = logging.getLogger(__name__)

INDEX_HTML = Path(__file__).with_name("index.html")
HISTORY_CANDLES = {"1s": 900, "1m": 500, "5m": 300}  # default 300 for other intervals
MAX_KEPT = 3000
BOOK_MIN_INTERVAL_MS = 100  # bookTicker can fire hundreds of times per second


def _candle(k: Kline) -> dict[str, Any]:
    return {"time": k.open_time // 1000, "open": k.open, "high": k.high, "low": k.low,
            "close": k.close, "volume": k.volume, "closed": k.is_closed}


class Dashboard:
    def __init__(self, cfg: AppConfig, host: str, port: int, paper: PaperTrader | None = None,
                 api_key: str | None = None, executor: LiveExecutor | None = None) -> None:
        self.executor = executor  # shadow/live orders on top of the paper decisions
        self.cfg = cfg
        self.host, self.port = host, port
        self.paper = paper  # live paper trader (no real orders); None = price monitor only
        self.api_key = api_key  # Predict.fun, only needed on mainnet
        self.candles: dict[str, dict[int, dict[str, Any]]] = {i: {} for i in cfg.websocket.kline_intervals}
        self.book: dict[str, Any] | None = None
        self.depth_imb: dict[str, float] | None = None  # depth imbalance per level count
        self.recorder: BookRecorder | None = None
        self.connected = False
        self._last_book_sent = 0
        self._last_depth_sent = 0
        self._server: Any = None

    # ---- state ------------------------------------------------------------------

    def _put(self, interval: str, c: dict[str, Any]) -> None:
        d = self.candles.get(interval)
        if d is None:
            return
        d[c["time"]] = c
        if len(d) > MAX_KEPT:
            for t in sorted(d)[: len(d) - MAX_KEPT]:
                del d[t]

    def snapshot(self) -> str:
        return json.dumps({
            "t": "snapshot",
            "symbol": self.cfg.market.symbol,
            "round_interval": self.cfg.market.round_interval,
            "round_seconds": interval_ms(self.cfg.market.round_interval) // 1000,
            "rule": self.cfg.settlement.rule.value,
            "connected": self.connected,
            "server_ms": now_ms(),
            "book": self.book,
            "depth_imb": self.depth_imb,
            "paper": None if self.paper is None else self._paper_view(),
            "candles": {i: [d[t] for t in sorted(d)] for i, d in self.candles.items()},
        })

    def _push(self, msg: dict[str, Any]) -> None:
        if self._server is not None:
            broadcast(self._server.connections, json.dumps(msg))

    # ---- http / ws --------------------------------------------------------------

    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        path = request.path.split("?")[0]
        if path == "/ws":
            return None
        if path in ("/", "/index.html"):
            resp = connection.respond(HTTPStatus.OK, INDEX_HTML.read_text(encoding="utf-8"))
            resp.headers["Content-Type"] = "text/html; charset=utf-8"
            return resp
        return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")

    async def _handler(self, ws: ServerConnection) -> None:
        await ws.send(self.snapshot())
        async for raw in ws:  # the only messages accepted: STOP / RESUME of real orders
            try:
                msg = json.loads(raw)
                cmd = msg.get("cmd")
            except (ValueError, AttributeError):
                continue
            # Browsers let ANY website open a WebSocket to 127.0.0.1, so only this page may send commands.
            origin = ws.request.headers.get("Origin") if ws.request is not None else None
            if origin not in {f"http://{h}:{self.port}" for h in (self.host, "127.0.0.1", "localhost")}:
                log_event(log, "ui.command_rejected", logging.WARNING, origin=origin, cmd=cmd)
                continue
            if self.executor is not None and cmd in ("stop", "resume"):
                self.executor.set_stopped(cmd == "stop")
                self._push_paper()
            elif self.executor is not None and cmd == "sell_quote":
                try:
                    fraction = float(msg.get("fraction", 1.0))
                except (TypeError, ValueError):
                    continue
                res = await self.executor.sell_quote(fraction)
                await ws.send(json.dumps({"t": "sell_quote", **res}, default=str))
                self._push_paper()
            elif self.executor is not None and cmd == "sell_confirm":
                qid = str(msg.get("quote_id", ""))
                # the sell waits up to ~30s for Binance: do not block this socket meanwhile
                asyncio.get_running_loop().create_task(self._sell(ws, qid))

    async def _sell(self, ws: ServerConnection, quote_id: str) -> None:
        assert self.executor is not None
        try:
            res = await self.executor.sell_confirm(quote_id)
        except Exception as e:  # report instead of dying silently in a task
            res = {"error": repr(e)}
        try:
            await ws.send(json.dumps({"t": "sell_result", **res}, default=str))
        except Exception:
            pass
        self._push_paper()

    # ---- data -------------------------------------------------------------------

    async def _load_history(self, rest: BinanceRestClient) -> None:
        now = now_ms()
        for interval in self.candles:
            step = interval_ms(interval)
            n = HISTORY_CANDLES.get(interval, 300)
            try:
                ks = await rest.get_klines(self.cfg.market.symbol, interval, floor_to(now, step) - n * step, now + 1)
            except BinanceRestError as e:
                log_event(log, "ui.history_failed", logging.ERROR, interval=interval, error=str(e))
                continue
            for k in ks:
                self._put(interval, _candle(k))
            if self.paper is not None and interval == "1s":
                self.paper.seed_seconds(ks)
            if self.executor is not None and interval == self.cfg.market.round_interval:
                # real orders from before a restart whose round has closed since: book them now
                for k in ks:
                    if k.is_closed:
                        self.executor.on_settle(k.open_time, settle_kline(k, self.cfg.settlement.rule, interval))
            log_event(log, "ui.history_loaded", interval=interval, candles=len(ks))

    async def _pump(self, feed: LiveFeed) -> None:
        async for ev in feed.events():
            match ev:
                case KlineEvent(kline=k):
                    c = _candle(k)
                    self._put(k.interval, c)
                    self._push({"t": "kline", "interval": k.interval, "candle": c})
                    if self.paper is not None and self.paper.on_kline(k):
                        self._push_paper()  # a bet opened, was skipped, or settled: show it right away
                case BookTickerEvent(ticker=b):
                    self.book = {"bid": b.bid, "ask": b.ask, "bid_qty": b.bid_qty, "ask_qty": b.ask_qty, "ts": b.recv_time}
                    if b.recv_time - self._last_book_sent >= BOOK_MIN_INTERVAL_MS:
                        self._last_book_sent = b.recv_time
                        self._push({"t": "book", **self.book})
                case DepthEvent(depth=d):
                    if self.recorder is not None:
                        self.recorder.handle(ev)
                    smp = sample_from_depth(d, d.recv_time)
                    if smp is not None:
                        self.depth_imb = {str(n): round(imbalance(*smp.depth(n)), 4) for n in LEVELS}
                        if d.recv_time - self._last_depth_sent >= BOOK_MIN_INTERVAL_MS:
                            self._last_depth_sent = d.recv_time
                            self._push({"t": "depth", "imb": self.depth_imb})
                case ConnectionEvent(state=s):
                    self.connected = s == "connected"
                    self._push({"t": "conn", "connected": self.connected, "reason": ev.reason})
                case GapEvent():
                    self._push({"t": "gap", "interval": ev.interval, "filled": ev.filled, "expected": ev.expected})

    def _paper_view(self) -> dict[str, Any]:
        assert self.paper is not None
        view = self.paper.snapshot()
        view["live"] = None if self.executor is None else self.executor.status(self.paper.last_odds)
        return view

    def _push_paper(self) -> None:
        if self.paper is not None:
            self._push({"t": "paper", "paper": self._paper_view()})

    async def _paper_ticker(self) -> None:
        """Refresh the paper panel every second (countdowns, current odds) even when nothing traded."""
        while True:
            await asyncio.sleep(1.0)
            self._push_paper()

    def _on_odds(self, sample: OddsSample | None) -> None:
        if self.paper is not None:
            self.paper.on_odds(sample)

    async def run(self) -> None:
        cache = KlineCache(self.cfg.data.cache_path)
        store = BookStore(self.cfg.data.book_path, self.cfg.market.symbol) if self.cfg.websocket.depth_levels else None
        paper_store = PaperStore(self.cfg.paper.trades_path) if self.paper is not None else None
        if self.paper is not None:
            self.paper.store = paper_store
            self.paper.reload()
        if store is not None or self.paper is not None:  # the dashboard also records the book
            self.recorder = BookRecorder(store, self.paper.on_book if self.paper is not None else None)
        tasks: list[asyncio.Task[None]] = []
        try:
            async with BinanceRestClient(self.cfg.rest.base_url, self.cfg.rest.timeout_s,
                                         self.cfg.rest.max_retries) as rest, \
                    PredictClient(self.cfg.predict.base_url, self.api_key, self.cfg.predict.timeout_s) as predict:
                offset = await sync_clock(rest)  # before anything reads the time
                print(describe_offset(offset), flush=True)
                tasks.append(asyncio.create_task(keep_clock_synced(rest)))
                await self._load_history(rest)
                stream = BinanceStream(self.cfg.websocket, self.cfg.market.symbol)
                feed = LiveFeed(stream, rest, self.cfg.websocket.kline_intervals, cache=cache)
                if self.paper is not None and self.paper.use_odds:
                    poller = OddsPoller(self.cfg, predict)  # live only; `data record-odds` is what stores them
                    tasks.append(asyncio.create_task(poller.run(self._on_odds)))
                    if self.executor is not None:
                        self.executor.tokens = lambda r: poller.outcome_tokens(r)  # type: ignore[assignment,return-value]
                        self.paper.on_entry = self.executor.on_entry
                        self.paper.on_settle = self.executor.on_settle
                        self.executor.recheck = self.paper.recheck
                        tasks.append(asyncio.create_task(self.executor.redeem_loop()))
                        tasks.append(asyncio.create_task(self.executor.topic_loop()))
                        tasks.append(asyncio.create_task(self.executor.account_loop()))
                        tasks.append(asyncio.create_task(self.executor.reconcile_loop()))
                        tasks.append(asyncio.create_task(self.executor.position_loop()))
                if self.paper is not None:
                    tasks.append(asyncio.create_task(self._paper_ticker()))
                async with serve(self._handler, self.host, self.port,
                                 process_request=self._process_request) as server:
                    self._server = server
                    print(f"Dashboard: http://{self.host}:{self.port}  (Ctrl+C to stop)", flush=True)
                    await self._pump(feed)
        finally:
            for t in tasks:
                t.cancel()
            cache.close()
            if store is not None:
                store.close()
            if paper_store is not None:
                paper_store.close()


async def cmd_ui(cfg: AppConfig, host: str, port: int, paper: PaperTrader | None = None,
                 api_key: str | None = None, executor: LiveExecutor | None = None) -> int:
    await Dashboard(cfg, host, port, paper, api_key, executor).run()
    return 0
