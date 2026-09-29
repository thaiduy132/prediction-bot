"""Poll Predict.fun once per `poll_interval_s` for the odds of the CURRENT Up/Down round.

`OddsPoller` is the reusable part (the paper-trading dashboard consumes its samples live);
`record_odds` wraps it to store every sample for later backtests.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from collections.abc import Callable
from typing import Any

from bot.config import AppConfig
from bot.data.odds_store import OddsSample, OddsStore
from bot.data.predict_client import PredictClient, PredictError, round_slug
from bot.logging_setup import log_event
from bot.timeutil import interval_ms, now_ms

log = logging.getLogger(__name__)

LIST_REFRESH_MS = 60_000  # re-list open markets this often (new rounds appear ahead of time)
MISSING_RETRY_MS = 10_000  # when the current round's market is not in the list, retry this often
RESOLVE_EVERY_MS = 30_000
RESOLVE_AFTER_MS = 60_000  # a round is checked for resolution this long after it ends


def _round_fields(m: dict[str, Any]) -> tuple[str | None, float | None, float | None, str | None, str | None]:
    v = m.get("variantData") or {}
    res = m.get("resolution")
    return (v.get("priceFeedProvider"), v.get("startPrice"), v.get("endPrice"), m.get("status"),
            None if res is None else json.dumps(res, default=str) if not isinstance(res, str) else res)


class OddsPoller:
    """Reads the current round's order book once per tick. Stores rows only if given a store."""

    def __init__(self, cfg: AppConfig, client: PredictClient, store: OddsStore | None = None) -> None:
        self.cfg = cfg
        self.client = client
        self.store = store
        self.round_ms = interval_ms(cfg.market.round_interval)
        self.counts: Counter[str] = Counter()
        self.last: OddsSample | None = None
        self._markets: dict[str, dict[str, Any]] = {}
        self._listed_at = 0
        self._resolved_at = 0

    async def _refresh_list(self) -> None:
        self._markets = {m["categorySlug"]: m for m in await self.client.open_updown_markets()}
        self._listed_at = now_ms()

    async def _resolve_finished(self, now: int) -> None:
        self._resolved_at = now
        if self.store is None:
            return
        for start_ms, market_id in self.store.unresolved_rounds(now - self.round_ms - RESOLVE_AFTER_MS + 1):
            m = await self.client.market(market_id)
            if m is not None:
                prov, sp, ep, status, res = _round_fields(m)
                self.store.upsert_round(start_ms, market_id, m["categorySlug"], prov, sp, ep, status, res)
                self.counts["rounds_resolved"] += status == "RESOLVED"

    async def tick(self) -> OddsSample | None:
        """One poll. Returns the sample read (possibly with no quote), or None if the round's market is unknown."""
        now = now_ms()
        start_ms = now // self.round_ms * self.round_ms
        slug = round_slug(self.cfg.market.symbol, self.round_ms // 1000, start_ms // 1000)
        age = now - self._listed_at
        if slug not in self._markets and age >= MISSING_RETRY_MS or age >= LIST_REFRESH_MS:
            await self._refresh_list()
        m = self._markets.get(slug)
        if m is None:
            self.counts["no_market_for_round"] += 1
            return None
        book = await self.client.orderbook(m["id"])
        if book is None:
            sample = OddsSample(start_ms, now, m["id"], None, None, 0, 0, 0, 0)
            self.counts["samples_no_book"] += 1
        else:
            sample = OddsSample(start_ms, now, m["id"], book.bid, book.ask, book.bid_qty, book.ask_qty,
                                book.bid_depth5, book.ask_depth5)
            self.counts["samples_with_quote" if sample.has_quote else "samples_empty_book"] += 1
        self.last = sample
        if self.store is not None:
            prov, sp, ep, status, res = _round_fields(m)
            self.store.upsert_round(start_ms, m["id"], slug, prov, sp, ep, status, res)
            self.store.add_sample(sample)
        if now - self._resolved_at >= RESOLVE_EVERY_MS:
            await self._resolve_finished(now)
        return sample

    async def run(self, on_sample: Callable[[OddsSample | None], None] | None = None) -> None:
        """Poll forever; errors are logged and retried."""
        loop = asyncio.get_running_loop()
        while True:
            t0 = loop.time()
            try:
                sample = await self.tick()
                if on_sample is not None:
                    on_sample(sample)
            except PredictError as e:
                self.counts["errors"] += 1
                log_event(log, "odds.error", logging.WARNING, error=str(e))
                await asyncio.sleep(2.0)
            await asyncio.sleep(max(0.0, self.cfg.predict.poll_interval_s - (loop.time() - t0)))


async def record_odds(cfg: AppConfig, api_key: str | None, duration_s: float | None) -> dict[str, Any]:
    store = OddsStore(cfg.predict.odds_path, cfg.market.symbol)
    async with PredictClient(cfg.predict.base_url, api_key, cfg.predict.timeout_s) as client:
        poller = OddsPoller(cfg, client, store)
        try:
            await asyncio.wait_for(poller.run(), timeout=duration_s)
        except TimeoutError:
            pass
    out = {"host": cfg.predict.base_url, "counts": dict(poller.counts), "stored": store.summary()}
    store.close()
    return out
