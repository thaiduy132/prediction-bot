"""Binance spot REST client (public market data only, no API key)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from bot.logging_setup import log_event
from bot.models import Kline, kline_from_rest
from bot.timeutil import interval_ms, now_ms

log = logging.getLogger(__name__)

KLINES_LIMIT = 1000


class BinanceRestError(RuntimeError):
    pass


class BinanceRestClient:
    def __init__(
        self,
        base_url: str = "https://api.binance.com",
        timeout_s: float = 10.0,
        max_retries: int = 5,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], int] = now_ms,
    ) -> None:
        self._client = client or httpx.AsyncClient(base_url=base_url, timeout=timeout_s)
        self._owns_client = client is None
        self._max_retries = max_retries
        self._sleep = sleep
        self._clock = clock

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> BinanceRestClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        delay = 1.0
        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._client.get(path, params=params)
            except httpx.TransportError as e:
                if attempt == self._max_retries:
                    raise BinanceRestError(f"GET {path} failed: {e!r}") from e
                log_event(log, "rest.retry", logging.WARNING, path=path, attempt=attempt, error=repr(e))
                await self._sleep(delay)
                delay = min(delay * 2, 30.0)
                continue

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 418:
                # IP banned for ignoring 429s: never hammer, surface immediately.
                raise BinanceRestError(f"GET {path}: IP banned (418), retry-after={resp.headers.get('Retry-After')}")
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == self._max_retries:
                    raise BinanceRestError(f"GET {path}: HTTP {resp.status_code} {resp.text[:200]}")
                wait = float(resp.headers.get("Retry-After", delay))
                log_event(log, "rest.retry", logging.WARNING, path=path, attempt=attempt, status=resp.status_code, wait_s=wait)
                await self._sleep(wait)
                delay = min(delay * 2, 30.0)
                continue
            raise BinanceRestError(f"GET {path}: HTTP {resp.status_code} {resp.text[:200]}")
        raise AssertionError("unreachable")

    async def server_time(self) -> int:
        return int((await self._get("/api/v3/time"))["serverTime"])

    async def get_klines_page(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> list[Kline]:
        """One page (<= 1000 rows) of klines whose open_time is in [start_ms, end_ms]."""
        rows = await self._get(
            "/api/v3/klines",
            {"symbol": symbol.upper(), "interval": interval, "startTime": start_ms, "endTime": end_ms, "limit": KLINES_LIMIT},
        )
        now = self._clock()
        return [kline_from_rest(r, symbol, interval, now) for r in rows]

    async def get_klines(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> list[Kline]:
        """All klines with open_time in [start_ms, end_ms), paginated. Includes the forming candle if in range."""
        step = interval_ms(interval)
        out: list[Kline] = []
        cursor = start_ms
        while cursor < end_ms:
            page = await self.get_klines_page(symbol, interval, cursor, end_ms - 1)
            if not page:
                break
            out.extend(k for k in page if start_ms <= k.open_time < end_ms)
            cursor = page[-1].open_time + step
            if len(page) < KLINES_LIMIT:
                break
        return out
