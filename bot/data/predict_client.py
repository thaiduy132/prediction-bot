"""Predict.fun REST client (the venue behind Binance Wallet prediction markets). Read-only.

Docs: https://dev.predict.fun/ . Mainnet needs an `x-api-key` header; the testnet does not.
Order books quote the YES side only: the first outcome ("Up" in Up/Down rounds) is YES,
and DOWN's price is the complement (1 - price) at the market's decimal precision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from bot.logging_setup import log_event

log = logging.getLogger(__name__)


class PredictError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OddsBook:
    """Top of the Up (YES) order book. Prices are probabilities in [0, 1]."""

    bid: float | None
    ask: float | None
    bid_qty: float
    ask_qty: float
    bid_depth5: float  # total qty over the top 5 bid levels
    ask_depth5: float
    update_ms: int | None

    @property
    def mid(self) -> float | None:
        return None if self.bid is None or self.ask is None else (self.bid + self.ask) / 2


def parse_orderbook(payload: dict[str, Any]) -> OddsBook:
    d = payload["data"]
    bids, asks = d.get("bids") or [], d.get("asks") or []  # [[price, qty], ...], best first
    return OddsBook(
        bid=float(bids[0][0]) if bids else None,
        ask=float(asks[0][0]) if asks else None,
        bid_qty=float(bids[0][1]) if bids else 0.0,
        ask_qty=float(asks[0][1]) if asks else 0.0,
        bid_depth5=sum(float(q) for _, q in bids[:5]),
        ask_depth5=sum(float(q) for _, q in asks[:5]),
        update_ms=int(d["updateTimestampMs"]) if d.get("updateTimestampMs") else None,
    )


def round_slug(symbol: str, round_seconds: int, start_s: int) -> str:
    """'BTCUSDT', 300, 1790659200 -> 'btc-updown-5m-1790659200' (Predict.fun's category slug)."""
    base = symbol.upper().removesuffix("USDT").lower()
    unit = f"{round_seconds // 60}m"
    return f"{base}-updown-{unit}-{start_s}"


class PredictClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout_s: float = 10.0,
                 client: httpx.AsyncClient | None = None) -> None:
        headers = {"x-api-key": api_key} if api_key else {}
        self._client = client or httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_s, headers=headers)
        self._owns = client is None

    async def aclose(self) -> None:
        if self._owns:
            await self._client.aclose()

    async def __aenter__(self) -> PredictClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _get(self, path: str, params: dict[str, Any] | None = None, allow_404: bool = False) -> Any:
        try:
            r = await self._client.get(path, params=params)
        except httpx.TransportError as e:
            raise PredictError(f"GET {path} failed: {e!r}") from e
        if r.status_code == 404 and allow_404:
            return None
        if r.status_code in (401, 403):
            raise PredictError(f"GET {path}: HTTP {r.status_code}, this host needs a valid PREDICT_API_KEY")
        if r.status_code == 429:
            raise PredictError(f"GET {path}: rate limited (429), raise predict.poll_interval_s")
        if r.status_code != 200:
            raise PredictError(f"GET {path}: HTTP {r.status_code} {r.text[:200]}")
        return r.json()

    async def open_updown_markets(self) -> list[dict[str, Any]]:
        """All currently OPEN crypto Up/Down markets (paginated)."""
        out: list[dict[str, Any]] = []
        after: str | None = None
        for _ in range(20):
            params: dict[str, Any] = {"first": 100, "marketVariant": "CRYPTO_UP_DOWN", "status": "OPEN"}
            if after:
                params["after"] = after
            j = await self._get("/v1/markets", params)
            out += j["data"]
            after = j.get("cursor")
            if len(j["data"]) < 100 or not after:
                break
        return out

    async def market(self, market_id: int) -> dict[str, Any] | None:
        j = await self._get(f"/v1/markets/{market_id}", allow_404=True)
        return None if j is None else j["data"]

    async def orderbook(self, market_id: int) -> OddsBook | None:
        """None when the market has no order book (404), which is normal for untraded markets."""
        j = await self._get(f"/v1/markets/{market_id}/orderbook", allow_404=True)
        if j is None:
            log_event(log, "predict.no_orderbook", logging.DEBUG, market_id=market_id)
            return None
        return parse_orderbook(j)
