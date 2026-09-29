"""Market data value objects and Binance payload parsers.

Prices are kept as float. Binance quotes BTCUSDT with <= 8 decimals (~13 significant
digits), well inside double precision, and float rounding is monotonic, so comparisons
like close >= open give the same answer as exact decimal comparison (ties included,
because identical strings parse to identical floats).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bot.timeutil import interval_ms


@dataclass(frozen=True, slots=True)
class Kline:
    symbol: str
    interval: str
    open_time: int  # ms, inclusive
    close_time: int  # ms, Binance convention: open_time + interval_ms - 1
    open: float
    high: float
    low: float
    close: float
    volume: float  # base asset
    quote_volume: float
    trades: int
    taker_buy_base: float
    taker_buy_quote: float
    is_closed: bool = True

    @property
    def interval_ms(self) -> int:
        return interval_ms(self.interval)


@dataclass(frozen=True, slots=True)
class BookTicker:
    symbol: str
    update_id: int
    bid: float
    bid_qty: float
    ask: float
    ask_qty: float
    recv_time: int  # local receive time, ms (spot bookTicker carries no event time)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid


def kline_from_rest(row: list[Any], symbol: str, interval: str, now_ms: int) -> Kline:
    """Parse one row of GET /api/v3/klines.

    Row layout: [open_time, open, high, low, close, volume, close_time, quote_volume,
    trades, taker_buy_base, taker_buy_quote, ignore].
    REST returns the still-forming candle too; mark it not closed.
    """
    close_time = int(row[6])
    return Kline(
        symbol=symbol.upper(),
        interval=interval,
        open_time=int(row[0]),
        close_time=close_time,
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
        quote_volume=float(row[7]),
        trades=int(row[8]),
        taker_buy_base=float(row[9]),
        taker_buy_quote=float(row[10]),
        is_closed=close_time < now_ms,
    )


def kline_from_ws(data: dict[str, Any]) -> Kline:
    """Parse the `data` object of a `<symbol>@kline_<interval>` stream message."""
    k = data["k"]
    return Kline(
        symbol=k["s"].upper(),
        interval=k["i"],
        open_time=int(k["t"]),
        close_time=int(k["T"]),
        open=float(k["o"]),
        high=float(k["h"]),
        low=float(k["l"]),
        close=float(k["c"]),
        volume=float(k["v"]),
        quote_volume=float(k["q"]),
        trades=int(k["n"]),
        taker_buy_base=float(k["V"]),
        taker_buy_quote=float(k["Q"]),
        is_closed=bool(k["x"]),
    )


def book_ticker_from_ws(data: dict[str, Any], recv_time: int) -> BookTicker:
    """Parse the `data` object of a `<symbol>@bookTicker` stream message."""
    return BookTicker(
        symbol=data["s"].upper(),
        update_id=int(data["u"]),
        bid=float(data["b"]),
        bid_qty=float(data["B"]),
        ask=float(data["a"]),
        ask_qty=float(data["A"]),
        recv_time=recv_time,
    )
