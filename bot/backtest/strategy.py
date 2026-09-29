"""Strategy interface for backtests plus two baselines.

A strategy sees ONLY what was known at decision time: the round's open price and the
1s candles that had already closed. It must never look at the round's final candle.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from bot.backtest.features import book_features
from bot.data.book_store import BookSample
from bot.models import Kline


class Side(StrEnum):
    UP = "UP"
    DOWN = "DOWN"


@dataclass(frozen=True, slots=True)
class RoundContext:
    round_open_time: int  # ms
    round_open: float  # open of the round candle (reference price)
    decision_time: int  # ms; every candle in `recent` closed at or before this
    recent: Sequence[Kline]  # closed 1s candles from round open up to decision_time, in order
    book: Sequence[BookSample] = ()  # recorded book samples that were known before decision_time

    @property
    def last_price(self) -> float:
        return self.recent[-1].close

    @property
    def move_bps(self) -> float:
        """Move from the round open to the latest price, in basis points."""
        return (self.last_price / self.round_open - 1) * 10_000


class Strategy(Protocol):
    name: str
    needs_book: bool  # True: rounds without recorded order-book data are skipped

    def decide(self, ctx: RoundContext) -> Side | None:
        """Return a side to bet on, or None to skip this round."""
        ...


@dataclass(frozen=True, slots=True)
class BookParams:
    levels: int = 10  # depth levels used for imbalance (1 | 5 | 10 | 20)
    window_s: int = 10  # average imbalance over the last N seconds
    min_imbalance: float = 0.2  # |imbalance| needed to act
    max_spread_bps: float = 2.0  # skip when the spread is wider (thin or stressed book)


def _book_side(ctx: RoundContext, p: BookParams) -> Side | None:
    """Side the book leans to, or None if the book is unusable or not lopsided enough."""
    f = book_features(ctx.book, p.levels, p.window_s)
    if f is None or f.spread_bps > p.max_spread_bps or abs(f.imbalance) < p.min_imbalance:
        return None
    return Side.UP if f.imbalance > 0 else Side.DOWN


@dataclass(frozen=True, slots=True)
class BookImbalance:
    """Bet with the heavier side of the order book, ignoring price movement."""

    params: BookParams = BookParams()
    name: str = "book"
    needs_book: bool = True

    def decide(self, ctx: RoundContext) -> Side | None:
        return _book_side(ctx, self.params)


@dataclass(frozen=True, slots=True)
class MomentumBook:
    """Follow the price move only when the order book leans the same way (skip if it disagrees)."""

    min_move_bps: float = 1.0
    params: BookParams = BookParams()
    name: str = "momentum_book"
    needs_book: bool = True

    def decide(self, ctx: RoundContext) -> Side | None:
        if abs(ctx.move_bps) < self.min_move_bps:
            return None
        move_side = Side.UP if ctx.move_bps > 0 else Side.DOWN
        return move_side if _book_side(ctx, self.params) is move_side else None


@dataclass(frozen=True, slots=True)
class Momentum:
    """Bet that the move so far continues (or reverses if contrarian) when it is big enough."""

    min_move_bps: float = 1.0
    contrarian: bool = False
    needs_book: bool = False

    @property
    def name(self) -> str:
        return "reversal" if self.contrarian else "momentum"

    def decide(self, ctx: RoundContext) -> Side | None:
        m = ctx.move_bps
        if abs(m) < self.min_move_bps:
            return None
        up = m > 0
        if self.contrarian:
            up = not up
        return Side.UP if up else Side.DOWN


@dataclass(frozen=True, slots=True)
class AlwaysUp:
    """Reference: any 'edge' must beat this, since BTC drifts and ties may count as UP."""

    name: str = "always_up"
    needs_book: bool = False

    def decide(self, ctx: RoundContext) -> Side | None:
        return Side.UP


def make_strategy(name: str, min_move_bps: float, book: BookParams = BookParams()) -> Strategy:
    match name:
        case "momentum":
            return Momentum(min_move_bps)
        case "reversal":
            return Momentum(min_move_bps, contrarian=True)
        case "always_up":
            return AlwaysUp()
        case "book":
            return BookImbalance(book)
        case "momentum_book":
            return MomentumBook(min_move_bps, book)
    raise ValueError(f"unknown strategy {name!r} (momentum | reversal | always_up | book | momentum_book)")
