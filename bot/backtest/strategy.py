"""Strategy interface for backtests plus two baselines.

A strategy sees ONLY what was known at decision time: the round's open price and the
1s candles that had already closed. It must never look at the round's final candle.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from bot.backtest.calibration import CalibrationTable, sigma_bps, z_score
from bot.backtest.features import book_features
from bot.backtest.pricing import Side, entry_price, expected_profit
from bot.data.book_store import BookSample
from bot.data.odds_store import OddsSample
from bot.models import Kline

__all__ = ["Side"]  # re-exported: most callers import Side from here


@dataclass(frozen=True, slots=True)
class RoundContext:
    round_open_time: int  # ms
    round_open: float  # open of the round candle (reference price)
    decision_time: int  # ms; every candle in `recent` closed at or before this
    recent: Sequence[Kline]  # closed 1s candles from round open up to decision_time, in order
    book: Sequence[BookSample] = ()  # recorded book samples that were known before decision_time
    lookback: Sequence[Kline] = ()  # closed 1s candles of the SIGMA_WINDOW_S seconds before decision_time
    quote: OddsSample | None = None  # latest odds quote known at decision_time (odds mode only)

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
    # Optional attribute `needs_odds` (default False): the strategy reads ctx.quote, so rounds
    # without a fresh quote are skipped before it is asked.

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

    def accept_retry(self, ctx: RoundContext, side: Side, price: float, max_price: float) -> str | None:
        """A retry after a failed order is a fresh decision: same direction now, and not too expensive."""
        if self.decide(ctx) is not side:
            return f"move no longer points {side.value} ({ctx.move_bps:+.2f} bps)"
        if price > max_price:
            return f"price {price:.3f} above the retry cap {max_price:.2f}"
        return None


@dataclass(frozen=True, slots=True)
class AlwaysUp:
    """Reference: any 'edge' must beat this, since BTC drifts and ties may count as UP."""

    name: str = "always_up"
    needs_book: bool = False

    def decide(self, ctx: RoundContext) -> Side | None:
        return Side.UP


@dataclass(frozen=True, slots=True)
class ValueParams:
    fee_bps: float = 0.0  # charged on the stake, same as the backtest's fee_bps
    min_edge: float = 0.03  # required expected profit per unit staked, after the fee
    min_samples: int = 100  # ignore |z| buckets calibrated on fewer rounds than this


@dataclass(frozen=True, slots=True)
class Value:
    """Bet only when the market price is below the calibrated win probability.

    p = measured win rate of "follow the move" for this |z| bucket (see bot.backtest.calibration).
    Expected profit per unit staked on a side that wins with probability q at price c:
        q / c - 1 - fee
    Both sides are checked: following the move (q = p) and fading it (q = 1 - p). The better one
    is taken if it clears `min_edge`; otherwise the round is skipped.
    """

    table: CalibrationTable
    params: ValueParams = ValueParams()
    name: str = "value"
    needs_book: bool = False
    needs_odds: bool = True

    def __repr__(self) -> str:  # short and stable: used as the paper-trading history label
        return f"Value({self.table.label}, {self.params})"

    def estimate(self, ctx: RoundContext, strict: bool = True) -> tuple[Side, float, float] | None:
        """(side the move points to, calibrated P(that side wins), z) or None if not estimable.

        strict=False (retries a few seconds later) uses the time actually left in the round: z already
        scales the move by sqrt(time left), so the table measured at decision_s is a close approximation.
        """
        elapsed = (ctx.decision_time - ctx.round_open_time) // 1000
        if strict and elapsed != self.table.decision_s:
            raise ValueError(f"calibration is for second {self.table.decision_s}, decision is at {elapsed}")
        sig = sigma_bps(ctx.lookback)
        move = ctx.move_bps
        if sig is None or move == 0:
            return None
        z = z_score(move, sig, self.table.round_s - (self.table.decision_s if strict else elapsed))
        b = self.table.lookup_pooled(abs(z), self.params.min_samples)
        if b.n < self.params.min_samples:  # the whole table is too small
            return None
        return (Side.UP if move > 0 else Side.DOWN), b.rate, z

    def decide(self, ctx: RoundContext) -> Side | None:
        est = self.estimate(ctx)
        if est is None or ctx.quote is None:
            return None
        follow, p, _ = est
        fade = Side.DOWN if follow is Side.UP else Side.UP
        fee = self.params.fee_bps / 10_000
        best: tuple[Side, float] | None = None
        for side, q in ((follow, p), (fade, 1.0 - p)):
            price = entry_price(side, ctx.quote)
            if price is None or not 0 < price < 1:
                continue
            ev = q / price - 1 - fee
            if ev >= self.params.min_edge and (best is None or ev > best[1]):
                best = (side, ev)
        return None if best is None else best[0]

    def accept_retry(self, ctx: RoundContext, side: Side, price: float, max_price: float) -> str | None:
        """Re-estimate the win probability NOW and require the same edge at the NEW price."""
        est = self.estimate(ctx, strict=False)
        if est is None:
            return "cannot re-estimate the probability now"
        follow, p, _ = est
        q = p if side is follow else 1.0 - p
        ev = q / price - 1 - self.params.fee_bps / 10_000
        if ev < self.params.min_edge:
            return f"no edge left at {price:.3f} (win prob now ~{q:.2f}, expected {ev:+.1%})"
        return None


@dataclass(frozen=True, slots=True)
class MomentumValueParams:
    fee_rate: float = 0.02  # Predict.fun feeRateBps 200, applied with the real formula (pricing.buy_fee_fraction)
    min_ev: float = 0.0  # required expected profit per 1$ after the fee
    min_samples: int = 100  # ignore |z| buckets calibrated on fewer rounds than this
    min_move_bps: float = 1.0  # same floor as momentum


@dataclass(frozen=True, slots=True)
class MomentumValue:
    """Momentum, but only when the price is below the calibrated chance that the move holds.

    At the decision second: the side the move points to (like `momentum`), its win probability p from
    the |z| calibration table (like `value`), and its ask. Buy it only if
        p / price * (1 - fee fraction) - 1 >= min_ev
    Never bets against the move: in paper trading the `value` fades (cheap long shots) carried most of
    its variance. On 420 momentum paper trades this filter kept 172, +14.1%/trade vs +3.0% unfiltered.
    """

    table: CalibrationTable
    params: MomentumValueParams = MomentumValueParams()
    name: str = "momentum_value"
    needs_book: bool = False
    needs_odds: bool = True

    def __repr__(self) -> str:  # short and stable: used as the paper-trading history label
        return f"MomentumValue({self.table.label}, {self.params})"

    def _edge(self, ctx: RoundContext, strict: bool) -> tuple[Side, float, float] | str:
        """(follow side, win prob, expected profit at the current ask) or the reason there is none."""
        if abs(ctx.move_bps) < self.params.min_move_bps:
            return "move too small"
        est = Value(self.table, ValueParams(min_samples=self.params.min_samples)).estimate(ctx, strict=strict)
        if est is None:
            return "cannot estimate the win probability"
        side, p, _ = est
        if ctx.quote is None:
            return "no quote"
        price = entry_price(side, ctx.quote)
        if price is None or not 0 < price < 1:
            return "nothing to buy on that side"
        return side, p, expected_profit(p, price, self.params.fee_rate)

    def decide(self, ctx: RoundContext) -> Side | None:
        e = self._edge(ctx, strict=True)
        if isinstance(e, str):
            return None
        side, _, ev = e
        return side if ev >= self.params.min_ev else None

    def accept_retry(self, ctx: RoundContext, side: Side, price: float, max_price: float) -> str | None:
        """Retry = fresh decision: the move must still point to `side` and the NEW price must keep the edge."""
        if abs(ctx.move_bps) < self.params.min_move_bps or (ctx.move_bps > 0) != (side is Side.UP):
            return f"move no longer points {side.value} ({ctx.move_bps:+.2f} bps)"
        est = Value(self.table, ValueParams(min_samples=self.params.min_samples)).estimate(ctx, strict=False)
        if est is None:
            return "cannot re-estimate the probability now"
        ev = expected_profit(est[1], price, self.params.fee_rate)
        if ev < self.params.min_ev:
            return f"no edge left at {price:.3f} (win prob now ~{est[1]:.2f}, expected {ev:+.1%})"
        return None


def accept_retry(strategy: Strategy, ctx: RoundContext, side: Side, price: float, max_price: float) -> str | None:
    """None if a retry at `price` is still a good bet for `strategy`, else why not."""
    check = getattr(strategy, "accept_retry", None)
    if check is not None:
        return check(ctx, side, price, max_price)
    return None if price <= max_price else f"price {price:.3f} above the retry cap {max_price:.2f}"


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
        case "value" | "momentum_value":
            raise ValueError(f"the {name} strategy needs a calibration table: build it with its table")
    raise ValueError(f"unknown strategy {name!r} (momentum | reversal | always_up | book | momentum_book | value)")
