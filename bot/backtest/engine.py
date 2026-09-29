"""Round-by-round backtest over cached candles. Pure: no I/O, no clock."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from bot.backtest.pricing import bet_pnl, entry_price
from bot.backtest.strategy import RoundContext, Side, Strategy
from bot.data.book_store import BookSample
from bot.data.odds_store import OddsSample
from bot.models import Kline
from bot.settlement import Outcome, SettlementRule, settle_kline
from bot.timeutil import interval_ms

# Skip a round when too many of the 1s candles before the decision are missing.
MIN_SECONDS_COVERAGE = 0.9
# Book samples handed to a strategy, and how stale the newest one may be at decision time.
BOOK_LOOKBACK_S = 60
BOOK_MAX_STALE_MS = 5_000
# The odds quote used for an entry must be this fresh (we poll about once a second).
ODDS_MAX_STALE_MS = 3_000


@dataclass(frozen=True, slots=True)
class Trade:
    round_open_time: int
    side: Side
    outcome: Outcome
    pnl: float  # in units of stake
    move_bps: float  # move at decision time
    entry_price: float | None = None  # odds mode: price paid per share (a probability)


@dataclass(slots=True)
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    rounds_total: int = 0
    rounds_skipped_no_data: int = 0
    rounds_skipped_no_book: int = 0
    rounds_skipped_no_odds: int = 0
    rounds_skipped_price: int = 0  # entry price above max_entry_price
    rounds_skipped_by_strategy: int = 0
    outcomes: dict[str, int] = field(default_factory=dict)  # every round's real outcome


@dataclass(frozen=True, slots=True)
class Entry:
    side: Side
    price: float | None  # per-share price paid (odds mode); None in fixed-payout mode
    move_bps: float  # price move from the round open at decision time


def decide_entry(
    strategy: Strategy,
    round_open_time: int,
    round_open: float,
    decision_time: int,
    recent: Sequence[Kline],
    book: Sequence[BookSample],
    odds: Sequence[OddsSample] | None,
    max_entry_price: float = 1.0,
) -> tuple[Entry | None, str | None]:
    """The single place that decides whether and what to bet, shared by backtest and paper trading.

    Returns (entry, None), or (None, reason) with reason one of no_data, no_book, by_strategy,
    no_odds, price. `odds` are the quotes recorded for THIS round, or None for fixed-payout mode.
    Everything passed in must already be known at `decision_time`.
    """
    decision_s = (decision_time - round_open_time) // 1000
    if len(recent) < MIN_SECONDS_COVERAGE * decision_s or recent[-1].open_time != decision_time - 1000:
        return None, "no_data"
    if getattr(strategy, "needs_book", False) and (not book or book[-1].ts < decision_time - BOOK_MAX_STALE_MS):
        return None, "no_book"

    ctx = RoundContext(round_open_time, round_open, decision_time, recent, book)
    side = strategy.decide(ctx)
    if side is None:
        return None, "by_strategy"
    if odds is None:
        return Entry(side, None, ctx.move_bps), None

    # Odds mode: pay the quote that was known at decision time, never a later one.
    known = [q for q in odds if decision_time - ODDS_MAX_STALE_MS <= q.ts <= decision_time]
    price = entry_price(side, known[-1]) if known else None
    if price is None or not 0 < price < 1:
        return None, "no_odds"
    if price > max_entry_price:
        return None, "price"
    return Entry(side, price, ctx.move_bps), None


def run_backtest(
    rounds: Sequence[Kline],
    seconds: Sequence[Kline],
    strategy: Strategy,
    rule: SettlementRule,
    round_interval: str = "5m",
    decision_s: int = 270,
    payout_ratio: float = 0.9,
    book_samples: Sequence[BookSample] = (),
    odds_samples: Sequence[OddsSample] | None = None,
    fee_bps: float = 0.0,
    max_entry_price: float = 1.0,
) -> BacktestResult:
    """Without `odds_samples`, `payout_ratio` is the net profit per unit stake on a win (a loss
    costs 1, VOID refunds). With `odds_samples` (recorded Predict.fun quotes) each bet is priced
    at the quote known at decision time and `payout_ratio` is ignored."""
    step = interval_ms(round_interval)
    if not 0 < decision_s * 1000 < step:
        raise ValueError(f"decision_s must be inside the round (0 < {decision_s} < {step // 1000})")
    by_open = {k.open_time: k for k in seconds if k.is_closed}
    book_by_ts = {b.ts: b for b in book_samples}
    odds_by_round: dict[int, list[OddsSample]] = {}
    for q in odds_samples or ():
        odds_by_round.setdefault(q.round_start, []).append(q)
    res = BacktestResult()

    for r in rounds:
        outcome = settle_kline(r, rule, round_interval)
        res.rounds_total += 1
        res.outcomes[outcome.value] = res.outcomes.get(outcome.value, 0) + 1

        decision_time = r.open_time + decision_s * 1000
        recent = [by_open[t] for t in range(r.open_time, decision_time, 1000) if t in by_open]
        # A book sample for second ts is only knowable at ts + 1000, hence ts < decision_time.
        book = [book_by_ts[t] for t in range(decision_time - BOOK_LOOKBACK_S * 1000, decision_time, 1000)
                if t in book_by_ts]
        entry, skip = decide_entry(
            strategy, r.open_time, r.open, decision_time, recent, book,
            None if odds_samples is None else odds_by_round.get(r.open_time, ()), max_entry_price)
        if entry is None:
            setattr(res, f"rounds_skipped_{skip}", getattr(res, f"rounds_skipped_{skip}") + 1)
            continue

        if entry.price is None:  # fixed-payout mode
            if outcome is Outcome.VOID:
                pnl = 0.0
            else:
                pnl = payout_ratio if outcome.value == entry.side.value else -1.0
        else:
            pnl = bet_pnl(entry.side, outcome, entry.price, fee_bps)
        res.trades.append(Trade(r.open_time, entry.side, outcome, pnl, entry.move_bps, entry.price))
    return res
