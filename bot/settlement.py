"""Round settlement: decide whether a 5m round resolved UP, DOWN or VOID.

The default rule (close >= open -> UP) is an ASSUMPTION. Every Binance Prediction
Market contract must be checked against its official rules: tie handling, the exact
reference price (spot last trade, index, mark...), and the exact start/end instants.
The rule is therefore selected from config, never hard-coded at call sites.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from bot.models import Kline
from bot.timeutil import interval_ms


class Outcome(StrEnum):
    UP = "UP"
    DOWN = "DOWN"
    VOID = "VOID"  # round cancelled / refunded


class SettlementRule(StrEnum):
    CLOSE_GTE_OPEN = "close_gte_open"  # tie -> UP   (default)
    CLOSE_GT_OPEN = "close_gt_open"  # tie -> DOWN
    TIE_VOID = "tie_void"  # tie -> VOID (stakes refunded)


class SettlementError(ValueError):
    pass


def settle(open_price: float, close_price: float, rule: SettlementRule) -> Outcome:
    if not (open_price > 0 and close_price > 0):
        raise SettlementError(f"non-positive price: open={open_price} close={close_price}")
    if close_price > open_price:
        return Outcome.UP
    if close_price < open_price:
        return Outcome.DOWN
    # exact tie
    match rule:
        case SettlementRule.CLOSE_GTE_OPEN:
            return Outcome.UP
        case SettlementRule.CLOSE_GT_OPEN:
            return Outcome.DOWN
        case SettlementRule.TIE_VOID:
            return Outcome.VOID
    raise SettlementError(f"unknown settlement rule: {rule!r}")


def settle_kline(k: Kline, rule: SettlementRule, round_interval: str = "5m") -> Outcome:
    """Settle a round from its candle. Refuses candles that are not final or misaligned."""
    step = interval_ms(round_interval)
    if k.interval != round_interval:
        raise SettlementError(f"expected {round_interval} candle, got {k.interval}")
    if not k.is_closed:
        raise SettlementError(f"candle {k.open_time} is not closed yet")
    if k.open_time % step != 0:
        raise SettlementError(f"candle open_time {k.open_time} not aligned to {round_interval}")
    if k.close_time != k.open_time + step - 1:
        raise SettlementError(f"candle {k.open_time} has unexpected close_time {k.close_time}")
    return settle(k.open, k.close, rule)


def cross_check_with_1m(k5: Kline, k1s: Sequence[Kline], rel_tol: float = 0.0) -> list[str]:
    """Verify a 5m candle against its five 1m candles. Returns a list of problems (empty = OK).

    Binance builds both from the same trades, so open/close must match exactly and
    high/low/volume must agree. A mismatch means corrupt or partial data.
    """
    problems: list[str] = []
    step1 = interval_ms("1m")
    n = interval_ms(k5.interval) // step1
    expected = [k5.open_time + i * step1 for i in range(n)]
    got = sorted(k1s, key=lambda k: k.open_time)
    if [k.open_time for k in got] != expected:
        return [f"1m candles for {k5.open_time} incomplete: {[k.open_time for k in got]}"]

    def differs(a: float, b: float) -> bool:
        return abs(a - b) > rel_tol * max(abs(a), abs(b))

    if differs(k5.open, got[0].open):
        problems.append(f"open mismatch {k5.open} vs 1m {got[0].open}")
    if differs(k5.close, got[-1].close):
        problems.append(f"close mismatch {k5.close} vs 1m {got[-1].close}")
    if differs(k5.high, max(k.high for k in got)):
        problems.append("high mismatch")
    if differs(k5.low, min(k.low for k in got)):
        problems.append("low mismatch")
    if abs(k5.volume - sum(k.volume for k in got)) > 1e-6 * max(1.0, k5.volume):
        problems.append("volume mismatch")
    return problems
