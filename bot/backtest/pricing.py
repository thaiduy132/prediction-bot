"""Turn a recorded quote into an entry price and a bet's result.

Odds come from Predict.fun (the venue behind Binance's prediction markets); orders and
fees are Binance's. One share pays 1 if it wins, so buying at price p with stake 1 buys 1/p
shares. Buying UP costs the ask of the YES book; buying DOWN costs 1 - (best YES bid).
"""

from __future__ import annotations

from bot.backtest.strategy import Side
from bot.data.odds_store import OddsSample
from bot.settlement import Outcome


def entry_price(side: Side, q: OddsSample) -> float | None:
    """Price paid per share to buy `side` right now, or None if that side has no offer."""
    if side is Side.UP:
        return q.up_ask
    return None if q.up_bid is None else 1.0 - q.up_bid


def bet_pnl(side: Side, outcome: Outcome, price: float, fee_bps: float = 0.0) -> float:
    """Profit per 1 unit staked. `fee_bps` is charged on the stake of every executed bet.

    VOID (a tie under the `tie_void` rule) refunds the stake, so only the fee is lost.
    """
    fee = fee_bps / 10_000
    if outcome is Outcome.VOID:
        return -fee
    return (1.0 / price - 1.0 - fee) if outcome.value == side.value else (-1.0 - fee)
