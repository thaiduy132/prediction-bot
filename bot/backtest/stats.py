"""Turn trades into numbers that say whether a result is skill or luck."""

from __future__ import annotations

import math

from bot.backtest.engine import BacktestResult


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def binomial_p_value(wins: int, n: int, p0: float) -> float:
    """One-sided exact P(X >= wins) for X ~ Binomial(n, p0): chance of doing this well by luck."""
    if n == 0:
        return 1.0
    if p0 <= 0:
        return 1.0 if wins == 0 else 0.0
    if p0 >= 1:
        return 1.0
    lp, lq = math.log(p0), math.log1p(-p0)
    logs = [
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * lp + (n - k) * lq
        for k in range(wins, n + 1)
    ]
    m = max(logs)
    return min(1.0, math.exp(m) * sum(math.exp(x - m) for x in logs))


def poisson_binomial_p_value(wins: int, probs: list[float]) -> float:
    """One-sided P(X >= wins) when bet i wins independently with probability probs[i].

    Exact by dynamic programming up to 2000 bets; beyond that a normal approximation.
    """
    n = len(probs)
    if n == 0:
        return 1.0
    if n > 2000:
        mean = sum(probs)
        sd = math.sqrt(sum(p * (1 - p) for p in probs))
        return 0.5 * math.erfc((wins - 0.5 - mean) / (sd * math.sqrt(2))) if sd > 0 else float(wins <= mean)
    dist = [1.0]
    for p in probs:
        nxt = [0.0] * (len(dist) + 1)
        for k, v in enumerate(dist):
            nxt[k] += v * (1 - p)
            nxt[k + 1] += v * p
        dist = nxt
    return min(1.0, sum(dist[wins:]))


def summarize(res: BacktestResult, payout_ratio: float, fee_bps: float = 0.0) -> dict[str, object]:
    trades = res.trades
    decided = [t for t in trades if t.outcome.value != "VOID"]  # VOIDs are refunded: no info on skill
    won = [t for t in decided if t.outcome.value == t.side.value]
    wins, n = len(won), len(decided)
    losses, voids = n - wins, len(trades) - n
    odds_mode = bool(trades) and all(t.entry_price is not None for t in trades)
    if odds_mode:
        # A bet at price p only breaks even if it wins with probability p * (1 + fee).
        q = [min(1.0, t.entry_price * (1 + fee_bps / 10_000)) for t in decided if t.entry_price is not None]
        breakeven = sum(q) / len(q) if q else 0.0
        p_value = poisson_binomial_p_value(wins, q)
    else:
        breakeven = 1 / (1 + payout_ratio)
        p_value = binomial_p_value(wins, n, breakeven)
    lo, hi = wilson_interval(wins, n)

    equity = peak = max_dd = 0.0
    for t in trades:
        equity += t.pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    total = res.rounds_total
    up = res.outcomes.get("UP", 0)
    return {
        "rounds_total": total,
        "rounds_skipped_no_data": res.rounds_skipped_no_data,
        "rounds_skipped_no_book": res.rounds_skipped_no_book,
        "rounds_skipped_no_odds": res.rounds_skipped_no_odds,
        "rounds_skipped_price": res.rounds_skipped_price,
        "rounds_skipped_by_strategy": res.rounds_skipped_by_strategy,
        "market_up_rate": round(up / total, 4) if total else None,
        "bets": len(trades),
        "wins": wins,
        "losses": losses,
        "voids": voids,
        "win_rate": round(wins / n, 4) if n else None,
        "win_rate_ci95": [round(lo, 4), round(hi, 4)],
        "breakeven_win_rate": round(breakeven, 4),
        "pnl_units": round(sum(t.pnl for t in trades), 3),
        "roi_per_bet": round(sum(t.pnl for t in trades) / len(trades), 4) if trades else None,
        "max_drawdown_units": round(max_dd, 3),
        "p_value_vs_breakeven": round(p_value, 4),
        **({"avg_entry_price": round(sum(t.entry_price for t in trades if t.entry_price is not None) / len(trades), 4),
            "fee_bps": fee_bps} if odds_mode else {}),
    }
