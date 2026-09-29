from bot.backtest.engine import run_backtest
from bot.backtest.stats import binomial_p_value, summarize, wilson_interval
from bot.backtest.strategy import AlwaysUp, Momentum
from bot.models import Kline
from bot.settlement import SettlementRule

STEP = 300_000
T0 = 1_800_000_000_000 // STEP * STEP


def k(interval: str, open_time: int, o: float, c: float) -> Kline:
    ms = 1000 if interval == "1s" else STEP
    return Kline("BTCUSDT", interval, open_time, open_time + ms - 1, o, max(o, c), min(o, c), c, 1, 1, 1, 1, 1)


def seconds_for(round_open: int, start: float, mid: float, end_secs: int = 300) -> list[Kline]:
    """1s candles drifting linearly from `start` to `mid` over the first 270s."""
    out = []
    for i in range(end_secs):
        px = start + (mid - start) * min(i + 1, 270) / 270
        out.append(k("1s", round_open + i * 1000, px, px))
    return out


def test_momentum_wins_when_move_continues():
    r = k("5m", T0, 100.0, 101.0)  # closes UP
    secs = seconds_for(T0, 100.0, 100.5)  # up 50bps at decision
    res = run_backtest([r], secs, Momentum(1.0), SettlementRule.CLOSE_GTE_OPEN, decision_s=270, payout_ratio=0.9)
    assert len(res.trades) == 1 and res.trades[0].pnl == 0.9


def test_momentum_loses_when_move_reverses():
    r = k("5m", T0, 100.0, 99.0)  # closes DOWN
    res = run_backtest([r], seconds_for(T0, 100.0, 100.5), Momentum(1.0), SettlementRule.CLOSE_GTE_OPEN)
    assert res.trades[0].pnl == -1.0


def test_small_move_is_skipped():
    r = k("5m", T0, 100.0, 101.0)
    res = run_backtest([r], seconds_for(T0, 100.0, 100.001), Momentum(1.0), SettlementRule.CLOSE_GTE_OPEN)
    assert not res.trades and res.rounds_skipped_by_strategy == 1


def test_missing_seconds_skip_round():
    r = k("5m", T0, 100.0, 101.0)
    res = run_backtest([r], [], AlwaysUp(), SettlementRule.CLOSE_GTE_OPEN)
    assert res.rounds_skipped_no_data == 1 and not res.trades


def test_strategy_cannot_see_after_decision():
    seen = []

    class Spy:
        name = "spy"

        def decide(self, ctx):
            seen.append(max(c.close_time for c in ctx.recent))
            return None

    run_backtest([k("5m", T0, 100.0, 101.0)], seconds_for(T0, 100.0, 100.5), Spy(), SettlementRule.CLOSE_GTE_OPEN)
    assert seen[0] < T0 + 270_000


def test_tie_void_refunds():
    r = k("5m", T0, 100.0, 100.0)
    res = run_backtest([r], seconds_for(T0, 100.0, 100.5), Momentum(1.0), SettlementRule.TIE_VOID)
    assert res.trades[0].pnl == 0.0


def test_stats_sanity():
    assert binomial_p_value(50, 100, 0.5) > 0.4
    assert binomial_p_value(70, 100, 0.5) < 0.001
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
    r = k("5m", T0, 100.0, 101.0)
    res = run_backtest([r], seconds_for(T0, 100.0, 100.5), AlwaysUp(), SettlementRule.CLOSE_GTE_OPEN)
    s = summarize(res, 0.9)
    assert s["bets"] == 1 and s["wins"] == 1 and s["pnl_units"] == 0.9
