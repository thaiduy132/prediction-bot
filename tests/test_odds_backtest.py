import pytest

from bot.backtest.engine import run_backtest
from bot.backtest.pricing import bet_pnl, entry_price
from bot.backtest.stats import poisson_binomial_p_value, summarize
from bot.backtest.strategy import AlwaysUp, Momentum, Side
from bot.data.odds_store import OddsSample
from bot.models import Kline
from bot.settlement import Outcome, SettlementRule

STEP = 300_000
T0 = 1_800_000_000_000 // STEP * STEP
DECISION = T0 + 270_000


def kline(interval: str, t: int, o: float, c: float) -> Kline:
    ms = 1000 if interval == "1s" else STEP
    return Kline("BTCUSDT", interval, t, t + ms - 1, o, max(o, c), min(o, c), c, 1, 1, 1, 1, 1)


def secs(px_end: float) -> list[Kline]:
    px = lambda i: 100 + (px_end - 100) * min(i + 1, 270) / 270  # noqa: E731
    return [kline("1s", T0 + i * 1000, px(i), px(i)) for i in range(300)]


def quote(ts: int, bid: float | None, ask: float | None, start: int = T0) -> OddsSample:
    return OddsSample(start, ts, 1, bid, ask, 10, 10, 50, 50)


def run(strategy, close: float, odds, fee_bps=0.0, max_price=1.0, rule=SettlementRule.CLOSE_GTE_OPEN):
    return run_backtest([kline("5m", T0, 100.0, close)], secs(100.5), strategy, rule,
                        odds_samples=odds, fee_bps=fee_bps, max_entry_price=max_price)


def test_entry_prices():
    q = quote(0, 0.40, 0.45)
    assert entry_price(Side.UP, q) == 0.45
    assert entry_price(Side.DOWN, q) == pytest.approx(0.60)  # 1 - best YES bid
    assert entry_price(Side.DOWN, quote(0, None, 0.45)) is None


def test_bet_pnl_win_loss_void_fee():
    assert bet_pnl(Side.UP, Outcome.UP, 0.5) == pytest.approx(1.0)  # 1/0.5 - 1
    assert bet_pnl(Side.UP, Outcome.DOWN, 0.5) == -1.0
    assert bet_pnl(Side.UP, Outcome.VOID, 0.5, fee_bps=100) == pytest.approx(-0.01)
    assert bet_pnl(Side.UP, Outcome.UP, 0.5, fee_bps=100) == pytest.approx(0.99)


def test_win_is_priced_at_the_quote_not_fixed_payout():
    res = run(AlwaysUp(), 101.0, [quote(DECISION - 500, 0.79, 0.80)])
    t = res.trades[0]
    assert t.entry_price == 0.80 and t.pnl == pytest.approx(0.25)  # 1/0.8 - 1, not 0.9


def test_expensive_favourite_loses_money_on_a_miss():
    res = run(Momentum(1.0), 99.0, [quote(DECISION - 500, 0.94, 0.95)])
    assert res.trades[0].pnl == -1.0


def test_uses_latest_quote_at_or_before_decision_never_later():
    odds = [quote(DECISION - 2000, 0.10, 0.30), quote(DECISION - 500, 0.20, 0.40), quote(DECISION + 500, 0.90, 0.99)]
    assert run(AlwaysUp(), 101.0, odds).trades[0].entry_price == 0.40


def test_stale_or_missing_quote_skips_round():
    assert run(AlwaysUp(), 101.0, [quote(DECISION - 10_000, 0.4, 0.5)]).rounds_skipped_no_odds == 1
    assert run(AlwaysUp(), 101.0, []).rounds_skipped_no_odds == 1
    assert run(AlwaysUp(), 101.0, [quote(DECISION - 500, 0.4, 0.5, start=T0 - STEP)]).rounds_skipped_no_odds == 1


def test_quote_from_another_round_is_not_used():
    res = run(AlwaysUp(), 101.0, [quote(DECISION - 500, 0.4, 0.5, start=T0 + STEP)])
    assert not res.trades and res.rounds_skipped_no_odds == 1


def test_one_sided_book_skips_that_side():
    # only asks quoted: UP can be bought, DOWN cannot
    odds = [quote(DECISION - 500, None, 0.5)]
    assert len(run(AlwaysUp(), 101.0, odds).trades) == 1
    assert run(Momentum(1.0, contrarian=True), 101.0, odds).rounds_skipped_no_odds == 1  # reversal -> DOWN


def test_max_entry_price_filter():
    res = run(AlwaysUp(), 101.0, [quote(DECISION - 500, 0.94, 0.95)], max_price=0.9)
    assert not res.trades and res.rounds_skipped_price == 1


def test_tie_void_refunds_and_fee_applies():
    res = run(AlwaysUp(), 100.0, [quote(DECISION - 500, 0.4, 0.5)], fee_bps=200, rule=SettlementRule.TIE_VOID)
    assert res.trades[0].pnl == pytest.approx(-0.02)


def test_summary_breakeven_follows_prices_and_fee():
    res = run(AlwaysUp(), 101.0, [quote(DECISION - 500, 0.79, 0.80)], fee_bps=100)
    s = summarize(res, payout_ratio=0.9, fee_bps=100)
    assert s["avg_entry_price"] == 0.8 and s["breakeven_win_rate"] == pytest.approx(0.808, abs=1e-3)
    assert s["pnl_units"] == pytest.approx(0.24)


def test_poisson_binomial_matches_binomial_case():
    from bot.backtest.stats import binomial_p_value

    assert poisson_binomial_p_value(60, [0.5] * 100) == pytest.approx(binomial_p_value(60, 100, 0.5), rel=1e-9)
    assert poisson_binomial_p_value(0, [0.3, 0.9]) == pytest.approx(1.0)
    assert poisson_binomial_p_value(2, [0.3, 0.9]) == pytest.approx(0.27)
