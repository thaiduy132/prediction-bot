import pytest

from bot.backtest.calibration import CalibrationTable, empty_buckets
from bot.backtest.engine import run_backtest
from bot.backtest.pricing import buy_fee_fraction, expected_profit
from bot.backtest.strategy import MomentumValue, MomentumValueParams, RoundContext, Side
from bot.data.odds_store import OddsSample
from bot.models import Kline
from bot.settlement import SettlementRule

STEP = 300_000
T0 = 1_800_000_000_000 // STEP * STEP


def k1s(t: int, px: float) -> Kline:
    return Kline("BTCUSDT", "1s", t, t + 999, px, px, px, px, 1, 1, 1, 1, 1)


def zigzag(start: int, n: int) -> list[Kline]:
    return [k1s(start + i * 1000, 100.0 + (0.01 if i % 2 else -0.01)) for i in range(n)]  # sigma = 2 bps


def table(follow_win: float = 0.72) -> CalibrationTable:
    b = empty_buckets()
    for x in b:
        x.n, x.wins = 1000, int(follow_win * 1000) - 1  # rate = (wins+1)/(n+2) ~ follow_win
    return CalibrationTable(60, 300, "close_gte_open", "BTCUSDT", "2026-09-15T00:00:00Z", "2026-09-25T00:00:00Z", b)


def quote(bid: float, ask: float, ts: int = T0 + 59_500) -> OddsSample:
    return OddsSample(T0, ts, 1, bid, ask, 10, 10, 50, 50)


def ctx(move_bps: float, q: OddsSample | None, second: int = 60) -> RoundContext:
    dt = T0 + second * 1000
    last = 100 * (1 + move_bps / 1e4)
    recent = [k1s(T0 + i * 1000, 100.0) for i in range(second - 1)] + [k1s(dt - 1000, last)]
    return RoundContext(T0, 100.0, dt, recent, (), zigzag(dt - 300_000, 300), q)


def test_real_fee_formula_matches_binance_quotes():
    # measured 2026-09-30: UP at 0.10 -> fee 0.2 of 10 shares; DOWN at 0.90 -> fee 0.00247 of 1.111 shares
    assert buy_fee_fraction(0.10) == pytest.approx(0.02)
    assert buy_fee_fraction(0.90) * (1 / 0.90) == pytest.approx(0.002469, abs=1e-5)
    assert expected_profit(0.72, 0.60) == pytest.approx(0.72 / 0.60 * (1 - 0.02 * 0.4 / 0.6) - 1)


def test_buys_the_leader_when_it_is_cheaper_than_its_win_probability():
    s = MomentumValue(table(0.72))
    assert s.decide(ctx(+20, quote(0.62, 0.64))) is Side.UP  # 0.72 at 0.64: +11.0%
    assert s.decide(ctx(-20, quote(0.36, 0.38))) is Side.DOWN  # DOWN costs 1 - 0.36 = 0.64


def test_skips_when_the_leader_is_already_too_expensive():
    s = MomentumValue(table(0.72))
    assert s.decide(ctx(+20, quote(0.73, 0.75))) is None  # 0.72 at 0.75: negative


def test_never_fades_the_move_even_if_the_other_side_is_cheap():
    s = MomentumValue(table(0.72))
    # leader UP costs 0.95 (bad); DOWN costs 0.07 with a 28% chance: value would buy it, momentum_value must not
    assert s.decide(ctx(+20, quote(0.93, 0.95))) is None


def test_min_ev_threshold_and_tiny_moves():
    assert MomentumValue(table(0.72), MomentumValueParams(min_ev=0.15)).decide(ctx(+20, quote(0.62, 0.64))) is None
    assert MomentumValue(table(0.72)).decide(ctx(+0.5, quote(0.40, 0.42))) is None  # below 1 bps


def test_missing_inputs_skip():
    s = MomentumValue(table(0.72))
    assert s.decide(ctx(+20, None)) is None
    c = ctx(+20, quote(0.62, 0.64))
    assert s.decide(RoundContext(c.round_open_time, c.round_open, c.decision_time, c.recent, (), (), c.quote)) is None


def test_retry_is_a_fresh_edge_check():
    s = MomentumValue(table(0.80))
    later = ctx(+20, quote(0.70, 0.71), second=75)
    assert s.accept_retry(later, Side.UP, 0.71, 0.85) is None  # still +10% at the higher price
    assert "no edge left" in s.accept_retry(later, Side.UP, 0.84, 0.85)
    assert "no longer points UP" in s.accept_retry(ctx(-20, quote(0.3, 0.31), second=75), Side.UP, 0.4, 0.85)


def test_repr_is_short_for_paper_labels():
    r = repr(MomentumValue(table()))
    assert r.startswith("MomentumValue(BTCUSDT t=60s 2026-09-15..2026-09-25") and len(r) < 200


def test_runs_through_the_backtest_engine_with_real_fee_pnl():
    c = ctx(+20, None)
    secs = sorted({k.open_time: k for k in list(c.lookback) + list(c.recent) +
                   [k1s(T0 + i * 1000, 100.3) for i in range(60, 300)]}.values(), key=lambda k: k.open_time)
    r5 = Kline("BTCUSDT", "5m", T0, T0 + STEP - 1, 100.0, 100.3, 100.0, 100.3, 1, 1, 1, 1, 1)
    res = run_backtest([r5], secs, MomentumValue(table(0.72)), SettlementRule.CLOSE_GTE_OPEN, decision_s=60,
                       odds_samples=[quote(0.62, 0.64)])
    assert len(res.trades) == 1 and res.trades[0].side is Side.UP and res.trades[0].entry_price == 0.64


def test_sparse_strong_buckets_are_pooled_upward():
    from bot.backtest.calibration import Bucket
    b = empty_buckets()
    counts = {0: (761, 414), 1: (709, 431), 2: (552, 381), 3: (357, 260), 4: (197, 140), 5: (138, 116),
              6: (86, 66), 7: (18, 17), 8: (2, 1), 9: (1, 0)}  # the real t=60 table shape
    for i, (n, w) in counts.items():
        b[i].n, b[i].wins = n, w
    t = CalibrationTable(60, 300, "close_gte_open", "BTCUSDT", "a", "b", b)
    assert t.lookup_pooled(0.9, 100) is t.buckets[3]  # enough rounds: unchanged
    p = t.lookup_pooled(1.62, 100)  # 86 rounds alone -> pooled with all stronger: 86+18+2+1
    assert (p.lo, p.n, p.wins) == (1.5, 107, 84)
    assert (t.lookup_pooled(3.5, 100).lo, t.lookup_pooled(3.5, 100).n) == (1.5, 107)  # 1 round; pooled down to >= 100
    assert Bucket(0, None, 107, 84).rate < t.buckets[7].rate  # conservative vs the strongest bucket


def test_retry_with_a_strong_move_is_judged_not_dropped():
    # the 02/10 10:30 case: decided at |z|~1, retry at |z|~1.6 with the price at 0.84
    b = empty_buckets()
    for i, (n, w) in {3: (357, 260), 6: (86, 66), 7: (18, 17), 8: (2, 1), 9: (1, 0)}.items():
        b[i].n, b[i].wins = n, w
    s = MomentumValue(CalibrationTable(60, 300, "close_gte_open", "BTCUSDT", "a", "b", b))
    strong = ctx(+50, quote(0.83, 0.84), second=74)  # zigzag sigma 2 bps, 226 s left -> |z| ~ 1.66
    why = s.accept_retry(strong, Side.UP, 0.84, 0.85)
    assert why is not None and "no edge left" in why  # a real answer (p ~0.79 < 0.84), not "cannot estimate"
    assert s.accept_retry(strong, Side.UP, 0.70, 0.85) is None  # cheaper: worth buying
