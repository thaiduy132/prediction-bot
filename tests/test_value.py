import math

import pytest

from bot.backtest.calibration import (
    Bucket, CalibrationTable, build_table, empty_buckets, load_table, save_tables, sigma_bps, z_score,
)
from bot.backtest.engine import decide_entry, run_backtest
from bot.backtest.strategy import RoundContext, Side, Value, ValueParams
from bot.data.odds_store import OddsSample
from bot.models import Kline
from bot.paper.trader import PaperTrader
from bot.settlement import SettlementRule

STEP = 300_000
T0 = 1_800_000_000_000 // STEP * STEP


def k1s(t: int, px: float) -> Kline:
    return Kline("BTCUSDT", "1s", t, t + 999, px, px, px, px, 1, 1, 1, 1, 1)


def k5m(t: int, o: float, c: float, closed: bool = True) -> Kline:
    return Kline("BTCUSDT", "5m", t, t + STEP - 1, o, max(o, c), min(o, c), c, 1, 1, 1, 1, 1, closed)


def zigzag(start: int, n: int, base: float = 100.0, amp: float = 0.01) -> list[Kline]:
    """n 1s candles alternating +-amp around base: sigma = amp/base*1e4 bps, no drift."""
    return [k1s(start + i * 1000, base + (amp if i % 2 else -amp)) for i in range(n)]


def table(rate_by_bucket: dict[int, tuple[int, int]], decision_s: int = 60) -> CalibrationTable:
    b = empty_buckets()
    for i, (n, w) in rate_by_bucket.items():
        b[i].n, b[i].wins = n, w
    return CalibrationTable(decision_s, 300, "close_gte_open", "BTCUSDT", "2026-01-01T00:00:00Z",
                            "2026-01-10T00:00:00Z", b)


def quote(bid: float | None, ask: float | None, ts: int = T0 + 59_500) -> OddsSample:
    return OddsSample(T0, ts, 1, bid, ask, 10, 10, 50, 50)


# ---- sigma / z -------------------------------------------------------------------------

def test_sigma_of_zigzag_and_missing_data():
    s = sigma_bps(zigzag(0, 300))
    assert s == pytest.approx(0.02 / 100 * 1e4, rel=1e-3)  # each step moves 2*amp
    assert sigma_bps(zigzag(0, 100)) is None  # too little of the window
    assert sigma_bps([k1s(0, 100.0)] * 300) is None  # zero volatility is not usable


def test_z_score_scales_with_time_left():
    assert z_score(4.0, 1.0, 16) == 1.0
    assert z_score(4.0, 1.0, 4) == 2.0


def test_bucket_rate_is_shrunk():
    assert Bucket(0, 1, 0, 0).rate == 0.5
    assert Bucket(0, 1, 8, 8).rate == pytest.approx(0.9)


# ---- build / save / load -------------------------------------------------------------------

def rounds_series(n_rounds: int, continue_move: bool) -> list[Kline]:
    """Rounds that drift up +5bps by second 60, then either keep going up or fall back below open."""
    out = zigzag(T0 - 300_000, 300)  # lookback for the first round
    for r in range(n_rounds):
        a = T0 + r * STEP
        for i in range(300):
            if i < 60:
                px = 100 * (1 + 5e-4 * (i + 1) / 60)
            else:
                px = 100 * (1 + 5e-4 * (1 + (i - 59) / 240)) if continue_move else 100 * (1 + 5e-4 - 1e-3 * (i - 59) / 240)
            px += 0.01 if i % 2 else -0.01  # keep sigma > 0
            out.append(k1s(a + i * 1000, px))
    return out


def test_build_table_counts_follow_wins_and_losses():
    up = build_table(rounds_series(4, True), 60, SettlementRule.CLOSE_GTE_OPEN, "btcusdt")
    down = build_table(rounds_series(4, False), 60, SettlementRule.CLOSE_GTE_OPEN, "btcusdt")
    assert sum(b.n for b in up.buckets) == 4 and sum(b.wins for b in up.buckets) == 4
    assert sum(b.n for b in down.buckets) == 4 and sum(b.wins for b in down.buckets) == 0


def test_build_table_skips_rounds_with_missing_seconds():
    data = [k for k in rounds_series(3, True) if k.open_time != T0 + STEP + 299_000]  # round 2 has no close
    t = build_table(data, 60, SettlementRule.CLOSE_GTE_OPEN, "BTCUSDT")
    assert sum(b.n for b in t.buckets) == 2


def test_save_keeps_other_tables_and_load_errors_are_explicit(tmp_path):
    p = tmp_path / "cal.json"
    save_tables(p, [table({2: (100, 70)}, 60)])
    save_tables(p, [table({2: (100, 90)}, 270)])
    assert load_table(p, 60).buckets[2].wins == 70 and load_table(p, 270).buckets[2].wins == 90
    with pytest.raises(KeyError, match="calibrate"):
        load_table(p, 120)
    with pytest.raises(FileNotFoundError, match="calibrate"):
        load_table(tmp_path / "nope.json", 60)


# ---- the value strategy -------------------------------------------------------------------

def ctx_for(move_bps: float, q: OddsSample | None, decision_s: int = 60) -> RoundContext:
    dt = T0 + decision_s * 1000
    last = 100 * (1 + move_bps / 1e4)
    recent = [k1s(T0 + i * 1000, 100.0) for i in range(decision_s - 1)] + [k1s(dt - 1000, last)]
    return RoundContext(T0, 100.0, dt, recent, (), zigzag(dt - 300_000, 300), q)


# zigzag sigma = 2 bps; with 240s left, move 20 bps -> z = 20/(2*sqrt(240)) = 0.645 -> bucket [0.5, 0.75) = index 2
MOVE = 20.0
IDX = 2


def test_z_bucket_used_by_the_test_setup():
    assert 0.5 <= z_score(MOVE, 2.0, 240) < 0.75


def test_buys_the_follow_side_when_it_is_cheap():
    v = Value(table({IDX: (1000, 720)}), ValueParams(fee_bps=200, min_edge=0.03))
    assert v.decide(ctx_for(MOVE, quote(0.60, 0.62))) is Side.UP  # 0.72/0.62 - 1 - 0.02 = +14%


def test_skips_when_price_already_reflects_the_odds():
    v = Value(table({IDX: (1000, 720)}), ValueParams(fee_bps=200, min_edge=0.03))
    assert v.decide(ctx_for(MOVE, quote(0.70, 0.72))) is None  # UP ev = -2%, DOWN = 0.28/0.30 - 1.02 < 0


def test_fades_when_the_leader_is_overpriced():
    v = Value(table({IDX: (1000, 720)}), ValueParams(fee_bps=200, min_edge=0.03))
    assert v.decide(ctx_for(MOVE, quote(0.80, 0.81))) is Side.DOWN  # DOWN costs 0.20, wins 28%: +38%


def test_direction_follows_the_sign_of_the_move():
    v = Value(table({IDX: (1000, 720)}), ValueParams(fee_bps=0, min_edge=0.03))
    assert v.decide(ctx_for(-MOVE, quote(0.38, 0.40))) is Side.DOWN  # DOWN costs 0.62, wins 72%


def test_thin_bucket_and_missing_inputs_skip():
    v = Value(table({IDX: (50, 45)}), ValueParams(min_samples=100))
    assert v.decide(ctx_for(MOVE, quote(0.10, 0.11))) is None
    v = Value(table({IDX: (1000, 720)}))
    assert v.decide(ctx_for(MOVE, None)) is None
    assert v.decide(ctx_for(0.0, quote(0.10, 0.11))) is None
    c = ctx_for(MOVE, quote(0.10, 0.11))
    no_lookback = RoundContext(c.round_open_time, c.round_open, c.decision_time, c.recent, (), (), c.quote)
    assert v.decide(no_lookback) is None


def test_refuses_a_table_for_another_decision_second():
    v = Value(table({IDX: (1000, 720)}, decision_s=270))
    with pytest.raises(ValueError, match="calibration is for second 270"):
        v.decide(ctx_for(MOVE, quote(0.6, 0.62)))


def test_repr_is_short_and_stable_for_paper_labels():
    r = repr(Value(table({IDX: (1000, 720)})))
    assert "BTCUSDT t=60s 2026-01-01..2026-01-10" in r and len(r) < 200 and "wins" not in r


# ---- wiring through the engine and paper trader --------------------------------------------------

def test_decide_entry_checks_odds_before_asking_value():
    v = Value(table({IDX: (1000, 720)}))
    c = ctx_for(MOVE, None)
    entry, why = decide_entry(v, T0, 100.0, c.decision_time, c.recent, (), [], lookback=c.lookback)
    assert entry is None and why == "no_odds"
    entry, why = decide_entry(v, T0, 100.0, c.decision_time, c.recent, (), [quote(0.60, 0.62)],
                              lookback=c.lookback)
    assert why is None and entry.side is Side.UP and entry.price == 0.62


def test_backtest_prices_value_bets_and_uses_lookback_before_the_round():
    c = ctx_for(MOVE, None)
    seconds = list(c.lookback) + [k for k in c.recent if k.open_time >= T0] + \
        [k1s(T0 + i * 1000, 100.3) for i in range(60, 300)]
    seconds = sorted({k.open_time: k for k in seconds}.values(), key=lambda k: k.open_time)
    res = run_backtest([k5m(T0, 100.0, 100.3)], seconds, Value(table({IDX: (1000, 720)})),
                       SettlementRule.CLOSE_GTE_OPEN, decision_s=60, odds_samples=[quote(0.60, 0.62)])
    assert len(res.trades) == 1 and res.trades[0].pnl == pytest.approx(1 / 0.62 - 1)


def test_paper_trader_uses_seeded_history_for_volatility():
    tr = PaperTrader(Value(table({IDX: (1000, 720)})), SettlementRule.CLOSE_GTE_OPEN, decision_s=60,
                     clock=lambda: T0 + 60_001)
    tr.seed_seconds(zigzag(T0 - 300_000, 300))  # history loaded at startup, before the round
    tr.on_kline(k5m(T0, 100.0, 100.0, closed=False))
    for i in range(60):
        px = 100.0 if i < 59 else 100.2
        if i == 59:
            tr.on_odds(quote(0.60, 0.62, ts=T0 + 59_500))
        tr.on_kline(k1s(T0 + i * 1000, px))
    assert tr.last_decision["result"] == "bet" and tr.open[T0].side is Side.UP
    assert math.isclose(tr.open[T0].entry_price, 0.62)
