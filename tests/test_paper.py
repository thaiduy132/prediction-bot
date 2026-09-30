import pytest

from bot.backtest.strategy import AlwaysUp, Momentum, Side
from bot.data.odds_store import OddsSample
from bot.models import Kline
from bot.paper.store import PaperStore
from bot.paper.trader import PaperTrader
from bot.settlement import SettlementRule

STEP = 300_000
T0 = 1_800_000_000_000 // STEP * STEP
DECISION = T0 + 270_000


def k1s(t: int, px: float) -> Kline:
    return Kline("BTCUSDT", "1s", t, t + 999, px, px, px, px, 1, 1, 1, 1, 1)


def k5m(closed: bool, o: float, c: float, t: int = T0) -> Kline:
    return Kline("BTCUSDT", "5m", t, t + STEP - 1, o, max(o, c), min(o, c), c, 1, 1, 1, 1, 1, closed)


def quote(ts: int, bid: float, ask: float, start: int = T0) -> OddsSample:
    return OddsSample(start, ts, 1, bid, ask, 10, 10, 50, 50)


def feed_round(tr: PaperTrader, start: int = T0, end_px: float = 100.5, odds=(0.79, 0.80), upto: int = 270) -> None:
    """One round: 5m candle forming, odds quote, then 1s candles for seconds [0, upto)."""
    tr.on_kline(k5m(False, 100.0, 100.0, start))
    for i in range(upto):
        t = start + i * 1000
        if i == 269:
            tr.on_odds(quote(t, odds[0], odds[1], start))
        tr.on_kline(k1s(t, 100 + (end_px - 100) * (i + 1) / 270))


def trader(strategy=None, **kw) -> PaperTrader:
    return PaperTrader(strategy or Momentum(1.0), SettlementRule.CLOSE_GTE_OPEN, decision_s=270,
                       clock=lambda: DECISION + 1, **kw)


def test_opens_bet_at_decision_second_priced_from_odds():
    tr = trader()
    feed_round(tr)
    assert list(tr.open) == [T0]
    b = tr.open[T0]
    assert b.side is Side.UP and b.entry_price == 0.80 and b.move_bps == pytest.approx(50.0, rel=1e-3)
    assert tr.snapshot()["open"][0]["entry_price"] == 0.80


def test_no_bet_before_decision_second():
    tr = trader()
    feed_round(tr, upto=269)  # last candle seen is second 268, decision needs second 269
    assert not tr.open and tr.last_decision is None


def test_settles_when_round_candle_closes_and_books_pnl():
    tr = trader()
    feed_round(tr)
    assert tr.on_kline(k5m(True, 100.0, 101.0)) is True
    assert not tr.open
    t = tr.result.trades[0]
    assert t.pnl == pytest.approx(0.25) and t.outcome.value == "UP"
    snap = tr.snapshot()
    assert snap["stats"]["bets"] == 1 and snap["stats"]["pnl_units"] == 0.25 and snap["curve"] == [0.25]
    assert snap["trades"][0]["cum_pnl"] == 0.25


def test_losing_bet_and_cumulative_pnl_across_rounds():
    tr = trader()
    feed_round(tr, T0)
    tr.on_kline(k5m(True, 100.0, 99.0, T0))  # momentum said UP, round closed DOWN -> -1
    feed_round(tr, T0 + STEP, end_px=100.5)
    tr.on_kline(k5m(True, 100.0, 101.0, T0 + STEP))  # +0.25
    s = tr.snapshot()
    assert s["curve"] == [-1.0, -0.75] and s["stats"]["wins"] == 1 and s["stats"]["losses"] == 1


def test_missing_odds_skips_and_is_reported():
    tr = trader()
    tr.on_kline(k5m(False, 100.0, 100.0))
    for i in range(270):
        tr.on_kline(k1s(T0 + i * 1000, 100 + 0.5 * (i + 1) / 270))
    assert not tr.open
    assert tr.last_decision["result"] == "skip: no_odds" and tr.snapshot()["skipped"] == {"no_odds": 1}


def test_gap_in_last_second_is_no_data_not_a_bet():
    tr = trader()
    tr.on_kline(k5m(False, 100.0, 100.0))
    tr.on_odds(quote(T0 + 265_000, 0.79, 0.80))
    for i in range(268):  # second 268 and 269 never arrive; 270 arrives late
        tr.on_kline(k1s(T0 + i * 1000, 100.5))
    tr.on_kline(k1s(T0 + 270_000, 100.5))
    assert not tr.open and tr.last_decision["result"] == "skip: no_data"


def test_started_after_the_decision_second_skips_instead_of_deciding_late():
    tr = trader(AlwaysUp())
    tr.seed_seconds([k1s(T0 + i * 1000, 100.5) for i in range(270)])  # history incl. the decision second
    tr.on_kline(k5m(False, 100.0, 100.0))
    tr.on_odds(quote(T0 + 272_000, 0.79, 0.80))
    tr.on_kline(k1s(T0 + 273_000, 100.5))  # first live candle is already past second 269
    assert not tr.open and tr.last_decision["result"] == "skip: no_data"
    tr.on_kline(k1s(T0 + 274_000, 100.5))
    assert tr.counts == {"skipped_no_data": 1}  # decided (skipped) once, not again


def test_decides_once_per_round():
    tr = trader(AlwaysUp())
    feed_round(tr)
    tr.on_kline(k1s(T0 + 270_000, 100.5))
    tr.on_kline(k1s(T0 + 271_000, 100.5))
    assert len(tr.open) == 1 and tr.counts == {}


def test_max_entry_price_skips_expensive_favourite():
    tr = trader(max_entry_price=0.9)
    feed_round(tr, odds=(0.94, 0.95))
    assert not tr.open and tr.last_decision["result"] == "skip: price"


def test_odds_from_another_round_are_ignored():
    tr = trader()
    tr.on_kline(k5m(False, 100.0, 100.0))
    tr.on_odds(quote(T0 + 265_000, 0.79, 0.80, start=T0 - STEP))
    for i in range(270):
        tr.on_kline(k1s(T0 + i * 1000, 100 + 0.5 * (i + 1) / 270))
    assert not tr.open and tr.last_decision["result"] == "skip: no_odds"


def test_fixed_payout_mode_needs_no_odds():
    tr = trader(use_odds=False, payout_ratio=0.9)
    tr.on_kline(k5m(False, 100.0, 100.0))
    for i in range(270):
        tr.on_kline(k1s(T0 + i * 1000, 100 + 0.5 * (i + 1) / 270))
    tr.on_kline(k5m(True, 100.0, 101.0))
    assert tr.result.trades[0].pnl == 0.9 and tr.snapshot()["pricing"] == "fixed payout"


def test_trades_survive_restart(tmp_path):
    db = tmp_path / "p.sqlite"
    tr = trader(store=PaperStore(db))
    feed_round(tr)
    tr.on_kline(k5m(True, 100.0, 101.0))
    tr2 = trader(store=PaperStore(db))
    assert tr2.snapshot()["stats"]["pnl_units"] == 0.25 and len(tr2.snapshot()["trades"]) == 1
    other = PaperTrader(Momentum(5.0), SettlementRule.CLOSE_GTE_OPEN, store=PaperStore(db))  # different params
    assert other.snapshot()["stats"]["bets"] == 0
