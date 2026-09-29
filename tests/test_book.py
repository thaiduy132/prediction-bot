import pytest

from bot.backtest.engine import run_backtest
from bot.backtest.features import book_features, imbalance
from bot.backtest.strategy import BookImbalance, BookParams, MomentumBook, Side
from bot.data.book_store import BookSample, BookSampler, BookStore, sample_from_depth
from bot.models import DepthSnapshot, Kline
from bot.settlement import SettlementRule

STEP = 300_000
T0 = 1_800_000_000_000 // STEP * STEP


def depth(recv: int, bid_qty: float, ask_qty: float, n: int = 20) -> DepthSnapshot:
    return DepthSnapshot(1, tuple((100.0 - i * 0.01, bid_qty) for i in range(n)),
                         tuple((100.01 + i * 0.01, ask_qty) for i in range(n)), recv)


def sample(ts: int, bq: float, aq: float) -> BookSample:
    return BookSample(ts, 100.0, 100.01, bq, aq, 5 * bq, 5 * aq, 10 * bq, 10 * aq, 20 * bq, 20 * aq)


def kline(interval: str, t: int, o: float, c: float) -> Kline:
    ms = 1000 if interval == "1s" else STEP
    return Kline("BTCUSDT", interval, t, t + ms - 1, o, max(o, c), min(o, c), c, 1, 1, 1, 1, 1)


def secs(px_end: float) -> list[Kline]:
    return [kline("1s", T0 + i * 1000, 100 + (px_end - 100) * min(i + 1, 270) / 270,
                  100 + (px_end - 100) * min(i + 1, 270) / 270) for i in range(300)]


def book_run(bq: float, aq: float, first_ts: int = T0 + 200_000) -> list[BookSample]:
    return [sample(t, bq, aq) for t in range(first_ts, T0 + 270_000, 1000)]


def test_imbalance_bounds():
    assert imbalance(3, 1) == 0.5 and imbalance(1, 3) == -0.5 and imbalance(0, 0) == 0.0


def test_sample_cumulative_depth():
    s = sample_from_depth(depth(1000, 2.0, 1.0), 1000)
    assert (s.bq1, s.aq1, s.bq5, s.aq5, s.bq20, s.aq20) == (2, 1, 10, 5, 40, 20)


def test_sampler_emits_last_state_of_finished_second():
    sm = BookSampler()
    assert sm.offer(depth(1_100, 1, 1)) is None
    assert sm.offer(depth(1_900, 5, 1)) is None  # same second, replaces
    out = sm.offer(depth(2_050, 1, 1))  # new second -> previous second is done
    assert out is not None and out.ts == 1000 and out.bq1 == 5


def test_store_roundtrip(tmp_path):
    st = BookStore(tmp_path / "b.sqlite", "btcusdt")
    for ts in (1000, 2000, 3000):
        st.add(sample(ts, 2, 1))
    st.flush()
    got = st.get_range(1000, 3000)
    assert [g.ts for g in got] == [1000, 2000] and got[0].bq10 == 20
    assert st.coverage() == (1000, 3000, 3)
    st.close()


def test_features_window_and_micro():
    f = book_features([sample(t, 3, 1) for t in range(0, 30_000, 1000)], levels=10, window_s=10)
    assert f.samples == 10 and f.imbalance == pytest.approx(0.5) and f.microprice_bps > 0
    assert book_features([]) is None


def test_book_strategy_follows_heavier_side():
    r = kline("5m", T0, 100.0, 101.0)
    res = run_backtest([r], secs(100.5), BookImbalance(BookParams(min_imbalance=0.2)),
                       SettlementRule.CLOSE_GTE_OPEN, book_samples=book_run(3, 1))
    assert res.trades[0].side is Side.UP and res.trades[0].pnl > 0
    res = run_backtest([r], secs(100.5), BookImbalance(BookParams(min_imbalance=0.2)),
                       SettlementRule.CLOSE_GTE_OPEN, book_samples=book_run(1, 3))
    assert res.trades[0].side is Side.DOWN and res.trades[0].pnl < 0


def test_momentum_book_skips_when_book_disagrees():
    r = kline("5m", T0, 100.0, 101.0)
    strat = MomentumBook(1.0, BookParams(min_imbalance=0.2))
    agree = run_backtest([r], secs(100.5), strat, SettlementRule.CLOSE_GTE_OPEN, book_samples=book_run(3, 1))
    against = run_backtest([r], secs(100.5), strat, SettlementRule.CLOSE_GTE_OPEN, book_samples=book_run(1, 3))
    assert len(agree.trades) == 1 and not against.trades and against.rounds_skipped_by_strategy == 1


def test_wide_spread_is_skipped():
    r = kline("5m", T0, 100.0, 101.0)
    wide = [BookSample(t, 100.0, 100.5, 3, 1, 15, 5, 30, 10, 60, 20) for t in range(T0 + 200_000, T0 + 270_000, 1000)]
    res = run_backtest([r], secs(100.5), BookImbalance(BookParams(max_spread_bps=2.0)),
                       SettlementRule.CLOSE_GTE_OPEN, book_samples=wide)
    assert not res.trades


def test_no_book_data_skips_and_counts():
    r = kline("5m", T0, 100.0, 101.0)
    res = run_backtest([r], secs(100.5), BookImbalance(), SettlementRule.CLOSE_GTE_OPEN)
    assert res.rounds_skipped_no_book == 1 and not res.trades


def test_no_lookahead_book_samples_stop_before_decision():
    seen = []

    class Spy:
        name, needs_book = "spy", True

        def decide(self, ctx):
            seen.append(max(b.ts for b in ctx.book))
            return None

    late = book_run(1, 1) + [sample(T0 + 270_000, 9, 1), sample(T0 + 271_000, 9, 1)]  # after decision
    run_backtest([kline("5m", T0, 100.0, 101.0)], secs(100.5), Spy(), SettlementRule.CLOSE_GTE_OPEN, book_samples=late)
    assert seen[0] < T0 + 270_000


def test_stale_book_is_rejected():
    r = kline("5m", T0, 100.0, 101.0)
    stale = [sample(t, 3, 1) for t in range(T0 + 100_000, T0 + 200_000, 1000)]  # ends 70s before decision
    res = run_backtest([r], secs(100.5), BookImbalance(), SettlementRule.CLOSE_GTE_OPEN, book_samples=stale)
    assert res.rounds_skipped_no_book == 1
