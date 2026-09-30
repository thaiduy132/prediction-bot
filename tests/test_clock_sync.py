import pytest

from bot import clock_sync, timeutil
from bot.backtest.strategy import AlwaysUp
from bot.clock_sync import describe_offset, keep_clock_synced, measure_offset, sync_clock
from bot.data.binance_rest import BinanceRestError
from bot.data.odds_store import OddsSample
from bot.models import Kline
from bot.paper.trader import PaperTrader
from bot.settlement import SettlementRule

STEP = 300_000
T0 = 1_800_000_000_000 // STEP * STEP
SKEW = 146_000  # machine clock this much behind Binance, as seen on 2026-09-30


@pytest.fixture(autouse=True)
def reset_offset():
    timeutil.set_clock_offset(0)
    yield
    timeutil.set_clock_offset(0)


class FakeRest:
    """server_time() answers in Binance time; the machine clock is SKEW behind and each call takes `rtts[i]`."""

    def __init__(self, rtts: list[int], fail: bool = False) -> None:
        self.rtts, self.fail, self.local, self.calls = list(rtts), fail, T0, 0

    def clock(self) -> int:
        return self.local

    async def server_time(self) -> int:
        if self.fail:
            raise BinanceRestError("down")
        rtt = self.rtts[self.calls % len(self.rtts)]
        self.calls += 1
        self.local += rtt // 2
        server = self.local + SKEW
        self.local += rtt - rtt // 2
        return server


async def test_measure_offset_picks_the_fastest_round_trip():
    rest = FakeRest([400, 40, 900])
    offset, rtt = await measure_offset(rest, samples=3, clock=rest.clock)
    assert rtt == 40 and offset == SKEW


async def test_now_ms_follows_the_measured_offset(monkeypatch):
    monkeypatch.setattr(clock_sync, "local_ms", lambda: T0)
    rest = FakeRest([50])
    rest.local = T0
    monkeypatch.setattr(clock_sync, "measure_offset", lambda r: _ret((SKEW, 50)))
    assert await sync_clock(rest) == SKEW
    assert timeutil.clock_offset_ms() == SKEW
    assert abs(timeutil.now_ms() - timeutil.local_ms() - SKEW) <= 1


async def _ret(v):
    return v


async def test_sync_clock_fails_loudly_when_binance_is_unreachable():
    with pytest.raises(BinanceRestError):
        await sync_clock(FakeRest([10], fail=True))
    assert timeutil.clock_offset_ms() == 0


async def test_resync_updates_offset_and_keeps_it_on_failure(monkeypatch):
    results = iter([(SKEW, 30), BinanceRestError("x"), (SKEW + 5_000, 30)])

    async def fake_measure(rest):
        r = next(results)
        if isinstance(r, Exception):
            raise r
        return r

    ticks = 0

    async def fake_sleep(_):
        nonlocal ticks
        ticks += 1
        if ticks > 3:
            raise StopAsyncIteration

    seen = []
    monkeypatch.setattr(clock_sync, "measure_offset", fake_measure)
    orig = timeutil.set_clock_offset
    monkeypatch.setattr(clock_sync, "set_clock_offset", lambda v: (seen.append(v), orig(v)))
    with pytest.raises(StopAsyncIteration):
        await keep_clock_synced(object(), interval_s=0, sleep=fake_sleep)
    assert seen == [SKEW, SKEW + 5_000]  # the failure in between kept the old offset
    assert timeutil.clock_offset_ms() == SKEW + 5_000


def test_describe_offset():
    assert "khớp" in describe_offset(-200)
    assert "chậm 146.0s" in describe_offset(SKEW)
    assert "nhanh 3.0s" in describe_offset(-3_000)


# ---- the bug this fixes: odds stamped with the machine clock never matched Binance-time decisions ----

def k1s(t: int, px: float) -> Kline:
    return Kline("BTCUSDT", "1s", t, t + 999, px, px, px, px, 1, 1, 1, 1, 1)


def run_round(offset_ms: int) -> dict:
    """The poller stamps quotes with now_ms() and derives the round from it, like OddsPoller does."""
    timeutil.set_clock_offset(offset_ms)
    true_decision = T0 + 60_000
    machine_now = true_decision - SKEW  # what the machine clock reads at that moment
    stamped = machine_now + offset_ms  # == now_ms() at that moment
    quote = OddsSample(stamped // STEP * STEP, stamped - 500, 1, 0.52, 0.53, 10, 10, 50, 50)
    tr = PaperTrader(AlwaysUp(), SettlementRule.CLOSE_GTE_OPEN, decision_s=60, clock=lambda: stamped)
    tr.on_odds(quote)
    for i in range(60):
        tr.on_kline(k1s(T0 + i * 1000, 100.0))
    return tr.last_decision


def test_unsynced_clock_reproduces_no_odds():
    assert run_round(0)["result"] == "skip: no_odds"


def test_synced_clock_finds_the_quote():
    d = run_round(SKEW)
    assert d["result"] == "bet" and d["price"] == 0.53
