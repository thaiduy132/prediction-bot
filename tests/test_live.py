import asyncio
import json

import pytest

from bot.backtest.strategy import Side
from bot.config import AppConfig
from bot.data.odds_recorder import OddsPoller
from bot.live.baw import BawClient, BawError
from bot.live.executor import ExecConfig, LiveExecutor, find_key, parse_quote
from bot.live.journal import LiveJournal
from bot.live.risk import RiskLimits, RiskManager
from bot.paper.trader import OpenBet
from bot.settlement import Outcome

T0 = 1_800_000_000_000 // 300_000 * 300_000
NOW = T0 + 60_000
TOKENS = {Side.UP: "111", Side.DOWN: "222"}


class FakeBaw:
    """Records argv; answers from a script keyed by the baw sub-command."""

    def __init__(self, replies: dict[str, object]) -> None:
        self.replies, self.calls = replies, []

    async def __call__(self, argv: list[str], timeout_s: float) -> tuple[int, str, str]:
        self.calls.append(argv)
        words = [a for a in argv[1:] if not a.startswith("--")][:3]
        key = " ".join(argv[1:3]) if argv[1] == "wallet" else " ".join(words)
        r = self.replies.get(key, {"success": False, "error": {"name": "NOT_SCRIPTED", "message": key}})
        if isinstance(r, list):
            r = r.pop(0) if len(r) > 1 else r[0]
        if isinstance(r, Exception):
            raise r
        return 0, r if isinstance(r, str) else json.dumps(r), ""

    def called(self, sub: str) -> list[list[str]]:
        return [a for a in self.calls if " ".join(a[1:1 + len(sub.split())]) == sub]


SLUG = f"btc-updown-5m-{T0 // 1000}"
SEARCH_OK = {"success": True, "data": [
    {"marketTopicId": 99, "slug": "btc-updown-15m-1", "markets": [{"outcomes": []}]},
    {"marketTopicId": 6207442, "slug": SLUG, "markets": [{"externalId": "2766638", "outcomes": [
        {"name": "Up", "tokenId": "111"}, {"name": "Down", "tokenId": "222"}]}]}]}
QUOTE_OK = {"success": True, "data": {"quoteId": "q-1", "averagePrice": "0.55", "feeAmount": "0.04", "slippageBps": 200}}
ORDER_OK = {"success": True, "data": {"orderId": "o-9", "status": "SUBMITTED"}}


def make(tmp_path, replies=None, live=True, limits=None, clock=lambda: NOW + 500, tokens=lambda r: TOKENS):
    base = {"prediction market search": SEARCH_OK, "prediction trade quote": QUOTE_OK,
            "prediction trade place-order": ORDER_OK}
    fake = FakeBaw({**base, **(replies or {})})
    journal = LiveJournal(tmp_path / "live.sqlite", clock=lambda: NOW)
    lim = limits or RiskLimits(1.0, 3.0, 20, 1, tmp_path / "STOP")
    ex = LiveExecutor(ExecConfig(live, stake_usd=1.0), BawClient("baw", runner=fake), journal,
                      RiskManager(lim, journal, clock=lambda: NOW), tokens,
                      slug_for=lambda r: f"btc-updown-5m-{r // 1000}", clock=clock, sleep=no_sleep)
    return ex, fake, journal


async def no_sleep(_: float) -> None:
    return None


def bet(side=Side.UP, price=0.55) -> OpenBet:
    return OpenBet(T0, side, price, 12.0, NOW)


def rows(journal):
    return journal.recent(50)[::-1]


# ---- baw client -------------------------------------------------------------------------------

async def test_baw_quote_argv_and_success():
    fake = FakeBaw({"prediction trade quote": QUOTE_OK})
    data = await BawClient("baw", runner=fake).quote(56, "111", "BUY", 1.0, slippage_bps=200)
    assert data["quoteId"] == "q-1"
    argv = fake.calls[0]
    assert argv[0] == "baw" and argv[-1] == "--json"
    assert argv[argv.index("--tokenId") + 1] == "111" and argv[argv.index("--amount") + 1] == "1"
    assert argv[argv.index("--side") + 1] == "BUY" and "--marketTopicId" not in argv


async def test_baw_errors_are_explicit():
    fake = FakeBaw({"wallet status": {"success": False, "error": {"code": 10003000, "name": "NOT_LOGGED_IN",
                                                                "message": "Not logged in"}}})
    with pytest.raises(BawError) as e:
        await BawClient("baw", runner=fake).wallet_status()
    assert e.value.not_logged_in
    with pytest.raises(BawError, match="non-JSON"):
        await BawClient("baw", runner=FakeBaw({"wallet status": "oops"})).wallet_status()
    with pytest.raises(BawError, match="not found"):
        await BawClient("baw", runner=FakeBaw({"wallet status": FileNotFoundError()})).wallet_status()


# ---- risk -----------------------------------------------------------------------------------

def test_risk_limits(tmp_path):
    j = LiveJournal(tmp_path / "j.sqlite", clock=lambda: NOW)
    lim = RiskLimits(1.0, 2.5, 3, 1, tmp_path / "STOP")
    r = RiskManager(lim, j, clock=lambda: NOW)
    assert r.check() is None
    rid = j.add(T0, "live", "UP", 1.0, "submitted")
    assert "still open" in r.check()
    j.update(rid, status="lost", pnl_usd=-1.0)
    assert r.check() is None  # lost 1, next bet could make it 2 <= 2.5
    j.update(j.add(T0, "live", "UP", 1.0, "submitted"), status="lost", pnl_usd=-1.0)
    assert "daily loss limit" in r.check()  # lost 2, next could make 3 > 2.5
    (tmp_path / "STOP").touch()
    assert "kill switch" in r.check()


def test_risk_counts_only_real_orders_and_bets_per_day(tmp_path):
    j = LiveJournal(tmp_path / "j.sqlite", clock=lambda: NOW)
    r = RiskManager(RiskLimits(1.0, 100.0, 2, 5, tmp_path / "STOP"), j, clock=lambda: NOW)
    j.add(T0, "shadow", "UP", 1.0, "quoted")
    j.add(T0, "live", "UP", 1.0, "failed")
    assert r.check() is None
    for _ in range(2):
        j.update(j.add(T0, "live", "UP", 1.0, "submitted"), status="won", pnl_usd=0.5)
    assert "daily bet limit" in r.check()


def test_risk_rejects_stake_above_daily_limit(tmp_path):
    with pytest.raises(ValueError):
        RiskLimits(5.0, 3.0, 10, 1, tmp_path / "STOP")


# ---- executor ---------------------------------------------------------------------------------

async def test_shadow_quotes_but_never_places_an_order(tmp_path):
    ex, fake, j = make(tmp_path, live=False)
    await ex.execute(bet())
    assert fake.called("prediction trade quote") and not fake.called("prediction trade place-order")
    r = rows(j)[0]
    assert r["mode"] == "shadow" and r["status"] == "quoted" and r["quote_price"] == 0.55 and r["quote_fee"] == 0.04


async def test_live_places_the_quoted_order_and_journals_it(tmp_path):
    ex, fake, j = make(tmp_path)
    await ex.execute(bet(Side.DOWN, 0.55))
    q = fake.called("prediction trade quote")[0]
    assert q[q.index("--tokenId") + 1] == "222"  # DOWN token
    assert q[q.index("--marketTopicId") + 1] == "6207442"
    po = fake.called("prediction trade place-order")[0]
    assert po[po.index("--quoteId") + 1] == "q-1" and po[po.index("--slippageBps") + 1] == "200"
    r = j.row(rows(j)[0]["id"])
    assert r["status"] == "submitted" and r["order_id"] == "o-9" and json.loads(r["order_json"])["orderId"] == "o-9"


def history(status: str, order_id: str = "o-9") -> dict:
    return {"success": True, "data": {"total": 1, "orders": [{"orderId": order_id, "status": status}]}}


def ended(shares: float, avg: float, token: str = "111") -> dict:
    return {"success": True, "data": {"summary": {"todayRealizedPnl": -1.5}, "counts": {},
                                      "positions": [{"tokenId": token, "shares": shares, "avgPrice": avg}]}}


async def test_submitted_is_not_a_position_until_binance_confirms_the_fill(tmp_path):
    ex, _, j = make(tmp_path, replies={"prediction order history": history("SUBMITTED"),
                                       "prediction position list": ended(0, 0, "x")})
    await ex.execute(bet(Side.UP))
    ex.on_settle(T0, Outcome.UP)
    assert rows(j)[0]["status"] == "submitted"  # never booked as a win without a confirmed fill
    await ex.reconcile()
    assert rows(j)[0]["status"] == "submitted"


async def test_failed_order_on_binance_is_never_a_win(tmp_path):
    # what happened on 2026-09-30: place-order answered, then Binance marked the order FAILED
    ex, _, j = make(tmp_path, replies={"prediction order history": history("FAILED"),
                                       "prediction position list": ended(0, 0, "x")})
    await ex.execute(bet(Side.UP))
    await ex.reconcile()
    ex.on_settle(T0, Outcome.UP)
    r = rows(j)[0]
    assert r["status"] == "failed" and r["pnl_usd"] is None and "FAILED" in r["reason"]
    assert ex.risk.check() is None  # no money at risk


async def test_filled_order_is_settled_then_corrected_with_binance_numbers(tmp_path):
    ex, _, j = make(tmp_path, replies={"prediction order history": history("FILLED"),
                                       "prediction position list": ended(1.72, 0.5787)})
    await ex.execute(bet(Side.UP))
    await ex.reconcile()
    assert rows(j)[0]["status"] == "filled" and "still open" in ex.risk.check()
    ex.on_settle(T0, Outcome.UP)
    assert rows(j)[0]["status"] == "won" and rows(j)[0]["pnl_source"] == "estimate"
    await ex.reconcile()
    r = rows(j)[0]
    assert r["pnl_source"] == "binance" and r["pnl_usd"] == pytest.approx(1.72 - 1.72 * 0.5787, abs=1e-6)


async def test_losing_fill_uses_the_real_cost(tmp_path):
    ex, _, j = make(tmp_path, replies={"prediction order history": history("FILLED"),
                                       "prediction position list": ended(2.55, 0.3878)})
    await ex.execute(bet(Side.UP))
    await ex.reconcile()
    ex.on_settle(T0, Outcome.DOWN)
    await ex.reconcile()
    assert rows(j)[0]["status"] == "lost" and rows(j)[0]["pnl_usd"] == pytest.approx(-2.55 * 0.3878, abs=1e-6)


async def test_fill_confirmed_after_the_round_closed_is_still_settled(tmp_path):
    ex, _, j = make(tmp_path, replies={"prediction order history": history("FILLED"),
                                       "prediction position list": ended(1.8, 0.55)})
    await ex.execute(bet(Side.UP))
    ex.on_settle(T0, Outcome.DOWN)  # candle closed while the order was still "submitted"
    await ex.reconcile()
    assert rows(j)[0]["status"] == "lost" and rows(j)[0]["pnl_source"] == "binance"


def test_risk_uses_the_worse_of_journal_and_binance(tmp_path):
    j = LiveJournal(tmp_path / "j.sqlite", clock=lambda: NOW)
    r = RiskManager(RiskLimits(1.0, 3.0, 20, 1, tmp_path / "STOP"), j, clock=lambda: NOW)
    j.update(j.add(T0, "live", "UP", 1.0, "submitted"), status="won", pnl_usd=2.5)  # a wrong booking
    r.external_realized = lambda: -2.26  # what Binance says
    assert "daily loss limit" in r.check()


async def test_price_moved_skips_without_ordering(tmp_path):
    ex, fake, j = make(tmp_path)
    await ex.execute(bet(Side.UP, 0.50))  # quote says 0.55 > 0.50 + 0.02
    assert not fake.called("prediction trade place-order") and "price moved" in rows(j)[0]["reason"]


async def test_too_late_skips(tmp_path):
    ex, fake, j = make(tmp_path, clock=lambda: NOW + 9_000)
    await ex.execute(bet())
    assert not fake.called("prediction trade place-order") and "too late" in rows(j)[0]["reason"]


async def test_risk_block_means_no_quote_at_all(tmp_path):
    (tmp_path / "STOP").touch()
    ex, fake, j = make(tmp_path)
    await ex.execute(bet())
    assert fake.calls == [] and "kill switch" in rows(j)[0]["reason"]  # not even a search


async def test_round_not_on_binance_and_unreadable_quote(tmp_path):
    ex, fake, j = make(tmp_path, replies={"prediction market search": {"success": True, "data": []}})
    await ex.execute(bet())
    assert not fake.called("prediction trade quote") and rows(j)[0]["reason"] == "round not found on Binance"
    ex, fake, j = make(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path,
                       replies={"prediction trade quote": {"success": True, "data": {"weird": 1}}})
    await ex.execute(bet())
    assert rows(j)[0]["status"] == "failed" and not fake.called("prediction trade place-order")


async def test_token_mismatch_with_predict_blocks_the_trade(tmp_path):
    ex, fake, j = make(tmp_path, tokens=lambda r: {Side.UP: "999", Side.DOWN: "222"})
    await ex.execute(bet(Side.UP))
    assert not fake.called("prediction trade quote") and "differ" in rows(j)[0]["reason"]


async def test_topic_is_cached_after_prefetch(tmp_path):
    ex, fake, _ = make(tmp_path)
    assert (await ex.topic(T0)).topic_id == "6207442"
    await ex.execute(bet())
    assert len(fake.called("prediction market search")) == 1  # the decision reused the cached topic


async def test_failed_order_is_journaled_and_not_counted(tmp_path):
    ex, _, j = make(tmp_path, replies={"prediction trade quote": QUOTE_OK, "prediction trade place-order":
                                       {"success": False, "error": {"name": "INSUFFICIENT_BALANCE", "message": "x"}}})
    await ex.execute(bet())
    assert rows(j)[0]["status"] == "failed" and "INSUFFICIENT_BALANCE" in rows(j)[0]["reason"]
    assert ex.risk.check() is None  # a failed order is not an open position


async def test_on_entry_runs_in_the_background(tmp_path):
    ex, fake, j = make(tmp_path)
    ex.on_entry(bet())
    assert fake.calls == []  # nothing happened synchronously
    await asyncio.gather(*list(ex._tasks))
    assert rows(j)[0]["status"] == "submitted"


async def test_redeem_pending_claims_winning_tokens(tmp_path):
    ex, fake, _ = make(tmp_path, replies={
        "prediction position list": {"success": True, "data": {"summary": {}, "counts": {"pendingClaimCount": 2},
                                                               "positions": [{"tokenId": "111"}, {"tokenId": "333"}]}},
        "prediction trade redeem": {"success": True, "data": {}}})
    await ex.redeem_pending()
    r = fake.called("prediction trade redeem")[0]
    assert r[r.index("--tokenIds") + 1] == "111,333"


def test_parse_quote_nested_and_find_key():
    assert parse_quote({"quote": {"id": 1, "quoteId": 7, "averagePrice": 0.4, "feeAmount": "0.1"}}) == ("7", 0.4, 0.1)
    assert find_key([{"a": {"b": {"orderId": "z"}}}], "orderId") == "z"
    assert parse_quote({}) == (None, None, None)


def test_status_for_dashboard(tmp_path):
    ex, _, _ = make(tmp_path, live=False)
    s = ex.status()
    assert s["mode"] == "shadow" and s["today"]["bets"] == 0 and s["blocked"] is None


# ---- token ids from Predict.fun ---------------------------------------------------------------

def test_outcome_tokens_from_predict_market():
    p = OddsPoller(AppConfig(), client=None)  # type: ignore[arg-type]
    p._markets = {f"btc-updown-5m-{T0 // 1000}": {"outcomes": [{"name": "Up", "onChainId": "111"},
                                                             {"name": "Down", "onChainId": "222"}]}}
    assert p.outcome_tokens(T0) == {"UP": "111", "DOWN": "222"}
    assert p.outcome_tokens(T0 + 300_000) is None


# ---- dashboard controls and account ------------------------------------------------------------

async def test_stop_and_resume_toggle_the_kill_switch(tmp_path):
    ex, fake, _ = make(tmp_path)
    ex.set_stopped(True)
    assert (tmp_path / "STOP").exists() and ex.status()["stopped"] and "kill switch" in ex.status()["blocked"]
    await ex.execute(bet())
    assert fake.calls == []
    ex.set_stopped(False)
    assert not (tmp_path / "STOP").exists() and ex.status()["blocked"] is None


async def test_refresh_account_reads_usdt_and_positions(tmp_path):
    ex, _, _ = make(tmp_path, replies={
        "wallet balance": {"success": True, "data": [
            {"symbol": "USDT", "address": "0x55d398326f99059fF775485246999027B3197955", "balance": "11.28"},
            {"symbol": "BNB", "address": "0x0", "balance": "0.1"}]},
        "prediction position list": {"success": True, "data": {"summary": {"walletBalance": 11.28},
            "counts": {"ongoingCount": 1, "endedCount": 3, "pendingClaimCount": 0}, "positions": [{"tokenId": "1"}]}}})
    await ex.refresh_account()
    a = ex.status()["account"]
    assert a["usdt"] == 11.28 and a["start_usdt"] == 11.28 and a["positions"] == 1 and a["error"] is None


async def test_open_order_from_before_a_restart_is_settled(tmp_path):
    ex, _, j = make(tmp_path, replies={"prediction order history": history("FILLED"),
                                       "prediction position list": ended(0, 0, "x")})
    await ex.execute(bet(Side.UP))  # order filled, then the bot restarts: paper forgets it
    await ex.reconcile()
    ex2, _, j2 = make(tmp_path)  # same journal file
    assert "still open" in ex2.risk.check()
    ex2.on_settle(T0, Outcome.DOWN)  # the round's candle, seen in history after restart
    assert rows(j2)[0]["status"] == "lost" and ex2.risk.check() is None


def test_paper_trader_reports_every_closed_round():
    from bot.backtest.strategy import AlwaysUp
    from bot.models import Kline
    from bot.paper.trader import PaperTrader
    from bot.settlement import SettlementRule

    seen = []
    tr = PaperTrader(AlwaysUp(), SettlementRule.CLOSE_GTE_OPEN, decision_s=60, clock=lambda: NOW)
    tr.on_settle = lambda r, o: seen.append((r, o))
    tr.on_kline(Kline("BTCUSDT", "5m", T0, T0 + 299_999, 100, 101, 99, 101, 1, 1, 1, 1, 1, True))
    assert seen == [(T0, Outcome.UP)]  # no paper bet in that round, still reported


# ---- retry when Binance fails a market order ---------------------------------------------------

def hist_seq(*statuses: str) -> list[dict]:
    """order history replies: each attempt's order id is o-9 (FakeBaw returns the same order id)."""
    return [{"success": True, "data": {"orders": [{"orderId": "o-9", "status": s,
                                                    "errorMessage": "Failed to execute the market order"}]}}
            for s in statuses]


async def test_failed_order_is_retried_and_the_retry_fills(tmp_path):
    ex, fake, j = make(tmp_path, replies={"prediction order history": hist_seq("SUBMITTED", "FAILED", "FILLED")})
    await ex.execute(bet())
    assert len(fake.called("prediction trade place-order")) == 2
    assert len(fake.called("prediction trade quote")) == 2  # a fresh quote for the retry
    r = rows(j)
    assert [x["status"] for x in r] == ["failed", "filled"]
    assert "Failed to execute the market order" in r[0]["reason"] and r[1]["reason"].startswith("retry 1/3")


async def test_gives_up_after_three_retries(tmp_path):
    ex, fake, j = make(tmp_path, replies={"prediction order history": hist_seq("FAILED")})
    await ex.execute(bet())
    assert len(fake.called("prediction trade place-order")) == 4  # first try + 3 retries
    assert [x["status"] for x in rows(j)] == ["failed"] * 4
    assert "gave up after 3 retries" in ex.last_action["detail"]
    assert ex.risk.check() is None  # nothing was spent, nothing is open


async def test_no_retry_after_the_retry_window(tmp_path):
    t = {"now": NOW + 500}
    ex, fake, _ = make(tmp_path, replies={"prediction order history": hist_seq("FAILED")}, clock=lambda: t["now"])

    async def slow_sleep(_):
        t["now"] += 70_000  # the FAILED status is only seen 70s after the decision

    ex._sleep = slow_sleep
    await ex.execute(bet())
    assert len(fake.called("prediction trade place-order")) == 1 and "not retrying" in ex.last_action["detail"]


async def test_retry_is_skipped_when_the_price_ran_away(tmp_path):
    quotes = [QUOTE_OK, {"success": True, "data": {"quoteId": "q-2", "averagePrice": "0.70", "feeAmount": "0.01"}}]
    ex, fake, j = make(tmp_path, replies={"prediction order history": hist_seq("FAILED"),
                                          "prediction trade quote": quotes})
    await ex.execute(bet(Side.UP, 0.55))
    assert len(fake.called("prediction trade place-order")) == 1
    assert "retry 1/3: price moved 0.550 -> 0.700" in rows(j)[-1]["reason"]


async def test_retry_respects_the_kill_switch(tmp_path):
    ex, fake, j = make(tmp_path, replies={"prediction order history": hist_seq("FAILED")})
    orig = ex._wait_fill

    async def fail_then_stop(row):
        res = await orig(row)
        (tmp_path / "STOP").touch()  # user pressed Stop while the first order was pending
        return res

    ex._wait_fill = fail_then_stop
    await ex.execute(bet())
    assert len(fake.called("prediction trade place-order")) == 1 and "kill switch" in rows(j)[-1]["reason"]


async def test_filled_first_time_means_no_retry(tmp_path):
    ex, fake, j = make(tmp_path, replies={"prediction order history": hist_seq("FILLED")})
    await ex.execute(bet())
    assert len(fake.called("prediction trade place-order")) == 1 and rows(j)[0]["status"] == "filled"


# ---- selling the open position from the dashboard ---------------------------------------------

from bot.data.odds_store import OddsSample  # noqa: E402


def ongoing(shares: float, avg: float, token: str = "111") -> dict:
    return {"success": True, "data": {"summary": {}, "counts": {"ongoingCount": 1},
                                      "positions": [{"tokenId": token, "shares": shares, "avgPrice": avg}]}}


SELL_QUOTE = {"success": True, "data": {"quoteId": "s-1", "side": "SELL", "amountIn": "1.8", "amountOut": "1.23",
                                         "averagePrice": 0.69, "feeAmount": "0.012"}}


async def filled_position(tmp_path, extra=None):
    replies = {"prediction order history": history("FILLED"), "prediction position list": ongoing(1.8, 0.55)}
    replies.update(extra or {})
    ex, fake, j = make(tmp_path, replies=replies)
    await ex.execute(bet(Side.UP, 0.55))
    await ex.refresh_position()
    return ex, fake, j


async def test_position_is_valued_at_the_current_bid(tmp_path):
    ex, _, _ = await filled_position(tmp_path)
    v = ex.position_view(OddsSample(T0, NOW + 30_000, 1, 0.69, 0.71, 10, 10, 50, 50))
    assert v["source"] == "binance" and v["shares"] == 1.8 and v["sell_price"] == 0.69
    fee = 0.02 * 0.31 * 1.8
    assert v["value_now"] == pytest.approx(1.8 * 0.69 - fee, abs=1e-4)
    assert v["pnl_if_sold"] == pytest.approx(1.8 * 0.69 - fee - 1.8 * 0.55, abs=1e-4)
    down = OddsSample(T0, NOW, 1, 0.69, 0.71, 10, 10, 50, 50)
    ex.position["side"] = "DOWN"
    assert ex.position_view(down)["sell_price"] == pytest.approx(0.29)  # selling DOWN hits 1 - Up ask
    assert ex.position_view(OddsSample(T0 + 300_000, NOW, 1, 0.5, 0.51, 1, 1, 1, 1))["sell_price"] is None


async def test_sell_quote_then_confirm_closes_the_position(tmp_path):
    ex, fake, j = await filled_position(tmp_path, {
        "prediction trade quote": [QUOTE_OK, SELL_QUOTE],
        "prediction trade place-order": [ORDER_OK, {"success": True, "data": {"orderId": "sell-1"}}],
        "prediction order history": [history("FILLED")["data"] and history("FILLED"),
                                     {"success": True, "data": {"orders": [
                                         {"orderId": "sell-1", "status": "FILLED", "filledShareQty": 1.8,
                                          "filledUsdtAmount": 1.22}]}}]})
    q = await ex.sell_quote(1.0)
    sell_argv = fake.called("prediction trade quote")[-1]
    assert sell_argv[sell_argv.index("--side") + 1] == "SELL" and sell_argv[sell_argv.index("--amount") + 1] == "1.8"
    assert q["quote_id"] == "s-1" and q["proceeds"] == 1.23 and q["pnl_realized"] == pytest.approx(1.23 - 0.99)
    assert len(fake.called("prediction trade place-order")) == 1  # quoting sold nothing
    res = await ex.sell_confirm("s-1")
    assert res["ok"] and res["closed"] and res["usd"] == 1.22
    r = j.row(rows(j)[0]["id"])
    assert r["status"] == "sold" and r["sold_shares"] == 1.8 and r["pnl_usd"] == pytest.approx(1.22 - 0.55 * 1.8)
    ex.on_settle(T0, Outcome.DOWN)  # the round ending later must not re-book a sold position
    assert j.row(r["id"])["status"] == "sold"


async def test_partial_sell_then_round_settles_with_both_parts(tmp_path):
    ex, _, j = await filled_position(tmp_path, {
        "prediction trade quote": [QUOTE_OK, {"success": True, "data": {"quoteId": "s-2", "amountIn": "0.9",
                                                                        "amountOut": "0.62", "averagePrice": 0.69}}],
        "prediction trade place-order": [ORDER_OK, {"success": True, "data": {"orderId": "sell-2"}}],
        "prediction order history": [history("FILLED"), {"success": True, "data": {"orders": [
            {"orderId": "sell-2", "status": "FILLED", "filledShareQty": 0.9, "filledUsdtAmount": 0.62}]}}]})
    q = await ex.sell_quote(0.5)
    assert q["shares"] == 0.9
    res = await ex.sell_confirm("s-2")
    assert res["ok"] and not res["closed"]
    row = j.row(rows(j)[0]["id"])
    assert row["status"] == "filled" and row["sold_usd"] == 0.62
    ex.on_settle(T0, Outcome.UP)
    row = j.row(row["id"])
    remaining = 1.0 / 0.55 - 0.04 - 0.9  # estimate from the buy quote, until Binance reconciles it
    assert row["status"] == "won" and row["pnl_usd"] == pytest.approx(0.62 + remaining - 1.0, abs=1e-6)


async def test_expired_or_unknown_quote_is_not_sold(tmp_path):
    t = {"now": NOW + 500}
    ex, fake, _ = await filled_position(tmp_path, {"prediction trade quote": [QUOTE_OK, SELL_QUOTE]})
    ex._clock = lambda: t["now"]
    await ex.sell_quote(1.0)
    assert (await ex.sell_confirm("other"))["error"] == "no such pending sell quote"
    await ex.sell_quote(1.0)
    t["now"] += 16_000
    assert "expired" in (await ex.sell_confirm("s-1"))["error"]
    assert len(fake.called("prediction trade place-order")) == 1  # only the original buy


async def test_selling_is_allowed_while_buys_are_stopped(tmp_path):
    ex, fake, _ = await filled_position(tmp_path, {"prediction trade quote": [QUOTE_OK, SELL_QUOTE]})
    ex.set_stopped(True)
    assert (await ex.sell_quote(1.0))["quote_id"] == "s-1"


async def test_sell_quote_without_position_or_in_shadow(tmp_path):
    ex, _, _ = make(tmp_path)
    assert (await ex.sell_quote(1.0))["error"] == "no open position"
    sh, _, _ = make(tmp_path / "s" if (tmp_path / "s").mkdir() is None else tmp_path, live=False)
    assert "live" in (await sh.sell_quote(1.0))["error"]


async def test_failed_sell_keeps_the_position(tmp_path):
    ex, _, j = await filled_position(tmp_path, {
        "prediction trade quote": [QUOTE_OK, SELL_QUOTE],
        "prediction trade place-order": [ORDER_OK, {"success": True, "data": {"orderId": "sell-3"}}],
        "prediction order history": [history("FILLED"), {"success": True, "data": {"orders": [
            {"orderId": "sell-3", "status": "FAILED", "errorMessage": "Failed to execute the market order"}]}}]})
    await ex.sell_quote(1.0)
    res = await ex.sell_confirm("s-1")
    assert "Failed to execute" in res["error"] and j.row(rows(j)[0]["id"])["status"] == "filled"


# ---- retries are re-decided on the current situation ----------------------------------------------

async def test_retry_buys_at_a_higher_price_when_the_strategy_still_agrees(tmp_path):
    # the case seen live: first order FAILED, then the price ran 0.56 -> 0.71 in our favour
    quotes = [QUOTE_OK, {"success": True, "data": {"quoteId": "q-2", "averagePrice": "0.71", "feeAmount": "0.01"}}]
    ex, fake, j = make(tmp_path, replies={"prediction order history": hist_seq("FAILED", "FILLED"),
                                          "prediction trade quote": quotes})
    seen = []
    ex.recheck = lambda b, price, cap: seen.append((b.side, price, cap)) or None  # strategy says: still buy
    await ex.execute(bet(Side.UP, 0.55))
    assert len(fake.called("prediction trade place-order")) == 2 and rows(j)[-1]["status"] == "filled"
    assert seen == [(Side.UP, 0.71, 0.85)]


async def test_retry_is_dropped_when_the_strategy_no_longer_agrees(tmp_path):
    quotes = [QUOTE_OK, {"success": True, "data": {"quoteId": "q-2", "averagePrice": "0.90", "feeAmount": "0.01"}}]
    ex, fake, j = make(tmp_path, replies={"prediction order history": hist_seq("FAILED"),
                                          "prediction trade quote": quotes})
    ex.recheck = lambda b, price, cap: f"price {price:.3f} above the retry cap {cap:.2f}" if price > cap else None
    await ex.execute(bet(Side.UP, 0.55))
    assert len(fake.called("prediction trade place-order")) == 1
    assert rows(j)[-1]["reason"] == "retry 1/3: price 0.900 above the retry cap 0.85"


async def test_first_attempt_still_uses_the_decision_price_tolerance(tmp_path):
    ex, fake, j = make(tmp_path)
    ex.recheck = lambda *a: None
    await ex.execute(bet(Side.UP, 0.50))  # quote 0.55 > 0.50 + 0.02 on the FIRST attempt
    assert not fake.called("prediction trade place-order") and "price moved" in rows(j)[0]["reason"]


def test_momentum_and_value_retry_rules():
    from bot.backtest.calibration import empty_buckets, CalibrationTable
    from bot.backtest.strategy import Momentum, RoundContext, Value, ValueParams
    from bot.models import Kline

    def k(t, px):
        return Kline("BTCUSDT", "1s", t, t + 999, px, px, px, px, 1, 1, 1, 1, 1)

    zig = [k(T0 - 300_000 + i * 1000, 100.0 + (0.01 if i % 2 else -0.01)) for i in range(300)]
    def ctx(last_px, second=75, quote=None):
        dt_ = T0 + second * 1000
        recent = [k(T0 + i * 1000, 100.0) for i in range(second - 1)] + [k(dt_ - 1000, last_px)]
        look = [x for x in zig + recent if dt_ - 300_000 <= x.open_time < dt_]
        return RoundContext(T0, 100.0, dt_, recent, (), look, quote)

    m = Momentum(1.0)
    assert m.accept_retry(ctx(100.2), Side.UP, 0.71, 0.85) is None
    assert "above the retry cap" in m.accept_retry(ctx(100.2), Side.UP, 0.90, 0.85)
    assert "no longer points UP" in m.accept_retry(ctx(99.8), Side.UP, 0.40, 0.85)

    b = empty_buckets()
    for x in b:
        x.n, x.wins = 1000, 900  # every |z| bucket: following the move wins 90%
    v = Value(CalibrationTable(60, 300, "close_gte_open", "BTCUSDT", "a", "b", b), ValueParams(fee_bps=200))
    assert v.accept_retry(ctx(100.2), Side.UP, 0.71, 0.85) is None  # 0.9/0.71 - 1.02 = +24.8%
    assert "no edge left" in v.accept_retry(ctx(100.2), Side.UP, 0.89, 0.85)  # 0.9/0.89 - 1.02 < 3%


def test_paper_trader_recheck_uses_the_latest_second():
    from bot.backtest.strategy import Momentum
    from bot.models import Kline
    from bot.paper.trader import OpenBet, PaperTrader
    from bot.settlement import SettlementRule

    tr = PaperTrader(Momentum(1.0), SettlementRule.CLOSE_GTE_OPEN, decision_s=60, clock=lambda: NOW)
    tr.on_kline(Kline("BTCUSDT", "5m", T0, T0 + 299_999, 100, 100, 100, 100, 1, 1, 1, 1, 1, False))
    for i in range(80):  # price went up, then by second 79 it is back below the open
        px = 100.2 if i < 70 else 99.9
        tr._seconds[T0 + i * 1000] = Kline("BTCUSDT", "1s", T0 + i * 1000, T0 + i * 1000 + 999, px, px, px, px,
                                           1, 1, 1, 1, 1)
    assert tr.current_context(T0).decision_time == T0 + 80_000
    assert "no longer points UP" in tr.recheck(OpenBet(T0, Side.UP, 0.6, 20.0, NOW), 0.65, 0.85)
    assert tr.recheck(OpenBet(T0, Side.DOWN, 0.4, -10.0, NOW), 0.45, 0.85) is None


# ---- stake scaling with the account ---------------------------------------------------------------

def test_stake_steps_up_at_each_doubling_and_back_down():
    from bot.live.sizing import StakeScaling
    s = StakeScaling(1.0, 11.28, 2.0, 1.5, 5.0)
    assert s.stake(None) == 1.0 and s.stake(13.96) == 1.0 and s.stake(22.55) == 1.0
    assert s.stake(22.56) == 1.5 and s.next_threshold(22.56) == 45.12
    assert s.stake(45.12) == 2.25 and s.stake(90.24) == 3.38
    assert s.stake(10_000) == 5.0 and s.next_threshold(10_000) is None  # capped
    assert s.stake(20.0) == 1.0  # fell back below 2x: back to the base stake
    assert StakeScaling(1.0).stake(1_000) == 1.0 and StakeScaling(1.0).next_threshold(1_000) is None  # off


async def test_executor_bets_the_scaled_stake_and_risk_checks_it(tmp_path):
    from bot.live.sizing import StakeScaling
    ex, fake, j = make(tmp_path, replies={"prediction order history": hist_seq("FILLED")})
    ex.scaling = StakeScaling(1.0, 11.28, 2.0, 1.5, 5.0)
    ex.account["usdt"] = 23.0  # >= 22.56
    await ex.execute(bet())
    q = fake.called("prediction trade quote")[0]
    assert q[q.index("--amount") + 1] == "1.5" and rows(j)[0]["stake_usd"] == 1.5
    s = ex.status()
    assert s["stake_usd"] == 1.5 and s["scaling"]["tier"] == 1 and s["scaling"]["next_step_at_usd"] == 45.12


async def test_open_stake_counts_toward_equity(tmp_path):
    from bot.live.sizing import StakeScaling
    ex, _, _ = make(tmp_path, replies={"prediction order history": hist_seq("FILLED")})
    ex.scaling = StakeScaling(1.0, 11.28, 2.0, 1.5, 5.0)
    ex.account["usdt"] = 22.0
    assert ex.current_stake() == 1.0
    await ex.execute(bet())  # 1$ now sits in an open position: 22 + 1 = 23 >= 22.56
    assert ex.equity_usd() == 23.0 and ex.current_stake() == 1.5


async def test_attempt_without_an_explicit_stake_uses_the_configured_one(tmp_path):
    ex, fake, j = make(tmp_path, live=False)
    await ex._attempt(bet(), 0)
    q = fake.called("prediction trade quote")[0]
    assert q[q.index("--amount") + 1] == "1" and rows(j)[0]["stake_usd"] == 1.0
