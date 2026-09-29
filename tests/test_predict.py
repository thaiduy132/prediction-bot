import httpx
import pytest

from bot.data.odds_store import OddsSample, OddsStore
from bot.data.predict_client import PredictClient, PredictError, parse_orderbook, round_slug


def test_round_slug():
    assert round_slug("BTCUSDT", 300, 1790659200) == "btc-updown-5m-1790659200"
    assert round_slug("ETHUSDT", 900, 5) == "eth-updown-15m-5"


def test_parse_orderbook_yes_side():
    ob = parse_orderbook({"data": {"asks": [[0.37, 100.0], [0.45, 7.5]], "updateTimestampMs": 5,
                                   "bids": [[0.33, 50.0], [0.17, 6.0], [0.1, 1.0]]}})
    assert (ob.bid, ob.ask, ob.bid_qty, ob.ask_qty) == (0.33, 0.37, 50.0, 100.0)
    assert ob.mid == pytest.approx(0.35) and ob.bid_depth5 == 57.0 and ob.update_ms == 5


def test_parse_empty_orderbook():
    ob = parse_orderbook({"data": {"asks": [], "bids": []}})
    assert ob.bid is None and ob.ask is None and ob.mid is None and ob.bid_depth5 == 0.0


def _client(handler):
    return PredictClient("https://x", client=httpx.AsyncClient(base_url="https://x", transport=httpx.MockTransport(handler)))


async def test_orderbook_404_is_none_and_auth_error_is_explicit():
    async with _client(lambda r: httpx.Response(404, json={})) as c:
        assert await c.orderbook(1) is None
    async with _client(lambda r: httpx.Response(401, json={})) as c:
        with pytest.raises(PredictError, match="PREDICT_API_KEY"):
            await c.orderbook(1)


async def test_open_markets_paginates():
    def handler(req: httpx.Request) -> httpx.Response:
        if "after" in req.url.params:
            return httpx.Response(200, json={"data": [{"categorySlug": "b"}], "cursor": "z"})
        return httpx.Response(200, json={"data": [{"categorySlug": str(i)} for i in range(100)], "cursor": "c1"})

    async with _client(handler) as c:
        assert len(await c.open_updown_markets()) == 101


def test_odds_store_roundtrip(tmp_path):
    st = OddsStore(tmp_path / "o.sqlite", "btcusdt")
    st.upsert_round(1000, 7, "btc-updown-5m-1", "CHAINLINK", 83000.0, None, "OPEN", None)
    st.add_sample(OddsSample(1000, 1500, 7, 0.4, 0.45, 10, 20, 30, 40))
    st.add_sample(OddsSample(1000, 2500, 7, None, None, 0, 0, 0, 0))
    got = st.samples_for_round(1000)
    assert [g.has_quote for g in got] == [True, False]
    assert st.unresolved_rounds(10_000_000) == [(1000, 7)]
    st.upsert_round(1000, 7, "btc-updown-5m-1", "CHAINLINK", 83000.0, 83100.0, "RESOLVED", "UP")
    assert st.unresolved_rounds(10_000_000) == []
    assert st.summary() == {"rounds": 1, "samples": 2, "samples_with_quote": 1}
