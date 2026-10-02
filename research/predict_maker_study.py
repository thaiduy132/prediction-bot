"""Study of the kacho.io method on Predict.fun (the venue behind Binance Wallet prediction markets).

The article (Polymarket esports, Jan-Mar 2026): fair value from an external reference, resting LIMIT
orders at >= 7% edge, pairs locked when both sides fill (+8,293$), unhedged leftovers lost (-3,185$).

Here: every Predict.fun sports/esports market is linked to a Polymarket market (polymarketConditionIds),
so Polymarket is the external fair value. For each past trade on Predict (GET /v1/orders/matches) we
know what the resting (maker) side bought and at what price, and Polymarket's price history at that
minute gives the fair value. That tells how much flow a maker posting at >= X% edge would have met.

Upper bound caveat: those fills went to the makers who were there; a new maker competes for them.

Usage: python research/predict_maker_study.py [--max-markets 700] [--refresh]
Writes raw data to data/research/ (gitignored).
"""

from __future__ import annotations

import argparse
import bisect
import collections
import datetime as dt
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "research"
WEI = 1e18
KINDS = ("SPORTS_MATCH", "SPORTS_TEAM_MATCH", "ESPORTS_CS2", "ESPORTS_LOL", "ESPORTS_DOTA2")


def get(c: httpx.Client, url: str, params: dict | list | None = None, tries: int = 4) -> dict | list:
    for i in range(tries):
        r = c.get(url, params=params)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(1.5 * (i + 1))
            continue
        r.raise_for_status()
    r.raise_for_status()
    return {}


def collect(max_markets: int, statuses: tuple[str, ...] = ("OPEN", "RESOLVED")) -> dict:
    load_dotenv(ROOT / ".env")
    pf = httpx.Client(base_url="https://api.predict.fun", headers={"x-api-key": os.environ["PREDICT_API_KEY"]},
                      timeout=30)
    pub = httpx.Client(timeout=30)
    markets, after = [], None
    for status in statuses:
        after = None
        for _ in range(80):
            p = {"first": 100, "status": status}
            if after:
                p["after"] = after
            j = get(pf, "/v1/markets", p)
            markets += [m for m in j["data"] if m.get("marketVariant") in KINDS and m.get("polymarketConditionIds")]
            after = j.get("cursor")
            if len(j["data"]) < 100 or not after:
                break
    markets = markets[:max_markets]
    print(f"{len(markets)} linked sports/esports markets ({'+'.join(statuses)})", file=sys.stderr)

    cids = sorted({m["polymarketConditionIds"][0] for m in markets})
    poly = {}
    for i in range(0, len(cids), 20):
        for pm in get(pub, "https://gamma-api.polymarket.com/markets",
                      [("condition_ids", x) for x in cids[i:i + 20]] + [("limit", "50")]):
            poly[pm["conditionId"]] = pm

    trades, history = {}, {}
    for n, m in enumerate(markets):
        rows, after = [], None
        for _ in range(20):
            p = {"marketId": m["id"], "first": 100}
            if after:
                p["after"] = after
            j = get(pf, "/v1/orders/matches", p)
            rows += j.get("data") or []
            after = j.get("cursor")
            if not after or len(j.get("data") or []) < 100:
                break
        trades[m["id"]] = rows
        pm = poly.get(m["polymarketConditionIds"][0])
        if rows and pm and pm.get("clobTokenIds"):
            ts = sorted(int(dt.datetime.fromisoformat(r["executedAt"].replace("Z", "+00:00")).timestamp())
                        for r in rows)
            # prices-history rejects long ranges at 1-minute fidelity: fetch <= 5-day windows around trades
            windows, start = [], ts[0]
            for i in range(1, len(ts) + 1):
                if i == len(ts) or ts[i] - start > 5 * 86400:
                    # +2 h after the last trade of the window: markouts (fair value after the fill)
                    windows.append((start - 3600, min(int(time.time()), ts[i - 1] + 7200)))
                    if i < len(ts):
                        start = ts[i]
            for k, tok in enumerate(json.loads(pm["clobTokenIds"])):
                pts: dict[int, float] = {}
                for a, b in windows:
                    for fid in (1, 5):
                        try:
                            h = get(pub, "https://clob.polymarket.com/prices-history",
                                    {"market": tok, "startTs": a, "endTs": b, "fidelity": fid})
                        except httpx.HTTPStatusError:
                            continue
                        pts.update({x["t"]: x["p"] for x in h.get("history", [])})
                        break
                history[f"{m['id']}:{k}"] = [{"t": t, "p": p} for t, p in sorted(pts.items())]
        if n % 50 == 0:
            print(f"  {n}/{len(markets)} markets, {sum(len(v) for v in trades.values())} trades", file=sys.stderr)
    return {"markets": markets, "poly": poly, "trades": trades, "history": history}


def maker_fills(data: dict) -> list[dict]:
    """One row per maker fill: what the resting side bought, at what cost, vs Polymarket fair then."""
    out = []
    hist = {k: ([x["t"] for x in v], [x["p"] for x in v]) for k, v in data["history"].items()}
    for m in data["markets"]:
        mid = m["id"]
        for e in data["trades"].get(str(mid), data["trades"].get(mid, [])):
            t = int(dt.datetime.fromisoformat(e["executedAt"].replace("Z", "+00:00")).timestamp())
            for mk in e["makers"]:
                k = int(mk["outcome"]["indexSet"]) - 1  # outcome index (0 = first outcome)
                price = float(mk["price"]) / WEI
                shares = float(mk["amount"]) / WEI
                if mk["quoteType"] == "Bid":  # maker bought outcome k at `price`
                    long_k, cost = k, price
                else:  # maker sold outcome k at `price` == bought the other outcome at 1 - price
                    long_k, cost = 1 - k, 1 - price
                h = hist.get(f"{mid}:{long_k}")
                if not h or not h[0]:
                    continue
                i = bisect.bisect_right(h[0], t) - 1
                if i < 0 or t - h[0][i] > 900:  # fair value must be from the last 15 minutes
                    continue
                fair = h[1][i]
                if not 0 < cost < 1 or not 0 < fair < 1:
                    continue
                status = mk["outcome"].get("status") if long_k == k else None
                out.append(dict(kind=m["marketVariant"], market=mid, t=t, cost=cost, fair=fair, shares=shares,
                                usd=shares * cost, edge=(fair - cost) / cost, signer=mk.get("signer"),
                                taker_signer=e["taker"].get("signer"), resolved_won=status))
    return out


def report(fills: list[dict]) -> None:
    if not fills:
        print("no fills with a fair value")
        return
    span_days = max(1e-9, (max(f["t"] for f in fills) - min(f["t"] for f in fills)) / 86400)
    print(f"{len(fills)} maker fills with a Polymarket fair value, over {span_days:.1f} days\n")
    print(f"{'market kind':>20} {'fills':>6} {'$ volume':>9} {'$/day':>7} {'median edge':>12} "
          f"{'$ at >=3% edge':>15} {'$ at >=7%':>10} {'$/day >=7%':>11} {'top-3 makers share':>19}")
    by = collections.defaultdict(list)
    for f in fills:
        by[f["kind"]].append(f)
    by["ALL"] = fills
    for kind, v in sorted(by.items(), key=lambda x: x[0] != "ALL"):
        vol = sum(f["usd"] for f in v)
        v3 = sum(f["usd"] for f in v if f["edge"] >= 0.03)
        v7 = sum(f["usd"] for f in v if f["edge"] >= 0.07)
        makers = collections.Counter()
        for f in v:
            makers[f["signer"]] += f["usd"]
        top3 = sum(x for _, x in makers.most_common(3)) / vol if vol else 0
        print(f"{kind:>20} {len(v):>6} {vol:>9.0f} {vol / span_days:>7.0f} {st.median(f['edge'] for f in v):>+12.1%} "
              f"{v3:>15.0f} {v7:>10.0f} {v7 / span_days:>11.1f} {top3:>18.0%}")
    big = [f for f in fills if f["edge"] >= 0.07]
    if big:
        print(f"\nfills at >= 7% edge: {len(big)}, ${sum(f['usd'] for f in big):.0f}; distinct makers that got them: "
              f"{len({f['signer'] for f in big})}; distinct takers who gave them: {len({f['taker_signer'] for f in big})}")
        print("edge distribution of those fills:", {f"{lo:.0%}+": sum(1 for f in big if f['edge'] >= lo)
                                                    for lo in (0.07, 0.15, 0.3, 0.5)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-markets", type=int, default=700)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--status", choices=("OPEN", "RESOLVED", "BOTH"), default="BOTH")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "" if a.status == "BOTH" else f"_{a.status.lower()}"
    raw = OUT / f"predict_maker_raw{suffix}.json"
    if a.refresh or not raw.exists():
        data = collect(a.max_markets, ("OPEN", "RESOLVED") if a.status == "BOTH" else (a.status,))
        raw.write_text(json.dumps(data), encoding="utf-8")
    data = json.loads(raw.read_text(encoding="utf-8"))
    fills = maker_fills(data)
    (OUT / f"predict_maker_fills{suffix}.json").write_text(json.dumps(fills), encoding="utf-8")
    report(fills)


if __name__ == "__main__":
    main()
