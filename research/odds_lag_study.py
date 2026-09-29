"""How many seconds after a BTC move does the Predict.fun Up price react?  (stdlib only)

Aligns recorded odds (data record-odds, ~1 sample/second) with Binance 1s candles on a per-second
grid and measures, for lags L = -3..+10 s:

    corr( change of the Up mid-price during second s ,  BTC return during second s - L )

If odds are set by fast market makers, the correlation peaks at L = 0 (same second, the best this
1-second polling can resolve). A peak at L >= 2 means odds lag the spot price by that many seconds,
which is time a faster bot could use. It also runs an event study on the largest 1s moves.

Usage:  python research/odds_lag_study.py --config config.yaml [--start 2026-09-29 --end 2026-10-10]
Needs thousands of paired seconds (hours of `data record-odds`) before the answer means anything.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.config import load_config  # noqa: E402
from bot.data.kline_cache import KlineCache  # noqa: E402
from bot.data.odds_store import OddsStore  # noqa: E402
from bot.timeutil import ms_to_iso, parse_utc  # noqa: E402

LAGS = range(-3, 11)


def corr(xs: list[float], ys: list[float]) -> tuple[float, float, int]:
    n = len(xs)
    if n < 30:
        return 0.0, 0.0, n
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return 0.0, 0.0, n
    r = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / math.sqrt(sxx * syy)
    r = max(-0.999999, min(0.999999, r))
    return r, r * math.sqrt((n - 2) / (1 - r * r)), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--start", help="UTC (default: first recorded quote)")
    ap.add_argument("--end", help="UTC (default: last recorded quote)")
    a = ap.parse_args()
    cfg = load_config(a.config)

    store = OddsStore(cfg.predict.odds_path, cfg.market.symbol)
    cov = store.coverage()
    if cov is None:
        sys.exit(f"no quotes in {cfg.predict.odds_path}: run `python -m bot data record-odds` for a few hours")
    start = parse_utc(a.start) if a.start else cov[0]
    end = parse_utc(a.end) if a.end else cov[1] + 1000
    samples = store.get_range(start, end)
    store.close()

    # Up mid-price per second (last quote seen in that second), only when both sides are quoted.
    mid: dict[int, tuple[float, int]] = {}
    for q in samples:
        if q.up_bid is not None and q.up_ask is not None:
            mid[q.ts // 1000 * 1000] = ((q.up_bid + q.up_ask) / 2, q.round_start)

    cache = KlineCache(cfg.data.cache_path)
    secs = cache.get_range(cfg.market.symbol, "1s", start // 1000 * 1000 - 20_000, end + 20_000)
    cache.close()
    close = {k.open_time: k.close for k in secs}
    ret = {t: (close[t] / close[t - 1000] - 1) * 1e4 for t in close if t - 1000 in close}  # bps during second t

    # dmid[s]: change of the mid between second s-1 and s, same round only.
    dmid = {s: m - mid[s - 1000][0] for s, (m, rnd) in mid.items() if s - 1000 in mid and mid[s - 1000][1] == rnd}

    print(f"odds {ms_to_iso(start)} .. {ms_to_iso(end)}: {len(samples)} quotes, {len(mid)} seconds with a mid, "
          f"{len(dmid)} one-second mid changes, {len(ret)} BTC 1s returns in the cache")
    if len(dmid) < 1000:
        print("WARNING: fewer than 1000 paired seconds; treat everything below as noise. Record more odds.")
    if len(ret) < len(dmid) // 2:
        print("NOTE: many 1s candles missing from the cache for this range; run `data record` (or `data fetch`/"
              "`run --mode backtest`) so the 1s klines get cached too.")

    print(f"\n{'lag L (s)':>9} {'pairs':>7} {'corr(dmid[s], ret[s-L])':>24} {'t-stat':>7}   (L>0: odds react L s AFTER BTC)")
    best = (0.0, 0)
    for lag in LAGS:
        xs, ys = [], []
        for s, dm in dmid.items():
            r = ret.get(s - lag * 1000)
            if r is not None:
                xs.append(r)
                ys.append(dm)
        c, t, n = corr(xs, ys)
        best = max(best, (c, lag))
        print(f"{lag:>9} {n:>7} {c:>+24.3f} {t:>+7.1f}")
    print(f"\nstrongest reaction at lag {best[1]} s (corr {best[0]:+.3f})")

    # Event study: after the biggest 1s BTC moves, how does the Up mid move over the next seconds?
    moves = sorted((abs(r), t, r) for t, r in ret.items() if t in mid)
    top = moves[-max(1, len(moves) // 100):]  # top 1% of moves that happened while we had a quote
    path: dict[int, list[float]] = defaultdict(list)
    for _, t, r in top:
        base = mid[t - 1000][0] if t - 1000 in mid else None
        if base is None:
            continue
        for k in range(0, 11):
            m = mid.get(t + k * 1000)
            if m is not None and m[1] == mid[t][1]:
                path[k].append((m[0] - base) * (1 if r > 0 else -1))
    if path:
        print(f"\nEvent study: {len(top)} largest 1s BTC moves (|ret| >= {top[0][0]:.2f} bps). "
              "Up-mid change in the direction of the move, since the second before it:")
        final = sum(path[10]) / len(path[10]) if path.get(10) else None
        for k in range(0, 11):
            if path[k]:
                avg = sum(path[k]) / len(path[k])
                share = f"{avg / final * 100:>5.0f}% of the 10s reaction" if final else ""
                print(f"  +{k:>2}s  n={len(path[k]):>4}  {avg:+.4f}  {share}")
        print("  If most of the reaction already shows at +0s, odds keep up with BTC within the polling "
              "resolution (~1s): no lag to exploit with this setup.")


if __name__ == "__main__":
    main()
