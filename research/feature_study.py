"""Which features carry information about the direction of a 5m round? (stdlib only)

For every complete round and several decision times t (seconds after the round opens) it builds
features from the 1s candles known at t and measures:

  1. how much of the outcome the move so far already explains (accuracy of following it),
  2. whether a plain random-walk fair value Phi(z) is calibrated,
  3. whether any feature explains what Phi(z) does NOT (residual correlation, with t-stat) and
     whether it predicts the remaining move (Spearman).

Usage:  python research/feature_study.py --config config.yaml --start 2026-09-15 --end 2026-09-29
Nothing here places orders. Only Binance klines are used, so book/odds features are not covered.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.config import load_config  # noqa: E402
from bot.data.binance_rest import BinanceRestClient  # noqa: E402
from bot.data.kline_cache import KlineCache, load_klines  # noqa: E402
from bot.timeutil import parse_utc  # noqa: E402

ROUND_S = 300
DECISIONS = (60, 120, 180, 240, 270)
FEATURES = ["ret10", "ret30", "ret60", "flow30", "flow_all", "trade_int", "prev_ret", "vol_ratio"]


def phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def pearson(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """(corr, t-stat under the null of zero correlation)."""
    n = len(xs)
    if n < 10:
        return 0.0, 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return 0.0, 0.0
    r = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / math.sqrt(sxx * syy)
    r = max(-0.999999, min(0.999999, r))
    return r, r * math.sqrt((n - 2) / (1 - r * r))


def ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=v.__getitem__)
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2
        i = j + 1
    return out


def spearman(xs: list[float], ys: list[float]) -> tuple[float, float]:
    return pearson(ranks(xs), ranks(ys))


class Series:
    """1s data on a dense second grid; missing seconds are None."""

    def __init__(self, klines: list, start_ms: int, end_ms: int) -> None:
        self.t0 = start_ms
        n = (end_ms - start_ms) // 1000
        self.o: list[float | None] = [None] * n
        self.c: list[float | None] = [None] * n
        self.v: list[float] = [0.0] * n
        self.tb: list[float] = [0.0] * n
        self.n_trades: list[float] = [0.0] * n
        for k in klines:
            i = (k.open_time - start_ms) // 1000
            if 0 <= i < n:
                self.o[i], self.c[i] = k.open, k.close
                self.v[i], self.tb[i], self.n_trades[i] = k.volume, k.taker_buy_base, float(k.trades)
        self.n = n

    def complete(self, a: int, b: int) -> bool:
        return all(self.c[i] is not None for i in range(a, b))


def sigma_bps(s: Series, end: int, window: int = 300) -> float | None:
    """RMS of 1s returns (bps) over the `window` seconds before index `end`."""
    rets = []
    for i in range(end - window + 1, end):
        a, b = s.c[i - 1], s.c[i]
        if a and b:
            rets.append((b / a - 1) * 1e4)
    if len(rets) < window * 0.9:
        return None
    return math.sqrt(sum(r * r for r in rets) / len(rets))


def flow(s: Series, a: int, b: int) -> float:
    vol = sum(s.v[a:b])
    return (2 * sum(s.tb[a:b]) - vol) / vol if vol > 0 else 0.0


def build_samples(s: Series, first_round: int, last_round: int) -> dict[int, list[dict[str, float]]]:
    """decision second t -> list of samples (one per usable round)."""
    out: dict[int, list[dict[str, float]]] = defaultdict(list)
    for r in range(first_round, last_round):
        a = (r - first_round) * ROUND_S  # index of the round's first second on the dense grid
        if a < 2 * ROUND_S or a + ROUND_S > s.n or not s.complete(a, a + ROUND_S):
            continue
        open0 = s.o[a]
        close_end = s.c[a + ROUND_S - 1]
        prev_open, prev_close = s.o[a - ROUND_S], s.c[a - 1]
        if not (open0 and close_end and prev_open and prev_close):
            continue
        y = 1.0 if close_end >= open0 else 0.0
        hour = datetime.fromtimestamp((s.t0 + a * 1000) / 1000, tz=UTC).hour
        for t in DECISIONS:
            c = s.c[a + t - 1]
            sig = sigma_bps(s, a + t)
            sig_prev = sigma_bps(s, a)  # calm/volatile regime before the round started
            base_trades = sum(s.n_trades[a - ROUND_S:a]) / (ROUND_S / 30)
            if not c or not sig or not sig_prev or base_trades <= 0:
                continue
            move = (c / open0 - 1) * 1e4
            rem = ROUND_S - t
            z = move / (sig * math.sqrt(rem))
            c10, c30, c60 = s.c[a + t - 11], s.c[a + t - 31] if t >= 31 else None, s.c[a + t - 61] if t >= 61 else None
            out[t].append({
                "y": y, "move": move, "z": z, "sigma": sig, "hour": float(hour),
                "rem_ret": (close_end / c - 1) * 1e4,  # what is left to happen, in bps
                "ret10": (c / c10 - 1) * 1e4 if c10 else 0.0,
                "ret30": (c / c30 - 1) * 1e4 if c30 else 0.0,
                "ret60": (c / c60 - 1) * 1e4 if c60 else 0.0,
                "flow30": flow(s, a + t - 30, a + t),
                "flow_all": flow(s, a, a + t),
                "trade_int": math.log(max(1e-9, sum(s.n_trades[a + t - 30:a + t]) / base_trades)),
                "prev_ret": (prev_close / prev_open - 1) * 1e4,
                "vol_ratio": math.log(sig / sig_prev),
            })
    return out


def report(samples: dict[int, list[dict[str, float]]]) -> None:
    print("\n=== 1. How much of the outcome does the move so far already explain? ===")
    print(f"{'t(s)':>5} {'rounds':>7} {'left(s)':>8} {'follow-move win%':>17} {'|move| bps (mean)':>18} {'up rate':>8}")
    for t in DECISIONS:
        rows = [r for r in samples[t] if r["move"] != 0]
        if not rows:
            continue
        win = sum(1 for r in rows if (r["move"] > 0) == (r["y"] == 1.0)) / len(rows)
        print(f"{t:>5} {len(rows):>7} {ROUND_S - t:>8} {win * 100:>16.1f}% {sum(abs(r['move']) for r in rows) / len(rows):>18.2f} "
              f"{sum(r['y'] for r in rows) / len(rows) * 100:>7.1f}%")

    print("\n=== 2. Is the random-walk fair value Phi(z) calibrated? (z = move / (sigma_1s * sqrt(seconds left))) ===")
    for t in (120, 270):
        rows = samples[t]
        print(f"\n t={t}s  ({len(rows)} rounds)   bucket of Phi(z)      n   mean Phi(z)   actual up-rate")
        bins = [(0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.0001)]
        for lo, hi in bins:
            b = [r for r in rows if lo <= phi(r["z"]) < hi]
            if b:
                print(f"{'':>26}[{lo:.1f},{min(hi, 1):.1f}) {len(b):>6} {sum(phi(r['z']) for r in b) / len(b):>12.3f} "
                      f"{sum(r['y'] for r in b) / len(b):>16.3f}")
        brier_rw = sum((phi(r["z"]) - r["y"]) ** 2 for r in rows) / len(rows)
        brier_const = sum((0.5 - r["y"]) ** 2 for r in rows) / len(rows)
        print(f"{'':>26}Brier: fair value {brier_rw:.4f}  vs  always-0.5 {brier_const:.4f}  (lower = better)")

    print("\n=== 3. Does a feature explain what Phi(z) misses?  corr(residual y-Phi(z), feature)  [t-stat] ===")
    print("    |t| > 3 is worth a look (35 tests here, so ~2 will look 'significant' by chance).")
    print(f"{'feature':>10} " + " ".join(f"{'t=' + str(t):>16}" for t in DECISIONS))
    for f in FEATURES:
        cells = []
        for t in DECISIONS:
            rows = samples[t]
            r, ts = pearson([x[f] for x in rows], [x["y"] - phi(x["z"]) for x in rows])
            cells.append(f"{r:+.3f} [{ts:+.1f}]")
        print(f"{f:>10} " + " ".join(f"{c:>16}" for c in cells))

    print("\n=== 3b. Same test against a CALIBRATED baseline (empirical up-rate per z-bucket, 20 buckets) ===")
    print("    Removes the over-confidence of Phi(z), so what is left is information beyond the move itself.")
    print(f"{'feature':>10} " + " ".join(f"{'t=' + str(t):>16}" for t in DECISIONS))
    resid: dict[int, list[float]] = {}
    for t in DECISIONS:
        rows = samples[t]
        order = sorted(range(len(rows)), key=lambda i: rows[i]["z"])
        res = [0.0] * len(rows)
        size = max(1, len(rows) // 20)
        for s0 in range(0, len(rows), size):
            idx = order[s0:s0 + size] if s0 + size * 2 <= len(rows) else order[s0:]
            p = sum(rows[i]["y"] for i in idx) / len(idx)
            for i in idx:
                res[i] = rows[i]["y"] - p
            if s0 + size * 2 > len(rows):
                break
        resid[t] = res
    for f in FEATURES:
        cells = []
        for t in DECISIONS:
            r, ts = pearson([x[f] for x in samples[t]], resid[t])
            cells.append(f"{r:+.3f} [{ts:+.1f}]")
        print(f"{f:>10} " + " ".join(f"{c:>16}" for c in cells))

    print("\n=== 4. Does a feature predict the REMAINING move (bps)?  Spearman [t-stat] ===")
    print(f"{'feature':>10} " + " ".join(f"{'t=' + str(t):>16}" for t in DECISIONS))
    for f in ["move", "z"] + FEATURES:
        cells = []
        for t in DECISIONS:
            rows = samples[t]
            r, ts = spearman([x[f] for x in rows], [x["rem_ret"] for x in rows])
            cells.append(f"{r:+.3f} [{ts:+.1f}]")
        print(f"{f:>10} " + " ".join(f"{c:>16}" for c in cells))

    print("\n=== 5. Volatility and time of day (sigma = RMS 1s return, bps; higher = harder to call) ===")
    rows = samples[270]
    by_hour: dict[int, list[dict[str, float]]] = defaultdict(list)
    for x in rows:
        by_hour[int(x["hour"])].append(x)
    print(f"{'UTC hour':>9} {'rounds':>7} {'sigma_1s':>9} {'|5m move| bps':>14} {'up rate':>8}")
    for h in sorted(by_hour):
        b = by_hour[h]
        print(f"{h:>9} {len(b):>7} {sum(x['sigma'] for x in b) / len(b):>9.2f} "
              f"{sum(abs(x['move'] + x['rem_ret']) for x in b) / len(b):>14.2f} {sum(x['y'] for x in b) / len(b) * 100:>7.1f}%")
    r, ts = pearson([x["sigma"] for x in rows], [abs(x["rem_ret"]) for x in rows])
    print(f"\ncorr(sigma_1s before decision, |remaining move|) = {r:+.3f} [t={ts:+.1f}]  -> volatility clusters: {'yes' if ts > 3 else 'weak'}")


async def load(cfg, start_ms: int, end_ms: int) -> list:
    cache = KlineCache(cfg.data.cache_path)
    try:
        async with BinanceRestClient(cfg.rest.base_url, cfg.rest.timeout_s, cfg.rest.max_retries) as rest:
            return await load_klines(rest, cache, cfg.market.symbol, "1s", start_ms, end_ms)
    finally:
        cache.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    a = p.parse_args()
    cfg = load_config(a.config)
    start_ms, end_ms = parse_utc(a.start), parse_utc(a.end)
    start_ms -= start_ms % (ROUND_S * 1000)
    end_ms -= end_ms % (ROUND_S * 1000)
    print(f"loading 1s klines {a.start} .. {a.end} (cached after the first run)...", file=sys.stderr)
    klines = asyncio.run(load(cfg, start_ms, end_ms))
    print(f"{len(klines)} candles", file=sys.stderr)
    s = Series(klines, start_ms, end_ms)
    samples = build_samples(s, start_ms // 1000 // ROUND_S, end_ms // 1000 // ROUND_S)
    print(f"Symbol {cfg.market.symbol}, {a.start} .. {a.end}, complete rounds used per decision time: "
          + ", ".join(f"t={t}: {len(samples[t])}" for t in DECISIONS))
    report(samples)


if __name__ == "__main__":
    main()
