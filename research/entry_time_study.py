"""If we follow the move so far at second t of the round, what win rate do we get and what is the
most we can PAY per share and still break even?  (stdlib only, Binance klines only)

Break-even price for a share that wins with probability w and costs p, with fee f charged on the stake:
    w * (1/p - 1 - f) + (1 - w) * (-1 - f) = 0   ->   p = w / (1 + f)

Usage: python research/entry_time_study.py --config config.yaml --start 2026-09-15 --end 2026-09-29 [--t 60 240] [--fee-bps 200]
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import feature_study as fs  # noqa: E402
from bot.config import load_config  # noqa: E402
from bot.timeutil import parse_utc  # noqa: E402

# |z| buckets: how far the move is, in units of "typical remaining noise"
BUCKETS = [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 99.0)]


def wilson_low(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--t", type=int, nargs="+", default=[60, 240])
    ap.add_argument("--fee-bps", type=float, default=200.0)
    ap.add_argument("--train-frac", type=float, default=0.7, help="share of the period used to pick buckets; the rest tests them")
    a = ap.parse_args()
    fee = a.fee_bps / 10_000
    cfg = load_config(a.config)
    step = fs.ROUND_S * 1000
    start_ms, end_ms = parse_utc(a.start) // step * step, parse_utc(a.end) // step * step
    klines = asyncio.run(fs.load(cfg, start_ms, end_ms))
    s = fs.Series(klines, start_ms, end_ms)
    fs.DECISIONS = tuple(a.t)
    samples = fs.build_samples(s, start_ms // 1000 // fs.ROUND_S, end_ms // 1000 // fs.ROUND_S)
    print(f"{cfg.market.symbol} {a.start}..{a.end}, fee {a.fee_bps:g} bps. Break-even price = win rate / (1 + fee).")

    for t in a.t:
        rows = [r for r in samples[t] if r["move"] != 0]
        cut = int(len(rows) * a.train_frac)
        print(f"\n=== Decide at second {t} ({fs.ROUND_S - t}s left) - follow the move; {len(rows)} rounds "
              f"(first {cut} = train, last {len(rows) - cut} = test, in time order) ===")
        print(f"{'|z| bucket':>12} {'rounds':>7} {'win% (all)':>11} {'win% (test)':>12} {'95% low (all)':>14} "
              f"{'max price, fee':>15} {'max price, no fee':>18}")
        for lo, hi in BUCKETS:
            b_all = [r for r in rows if lo <= abs(r["z"]) < hi]
            b_test = [r for r in rows[cut:] if lo <= abs(r["z"]) < hi]
            if not b_all:
                continue
            w = lambda b: sum(1 for r in b if (r["move"] > 0) == (r["y"] == 1.0))  # noqa: E731
            wa, wt = w(b_all), w(b_test)
            win_all = wa / len(b_all)
            print(f"{lo:>5.2f}-{min(hi, 9.99):<5.2f} {len(b_all):>7} {win_all * 100:>10.1f}% "
                  f"{(wt / len(b_test) * 100 if b_test else float('nan')):>11.1f}% {wilson_low(wa, len(b_all)) * 100:>13.1f}% "
                  f"{wilson_low(wa, len(b_all)) / (1 + fee):>15.3f} {win_all:>18.3f}")
        print(f"{'':>12} 'max price, fee' uses the LOWER 95% bound of the win rate (conservative).")


if __name__ == "__main__":
    main()
