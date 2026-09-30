"""Late in a round, is the near-certain leader overpriced (and the long shot underpriced)?

Uses only recorded Predict.fun data: the order book of each round (data record-odds) and the
OFFICIAL winner Predict.fun published (Chainlink BTC/USDT settlement), not Binance candles.

For each resolved round and each decision second S it takes the last quote known at S, finds the
leader (mid > 0.5), and books a hypothetical 1$ buy of the leader and of the long shot at the
ask actually on the book, with the fee formula observed in real Binance quotes:
    fee (in shares) = 2% * min(p, 1 - p) / p  of the shares bought
    -> long shot (p < 0.5): 2% of the payout;  leader: 2% * (1 - p) / p of the payout

Clock: quotes recorded between 2026-09-29 10:20 and 2026-09-30 01:42 UTC carry the machine clock,
which was 145 s behind Binance (measured by aligning Up-price moves with BTC 1 s returns); those
timestamps are corrected. Later quotes already use Binance time.

Usage: python research/longshot_study.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.config import load_config  # noqa: E402

SKEWED = (1790662800000, 1790718120000)  # machine-clock period, UTC ms
SKEW_MS = 145_000
SECONDS = (240, 255, 270, 280, 285)
BUCKETS = [(0.5, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.97), (0.97, 0.98), (0.98, 0.99), (0.99, 1.01)]
FEE = 0.02


def pnl_per_dollar(price: float, won: bool) -> float:
    fee_frac = FEE * min(price, 1 - price) / price
    return (1 / price) * (1 - fee_frac) - 1 if won else -1.0


def wilson(w: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    db = sqlite3.connect(f"file:{cfg.predict.odds_path}?mode=ro", uri=True)

    winners = {}
    for rs, res in db.execute("SELECT round_start, resolution FROM odds_rounds WHERE status='RESOLVED'"):
        try:
            r = json.loads(res)
        except (TypeError, ValueError):
            continue
        if r.get("status") == "WON" and r.get("name") in ("Up", "Down"):
            winners[rs] = r["name"].upper()

    quotes: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    for rs, ts, bid, ask, bq, aq in db.execute(
            "SELECT round_start, ts, up_bid, up_ask, bid_qty, ask_qty FROM odds_samples "
            "WHERE up_bid IS NOT NULL AND up_ask IS NOT NULL ORDER BY ts"):
        if rs not in winners:
            continue
        true_ts = ts + SKEW_MS if SKEWED[0] <= ts < SKEWED[1] else ts
        quotes[rs].append((true_ts - rs, bid, ask, bq, aq))

    print(f"{len(winners)} rounds with an official Predict.fun winner; {len(quotes)} of them have two-sided quotes.")
    print(f"Up won {sum(w == 'UP' for w in winners.values())}, Down won {sum(w == 'DOWN' for w in winners.values())}.\n")

    for S in SECONDS:
        rows = []
        for rs, qs in quotes.items():
            known = [q for q in qs if S * 1000 - 3000 <= q[0] <= S * 1000]
            if not known:
                continue
            _, bid, ask, bq, aq = known[-1]
            mid = (bid + ask) / 2
            leader = "UP" if mid >= 0.5 else "DOWN"
            lead_mid = mid if leader == "UP" else 1 - mid
            # buy prices actually on the book; UP costs the ask, DOWN costs 1 - bid
            lead_px, lead_depth = (ask, aq) if leader == "UP" else (1 - bid, bq)
            dog_px, dog_depth = (1 - bid, bq) if leader == "UP" else (ask, aq)
            won_lead = winners[rs] == leader
            rows.append((lead_mid, lead_px, dog_px, won_lead, lead_depth * lead_px, dog_depth * dog_px))

        print(f"=== second {S} ({300 - S}s left): {len(rows)} rounds ===")
        print(f"{'leader priced':>14} {'n':>4} {'leader won':>11} {'95% CI':>13} {'avg buy px':>11} "
              f"{'PnL/1$ lead':>12} {'long shot px':>13} {'PnL/1$ shot':>12} {'shot depth $':>13}")
        for lo, hi in BUCKETS:
            b = [r for r in rows if lo <= r[0] < hi]
            if not b:
                continue
            n, w = len(b), sum(r[3] for r in b)
            ci = wilson(w, n)
            lead_pnl = sum(pnl_per_dollar(r[1], r[3]) for r in b if 0 < r[1] < 1) / n
            shot = [r for r in b if 0 < r[2] < 1]
            shot_pnl = sum(pnl_per_dollar(r[2], not r[3]) for r in shot) / len(shot) if shot else float("nan")
            depth = sorted(r[5] for r in shot)[len(shot) // 2] if shot else 0
            print(f"{lo:>6.2f}-{min(hi, 1):<6.2f} {n:>5} {w / n * 100:>10.1f}% {ci[0] * 100:>5.1f}-{ci[1] * 100:<5.1f}% "
                  f"{sum(r[1] for r in b) / n:>11.3f} {lead_pnl:>+12.3f} "
                  f"{(sum(r[2] for r in shot) / len(shot) if shot else float('nan')):>13.3f} {shot_pnl:>+12.3f} {depth:>13.2f}")
        print()


if __name__ == "__main__":
    main()
