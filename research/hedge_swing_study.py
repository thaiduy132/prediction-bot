"""Can the kacho.io Polymarket method ("buy one side cheap, later buy the other side so that the
pair costs < 1$, locking the difference") work on Predict.fun BTC 5m rounds?

The article (esports, Jan-Mar 2026): limit orders at >= 7% edge vs sportsbook fair value; 1,075 locked
pairs made +8,293$, unhedged leftovers lost -3,185$, edge decayed with competition and fees.

Here, on recorded Predict.fun books (1 s samples) and official results:
  entry  : at second 60 buy the side the BTC move points to (momentum; optionally only when
           momentum_value sees an edge), at the ask, 1 share
  hedge  : any later second where buying the other side makes the pair cost <= 1 - margin after
           fees, buy it at the price seen `latency` seconds later (if still <= the threshold);
           the pair then pays 1$ for sure (minus the winning leg's fee)
  else   : hold the single share to the end (directional)

Fees: real formula measured on Binance quotes, fee shares = 2% * min(p, 1-p) / p of the bought shares.
Usage: python research/hedge_swing_study.py --config config.yaml
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.backtest.calibration import load_table, sigma_bps, z_score  # noqa: E402
from bot.backtest.pricing import buy_fee_fraction, expected_profit  # noqa: E402
from bot.config import load_config  # noqa: E402
from bot.models import Kline  # noqa: E402

FIX = 1790719740000  # odds timestamps are Binance-aligned from here on
ENTRY_S = 60


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    odds = sqlite3.connect(f"file:{cfg.predict.odds_path}?mode=ro", uri=True)
    kl = sqlite3.connect(f"file:{cfg.data.cache_path}?mode=ro", uri=True)
    table = load_table(cfg.backtest.calibration_path, ENTRY_S)

    winners = {rs: json.loads(r).get("name", "").upper() for rs, r in odds.execute(
        "SELECT round_start, resolution FROM odds_rounds WHERE status='RESOLVED' AND round_start>=?", (FIX,))}
    books: dict[int, tuple[list[int], list[tuple[float, float]]]] = {}
    for rs, ts, b, a in odds.execute("SELECT round_start, ts, up_bid, up_ask FROM odds_samples WHERE ts>=? AND "
                                     "up_bid IS NOT NULL AND up_ask IS NOT NULL ORDER BY ts", (FIX,)):
        if rs in winners:
            t, v = books.setdefault(rs, ([], []))
            t.append(ts - rs)
            v.append((b, a))
    closes = dict(kl.execute("SELECT open_time, close FROM klines WHERE symbol=? AND interval='1s' AND open_time>=?",
                             (cfg.market.symbol, FIX - 400_000)))
    opens5 = dict(kl.execute("SELECT open_time, open FROM klines WHERE symbol=? AND interval='5m' AND open_time>=?",
                             (cfg.market.symbol, FIX)))

    def at(rs: int, ms: int, stale: int = 3000) -> tuple[float, float] | None:
        t, v = books[rs]
        i = bisect.bisect_right(t, ms) - 1
        return v[i] if i >= 0 and ms - t[i] <= stale else None

    def buy_price(q: tuple[float, float], side: str) -> float:
        return q[1] if side == "UP" else 1 - q[0]

    entries = []
    for rs in sorted(books):
        dec = rs + ENTRY_S * 1000
        last, op = closes.get(dec - 1000), opens5.get(rs, closes.get(rs))
        q = at(rs, ENTRY_S * 1000)
        if last is None or op is None or q is None:
            continue
        move = (last / op - 1) * 1e4
        if abs(move) < 1:
            continue
        look = [Kline("x", "1s", t, t + 999, closes[t], closes[t], closes[t], closes[t], 0, 0, 0, 0, 0)
                for t in range(dec - 300_000, dec, 1000) if t in closes]
        sig = sigma_bps(look)
        if sig is None:
            continue
        side = "UP" if move > 0 else "DOWN"
        p_entry = buy_price(q, side)
        if not 0 < p_entry < 1:
            continue
        p_win = table.lookup_pooled(abs(z_score(move, sig, 300 - ENTRY_S)), 100).rate
        entries.append(dict(rs=rs, side=side, px=p_entry, won=winners[rs] == side,
                            mv_ev=expected_profit(p_win, p_entry)))
    print(f"{len(winners)} resolved rounds with 1 s books since the clock fix; {len(entries)} momentum entries at second "
          f"{ENTRY_S}; {sum(e['mv_ev'] >= 0 for e in entries)} of them pass the momentum_value filter.\n")

    def directional(e: dict) -> float:  # per 1 share bought at e['px']
        return (1 - buy_fee_fraction(e["px"])) - e["px"] if e["won"] else -e["px"]

    def run(sel: list[dict], margin: float | None, latency_s: int) -> tuple[float, int, float, float]:
        total, locked, lock_profit, missed_wins = 0.0, 0, 0.0, 0
        for e in sel:
            other = "DOWN" if e["side"] == "UP" else "UP"
            done = None
            if margin is not None:
                t, v = books[e["rs"]]
                for i in range(bisect.bisect_right(t, ENTRY_S * 1000), len(t)):
                    if t[i] > 295_000:
                        break
                    q_other = buy_price(v[i], other)
                    if not 0 < q_other < 1 or e["px"] + q_other > 1 - margin:
                        continue
                    fill = at(e["rs"], t[i] + latency_s * 1000)  # what we can actually buy after the delay
                    if fill is None:
                        continue
                    q_fill = buy_price(fill, other)
                    if 0 < q_fill < 1 and e["px"] + q_fill <= 1 - margin:
                        win_px = e["px"] if e["won"] else q_fill  # the leg that pays carries its fee
                        done = 1 - buy_fee_fraction(win_px) - e["px"] - q_fill
                        break
            if done is None:
                total += directional(e)
            else:
                locked += 1
                lock_profit += done
                total += done
                missed_wins += e["won"]
        n = max(1, len(sel))
        return total / n, locked, (lock_profit / locked if locked else 0.0), missed_wins / max(1, locked)

    for title, sel in (("ALL momentum entries", entries), ("momentum_value entries (EV >= 0)",
                                                           [e for e in entries if e["mv_ev"] >= 0])):
        base, *_ = run(sel, None, 0)
        print(f"=== {title}: {len(sel)} rounds | hold to the end: {base:+.4f}$/share ===")
        print(f"{'lock margin':>12} {'latency':>8} {'locked':>8} {'avg locked profit':>18} {'locked rounds the entry had won':>32} "
              f"{'result $/share':>15} {'vs holding':>11}")
        for margin in (0.0, 0.03, 0.07, 0.15):
            for lat in (0, 3, 10):
                avg, locked, lp, mw = run(sel, margin, lat)
                print(f"{margin:>12.2f} {lat:>7}s {locked:>4} ({locked / max(1, len(sel)) * 100:>3.0f}%) {lp:>+18.4f} "
                      f"{mw * 100:>31.0f}% {avg:>+15.4f} {avg - base:>+11.4f}")
        print()


if __name__ == "__main__":
    main()
