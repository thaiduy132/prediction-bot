"""Measured win rate of "follow the move so far", by how big the move is relative to noise.

    z = move_bps / (sigma_1s_bps * sqrt(seconds left in the round))

sigma_1s_bps is the RMS of 1-second close-to-close returns over the SIGMA_WINDOW_S seconds before
the decision. A pure random walk would win with probability Phi(|z|), but real rounds are less
predictable at the extremes (14 days of BTCUSDT: 96.5% measured vs 99.0% predicted with 30s left),
so strategies use the rate measured per |z| bucket instead of a formula.

These functions are the ONLY definition of sigma and z: calibration and live decisions both use them.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bot.models import Kline
from bot.settlement import Outcome, SettlementRule, settle
from bot.timeutil import ms_to_iso

SIGMA_WINDOW_S = 300
MIN_SIGMA_COVERAGE = 0.9  # share of the window's 1s returns that must be present
Z_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)  # last bucket is [4, inf)


def sigma_bps(candles: Sequence[Kline], window_s: int = SIGMA_WINDOW_S) -> float | None:
    """RMS of consecutive 1s close-to-close returns in bps; None if too many seconds are missing."""
    rets = [(b.close / a.close - 1) * 10_000
            for a, b in zip(candles, candles[1:], strict=False) if b.open_time - a.open_time == 1000]
    if not rets or len(rets) < MIN_SIGMA_COVERAGE * (window_s - 1):
        return None
    s = math.sqrt(sum(r * r for r in rets) / len(rets))
    return s if s > 0 else None


def z_score(move_bps: float, sigma: float, remaining_s: float) -> float:
    return move_bps / (sigma * math.sqrt(max(remaining_s, 1.0)))


def bucket_index(abs_z: float) -> int:
    for i in range(len(Z_EDGES) - 1, -1, -1):
        if abs_z >= Z_EDGES[i]:
            return i
    return 0


@dataclass(slots=True)
class Bucket:
    lo: float
    hi: float | None  # None = no upper bound
    n: int = 0
    wins: int = 0  # rounds where the side the move pointed to won

    @property
    def rate(self) -> float:
        """Win rate shrunk towards 1/2 (Laplace), so thin buckets are not over-trusted."""
        return (self.wins + 1) / (self.n + 2)


@dataclass(slots=True)
class CalibrationTable:
    decision_s: int
    round_s: int
    rule: str
    symbol: str
    start: str  # ISO range the table was measured on (for in-sample warnings)
    end: str
    buckets: list[Bucket] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.symbol} t={self.decision_s}s {self.start[:10]}..{self.end[:10]}"

    def lookup(self, abs_z: float) -> Bucket:
        return self.buckets[bucket_index(abs_z)]

    def lookup_pooled(self, abs_z: float, min_samples: int) -> Bucket:
        """The |z| bucket, or, when it has fewer than `min_samples` rounds, that bucket pooled with
        every STRONGER bucket (then weaker ones if still short).

        Strong moves are rare (t=60: |z| 1.5-2 had 86 rounds, >= 2 only 21), and they are exactly where
        retries land, since a market order fails when the price runs our way. Pooling upward is
        conservative: win rates rise with |z|, so the pooled rate is not above the true one here.
        """
        i = bucket_index(abs_z)
        if self.buckets[i].n >= min_samples:
            return self.buckets[i]
        lo, hi = i, len(self.buckets) - 1
        pooled = Bucket(self.buckets[i].lo, None, sum(b.n for b in self.buckets[i:]), sum(b.wins for b in self.buckets[i:]))
        while pooled.n < min_samples and lo > 0:
            lo -= 1
            pooled = Bucket(self.buckets[lo].lo, None, pooled.n + self.buckets[lo].n, pooled.wins + self.buckets[lo].wins)
        return pooled

    def to_json(self) -> dict[str, Any]:
        return {"decision_s": self.decision_s, "round_s": self.round_s, "rule": self.rule, "symbol": self.symbol,
                "start": self.start, "end": self.end,
                "buckets": [{"lo": b.lo, "hi": b.hi, "n": b.n, "wins": b.wins} for b in self.buckets]}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> CalibrationTable:
        return cls(d["decision_s"], d["round_s"], d["rule"], d["symbol"], d["start"], d["end"],
                   [Bucket(b["lo"], b["hi"], b["n"], b["wins"]) for b in d["buckets"]])


def empty_buckets() -> list[Bucket]:
    return [Bucket(lo, Z_EDGES[i + 1] if i + 1 < len(Z_EDGES) else None) for i, lo in enumerate(Z_EDGES)]


def build_table(seconds: Sequence[Kline], decision_s: int, rule: SettlementRule, symbol: str,
                round_s: int = 300) -> CalibrationTable:
    """Measure the table from closed 1s candles. Rounds with any needed second missing are skipped."""
    if not 0 < decision_s < round_s:
        raise ValueError(f"decision_s must be inside the round (0 < {decision_s} < {round_s})")
    by_open = {k.open_time: k for k in seconds if k.is_closed}
    step = round_s * 1000
    times = sorted(by_open)
    table = CalibrationTable(decision_s, round_s, rule.value, symbol.upper(),
                             ms_to_iso(times[0]) if times else "", ms_to_iso(times[-1] + 1000) if times else "",
                             empty_buckets())
    if not times:
        return table
    first = -(-times[0] // step) * step  # first round start inside the data
    for r in range(first, times[-1] - step + 2000, step):
        dec = r + decision_s * 1000
        o, c_dec, c_end = by_open.get(r), by_open.get(dec - 1000), by_open.get(r + step - 1000)
        if o is None or c_dec is None or c_end is None:
            continue
        lookback = [by_open[t] for t in range(dec - SIGMA_WINDOW_S * 1000, dec, 1000) if t in by_open]
        sig = sigma_bps(lookback)
        move = (c_dec.close / o.open - 1) * 10_000
        if sig is None or move == 0:
            continue
        outcome = settle(o.open, c_end.close, rule)
        if outcome is Outcome.VOID:
            continue
        b = table.lookup(abs(z_score(move, sig, round_s - decision_s)))
        b.n += 1
        b.wins += (outcome is Outcome.UP) == (move > 0)
    return table


def save_tables(path: str | Path, tables: Sequence[CalibrationTable]) -> None:
    """Write tables into `path`, keeping entries for other decision seconds already there."""
    path = Path(path)
    data: dict[str, Any] = {"tables": {}}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    for t in tables:
        data["tables"][str(t.decision_s)] = t.to_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def load_table(path: str | Path, decision_s: int) -> CalibrationTable:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python -m bot data calibrate --start .. --end .. "
                                f"--decision-s {decision_s}` first")
    tables = json.loads(path.read_text(encoding="utf-8"))["tables"]
    if str(decision_s) not in tables:
        raise KeyError(f"{path} has no table for decision second {decision_s} (has: {', '.join(sorted(tables))}); "
                       f"run `python -m bot data calibrate ... --decision-s {decision_s}`")
    return CalibrationTable.from_json(tables[str(decision_s)])
