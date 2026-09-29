"""Order-book features from recorded 1-second samples."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from bot.data.book_store import BookSample


def imbalance(bid_qty: float, ask_qty: float) -> float:
    """(bid - ask) / (bid + ask) in [-1, 1]; > 0 means more resting buy size than sell size."""
    total = bid_qty + ask_qty
    return (bid_qty - ask_qty) / total if total > 0 else 0.0


@dataclass(frozen=True, slots=True)
class BookFeatures:
    spread_bps: float  # latest spread relative to mid
    imbalance: float  # depth imbalance, averaged over the window
    imbalance_last: float  # depth imbalance of the latest sample only
    microprice_bps: float  # size-weighted top-of-book price vs mid; > 0 leans up
    samples: int  # how many samples the average used


def book_features(samples: Sequence[BookSample], levels: int = 10, window_s: int = 10) -> BookFeatures | None:
    """Features from the most recent `window_s` seconds of `samples` (ascending). None if empty."""
    if not samples:
        return None
    last = samples[-1]
    recent = [s for s in samples if s.ts > last.ts - window_s * 1000]
    imbs = [imbalance(*s.depth(levels)) for s in recent]
    mid = (last.bid + last.ask) / 2
    top = last.bq1 + last.aq1
    micro = (last.bq1 * last.ask + last.aq1 * last.bid) / top if top > 0 else mid
    return BookFeatures(
        spread_bps=(last.ask - last.bid) / mid * 10_000,
        imbalance=sum(imbs) / len(imbs),
        imbalance_last=imbs[-1],
        microprice_bps=(micro / mid - 1) * 10_000,
        samples=len(recent),
    )
