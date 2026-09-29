"""Pure helpers for stream integrity: gap tracking and reconnect backoff."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class ObsKind(StrEnum):
    FIRST = "first"  # first closed candle seen
    NEXT = "next"  # exactly the expected successor
    GAP = "gap"  # skipped one or more candles
    DUPLICATE = "duplicate"  # already seen (or older): drop it


@dataclass(frozen=True, slots=True)
class Observation:
    kind: ObsKind
    missing_start: int | None = None  # half-open [start, end) of missing open_times
    missing_end: int | None = None


class GapTracker:
    """Tracks the last CLOSED candle open_time of one interval."""

    def __init__(self, step_ms: int) -> None:
        self.step = step_ms
        self.last: int | None = None

    def observe(self, open_time: int) -> Observation:
        if open_time % self.step != 0:
            raise ValueError(f"open_time {open_time} not aligned to step {self.step}")
        if self.last is None:
            self.last = open_time
            return Observation(ObsKind.FIRST)
        if open_time <= self.last:
            return Observation(ObsKind.DUPLICATE)
        expected = self.last + self.step
        self.last = open_time
        if open_time == expected:
            return Observation(ObsKind.NEXT)
        return Observation(ObsKind.GAP, expected, open_time)


class Backoff:
    """Exponential backoff with symmetric jitter: initial * factor^n, capped, +/- jitter."""

    def __init__(
        self,
        initial_s: float,
        max_s: float,
        factor: float = 2.0,
        jitter: float = 0.2,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self.initial, self.max, self.factor, self.jitter, self.rng = initial_s, max_s, factor, jitter, rng
        self.attempt = 0

    def next(self) -> float:
        base = min(self.max, self.initial * self.factor**self.attempt)
        self.attempt += 1
        return max(0.0, base * (1 + self.jitter * (2 * self.rng() - 1)))

    def reset(self) -> None:
        self.attempt = 0
