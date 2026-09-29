"""Time helpers. All timestamps inside the bot are UTC epoch milliseconds (int)."""

from __future__ import annotations

import time
from datetime import UTC, datetime

_UNIT_MS = {"s": 1_000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}


def interval_ms(interval: str) -> int:
    """'1m' -> 60000, '5m' -> 300000, '1h' -> 3600000 ..."""
    if len(interval) < 2 or interval[-1] not in _UNIT_MS or not interval[:-1].isdigit():
        raise ValueError(f"unsupported interval: {interval!r}")
    n = int(interval[:-1])
    if n <= 0:
        raise ValueError(f"unsupported interval: {interval!r}")
    return n * _UNIT_MS[interval[-1]]


def floor_to(ts_ms: int, step_ms: int) -> int:
    return ts_ms - (ts_ms % step_ms)


def ceil_to(ts_ms: int, step_ms: int) -> int:
    r = ts_ms % step_ms
    return ts_ms if r == 0 else ts_ms + (step_ms - r)


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def parse_utc(s: str) -> int:
    """Parse '2026-09-01', '2026-09-01T12:00', '2026-09-01T12:00:00Z' (naive = UTC) -> epoch ms."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat().replace("+00:00", "Z")
