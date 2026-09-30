"""Align the bot's clock with Binance server time instead of trusting the machine clock.

Live commands call `sync_clock` once at startup (it raises if Binance cannot be reached, because
running on a wrong clock silently skips every round) and then run `keep_clock_synced`, which
re-measures every minute so a laptop sleep or an NTP step is corrected within that interval.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from bot.data.binance_rest import BinanceRestClient, BinanceRestError
from bot.logging_setup import log_event
from bot.timeutil import clock_offset_ms, local_ms, set_clock_offset

log = logging.getLogger(__name__)

SAMPLES = 3  # keep the measurement with the smallest round trip: the least network noise
RESYNC_INTERVAL_S = 60.0
JUMP_WARN_MS = 1_000  # offset changed this much since the last sync: the machine clock jumped or drifts


async def measure_offset(rest: BinanceRestClient, samples: int = SAMPLES,
                         clock: Callable[[], int] = local_ms) -> tuple[int, int]:
    """(server - local offset in ms, round trip in ms) of the best of `samples` requests."""
    best: tuple[int, int] | None = None
    for _ in range(samples):
        t0 = clock()
        server = await rest.server_time()
        t1 = clock()
        rtt = t1 - t0
        offset = server - (t0 + t1) // 2
        if best is None or rtt < best[1]:
            best = (offset, rtt)
    assert best is not None
    return best


async def sync_clock(rest: BinanceRestClient) -> int:
    """Measure and apply the offset. Raises BinanceRestError if the server time cannot be read."""
    offset, rtt = await measure_offset(rest)
    set_clock_offset(offset)
    log_event(log, "clock.synced", offset_ms=offset, rtt_ms=rtt)
    return offset


async def keep_clock_synced(rest: BinanceRestClient, interval_s: float = RESYNC_INTERVAL_S,
                            sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
    """Re-measure forever. On failure keep the last offset (it is still the best estimate)."""
    while True:
        await sleep(interval_s)
        before = clock_offset_ms()
        try:
            offset, rtt = await measure_offset(rest)
        except BinanceRestError as e:
            log_event(log, "clock.sync_failed", logging.WARNING, error=str(e), kept_offset_ms=before)
            continue
        set_clock_offset(offset)
        if abs(offset - before) >= JUMP_WARN_MS:
            log_event(log, "clock.jump", logging.WARNING, previous_offset_ms=before, offset_ms=offset, rtt_ms=rtt)


def describe_offset(offset_ms: int) -> str:
    if abs(offset_ms) < 1000:
        return f"Đồng hồ máy khớp Binance (lệch {-offset_ms} ms)."
    side = "chậm" if offset_ms > 0 else "nhanh"
    return (f"Đồng hồ máy {side} {abs(offset_ms) / 1000:.1f}s so với Binance: bot dùng giờ Binance thay cho giờ máy "
            "(nên sửa NTP của máy).")
