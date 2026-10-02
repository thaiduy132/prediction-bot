"""Hard limits checked before every real order. Any failing check means: do not trade."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from bot.live.journal import LiveJournal
from bot.timeutil import floor_to, now_ms

DAY_MS = 86_400_000


@dataclass(frozen=True, slots=True)
class RiskLimits:
    stake_usd: float  # per bet
    max_daily_loss_usd: float  # worst case: open bets are counted as lost
    max_bets_per_day: int
    max_open_positions: int
    kill_switch_path: Path  # if this file exists, no new orders

    def __post_init__(self) -> None:
        if self.stake_usd <= 0 or self.stake_usd > self.max_daily_loss_usd:
            raise ValueError("stake_usd must be > 0 and <= max_daily_loss_usd")


class RiskManager:
    def __init__(self, limits: RiskLimits, journal: LiveJournal, clock: Callable[[], int] = now_ms,
                 external_realized: Callable[[], float | None] = lambda: None) -> None:
        self.limits, self.journal, self._clock = limits, journal, clock
        # Binance's own "today realized PnL"; the worse of it and the journal is used, so a bookkeeping
        # error in the bot can never loosen the loss limit.
        self.external_realized = external_realized

    def check(self, stake_usd: float | None = None) -> str | None:
        """None if a new bet of `stake_usd` (default: limits.stake_usd) is allowed, else the reason."""
        lim = self.limits
        stake = lim.stake_usd if stake_usd is None else stake_usd
        if lim.kill_switch_path.exists():
            return f"kill switch ({lim.kill_switch_path} exists)"
        s = self.journal.day_stats(floor_to(self._clock(), DAY_MS))
        if s.open_count >= lim.max_open_positions:
            return f"{s.open_count} position(s) still open"
        if s.bets >= lim.max_bets_per_day:
            return f"daily bet limit reached ({s.bets})"
        realized = s.realized_pnl_usd
        ext = self.external_realized()
        if ext is not None:
            realized = min(realized, ext)
        worst = -realized + s.open_stake_usd + stake
        if worst > lim.max_daily_loss_usd:
            return (f"daily loss limit: realized {realized:+.2f}$, open {s.open_stake_usd:.2f}$, "
                    f"next bet {stake:.2f}$ could reach -{worst:.2f}$ > -{lim.max_daily_loss_usd:.2f}$")
        return None
