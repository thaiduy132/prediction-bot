"""Stake per bet that grows with the account, in steps, and shrinks back when it falls.

tier = how many times equity has multiplied by `step_multiple` over `base_capital_usd`
stake = base_stake_usd * factor ** tier, capped at max_stake_usd

Default use: base capital 11.28$ (the deposit), step x2, factor 1.5 -> 1$ until 22.56$, 1.5$ from
22.56$, 2.25$ from 45.12$, ... Equity below a threshold drops back to the lower tier, so losses
are never pressed with a bigger stake.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StakeScaling:
    base_stake_usd: float
    base_capital_usd: float | None = None  # None: always base_stake_usd
    step_multiple: float = 2.0
    factor: float = 1.5
    max_stake_usd: float = 5.0

    def __post_init__(self) -> None:
        if self.step_multiple <= 1 or self.factor <= 0 or self.base_stake_usd <= 0:
            raise ValueError("step_multiple must be > 1, factor and base_stake_usd > 0")

    def tier(self, equity_usd: float | None) -> int:
        if self.base_capital_usd is None or equity_usd is None or equity_usd <= 0:
            return 0
        ratio = equity_usd / self.base_capital_usd
        if ratio < self.step_multiple:
            return 0
        return int(math.floor(math.log(ratio) / math.log(self.step_multiple) + 1e-9))

    def stake(self, equity_usd: float | None) -> float:
        raw = self.base_stake_usd * self.factor ** self.tier(equity_usd)
        return round(min(raw, max(self.max_stake_usd, self.base_stake_usd)), 2)

    def next_threshold(self, equity_usd: float | None) -> float | None:
        """Equity at which the stake steps up next, or None if scaling is off or capped."""
        if self.base_capital_usd is None:
            return None
        t = self.tier(equity_usd)
        if self.base_stake_usd * self.factor ** t >= self.max_stake_usd:
            return None
        return round(self.base_capital_usd * self.step_multiple ** (t + 1), 2)
