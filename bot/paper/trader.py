"""Live paper trader: same decision code as the backtest, fed by live data, no real orders.

Flow per round: at `decision_s` seconds after the round opens (when that second's 1s candle
closes) ask the strategy; if it bets, "buy" at the current Predict.fun quote; when the round's
5m candle closes, settle with the same rule as the backtest and book the PnL.
Driven by events (on_kline / on_book / on_odds) and holds no I/O other than the optional store.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from bot.backtest.calibration import SIGMA_WINDOW_S
from bot.backtest.engine import BacktestResult, Trade, decide_entry
from bot.backtest.pricing import bet_pnl
from bot.backtest.stats import summarize
from bot.backtest.strategy import RoundContext, Side, Strategy, accept_retry
from bot.data.book_store import BookSample
from bot.data.odds_store import OddsSample
from bot.models import Kline
from bot.paper.store import PaperStore, PaperTradeRow
from bot.settlement import Outcome, SettlementRule, settle_kline
from bot.timeutil import floor_to, interval_ms, now_ms

KEEP_ROUNDS = 3  # per-round bookkeeping older than this many rounds is dropped
SHOWN_TRADES = 100


@dataclass(frozen=True, slots=True)
class OpenBet:
    round_open: int
    side: Side
    entry_price: float | None
    move_bps: float
    opened_at: int


class PaperTrader:
    def __init__(
        self,
        strategy: Strategy,
        rule: SettlementRule,
        round_interval: str = "5m",
        decision_s: int = 270,
        fee_bps: float = 0.0,
        max_entry_price: float = 1.0,
        use_odds: bool = True,
        payout_ratio: float = 0.9,
        store: PaperStore | None = None,
        clock: Callable[[], int] = now_ms,
    ) -> None:
        self.strategy, self.rule, self.round_interval = strategy, rule, round_interval
        self.step = interval_ms(round_interval)
        if not 0 < decision_s * 1000 < self.step:
            raise ValueError(f"decision_s must be inside the round (0 < {decision_s} < {self.step // 1000})")
        self.decision_s, self.fee_bps, self.max_entry_price = decision_s, fee_bps, max_entry_price
        self.use_odds, self.payout_ratio = use_odds, payout_ratio
        self.store, self._clock = store, clock
        self.label = f"{strategy!r}|{decision_s}|{'odds' if use_odds else 'fixed'}|fee{fee_bps}"

        self.result = BacktestResult()
        self.times: dict[int, tuple[int, int]] = {}  # round_open -> (opened_at, settled_at)
        self.open: dict[int, OpenBet] = {}
        self.last_decision: dict[str, Any] | None = None
        self.last_odds: OddsSample | None = None
        self._seconds: dict[int, Kline] = {}
        self._round_open_px: dict[int, float] = {}
        self._book: deque[BookSample] = deque(maxlen=90)
        self._odds: deque[OddsSample] = deque(maxlen=30)
        self._decided: set[int] = set()
        self._seen_rounds: set[int] = set()
        self.counts: Counter[str] = Counter()
        # Optional hooks for the shadow/live executor (bot.live.executor): called when a bet is taken
        # and when the round it was taken in settles. They must return quickly.
        self.on_entry: Callable[[OpenBet], None] | None = None
        self.on_settle: Callable[[int, Outcome], None] | None = None
        self.reload()

    def reload(self) -> None:
        """(Re)load this strategy's past trades from the store; `store` may be attached after __init__."""
        if self.store is None:
            return
        self.result.trades.clear()
        self.times.clear()
        for r in self.store.load(self.label):
            self.result.trades.append(Trade(r.round_open, Side(r.side), Outcome(r.outcome), r.pnl,
                                            r.move_bps, r.entry_price))
            self.times[r.round_open] = (r.opened_at, r.settled_at)

    # ---- inputs ---------------------------------------------------------------------

    def current_context(self, round_open: int) -> RoundContext | None:
        """The round as it stands now (latest closed second), for re-deciding on a retry."""
        last = max((t for t in self._seconds if round_open <= t < round_open + self.step), default=None)
        if last is None:
            return None
        decision_time = last + 1000
        recent = [self._seconds[t] for t in range(round_open, decision_time, 1000) if t in self._seconds]
        px = self._round_open_px.get(round_open)
        if px is None and round_open in self._seconds:
            px = self._seconds[round_open].open
        if px is None or not recent:
            return None
        lookback = [self._seconds[t] for t in range(decision_time - SIGMA_WINDOW_S * 1000, decision_time, 1000)
                    if t in self._seconds]
        quotes = [q for q in self._odds if q.round_start == round_open]
        return RoundContext(round_open, px, decision_time, recent, [b for b in self._book if b.ts < decision_time],
                            lookback, quotes[-1] if quotes else None)

    def recheck(self, bet: OpenBet, price: float, max_price: float) -> str | None:
        """Hook for the live executor: is buying `bet.side` at `price` still a good decision now?"""
        ctx = self.current_context(bet.round_open)
        if ctx is None:
            return "no fresh market data for this round"
        return accept_retry(self.strategy, ctx, bet.side, price, max_price)

    def seed_seconds(self, klines: list[Kline]) -> None:
        """Store past closed 1s candles (e.g. history loaded at startup) without deciding on them,
        so the volatility lookback is full from the first live round."""
        for k in klines:
            if k.interval == "1s" and k.is_closed:
                self._seconds[k.open_time] = k

    def on_book(self, s: BookSample) -> None:
        self._book.append(s)

    def on_odds(self, q: OddsSample | None) -> bool:
        if q is None:
            return False
        self._odds.append(q)
        self.last_odds = q
        return True

    def on_kline(self, k: Kline) -> bool:
        """Feed any kline event. Returns True when trades/positions changed."""
        if k.interval == "1s" and k.is_closed:
            return self._on_second(k)
        if k.interval == self.round_interval:
            if not k.is_closed:
                self._round_open_px[k.open_time] = k.open
                return False
            return self._on_round_closed(k)
        return False

    # ---- decisions --------------------------------------------------------------------

    def _on_second(self, k: Kline) -> bool:
        self._seconds[k.open_time] = k
        round_open = floor_to(k.open_time, self.step)
        decision_time = round_open + self.decision_s * 1000
        self._prune(round_open)
        if round_open in self._decided or k.open_time < decision_time - 1000:
            return False
        self._decided.add(round_open)
        if k.open_time > decision_time - 1000:
            # The decision second's candle never arrived live (we started mid-round or it was lost):
            # deciding now would use a later moment than the backtest does, so skip the round.
            self._record_skip(round_open, "no_data")
            return True

        recent = [self._seconds[t] for t in range(round_open, decision_time, 1000) if t in self._seconds]
        book = [b for b in self._book if b.ts < decision_time]  # a sample is knowable from ts + 1000
        px = self._round_open_px.get(round_open)
        if px is None and round_open in self._seconds:
            px = self._seconds[round_open].open  # 1s candle at the round start has the same open
        if px is None or not recent:
            self._record_skip(round_open, "no_data")
            return True
        odds = [q for q in self._odds if q.round_start == round_open] if self.use_odds else None
        lookback = [self._seconds[t] for t in range(decision_time - SIGMA_WINDOW_S * 1000, decision_time, 1000)
                    if t in self._seconds]
        entry, skip = decide_entry(self.strategy, round_open, px, decision_time, recent, book, odds,
                                   self.max_entry_price, lookback)
        if entry is None:
            self._record_skip(round_open, skip or "unknown")
            return True
        bet = OpenBet(round_open, entry.side, entry.price, entry.move_bps, self._clock())
        self.open[round_open] = bet
        if self.on_entry is not None:
            self.on_entry(bet)
        self.last_decision = {"round_open": round_open, "result": "bet", "side": entry.side.value,
                              "price": entry.price, "at": self._clock()}
        return True

    def _record_skip(self, round_open: int, reason: str) -> None:
        self.counts[f"skipped_{reason}"] += 1
        setattr(self.result, f"rounds_skipped_{reason}", getattr(self.result, f"rounds_skipped_{reason}") + 1)
        self.last_decision = {"round_open": round_open, "result": f"skip: {reason}", "at": self._clock()}

    # ---- settlement -------------------------------------------------------------------

    def _on_round_closed(self, k: Kline) -> bool:
        if k.open_time in self._seen_rounds:
            return False
        outcome = settle_kline(k, self.rule, self.round_interval)
        self._seen_rounds.add(k.open_time)
        self.result.rounds_total += 1
        self.result.outcomes[outcome.value] = self.result.outcomes.get(outcome.value, 0) + 1
        if self.on_settle is not None:
            # every closed round, not only rounds with a paper bet: real orders placed before a
            # restart must still be settled
            self.on_settle(k.open_time, outcome)
        bet = self.open.pop(k.open_time, None)
        if bet is None:
            return False
        if bet.entry_price is None:
            pnl = 0.0 if outcome is Outcome.VOID else (self.payout_ratio if outcome.value == bet.side.value else -1.0)
        else:
            pnl = bet_pnl(bet.side, outcome, bet.entry_price, self.fee_bps)
        settled_at = self._clock()
        self.result.trades.append(Trade(k.open_time, bet.side, outcome, pnl, bet.move_bps, bet.entry_price))
        self.times[k.open_time] = (bet.opened_at, settled_at)
        if self.store is not None:
            self.store.save(self.label, PaperTradeRow(k.open_time, bet.side.value, bet.entry_price, bet.move_bps,
                                                      bet.opened_at, outcome.value, pnl, settled_at))
        return True

    def _prune(self, current_round: int) -> None:
        cutoff = current_round - KEEP_ROUNDS * self.step
        for d in (self._seconds, self._round_open_px):
            for t in [t for t in d if t < cutoff]:
                del d[t]
        self._decided = {t for t in self._decided if t >= cutoff}
        self._seen_rounds = {t for t in self._seen_rounds if t >= cutoff}

    # ---- output for the dashboard --------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        stats = summarize(self.result, self.payout_ratio, self.fee_bps)
        cum, curve, rows = 0.0, [], []
        for t in self.result.trades:
            cum += t.pnl
            curve.append(round(cum, 6))
            opened, settled = self.times.get(t.round_open_time, (None, None))
            rows.append({"round_open": t.round_open_time, "side": t.side.value, "entry_price": t.entry_price,
                         "outcome": t.outcome.value, "pnl": round(t.pnl, 6), "cum_pnl": round(cum, 6),
                         "move_bps": round(t.move_bps, 3), "opened_at": opened, "settled_at": settled})
        now = self._clock()
        cur_round = floor_to(now, self.step)
        q = self.last_odds
        return {
            "enabled": True,
            "strategy": self.strategy.name,
            "pricing": "odds" if self.use_odds else "fixed payout",
            "decision_s": self.decision_s, "fee_bps": self.fee_bps, "max_entry_price": self.max_entry_price,
            "payout_ratio": None if self.use_odds else self.payout_ratio,
            "stats": {k: stats.get(k) for k in (
                "bets", "wins", "losses", "voids", "win_rate", "breakeven_win_rate", "pnl_units", "roi_per_bet",
                "max_drawdown_units", "avg_entry_price", "p_value_vs_breakeven")},
            "skipped": {k[len("rounds_skipped_"):]: v for k, v in stats.items()
                        if k.startswith("rounds_skipped_") and v},
            "open": [{"round_open": b.round_open, "side": b.side.value, "entry_price": b.entry_price,
                      "move_bps": round(b.move_bps, 3), "opened_at": b.opened_at,
                      "ends_at": b.round_open + self.step} for b in self.open.values()],
            "trades": rows[-SHOWN_TRADES:],
            "curve": curve,
            "last_decision": self.last_decision,
            "next_decision_at": (cur_round + self.decision_s * 1000
                                 if cur_round not in self._decided else cur_round + self.step + self.decision_s * 1000),
            "odds": None if q is None else {"up_bid": q.up_bid, "up_ask": q.up_ask, "ts": q.ts,
                                            "round_start": q.round_start},
        }
