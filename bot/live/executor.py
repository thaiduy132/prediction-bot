"""Turns a strategy decision into a Binance prediction-market order (live) or a real quote (shadow).

Flow per decision, run as a background task so market data keeps flowing:
  risk check -> token id of the side -> real quote -> price/deadline re-check -> [live] place order
Every step is journaled, including the raw JSON of quotes and orders.

The paper trader keeps computing its own simulated result next to this; the journal holds what
really happened. Settlement is booked from the round's Binance 5m candle (same rule as paper)
as an ESTIMATE; `baw prediction position ...` is the source of truth and pending wins are redeemed.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from bot.backtest.strategy import Side
from bot.live.baw import BawClient, BawError
from bot.live.journal import LiveJournal
from bot.live.risk import RiskManager
from bot.logging_setup import log_event
from bot.paper.trader import OpenBet
from bot.settlement import Outcome
from bot.timeutil import now_ms

log = logging.getLogger(__name__)

TokenLookup = Callable[[int], dict[Side, str] | None]  # round_open -> {UP: token id, DOWN: token id}
SEARCH_QUERY = "BTC Up or Down 5m"  # Binance lists only the round that is trading NOW
TOPIC_REFRESH_S = 10.0
ACCOUNT_REFRESH_S = 60.0
RECONCILE_S = 15.0
ORDER_FAILED = ("FAILED", "CANCELLED", "EXPIRED")
USDT_BSC = "0x55d398326f99059ff775485246999027b3197955"


@dataclass(frozen=True, slots=True)
class Topic:
    """Binance's handle for one round: quote needs marketTopicId (verified: it is required)."""

    topic_id: str
    tokens: dict[str, str]  # "UP"/"DOWN" -> ERC1155 token id (verified equal to Predict.fun onChainId)


def topic_from_search(results: Any, slug: str) -> Topic | None:
    for t in results if isinstance(results, list) else []:
        if not isinstance(t, dict) or t.get("slug") != slug or not t.get("markets"):
            continue
        outcomes = t["markets"][0].get("outcomes") or []
        tokens = {str(o.get("name", "")).upper(): str(o["tokenId"]) for o in outcomes if o.get("tokenId")}
        if {"UP", "DOWN"} <= tokens.keys() and t.get("marketTopicId") is not None:
            return Topic(str(t["marketTopicId"]), tokens)
    return None


@dataclass(frozen=True, slots=True)
class ExecConfig:
    live: bool  # False = shadow: quote only, never place an order
    chain_id: int = 56  # BNB Smart Chain (Predict.fun)
    stake_usd: float = 1.0
    slippage_bps: int = 200  # passed to quote and place-order
    max_price_slippage: float = 0.02  # skip if the quote is this much worse than the decision price
    deadline_ms: int = 5_000  # skip if the order could not be sent this long after the decision
    # Binance marks some MARKET orders FAILED ("Failed to execute the market order", ~12s after
    # sending, nothing spent). Such an order is retried with a fresh quote, within this window.
    max_retries: int = 3
    retry_window_ms: int = 60_000  # no new attempt later than this after the decision
    fill_poll_ms: int = 2_000  # how often to ask Binance whether the order filled
    fill_timeout_ms: int = 30_000  # stop asking after this; the reconcile loop takes over


def find_key(data: Any, *names: str) -> Any:
    """First value for any of `names`, searching nested dicts/lists breadth-first. The reply
    shape of quote/place-order could not be inspected without a signed-in wallet."""
    queue = [data]
    while queue:
        cur = queue.pop(0)
        if isinstance(cur, dict):
            for n in names:
                if cur.get(n) is not None:
                    return cur[n]
            queue.extend(cur.values())
        elif isinstance(cur, list):
            queue.extend(cur)
    return None


def _float(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def position_items(data: Any) -> list[Any]:
    """Rows of `baw prediction position list`. Observed shape (2026-09-30):
    {"summary": {...}, "counts": {"ongoingCount", "endedCount", "pendingClaimCount"}, "positions": [...]}"""
    if isinstance(data, list):
        return data
    items = find_key(data, "positions", "list", "items", "rows")
    return items if isinstance(items, list) else []


def parse_quote(data: Any) -> tuple[str | None, float | None, float | None]:
    """(quote id, average price per share, fee amount) from a `trade quote` reply."""
    qid = find_key(data, "quoteId", "quote_id")
    price = _float(find_key(data, "averagePrice", "avgPrice", "price"))
    fee = _float(find_key(data, "feeAmount", "fee"))
    return (None if qid is None else str(qid)), price, fee


class LiveExecutor:
    def __init__(self, cfg: ExecConfig, baw: BawClient, journal: LiveJournal, risk: RiskManager,
                 tokens: TokenLookup, slug_for: Callable[[int], str], round_ms: int = 300_000,
                 clock: Callable[[], int] = now_ms,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        self._sleep = sleep
        self.cfg, self.baw, self.journal, self.risk, self.tokens, self._clock = cfg, baw, journal, risk, tokens, clock
        self.slug_for, self.round_ms = slug_for, round_ms  # tokens: Predict.fun ids, used as a cross-check
        self._topics: dict[int, Topic] = {}
        # Real wallet state read from Binance (the ground truth the dashboard shows first).
        self.account: dict[str, Any] = {"usdt": None, "start_usdt": None, "positions": None, "pending_claim": None,
                                        "today_realized": None, "updated_at": None, "error": None}
        self._outcomes: dict[int, Outcome] = {}  # closed rounds seen, for orders confirmed filled late
        if hasattr(risk, "external_realized"):
            risk.external_realized = lambda: self.account.get("today_realized")
        self.mode = "live" if cfg.live else "shadow"
        self.last_action: dict[str, Any] | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    # ---- hooks called by the paper trader ---------------------------------------------------

    def on_entry(self, bet: OpenBet) -> None:
        task = asyncio.get_running_loop().create_task(self._execute_safely(bet))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def on_settle(self, round_open: int, outcome: Outcome) -> None:
        """Book FILLED orders of a closed round from its 5m candle (estimate until reconciled)."""
        self._outcomes[round_open] = outcome
        for old in [r for r in self._outcomes if r < round_open - 12 * self.round_ms]:
            del self._outcomes[old]
        for row in self.journal.filled_for_round(round_open):
            price, stake = row["quote_price"], row["stake_usd"]
            if outcome is Outcome.VOID:
                status, pnl = "void", 0.0
            elif outcome.value == row["side"]:
                # stake buys stake/price shares paying 1$ each; the BUY fee (feeAmount) is in shares
                status, pnl = "won", (stake / price - (row["quote_fee"] or 0.0) - stake) if price else 0.0
            else:
                status, pnl = "lost", -stake
            self.journal.update(row["id"], status=status, pnl_usd=round(pnl, 6), pnl_source="estimate")
            self._act(round_open, status, f"{pnl:+.3f}$ (estimate, Binance figures follow)")

    async def reconcile(self) -> None:
        """Replace the bot's assumptions with Binance's records: fill status and real PnL."""
        rows = self.journal.to_reconcile()
        if not rows:
            return
        try:
            hist = await self.baw.order_history(limit=100)
            ended = await self.baw.positions("ENDED")
        except BawError as e:
            log_event(log, "live.reconcile_failed", logging.WARNING, error=str(e))
            return
        orders = hist.get("orders") if isinstance(hist, dict) else hist
        status_by_id = {str(o.get("orderId")): str(o.get("status", "")).upper()
                        for o in orders or [] if isinstance(o, dict)}
        pos_by_token = {str(p.get("tokenId")): p for p in position_items(ended) if isinstance(p, dict)}
        for row in rows:
            st = status_by_id.get(str(row["order_id"]))
            if st is None:
                continue
            if st in ORDER_FAILED:
                self.journal.update(row["id"], status="failed", pnl_usd=None, pnl_source="binance",
                                    reason=f"order {st} on Binance (no money spent)")
                self._act(row["round_open"], "failed", f"order {st} on Binance, not filled")
                continue
            if st != "FILLED":
                continue  # PENDING / SUBMITTED / PARTIALLY_FILLED: ask again next time
            status = row["status"]
            if status == "submitted":
                status = "filled"
                self.journal.update(row["id"], status="filled")
                outcome = self._outcomes.get(row["round_open"])
                if outcome is not None:  # the round already closed before the fill was confirmed
                    self.on_settle(row["round_open"], outcome)
                    status = self.journal.row(row["id"])["status"]
            p = pos_by_token.get(str(row["token_id"]))
            if status in ("won", "lost", "void") and p is not None:
                shares, avg = float(p.get("shares") or 0), float(p.get("avgPrice") or 0)
                payout = shares if status == "won" else shares / 2 if status == "void" else 0.0
                pnl = payout - avg * shares
                self.journal.update(row["id"], pnl_usd=round(pnl, 6), pnl_source="binance", quote_price=avg)
                self._act(row["round_open"], status, f"{pnl:+.3f}$ (Binance: {shares} shares @ {avg:.4f})")

    async def reconcile_loop(self) -> None:
        while True:
            try:
                await self.reconcile()
            except Exception as e:  # keep reconciling even if one pass hits an unexpected shape
                log_event(log, "live.reconcile_crash", logging.ERROR, error=repr(e))
            await asyncio.sleep(RECONCILE_S)

    # ---- the order flow ----------------------------------------------------------------------

    async def _execute_safely(self, bet: OpenBet) -> None:
        try:
            await self.execute(bet)
        except Exception as e:  # never lose an error inside a background task
            log_event(log, "live.crash", logging.ERROR, round_open=bet.round_open, error=repr(e))
            self._act(bet.round_open, "failed", f"unexpected error: {e!r}")

    async def execute(self, bet: OpenBet) -> None:
        """Send the order; if Binance fails it, retry with a fresh quote up to `max_retries` times."""
        n = self.cfg.max_retries
        for attempt in range(n + 1):
            row = await self._attempt(bet, attempt)
            if row is None or not self.cfg.live:
                return  # not sent (skipped, failed before sending) or shadow
            status, error = await self._wait_fill(row)
            if status != "FAILED":
                return  # FILLED, or still unknown: the reconcile loop follows it up
            if attempt == n:
                self._act(bet.round_open, "failed", f"gave up after {n} retries ({error})")
                return
            if self._clock() - bet.opened_at > self.cfg.retry_window_ms:
                self._act(bet.round_open, "failed", f"not retrying: more than {self.cfg.retry_window_ms // 1000}s "
                                                    "after the decision")
                return
            self._act(bet.round_open, "retry", f"order FAILED ({error}); retry {attempt + 1}/{n} with a fresh quote")

    async def _attempt(self, bet: OpenBet, attempt: int) -> int | None:
        """One quote (+ order when live). Returns the journal row id if an order was sent."""
        side = bet.side.value
        tag = f"retry {attempt}/{self.cfg.max_retries}: " if attempt else ""
        reason = self.risk.check() if self.cfg.live else None
        if reason:
            self.journal.add(bet.round_open, self.mode, side, self.cfg.stake_usd, "skipped",
                             decided_price=bet.entry_price, reason=tag + reason)
            self._act(bet.round_open, "skipped", tag + reason)
            return None
        topic = await self.topic(bet.round_open)
        token = None if topic is None else topic.tokens.get(side)
        row = self.journal.add(bet.round_open, self.mode, side, self.cfg.stake_usd, "skipped",
                               token_id=token, decided_price=bet.entry_price, reason=tag.strip(" :") or None)
        if topic is None or token is None:
            self.journal.update(row, reason=tag + "round not found on Binance")
            self._act(bet.round_open, "skipped", tag + "round not found on Binance")
            return None
        predict = self.tokens(bet.round_open)
        if predict is not None and predict.get(bet.side) != token:
            # Odds come from Predict.fun: never buy a token that is not the one those odds are for.
            self.journal.update(row, reason="Binance and Predict.fun token ids differ")
            self._act(bet.round_open, "skipped", "Binance and Predict.fun token ids differ")
            return None
        try:
            q = await self.baw.quote(self.cfg.chain_id, token, "BUY", self.cfg.stake_usd,
                                     topic_id=topic.topic_id, slippage_bps=self.cfg.slippage_bps)
        except BawError as e:
            self.journal.update(row, status="failed", reason=f"{tag}quote: {e}")
            self._act(bet.round_open, "failed", f"{tag}quote: {e}")
            return None
        qid, price, fee = parse_quote(q)
        self.journal.update(row, status="quoted", quote_id=qid, quote_price=price, quote_fee=fee, quote_json=q)
        if qid is None or price is None:
            self.journal.update(row, status="failed", reason="could not read quoteId/averagePrice from the reply")
            self._act(bet.round_open, "failed", "unreadable quote (raw JSON is in the journal)")
            return None
        if bet.entry_price is not None and price > bet.entry_price + self.cfg.max_price_slippage:
            # retries keep the ORIGINAL decision price as the limit: the edge was measured there
            msg = f"{tag}price moved {bet.entry_price:.3f} -> {price:.3f}"
            self.journal.update(row, status="skipped", reason=msg)
            self._act(bet.round_open, "skipped", msg)
            return None
        late = self._clock() - bet.opened_at
        limit = self.cfg.deadline_ms if attempt == 0 else self.cfg.retry_window_ms
        if late > limit:
            self.journal.update(row, status="skipped", reason=f"{tag}too late ({late} ms after the decision)")
            self._act(bet.round_open, "skipped", f"{tag}too late ({late} ms)")
            return None
        if not self.cfg.live:
            self._act(bet.round_open, "quoted", f"{side} {self.cfg.stake_usd}$ @ {price:.3f} fee {fee} (shadow)")
            return None

        # ---- real money from here -------------------------------------------------------------
        try:
            o = await self.baw.place_order(qid, self.cfg.slippage_bps)
        except BawError as e:
            self.journal.update(row, status="failed", reason=f"{tag}place-order: {e}")
            self._act(bet.round_open, "failed", f"{tag}place-order: {e}")
            return None
        oid = find_key(o, "orderId", "order_id", "id")
        self.journal.update(row, status="submitted", order_id=None if oid is None else str(oid), order_json=o)
        self._act(bet.round_open, "submitted", f"{tag}{side} {self.cfg.stake_usd}$ @ ~{price:.3f} order {oid} "
                                               "(waiting for Binance to confirm the fill)")
        return row if oid is not None else None

    async def _wait_fill(self, row: int) -> tuple[str | None, str | None]:
        """Ask Binance about this order until it is FILLED or FAILED. Returns (status, error message)."""
        order_id = str(self.journal.row(row)["order_id"])
        for _ in range(max(1, self.cfg.fill_timeout_ms // self.cfg.fill_poll_ms)):
            await self._sleep(self.cfg.fill_poll_ms / 1000)
            try:
                hist = await self.baw.order_history(limit=20)
            except BawError:
                continue
            orders = hist.get("orders") if isinstance(hist, dict) else hist
            o = next((x for x in orders or [] if isinstance(x, dict) and str(x.get("orderId")) == order_id), None)
            st = str((o or {}).get("status", "")).upper()
            if st == "FILLED":
                self.journal.update(row, status="filled")
                self._act(self.journal.row(row)["round_open"], "filled", f"order {order_id} filled")
                return "FILLED", None
            if st in ORDER_FAILED:
                err = (o or {}).get("errorMessage") or st
                self.journal.update(row, status="failed", pnl_usd=None, pnl_source="binance",
                                    reason=f"order {st} on Binance: {err} (no money spent)")
                return "FAILED", str(err)
        return None, None

    # ---- which Binance market is this round -----------------------------------------------------

    async def topic(self, round_open: int) -> Topic | None:
        """Cached Binance topic of that round; searched now if the prefetch has not found it yet."""
        if round_open not in self._topics:
            await self._search(round_open)
        return self._topics.get(round_open)

    async def _search(self, round_open: int) -> None:
        try:
            res = await self.baw.search_markets(SEARCH_QUERY, 50)
        except BawError as e:
            log_event(log, "live.search_failed", logging.WARNING, error=str(e))
            return
        t = topic_from_search(res, self.slug_for(round_open))
        if t is not None:
            self._topics[round_open] = t
            for old in [r for r in self._topics if r < round_open - 2 * self.round_ms]:
                del self._topics[old]

    async def topic_loop(self) -> None:
        """Find each round's topic soon after it opens, so the decision does not wait on a search."""
        while True:
            current = self._clock() // self.round_ms * self.round_ms
            if current not in self._topics:
                await self._search(current)
            await asyncio.sleep(TOPIC_REFRESH_S)

    async def redeem_pending(self) -> None:
        """Claim resolved winning positions so the money comes back to the wallet."""
        if not self.cfg.live:
            return
        try:
            data = await self.baw.positions("PENDING_CLAIM")
        except BawError as e:
            log_event(log, "live.redeem_list_failed", logging.WARNING, error=str(e))
            return
        items = position_items(data)
        ids = sorted({str(t) for it in items if isinstance(it, dict)
                      for t in [find_key(it, "tokenId", "token_id")] if t is not None})
        if not ids:
            return
        try:
            await self.baw.redeem(ids, self.cfg.chain_id)
            log_event(log, "live.redeemed", token_ids=ids)
        except BawError as e:
            log_event(log, "live.redeem_failed", logging.WARNING, error=str(e), token_ids=ids)

    async def refresh_account(self) -> None:
        try:
            bal = await self.baw.run("wallet", "balance", "--binanceChainId", str(self.cfg.chain_id))
            usdt = sum(float(b.get("balance") or 0) for b in (bal or []) if isinstance(b, dict)
                       and str(b.get("address", "")).lower() == USDT_BSC)
            ongoing = await self.baw.positions("ONGOING")
            pending = await self.baw.positions("PENDING_CLAIM")
        except (BawError, ValueError, TypeError) as e:
            self.account["error"] = str(e)
            return
        counts = (pending or {}).get("counts") if isinstance(pending, dict) else None
        summary = (pending or {}).get("summary") if isinstance(pending, dict) else None
        today = _float((summary or {}).get("todayRealizedPnl"))
        if self.account["start_usdt"] is None:
            self.account["start_usdt"] = usdt
        self.account.update(
            usdt=usdt,
            positions=counts.get("ongoingCount") if counts else len(position_items(ongoing)),
            pending_claim=counts.get("pendingClaimCount") if counts else len(position_items(pending)),
            today_realized=today, updated_at=self._clock(), error=None)

    async def account_loop(self) -> None:
        while True:
            await self.refresh_account()
            await asyncio.sleep(ACCOUNT_REFRESH_S)

    def set_stopped(self, stopped: bool) -> None:
        """Dashboard STOP/RESUME: the kill-switch file is what the risk manager checks."""
        path = self.risk.limits.kill_switch_path
        if stopped:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("stopped from the dashboard\n", encoding="utf-8")
        elif path.exists():
            path.unlink()
        self._act(self._clock() // self.round_ms * self.round_ms, "stopped" if stopped else "resumed",
                  "new orders blocked" if stopped else "new orders allowed")

    async def redeem_loop(self, every_s: float = 300.0) -> None:
        while True:
            await asyncio.sleep(every_s)
            await self.redeem_pending()

    # ---- status for the dashboard ----------------------------------------------------------------

    def _act(self, round_open: int, result: str, detail: str) -> None:
        self.last_action = {"round_open": round_open, "result": result, "detail": detail, "at": self._clock()}
        log_event(log, f"live.{result}", mode=self.mode, round_open=round_open, detail=detail)

    def status(self) -> dict[str, Any]:
        from bot.live.risk import DAY_MS
        from bot.timeutil import floor_to

        s = self.journal.day_stats(floor_to(self._clock(), DAY_MS))
        lim = self.risk.limits
        return {
            "mode": self.mode, "stake_usd": self.cfg.stake_usd,
            "limits": {"max_daily_loss_usd": lim.max_daily_loss_usd, "max_bets_per_day": lim.max_bets_per_day},
            "today": {"bets": s.bets, "realized_pnl_usd": round(s.realized_pnl_usd, 4), "open": s.open_count},
            "blocked": self.risk.check() if self.cfg.live else None,
            "stopped": self.risk.limits.kill_switch_path.exists(),
            "account": self.account,
            "last_action": self.last_action,
            "recent": self.journal.recent(15),
        }
