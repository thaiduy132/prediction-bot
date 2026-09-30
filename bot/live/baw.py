"""Thin async wrapper around the Binance Agentic Wallet CLI (`baw`, npm @binance/agentic-wallet).

Every call runs `baw ... --json` as a subprocess (no shell) and returns the `data` field of a
`{"success": true, "data": ...}` reply, or raises BawError. Authentication lives entirely inside
`baw` (the user runs `baw auth signin` once); this bot never sees keys or wallet secrets.

Command names and flags follow `baw prediction ... --help` of version 1.10.0. The SHAPE of the
data returned by quote/place-order was not observable without a signed-in wallet, so callers
read it through the tolerant helpers in bot.live.executor and journal the raw JSON.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

from bot.logging_setup import log_event

log = logging.getLogger(__name__)

# (argv) -> (exit code, stdout, stderr); injectable for tests
Runner = Callable[[list[str], float], Awaitable[tuple[int, str, str]]]


class BawError(RuntimeError):
    def __init__(self, message: str, code: int | None = None, name: str | None = None) -> None:
        super().__init__(message)
        self.code, self.name = code, name

    @property
    def not_logged_in(self) -> bool:
        return self.name == "NOT_LOGGED_IN"


def baw_env() -> dict[str, str]:
    """Environment for `baw`. Node ignores HTTP(S)_PROXY unless NODE_USE_ENV_PROXY=1 (Node >= 22.21),
    so behind a proxy every call timed out while Python (httpx) worked."""
    env = dict(os.environ)
    env.setdefault("NODE_USE_ENV_PROXY", "1")
    return env


async def _subprocess_runner(argv: list[str], timeout_s: float) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=baw_env())
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout_s)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise BawError(f"`{' '.join(argv[:4])}` timed out after {timeout_s}s") from None
    return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


def _num(v: float | int) -> str:
    return f"{v:.6f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


class BawClient:
    def __init__(self, exe: str = "baw", timeout_s: float = 20.0, runner: Runner = _subprocess_runner) -> None:
        self.exe, self.timeout_s, self._run_argv = exe, timeout_s, runner

    async def run(self, *args: str) -> Any:
        argv = [self.exe, *args, "--json"]
        try:
            code, out, err = await self._run_argv(argv, self.timeout_s)
        except FileNotFoundError:
            raise BawError(f"`{self.exe}` not found: npm install -g @binance/agentic-wallet") from None
        try:
            reply = json.loads(out)
        except json.JSONDecodeError:
            raise BawError(f"`{' '.join(args[:3])}` returned non-JSON (exit {code}): {(out or err)[:300]}") from None
        if not isinstance(reply, dict) or not reply.get("success"):
            e = (reply or {}).get("error") or {} if isinstance(reply, dict) else {}
            raise BawError(f"`{' '.join(args[:3])}` failed: {e.get('name')} {e.get('message')}".strip(),
                           e.get("code"), e.get("name"))
        return reply.get("data")

    # ---- read-only -----------------------------------------------------------------------

    async def wallet_status(self) -> Any:
        return await self.run("wallet", "status")

    async def search_markets(self, query: str, limit: int = 20) -> Any:
        return await self.run("prediction", "market", "search", "--query", query, "--limit", str(limit))

    async def market_detail(self, topic_id: str) -> Any:
        return await self.run("prediction", "market", "detail", "--marketTopicId", str(topic_id))

    async def order_history(self, status: str | None = None, limit: int = 50) -> Any:
        args = ["prediction", "order", "history", "--limit", str(limit)]
        if status:
            args += ["--status", status]
        return await self.run(*args)

    async def positions(self, tab: str = "ONGOING", limit: int = 100) -> Any:
        return await self.run("prediction", "position", "list", "--tab", tab, "--limit", str(limit))

    async def quote(self, chain_id: int, token_id: str, side: str, amount_usd: float,
                    topic_id: str | None = None, slippage_bps: int | None = None,
                    order_type: str = "MARKET", price_limit: float | None = None) -> Any:
        """A price quote. It does NOT place an order and costs nothing."""
        args = ["prediction", "trade", "quote", "--binanceChainId", str(chain_id), "--tokenId", str(token_id),
                "--side", side, "--amount", _num(amount_usd), "--orderType", order_type]
        if topic_id:
            args += ["--marketTopicId", str(topic_id)]
        if slippage_bps is not None:
            args += ["--slippageBps", str(slippage_bps)]
        if price_limit is not None:
            args += ["--priceLimit", _num(price_limit)]
        return await self.run(*args)

    # ---- spends money ----------------------------------------------------------------------

    async def place_order(self, quote_id: str, slippage_bps: int, order_type: str = "MARKET",
                          price_limit: float | None = None) -> Any:
        """Submit a REAL order for a previous quote. Only the live executor calls this."""
        args = ["prediction", "trade", "place-order", "--quoteId", str(quote_id), "--slippageBps", str(slippage_bps),
                "--orderType", order_type]
        if price_limit is not None:
            args += ["--priceLimit", _num(price_limit)]
        log_event(log, "baw.place_order", quote_id=quote_id, slippage_bps=slippage_bps, order_type=order_type)
        return await self.run(*args)

    async def redeem(self, token_ids: list[str], chain_id: int | None = None) -> Any:
        args = ["prediction", "trade", "redeem", "--tokenIds", ",".join(token_ids)]
        if chain_id is not None:
            args += ["--binanceChainId", str(chain_id)]
        return await self.run(*args)
