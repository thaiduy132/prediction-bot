"""Command line entry point: python -m bot <command> ..."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter

from bot.config import AppConfig, load_config
from bot.data.binance_rest import BinanceRestClient
from bot.data.binance_ws import BinanceStream, BookTickerEvent, ConnectionEvent, GapEvent, KlineEvent
from bot.data.feed import LiveFeed
from bot.data.kline_cache import KlineCache, load_klines
from bot.logging_setup import log_event, setup_logging
from bot.secrets import load_secrets
from bot.settlement import cross_check_with_1m, settle_kline
from bot.storage.db import connect
from bot.timeutil import interval_ms, ms_to_iso, now_ms, parse_utc

log = logging.getLogger("bot")

MODES = ("backtest", "paper", "shadow", "live")


def _rest(cfg: AppConfig) -> BinanceRestClient:
    return BinanceRestClient(cfg.rest.base_url, cfg.rest.timeout_s, cfg.rest.max_retries)


# ---- data fetch ----------------------------------------------------------------

async def cmd_data_fetch(cfg: AppConfig, start: str, end: str) -> int:
    """Fill the cache for [start, end) and print a data-quality + settlement summary."""
    start_ms, end_ms = parse_utc(start), parse_utc(end)
    sym, rnd = cfg.market.symbol, cfg.market.round_interval
    cache = KlineCache(cfg.data.cache_path)
    try:
        async with _rest(cfg) as rest:
            k5 = await load_klines(rest, cache, sym, rnd, start_ms, end_ms)
            k1 = await load_klines(rest, cache, sym, "1m", start_ms, end_ms)
        by_open = {}
        for k in k1:
            by_open.setdefault(k.open_time - k.open_time % interval_ms(rnd), []).append(k)
        mismatches = []
        for k in k5:
            probs = cross_check_with_1m(k, by_open.get(k.open_time, []))
            if probs:
                mismatches.append({"open_time": ms_to_iso(k.open_time), "problems": probs})
        outcomes = Counter(settle_kline(k, cfg.settlement.rule, rnd).value for k in k5)
        ties = sum(1 for k in k5 if k.close == k.open)
        summary = {
            "symbol": sym,
            "range": [ms_to_iso(start_ms), ms_to_iso(end_ms)],
            f"candles_{rnd}": len(k5),
            "candles_1m": len(k1),
            f"exchange_gaps_{rnd}": len(cache.known_gaps(sym, rnd, start_ms, end_ms)),
            "exchange_gaps_1m": len(cache.known_gaps(sym, "1m", start_ms, end_ms)),
            "cross_check_1m_vs_5m_mismatches": len(mismatches),
            "settlement_rule": cfg.settlement.rule.value,
            "outcomes": dict(outcomes),
            "exact_ties": ties,
            "up_rate": round(outcomes["UP"] / len(k5), 4) if k5 else None,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        for m in mismatches[:10]:
            print(json.dumps(m, ensure_ascii=False), file=sys.stderr)
        return 0 if not mismatches else 2
    finally:
        cache.close()


# ---- data stream -----------------------------------------------------------------

async def cmd_data_stream(cfg: AppConfig, duration_s: float) -> int:
    """Run the live feed for a while and log closed candles, gaps and reconnects."""
    status_conn = connect(cfg.data.cache_path)
    cache = KlineCache(cfg.data.cache_path)
    counts: Counter[str] = Counter()
    async with _rest(cfg) as rest:
        stream = BinanceStream(cfg.websocket, cfg.market.symbol)
        feed = LiveFeed(stream, rest, cfg.websocket.kline_intervals, cache=cache, status_conn=status_conn)

        async def consume() -> None:
            async for ev in feed.events():
                match ev:
                    case BookTickerEvent():
                        counts["book_ticker"] += 1
                    case KlineEvent(kline=k, source=src) if k.is_closed:
                        counts[f"closed_{k.interval}_{src}"] += 1
                        log_event(log, "kline.closed", interval=k.interval, open_time=ms_to_iso(k.open_time),
                                  open=k.open, close=k.close, source=src,
                                  outcome=settle_kline(k, cfg.settlement.rule, k.interval).value)
                    case KlineEvent():
                        counts["kline_update"] += 1
                    case GapEvent():
                        counts["gaps"] += 1
                    case ConnectionEvent(state=s):
                        counts[f"conn_{s}"] += 1

        try:
            await asyncio.wait_for(consume(), timeout=duration_s)
        except TimeoutError:
            pass
    status_conn.close()
    cache.close()
    book = feed.last_book
    print(json.dumps({"duration_s": duration_s, "counts": dict(counts),
                      "last_book": None if book is None else {"bid": book.bid, "ask": book.ask, "spread": book.spread}},
                     indent=2))
    return 0


async def cmd_data_clock(cfg: AppConfig) -> int:
    async with _rest(cfg) as rest:
        t0 = now_ms()
        server = await rest.server_time()
        t1 = now_ms()
    skew = server - (t0 + t1) // 2
    print(json.dumps({"server_time": ms_to_iso(server), "local_minus_server_ms": -skew, "rtt_ms": t1 - t0}, indent=2))
    if abs(skew) > 1000:
        print("CẢNH BÁO: đồng hồ máy lệch > 1s so với Binance, hãy bật NTP.", file=sys.stderr)
        return 2
    return 0


# ---- parser --------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m bot")
    p.add_argument("--config", default=None, help="YAML config file (default: built-in defaults)")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the bot in a mode")
    run.add_argument("--mode", choices=MODES, required=True)
    run.add_argument("--config", dest="run_config", default=None)
    run.add_argument("--i-understand-the-risk", action="store_true")

    rep = sub.add_parser("report", help="print evaluation report")
    rep.add_argument("--config", dest="run_config", default=None)

    data = sub.add_parser("data", help="market data utilities")
    dsub = data.add_subparsers(dest="data_command", required=True)
    f = dsub.add_parser("fetch", help="download klines into the cache and check quality")
    f.add_argument("--start", required=True, help="UTC, e.g. 2026-09-01 or 2026-09-01T12:00")
    f.add_argument("--end", required=True)
    s = dsub.add_parser("stream", help="watch the live feed")
    s.add_argument("--duration", type=float, default=180.0, help="seconds")
    dsub.add_parser("clock", help="compare local clock with Binance server time")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(getattr(args, "run_config", None) or args.config)
    setup_logging(cfg.logging.level, cfg.logging.file)
    load_secrets()

    if args.command == "data":
        match args.data_command:
            case "fetch":
                return asyncio.run(cmd_data_fetch(cfg, args.start, args.end))
            case "stream":
                return asyncio.run(cmd_data_stream(cfg, args.duration))
            case "clock":
                return asyncio.run(cmd_data_clock(cfg))
    if args.command == "run":
        print(f"mode '{args.mode}' chưa được triển khai (bước 2-5).", file=sys.stderr)
        return 1
    if args.command == "report":
        print("report chưa được triển khai (bước 6).", file=sys.stderr)
        return 1
    return 1
