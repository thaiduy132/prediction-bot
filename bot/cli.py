"""Command line entry point: python -m bot <command> ..."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass

from bot.backtest.engine import run_backtest
from bot.backtest.stats import summarize
from bot.backtest.strategy import BookParams, Strategy, make_strategy
from bot.config import AppConfig, load_config
from bot.data.binance_rest import BinanceRestClient
from bot.data.binance_ws import BinanceStream, BookTickerEvent, ConnectionEvent, DepthEvent, GapEvent, KlineEvent
from bot.data.book_store import BookRecorder, BookStore
from bot.data.feed import LiveFeed
from bot.data.kline_cache import KlineCache, load_klines
from bot.data.odds_store import OddsStore
from bot.logging_setup import log_event, setup_logging
from bot.paper.trader import PaperTrader
from bot.secrets import load_secrets
from bot.settlement import cross_check_with_1m, settle_kline
from bot.storage.db import connect
from bot.timeutil import ceil_to, floor_to, interval_ms, ms_to_iso, now_ms, parse_utc

log = logging.getLogger("bot")

MODES = ("backtest", "paper", "shadow", "live")
STRATEGIES = ("momentum", "reversal", "always_up", "book", "momentum_book")


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


# ---- backtest --------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyParams:
    """Strategy + pricing settings: config.yaml `backtest:` with CLI overrides. Shared by backtest and paper."""

    name: str
    strategy: Strategy
    decision_s: int
    min_move: float
    payout: float
    book_params: BookParams
    use_odds: bool
    fee_bps: float
    max_entry_price: float


def _strategy_params(cfg: AppConfig, args: argparse.Namespace) -> StrategyParams:
    bt = cfg.backtest
    name = args.strategy or bt.strategy
    min_move = bt.min_move_bps if args.min_move_bps is None else args.min_move_bps
    book_params = BookParams(
        levels=args.book_levels or bt.book_levels,
        window_s=args.book_window_s or bt.book_window_s,
        min_imbalance=bt.min_imbalance if args.min_imbalance is None else args.min_imbalance,
        max_spread_bps=bt.max_spread_bps if args.max_spread_bps is None else args.max_spread_bps,
    )
    return StrategyParams(
        name=name,
        strategy=make_strategy(name, min_move, book_params),
        decision_s=args.decision_s or bt.decision_s,
        min_move=min_move,
        payout=args.payout or bt.payout_ratio,
        book_params=book_params,
        use_odds=args.odds or bt.use_odds,
        fee_bps=bt.fee_bps if args.fee_bps is None else args.fee_bps,
        max_entry_price=bt.max_entry_price if args.max_entry_price is None else args.max_entry_price,
    )


async def cmd_backtest(cfg: AppConfig, args: argparse.Namespace) -> int:
    if not (args.start and args.end):
        print("backtest cần --start và --end (UTC, vd 2026-09-01).", file=sys.stderr)
        return 1
    bt = cfg.backtest
    p = _strategy_params(cfg, args)
    name, decision_s, min_move, payout, book_params, strategy = (
        p.name, p.decision_s, p.min_move, p.payout, p.book_params, p.strategy)
    start_ms, end_ms = parse_utc(args.start), parse_utc(args.end)
    sym, rnd = cfg.market.symbol, cfg.market.round_interval

    book_samples = []
    if strategy.needs_book:
        store = BookStore(cfg.data.book_path, sym)
        try:
            cov = store.coverage()
            if cov is None:
                print("Chưa có dữ liệu order book. Chạy `python -m bot data record` (hoặc `ui`) một lúc trước.",
                      file=sys.stderr)
                return 1
            # only backtest rounds that the recording actually covers
            step = interval_ms(rnd)
            start_ms, end_ms = max(start_ms, ceil_to(cov[0], step)), min(end_ms, floor_to(cov[1] + 1000, step))
            if end_ms <= start_ms:
                print(f"Khoảng --start/--end không trùng dữ liệu book ({ms_to_iso(cov[0])} .. {ms_to_iso(cov[1])}).",
                      file=sys.stderr)
                return 1
            book_samples = store.get_range(start_ms - 60_000, end_ms)
        finally:
            store.close()

    use_odds, fee_bps, max_price = p.use_odds, p.fee_bps, p.max_entry_price
    odds_samples = None
    if use_odds:
        ostore = OddsStore(cfg.predict.odds_path, sym)
        try:
            cov = ostore.coverage()
            if cov is None:
                print("Chưa có odds đã ghi. Chạy `python -m bot data record-odds` một lúc trước.", file=sys.stderr)
                return 1
            step = interval_ms(rnd)
            start_ms, end_ms = max(start_ms, ceil_to(cov[0], step)), min(end_ms, floor_to(cov[1] + 1000, step))
            if end_ms <= start_ms:
                print(f"Khoảng --start/--end không trùng odds đã ghi ({ms_to_iso(cov[0])} .. {ms_to_iso(cov[1])}).",
                      file=sys.stderr)
                return 1
            odds_samples = ostore.get_range(start_ms, end_ms)
        finally:
            ostore.close()

    cache = KlineCache(cfg.data.cache_path)
    try:
        async with _rest(cfg) as rest:
            rounds = await load_klines(rest, cache, sym, rnd, start_ms, end_ms)
            seconds = await load_klines(rest, cache, sym, "1s", start_ms, end_ms)
    finally:
        cache.close()

    res = run_backtest(rounds, seconds, strategy, cfg.settlement.rule, rnd, decision_s, payout, book_samples,
                       odds_samples, fee_bps, max_price)
    summary = {
        "symbol": sym, "range": [ms_to_iso(start_ms), ms_to_iso(end_ms)], "strategy": strategy.name,
        "decision_s": decision_s, "min_move_bps": min_move,
        **({"pricing": "predict.fun odds", "odds_samples": len(odds_samples), "max_entry_price": max_price}
           if odds_samples is not None else {"pricing": "fixed payout", "payout_ratio": payout}),
        **({"book": {"levels": book_params.levels, "window_s": book_params.window_s,
                     "min_imbalance": book_params.min_imbalance, "max_spread_bps": book_params.max_spread_bps,
                     "samples": len(book_samples)}} if strategy.needs_book else {}),
        "settlement_rule": cfg.settlement.rule.value, **summarize(res, payout, fee_bps),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    bt.trades_path.parent.mkdir(parents=True, exist_ok=True)
    with bt.trades_path.open("w", encoding="utf-8") as f:
        for t in res.trades:
            f.write(json.dumps({"round_open": ms_to_iso(t.round_open_time), "side": t.side.value,
                                "outcome": t.outcome.value, "pnl": round(t.pnl, 6), "move_bps": round(t.move_bps, 3),
                                "entry_price": t.entry_price}) + "\n")
    print(f"trades -> {bt.trades_path}", file=sys.stderr)
    return 0


# ---- dashboard / paper trading ------------------------------------------------------

def _run_dashboard(cfg: AppConfig, args: argparse.Namespace, secrets: object, paper: bool) -> int:
    from bot.ui.server import cmd_ui

    trader = None
    if paper:
        p = _strategy_params(cfg, args)
        if p.strategy.needs_book and not cfg.websocket.depth_levels:
            print("Chiến lược này cần order book: đặt websocket.depth_levels > 0.", file=sys.stderr)
            return 1
        trader = PaperTrader(p.strategy, cfg.settlement.rule, cfg.market.round_interval, p.decision_s, p.fee_bps,
                             p.max_entry_price, p.use_odds, p.payout)
        print(f"Paper trading: {p.name}, quyết định ở giây {p.decision_s}, "
              f"giá {'odds Predict.fun' if p.use_odds else f'payout cố định {p.payout}'}, phí {p.fee_bps} bps. "
              "Không đặt lệnh thật.", file=sys.stderr)
    key_secret = getattr(secrets, "predict_api_key", None)
    key = key_secret.get_secret_value() if key_secret else None
    try:
        return asyncio.run(cmd_ui(cfg, args.host, args.port, trader, key))
    except KeyboardInterrupt:
        return 0


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


async def cmd_data_record(cfg: AppConfig, duration_s: float | None) -> int:
    """Record 1s order-book samples and closed candles until stopped."""
    if not cfg.websocket.depth_levels:
        print("Cần websocket.depth_levels > 0 để ghi order book.", file=sys.stderr)
        return 1
    store = BookStore(cfg.data.book_path, cfg.market.symbol)
    cache = KlineCache(cfg.data.cache_path)
    recorder = BookRecorder(store)
    counts: Counter[str] = Counter()
    async with _rest(cfg) as rest:
        stream = BinanceStream(cfg.websocket, cfg.market.symbol)
        feed = LiveFeed(stream, rest, cfg.websocket.kline_intervals, cache=cache)

        async def consume() -> None:
            last_report = now_ms()
            async for ev in feed.events():
                recorder.handle(ev)
                if isinstance(ev, ConnectionEvent):
                    counts[f"conn_{ev.state}"] += 1
                elif isinstance(ev, GapEvent):
                    counts["gaps"] += 1
                if now_ms() - last_report >= 30_000:
                    last_report = now_ms()
                    print(f"recorded {recorder.recorded} book samples", file=sys.stderr, flush=True)

        try:
            await asyncio.wait_for(consume(), timeout=duration_s)
        except TimeoutError:
            pass
        finally:
            store.flush()
    cov = store.coverage()
    store.close()
    cache.close()
    print(json.dumps({"recorded_this_run": recorder.recorded, "counts": dict(counts),
                      "book_coverage": None if cov is None else
                      {"from": ms_to_iso(cov[0]), "to": ms_to_iso(cov[1]), "samples": cov[2]}}, indent=2))
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

def _add_strategy_args(p: argparse.ArgumentParser) -> None:
    """Strategy/pricing overrides of config.yaml `backtest:`; shared by `run` and `ui`."""
    p.add_argument("--strategy", choices=STRATEGIES, help="override backtest.strategy")
    p.add_argument("--decision-s", type=int, help="override backtest.decision_s")
    p.add_argument("--min-move-bps", type=float, help="override backtest.min_move_bps")
    p.add_argument("--payout", type=float, help="override backtest.payout_ratio")
    p.add_argument("--odds", action="store_true", help="price bets from Predict.fun odds instead of a fixed payout")
    p.add_argument("--fee-bps", type=float, help="override backtest.fee_bps (Binance fee, on the stake)")
    p.add_argument("--max-entry-price", type=float, help="override backtest.max_entry_price (0..1)")
    p.add_argument("--min-imbalance", type=float, help="override backtest.min_imbalance (0..1)")
    p.add_argument("--book-levels", type=int, choices=(1, 5, 10, 20), help="override backtest.book_levels")
    p.add_argument("--book-window-s", type=int, help="override backtest.book_window_s")
    p.add_argument("--max-spread-bps", type=float, help="override backtest.max_spread_bps")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m bot")
    p.add_argument("--config", default=None, help="YAML config file (default: built-in defaults)")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the bot in a mode")
    run.add_argument("--mode", choices=MODES, required=True)
    run.add_argument("--config", dest="run_config", default=None)
    run.add_argument("--i-understand-the-risk", action="store_true")
    run.add_argument("--start", help="backtest: UTC start, e.g. 2026-09-01")
    run.add_argument("--end", help="backtest: UTC end (exclusive)")
    _add_strategy_args(run)
    run.add_argument("--host", default="127.0.0.1", help="paper: dashboard host")
    run.add_argument("--port", type=int, default=8765, help="paper: dashboard port")

    ui = sub.add_parser("ui", help="local web dashboard to monitor the live price")
    ui.add_argument("--config", dest="run_config", default=None)
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--paper", action="store_true",
                    help="also paper-trade live (simulated bets, no real orders) and show them with the PnL")
    _add_strategy_args(ui)

    rep = sub.add_parser("report", help="print evaluation report")
    rep.add_argument("--config", dest="run_config", default=None)

    data = sub.add_parser("data", help="market data utilities")
    dsub = data.add_subparsers(dest="data_command", required=True)
    f = dsub.add_parser("fetch", help="download klines into the cache and check quality")
    f.add_argument("--start", required=True, help="UTC, e.g. 2026-09-01 or 2026-09-01T12:00")
    f.add_argument("--end", required=True)
    s = dsub.add_parser("stream", help="watch the live feed")
    s.add_argument("--duration", type=float, default=180.0, help="seconds")
    r = dsub.add_parser("record", help="record 1s order-book samples (+ candles) for order-book backtests")
    r.add_argument("--duration", type=float, default=None, help="seconds (default: until Ctrl+C)")
    ro = dsub.add_parser("record-odds", help="record Predict.fun odds of the current Up/Down round (needs no key on testnet)")
    ro.add_argument("--duration", type=float, default=None, help="seconds (default: until Ctrl+C)")
    dsub.add_parser("clock", help="compare local clock with Binance server time")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(getattr(args, "run_config", None) or args.config)
    setup_logging(cfg.logging.level, cfg.logging.file)
    secrets = load_secrets()

    if args.command == "data":
        match args.data_command:
            case "fetch":
                return asyncio.run(cmd_data_fetch(cfg, args.start, args.end))
            case "stream":
                return asyncio.run(cmd_data_stream(cfg, args.duration))
            case "record":
                try:
                    return asyncio.run(cmd_data_record(cfg, args.duration))
                except KeyboardInterrupt:
                    return 0
            case "record-odds":
                from bot.data.odds_recorder import record_odds

                key = secrets.predict_api_key.get_secret_value() if secrets.predict_api_key else None
                try:
                    print(json.dumps(asyncio.run(record_odds(cfg, key, args.duration)), indent=2))
                except KeyboardInterrupt:
                    pass
                return 0
            case "clock":
                return asyncio.run(cmd_data_clock(cfg))
    if args.command == "ui":
        return _run_dashboard(cfg, args, secrets, paper=args.paper)
    if args.command == "run" and args.mode == "paper":
        return _run_dashboard(cfg, args, secrets, paper=True)
    if args.command == "run" and args.mode == "backtest":
        return asyncio.run(cmd_backtest(cfg, args))
    if args.command == "run":
        print(f"mode '{args.mode}' chưa được triển khai (bước 2-5).", file=sys.stderr)
        return 1
    if args.command == "report":
        print("report chưa được triển khai (bước 6).", file=sys.stderr)
        return 1
    return 1
