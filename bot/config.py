"""Typed configuration loaded from YAML. Secrets never live here (see bot.secrets)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from bot.settlement import SettlementRule
from bot.timeutil import interval_ms


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MarketConfig(_Strict):
    symbol: str = "BTCUSDT"
    round_interval: str = "5m"

    @field_validator("symbol")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @field_validator("round_interval")
    @classmethod
    def _valid_interval(cls, v: str) -> str:
        interval_ms(v)
        return v


class SettlementConfig(_Strict):
    rule: SettlementRule = SettlementRule.CLOSE_GTE_OPEN
    # Informational: where the settlement price comes from. Only Binance spot klines
    # are implemented; if the real market uses another reference, backtests are biased.
    price_source: str = "binance_spot_kline"


# Binance's public market-data mirrors. Same data as api/stream.binance.com, but
# they don't apply the region block (HTTP 451) that the main hosts return in some
# locations (e.g. US cloud servers). Only public data is served there, which is
# all this bot reads; signed endpoints will need the main hosts.
MARKET_DATA_REST_URL = "https://data-api.binance.vision"
MARKET_DATA_WS_URL = "wss://data-stream.binance.vision:443"


class WebSocketConfig(_Strict):
    base_url: str = MARKET_DATA_WS_URL
    kline_intervals: list[str] = Field(default_factory=lambda: ["1s", "1m", "5m"])
    book_ticker: bool = True
    # Partial order book stream (top N levels). 0 disables. Needed for imbalance features.
    depth_levels: int = Field(20, description="0 | 5 | 10 | 20")
    depth_speed_ms: int = Field(100, description="100 | 1000")
    backoff_initial_s: float = Field(1.0, gt=0)
    backoff_max_s: float = Field(60.0, gt=0)
    backoff_factor: float = Field(2.0, ge=1)
    backoff_jitter: float = Field(0.2, ge=0, le=1)
    # connection counted as healthy (backoff reset) after this many seconds
    stable_after_s: float = Field(60.0, gt=0)
    # no message at all for this long -> assume dead socket and reconnect
    stale_timeout_s: float = Field(30.0, gt=0)
    # Binance drops connections after 24h; reconnect ourselves slightly before
    max_connection_age_s: float = Field(23.5 * 3600, gt=0)


    @field_validator("depth_levels")
    @classmethod
    def _levels(cls, v: int) -> int:
        if v not in (0, 5, 10, 20):
            raise ValueError("depth_levels must be 0, 5, 10 or 20")
        return v

    @field_validator("depth_speed_ms")
    @classmethod
    def _speed(cls, v: int) -> int:
        if v not in (100, 1000):
            raise ValueError("depth_speed_ms must be 100 or 1000")
        return v


class RestConfig(_Strict):
    base_url: str = MARKET_DATA_REST_URL
    timeout_s: float = Field(10.0, gt=0)
    max_retries: int = Field(5, ge=0)


class DataConfig(_Strict):
    cache_path: Path = Path("data/klines.sqlite")
    book_path: Path = Path("data/book.sqlite")  # order-book samples recorded live (1/second)


class BacktestConfig(_Strict):
    strategy: str = "momentum"  # momentum | reversal | always_up | book | momentum_book
    decision_s: int = Field(270, gt=0)  # seconds after round open when the bet is placed
    min_move_bps: float = Field(1.0, ge=0)  # momentum/reversal: skip if |move| is smaller
    # Net profit per unit stake on a win; a loss costs 1. MUST match the market's real
    # payout, otherwise PnL is meaningless. 0.9 is a placeholder, not a fact.
    payout_ratio: float = Field(0.9, gt=0)
    # Odds mode (`--odds`): bets are priced from recorded Predict.fun quotes instead of payout_ratio.
    # Fees are Binance's, not Predict.fun's; charged on the stake of every executed bet. Fill in
    # the fee that really applies to you, 0 means "no fee" and flatters the result.
    use_odds: bool = False
    fee_bps: float = Field(0.0, ge=0)
    max_entry_price: float = Field(1.0, gt=0, le=1)  # skip bets that cost more than this per share
    # order-book strategies (need recorded data, see `python -m bot data record`)
    book_levels: int = Field(10, description="1 | 5 | 10 | 20 levels used for imbalance")
    book_window_s: int = Field(10, gt=0)  # average imbalance over the last N seconds
    min_imbalance: float = Field(0.2, ge=0, le=1)  # |bid-ask depth imbalance| needed to act
    max_spread_bps: float = Field(2.0, ge=0)  # skip when the spread is wider than this
    trades_path: Path = Path("data/backtest_trades.jsonl")


class PredictConfig(_Strict):
    # Predict.fun is the venue behind Binance Wallet prediction markets. The testnet needs no
    # API key but has NO liquidity on BTC rounds (verified 2026-09-29); real odds need mainnet
    # (https://api.predict.fun) and PREDICT_API_KEY in .env.
    base_url: str = "https://api-testnet.predict.fun"
    poll_interval_s: float = Field(1.0, ge=0.2)  # mainnet default limit is 240 req/min
    timeout_s: float = Field(10.0, gt=0)
    odds_path: Path = Path("data/odds.sqlite")


class PaperConfig(_Strict):
    trades_path: Path = Path("data/paper.sqlite")  # simulated bets, kept across restarts


class LoggingConfig(_Strict):
    level: str = "INFO"
    file: Path | None = None  # JSON lines; stdout always gets JSON too


class AppConfig(_Strict):
    market: MarketConfig = MarketConfig()
    settlement: SettlementConfig = SettlementConfig()
    websocket: WebSocketConfig = WebSocketConfig()
    rest: RestConfig = RestConfig()
    data: DataConfig = DataConfig()
    backtest: BacktestConfig = BacktestConfig()
    predict: PredictConfig = PredictConfig()
    paper: PaperConfig = PaperConfig()
    logging: LoggingConfig = LoggingConfig()


def load_config(path: str | Path | None) -> AppConfig:
    if path is None:
        return AppConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return AppConfig.model_validate(raw)
