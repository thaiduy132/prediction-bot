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


class WebSocketConfig(_Strict):
    base_url: str = "wss://stream.binance.com:9443"
    kline_intervals: list[str] = Field(default_factory=lambda: ["1m", "5m"])
    book_ticker: bool = True
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


class RestConfig(_Strict):
    base_url: str = "https://api.binance.com"
    timeout_s: float = Field(10.0, gt=0)
    max_retries: int = Field(5, ge=0)


class DataConfig(_Strict):
    cache_path: Path = Path("data/klines.sqlite")


class LoggingConfig(_Strict):
    level: str = "INFO"
    file: Path | None = None  # JSON lines; stdout always gets JSON too


class AppConfig(_Strict):
    market: MarketConfig = MarketConfig()
    settlement: SettlementConfig = SettlementConfig()
    websocket: WebSocketConfig = WebSocketConfig()
    rest: RestConfig = RestConfig()
    data: DataConfig = DataConfig()
    logging: LoggingConfig = LoggingConfig()


def load_config(path: str | Path | None) -> AppConfig:
    if path is None:
        return AppConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return AppConfig.model_validate(raw)
