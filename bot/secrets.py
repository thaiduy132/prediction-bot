"""Secrets come only from the environment / .env file, never from config.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr

from bot.logging_setup import register_secret


@dataclass(frozen=True)
class Secrets:
    jev_api_key: SecretStr | None
    binance_api_key: SecretStr | None
    binance_api_secret: SecretStr | None


def _get(name: str) -> SecretStr | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    register_secret(value)  # redact from every log line from now on
    return SecretStr(value)


def load_secrets(env_file: str | Path = ".env") -> Secrets:
    # Real environment variables win over .env.
    load_dotenv(env_file, override=False)
    return Secrets(
        jev_api_key=_get("JEV_API_KEY"),
        binance_api_key=_get("BINANCE_API_KEY"),
        binance_api_secret=_get("BINANCE_API_SECRET"),
    )
