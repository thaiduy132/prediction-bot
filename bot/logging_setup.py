"""Structured JSON logging with secret redaction.

Usage:
    log = logging.getLogger(__name__)
    log_event(log, "ws.connected", url=url, attempt=3)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from bot.timeutil import ms_to_iso

_SECRETS: set[str] = set()
_REDACTED = "***REDACTED***"


def register_secret(value: str) -> None:
    if len(value) >= 4:
        _SECRETS.add(value)


def redact(text: str) -> str:
    for s in _SECRETS:
        if s in text:
            text = text.replace(s, _REDACTED)
    return text


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": ms_to_iso(int(record.created * 1000)),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return redact(json.dumps(payload, default=str, ensure_ascii=False))


def setup_logging(level: str = "INFO", file: Path | None = None) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())
    fmt = JsonFormatter()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if file is not None:
        file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(file, encoding="utf-8"))
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)
    # third-party libraries are chatty at INFO/DEBUG
    for noisy in ("httpx", "httpcore", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, event, extra={"fields": fields})
