"""
📚 busel — Structured JSON Event Stream
Emits one JSON object per line to checkpoints/busel.log.jsonl for downstream
consumers (Telegram bot, web dashboard, analytics). Every training step,
checkpoint save, curriculum change, and error is recorded as a discrete event
with a stable schema.

Public API:
    setup_logging(log_dir="checkpoints", level=logging.INFO) -> logging.Logger
    log_event(logger, event: str, **fields) -> None
    JSONFormatter (logging.Formatter subclass — append-only, no I/O)

Event schema (all fields always present, may be null):
    {
      "ts":          "<ISO-8601 UTC timestamp>",
      "level":       "INFO" | "WARNING" | "ERROR" | "DEBUG",
      "logger":      "<logger name>",
      "event":       "<short snake_case event name>",
      "step":        <int or null>,
      "loss":        <float or null>,
      "lr":          <float or null>,
      "aux_loss":    <float or null>,
      "tokens_per_s":<float or null>,
      "vram_mb":     <float or null>,
      "extra":       {<arbitrary extra fields>}
    }
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOGGER_NAME = "busel"
_FILE_HANDLER_ATTR = "_busel_file_handler"
_LOCK = threading.Lock()


class JSONFormatter(logging.Formatter):
    """Formats LogRecord as a single-line JSON object (no embedded newlines).

    All structured fields are placed under a top-level "extra" object so that
    consumers can pick exactly the fields they care about. Step, loss, lr,
    aux_loss, tokens_per_s, vram_mb are hoisted to top level for convenience
    because that's what every step event will carry.
    """

    HOISTED = ("step", "loss", "lr", "aux_loss", "tokens_per_s", "vram_mb")

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        payload: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        extra: dict[str, Any] = {}
        for k, v in record.__dict__.items():
            if k in _STANDARD_LOGRECORD_ATTRS:
                continue
            extra[k] = _safe(v)
        for field in self.HOISTED:
            if field in extra:
                payload[field] = extra.pop(field)
        if extra:
            payload["extra"] = extra
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


_STANDARD_LOGRECORD_ATTRS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
})


def _safe(v: Any) -> Any:
    """Make a value JSON-serializable; fall back to its repr."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _safe(val) for k, val in v.items()}
    return repr(v)


def setup_logging(
    log_dir: str | os.PathLike = "checkpoints",
    level: int = logging.INFO,
    log_filename: str = "busel.log.jsonl",
) -> logging.Logger:
    """Configure the busel logger with a JSON file handler.

    Idempotent: if setup_logging has already been called, the existing
    handler is reused and the level is updated. Re-runs of train.py resume
    the same log file (append mode).

    Args:
        log_dir: Directory for the JSONL log. Created if missing.
        level: Minimum log level (defaults to INFO).
        log_filename: Name of the log file inside log_dir.

    Returns:
        The configured "busel" logger.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    with _LOCK:
        existing = getattr(logger, _FILE_HANDLER_ATTR, None)
        if existing is not None:
            existing.setLevel(level)
            logger.setLevel(level)
            return logger

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(Path(log_dir) / log_filename, mode="a", encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        setattr(logger, _FILE_HANDLER_ATTR, handler)
    return logger


def get_logger() -> logging.Logger:
    """Return the busel logger. Calls setup_logging() with defaults if needed."""
    logger = logging.getLogger(_LOGGER_NAME)
    if not getattr(logger, _FILE_HANDLER_ATTR, None):
        return setup_logging()
    return logger


def log_event(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Emit a structured event. Convenience wrapper over logger.log().

    Args:
        event: Short snake_case event name (e.g. "step_complete", "checkpoint_saved").
        level: Log level (defaults to INFO).
        **fields: Arbitrary key-value pairs. Standard step fields
            (step, loss, lr, aux_loss, tokens_per_s, vram_mb) are hoisted to
            the top level of the JSON object.
    """
    logger = get_logger()
    logger.log(level, event, extra=fields)
