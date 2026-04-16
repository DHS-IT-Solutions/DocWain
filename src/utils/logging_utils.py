"""
Enhanced logging utilities with correlation ID support and JSON formatting.

This module provides:
- CorrelationLoggerAdapter: Logger adapter that auto-injects correlation ID
- JSONFormatter: Structured JSON log formatter for production
- configure_logging(): Application-wide logging configuration

Usage:
    from src.utils.logging_utils import get_logger, configure_logging

    # At application startup
    configure_logging(json_format=True, log_level="INFO")

    # In any module
    logger = get_logger(__name__)
    logger.info("Processing request", extra={"user_id": "123"})
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time as _time
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple


def get_correlation_id_from_context() -> str:
    """
    Get correlation ID from the middleware context.

    Returns:
        Correlation ID or "n/a" if not in request context.
    """
    try:
        from src.middleware.correlation import get_correlation_id
        return get_correlation_id()
    except ImportError:
        return "n/a"


def get_correlation_id(
    activity: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> str:
    """
    Resolve a correlation id from headers/activity or generate a new one.

    This is a legacy function maintained for backward compatibility.
    For new code, use the middleware context via get_correlation_id_from_context().

    Args:
        activity: Optional Teams activity dict.
        headers: Optional request headers dict.

    Returns:
        A correlation ID string.
    """
    header_id = None
    if headers:
        header_id = (
            headers.get("x-correlation-id")
            or headers.get("x-request-id")
            or headers.get("X-Correlation-Id")
            or headers.get("X-Request-Id")
        )
    activity_id = None
    if activity:
        activity_id = activity.get("id") or activity.get("activityId")
    return str(header_id or activity_id or uuid.uuid4())


class CorrelationLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that automatically injects correlation ID into log records.

    This adapter retrieves the correlation ID from the middleware context
    and adds it to every log message.

    Usage:
        logger = CorrelationLoggerAdapter(logging.getLogger(__name__))
        logger.info("Processing request")  # correlation_id auto-added
    """

    def __init__(
        self,
        logger: logging.Logger,
        extra: Optional[Mapping[str, Any]] = None,
    ):
        """
        Initialize the adapter.

        Args:
            logger: The underlying logger.
            extra: Optional extra fields to include in all log records.
        """
        super().__init__(logger, extra or {})

    def process(
        self,
        msg: str,
        kwargs: MutableMapping[str, Any],
    ) -> Tuple[str, MutableMapping[str, Any]]:
        """
        Process log message to add correlation ID.

        Args:
            msg: The log message.
            kwargs: Keyword arguments passed to the logging call.

        Returns:
            Tuple of (message, kwargs) with correlation_id added.
        """
        extra = dict(self.extra) if self.extra else {}
        extra["correlation_id"] = get_correlation_id_from_context()

        # Merge with any extra provided in the log call
        if "extra" in kwargs:
            extra.update(kwargs["extra"])
        kwargs["extra"] = extra

        return msg, kwargs


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.

    Outputs logs as single-line JSON objects for easy parsing by log aggregators.

    Example output:
        {"timestamp": "2024-01-15T10:30:00Z", "level": "INFO", "message": "Request processed", ...}
    """

    def __init__(
        self,
        include_extra: bool = True,
        timestamp_format: str = "iso",
    ):
        """
        Initialize the JSON formatter.

        Args:
            include_extra: Whether to include extra fields in output.
            timestamp_format: "iso" for ISO 8601, "epoch" for Unix timestamp.
        """
        super().__init__()
        self.include_extra = include_extra
        self.timestamp_format = timestamp_format

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON string representation of the log record.
        """
        log_data: Dict[str, Any] = {
            "timestamp": self._format_timestamp(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add correlation ID if present
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id

        # Add location info
        log_data["location"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if self.include_extra:
            extra_fields = self._extract_extra_fields(record)
            if extra_fields:
                log_data["extra"] = extra_fields

        return json.dumps(log_data, default=str, ensure_ascii=False)

    def _format_timestamp(self, record: logging.LogRecord) -> str | float:
        """Format the timestamp based on configuration."""
        if self.timestamp_format == "epoch":
            return record.created
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return dt.isoformat()

    def _extract_extra_fields(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Extract extra fields from the log record."""
        standard_attrs = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "exc_info", "exc_text", "thread", "threadName",
            "message", "correlation_id", "taskName",
        }
        return {
            key: value
            for key, value in record.__dict__.items()
            if key not in standard_attrs and not key.startswith("_")
        }


class ConsoleFormatter(logging.Formatter):
    """
    Production console formatter — compact, scannable, systemd-friendly.

    journalctl already supplies timestamp + hostname + unit, so we omit those.
    Format:  LEVEL  short_logger [corr_id] message  {extras}
    """

    # Shorten common logger prefixes for readability
    _PREFIX_MAP = {
        "src.api.": "",
        "src.llm.": "llm.",
        "src.intelligence.": "intel.",
        "src.serving.": "serve.",
        "src.doc_understanding.": "docai.",
        "src.storage.": "store.",
        "src.embedding.": "embed.",
        "src.kg.": "kg.",
        "src.middleware.": "mw.",
    }

    _LEVEL_SYMBOLS = {
        "DEBUG": "DBG",
        "INFO": "INF",
        "WARNING": "WRN",
        "ERROR": "ERR",
        "CRITICAL": "CRT",
    }

    def __init__(self, include_correlation_id: bool = True):
        self.include_correlation_id = include_correlation_id
        super().__init__()

    def _short_name(self, name: str) -> str:
        for prefix, replacement in self._PREFIX_MAP.items():
            if name.startswith(prefix):
                return replacement + name[len(prefix):]
        return name

    def format(self, record: logging.LogRecord) -> str:
        lvl = self._LEVEL_SYMBOLS.get(record.levelname, record.levelname[:3])
        logger_name = self._short_name(record.name)
        message = record.getMessage()

        # Build correlation snippet
        corr = ""
        if self.include_correlation_id:
            cid = getattr(record, "correlation_id", None) or ""
            if cid and cid not in ("unknown", "n/a"):
                corr = f" [{cid[:8]}]"

        # Build extras snippet (only for meaningful extra fields)
        extras = ""
        extra_fields = getattr(record, "_extra_fields", None)
        if extra_fields:
            pairs = " ".join(f"{k}={v}" for k, v in extra_fields.items())
            extras = f"  {{{pairs}}}"

        line = f"{lvl}  {logger_name}{corr}  {message}{extras}"

        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            return f"{line}\n{exc_text}"

        return line


# ---------------------------------------------------------------------------
# Pipeline log context: tracks which profile_id is active on this thread
# so RedisLogHandler can route logs to the correct Redis list.
# ---------------------------------------------------------------------------
_pipeline_ctx = threading.local()

_LIVE_LOG_TTL = 86400  # 24 hours
_LIVE_LOG_MAX_ENTRIES = 5000


def set_pipeline_profile(profile_id: Optional[str]) -> None:
    """Set the active pipeline profile for the current thread."""
    _pipeline_ctx.profile_id = profile_id


def get_pipeline_profile() -> Optional[str]:
    """Get the active pipeline profile for the current thread."""
    return getattr(_pipeline_ctx, "profile_id", None)


def clear_pipeline_profile() -> None:
    """Clear the pipeline profile for the current thread."""
    _pipeline_ctx.profile_id = None


class RedisLogHandler(logging.Handler):
    """Logging handler that pushes log records to a Redis list per profile.

    Only emits when a pipeline profile is active on the current thread
    (set via ``set_pipeline_profile``).  Each entry is a JSON dict matching
    the structure expected by ``Overall_live_logs``.
    """

    def __init__(self, level: int = logging.DEBUG):
        super().__init__(level)
        self._redis_client = None
        self._init_attempted = False

    def _get_redis(self):
        if self._init_attempted:
            return self._redis_client
        self._init_attempted = True
        try:
            from src.api.dw_newron import get_redis_client
            self._redis_client = get_redis_client()
        except Exception:
            self._redis_client = None
        return self._redis_client

    def emit(self, record: logging.LogRecord) -> None:
        profile_id = get_pipeline_profile()
        if not profile_id:
            return
        try:
            client = self._get_redis()
            if not client:
                return
            key = f"dw:live_logs:{profile_id}"
            entry = {
                "timestamp": record.created,
                "ts_iso": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "function": record.funcName,
                "line": record.lineno,
            }
            correlation_id = getattr(record, "correlation_id", None)
            if correlation_id:
                entry["correlation_id"] = correlation_id
            client.rpush(key, json.dumps(entry, default=str))
            client.expire(key, _LIVE_LOG_TTL)
            # Trim to prevent unbounded growth
            client.ltrim(key, -_LIVE_LOG_MAX_ENTRIES, -1)
        except Exception:
            pass  # Best-effort, never block the pipeline


def get_live_logs(profile_id: str) -> list:
    """Retrieve all live log entries from Redis for a profile."""
    try:
        from src.api.dw_newron import get_redis_client
        client = get_redis_client()
        if not client:
            return []
        raw_entries = client.lrange(f"dw:live_logs:{profile_id}", 0, -1)
        logs = []
        for raw in raw_entries:
            try:
                logs.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                pass
        return logs
    except Exception:
        return []


def clear_live_logs(profile_id: str) -> None:
    """Clear live logs for a profile (called at start of a new pipeline run)."""
    try:
        from src.api.dw_newron import get_redis_client
        client = get_redis_client()
        if client:
            client.delete(f"dw:live_logs:{profile_id}")
    except Exception:
        pass


def configure_logging(
    log_level: str = "INFO",
    json_format: bool = False,
    include_correlation_id: bool = True,
    log_dir: Optional[str] = None,
    max_bytes: int = 50 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """
    Configure application-wide logging.

    Args:
        log_level: Root log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: Use JSON formatter for structured logging.
        include_correlation_id: Include correlation ID in log output.
        log_dir: Directory for log files. Defaults to LOG_DIR env var or "logs/".
        max_bytes: Max size per log file before rotation (default 50MB).
        backup_count: Number of rotated log files to keep (default 5).
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console: human-readable (journalctl adds timestamp/host/unit already)
    # File: JSON for machine parsing / log aggregation
    console_formatter = ConsoleFormatter(include_correlation_id=include_correlation_id)
    json_formatter = JSONFormatter(include_extra=True)

    # Resolve correlation ID filter (shared across handlers)
    correlation_filter = None
    try:
        from src.middleware.correlation import CorrelationIdFilter
        correlation_filter = CorrelationIdFilter()
    except ImportError:
        pass

    # --- Console handler (human-readable for systemd/journalctl) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    if correlation_filter:
        console_handler.addFilter(correlation_filter)
    root_logger.addHandler(console_handler)

    # --- Rotating file handler (JSON for log aggregation) ---
    log_dir = log_dir or os.getenv("LOG_DIR", "logs")
    try:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path / "docwain.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(json_formatter)
        if correlation_filter:
            file_handler.addFilter(correlation_filter)
        root_logger.addHandler(file_handler)
    except OSError as exc:
        root_logger.warning("Could not create file log handler at %s: %s", log_dir, exc)

    # --- Redis log handler (captures terminal output for live progress) ---
    redis_handler = RedisLogHandler(level=level)
    if correlation_filter:
        redis_handler.addFilter(correlation_filter)
    root_logger.addHandler(redis_handler)

    # Route Python warnings through logging so they get formatted properly
    logging.captureWarnings(True)

    # Suppress noisy third-party loggers
    _noisy_loggers = {
        "httpx": logging.WARNING,
        "httpcore": logging.WARNING,
        "urllib3": logging.WARNING,
        "uvicorn.access": logging.WARNING,
        "uvicorn.error": logging.INFO,
        "neo4j.notifications": logging.ERROR,
        "neo4j.pool": logging.ERROR,
        "azure.core.pipeline.policies.http_logging_policy": logging.WARNING,
        "pymongo.client": logging.WARNING,
        "pymongo.connection": logging.WARNING,
        "sentence_transformers": logging.WARNING,
        "py.warnings": logging.ERROR,          # suppress CosmosDB/Qdrant UserWarnings
        "qdrant_client": logging.WARNING,
        "transformers": logging.WARNING,
        "torch": logging.WARNING,
    }
    for logger_name, lvl_override in _noisy_loggers.items():
        logging.getLogger(logger_name).setLevel(lvl_override)


def get_logger(
    name: str,
    correlation_id: Optional[str] = None,
) -> logging.LoggerAdapter:
    """
    Return a logger adapter that injects correlation_id into log records.

    If correlation_id is not provided, it will be automatically retrieved
    from the request context.

    Args:
        name: Logger name (usually __name__).
        correlation_id: Optional explicit correlation ID.

    Returns:
        A LoggerAdapter with correlation ID support.
    """
    if correlation_id:
        # Use simple adapter with explicit correlation ID
        extra = {"correlation_id": correlation_id}
        return logging.LoggerAdapter(logging.getLogger(name), extra)

    # Use auto-injecting adapter
    return CorrelationLoggerAdapter(logging.getLogger(name))


__all__ = [
    "get_correlation_id",
    "get_logger",
    "configure_logging",
    "CorrelationLoggerAdapter",
    "JSONFormatter",
    "ConsoleFormatter",
    "RedisLogHandler",
    "set_pipeline_profile",
    "get_pipeline_profile",
    "clear_pipeline_profile",
    "get_live_logs",
    "clear_live_logs",
]
