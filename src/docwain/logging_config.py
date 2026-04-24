"""Centralized log-hygiene for the DocWain application and Celery workers.

Call ``apply_log_hygiene()`` once at process startup (before any other imports
that might emit logs) to:

1. Demote Azure SDK and urllib3 loggers to WARNING — they produce 44 KB of
   DEBUG-level HTTP chatter per Celery log tail when left at default.
2. Attach a ``HealthCheckPathFilter`` to the correlation-middleware logger so
   that high-frequency poll endpoints (/api/extract/progress,
   /api/train/progress) are silently dropped at INFO level while WARN+ events
   on those paths still pass through.
"""

from __future__ import annotations

import logging

__all__ = ["apply_log_hygiene", "HealthCheckPathFilter"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEALTHCHECK_PATHS: tuple[str, ...] = (
    "/api/extract/progress",
    "/api/train/progress",
)

AZURE_SDK_LOGGERS: tuple[str, ...] = (
    "azure",
    "azure.core",
    "azure.core.pipeline.policies.http_logging_policy",
    "azure.storage",
    "azure.storage.blob",
    "azure.identity",
    "urllib3",
)

# Idempotency guard — apply_log_hygiene() is a no-op after the first call.
_APPLIED: bool = False


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


class HealthCheckPathFilter(logging.Filter):
    """Drop INFO-level log records whose message contains a health-check path.

    Records at WARNING or above always pass through so that errors on
    /api/extract/progress are never silently swallowed.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        if record.levelno >= logging.WARNING:
            return True
        msg = record.getMessage()
        for path in HEALTHCHECK_PATHS:
            if path in msg:
                return False
        return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_log_hygiene() -> None:
    """Apply all log-hygiene settings for this process.

    Idempotent: safe to call multiple times.  Logger-level assignments are
    inherently idempotent (setting WARNING twice is a no-op).  The filter
    attachment uses a module-level flag so the HealthCheckPathFilter is added
    to the correlation logger exactly once per process.
    """
    global _APPLIED  # noqa: PLW0603

    # 1. Silence Azure SDK and urllib3 DEBUG chatter — always safe to re-apply.
    for name in AZURE_SDK_LOGGERS:
        lgr = logging.getLogger(name)
        if lgr.level < logging.WARNING:
            lgr.setLevel(logging.WARNING)

    # 2. Filter health-check poll paths from the correlation-middleware logger.
    #    Guard against duplicate filter additions across repeated calls.
    if not _APPLIED:
        correlation_logger = logging.getLogger("src.middleware.correlation")
        existing_filter_types = {type(f) for f in correlation_logger.filters}
        if HealthCheckPathFilter not in existing_filter_types:
            correlation_logger.addFilter(HealthCheckPathFilter())
        _APPLIED = True
