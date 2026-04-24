"""Verify log-hygiene settings apply as expected."""
import logging

from src.docwain.logging_config import (
    apply_log_hygiene,
    HealthCheckPathFilter,
)


def test_apply_log_hygiene_silences_azure_sdk_to_warn():
    azure_logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
    azure_logger.setLevel(logging.DEBUG)
    apply_log_hygiene()
    assert azure_logger.level == logging.WARNING


def test_apply_log_hygiene_silences_azure_storage_blob():
    blob_logger = logging.getLogger("azure.storage.blob")
    blob_logger.setLevel(logging.DEBUG)
    apply_log_hygiene()
    assert blob_logger.level >= logging.WARNING


def test_healthcheck_filter_demotes_progress_polls():
    filter_ = HealthCheckPathFilter()
    rec = logging.LogRecord(
        name="src.middleware.correlation", level=logging.INFO,
        pathname="x", lineno=1,
        msg="Request completed: GET /api/extract/progress -> 200 (1300.0ms)",
        args=(), exc_info=None,
    )
    assert filter_.filter(rec) is False


def test_healthcheck_filter_allows_normal_paths():
    filter_ = HealthCheckPathFilter()
    rec = logging.LogRecord(
        name="src.middleware.correlation", level=logging.INFO,
        pathname="x", lineno=1,
        msg="Request completed: POST /api/ask -> 200 (520.0ms)",
        args=(), exc_info=None,
    )
    assert filter_.filter(rec) is True


def test_healthcheck_filter_allows_errors_on_progress_paths():
    filter_ = HealthCheckPathFilter()
    rec = logging.LogRecord(
        name="src.middleware.correlation", level=logging.ERROR,
        pathname="x", lineno=1,
        msg="Request completed: GET /api/extract/progress -> 500",
        args=(), exc_info=None,
    )
    assert filter_.filter(rec) is True
