"""One-liner stage tracking decorator for Celery pipeline tasks.

Every handler has to: (a) write ``update_stage(IN_PROGRESS)`` at entry,
(b) ``append_audit_log(..._STARTED)``, (c) update ``pipeline_status``,
(d) do the work, (e) at exit write ``update_stage(COMPLETED, summary=...)``,
``append_audit_log(..._COMPLETED)``, advance pipeline_status, and on
exceptions emit the ``FAILED`` triplet. Every handler forgetting even
one of those lines is where our observability gaps came from.

This decorator centralizes all six writes. Apply it once per Celery
task and the stage metadata is guaranteed.

Usage:
    @app.task(bind=True, name="src.tasks.extraction.extract_document")
    @tracked_stage(
        stage="extraction",
        audit_event="EXTRACTION",
        pipeline_in_progress=PIPELINE_EXTRACTION_IN_PROGRESS,
        pipeline_completed=PIPELINE_EXTRACTION_COMPLETED,
        pipeline_failed=PIPELINE_EXTRACTION_FAILED,
    )
    def extract_document(self, document_id, subscription_id, profile_id):
        ... # return a dict — its keys land in stage.summary
        return {"entity_count": 42, "blob_path": "..."}

The wrapped function should return a ``dict`` (or ``None``) — whatever
it returns is stored as the stage summary alongside ``duration_seconds``.
Exceptions propagate to Celery so retries / soft-time-limits still work.
"""
from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Dict, Optional

from celery.exceptions import SoftTimeLimitExceeded

from src.api.document_status import (
    append_audit_log,
    update_pipeline_status,
    update_stage,
)
from src.api.statuses import (
    STAGE_COMPLETED,
    STAGE_FAILED,
    STAGE_IN_PROGRESS,
)

logger = logging.getLogger(__name__)


def tracked_stage(
    *,
    stage: str,
    audit_event: str,
    pipeline_in_progress: Optional[str] = None,
    pipeline_completed: Optional[str] = None,
    pipeline_failed: Optional[str] = None,
) -> Callable:
    """Wrap a Celery task body so stage + audit + pipeline writes happen automatically.

    ``stage`` is the key under the Mongo document (e.g. ``"extraction"``,
    ``"screening"``, ``"embedding"``, ``"knowledge_graph"``). ``audit_event``
    is the uppercase prefix — ``EXTRACTION`` becomes ``EXTRACTION_STARTED``
    / ``EXTRACTION_COMPLETED`` / ``EXTRACTION_FAILED`` automatically.
    ``pipeline_*`` values, when set, are written to the top-level
    ``pipeline_status`` field at each transition; pass ``None`` to skip.
    """

    def _decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def _wrapper(self, document_id: str, *args, **kwargs):
            start_time = time.time()
            task_id = getattr(getattr(self, "request", None), "id", None)

            update_stage(
                document_id, stage, status=STAGE_IN_PROGRESS,
                celery_task_id=task_id,
            )
            if pipeline_in_progress:
                update_pipeline_status(document_id, pipeline_in_progress)
            append_audit_log(
                document_id, f"{audit_event}_STARTED",
                celery_task_id=task_id,
            )

            try:
                result = fn(self, document_id, *args, **kwargs)
            except SoftTimeLimitExceeded:
                duration_seconds = round(time.time() - start_time, 2)
                error = {"message": f"{stage} timed out", "code": "TIMEOUT"}
                update_stage(document_id, stage, status=STAGE_FAILED, error=error)
                if pipeline_failed:
                    update_pipeline_status(document_id, pipeline_failed)
                append_audit_log(
                    document_id, f"{audit_event}_FAILED",
                    error="timeout", duration_seconds=duration_seconds,
                )
                raise
            except Exception as exc:  # noqa: BLE001
                duration_seconds = round(time.time() - start_time, 2)
                error = {"message": str(exc), "code": f"{stage.upper()}_ERROR"}
                update_stage(document_id, stage, status=STAGE_FAILED, error=error)
                if pipeline_failed:
                    update_pipeline_status(document_id, pipeline_failed)
                append_audit_log(
                    document_id, f"{audit_event}_FAILED",
                    error=str(exc), duration_seconds=duration_seconds,
                )
                # Celery retry semantics require bubbling the exception
                raise

            duration_seconds = round(time.time() - start_time, 2)
            summary: Dict[str, Any] = dict(result) if isinstance(result, dict) else {}
            summary["duration_seconds"] = duration_seconds

            # Common ergonomic: if the task returned a blob_path, lift it
            # to the top-level stage.blob_path key too so readers don't
            # have to dig into summary.
            blob_path = summary.get("blob_path")
            update_stage(
                document_id, stage, status=STAGE_COMPLETED,
                summary=summary, blob_path=blob_path, error=None,
            )
            if pipeline_completed:
                update_pipeline_status(document_id, pipeline_completed)
            append_audit_log(
                document_id, f"{audit_event}_COMPLETED",
                duration_seconds=duration_seconds,
                **{k: v for k, v in summary.items()
                   if k in ("entity_count", "table_count", "risk_level",
                            "plugins_run", "flags_count", "chunk_count",
                            "node_count", "edge_count", "blob_path")},
            )
            return result

        return _wrapper

    return _decorator


__all__ = ["tracked_stage"]
