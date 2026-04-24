"""Weekend full-set Researcher refresh.

Enumerates documents in TRAINING_COMPLETED (past-HITL-training) state and
re-dispatches `run_researcher_agent` for each. Cadence: Celery beat weekly,
typically Sunday 03:00 UTC (configurable).

Isolation: runs on `researcher_queue` same as the per-doc Researcher task.
Never touches `pipeline_status`. Flag-gated via
`Config.Researcher.WEEKEND_REFRESH_ENABLED`.

Spec: project_researcher_agent_vision.md + 2026-04-24-unified-docwain-wave2b-plan.md
"""
from __future__ import annotations

import logging

from src.celery_app import app

logger = logging.getLogger(__name__)


def _get_documents_collection():
    """Return the MongoDB documents collection. Wrapped for test monkeypatching."""
    try:
        from src.api.dw_newron import get_mongo_collection
        return get_mongo_collection("documents")
    except Exception:
        try:
            from src.api.document_status import get_documents_collection
            return get_documents_collection()
        except Exception as exc:
            logger.error("Cannot obtain documents collection: %s", exc)
            raise


@app.task(bind=True, name="src.tasks.researcher_refresh.researcher_weekly_refresh",
           max_retries=0, soft_time_limit=3600)
def researcher_weekly_refresh(self):
    """Weekly full-set Researcher refresh across all TRAINING_COMPLETED docs."""
    try:
        from src.api.config import Config
        researcher_cfg = getattr(Config, "Researcher", None)
        enabled = getattr(researcher_cfg, "ENABLED", False) if researcher_cfg else False
    except Exception:
        enabled = False

    if not enabled:
        logger.info("Researcher disabled; skipping weekly refresh.")
        return {"skipped": True, "dispatched_count": 0, "failed_count": 0}

    try:
        from src.tasks.researcher import run_researcher_agent
    except Exception as exc:
        logger.error("Cannot import run_researcher_agent: %s", exc)
        return {"skipped": True, "dispatched_count": 0, "failed_count": 0,
                "error": f"import failed: {exc}"}

    col = _get_documents_collection()

    projection = {"document_id": 1, "subscription_id": 1, "profile_id": 1,
                  "pipeline_status": 1}
    filter_ = {"pipeline_status": {"$in": [
        "TRAINING_COMPLETED",
        "TRAINING_PARTIALLY_COMPLETED",
    ]}}

    dispatched_count = 0
    failed_count = 0

    try:
        cursor = col.find(filter_, projection)
    except TypeError:
        cursor = col.find(filter_)

    for doc in cursor:
        doc_id = doc.get("document_id") or doc.get("_id")
        sub_id = doc.get("subscription_id")
        profile_id = doc.get("profile_id")
        if not (doc_id and sub_id and profile_id):
            continue
        try:
            run_researcher_agent.delay(doc_id, sub_id, profile_id)
            dispatched_count += 1
        except Exception as exc:
            failed_count += 1
            logger.warning("Researcher refresh dispatch failed for %s: %s", doc_id, exc)

    logger.info("Weekly Researcher refresh: dispatched=%d failed=%d",
                dispatched_count, failed_count)
    return {"dispatched_count": dispatched_count, "failed_count": failed_count,
            "skipped": False}
