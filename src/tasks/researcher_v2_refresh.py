"""Continuous-refresh tasks — on-upload, scheduled, watchlist.

Runs on `researcher_refresh_queue` (low priority, isolated). Only writes
to insights collections + researcher_v2.* fields. Per spec Section 9.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.api.config import insight_flag_enabled
from src.intelligence.insights.staleness import mark_stale_for_documents
from src.tasks.researcher_v2 import (
    run_researcher_v2_for_doc,
    run_researcher_v2_for_profile,
)

logger = logging.getLogger(__name__)


def resolve_default_index_collection():
    """Hook — production wiring binds the Mongo `insights_index` collection."""
    raise NotImplementedError


def fetch_active_profile_documents(*, profile_id: str) -> List[Dict[str, Any]]:
    """Hook — production wiring fetches all documents for the profile."""
    raise NotImplementedError


def refresh_for_new_doc(
    *,
    document_id: str,
    profile_id: str,
    subscription_id: str,
    document_text: str,
    domain_hint: str = "generic",
) -> Dict[str, Any]:
    if not insight_flag_enabled("REFRESH_ON_UPLOAD_ENABLED"):
        return {"status": "skipped_flag_off"}
    if insight_flag_enabled("REFRESH_INCREMENTAL_ENABLED"):
        coll = resolve_default_index_collection()
        mark_stale_for_documents(
            collection=coll, profile_id=profile_id, document_ids=[document_id]
        )
    run_researcher_v2_for_doc(
        document_id=document_id,
        profile_id=profile_id,
        subscription_id=subscription_id,
        document_text=document_text,
        domain_hint=domain_hint,
    )
    return {"status": "ok"}


def refresh_scheduled_pass(
    *, profile_id: str, subscription_id: str, domain_hint: str = "generic"
) -> Dict[str, Any]:
    if not insight_flag_enabled("REFRESH_SCHEDULED_ENABLED"):
        return {"status": "skipped_flag_off"}
    docs = fetch_active_profile_documents(profile_id=profile_id)
    run_researcher_v2_for_profile(
        profile_id=profile_id,
        subscription_id=subscription_id,
        documents=docs,
        domain_hint=domain_hint,
    )
    return {"status": "ok"}


# Celery task bindings — registered if Celery app is available
try:
    from src.celery_app import app as _celery_app

    @_celery_app.task(name="src.tasks.researcher_v2_refresh.refresh_for_new_doc_task", bind=True)
    def refresh_for_new_doc_task(self, *, document_id, profile_id, subscription_id, document_text, domain_hint="generic"):
        return refresh_for_new_doc(
            document_id=document_id,
            profile_id=profile_id,
            subscription_id=subscription_id,
            document_text=document_text,
            domain_hint=domain_hint,
        )

    @_celery_app.task(name="src.tasks.researcher_v2_refresh.refresh_scheduled_pass_task", bind=True)
    def refresh_scheduled_pass_task(self, *, profile_id, subscription_id, domain_hint="generic"):
        return refresh_scheduled_pass(
            profile_id=profile_id,
            subscription_id=subscription_id,
            domain_hint=domain_hint,
        )
except Exception as _exc:  # pragma: no cover
    logger.debug("Celery binding for researcher_v2_refresh deferred: %s", _exc)
