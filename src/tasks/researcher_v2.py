"""Researcher Agent v2 Celery task entry points.

Per-doc and profile-level. Runs on `researcher_v2_queue`. Isolated from
`extraction_queue`, `embedding_queue`, `kg_queue`. Writes ONLY to
researcher_v2.* fields and the insights collections (Mongo + Qdrant +
Neo4j) — never touches pipeline_status (per feedback_mongo_status_stability.md).

Each insight type has its own flag (INSIGHTS_TYPE_*_ENABLED). The task
runs only enabled types, and short-circuits entirely if no type is
enabled (avoids wasted LLM calls when feature is fully off).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.api.config import insight_flag_enabled
from src.intelligence.researcher_v2.runner import (
    run_per_doc_insight_pass, DocPassInputs,
)
from src.intelligence.researcher_v2.profile_passes import (
    run_profile_pass, ProfilePassInputs,
)
from src.intelligence.insights.store import InsightStore
from src.intelligence.adapters.schema import Adapter

logger = logging.getLogger(__name__)


_TYPE_FLAGS = {
    "anomaly": "INSIGHTS_TYPE_ANOMALY_ENABLED",
    "gap": "INSIGHTS_TYPE_GAP_ENABLED",
    "comparison": "INSIGHTS_TYPE_COMPARISON_ENABLED",
    "scenario": "INSIGHTS_TYPE_SCENARIO_ENABLED",
    "trend": "INSIGHTS_TYPE_TREND_ENABLED",
    "recommendation": "INSIGHTS_TYPE_RECOMMENDATION_ENABLED",
    "conflict": "INSIGHTS_TYPE_CONFLICT_ENABLED",
    "projection": "INSIGHTS_TYPE_PROJECTION_ENABLED",
    "next_action": "INSIGHTS_TYPE_RECOMMENDATION_ENABLED",
}

_PROFILE_TYPES = ("comparison", "conflict", "trend", "projection")


def resolve_default_store() -> InsightStore:
    """Hook for tests + production wiring. Real implementation injects
    Mongo + Qdrant + Neo4j clients."""
    raise NotImplementedError("wire me from src.api startup")


def resolve_default_adapter(*, domain: str, subscription_id: str) -> Adapter:
    """Hook for tests + production wiring. Real implementation uses
    AdapterStore from src.intelligence.adapters.store."""
    raise NotImplementedError("wire me from src.api startup")


def resolve_default_llm():
    raise NotImplementedError("wire me from src.api startup")


def _enabled_types(types: List[str]) -> List[str]:
    return [t for t in types if insight_flag_enabled(_TYPE_FLAGS.get(t, ""))]


def _resolve_domain(*, document_text: str, hint: str) -> str:
    """Choose the adapter domain.

    If hint is anything other than "generic" or empty, honour it. Otherwise
    auto-detect via the existing classifier (gated by ADAPTER_AUTO_DETECT_ENABLED).
    Falls back to "generic" on any classifier failure or low confidence.
    """
    if hint and hint != "generic":
        return hint
    if not insight_flag_enabled("ADAPTER_AUTO_DETECT_ENABLED"):
        return "generic"
    try:
        from src.intelligence.adapters.detect import detect_domain
        result = detect_domain(document_text)
        return result.domain  # already falls back to "generic" on low confidence
    except Exception as exc:
        logger.debug("auto-detect skipped: %s", exc)
        return "generic"


def run_researcher_v2_for_doc(
    *,
    document_id: str,
    profile_id: str,
    subscription_id: str,
    document_text: str,
    domain_hint: str = "generic",
) -> Dict[str, Any]:
    enabled = _enabled_types(list(_TYPE_FLAGS.keys()))
    if not enabled:
        return {"status": "skipped_flag_off"}
    domain = _resolve_domain(document_text=document_text, hint=domain_hint)
    adapter = resolve_default_adapter(domain=domain, subscription_id=subscription_id)
    store = resolve_default_store()
    llm_call = resolve_default_llm()
    written = 0
    for itype in enabled:
        if itype not in adapter.researcher.insight_types:
            continue
        if itype in _PROFILE_TYPES:
            continue
        result = run_per_doc_insight_pass(DocPassInputs(
            adapter=adapter,
            insight_type=itype,
            document_id=document_id,
            document_text=document_text,
            profile_id=profile_id,
            subscription_id=subscription_id,
            kb_provider=None,
            llm_call=llm_call,
        ))
        for insight in result.insights:
            try:
                store.write(insight)
                written += 1
            except Exception as exc:
                logger.warning("insight write failed: %s", exc)
    return {"status": "ok", "written": written}


def run_researcher_v2_for_profile(
    *,
    profile_id: str,
    subscription_id: str,
    documents: List[Dict[str, Any]],
    domain_hint: str = "generic",
) -> Dict[str, Any]:
    enabled = _enabled_types(list(_PROFILE_TYPES))
    if not enabled:
        return {"status": "skipped_flag_off"}
    adapter = resolve_default_adapter(domain=domain_hint, subscription_id=subscription_id)
    store = resolve_default_store()
    llm_call = resolve_default_llm()
    written = 0
    for itype in enabled:
        if itype not in adapter.researcher.insight_types:
            continue
        result = run_profile_pass(ProfilePassInputs(
            adapter=adapter,
            insight_type=itype,
            documents=documents,
            profile_id=profile_id,
            subscription_id=subscription_id,
            kb_provider=None,
            llm_call=llm_call,
        ))
        for insight in result.insights:
            try:
                store.write(insight)
                written += 1
            except Exception as exc:
                logger.warning("profile insight write failed: %s", exc)
    return {"status": "ok", "written": written}


def write_doc_status(
    *,
    collection,
    document_id: str,
    status: str,
    adapter_version: str,
    written_count: int,
) -> None:
    """Write per-doc researcher_v2 status. Only touches researcher_v2.* keys."""
    collection.update_one(
        {"document_id": document_id},
        {"$set": {
            "researcher_v2.status": status,
            "researcher_v2.adapter_version": adapter_version,
            "researcher_v2.written_count": written_count,
        }},
        upsert=True,
    )


# Celery task bindings — registered if Celery app is available
try:
    from src.celery_app import app as _celery_app

    @_celery_app.task(name="src.tasks.researcher_v2.run_researcher_v2_for_doc_task", bind=True)
    def run_researcher_v2_for_doc_task(self, *, document_id, profile_id, subscription_id, document_text, domain_hint="generic"):
        return run_researcher_v2_for_doc(
            document_id=document_id,
            profile_id=profile_id,
            subscription_id=subscription_id,
            document_text=document_text,
            domain_hint=domain_hint,
        )

    @_celery_app.task(name="src.tasks.researcher_v2.run_researcher_v2_for_profile_task", bind=True)
    def run_researcher_v2_for_profile_task(self, *, profile_id, subscription_id, documents, domain_hint="generic"):
        return run_researcher_v2_for_profile(
            profile_id=profile_id,
            subscription_id=subscription_id,
            documents=documents,
            domain_hint=domain_hint,
        )
except Exception as _exc:  # pragma: no cover
    logger.debug("Celery binding for researcher_v2 deferred: %s", _exc)
