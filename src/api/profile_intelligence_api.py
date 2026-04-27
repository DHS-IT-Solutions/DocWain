"""Profile Intelligence API — serves auto-generated document insights."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.utils.logging_utils import get_logger

# Manual regenerate is intentionally synchronous over the same Celery task so
# the UI can fire-and-forget; no separate code path means no schema drift.

logger = get_logger(__name__)

profile_intelligence_router = APIRouter(prefix="/profiles", tags=["Profile Intelligence"])


@profile_intelligence_router.get("/{profile_id}/intelligence", summary="Get profile intelligence report")
async def get_profile_intelligence(profile_id: str):
    """Return the auto-generated intelligence report for a profile.

    Contains:
    - Profile overview (summary, key metrics, overall insights)
    - Per-document briefs (key facts, entities, insights per document)
    - Cross-document analysis (comparisons, trends, anomalies, rankings)

    UAT Issue #5 (2026-04-27): use the shared dataHandler db (which has
    a 20s serverSelectionTimeout and survives transient CosmosDB topology
    drops) and add a single retry on ServerSelectionTimeoutError so a
    transient drop doesn't propagate as 500 to the user.
    """
    import time
    from pymongo.errors import ServerSelectionTimeoutError, AutoReconnect
    from src.api.dataHandler import db

    last_exc = None
    for attempt in range(2):  # initial + 1 retry
        try:
            report = db["profile_intelligence"].find_one(
                {"profile_id": profile_id},
                {"_id": 0},
            )
            if not report:
                return {
                    "profile_id": profile_id,
                    "status": "pending",
                    "message": "No intelligence report yet. Reports generate automatically after documents complete processing.",
                    "profile_overview": None,
                    "document_briefs": [],
                    "cross_document_analysis": None,
                }
            return report
        except (ServerSelectionTimeoutError, AutoReconnect) as exc:
            last_exc = exc
            logger.warning(
                "Mongo timeout on profile_intelligence (attempt %d/2): %s",
                attempt + 1, exc,
            )
            if attempt == 0:
                time.sleep(0.3)  # short backoff; cluster usually recovers
                continue
            # Both attempts failed — return a friendly 503 instead of 500
            raise HTTPException(
                status_code=503,
                detail="Profile intelligence is temporarily unavailable. Please retry in a moment.",
            )


@profile_intelligence_router.post(
    "/{profile_id}/intelligence/regenerate",
    summary="Force-regenerate profile intelligence for every doc in the profile",
)
async def regenerate_profile_intelligence_api(
    profile_id: str,
    subscription_id: Optional[str] = Query(None, description="Optional subscription scope"),
    force: bool = Query(False, description="Rebuild every brief (true) vs only insufficient ones (false)"),
    sync: bool = Query(False, description="Run inline (true) vs dispatch to Celery (false)"),
):
    """Force a fresh profile-intelligence build.

    Default (``sync=false``) dispatches one Celery task per document onto the
    ``profile_intelligence_queue`` and returns immediately with the count.
    ``sync=true`` runs the regeneration inline — useful for ad-hoc fixes from
    a shell, but blocking from the API process.
    """
    if sync:
        try:
            from src.intelligence.profile_intelligence import regenerate_profile_intelligence
            return regenerate_profile_intelligence(profile_id, subscription_id, force=bool(force))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"regenerate failed: {exc}")

    from src.api.document_status import get_documents_collection
    docs_col = get_documents_collection()
    if docs_col is None:
        raise HTTPException(status_code=503, detail="documents collection unavailable")
    q: Dict[str, Any] = {
        "profile_id": profile_id,
        "status": {"$in": [
            "EMBEDDING_COMPLETED",
            "TRAINING_COMPLETED",
            "TRAINING_PARTIALLY_COMPLETED",
            "SCREENING_COMPLETED",
            "UNDER_REVIEW",
        ]},
    }
    if subscription_id:
        q["subscription_id"] = subscription_id
    docs = list(docs_col.find(q, {"document_id": 1, "subscription_id": 1}))

    try:
        from src.tasks.profile_intelligence import generate_profile_intelligence_task
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"profile_intelligence task unavailable: {exc}")

    dispatched = 0
    skipped = 0
    for d in docs:
        doc_id = d.get("document_id")
        sub = d.get("subscription_id") or subscription_id
        if not doc_id or not sub:
            skipped += 1
            continue
        try:
            generate_profile_intelligence_task.delay(doc_id, profile_id, sub)
            dispatched += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("regenerate dispatch failed for %s: %s", doc_id, exc)
            skipped += 1

    return {
        "profile_id": profile_id,
        "subscription_id": subscription_id,
        "scanned": len(docs),
        "dispatched": dispatched,
        "skipped": skipped,
        "force": bool(force),
    }


@profile_intelligence_router.post(
    "/{profile_id}/researcher/backfill",
    summary="Dispatch Researcher Agent for every extraction-complete doc in the profile",
)
async def backfill_profile_researcher(profile_id: str):
    """Dispatch ``run_researcher_agent`` as Celery tasks for every doc in the
    profile that has completed extraction but doesn't have a Researcher
    Agent output yet.

    Use after:
      * upgrading the researcher payload shape
      * onboarding a profile that was ingested before Researcher Agent
        was wired up
      * after any bulk re-extraction event

    Returns immediately with counts; tasks run async on researcher_queue.
    """
    from src.api.document_status import get_documents_collection
    col = get_documents_collection()
    if col is None:
        raise HTTPException(status_code=503, detail="documents collection unavailable")
    candidates = list(col.find(
        {
            "profile_id": profile_id,
            "status": {"$in": [
                "EXTRACTION_COMPLETED", "UNDER_REVIEW",
                "SCREENING_COMPLETED", "TRAINING_COMPLETED",
            ]},
        },
        {"_id": 1, "document_id": 1, "subscription_id": 1, "profile_id": 1, "researcher": 1},
    ))
    dispatched: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    try:
        from src.tasks.researcher import run_researcher_agent
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"researcher task unavailable: {exc}")
    for d in candidates:
        doc_id = str(d.get("document_id") or d.get("_id"))
        sub_id = d.get("subscription_id")
        prof_id = d.get("profile_id")
        researcher = d.get("researcher") or {}
        if researcher.get("status") == "RESEARCHER_COMPLETED" and researcher.get("insights"):
            skipped.append({"document_id": doc_id, "reason": "already_complete"})
            continue
        if not sub_id or not prof_id:
            skipped.append({"document_id": doc_id, "reason": "missing_ids"})
            continue
        try:
            task = run_researcher_agent.delay(doc_id, sub_id, prof_id)
            dispatched.append({"document_id": doc_id, "task_id": getattr(task, "id", None)})
        except Exception as exc:  # noqa: BLE001
            failed.append({"document_id": doc_id, "error": str(exc)[:200]})
    return {
        "profile_id": profile_id,
        "scanned": len(candidates),
        "dispatched": len(dispatched),
        "skipped": len(skipped),
        "failed": len(failed),
        "details": {"dispatched": dispatched, "skipped": skipped, "failed": failed},
    }


@profile_intelligence_router.get(
    "/{profile_id}/insights",
    summary="Get corpus-level insights (fast, aggregates over hot cache + Researcher Agent)",
)
async def get_profile_insights(
    profile_id: str,
    min_doc_count: int = Query(2, ge=1, description="Minimum doc appearances for 'prevalent' entities"),
    limit_entities: int = Query(30, ge=1, le=200, description="Max entities returned per type bucket"),
    include_researcher: bool = Query(True, description="Merge per-doc Researcher Agent insights"),
):
    """Corpus-level insights — the shortest path from ingested documents to
    actionable cross-document intelligence.

    Builds a single response from data already produced during ingestion:
      • Dominant domain (hot-cache profile_domain)
      • Prevalent entities (hot-cache, grouped by type) — "who / what / where
        shows up across the corpus?"
      • Top relationships by confidence (hot-cache relationships)
      • Aggregated Researcher Agent insights across documents (summary +
        anomalies + recommendations + questions_to_ask) when available
      • Document counts and status snapshot

    Zero LLM round trips — every byte comes from Redis or Mongo.
    Intended for a UI "Insights" tab that loads in <1s.
    """
    from src.api.config import Config
    from src.api.document_status import get_documents_collection
    from src.intelligence.hot_cache import (
        get_prevalent_entities,
        get_profile_domain,
        get_top_relationships,
    )

    redis_client = None
    try:
        from src.api.dw_newron import get_redis_client
        redis_client = get_redis_client()
    except Exception:
        pass

    # --- Prevalent entities (hot cache) -----------------------------------
    prevalent = get_prevalent_entities(
        redis_client, profile_id,
        min_doc_count=min_doc_count, limit=limit_entities,
    ) if redis_client else []
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for ent in prevalent:
        by_type.setdefault(ent.get("type") or "other", []).append(ent)

    # --- Top relationships (hot cache) ------------------------------------
    top_relationships: List[Dict[str, Any]] = []
    if redis_client:
        try:
            top_relationships = get_top_relationships(redis_client, profile_id, limit=20) or []
        except Exception as exc:  # noqa: BLE001
            logger.debug("top_relationships lookup failed: %s", exc)

    # --- Profile dominant domain -----------------------------------------
    dominant_domain = get_profile_domain(redis_client, profile_id) if redis_client else "general"

    # --- Doc-level state snapshot + optional Researcher merge -------------
    docs_col = get_documents_collection()
    projection = {
        "document_id": 1, "status": 1, "source_file": 1,
        "researcher": 1, "updated_at": 1,
    }
    docs = list(docs_col.find({"profile_id": profile_id}, projection)) if docs_col is not None else []

    by_status: Dict[str, int] = {}
    researcher_summaries: List[Dict[str, Any]] = []
    all_anomalies: List[Dict[str, Any]] = []
    all_recommendations: List[Dict[str, Any]] = []
    all_questions: List[Dict[str, Any]] = []
    for d in docs:
        st = d.get("status") or "UNKNOWN"
        by_status[st] = by_status.get(st, 0) + 1
        r = d.get("researcher") or {}
        if include_researcher and r.get("status") == "RESEARCHER_COMPLETED":
            insights = r.get("insights") or {}
            doc_meta = {
                "document_id": d.get("document_id"),
                "document_name": d.get("source_file"),
                "confidence": r.get("confidence"),
            }
            if insights.get("summary"):
                researcher_summaries.append({**doc_meta, "summary": insights["summary"]})
            for item in (insights.get("anomalies") or []):
                all_anomalies.append({**doc_meta, "anomaly": item})
            for item in (insights.get("recommendations") or []):
                all_recommendations.append({**doc_meta, "recommendation": item})
            for item in (insights.get("questions_to_ask") or []):
                all_questions.append({**doc_meta, "question": item})

    return {
        "profile_id": profile_id,
        "dominant_domain": dominant_domain,
        "document_counts": {
            "total": len(docs),
            "by_status": by_status,
        },
        "prevalent_entities": {
            "by_type": by_type,
            "total": len(prevalent),
        },
        "top_relationships": top_relationships,
        "researcher": {
            "enabled": include_researcher,
            "summaries": researcher_summaries,
            "anomalies": all_anomalies,
            "recommendations": all_recommendations,
            "questions_to_ask": all_questions,
            "docs_with_insights": len(researcher_summaries),
        },
    }
