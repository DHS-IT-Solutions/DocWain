"""Profile Intelligence API — serves auto-generated document insights."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

profile_intelligence_router = APIRouter(prefix="/profiles", tags=["Profile Intelligence"])


@profile_intelligence_router.get("/{profile_id}/intelligence", summary="Get profile intelligence report")
async def get_profile_intelligence(profile_id: str):
    """Return the auto-generated intelligence report for a profile.

    Contains:
    - Profile overview (summary, key metrics, overall insights)
    - Per-document briefs (key facts, entities, insights per document)
    - Cross-document analysis (comparisons, trends, anomalies, rankings)
    """
    from pymongo import MongoClient
    from src.api.config import Config

    client = MongoClient(Config.MongoDB.URI, serverSelectionTimeoutMS=5000)
    db = client[Config.MongoDB.DB]
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
