"""Insights surface endpoints — read-only. Lookup against Mongo control-plane index.

All endpoints flag-gated by INSIGHTS_DASHBOARD_ENABLED. When flag is off,
the router returns 404 for every path — no behavioural change to /api/ask.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.config import insight_flag_enabled

insights_router = APIRouter(prefix="/profiles/v2", tags=["Insights"])


def list_insights_for_profile(
    *,
    profile_id: str,
    insight_types: Optional[List[str]] = None,
    severities: Optional[List[str]] = None,
    domain: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Hook — production wiring binds InsightStore.list_for_profile."""
    raise NotImplementedError


def get_insight_full(*, insight_id: str) -> Optional[Dict[str, Any]]:
    """Hook — production wiring binds Qdrant insight payload fetch."""
    raise NotImplementedError


def _gate():
    if not insight_flag_enabled("INSIGHTS_DASHBOARD_ENABLED"):
        raise HTTPException(status_code=404, detail="Feature not enabled")


@insights_router.get("/{profile_id}/insights")
async def list_insights(
    profile_id: str,
    insight_type: Optional[str] = None,
    severity: Optional[str] = None,
    domain: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
):
    _gate()
    types = insight_type.split(",") if insight_type else None
    sevs = severity.split(",") if severity else None
    rows = list_insights_for_profile(
        profile_id=profile_id,
        insight_types=types,
        severities=sevs,
        domain=domain,
        since=since,
        limit=limit,
        offset=offset,
    )
    domains_present = sorted({r.get("domain", "") for r in rows if r.get("domain")})
    last_refresh = max((r.get("refreshed_at", "") for r in rows), default="")
    stale_count = sum(1 for r in rows if r.get("stale"))
    return {
        "profile_id": profile_id,
        "total": len(rows),
        "stale_count": stale_count,
        "insights": rows,
        "domains_present": domains_present,
        "last_refresh": last_refresh,
    }


@insights_router.get("/{profile_id}/insights/{insight_id}")
async def get_insight(profile_id: str, insight_id: str):
    _gate()
    obj = get_insight_full(insight_id=insight_id)
    if obj is None or obj.get("profile_id") != profile_id:
        raise HTTPException(status_code=404, detail="Not found")
    return obj


@insights_router.get("/{profile_id}/refresh-status")
async def refresh_status(profile_id: str):
    _gate()
    return {
        "profile_id": profile_id,
        "last_on_upload_refresh": None,
        "last_scheduled_run": None,
        "pending_watchlist_evaluations": 0,
        "stale_insight_count": 0,
    }
