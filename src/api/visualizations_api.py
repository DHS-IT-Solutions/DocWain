"""Visualization spec endpoint."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from src.api.config import insight_flag_enabled

visualizations_router = APIRouter(prefix="/profiles/v2", tags=["Visualizations"])


def list_visualizations_for_profile(*, profile_id: str) -> List[Dict[str, Any]]:
    raise NotImplementedError


@visualizations_router.get("/{profile_id}/visualizations")
async def list_viz(profile_id: str):
    if not insight_flag_enabled("VIZ_ENABLED"):
        raise HTTPException(status_code=404, detail="Feature not enabled")
    return {
        "profile_id": profile_id,
        "visualizations": list_visualizations_for_profile(profile_id=profile_id),
    }
