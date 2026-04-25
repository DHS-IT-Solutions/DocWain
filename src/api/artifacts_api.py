"""Artifacts list endpoint."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from src.api.config import insight_flag_enabled

artifacts_router = APIRouter(prefix="/profiles/v2", tags=["Artifacts"])


def list_artifacts_for_profile(*, profile_id: str) -> List[Dict[str, Any]]:
    raise NotImplementedError


@artifacts_router.get("/{profile_id}/artifacts")
async def list_artifacts(profile_id: str):
    if not insight_flag_enabled("ACTIONS_ARTIFACT_ENABLED"):
        raise HTTPException(status_code=404, detail="Feature not enabled")
    try:
        artifacts = list_artifacts_for_profile(profile_id=profile_id)
    except NotImplementedError:
        artifacts = []
    return {"profile_id": profile_id, "artifacts": artifacts}
