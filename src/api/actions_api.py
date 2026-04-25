"""Actions surface endpoints — list + execute. Gated by per-action-type flags.

The action layer surface is gated as a whole behind any of:
  - ACTIONS_ARTIFACT_ENABLED
  - ACTIONS_FORM_FILL_ENABLED
  - ACTIONS_PLAN_ENABLED
  - ACTIONS_REMINDER_ENABLED
If none enabled → 404. Per-action filtering happens inside list_actions_for_profile.
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Body, HTTPException

from src.api.config import insight_flag_enabled

actions_router = APIRouter(prefix="/profiles/v2", tags=["Actions"])


_ACTION_FLAGS = (
    "ACTIONS_ARTIFACT_ENABLED",
    "ACTIONS_FORM_FILL_ENABLED",
    "ACTIONS_PLAN_ENABLED",
    "ACTIONS_REMINDER_ENABLED",
)


def list_actions_for_profile(*, profile_id: str) -> List[Dict[str, Any]]:
    raise NotImplementedError


def execute_action(*, profile_id: str, action_id: str, inputs: Dict[str, Any], confirmed: bool) -> Dict[str, Any]:
    raise NotImplementedError


def _gate():
    if not any(insight_flag_enabled(f) for f in _ACTION_FLAGS):
        raise HTTPException(status_code=404, detail="Feature not enabled")


@actions_router.get("/{profile_id}/actions")
async def list_actions(profile_id: str):
    _gate()
    return {"profile_id": profile_id, "actions": list_actions_for_profile(profile_id=profile_id)}


@actions_router.post("/{profile_id}/actions/{action_id}/execute")
async def execute_action_endpoint(
    profile_id: str,
    action_id: str,
    body: Dict[str, Any] = Body(default_factory=dict),
):
    _gate()
    inputs = body.get("inputs") or {}
    confirmed = bool(body.get("confirmed", False))
    return execute_action(
        profile_id=profile_id,
        action_id=action_id,
        inputs=inputs,
        confirmed=confirmed,
    )
