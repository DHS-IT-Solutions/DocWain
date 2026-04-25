"""Action runner — gated, audited, idempotent.

v1 ships side-effect-free actions only. External-side-effect actions
declared in adapter YAML are detected via _side_effect marker and
disabled at runtime unless ACTIONS_EXTERNAL_SIDE_EFFECTS_ENABLED=true.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from src.api.config import insight_flag_enabled
from src.intelligence.adapters.schema import ActionTemplate

logger = logging.getLogger(__name__)


@dataclass
class ActionExecutionResult:
    status: str
    preview: Optional[str] = None
    output: Dict[str, Any] = field(default_factory=dict)


class ActionRunner:
    def __init__(
        self,
        *,
        handlers: Dict[str, Callable[..., Dict[str, Any]]],
        audit_writer: Callable[..., None],
    ):
        self._handlers = handlers
        self._audit = audit_writer

    def execute(
        self,
        *,
        action: ActionTemplate,
        profile_id: str,
        inputs: Dict[str, Any],
        confirmed: bool,
    ) -> ActionExecutionResult:
        if action.requires_confirmation and not confirmed:
            preview = self._build_preview(action=action, inputs=inputs)
            return ActionExecutionResult(status="needs_confirmation", preview=preview)
        side_effect = getattr(action, "_side_effect", None)
        if side_effect == "external" and not insight_flag_enabled(
            "ACTIONS_EXTERNAL_SIDE_EFFECTS_ENABLED"
        ):
            return ActionExecutionResult(status="external_side_effects_disabled")
        handler = self._handlers.get(action.action_type)
        if handler is None:
            return ActionExecutionResult(status="unknown_action_type")
        try:
            output = handler(action=action, profile_id=profile_id, inputs=inputs)
        except Exception as exc:
            logger.exception("action handler raised: %s", exc)
            return ActionExecutionResult(status="failed")
        try:
            self._audit(
                action_id=action.action_id,
                profile_id=profile_id,
                inputs=inputs,
                output=output,
            )
        except Exception as exc:
            logger.warning("audit write failed: %s", exc)
        return ActionExecutionResult(status="executed", output=output)

    def _build_preview(self, *, action: ActionTemplate, inputs: Dict[str, Any]) -> str:
        return (
            f"Action: {action.title}\n"
            f"Type: {action.action_type}\n"
            f"Inputs: {inputs}\n"
        )
