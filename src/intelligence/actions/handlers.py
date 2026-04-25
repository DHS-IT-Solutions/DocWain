"""Per-type action handlers. v1: side-effect-free.

artifact      — render template, return content (production wires Blob upload)
form_fill     — produce structured form data from inputs
plan          — produce a checklist from steps
reminder      — schedule an in-system reminder (no external email/SMS in v1)
"""
from __future__ import annotations

import os
from typing import Any, Dict


def _render(template: str, *, profile_id: str, inputs: Dict[str, Any]) -> str:
    out = template.replace("{{ profile_id }}", profile_id)
    for k, v in inputs.items():
        out = out.replace(f"{{{{ inputs['{k}'] }}}}", str(v))
    return out


def artifact_handler(*, action, profile_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    template_root = os.environ.get("ACTION_TEMPLATE_ROOT", "src/intelligence/actions/templates")
    template_path = os.path.join(template_root, action.artifact_template or "")
    if not os.path.exists(template_path):
        return {"status": "template_not_found", "template": action.artifact_template}
    with open(template_path, "r", encoding="utf-8") as fh:
        template = fh.read()
    content = _render(template, profile_id=profile_id, inputs=inputs)
    return {
        "status": "rendered",
        "artifact_blob_url": f"blob://artifacts/{profile_id}/{action.action_id}.md",
        "artifact_content": content,
    }


def form_fill_handler(*, action, profile_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "filled",
        "form_id": action.action_id,
        "profile_id": profile_id,
        "form_data": dict(inputs),
    }


def plan_handler(*, action, profile_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    steps = list(inputs.get("steps") or [])
    checklist = [{"step": s, "done": False} for s in steps]
    return {"status": "planned", "profile_id": profile_id, "checklist": checklist}


def reminder_handler(*, action, profile_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "scheduled_in_system",
        "profile_id": profile_id,
        "fire_at": inputs.get("fire_at"),
        "message": inputs.get("message", ""),
    }
