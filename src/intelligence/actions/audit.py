"""Action execution audit log — records every executed action.

Per spec Section 11.1 — writes to actions_audit Mongo collection.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict


def make_audit_writer(*, collection) -> Callable[..., None]:
    def write(*, action_id: str, profile_id: str, inputs: Dict[str, Any], output: Dict[str, Any]) -> None:
        collection.insert_one({
            "action_id": action_id,
            "profile_id": profile_id,
            "inputs": inputs,
            "output": output,
            "at": datetime.now(tz=timezone.utc).isoformat(),
        })
    return write
