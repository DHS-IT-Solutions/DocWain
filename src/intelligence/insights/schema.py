"""Insight + Action data model.

Canonical schema for the Insights Portal. Persisted to Qdrant payload +
Neo4j Insight nodes + Mongo control-plane index per spec Section 5.1.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

INSIGHT_TYPES = (
    "anomaly", "gap", "comparison", "scenario", "trend",
    "recommendation", "conflict", "projection", "next_action",
)
SEVERITIES = ("info", "notice", "warn", "critical")
ACTION_TYPES = ("artifact", "form_fill", "alert", "plan", "reminder")


@dataclass
class EvidenceSpan:
    document_id: str
    page: int
    char_start: int
    char_end: int
    quote: str


@dataclass
class KbRef:
    kb_id: str
    ref: str
    label: str = ""


@dataclass
class Insight:
    insight_id: str
    profile_id: str
    subscription_id: str
    document_ids: List[str]
    domain: str
    insight_type: str
    headline: str
    body: str
    evidence_doc_spans: List[EvidenceSpan]
    confidence: float
    severity: str
    adapter_version: str
    external_kb_refs: List[KbRef] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    refreshed_at: str = ""
    stale: bool = False
    feature_flags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.insight_type not in INSIGHT_TYPES:
            raise ValueError(
                f"insight_type must be one of {INSIGHT_TYPES}, got {self.insight_type!r}"
            )
        if self.severity not in SEVERITIES:
            raise ValueError(
                f"severity must be one of {SEVERITIES}, got {self.severity!r}"
            )
        if not self.created_at:
            self.created_at = datetime.now(tz=timezone.utc).isoformat()
        if not self.refreshed_at:
            self.refreshed_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Action:
    action_id: str
    profile_id: str
    subscription_id: str
    domain: str
    action_type: str
    title: str
    description: str
    preview: str
    requires_confirmation: bool
    produces_artifact: bool = False
    artifact_template: Optional[str] = None
    input_schema: Dict[str, Any] = field(default_factory=dict)
    executed_at: Optional[str] = None
    execution_status: str = "pending"
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.action_type not in ACTION_TYPES:
            raise ValueError(
                f"action_type must be one of {ACTION_TYPES}, got {self.action_type!r}"
            )
