"""Pydantic models for the Teams service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TeamsDocument:
    """A document being processed through the Teams pipeline."""

    document_id: str
    tenant_id: str
    user_id: str
    filename: str
    source: str  # "attachment" | "onedrive"
    source_url: Optional[str] = None
    pipeline: str = "full"  # "express" | "full"
    status: str = "downloading"
    progress: Dict[str, Any] = field(default_factory=dict)
    teams_message_id: Optional[str] = None
    teams_conversation_id: Optional[str] = None


@dataclass
class TenantInfo:
    """Auto-provisioned tenant record."""

    tenant_id: str
    display_name: str
    qdrant_collection: str
    settings: Dict[str, Any] = field(default_factory=lambda: {
        "kg_enabled": True,
        "max_documents": 1000,
        "express_pipeline": True,
    })
    document_count: int = 0
