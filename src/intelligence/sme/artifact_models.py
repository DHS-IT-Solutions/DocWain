"""Pydantic models for individual artifact items.

Every item carries provenance (``evidence``) and ``confidence``; the verifier
depends on both. Phase 2 builders produce per-type subclasses, but the unified
``ArtifactItem`` is the contract consumed by verifier, storage, retrieval, and
the serving layer. See ERRATA §3 for the ``.text`` contract.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceRef(BaseModel):
    """Immutable reference back to the source chunk.

    ``quote`` is optional but when present is validated by the verifier against
    the chunk's actual text (evidence-validity check 2).
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str
    chunk_id: str
    quote: str | None = None


class ArtifactItem(BaseModel):
    """Unified item contract across every artifact type.

    Phase 2 produces heterogeneous per-type models (``DossierSection``,
    ``InferredEdge``, …); they all expose a ``.text`` surface (ERRATA §3) and
    serialize down to this shape for storage + retrieval + verification.
    """

    model_config = ConfigDict(extra="forbid")

    item_id: str
    artifact_type: Literal[
        "dossier", "insight", "comparison", "kg_edge", "recommendation"
    ]
    subscription_id: str
    profile_id: str
    text: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    inference_path: list[dict[str, Any]] = Field(default_factory=list)
    domain_tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
