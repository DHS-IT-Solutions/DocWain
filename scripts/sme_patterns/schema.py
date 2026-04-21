"""Schema for Phase 6 pattern mining.

These models define the Phase 1 / 2 -> Phase 6 seam. ``trace_loader.py`` is the
only module that reads raw JSONL; everything else operates on SynthesisRun /
QueryRun. Changing a field in upstream trace writers requires updating this
file and the loader in lockstep.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class VerifierDrop(BaseModel):
    item_id: str
    builder: str
    reason_code: str
    detail: str = ""


class BuilderTrace(BaseModel):
    builder_name: str
    items_produced: int = 0
    items_persisted: int = 0
    duration_ms: float | None = None
    errors: list[str] = Field(default_factory=list)


class SynthesisRun(BaseModel):
    subscription_id: str
    profile_id: str
    synthesis_id: str
    started_at: datetime
    completed_at: datetime | None = None
    adapter_version: str
    adapter_content_hash: str
    profile_domain: str
    per_builder: dict[str, BuilderTrace] = Field(default_factory=dict)
    verifier_drops: list[VerifierDrop] = Field(default_factory=list)


class QueryFeedback(BaseModel):
    rating: Literal[-1, 0, 1] | None = None
    edited: bool = False
    follow_up_count: int = 0
    source: str = "implicit"


class QueryRun(BaseModel):
    subscription_id: str
    profile_id: str
    profile_domain: str
    query_id: str
    query_text: str
    query_fingerprint: str
    intent: str
    format_hint: str | None = None
    adapter_version: str
    adapter_persona_role: str = ""
    retrieval_layers: dict[str, int] = Field(default_factory=dict)
    pack_tokens: int = 0
    reasoner_prompt_hash: str = ""
    response_len_tokens: int = 0
    citation_verifier_drops: int = 0
    honest_compact_fallback: bool = False
    url_present: bool = False
    url_fetch_ok: bool | None = None
    timing_ms: dict[str, float] = Field(default_factory=dict)
    feedback: QueryFeedback | None = None
    captured_at: datetime


class ClusterType(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    ARTIFACT_UTILITY = "artifact_utility"
    PERSONA_EFFECT = "persona_effect"


class Cluster(BaseModel):
    cluster_id: str
    cluster_type: ClusterType
    size: int
    subscription_ids: list[str] = Field(default_factory=list)
    primary_intent: str | None = None
    profile_domain: str | None = None
    fingerprint_samples: list[str] = Field(default_factory=list)
    short_description: str
    signal_score: float  # pass-specific interpretation; always in [0.0, 1.0] or a rate
    evidence: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


class TrainingCandidate(BaseModel):
    candidate_id: str
    cluster_ids: list[str]
    months_present: int
    total_volume: int
    stabilization_score: float
    dominant_intent: str
    dominant_domain: str
    short_description: str

    @model_validator(mode="after")
    def _require_stabilization(self) -> "TrainingCandidate":
        if self.months_present < 2:
            raise ValueError("TrainingCandidate requires months_present >= 2")
        return self


class PatternReport(BaseModel):
    run_id: str
    period_start: datetime
    period_end: datetime
    num_synth_runs: int
    num_query_runs: int
    successes: list[Cluster] = Field(default_factory=list)
    failures: list[Cluster] = Field(default_factory=list)
    artifact_utility: list[Cluster] = Field(default_factory=list)
    persona_effect: list[Cluster] = Field(default_factory=list)
    training_candidates: list[TrainingCandidate] = Field(default_factory=list)
    rollback_links: list[str] = Field(default_factory=list)
