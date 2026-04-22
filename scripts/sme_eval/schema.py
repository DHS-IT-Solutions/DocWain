"""Schema for DocWain SME evaluation set.

Pydantic models are the single source of truth for eval queries, results,
and metric outputs. The YAML query files in tests/sme_evalset_v1/queries/
must deserialize cleanly into EvalQuery instances.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Domain = Literal["finance", "legal", "hr", "medical", "it_support", "generic"]

Intent = Literal[
    "greeting", "identity", "lookup", "list", "count",
    "summarize", "compare", "overview", "investigate",
    "extract", "aggregate",
    "diagnose", "analyze", "recommend",
]


class EvalQuery(BaseModel):
    """One query in the evaluation set."""

    query_id: str
    query_text: str
    intent: Intent
    profile_domain: Domain
    subscription_id: str
    profile_id: str
    expected_behavior: str = ""
    tags: list[str] = Field(default_factory=list)

    urls_in_query: list[str] = Field(default_factory=list)
    gold_answer_snippets: list[str] = Field(default_factory=list)


class LatencyBreakdown(BaseModel):
    """Timing captured per query."""

    ttft_ms: float | None = None
    total_ms: float
    retrieval_ms: float | None = None
    generation_ms: float | None = None


class EvalResult(BaseModel):
    """One query run against DocWain, fully captured."""

    query: EvalQuery
    response_text: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    latency: LatencyBreakdown
    run_id: str
    captured_at: datetime
    api_status: int
    api_error: str | None = None


class MetricResult(BaseModel):
    """One metric's computed value over an EvalResult or a batch."""

    metric_name: str
    value: float
    details: dict[str, Any] = Field(default_factory=dict)


class BaselineSnapshot(BaseModel):
    """Full baseline captured on one run; frozen and committed."""

    run_id: str
    captured_at: datetime
    git_sha: str
    api_base_url: str
    num_queries: int
    per_domain_counts: dict[Domain, int]

    ragas: dict[str, float | int]
    sme_metrics: dict[str, MetricResult]

    latency_p50_per_intent: dict[Intent, float]
    latency_p95_per_intent: dict[Intent, float]
    latency_p99_per_intent: dict[Intent, float]

    human_rated_sme_score_avg: float | None = None
    human_rated_count: int = 0
