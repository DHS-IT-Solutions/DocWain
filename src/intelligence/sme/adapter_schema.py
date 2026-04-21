"""Pydantic schema for SME adapter YAMLs (spec §5).

Strict: extra fields rejected; semver enforced; dossier weights must sum to 1.0.

The ``Adapter`` model also carries two runtime-injected fields — ``content_hash``
and ``source_path`` — populated by :class:`AdapterLoader` after parsing, so
consumers downstream (synthesizer, retrieval, recommendation grounding) can read
the resolved hash/version directly off the returned model (ERRATA §1).
"""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")
_DOMAIN_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{1,31}$")


class _Strict(BaseModel):
    """Base class: all adapter submodels reject unknown fields."""

    model_config = ConfigDict(extra="forbid")


class Persona(_Strict):
    """Persona section: role, voice, grounding rules."""

    role: str
    voice: str
    grounding_rules: list[str] = Field(default_factory=list)


class DossierConfig(_Strict):
    """Dossier config: section weights + prompt template path."""

    section_weights: dict[str, float]
    prompt_template: str

    @model_validator(mode="after")
    def _weights_sum_one(self) -> "DossierConfig":
        total = sum(self.section_weights.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"dossier.section_weights must sum to 1.0 (got {total:.3f})"
            )
        return self


class InsightDetector(_Strict):
    """Insight detector config: type, rule name, and per-rule params."""

    type: Literal["trend", "anomaly", "gap", "risk", "opportunity", "conflict"]
    rule: str
    params: dict[str, Any] = Field(default_factory=dict)


class ComparisonAxis(_Strict):
    """Comparison axis used by comparative-register builder."""

    name: str
    dimension: str
    unit: str | None = None


class KGInferenceRule(_Strict):
    """KG multi-hop inference rule."""

    pattern: str
    produces: str
    confidence_floor: float = Field(ge=0.0, le=1.0)
    max_hops: int = Field(ge=2, le=5)


class RecommendationFrame(_Strict):
    """Recommendation frame used by recommendation-bank builder."""

    frame: str
    template: str
    requires: dict[str, list[str]] = Field(default_factory=dict)


class ResponsePersonaPrompts(_Strict):
    """Per-intent prompt template paths."""

    diagnose: str
    analyze: str
    recommend: str


class RetrievalCaps(_Strict):
    """Retrieval caps: per-intent max_pack_tokens."""

    max_pack_tokens: dict[str, int]


class OutputCaps(_Strict):
    """Per-intent output-length caps (optional)."""

    analyze: int | None = None
    diagnose: int | None = None
    recommend: int | None = None
    investigate: int | None = None
    summarize: int | None = None
    compare: int | None = None


class Adapter(_Strict):
    """Top-level adapter model.

    Fields ``content_hash`` and ``source_path`` are runtime-injected by
    :class:`AdapterLoader` after YAML parse; they are NOT part of the YAML
    content authored by operators (ERRATA §1).
    """

    domain: str
    version: str
    persona: Persona
    dossier: DossierConfig
    insight_detectors: list[InsightDetector] = Field(default_factory=list)
    comparison_axes: list[ComparisonAxis] = Field(default_factory=list)
    kg_inference_rules: list[KGInferenceRule] = Field(default_factory=list)
    recommendation_frames: list[RecommendationFrame] = Field(default_factory=list)
    response_persona_prompts: ResponsePersonaPrompts
    retrieval_caps: RetrievalCaps
    output_caps: OutputCaps

    # Runtime-injected by AdapterLoader after parsing; NOT part of the authored
    # YAML. Exposed so callers (Phase 2+ synthesizer, Phase 4 recommendation
    # grounding) can read them directly off the Adapter instance. See ERRATA §1.
    content_hash: str | None = None
    source_path: str | None = None

    @field_validator("version")
    @classmethod
    def _semver(cls, v: str) -> str:
        if not _SEMVER_RE.match(v):
            raise ValueError(f"version must be semver (got {v!r})")
        return v

    @field_validator("domain")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not _DOMAIN_SLUG_RE.match(v):
            raise ValueError(f"domain must be lowercase slug (got {v!r})")
        return v
