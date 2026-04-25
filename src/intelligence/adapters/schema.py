"""Adapter YAML schema — parsed dataclasses.

Adapters live in Azure Blob (per feedback_adapter_yaml_blob.md). Code
ships only the generic.yaml fallback, uploaded to Blob at first deploy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class AppliesWhen:
    domain_classifier_labels: List[str] = field(default_factory=list)
    doc_type_hints: List[str] = field(default_factory=list)
    keyword_evidence_min: int = 0
    keywords: List[str] = field(default_factory=list)


@dataclass
class InsightTypeConfig:
    prompt_template: str = ""
    enabled: bool = True
    requires_min_docs: int = 1


@dataclass
class ResearcherSection:
    insight_types: Dict[str, InsightTypeConfig] = field(default_factory=dict)


@dataclass
class SanctionedKb:
    kb_id: str
    ref: str
    describes: str = ""


@dataclass
class KnowledgeConfig:
    sanctioned_kbs: List[SanctionedKb] = field(default_factory=list)
    citation_rule: str = "doc_grounded_first"


@dataclass
class Watchlist:
    id: str
    description: str
    eval: str
    fires_insight_type: str


@dataclass
class ActionTemplate:
    action_id: str
    title: str
    action_type: str  # artifact | form_fill | alert | plan | reminder
    artifact_template: Optional[str] = None
    requires_confirmation: bool = True
    input_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualizationSpec:
    viz_id: str
    insight_types: List[str]


@dataclass
class Adapter:
    name: str
    version: str
    description: str
    applies_when: AppliesWhen
    researcher: ResearcherSection
    knowledge: KnowledgeConfig
    watchlists: List[Watchlist] = field(default_factory=list)
    actions: List[ActionTemplate] = field(default_factory=list)
    visualizations: List[VisualizationSpec] = field(default_factory=list)


def parse_adapter_yaml(text: str) -> Adapter:
    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid adapter YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("adapter YAML root must be a mapping")
    aw = raw.get("applies_when") or {}
    applies_when = AppliesWhen(
        domain_classifier_labels=list(aw.get("domain_classifier_labels") or []),
        doc_type_hints=list(aw.get("doc_type_hints") or []),
        keyword_evidence_min=int(aw.get("keyword_evidence_min") or 0),
        keywords=list(aw.get("keywords") or []),
    )
    r = raw.get("researcher") or {}
    insight_types = {
        name: InsightTypeConfig(
            prompt_template=str(v.get("prompt_template") or ""),
            enabled=bool(v.get("enabled", True)),
            requires_min_docs=int(v.get("requires_min_docs") or 1),
        )
        for name, v in (r.get("insight_types") or {}).items()
    }
    researcher = ResearcherSection(insight_types=insight_types)
    k = raw.get("knowledge") or {}
    sanctioned = [
        SanctionedKb(
            kb_id=str(kb.get("kb_id") or ""),
            ref=str(kb.get("ref") or ""),
            describes=str(kb.get("describes") or ""),
        )
        for kb in (k.get("sanctioned_kbs") or [])
    ]
    knowledge = KnowledgeConfig(
        sanctioned_kbs=sanctioned,
        citation_rule=str(k.get("citation_rule") or "doc_grounded_first"),
    )
    watchlists = [
        Watchlist(
            id=str(w.get("id") or ""),
            description=str(w.get("description") or ""),
            eval=str(w.get("eval") or ""),
            fires_insight_type=str(w.get("fires_insight_type") or ""),
        )
        for w in (raw.get("watchlists") or [])
    ]
    actions = [
        ActionTemplate(
            action_id=str(a.get("action_id") or ""),
            title=str(a.get("title") or ""),
            action_type=str(a.get("action_type") or ""),
            artifact_template=a.get("artifact_template"),
            requires_confirmation=bool(a.get("requires_confirmation", True)),
            input_schema=dict(a.get("input_schema") or {}),
        )
        for a in (raw.get("actions") or [])
    ]
    visualizations = [
        VisualizationSpec(
            viz_id=str(v.get("viz_id") or ""),
            insight_types=list(v.get("insight_types") or []),
        )
        for v in (raw.get("visualizations") or [])
    ]
    return Adapter(
        name=str(raw.get("name") or ""),
        version=str(raw.get("version") or ""),
        description=str(raw.get("description") or ""),
        applies_when=applies_when,
        researcher=researcher,
        knowledge=knowledge,
        watchlists=watchlists,
        actions=actions,
        visualizations=visualizations,
    )
