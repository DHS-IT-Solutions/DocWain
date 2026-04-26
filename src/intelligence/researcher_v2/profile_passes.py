"""Profile-level (cross-doc) researcher v2 passes.

For comparison / conflict / trend / projection — passes that require
≥2 documents. Document_ids on emitted insights point to all docs that
contributed evidence.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.intelligence.adapters.schema import Adapter
from src.intelligence.insights.schema import Insight, EvidenceSpan, KbRef
from src.intelligence.knowledge.provider import KnowledgeProvider
from src.intelligence.researcher_v2.parser import (
    parse_typed_insight_response, ParseError,
)
from src.docwain.prompts.researcher_v2_generic import (
    SYSTEM_PROMPT,
    build_typed_insight_prompt,
)

logger = logging.getLogger(__name__)


LlmCall = Callable[..., str]


@dataclass
class ProfilePassInputs:
    adapter: Adapter
    insight_type: str
    documents: List[Dict[str, Any]]
    profile_id: str
    subscription_id: str
    kb_provider: Optional[KnowledgeProvider]
    llm_call: LlmCall


@dataclass
class ProfilePassResult:
    insights: List[Insight] = field(default_factory=list)
    skipped_reason: Optional[str] = None


def _join_docs(docs):
    parts = []
    for d in docs:
        parts.append(f"=== document_id: {d['document_id']} ===\n{d.get('text', '')}")
    return "\n\n".join(parts)


def run_profile_pass(inp: ProfilePassInputs) -> ProfilePassResult:
    cfg = inp.adapter.researcher.insight_types.get(inp.insight_type)
    if cfg is None or not cfg.enabled:
        return ProfilePassResult(skipped_reason="type_disabled")
    if len(inp.documents) < cfg.requires_min_docs:
        return ProfilePassResult(skipped_reason="below_min_docs")
    user_prompt = build_typed_insight_prompt(
        insight_type=inp.insight_type,
        domain_name=inp.adapter.name,
        document_text=_join_docs(inp.documents),
        kb_context="",
        domain_focus=cfg.domain_focus,
    )
    try:
        raw = inp.llm_call(system=SYSTEM_PROMPT, user=user_prompt)
    except Exception as exc:
        logger.warning("profile-pass LLM failed: %s", exc)
        return ProfilePassResult(skipped_reason="llm_error")
    try:
        parsed = parse_typed_insight_response(raw)
    except ParseError as exc:
        logger.debug("parse error: %s", exc)
        return ProfilePassResult(skipped_reason="parse_error")
    adapter_version = f"{inp.adapter.name}@{inp.adapter.version}"
    insights: List[Insight] = []
    for p in parsed:
        document_ids = sorted({s["document_id"] for s in p.evidence_doc_spans})
        spans = [EvidenceSpan(**s) for s in p.evidence_doc_spans]
        kb_refs = [KbRef(**r) for r in p.external_kb_refs]
        try:
            insights.append(Insight(
                insight_id=str(uuid.uuid4()),
                profile_id=inp.profile_id,
                subscription_id=inp.subscription_id,
                document_ids=document_ids,
                domain=inp.adapter.name,
                insight_type=inp.insight_type,
                headline=p.headline,
                body=p.body,
                evidence_doc_spans=spans,
                external_kb_refs=kb_refs,
                confidence=p.confidence,
                severity=p.severity,
                adapter_version=adapter_version,
            ))
        except ValueError as exc:
            logger.debug("dropping invalid profile insight: %s", exc)
            continue
    return ProfilePassResult(insights=insights)
