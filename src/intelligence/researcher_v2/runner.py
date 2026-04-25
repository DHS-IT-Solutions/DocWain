"""Per-doc + profile-level researcher v2 runner.

Loads adapter, builds prompts, calls LLM, parses, constructs Insight
objects, applies validators (citation, body-separation), returns
results. Idempotency keys are computed at write time by InsightStore.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.intelligence.adapters.schema import Adapter
from src.intelligence.insights.schema import Insight, EvidenceSpan, KbRef
from src.intelligence.knowledge.provider import KnowledgeProvider
from src.intelligence.knowledge.template_resolver import resolve_template
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
class DocPassInputs:
    adapter: Adapter
    insight_type: str
    document_id: str
    document_text: str
    profile_id: str
    subscription_id: str
    kb_provider: Optional[KnowledgeProvider]
    llm_call: LlmCall


@dataclass
class DocPassResult:
    insights: List[Insight] = field(default_factory=list)
    skipped_reason: Optional[str] = None
    parse_error: Optional[str] = None


def run_per_doc_insight_pass(inp: DocPassInputs) -> DocPassResult:
    cfg = inp.adapter.researcher.insight_types.get(inp.insight_type)
    if cfg is None or not cfg.enabled:
        return DocPassResult(skipped_reason="type_disabled")
    kb_context = ""
    if inp.kb_provider is not None and cfg.prompt_template:
        kb_context = resolve_template(cfg.prompt_template, kb=inp.kb_provider)
    user_prompt = build_typed_insight_prompt(
        insight_type=inp.insight_type,
        domain_name=inp.adapter.name,
        document_text=inp.document_text,
        kb_context=kb_context,
    )
    try:
        raw = inp.llm_call(system=SYSTEM_PROMPT, user=user_prompt)
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return DocPassResult(skipped_reason="llm_error")
    try:
        parsed = parse_typed_insight_response(raw)
    except ParseError as exc:
        return DocPassResult(parse_error=str(exc))
    adapter_version = f"{inp.adapter.name}@{inp.adapter.version}"
    insights: List[Insight] = []
    for p in parsed:
        spans = [EvidenceSpan(**s) for s in p.evidence_doc_spans]
        kb_refs = [KbRef(**r) for r in p.external_kb_refs]
        try:
            insights.append(Insight(
                insight_id=str(uuid.uuid4()),
                profile_id=inp.profile_id,
                subscription_id=inp.subscription_id,
                document_ids=[inp.document_id],
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
            logger.debug("dropping invalid insight: %s", exc)
            continue
    return DocPassResult(insights=insights)
