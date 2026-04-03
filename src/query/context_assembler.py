"""Build structured multi-block context for Phase 3 response generation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.query.planner import QueryPlan
from src.query.executor import ExecutionResult, StepResult

logger = logging.getLogger(__name__)

# Default token budget (estimated as chars / 4)
_DEFAULT_TOKEN_BUDGET = 12000


def assemble_context(
    execution_result: ExecutionResult,
    profile_intelligence: Optional[Dict[str, Any]] = None,
    plan: Optional[QueryPlan] = None,
    token_budget: int = _DEFAULT_TOKEN_BUDGET,
) -> str:
    """Produce a multi-block context string for Phase 3 generation.

    Blocks (in order):
        <profile_context>   — domain, doc count, entity summary, insights
        <retrieved_evidence> — from search steps with source attribution
        <kg_context>         — entities and relationships from KG steps
        <knowledge_pack_context> — domain knowledge with citations
        <cross_reference_results> — conflicts / agreements
        <spreadsheet_data>   — structured tabular results

    Args:
        execution_result: Output from PlanExecutor.execute().
        profile_intelligence: Pre-computed profile-level intelligence dict.
        plan: The QueryPlan (used for metadata hints).
        token_budget: Approximate max tokens (chars / 4).

    Returns:
        A single string with XML-tagged blocks.
    """
    profile_intelligence = profile_intelligence or {}
    char_budget = token_budget * 4  # rough chars-to-tokens ratio
    blocks: List[str] = []
    chars_used = 0

    # 1. Profile context (always included, typically small)
    profile_block = _build_profile_block(profile_intelligence, plan)
    if profile_block:
        blocks.append(profile_block)
        chars_used += len(profile_block)

    # Classify step results by action type
    search_results: List[Dict[str, Any]] = []
    kg_results: List[Dict[str, Any]] = []
    knowledge_results: List[Dict[str, Any]] = []
    xref_results: List[Dict[str, Any]] = []
    spreadsheet_results: List[Dict[str, Any]] = []

    for step_id, sr in execution_result.step_results.items():
        if sr.error:
            continue
        if sr.action == "search":
            search_results.extend(sr.data)
        elif sr.action == "knowledge_search":
            knowledge_results.extend(sr.data)
        elif sr.action in ("kg_lookup", "kg_search"):
            kg_results.extend(sr.data)
        elif sr.action == "cross_reference":
            xref_results.extend(sr.data)
        elif sr.action == "spreadsheet_query":
            spreadsheet_results.extend(sr.data)

    # 2. Retrieved evidence (largest block — gets most budget)
    evidence_budget = int((char_budget - chars_used) * 0.50)
    evidence_block = _build_evidence_block(search_results, evidence_budget)
    if evidence_block:
        blocks.append(evidence_block)
        chars_used += len(evidence_block)

    # 3. KG context
    kg_budget = int((char_budget - chars_used) * 0.30)
    kg_block = _build_kg_block(kg_results, kg_budget)
    if kg_block:
        blocks.append(kg_block)
        chars_used += len(kg_block)

    # 4. Knowledge pack context
    kp_budget = int((char_budget - chars_used) * 0.40)
    kp_block = _build_knowledge_pack_block(knowledge_results, kp_budget)
    if kp_block:
        blocks.append(kp_block)
        chars_used += len(kp_block)

    # 5. Cross-reference results
    xref_block = _build_xref_block(xref_results)
    if xref_block:
        blocks.append(xref_block)
        chars_used += len(xref_block)

    # 6. Spreadsheet data
    ss_budget = int((char_budget - chars_used) * 0.50) if chars_used < char_budget else 500
    ss_block = _build_spreadsheet_block(spreadsheet_results, ss_budget)
    if ss_block:
        blocks.append(ss_block)

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Block builders
# ---------------------------------------------------------------------------

def _build_profile_block(
    profile: Dict[str, Any],
    plan: Optional[QueryPlan],
) -> str:
    if not profile and plan is None:
        return ""

    lines: List[str] = []
    if profile.get("primary_domain"):
        lines.append(f"Domain: {profile['primary_domain']}")
    if profile.get("doc_count"):
        lines.append(f"Documents in profile: {profile['doc_count']}")
    if profile.get("entity_summary"):
        lines.append(f"Key entities: {profile['entity_summary']}")
    if profile.get("insights"):
        insights = profile["insights"]
        if isinstance(insights, list):
            for ins in insights[:5]:
                lines.append(f"- {ins}")
        elif isinstance(insights, str):
            lines.append(insights)
    if plan is not None:
        lines.append(f"Query intent: {plan.intent}")
        lines.append(f"Domain pack: {plan.domain_pack}")

    if not lines:
        return ""
    inner = "\n".join(lines)
    return f"<profile_context>\n{inner}\n</profile_context>"


def _build_evidence_block(
    results: List[Dict[str, Any]],
    char_budget: int,
) -> str:
    if not results:
        return ""

    # Sort by relevance descending
    sorted_results = sorted(results, key=lambda r: r.get("relevance_score", 0), reverse=True)
    lines: List[str] = []
    chars = 0

    for idx, r in enumerate(sorted_results, 1):
        content = r.get("content", "").strip()
        if not content:
            continue
        source = r.get("source", "unknown")
        page = r.get("page")
        section = r.get("section", "")
        score = r.get("relevance_score", 0)

        header_parts = [f"Source: {source}"]
        if section:
            header_parts.append(f"Section: {section}")
        if page is not None:
            header_parts.append(f"Page: {page}")
        header_parts.append(f"Score: {score:.3f}")
        header = ", ".join(header_parts)

        entry = f"[{idx}] {header}\n{content}"
        entry_len = len(entry)
        if chars + entry_len > char_budget:
            # Try truncating content to fit
            remaining = char_budget - chars - len(f"[{idx}] {header}\n") - 20
            if remaining > 100:
                entry = f"[{idx}] {header}\n{content[:remaining]}..."
                lines.append(entry)
            break
        lines.append(entry)
        chars += entry_len

    if not lines:
        return ""
    inner = "\n\n".join(lines)
    return f"<retrieved_evidence>\n{inner}\n</retrieved_evidence>"


def _build_kg_block(
    results: List[Dict[str, Any]],
    char_budget: int,
) -> str:
    if not results:
        return ""

    entities_seen: Dict[str, str] = {}
    relationships: List[str] = []

    for r in results:
        ename = r.get("entity_name", "")
        etype = r.get("entity_type", "")
        if ename and ename not in entities_seen:
            entities_seen[ename] = etype or "Entity"

        rel = r.get("relationship")
        if rel:
            pred = rel.get("predicate", "related_to")
            target = rel.get("target", "")
            if target:
                relationships.append(f"{ename} --[{pred}]--> {target}")

    lines: List[str] = []
    if entities_seen:
        entity_str = "; ".join(f"{name} ({typ})" for name, typ in entities_seen.items())
        lines.append(f"Entities: {entity_str}")
    if relationships:
        lines.append("Relationships:")
        for rel_line in relationships[:20]:
            lines.append(f"  - {rel_line}")

    if not lines:
        return ""
    inner = "\n".join(lines)
    if len(inner) > char_budget:
        inner = inner[:char_budget] + "..."
    return f"<kg_context>\n{inner}\n</kg_context>"


def _build_knowledge_pack_block(
    results: List[Dict[str, Any]],
    char_budget: int,
) -> str:
    if not results:
        return ""

    sorted_results = sorted(results, key=lambda r: r.get("relevance_score", 0), reverse=True)
    lines: List[str] = []
    chars = 0

    for idx, r in enumerate(sorted_results, 1):
        content = r.get("content", "").strip()
        if not content:
            continue
        source = r.get("source", "knowledge_pack")
        citation = r.get("citation", "")
        domain = r.get("domain", "")

        header_parts = [f"Source: {source}"]
        if domain:
            header_parts.append(f"Domain: {domain}")
        if citation:
            header_parts.append(f"Citation: {citation}")
        header = ", ".join(header_parts)

        entry = f"[KP-{idx}] {header}\n{content}"
        entry_len = len(entry)
        if chars + entry_len > char_budget:
            remaining = char_budget - chars - len(f"[KP-{idx}] {header}\n") - 20
            if remaining > 100:
                entry = f"[KP-{idx}] {header}\n{content[:remaining]}..."
                lines.append(entry)
            break
        lines.append(entry)
        chars += entry_len

    if not lines:
        return ""
    inner = "\n\n".join(lines)
    return f"<knowledge_pack_context>\n{inner}\n</knowledge_pack_context>"


def _build_xref_block(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""

    lines: List[str] = []
    for r in results:
        xtype = r.get("type", "info")
        content = r.get("content", "")
        if content:
            lines.append(f"[{xtype.upper()}] {content}")

    if not lines:
        return ""
    inner = "\n".join(lines)
    return f"<cross_reference_results>\n{inner}\n</cross_reference_results>"


def _build_spreadsheet_block(
    results: List[Dict[str, Any]],
    char_budget: int,
) -> str:
    if not results:
        return ""

    lines: List[str] = []
    chars = 0

    for idx, r in enumerate(sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True), 1):
        content = r.get("content", "").strip()
        source = r.get("source", "spreadsheet")
        if not content:
            continue
        entry = f"[TABLE-{idx}] Source: {source}\n{content}"
        entry_len = len(entry)
        if chars + entry_len > char_budget:
            break
        lines.append(entry)
        chars += entry_len

    if not lines:
        return ""
    inner = "\n\n".join(lines)
    return f"<spreadsheet_data>\n{inner}\n</spreadsheet_data>"


__all__ = ["assemble_context"]
