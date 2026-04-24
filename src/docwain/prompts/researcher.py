"""Researcher Agent prompt for DocWain.

Called by `src.tasks.researcher.run_researcher_agent` during the training stage.
Produces domain-aware insights from an extracted document. Writes to Qdrant
payload + Neo4j Insight nodes.

Spec: 2026-04-24-unified-docwain-engineering-layer-design.md §5.6
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


RESEARCHER_SYSTEM_PROMPT = (
    "You are DocWain's Researcher Agent. Given an extracted document, produce "
    "structured domain-aware insights — the kind a domain expert would surface "
    "proactively without being asked. You will be called during ingestion, not "
    "at query time; your output is persisted and served to users later.\n\n"
    "Output ONLY valid JSON (no prose, no markdown fences):\n"
    "{\n"
    '  "summary": string (2-4 sentences),\n'
    '  "key_facts": [string, ...] (5-10 factual bullets verbatim from the doc),\n'
    '  "entities": [ { "text": string, "type": string } ],\n'
    '  "recommendations": [string, ...] (actionable suggestions based on doc content),\n'
    '  "anomalies": [string, ...] (anything unusual, inconsistent, or risky),\n'
    '  "questions_to_ask": [string, ...] (questions a user might want to ask about this doc),\n'
    '  "confidence": number 0..1\n'
    "}\n\n"
    "Rules:\n"
    "- Ground every key_fact and anomaly in text explicitly present in the document.\n"
    "- Do not fabricate numbers, names, or dates.\n"
    "- recommendations and questions_to_ask may be inferential but must be "
    "  supported by what the document actually contains.\n"
    "- Keep lists concise — quality over quantity."
)


@dataclass
class ResearcherInsights:
    summary: str = ""
    key_facts: List[str] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)
    questions_to_ask: List[str] = field(default_factory=list)
    confidence: float = 0.0


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    m = re.match(r"```(?:json)?\s*(.*?)```", t, flags=re.DOTALL)
    return m.group(1).strip() if m else t


def _first_json_object(text: str) -> str:
    t = _strip_code_fence(text)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        return t
    return t[start : end + 1]


def _str_list(value) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(x) for x in value if isinstance(x, (str, int, float))]


def parse_researcher_response(text: str) -> ResearcherInsights:
    try:
        data = json.loads(_first_json_object(text))
    except Exception:
        return ResearcherInsights()
    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    entities = data.get("entities") or []
    if not isinstance(entities, list):
        entities = []
    return ResearcherInsights(
        summary=str(data.get("summary") or ""),
        key_facts=_str_list(data.get("key_facts")),
        entities=entities,
        recommendations=_str_list(data.get("recommendations")),
        anomalies=_str_list(data.get("anomalies")),
        questions_to_ask=_str_list(data.get("questions_to_ask")),
        confidence=confidence,
    )


def build_user_prompt(*, document_text: str, doc_type_hint: str = "generic",
                      max_chars: int = 16000) -> str:
    truncated = document_text[:max_chars]
    return (
        f"Document type hint: {doc_type_hint}\n\n"
        f"Document text:\n\n{truncated}\n\n"
        "Return the insights JSON."
    )
