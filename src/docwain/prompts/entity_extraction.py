"""Entity + relation extraction prompt for DocWain.

Called by `src.tasks.kg._canonical_to_graph_payload` before building the
GraphIngestPayload. Empty-on-error fallback keeps KG ingestion resilient.

Spec: 2026-04-24-unified-docwain-engineering-layer-design.md §5.3
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


ENTITY_EXTRACTION_SYSTEM_PROMPT = (
    "You extract named entities and relationships from document text. Given the "
    "concatenated text of a document, return a JSON object describing the "
    "entities present and their relationships.\n\n"
    "Entity types to use: PERSON, ORGANIZATION, LOCATION, DATE, MONEY, PRODUCT, "
    "EVENT, DOCUMENT_REF, OTHER. Set type=OTHER for anything that doesn't fit.\n\n"
    "Output ONLY valid JSON (no prose, no markdown fences):\n"
    "{\n"
    '  "entities": [\n'
    '    { "text": string, "type": string, "confidence": number 0..1 }\n'
    "  ],\n"
    '  "relationships": [\n'
    '    { "source": string, "target": string, "type": string }\n'
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- Extract only entities that are explicitly mentioned in the provided text.\n"
    "- Do not infer or synthesize entities not in the text.\n"
    "- For relationships, `source` and `target` must match entity `text` values "
    "exactly.\n"
    "- Confidence reflects how unambiguous the mention is (higher = clearer)."
)


@dataclass
class ExtractedEntities:
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    m = re.match(r"```(?:json)?\s*(.*?)```", t, flags=re.DOTALL)
    return m.group(1).strip() if m else t


def _first_json_object(text: str) -> str:
    t = _strip_code_fence(text)
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return t
    return t[start : end + 1]


def parse_entity_response(text: str) -> ExtractedEntities:
    """Best-effort parse; empty on failure."""
    try:
        data = json.loads(_first_json_object(text))
    except Exception:
        return ExtractedEntities()
    entities = data.get("entities") or []
    relationships = data.get("relationships") or []
    if not isinstance(entities, list):
        entities = []
    if not isinstance(relationships, list):
        relationships = []
    return ExtractedEntities(entities=entities, relationships=relationships)


def build_user_prompt(*, document_text: str, max_chars: int = 16000) -> str:
    """Build the user-prompt payload for the entity extractor call."""
    truncated = document_text[:max_chars]
    return f"Document text:\n\n{truncated}\n\nReturn the entity + relationship JSON."
