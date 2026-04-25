"""Parse LLM responses for typed insight passes.

The model is prompted to return JSON with shape:
  {"insights": [{"headline", "body", "evidence_doc_spans", "external_kb_refs",
                 "confidence", "severity"}]}

Malformed entries (missing required fields, non-numeric confidence) are
dropped silently. Completely malformed JSON raises ParseError.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


class ParseError(ValueError):
    pass


@dataclass
class ParsedInsight:
    headline: str
    body: str
    evidence_doc_spans: List[Dict[str, Any]]
    external_kb_refs: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    severity: str = "notice"


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_fence(text: str) -> str:
    text = text.strip()
    m = _FENCE.match(text)
    return m.group(1).strip() if m else text


def parse_typed_insight_response(text: str) -> List[ParsedInsight]:
    body = _strip_fence(text)
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ParseError("response root must be a JSON object")
    items = data.get("insights") or []
    if not isinstance(items, list):
        return []
    parsed: List[ParsedInsight] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        headline = str(raw.get("headline") or "").strip()
        body_text = str(raw.get("body") or "").strip()
        spans = raw.get("evidence_doc_spans") or []
        if not headline or not body_text or not isinstance(spans, list) or not spans:
            continue
        try:
            confidence = float(raw.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        severity = str(raw.get("severity") or "notice").lower()
        if severity not in ("info", "notice", "warn", "critical"):
            severity = "notice"
        parsed.append(ParsedInsight(
            headline=headline,
            body=body_text,
            evidence_doc_spans=[
                {
                    "document_id": str(s.get("document_id") or ""),
                    "page": int(s.get("page") or 0),
                    "char_start": int(s.get("char_start") or 0),
                    "char_end": int(s.get("char_end") or 0),
                    "quote": str(s.get("quote") or ""),
                }
                for s in spans
                if isinstance(s, dict)
            ],
            external_kb_refs=[
                {
                    "kb_id": str(r.get("kb_id") or ""),
                    "ref": str(r.get("ref") or ""),
                    "label": str(r.get("label") or ""),
                }
                for r in (raw.get("external_kb_refs") or [])
                if isinstance(r, dict)
            ],
            confidence=confidence,
            severity=severity,
        ))
    return parsed
