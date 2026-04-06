"""Analytical Reasoner — structured analysis with data-grounded recommendations.

Specialized reasoning path for ANALYTICAL queries that produces structured
output with cited data points, pattern analysis, risk flags, and actionable
recommendations.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain-specific system prompts
# ---------------------------------------------------------------------------

DOMAIN_PROMPTS = {
    "financial": "You are a financial analyst. Focus on trends, ratios, variances, and actionable recommendations.",
    "healthcare": "You are a clinical analyst. Focus on protocol comparison, outcome tracking, and compliance gaps.",
    "legal": "You are a legal analyst. Focus on clause comparison, obligation tracking, and risk exposure.",
    "hr": "You are an HR analyst. Focus on compensation benchmarking, skill gaps, and attrition patterns.",
    "general": "You are a document analyst. Focus on patterns, anomalies, and actionable insights.",
}

# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------


@dataclass
class AnalyticalResponse:
    """Structured analytical response with grounded observations and recommendations."""

    key_data_points: List[Dict[str, Any]]
    analysis: str
    risks: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    confidence_level: str
    data_sufficiency: str
    raw_answer: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _format_source_chunks(source_chunks: List[Dict[str, Any]]) -> str:
    """Format source chunks with citation labels."""
    parts = []
    for i, chunk in enumerate(source_chunks, 1):
        text = (
            chunk.get("text")
            or chunk.get("canonical_text")
            or chunk.get("content")
            or ""
        )
        source_name = chunk.get("source_name", f"Source {i}")
        page = chunk.get("page")
        label = f"[{source_name}"
        if page is not None:
            label += f", p.{page}"
        label += "]"
        parts.append(f"[{i}] {label}:\n{text}")
    return "\n\n".join(parts)


def _format_enrichment(enrichment: Dict[str, Any]) -> str:
    """Format pre-computed intelligence for the prompt."""
    if not enrichment:
        return "None available."

    sections = []
    for key, value in enrichment.items():
        if value is None:
            continue
        if isinstance(value, dict):
            formatted = json.dumps(value, indent=2, default=str)
        elif isinstance(value, list):
            formatted = json.dumps(value, indent=2, default=str)
        else:
            formatted = str(value)
        sections.append(f"### {key.replace('_', ' ').title()}\n{formatted}")

    return "\n\n".join(sections) if sections else "None available."


def build_analytical_prompt(
    query: str,
    source_chunks: List[Dict[str, Any]],
    enrichment: Dict[str, Any] = None,
    domain: str = "general",
) -> str:
    """Build the analytical reasoning prompt.

    The prompt enforces structured output:
    1. Key Data Points (cited)
    2. Analysis (reasoned)
    3. Risks (flagged)
    4. Recommendations (advisory)

    Enrichment is injected as CONTEXT, not as answer source.
    """
    domain_prompt = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["general"])
    formatted_sources = _format_source_chunks(source_chunks)
    formatted_enrichment = _format_enrichment(enrichment)

    prompt = f"""SYSTEM: {domain_prompt}

SOURCE DATA:
{formatted_sources}

COMPUTED CONTEXT (use to inform reasoning, do NOT cite directly):
{formatted_enrichment}

USER QUERY: {query}

INSTRUCTIONS:
Analyze the source data and provide:

1. KEY DATA POINTS: List specific facts from the source data. Each must cite its source.
2. ANALYSIS: What patterns, trends, or insights does the data show?
3. RISKS: What anomalies, concerning trends, or gaps should be flagged? Rate each HIGH/MEDIUM/LOW.
4. RECOMMENDATIONS: What actions should be considered? Rate confidence HIGH/MEDIUM/LOW.

Rules:
- Every observation must cite source data
- Clearly separate facts ("the data shows") from inference ("this suggests")
- If fewer than 3 data points exist for a trend, state "insufficient data"
- Frame recommendations as "the data suggests" not "you should"
- If data contradicts itself, surface both sides

Respond in this exact format:
## Key Data Points
- [fact] (Source: [reference])

## Analysis
[analysis text]

## Risks
- [risk] — Severity: [HIGH/MEDIUM/LOW]

## Recommendations
- [recommendation] — Confidence: [HIGH/MEDIUM/LOW]"""

    return prompt


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"##\s*(Key Data Points|Analysis|Risks|Recommendations)\s*\n",
    re.IGNORECASE,
)


def _parse_list_items(text: str) -> List[str]:
    """Extract bullet-point items from a text block."""
    items = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("- ") or line.startswith("* "):
            items.append(line[2:].strip())
        elif line.startswith("•"):
            items.append(line[1:].strip())
    return items


def _parse_data_points(text: str) -> List[Dict[str, Any]]:
    """Parse key data points with source citations."""
    items = _parse_list_items(text)
    results = []
    source_re = re.compile(r"\(Source:\s*(.+?)\)\s*$")
    for item in items:
        m = source_re.search(item)
        if m:
            fact = source_re.sub("", item).strip()
            source = m.group(1).strip()
        else:
            fact = item
            source = "unspecified"
        results.append({"fact": fact, "source": source})
    return results


def _parse_risks(text: str) -> List[Dict[str, Any]]:
    """Parse risks with severity ratings."""
    items = _parse_list_items(text)
    results = []
    severity_re = re.compile(r"—\s*Severity:\s*(HIGH|MEDIUM|LOW)\s*$", re.IGNORECASE)
    for item in items:
        m = severity_re.search(item)
        if m:
            risk = severity_re.sub("", item).strip()
            severity = m.group(1).upper()
        else:
            risk = item
            severity = "MEDIUM"
        results.append({"risk": risk, "severity": severity})
    return results


def _parse_recommendations(text: str) -> List[Dict[str, Any]]:
    """Parse recommendations with confidence ratings."""
    items = _parse_list_items(text)
    results = []
    conf_re = re.compile(r"—\s*Confidence:\s*(HIGH|MEDIUM|LOW)\s*$", re.IGNORECASE)
    for item in items:
        m = conf_re.search(item)
        if m:
            rec = conf_re.sub("", item).strip()
            confidence = m.group(1).upper()
        else:
            rec = item
            confidence = "MEDIUM"
        results.append({"recommendation": rec, "confidence": confidence})
    return results


def _determine_confidence(
    data_points: List[Dict[str, Any]],
    risks: List[Dict[str, Any]],
) -> str:
    """Determine overall confidence based on data richness."""
    if len(data_points) >= 5:
        return "HIGH"
    elif len(data_points) >= 2:
        return "MEDIUM"
    return "LOW"


def _determine_sufficiency(data_points: List[Dict[str, Any]]) -> str:
    """Determine data sufficiency."""
    if len(data_points) >= 5:
        return "sufficient"
    elif len(data_points) >= 2:
        return "partial"
    return "insufficient"


def parse_analytical_response(raw: str) -> AnalyticalResponse:
    """Parse raw LLM output into an AnalyticalResponse.

    If parsing fails, returns raw text with empty structured fields.
    """
    sections: Dict[str, str] = {}
    splits = _SECTION_RE.split(raw)

    # splits alternates: [preamble, header1, content1, header2, content2, ...]
    if len(splits) >= 3:
        i = 1
        while i < len(splits) - 1:
            header = splits[i].strip().lower()
            content = splits[i + 1]
            sections[header] = content
            i += 2

    if not sections:
        return AnalyticalResponse(
            key_data_points=[],
            analysis="",
            risks=[],
            recommendations=[],
            confidence_level="LOW",
            data_sufficiency="insufficient",
            raw_answer=raw,
        )

    data_points = _parse_data_points(sections.get("key data points", ""))
    analysis = sections.get("analysis", "").strip()
    risks = _parse_risks(sections.get("risks", ""))
    recommendations = _parse_recommendations(sections.get("recommendations", ""))

    return AnalyticalResponse(
        key_data_points=data_points,
        analysis=analysis,
        risks=risks,
        recommendations=recommendations,
        confidence_level=_determine_confidence(data_points, risks),
        data_sufficiency=_determine_sufficiency(data_points),
        raw_answer=raw,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reason_analytical(
    query: str,
    source_chunks: List[Dict[str, Any]],
    enrichment: Dict[str, Any] = None,
    domain: str = "general",
    llm_fn=None,
) -> AnalyticalResponse:
    """Generate an analytical response with structured output.

    Args:
        query: The user's analytical question.
        source_chunks: Retrieved evidence chunks with text and metadata.
        enrichment: Pre-computed document intelligence (temporal, numerical, etc.).
        domain: Domain hint for prompt specialization.
        llm_fn: Callable that takes a prompt string and returns a response string.
                 If None, uses the default LLM gateway.

    Returns:
        AnalyticalResponse with parsed structured fields.
    """
    prompt = build_analytical_prompt(query, source_chunks, enrichment, domain)

    if llm_fn is None:
        from src.llm.gateway import get_llm_gateway
        gw = get_llm_gateway()
        raw = gw.generate(prompt, temperature=0.3, max_tokens=4096)
    else:
        raw = llm_fn(prompt)

    logger.info(
        "[ANALYTICAL_REASONER] query=%r domain=%s chunks=%d raw_len=%d",
        query[:80], domain, len(source_chunks), len(raw),
    )

    return parse_analytical_response(raw)


def reason_analytical_stream(
    query: str,
    source_chunks: List[Dict[str, Any]],
    enrichment: Dict[str, Any] = None,
    domain: str = "general",
    llm_fn=None,
) -> Generator[str, None, None]:
    """Streaming version -- yields chunks, final chunk includes structured metadata.

    The final yielded chunk is a JSON object with the parsed analytical structure,
    prefixed with '\\n---ANALYTICAL_META---\\n' to allow the caller to detect it.
    """
    prompt = build_analytical_prompt(query, source_chunks, enrichment, domain)

    accumulated = []

    if llm_fn is None:
        from src.llm.gateway import get_llm_gateway
        gw = get_llm_gateway()
        stream = gw.generate_stream(prompt, temperature=0.3, max_tokens=4096)
    else:
        # llm_fn for streaming should return an iterable of chunks
        result = llm_fn(prompt)
        if isinstance(result, str):
            stream = [result]
        else:
            stream = result

    for chunk in stream:
        accumulated.append(chunk)
        yield chunk

    # After streaming completes, parse the full response and yield metadata
    full_text = "".join(accumulated)
    parsed = parse_analytical_response(full_text)
    meta = json.dumps(parsed.to_dict(), default=str)
    yield f"\n---ANALYTICAL_META---\n{meta}"
