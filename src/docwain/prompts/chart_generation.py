"""Chart / DOCWAIN_VIZ generation support.

DocWain already emits `<!--DOCWAIN_VIZ ... -->` blocks from the existing
Reasoner system prompt. This module adds:

- A dedicated system-prompt fragment that can be appended to the Reasoner
  prompt when a query warrants a chart (routed by `should_emit_chart`).
- A canonical parser for `DOCWAIN_VIZ` blocks (used by the frontend and by
  tests).
- A heuristic query classifier that determines when a chart is appropriate.

Spec: 2026-04-24-unified-docwain-engineering-layer-design.md §5.4
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


CHART_GENERATION_SYSTEM_PROMPT = (
    "When the user's question asks for a comparison, trend, distribution, or "
    "aggregation across multiple values that can be visualized, emit a "
    "DOCWAIN_VIZ block after your natural-language answer. Format:\n\n"
    "<!--DOCWAIN_VIZ\n"
    "{\n"
    '  "chart_type": "bar" | "line" | "pie" | "horizontal_bar" | "table",\n'
    '  "title": string,\n'
    '  "labels": [string, ...],\n'
    '  "values": [number, ...] or [[number, ...], ...] for multi-series,\n'
    '  "unit": string (e.g., "USD", "%", "count")\n'
    "}\n"
    "-->\n\n"
    "Rules:\n"
    "- Emit DOCWAIN_VIZ only when the data is meaningfully visualizable. Do "
    "  not emit for single-value facts or yes/no answers.\n"
    "- Use only values grounded in the retrieved documents. Do not fabricate.\n"
    "- If the chart would have fewer than 2 data points, omit it."
)


_CHART_KEYWORDS = (
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bchart\b",
    r"\bgraph\b",
    r"\bplot\b",
    r"\btrend\b",
    r"\bover time\b",
    r"\bmonthly\b",
    r"\byearly\b",
    r"\bquarterly\b",
    r"\bdistribution\b",
    r"\bbreakdown\b",
    r"\bversus\b",
    r"\bvs\.?\b",
)

_CHART_RE = re.compile("|".join(_CHART_KEYWORDS), re.IGNORECASE)


def should_emit_chart(query: str) -> bool:
    """Heuristic: does this query warrant a chart / DOCWAIN_VIZ in the response?"""
    if not query:
        return False
    return bool(_CHART_RE.search(query))


_VIZ_BLOCK_RE = re.compile(r"<!--\s*DOCWAIN_VIZ\s*(.+?)\s*-->", re.DOTALL)


def extract_viz_block(response_text: str) -> Optional[Dict[str, Any]]:
    """Extract a DOCWAIN_VIZ JSON payload from an HTML comment block. None on failure."""
    if not response_text:
        return None
    match = _VIZ_BLOCK_RE.search(response_text)
    if not match:
        return None
    payload_text = match.group(1).strip()
    try:
        data = json.loads(payload_text)
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None
