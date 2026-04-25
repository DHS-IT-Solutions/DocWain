"""Helpers for /api/ask proactive insight injection.

Lookup-only — no LLM calls, no network beyond the existing Mongo index
read. 50ms hard budget per spec Section 13.2.

Per OQ4: always-on once INSIGHTS_PROACTIVE_INJECTION is enabled. Severity
filtering ('notice'+) is a quality guard, not an opt-out.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Set

INJECTION_BUDGET_MS = 50.0
DEFAULT_TOP_N = 5
_SEVERITY_ORDER = {"info": 0, "notice": 1, "warn": 2, "critical": 3}


def select_insights_for_query(
    *,
    query: str,
    profile_insights: List[Dict[str, Any]],
    query_entities: Set[str],
    top_n: int = DEFAULT_TOP_N,
) -> List[Dict[str, Any]]:
    deadline = time.perf_counter() + (INJECTION_BUDGET_MS / 1000.0)
    selected: List[Dict[str, Any]] = []
    for row in profile_insights:
        if time.perf_counter() > deadline:
            break
        sev = row.get("severity", "info")
        if _SEVERITY_ORDER.get(sev, 0) < _SEVERITY_ORDER["notice"]:
            continue
        tags = set(row.get("tags") or [])
        relevance_score = _SEVERITY_ORDER.get(sev, 0)
        if query_entities and not (tags & query_entities):
            relevance_score -= 1
        row_with_score = dict(row)
        row_with_score["__relevance"] = relevance_score
        selected.append(row_with_score)
    selected.sort(key=lambda r: -r["__relevance"])
    return [{k: v for k, v in r.items() if k != "__relevance"} for r in selected[:top_n]]


def format_related_findings(insights: Iterable[Dict[str, Any]]) -> str:
    rows = list(insights)
    if not rows:
        return ""
    lines = ["", "Related findings:"]
    for r in rows:
        sev = r.get("severity", "")
        marker = "!" if sev in ("warn", "critical") else "•"
        lines.append(f"  {marker} {r.get('headline', '')}")
    return "\n".join(lines)
