"""Visualization spec generator.

Called at insight-write time + at profile-list time. Produces JSON specs
the frontend can render directly. v1 ships timeline, comparison_table,
trend_chart per spec Section 5.3.
"""
from __future__ import annotations

from typing import Any, Dict, List

from src.intelligence.insights.schema import Insight


def generate_visualizations_for_insight(insight: Insight) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if insight.insight_type == "trend":
        out.append({
            "viz_id": "trend_chart",
            "type": "trend_chart",
            "source_insight_ids": [insight.insight_id],
            "data": {
                "headline": insight.headline,
                "domain": insight.domain,
                "refreshed_at": insight.refreshed_at,
            },
        })
    if insight.insight_type == "comparison":
        out.append({
            "viz_id": "comparison_table",
            "type": "comparison_table",
            "source_insight_ids": [insight.insight_id],
            "data": {
                "headline": insight.headline,
                "documents": insight.document_ids,
            },
        })
    return out


def generate_visualizations_for_profile(insights: List[Insight]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    events = []
    for ins in sorted(insights, key=lambda i: i.refreshed_at):
        events.append({
            "at": ins.refreshed_at,
            "headline": ins.headline,
            "insight_type": ins.insight_type,
            "severity": ins.severity,
        })
    if events:
        out.append({
            "viz_id": "timeline",
            "type": "timeline",
            "source_insight_ids": [i.insight_id for i in insights],
            "data": {"events": events},
        })
    for ins in insights:
        out.extend(generate_visualizations_for_insight(ins))
    return out
