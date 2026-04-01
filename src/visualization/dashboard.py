"""Multi-document dashboard composer.

Composes a mini-dashboard with tables and charts when a query spans
multiple documents. The V2 model decides composition based on data shape.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def compose_dashboard(
    data: dict,
    query: str,
    *,
    max_sections: int = 5,
) -> dict:
    """Compose a multi-document dashboard from structured data.

    Parameters
    ----------
    data:
        Optional keys:
        - ``"documents"`` — list of document names (str)
        - ``"values"``    — list of numeric values
        - ``"dates"``     — list of date strings
    query:
        The user query that triggered this dashboard.
    max_sections:
        Maximum number of sections to include in the result.

    Returns
    -------
    dict
        ``{"sections": [...], "query": str, "document_count": int}``
        If *data* is empty, returns ``{"sections": [], "query": query}``.
    """
    if not data:
        return {"sections": [], "query": query}

    documents: List[str] = data.get("documents") or []
    values: List[Any] = data.get("values") or []
    dates: List[str] = data.get("dates") or []

    sections: List[dict] = []

    # ── Section 1: Summary table ─────────────────────────────────────────────
    if documents and values:
        rows = []
        for i, doc in enumerate(documents):
            row: dict = {
                "document": doc,
                "value": values[i] if i < len(values) else None,
            }
            if dates and i < len(dates):
                row["date"] = dates[i]
            rows.append(row)

        sections.append({
            "type": "table",
            "title": "Summary",
            "data": rows,
        })

    # ── Section 2: Value bar chart ───────────────────────────────────────────
    if values and len(values) >= 2:
        labels = documents if documents else [str(i) for i in range(len(values))]
        sections.append({
            "type": "chart",
            "chart_type": "bar",
            "title": "Values",
            "x": labels[: len(values)],
            "y": list(values),
        })

    # ── Section 3: Timeline line chart ───────────────────────────────────────
    if dates and values and len(values) >= 2 and len(dates) >= 2:
        sections.append({
            "type": "chart",
            "chart_type": "line",
            "title": "Timeline",
            "x": list(dates),
            "y": list(values[: len(dates)]),
        })

    # Enforce max_sections cap
    sections = sections[:max_sections]

    return {
        "sections": sections,
        "query": query,
        "document_count": len(documents),
    }
