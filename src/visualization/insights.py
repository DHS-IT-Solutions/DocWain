"""Insight Engine — categorizes model output insights and routes them to visualizations."""

import re
from typing import Any

from src.utils.logging_utils import get_logger

logger = get_logger("insights")

# ---------------------------------------------------------------------------
# Category pattern definitions
# ---------------------------------------------------------------------------

_CATEGORY_PATTERNS: dict[str, list[str]] = {
    "pattern_recognition": [
        r"consistently",
        r"pattern",
        r"repeatedly",
        r"across \d+",
        r"always",
        r"every time",
        r"regularly",
    ],
    "anomaly_detection": [
        r"\d+x (lower|higher|more|less)",
        r"anomal",
        r"unusual",
        r"outlier",
        r"significantly (lower|higher|different)",
        r"deviates",
        r"unexpected",
    ],
    "trend_analysis": [
        r"increas",
        r"decreas",
        r"trend",
        r"growth",
        r"over (the)? (last|past|next)",
        r"quarter-over-quarter",
        r"year-over-year",
        r"rising",
        r"falling",
    ],
    "comparative_analysis": [
        r"compar",
        r"differ",
        r"version \d+",
        r"versus",
        r"unlike",
        r"in contrast",
        r"removes? the",
    ],
    "gap_analysis": [
        r"missing",
        r"gap",
        r"covers? \d+ of \d+",
        r"incomplete",
        r"absent",
        r"lacks?",
    ],
}

# Chart type for each category
_CATEGORY_CHART_MAP: dict[str, str] = {
    "pattern_recognition": "bar",
    "anomaly_detection": "bar",
    "trend_analysis": "line",
    "comparative_analysis": "bar",
    "gap_analysis": "pie",
}

_DEFAULT_CATEGORY = "pattern_recognition"


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def categorize_insight(text: str) -> dict[str, Any]:
    """Categorize an insight text string.

    Returns a dict with:
      - "category": best-matching category name (str)
      - "score": number of matching patterns for that category (int)
      - "text": the original input text (str)
    """
    text_lower = text.lower()
    scores: dict[str, int] = {}

    for category, patterns in _CATEGORY_PATTERNS.items():
        count = sum(1 for p in patterns if re.search(p, text_lower))
        scores[category] = count

    best_category = max(scores, key=lambda c: scores[c])
    best_score = scores[best_category]

    # If no pattern matched at all, fall back to default category with score 0
    if best_score == 0:
        best_category = _DEFAULT_CATEGORY

    logger.debug("categorize_insight scores=%s best=%s", scores, best_category)

    return {
        "category": best_category,
        "score": best_score,
        "text": text,
    }


def insight_to_visualization(category: str, data: Any, title: str = "") -> dict[str, Any]:
    """Map a category to a visualization specification.

    Returns a dict with:
      - "chart_type": chart type string (str)
      - "data": the provided data (Any)
      - "title": the provided title (str)
    """
    chart_type = _CATEGORY_CHART_MAP.get(category, "bar")

    return {
        "chart_type": chart_type,
        "data": data,
        "title": title,
    }


def classify_severity(category: str, confidence: float) -> str:
    """Classify severity of an insight based on category and confidence score.

    Rules (evaluated in order):
      1. anomaly_detection + confidence >= 0.8  -> "critical"
      2. anomaly_detection                       -> "warning"
      3. confidence >= 0.8                       -> "warning"
      4. else                                    -> "info"
    """
    if category == "anomaly_detection" and confidence >= 0.8:
        return "critical"
    if category == "anomaly_detection":
        return "warning"
    if confidence >= 0.8:
        return "warning"
    return "info"
