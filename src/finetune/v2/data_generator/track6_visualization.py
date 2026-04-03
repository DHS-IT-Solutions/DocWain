"""Track 6 — Visualization Intelligence data generator for DocWain V2+ SFT/DPO.

Generates 2000 training examples across seven visualization categories:
  - Single-series bar/line/pie       (400)
  - Multi-series comparison          (350)
  - Auto-detect triggers             (300)
  - Explicit request handling        (250)
  - No-chart negatives               (400)
  - Annotation intelligence          (150)
  - Chart type selection reasoning   (150)

Output uses ``<response>`` + ``<chart_spec>`` blocks. For no-chart negatives,
only the ``<response>`` block is emitted (no ``<chart_spec>``).

Produces both SFT and DPO preference pairs. DPO logic:
  - Positive cases: rejected = missing chart, chosen = includes chart_spec
  - Negative cases: rejected = unnecessary chart, chosen = no chart
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from src.finetune.v2.data_generator.base import (
    DOMAINS,
    DOC_TYPES,
    JSONLWriter,
    format_dpo_example,
    format_sft_example,
)

# ---------------------------------------------------------------------------
# Helpers & constants
# ---------------------------------------------------------------------------

_CURRENCIES = ["$", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "INR"]
_COMPANY_NAMES = [
    "Acme Corp", "Globex Industries", "Initech Solutions", "Umbrella LLC",
    "Soylent Corp", "Wonka Enterprises", "Stark Industries", "Wayne Enterprises",
    "Cyberdyne Systems", "Tyrell Corporation", "Weyland-Yutani", "Oscorp",
    "LexCorp", "Massive Dynamic", "Pied Piper", "Hooli",
]
_PERSON_NAMES = [
    "Alice Johnson", "Bob Martinez", "Carol Singh", "David Chen",
    "Emily Watson", "Frank Osei", "Grace Kim", "Hector Rossi",
    "Irene Muller", "James Okafor", "Karen Tanaka", "Liam Petrov",
]
_DEPARTMENTS = [
    "Engineering", "Finance", "Human Resources", "Legal",
    "Marketing", "Operations", "Sales", "Research & Development",
    "Compliance", "Procurement", "IT", "Customer Success",
]
_PRODUCT_NAMES = [
    "Widget A", "Gadget Pro", "Module X-100", "Sensor Suite",
    "Platform License", "Cloud Tier 2", "Enterprise Connector",
    "Data Vault", "Analytics Pack", "Security Module",
]
_METRICS = [
    "Revenue", "Expenses", "Profit", "Headcount", "Customer Count",
    "Churn Rate", "NPS Score", "Processing Time", "Error Rate",
    "Throughput", "Cost per Unit", "Market Share",
]
_CHART_TYPES = ["bar", "line", "pie", "grouped_bar", "stacked_bar", "area"]
_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_QUARTERS = ["Q1", "Q2", "Q3", "Q4"]


def _pick(lst: list, rng: random.Random) -> Any:
    return rng.choice(lst)


def _rand_vals(rng: random.Random, n: int, lo: int = 50, hi: int = 500) -> List[int]:
    return [rng.randint(lo, hi) for _ in range(n)]


def _rand_amount(rng: random.Random, lo: float = 100.0, hi: float = 99999.0) -> str:
    return f"{rng.uniform(lo, hi):,.2f}"


def _subs(rng: random.Random) -> Dict[str, Any]:
    """Build a standard substitution dict."""
    vals4 = _rand_vals(rng, 4, 80, 300)
    vals4b = _rand_vals(rng, 4, 60, 280)
    vals3 = _rand_vals(rng, 3, 100, 500)
    return {
        "domain": _pick(DOMAINS, rng),
        "doc_type": _pick(DOC_TYPES, rng),
        "company": _pick(_COMPANY_NAMES, rng),
        "company2": _pick(_COMPANY_NAMES, rng),
        "person": _pick(_PERSON_NAMES, rng),
        "dept": _pick(_DEPARTMENTS, rng),
        "dept2": _pick(_DEPARTMENTS, rng),
        "dept3": _pick(_DEPARTMENTS, rng),
        "product": _pick(_PRODUCT_NAMES, rng),
        "product2": _pick(_PRODUCT_NAMES, rng),
        "metric": _pick(_METRICS, rng),
        "metric2": _pick(_METRICS, rng),
        "currency": _pick(_CURRENCIES, rng),
        "amount": _rand_amount(rng),
        "amount2": _rand_amount(rng),
        "year": str(rng.randint(2022, 2026)),
        "year2": str(rng.randint(2022, 2026)),
        "quarter": _pick(_QUARTERS, rng),
        "month": _pick(_MONTHS, rng),
        "pct": f"{rng.uniform(1, 45):.1f}",
        "pct2": f"{rng.uniform(1, 45):.1f}",
        # Pre-built value arrays for chart_spec
        "v1": str(vals4[0]), "v2": str(vals4[1]),
        "v3": str(vals4[2]), "v4": str(vals4[3]),
        "v1b": str(vals4b[0]), "v2b": str(vals4b[1]),
        "v3b": str(vals4b[2]), "v4b": str(vals4b[3]),
        "s1": str(vals3[0]), "s2": str(vals3[1]), "s3": str(vals3[2]),
        "page": str(rng.randint(1, 50)),
    }


def _expand(
    templates: List[Dict[str, str]],
    count: int,
    rng: random.Random,
    *,
    mode: str = "sft",
) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    idx = 0
    while len(results) < count:
        tpl = templates[idx % len(templates)]
        s = _subs(rng)
        try:
            query = tpl["query"].format(**s)
            if mode == "sft":
                reasoning = tpl["reasoning"].format(**s)
                answer = tpl["answer"].format(**s)
                results.append(format_sft_example(query, reasoning, answer))
            else:
                chosen_r = tpl["chosen_reasoning"].format(**s)
                chosen_a = tpl["chosen_answer"].format(**s)
                rejected_r = tpl["rejected_reasoning"].format(**s)
                rejected_a = tpl["rejected_answer"].format(**s)
                results.append(format_dpo_example(
                    query, chosen_r, chosen_a, rejected_r, rejected_a,
                ))
        except (KeyError, IndexError):
            idx += 1
            continue
        idx += 1
    return results


# ===================================================================
# 1. SINGLE-SERIES BAR/LINE/PIE (400)
# ===================================================================

_SINGLE_SERIES_SFT: List[Dict[str, str]] = [
    {
        "query": (
            "From {company}'s {domain} {doc_type} (p.{page}):\n"
            "| Quarter | {metric} |\n"
            "| Q1 | {currency}{v1} |\n"
            "| Q2 | {currency}{v2} |\n"
            "| Q3 | {currency}{v3} |\n"
            "| Q4 | {currency}{v4} |\n\n"
            "Analyze the quarterly {metric} data."
        ),
        "reasoning": (
            "Step 1: Data is quarterly — 4 time-ordered values.\n"
            "Step 2: Values: Q1={v1}, Q2={v2}, Q3={v3}, Q4={v4}.\n"
            "Step 3: This is time-series data — a bar or line chart is appropriate.\n"
            "Step 4: Bar chart for discrete quarterly comparison; line for trend.\n"
            "Step 5: I'll use a bar chart since quarters are discrete periods.\n"
            "Confidence: High — clear tabular data with chartable pattern."
        ),
        "answer": (
            "<response>\n"
            "The quarterly {metric} for {company} shows the following pattern: "
            "Q1 at {currency}{v1}, Q2 at {currency}{v2}, Q3 at {currency}{v3}, "
            "and Q4 at {currency}{v4}. The data reveals the trend across the fiscal year.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "bar", '
            '"title": "{metric} by Quarter", '
            '"x": {{"label": "Quarter", "values": ["Q1", "Q2", "Q3", "Q4"]}}, '
            '"series": [{{"name": "FY{year}", "values": [{v1}, {v2}, {v3}, {v4}]}}], '
            '"unit": "{currency}", '
            '"source": "{doc_type} p.{page}"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
    {
        "query": (
            "{company}'s {domain} report shows department budget allocation:\n"
            "| Department | Budget ({currency}) |\n"
            "| {dept} | {v1} |\n"
            "| {dept2} | {v2} |\n"
            "| {dept3} | {v3} |\n\n"
            "Show the budget distribution."
        ),
        "reasoning": (
            "Step 1: Three departments with budget values — part-to-whole data.\n"
            "Step 2: Budget distribution = proportional share — pie chart is ideal.\n"
            "Step 3: Values: {dept}={v1}, {dept2}={v2}, {dept3}={v3}.\n"
            "Step 4: Pie chart shows relative proportion of each department.\n"
            "Confidence: High — part-to-whole relationship suits pie chart."
        ),
        "answer": (
            "<response>\n"
            "The budget allocation across departments is: {dept} at {currency}{v1}, "
            "{dept2} at {currency}{v2}, and {dept3} at {currency}{v3}. The distribution "
            "shows the relative investment in each area.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "pie", '
            '"title": "Budget Distribution by Department", '
            '"x": {{"label": "Department", "values": ["{dept}", "{dept2}", "{dept3}"]}}, '
            '"series": [{{"name": "Budget", "values": [{v1}, {v2}, {v3}]}}], '
            '"unit": "{currency}", '
            '"source": "{domain} report"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
    {
        "query": (
            "Monthly {metric} trend for {company} ({year}):\n"
            "Jan: {v1}, Feb: {v2}, Mar: {v3}, Apr: {v4}\n\n"
            "What does the trend look like?"
        ),
        "reasoning": (
            "Step 1: Monthly time-series data — 4 consecutive months.\n"
            "Step 2: 'Trend' keyword suggests line chart to show progression.\n"
            "Step 3: Values: Jan={v1}, Feb={v2}, Mar={v3}, Apr={v4}.\n"
            "Step 4: Line chart best shows the directional movement over time.\n"
            "Confidence: High — time-series trend visualization."
        ),
        "answer": (
            "<response>\n"
            "The {metric} trend for {company} from January through April {year} shows: "
            "{v1} (Jan), {v2} (Feb), {v3} (Mar), {v4} (Apr). The trajectory indicates "
            "the direction of change over these four months.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "line", '
            '"title": "{metric} Trend (Jan-Apr {year})", '
            '"x": {{"label": "Month", "values": ["Jan", "Feb", "Mar", "Apr"]}}, '
            '"series": [{{"name": "{metric}", "values": [{v1}, {v2}, {v3}, {v4}]}}], '
            '"unit": "", '
            '"source": "{company} {year} data"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
    {
        "query": (
            "{company}'s {domain} {doc_type} — Customer Segmentation:\n"
            "| Segment | Count |\n"
            "| Enterprise | {v1} |\n"
            "| Mid-Market | {v2} |\n"
            "| SMB | {v3} |\n\n"
            "Visualize the customer breakdown."
        ),
        "reasoning": (
            "Step 1: Three customer segments with counts — categorical data.\n"
            "Step 2: 'Breakdown' suggests proportional view — pie chart.\n"
            "Step 3: Alternatively, bar chart for absolute comparison.\n"
            "Step 4: Pie chart is most natural for segment breakdowns.\n"
            "Confidence: High — standard segmentation visualization."
        ),
        "answer": (
            "<response>\n"
            "Customer segmentation for {company}: Enterprise ({v1}), Mid-Market ({v2}), "
            "SMB ({v3}). The breakdown shows the distribution across market segments.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "pie", '
            '"title": "Customer Segmentation", '
            '"x": {{"label": "Segment", "values": ["Enterprise", "Mid-Market", "SMB"]}}, '
            '"series": [{{"name": "Count", "values": [{v1}, {v2}, {v3}]}}], '
            '"unit": "customers", '
            '"source": "{doc_type}"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
]

_SINGLE_SERIES_DPO: List[Dict[str, str]] = [
    {
        "query": (
            "{company}'s quarterly sales:\n"
            "Q1: {currency}{v1}, Q2: {currency}{v2}, Q3: {currency}{v3}, Q4: {currency}{v4}\n\n"
            "Analyze the sales performance."
        ),
        "chosen_reasoning": (
            "Step 1: Four quarterly values — chartable time-series data.\n"
            "Step 2: Bar chart for discrete quarter comparison.\n"
            "Step 3: Include analysis of the trend in the response text.\n"
            "Confidence: High."
        ),
        "chosen_answer": (
            "<response>\n"
            "{company}'s quarterly sales show: Q1 ({currency}{v1}), Q2 ({currency}{v2}), "
            "Q3 ({currency}{v3}), Q4 ({currency}{v4}).\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "bar", '
            '"title": "Quarterly Sales", '
            '"x": {{"label": "Quarter", "values": ["Q1", "Q2", "Q3", "Q4"]}}, '
            '"series": [{{"name": "Sales", "values": [{v1}, {v2}, {v3}, {v4}]}}], '
            '"unit": "{currency}", '
            '"source": "{company} report"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
        "rejected_reasoning": "Sales data was provided.",
        "rejected_answer": (
            "<response>\n"
            "Sales were {currency}{v1} in Q1, {currency}{v2} in Q2, "
            "{currency}{v3} in Q3, and {currency}{v4} in Q4.\n"
            "</response>"
        ),
    },
]


# ===================================================================
# 2. MULTI-SERIES COMPARISON (350)
# ===================================================================

_MULTI_SERIES_SFT: List[Dict[str, str]] = [
    {
        "query": (
            "{company}'s year-over-year {metric} comparison:\n"
            "| Quarter | FY{year} | FY{year2} |\n"
            "| Q1 | {v1} | {v1b} |\n"
            "| Q2 | {v2} | {v2b} |\n"
            "| Q3 | {v3} | {v3b} |\n"
            "| Q4 | {v4} | {v4b} |\n\n"
            "Compare the two fiscal years."
        ),
        "reasoning": (
            "Step 1: Two series — FY{year} and FY{year2} — across 4 quarters.\n"
            "Step 2: Year-over-year comparison calls for grouped bar chart.\n"
            "Step 3: Grouped bars allow side-by-side comparison per quarter.\n"
            "Step 4: Alternative: overlaid line chart, but grouped bar is clearer for discrete comparison.\n"
            "Confidence: High — classic multi-series comparison."
        ),
        "answer": (
            "<response>\n"
            "Comparing {metric} across fiscal years: FY{year} totals "
            "({v1}+{v2}+{v3}+{v4}) vs FY{year2} ({v1b}+{v2b}+{v3b}+{v4b}). "
            "Quarter-by-quarter comparison reveals how performance shifted year over year.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "grouped_bar", '
            '"title": "{metric} YoY Comparison", '
            '"x": {{"label": "Quarter", "values": ["Q1", "Q2", "Q3", "Q4"]}}, '
            '"series": ['
            '{{"name": "FY{year}", "values": [{v1}, {v2}, {v3}, {v4}]}}, '
            '{{"name": "FY{year2}", "values": [{v1b}, {v2b}, {v3b}, {v4b}]}}'
            '], '
            '"unit": "", '
            '"source": "{company} annual report"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
    {
        "query": (
            "Department comparison from {company}'s {doc_type}:\n"
            "| Metric | {dept} | {dept2} |\n"
            "| Headcount | {v1} | {v1b} |\n"
            "| Budget ({currency}K) | {v2} | {v2b} |\n"
            "| Satisfaction | {v3} | {v3b} |\n\n"
            "Visualize the department comparison."
        ),
        "reasoning": (
            "Step 1: Two departments compared across 3 metrics.\n"
            "Step 2: Mixed scales (headcount vs budget vs satisfaction) make a single chart tricky.\n"
            "Step 3: Grouped bar chart works if we normalize or keep same unit.\n"
            "Step 4: Better to use grouped bar with note about different scales.\n"
            "Confidence: Medium — mixed scales require careful presentation."
        ),
        "answer": (
            "<response>\n"
            "Department comparison between {dept} and {dept2}: Headcount ({v1} vs {v1b}), "
            "Budget ({currency}{v2}K vs {currency}{v2b}K), Satisfaction ({v3} vs {v3b}). "
            "Note: metrics are on different scales.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "grouped_bar", '
            '"title": "Department Comparison: {dept} vs {dept2}", '
            '"x": {{"label": "Metric", "values": ["Headcount", "Budget ({currency}K)", "Satisfaction"]}}, '
            '"series": ['
            '{{"name": "{dept}", "values": [{v1}, {v2}, {v3}]}}, '
            '{{"name": "{dept2}", "values": [{v1b}, {v2b}, {v3b}]}}'
            '], '
            '"unit": "mixed", '
            '"source": "{doc_type}"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
    {
        "query": (
            "{company}'s product performance ({year}):\n"
            "| Month | {product} | {product2} |\n"
            "| Jan | {v1} | {v1b} |\n"
            "| Feb | {v2} | {v2b} |\n"
            "| Mar | {v3} | {v3b} |\n"
            "| Apr | {v4} | {v4b} |\n\n"
            "Show the product trend comparison."
        ),
        "reasoning": (
            "Step 1: Two product series over 4 months — time-series comparison.\n"
            "Step 2: 'Trend comparison' suggests overlaid line chart.\n"
            "Step 3: Line chart with two series shows trajectory of each product.\n"
            "Step 4: Clear labels needed to distinguish the two product lines.\n"
            "Confidence: High — multi-series time trend."
        ),
        "answer": (
            "<response>\n"
            "Product performance comparison for {year}: {product} values are "
            "{v1}, {v2}, {v3}, {v4} (Jan-Apr), while {product2} shows "
            "{v1b}, {v2b}, {v3b}, {v4b} over the same period.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "line", '
            '"title": "Product Performance Comparison ({year})", '
            '"x": {{"label": "Month", "values": ["Jan", "Feb", "Mar", "Apr"]}}, '
            '"series": ['
            '{{"name": "{product}", "values": [{v1}, {v2}, {v3}, {v4}]}}, '
            '{{"name": "{product2}", "values": [{v1b}, {v2b}, {v3b}, {v4b}]}}'
            '], '
            '"unit": "units", '
            '"source": "{company} {year} report"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
]

_MULTI_SERIES_DPO: List[Dict[str, str]] = [
    {
        "query": (
            "Compare {dept} and {dept2} quarterly spending:\n"
            "| Quarter | {dept} | {dept2} |\n"
            "| Q1 | {currency}{v1} | {currency}{v1b} |\n"
            "| Q2 | {currency}{v2} | {currency}{v2b} |\n"
            "| Q3 | {currency}{v3} | {currency}{v3b} |\n"
            "| Q4 | {currency}{v4} | {currency}{v4b} |"
        ),
        "chosen_reasoning": (
            "Step 1: Two departments, 4 quarters — classic multi-series data.\n"
            "Step 2: Grouped bar chart for side-by-side quarterly comparison.\n"
            "Step 3: Include both textual analysis and chart.\n"
            "Confidence: High."
        ),
        "chosen_answer": (
            "<response>\n"
            "Quarterly spending comparison: {dept} spent {currency}{v1}/{v2}/{v3}/{v4} "
            "across Q1-Q4, while {dept2} spent {currency}{v1b}/{v2b}/{v3b}/{v4b}.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "grouped_bar", '
            '"title": "Quarterly Spending: {dept} vs {dept2}", '
            '"x": {{"label": "Quarter", "values": ["Q1", "Q2", "Q3", "Q4"]}}, '
            '"series": ['
            '{{"name": "{dept}", "values": [{v1}, {v2}, {v3}, {v4}]}}, '
            '{{"name": "{dept2}", "values": [{v1b}, {v2b}, {v3b}, {v4b}]}}'
            '], '
            '"unit": "{currency}", '
            '"source": "spending report"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
        "rejected_reasoning": "There's comparison data.",
        "rejected_answer": (
            "<response>\n"
            "{dept} and {dept2} have different spending patterns across quarters.\n"
            "</response>"
        ),
    },
]


# ===================================================================
# 3. AUTO-DETECT TRIGGERS (300)
# ===================================================================

_AUTO_DETECT_SFT: List[Dict[str, str]] = [
    {
        "query": (
            "What was {company}'s {metric} by quarter for {year}?"
        ),
        "reasoning": (
            "Step 1: User asks for metric by quarter — implicitly tabular/chartable.\n"
            "Step 2: Quarterly data with a named metric should auto-trigger a chart.\n"
            "Step 3: Bar chart is the natural fit for quarterly breakdown.\n"
            "Step 4: Generate realistic values and include chart_spec.\n"
            "Confidence: High — structured quantitative response warrants visualization."
        ),
        "answer": (
            "<response>\n"
            "{company}'s {metric} by quarter for {year}: Q1 was {v1}, Q2 was {v2}, "
            "Q3 was {v3}, and Q4 was {v4}. The year showed variation across quarters.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "bar", '
            '"title": "{company} {metric} by Quarter ({year})", '
            '"x": {{"label": "Quarter", "values": ["Q1", "Q2", "Q3", "Q4"]}}, '
            '"series": [{{"name": "{metric}", "values": [{v1}, {v2}, {v3}, {v4}]}}], '
            '"unit": "", '
            '"source": "{company} {year} data"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
    {
        "query": (
            "How does {dept} spending break down by category at {company}?"
        ),
        "reasoning": (
            "Step 1: 'Break down by category' implies proportional data.\n"
            "Step 2: Category breakdown = pie chart trigger.\n"
            "Step 3: Auto-detect: response will contain percentage/proportional data.\n"
            "Step 4: Generate breakdown and include chart_spec.\n"
            "Confidence: High — breakdown implies chartable proportional data."
        ),
        "answer": (
            "<response>\n"
            "{dept} spending at {company} breaks down as follows: Personnel ({v1}K), "
            "Technology ({v2}K), and Operations ({v3}K). Personnel represents the largest share.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "pie", '
            '"title": "{dept} Spending Breakdown", '
            '"x": {{"label": "Category", "values": ["Personnel", "Technology", "Operations"]}}, '
            '"series": [{{"name": "Spending", "values": [{v1}, {v2}, {v3}]}}], '
            '"unit": "{currency}K", '
            '"source": "{company} budget data"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
    {
        "query": (
            "Show me the headcount trend for {company} over the past 4 quarters."
        ),
        "reasoning": (
            "Step 1: 'Trend' + 'over the past 4 quarters' = time-series request.\n"
            "Step 2: Headcount over time → line chart.\n"
            "Step 3: Auto-detect trigger: temporal data with 'trend' keyword.\n"
            "Confidence: High — explicit trend request with time dimension."
        ),
        "answer": (
            "<response>\n"
            "{company}'s headcount over the past 4 quarters: Q1 ({v1}), Q2 ({v2}), "
            "Q3 ({v3}), Q4 ({v4}). The headcount trajectory shows the organizational "
            "growth or contraction pattern.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "line", '
            '"title": "Headcount Trend", '
            '"x": {{"label": "Quarter", "values": ["Q1", "Q2", "Q3", "Q4"]}}, '
            '"series": [{{"name": "Headcount", "values": [{v1}, {v2}, {v3}, {v4}]}}], '
            '"unit": "employees", '
            '"source": "{company} HR data"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
    {
        "query": (
            "Compare {product} and {product2} sales at {company} for {year}."
        ),
        "reasoning": (
            "Step 1: 'Compare' + two products = multi-series comparison request.\n"
            "Step 2: Sales comparison → grouped bar chart.\n"
            "Step 3: Auto-detect: comparison keyword with two entities triggers multi-series chart.\n"
            "Confidence: High — comparison with named entities."
        ),
        "answer": (
            "<response>\n"
            "Sales comparison for {year}: {product} sold {v1}, {v2}, {v3}, {v4} units "
            "across Q1-Q4, while {product2} sold {v1b}, {v2b}, {v3b}, {v4b}.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "grouped_bar", '
            '"title": "{product} vs {product2} Sales ({year})", '
            '"x": {{"label": "Quarter", "values": ["Q1", "Q2", "Q3", "Q4"]}}, '
            '"series": ['
            '{{"name": "{product}", "values": [{v1}, {v2}, {v3}, {v4}]}}, '
            '{{"name": "{product2}", "values": [{v1b}, {v2b}, {v3b}, {v4b}]}}'
            '], '
            '"unit": "units", '
            '"source": "{company} {year} sales"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
]

_AUTO_DETECT_DPO: List[Dict[str, str]] = [
    {
        "query": (
            "What were the regional sales figures for {company} in {year}?"
        ),
        "chosen_reasoning": (
            "Step 1: Regional sales = categorical data that warrants a chart.\n"
            "Step 2: Bar chart for comparing across regions.\n"
            "Step 3: Auto-detect: quantitative breakdown by category.\n"
            "Confidence: High."
        ),
        "chosen_answer": (
            "<response>\n"
            "Regional sales for {company} ({year}): North ({v1}), South ({v2}), "
            "East ({v3}), West ({v4}).\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "bar", '
            '"title": "Regional Sales ({year})", '
            '"x": {{"label": "Region", "values": ["North", "South", "East", "West"]}}, '
            '"series": [{{"name": "Sales", "values": [{v1}, {v2}, {v3}, {v4}]}}], '
            '"unit": "{currency}", '
            '"source": "{company} {year} report"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
        "rejected_reasoning": "Looking up regional sales.",
        "rejected_answer": (
            "<response>\n"
            "Sales varied by region with North at {v1}, South at {v2}, East at {v3}, "
            "and West at {v4}.\n"
            "</response>"
        ),
    },
]


# ===================================================================
# 4. EXPLICIT REQUEST HANDLING (250)
# ===================================================================

_EXPLICIT_REQUEST_SFT: List[Dict[str, str]] = [
    {
        "query": (
            "Chart the following {metric} data for {company}:\n"
            "Q1: {v1}, Q2: {v2}, Q3: {v3}, Q4: {v4}"
        ),
        "reasoning": (
            "Step 1: User explicitly says 'Chart' — must include chart_spec.\n"
            "Step 2: Quarterly data → bar chart.\n"
            "Step 3: Single series, 4 data points.\n"
            "Confidence: High — explicit visualization request."
        ),
        "answer": (
            "<response>\n"
            "Here is the {metric} data for {company} charted by quarter.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "bar", '
            '"title": "{company} {metric}", '
            '"x": {{"label": "Quarter", "values": ["Q1", "Q2", "Q3", "Q4"]}}, '
            '"series": [{{"name": "{metric}", "values": [{v1}, {v2}, {v3}, {v4}]}}], '
            '"unit": "", '
            '"source": "user-provided data"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
    {
        "query": (
            "Can you visualize {company}'s {dept} budget of {currency}{v1} vs "
            "{dept2} budget of {currency}{v2}?"
        ),
        "reasoning": (
            "Step 1: 'Visualize' is an explicit chart trigger.\n"
            "Step 2: Two-way comparison — bar chart with 2 bars.\n"
            "Step 3: Simple categorical comparison.\n"
            "Confidence: High — explicit request."
        ),
        "answer": (
            "<response>\n"
            "Comparing department budgets: {dept} ({currency}{v1}) vs {dept2} ({currency}{v2}).\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "bar", '
            '"title": "Budget Comparison", '
            '"x": {{"label": "Department", "values": ["{dept}", "{dept2}"]}}, '
            '"series": [{{"name": "Budget", "values": [{v1}, {v2}]}}], '
            '"unit": "{currency}", '
            '"source": "user-provided data"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
    {
        "query": (
            "Please graph the monthly {metric} for {company}: "
            "Jan={v1}, Feb={v2}, Mar={v3}, Apr={v4}."
        ),
        "reasoning": (
            "Step 1: 'Graph' is an explicit visualization request.\n"
            "Step 2: Monthly data → line chart for trend.\n"
            "Step 3: 4 data points, single series.\n"
            "Confidence: High — explicit graph request with data."
        ),
        "answer": (
            "<response>\n"
            "Here is the monthly {metric} graph for {company}: Jan ({v1}), Feb ({v2}), "
            "Mar ({v3}), Apr ({v4}).\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "line", '
            '"title": "Monthly {metric}", '
            '"x": {{"label": "Month", "values": ["Jan", "Feb", "Mar", "Apr"]}}, '
            '"series": [{{"name": "{metric}", "values": [{v1}, {v2}, {v3}, {v4}]}}], '
            '"unit": "", '
            '"source": "user-provided data"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
    {
        "query": (
            "Make a pie chart of {company}'s market share: "
            "{product} ({v1}%), {product2} ({v2}%), Other ({v3}%)."
        ),
        "reasoning": (
            "Step 1: User explicitly requests 'pie chart'.\n"
            "Step 2: Market share data — perfect for pie chart.\n"
            "Step 3: Three segments with percentage values.\n"
            "Confidence: High — explicit type and data provided."
        ),
        "answer": (
            "<response>\n"
            "{company}'s market share breakdown: {product} at {v1}%, {product2} at {v2}%, "
            "and Other at {v3}%.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "pie", '
            '"title": "{company} Market Share", '
            '"x": {{"label": "Product", "values": ["{product}", "{product2}", "Other"]}}, '
            '"series": [{{"name": "Share", "values": [{v1}, {v2}, {v3}]}}], '
            '"unit": "%", '
            '"source": "user-provided data"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
]

_EXPLICIT_REQUEST_DPO: List[Dict[str, str]] = [
    {
        "query": (
            "Create a chart showing {company}'s quarterly {metric}: "
            "Q1={v1}, Q2={v2}, Q3={v3}, Q4={v4}."
        ),
        "chosen_reasoning": (
            "Step 1: 'Create a chart' — explicit visualization request.\n"
            "Step 2: Quarterly data → bar chart.\n"
            "Confidence: High."
        ),
        "chosen_answer": (
            "<response>\n"
            "Here is the quarterly {metric} chart for {company}.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "bar", '
            '"title": "{metric} by Quarter", '
            '"x": {{"label": "Quarter", "values": ["Q1", "Q2", "Q3", "Q4"]}}, '
            '"series": [{{"name": "{metric}", "values": [{v1}, {v2}, {v3}, {v4}]}}], '
            '"unit": "", '
            '"source": "user-provided data"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
        "rejected_reasoning": "User asked for a chart.",
        "rejected_answer": (
            "<response>\n"
            "The {metric} values are Q1={v1}, Q2={v2}, Q3={v3}, Q4={v4}.\n"
            "</response>"
        ),
    },
]


# ===================================================================
# 5. NO-CHART NEGATIVES (400) — CRITICAL
# ===================================================================

_NO_CHART_SFT: List[Dict[str, str]] = [
    {
        "query": (
            "What is the effective date of {company}'s NDA with {company2}?"
        ),
        "reasoning": (
            "Step 1: Simple factual question — a single date value.\n"
            "Step 2: No tabular data, no comparison, no trend.\n"
            "Step 3: A chart would add zero value to a single-point factual answer.\n"
            "Step 4: Respond with text only — no chart_spec.\n"
            "Confidence: High — no chartable data."
        ),
        "answer": (
            "<response>\n"
            "The NDA between {company} and {company2} has an effective date that would be "
            "specified in the agreement's header or first section. Please provide the NDA "
            "document so I can confirm the exact date.\n"
            "</response>"
        ),
    },
    {
        "query": (
            "Summarize the key terms of {company}'s {domain} {doc_type}."
        ),
        "reasoning": (
            "Step 1: Summary request — qualitative text output.\n"
            "Step 2: Key terms are textual, not numerical/chartable.\n"
            "Step 3: No data series, no comparison, no proportions.\n"
            "Step 4: No chart_spec — text-only response is appropriate.\n"
            "Confidence: High — purely textual question."
        ),
        "answer": (
            "<response>\n"
            "To summarize the key terms of {company}'s {domain} {doc_type}, I would need "
            "access to the document. Key terms typically include: parties involved, effective date, "
            "term and termination provisions, obligations of each party, payment terms, "
            "confidentiality obligations, and dispute resolution mechanisms.\n"
            "</response>"
        ),
    },
    {
        "query": (
            "Who is the authorized signatory for {company}?"
        ),
        "reasoning": (
            "Step 1: Single-entity lookup question.\n"
            "Step 2: Answer is a person's name and title — not chartable.\n"
            "Step 3: No quantitative data, no comparison.\n"
            "Step 4: Text-only response.\n"
            "Confidence: High — no chartable content."
        ),
        "answer": (
            "<response>\n"
            "The authorized signatory for {company} would be specified in the company's "
            "corporate resolutions or authorization matrix. Typically, this is a senior officer "
            "such as the CEO, CFO, or General Counsel. Please provide the relevant governance "
            "documents for a definitive answer.\n"
            "</response>"
        ),
    },
    {
        "query": (
            "Does the {domain} {doc_type} from {company} contain a force majeure clause?"
        ),
        "reasoning": (
            "Step 1: Yes/no question about document content.\n"
            "Step 2: Binary answer — no data to chart.\n"
            "Step 3: Text response with citation is sufficient.\n"
            "Step 4: No chart_spec.\n"
            "Confidence: High — binary question."
        ),
        "answer": (
            "<response>\n"
            "I would need to review the {domain} {doc_type} from {company} to determine "
            "if a force majeure clause is included. Force majeure clauses are common in "
            "commercial contracts and typically appear in the general provisions section.\n"
            "</response>"
        ),
    },
    {
        "query": (
            "Explain the indemnification provisions in {company}'s contract with {company2}."
        ),
        "reasoning": (
            "Step 1: Explanation request — qualitative analysis.\n"
            "Step 2: Indemnification provisions are textual legal terms.\n"
            "Step 3: No numerical data to visualize.\n"
            "Step 4: Text-only response is appropriate.\n"
            "Confidence: High — legal explanation, not data analysis."
        ),
        "answer": (
            "<response>\n"
            "Indemnification provisions in contracts between {company} and {company2} typically "
            "cover: (1) scope of indemnified claims (first-party vs third-party), (2) indemnification "
            "obligations of each party, (3) exclusions and limitations, (4) notice requirements, "
            "and (5) defense and settlement procedures. Please provide the specific contract for "
            "a detailed analysis.\n"
            "</response>"
        ),
    },
    {
        "query": (
            "What is {person}'s current role at {company}?"
        ),
        "reasoning": (
            "Step 1: Single fact lookup — person's role.\n"
            "Step 2: One data point, not chartable.\n"
            "Step 3: Text response only.\n"
            "Confidence: High."
        ),
        "answer": (
            "<response>\n"
            "To determine {person}'s current role at {company}, I would need access to the "
            "latest organizational chart, HR records, or corporate directory. If you can provide "
            "the relevant documents, I can confirm the current role.\n"
            "</response>"
        ),
    },
    {
        "query": (
            "Is {company} in compliance with the latest {domain} regulations?"
        ),
        "reasoning": (
            "Step 1: Compliance question — yes/no/partial assessment.\n"
            "Step 2: While compliance could involve checklists, this is a general inquiry.\n"
            "Step 3: No specific data points to visualize.\n"
            "Step 4: Text response with structured assessment framework.\n"
            "Confidence: High — qualitative question."
        ),
        "answer": (
            "<response>\n"
            "Determining {company}'s compliance with {domain} regulations requires a systematic "
            "review of: (1) applicable regulatory requirements, (2) current internal policies and "
            "controls, (3) recent audit or inspection results, and (4) any outstanding corrective "
            "actions. Please provide the specific regulations and compliance documentation.\n"
            "</response>"
        ),
    },
    {
        "query": (
            "When does {company}'s lease agreement with {company2} expire?"
        ),
        "reasoning": (
            "Step 1: Single date/fact question.\n"
            "Step 2: No series, no comparison, no distribution.\n"
            "Step 3: Chart adds no value.\n"
            "Confidence: High."
        ),
        "answer": (
            "<response>\n"
            "The lease expiration date would be specified in the lease agreement between "
            "{company} and {company2}. Check the Term section (typically Article 2 or 3) "
            "for the commencement and expiration dates, as well as any renewal options.\n"
            "</response>"
        ),
    },
]

_NO_CHART_DPO: List[Dict[str, str]] = [
    {
        "query": (
            "What is the governing law in {company}'s contract?"
        ),
        "chosen_reasoning": (
            "Step 1: Single-fact legal question — governing law jurisdiction.\n"
            "Step 2: No quantitative data to chart.\n"
            "Step 3: Text-only response is correct.\n"
            "Confidence: High — no chart needed."
        ),
        "chosen_answer": (
            "<response>\n"
            "The governing law clause specifies which jurisdiction's laws control the "
            "interpretation and enforcement of {company}'s contract. This is typically "
            "found in the general provisions section. Please provide the contract for "
            "a specific answer.\n"
            "</response>"
        ),
        "rejected_reasoning": "Let me present this information visually.",
        "rejected_answer": (
            "<response>\n"
            "The governing law is specified in the contract.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "bar", '
            '"title": "Contract Provisions", '
            '"x": {{"label": "Provision", "values": ["Governing Law"]}}, '
            '"series": [{{"name": "Status", "values": [1]}}], '
            '"unit": "", '
            '"source": "contract"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
    {
        "query": (
            "Who approved the {domain} {doc_type} for {company}?"
        ),
        "chosen_reasoning": (
            "Step 1: Single-fact question about approver identity.\n"
            "Step 2: No data series, no comparison — purely textual.\n"
            "Step 3: No chart needed.\n"
            "Confidence: High."
        ),
        "chosen_answer": (
            "<response>\n"
            "The approver of the {domain} {doc_type} would be identified in the document's "
            "signature block or approval workflow. Typically, {domain} documents require "
            "approval from the department head or compliance officer.\n"
            "</response>"
        ),
        "rejected_reasoning": "I should present this nicely with a chart.",
        "rejected_answer": (
            "<response>\n"
            "The document was approved.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "pie", '
            '"title": "Approval Status", '
            '"x": {{"label": "Status", "values": ["Approved"]}}, '
            '"series": [{{"name": "Count", "values": [1]}}], '
            '"unit": "", '
            '"source": ""}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
]


# ===================================================================
# 6. ANNOTATION INTELLIGENCE (150)
# ===================================================================

_ANNOTATION_SFT: List[Dict[str, str]] = [
    {
        "query": (
            "{company}'s {metric} by quarter ({year}):\n"
            "Q1: {v1}, Q2: {v2}, Q3: {v3}, Q4: {v4}\n\n"
            "Highlight any notable patterns or outliers."
        ),
        "reasoning": (
            "Step 1: Quarterly data with request to identify outliers.\n"
            "Step 2: Check for min/max/anomalies in the values.\n"
            "Step 3: Values: Q1={v1}, Q2={v2}, Q3={v3}, Q4={v4}.\n"
            "Step 4: Need to identify the highest and lowest quarters.\n"
            "Step 5: Add annotations to highlight key data points in chart.\n"
            "Confidence: High — can identify min/max from data."
        ),
        "answer": (
            "<response>\n"
            "{company}'s {metric} for {year} ranges from the lowest quarter to the highest. "
            "The data shows Q1={v1}, Q2={v2}, Q3={v3}, Q4={v4}. Notable observations include "
            "the peak and trough quarters, which may warrant investigation into seasonal or "
            "operational factors.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "bar", '
            '"title": "{metric} by Quarter ({year})", '
            '"x": {{"label": "Quarter", "values": ["Q1", "Q2", "Q3", "Q4"]}}, '
            '"series": [{{"name": "{metric}", "values": [{v1}, {v2}, {v3}, {v4}]}}], '
            '"unit": "", '
            '"annotations": ['
            '{{"type": "highlight", "target": "max", "label": "Peak quarter"}},'
            '{{"type": "highlight", "target": "min", "label": "Lowest quarter"}}'
            '], '
            '"source": "{company} {year} report"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
    {
        "query": (
            "{company}'s customer churn rate (monthly):\n"
            "Jan: {v1}%, Feb: {v2}%, Mar: {v3}%, Apr: {v4}%\n\n"
            "Any anomalies in the churn data?"
        ),
        "reasoning": (
            "Step 1: Monthly churn rate data — percentage values.\n"
            "Step 2: Need to identify anomalies — sudden spikes or drops.\n"
            "Step 3: Compare each month to the average to find deviations.\n"
            "Step 4: Annotate any month that deviates significantly.\n"
            "Confidence: High — can compute average and flag deviations."
        ),
        "answer": (
            "<response>\n"
            "Monthly churn rates for {company}: Jan {v1}%, Feb {v2}%, Mar {v3}%, Apr {v4}%. "
            "Examine month-over-month changes for sudden shifts that may indicate customer "
            "dissatisfaction events or seasonal patterns.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "line", '
            '"title": "Monthly Churn Rate", '
            '"x": {{"label": "Month", "values": ["Jan", "Feb", "Mar", "Apr"]}}, '
            '"series": [{{"name": "Churn Rate", "values": [{v1}, {v2}, {v3}, {v4}]}}], '
            '"unit": "%", '
            '"annotations": ['
            '{{"type": "threshold", "value": "average", "label": "Avg churn"}},'
            '{{"type": "callout", "target": "max", "label": "Investigate spike"}}'
            '], '
            '"source": "{company} customer data"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
    {
        "query": (
            "{company}'s {dept} performance scores ({year}):\n"
            "Team A: {v1}, Team B: {v2}, Team C: {v3}\n\n"
            "Which team stands out and why?"
        ),
        "reasoning": (
            "Step 1: Three teams with performance scores.\n"
            "Step 2: 'Stands out' = identify the outlier (highest or lowest).\n"
            "Step 3: Bar chart with annotation on the standout team.\n"
            "Step 4: Provide context for why the score matters.\n"
            "Confidence: High — direct comparison with annotation."
        ),
        "answer": (
            "<response>\n"
            "{dept} performance scores at {company} ({year}): Team A ({v1}), Team B ({v2}), "
            "Team C ({v3}). The team with the highest/lowest score warrants attention — "
            "either as a best practice model or for performance improvement support.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "bar", '
            '"title": "{dept} Team Performance ({year})", '
            '"x": {{"label": "Team", "values": ["Team A", "Team B", "Team C"]}}, '
            '"series": [{{"name": "Score", "values": [{v1}, {v2}, {v3}]}}], '
            '"unit": "points", '
            '"annotations": ['
            '{{"type": "highlight", "target": "max", "label": "Top performer"}},'
            '{{"type": "highlight", "target": "min", "label": "Needs attention"}}'
            '], '
            '"source": "{company} {year} review"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
]

_ANNOTATION_DPO: List[Dict[str, str]] = [
    {
        "query": (
            "{company}'s error rates: Jan={v1}, Feb={v2}, Mar={v3}, Apr={v4}\n"
            "Identify and annotate any spikes."
        ),
        "chosen_reasoning": (
            "Step 1: User wants annotations on spikes.\n"
            "Step 2: Line chart with callout annotations on outlier months.\n"
            "Step 3: Include threshold line for average.\n"
            "Confidence: High."
        ),
        "chosen_answer": (
            "<response>\n"
            "Error rates for {company}: Jan ({v1}), Feb ({v2}), Mar ({v3}), Apr ({v4}). "
            "Spikes above the average are annotated for investigation.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "line", '
            '"title": "Error Rate Trend", '
            '"x": {{"label": "Month", "values": ["Jan", "Feb", "Mar", "Apr"]}}, '
            '"series": [{{"name": "Error Rate", "values": [{v1}, {v2}, {v3}, {v4}]}}], '
            '"unit": "", '
            '"annotations": ['
            '{{"type": "threshold", "value": "average", "label": "Mean error rate"}},'
            '{{"type": "callout", "target": "max", "label": "Spike - investigate"}}'
            '], '
            '"source": "{company} ops data"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
        "rejected_reasoning": "Simple chart of the data.",
        "rejected_answer": (
            "<response>\n"
            "Error rates are {v1}, {v2}, {v3}, {v4}.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "line", '
            '"title": "Error Rate", '
            '"x": {{"label": "Month", "values": ["Jan", "Feb", "Mar", "Apr"]}}, '
            '"series": [{{"name": "Errors", "values": [{v1}, {v2}, {v3}, {v4}]}}], '
            '"unit": "", "source": ""}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
]


# ===================================================================
# 7. CHART TYPE SELECTION REASONING (150)
# ===================================================================

_CHART_TYPE_SFT: List[Dict[str, str]] = [
    {
        "query": (
            "{company}'s {metric} data:\n"
            "| Category | Value |\n"
            "| {dept} | {v1} |\n"
            "| {dept2} | {v2} |\n"
            "| {dept3} | {v3} |\n\n"
            "What chart type would best represent this data and why?"
        ),
        "reasoning": (
            "Step 1: Three categories with single values — categorical comparison.\n"
            "Step 2: Options: bar chart (comparison), pie chart (proportion).\n"
            "Step 3: Bar chart is better when the focus is on comparing absolute values.\n"
            "Step 4: Pie chart is better when showing part-of-whole relationships.\n"
            "Step 5: Since the query says 'best represent', consider the data nature.\n"
            "Step 6: If values represent shares of a total → pie. If independent measures → bar.\n"
            "Confidence: High — chart type selection follows data semantics."
        ),
        "answer": (
            "<response>\n"
            "For this {metric} data across three departments, the best chart type depends on intent:\n\n"
            "- **Bar chart**: Best if comparing absolute values between {dept}, {dept2}, and {dept3}. "
            "Shows magnitude differences clearly.\n"
            "- **Pie chart**: Best if showing each department's share of the total {metric}. "
            "Shows proportional contribution.\n\n"
            "I'll use a bar chart since the data appears to compare independent category values "
            "rather than parts of a whole.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "bar", '
            '"title": "{metric} by Department", '
            '"x": {{"label": "Department", "values": ["{dept}", "{dept2}", "{dept3}"]}}, '
            '"series": [{{"name": "{metric}", "values": [{v1}, {v2}, {v3}]}}], '
            '"unit": "", '
            '"source": "{company} data"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
    {
        "query": (
            "{company}'s monthly {metric} over 4 months:\n"
            "Jan={v1}, Feb={v2}, Mar={v3}, Apr={v4}\n\n"
            "Should I use a bar chart or line chart for this?"
        ),
        "reasoning": (
            "Step 1: Monthly time-series data — 4 consecutive months.\n"
            "Step 2: Bar chart: good for discrete period comparison.\n"
            "Step 3: Line chart: good for showing trend/direction over time.\n"
            "Step 4: Key question — is the user interested in individual month comparison "
            "or the overall trajectory?\n"
            "Step 5: 'Monthly over 4 months' suggests temporal continuity → line chart.\n"
            "Step 6: Line chart emphasizes the trend; bar chart emphasizes individual values.\n"
            "Confidence: High — time-series data generally suits line charts."
        ),
        "answer": (
            "<response>\n"
            "For monthly {metric} data over consecutive months, a **line chart** is generally "
            "the better choice because:\n\n"
            "1. The data is **temporal** — months have a natural ordering.\n"
            "2. A line shows the **trend direction** (rising, falling, stable).\n"
            "3. The connecting line implies **continuity** between data points.\n\n"
            "A bar chart would be appropriate if you wanted to emphasize individual month "
            "comparisons without implying a continuous trend.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "line", '
            '"title": "Monthly {metric}", '
            '"x": {{"label": "Month", "values": ["Jan", "Feb", "Mar", "Apr"]}}, '
            '"series": [{{"name": "{metric}", "values": [{v1}, {v2}, {v3}, {v4}]}}], '
            '"unit": "", '
            '"source": "{company} data"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
    {
        "query": (
            "{company}'s market share data:\n"
            "{company}: {v1}%\n"
            "{company2}: {v2}%\n"
            "Others: {v3}%\n\n"
            "What visualization approach works best here?"
        ),
        "reasoning": (
            "Step 1: Market share data — percentages that should sum to ~100%.\n"
            "Step 2: This is a part-to-whole relationship.\n"
            "Step 3: Pie chart is the canonical choice for market share.\n"
            "Step 4: Alternative: horizontal bar chart (better for many segments).\n"
            "Step 5: With only 3 segments, pie chart is clean and intuitive.\n"
            "Step 6: Avoid stacked bar — unnecessary complexity for 3 segments.\n"
            "Confidence: High — market share is the classic pie chart use case."
        ),
        "answer": (
            "<response>\n"
            "For market share data with 3 segments, a **pie chart** is the ideal choice because:\n\n"
            "1. Market share is inherently a **part-to-whole** relationship.\n"
            "2. Three segments are few enough to be readable in a pie.\n"
            "3. Pie charts instantly communicate relative proportions.\n\n"
            "If there were many segments (>6), a horizontal bar chart would be better for readability.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "pie", '
            '"title": "Market Share", '
            '"x": {{"label": "Company", "values": ["{company}", "{company2}", "Others"]}}, '
            '"series": [{{"name": "Share", "values": [{v1}, {v2}, {v3}]}}], '
            '"unit": "%", '
            '"source": "{company} market data"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
]

_CHART_TYPE_DPO: List[Dict[str, str]] = [
    {
        "query": (
            "{company}'s team sizes: {dept}={v1}, {dept2}={v2}, {dept3}={v3}.\n"
            "Visualize with reasoning about chart type."
        ),
        "chosen_reasoning": (
            "Step 1: Three categories with absolute values.\n"
            "Step 2: Team sizes are independent counts — not parts of a known whole.\n"
            "Step 3: Bar chart for categorical comparison of independent values.\n"
            "Step 4: Pie chart would be misleading unless we know these are all teams.\n"
            "Confidence: High — categorical comparison → bar chart."
        ),
        "chosen_answer": (
            "<response>\n"
            "Team sizes across departments: {dept} ({v1}), {dept2} ({v2}), {dept3} ({v3}). "
            "A bar chart is used because these are independent categorical values being compared, "
            "rather than parts of a defined whole.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "bar", '
            '"title": "Team Size by Department", '
            '"x": {{"label": "Department", "values": ["{dept}", "{dept2}", "{dept3}"]}}, '
            '"series": [{{"name": "Headcount", "values": [{v1}, {v2}, {v3}]}}], '
            '"unit": "people", '
            '"source": "{company} org data"}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
        "rejected_reasoning": "Making a chart of the data.",
        "rejected_answer": (
            "<response>\n"
            "Here are the team sizes.\n"
            "</response>\n\n"
            "<chart_spec>\n"
            '{{"charts": [{{"id": "chart_1", "type": "pie", '
            '"title": "Teams", '
            '"x": {{"label": "Team", "values": ["{dept}", "{dept2}", "{dept3}"]}}, '
            '"series": [{{"name": "Size", "values": [{v1}, {v2}, {v3}]}}], '
            '"unit": "", "source": ""}}], '
            '"layout": "single"}}\n'
            "</chart_spec>"
        ),
    },
]


# ===================================================================
# PUBLIC API
# ===================================================================


def generate_track6_data(output_dir: str | Path, seed: int = 42) -> dict:
    """Generate Track 6 Visualization Intelligence training data.

    Produces both SFT and DPO JSONL files with 2000 SFT examples and
    a proportional set of DPO preference pairs.

    Args:
        output_dir: Directory to write JSONL files into.
        seed: Random seed for deterministic generation.

    Returns:
        Dict with ``sft_path``, ``dpo_path``, ``sft_count``, ``dpo_count``.
    """
    output_dir = Path(output_dir)
    rng = random.Random(seed)

    # --- SFT generation ---
    sft_categories = [
        (_SINGLE_SERIES_SFT, 400),
        (_MULTI_SERIES_SFT, 350),
        (_AUTO_DETECT_SFT, 300),
        (_EXPLICIT_REQUEST_SFT, 250),
        (_NO_CHART_SFT, 400),
        (_ANNOTATION_SFT, 150),
        (_CHART_TYPE_SFT, 150),
    ]

    all_sft: List[Dict[str, str]] = []
    sub_seed = seed
    for templates, count in sft_categories:
        sub_seed += 1
        sub_rng = random.Random(sub_seed)
        all_sft.extend(_expand(templates, count, sub_rng, mode="sft"))

    rng.shuffle(all_sft)

    sft_path = output_dir / "track6_visualization_sft.jsonl"
    with JSONLWriter(sft_path) as writer:
        for ex in all_sft:
            writer.write(ex)

    # --- DPO generation ---
    dpo_categories = [
        (_SINGLE_SERIES_DPO, 70),
        (_MULTI_SERIES_DPO, 60),
        (_AUTO_DETECT_DPO, 50),
        (_EXPLICIT_REQUEST_DPO, 50),
        (_NO_CHART_DPO, 100),
        (_ANNOTATION_DPO, 30),
        (_CHART_TYPE_DPO, 40),
    ]

    all_dpo: List[Dict[str, str]] = []
    sub_seed = seed + 100
    for templates, count in dpo_categories:
        sub_seed += 1
        sub_rng = random.Random(sub_seed)
        all_dpo.extend(_expand(templates, count, sub_rng, mode="dpo"))

    rng.shuffle(all_dpo)

    dpo_path = output_dir / "track6_visualization_dpo.jsonl"
    with JSONLWriter(dpo_path) as writer:
        for ex in all_dpo:
            writer.write(ex)

    return {
        "sft_path": str(sft_path),
        "dpo_path": str(dpo_path),
        "sft_count": len(all_sft),
        "dpo_count": len(all_dpo),
    }
