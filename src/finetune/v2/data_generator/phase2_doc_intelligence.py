"""Phase 2 Document Intelligence data generator for DocWain V2+ SFT.

Generates 20K training examples (at scale=1.0) across four categories:
  - Table Understanding   (40% = 8K)
  - Layout Comprehension  (25% = 5K)
  - OCR Correction        (20% = 4K)
  - Cross-Document Reasoning (15% = 3K)

Each example includes a ``<think>`` reasoning block for chain-of-thought
training via the Qwen3 chat template.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.finetune.v2.data_generator.base import (
    DOMAINS,
    DOC_TYPES,
    JSONLWriter,
    format_sft_example,
)

# ---------------------------------------------------------------------------
# Helpers
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


def _pick(lst: list, rng: random.Random) -> Any:
    return rng.choice(lst)


def _picks(lst: list, n: int, rng: random.Random) -> list:
    return [rng.choice(lst) for _ in range(n)]


def _rand_amount(rng: random.Random, lo: float = 100.0, hi: float = 99999.0) -> str:
    return f"{rng.uniform(lo, hi):,.2f}"


def _rand_int(rng: random.Random, lo: int = 1, hi: int = 500) -> int:
    return rng.randint(lo, hi)


def _expand_templates(
    templates: List[Dict[str, str]],
    target_count: int,
    rng: random.Random,
) -> List[Dict[str, str]]:
    """Cycle through templates with domain/doc_type variation to reach target_count."""
    results: List[Dict[str, str]] = []
    idx = 0
    while len(results) < target_count:
        tpl = templates[idx % len(templates)]
        domain = _pick(DOMAINS, rng)
        doc_type = _pick(DOC_TYPES, rng)
        company = _pick(_COMPANY_NAMES, rng)
        person = _pick(_PERSON_NAMES, rng)
        dept = _pick(_DEPARTMENTS, rng)
        product = _pick(_PRODUCT_NAMES, rng)
        currency = _pick(_CURRENCIES, rng)
        amount = _rand_amount(rng)
        amount2 = _rand_amount(rng)
        amount3 = _rand_amount(rng)
        qty = _rand_int(rng, 1, 200)
        year = rng.randint(2020, 2026)
        quarter = rng.choice(["Q1", "Q2", "Q3", "Q4"])
        month = rng.choice([
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ])
        pct = f"{rng.uniform(1, 45):.1f}"
        row_num = rng.randint(2, 15)
        col_num = rng.randint(2, 8)

        subs = {
            "domain": domain,
            "doc_type": doc_type,
            "company": company,
            "person": person,
            "dept": dept,
            "product": product,
            "currency": currency,
            "amount": amount,
            "amount2": amount2,
            "amount3": amount3,
            "qty": str(qty),
            "year": str(year),
            "quarter": quarter,
            "month": month,
            "pct": pct,
            "row_num": str(row_num),
            "col_num": str(col_num),
        }

        try:
            query = tpl["query"].format(**subs)
            reasoning = tpl["reasoning"].format(**subs)
            answer = tpl["answer"].format(**subs)
        except (KeyError, IndexError):
            idx += 1
            continue

        results.append(format_sft_example(query, reasoning, answer))
        idx += 1

    return results


# ===================================================================
# TABLE UNDERSTANDING TEMPLATES
# ===================================================================

_TABLE_SIMPLE_TEMPLATES: List[Dict[str, str]] = [
    {
        "query": (
            "Given the following {domain} {doc_type} table:\n"
            "| Item | Quantity | Unit Price | Total |\n"
            "| {product} | {qty} | {currency}{amount} | {currency}{amount2} |\n\n"
            "What is the total amount for {product}?"
        ),
        "reasoning": (
            "Step 1: Identify the table — single table with columns Item, Quantity, Unit Price, Total.\n"
            "Step 2: Locate row for '{product}' — row 1.\n"
            "Step 3: Read the Total column value: {currency}{amount2}.\n"
            "Confidence: High — direct single-cell lookup."
        ),
        "answer": "The total amount for {product} is {currency}{amount2}.",
    },
    {
        "query": (
            "In this {domain} report table:\n"
            "| Department | {quarter} Budget | {quarter} Actual | Variance |\n"
            "| {dept} | {currency}{amount} | {currency}{amount2} | {currency}{amount3} |\n\n"
            "What was the {quarter} actual spend for {dept}?"
        ),
        "reasoning": (
            "Step 1: Table identified — budget vs actual comparison for {quarter}.\n"
            "Step 2: Locate '{dept}' row.\n"
            "Step 3: Read '{quarter} Actual' column: {currency}{amount2}.\n"
            "Confidence: High — single cell read."
        ),
        "answer": "The {quarter} actual spend for {dept} was {currency}{amount2}.",
    },
    {
        "query": (
            "From the {domain} {doc_type}:\n"
            "| Employee | Title | Start Date |\n"
            "| {person} | Senior Analyst | {month} {year} |\n\n"
            "When did {person} start?"
        ),
        "reasoning": (
            "Step 1: Single table with Employee, Title, Start Date.\n"
            "Step 2: Locate '{person}' row.\n"
            "Step 3: Read Start Date: {month} {year}.\n"
            "Confidence: High — direct lookup."
        ),
        "answer": "{person} started in {month} {year}.",
    },
    {
        "query": (
            "Review this {domain} inventory table:\n"
            "| SKU | Product | Stock |\n"
            "| SKU-{qty} | {product} | {row_num} units |\n\n"
            "How many units of {product} are in stock?"
        ),
        "reasoning": (
            "Step 1: Inventory table with SKU, Product, Stock columns.\n"
            "Step 2: Find '{product}' — SKU-{qty}.\n"
            "Step 3: Stock = {row_num} units.\n"
            "Confidence: High — direct read."
        ),
        "answer": "There are {row_num} units of {product} in stock.",
    },
    {
        "query": (
            "From the {domain} {doc_type} summary:\n"
            "| Metric | Value |\n"
            "| Revenue | {currency}{amount} |\n"
            "| Expenses | {currency}{amount2} |\n\n"
            "What is the reported revenue?"
        ),
        "reasoning": (
            "Step 1: Two-column summary table — Metric and Value.\n"
            "Step 2: Locate 'Revenue' row.\n"
            "Step 3: Value = {currency}{amount}.\n"
            "Confidence: High — simple key-value lookup."
        ),
        "answer": "The reported revenue is {currency}{amount}.",
    },
    {
        "query": (
            "In this {domain} vendor table:\n"
            "| Vendor | Contact | Rating |\n"
            "| {company} | {person} | {row_num}/10 |\n\n"
            "What rating does {company} have?"
        ),
        "reasoning": (
            "Step 1: Vendor evaluation table.\n"
            "Step 2: Find '{company}' row.\n"
            "Step 3: Rating column = {row_num}/10.\n"
            "Confidence: High — direct read."
        ),
        "answer": "{company} has a rating of {row_num}/10.",
    },
    {
        "query": (
            "From the {domain} {doc_type}:\n"
            "| Category | {year} |\n"
            "| {dept} | {currency}{amount} |\n\n"
            "What is the {year} value for {dept}?"
        ),
        "reasoning": (
            "Step 1: Two-column table with Category and year {year}.\n"
            "Step 2: Locate '{dept}'.\n"
            "Step 3: Value = {currency}{amount}.\n"
            "Confidence: High."
        ),
        "answer": "The {year} value for {dept} is {currency}{amount}.",
    },
    {
        "query": (
            "Table from {company}'s {domain} {doc_type}:\n"
            "| Line | Description | Amount |\n"
            "| 1 | {product} delivery | {currency}{amount} |\n\n"
            "What is the amount on line 1?"
        ),
        "reasoning": (
            "Step 1: Line-item table from {company}.\n"
            "Step 2: Row with Line = 1.\n"
            "Step 3: Amount = {currency}{amount}.\n"
            "Confidence: High — single cell."
        ),
        "answer": "The amount on line 1 is {currency}{amount}.",
    },
]

_TABLE_MEDIUM_TEMPLATES: List[Dict[str, str]] = [
    {
        "query": (
            "Given these two tables from a {domain} {doc_type}:\n\n"
            "Table 1 — Order Summary:\n"
            "| Item | Qty | Unit Price |\n"
            "| {product} | {qty} | {currency}{amount} |\n\n"
            "Table 2 — Tax Rates:\n"
            "| Region | Tax Rate |\n"
            "| Domestic | {pct}% |\n\n"
            "Calculate the total cost of {product} including domestic tax."
        ),
        "reasoning": (
            "Step 1: From Table 1 — {product}: qty={qty}, unit price={currency}{amount}.\n"
            "Step 2: Subtotal = {qty} x {currency}{amount}.\n"
            "Step 3: From Table 2 — domestic tax rate = {pct}%.\n"
            "Step 4: Tax amount = subtotal x {pct}/100.\n"
            "Step 5: Total = subtotal + tax.\n"
            "Confidence: High — straightforward cross-table computation with two tables."
        ),
        "answer": (
            "The subtotal for {product} is {qty} x {currency}{amount}. "
            "Applying the domestic tax rate of {pct}% gives the total inclusive of tax."
        ),
    },
    {
        "query": (
            "From a {domain} {doc_type} with merged header cells:\n\n"
            "| | {quarter} {year} | | |\n"
            "| Department | Budget | Actual | Variance |\n"
            "| {dept} | {currency}{amount} | {currency}{amount2} | {currency}{amount3} |\n\n"
            "What is the variance for {dept} in {quarter} {year}?"
        ),
        "reasoning": (
            "Step 1: Table has merged header spanning '{quarter} {year}' across Budget/Actual/Variance.\n"
            "Step 2: Locate '{dept}' row.\n"
            "Step 3: Variance column (col 4) = {currency}{amount3}.\n"
            "Confidence: High — merged header understood, single value extraction."
        ),
        "answer": "The variance for {dept} in {quarter} {year} is {currency}{amount3}.",
    },
    {
        "query": (
            "Analyse this {domain} multi-row table:\n"
            "| Region | {product} Sales | {product} Returns | Net |\n"
            "| North | {currency}{amount} | {currency}{amount3} | ? |\n"
            "| South | {currency}{amount2} | {currency}{amount3} | ? |\n\n"
            "Which region has higher net sales?"
        ),
        "reasoning": (
            "Step 1: Table has two regions and computed Net = Sales - Returns.\n"
            "Step 2: North net = {currency}{amount} - {currency}{amount3}.\n"
            "Step 3: South net = {currency}{amount2} - {currency}{amount3}.\n"
            "Step 4: Compare the two net values to determine which is higher.\n"
            "Confidence: Medium — requires computation and comparison."
        ),
        "answer": (
            "To determine the higher net: North net = {currency}{amount} - {currency}{amount3}, "
            "South net = {currency}{amount2} - {currency}{amount3}. Compare these values to identify the region with higher net sales."
        ),
    },
    {
        "query": (
            "This {domain} {doc_type} has a pivot table:\n"
            "| | Q1 | Q2 | Q3 | Q4 | Annual |\n"
            "| {dept} | {currency}{amount} | {currency}{amount2} | {currency}{amount3} | {currency}{amount} | ? |\n\n"
            "What should the Annual column show for {dept}?"
        ),
        "reasoning": (
            "Step 1: Pivot table with quarterly columns and Annual total.\n"
            "Step 2: Annual = Q1 + Q2 + Q3 + Q4 for {dept}.\n"
            "Step 3: Sum = {currency}{amount} + {currency}{amount2} + {currency}{amount3} + {currency}{amount}.\n"
            "Confidence: High — straightforward summation across row."
        ),
        "answer": (
            "The Annual column for {dept} should be the sum of Q1 through Q4: "
            "{currency}{amount} + {currency}{amount2} + {currency}{amount3} + {currency}{amount}."
        ),
    },
    {
        "query": (
            "From a {domain} comparison table:\n"
            "| Vendor | Price | Lead Time (days) | Rating |\n"
            "| {company} | {currency}{amount} | {qty} | {row_num}/10 |\n"
            "| Globex Industries | {currency}{amount2} | {col_num} | 7/10 |\n\n"
            "Which vendor offers the lower price?"
        ),
        "reasoning": (
            "Step 1: Comparison table with 2 vendors.\n"
            "Step 2: {company} price = {currency}{amount}.\n"
            "Step 3: Globex Industries price = {currency}{amount2}.\n"
            "Step 4: Compare the two price values.\n"
            "Confidence: High — direct numeric comparison."
        ),
        "answer": (
            "Comparing prices: {company} at {currency}{amount} vs Globex Industries at {currency}{amount2}. "
            "The vendor with the lower numeric value offers the better price."
        ),
    },
    {
        "query": (
            "Given this {domain} {doc_type} with a percentage column:\n"
            "| Category | Amount | % of Total |\n"
            "| {dept} | {currency}{amount} | {pct}% |\n"
            "| Other | {currency}{amount2} | ? |\n\n"
            "What percentage does 'Other' represent?"
        ),
        "reasoning": (
            "Step 1: Two categories with % of Total column.\n"
            "Step 2: {dept} = {pct}%, so Other = 100% - {pct}%.\n"
            "Step 3: Computed Other % = (100 - {pct})%.\n"
            "Confidence: High — complementary percentage."
        ),
        "answer": "The 'Other' category represents {pct}% subtracted from 100%, i.e. the complementary share.",
    },
    {
        "query": (
            "Multi-table {domain} {doc_type}:\n\n"
            "Table A — Headcount:\n"
            "| Department | FTEs |\n"
            "| {dept} | {qty} |\n\n"
            "Table B — Cost per FTE:\n"
            "| Department | Annual Cost |\n"
            "| {dept} | {currency}{amount} |\n\n"
            "What is the total annual cost for {dept}?"
        ),
        "reasoning": (
            "Step 1: Table A gives FTEs for {dept} = {qty}.\n"
            "Step 2: Table B gives cost per FTE = {currency}{amount}.\n"
            "Step 3: Total = {qty} x {currency}{amount}.\n"
            "Confidence: High — cross-table multiplication."
        ),
        "answer": "The total annual cost for {dept} is {qty} x {currency}{amount}.",
    },
    {
        "query": (
            "From the {domain} {doc_type} with rowspan:\n"
            "| Category | Sub-Category | {year} |\n"
            "| {dept} (spans 2 rows) | Operations | {currency}{amount} |\n"
            "| | Strategy | {currency}{amount2} |\n\n"
            "What is the total for {dept}?"
        ),
        "reasoning": (
            "Step 1: Table has merged/rowspan cell for '{dept}' spanning 2 sub-rows.\n"
            "Step 2: Operations = {currency}{amount}, Strategy = {currency}{amount2}.\n"
            "Step 3: Total = {currency}{amount} + {currency}{amount2}.\n"
            "Confidence: High — rowspan handling + summation."
        ),
        "answer": "The total for {dept} is {currency}{amount} + {currency}{amount2}.",
    },
]

_TABLE_HARD_TEMPLATES: List[Dict[str, str]] = [
    {
        "query": (
            "Complex {domain} {doc_type} with nested tables:\n\n"
            "Outer Table:\n"
            "| Section | Details |\n"
            "| {dept} | [Nested Table] |\n\n"
            "Nested Table inside '{dept}' cell:\n"
            "| Sub-Item | {quarter} | {year} Total |\n"
            "| {product} | {currency}{amount} | {currency}{amount2} |\n"
            "| Overhead | {currency}{amount3} | {currency}{amount} |\n\n"
            "What is the {quarter} total for all sub-items under {dept}?"
        ),
        "reasoning": (
            "Step 1: Outer table has a nested table in the Details cell of '{dept}'.\n"
            "Step 2: Navigate into nested table — 2 sub-items.\n"
            "Step 3: {quarter} column: {product} = {currency}{amount}, Overhead = {currency}{amount3}.\n"
            "Step 4: Sum = {currency}{amount} + {currency}{amount3}.\n"
            "Step 5: Cross-reference with {year} Total for validation.\n"
            "Confidence: Medium — nested table traversal required."
        ),
        "answer": (
            "The {quarter} total for sub-items under {dept} is "
            "{currency}{amount} + {currency}{amount3} (sum of {product} and Overhead)."
        ),
    },
    {
        "query": (
            "Cross-table reference in {domain} {doc_type}:\n\n"
            "Table 1 — Master Price List:\n"
            "| Code | Item | Base Price |\n"
            "| P-{qty} | {product} | {currency}{amount} |\n\n"
            "Table 2 — Discount Schedule:\n"
            "| Code | Tier | Discount |\n"
            "| P-{qty} | Gold | {pct}% |\n\n"
            "Table 3 — Order:\n"
            "| Code | Quantity |\n"
            "| P-{qty} | {row_num} |\n\n"
            "Calculate the discounted total for order code P-{qty}."
        ),
        "reasoning": (
            "Step 1: Table 1 — P-{qty} base price = {currency}{amount}.\n"
            "Step 2: Table 2 — P-{qty} Gold tier discount = {pct}%.\n"
            "Step 3: Table 3 — order quantity = {row_num}.\n"
            "Step 4: Discounted unit price = {currency}{amount} x (1 - {pct}/100).\n"
            "Step 5: Total = discounted unit price x {row_num}.\n"
            "Confidence: Medium — requires joining 3 tables on code P-{qty}."
        ),
        "answer": (
            "Discounted unit price = {currency}{amount} x (1 - {pct}%). "
            "Total = discounted price x {row_num} units."
        ),
    },
    {
        "query": (
            "Hierarchical {domain} table:\n"
            "| Level | Category | Sub-Category | {quarter} {year} |\n"
            "| 1 | {dept} | — | SUBTOTAL |\n"
            "| 2 | | {product} | {currency}{amount} |\n"
            "| 2 | | Consulting | {currency}{amount2} |\n"
            "| 1 | Operations | — | SUBTOTAL |\n"
            "| 2 | | Logistics | {currency}{amount3} |\n\n"
            "What is the subtotal for {dept}?"
        ),
        "reasoning": (
            "Step 1: Hierarchical table — Level 1 = category totals, Level 2 = sub-items.\n"
            "Step 2: Under {dept}: {product} = {currency}{amount}, Consulting = {currency}{amount2}.\n"
            "Step 3: Subtotal = {currency}{amount} + {currency}{amount2}.\n"
            "Step 4: Verify this should match the SUBTOTAL marker in the Level 1 row.\n"
            "Confidence: High — tree-structured aggregation."
        ),
        "answer": (
            "The subtotal for {dept} is {currency}{amount} + {currency}{amount2}."
        ),
    },
    {
        "query": (
            "Multi-year cross-reference {domain} analysis:\n\n"
            "Table A — {year} Actuals:\n"
            "| Dept | Revenue | Cost |\n"
            "| {dept} | {currency}{amount} | {currency}{amount2} |\n\n"
            "Table B — Previous Year:\n"
            "| Dept | Revenue | Cost |\n"
            "| {dept} | {currency}{amount3} | {currency}{amount} |\n\n"
            "Calculate year-over-year revenue growth for {dept}."
        ),
        "reasoning": (
            "Step 1: Table A — {dept} current revenue = {currency}{amount}.\n"
            "Step 2: Table B — {dept} prior revenue = {currency}{amount3}.\n"
            "Step 3: YoY growth = (current - prior) / prior x 100.\n"
            "Step 4: = ({currency}{amount} - {currency}{amount3}) / {currency}{amount3} x 100%.\n"
            "Confidence: Medium — cross-table temporal comparison."
        ),
        "answer": (
            "Year-over-year revenue growth for {dept} = "
            "({currency}{amount} - {currency}{amount3}) / {currency}{amount3} x 100%."
        ),
    },
    {
        "query": (
            "Nested + pivot {domain} {doc_type}:\n\n"
            "| Region | {dept} Q1 | {dept} Q2 | {dept} Total | Grand Total |\n"
            "| North | {currency}{amount} | {currency}{amount2} | ? | ? |\n"
            "| South | {currency}{amount3} | {currency}{amount} | ? | ? |\n\n"
            "Additional context table:\n"
            "| Region | Non-{dept} Total |\n"
            "| North | {currency}{amount3} |\n"
            "| South | {currency}{amount2} |\n\n"
            "Fill in all missing values."
        ),
        "reasoning": (
            "Step 1: {dept} Total per region = Q1 + Q2.\n"
            "Step 2: North {dept} Total = {currency}{amount} + {currency}{amount2}.\n"
            "Step 3: South {dept} Total = {currency}{amount3} + {currency}{amount}.\n"
            "Step 4: Grand Total = {dept} Total + Non-{dept} Total (from context table).\n"
            "Step 5: North Grand = North {dept} Total + {currency}{amount3}.\n"
            "Step 6: South Grand = South {dept} Total + {currency}{amount2}.\n"
            "Confidence: Medium — cross-table join with computed values."
        ),
        "answer": (
            "North {dept} Total = {currency}{amount} + {currency}{amount2}; "
            "South {dept} Total = {currency}{amount3} + {currency}{amount}; "
            "Grand Totals include non-{dept} figures from the context table."
        ),
    },
    {
        "query": (
            "Sparse {domain} matrix table:\n"
            "| | {product} | Gadget Pro | Module X-100 |\n"
            "| {company} | {currency}{amount} | — | {currency}{amount3} |\n"
            "| Globex | — | {currency}{amount2} | {currency}{amount} |\n"
            "| Initech | {currency}{amount2} | {currency}{amount3} | — |\n\n"
            "Which vendor-product pair has the highest quoted price?"
        ),
        "reasoning": (
            "Step 1: Sparse matrix — '—' means no quote.\n"
            "Step 2: Non-null values: {company}/{product}={currency}{amount}, "
            "{company}/Module X-100={currency}{amount3}, Globex/Gadget Pro={currency}{amount2}, "
            "Globex/Module X-100={currency}{amount}, Initech/{product}={currency}{amount2}, "
            "Initech/Gadget Pro={currency}{amount3}.\n"
            "Step 3: Compare all values to find maximum.\n"
            "Confidence: Medium — sparse table with multiple comparisons."
        ),
        "answer": (
            "Comparing all non-null entries across the sparse matrix, "
            "the highest quoted price can be identified by comparing: "
            "{currency}{amount}, {currency}{amount2}, and {currency}{amount3}."
        ),
    },
    {
        "query": (
            "Multi-level grouped {domain} {doc_type}:\n\n"
            "| Group | Sub-Group | Item | {year} Value |\n"
            "| Assets | Current | Cash | {currency}{amount} |\n"
            "| Assets | Current | Receivables | {currency}{amount2} |\n"
            "| Assets | Non-Current | Equipment | {currency}{amount3} |\n"
            "| Liabilities | Current | Payables | {currency}{amount} |\n\n"
            "What is the total Current Assets?"
        ),
        "reasoning": (
            "Step 1: Multi-level grouping — Group > Sub-Group > Item.\n"
            "Step 2: Filter: Group='Assets', Sub-Group='Current'.\n"
            "Step 3: Matching items: Cash={currency}{amount}, Receivables={currency}{amount2}.\n"
            "Step 4: Total Current Assets = {currency}{amount} + {currency}{amount2}.\n"
            "Confidence: High — hierarchical filter + sum."
        ),
        "answer": "Total Current Assets = {currency}{amount} + {currency}{amount2}.",
    },
    {
        "query": (
            "Conditional cross-table in {domain} {doc_type}:\n\n"
            "Table 1 — Thresholds:\n"
            "| Metric | Warning | Critical |\n"
            "| Spend | {currency}{amount2} | {currency}{amount} |\n\n"
            "Table 2 — Actual:\n"
            "| Department | Spend |\n"
            "| {dept} | {currency}{amount3} |\n\n"
            "Is {dept} in warning or critical status?"
        ),
        "reasoning": (
            "Step 1: Table 1 thresholds — Warning at {currency}{amount2}, Critical at {currency}{amount}.\n"
            "Step 2: Table 2 — {dept} actual spend = {currency}{amount3}.\n"
            "Step 3: Compare actual against thresholds to determine status.\n"
            "Step 4: If actual >= critical then Critical; elif actual >= warning then Warning; else Normal.\n"
            "Confidence: Medium — conditional logic across tables."
        ),
        "answer": (
            "Comparing {dept}'s spend of {currency}{amount3} against thresholds: "
            "Warning={currency}{amount2}, Critical={currency}{amount}. "
            "Status depends on which threshold is exceeded."
        ),
    },
]

# ===================================================================
# LAYOUT COMPREHENSION TEMPLATES
# ===================================================================

_LAYOUT_TEMPLATES: List[Dict[str, str]] = [
    {
        "query": (
            "A {domain} {doc_type} has the following layout:\n"
            "--- Page Header: '{company} — Confidential' ---\n"
            "# Section 1: Overview\n"
            "Main body text about {product} deployment.\n"
            "## 1.1 Background\n"
            "Additional context for {dept}.\n"
            "--- Page Footer: 'Page 1 of 5' ---\n\n"
            "Identify the header hierarchy and separate page headers/footers from content."
        ),
        "reasoning": (
            "Step 1: Page header = '{company} — Confidential' — decorative, not content.\n"
            "Step 2: Page footer = 'Page 1 of 5' — pagination, not content.\n"
            "Step 3: Content hierarchy: H1='Section 1: Overview' > H2='1.1 Background'.\n"
            "Step 4: Main body belongs under H1; additional context under H2.\n"
            "Confidence: High — clear structural markers."
        ),
        "answer": (
            "Header hierarchy: H1='Section 1: Overview', H2='1.1 Background'. "
            "Page header ('{company} — Confidential') and footer ('Page 1 of 5') are "
            "document metadata, not content."
        ),
    },
    {
        "query": (
            "A {domain} {doc_type} uses a two-column layout:\n"
            "LEFT COLUMN: Main policy text about {product} standards.\n"
            "RIGHT COLUMN: Sidebar with key definitions and {dept} contacts.\n\n"
            "What is the correct reading order?"
        ),
        "reasoning": (
            "Step 1: Detect two-column layout — left and right columns.\n"
            "Step 2: Left column = primary content (policy text).\n"
            "Step 3: Right column = supplementary sidebar (definitions, contacts).\n"
            "Step 4: Reading order: left column first (top to bottom), then right sidebar.\n"
            "Confidence: High — standard two-column document layout."
        ),
        "answer": (
            "Reading order: (1) Left column — main policy text from top to bottom, "
            "(2) Right sidebar — definitions and contacts. The sidebar is supplementary."
        ),
    },
    {
        "query": (
            "A {domain} form has these field-label pairs:\n"
            "Label: 'Company Name' → Field: '{company}'\n"
            "Label: 'Contact Person' → Field: '{person}'\n"
            "Label: 'Department' → Field: '{dept}'\n"
            "Label: 'Amount ({currency})' → Field: '{amount}'\n\n"
            "Extract all field-label associations."
        ),
        "reasoning": (
            "Step 1: Form layout with label-field pairs.\n"
            "Step 2: Association 1: 'Company Name' → '{company}'.\n"
            "Step 3: Association 2: 'Contact Person' → '{person}'.\n"
            "Step 4: Association 3: 'Department' → '{dept}'.\n"
            "Step 5: Association 4: 'Amount ({currency})' → '{amount}'.\n"
            "Confidence: High — clear label:field pairing."
        ),
        "answer": (
            "Field-label associations: Company Name={company}, Contact Person={person}, "
            "Department={dept}, Amount({currency})={amount}."
        ),
    },
    {
        "query": (
            "This {domain} {doc_type} has section boundaries:\n"
            "=== SECTION A: {dept} Overview ===\n"
            "Paragraph about {company}'s {dept} operations.\n"
            "=== SECTION B: Financial Summary ===\n"
            "Revenue: {currency}{amount}\n"
            "=== SECTION C: Appendix ===\n"
            "Supporting data tables.\n\n"
            "Identify all section boundaries and their content types."
        ),
        "reasoning": (
            "Step 1: Three sections delimited by '=== SECTION X ===' markers.\n"
            "Step 2: Section A — narrative/descriptive content about {dept}.\n"
            "Step 3: Section B — financial data (revenue figure).\n"
            "Step 4: Section C — supplementary (appendix with tables).\n"
            "Confidence: High — explicit section markers."
        ),
        "answer": (
            "Section boundaries: A='{dept} Overview' (narrative), "
            "B='Financial Summary' (quantitative), C='Appendix' (supplementary data)."
        ),
    },
    {
        "query": (
            "OCR output of a {domain} {doc_type} shows:\n"
            "Line 1: '{company}'\n"
            "Line 2: '{person}, {dept}'\n"
            "Line 3: '---'\n"
            "Line 4: 'RE: {product} Contract #{qty}'\n"
            "Line 5: 'Dear {person},'\n"
            "Line 6: 'Please find enclosed...'\n\n"
            "Classify each line by layout role."
        ),
        "reasoning": (
            "Step 1: Line 1 — company letterhead/header.\n"
            "Step 2: Line 2 — sender identification.\n"
            "Step 3: Line 3 — visual separator.\n"
            "Step 4: Line 4 — subject/reference line.\n"
            "Step 5: Line 5 — salutation.\n"
            "Step 6: Line 6 — body content start.\n"
            "Confidence: High — standard business letter layout."
        ),
        "answer": (
            "Layout roles: Line 1=Letterhead, Line 2=Sender Info, Line 3=Separator, "
            "Line 4=Subject Line, Line 5=Salutation, Line 6=Body."
        ),
    },
    {
        "query": (
            "Multi-page {domain} {doc_type} layout:\n"
            "Page 1: Title page with '{company} — {doc_type}' centered.\n"
            "Page 2: Table of Contents.\n"
            "Page 3-7: Main content with running header '{dept} Report — {year}'.\n"
            "Page 8: Signature page with '{person}' and date fields.\n\n"
            "Describe the document structure."
        ),
        "reasoning": (
            "Step 1: Page 1 = title page (non-content, metadata).\n"
            "Step 2: Page 2 = navigation (TOC).\n"
            "Step 3: Pages 3-7 = primary content with running header.\n"
            "Step 4: Page 8 = signature/execution page.\n"
            "Step 5: Running header '{dept} Report — {year}' is repeated metadata.\n"
            "Confidence: High — standard multi-page document structure."
        ),
        "answer": (
            "Document structure: Title page (p1), TOC (p2), Main content (p3-7) with "
            "running header, Signature page (p8). Running headers are metadata."
        ),
    },
    {
        "query": (
            "A {domain} {doc_type} contains a mixed layout:\n"
            "HEADER BAR: '{company} | {dept} | {month} {year}'\n"
            "[LEFT PANEL - 70%]: Detailed analysis of {product}.\n"
            "[RIGHT PANEL - 30%]: Key metrics box: Revenue {currency}{amount}.\n"
            "[FULL WIDTH]: Conclusion paragraph by {person}.\n"
            "[FOOTER]: 'Document ID: DOC-{qty}'\n\n"
            "Map the layout zones."
        ),
        "reasoning": (
            "Step 1: Header bar — spanning full width, contains metadata.\n"
            "Step 2: Two-panel section — left (70%) is main content, right (30%) is metrics sidebar.\n"
            "Step 3: Full-width section — conclusion, returns to single-column.\n"
            "Step 4: Footer — document identifier, metadata.\n"
            "Confidence: High — explicit layout zone indicators."
        ),
        "answer": (
            "Layout zones: Header bar (metadata), Left panel 70% (main analysis), "
            "Right panel 30% (key metrics), Full-width (conclusion), Footer (document ID)."
        ),
    },
    {
        "query": (
            "A scanned {domain} form has the following spatial layout:\n"
            "Top-left box: Logo area\n"
            "Top-right: 'Form #{qty}    Date: {month} {year}'\n"
            "Row 1: [Label: 'Applicant'] [Field: '{person}']\n"
            "Row 2: [Label: 'Organization'] [Field: '{company}']\n"
            "Row 3: [Label: 'Requested Amount'] [Field: '{currency}{amount}']\n"
            "Bottom: [Checkbox: 'I agree to terms'] [Signature line]\n\n"
            "Extract the form structure."
        ),
        "reasoning": (
            "Step 1: Top area — logo (decorative) + form number and date (metadata).\n"
            "Step 2: Rows 1-3 — structured label-field pairs.\n"
            "Step 3: Bottom area — consent checkbox + signature (action elements).\n"
            "Step 4: Field extraction: Applicant={person}, Organization={company}, Amount={currency}{amount}.\n"
            "Confidence: High — structured form with clear spatial layout."
        ),
        "answer": (
            "Form structure: Header (logo + form #{qty}, {month} {year}), "
            "Data fields (Applicant={person}, Organization={company}, Amount={currency}{amount}), "
            "Footer (consent checkbox + signature line)."
        ),
    },
    {
        "query": (
            "A {domain} {doc_type} has interleaved content:\n"
            "Paragraph: 'The {dept} review found that...'\n"
            "[EMBEDDED TABLE]\n"
            "| Item | Status |\n"
            "| {product} | Approved |\n"
            "[/TABLE]\n"
            "Paragraph: 'Based on the above, {person} recommends...'\n"
            "[FOOTNOTE 1]: 'See Appendix B for details.'\n\n"
            "Separate inline content from structural elements."
        ),
        "reasoning": (
            "Step 1: First paragraph — narrative content (inline).\n"
            "Step 2: Embedded table — structural element breaking text flow.\n"
            "Step 3: Second paragraph — narrative content referencing the table.\n"
            "Step 4: Footnote — supplementary metadata linked to content.\n"
            "Confidence: High — clear delineation between inline and structural."
        ),
        "answer": (
            "Inline content: two paragraphs. Structural elements: embedded table "
            "(1 row, {product}=Approved). Metadata: Footnote 1 (cross-reference to Appendix B)."
        ),
    },
    {
        "query": (
            "A {domain} {doc_type} has nested list structure:\n"
            "1. {dept} Requirements\n"
            "   a. Staffing: {qty} FTEs\n"
            "   b. Budget: {currency}{amount}\n"
            "      i. Capital: {currency}{amount2}\n"
            "      ii. Operating: {currency}{amount3}\n"
            "2. Timeline\n"
            "   a. Start: {month} {year}\n\n"
            "Parse the hierarchy."
        ),
        "reasoning": (
            "Step 1: Three-level nested list structure.\n"
            "Step 2: Level 1 (numeric): '1. {dept} Requirements', '2. Timeline'.\n"
            "Step 3: Level 2 (alpha): under item 1 — a. Staffing, b. Budget; under item 2 — a. Start.\n"
            "Step 4: Level 3 (roman): under 1.b — i. Capital, ii. Operating.\n"
            "Confidence: High — standard outline numbering."
        ),
        "answer": (
            "Hierarchy: 1. {dept} Requirements > (a) Staffing={qty} FTEs, "
            "(b) Budget={currency}{amount} > (i) Capital={currency}{amount2}, "
            "(ii) Operating={currency}{amount3}; 2. Timeline > (a) Start={month} {year}."
        ),
    },
]

# ===================================================================
# OCR CORRECTION TEMPLATES
# ===================================================================

_OCR_TEMPLATES: List[Dict[str, str]] = [
    {
        "query": (
            "OCR text from a {domain} {doc_type}: 'The tota1 amount is {currency}{amount} "
            "for the iten dellvered to {company}.'\n"
            "Correct any OCR errors."
        ),
        "reasoning": (
            "Step 1: 'tota1' — digit '1' confused with letter 'l' → 'total'.\n"
            "Step 2: 'iten' — missing letter → 'item'.\n"
            "Step 3: 'dellvered' — double 'l' error → 'delivered'.\n"
            "Step 4: {currency}{amount} and {company} appear correct.\n"
            "Confidence: High — common OCR character confusions."
        ),
        "answer": (
            "Corrected: 'The total amount is {currency}{amount} for the item delivered to {company}.'"
        ),
    },
    {
        "query": (
            "OCR output from {domain} {doc_type}: 'lnvoice #l{qty} dated {month} {year}, "
            "payab1e to {company}, arnount: {currency}{amount}.'\n"
            "Fix OCR errors."
        ),
        "reasoning": (
            "Step 1: 'lnvoice' — lowercase 'l' instead of 'I' → 'Invoice'.\n"
            "Step 2: '#l{qty}' — 'l' confused with '1' in number → '#{qty}' (check context).\n"
            "Step 3: 'payab1e' — '1' for 'l' → 'payable'.\n"
            "Step 4: 'arnount' — 'rn' is OCR for 'm' → 'amount'.\n"
            "Confidence: High — classic l/1/I and rn/m confusions."
        ),
        "answer": (
            "Corrected: 'Invoice #{qty} dated {month} {year}, payable to {company}, amount: {currency}{amount}.'"
        ),
    },
    {
        "query": (
            "OCR from scanned {domain} table:\n"
            "| Oepartment | 8udget | Actua1 |\n"
            "| {dept} | {currency}{amount} | {currency}{arnount2} |\n\n"
            "Correct the OCR errors in this table."
        ),
        "reasoning": (
            "Step 1: 'Oepartment' — 'O' confused with 'D' → 'Department'.\n"
            "Step 2: '8udget' — '8' confused with 'B' → 'Budget'.\n"
            "Step 3: 'Actua1' — '1' confused with 'l' → 'Actual'.\n"
            "Step 4: '{currency}{arnount2}' — if 'arnount' present, rn→m → '{currency}{amount2}'.\n"
            "Confidence: High — header-level OCR corrections with character shape confusion."
        ),
        "answer": (
            "Corrected table:\n"
            "| Department | Budget | Actual |\n"
            "| {dept} | {currency}{amount} | {currency}{amount2} |"
        ),
    },
    {
        "query": (
            "Handwritten note OCR from {domain} context: "
            "'Appr0ved by {person} on {month} {year}. Arnount: {currency}{amount}. "
            "P1ease process irnrnediately.'\n"
            "Interpret and correct."
        ),
        "reasoning": (
            "Step 1: 'Appr0ved' — '0' (zero) for 'o' → 'Approved'.\n"
            "Step 2: 'Arnount' — 'rn' for 'm' → 'Amount'.\n"
            "Step 3: 'P1ease' — '1' for 'l' → 'Please'.\n"
            "Step 4: 'irnrnediately' — double 'rn' for 'mm' → 'immediately'.\n"
            "Step 5: Handwriting context increases uncertainty for similar-looking chars.\n"
            "Confidence: Medium — handwriting adds ambiguity but context resolves it."
        ),
        "answer": (
            "Corrected: 'Approved by {person} on {month} {year}. Amount: {currency}{amount}. "
            "Please process immediately.'"
        ),
    },
    {
        "query": (
            "Misaligned OCR table from {domain} {doc_type}:\n"
            "Oescription     Qty  Price\n"
            "{product}   {qty}   {currency}{amount}\n"
            "  Sh1pping             {currency}{amount3}\n"
            "TOTA L                 {currency}{amount2}\n\n"
            "Recover the table structure and fix errors."
        ),
        "reasoning": (
            "Step 1: 'Oescription' → 'Description' (O/D confusion).\n"
            "Step 2: Misalignment — columns are space-separated, need re-alignment.\n"
            "Step 3: 'Sh1pping' → 'Shipping' (1/i confusion).\n"
            "Step 4: 'TOTA L' — space in word → 'TOTAL'.\n"
            "Step 5: Shipping row has no Qty — it's a fee line.\n"
            "Step 6: Reconstruct aligned table.\n"
            "Confidence: Medium — misalignment requires structure inference."
        ),
        "answer": (
            "Corrected table:\n"
            "| Description | Qty | Price |\n"
            "| {product} | {qty} | {currency}{amount} |\n"
            "| Shipping | — | {currency}{amount3} |\n"
            "| TOTAL | — | {currency}{amount2} |"
        ),
    },
    {
        "query": (
            "OCR text from {domain} {doc_type}: 'C0ntract between {company} (hereinafter "
            "\"Party A\") and G1obex lndustries (hereinafter \"Party 8\"). Effective "
            "date: O1/{month}/{year}.'\n"
            "Correct all OCR errors."
        ),
        "reasoning": (
            "Step 1: 'C0ntract' — '0' for 'o' → 'Contract'.\n"
            "Step 2: 'G1obex' — '1' for 'l' → 'Globex'.\n"
            "Step 3: 'lndustries' — 'l' for 'I' → 'Industries'.\n"
            "Step 4: 'Party 8' — '8' for 'B' → 'Party B'.\n"
            "Step 5: 'O1' — 'O' for '0' and/or '1' for 'l' → '01' (date context).\n"
            "Confidence: High — multiple classic OCR confusions."
        ),
        "answer": (
            "Corrected: 'Contract between {company} (hereinafter \"Party A\") and "
            "Globex Industries (hereinafter \"Party B\"). Effective date: 01/{month}/{year}.'"
        ),
    },
    {
        "query": (
            "OCR from a {domain} {doc_type} with mixed fonts:\n"
            "'CONFI0ENTIAL — {company}\n"
            "Prepared by: {person}, {dept}\n"
            "Oate: {month} {year}\n"
            "Arnount Due: {currency}{amount}\n"
            "Status: APPROVEO'\n\n"
            "Fix all OCR issues."
        ),
        "reasoning": (
            "Step 1: 'CONFI0ENTIAL' — '0' for 'D' → 'CONFIDENTIAL'.\n"
            "Step 2: 'Oate' — 'O' for 'D' → 'Date'.\n"
            "Step 3: 'Arnount' — 'rn' for 'm' → 'Amount'.\n"
            "Step 4: 'APPROVEO' — 'O' for 'D' → 'APPROVED'.\n"
            "Step 5: Mixed fonts (bold headers) cause more O/D confusion in caps.\n"
            "Confidence: High — consistent O/D pattern in uppercase text."
        ),
        "answer": (
            "Corrected:\n"
            "'CONFIDENTIAL — {company}\n"
            "Prepared by: {person}, {dept}\n"
            "Date: {month} {year}\n"
            "Amount Due: {currency}{amount}\n"
            "Status: APPROVED'"
        ),
    },
    {
        "query": (
            "OCR scan of {domain} {doc_type} with degraded quality:\n"
            "'Sch edu1e of Payrnents\n"
            "Mi1estone 1: {currency}{amount} upon s1gning\n"
            "Mi1estone 2: {currency}{amount2} upon de1ivery\n"
            "Mi1estone 3: {currency}{amount3} upon cornp1etion'\n\n"
            "Recover the original text."
        ),
        "reasoning": (
            "Step 1: 'Sch edu1e' — space in word + '1' for 'l' → 'Schedule'.\n"
            "Step 2: 'Payrnents' — 'rn' for 'm' → 'Payments'.\n"
            "Step 3: 'Mi1estone' (x3) — '1' for 'l' → 'Milestone'.\n"
            "Step 4: 's1gning' — '1' for 'i' → 'signing'.\n"
            "Step 5: 'de1ivery' — '1' for 'l' → 'delivery'.\n"
            "Step 6: 'cornp1etion' — 'rn' for 'm' + '1' for 'l' → 'completion'.\n"
            "Confidence: Medium — degraded scan with multiple error types."
        ),
        "answer": (
            "Corrected:\n"
            "'Schedule of Payments\n"
            "Milestone 1: {currency}{amount} upon signing\n"
            "Milestone 2: {currency}{amount2} upon delivery\n"
            "Milestone 3: {currency}{amount3} upon completion'"
        ),
    },
    {
        "query": (
            "OCR from faded {domain} document:\n"
            "'TO: {person}\n"
            "FR0M: {company}, {dept}\n"
            "RE: Purchase 0rder #{qty}\n"
            "P1ease conf1rm rece1pt of {qty} un1ts of {product} "
            "at {currency}{amount} per un1t.'\n\n"
            "Correct OCR errors."
        ),
        "reasoning": (
            "Step 1: 'FR0M' — '0' for 'O' → 'FROM'.\n"
            "Step 2: '0rder' — '0' for 'O' → 'Order'.\n"
            "Step 3: 'P1ease' — '1' for 'l' → 'Please'.\n"
            "Step 4: 'conf1rm' — '1' for 'i' → 'confirm'.\n"
            "Step 5: 'rece1pt' — '1' for 'i' → 'receipt'.\n"
            "Step 6: 'un1ts'/'un1t' — '1' for 'i' → 'units'/'unit'.\n"
            "Confidence: High — consistent 0/O and 1/i/l pattern from fading."
        ),
        "answer": (
            "Corrected:\n"
            "'TO: {person}\n"
            "FROM: {company}, {dept}\n"
            "RE: Purchase Order #{qty}\n"
            "Please confirm receipt of {qty} units of {product} "
            "at {currency}{amount} per unit.'"
        ),
    },
    {
        "query": (
            "OCR of handwritten {domain} note:\n"
            "'Revlew cornpleted for {company}. Recornrnend approva1 "
            "of {currency}{amount}. Need slgnature frorn {person} by EOD.'\n\n"
            "Interpret the handwriting OCR."
        ),
        "reasoning": (
            "Step 1: 'Revlew' — 'l' for 'i' → 'Review'.\n"
            "Step 2: 'cornpleted' — 'rn' for 'm' → 'completed'.\n"
            "Step 3: 'Recornrnend' — double 'rn' for 'mm' → 'Recommend'.\n"
            "Step 4: 'approva1' — '1' for 'l' → 'approval'.\n"
            "Step 5: 'slgnature' — 'l' for 'i' → 'signature'.\n"
            "Step 6: 'frorn' — 'rn' for 'm' → 'from'.\n"
            "Confidence: Medium — handwriting compounds ambiguity."
        ),
        "answer": (
            "Corrected: 'Review completed for {company}. Recommend approval "
            "of {currency}{amount}. Need signature from {person} by EOD.'"
        ),
    },
]

# ===================================================================
# CROSS-DOCUMENT REASONING TEMPLATES
# ===================================================================

_CROSS_DOC_TEMPLATES: List[Dict[str, str]] = [
    {
        "query": (
            "Document 1 — Original Contract ({domain}):\n"
            "'{company} agrees to deliver {product} for {currency}{amount} by {month} {year}.'\n\n"
            "Document 2 — Amendment #1:\n"
            "'The delivery date is extended to {month} {year}. Price revised to {currency}{amount2}.'\n\n"
            "What are the current terms?"
        ),
        "reasoning": (
            "Step 1: Original contract — price={currency}{amount}, deadline={month} {year}.\n"
            "Step 2: Amendment #1 supersedes — new deadline={month} {year}, new price={currency}{amount2}.\n"
            "Step 3: Amendment takes precedence over original for conflicting terms.\n"
            "Step 4: Non-amended terms (product={product}, parties) remain from original.\n"
            "Confidence: High — standard contract amendment precedence."
        ),
        "answer": (
            "Current terms per Amendment #1: {company} delivers {product} for "
            "{currency}{amount2} by {month} {year}. Amendment supersedes original pricing."
        ),
    },
    {
        "query": (
            "Document 1 — Purchase Order from {company}:\n"
            "'PO #{qty}: {row_num} units of {product} at {currency}{amount}/unit.'\n\n"
            "Document 2 — Invoice from supplier:\n"
            "'Invoice for PO #{qty}: {col_num} units of {product} at {currency}{amount2}/unit.'\n\n"
            "Identify discrepancies."
        ),
        "reasoning": (
            "Step 1: PO #{qty} — ordered {row_num} units at {currency}{amount}/unit.\n"
            "Step 2: Invoice references PO #{qty} — billed {col_num} units at {currency}{amount2}/unit.\n"
            "Step 3: Quantity discrepancy: PO={row_num}, Invoice={col_num}.\n"
            "Step 4: Price discrepancy: PO={currency}{amount}, Invoice={currency}{amount2}.\n"
            "Confidence: High — clear numeric mismatches between documents."
        ),
        "answer": (
            "Discrepancies found: (1) Quantity — PO says {row_num} units, invoice says {col_num} units. "
            "(2) Unit price — PO says {currency}{amount}, invoice says {currency}{amount2}."
        ),
    },
    {
        "query": (
            "Document 1 — Resume of {person}:\n"
            "'Experience: {row_num} years in {domain}. Skills: {product}, data analysis. "
            "Education: MBA, {year}.'\n\n"
            "Document 2 — Job Description at {company}:\n"
            "'Required: {col_num}+ years {domain} experience. Must know {product}. MBA preferred.'\n\n"
            "Assess candidate fit."
        ),
        "reasoning": (
            "Step 1: Resume — {row_num} years experience, knows {product}, has MBA ({year}).\n"
            "Step 2: JD requires — {col_num}+ years, {product} knowledge, MBA preferred.\n"
            "Step 3: Experience match: {row_num} years vs {col_num}+ required.\n"
            "Step 4: Skills match: {product} — yes.\n"
            "Step 5: Education: MBA — meets 'preferred' criterion.\n"
            "Confidence: Medium — experience years need numeric comparison."
        ),
        "answer": (
            "{person} has {row_num} years experience (requirement: {col_num}+), "
            "knows {product} (required), and holds an MBA (preferred). "
            "Fit assessment depends on whether {row_num} >= {col_num}."
        ),
    },
    {
        "query": (
            "Document 1 — Company Policy ({company}):\n"
            "'All {domain} expenditures over {currency}{amount} require {dept} approval. "
            "Travel expenses capped at {currency}{amount3} per trip.'\n\n"
            "Document 2 — Compliance Report:\n"
            "'{person} submitted {domain} expense of {currency}{amount2} without {dept} approval. "
            "Travel claim: {currency}{amount} for {month} trip.'\n\n"
            "Identify compliance violations."
        ),
        "reasoning": (
            "Step 1: Policy — expenditures > {currency}{amount} need {dept} approval; travel cap = {currency}{amount3}.\n"
            "Step 2: {person}'s expense = {currency}{amount2} — check if > {currency}{amount} threshold.\n"
            "Step 3: If {currency}{amount2} > {currency}{amount}: violation (no {dept} approval obtained).\n"
            "Step 4: Travel claim = {currency}{amount} — check against {currency}{amount3} cap.\n"
            "Confidence: Medium — requires numeric threshold comparisons across documents."
        ),
        "answer": (
            "Potential violations: (1) {person}'s expense of {currency}{amount2} may exceed the "
            "{currency}{amount} threshold requiring {dept} approval. (2) Travel claim of "
            "{currency}{amount} should be checked against the {currency}{amount3} per-trip cap."
        ),
    },
    {
        "query": (
            "Document 1 — Lease Agreement:\n"
            "'{company} leases office space at {currency}{amount}/month for {row_num} years "
            "starting {month} {year}. Annual escalation: {pct}%.'\n\n"
            "Document 2 — Renewal Notice:\n"
            "'Lease renewal for {company}: new rate {currency}{amount2}/month, term {col_num} years.'\n\n"
            "Document 3 — {dept} Budget:\n"
            "'Allocated {currency}{amount3} annually for office lease.'\n\n"
            "Is the budget sufficient for the renewal?"
        ),
        "reasoning": (
            "Step 1: Original lease — {currency}{amount}/month, {row_num} years, {pct}% escalation.\n"
            "Step 2: Renewal — {currency}{amount2}/month, {col_num} years.\n"
            "Step 3: Annual renewal cost = {currency}{amount2} x 12.\n"
            "Step 4: Budget allocation = {currency}{amount3}/year.\n"
            "Step 5: Compare annual renewal cost vs budget.\n"
            "Confidence: Medium — requires cross-document calculation."
        ),
        "answer": (
            "Annual renewal cost = {currency}{amount2} x 12 months. "
            "Budget = {currency}{amount3}. Sufficiency depends on whether "
            "annual cost <= {currency}{amount3}."
        ),
    },
    {
        "query": (
            "Document 1 — Insurance Policy:\n"
            "'{company} policy covers {domain} liability up to {currency}{amount}. "
            "Deductible: {currency}{amount3}. Effective through {month} {year}.'\n\n"
            "Document 2 — Claim Filing:\n"
            "'{person} filed claim for {currency}{amount2} on behalf of {company}. "
            "Incident date: {month} {year}. Category: {domain} liability.'\n\n"
            "Will the claim be covered?"
        ),
        "reasoning": (
            "Step 1: Policy — coverage up to {currency}{amount}, deductible {currency}{amount3}, "
            "effective through {month} {year}.\n"
            "Step 2: Claim — {currency}{amount2}, incident {month} {year}, category {domain} liability.\n"
            "Step 3: Date check — is incident within policy period?\n"
            "Step 4: Amount check — {currency}{amount2} vs coverage limit {currency}{amount}.\n"
            "Step 5: Payout = min(claim, coverage) - deductible.\n"
            "Confidence: Medium — multiple cross-document conditions to verify."
        ),
        "answer": (
            "Claim of {currency}{amount2} against coverage limit of {currency}{amount} with "
            "deductible {currency}{amount3}. Coverage depends on: (1) incident date within policy period, "
            "(2) claim amount vs limit. Payout = min(claim, limit) - deductible."
        ),
    },
    {
        "query": (
            "Document 1 — {domain} Audit Report:\n"
            "'{company} {dept} audit found {row_num} non-conformances. "
            "Critical: {col_num}. Corrective action deadline: {month} {year}.'\n\n"
            "Document 2 — Corrective Action Plan:\n"
            "'{person} submitted plan addressing {col_num} critical findings. "
            "Estimated completion: {month} {year}. Budget: {currency}{amount}.'\n\n"
            "Are all critical findings addressed?"
        ),
        "reasoning": (
            "Step 1: Audit found {row_num} total non-conformances, {col_num} critical.\n"
            "Step 2: Corrective plan addresses {col_num} findings — matches critical count.\n"
            "Step 3: Deadline alignment — audit deadline {month} {year} vs plan completion {month} {year}.\n"
            "Step 4: Remaining non-critical = {row_num} - {col_num} — not addressed in plan.\n"
            "Confidence: Medium — count matching across documents."
        ),
        "answer": (
            "The corrective action plan addresses {col_num} critical findings (matching audit count). "
            "However, {row_num} - {col_num} non-critical findings may not be covered. "
            "Timeline alignment needs verification."
        ),
    },
    {
        "query": (
            "Document 1 — Statement of Work:\n"
            "'{company} will deliver {product} in {row_num} phases. "
            "Total value: {currency}{amount}. Acceptance criteria in Exhibit A.'\n\n"
            "Document 2 — Deliverable Acceptance Report:\n"
            "'Phase 1 accepted by {person}. Phase 2 rejected — {product} "
            "failed performance test. Remediation needed.'\n\n"
            "What is the project status?"
        ),
        "reasoning": (
            "Step 1: SOW — {row_num} phases, total {currency}{amount}.\n"
            "Step 2: Acceptance report — Phase 1 accepted, Phase 2 rejected.\n"
            "Step 3: Completion = 1/{row_num} phases fully accepted.\n"
            "Step 4: Phase 2 blocked — {product} performance failure requires remediation.\n"
            "Step 5: Remaining phases ({row_num} - 2) not yet started or reported.\n"
            "Confidence: High — clear status from acceptance report."
        ),
        "answer": (
            "Project status: 1 of {row_num} phases accepted. Phase 2 rejected "
            "({product} performance failure). Remediation required before proceeding. "
            "Total contract value: {currency}{amount}."
        ),
    },
    {
        "query": (
            "Document 1 — NDA between {company} and Globex Industries:\n"
            "'Confidential information includes {domain} data, {product} specifications, "
            "and pricing. Duration: {row_num} years from {month} {year}.'\n\n"
            "Document 2 — Email from {person}:\n"
            "'Shared {product} pricing details with external consultant on {month} {year}.'\n\n"
            "Is there a potential NDA breach?"
        ),
        "reasoning": (
            "Step 1: NDA covers {product} specifications and pricing for {row_num} years from {month} {year}.\n"
            "Step 2: Email — {person} shared {product} pricing with external party.\n"
            "Step 3: Check if sharing date is within NDA duration.\n"
            "Step 4: External consultant may not be a covered party under the NDA.\n"
            "Step 5: Pricing is explicitly listed as confidential information.\n"
            "Confidence: High — pricing is clearly covered; breach likely if within duration."
        ),
        "answer": (
            "Potential NDA breach: {person} shared {product} pricing (explicitly confidential) "
            "with an external consultant. If the sharing date falls within the {row_num}-year "
            "NDA period starting {month} {year}, this constitutes a violation."
        ),
    },
    {
        "query": (
            "Document 1 — {company} Procurement Policy:\n"
            "'All purchases over {currency}{amount3} require 3 competitive bids. "
            "{dept} must approve vendors.'\n\n"
            "Document 2 — Purchase Record:\n"
            "'Vendor: Initech Solutions. Amount: {currency}{amount}. "
            "Approved by: {person}. Bids received: 1.'\n\n"
            "Identify policy violations."
        ),
        "reasoning": (
            "Step 1: Policy — purchases > {currency}{amount3} need 3 bids + {dept} approval.\n"
            "Step 2: Purchase — {currency}{amount}, 1 bid only, approved by {person}.\n"
            "Step 3: If {currency}{amount} > {currency}{amount3}: bid requirement triggered.\n"
            "Step 4: Only 1 bid received — violation (need 3).\n"
            "Step 5: Check if {person} is in {dept} — approver authorization unclear.\n"
            "Confidence: High — clear bid count violation."
        ),
        "answer": (
            "Policy violations: (1) Only 1 bid received for {currency}{amount} purchase "
            "(policy requires 3 bids for amounts over {currency}{amount3}). "
            "(2) Verify {person}'s authority — {dept} approval required."
        ),
    },
]

# ===================================================================
# PUBLIC API
# ===================================================================


def generate_table_examples(count: int = 8000, *, seed: int = 42) -> List[Dict[str, str]]:
    """Generate table understanding SFT examples across three difficulty tiers.

    Distribution: ~37.5% simple, ~37.5% medium, ~25% hard.
    """
    rng = random.Random(seed)
    simple_n = int(count * 0.375)
    medium_n = int(count * 0.375)
    hard_n = count - simple_n - medium_n

    simple = _expand_templates(_TABLE_SIMPLE_TEMPLATES, simple_n, rng)
    medium = _expand_templates(_TABLE_MEDIUM_TEMPLATES, medium_n, rng)
    hard = _expand_templates(_TABLE_HARD_TEMPLATES, hard_n, rng)

    combined = simple + medium + hard
    rng.shuffle(combined)
    return combined


def generate_layout_examples(count: int = 5000, *, seed: int = 43) -> List[Dict[str, str]]:
    """Generate layout comprehension SFT examples."""
    rng = random.Random(seed)
    return _expand_templates(_LAYOUT_TEMPLATES, count, rng)


def generate_ocr_examples(count: int = 4000, *, seed: int = 44) -> List[Dict[str, str]]:
    """Generate OCR correction SFT examples."""
    rng = random.Random(seed)
    return _expand_templates(_OCR_TEMPLATES, count, rng)


def generate_cross_ref_examples(count: int = 3000, *, seed: int = 45) -> List[Dict[str, str]]:
    """Generate cross-document reasoning SFT examples."""
    rng = random.Random(seed)
    return _expand_templates(_CROSS_DOC_TEMPLATES, count, rng)


def generate_phase2_data(output_dir: Path, scale: float = 1.0) -> Dict[str, int]:
    """Generate all Phase 2 document intelligence training data.

    Args:
        output_dir: Directory to write JSONL files into.
        scale: Scaling factor (1.0 = 20K total examples).

    Returns:
        Dict mapping category name to number of examples written.
    """
    output_dir = Path(output_dir)
    stats: Dict[str, int] = {}

    categories = [
        ("table_understanding", generate_table_examples, int(8000 * scale)),
        ("layout_comprehension", generate_layout_examples, int(5000 * scale)),
        ("ocr_correction", generate_ocr_examples, int(4000 * scale)),
        ("cross_document_reasoning", generate_cross_ref_examples, int(3000 * scale)),
    ]

    for name, gen_fn, count in categories:
        if count < 1:
            count = 1
        examples = gen_fn(count=count)
        path = output_dir / f"phase2_{name}.jsonl"
        with JSONLWriter(path) as writer:
            for ex in examples:
                writer.write(ex)
        stats[name] = len(examples)

    return stats
