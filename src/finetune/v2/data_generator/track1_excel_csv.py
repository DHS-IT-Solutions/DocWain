"""Track 1 — Excel/CSV Intelligence data generator for DocWain V2+ SFT.

Generates 2500 training examples across nine categories:
  - Single-sheet tabular QA            (400)
  - Multi-sheet reasoning              (350)
  - Formula-aware understanding        (300)
  - Merged cell & named range handling (250)
  - CSV delimiter detection            (200)
  - Large spreadsheet chunking         (200)
  - Data type inference                (250)
  - Spreadsheet-to-insight             (300)
  - Negatives & edge cases             (250)

Each example uses the ``<spreadsheet>`` XML context format and includes
chain-of-thought reasoning via ``format_sft_example``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

from src.finetune.v2.data_generator.base import (
    DOMAINS,
    DOC_TYPES,
    JSONLWriter,
    format_sft_example,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_CURRENCIES = ["$", "EUR ", "GBP ", "JPY ", "AUD ", "CAD ", "CHF ", "INR "]
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
_QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_DELIMITERS = [
    ("tab", "\\t", "\t"),
    ("semicolon", ";", ";"),
    ("pipe", "|", "|"),
    ("comma", ",", ","),
]
_DATA_TYPES = ["date", "currency", "percentage", "phone_number", "integer", "decimal"]
_FORMULAS = ["SUM", "AVERAGE", "VLOOKUP", "COUNT", "MAX", "MIN", "IF", "COUNTIF"]
_SHEET_NAMES = [
    "Summary", "Revenue", "Expenses", "Headcount", "Forecast",
    "Raw Data", "Pivot", "Dashboard", "YoY Comparison", "Assumptions",
    "Regional", "Products", "Customers", "Inventory", "Transactions",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick(lst: list, rng: random.Random) -> Any:
    return rng.choice(lst)


def _rand_amount(rng: random.Random, lo: float = 100.0, hi: float = 99999.0) -> str:
    return f"{rng.uniform(lo, hi):,.2f}"


def _rand_int(rng: random.Random, lo: int = 1, hi: int = 500) -> int:
    return rng.randint(lo, hi)


def _rand_pct(rng: random.Random) -> str:
    return f"{rng.uniform(0.5, 55.0):.1f}"


def _make_filename(rng: random.Random, ext: str = "xlsx") -> str:
    company = _pick(_COMPANY_NAMES, rng).replace(" ", "_")
    year = rng.randint(2020, 2026)
    quarter = _pick(_QUARTERS, rng)
    return f"{company}_{quarter}_{year}.{ext}"


def _spreadsheet_ctx(filename: str, sheets: List[Dict[str, str]]) -> str:
    """Build a <spreadsheet> XML block from sheet dicts."""
    parts = [f'<spreadsheet source="{filename}">']
    for s in sheets:
        parts.append(
            f'  <sheet name="{s["name"]}" rows="{s["rows"]}" cols="{s["cols"]}">'
        )
        parts.append(f'    <headers>{s["headers"]}</headers>')
        parts.append(f"    <sample_rows>")
        for row in s["sample_rows"]:
            parts.append(f"      {row}")
        parts.append(f"    </sample_rows>")
        parts.append(f'    <summary>{s["summary"]}</summary>')
        parts.append(f"  </sheet>")
    parts.append("</spreadsheet>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Category 1: Single-sheet tabular QA (400)
# ---------------------------------------------------------------------------

def _gen_single_sheet_qa(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    templates = [
        # Revenue lookup
        lambda: _single_sheet_revenue(rng),
        # Row count
        lambda: _single_sheet_row_count(rng),
        # Filtering
        lambda: _single_sheet_filter(rng),
        # Aggregation
        lambda: _single_sheet_aggregate(rng),
        # Specific cell
        lambda: _single_sheet_cell(rng),
    ]
    idx = 0
    while len(results) < count:
        fn = templates[idx % len(templates)]
        results.append(fn())
        idx += 1
    return results


def _single_sheet_revenue(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng)
    company = _pick(_COMPANY_NAMES, rng)
    quarter = _pick(_QUARTERS, rng)
    cur = _pick(_CURRENCIES, rng)
    products = rng.sample(_PRODUCT_NAMES, k=min(4, len(_PRODUCT_NAMES)))
    amounts = [rng.uniform(5000, 80000) for _ in products]
    total = sum(amounts)
    rows = [f"{p} | {cur}{a:,.2f}" for p, a in zip(products, amounts)]
    sheet = {
        "name": "Revenue",
        "rows": str(len(products)),
        "cols": "2",
        "headers": "Product | Revenue",
        "sample_rows": rows,
        "summary": f"{len(products)} rows, 2 columns. Total revenue: {cur}{total:,.2f}",
    }
    ctx = _spreadsheet_ctx(fname, [sheet])
    query = f"{ctx}\n\nWhat is the total revenue in {quarter} from {company}'s spreadsheet?"
    reasoning = (
        f"Step 1: The spreadsheet '{fname}' has one sheet 'Revenue' with {len(products)} product rows.\n"
        f"Step 2: I need to sum the Revenue column across all rows.\n"
        f"Step 3: Sum = {' + '.join(f'{cur}{a:,.2f}' for a in amounts)} = {cur}{total:,.2f}.\n"
        f"Confidence: High -- direct column summation from a single sheet."
    )
    answer = f"The total revenue is **{cur}{total:,.2f}** across {len(products)} products."
    return format_sft_example(query, reasoning, answer)


def _single_sheet_row_count(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng)
    dept = _pick(_DEPARTMENTS, rng)
    n_rows = rng.randint(15, 200)
    names = rng.sample(_PERSON_NAMES, k=min(3, len(_PERSON_NAMES)))
    rows = [f"{n} | {dept} | Active" for n in names]
    sheet = {
        "name": "Employees",
        "rows": str(n_rows),
        "cols": "3",
        "headers": "Name | Department | Status",
        "sample_rows": rows + ["... (additional rows)"],
        "summary": f"{n_rows} rows, 3 columns. Departments represented: {dept} and others.",
    }
    ctx = _spreadsheet_ctx(fname, [sheet])
    query = f"{ctx}\n\nHow many employee records are in this spreadsheet?"
    reasoning = (
        f"Step 1: The sheet 'Employees' metadata says rows=\"{n_rows}\".\n"
        f"Step 2: Each row represents one employee record.\n"
        f"Step 3: Total employee records = {n_rows}.\n"
        f"Confidence: High -- row count from metadata."
    )
    answer = f"There are **{n_rows}** employee records in the spreadsheet."
    return format_sft_example(query, reasoning, answer)


def _single_sheet_filter(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng)
    cur = _pick(_CURRENCIES, rng)
    threshold = rng.randint(10000, 50000)
    products = rng.sample(_PRODUCT_NAMES, k=min(5, len(_PRODUCT_NAMES)))
    amounts = [rng.uniform(1000, 80000) for _ in products]
    above = [(p, a) for p, a in zip(products, amounts) if a > threshold]
    rows = [f"{p} | {cur}{a:,.2f}" for p, a in zip(products, amounts)]
    sheet = {
        "name": "Sales",
        "rows": str(len(products)),
        "cols": "2",
        "headers": "Product | Amount",
        "sample_rows": rows,
        "summary": f"{len(products)} rows, 2 columns.",
    }
    ctx = _spreadsheet_ctx(fname, [sheet])
    query = f"{ctx}\n\nWhich products have sales above {cur}{threshold:,}?"
    above_list = ", ".join(p for p, _ in above) if above else "None"
    reasoning = (
        f"Step 1: Filter the 'Amount' column for values > {cur}{threshold:,}.\n"
        f"Step 2: Check each row:\n"
        + "\n".join(f"  - {p}: {cur}{a:,.2f} {'> ' if a > threshold else '<= '}{cur}{threshold:,}" for p, a in zip(products, amounts))
        + f"\nStep 3: Products above threshold: {above_list}.\n"
        f"Confidence: High -- straightforward comparison."
    )
    answer = f"Products with sales above {cur}{threshold:,}: **{above_list}**."
    return format_sft_example(query, reasoning, answer)


def _single_sheet_aggregate(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng)
    cur = _pick(_CURRENCIES, rng)
    dept = _pick(_DEPARTMENTS, rng)
    months = rng.sample(_MONTHS, k=4)
    values = [rng.uniform(5000, 50000) for _ in months]
    avg = sum(values) / len(values)
    rows = [f"{dept} | {m} | {cur}{v:,.2f}" for m, v in zip(months, values)]
    sheet = {
        "name": "Budget",
        "rows": str(len(months)),
        "cols": "3",
        "headers": "Department | Month | Spend",
        "sample_rows": rows,
        "summary": f"{len(months)} rows, 3 columns. Average spend: {cur}{avg:,.2f}",
    }
    ctx = _spreadsheet_ctx(fname, [sheet])
    query = f"{ctx}\n\nWhat is the average monthly spend for {dept}?"
    reasoning = (
        f"Step 1: Filter rows where Department = '{dept}'.\n"
        f"Step 2: Collect Spend values: {', '.join(f'{cur}{v:,.2f}' for v in values)}.\n"
        f"Step 3: Average = ({' + '.join(f'{v:,.2f}' for v in values)}) / {len(values)} = {cur}{avg:,.2f}.\n"
        f"Confidence: High -- simple arithmetic mean."
    )
    answer = f"The average monthly spend for {dept} is **{cur}{avg:,.2f}**."
    return format_sft_example(query, reasoning, answer)


def _single_sheet_cell(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng)
    person = _pick(_PERSON_NAMES, rng)
    dept = _pick(_DEPARTMENTS, rng)
    year = rng.randint(2020, 2026)
    salary = rng.uniform(40000, 150000)
    cur = _pick(_CURRENCIES, rng)
    sheet = {
        "name": "Payroll",
        "rows": "50",
        "cols": "4",
        "headers": "Employee | Department | Year | Salary",
        "sample_rows": [
            f"{person} | {dept} | {year} | {cur}{salary:,.2f}",
            "... (additional rows)",
        ],
        "summary": f"50 rows, 4 columns. Payroll data for {year}.",
    }
    ctx = _spreadsheet_ctx(fname, [sheet])
    query = f"{ctx}\n\nWhat is {person}'s salary?"
    reasoning = (
        f"Step 1: Look up '{person}' in the Payroll sheet.\n"
        f"Step 2: Read the Salary column: {cur}{salary:,.2f}.\n"
        f"Confidence: High -- direct cell lookup."
    )
    answer = f"{person}'s salary is **{cur}{salary:,.2f}**."
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Category 2: Multi-sheet reasoning (350)
# ---------------------------------------------------------------------------

def _gen_multi_sheet(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    templates = [
        lambda: _multi_sheet_cross_ref(rng),
        lambda: _multi_sheet_join(rng),
        lambda: _multi_sheet_compare(rng),
    ]
    idx = 0
    while len(results) < count:
        fn = templates[idx % len(templates)]
        results.append(fn())
        idx += 1
    return results


def _multi_sheet_cross_ref(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng)
    cur = _pick(_CURRENCIES, rng)
    products = rng.sample(_PRODUCT_NAMES, k=3)
    revenues = [rng.uniform(10000, 90000) for _ in products]
    costs = [rng.uniform(5000, r * 0.8) for r in revenues]
    profits = [r - c for r, c in zip(revenues, costs)]
    best_idx = profits.index(max(profits))
    sheet1 = {
        "name": "Revenue",
        "rows": "3", "cols": "2",
        "headers": "Product | Revenue",
        "sample_rows": [f"{p} | {cur}{r:,.2f}" for p, r in zip(products, revenues)],
        "summary": "3 rows, 2 columns. Product revenue.",
    }
    sheet2 = {
        "name": "Costs",
        "rows": "3", "cols": "2",
        "headers": "Product | Cost",
        "sample_rows": [f"{p} | {cur}{c:,.2f}" for p, c in zip(products, costs)],
        "summary": "3 rows, 2 columns. Product costs.",
    }
    ctx = _spreadsheet_ctx(fname, [sheet1, sheet2])
    query = f"{ctx}\n\nWhich product has the highest profit margin? (Profit = Revenue - Cost)"
    reasoning = (
        f"Step 1: Cross-reference 'Revenue' and 'Costs' sheets by Product.\n"
        + "\n".join(
            f"Step {i+2}: {p}: Revenue {cur}{r:,.2f} - Cost {cur}{c:,.2f} = Profit {cur}{pr:,.2f}"
            for i, (p, r, c, pr) in enumerate(zip(products, revenues, costs, profits))
        )
        + f"\nStep {len(products)+2}: Highest profit: {products[best_idx]} at {cur}{profits[best_idx]:,.2f}.\n"
        f"Confidence: High -- straightforward cross-sheet subtraction."
    )
    answer = (
        f"**{products[best_idx]}** has the highest profit at {cur}{profits[best_idx]:,.2f} "
        f"(Revenue {cur}{revenues[best_idx]:,.2f} - Cost {cur}{costs[best_idx]:,.2f})."
    )
    return format_sft_example(query, reasoning, answer)


def _multi_sheet_join(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng)
    person = _pick(_PERSON_NAMES, rng)
    dept = _pick(_DEPARTMENTS, rng)
    cur = _pick(_CURRENCIES, rng)
    salary = rng.uniform(50000, 120000)
    bonus_pct = rng.uniform(5, 25)
    bonus = salary * bonus_pct / 100
    sheet1 = {
        "name": "Employees",
        "rows": "30", "cols": "3",
        "headers": "Employee ID | Name | Department",
        "sample_rows": [f"EMP-{rng.randint(100,999)} | {person} | {dept}"],
        "summary": "30 rows, 3 columns. Employee directory.",
    }
    sheet2 = {
        "name": "Compensation",
        "rows": "30", "cols": "3",
        "headers": "Name | Base Salary | Bonus %",
        "sample_rows": [f"{person} | {cur}{salary:,.2f} | {bonus_pct:.1f}%"],
        "summary": "30 rows, 3 columns. Compensation details.",
    }
    ctx = _spreadsheet_ctx(fname, [sheet1, sheet2])
    query = f"{ctx}\n\nWhat is {person}'s total compensation including bonus?"
    reasoning = (
        f"Step 1: Find {person} in 'Employees' sheet -- department: {dept}.\n"
        f"Step 2: Look up {person} in 'Compensation' sheet -- Base: {cur}{salary:,.2f}, Bonus: {bonus_pct:.1f}%.\n"
        f"Step 3: Bonus amount = {cur}{salary:,.2f} x {bonus_pct:.1f}% = {cur}{bonus:,.2f}.\n"
        f"Step 4: Total = {cur}{salary:,.2f} + {cur}{bonus:,.2f} = {cur}{salary + bonus:,.2f}.\n"
        f"Confidence: High -- cross-sheet join on Name, then simple calculation."
    )
    answer = f"{person}'s total compensation is **{cur}{salary + bonus:,.2f}** (base {cur}{salary:,.2f} + bonus {cur}{bonus:,.2f})."
    return format_sft_example(query, reasoning, answer)


def _multi_sheet_compare(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng)
    cur = _pick(_CURRENCIES, rng)
    dept = _pick(_DEPARTMENTS, rng)
    y1, y2 = rng.randint(2021, 2023), rng.randint(2024, 2026)
    budget_y1 = rng.uniform(100000, 500000)
    budget_y2 = rng.uniform(100000, 500000)
    change_pct = ((budget_y2 - budget_y1) / budget_y1) * 100
    direction = "increase" if change_pct > 0 else "decrease"
    sheet1 = {
        "name": str(y1),
        "rows": "8", "cols": "2",
        "headers": "Department | Budget",
        "sample_rows": [f"{dept} | {cur}{budget_y1:,.2f}"],
        "summary": f"8 rows, 2 columns. FY{y1} budget.",
    }
    sheet2 = {
        "name": str(y2),
        "rows": "8", "cols": "2",
        "headers": "Department | Budget",
        "sample_rows": [f"{dept} | {cur}{budget_y2:,.2f}"],
        "summary": f"8 rows, 2 columns. FY{y2} budget.",
    }
    ctx = _spreadsheet_ctx(fname, [sheet1, sheet2])
    query = f"{ctx}\n\nHow did {dept}'s budget change from {y1} to {y2}?"
    reasoning = (
        f"Step 1: From sheet '{y1}' -- {dept} budget: {cur}{budget_y1:,.2f}.\n"
        f"Step 2: From sheet '{y2}' -- {dept} budget: {cur}{budget_y2:,.2f}.\n"
        f"Step 3: Change = {cur}{budget_y2:,.2f} - {cur}{budget_y1:,.2f} = {cur}{budget_y2 - budget_y1:,.2f}.\n"
        f"Step 4: Percentage change = {change_pct:+.1f}%.\n"
        f"Confidence: High -- direct comparison across two year-specific sheets."
    )
    answer = (
        f"{dept}'s budget had a **{abs(change_pct):.1f}% {direction}** from {y1} to {y2}, "
        f"going from {cur}{budget_y1:,.2f} to {cur}{budget_y2:,.2f}."
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Category 3: Formula-aware understanding (300)
# ---------------------------------------------------------------------------

def _gen_formula_aware(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    templates = [
        lambda: _formula_sum(rng),
        lambda: _formula_average(rng),
        lambda: _formula_vlookup(rng),
        lambda: _formula_if(rng),
    ]
    idx = 0
    while len(results) < count:
        fn = templates[idx % len(templates)]
        results.append(fn())
        idx += 1
    return results


def _formula_sum(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng)
    cur = _pick(_CURRENCIES, rng)
    items = rng.sample(_PRODUCT_NAMES, k=4)
    values = [rng.uniform(1000, 30000) for _ in items]
    total = sum(values)
    rows = [f"{item} | {cur}{v:,.2f}" for item, v in zip(items, values)]
    rows.append(f"Total | =SUM(B2:B{len(items)+1}) -> {cur}{total:,.2f}")
    sheet = {
        "name": "Invoice",
        "rows": str(len(items) + 1), "cols": "2",
        "headers": "Item | Amount",
        "sample_rows": rows,
        "summary": f"{len(items)} line items plus SUM total row. Total: {cur}{total:,.2f}",
    }
    ctx = _spreadsheet_ctx(fname, [sheet])
    query = f"{ctx}\n\nThe Total row uses =SUM(B2:B{len(items)+1}). What value does this formula compute and is it correct?"
    reasoning = (
        f"Step 1: The formula =SUM(B2:B{len(items)+1}) sums cells B2 through B{len(items)+1}.\n"
        f"Step 2: Values in range: {', '.join(f'{cur}{v:,.2f}' for v in values)}.\n"
        f"Step 3: Manual sum = {cur}{total:,.2f}.\n"
        f"Step 4: Displayed value matches: {cur}{total:,.2f}. Formula is correct.\n"
        f"Confidence: High -- verified by manual addition."
    )
    answer = (
        f"The SUM formula computes **{cur}{total:,.2f}**, which is correct. "
        f"It adds {len(items)} line item amounts."
    )
    return format_sft_example(query, reasoning, answer)


def _formula_average(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng)
    dept = _pick(_DEPARTMENTS, rng)
    scores = [rng.randint(60, 100) for _ in range(5)]
    avg = sum(scores) / len(scores)
    names = rng.sample(_PERSON_NAMES, k=5)
    rows = [f"{n} | {s}" for n, s in zip(names, scores)]
    rows.append(f"Average | =AVERAGE(B2:B6) -> {avg:.1f}")
    sheet = {
        "name": "Performance",
        "rows": "6", "cols": "2",
        "headers": "Employee | Score",
        "sample_rows": rows,
        "summary": f"5 employees + average row. Dept: {dept}.",
    }
    ctx = _spreadsheet_ctx(fname, [sheet])
    query = f"{ctx}\n\nWhat does the AVERAGE formula in the Performance sheet compute?"
    reasoning = (
        f"Step 1: =AVERAGE(B2:B6) averages the Score column for 5 employees.\n"
        f"Step 2: Scores: {', '.join(str(s) for s in scores)}.\n"
        f"Step 3: Average = ({' + '.join(str(s) for s in scores)}) / 5 = {avg:.1f}.\n"
        f"Confidence: High -- verified arithmetic."
    )
    answer = f"The AVERAGE formula computes **{avg:.1f}**, the mean of {len(scores)} employee scores."
    return format_sft_example(query, reasoning, answer)


def _formula_vlookup(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng)
    cur = _pick(_CURRENCIES, rng)
    product = _pick(_PRODUCT_NAMES, rng)
    price = rng.uniform(50, 5000)
    products = rng.sample(_PRODUCT_NAMES, k=5)
    if product not in products:
        products[0] = product
    prices = {p: rng.uniform(50, 5000) for p in products}
    prices[product] = price
    rows = [f"{p} | {cur}{pr:,.2f}" for p, pr in prices.items()]
    sheet1 = {
        "name": "Prices",
        "rows": "5", "cols": "2",
        "headers": "Product | Unit Price",
        "sample_rows": rows,
        "summary": "5 products with unit prices.",
    }
    qty = rng.randint(5, 100)
    sheet2 = {
        "name": "Orders",
        "rows": "1", "cols": "3",
        "headers": "Product | Qty | Price (looked up)",
        "sample_rows": [f"{product} | {qty} | =VLOOKUP(A2,Prices!A:B,2,FALSE) -> {cur}{price:,.2f}"],
        "summary": "1 order row with VLOOKUP price reference.",
    }
    ctx = _spreadsheet_ctx(fname, [sheet1, sheet2])
    query = f"{ctx}\n\nExplain how the VLOOKUP formula retrieves the price for {product} and compute the order total."
    total = price * qty
    reasoning = (
        f"Step 1: VLOOKUP(A2, Prices!A:B, 2, FALSE) looks up the value in cell A2 ('{product}') "
        f"in column A of the 'Prices' sheet and returns column 2 (Unit Price).\n"
        f"Step 2: Match found: {product} -> {cur}{price:,.2f}.\n"
        f"Step 3: Order total = Qty x Price = {qty} x {cur}{price:,.2f} = {cur}{total:,.2f}.\n"
        f"Confidence: High -- exact match VLOOKUP with FALSE parameter."
    )
    answer = (
        f"The VLOOKUP retrieves **{cur}{price:,.2f}** for {product} from the Prices sheet. "
        f"Order total: {qty} x {cur}{price:,.2f} = **{cur}{total:,.2f}**."
    )
    return format_sft_example(query, reasoning, answer)


def _formula_if(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng)
    cur = _pick(_CURRENCIES, rng)
    threshold = rng.randint(10000, 50000)
    names = rng.sample(_PERSON_NAMES, k=4)
    sales = [rng.uniform(5000, 70000) for _ in names]
    statuses = ["Over Target" if s > threshold else "Under Target" for s in sales]
    rows = [
        f"{n} | {cur}{s:,.2f} | =IF(B{i+2}>{threshold},\"Over Target\",\"Under Target\") -> {st}"
        for i, (n, s, st) in enumerate(zip(names, sales, statuses))
    ]
    sheet = {
        "name": "Sales Review",
        "rows": str(len(names)), "cols": "3",
        "headers": "Rep | Sales | Status",
        "sample_rows": rows,
        "summary": f"{len(names)} reps. Target: {cur}{threshold:,}.",
    }
    ctx = _spreadsheet_ctx(fname, [sheet])
    query = f"{ctx}\n\nHow does the IF formula determine each rep's status?"
    over = [n for n, s in zip(names, sales) if s > threshold]
    under = [n for n, s in zip(names, sales) if s <= threshold]
    reasoning = (
        f"Step 1: The IF formula checks if Sales (col B) > {cur}{threshold:,}.\n"
        f"Step 2: If true, returns 'Over Target'; otherwise 'Under Target'.\n"
        f"Step 3: Results:\n"
        + "\n".join(f"  - {n}: {cur}{s:,.2f} -> {st}" for n, s, st in zip(names, sales, statuses))
        + f"\nStep 4: {len(over)} reps over target, {len(under)} under.\n"
        f"Confidence: High -- deterministic conditional logic."
    )
    answer = (
        f"The IF formula compares each rep's sales to the {cur}{threshold:,} target. "
        f"Over target: {', '.join(over) if over else 'None'}. "
        f"Under target: {', '.join(under) if under else 'None'}."
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Category 4: Merged cell & named range handling (250)
# ---------------------------------------------------------------------------

def _gen_merged_cells(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    templates = [lambda: _merged_header(rng), lambda: _named_range(rng)]
    idx = 0
    while len(results) < count:
        fn = templates[idx % len(templates)]
        results.append(fn())
        idx += 1
    return results


def _merged_header(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng)
    cur = _pick(_CURRENCIES, rng)
    q1, q2 = "Q1", "Q2"
    year = rng.randint(2022, 2026)
    depts = rng.sample(_DEPARTMENTS, k=3)
    q1_vals = [rng.uniform(20000, 100000) for _ in depts]
    q2_vals = [rng.uniform(20000, 100000) for _ in depts]
    header_row = f"[merged: {year} Budget] | | | |"
    sub_header = "Department | Q1 Budget | Q1 Actual | Q2 Budget | Q2 Actual"
    rows = [
        f"{d} | {cur}{v1:,.2f} | {cur}{rng.uniform(v1*0.8, v1*1.2):,.2f} | {cur}{v2:,.2f} | {cur}{rng.uniform(v2*0.8, v2*1.2):,.2f}"
        for d, v1, v2 in zip(depts, q1_vals, q2_vals)
    ]
    sheet = {
        "name": "Budget Summary",
        "rows": str(len(depts) + 2), "cols": "5",
        "headers": f"{header_row}\n    {sub_header}",
        "sample_rows": rows,
        "summary": f"Merged header spanning '{year} Budget' across 5 columns. {len(depts)} department rows.",
    }
    ctx = _spreadsheet_ctx(fname, [sheet])
    target_dept = _pick(depts, rng)
    query = f"{ctx}\n\nThe header row has a merged cell spanning all columns. What is the Q1 Budget for {target_dept}?"
    dept_idx = depts.index(target_dept)
    val = q1_vals[dept_idx]
    reasoning = (
        f"Step 1: The first row is a merged header '{year} Budget' spanning columns A-E.\n"
        f"Step 2: The actual column headers are in row 2: Department, Q1 Budget, Q1 Actual, Q2 Budget, Q2 Actual.\n"
        f"Step 3: Locate '{target_dept}' row and read 'Q1 Budget' column: {cur}{val:,.2f}.\n"
        f"Confidence: High -- merged header does not affect data row interpretation."
    )
    answer = f"The Q1 Budget for {target_dept} is **{cur}{val:,.2f}** (under the merged '{year} Budget' header)."
    return format_sft_example(query, reasoning, answer)


def _named_range(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng)
    cur = _pick(_CURRENCIES, rng)
    range_name = _pick(["TaxRate", "DiscountPct", "ExchangeRate", "InflationAdj", "OverheadMultiplier"], rng)
    range_val = rng.uniform(0.01, 0.35)
    base = rng.uniform(10000, 200000)
    computed = base * (1 + range_val)
    sheet = {
        "name": "Calculations",
        "rows": "5", "cols": "3",
        "headers": "Label | Value | Formula",
        "sample_rows": [
            f"Base Amount | {cur}{base:,.2f} | (input)",
            f"{range_name} | {range_val:.4f} | (named range: {range_name})",
            f"Adjusted Amount | {cur}{computed:,.2f} | =B2*(1+{range_name})",
        ],
        "summary": f"Calculation sheet using named range '{range_name}' = {range_val:.4f}.",
    }
    ctx = _spreadsheet_ctx(fname, [sheet])
    query = f"{ctx}\n\nThe spreadsheet uses a named range '{range_name}'. How is the Adjusted Amount computed?"
    reasoning = (
        f"Step 1: Named range '{range_name}' is defined as {range_val:.4f}.\n"
        f"Step 2: The formula =B2*(1+{range_name}) takes the Base Amount ({cur}{base:,.2f}) "
        f"and multiplies by (1 + {range_val:.4f}) = {1 + range_val:.4f}.\n"
        f"Step 3: Result = {cur}{base:,.2f} x {1 + range_val:.4f} = {cur}{computed:,.2f}.\n"
        f"Confidence: High -- named range resolves to a constant value."
    )
    answer = (
        f"The Adjusted Amount ({cur}{computed:,.2f}) is computed as Base Amount x (1 + {range_name}), "
        f"where {range_name} = {range_val:.4f}."
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Category 5: CSV delimiter detection (200)
# ---------------------------------------------------------------------------

def _gen_csv_delimiter(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    idx = 0
    while len(results) < count:
        delim_name, delim_display, delim_char = _pick(_DELIMITERS, rng)
        fname = _make_filename(rng, ext="csv")
        domain = _pick(DOMAINS, rng)
        headers = rng.sample(["Name", "Date", "Amount", "Status", "Region", "Category", "ID"], k=4)
        sample_vals = [
            [f"Row{r}Col{c}" for c in range(4)]
            for r in range(3)
        ]
        header_line = delim_char.join(headers)
        data_lines = [delim_char.join(row) for row in sample_vals]
        raw_preview = header_line + "\n" + "\n".join(data_lines)
        query = (
            f"The following CSV file '{fname}' was received from a {domain} system. "
            f"Detect the delimiter and parse the data:\n\n```\n{raw_preview}\n```"
        )
        reasoning = (
            f"Step 1: Examine the raw text for consistent separators.\n"
            f"Step 2: Check common delimiters -- comma, tab, semicolon, pipe.\n"
            f"Step 3: The character '{delim_display}' appears consistently between fields in every line.\n"
            f"Step 4: Header count ({len(headers)}) matches value count per row ({len(headers)}) when split by '{delim_display}'.\n"
            f"Step 5: Delimiter identified: {delim_name}.\n"
            f"Confidence: High -- consistent column count across all rows."
        )
        answer = (
            f"**Delimiter detected:** {delim_name} (`{delim_display}`)\n\n"
            f"**Parsed headers:** {', '.join(headers)}\n\n"
            f"**Row count:** {len(data_lines)} data rows\n\n"
            f"The file uses {delim_name}-separated values with {len(headers)} columns."
        )
        results.append(format_sft_example(query, reasoning, answer))
        idx += 1
    return results


# ---------------------------------------------------------------------------
# Category 6: Large spreadsheet chunking (200)
# ---------------------------------------------------------------------------

def _gen_large_chunking(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    templates = [lambda: _chunked_query(rng), lambda: _chunked_aggregation(rng)]
    idx = 0
    while len(results) < count:
        fn = templates[idx % len(templates)]
        results.append(fn())
        idx += 1
    return results


def _chunked_query(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng)
    cur = _pick(_CURRENCIES, rng)
    total_rows = rng.choice([100000, 250000, 500000, 1000000])
    chunk_size = rng.choice([5000, 10000, 50000])
    n_chunks = total_rows // chunk_size
    current_chunk = rng.randint(1, min(n_chunks, 20))
    start_row = (current_chunk - 1) * chunk_size + 1
    end_row = current_chunk * chunk_size
    product = _pick(_PRODUCT_NAMES, rng)
    matches_in_chunk = rng.randint(5, 50)
    chunk_total = rng.uniform(50000, 500000)
    sheet = {
        "name": "Transactions",
        "rows": str(total_rows), "cols": "6",
        "headers": "TxID | Date | Product | Qty | Unit Price | Total",
        "sample_rows": [
            f"TX-{start_row} | 2024-01-15 | {product} | 10 | {cur}50.00 | {cur}500.00",
            f"TX-{start_row+1} | 2024-01-15 | Widget A | 5 | {cur}100.00 | {cur}500.00",
            f"... (chunk {current_chunk} of {n_chunks}: rows {start_row}-{end_row})",
        ],
        "summary": (
            f"{total_rows:,} total rows, chunked into {n_chunks} segments of {chunk_size:,}. "
            f"Showing chunk {current_chunk} (rows {start_row:,}-{end_row:,})."
        ),
    }
    ctx = _spreadsheet_ctx(fname, [sheet])
    query = (
        f"{ctx}\n\nThis is chunk {current_chunk} of {n_chunks} from a {total_rows:,}-row spreadsheet. "
        f"How many transactions for '{product}' appear in this chunk?"
    )
    reasoning = (
        f"Step 1: This is a large spreadsheet ({total_rows:,} rows) chunked into {n_chunks} segments.\n"
        f"Step 2: Current chunk {current_chunk} covers rows {start_row:,}-{end_row:,}.\n"
        f"Step 3: Scanning this chunk for Product = '{product}'.\n"
        f"Step 4: Found {matches_in_chunk} matching transactions in this chunk.\n"
        f"Step 5: Note -- this is only chunk {current_chunk}/{n_chunks}; other chunks may contain more.\n"
        f"Confidence: Medium -- result is chunk-scoped, not global."
    )
    answer = (
        f"In chunk {current_chunk} (rows {start_row:,}-{end_row:,}), there are **{matches_in_chunk}** "
        f"transactions for '{product}'. Note: This represents only {chunk_size/total_rows*100:.1f}% "
        f"of the full {total_rows:,}-row dataset."
    )
    return format_sft_example(query, reasoning, answer)


def _chunked_aggregation(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng)
    cur = _pick(_CURRENCIES, rng)
    total_rows = rng.choice([150000, 300000, 750000])
    n_chunks = rng.randint(5, 15)
    chunk_totals = [rng.uniform(100000, 2000000) for _ in range(n_chunks)]
    grand_total = sum(chunk_totals)
    dept = _pick(_DEPARTMENTS, rng)
    chunk_summaries = "\n".join(
        f"Chunk {i+1}: {cur}{t:,.2f}" for i, t in enumerate(chunk_totals)
    )
    query = (
        f"A {total_rows:,}-row spreadsheet '{fname}' was processed in {n_chunks} chunks for {dept}. "
        f"The per-chunk totals for the 'Amount' column are:\n\n{chunk_summaries}\n\n"
        f"What is the grand total across all chunks?"
    )
    reasoning = (
        f"Step 1: The spreadsheet was split into {n_chunks} chunks due to its size ({total_rows:,} rows).\n"
        f"Step 2: Each chunk was independently aggregated.\n"
        f"Step 3: Grand total = sum of chunk totals = {' + '.join(f'{cur}{t:,.2f}' for t in chunk_totals[:3])} + ... = {cur}{grand_total:,.2f}.\n"
        f"Confidence: High -- simple sum of pre-aggregated chunks."
    )
    answer = (
        f"The grand total across all {n_chunks} chunks is **{cur}{grand_total:,.2f}** "
        f"({total_rows:,} rows processed)."
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Category 7: Data type inference (250)
# ---------------------------------------------------------------------------

def _gen_data_type_inference(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    idx = 0
    while len(results) < count:
        results.append(_data_type_example(rng))
        idx += 1
    return results


def _data_type_example(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng, ext=rng.choice(["xlsx", "csv"]))
    domain = _pick(DOMAINS, rng)
    # Build a table with various data types that need inference
    columns = rng.sample([
        ("Date", ["2024-03-15", "03/15/2024", "15-Mar-2024", "March 15, 2024"], "date"),
        ("Revenue", ["$12,345.67", "EUR 8,900.00", "GBP 5,432.10", "15000"], "currency"),
        ("Growth", ["12.5%", "-3.2%", "0.8%", "+45.0%"], "percentage"),
        ("Phone", ["(555) 123-4567", "+1-555-987-6543", "555.123.4567", "5551234567"], "phone_number"),
        ("Quantity", ["1,234", "567", "89012", "3"], "integer"),
        ("Rate", ["3.14159", "2.718", "0.0045", "99.9"], "decimal"),
    ], k=4)

    col_names = [c[0] for c in columns]
    sample_vals = [c[1] for c in columns]
    inferred_types = [c[2] for c in columns]

    rows = [" | ".join(row) for row in zip(*sample_vals)]
    sheet = {
        "name": "Data",
        "rows": str(len(rows)), "cols": str(len(col_names)),
        "headers": " | ".join(col_names),
        "sample_rows": rows,
        "summary": f"{len(rows)} rows, {len(col_names)} columns. Mixed data types from {domain} system.",
    }
    ctx = _spreadsheet_ctx(fname, [sheet])
    query = f"{ctx}\n\nInfer the data type for each column and note any formatting inconsistencies."
    reasoning = (
        "Step 1: Analyze each column's sample values to determine data type.\n"
        + "\n".join(
            f"Step {i+2}: Column '{cn}' -- values: {', '.join(repr(v) for v in sv[:2])}... "
            f"Pattern matches {it} type."
            for i, (cn, sv, it) in enumerate(columns)
        )
        + f"\nStep {len(columns)+2}: Check for formatting inconsistencies within each column.\n"
        f"Confidence: High -- patterns are clear from sample values."
    )
    type_lines = "\n".join(
        f"- **{cn}**: `{it}` (e.g., {sv[0]})"
        for cn, sv, it in columns
    )
    inconsistencies = []
    for cn, sv, it in columns:
        if it == "date" and len(set(len(v) for v in sv)) > 1:
            inconsistencies.append(f"'{cn}' has mixed date formats (ISO, US, European)")
        elif it == "phone_number":
            inconsistencies.append(f"'{cn}' has inconsistent phone formats (parentheses, dashes, dots)")
        elif it == "currency" and any(c.isalpha() for vals in sv for c in vals[:3]):
            inconsistencies.append(f"'{cn}' mixes currency symbol styles")
    incon_text = "\n".join(f"- {i}" for i in inconsistencies) if inconsistencies else "- None detected"
    answer = (
        f"**Inferred data types:**\n{type_lines}\n\n"
        f"**Formatting inconsistencies:**\n{incon_text}"
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Category 8: Spreadsheet-to-insight (300)
# ---------------------------------------------------------------------------

def _gen_spreadsheet_insight(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    templates = [lambda: _insight_summary(rng), lambda: _insight_anomaly(rng), lambda: _insight_trend(rng)]
    idx = 0
    while len(results) < count:
        fn = templates[idx % len(templates)]
        results.append(fn())
        idx += 1
    return results


def _insight_summary(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng)
    cur = _pick(_CURRENCIES, rng)
    company = _pick(_COMPANY_NAMES, rng)
    depts = rng.sample(_DEPARTMENTS, k=5)
    values = [rng.uniform(50000, 500000) for _ in depts]
    total = sum(values)
    max_idx = values.index(max(values))
    min_idx = values.index(min(values))
    rows = [f"{d} | {cur}{v:,.2f} | {v/total*100:.1f}%" for d, v in zip(depts, values)]
    sheet = {
        "name": "Annual Budget",
        "rows": str(len(depts)), "cols": "3",
        "headers": "Department | Budget | % of Total",
        "sample_rows": rows,
        "summary": f"{len(depts)} departments. Total: {cur}{total:,.2f}.",
    }
    ctx = _spreadsheet_ctx(fname, [sheet])
    query = f"{ctx}\n\nSummarize this spreadsheet from {company}. Identify key patterns."
    reasoning = (
        f"Step 1: This is a departmental budget breakdown for {company}.\n"
        f"Step 2: Total budget: {cur}{total:,.2f} across {len(depts)} departments.\n"
        f"Step 3: Largest allocation: {depts[max_idx]} at {cur}{values[max_idx]:,.2f} ({values[max_idx]/total*100:.1f}%).\n"
        f"Step 4: Smallest allocation: {depts[min_idx]} at {cur}{values[min_idx]:,.2f} ({values[min_idx]/total*100:.1f}%).\n"
        f"Step 5: Spread ratio (max/min): {values[max_idx]/values[min_idx]:.1f}x.\n"
        f"Confidence: High -- complete data visible."
    )
    answer = (
        f"**Summary of {company}'s Annual Budget:**\n\n"
        f"- **Total budget:** {cur}{total:,.2f} across {len(depts)} departments\n"
        f"- **Largest allocation:** {depts[max_idx]} ({values[max_idx]/total*100:.1f}%)\n"
        f"- **Smallest allocation:** {depts[min_idx]} ({values[min_idx]/total*100:.1f}%)\n"
        f"- **Key pattern:** Budget concentration -- top department receives "
        f"{values[max_idx]/values[min_idx]:.1f}x more than the smallest"
    )
    return format_sft_example(query, reasoning, answer)


def _insight_anomaly(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng)
    cur = _pick(_CURRENCIES, rng)
    months = _MONTHS[:6]
    normal_range = (8000, 15000)
    values = [rng.uniform(*normal_range) for _ in months]
    anomaly_idx = rng.randint(1, 4)
    anomaly_val = rng.uniform(40000, 80000)
    values[anomaly_idx] = anomaly_val
    rows = [f"{m} | {cur}{v:,.2f}" for m, v in zip(months, values)]
    sheet = {
        "name": "Monthly Expenses",
        "rows": "6", "cols": "2",
        "headers": "Month | Expenses",
        "sample_rows": rows,
        "summary": f"6 months of expense data. Range: {cur}{min(values):,.2f} - {cur}{max(values):,.2f}.",
    }
    ctx = _spreadsheet_ctx(fname, [sheet])
    query = f"{ctx}\n\nAnalyze this expense data for anomalies or unusual patterns."
    normal_avg = sum(v for i, v in enumerate(values) if i != anomaly_idx) / (len(values) - 1)
    deviation = (anomaly_val - normal_avg) / normal_avg * 100
    reasoning = (
        f"Step 1: Compute typical range -- excluding outliers, expenses average {cur}{normal_avg:,.2f}.\n"
        f"Step 2: {months[anomaly_idx]} shows {cur}{anomaly_val:,.2f}, which is {deviation:.0f}% above average.\n"
        f"Step 3: All other months fall within {cur}{normal_range[0]:,}-{cur}{normal_range[1]:,} range.\n"
        f"Step 4: This is a clear outlier requiring investigation.\n"
        f"Confidence: High -- statistical anomaly is unambiguous."
    )
    answer = (
        f"**Anomaly detected:**\n\n"
        f"- **{months[anomaly_idx]}**: {cur}{anomaly_val:,.2f} -- **{deviation:.0f}% above** the "
        f"typical monthly average of {cur}{normal_avg:,.2f}\n"
        f"- All other months fall within a normal range of {cur}{normal_range[0]:,}-{cur}{normal_range[1]:,}\n"
        f"- **Recommendation:** Investigate the {months[anomaly_idx]} spike for unusual transactions"
    )
    return format_sft_example(query, reasoning, answer)


def _insight_trend(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng)
    cur = _pick(_CURRENCIES, rng)
    quarters = ["Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023", "Q1 2024", "Q2 2024"]
    base = rng.uniform(100000, 300000)
    growth_rate = rng.uniform(0.02, 0.12)
    direction = rng.choice(["up", "down"])
    multiplier = 1 + growth_rate if direction == "up" else 1 - growth_rate
    values = [base * (multiplier ** i) for i in range(len(quarters))]
    rows = [f"{q} | {cur}{v:,.2f}" for q, v in zip(quarters, values)]
    overall_change = (values[-1] - values[0]) / values[0] * 100
    sheet = {
        "name": "Quarterly Revenue",
        "rows": "6", "cols": "2",
        "headers": "Quarter | Revenue",
        "sample_rows": rows,
        "summary": f"6 quarters of revenue data showing {'upward' if direction == 'up' else 'downward'} trend.",
    }
    ctx = _spreadsheet_ctx(fname, [sheet])
    query = f"{ctx}\n\nIdentify the trend in this quarterly revenue data and project the next quarter."
    next_val = values[-1] * multiplier
    reasoning = (
        f"Step 1: Analyze quarter-over-quarter changes.\n"
        f"Step 2: Each quarter {'increases' if direction == 'up' else 'decreases'} by approximately {growth_rate*100:.1f}%.\n"
        f"Step 3: Overall change from {quarters[0]} to {quarters[-1]}: {overall_change:+.1f}%.\n"
        f"Step 4: Assuming trend continues, Q3 2024 projection: {cur}{next_val:,.2f}.\n"
        f"Confidence: Medium -- projection assumes trend continuation."
    )
    answer = (
        f"**Trend Analysis:**\n\n"
        f"- **Direction:** {'Upward' if direction == 'up' else 'Downward'} trend\n"
        f"- **Rate:** ~{growth_rate*100:.1f}% per quarter\n"
        f"- **Overall change:** {overall_change:+.1f}% over {len(quarters)} quarters\n"
        f"- **Q3 2024 projection:** {cur}{next_val:,.2f} (assuming trend continues)"
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Category 9: Negatives & edge cases (250)
# ---------------------------------------------------------------------------

def _gen_edge_cases(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    templates = [
        lambda: _edge_empty_sheet(rng),
        lambda: _edge_pivot_table(rng),
        lambda: _edge_password_protected(rng),
        lambda: _edge_mismatched_columns(rng),
        lambda: _edge_no_headers(rng),
    ]
    idx = 0
    while len(results) < count:
        fn = templates[idx % len(templates)]
        results.append(fn())
        idx += 1
    return results


def _edge_empty_sheet(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng)
    domain = _pick(DOMAINS, rng)
    sheet = {
        "name": "Data",
        "rows": "0", "cols": "0",
        "headers": "(empty)",
        "sample_rows": ["(no data)"],
        "summary": "Empty sheet -- 0 rows, 0 columns.",
    }
    ctx = _spreadsheet_ctx(fname, [sheet])
    query = f"{ctx}\n\nExtract the key metrics from this {domain} spreadsheet."
    reasoning = (
        f"Step 1: Examine the sheet 'Data' -- rows=0, cols=0.\n"
        f"Step 2: Headers show '(empty)' and sample rows show '(no data)'.\n"
        f"Step 3: This sheet contains no data to analyze.\n"
        f"Confidence: High -- the sheet is definitively empty."
    )
    answer = (
        f"The spreadsheet '{fname}' contains an **empty sheet** with no data. "
        f"No metrics can be extracted. Please verify the file was exported correctly "
        f"or check if data resides on a different sheet."
    )
    return format_sft_example(query, reasoning, answer)


def _edge_pivot_table(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng)
    cur = _pick(_CURRENCIES, rng)
    depts = rng.sample(_DEPARTMENTS, k=3)
    quarters = _QUARTERS
    values = {d: {q: rng.uniform(10000, 80000) for q in quarters} for d in depts}
    row_totals = {d: sum(values[d].values()) for d in depts}
    col_totals = {q: sum(values[d][q] for d in depts) for q in quarters}
    grand_total = sum(row_totals.values())
    rows = []
    for d in depts:
        row = f"{d} | " + " | ".join(f"{cur}{values[d][q]:,.2f}" for q in quarters) + f" | {cur}{row_totals[d]:,.2f}"
        rows.append(row)
    rows.append("Grand Total | " + " | ".join(f"{cur}{col_totals[q]:,.2f}" for q in quarters) + f" | {cur}{grand_total:,.2f}")
    sheet = {
        "name": "PivotTable1",
        "rows": str(len(depts) + 1), "cols": str(len(quarters) + 2),
        "headers": "Department | " + " | ".join(quarters) + " | Row Total",
        "sample_rows": rows,
        "summary": f"Pivot table: {len(depts)} departments x {len(quarters)} quarters. Grand total: {cur}{grand_total:,.2f}.",
    }
    ctx = _spreadsheet_ctx(fname, [sheet])
    query = f"{ctx}\n\nThis appears to be a pivot table. Verify the row and column totals are consistent."
    # Introduce a verification narrative
    reasoning = (
        f"Step 1: This is a pivot table with {len(depts)} departments and {len(quarters)} quarters.\n"
        f"Step 2: Verify row totals -- sum each department's quarterly values.\n"
        + "\n".join(
            f"  - {d}: {' + '.join(f'{cur}{values[d][q]:,.2f}' for q in quarters)} = {cur}{row_totals[d]:,.2f} (matches)"
            for d in depts
        )
        + f"\nStep 3: Verify column totals -- sum each quarter across departments.\n"
        + "\n".join(
            f"  - {q}: {' + '.join(f'{cur}{values[d][q]:,.2f}' for d in depts)} = {cur}{col_totals[q]:,.2f} (matches)"
            for q in quarters
        )
        + f"\nStep 4: Grand total = {cur}{grand_total:,.2f} -- consistent with both row and column sums.\n"
        f"Confidence: High -- all totals verified."
    )
    answer = (
        f"**Pivot table verification: All totals are consistent.**\n\n"
        f"- Row totals: {len(depts)} departments verified\n"
        f"- Column totals: {len(quarters)} quarters verified\n"
        f"- Grand total: {cur}{grand_total:,.2f} -- matches both row and column sums"
    )
    return format_sft_example(query, reasoning, answer)


def _edge_password_protected(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng)
    domain = _pick(DOMAINS, rng)
    query = (
        f"<spreadsheet source=\"{fname}\">\n"
        f"  <error>Unable to open: workbook is password-protected</error>\n"
        f"</spreadsheet>\n\n"
        f"Extract the {domain} data from this spreadsheet."
    )
    reasoning = (
        f"Step 1: The spreadsheet parser returned an error: 'workbook is password-protected'.\n"
        f"Step 2: No sheet data, headers, or rows are available.\n"
        f"Step 3: I cannot extract any data without the password to decrypt the file.\n"
        f"Confidence: N/A -- cannot process the file."
    )
    answer = (
        f"The spreadsheet '{fname}' is **password-protected** and cannot be opened. "
        f"To extract data, please provide the workbook password or an unprotected copy of the file."
    )
    return format_sft_example(query, reasoning, answer)


def _edge_mismatched_columns(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng, ext="csv")
    domain = _pick(DOMAINS, rng)
    headers = ["ID", "Name", "Amount", "Date"]
    rows = [
        "1 | Alice | $500 | 2024-01-01",
        "2 | Bob | $300",  # missing Date
        "3 | Carol | $700 | 2024-01-03 | Extra",  # extra column
    ]
    sheet = {
        "name": "Sheet1",
        "rows": "3", "cols": "4",
        "headers": " | ".join(headers),
        "sample_rows": rows,
        "summary": "3 rows with inconsistent column counts (3, 4, and 5 fields).",
    }
    ctx = _spreadsheet_ctx(fname, [sheet])
    query = f"{ctx}\n\nParse this {domain} CSV data and flag any structural issues."
    reasoning = (
        f"Step 1: Header defines 4 columns: {', '.join(headers)}.\n"
        f"Step 2: Row 1 has 4 fields -- OK.\n"
        f"Step 3: Row 2 has 3 fields -- MISSING 'Date' column.\n"
        f"Step 4: Row 3 has 5 fields -- EXTRA unnamed column.\n"
        f"Step 5: 2 out of 3 rows have structural issues.\n"
        f"Confidence: High -- column count mismatches are unambiguous."
    )
    answer = (
        f"**Structural issues detected:**\n\n"
        f"- **Row 2:** Missing value for 'Date' column (3 fields instead of 4)\n"
        f"- **Row 3:** Extra field beyond expected 4 columns\n"
        f"- **Row 1:** Valid (4 fields matching header)\n\n"
        f"**Recommendation:** Clean the CSV by padding missing fields with NULL "
        f"and trimming or relocating extra fields."
    )
    return format_sft_example(query, reasoning, answer)


def _edge_no_headers(rng: random.Random) -> Dict[str, str]:
    fname = _make_filename(rng, ext="csv")
    domain = _pick(DOMAINS, rng)
    rows = [
        "2024-01-15 | Acme Corp | 12500.00 | Paid",
        "2024-02-01 | Globex | 8900.50 | Pending",
        "2024-02-14 | Initech | 3200.00 | Paid",
    ]
    sheet = {
        "name": "Sheet1",
        "rows": "3", "cols": "4",
        "headers": "(no header row detected)",
        "sample_rows": rows,
        "summary": "3 rows, 4 columns. No header row. Appears to be transactional data.",
    }
    ctx = _spreadsheet_ctx(fname, [sheet])
    query = f"{ctx}\n\nThis {domain} CSV has no header row. Infer column names from the data."
    reasoning = (
        f"Step 1: No explicit headers -- must infer from data patterns.\n"
        f"Step 2: Column 1 values (2024-01-15, 2024-02-01, 2024-02-14) -- date format -> 'Date'.\n"
        f"Step 3: Column 2 values (Acme Corp, Globex, Initech) -- company names -> 'Vendor' or 'Company'.\n"
        f"Step 4: Column 3 values (12500.00, 8900.50, 3200.00) -- numeric with decimals -> 'Amount'.\n"
        f"Step 5: Column 4 values (Paid, Pending) -- status labels -> 'Status'.\n"
        f"Confidence: High -- patterns are consistent and recognizable."
    )
    answer = (
        f"**Inferred column headers:**\n\n"
        f"1. **Date** -- ISO date format (YYYY-MM-DD)\n"
        f"2. **Vendor** -- company/organization names\n"
        f"3. **Amount** -- monetary values (decimal)\n"
        f"4. **Status** -- payment status (Paid/Pending)\n\n"
        f"These inferred headers are consistent across all 3 data rows."
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_track1_data(output_dir: str | Path, seed: int = 42) -> dict:
    """Generate Track 1 Excel/CSV intelligence training data.

    Args:
        output_dir: Directory to write JSONL files into.
        seed: Random seed for reproducibility.

    Returns:
        Dict with ``sft_path`` and ``sft_count``.
    """
    rng = random.Random(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_path = output_dir / "track1_excel_csv_sft.jsonl"

    categories = [
        ("single_sheet_qa", _gen_single_sheet_qa, 400),
        ("multi_sheet_reasoning", _gen_multi_sheet, 350),
        ("formula_aware", _gen_formula_aware, 300),
        ("merged_cell_named_range", _gen_merged_cells, 250),
        ("csv_delimiter", _gen_csv_delimiter, 200),
        ("large_chunking", _gen_large_chunking, 200),
        ("data_type_inference", _gen_data_type_inference, 250),
        ("spreadsheet_insight", _gen_spreadsheet_insight, 300),
        ("edge_cases", _gen_edge_cases, 250),
    ]

    all_examples: List[Dict[str, str]] = []
    for name, gen_fn, count in categories:
        # Derive a sub-seed for each category for reproducibility
        sub_seed = seed + hash(name) % 10000
        cat_rng = random.Random(sub_seed)
        examples = gen_fn(count, cat_rng)
        all_examples.extend(examples)

    rng.shuffle(all_examples)

    with JSONLWriter(sft_path) as writer:
        for ex in all_examples:
            writer.write(ex)

    return {
        "sft_path": str(sft_path),
        "sft_count": len(all_examples),
    }


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/finetune/v2")
    result = generate_track1_data(out)
    print(f"Track 1 Excel/CSV: {result['sft_count']} SFT examples -> {result['sft_path']}")
