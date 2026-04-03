"""Track 2 — Layout Intelligence data generator for DocWain V2+ SFT/DPO.

Generates 2500 SFT examples and ~200 DPO preference pairs across nine categories:
  - Nested tables 3+ levels             (300)
  - Merged cell reasoning                (300)
  - Multi-column with interruptions      (250)
  - Mixed form + prose                   (250)
  - Page-spanning structures             (200)
  - Hierarchical headings                (200)
  - Document type adaptation             (300)
  - Completeness verification            (400)
  - Edge cases & negatives               (300)

Completeness verification examples include explicit field checklists.
DPO pairs contrast complete vs. incomplete extraction.
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
    format_dpo_example,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

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
_ADDRESSES = [
    "123 Main St, Suite 400, New York, NY 10001",
    "456 Oak Avenue, Building B, San Francisco, CA 94105",
    "789 Elm Drive, Floor 12, Chicago, IL 60601",
    "321 Pine Road, Unit 7, Austin, TX 78701",
    "654 Maple Lane, London EC2V 8AS, UK",
    "987 Cedar Blvd, Toronto ON M5V 2T6, Canada",
]
_FORM_FIELDS = {
    "contract": [
        "Party A Name", "Party A Address", "Party B Name", "Party B Address",
        "Effective Date", "Termination Date", "Governing Law", "Payment Terms",
        "Scope of Work", "Liability Cap", "Confidentiality Clause",
        "Signature Party A", "Signature Party B",
    ],
    "invoice": [
        "Invoice Number", "Invoice Date", "Due Date", "Vendor Name",
        "Vendor Address", "Bill To", "Line Items", "Subtotal",
        "Tax Rate", "Tax Amount", "Total Amount", "Payment Method",
    ],
    "insurance_claim": [
        "Claim Number", "Policy Number", "Claimant Name", "Claimant Address",
        "Date of Incident", "Location of Incident", "Description of Loss",
        "Estimated Amount", "Supporting Documents", "Adjuster Name",
        "Status", "Date Filed",
    ],
    "medical_record": [
        "Patient Name", "Patient ID", "Date of Birth", "Gender",
        "Attending Physician", "Visit Date", "Chief Complaint",
        "Diagnosis (ICD-10)", "Treatment Plan", "Medications",
        "Follow-up Date", "Vitals",
    ],
    "resume": [
        "Full Name", "Contact Email", "Phone Number", "Address",
        "Summary/Objective", "Work Experience", "Education",
        "Skills", "Certifications", "Languages", "References",
    ],
    "purchase_order": [
        "PO Number", "Date", "Vendor Name", "Vendor Address",
        "Ship To", "Line Items", "Quantities", "Unit Prices",
        "Subtotal", "Shipping Cost", "Tax", "Total",
        "Authorized By", "Delivery Date",
    ],
    "policy_document": [
        "Policy Title", "Policy Number", "Effective Date", "Review Date",
        "Department", "Scope", "Purpose", "Definitions",
        "Policy Statement", "Procedures", "Responsibilities",
        "Compliance Requirements", "Approval Authority",
    ],
    "compliance_report": [
        "Report Title", "Report Date", "Reporting Period",
        "Auditor Name", "Department Audited", "Scope",
        "Findings Summary", "Risk Level", "Recommendations",
        "Management Response", "Follow-up Actions", "Status",
    ],
}
_HEADING_STRUCTURES = [
    ["1. Introduction", "1.1 Purpose", "1.1.1 Background", "1.1.1.1 Historical Context",
     "1.1.1.1.1 Origin", "1.2 Scope", "2. Methodology", "2.1 Approach",
     "2.1.1 Data Collection", "2.2 Analysis Framework", "3. Findings"],
    ["I. Executive Summary", "II. Background", "II.A. Context", "II.A.1. Market",
     "II.A.1.a. Demographics", "II.B. Prior Work", "III. Analysis",
     "III.A. Quantitative", "III.A.1. Statistics", "IV. Recommendations"],
    ["Section 1: Overview", "1.1 Goals", "1.1.1 Short-term", "1.1.1.1 Q1 Targets",
     "1.1.1.1.1 Revenue Goals", "1.1.2 Long-term", "Section 2: Strategy",
     "2.1 Market Entry", "2.1.1 Phase 1", "Section 3: Budget"],
]


def _pick(lst: list, rng: random.Random) -> Any:
    return rng.choice(lst)


def _rand_amount(rng: random.Random, lo: float = 100.0, hi: float = 99999.0) -> str:
    return f"${rng.uniform(lo, hi):,.2f}"


def _rand_date(rng: random.Random) -> str:
    y = rng.randint(2021, 2026)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    return f"{y}-{m:02d}-{d:02d}"


def _get_fields_for_type(doc_type: str, rng: random.Random) -> List[str]:
    """Get expected fields for a document type, falling back to generic if unknown."""
    if doc_type in _FORM_FIELDS:
        return _FORM_FIELDS[doc_type]
    # fallback to a random known type's fields
    return _FORM_FIELDS[_pick(list(_FORM_FIELDS.keys()), rng)]


# ---------------------------------------------------------------------------
# Category 1: Nested tables 3+ levels (300)
# ---------------------------------------------------------------------------

def _gen_nested_tables(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    idx = 0
    while len(results) < count:
        results.append(_nested_table_example(rng, idx))
        idx += 1
    return results


def _nested_table_example(rng: random.Random, variant: int) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    doc_type = _pick(DOC_TYPES, rng)
    company = _pick(_COMPANY_NAMES, rng)
    dept = _pick(_DEPARTMENTS, rng)
    person = _pick(_PERSON_NAMES, rng)

    # Build a 3-level nested table layout
    level3_items = [
        f"        | Task {rng.randint(1,9)}.{rng.randint(1,5)}.{j} | {_rand_date(rng)} | {_pick(['Done', 'In Progress', 'Pending'], rng)}"
        for j in range(1, rng.randint(3, 5))
    ]
    level2_items = [
        f"      | Milestone {rng.randint(1,5)}.{i} | {_rand_date(rng)} | {_rand_amount(rng, 5000, 50000)}\n"
        + "\n".join(level3_items)
        for i in range(1, rng.randint(3, 4))
    ]
    layout = (
        f"Document: {domain} {doc_type} from {company}\n\n"
        f"[Outer Table: Project Summary]\n"
        f"| Project | Lead | Budget |\n"
        f"| {dept} Upgrade | {person} | {_rand_amount(rng, 100000, 500000)} |\n\n"
        f"  [Nested Table Level 2: Milestones]\n"
        f"  | Milestone | Due Date | Allocated |\n"
        + "\n".join(level2_items) + "\n\n"
        f"    [Nested Table Level 3: Tasks]\n"
        f"    | Task ID | Deadline | Status |\n"
        + "\n".join(level3_items)
    )

    query = (
        f"The following {domain} document has tables nested 3 levels deep:\n\n"
        f"{layout}\n\n"
        f"Extract the hierarchical structure and identify which tasks belong to which milestone."
    )
    n_milestones = len(level2_items)
    n_tasks = len(level3_items)
    reasoning = (
        f"Step 1: Identify the outer table -- Project Summary with 1 row (project: {dept} Upgrade).\n"
        f"Step 2: Inside the project row, find Level 2 table -- {n_milestones} milestones.\n"
        f"Step 3: Each milestone contains Level 3 task tables with {n_tasks} tasks shown.\n"
        f"Step 4: The nesting hierarchy is: Project -> Milestone -> Task.\n"
        f"Step 5: Parent-child relationships are determined by indentation and containment.\n"
        f"Confidence: High -- nesting levels are clearly demarcated."
    )
    answer = (
        f"**Hierarchical Structure (3 levels):**\n\n"
        f"- **Level 1 (Project):** {dept} Upgrade, led by {person}\n"
        f"- **Level 2 (Milestones):** {n_milestones} milestones with dates and budgets\n"
        f"- **Level 3 (Tasks):** {n_tasks} tasks per milestone with status tracking\n\n"
        f"Each task is a child of its enclosing milestone, which in turn belongs to the project."
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Category 2: Merged cell reasoning (300)
# ---------------------------------------------------------------------------

def _gen_merged_cells(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    templates = [
        lambda: _merged_spanning_header(rng),
        lambda: _merged_row_group(rng),
        lambda: _merged_complex(rng),
    ]
    idx = 0
    while len(results) < count:
        fn = templates[idx % len(templates)]
        results.append(fn())
        idx += 1
    return results


def _merged_spanning_header(rng: random.Random) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    company = _pick(_COMPANY_NAMES, rng)
    year = rng.randint(2022, 2026)
    depts = rng.sample(_DEPARTMENTS, k=3)
    layout = (
        f"[Table from {company}'s {domain} report]\n\n"
        f"| [MERGED: {year} Annual Performance] |\n"
        f"|------|------|------|------|\n"
        f"| Department | Revenue | Expenses | Margin |\n"
    )
    revenues = []
    expenses = []
    for d in depts:
        rev = rng.uniform(100000, 800000)
        exp = rng.uniform(50000, rev * 0.9)
        margin = (rev - exp) / rev * 100
        revenues.append(rev)
        expenses.append(exp)
        layout += f"| {d} | ${rev:,.2f} | ${exp:,.2f} | {margin:.1f}% |\n"

    query = (
        f"{layout}\n"
        f"The header '{year} Annual Performance' is a merged cell spanning all 4 columns. "
        f"Which department has the highest margin?"
    )
    margins = [(rev - exp) / rev * 100 for rev, exp in zip(revenues, expenses)]
    best_idx = margins.index(max(margins))
    reasoning = (
        f"Step 1: The merged header '{year} Annual Performance' spans columns A-D. "
        f"The actual column headers are in the next row.\n"
        f"Step 2: Calculate margin for each department:\n"
        + "\n".join(
            f"  - {d}: ({revenues[i]:,.2f} - {expenses[i]:,.2f}) / {revenues[i]:,.2f} = {margins[i]:.1f}%"
            for i, d in enumerate(depts)
        )
        + f"\nStep 3: Highest margin: {depts[best_idx]} at {margins[best_idx]:.1f}%.\n"
        f"Confidence: High -- merged cell correctly interpreted as title, not data."
    )
    answer = (
        f"**{depts[best_idx]}** has the highest margin at **{margins[best_idx]:.1f}%** "
        f"(Revenue: ${revenues[best_idx]:,.2f}, Expenses: ${expenses[best_idx]:,.2f})."
    )
    return format_sft_example(query, reasoning, answer)


def _merged_row_group(rng: random.Random) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    company = _pick(_COMPANY_NAMES, rng)
    dept = _pick(_DEPARTMENTS, rng)
    people = rng.sample(_PERSON_NAMES, k=4)
    roles = ["Manager", "Senior Analyst", "Analyst", "Associate"]
    layout = (
        f"[Table from {company} {domain} org chart]\n\n"
        f"| Department | Employee | Role | Salary |\n"
        f"| [MERGED: {dept} spans 4 rows] | {people[0]} | {roles[0]} | {_rand_amount(rng, 80000, 150000)} |\n"
        f"|  | {people[1]} | {roles[1]} | {_rand_amount(rng, 60000, 100000)} |\n"
        f"|  | {people[2]} | {roles[2]} | {_rand_amount(rng, 50000, 80000)} |\n"
        f"|  | {people[3]} | {roles[3]} | {_rand_amount(rng, 40000, 65000)} |\n"
    )
    query = (
        f"{layout}\n"
        f"The Department cell '{dept}' is merged vertically across 4 rows. "
        f"How many employees are in {dept} and what are their roles?"
    )
    reasoning = (
        f"Step 1: The merged cell in column A reads '{dept}' and spans rows 2-5.\n"
        f"Step 2: This means all 4 rows belong to the {dept} department.\n"
        f"Step 3: Employees and roles:\n"
        + "\n".join(f"  - {p}: {r}" for p, r in zip(people, roles))
        + f"\nStep 4: Total: 4 employees in {dept}.\n"
        f"Confidence: High -- merged cell semantics are clear."
    )
    answer = (
        f"The **{dept}** department has **4 employees**:\n"
        + "\n".join(f"- {p} ({r})" for p, r in zip(people, roles))
    )
    return format_sft_example(query, reasoning, answer)


def _merged_complex(rng: random.Random) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    q1_rev = rng.uniform(100000, 400000)
    q2_rev = rng.uniform(100000, 400000)
    q1_domestic = rng.uniform(50000, q1_rev * 0.7)
    q1_intl = q1_rev - q1_domestic
    q2_domestic = rng.uniform(50000, q2_rev * 0.7)
    q2_intl = q2_rev - q2_domestic
    layout = (
        f"| [MERGED: Revenue Breakdown] | | | | |\n"
        f"|------|------|------|------|------|\n"
        f"| | [MERGED: Q1] | | [MERGED: Q2] | |\n"
        f"| Region | Domestic | International | Domestic | International |\n"
        f"| Actual | ${q1_domestic:,.2f} | ${q1_intl:,.2f} | ${q2_domestic:,.2f} | ${q2_intl:,.2f} |\n"
    )
    query = (
        f"This {domain} table has both horizontal and vertical merged cells:\n\n{layout}\n"
        f"What is the total international revenue across both quarters?"
    )
    intl_total = q1_intl + q2_intl
    reasoning = (
        f"Step 1: Top merged header 'Revenue Breakdown' spans all columns (title).\n"
        f"Step 2: Second row has two merged headers: 'Q1' spans columns 2-3, 'Q2' spans columns 4-5.\n"
        f"Step 3: Sub-columns under each quarter: Domestic and International.\n"
        f"Step 4: International revenue: Q1 = ${q1_intl:,.2f}, Q2 = ${q2_intl:,.2f}.\n"
        f"Step 5: Total international = ${q1_intl:,.2f} + ${q2_intl:,.2f} = ${intl_total:,.2f}.\n"
        f"Confidence: High -- merged cell hierarchy correctly parsed."
    )
    answer = (
        f"Total international revenue across Q1 and Q2 is **${intl_total:,.2f}** "
        f"(Q1: ${q1_intl:,.2f} + Q2: ${q2_intl:,.2f})."
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Category 3: Multi-column with interruptions (250)
# ---------------------------------------------------------------------------

def _gen_multicolumn_interruptions(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    templates = [
        lambda: _multicolumn_figure(rng),
        lambda: _multicolumn_callout(rng),
    ]
    idx = 0
    while len(results) < count:
        fn = templates[idx % len(templates)]
        results.append(fn())
        idx += 1
    return results


def _multicolumn_figure(rng: random.Random) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    doc_type = _pick(DOC_TYPES, rng)
    company = _pick(_COMPANY_NAMES, rng)
    layout = (
        f"[Two-column layout from {company}'s {domain} {doc_type}]\n\n"
        f"LEFT COLUMN (paragraphs 1-3):\n"
        f"  P1: The {domain} review for FY2024 showed significant improvements...\n"
        f"  P2: Key performance indicators exceeded targets by 12%...\n\n"
        f"[FIGURE: Bar chart showing quarterly performance - spans both columns]\n"
        f"  Caption: Figure 2.1 - Quarterly KPI Achievement Rate\n\n"
        f"LEFT COLUMN (paragraph 4):\n"
        f"  P4: Following the mid-year adjustment, the team realigned...\n\n"
        f"RIGHT COLUMN (paragraphs 1-3):\n"
        f"  P1: Risk mitigation strategies were implemented across...\n"
        f"  P2: The compliance team reported zero critical findings...\n"
        f"  P3: Internal audit confirmed adherence to all regulatory...\n\n"
        f"RIGHT COLUMN (paragraph 4):\n"
        f"  P4: Looking ahead to FY2025, the outlook remains positive..."
    )
    query = (
        f"{layout}\n\n"
        f"This document uses a two-column layout interrupted by a full-width figure. "
        f"Extract the correct reading order of the text."
    )
    reasoning = (
        f"Step 1: Identify layout -- two-column format with a full-width interruption.\n"
        f"Step 2: Left column paragraphs 1-3 appear before the figure.\n"
        f"Step 3: A full-width figure (bar chart) interrupts both columns.\n"
        f"Step 4: After the figure, the left column continues with P4, then the right column has P1-P4.\n"
        f"Step 5: Reading order must follow column-first, then respect the figure break.\n"
        f"Step 6: Correct order: Left P1 -> Left P2 -> Left P3 (missing in layout, implied) -> "
        f"Figure -> Left P4 -> Right P1 -> Right P2 -> Right P3 -> Right P4.\n"
        f"Confidence: High -- figure clearly spans both columns, acting as a section break."
    )
    answer = (
        f"**Correct reading order:**\n\n"
        f"1. Left column, paragraphs 1-2 (pre-figure)\n"
        f"2. Figure 2.1 (full-width bar chart with caption)\n"
        f"3. Left column, paragraph 4 (post-figure)\n"
        f"4. Right column, paragraphs 1-3\n"
        f"5. Right column, paragraph 4\n\n"
        f"The figure interrupts the two-column flow and should be read as a separate element "
        f"between the left column's pre- and post-figure content."
    )
    return format_sft_example(query, reasoning, answer)


def _multicolumn_callout(rng: random.Random) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    company = _pick(_COMPANY_NAMES, rng)
    callout_type = _pick(["Warning", "Note", "Important", "Tip", "Caution"], rng)
    callout_text = _pick([
        "Failure to comply may result in penalties up to $50,000.",
        "This section supersedes all prior agreements dated before 2023.",
        "Supporting documentation must be submitted within 30 calendar days.",
        "Contact the compliance hotline at 1-800-555-0199 for questions.",
        "All figures are unaudited and subject to final review.",
    ], rng)
    layout = (
        f"[Three-column layout from {company}'s {domain} report]\n\n"
        f"COL 1: Background information on regulatory changes...\n"
        f"COL 2: Analysis of impact on {company}'s operations...\n"
        f"COL 3: Recommended action items for Q3...\n\n"
        f"[CALLOUT BOX - {callout_type} - spans columns 1-3]\n"
        f"  {callout_text}\n\n"
        f"COL 1 (continued): Implementation timeline begins...\n"
        f"COL 2 (continued): Budget implications include...\n"
        f"COL 3 (continued): Responsible parties are..."
    )
    query = (
        f"{layout}\n\n"
        f"A {callout_type.lower()} callout box interrupts the three-column layout. "
        f"How should this content be structured when extracting text?"
    )
    reasoning = (
        f"Step 1: Three-column layout detected with a full-width callout box.\n"
        f"Step 2: The callout is a '{callout_type}' box spanning all three columns.\n"
        f"Step 3: Content before callout: Col 1, Col 2, Col 3 (first sections).\n"
        f"Step 4: Callout content: \"{callout_text}\"\n"
        f"Step 5: Content after callout: Col 1, Col 2, Col 3 (continued sections).\n"
        f"Step 6: The callout should be extracted as a distinct highlighted element.\n"
        f"Confidence: High -- callout box is clearly demarcated."
    )
    answer = (
        f"**Extraction structure:**\n\n"
        f"1. **Pre-callout text** (3 columns): Background, Analysis, Recommendations\n"
        f"2. **[{callout_type}]**: \"{callout_text}\"\n"
        f"3. **Post-callout text** (3 columns): Implementation, Budget, Responsibilities\n\n"
        f"The callout box should be treated as a standalone highlighted element that "
        f"interrupts the column flow. It applies to all three columns' content."
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Category 4: Mixed form + prose (250)
# ---------------------------------------------------------------------------

def _gen_mixed_form_prose(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    idx = 0
    while len(results) < count:
        results.append(_mixed_form_example(rng, idx))
        idx += 1
    return results


def _mixed_form_example(rng: random.Random, variant: int) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    company = _pick(_COMPANY_NAMES, rng)
    person = _pick(_PERSON_NAMES, rng)
    address = _pick(_ADDRESSES, rng)
    date = _rand_date(rng)
    form_type = _pick(["Application Form", "Registration Form", "Claim Form", "Enrollment Form"], rng)

    instruction_text = _pick([
        "Complete all fields in ink. Use block capitals. Do not leave any section blank.",
        "This form must be accompanied by two forms of identification and a utility bill.",
        "Submissions received after the deadline will be subject to a late processing fee.",
        "If you require assistance completing this form, contact the help desk at ext. 4500.",
    ], rng)

    field_values = {
        "Full Name": person,
        "Address": address,
        "Date of Application": date,
        "Organization": company,
        "Reference Number": f"REF-{rng.randint(10000, 99999)}",
    }

    layout = (
        f"[{domain.upper()} {form_type}]\n\n"
        f"INSTRUCTIONS:\n"
        f"{instruction_text}\n\n"
        f"SECTION 1: Applicant Information\n"
        + "\n".join(f"  {k}: [{v}]" for k, v in field_values.items())
        + "\n\n"
        f"SECTION 2: Declaration\n"
        f"By signing below, I certify that all information provided is accurate and complete.\n"
        f"I understand that false statements may result in disqualification or legal action.\n\n"
        f"Signature: [signed: {person}]    Date: [{date}]"
    )

    query = (
        f"{layout}\n\n"
        f"Extract all form fields and their values from this {domain} {form_type}. "
        f"Separate instructions/prose from structured data."
    )
    reasoning = (
        f"Step 1: Document type: {form_type} with mixed form fields and prose instructions.\n"
        f"Step 2: Instructions section contains guidance text (not extractable fields).\n"
        f"Step 3: Section 1 contains structured form fields:\n"
        + "\n".join(f"  - {k}: {v}" for k, v in field_values.items())
        + f"\nStep 4: Section 2 is a prose declaration with an embedded signature field.\n"
        f"Step 5: Signature and date are extractable fields within the prose.\n"
        f"Confidence: High -- form fields clearly bracketed, prose is contextual."
    )
    answer = (
        f"**Extracted Form Fields:**\n\n"
        + "\n".join(f"- **{k}:** {v}" for k, v in field_values.items())
        + f"\n- **Signature:** {person}\n"
        f"- **Signature Date:** {date}\n\n"
        f"**Prose/Instructions (not field data):**\n"
        f"- Instructions: \"{instruction_text}\"\n"
        f"- Declaration text (standard legal language, not a fillable field)"
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Category 5: Page-spanning structures (200)
# ---------------------------------------------------------------------------

def _gen_page_spanning(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    templates = [lambda: _page_spanning_table(rng), lambda: _page_spanning_form(rng)]
    idx = 0
    while len(results) < count:
        fn = templates[idx % len(templates)]
        results.append(fn())
        idx += 1
    return results


def _page_spanning_table(rng: random.Random) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    company = _pick(_COMPANY_NAMES, rng)
    n_total_rows = rng.randint(25, 80)
    page1_rows = rng.randint(12, 20)
    page2_rows = n_total_rows - page1_rows
    headers = "Item | Description | Qty | Unit Price | Total"
    products = rng.sample(_PRODUCT_NAMES, k=min(3, len(_PRODUCT_NAMES)))

    page1_sample = [f"{products[i % len(products)]} | Desc {i+1} | {rng.randint(1,50)} | ${rng.uniform(10,500):,.2f} | ${rng.uniform(100,5000):,.2f}" for i in range(3)]
    page2_sample = [f"{products[i % len(products)]} | Desc {page1_rows+i+1} | {rng.randint(1,50)} | ${rng.uniform(10,500):,.2f} | ${rng.uniform(100,5000):,.2f}" for i in range(3)]

    layout = (
        f"--- PAGE 1 of 2 ---\n"
        f"[Table: {company} {domain} Line Items]\n"
        f"| {headers} |\n"
        + "\n".join(f"| {r} |" for r in page1_sample)
        + f"\n... ({page1_rows} rows on this page)\n"
        f"[TABLE CONTINUES ON NEXT PAGE]\n\n"
        f"--- PAGE 2 of 2 ---\n"
        f"[Table continued from previous page]\n"
        f"| {headers} |\n"
        + "\n".join(f"| {r} |" for r in page2_sample)
        + f"\n... ({page2_rows} rows on this page)\n"
        f"[END OF TABLE]"
    )
    query = (
        f"{layout}\n\n"
        f"This table spans two pages. How should the data be reconstructed?"
    )
    reasoning = (
        f"Step 1: Table starts on page 1 with {page1_rows} rows.\n"
        f"Step 2: Table continues on page 2 with {page2_rows} rows.\n"
        f"Step 3: Headers are repeated on page 2 ('{headers}') -- these are continuation headers, not new data.\n"
        f"Step 4: Total rows when merged: {page1_rows} + {page2_rows} = {n_total_rows}.\n"
        f"Step 5: The repeated header row on page 2 should be de-duplicated.\n"
        f"Confidence: High -- explicit page break markers and continuation note."
    )
    answer = (
        f"**Table reconstruction:**\n\n"
        f"- Merge page 1 ({page1_rows} rows) and page 2 ({page2_rows} rows) into a single table\n"
        f"- Remove the duplicate header row on page 2\n"
        f"- Total reconstructed rows: **{n_total_rows}**\n"
        f"- Columns: {headers}\n\n"
        f"The table should be treated as one continuous dataset spanning two pages."
    )
    return format_sft_example(query, reasoning, answer)


def _page_spanning_form(rng: random.Random) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    person = _pick(_PERSON_NAMES, rng)
    company = _pick(_COMPANY_NAMES, rng)
    layout = (
        f"--- PAGE 1 ---\n"
        f"[{domain.upper()} CLAIM FORM - Page 1 of 3]\n"
        f"Section A: Claimant Information\n"
        f"  Name: [{person}]\n"
        f"  Company: [{company}]\n"
        f"  Date: [{_rand_date(rng)}]\n\n"
        f"Section B: Incident Details\n"
        f"  Date of Incident: [{_rand_date(rng)}]\n"
        f"  Location: [Main Office, Floor 3]\n"
        f"  [CONTINUED ON PAGE 2]\n\n"
        f"--- PAGE 2 ---\n"
        f"Section B (continued):\n"
        f"  Description: [Equipment malfunction during routine maintenance...]\n"
        f"  Witnesses: [{_pick(_PERSON_NAMES, rng)}, {_pick(_PERSON_NAMES, rng)}]\n\n"
        f"Section C: Financial Impact\n"
        f"  Estimated Loss: [{_rand_amount(rng, 5000, 50000)}]\n"
        f"  [CONTINUED ON PAGE 3]\n\n"
        f"--- PAGE 3 ---\n"
        f"Section C (continued):\n"
        f"  Insurance Coverage: [Policy #{rng.randint(100000, 999999)}]\n"
        f"  Deductible: [{_rand_amount(rng, 500, 5000)}]\n\n"
        f"Section D: Signatures\n"
        f"  Claimant Signature: [signed: {person}]\n"
        f"  Date: [{_rand_date(rng)}]"
    )
    query = (
        f"{layout}\n\n"
        f"This claim form spans 3 pages with sections split across page breaks. "
        f"Reconstruct the complete form structure."
    )
    reasoning = (
        f"Step 1: Form spans 3 pages with 4 sections (A-D).\n"
        f"Step 2: Section A (page 1): Complete -- Claimant info.\n"
        f"Step 3: Section B starts on page 1, continues on page 2 -- split across pages.\n"
        f"Step 4: Section C starts on page 2, continues on page 3 -- split across pages.\n"
        f"Step 5: Section D (page 3): Complete -- Signatures.\n"
        f"Step 6: Must merge continued sections to reconstruct complete form.\n"
        f"Confidence: High -- continuation markers are explicit."
    )
    answer = (
        f"**Reconstructed form structure:**\n\n"
        f"- **Section A** (complete, page 1): Claimant info -- {person}, {company}\n"
        f"- **Section B** (pages 1-2): Incident details -- date, location, description, witnesses\n"
        f"- **Section C** (pages 2-3): Financial impact -- estimated loss, insurance, deductible\n"
        f"- **Section D** (complete, page 3): Signatures\n\n"
        f"Two sections (B and C) are split across page boundaries and must be merged."
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Category 6: Hierarchical headings (200)
# ---------------------------------------------------------------------------

def _gen_hierarchical_headings(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    idx = 0
    while len(results) < count:
        results.append(_heading_tree_example(rng, idx))
        idx += 1
    return results


def _heading_tree_example(rng: random.Random, variant: int) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    doc_type = _pick(DOC_TYPES, rng)
    company = _pick(_COMPANY_NAMES, rng)
    headings = _pick(_HEADING_STRUCTURES, rng)
    n_levels = 5

    heading_block = "\n".join(
        f"{'  ' * (h.count('.') + h.count(':'))}{'#' * min(h.count('.') + 1, 6)} {h}"
        for h in headings
    )

    layout = (
        f"[Document: {company} {domain} {doc_type}]\n\n"
        f"{heading_block}"
    )
    query = (
        f"{layout}\n\n"
        f"Build the heading hierarchy tree for this document. "
        f"Identify all nesting levels and parent-child relationships."
    )
    # Count nesting levels
    max_depth = max(h.count('.') + 1 for h in headings)
    top_level = [h for h in headings if '.' not in h.split(':')[-1].strip() and h.count('.') == 0]

    reasoning = (
        f"Step 1: Parse heading numbering to determine hierarchy.\n"
        f"Step 2: Identify nesting depth by numbering pattern (dots or roman numeral sub-levels).\n"
        f"Step 3: Maximum depth found: {max_depth} levels.\n"
        f"Step 4: Top-level headings: {len(top_level)}.\n"
        f"Step 5: Build parent-child tree based on numbering prefix.\n"
        f"Confidence: High -- heading numbering provides unambiguous hierarchy."
    )
    tree_lines = []
    for h in headings:
        depth = h.count('.')
        indent = "  " * depth
        tree_lines.append(f"{indent}- {h}")

    answer = (
        f"**Heading hierarchy ({max_depth}+ levels):**\n\n"
        + "\n".join(tree_lines)
        + f"\n\n**Statistics:**\n"
        f"- Total headings: {len(headings)}\n"
        f"- Maximum nesting depth: {max_depth} levels\n"
        f"- Top-level sections: {len(top_level)}"
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Category 7: Document type adaptation (300)
# ---------------------------------------------------------------------------

def _gen_doc_type_adaptation(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    idx = 0
    while len(results) < count:
        results.append(_doc_type_strategy(rng, idx))
        idx += 1
    return results


def _doc_type_strategy(rng: random.Random, variant: int) -> Dict[str, str]:
    doc_type = _pick(list(_FORM_FIELDS.keys()), rng)
    domain = _pick(DOMAINS, rng)
    company = _pick(_COMPANY_NAMES, rng)
    fields = _FORM_FIELDS[doc_type]

    # Different extraction strategies per doc type
    strategies = {
        "contract": "clause-based extraction with party identification and obligation mapping",
        "invoice": "line-item table extraction with header/footer field capture",
        "insurance_claim": "section-based form field extraction with narrative description parsing",
        "medical_record": "structured clinical data extraction with coded terminology (ICD-10, CPT)",
        "resume": "section-segmented extraction (experience, education, skills) with chronological ordering",
        "purchase_order": "tabular line-item extraction with metadata header parsing",
        "policy_document": "hierarchical section extraction with cross-reference resolution",
        "compliance_report": "findings-based extraction with risk classification",
    }
    strategy = strategies.get(doc_type, "general field extraction with section identification")

    # Build a realistic layout snippet for the doc type
    field_subset = rng.sample(fields, k=min(6, len(fields)))
    field_lines = "\n".join(f"  {f}: [sample value {i+1}]" for i, f in enumerate(field_subset))
    layout = (
        f"[Document detected: {doc_type.replace('_', ' ').title()}]\n"
        f"[Source: {company}, Domain: {domain}]\n\n"
        f"Visible fields:\n{field_lines}\n\n"
        f"[Additional content: tables, signatures, and prose sections]"
    )
    query = (
        f"{layout}\n\n"
        f"What extraction strategy should be used for this {doc_type.replace('_', ' ')} document? "
        f"List the expected fields and appropriate layout handling approach."
    )
    reasoning = (
        f"Step 1: Document type identified: {doc_type.replace('_', ' ').title()}.\n"
        f"Step 2: Domain context: {domain} -- this affects terminology and field semantics.\n"
        f"Step 3: Recommended strategy: {strategy}.\n"
        f"Step 4: Expected fields for {doc_type}: {', '.join(fields[:8])}{'...' if len(fields) > 8 else ''}.\n"
        f"Step 5: Currently visible: {len(field_subset)} of {len(fields)} expected fields.\n"
        f"Confidence: High -- document type is clearly identified."
    )
    answer = (
        f"**Document Type:** {doc_type.replace('_', ' ').title()}\n\n"
        f"**Extraction Strategy:** {strategy}\n\n"
        f"**Expected Fields ({len(fields)}):**\n"
        + "\n".join(f"- {f}" for f in fields)
        + f"\n\n**Layout Handling:**\n"
        f"- Parse structured form fields in bracketed sections\n"
        f"- Extract tabular data with header alignment\n"
        f"- Separate prose/instructions from field values\n"
        f"- Verify all {len(fields)} expected fields are captured"
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Category 8: Completeness verification (400 SFT + ~200 DPO)
# ---------------------------------------------------------------------------

def _gen_completeness_verification(
    sft_count: int, dpo_count: int, rng: random.Random
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    sft_results: List[Dict[str, str]] = []
    dpo_results: List[Dict[str, str]] = []

    # SFT examples
    idx = 0
    while len(sft_results) < sft_count:
        sft_results.append(_completeness_sft(rng, idx))
        idx += 1

    # DPO examples
    idx = 0
    while len(dpo_results) < dpo_count:
        dpo_results.append(_completeness_dpo(rng, idx))
        idx += 1

    return sft_results, dpo_results


def _completeness_sft(rng: random.Random, variant: int) -> Dict[str, str]:
    doc_type = _pick(list(_FORM_FIELDS.keys()), rng)
    domain = _pick(DOMAINS, rng)
    company = _pick(_COMPANY_NAMES, rng)
    person = _pick(_PERSON_NAMES, rng)
    all_fields = _FORM_FIELDS[doc_type]

    # Simulate partial extraction -- some fields found, some missing
    n_extracted = rng.randint(len(all_fields) - 4, len(all_fields))
    extracted_fields = rng.sample(all_fields, k=n_extracted)
    missing_fields = [f for f in all_fields if f not in extracted_fields]

    extracted_lines = "\n".join(
        f"  - {f}: [value_{i+1}]" for i, f in enumerate(extracted_fields)
    )
    layout = (
        f"[Extraction result from {company}'s {domain} {doc_type.replace('_', ' ')}]\n\n"
        f"Extracted fields:\n{extracted_lines}"
    )

    missing_text = ", ".join(missing_fields) if missing_fields else "None -- all fields captured"
    n_missing = len(missing_fields)
    confidence = "high" if n_missing == 0 else ("medium" if n_missing <= 2 else "low")

    query = (
        f"{layout}\n\n"
        f"Verify the completeness of this extraction against the expected schema for a "
        f"{doc_type.replace('_', ' ')} document."
    )
    reasoning = (
        f"Step 1: Document type: {doc_type.replace('_', ' ').title()} -- I know the expected field schema.\n"
        f"Step 2: Expected fields: {len(all_fields)} total.\n"
        f"Step 3: Extracted fields: {len(extracted_fields)}.\n"
        f"Step 4: Cross-reference each expected field against extracted list.\n"
        f"Step 5: Missing: {missing_text}.\n"
        f"Step 6: Completeness: {len(extracted_fields)}/{len(all_fields)} = "
        f"{len(extracted_fields)/len(all_fields)*100:.0f}%.\n"
        f"Confidence: {confidence}."
    )
    answer = (
        f"**Completeness Check:**\n"
        f"- Expected fields for {doc_type.replace('_', ' ')}: {', '.join(all_fields)}\n"
        f"- Extracted: {', '.join(extracted_fields)}\n"
        f"- Missing: {missing_text}\n"
        f"- Confidence: {confidence}\n\n"
        f"**Result:** {len(extracted_fields)}/{len(all_fields)} fields captured "
        f"({len(extracted_fields)/len(all_fields)*100:.0f}% complete)."
        + (f"\n\n**Action required:** Re-examine the document for: {', '.join(missing_fields)}" if missing_fields else "")
    )
    return format_sft_example(query, reasoning, answer)


def _completeness_dpo(rng: random.Random, variant: int) -> Dict[str, str]:
    doc_type = _pick(list(_FORM_FIELDS.keys()), rng)
    domain = _pick(DOMAINS, rng)
    company = _pick(_COMPANY_NAMES, rng)
    all_fields = _FORM_FIELDS[doc_type]

    # Chosen: catches all fields and flags missing ones
    # Rejected: silently omits some fields without mentioning them
    n_omitted = rng.randint(2, 4)
    omitted_fields = rng.sample(all_fields, k=n_omitted)
    kept_fields = [f for f in all_fields if f not in omitted_fields]

    extracted_lines = "\n".join(f"  - {f}: [value]" for f in all_fields)
    layout = (
        f"[{company} {domain} {doc_type.replace('_', ' ')}]\n\n"
        f"Document content:\n{extracted_lines}"
    )
    query = (
        f"{layout}\n\n"
        f"Extract all fields from this {doc_type.replace('_', ' ')} and verify completeness."
    )

    # Chosen response: complete extraction with verification
    chosen_reasoning = (
        f"Step 1: This is a {doc_type.replace('_', ' ')} with {len(all_fields)} expected fields.\n"
        f"Step 2: Systematically extract each field from the document.\n"
        f"Step 3: Cross-reference against the complete field schema.\n"
        f"Step 4: All {len(all_fields)} fields found -- extraction is complete.\n"
        f"Confidence: High -- verified against schema."
    )
    chosen_answer = (
        f"**Extracted Fields ({len(all_fields)}/{len(all_fields)}):**\n"
        + "\n".join(f"- **{f}:** [value]" for f in all_fields)
        + f"\n\n**Completeness Check:**\n"
        f"- Expected fields for {doc_type.replace('_', ' ')}: {', '.join(all_fields)}\n"
        f"- Extracted: {', '.join(all_fields)}\n"
        f"- Missing: None -- all fields captured\n"
        f"- Confidence: high"
    )

    # Rejected response: silently omits fields, no completeness check
    rejected_reasoning = (
        f"Step 1: This is a {doc_type.replace('_', ' ')}.\n"
        f"Step 2: Extract the visible fields.\n"
        f"Step 3: Return the extracted data."
    )
    rejected_answer = (
        f"**Extracted Fields:**\n"
        + "\n".join(f"- **{f}:** [value]" for f in kept_fields)
        + f"\n\nExtraction complete."
    )

    return format_dpo_example(
        query, chosen_reasoning, chosen_answer, rejected_reasoning, rejected_answer
    )


# ---------------------------------------------------------------------------
# Category 9: Edge cases & negatives (300)
# ---------------------------------------------------------------------------

def _gen_edge_cases(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    templates = [
        lambda: _edge_ambiguous_layout(rng),
        lambda: _edge_overlapping_tables(rng),
        lambda: _edge_empty_sections(rng),
        lambda: _edge_rotated_text(rng),
        lambda: _edge_watermark_interference(rng),
    ]
    idx = 0
    while len(results) < count:
        fn = templates[idx % len(templates)]
        results.append(fn())
        idx += 1
    return results


def _edge_ambiguous_layout(rng: random.Random) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    company = _pick(_COMPANY_NAMES, rng)
    layout = (
        f"[Ambiguous layout from {company}'s {domain} document]\n\n"
        f"Text block A (left-aligned):\n"
        f"  Revenue figures for the current quarter show...\n\n"
        f"Text block B (slightly offset, overlapping vertical space with A):\n"
        f"  Cost analysis for the same period indicates...\n\n"
        f"Text block C (below A and B, unclear alignment):\n"
        f"  The net margin calculation depends on which figures above...\n"
    )
    query = (
        f"{layout}\n\n"
        f"The layout is ambiguous -- blocks A and B overlap vertically. "
        f"What is the correct reading order, and how confident are you?"
    )
    reasoning = (
        f"Step 1: Block A is left-aligned, Block B is offset but overlaps A's vertical space.\n"
        f"Step 2: This could be a two-column layout (A left, B right) OR sequential text with indentation.\n"
        f"Step 3: Without exact coordinates, I cannot definitively determine the intended order.\n"
        f"Step 4: Block C references 'figures above' -- it depends on both A and B.\n"
        f"Step 5: Most likely: two-column layout with C as a full-width conclusion.\n"
        f"Confidence: Low -- layout is genuinely ambiguous without precise coordinates."
    )
    answer = (
        f"**Layout interpretation (uncertain):**\n\n"
        f"The layout is **ambiguous**. Two plausible interpretations:\n\n"
        f"1. **Two-column layout:** A (left) and B (right) are parallel, then C follows\n"
        f"2. **Sequential with indentation:** A -> B -> C in order\n\n"
        f"**Confidence: Low** -- precise bounding box coordinates are needed to resolve this ambiguity. "
        f"I recommend examining the original document's spatial layout before proceeding."
    )
    return format_sft_example(query, reasoning, answer)


def _edge_overlapping_tables(rng: random.Random) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    layout = (
        f"[Page from {domain} report with overlapping elements]\n\n"
        f"Table 1 (main content):\n"
        f"| Product | Q1 | Q2 |\n"
        f"| Widget A | $10,000 | $12,000 |\n\n"
        f"[OVERLAPPING: A sidebar table partially covers the right edge of Table 1]\n"
        f"Sidebar table:\n"
        f"| Metric | Value |\n"
        f"| Growth | 15% |\n"
        f"| Target | 20% |\n"
    )
    query = (
        f"{layout}\n\n"
        f"Two tables overlap on this page. How should the extraction handle this?"
    )
    reasoning = (
        f"Step 1: Table 1 is the main content table (Product x Quarter).\n"
        f"Step 2: A sidebar table overlaps Table 1's right edge.\n"
        f"Step 3: Overlapping regions may cause OCR misalignment or merged cells.\n"
        f"Step 4: Strategy: extract each table independently using their bounding boxes.\n"
        f"Step 5: Flag the overlap region for manual review.\n"
        f"Confidence: Medium -- overlap may cause data corruption in the shared region."
    )
    answer = (
        f"**Extraction approach for overlapping tables:**\n\n"
        f"1. Extract Table 1 (main) and Table 2 (sidebar) as **separate tables**\n"
        f"2. **Flag the overlap region** -- data in the shared area may be corrupted\n"
        f"3. Verify Table 1's rightmost column values against source\n"
        f"4. **Confidence: Medium** -- overlapping layout requires manual verification\n\n"
        f"Recommendation: Request the original digital file rather than relying on the scanned layout."
    )
    return format_sft_example(query, reasoning, answer)


def _edge_empty_sections(rng: random.Random) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    doc_type = _pick(list(_FORM_FIELDS.keys()), rng)
    fields = _FORM_FIELDS[doc_type]
    filled = rng.sample(fields, k=len(fields) // 2)
    empty = [f for f in fields if f not in filled]
    layout = (
        f"[{domain} {doc_type.replace('_', ' ')} extraction]\n\n"
        + "\n".join(f"  {f}: {'[value provided]' if f in filled else '[BLANK]'}" for f in fields)
    )
    query = (
        f"{layout}\n\n"
        f"Several sections of this {doc_type.replace('_', ' ')} are blank. "
        f"Determine which are required and which are optional."
    )
    # Treat first 60% of fields as required
    n_required = int(len(fields) * 0.6)
    required = fields[:n_required]
    optional = fields[n_required:]
    missing_required = [f for f in required if f in empty]
    reasoning = (
        f"Step 1: Document has {len(fields)} total fields, {len(filled)} filled, {len(empty)} blank.\n"
        f"Step 2: For a {doc_type.replace('_', ' ')}, typically the first {n_required} fields are required.\n"
        f"Step 3: Required fields: {', '.join(required)}.\n"
        f"Step 4: Missing required fields: {', '.join(missing_required) if missing_required else 'None'}.\n"
        f"Step 5: Optional blank fields: {', '.join([f for f in optional if f in empty]) or 'None'}.\n"
        f"Confidence: Medium -- 'required' determination is based on standard schemas."
    )
    answer = (
        f"**Field completeness analysis:**\n\n"
        f"- **Filled:** {len(filled)}/{len(fields)} fields\n"
        f"- **Blank:** {len(empty)}/{len(fields)} fields\n\n"
        f"**Missing required fields ({len(missing_required)}):** "
        f"{', '.join(missing_required) if missing_required else 'None'}\n\n"
        f"**Blank optional fields:** "
        f"{', '.join([f for f in optional if f in empty]) or 'None'}\n\n"
        + (f"**Action:** The form is incomplete. Missing required fields must be filled before processing."
           if missing_required else f"**Status:** All required fields are present. Form is valid for processing.")
    )
    return format_sft_example(query, reasoning, answer)


def _edge_rotated_text(rng: random.Random) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    rotation = _pick([90, 270, 45, -45], rng)
    text_content = _pick([
        "CONFIDENTIAL - DO NOT DISTRIBUTE",
        "DRAFT - For Internal Review Only",
        "APPROVED - Final Version",
        "SAMPLE - Not for Production Use",
    ], rng)
    layout = (
        f"[{domain} document page]\n\n"
        f"Main content: Standard left-to-right text about {domain} procedures...\n\n"
        f"[ROTATED TEXT: '{text_content}' at {rotation} degrees along the left margin]"
    )
    query = (
        f"{layout}\n\n"
        f"There is rotated text at {rotation} degrees. How should this be extracted and classified?"
    )
    is_watermark = "CONFIDENTIAL" in text_content or "DRAFT" in text_content
    classification = "watermark/stamp" if is_watermark else "label/annotation"
    reasoning = (
        f"Step 1: Rotated text detected at {rotation} degrees: '{text_content}'.\n"
        f"Step 2: Position: along the left margin (not inline with body text).\n"
        f"Step 3: Content analysis: '{text_content}' is a {classification}.\n"
        f"Step 4: This text should be extracted separately from the body content.\n"
        f"Step 5: Classification: {classification} -- metadata, not body text.\n"
        f"Confidence: High -- rotated margin text is typically a {classification}."
    )
    answer = (
        f"**Rotated text extraction:**\n\n"
        f"- **Text:** \"{text_content}\"\n"
        f"- **Rotation:** {rotation} degrees\n"
        f"- **Position:** Left margin\n"
        f"- **Classification:** {classification}\n\n"
        f"This text should be captured as **document metadata** (not body text) "
        f"and tagged as a {classification} in the extraction output."
    )
    return format_sft_example(query, reasoning, answer)


def _edge_watermark_interference(rng: random.Random) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    company = _pick(_COMPANY_NAMES, rng)
    watermark = _pick(["CONFIDENTIAL", "DRAFT", "COPY", company.upper()], rng)
    layout = (
        f"[Scanned {domain} document from {company}]\n\n"
        f"[WATERMARK: '{watermark}' diagonally across page, 50% opacity]\n\n"
        f"Underlying text (partially obscured):\n"
        f"  The annual {domain} review for {company} concludes that...\n"
        f"  [text partially obscured by watermark in middle section]\n"
        f"  ...recommendations include increasing the budget by 15%."
    )
    query = (
        f"{layout}\n\n"
        f"A diagonal watermark partially obscures the text. "
        f"Extract what is readable and flag obscured sections."
    )
    reasoning = (
        f"Step 1: Watermark '{watermark}' is diagonal at 50% opacity.\n"
        f"Step 2: Some text is readable despite the watermark.\n"
        f"Step 3: Middle section is marked as 'partially obscured'.\n"
        f"Step 4: Extract readable text, flag the gap, and note the watermark.\n"
        f"Confidence: Medium -- partial text recovery possible, but gaps exist."
    )
    answer = (
        f"**Extraction with watermark handling:**\n\n"
        f"**Watermark:** \"{watermark}\" (diagonal, 50% opacity)\n\n"
        f"**Extracted text:**\n"
        f"- \"The annual {domain} review for {company} concludes that...\"\n"
        f"- [OBSCURED SECTION -- watermark interference]\n"
        f"- \"...recommendations include increasing the budget by 15%.\"\n\n"
        f"**Confidence: Medium** -- middle section could not be reliably extracted. "
        f"Consider requesting a non-watermarked copy for complete extraction."
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_track2_data(output_dir: str | Path, seed: int = 42) -> dict:
    """Generate Track 2 Layout Intelligence training data.

    Args:
        output_dir: Directory to write JSONL files into.
        seed: Random seed for reproducibility.

    Returns:
        Dict with ``sft_path``, ``dpo_path``, ``sft_count``, ``dpo_count``.
    """
    rng = random.Random(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_path = output_dir / "track2_layout_sft.jsonl"
    dpo_path = output_dir / "track2_layout_dpo.jsonl"

    # Generate SFT examples per category
    sft_categories = [
        ("nested_tables", _gen_nested_tables, 300),
        ("merged_cells", _gen_merged_cells, 300),
        ("multicolumn_interruptions", _gen_multicolumn_interruptions, 250),
        ("mixed_form_prose", _gen_mixed_form_prose, 250),
        ("page_spanning", _gen_page_spanning, 200),
        ("hierarchical_headings", _gen_hierarchical_headings, 200),
        ("doc_type_adaptation", _gen_doc_type_adaptation, 300),
        # completeness_verification handled separately for DPO
        ("edge_cases", _gen_edge_cases, 300),
    ]

    all_sft: List[Dict[str, str]] = []
    for name, gen_fn, count in sft_categories:
        sub_seed = seed + hash(name) % 10000
        cat_rng = random.Random(sub_seed)
        examples = gen_fn(count, cat_rng)
        all_sft.extend(examples)

    # Completeness verification: 400 SFT + ~200 DPO
    comp_rng = random.Random(seed + hash("completeness") % 10000)
    comp_sft, comp_dpo = _gen_completeness_verification(400, 200, comp_rng)
    all_sft.extend(comp_sft)

    rng.shuffle(all_sft)

    with JSONLWriter(sft_path) as writer:
        for ex in all_sft:
            writer.write(ex)

    rng2 = random.Random(seed + 1)
    rng2.shuffle(comp_dpo)

    with JSONLWriter(dpo_path) as writer:
        for ex in comp_dpo:
            writer.write(ex)

    return {
        "sft_path": str(sft_path),
        "dpo_path": str(dpo_path),
        "sft_count": len(all_sft),
        "dpo_count": len(comp_dpo),
    }


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/finetune/v2")
    result = generate_track2_data(out)
    print(f"Track 2 Layout: {result['sft_count']} SFT examples -> {result['sft_path']}")
    print(f"Track 2 Layout: {result['dpo_count']} DPO examples -> {result['dpo_path']}")
