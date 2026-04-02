"""Phase 2.5 DPO Preference Pair generator for DocWain V2+ finetuning.

Generates 5K DPO preference pairs (at scale=1.0) across five corruption types:
  1. Reasoning corruption   — good vs sloppy/wrong reasoning
  2. Hallucination injection — plausible but fabricated values
  3. Over-confidence         — states "High confidence" on ambiguous data
  4. Omission corruption     — misses clearly present information
  5. Structure corruption    — breaks table/entity format

Each pair is formatted via ``format_dpo_example`` for TRL DPOTrainer.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

from src.finetune.v2.data_generator.base import (
    DOMAINS,
    DOC_TYPES,
    JSONLWriter,
    format_dpo_example,
)

# ---------------------------------------------------------------------------
# Helpers (same pattern as phase2_doc_intelligence)
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

CORRUPTION_TYPES = [
    "reasoning_corruption",
    "hallucination_injection",
    "over_confidence",
    "omission_corruption",
    "structure_corruption",
]


def _pick(lst: list, rng: random.Random) -> Any:
    return rng.choice(lst)


def _rand_amount(rng: random.Random, lo: float = 100.0, hi: float = 99999.0) -> str:
    return f"{rng.uniform(lo, hi):,.2f}"


def _rand_int(rng: random.Random, lo: int = 1, hi: int = 500) -> int:
    return rng.randint(lo, hi)


# ---------------------------------------------------------------------------
# DPO Template pairs: (query, chosen_reasoning, chosen_answer,
#                       rejected_reasoning, rejected_answer, corruption_type)
# ---------------------------------------------------------------------------

_DPO_TEMPLATES: List[Dict[str, str]] = [
    # 1. Reasoning corruption — invoice extraction
    {
        "corruption_type": "reasoning_corruption",
        "query": (
            "Extract the total amount from this {domain} invoice from {company}:\n"
            "| Item | Qty | Unit Price | Total |\n"
            "| {product} | {qty} | {currency}{amount} | {currency}{amount2} |\n"
            "| Shipping | 1 | {currency}{amount3} | {currency}{amount3} |\n\n"
            "What is the grand total?"
        ),
        "chosen_reasoning": (
            "Step 1: Identify the invoice table from {company} with two line items.\n"
            "Step 2: Line 1 total = {currency}{amount2} (for {product}).\n"
            "Step 3: Line 2 total = {currency}{amount3} (shipping).\n"
            "Step 4: Grand total = {currency}{amount2} + {currency}{amount3}.\n"
            "Confidence: High — both line totals are explicitly stated."
        ),
        "chosen_answer": (
            "The grand total is {currency}{amount2} + {currency}{amount3} "
            "(sum of {product} and shipping line items from {company}'s invoice)."
        ),
        "rejected_reasoning": (
            "I see a table. The total is {currency}{amount2}. Done."
        ),
        "rejected_answer": (
            "The total is {currency}{amount2}."
        ),
    },
    # 2. Hallucination injection — entity extraction
    {
        "corruption_type": "hallucination_injection",
        "query": (
            "From this {domain} {doc_type}, extract all named entities:\n\n"
            "'{person} signed the agreement on behalf of {company} on {month} 15, {year}. "
            "The contract covers {product} deployment in the {dept} department.'"
        ),
        "chosen_reasoning": (
            "Step 1: Identify entity types present.\n"
            "Step 2: Person = '{person}'.\n"
            "Step 3: Organization = '{company}'.\n"
            "Step 4: Date = '{month} 15, {year}'.\n"
            "Step 5: Product = '{product}'.\n"
            "Step 6: Department = '{dept}'.\n"
            "Confidence: High — all entities are explicitly mentioned in the text."
        ),
        "chosen_answer": (
            "Entities extracted:\n"
            "- Person: {person}\n"
            "- Organization: {company}\n"
            "- Date: {month} 15, {year}\n"
            "- Product: {product}\n"
            "- Department: {dept}"
        ),
        "rejected_reasoning": (
            "Step 1: I see several entities in this document.\n"
            "Step 2: Person = '{person}', also '{person}' appears to be the CFO.\n"
            "Step 3: Organization = '{company}' (headquartered in Delaware).\n"
            "Step 4: Date = '{month} 15, {year}', with a renewal date of {month} 15, 2027.\n"
            "Step 5: Product = '{product}'.\n"
            "Step 6: Department = '{dept}'.\n"
            "Confidence: High."
        ),
        "rejected_answer": (
            "Entities extracted:\n"
            "- Person: {person} (CFO)\n"
            "- Organization: {company} (Delaware)\n"
            "- Date: {month} 15, {year}; Renewal: {month} 15, 2027\n"
            "- Product: {product}\n"
            "- Department: {dept}"
        ),
    },
    # 3. Over-confidence — contract terms with ambiguity
    {
        "corruption_type": "over_confidence",
        "query": (
            "Review this {domain} contract clause from {company}:\n\n"
            "'The service fee shall be {currency}{amount} per {quarter}, subject to "
            "adjustment based on mutually agreed performance metrics.'\n\n"
            "What is the annual cost of this contract?"
        ),
        "chosen_reasoning": (
            "Step 1: The clause states a quarterly fee of {currency}{amount}.\n"
            "Step 2: Naive annual = 4 x {currency}{amount}.\n"
            "Step 3: However, the clause says 'subject to adjustment based on "
            "mutually agreed performance metrics' — the actual annual cost is uncertain.\n"
            "Confidence: Medium — base amount is clear but adjustments are undefined."
        ),
        "chosen_answer": (
            "The base annual cost would be 4 x {currency}{amount} = approximately "
            "{currency}{amount2} per year. However, the clause includes a performance-based "
            "adjustment mechanism that could change the actual amount. The final annual cost "
            "cannot be determined without knowing the agreed performance metrics."
        ),
        "rejected_reasoning": (
            "Step 1: Quarterly fee = {currency}{amount}.\n"
            "Step 2: Annual = 4 x {currency}{amount}.\n"
            "Confidence: High — the calculation is straightforward."
        ),
        "rejected_answer": (
            "The annual cost of this contract is exactly 4 x {currency}{amount}. "
            "This is definitive."
        ),
    },
    # 4. Omission corruption — financial table missing key info
    {
        "corruption_type": "omission_corruption",
        "query": (
            "Analyze this {domain} financial summary for {company}:\n\n"
            "| Metric | {quarter} {year} |\n"
            "| Revenue | {currency}{amount} |\n"
            "| COGS | {currency}{amount2} |\n"
            "| Operating Expenses | {currency}{amount3} |\n"
            "| Net Income | {currency}{amount} |\n\n"
            "Provide a complete financial overview."
        ),
        "chosen_reasoning": (
            "Step 1: Four financial metrics reported for {company} in {quarter} {year}.\n"
            "Step 2: Revenue = {currency}{amount}.\n"
            "Step 3: COGS = {currency}{amount2} (Cost of Goods Sold).\n"
            "Step 4: Operating Expenses = {currency}{amount3}.\n"
            "Step 5: Net Income = {currency}{amount}.\n"
            "Step 6: Gross Profit = Revenue - COGS = {currency}{amount} - {currency}{amount2}.\n"
            "Confidence: High — all four metrics are explicitly stated."
        ),
        "chosen_answer": (
            "Financial Overview for {company} ({quarter} {year}):\n"
            "- Revenue: {currency}{amount}\n"
            "- COGS: {currency}{amount2}\n"
            "- Operating Expenses: {currency}{amount3}\n"
            "- Net Income: {currency}{amount}\n"
            "- Derived Gross Profit: {currency}{amount} - {currency}{amount2}"
        ),
        "rejected_reasoning": (
            "Step 1: Looking at {company}'s financials.\n"
            "Step 2: Revenue is {currency}{amount}.\n"
            "Step 3: Net Income is {currency}{amount}.\n"
            "Confidence: High."
        ),
        "rejected_answer": (
            "Financial Overview for {company}:\n"
            "- Revenue: {currency}{amount}\n"
            "- Net Income: {currency}{amount}"
        ),
    },
    # 5. Structure corruption — table/entity format broken
    {
        "corruption_type": "structure_corruption",
        "query": (
            "Extract the vendor details from this {domain} {doc_type}:\n\n"
            "Vendor: {company}\n"
            "Contact: {person}\n"
            "Department: {dept}\n"
            "Amount: {currency}{amount}\n"
            "Date: {month} {year}"
        ),
        "chosen_reasoning": (
            "Step 1: Document contains structured vendor information.\n"
            "Step 2: Parse each field — Vendor, Contact, Department, Amount, Date.\n"
            "Step 3: All fields are clearly labeled and present.\n"
            "Confidence: High — straightforward field extraction."
        ),
        "chosen_answer": (
            "| Field | Value |\n"
            "| --- | --- |\n"
            "| Vendor | {company} |\n"
            "| Contact | {person} |\n"
            "| Department | {dept} |\n"
            "| Amount | {currency}{amount} |\n"
            "| Date | {month} {year} |"
        ),
        "rejected_reasoning": (
            "Step 1: I see vendor data.\n"
            "Step 2: Extracting fields.\n"
            "Confidence: High."
        ),
        "rejected_answer": (
            "vendor={company}, contact={person}, dept is {dept} and the amount "
            "is around {currency}{amount} for {month} {year} or thereabouts"
        ),
    },
    # 6. Reasoning corruption — OCR correction
    {
        "corruption_type": "reasoning_corruption",
        "query": (
            "This {domain} document was OCR-scanned. Correct any errors:\n\n"
            "'{company} lnvoice #INV-{qty}\n"
            "Billed to: {person}\n"
            "Arnount Due: {currency}{amount}\n"
            "Oue Date: {month} {year}'"
        ),
        "chosen_reasoning": (
            "Step 1: Scan for OCR artefacts in the text.\n"
            "Step 2: 'lnvoice' -> 'Invoice' (lowercase L instead of capital I).\n"
            "Step 3: 'Arnount' -> 'Amount' (OCR misread 'm' as 'rn').\n"
            "Step 4: 'Oue' -> 'Due' (OCR misread 'D' as 'O').\n"
            "Step 5: All other text appears correct.\n"
            "Confidence: High — common OCR substitution patterns identified."
        ),
        "chosen_answer": (
            "Corrected text:\n"
            "'{company} Invoice #INV-{qty}\n"
            "Billed to: {person}\n"
            "Amount Due: {currency}{amount}\n"
            "Due Date: {month} {year}'"
        ),
        "rejected_reasoning": (
            "The text looks mostly fine. I'll fix one typo."
        ),
        "rejected_answer": (
            "Corrected: '{company} Invoice #INV-{qty}\n"
            "Billed to: {person}\n"
            "Arnount Due: {currency}{amount}\n"
            "Due Date: {month} {year}'"
        ),
    },
    # 7. Hallucination injection — org structure
    {
        "corruption_type": "hallucination_injection",
        "query": (
            "From this {domain} org chart excerpt:\n\n"
            "CEO: {person}\n"
            "  VP of {dept}: [Vacant]\n"
            "    Director: {person}\n\n"
            "Who reports to whom?"
        ),
        "chosen_reasoning": (
            "Step 1: Org chart has three levels.\n"
            "Step 2: CEO = {person} (top level).\n"
            "Step 3: VP of {dept} = Vacant (reports to CEO).\n"
            "Step 4: Director = {person} (reports to VP of {dept}).\n"
            "Step 5: Note the VP position is vacant.\n"
            "Confidence: High — hierarchy is explicitly laid out."
        ),
        "chosen_answer": (
            "Reporting structure:\n"
            "- {person} (Director) reports to VP of {dept} (currently vacant)\n"
            "- VP of {dept} reports to {person} (CEO)\n"
            "Note: The VP of {dept} position is currently vacant."
        ),
        "rejected_reasoning": (
            "Step 1: The org chart shows a standard structure.\n"
            "Step 2: CEO = {person}, VP of {dept} = interim filled by {person}, "
            "Director = {person}.\n"
            "Step 3: There are also 3 team leads under the Director.\n"
            "Confidence: High."
        ),
        "rejected_answer": (
            "Reporting structure:\n"
            "- 3 Team Leads report to {person} (Director)\n"
            "- {person} (Director) reports to {person} (interim VP of {dept})\n"
            "- VP of {dept} reports to {person} (CEO)"
        ),
    },
    # 8. Over-confidence — compliance check with gaps
    {
        "corruption_type": "over_confidence",
        "query": (
            "Review this {domain} compliance checklist for {company}:\n\n"
            "- Data encryption: Implemented\n"
            "- Access logging: Implemented\n"
            "- Incident response plan: Under review\n"
            "- Third-party audit: Scheduled for {quarter} {year}\n\n"
            "Is {company} fully compliant?"
        ),
        "chosen_reasoning": (
            "Step 1: Four compliance items listed.\n"
            "Step 2: Data encryption — done.\n"
            "Step 3: Access logging — done.\n"
            "Step 4: Incident response plan — 'Under review' means NOT yet finalized.\n"
            "Step 5: Third-party audit — only 'Scheduled', not completed.\n"
            "Step 6: Two out of four items are incomplete.\n"
            "Confidence: High on assessment — {company} is NOT fully compliant."
        ),
        "chosen_answer": (
            "{company} is NOT fully compliant. While data encryption and access logging "
            "are implemented, the incident response plan is still under review and the "
            "third-party audit is only scheduled for {quarter} {year}, not yet completed. "
            "Two of four compliance items remain incomplete."
        ),
        "rejected_reasoning": (
            "Step 1: Checking compliance items.\n"
            "Step 2: Everything looks good — implemented or in progress.\n"
            "Confidence: High — {company} is compliant."
        ),
        "rejected_answer": (
            "{company} is fully compliant. All items are either implemented or "
            "actively being addressed, which demonstrates strong compliance posture."
        ),
    },
    # 9. Omission corruption — multi-field document
    {
        "corruption_type": "omission_corruption",
        "query": (
            "Summarize all key terms from this {domain} {doc_type} between "
            "{company} and Globex Industries:\n\n"
            "- Effective Date: {month} 1, {year}\n"
            "- Term: 24 months\n"
            "- Value: {currency}{amount}\n"
            "- Payment Terms: Net 30\n"
            "- Governing Law: State of Delaware\n"
            "- Termination Clause: 90-day written notice"
        ),
        "chosen_reasoning": (
            "Step 1: This is a {domain} {doc_type} between {company} and Globex Industries.\n"
            "Step 2: Six key terms identified.\n"
            "Step 3: Effective Date = {month} 1, {year}.\n"
            "Step 4: Term = 24 months (expiry around {month} {year} + 2 years).\n"
            "Step 5: Value = {currency}{amount}.\n"
            "Step 6: Payment Terms = Net 30.\n"
            "Step 7: Governing Law = Delaware.\n"
            "Step 8: Termination = 90-day written notice.\n"
            "Confidence: High — all terms explicitly stated."
        ),
        "chosen_answer": (
            "Key Terms Summary ({company} / Globex Industries {doc_type}):\n"
            "- Effective Date: {month} 1, {year}\n"
            "- Term: 24 months\n"
            "- Contract Value: {currency}{amount}\n"
            "- Payment Terms: Net 30\n"
            "- Governing Law: State of Delaware\n"
            "- Termination: 90-day written notice required"
        ),
        "rejected_reasoning": (
            "Step 1: Contract between {company} and Globex.\n"
            "Step 2: Value is {currency}{amount}.\n"
            "Step 3: It starts {month} {year}.\n"
            "Confidence: High."
        ),
        "rejected_answer": (
            "The contract is worth {currency}{amount} and starts {month} {year}."
        ),
    },
    # 10. Structure corruption — financial table formatting
    {
        "corruption_type": "structure_corruption",
        "query": (
            "Format the following {domain} data as a structured table:\n\n"
            "{company} {quarter} {year} Results:\n"
            "Revenue: {currency}{amount}\n"
            "Expenses: {currency}{amount2}\n"
            "Profit: {currency}{amount3}\n"
            "Headcount: {qty}"
        ),
        "chosen_reasoning": (
            "Step 1: Four data points to structure for {company} {quarter} {year}.\n"
            "Step 2: Create a clean two-column table with Metric and Value.\n"
            "Step 3: Include all four metrics with proper formatting.\n"
            "Confidence: High — straightforward restructuring."
        ),
        "chosen_answer": (
            "**{company} {quarter} {year} Results**\n\n"
            "| Metric | Value |\n"
            "| --- | --- |\n"
            "| Revenue | {currency}{amount} |\n"
            "| Expenses | {currency}{amount2} |\n"
            "| Profit | {currency}{amount3} |\n"
            "| Headcount | {qty} |"
        ),
        "rejected_reasoning": (
            "I'll format this data."
        ),
        "rejected_answer": (
            "{company} had revenue of {currency}{amount} and expenses "
            "of {currency}{amount2} and profit was {currency}{amount3} with "
            "{qty} people in {quarter} {year}"
        ),
    },
]


# ---------------------------------------------------------------------------
# Expansion with DPO-specific handling
# ---------------------------------------------------------------------------

def _expand_dpo_templates(
    templates: List[Dict[str, str]],
    target_count: int,
    rng: random.Random,
) -> List[Dict[str, str]]:
    """Cycle through DPO templates with domain/doc_type variation."""
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
        }

        try:
            query = tpl["query"].format(**subs)
            chosen_reasoning = tpl["chosen_reasoning"].format(**subs)
            chosen_answer = tpl["chosen_answer"].format(**subs)
            rejected_reasoning = tpl["rejected_reasoning"].format(**subs)
            rejected_answer = tpl["rejected_answer"].format(**subs)
        except (KeyError, IndexError):
            idx += 1
            continue

        results.append(format_dpo_example(
            query, chosen_reasoning, chosen_answer,
            rejected_reasoning, rejected_answer,
        ))
        idx += 1

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_dpo_pairs(count: int = 5000, *, seed: int = 50) -> List[Dict[str, str]]:
    """Generate DPO preference pairs across 5 corruption types.

    Args:
        count: Number of pairs to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts with prompt/chosen/rejected keys.
    """
    rng = random.Random(seed)
    results = _expand_dpo_templates(_DPO_TEMPLATES, count, rng)
    rng.shuffle(results)
    return results


def generate_phase25_data(output_dir: Path, scale: float = 1.0) -> int:
    """Generate Phase 2.5 DPO training data and write to JSONL.

    Args:
        output_dir: Directory to write the JSONL file into.
        scale: Scaling factor (1.0 = 5K pairs).

    Returns:
        Number of examples written.
    """
    output_dir = Path(output_dir)
    count = max(1, int(5000 * scale))
    examples = generate_dpo_pairs(count=count)

    path = output_dir / "phase25_dpo_pairs.jsonl"
    with JSONLWriter(path) as writer:
        for ex in examples:
            writer.write(ex)

    return len(examples)
