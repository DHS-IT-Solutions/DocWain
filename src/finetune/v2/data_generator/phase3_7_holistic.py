"""Phase 3.7 Holistic Reasoning data generator for DocWain V2+ SFT.

Generates 8K holistic reasoning examples (at scale=1.0) across 4 modes:
  - intent_decomposition   (2000)
  - evidence_synthesis      (2500)
  - depth_calibration       (1500)
  - domain_reasoning        (2000)
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
# Constants
# ---------------------------------------------------------------------------

REASONING_MODES = [
    "intent_decomposition",
    "evidence_synthesis",
    "depth_calibration",
    "domain_reasoning",
]

_MODE_COUNTS = {
    "intent_decomposition": 2000,
    "evidence_synthesis": 2500,
    "depth_calibration": 1500,
    "domain_reasoning": 2000,
}

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


def _rand_amount(rng: random.Random, lo: float = 100.0, hi: float = 99999.0) -> str:
    return f"{rng.uniform(lo, hi):,.2f}"


def _rand_pct(rng: random.Random) -> str:
    return f"{rng.uniform(1, 45):.1f}"


# ---------------------------------------------------------------------------
# Intent Decomposition templates — vague queries -> structured think blocks
# ---------------------------------------------------------------------------

_INTENT_TEMPLATES: List[Dict[str, str]] = [
    {
        "query": "Tell me about the {company} deal.",
        "reasoning": (
            "Intent Decomposition:\n"
            "1. Ambiguity analysis: 'the deal' is unspecified — could mean acquisition, "
            "partnership, sales contract, or investment.\n"
            "2. Likely intent: User wants a summary of a recent or notable transaction "
            "involving {company}.\n"
            "3. Sub-questions to resolve:\n"
            "   a) What type of deal? (need to check available documents)\n"
            "   b) Key terms? (value, parties, timeline)\n"
            "   c) Current status? (signed, pending, closed)\n"
            "4. Information needed: {domain} documents related to {company} transactions.\n"
            "5. Response plan: Provide overview of available deal information, ask for "
            "clarification if multiple deals exist.\n"
            "Confidence: Low — query is highly ambiguous without document context."
        ),
        "answer": (
            "I found references to {company} in the {domain} documents. To give you "
            "the most relevant information, could you clarify which deal you're referring to?\n\n"
            "Available {company} transactions in the system:\n"
            "1. {doc_type} — {currency}{amount} ({month} {year})\n"
            "2. Service agreement with {dept} department\n\n"
            "Please specify which one you'd like me to analyze, or I can provide "
            "a summary of all {company} transactions."
        ),
    },
    {
        "query": "What's going on with {dept}?",
        "reasoning": (
            "Intent Decomposition:\n"
            "1. Ambiguity analysis: 'what's going on' is extremely open-ended — could mean "
            "performance, headcount, issues, projects, budget, or recent changes.\n"
            "2. Likely intent: User wants a status overview of {dept}.\n"
            "3. Sub-questions to resolve:\n"
            "   a) Performance metrics? (KPIs, targets vs actual)\n"
            "   b) Staffing? (headcount, open roles, attrition)\n"
            "   c) Recent events? (reorgs, project launches, issues)\n"
            "   d) Budget status? (utilization, overruns)\n"
            "4. Information needed: Cross-reference {dept} data across HR, finance, "
            "and operations documents.\n"
            "5. Response plan: Provide a high-level dashboard-style overview.\n"
            "Confidence: Low — broad query requires multi-source synthesis."
        ),
        "answer": (
            "Here's a high-level overview of {dept} based on available documents:\n\n"
            "**Staffing:** Current headcount data available in HR records\n"
            "**Budget:** {quarter} {year} allocation of {currency}{amount}\n"
            "**Recent Activity:** References found in {domain} {doc_type} documents\n\n"
            "Would you like me to dive deeper into any specific aspect of {dept}? "
            "I can provide detailed analysis on performance, staffing, budget, or "
            "recent changes."
        ),
    },
    {
        "query": "Can you check the numbers for {person}?",
        "reasoning": (
            "Intent Decomposition:\n"
            "1. Ambiguity analysis: 'the numbers' could mean financial figures, "
            "performance metrics, expense reports, sales targets, or compensation.\n"
            "2. Likely intent: User wants quantitative data associated with {person}.\n"
            "3. Sub-questions to resolve:\n"
            "   a) Which numbers? (expenses, sales, KPIs, salary)\n"
            "   b) Time period? (current quarter, YTD, historical)\n"
            "   c) Context? ({person}'s role determines relevant metrics)\n"
            "4. Information needed: {person}'s profile and associated {domain} documents.\n"
            "5. Response plan: Surface available numeric data for {person}, ask for specifics.\n"
            "Confidence: Low — 'numbers' is too vague to provide a precise answer."
        ),
        "answer": (
            "I found several data points associated with {person} in the {domain} records:\n\n"
            "- {doc_type} reference: {currency}{amount}\n"
            "- Department: {dept}\n\n"
            "Could you specify which numbers you're looking for? For example:\n"
            "- Expense reports\n"
            "- Performance metrics\n"
            "- Financial transactions\n"
            "- Project allocations"
        ),
    },
]

# ---------------------------------------------------------------------------
# Evidence Synthesis templates — multi-chunk contexts
# ---------------------------------------------------------------------------

_EVIDENCE_TEMPLATES: List[Dict[str, str]] = [
    {
        "query": (
            "Based on all available documents, what is {company}'s financial position?"
        ),
        "reasoning": (
            "Evidence Synthesis (Triage -> Connect -> Resolve -> Narrate):\n\n"
            "TRIAGE — Relevant chunks identified:\n"
            "  Chunk 1: {domain} {doc_type} showing revenue of {currency}{amount}\n"
            "  Chunk 2: Expense report totaling {currency}{amount2}\n"
            "  Chunk 3: {dept} budget allocation of {currency}{amount3}\n"
            "  Chunk 4: Audit report noting {pct}% cost reduction target\n"
            "  Chunk 5: Vendor contract at {currency}{amount} annually\n\n"
            "CONNECT — Linking evidence:\n"
            "  Revenue ({currency}{amount}) vs Expenses ({currency}{amount2}) provides margin picture.\n"
            "  {dept} budget ({currency}{amount3}) is a component of total expenses.\n"
            "  Audit target of {pct}% reduction applies to operational costs.\n\n"
            "RESOLVE — Conflicts/gaps:\n"
            "  Revenue and vendor contract show same amount — may be coincidental or circular.\n"
            "  No balance sheet data available (assets, liabilities unknown).\n\n"
            "NARRATE — Synthesis:\n"
            "  {company} has documented revenue and expense streams but incomplete financial picture.\n"
            "Confidence: Medium — multiple sources but significant gaps remain."
        ),
        "answer": (
            "**{company} Financial Position Summary:**\n\n"
            "Based on {qty} document sources across {domain}:\n\n"
            "- **Revenue:** {currency}{amount} (from {doc_type})\n"
            "- **Total Expenses:** {currency}{amount2} (expense reports)\n"
            "- **{dept} Budget:** {currency}{amount3}\n"
            "- **Cost Reduction Target:** {pct}% (per audit report)\n\n"
            "**Gaps:** No balance sheet data, no cash flow statement, no year-over-year comparison.\n\n"
            "**Confidence:** Medium — based on {qty} partial sources; a complete financial "
            "assessment requires balance sheet and cash flow data."
        ),
    },
    {
        "query": (
            "Synthesize what we know about the {product} project from all documents."
        ),
        "reasoning": (
            "Evidence Synthesis (Triage -> Connect -> Resolve -> Narrate):\n\n"
            "TRIAGE — Relevant chunks:\n"
            "  Chunk 1: Project proposal for {product} — {currency}{amount} budget, {dept} lead\n"
            "  Chunk 2: {company} vendor quote for {product} components — {currency}{amount2}\n"
            "  Chunk 3: {person}'s status update — 65% complete\n"
            "  Chunk 4: Risk register — 3 open risks flagged\n"
            "  Chunk 5: {quarter} {year} board minutes — {product} discussed\n"
            "  Chunk 6: Procurement {doc_type} for {product} materials\n\n"
            "CONNECT:\n"
            "  Budget ({currency}{amount}) vs vendor quote ({currency}{amount2}) shows cost alignment.\n"
            "  65% complete with {quarter} board discussion suggests on-track.\n"
            "  {dept} is the accountable department under {person}.\n\n"
            "RESOLVE:\n"
            "  3 open risks need characterization — severity unknown from available docs.\n"
            "  No timeline/deadline found in available chunks.\n\n"
            "NARRATE:\n"
            "  {product} is a mid-flight project with reasonable budget alignment but open risks.\n"
            "Confidence: Medium — good coverage but risk details and timeline missing."
        ),
        "answer": (
            "**{product} Project Synthesis:**\n\n"
            "Compiled from 6 document sources:\n\n"
            "- **Budget:** {currency}{amount} (approved in project proposal)\n"
            "- **Vendor Cost:** {currency}{amount2} ({company})\n"
            "- **Progress:** 65% complete (per {person}'s update)\n"
            "- **Lead Department:** {dept}\n"
            "- **Board Visibility:** Discussed in {quarter} {year} board minutes\n"
            "- **Open Risks:** 3 items flagged in risk register\n\n"
            "**Gaps:** No project timeline or deadline found. Risk severity levels not documented.\n\n"
            "**Confidence:** Medium — reasonable source coverage but critical timeline data missing."
        ),
    },
]

# ---------------------------------------------------------------------------
# Depth Calibration templates — matched query-complexity -> response-depth
# ---------------------------------------------------------------------------

_DEPTH_TEMPLATES: List[Dict[str, str]] = [
    # Lookup -> brief
    {
        "query": "What is {company}'s address?",
        "reasoning": (
            "Depth Calibration:\n"
            "Query type: Simple lookup (factual, single-value).\n"
            "Required depth: Brief — direct answer with source citation.\n"
            "No analysis needed; this is a retrieval task.\n"
            "Confidence: High if found in documents."
        ),
        "answer": (
            "Based on the {domain} {doc_type} on file, {company}'s registered address "
            "is listed in the document header. Please refer to the {doc_type} for the "
            "exact address.\n\n"
            "**Confidence:** High — direct document lookup."
        ),
    },
    # Summarize -> moderate
    {
        "query": "Summarize the key terms of the {company} {doc_type}.",
        "reasoning": (
            "Depth Calibration:\n"
            "Query type: Summarization (multi-field extraction, moderate complexity).\n"
            "Required depth: Moderate — extract and organize key terms, no deep analysis.\n"
            "Fields to cover: parties, value, term, payment, governing law.\n"
            "Confidence: High if document is available."
        ),
        "answer": (
            "**Key Terms — {company} {doc_type}:**\n\n"
            "- **Parties:** {company} and counterparty\n"
            "- **Value:** {currency}{amount}\n"
            "- **Department:** {dept}\n"
            "- **Period:** {quarter} {year}\n"
            "- **Contact:** {person}\n\n"
            "**Confidence:** High — terms extracted directly from {doc_type}."
        ),
    },
    # Analyze -> thorough
    {
        "query": (
            "Analyze the risk implications of {company}'s {domain} strategy based "
            "on all available documents and provide recommendations."
        ),
        "reasoning": (
            "Depth Calibration:\n"
            "Query type: Deep analysis (multi-source, requires synthesis and recommendations).\n"
            "Required depth: Thorough — full evidence review, pattern identification, "
            "risk assessment, and actionable recommendations.\n"
            "Approach: Cross-reference {domain} documents, identify strategic risks, "
            "assess probability and impact, formulate mitigations.\n"
            "Confidence: Medium — analysis quality depends on document coverage."
        ),
        "answer": (
            "**{company} {domain} Strategy Risk Analysis:**\n\n"
            "**Evidence Base:** {domain} {doc_type} and related documents from {dept}\n\n"
            "**Identified Risks:**\n"
            "1. Concentration risk — {currency}{amount} committed to single initiative\n"
            "2. Execution risk — {dept} capacity to deliver within {quarter} {year} timeline\n"
            "3. Market risk — {domain} sector volatility may impact assumptions\n\n"
            "**Recommendations:**\n"
            "1. Diversify commitment across multiple initiatives\n"
            "2. Establish quarterly review gates with {person} as accountable executive\n"
            "3. Build contingency buffer of {pct}% above current budget\n\n"
            "**Confidence:** Medium — comprehensive analysis but limited to available documents. "
            "External market data would strengthen the assessment."
        ),
    },
]

# ---------------------------------------------------------------------------
# Domain Reasoning templates — legal/financial/HR/medical frameworks
# ---------------------------------------------------------------------------

_DOMAIN_TEMPLATES: List[Dict[str, str]] = [
    # Legal reasoning
    {
        "query": (
            "Review clause 7.3 of the {company} {doc_type} for enforceability issues:\n\n"
            "'Either party may terminate this agreement with 30 days written notice. "
            "Notwithstanding the foregoing, {company} may terminate immediately for cause.'"
        ),
        "reasoning": (
            "Domain Reasoning — Legal Framework:\n\n"
            "1. Clause structure: Two-part termination provision.\n"
            "2. Mutual right: 30-day written notice — standard and enforceable.\n"
            "3. Unilateral right: {company} can terminate immediately 'for cause'.\n"
            "4. Legal analysis:\n"
            "   - 'For cause' is not defined in this clause — enforceability risk.\n"
            "   - Asymmetric termination rights may be challenged as unconscionable.\n"
            "   - Notice requirement (written) is properly specified.\n"
            "5. Jurisdiction considerations: Enforceability varies by governing law.\n"
            "Confidence: High on identification of issues; enforceability depends on jurisdiction."
        ),
        "answer": (
            "**Clause 7.3 Enforceability Review:**\n\n"
            "**Issues Identified:**\n"
            "1. **Undefined 'for cause'** — The clause grants {company} immediate termination "
            "'for cause' but does not define what constitutes cause. This creates ambiguity "
            "and may be unenforceable or subject to judicial interpretation.\n"
            "2. **Asymmetric rights** — Only {company} has the immediate termination right. "
            "The counterparty may argue this is one-sided.\n\n"
            "**Recommendations:**\n"
            "- Define 'cause' explicitly (e.g., material breach, insolvency, regulatory action)\n"
            "- Consider granting reciprocal immediate termination rights\n"
            "- Add cure period before 'for cause' termination takes effect\n\n"
            "**Confidence:** High — legal analysis based on standard contract law principles."
        ),
    },
    # Financial reasoning
    {
        "query": (
            "Evaluate {company}'s working capital position:\n\n"
            "Current Assets: {currency}{amount}\n"
            "Current Liabilities: {currency}{amount2}\n"
            "Inventory: {currency}{amount3}\n"
            "Accounts Receivable: {currency}{amount}"
        ),
        "reasoning": (
            "Domain Reasoning — Financial Framework:\n\n"
            "1. Working Capital = Current Assets - Current Liabilities = "
            "{currency}{amount} - {currency}{amount2}.\n"
            "2. Current Ratio = Current Assets / Current Liabilities.\n"
            "3. Quick Ratio = (Current Assets - Inventory) / Current Liabilities = "
            "({currency}{amount} - {currency}{amount3}) / {currency}{amount2}.\n"
            "4. AR concentration: AR ({currency}{amount}) equals total current assets, "
            "suggesting assets are heavily receivables-based.\n"
            "5. Liquidity assessment: Need to check if working capital is positive and "
            "ratios meet industry benchmarks (typically >1.5 current, >1.0 quick).\n"
            "Confidence: High on calculation methodology; benchmarking requires industry data."
        ),
        "answer": (
            "**{company} Working Capital Analysis:**\n\n"
            "- **Working Capital:** {currency}{amount} - {currency}{amount2}\n"
            "- **Current Ratio:** {currency}{amount} / {currency}{amount2}\n"
            "- **Quick Ratio:** ({currency}{amount} - {currency}{amount3}) / {currency}{amount2}\n\n"
            "**Key Observation:** Accounts Receivable ({currency}{amount}) equals total "
            "current assets, indicating heavy reliance on receivables for liquidity. "
            "This creates collection risk.\n\n"
            "**Recommendation:** Diversify current asset composition and accelerate "
            "AR collection to improve liquidity resilience.\n\n"
            "**Confidence:** High — standard financial ratio analysis."
        ),
    },
    # HR reasoning
    {
        "query": (
            "Assess this {dept} attrition data for {company}:\n\n"
            "- Voluntary departures: {qty} in {year}\n"
            "- Average tenure of leavers: 1.8 years\n"
            "- Exit interview theme: limited growth opportunities\n"
            "- Industry average attrition: 12%"
        ),
        "reasoning": (
            "Domain Reasoning — HR Framework:\n\n"
            "1. Attrition count: {qty} voluntary departures in {year}.\n"
            "2. Tenure analysis: 1.8 years average — indicates early-career attrition, "
            "not retirement or long-service departure.\n"
            "3. Root cause signal: 'Limited growth opportunities' from exit interviews "
            "points to career development gap.\n"
            "4. Benchmarking: Need {dept} headcount to compute attrition rate and "
            "compare to 12% industry average.\n"
            "5. Cost impact: Early-tenure departures typically cost 50-200% of annual salary "
            "in replacement costs.\n"
            "Confidence: Medium — qualitative signal is strong but attrition rate "
            "cannot be computed without headcount denominator."
        ),
        "answer": (
            "**{dept} Attrition Assessment for {company} ({year}):**\n\n"
            "- **Volume:** {qty} voluntary departures\n"
            "- **Profile:** Early-career (1.8yr avg tenure) — not natural attrition\n"
            "- **Root Cause:** Career development gap (exit interview consensus)\n"
            "- **Industry Benchmark:** 12% average (rate comparison pending headcount data)\n\n"
            "**Recommendations:**\n"
            "1. Implement structured career pathing for {dept}\n"
            "2. Introduce mentorship program targeting 0-2 year tenure employees\n"
            "3. Conduct stay interviews with current staff to identify at-risk individuals\n\n"
            "**Confidence:** Medium — strong qualitative signal; quantitative rate requires headcount."
        ),
    },
    # Medical/clinical reasoning
    {
        "query": (
            "Review this {domain} clinical document summary for completeness:\n\n"
            "Patient ID: P-{qty}\n"
            "Diagnosis: Type 2 Diabetes\n"
            "Treatment: Metformin 500mg BID\n"
            "Last A1C: 7.2%\n"
            "Follow-up: Not specified"
        ),
        "reasoning": (
            "Domain Reasoning — Medical/Clinical Framework:\n\n"
            "1. Document type: Clinical summary for patient P-{qty}.\n"
            "2. Diagnosis: Type 2 Diabetes — established chronic condition.\n"
            "3. Treatment: Metformin 500mg BID — standard first-line therapy.\n"
            "4. Lab result: A1C 7.2% — slightly above 7.0% target per ADA guidelines.\n"
            "5. Completeness check:\n"
            "   - Present: diagnosis, medication, key lab value\n"
            "   - Missing: follow-up date (critical gap), BMI, blood pressure, "
            "     renal function (eGFR — required for Metformin), medication list, "
            "     comorbidities, lifestyle counseling notes\n"
            "Confidence: High — completeness gaps are objectively identifiable."
        ),
        "answer": (
            "**Clinical Document Completeness Review (P-{qty}):**\n\n"
            "**Present:**\n"
            "- Diagnosis: Type 2 Diabetes\n"
            "- Medication: Metformin 500mg BID\n"
            "- Lab: A1C 7.2% (slightly above 7.0% ADA target)\n\n"
            "**Missing (Critical):**\n"
            "- Follow-up appointment date\n"
            "- Renal function (eGFR) — required for Metformin safety\n"
            "- Complete medication list\n\n"
            "**Missing (Important):**\n"
            "- BMI, blood pressure\n"
            "- Comorbidity documentation\n"
            "- Lifestyle counseling notes\n\n"
            "**Confidence:** High — gaps identified against standard clinical documentation requirements."
        ),
    },
]

# Map mode -> template list
_MODE_TEMPLATE_MAP = {
    "intent_decomposition": _INTENT_TEMPLATES,
    "evidence_synthesis": _EVIDENCE_TEMPLATES,
    "depth_calibration": _DEPTH_TEMPLATES,
    "domain_reasoning": _DOMAIN_TEMPLATES,
}


# ---------------------------------------------------------------------------
# Template expansion
# ---------------------------------------------------------------------------

def _expand_holistic_templates(
    templates: List[Dict[str, str]],
    target_count: int,
    rng: random.Random,
) -> List[Dict[str, str]]:
    """Expand holistic templates with random variation."""
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
        qty = str(rng.randint(5, 200))
        year = str(rng.randint(2020, 2026))
        quarter = rng.choice(["Q1", "Q2", "Q3", "Q4"])
        month = rng.choice([
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ])
        pct = _rand_pct(rng)

        subs = {
            "domain": domain, "doc_type": doc_type, "company": company,
            "person": person, "dept": dept, "product": product,
            "currency": currency, "amount": amount, "amount2": amount2,
            "amount3": amount3, "qty": qty, "year": year,
            "quarter": quarter, "month": month, "pct": pct,
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_holistic_examples(count: int = 8000, *, seed: int = 70) -> List[Dict[str, str]]:
    """Generate holistic reasoning SFT examples across 4 modes.

    Distribution matches _MODE_COUNTS, scaled proportionally to ``count``.

    Args:
        count: Total number of examples to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of SFT examples with ``text`` key.
    """
    rng = random.Random(seed)
    total_base = sum(_MODE_COUNTS.values())

    # Compute per-mode counts, ensuring they sum to exactly `count`
    raw_counts = {mode: max(1, int(count * _MODE_COUNTS[mode] / total_base))
                  for mode in REASONING_MODES}
    allocated = sum(raw_counts.values())
    remainder = count - allocated
    sorted_modes = sorted(REASONING_MODES, key=lambda m: _MODE_COUNTS[m], reverse=True)
    i = 0
    while remainder > 0:
        raw_counts[sorted_modes[i % len(sorted_modes)]] += 1
        remainder -= 1
        i += 1

    results: List[Dict[str, str]] = []
    for mode in REASONING_MODES:
        templates = _MODE_TEMPLATE_MAP[mode]
        results.extend(_expand_holistic_templates(templates, raw_counts[mode], rng))

    rng.shuffle(results)
    return results[:count]


def generate_phase37_data(output_dir: Path, scale: float = 1.0) -> int:
    """Generate Phase 3.7 holistic reasoning training data and write to JSONL.

    Args:
        output_dir: Directory to write the JSONL file into.
        scale: Scaling factor (1.0 = 8K examples).

    Returns:
        Number of examples written.
    """
    output_dir = Path(output_dir)
    count = max(1, int(8000 * scale))
    examples = generate_holistic_examples(count=count)

    path = output_dir / "phase37_holistic.jsonl"
    with JSONLWriter(path) as writer:
        for ex in examples:
            writer.write(ex)

    return len(examples)
