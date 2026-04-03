"""Track 4 — Context & Reasoning data generator for DocWain V2+ SFT/DPO.

Generates 2000 training examples across eight reasoning categories:
  - Multi-document contradiction resolution (300)
  - Temporal reasoning                      (300)
  - Implicit intent decomposition           (300)
  - Causal chain reasoning                  (250)
  - Quantitative reasoning                  (250)
  - Counterfactual analysis                 (150)
  - Uncertainty calibration                 (250)
  - Refusal with explanation                (200)

Produces both SFT examples (with ``<think>`` reasoning blocks) and DPO
preference pairs where the *chosen* response demonstrates deep multi-step
reasoning while the *rejected* response is shallow summarization.
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
_CLAUSES = [
    "Section 4.2 (Indemnification)", "Section 7.1 (Termination)",
    "Clause 3.3 (Liability Cap)", "Clause 5.6 (Force Majeure)",
    "Article 2.4 (Payment Terms)", "Article 9.1 (Confidentiality)",
    "Section 6.3 (Non-Compete)", "Clause 8.2 (Warranty)",
    "Section 11.4 (Dispute Resolution)", "Clause 1.7 (Definitions)",
]
_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_QUARTERS = ["Q1", "Q2", "Q3", "Q4"]


def _pick(lst: list, rng: random.Random) -> Any:
    return rng.choice(lst)


def _rand_amount(rng: random.Random, lo: float = 100.0, hi: float = 99999.0) -> str:
    return f"{rng.uniform(lo, hi):,.2f}"


def _rand_pct(rng: random.Random, lo: float = 1.0, hi: float = 45.0) -> str:
    return f"{rng.uniform(lo, hi):.1f}"


def _subs(rng: random.Random) -> Dict[str, str]:
    """Build a standard substitution dict."""
    return {
        "domain": _pick(DOMAINS, rng),
        "doc_type": _pick(DOC_TYPES, rng),
        "company": _pick(_COMPANY_NAMES, rng),
        "company2": _pick(_COMPANY_NAMES, rng),
        "person": _pick(_PERSON_NAMES, rng),
        "person2": _pick(_PERSON_NAMES, rng),
        "dept": _pick(_DEPARTMENTS, rng),
        "dept2": _pick(_DEPARTMENTS, rng),
        "product": _pick(_PRODUCT_NAMES, rng),
        "currency": _pick(_CURRENCIES, rng),
        "amount": _rand_amount(rng),
        "amount2": _rand_amount(rng),
        "amount3": _rand_amount(rng),
        "year": str(rng.randint(2020, 2026)),
        "year2": str(rng.randint(2020, 2026)),
        "quarter": _pick(_QUARTERS, rng),
        "month": _pick(_MONTHS, rng),
        "pct": _rand_pct(rng),
        "pct2": _rand_pct(rng),
        "clause": _pick(_CLAUSES, rng),
        "clause2": _pick(_CLAUSES, rng),
        "qty": str(rng.randint(1, 200)),
    }


def _expand(
    templates: List[Dict[str, str]],
    count: int,
    rng: random.Random,
    *,
    mode: str = "sft",
) -> List[Dict[str, str]]:
    """Expand templates to reach *count* examples.

    mode='sft' uses format_sft_example; mode='dpo' uses format_dpo_example.
    """
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
# 1. MULTI-DOCUMENT CONTRADICTION RESOLUTION (300 SFT + DPO)
# ===================================================================

_CONTRADICTION_SFT: List[Dict[str, str]] = [
    {
        "query": (
            "Document A ({domain} {doc_type} from {company}, dated {month} {year}):\n"
            "\"{person} holds the title of Chief Financial Officer.\"\n\n"
            "Document B ({domain} {doc_type} from {company}, dated {month} {year2}):\n"
            "\"{person2} is listed as the Chief Financial Officer.\"\n\n"
            "Who is the CFO of {company}?"
        ),
        "reasoning": (
            "Step 1: Identify the contradiction — Document A names {person} as CFO, "
            "Document B names {person2} as CFO.\n"
            "Step 2: Assess document recency — Document A is from {year}, Document B from {year2}.\n"
            "Step 3: The more recent document is more likely to reflect the current state.\n"
            "Step 4: Consider that CFO roles change over time; a later filing supersedes an earlier one.\n"
            "Step 5: Check document types — both are {domain} {doc_type}s, so equal authority.\n"
            "Confidence: Medium — temporal ordering is the strongest signal, but without "
            "explicit confirmation of a role change, partial uncertainty remains."
        ),
        "answer": (
            "There is a contradiction between the two documents. Document A ({year}) lists "
            "{person} as CFO, while Document B ({year2}) lists {person2}. Based on document "
            "recency, the more recent filing is the stronger indicator. However, I recommend "
            "verifying against the latest corporate filings to confirm the current CFO."
        ),
    },
    {
        "query": (
            "Contract v1.0 ({company}, {year}):\n"
            "\"The liability cap shall not exceed {currency}{amount}.\"\n\n"
            "Amendment v1.2 ({company}, {year2}):\n"
            "\"The liability cap is revised to {currency}{amount2}.\"\n\n"
            "What is the current liability cap for the {company} contract?"
        ),
        "reasoning": (
            "Step 1: Contradiction — v1.0 says {currency}{amount}, Amendment v1.2 says {currency}{amount2}.\n"
            "Step 2: Amendments supersede original contract clauses by legal convention.\n"
            "Step 3: v1.2 is a later version number and dated {year2} vs {year}.\n"
            "Step 4: Unless the amendment was voided or further amended, v1.2 controls.\n"
            "Confidence: High — amendment clearly supersedes the original."
        ),
        "answer": (
            "The current liability cap is {currency}{amount2}, as established by Amendment v1.2 ({year2}). "
            "This supersedes the original {currency}{amount} cap in Contract v1.0 ({year})."
        ),
    },
    {
        "query": (
            "Source 1 — {domain} audit report ({year}):\n"
            "\"{dept} headcount: {qty} full-time employees.\"\n\n"
            "Source 2 — {domain} financial statement ({year}):\n"
            "\"{dept} headcount: {amount} total staff (including contractors).\"\n\n"
            "How many people work in {dept}?"
        ),
        "reasoning": (
            "Step 1: Source 1 says {qty} FTEs; Source 2 says {amount} total staff.\n"
            "Step 2: These are not necessarily contradictory — the difference may be "
            "contractors vs full-time employees.\n"
            "Step 3: The audit report specifies 'full-time employees', the financial statement "
            "says 'total staff (including contractors)'.\n"
            "Step 4: Both can be simultaneously true if the difference represents contractors.\n"
            "Confidence: High — apparent contradiction is resolved by definitional difference."
        ),
        "answer": (
            "Both figures can be correct: {dept} has {qty} full-time employees (per the audit report) "
            "and a total staff count of {amount} when including contractors (per the financial statement). "
            "The discrepancy is due to different counting methodologies, not a true contradiction."
        ),
    },
    {
        "query": (
            "Policy Document A ({company}, {year}):\n"
            "\"All purchases above {currency}{amount} require three competitive bids.\"\n\n"
            "Policy Document B ({company}, {year2}):\n"
            "\"Emergency purchases are exempt from the competitive bidding requirement "
            "regardless of amount.\"\n\n"
            "Does a {currency}{amount2} emergency purchase require competitive bids?"
        ),
        "reasoning": (
            "Step 1: Policy A requires three bids above {currency}{amount}.\n"
            "Step 2: Policy B exempts emergency purchases from bidding.\n"
            "Step 3: These are not a contradiction but a general rule + exception.\n"
            "Step 4: {currency}{amount2} exceeds the {currency}{amount} threshold, "
            "triggering the general rule.\n"
            "Step 5: However, the emergency exemption in Policy B takes precedence as a "
            "specific exception to the general rule.\n"
            "Confidence: High — specific exception overrides general rule."
        ),
        "answer": (
            "No, a {currency}{amount2} emergency purchase does not require competitive bids. "
            "While Policy A requires three bids for amounts above {currency}{amount}, "
            "Policy B explicitly exempts emergency purchases from this requirement."
        ),
    },
    {
        "query": (
            "Report from {dept} ({year}):\n"
            "\"Project Alpha was completed under budget at {currency}{amount}.\"\n\n"
            "Report from {dept2} ({year}):\n"
            "\"Project Alpha final expenditure was {currency}{amount2}, exceeding the "
            "{currency}{amount3} budget.\"\n\n"
            "Was Project Alpha completed under or over budget?"
        ),
        "reasoning": (
            "Step 1: {dept} claims under-budget at {currency}{amount}.\n"
            "Step 2: {dept2} claims over-budget at {currency}{amount2} vs {currency}{amount3} budget.\n"
            "Step 3: The amounts differ — possible different scopes or accounting methods.\n"
            "Step 4: {dept} may be reporting only their portion; {dept2} may include "
            "cross-departmental costs.\n"
            "Step 5: Need to determine which report covers the full project scope.\n"
            "Confidence: Low — genuine contradiction requiring scope clarification."
        ),
        "answer": (
            "The two departments report conflicting budget outcomes for Project Alpha. {dept} "
            "reports {currency}{amount} (under budget), while {dept2} reports {currency}{amount2} "
            "against a {currency}{amount3} budget (over budget). This discrepancy likely stems from "
            "different scope definitions. I recommend consulting the project charter to determine "
            "which report covers the authoritative full project scope."
        ),
    },
    {
        "query": (
            "Document 1 — {domain} {doc_type} from {company} ({month} {year}):\n"
            "\"Payment terms: Net 30 days from invoice date.\"\n\n"
            "Document 2 — {domain} {doc_type} from {company} ({month} {year2}):\n"
            "\"Payment terms: Net 60 days from delivery date.\"\n\n"
            "What are the payment terms for {company}?"
        ),
        "reasoning": (
            "Step 1: Document 1 states Net 30 from invoice date ({year}).\n"
            "Step 2: Document 2 states Net 60 from delivery date ({year2}).\n"
            "Step 3: Both the duration (30 vs 60 days) and trigger (invoice vs delivery) differ.\n"
            "Step 4: Check recency — whichever document is more recent likely reflects current terms.\n"
            "Step 5: If these are different contracts or different document types, they may both "
            "be valid in their respective contexts.\n"
            "Confidence: Medium — depends on whether documents govern the same relationship."
        ),
        "answer": (
            "Two conflicting payment terms exist: Net 30 from invoice date (Document 1, {year}) "
            "and Net 60 from delivery date (Document 2, {year2}). If these documents govern the same "
            "commercial relationship, the more recent one typically controls. Verify which document "
            "is the operative agreement."
        ),
    },
]

_CONTRADICTION_DPO: List[Dict[str, str]] = [
    {
        "query": (
            "Document A ({domain} report, {year}):\n"
            "\"{company} reported annual revenue of {currency}{amount}.\"\n\n"
            "Document B ({domain} filing, {year2}):\n"
            "\"{company} reported annual revenue of {currency}{amount2}.\"\n\n"
            "What was {company}'s annual revenue?"
        ),
        "chosen_reasoning": (
            "Step 1: Identify contradiction — two revenue figures: {currency}{amount} ({year}) "
            "and {currency}{amount2} ({year2}).\n"
            "Step 2: Assess document types — a 'report' vs a 'filing'. Regulatory filings are "
            "typically audited and carry legal liability, making them more authoritative.\n"
            "Step 3: Check dates — the figures may refer to different fiscal years ({year} vs {year2}).\n"
            "Step 4: If same fiscal year, prefer the regulatory filing. If different years, both "
            "can be correct for their respective periods.\n"
            "Confidence: Medium — depends on whether the same fiscal year is referenced."
        ),
        "chosen_answer": (
            "The two documents report different revenue figures for {company}: {currency}{amount} "
            "(from a {year} report) and {currency}{amount2} (from a {year2} filing). If these cover "
            "different fiscal years, both may be accurate. If they cover the same period, the "
            "regulatory filing ({currency}{amount2}) is more authoritative as it carries legal "
            "liability for accuracy."
        ),
        "rejected_reasoning": "The documents show different revenue numbers.",
        "rejected_answer": (
            "The revenue was {currency}{amount} according to Document A and {currency}{amount2} "
            "according to Document B."
        ),
    },
    {
        "query": (
            "Lease v1 ({company}, {year}): \"Monthly rent: {currency}{amount}.\"\n"
            "Lease Amendment ({company}, {year2}): \"Monthly rent: {currency}{amount2}.\"\n\n"
            "What is the monthly rent?"
        ),
        "chosen_reasoning": (
            "Step 1: Original lease ({year}) sets rent at {currency}{amount}.\n"
            "Step 2: Amendment ({year2}) changes rent to {currency}{amount2}.\n"
            "Step 3: Amendments are legally binding modifications that supersede original terms.\n"
            "Step 4: Check if the amendment has an effective date or conditions.\n"
            "Step 5: Absent conditions, the amendment controls from its date.\n"
            "Confidence: High — amendments supersede originals by legal convention."
        ),
        "chosen_answer": (
            "The current monthly rent is {currency}{amount2}, per the Lease Amendment ({year2}). "
            "This supersedes the original {currency}{amount} in Lease v1 ({year}). Note: verify "
            "there are no subsequent amendments."
        ),
        "rejected_reasoning": "One document says one rent, the other says another.",
        "rejected_answer": "The rent is either {currency}{amount} or {currency}{amount2}.",
    },
]


# ===================================================================
# 2. TEMPORAL REASONING (300 SFT + DPO)
# ===================================================================

_TEMPORAL_SFT: List[Dict[str, str]] = [
    {
        "query": (
            "Given the following {domain} document versions for {company}:\n"
            "- v1.0 ({month} {year}): {clause} states the penalty is {pct}%.\n"
            "- v2.0 ({month} {year2}): {clause} amended to {pct2}%.\n\n"
            "As of the latest amendment, what is the penalty percentage in {clause}?"
        ),
        "reasoning": (
            "Step 1: Two versions exist — v1.0 ({year}) with {pct}% and v2.0 ({year2}) with {pct2}%.\n"
            "Step 2: v2.0 is the latest version, so {pct2}% is the current value.\n"
            "Step 3: Confirm the query asks for 'as of the latest amendment' — that is v2.0.\n"
            "Confidence: High — version ordering is clear."
        ),
        "answer": (
            "As of the latest amendment (v2.0, {year2}), the penalty in {clause} is {pct2}%. "
            "This replaces the original {pct}% from v1.0 ({year})."
        ),
    },
    {
        "query": (
            "{company}'s {domain} records show:\n"
            "- {month} {year}: {person} hired as {dept} Manager.\n"
            "- {month} {year2}: {person} promoted to VP of {dept}.\n"
            "- {month} {year2}: {person2} hired as {dept} Manager (replacement).\n\n"
            "Who is the current {dept} Manager?"
        ),
        "reasoning": (
            "Step 1: Timeline — {person} started as Manager in {year}, promoted to VP in {year2}.\n"
            "Step 2: {person2} replaced {person} as Manager in {year2}.\n"
            "Step 3: Current Manager = {person2}; {person} is now VP.\n"
            "Confidence: High — clear succession timeline."
        ),
        "answer": (
            "The current {dept} Manager is {person2}, who was hired as a replacement in {year2} "
            "after {person} was promoted to VP of {dept}."
        ),
    },
    {
        "query": (
            "Contract timeline for {company}:\n"
            "- Original contract signed {year}: term = 3 years.\n"
            "- Extension signed {year2}: term extended by 2 years.\n"
            "- Amendment signed {year2}: added auto-renewal clause for 1-year periods.\n\n"
            "What is the current contract status and when does it expire?"
        ),
        "reasoning": (
            "Step 1: Original term: {year} + 3 years.\n"
            "Step 2: Extension: + 2 years beyond original expiry.\n"
            "Step 3: Auto-renewal added in {year2} — 1-year periods.\n"
            "Step 4: Need to compute the base expiry, then consider auto-renewal.\n"
            "Step 5: Auto-renewal means the contract continues unless terminated.\n"
            "Confidence: Medium — depends on whether auto-renewal has been triggered."
        ),
        "answer": (
            "The original 3-year term from {year} was extended by 2 years, and an auto-renewal "
            "clause (1-year periods) was added in {year2}. The contract will continue to auto-renew "
            "annually unless either party provides notice of termination per the amendment's terms."
        ),
    },
    {
        "query": (
            "Policy revision history for {company}:\n"
            "- Rev A ({year}): Expense reports due within 30 days.\n"
            "- Rev B ({year2}): Changed to 15 days for amounts over {currency}{amount}.\n"
            "- Rev C ({year2}): All expense reports due within 15 days regardless of amount.\n\n"
            "As of the latest revision, what is the expense report deadline?"
        ),
        "reasoning": (
            "Step 1: Rev A — 30 days for all.\n"
            "Step 2: Rev B — 15 days only for amounts > {currency}{amount}; others presumably still 30.\n"
            "Step 3: Rev C — 15 days for all, regardless of amount.\n"
            "Step 4: Rev C is the latest and simplifies the rule.\n"
            "Confidence: High — latest revision is unambiguous."
        ),
        "answer": (
            "Per the latest revision (Rev C), all expense reports must be submitted within 15 days, "
            "regardless of amount. This supersedes the previous tiered deadlines."
        ),
    },
    {
        "query": (
            "Regulatory filings for {company} in the {domain} sector:\n"
            "- {quarter} {year}: Compliance score = {pct}%.\n"
            "- {quarter} {year2}: Compliance score = {pct2}%.\n\n"
            "What is the trend in {company}'s compliance performance?"
        ),
        "reasoning": (
            "Step 1: {quarter} {year} score = {pct}%.\n"
            "Step 2: {quarter} {year2} score = {pct2}%.\n"
            "Step 3: Compare — if {pct2}% > {pct}%, improving; if lower, declining.\n"
            "Step 4: Consider that only two data points limits trend analysis.\n"
            "Confidence: Medium — only two data points available."
        ),
        "answer": (
            "Comparing {quarter} {year} ({pct}%) to {quarter} {year2} ({pct2}%), {company}'s "
            "compliance score changed by {pct2}% - {pct}% percentage points. Note: with only "
            "two data points, this represents a directional indicator rather than a robust trend."
        ),
    },
]

_TEMPORAL_DPO: List[Dict[str, str]] = [
    {
        "query": (
            "{company} board minutes:\n"
            "- {year}: Approved budget of {currency}{amount} for {dept}.\n"
            "- {year2}: Revised budget to {currency}{amount2}.\n\n"
            "What is {dept}'s current budget?"
        ),
        "chosen_reasoning": (
            "Step 1: Two budget decisions — {year} at {currency}{amount}, {year2} at {currency}{amount2}.\n"
            "Step 2: Board revisions supersede earlier approvals.\n"
            "Step 3: The {year2} revision is the latest action, making {currency}{amount2} current.\n"
            "Step 4: Verify no subsequent revisions are mentioned.\n"
            "Confidence: High — clear temporal supersession."
        ),
        "chosen_answer": (
            "The current {dept} budget is {currency}{amount2}, as revised by the board in {year2}. "
            "This supersedes the original {currency}{amount} approved in {year}."
        ),
        "rejected_reasoning": "The budget was changed.",
        "rejected_answer": "The budget is {currency}{amount2}.",
    },
]


# ===================================================================
# 3. IMPLICIT INTENT DECOMPOSITION (300 SFT + DPO)
# ===================================================================

_INTENT_SFT: List[Dict[str, str]] = [
    {
        "query": (
            "I need to understand the risk exposure from the {company} deal."
        ),
        "reasoning": (
            "Step 1: The user's query is vague — 'risk exposure' could mean financial risk, "
            "legal risk, operational risk, or reputational risk.\n"
            "Step 2: 'The {company} deal' implies a specific transaction — likely a contract, "
            "acquisition, or partnership.\n"
            "Step 3: Decompose into analytical sub-tasks:\n"
            "  a) Identify the deal structure and counterparty obligations.\n"
            "  b) Assess financial exposure (liability caps, penalties, payment terms).\n"
            "  c) Review legal risk (indemnification, warranties, dispute resolution).\n"
            "  d) Evaluate operational risk (dependencies, SLAs, performance guarantees).\n"
            "  e) Check compliance risk (regulatory requirements, certifications).\n"
            "Step 4: Determine which documents to examine — contract, amendments, due diligence.\n"
            "Confidence: Medium — need to clarify which risk dimensions matter most."
        ),
        "answer": (
            "To assess risk exposure from the {company} deal, I recommend the following "
            "structured analysis plan:\n\n"
            "1. **Financial Risk**: Review liability caps, penalty clauses, payment terms, and "
            "total contract value.\n"
            "2. **Legal Risk**: Examine indemnification obligations, warranty scope, and dispute "
            "resolution mechanisms.\n"
            "3. **Operational Risk**: Assess SLAs, performance guarantees, and dependency risks.\n"
            "4. **Compliance Risk**: Verify regulatory requirements and certifications.\n\n"
            "Which dimensions would you like me to prioritize? And can you point me to the "
            "relevant deal documents?"
        ),
    },
    {
        "query": "Can you check if everything is in order with {person}'s paperwork?",
        "reasoning": (
            "Step 1: Vague query — 'everything in order' and 'paperwork' are undefined.\n"
            "Step 2: Context clues — {person} is likely an employee, candidate, or vendor contact.\n"
            "Step 3: Decompose 'paperwork' into likely document categories:\n"
            "  a) Employment: offer letter, contract, NDA, tax forms, ID verification.\n"
            "  b) Compliance: background check, certifications, training records.\n"
            "  c) Administrative: benefits enrollment, emergency contacts, bank details.\n"
            "Step 4: 'In order' means: documents exist, are signed, not expired, consistent.\n"
            "Step 5: Build a checklist approach for systematic review.\n"
            "Confidence: Low — need to know {person}'s role context to narrow scope."
        ),
        "answer": (
            "To verify {person}'s paperwork, I'll systematically check:\n\n"
            "1. **Employment Documents**: Signed offer letter, employment contract, NDA.\n"
            "2. **Identity & Tax**: Government ID, tax forms (W-4/W-9), work authorization.\n"
            "3. **Compliance**: Background check completion, required certifications, training records.\n"
            "4. **Administrative**: Benefits enrollment, emergency contact, direct deposit setup.\n\n"
            "Could you confirm {person}'s role (new hire, existing employee, contractor)? This will "
            "help me focus on the relevant document set."
        ),
    },
    {
        "query": "What's the status of the {dept} restructuring?",
        "reasoning": (
            "Step 1: 'Status' is open-ended — could mean progress, timeline, approvals, or impact.\n"
            "Step 2: 'Restructuring' implies organizational changes — headcount, reporting lines, "
            "budget reallocations.\n"
            "Step 3: Decompose into structured inquiry:\n"
            "  a) What phase is the restructuring in (planning, approved, in-progress, complete)?\n"
            "  b) What changes are planned (headcount, org chart, budget)?\n"
            "  c) What approvals are pending or granted?\n"
            "  d) What is the timeline and key milestones?\n"
            "  e) What is the impact on ongoing projects and staffing?\n"
            "Step 4: Identify relevant documents — board resolutions, HR plans, budget proposals.\n"
            "Confidence: Low — broad query requiring document identification first."
        ),
        "answer": (
            "To provide a comprehensive restructuring status, I need to assess:\n\n"
            "1. **Phase**: Is the restructuring in planning, approved, in-progress, or completed?\n"
            "2. **Scope**: What organizational changes are involved (headcount, reporting lines, "
            "budget reallocations)?\n"
            "3. **Approvals**: What board/executive approvals have been obtained or are pending?\n"
            "4. **Timeline**: Key milestones and target completion dates.\n"
            "5. **Impact**: Effects on ongoing projects, staffing, and budgets.\n\n"
            "Please share relevant documents (e.g., board resolutions, HR restructuring plans, "
            "budget proposals) so I can provide a detailed status assessment."
        ),
    },
    {
        "query": "Help me prepare for the {company} negotiation next week.",
        "reasoning": (
            "Step 1: 'Prepare for negotiation' is multi-faceted — could mean reviewing terms, "
            "identifying leverage, or drafting counter-proposals.\n"
            "Step 2: Decompose preparation into key areas:\n"
            "  a) Review current contract/proposal terms with {company}.\n"
            "  b) Identify key negotiation points (price, terms, SLAs).\n"
            "  c) Assess BATNA (Best Alternative to Negotiated Agreement).\n"
            "  d) Analyze {company}'s likely positions and constraints.\n"
            "  e) Prepare supporting data (market benchmarks, historical terms).\n"
            "Step 3: Determine available documents for analysis.\n"
            "Confidence: Low — need specifics about negotiation scope and available docs."
        ),
        "answer": (
            "To prepare for the {company} negotiation, I recommend this structured approach:\n\n"
            "1. **Terms Review**: Analyze the current proposal/contract and identify favorable "
            "and unfavorable clauses.\n"
            "2. **Key Issues**: List the top negotiation points (pricing, SLAs, liability, terms).\n"
            "3. **Leverage Analysis**: Identify your BATNA and areas of relative strength.\n"
            "4. **Counterparty Analysis**: Assess {company}'s likely priorities and constraints.\n"
            "5. **Data Package**: Compile market benchmarks and precedent from similar deals.\n\n"
            "Share the current proposal and any prior correspondence so I can begin detailed analysis."
        ),
    },
    {
        "query": "Are we compliant with the new {domain} regulations?",
        "reasoning": (
            "Step 1: 'New regulations' is vague — which specific regulation?\n"
            "Step 2: 'Compliant' requires a gap analysis: current state vs requirements.\n"
            "Step 3: Decompose into structured plan:\n"
            "  a) Identify the specific regulation(s) in question.\n"
            "  b) Map regulatory requirements to internal policies and controls.\n"
            "  c) Assess gaps between current state and requirements.\n"
            "  d) Evaluate remediation timeline and risk.\n"
            "  e) Check documentation and evidence of compliance.\n"
            "Step 4: Need access to regulatory text and current compliance reports.\n"
            "Confidence: Low — cannot determine compliance without specific regulation reference."
        ),
        "answer": (
            "To assess {domain} regulatory compliance, I need to:\n\n"
            "1. **Identify Regulations**: Which specific {domain} regulations are you referring to?\n"
            "2. **Requirements Mapping**: Map each regulatory requirement to your current policies.\n"
            "3. **Gap Analysis**: Identify areas where current controls fall short.\n"
            "4. **Evidence Review**: Verify documentation and audit trails exist.\n"
            "5. **Remediation Plan**: Prioritize gaps by risk severity and timeline.\n\n"
            "Please specify the regulation(s) and share current compliance reports so I can "
            "perform a targeted analysis."
        ),
    },
]

_INTENT_DPO: List[Dict[str, str]] = [
    {
        "query": "Look into the {company} situation for me.",
        "chosen_reasoning": (
            "Step 1: Extremely vague — 'situation' could mean financial status, legal dispute, "
            "partnership status, or operational issues.\n"
            "Step 2: Need to decompose into possible interpretations:\n"
            "  a) Financial: outstanding invoices, credit risk, payment history.\n"
            "  b) Legal: pending litigation, contract disputes, compliance issues.\n"
            "  c) Operational: service delivery, SLA performance, project status.\n"
            "  d) Strategic: partnership health, renewal decisions, expansion.\n"
            "Step 3: Present a structured inquiry plan and ask for clarification.\n"
            "Confidence: Low — need user to narrow scope."
        ),
        "chosen_answer": (
            "To investigate the {company} situation thoroughly, I can analyze across these dimensions:\n\n"
            "1. **Financial**: Payment history, outstanding invoices, credit exposure.\n"
            "2. **Legal/Contractual**: Active contracts, disputes, compliance status.\n"
            "3. **Operational**: Service delivery performance, SLA adherence.\n"
            "4. **Strategic**: Relationship health, renewal timeline, expansion opportunities.\n\n"
            "Which area is most pressing? And please share any relevant documents or background."
        ),
        "rejected_reasoning": "The user wants to know about {company}.",
        "rejected_answer": "I'd be happy to look into {company}. What would you like to know?",
    },
]


# ===================================================================
# 4. CAUSAL CHAIN REASONING (250 SFT + DPO)
# ===================================================================

_CAUSAL_SFT: List[Dict[str, str]] = [
    {
        "query": (
            "From the {domain} incident report for {company}:\n"
            "- {dept} missed the {quarter} {year} deadline for deliverable X.\n"
            "- Deliverable X was a prerequisite for {dept2}'s budget approval.\n"
            "- {dept2} budget was not approved, delaying hiring.\n"
            "- Without new hires, the {product} launch was postponed.\n\n"
            "Why was the {product} launch postponed?"
        ),
        "reasoning": (
            "Step 1: Build causal chain:\n"
            "  {dept} missed deadline -> deliverable X not completed\n"
            "  -> {dept2} budget not approved (prerequisite missing)\n"
            "  -> hiring delayed (no budget)\n"
            "  -> {product} launch postponed (insufficient staff)\n"
            "Step 2: Root cause = {dept} missing the {quarter} {year} deadline.\n"
            "Step 3: Three intermediate links connect root cause to final effect.\n"
            "Step 4: Each link is documented in the incident report.\n"
            "Confidence: High — complete causal chain documented."
        ),
        "answer": (
            "The {product} launch was postponed due to a causal chain originating in {dept}:\n\n"
            "1. {dept} missed the {quarter} {year} deadline for deliverable X.\n"
            "2. Since deliverable X was a prerequisite for {dept2}'s budget approval, the budget "
            "was not approved.\n"
            "3. Without budget approval, {dept2} could not proceed with planned hiring.\n"
            "4. The {product} launch required the new hires and was therefore postponed.\n\n"
            "Root cause: {dept}'s missed deadline on deliverable X."
        ),
    },
    {
        "query": (
            "{company}'s compliance failure analysis:\n"
            "- Vendor {company2} failed to renew their security certification.\n"
            "- {company} relied on {company2}'s certification for their own compliance.\n"
            "- {company}'s compliance report flagged a gap.\n"
            "- The regulator issued a warning letter to {company}.\n\n"
            "Why did {company} receive a regulatory warning?"
        ),
        "reasoning": (
            "Step 1: Causal chain:\n"
            "  {company2} certification lapsed -> {company}'s compliance dependent on it\n"
            "  -> {company}'s compliance report shows gap -> regulator issues warning.\n"
            "Step 2: Root cause = third-party vendor ({company2}) failed to renew certification.\n"
            "Step 3: {company}'s vulnerability was dependency on a vendor's compliance status.\n"
            "Step 4: Systemic issue — vendor management and monitoring gap.\n"
            "Confidence: High — clear documented chain."
        ),
        "answer": (
            "{company} received a regulatory warning because:\n\n"
            "1. Vendor {company2} failed to renew their security certification.\n"
            "2. {company}'s own compliance relied on {company2}'s active certification.\n"
            "3. This created a gap flagged in {company}'s compliance report.\n"
            "4. The regulator identified this gap and issued a warning.\n\n"
            "Root cause: Vendor dependency without adequate certification monitoring."
        ),
    },
    {
        "query": (
            "From {company}'s {year} annual review:\n"
            "- {dept} adopted a new approval workflow in {quarter}.\n"
            "- Processing time increased from 2 days to 8 days.\n"
            "- Three clients escalated complaints about delayed orders.\n"
            "- Client satisfaction scores dropped by {pct}%.\n\n"
            "What caused the drop in client satisfaction?"
        ),
        "reasoning": (
            "Step 1: Trace the chain:\n"
            "  New approval workflow -> processing time increased (2 to 8 days)\n"
            "  -> delayed orders -> client complaints -> satisfaction drop ({pct}%).\n"
            "Step 2: Root cause = the new {dept} approval workflow.\n"
            "Step 3: The workflow change was intended as an improvement but had unintended "
            "consequences on turnaround time.\n"
            "Step 4: The 4x increase in processing time was the critical bottleneck.\n"
            "Confidence: High — well-documented cause-effect chain in annual review."
        ),
        "answer": (
            "The {pct}% drop in client satisfaction traces back to {dept}'s new approval workflow:\n\n"
            "1. {dept} introduced a new approval workflow in {quarter}.\n"
            "2. Processing time quadrupled from 2 to 8 days.\n"
            "3. Extended processing caused order delays, triggering 3 client escalations.\n"
            "4. Cumulative delays drove client satisfaction down by {pct}%.\n\n"
            "The root cause is the workflow change itself — consider streamlining the new process "
            "or adding parallel approval paths."
        ),
    },
    {
        "query": (
            "Audit trail from {company}:\n"
            "- {person} submitted an expense claim for {currency}{amount}.\n"
            "- The claim exceeded the {currency}{amount3} auto-approval threshold.\n"
            "- {person2} (manager) was on leave; no backup approver configured.\n"
            "- Claim sat in queue for 45 days, triggering a compliance flag.\n\n"
            "Why was the compliance flag triggered?"
        ),
        "reasoning": (
            "Step 1: Chain: expense > threshold -> needs manual approval\n"
            "  -> approver on leave + no backup -> claim stuck 45 days\n"
            "  -> exceeded compliance deadline -> flag triggered.\n"
            "Step 2: Multiple contributing causes: threshold exceeded, approver absent, "
            "no delegation configured.\n"
            "Step 3: Root cause is process gap — no backup approver for amounts above threshold.\n"
            "Confidence: High — complete audit trail available."
        ),
        "answer": (
            "The compliance flag was triggered through this chain:\n\n"
            "1. {person}'s {currency}{amount} claim exceeded the {currency}{amount3} auto-approval limit.\n"
            "2. Manual approval was required from {person2}, who was on leave.\n"
            "3. No backup approver was configured in the system.\n"
            "4. The claim remained unprocessed for 45 days, exceeding the compliance deadline.\n\n"
            "Root cause: Process gap — no delegation or backup approver configuration for "
            "above-threshold expense claims."
        ),
    },
]

_CAUSAL_DPO: List[Dict[str, str]] = [
    {
        "query": (
            "{company} post-mortem:\n"
            "- Server migration scheduled for {month} {year}.\n"
            "- Migration script had an untested rollback procedure.\n"
            "- Data corruption occurred during migration.\n"
            "- Rollback failed, extending downtime to 72 hours.\n"
            "- {qty} client transactions were lost.\n\n"
            "Why were client transactions lost?"
        ),
        "chosen_reasoning": (
            "Step 1: Build full causal chain:\n"
            "  Migration scheduled -> script had untested rollback\n"
            "  -> data corruption during migration -> rollback attempted but failed\n"
            "  -> extended downtime (72 hrs) -> {qty} transactions lost.\n"
            "Step 2: Root cause analysis — two failures:\n"
            "  a) Primary: Data corruption during migration (execution failure).\n"
            "  b) Compounding: Untested rollback procedure (preparedness failure).\n"
            "Step 3: Had the rollback worked, transactions could have been recovered.\n"
            "Step 4: The untested rollback is the critical systemic failure.\n"
            "Confidence: High — post-mortem provides complete chain."
        ),
        "chosen_answer": (
            "Client transactions were lost due to a compound failure chain:\n\n"
            "1. The server migration encountered data corruption.\n"
            "2. The rollback procedure, which had never been tested, failed.\n"
            "3. The failed rollback extended downtime to 72 hours.\n"
            "4. During this extended outage, {qty} client transactions were irrecoverably lost.\n\n"
            "Root cause: The untested rollback procedure transformed a recoverable data corruption "
            "event into permanent data loss. Recommendation: mandate rollback testing for all "
            "migration procedures."
        ),
        "rejected_reasoning": "There was a server migration problem.",
        "rejected_answer": (
            "Client transactions were lost because the server migration failed and "
            "data was corrupted."
        ),
    },
]


# ===================================================================
# 5. QUANTITATIVE REASONING (250 SFT + DPO)
# ===================================================================

_QUANT_SFT: List[Dict[str, str]] = [
    {
        "query": (
            "From {company}'s {domain} {doc_type}:\n"
            "| Quarter | Revenue | Expenses |\n"
            "| {quarter} {year} | {currency}{amount} | {currency}{amount2} |\n\n"
            "What is the profit margin for {quarter} {year}?"
        ),
        "reasoning": (
            "Step 1: Profit = Revenue - Expenses = {currency}{amount} - {currency}{amount2}.\n"
            "Step 2: Profit Margin = (Profit / Revenue) x 100.\n"
            "Step 3: Compute the margin percentage.\n"
            "Step 4: Interpret — is this a healthy margin for the {domain} sector?\n"
            "Confidence: High — straightforward calculation from provided data."
        ),
        "answer": (
            "Profit for {quarter} {year}: {currency}{amount} (revenue) - {currency}{amount2} (expenses). "
            "Profit margin = (Revenue - Expenses) / Revenue x 100%. This gives the percentage of "
            "revenue retained as profit."
        ),
    },
    {
        "query": (
            "{company}'s {dept} budget data:\n"
            "| Year | Budget | Actual |\n"
            "| {year} | {currency}{amount} | {currency}{amount2} |\n\n"
            "What is the budget variance and is it favorable or unfavorable?"
        ),
        "reasoning": (
            "Step 1: Variance = Actual - Budget = {currency}{amount2} - {currency}{amount}.\n"
            "Step 2: If Variance < 0 (actual < budget), it's favorable (under budget).\n"
            "Step 3: If Variance > 0 (actual > budget), it's unfavorable (over budget).\n"
            "Step 4: Calculate percentage variance = (Variance / Budget) x 100.\n"
            "Confidence: High — standard variance calculation."
        ),
        "answer": (
            "Budget variance for {dept} in {year}: Actual ({currency}{amount2}) minus Budget "
            "({currency}{amount}). If the result is negative, the variance is favorable (under budget); "
            "if positive, it's unfavorable (over budget). Percentage variance = "
            "|(Actual - Budget) / Budget| x 100%."
        ),
    },
    {
        "query": (
            "Year-over-year comparison for {company}:\n"
            "- {year} revenue: {currency}{amount}\n"
            "- {year2} revenue: {currency}{amount2}\n\n"
            "What is the year-over-year growth rate?"
        ),
        "reasoning": (
            "Step 1: YoY Growth = ((Year2 - Year1) / Year1) x 100.\n"
            "Step 2: Year1 = {currency}{amount} ({year}), Year2 = {currency}{amount2} ({year2}).\n"
            "Step 3: Compute delta = {currency}{amount2} - {currency}{amount}.\n"
            "Step 4: Growth rate = delta / {currency}{amount} x 100.\n"
            "Step 5: Positive = growth, negative = decline.\n"
            "Confidence: High — standard calculation."
        ),
        "answer": (
            "Year-over-year growth rate = ({currency}{amount2} - {currency}{amount}) / {currency}{amount} x 100%. "
            "This measures the percentage change in revenue from {year} to {year2}."
        ),
    },
    {
        "query": (
            "{company}'s headcount report:\n"
            "| Department | Headcount | Avg Salary |\n"
            "| {dept} | {qty} | {currency}{amount} |\n"
            "| {dept2} | {qty} | {currency}{amount2} |\n\n"
            "What is the total payroll cost across both departments?"
        ),
        "reasoning": (
            "Step 1: {dept} payroll = {qty} x {currency}{amount}.\n"
            "Step 2: {dept2} payroll = {qty} x {currency}{amount2}.\n"
            "Step 3: Total = {dept} payroll + {dept2} payroll.\n"
            "Step 4: This assumes all employees are at the average salary.\n"
            "Confidence: High — straightforward multiplication and addition."
        ),
        "answer": (
            "Total payroll = ({qty} x {currency}{amount}) + ({qty} x {currency}{amount2}). "
            "{dept} costs {qty} x {currency}{amount} and {dept2} costs {qty} x {currency}{amount2}."
        ),
    },
    {
        "query": (
            "From {company}'s {domain} report:\n"
            "Total claims: {qty}\n"
            "Approved: {amount}\n"
            "Denied: {amount2}\n"
            "Pending: {amount3}\n\n"
            "What percentage of claims were approved?"
        ),
        "reasoning": (
            "Step 1: Approval rate = (Approved / Total) x 100.\n"
            "Step 2: Approved = {amount}, Total = {qty}.\n"
            "Step 3: Calculate {amount} / {qty} x 100.\n"
            "Step 4: Cross-check: Approved + Denied + Pending should = Total.\n"
            "Confidence: High — direct division."
        ),
        "answer": (
            "The claim approval rate is ({amount} / {qty}) x 100%. Cross-verification: "
            "approved ({amount}) + denied ({amount2}) + pending ({amount3}) should equal "
            "total ({qty})."
        ),
    },
]

_QUANT_DPO: List[Dict[str, str]] = [
    {
        "query": (
            "{company}'s {quarter} report:\n"
            "Revenue: {currency}{amount}\n"
            "COGS: {currency}{amount2}\n"
            "Operating Expenses: {currency}{amount3}\n\n"
            "Calculate gross margin, operating margin, and comment on profitability."
        ),
        "chosen_reasoning": (
            "Step 1: Gross Profit = Revenue - COGS = {currency}{amount} - {currency}{amount2}.\n"
            "Step 2: Gross Margin = Gross Profit / Revenue x 100.\n"
            "Step 3: Operating Profit = Revenue - COGS - OpEx = {currency}{amount} - "
            "{currency}{amount2} - {currency}{amount3}.\n"
            "Step 4: Operating Margin = Operating Profit / Revenue x 100.\n"
            "Step 5: Compare margins to assess operational efficiency.\n"
            "Step 6: Consider industry context ({domain} sector).\n"
            "Confidence: High — standard financial metrics."
        ),
        "chosen_answer": (
            "**Gross Margin**: ({currency}{amount} - {currency}{amount2}) / {currency}{amount} x 100%\n"
            "**Operating Margin**: ({currency}{amount} - {currency}{amount2} - {currency}{amount3}) "
            "/ {currency}{amount} x 100%\n\n"
            "The difference between gross and operating margin indicates the operational cost "
            "burden. A narrowing gap between the two suggests operating expenses are well-controlled."
        ),
        "rejected_reasoning": "Need to calculate margins.",
        "rejected_answer": "The margins can be calculated from the revenue and cost data.",
    },
]


# ===================================================================
# 6. COUNTERFACTUAL ANALYSIS (150 SFT + DPO)
# ===================================================================

_COUNTERFACTUAL_SFT: List[Dict[str, str]] = [
    {
        "query": (
            "Given {company}'s contract with {company2}:\n"
            "- {clause}: Liability cap of {currency}{amount}.\n"
            "- {clause2}: Indemnification for third-party IP claims.\n\n"
            "What if {clause} were removed from the contract?"
        ),
        "reasoning": (
            "Step 1: Removing {clause} eliminates the {currency}{amount} liability cap.\n"
            "Step 2: Without a cap, {company}'s liability exposure becomes unlimited.\n"
            "Step 3: Impact on risk profile — significantly increased financial risk.\n"
            "Step 4: Impact on {clause2} — indemnification claims would also be uncapped.\n"
            "Step 5: Business impact — may require additional insurance or reserves.\n"
            "Step 6: Negotiation impact — the other party gains significant leverage.\n"
            "Confidence: High — well-understood legal implications."
        ),
        "answer": (
            "Removing {clause} (Liability Cap of {currency}{amount}) would have these consequences:\n\n"
            "1. **Financial Risk**: Liability exposure becomes unlimited — any breach or claim could "
            "result in damages far exceeding {currency}{amount}.\n"
            "2. **Interaction with {clause2}**: Indemnification obligations for third-party IP claims "
            "would also be uncapped, compounding risk.\n"
            "3. **Insurance Impact**: May require significantly higher insurance coverage.\n"
            "4. **Negotiation Position**: The counterparty gains substantial leverage in any dispute.\n\n"
            "Recommendation: Retain the liability cap or negotiate a higher but defined limit."
        ),
    },
    {
        "query": (
            "{company}'s employment policy states:\n"
            "- Probation period: 90 days.\n"
            "- During probation, either party may terminate with 1 week notice.\n"
            "- After probation, termination requires 30 days notice.\n\n"
            "What if the probation period were extended to 180 days?"
        ),
        "reasoning": (
            "Step 1: Extending probation from 90 to 180 days doubles the at-will-like period.\n"
            "Step 2: Impact on employees — longer period of job insecurity, delayed benefits.\n"
            "Step 3: Impact on employer — more time to evaluate, but may affect recruitment.\n"
            "Step 4: Legal considerations — some jurisdictions limit probation length.\n"
            "Step 5: Retention risk — top candidates may reject offers with 180-day probation.\n"
            "Confidence: Medium — effects are predictable but jurisdiction-dependent."
        ),
        "answer": (
            "Extending the probation period from 90 to 180 days would:\n\n"
            "1. **Employee Impact**: Longer uncertainty period; delayed full employment protections.\n"
            "2. **Recruitment Risk**: May deter top candidates who have competing offers.\n"
            "3. **Employer Benefit**: Extended evaluation period for performance assessment.\n"
            "4. **Legal Risk**: Some jurisdictions cap probation periods — verify compliance.\n"
            "5. **Operational**: More terminations during probation may increase hiring costs.\n\n"
            "Consider alternatives: structured 90-day reviews with optional 90-day extension."
        ),
    },
    {
        "query": (
            "From {company}'s vendor agreement:\n"
            "- SLA: 99.9% uptime guarantee.\n"
            "- Penalty: {pct}% credit per hour of downtime exceeding SLA.\n\n"
            "What if the SLA were reduced to 99.5%?"
        ),
        "reasoning": (
            "Step 1: 99.9% uptime = max 8.76 hours downtime/year.\n"
            "Step 2: 99.5% uptime = max 43.8 hours downtime/year (5x more allowed).\n"
            "Step 3: Penalty triggers would decrease — vendor has 35+ more hours of buffer.\n"
            "Step 4: Business impact — more acceptable downtime, potential service degradation.\n"
            "Step 5: Financial impact — fewer penalty credits for {company}.\n"
            "Step 6: Risk assessment — depends on how critical the service is.\n"
            "Confidence: High — quantifiable difference."
        ),
        "answer": (
            "Reducing the SLA from 99.9% to 99.5% would:\n\n"
            "1. **Allowed Downtime**: Increase from ~8.76 hours/year to ~43.8 hours/year (5x).\n"
            "2. **Penalty Exposure**: Vendor faces far fewer penalty triggers, reducing "
            "the {pct}% credit protection.\n"
            "3. **Service Quality**: Permits significantly more downtime, which may impact "
            "business operations.\n"
            "4. **Cost Trade-off**: Vendor may offer lower pricing for reduced SLA.\n\n"
            "Recommendation: Maintain 99.9% for mission-critical services; 99.5% may be "
            "acceptable for non-critical systems."
        ),
    },
]

_COUNTERFACTUAL_DPO: List[Dict[str, str]] = [
    {
        "query": (
            "{company}'s NDA contains:\n"
            "- Non-compete clause: 2-year restriction within {domain} sector.\n"
            "- Non-solicitation: 1 year for employees and clients.\n\n"
            "What if the non-compete clause were removed?"
        ),
        "chosen_reasoning": (
            "Step 1: Removing non-compete eliminates 2-year {domain} sector restriction.\n"
            "Step 2: Impact on talent mobility — departing employees can immediately join competitors.\n"
            "Step 3: Impact on trade secrets — non-solicitation remains but doesn't prevent "
            "competitive employment.\n"
            "Step 4: Legal landscape — many jurisdictions are restricting non-competes anyway.\n"
            "Step 5: Retention implication — employees may feel less trapped, potentially improving morale.\n"
            "Step 6: Risk mitigation — strengthen NDA and non-solicitation to compensate.\n"
            "Confidence: Medium — depends on jurisdiction and specific competitive landscape."
        ),
        "chosen_answer": (
            "Removing the non-compete clause would have multi-dimensional effects:\n\n"
            "1. **Talent Mobility**: Ex-employees could immediately join {domain} competitors.\n"
            "2. **Competitive Risk**: Proprietary knowledge could reach competitors faster.\n"
            "3. **Legal Alignment**: Many jurisdictions are banning non-competes, so removal "
            "may pre-empt legal challenges.\n"
            "4. **Non-Solicitation Remains**: The 1-year non-solicitation still provides some protection.\n"
            "5. **Morale**: May improve employee satisfaction and recruitment competitiveness.\n\n"
            "Recommendation: If removing non-compete, strengthen trade secret protections "
            "and consider garden leave provisions."
        ),
        "rejected_reasoning": "Non-compete clauses restrict competition.",
        "rejected_answer": (
            "Without the non-compete, employees could work for competitors after leaving."
        ),
    },
]


# ===================================================================
# 7. UNCERTAINTY CALIBRATION (250 SFT + DPO)
# ===================================================================

_UNCERTAINTY_SFT: List[Dict[str, str]] = [
    {
        "query": (
            "Based on the {domain} {doc_type} from {company}:\n"
            "\"{person} is listed as authorized signatory for amounts up to {currency}{amount}.\"\n\n"
            "Can {person} approve a {currency}{amount2} transaction?"
        ),
        "reasoning": (
            "Step 1: {person}'s authorization limit is {currency}{amount}.\n"
            "Step 2: The transaction is {currency}{amount2}.\n"
            "Step 3: If {amount2} > {amount}, the answer is no — exceeds authority.\n"
            "Step 4: However, there may be co-signing provisions or escalation paths not in this doc.\n"
            "Step 5: The document only covers one authorization source.\n"
            "Confidence: Medium — I can confirm the stated limit, but cannot rule out "
            "other authorization mechanisms."
        ),
        "answer": (
            "Based on the available {doc_type}, {person}'s signatory authority is capped at "
            "{currency}{amount}. A {currency}{amount2} transaction would exceed this limit. "
            "However, I cannot fully determine the answer because: (1) there may be co-signing "
            "provisions allowing {person} to participate with another approver, and (2) other "
            "documents may grant additional authority. I recommend checking the full authorization "
            "matrix."
        ),
    },
    {
        "query": (
            "From {company}'s {domain} report:\n"
            "\"The project is expected to complete by {quarter} {year}.\"\n\n"
            "Will the project finish on time?"
        ),
        "reasoning": (
            "Step 1: The document says 'expected to complete' — this is a projection, not a fact.\n"
            "Step 2: No information on current progress, risks, or blockers.\n"
            "Step 3: Cannot assess probability of on-time completion without status data.\n"
            "Step 4: The word 'expected' implies uncertainty even in the source document.\n"
            "Confidence: Low — insufficient data to make a prediction."
        ),
        "answer": (
            "The {domain} report projects completion by {quarter} {year}, but I have low confidence "
            "in predicting actual on-time delivery because: (1) the source itself uses 'expected' "
            "rather than 'confirmed', (2) I have no visibility into current progress, resource "
            "availability, or risk factors, and (3) no status updates beyond this projection are "
            "available. I recommend requesting the latest status report before making planning decisions."
        ),
    },
    {
        "query": (
            "{company}'s {doc_type} mentions:\n"
            "\"Revenue growth is projected at {pct}% for {year}.\"\n\n"
            "How confident should we be in this projection?"
        ),
        "reasoning": (
            "Step 1: This is a forward-looking projection, inherently uncertain.\n"
            "Step 2: No basis or methodology for the projection is provided.\n"
            "Step 3: Cannot assess historical accuracy of {company}'s projections.\n"
            "Step 4: External factors (market conditions, competition) not accounted for.\n"
            "Step 5: Single-point estimate with no range or scenario analysis.\n"
            "Confidence: Low — projection without supporting analysis."
        ),
        "answer": (
            "Confidence in the {pct}% growth projection should be tempered by several factors:\n\n"
            "1. **No Methodology**: The document doesn't explain the basis for this figure.\n"
            "2. **No Range**: A single-point estimate without confidence intervals is less reliable.\n"
            "3. **Unknown Track Record**: I have no data on {company}'s historical projection accuracy.\n"
            "4. **External Factors**: Market conditions and competitive dynamics could significantly "
            "impact actual results.\n\n"
            "I recommend treating this as a directional indicator rather than a reliable forecast."
        ),
    },
    {
        "query": (
            "From partial OCR scan of {company}'s {doc_type}:\n"
            "\"{person} reports to [illegible] in the {dept} department.\"\n\n"
            "Who does {person} report to?"
        ),
        "reasoning": (
            "Step 1: The reporting manager's name is illegible in the OCR scan.\n"
            "Step 2: I cannot determine the name from this document alone.\n"
            "Step 3: The department ({dept}) is visible, which narrows possibilities.\n"
            "Step 4: Cross-referencing with an org chart could resolve this.\n"
            "Confidence: Very Low — critical information is missing from the source."
        ),
        "answer": (
            "I cannot determine who {person} reports to because the manager's name is illegible "
            "in the OCR scan. What I can confirm: {person} is in the {dept} department, and they "
            "report to someone within that department. To resolve this, I would need: (1) a higher "
            "quality scan of this document, (2) the current {dept} org chart, or (3) HR records "
            "showing reporting relationships."
        ),
    },
    {
        "query": (
            "{company}'s {domain} filing ({year}):\n"
            "\"Total assets: {currency}{amount}.\"\n\n"
            "No other financial documents are available.\n"
            "What is {company}'s financial health?"
        ),
        "reasoning": (
            "Step 1: Only one data point — total assets = {currency}{amount}.\n"
            "Step 2: Financial health requires: assets, liabilities, equity, revenue, cash flow.\n"
            "Step 3: Cannot compute debt-to-equity, current ratio, or any profitability metric.\n"
            "Step 4: Total assets alone says nothing about obligations or liquidity.\n"
            "Confidence: Very Low — grossly insufficient data for financial health assessment."
        ),
        "answer": (
            "I cannot meaningfully assess {company}'s financial health from total assets alone "
            "({currency}{amount}). A proper assessment requires:\n\n"
            "- **Liabilities**: To compute net worth and leverage ratios.\n"
            "- **Revenue & Expenses**: To assess profitability.\n"
            "- **Cash Flow**: To evaluate liquidity.\n"
            "- **Comparative Data**: Industry benchmarks or historical trends.\n\n"
            "The single asset figure of {currency}{amount} tells us the scale of the company but "
            "nothing about its obligations, profitability, or sustainability."
        ),
    },
]

_UNCERTAINTY_DPO: List[Dict[str, str]] = [
    {
        "query": (
            "{company}'s {doc_type} states:\n"
            "\"{person} was employed as of {month} {year}.\"\n\n"
            "Is {person} currently employed at {company}?"
        ),
        "chosen_reasoning": (
            "Step 1: Document confirms employment 'as of {month} {year}'.\n"
            "Step 2: No information about events after that date.\n"
            "Step 3: Employment status can change — resignation, termination, transfer.\n"
            "Step 4: Cannot extrapolate from a past-dated statement to current status.\n"
            "Confidence: Low — past-dated evidence cannot confirm current state."
        ),
        "chosen_answer": (
            "I can confirm that {person} was employed at {company} as of {month} {year}, based "
            "on the {doc_type}. However, I cannot determine their current employment status because "
            "this document only establishes a point-in-time fact. Employment may have ended since "
            "then. To confirm current status, check the latest HR records or employee directory."
        ),
        "rejected_reasoning": "The document says they work there.",
        "rejected_answer": "Yes, {person} is employed at {company}.",
    },
]


# ===================================================================
# 8. REFUSAL WITH EXPLANATION (200 SFT + DPO)
# ===================================================================

_REFUSAL_SFT: List[Dict[str, str]] = [
    {
        "query": (
            "Based on {company}'s {domain} {doc_type}, what is {person}'s Social Security Number?"
        ),
        "reasoning": (
            "Step 1: The query asks for a Social Security Number — this is PII.\n"
            "Step 2: Even if present in the document, exposing SSNs would violate "
            "data protection principles.\n"
            "Step 3: I should refuse this request and explain why.\n"
            "Step 4: Suggest alternative ways to verify identity if needed.\n"
            "Confidence: High — clear policy boundary."
        ),
        "answer": (
            "I cannot provide Social Security Numbers or similar personally identifiable "
            "information, even if present in the source documents. This is protected data "
            "subject to privacy regulations (GDPR, CCPA, etc.). If you need to verify "
            "{person}'s identity, please contact {company}'s HR department through secure channels."
        ),
    },
    {
        "query": (
            "The documents mention {company} has operations in several countries. "
            "Can you tell me if they are evading taxes?"
        ),
        "reasoning": (
            "Step 1: The query asks me to make a legal determination (tax evasion).\n"
            "Step 2: Tax evasion is a criminal matter requiring legal analysis beyond document review.\n"
            "Step 3: I can describe what the documents say about tax structures, but cannot "
            "conclude illegality.\n"
            "Step 4: This requires qualified legal and tax professionals.\n"
            "Confidence: N/A — outside scope of document analysis."
        ),
        "answer": (
            "I cannot determine whether {company} is evading taxes. Tax evasion is a legal "
            "conclusion that requires: (1) detailed analysis by qualified tax professionals, "
            "(2) access to complete financial records, and (3) legal interpretation of "
            "applicable tax codes across jurisdictions. I can describe the tax-related "
            "information present in the available documents, but any determination of "
            "legality must be made by appropriate legal counsel."
        ),
    },
    {
        "query": (
            "What will {company}'s stock price be in {quarter} {year}?"
        ),
        "reasoning": (
            "Step 1: The query asks for a stock price prediction.\n"
            "Step 2: Future stock prices are inherently unpredictable.\n"
            "Step 3: Document analysis cannot forecast market movements.\n"
            "Step 4: Providing such predictions could constitute financial advice.\n"
            "Confidence: N/A — impossible to predict."
        ),
        "answer": (
            "I cannot predict {company}'s future stock price. Stock prices are influenced by "
            "countless factors including market conditions, economic events, and investor "
            "sentiment that cannot be determined from document analysis alone. Providing such "
            "predictions would be speculative and could constitute financial advice. I can "
            "analyze {company}'s financial documents for historical performance and disclosed "
            "risks, but future market performance is outside my scope."
        ),
    },
    {
        "query": (
            "I found {person}'s medical records in the {company} HR files. "
            "Summarize their health conditions."
        ),
        "reasoning": (
            "Step 1: Medical records are among the most sensitive categories of personal data.\n"
            "Step 2: Accessing and summarizing health information raises HIPAA and privacy concerns.\n"
            "Step 3: Even if I have access, sharing medical details is inappropriate.\n"
            "Step 4: Should refuse and note the sensitivity.\n"
            "Confidence: High — clear ethical and legal boundary."
        ),
        "answer": (
            "I cannot summarize individual medical records or health conditions. Medical information "
            "is protected under health privacy regulations (e.g., HIPAA) and should only be accessed "
            "by authorized healthcare or HR personnel for legitimate purposes. If you need health-related "
            "information for accommodation or leave management, please work through your organization's "
            "designated privacy officer or HR compliance team."
        ),
    },
    {
        "query": (
            "Based on the {domain} documents, should {company} fire {person}?"
        ),
        "reasoning": (
            "Step 1: The query asks for a personnel action recommendation.\n"
            "Step 2: Termination decisions require context beyond document analysis — "
            "performance history, legal considerations, HR policies.\n"
            "Step 3: I should not make employment action recommendations.\n"
            "Step 4: I can present factual findings from documents but not recommend actions.\n"
            "Confidence: High — clear scope boundary."
        ),
        "answer": (
            "I cannot recommend personnel actions such as termination. This decision requires: "
            "(1) comprehensive performance evaluation, (2) HR and legal review, (3) consideration "
            "of employment law and contractual obligations, and (4) organizational context that "
            "extends well beyond document analysis. I can present factual findings from the "
            "available {domain} documents regarding {person}, but the decision must be made "
            "by appropriate management and HR personnel."
        ),
    },
]

_REFUSAL_DPO: List[Dict[str, str]] = [
    {
        "query": (
            "The {domain} {doc_type} from {company} mentions a whistleblower complaint. "
            "Who filed the complaint?"
        ),
        "chosen_reasoning": (
            "Step 1: Whistleblower identity is legally protected information.\n"
            "Step 2: Even if the identity is mentioned in the document, revealing it could "
            "violate whistleblower protection laws.\n"
            "Step 3: Exposing a whistleblower could lead to retaliation.\n"
            "Step 4: I should refuse and explain the legal protections.\n"
            "Confidence: High — legal and ethical boundary."
        ),
        "chosen_answer": (
            "I cannot identify the whistleblower, even if their identity appears in the documents. "
            "Whistleblower identities are protected by law to prevent retaliation. Revealing this "
            "information could violate whistleblower protection statutes and expose both the "
            "individual and your organization to legal liability. If you need information about "
            "the substance of the complaint, I can summarize the allegations without identifying "
            "the complainant."
        ),
        "rejected_reasoning": "The document mentions a whistleblower.",
        "rejected_answer": "I found the complaint in the document. Let me look for the name.",
    },
]


# ===================================================================
# PUBLIC API
# ===================================================================


def generate_track4_data(output_dir: str | Path, seed: int = 42) -> dict:
    """Generate Track 4 Context & Reasoning training data.

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
        (_CONTRADICTION_SFT, 300),
        (_TEMPORAL_SFT, 300),
        (_INTENT_SFT, 300),
        (_CAUSAL_SFT, 250),
        (_QUANT_SFT, 250),
        (_COUNTERFACTUAL_SFT, 150),
        (_UNCERTAINTY_SFT, 250),
        (_REFUSAL_SFT, 200),
    ]

    all_sft: List[Dict[str, str]] = []
    sub_seed = seed
    for templates, count in sft_categories:
        sub_seed += 1
        sub_rng = random.Random(sub_seed)
        all_sft.extend(_expand(templates, count, sub_rng, mode="sft"))

    rng.shuffle(all_sft)

    sft_path = output_dir / "track4_reasoning_sft.jsonl"
    with JSONLWriter(sft_path) as writer:
        for ex in all_sft:
            writer.write(ex)

    # --- DPO generation ---
    dpo_categories = [
        (_CONTRADICTION_DPO, 80),
        (_TEMPORAL_DPO, 60),
        (_INTENT_DPO, 60),
        (_CAUSAL_DPO, 50),
        (_QUANT_DPO, 50),
        (_COUNTERFACTUAL_DPO, 40),
        (_UNCERTAINTY_DPO, 50),
        (_REFUSAL_DPO, 40),
    ]

    all_dpo: List[Dict[str, str]] = []
    sub_seed = seed + 100
    for templates, count in dpo_categories:
        sub_seed += 1
        sub_rng = random.Random(sub_seed)
        all_dpo.extend(_expand(templates, count, sub_rng, mode="dpo"))

    rng.shuffle(all_dpo)

    dpo_path = output_dir / "track4_reasoning_dpo.jsonl"
    with JSONLWriter(dpo_path) as writer:
        for ex in all_dpo:
            writer.write(ex)

    return {
        "sft_path": str(sft_path),
        "dpo_path": str(dpo_path),
        "sft_count": len(all_sft),
        "dpo_count": len(all_dpo),
    }
