"""Phase 3.5 Insight data generator for DocWain V2+ SFT.

Generates 6K insight examples (at scale=1.0) across 7 categories:
  - pattern_recognition    (1000)
  - anomaly_detection      (1000)
  - trend_analysis         (1000)
  - comparative_analysis   (1000)
  - gap_analysis           (800)
  - holistic_synthesis     (700)
  - risk_assessment        (500)

Each example uses the DocWain Analysis Frame in ``<think>`` and wraps
the insight portion in ``<insight category="...">`` tags.
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

INSIGHT_CATEGORIES = [
    "pattern_recognition",
    "anomaly_detection",
    "trend_analysis",
    "comparative_analysis",
    "gap_analysis",
    "holistic_synthesis",
    "risk_assessment",
]

_CATEGORY_COUNTS = {
    "pattern_recognition": 1000,
    "anomaly_detection": 1000,
    "trend_analysis": 1000,
    "comparative_analysis": 1000,
    "gap_analysis": 800,
    "holistic_synthesis": 700,
    "risk_assessment": 500,
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
_METRICS = [
    "revenue", "headcount", "operating margin", "customer churn",
    "NPS score", "ticket volume", "defect rate", "cycle time",
]


def _pick(lst: list, rng: random.Random) -> Any:
    return rng.choice(lst)


def _rand_amount(rng: random.Random, lo: float = 100.0, hi: float = 99999.0) -> str:
    return f"{rng.uniform(lo, hi):,.2f}"


def _rand_pct(rng: random.Random) -> str:
    return f"{rng.uniform(1, 45):.1f}"


# ---------------------------------------------------------------------------
# Analysis Frame reasoning builder
# ---------------------------------------------------------------------------

def _analysis_frame(
    step1: str, step2: str, step3: str, step4: str, step5: str, confidence: str
) -> str:
    """Build a DocWain Analysis Frame reasoning block."""
    return (
        f"Step 1: What am I looking at? {step1}\n"
        f"Step 2: Key facts. {step2}\n"
        f"Step 3: Patterns. {step3}\n"
        f"Step 4: What's missing. {step4}\n"
        f"Step 5: Implications. {step5}\n"
        f"Step 6: Confidence assessment. {confidence}"
    )


def _insight_answer(category: str, summary: str, findings: str,
                    analysis: str, flags: str, confidence: str) -> str:
    """Build an insight answer with the standard structure."""
    return (
        f"<insight category=\"{category}\">\n"
        f"**Summary:** {summary}\n\n"
        f"**Key Findings:**\n{findings}\n\n"
        f"**Analysis:**\n{analysis}\n\n"
        f"**Risk/Opportunity Flags:**\n{flags}\n\n"
        f"**Confidence:** {confidence}\n"
        f"</insight>"
    )


# ---------------------------------------------------------------------------
# Templates per category
# ---------------------------------------------------------------------------

_PATTERN_TEMPLATES: List[Dict[str, str]] = [
    {
        "query": (
            "Analyze the following {domain} data for {company} across the last "
            "4 quarters and identify any patterns:\n\n"
            "| Quarter | {metric} |\n"
            "| Q1 | {currency}{amount} |\n"
            "| Q2 | {currency}{amount2} |\n"
            "| Q3 | {currency}{amount3} |\n"
            "| Q4 | {currency}{amount} |"
        ),
        "step1": "A quarterly {metric} table for {company} spanning 4 quarters in the {domain} domain.",
        "step2": "Q1={currency}{amount}, Q2={currency}{amount2}, Q3={currency}{amount3}, Q4={currency}{amount}.",
        "step3": "Q1 and Q4 have identical values, suggesting a cyclical or seasonal pattern. Q2 and Q3 differ.",
        "step4": "Year-over-year comparison data is absent. No breakdown by sub-segment.",
        "step5": "If cyclical, {company} should plan resource allocation around mid-year variation.",
        "confidence_frame": "Medium — pattern visible but only 4 data points; more history needed.",
        "summary": "{company} shows a cyclical {metric} pattern with Q1/Q4 alignment.",
        "findings": "- Q1 and Q4 {metric} are identical at {currency}{amount}\n- Mid-year quarters show variation",
        "analysis": "The data suggests seasonal cyclicality in {company}'s {metric}. The Q2-Q3 divergence warrants investigation into operational or market drivers.",
        "flags": "- Opportunity: Leverage predictable Q1/Q4 for planning\n- Risk: Mid-year volatility may impact forecasting",
        "confidence_answer": "Medium — limited to 4 quarters of data",
    },
    {
        "query": (
            "Review {company}'s {domain} vendor payment records:\n\n"
            "| Vendor | Payment Frequency | Avg Days to Pay |\n"
            "| {company} | Monthly | 28 |\n"
            "| Globex Industries | Monthly | 45 |\n"
            "| Initech Solutions | Quarterly | 15 |\n\n"
            "Identify payment patterns."
        ),
        "step1": "A vendor payment summary table with 3 vendors in the {domain} domain.",
        "step2": "Three vendors with varying payment frequencies and days-to-pay.",
        "step3": "Monthly payers have longer payment cycles (28-45 days); quarterly payer pays fastest (15 days).",
        "step4": "Missing: payment amounts, late payment penalties, vendor satisfaction data.",
        "step5": "The 45-day cycle for Globex may indicate cash flow management or approval bottlenecks.",
        "confidence_frame": "Medium — pattern clear but root causes unknown.",
        "summary": "Payment timing varies significantly across vendors with an inverse frequency-speed pattern.",
        "findings": "- Monthly vendors average 28-45 days to pay\n- Quarterly vendor pays in 15 days\n- Globex Industries is the slowest at 45 days",
        "analysis": "There appears to be an inverse relationship between payment frequency and speed. Monthly obligations may face more approval layers.",
        "flags": "- Risk: Globex's 45-day cycle may strain vendor relationships\n- Opportunity: Standardize payment terms to optimize cash flow",
        "confidence_answer": "Medium — pattern visible but limited sample size",
    },
]

_ANOMALY_TEMPLATES: List[Dict[str, str]] = [
    {
        "query": (
            "Flag any anomalies in this {domain} expense report for {company}:\n\n"
            "| Employee | Department | Amount | Category |\n"
            "| {person} | {dept} | {currency}{amount} | Travel |\n"
            "| {person} | {dept} | {currency}97,500.00 | Office Supplies |\n"
            "| {person} | {dept} | {currency}{amount3} | Training |"
        ),
        "step1": "An expense report with 3 line items from {company} employees in {domain}.",
        "step2": "{currency}97,500.00 in Office Supplies for {dept} is the standout figure.",
        "step3": "Office Supplies typically range {currency}50-5,000. A {currency}97,500.00 entry is 20-2000x normal.",
        "step4": "No receipts, approval chain, or historical comparison provided.",
        "step5": "This could indicate a data entry error, miscategorization, or potential fraud.",
        "confidence_frame": "High — the anomaly magnitude is extreme relative to category norms.",
        "summary": "A {currency}97,500.00 Office Supplies charge is a significant anomaly requiring investigation.",
        "findings": "- {currency}97,500.00 Office Supplies expense is 20-2000x typical range\n- Other expenses appear within normal bounds\n- {dept} department flagged",
        "analysis": "The Office Supplies amount is statistically anomalous. Likely causes: data entry error (extra digit), miscategorization of equipment purchase, or unauthorized expenditure.",
        "flags": "- Risk: Potential fraud or policy violation\n- Action: Immediate review of receipts and approval chain required",
        "confidence_answer": "High — anomaly is clear and significant",
    },
    {
        "query": (
            "Check this {domain} headcount report for {company} for anomalies:\n\n"
            "| Department | Q1 | Q2 | Q3 | Q4 |\n"
            "| {dept} | {qty} | {qty} | 0 | {qty} |\n"
            "| Engineering | 120 | 125 | 130 | 135 |"
        ),
        "step1": "A quarterly headcount report for two departments at {company}.",
        "step2": "{dept} drops to 0 in Q3 then recovers to {qty} in Q4. Engineering grows steadily.",
        "step3": "A department going to zero headcount in one quarter is extremely unusual.",
        "step4": "Missing: reason for Q3 drop, whether it was a reporting error or actual event.",
        "step5": "If real, this suggests a major restructuring event. If error, data quality is compromised.",
        "confidence_frame": "High — zero headcount for an active department is clearly anomalous.",
        "summary": "{dept} department shows a suspicious drop to zero headcount in Q3.",
        "findings": "- {dept} headcount: {qty} -> {qty} -> 0 -> {qty}\n- Q3 zero is inconsistent with Q4 recovery\n- Engineering shows normal steady growth",
        "analysis": "The Q3 zero for {dept} is almost certainly a data error or temporary organizational change (merger, rename). The Q4 recovery to {qty} confirms the department still exists.",
        "flags": "- Risk: Data integrity issue in HR systems\n- Action: Verify Q3 {dept} headcount with HR",
        "confidence_answer": "High — the anomaly pattern is unambiguous",
    },
]

_TREND_TEMPLATES: List[Dict[str, str]] = [
    {
        "query": (
            "Analyze the {metric} trend for {company} over {year}:\n\n"
            "| Month | {metric} |\n"
            "| Jan | {currency}{amount} |\n"
            "| Apr | {currency}{amount2} |\n"
            "| Jul | {currency}{amount3} |\n"
            "| Oct | {currency}{amount} |\n\n"
            "What trend do you observe?"
        ),
        "step1": "A quarterly-sampled {metric} trend for {company} in {year}.",
        "step2": "Jan={currency}{amount}, Apr={currency}{amount2}, Jul={currency}{amount3}, Oct={currency}{amount}.",
        "step3": "The series returns to its starting point by October, suggesting a cyclical rather than linear trend.",
        "step4": "Missing months between samples. Annual comparison data absent.",
        "step5": "If this cycle repeats annually, {company} can optimize for known seasonal patterns.",
        "confidence_frame": "Medium — quarterly sampling may miss intra-quarter dynamics.",
        "summary": "{company}'s {metric} shows a cyclical pattern returning to baseline by year-end.",
        "findings": "- {metric} returns to {currency}{amount} by October (same as January)\n- Mid-year values diverge then converge\n- Pattern suggests seasonality rather than growth",
        "analysis": "The data indicates a seasonal cycle in {company}'s {metric}. Without year-over-year data, it is unclear whether the baseline itself is trending up or down.",
        "flags": "- Opportunity: Seasonal planning can improve resource allocation\n- Risk: Flat year-end return may mask stagnation",
        "confidence_answer": "Medium — limited data points; annual repetition unconfirmed",
    },
    {
        "query": (
            "Review {company}'s {domain} hiring trend:\n\n"
            "2022: 50 hires\n2023: 75 hires\n2024: 110 hires\n2025: 160 hires\n\n"
            "Project the trajectory and assess sustainability."
        ),
        "step1": "A 4-year hiring series for {company} in {domain}.",
        "step2": "Hires: 50 -> 75 -> 110 -> 160. Each year increases by roughly 50% over the prior.",
        "step3": "Exponential growth pattern. Compound annual growth rate ~47%.",
        "step4": "Missing: attrition data, budget constraints, market labor supply.",
        "step5": "At this rate, 2026 would require ~235 hires. Sustainability depends on funding and market.",
        "confidence_frame": "Medium — trend is clear but sustainability is speculative.",
        "summary": "{company} is on an exponential hiring trajectory (~47% CAGR) that may be unsustainable.",
        "findings": "- Hiring has tripled from 50 to 160 over 4 years\n- Growth rate: ~47% compound annual\n- 2026 projection: ~235 hires needed",
        "analysis": "The trend is strongly positive but exponential hiring rarely sustains beyond 3-5 years without proportional revenue growth. {company} should evaluate if infrastructure and management capacity can absorb continued expansion.",
        "flags": "- Risk: Hiring burnout, culture dilution, budget strain\n- Opportunity: Strong growth signal for investors and market position",
        "confidence_answer": "Medium — trend extrapolation carries inherent uncertainty",
    },
]

_COMPARATIVE_TEMPLATES: List[Dict[str, str]] = [
    {
        "query": (
            "Compare these two {domain} vendors for {company}:\n\n"
            "| Criterion | {company} | Globex Industries |\n"
            "| Price | {currency}{amount} | {currency}{amount2} |\n"
            "| Delivery (days) | 14 | 7 |\n"
            "| Quality Rating | 8.5/10 | 9.2/10 |\n"
            "| Support SLA | 24h | 4h |"
        ),
        "step1": "A vendor comparison table with 4 criteria between {company} and Globex Industries.",
        "step2": "{company}: lower price, slower delivery, lower quality, slower support. Globex: higher price, faster across the board.",
        "step3": "Classic cost-vs-quality trade-off. Globex dominates on service; {company} on price.",
        "step4": "Missing: contract flexibility, scalability, track record, total cost of ownership.",
        "step5": "For mission-critical needs, Globex is preferable. For cost-sensitive, {company} wins.",
        "confidence_frame": "High — comparison data is clear; recommendation depends on priorities.",
        "summary": "Globex Industries outperforms on quality and speed; {company} is the budget option.",
        "findings": "- Price advantage: {company} ({currency}{amount} vs {currency}{amount2})\n- Speed advantage: Globex (7 vs 14 days)\n- Quality advantage: Globex (9.2 vs 8.5)\n- Support advantage: Globex (4h vs 24h SLA)",
        "analysis": "Globex wins 3 of 4 criteria. The price premium may be justified by faster delivery and better quality, especially for time-sensitive {domain} operations.",
        "flags": "- Risk: Choosing {company} purely on price may increase hidden costs\n- Opportunity: Negotiate Globex's price down given volume commitment",
        "confidence_answer": "High — data supports clear differentiation",
    },
    {
        "query": (
            "Compare {dept} performance across two divisions at {company}:\n\n"
            "Division A: {metric} = {currency}{amount}, headcount = {qty}\n"
            "Division B: {metric} = {currency}{amount2}, headcount = 85\n\n"
            "Which division is more efficient?"
        ),
        "step1": "A per-division performance comparison for {dept} at {company}.",
        "step2": "Division A: {currency}{amount} with {qty} staff. Division B: {currency}{amount2} with 85 staff.",
        "step3": "Need to compute per-capita {metric} for fair comparison.",
        "step4": "Missing: qualitative factors, overhead allocation, scope differences.",
        "step5": "Per-capita metric will reveal true efficiency; raw totals can be misleading.",
        "confidence_frame": "Medium — quantitative comparison is feasible but qualitative context missing.",
        "summary": "Division efficiency comparison requires per-capita analysis of {metric}.",
        "findings": "- Division A: {currency}{amount} / {qty} staff\n- Division B: {currency}{amount2} / 85 staff\n- Raw totals alone are insufficient for efficiency assessment",
        "analysis": "Dividing {metric} by headcount provides the per-capita figure. The division with higher per-capita {metric} is more efficient in narrow quantitative terms, though qualitative factors may alter the conclusion.",
        "flags": "- Risk: Headcount differences may reflect scope, not inefficiency\n- Opportunity: Cross-pollinate best practices from the more efficient division",
        "confidence_answer": "Medium — per-capita metric is indicative but not definitive",
    },
]

_GAP_TEMPLATES: List[Dict[str, str]] = [
    {
        "query": (
            "Review this {domain} compliance framework for {company} and identify gaps:\n\n"
            "Required controls: Data Encryption, Access Control, Audit Logging, "
            "Incident Response, Business Continuity, Vendor Assessment\n\n"
            "Implemented: Data Encryption, Access Control, Audit Logging"
        ),
        "step1": "A compliance gap analysis for {company} in {domain}.",
        "step2": "6 controls required, 3 implemented. Gap = 3 controls.",
        "step3": "The missing controls (Incident Response, Business Continuity, Vendor Assessment) are operational/strategic, not technical.",
        "step4": "Missing: timeline for remediation, risk ratings per gap, regulatory deadlines.",
        "step5": "Without Incident Response and Business Continuity, {company} is vulnerable to disruptions.",
        "confidence_frame": "High — gaps are clearly identified from the control lists.",
        "summary": "{company} has 3 of 6 required {domain} compliance controls missing.",
        "findings": "- Missing: Incident Response, Business Continuity, Vendor Assessment\n- Implemented: Data Encryption, Access Control, Audit Logging\n- 50% compliance gap",
        "analysis": "The implemented controls are technical (encryption, access, logging). The gaps are all process/governance controls, suggesting {company} prioritized technical implementation over operational preparedness.",
        "flags": "- Risk: Regulatory non-compliance exposure\n- Risk: No incident response capability\n- Opportunity: Bundled implementation of remaining 3 controls",
        "confidence_answer": "High — gap identification is objective",
    },
    {
        "query": (
            "Identify skill gaps in {company}'s {dept} team:\n\n"
            "Required skills: Python, SQL, Cloud Architecture, ML/AI, Data Governance\n"
            "Current team skills: Python, SQL, Cloud Architecture"
        ),
        "step1": "A skills gap analysis for {dept} at {company}.",
        "step2": "5 skills required, 3 present. Missing: ML/AI and Data Governance.",
        "step3": "Both gaps are in advanced/emerging areas, common for growing teams.",
        "step4": "Missing: proficiency levels, training budget, hiring timeline.",
        "step5": "ML/AI and Data Governance gaps could block strategic initiatives.",
        "confidence_frame": "High — skills inventory directly compared to requirements.",
        "summary": "{company}'s {dept} team lacks ML/AI and Data Governance capabilities.",
        "findings": "- 2 of 5 required skills are missing\n- Missing skills are advanced/strategic\n- Core technical skills (Python, SQL, Cloud) are covered",
        "analysis": "The team has a solid technical foundation but lacks the specialized skills needed for advanced analytics and compliance. Addressing these gaps requires either hiring specialists or structured upskilling programs.",
        "flags": "- Risk: Strategic project delays without ML/AI capability\n- Opportunity: Upskilling existing team builds retention and loyalty",
        "confidence_answer": "High — gap is clearly defined",
    },
]

_HOLISTIC_SYNTHESIS_TEMPLATES: List[Dict[str, str]] = [
    {
        "query": (
            "Synthesize these three data points about {company} in the {domain} sector:\n\n"
            "1. Revenue grew {pct}% year-over-year\n"
            "2. Employee satisfaction dropped from 82% to 68%\n"
            "3. Customer complaints increased by {qty} tickets/month\n\n"
            "What is the holistic picture?"
        ),
        "step1": "Three disparate signals about {company}: revenue, satisfaction, complaints.",
        "step2": "Revenue up {pct}%, satisfaction down 14 points, complaints up {qty}/month.",
        "step3": "Growth is coming at the cost of internal and external quality — classic growth-strain pattern.",
        "step4": "Missing: profitability data, root cause of satisfaction drop, customer retention rates.",
        "step5": "If unchecked, declining satisfaction and rising complaints will erode the revenue gains.",
        "confidence_frame": "Medium — synthesis is logical but causal links are inferred.",
        "summary": "{company} is experiencing growth-strain: revenue gains are offset by declining quality indicators.",
        "findings": "- Revenue growth of {pct}% is positive\n- Employee satisfaction fell 14 points (82% -> 68%)\n- Customer complaints rose by {qty}/month\n- Growth appears to be straining operations",
        "analysis": "The three signals paint a picture of unsustainable growth. Revenue increase without corresponding investment in employee experience and customer service typically leads to a 12-18 month correction. {company} should address operational capacity before further expansion.",
        "flags": "- Risk: Employee attrition if satisfaction continues declining\n- Risk: Customer churn from rising complaints\n- Opportunity: Investing in operations now can sustain growth trajectory",
        "confidence_answer": "Medium — causal chain is inferred from correlated signals",
    },
    {
        "query": (
            "Provide a holistic view of {company}'s {domain} transformation:\n\n"
            "- Cloud migration: 75% complete\n"
            "- Legacy system retirement: 40% complete\n"
            "- Staff retraining: 60% complete\n"
            "- Budget utilization: 90% of allocated {currency}{amount}"
        ),
        "step1": "Four transformation metrics for {company}'s {domain} modernization.",
        "step2": "Cloud: 75%, Legacy: 40%, Training: 60%, Budget: 90% used.",
        "step3": "Budget is nearly exhausted (90%) while legacy retirement lags (40%). Training outpaces legacy retirement.",
        "step4": "Missing: timeline, remaining budget, risk of running dual systems.",
        "step5": "With 90% budget used and only 40% legacy retired, a budget overrun is likely.",
        "confidence_frame": "High — the budget-progress mismatch is quantitatively clear.",
        "summary": "{company}'s transformation is at risk of budget overrun with legacy retirement significantly behind.",
        "findings": "- Cloud migration leads at 75%\n- Legacy retirement lags at 40% (bottleneck)\n- 90% of budget consumed with 40-75% progress\n- Staff retraining at 60% is on reasonable pace",
        "analysis": "The transformation has a pacing problem. Cloud migration has outrun legacy retirement, likely creating expensive dual-running costs. The 90% budget utilization with 40% legacy completion signals a funding gap for the remaining work.",
        "flags": "- Risk: Budget overrun likely (10% budget remaining for 60% of legacy work)\n- Risk: Dual-system costs during overlap period\n- Opportunity: Accelerate legacy retirement to reduce dual-running costs",
        "confidence_answer": "High — quantitative misalignment is evident",
    },
]

_RISK_TEMPLATES: List[Dict[str, str]] = [
    {
        "query": (
            "Assess the risks in this {domain} vendor contract for {company}:\n\n"
            "- Single-source dependency for critical component\n"
            "- Contract term: 5 years, no exit clause\n"
            "- Vendor financial rating: BBB-\n"
            "- Annual cost: {currency}{amount}"
        ),
        "step1": "A vendor risk assessment for {company} in {domain}.",
        "step2": "Single source, 5-year lock-in, no exit, BBB- rating, {currency}{amount}/year.",
        "step3": "Multiple risk factors compound: single source + no exit + borderline credit rating.",
        "step4": "Missing: alternative vendor options, business impact analysis, force majeure terms.",
        "step5": "If vendor defaults or underperforms, {company} has no contractual escape for 5 years.",
        "confidence_frame": "High — risk factors are clearly documented and compounding.",
        "summary": "High-risk vendor dependency with compounding factors: single source, lock-in, weak credit.",
        "findings": "- Single-source dependency for critical component\n- 5-year lock-in with no exit clause\n- BBB- credit rating (one notch above junk)\n- {currency}{amount} annual exposure",
        "analysis": "This contract concentrates risk dangerously. A BBB- rated vendor locked into a 5-year no-exit contract for a critical component creates a scenario where vendor financial distress could cascade into {company}'s operations with no mitigation path.",
        "flags": "- Risk (Critical): No diversification for critical component\n- Risk (High): No exit clause limits response options\n- Risk (Medium): BBB- rating indicates financial stress\n- Action: Negotiate exit clause or identify backup vendor immediately",
        "confidence_answer": "High — risk factors are explicit and well-documented",
    },
    {
        "query": (
            "Evaluate data security risks for {company}'s {domain} operations:\n\n"
            "- {qty} employees with admin access\n"
            "- Last security audit: 18 months ago\n"
            "- Encryption: At rest only (not in transit)\n"
            "- MFA: Enabled for 70% of accounts"
        ),
        "step1": "A security risk evaluation for {company} in {domain}.",
        "step2": "{qty} admins, stale audit (18 months), partial encryption, 70% MFA.",
        "step3": "Multiple security gaps compound: broad admin access + incomplete encryption + partial MFA.",
        "step4": "Missing: incident history, data classification, compliance requirements.",
        "step5": "The 30% without MFA are the most likely attack vector.",
        "confidence_frame": "High — security gaps are objectively identifiable.",
        "summary": "{company} has multiple compounding security risks, especially the 30% MFA gap and missing transit encryption.",
        "findings": "- {qty} admin accounts (potential over-provisioning)\n- Security audit 18 months stale\n- No encryption in transit\n- 30% of accounts lack MFA",
        "analysis": "The combination of incomplete MFA and unencrypted data in transit creates a viable attack path. The stale audit means emerging vulnerabilities are undetected. Admin access breadth amplifies the blast radius of any breach.",
        "flags": "- Risk (Critical): Data in transit unencrypted\n- Risk (High): 30% accounts without MFA\n- Risk (Medium): Overdue security audit\n- Action: Immediate MFA rollout and transit encryption",
        "confidence_answer": "High — security gaps are factual and measurable",
    },
]

# Map category -> template list
_CATEGORY_TEMPLATE_MAP = {
    "pattern_recognition": _PATTERN_TEMPLATES,
    "anomaly_detection": _ANOMALY_TEMPLATES,
    "trend_analysis": _TREND_TEMPLATES,
    "comparative_analysis": _COMPARATIVE_TEMPLATES,
    "gap_analysis": _GAP_TEMPLATES,
    "holistic_synthesis": _HOLISTIC_SYNTHESIS_TEMPLATES,
    "risk_assessment": _RISK_TEMPLATES,
}


# ---------------------------------------------------------------------------
# Template expansion
# ---------------------------------------------------------------------------

def _expand_insight_templates(
    templates: List[Dict[str, str]],
    category: str,
    target_count: int,
    rng: random.Random,
) -> List[Dict[str, str]]:
    """Expand insight templates with random variation."""
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
        metric = _pick(_METRICS, rng)
        amount = _rand_amount(rng)
        amount2 = _rand_amount(rng)
        amount3 = _rand_amount(rng)
        qty = str(rng.randint(5, 200))
        year = str(rng.randint(2020, 2026))
        pct = _rand_pct(rng)

        subs = {
            "domain": domain, "doc_type": doc_type, "company": company,
            "person": person, "dept": dept, "product": product,
            "currency": currency, "metric": metric, "amount": amount,
            "amount2": amount2, "amount3": amount3, "qty": qty,
            "year": year, "pct": pct,
        }

        try:
            query = tpl["query"].format(**subs)
            reasoning = _analysis_frame(
                tpl["step1"].format(**subs),
                tpl["step2"].format(**subs),
                tpl["step3"].format(**subs),
                tpl["step4"].format(**subs),
                tpl["step5"].format(**subs),
                tpl["confidence_frame"].format(**subs),
            )
            answer = _insight_answer(
                category,
                tpl["summary"].format(**subs),
                tpl["findings"].format(**subs),
                tpl["analysis"].format(**subs),
                tpl["flags"].format(**subs),
                tpl["confidence_answer"].format(**subs),
            )
        except (KeyError, IndexError):
            idx += 1
            continue

        results.append(format_sft_example(query, reasoning, answer))
        idx += 1

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_insight_examples(count: int = 6000, *, seed: int = 60) -> List[Dict[str, str]]:
    """Generate insight SFT examples across 7 categories.

    Distribution matches _CATEGORY_COUNTS, scaled proportionally to ``count``.

    Args:
        count: Total number of examples to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of SFT examples with ``text`` key.
    """
    rng = random.Random(seed)
    total_base = sum(_CATEGORY_COUNTS.values())

    # Compute per-category counts, ensuring they sum to exactly `count`
    raw_counts = {cat: max(1, int(count * _CATEGORY_COUNTS[cat] / total_base))
                  for cat in INSIGHT_CATEGORIES}
    allocated = sum(raw_counts.values())
    # Distribute remainder to largest categories
    remainder = count - allocated
    sorted_cats = sorted(INSIGHT_CATEGORIES, key=lambda c: _CATEGORY_COUNTS[c], reverse=True)
    i = 0
    while remainder > 0:
        raw_counts[sorted_cats[i % len(sorted_cats)]] += 1
        remainder -= 1
        i += 1

    results: List[Dict[str, str]] = []
    for cat in INSIGHT_CATEGORIES:
        templates = _CATEGORY_TEMPLATE_MAP[cat]
        results.extend(_expand_insight_templates(templates, cat, raw_counts[cat], rng))

    rng.shuffle(results)
    return results[:count]


def generate_phase35_data(output_dir: Path, scale: float = 1.0) -> int:
    """Generate Phase 3.5 insight training data and write to JSONL.

    Args:
        output_dir: Directory to write the JSONL file into.
        scale: Scaling factor (1.0 = 6K examples).

    Returns:
        Number of examples written.
    """
    output_dir = Path(output_dir)
    count = max(1, int(6000 * scale))
    examples = generate_insight_examples(count=count)

    path = output_dir / "phase35_insights.jsonl"
    with JSONLWriter(path) as writer:
        for ex in examples:
            writer.write(ex)

    return len(examples)
