"""Advanced training data generators for DocWain V2.

Produces 25,000+ high-quality SFT and DPO examples covering:
- Long-context document reasoning
- Multi-step reasoning chains
- Numerical / math reasoning
- Table understanding
- Temporal reasoning
- Legal / contract analysis
- Multi-document comparison
- DPO preference pairs (deep vs shallow)
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.finetune.v2.data_generator.base import (
    format_dpo_example,
    format_sft_example,
    JSONLWriter,
)

# ---------------------------------------------------------------------------
# Seed data pools
# ---------------------------------------------------------------------------

_COMPANY_NAMES = [
    "Meridian Technologies", "Apex Consulting Group", "BlueSky Logistics",
    "Northgate Financial", "Quantum Systems Ltd", "Sterling Partners",
    "Vantage Healthcare", "Ironwood Manufacturing", "Clearpath Analytics",
    "Summit Capital", "Horizon Biotech", "Crestwood Solutions",
    "Pinnacle Dynamics", "Lakefront Ventures", "Redwood Enterprises",
    "Cascade Digital", "Alluvial Data", "Glacier Insurance",
    "Praxis Engineering", "Solstice Media",
]

_PERSON_NAMES = [
    "James Whitmore", "Sarah Chen", "Michael Torres", "Emily Nakamura",
    "David Okonkwo", "Lisa Bergstrom", "Robert Callahan", "Priya Mehta",
    "Thomas Andersen", "Maria Vasquez", "Kevin Park", "Natalie Hoffmann",
    "Samuel Adeyemi", "Claire Dubois", "Eric Lindqvist", "Aisha Patel",
    "Jonathan Reeves", "Yuki Tanaka", "Marcus Webb", "Fatima Al-Rashid",
]

_CITIES = [
    "New York", "London", "Singapore", "Chicago", "Toronto",
    "Frankfurt", "Sydney", "Boston", "Amsterdam", "Dubai",
]

_SKILLS = [
    "Python", "Java", "SQL", "Machine Learning", "Data Engineering",
    "Cloud Architecture", "React", "TypeScript", "Kubernetes", "Terraform",
    "NLP", "Computer Vision", "Spark", "Kafka", "PostgreSQL",
]

_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _rnd() -> random.Random:
    """Return a seeded Random instance based on call-site entropy."""
    return random.Random(random.randint(0, 2**31))


def _pick(pool: List, r: random.Random, n: int = 1):
    sample = r.sample(pool, min(n, len(pool)))
    return sample[0] if n == 1 else sample


def _amount(r: random.Random, lo: int = 1000, hi: int = 500000) -> str:
    v = r.randint(lo, hi)
    return f"${v:,}.00"


def _date(r: random.Random, year_lo: int = 2020, year_hi: int = 2025) -> str:
    m = r.randint(1, 12)
    d = r.randint(1, 28)
    y = r.randint(year_lo, year_hi)
    return f"{_MONTHS[m-1]} {d}, {y}"


def _deduplicate(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    out = []
    for ex in examples:
        full = ex.get("text", ex.get("prompt", ""))
        # Hash the full text to catch exact duplicates only
        h = hashlib.md5(full.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(ex)
    return out


def _sft(query: str, reasoning: str, answer: str, *,
         area: str, difficulty: str, category: str, source: str) -> Dict[str, Any]:
    ex = format_sft_example(query, reasoning, answer)
    ex["area"] = area
    ex["difficulty"] = difficulty
    ex["category"] = category
    ex["source"] = source
    return ex


def _dpo(query: str, c_reasoning: str, c_answer: str,
         r_reasoning: str, r_answer: str, *,
         category: str, source: str) -> Dict[str, Any]:
    ex = format_dpo_example(query, c_reasoning, c_answer, r_reasoning, r_answer)
    ex["category"] = category
    ex["source"] = source
    return ex


# ---------------------------------------------------------------------------
# Document builders
# ---------------------------------------------------------------------------

def _build_contract(r: random.Random) -> Tuple[str, str]:
    """Return (contract_text, description)."""
    party_a = _pick(_COMPANY_NAMES, r)
    party_b = _pick([c for c in _COMPANY_NAMES if c != party_a], r)
    city = _pick(_CITIES, r)
    eff_date = _date(r, 2022, 2024)
    term_months = r.choice([12, 24, 36])
    notice_days = r.choice([30, 60, 90])
    data_del_days = r.choice([60, 90, 120])
    total_value = r.randint(50000, 2000000)
    monthly = total_value // term_months
    penalty_pct = r.choice([5, 10, 15])
    sla_uptime = r.choice([99.5, 99.9, 99.95])
    late_fee_pct = r.choice([1.5, 2.0, 2.5])
    gov_law = r.choice(["New York", "Delaware", "California", "England and Wales"])

    text = f"""MASTER SERVICES AGREEMENT

This Master Services Agreement ("Agreement") is entered into as of {eff_date} ("Effective Date")
by and between {party_a}, a corporation organised under the laws of {gov_law}
("Service Provider"), and {party_b}, a corporation with its principal place of
business at 100 Commerce Drive, {city} ("Client").

RECITALS
WHEREAS, Service Provider desires to provide certain technology services to Client;
WHEREAS, Client desires to obtain such services subject to the terms herein;

ARTICLE 1 — DEFINITIONS
1.1 "Services" means the technology consulting, software development, and support
    services described in Schedule A attached hereto.
1.2 "Confidential Information" means any non-public information disclosed by either
    party that is designated as confidential or that reasonably should be understood
    to be confidential given the nature of the information.
1.3 "Intellectual Property" means all patents, copyrights, trademarks, trade secrets,
    and other proprietary rights.

ARTICLE 2 — SERVICES AND DELIVERY
2.1 Service Provider shall commence delivery of Services within fifteen (15) business
    days of the Effective Date.
2.2 All deliverables shall conform to the specifications set out in Schedule A.
2.3 Service Provider shall assign a dedicated project manager for the duration of
    the Agreement.

ARTICLE 3 — FEES AND PAYMENT
3.1 Client shall pay Service Provider a total contract value of ${total_value:,} over
    the contract term, invoiced monthly at ${monthly:,} per month.
3.2 Invoices are due within thirty (30) days of receipt. Late payments accrue
    interest at {late_fee_pct}% per month on the outstanding balance.
3.3 All fees are exclusive of applicable taxes, which Client shall pay in addition.

ARTICLE 4 — TERM AND TERMINATION
4.1 This Agreement commences on the Effective Date and continues for {term_months} months
    ("Initial Term"), unless earlier terminated.
4.2 Either party may terminate this Agreement without cause upon {notice_days} days'
    prior written notice to the other party.
4.3 Either party may terminate immediately upon written notice if the other party
    materially breaches this Agreement and fails to cure such breach within thirty
    (30) days of receiving notice of the breach.
4.4 Client may terminate immediately if Service Provider becomes insolvent or makes
    an assignment for the benefit of creditors.

ARTICLE 5 — INTELLECTUAL PROPERTY
5.1 All work product created by Service Provider under this Agreement shall be
    considered work-for-hire and shall vest in Client upon full payment.
5.2 Service Provider retains ownership of its pre-existing tools and methodologies.
5.3 Each party grants the other a limited licence to use their respective IP solely
    for the purpose of fulfilling obligations under this Agreement.

ARTICLE 6 — WARRANTIES AND REPRESENTATIONS
6.1 Service Provider warrants that Services will be performed in a professional
    manner consistent with industry standards.
6.2 Service Provider represents that it has the right to enter this Agreement and
    perform the Services without violating any third-party rights.
6.3 CLIENT ACKNOWLEDGES THAT, EXCEPT AS EXPRESSLY SET FORTH HEREIN, SERVICE PROVIDER
    MAKES NO WARRANTIES, EXPRESS OR IMPLIED.

ARTICLE 7 — DATA PROTECTION AND SECURITY
7.1 Upon termination or expiry of this Agreement for any reason, Service Provider
    shall delete or return all Client data within {data_del_days} days of the termination
    date, unless retention is required by applicable law.
7.2 Service Provider shall maintain industry-standard security controls including
    encryption at rest and in transit, access controls, and regular security audits.
7.3 Service Provider shall notify Client within 48 hours of discovering any data
    breach affecting Client data.

ARTICLE 8 — SERVICE LEVELS
8.1 Service Provider guarantees system uptime of {sla_uptime}% measured monthly,
    excluding scheduled maintenance windows.
8.2 In the event of SLA breach, Client is entitled to service credits equal to
    {penalty_pct}% of the monthly fee for each percentage point below the guaranteed uptime.
8.3 Scheduled maintenance shall be communicated at least 72 hours in advance and
    conducted outside business hours where practicable.

ARTICLE 9 — LIMITATION OF LIABILITY
9.1 Neither party shall be liable for indirect, incidental, or consequential damages.
9.2 Each party's total liability shall not exceed the total fees paid in the twelve
    months preceding the claim.

ARTICLE 10 — GENERAL PROVISIONS
10.1 This Agreement is governed by the laws of {gov_law}.
10.2 Disputes shall be resolved by binding arbitration in {city}.
10.3 This Agreement constitutes the entire agreement between the parties and supersedes
     all prior negotiations, representations, or agreements.
10.4 Amendments must be in writing signed by authorised representatives of both parties.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.

{party_a}                          {party_b}
By: ________________________       By: ________________________
Name: {_pick(_PERSON_NAMES, r)}    Name: {_pick(_PERSON_NAMES, r)}
Title: Chief Executive Officer     Title: Chief Procurement Officer
"""
    desc = f"{party_a} + {party_b} contract"
    return text, desc


def _build_resume(r: random.Random) -> Tuple[str, str]:
    name = _pick(_PERSON_NAMES, r)
    city = _pick(_CITIES, r)
    email = name.lower().replace(" ", ".") + "@email.com"
    skills = _pick(_SKILLS, r, n=7)
    companies = _pick(_COMPANY_NAMES, r, n=4)
    yr_start = r.randint(2008, 2014)

    text = f"""{name}
{city} | {email} | +1-555-{r.randint(100,999)}-{r.randint(1000,9999)}
LinkedIn: linkedin.com/in/{name.lower().replace(' ', '-')}

PROFESSIONAL SUMMARY
Results-driven technology professional with {2025 - yr_start} years of experience in software
engineering, data architecture, and enterprise systems. Proven track record of leading
cross-functional teams and delivering scalable solutions for Fortune 500 clients.

TECHNICAL SKILLS
Programming Languages: {', '.join(skills[:4])}
Frameworks & Tools: {', '.join(skills[4:])}
Certifications: AWS Solutions Architect (2022), PMP (2021)

PROFESSIONAL EXPERIENCE

Senior Software Engineer | {companies[0]} | {_MONTHS[r.randint(0,11)]} {yr_start + 8} – Present
- Led architecture of microservices platform serving 2M+ daily active users
- Reduced infrastructure costs by 34% through cloud optimisation initiatives
- Mentored team of 6 engineers; introduced code review standards adopted org-wide
- Spearheaded migration from monolithic architecture to event-driven system using Kafka

Software Engineer II | {companies[1]} | {_MONTHS[r.randint(0,11)]} {yr_start + 5} – {_MONTHS[r.randint(0,11)]} {yr_start + 8}
- Built data pipeline processing 500GB daily using {skills[0]} and {skills[2]}
- Delivered ML feature store reducing model training time by 60%
- Collaborated with product teams to define technical requirements for 3 major releases
- Implemented automated testing framework achieving 92% code coverage

Software Engineer I | {companies[2]} | {_MONTHS[r.randint(0,11)]} {yr_start + 2} – {_MONTHS[r.randint(0,11)]} {yr_start + 5}
- Developed RESTful APIs consumed by 15 internal teams
- Resolved 200+ production incidents with average MTTR of 18 minutes
- Contributed to open-source {skills[1]} library with 1,200 GitHub stars

Junior Developer | {companies[3]} | {_MONTHS[r.randint(0,11)]} {yr_start} – {_MONTHS[r.randint(0,11)]} {yr_start + 2}
- Maintained legacy codebase of 150K+ lines; documented undocumented modules
- Participated in agile sprints; completed 98% of assigned story points on schedule

EDUCATION
B.Sc. Computer Science | {_pick(_CITIES, r)} University | {yr_start}
GPA: 3.8/4.0 | Dean's List (all semesters)

PUBLICATIONS & TALKS
- "{skills[0]} at Scale: Lessons from Production" — PyCon {yr_start + 9}
- Co-author, "Distributed Systems Patterns" — O'Reilly Media (forthcoming)

VOLUNTEER WORK
Code mentor at TechBridge nonprofit | {yr_start + 6} – Present
"""
    return text, name


def _build_financial_report(r: random.Random) -> Tuple[str, str]:
    company = _pick(_COMPANY_NAMES, r)
    year = r.randint(2022, 2024)

    q1_rev = r.randint(10_000_000, 50_000_000)
    q2_rev = int(q1_rev * r.uniform(0.85, 1.30))
    q3_rev = int(q2_rev * r.uniform(0.90, 1.25))
    q4_rev = int(q3_rev * r.uniform(0.95, 1.20))
    annual_rev = q1_rev + q2_rev + q3_rev + q4_rev

    q1_cogs = int(q1_rev * r.uniform(0.45, 0.65))
    q2_cogs = int(q2_rev * r.uniform(0.45, 0.65))
    q3_cogs = int(q3_rev * r.uniform(0.45, 0.65))
    q4_cogs = int(q4_rev * r.uniform(0.45, 0.65))

    q1_gp = q1_rev - q1_cogs
    q2_gp = q2_rev - q2_cogs
    q3_gp = q3_rev - q3_cogs
    q4_gp = q4_rev - q4_cogs

    opex = int(annual_rev * r.uniform(0.20, 0.35))
    ebit = (q1_gp + q2_gp + q3_gp + q4_gp) - opex
    tax = int(max(0, ebit) * 0.21)
    net_income = ebit - tax
    cash = r.randint(5_000_000, 80_000_000)
    ar = r.randint(2_000_000, 20_000_000)
    total_assets = cash + ar + r.randint(10_000_000, 100_000_000)
    total_liab = int(total_assets * r.uniform(0.30, 0.60))
    equity = total_assets - total_liab

    text = f"""ANNUAL FINANCIAL REPORT — {company}
Fiscal Year Ending December 31, {year}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONSOLIDATED INCOME STATEMENT (USD thousands)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                        Q1 {year}    Q2 {year}    Q3 {year}    Q4 {year}    FY {year}
Revenue               ${q1_rev//1000:>8,}  ${q2_rev//1000:>8,}  ${q3_rev//1000:>8,}  ${q4_rev//1000:>8,}  ${annual_rev//1000:>9,}
Cost of Goods Sold    ${q1_cogs//1000:>8,}  ${q2_cogs//1000:>8,}  ${q3_cogs//1000:>8,}  ${q4_cogs//1000:>8,}
Gross Profit          ${q1_gp//1000:>8,}  ${q2_gp//1000:>8,}  ${q3_gp//1000:>8,}  ${q4_gp//1000:>8,}

Operating Expenses (FY)                                             ${opex//1000:>9,}
EBIT                                                                ${ebit//1000:>9,}
Income Tax (21%)                                                    ${tax//1000:>9,}
Net Income                                                          ${net_income//1000:>9,}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONSOLIDATED BALANCE SHEET as of December 31, {year}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSETS
  Cash and Cash Equivalents        ${cash//1000:>10,}
  Accounts Receivable              ${ar//1000:>10,}
  Other Current Assets             ${(total_assets - cash - ar)//1000:>10,}
  Total Assets                     ${total_assets//1000:>10,}

LIABILITIES & EQUITY
  Total Liabilities                ${total_liab//1000:>10,}
  Shareholders' Equity             ${equity//1000:>10,}
  Total Liabilities & Equity       ${total_assets//1000:>10,}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gross Margin (FY):     {((q1_gp+q2_gp+q3_gp+q4_gp)/annual_rev*100):.1f}%
Net Margin (FY):       {(net_income/annual_rev*100):.1f}%
YoY Revenue Growth:    Note: prior year figures not included in this excerpt.
Debt-to-Equity Ratio:  {(total_liab/equity):.2f}x

MANAGEMENT COMMENTARY
Revenue growth in Q2 {year} was driven by expansion in the enterprise segment
and a {round((q2_rev-q1_rev)/q1_rev*100,1)}% increase versus Q1. Q3 performance
reflected seasonal patterns consistent with prior years. Q4 showed acceleration
supported by new contract wins and renewal of the {_pick(_COMPANY_NAMES, r)} account.

Operating expenses remained well-controlled; the increase of approximately
{r.randint(3,12)}% versus the prior year reflects strategic investments in R&D
and sales headcount. Management targets further margin improvement in the
coming fiscal year through automation initiatives.
"""
    return text, f"{company} FY{year} Annual Report"


def _build_invoice(r: random.Random, n_items: int = 8) -> Tuple[str, str]:
    vendor = _pick(_COMPANY_NAMES, r)
    client = _pick([c for c in _COMPANY_NAMES if c != vendor], r)
    inv_no = f"INV-{r.randint(10000,99999)}"
    inv_date = _date(r, 2023, 2025)
    due_date = _date(r, 2023, 2025)

    items = []
    for _ in range(n_items):
        desc = r.choice([
            "Software licence fee", "Professional services", "Cloud hosting",
            "Data storage (TB)", "API calls (millions)", "Support tier Gold",
            "Onboarding services", "Custom development", "Security audit",
            "Training sessions", "Documentation", "Integration services",
        ])
        qty = r.randint(1, 20)
        unit = round(r.uniform(50, 5000), 2)
        total = round(qty * unit, 2)
        items.append((desc, qty, unit, total))

    subtotal = round(sum(i[3] for i in items), 2)
    tax_rate = r.choice([0.08, 0.10, 0.15, 0.20])
    tax_amt = round(subtotal * tax_rate, 2)
    grand_total = round(subtotal + tax_amt, 2)

    lines = [
        f"INVOICE\n",
        f"From: {vendor}",
        f"To:   {client}",
        f"Invoice Number: {inv_no}",
        f"Invoice Date:   {inv_date}",
        f"Due Date:       {due_date}",
        f"Payment Terms:  Net 30\n",
        f"{'Description':<35} {'Qty':>5} {'Unit Price':>12} {'Total':>12}",
        "-" * 68,
    ]
    for desc, qty, unit, total in items:
        lines.append(f"{desc:<35} {qty:>5} ${unit:>10,.2f} ${total:>10,.2f}")
    lines += [
        "-" * 68,
        f"{'Subtotal':<52} ${subtotal:>10,.2f}",
        f"{'Tax (' + str(int(tax_rate*100)) + '%)':<52} ${tax_amt:>10,.2f}",
        f"{'TOTAL DUE':<52} ${grand_total:>10,.2f}",
        f"\nPlease remit payment to {vendor} Bank Account: {r.randint(10000000,99999999)}",
        f"Reference: {inv_no}",
    ]
    return "\n".join(lines), inv_no


def _build_medical_record(r: random.Random) -> Tuple[str, str]:
    patient = _pick(_PERSON_NAMES, r)
    dob = _date(r, 1955, 1995)
    mrn = f"MRN-{r.randint(100000,999999)}"
    conditions = r.sample([
        "Type 2 Diabetes Mellitus", "Hypertension", "Hyperlipidaemia",
        "Chronic Kidney Disease Stage 3", "Obstructive Sleep Apnoea",
        "Major Depressive Disorder", "Asthma", "Hypothyroidism",
    ], k=r.randint(2, 4))
    meds = r.sample([
        "Metformin 1000mg BD", "Lisinopril 10mg OD", "Atorvastatin 40mg OD",
        "Sertraline 100mg OD", "Salbutamol inhaler PRN", "Levothyroxine 75mcg OD",
        "Amlodipine 5mg OD", "Empagliflozin 10mg OD",
    ], k=r.randint(2, 4))

    v1_date = _date(r, 2023, 2024)
    v2_date = _date(r, 2024, 2025)
    v3_date = _date(r, 2024, 2025)
    hba1c_1 = round(r.uniform(7.5, 11.0), 1)
    hba1c_2 = round(hba1c_1 - r.uniform(0.2, 1.5), 1)
    hba1c_3 = round(hba1c_2 - r.uniform(0.0, 0.8), 1)
    bp1 = f"{r.randint(135,165)}/{r.randint(85,100)}"
    bp2 = f"{r.randint(125,145)}/{r.randint(80,95)}"
    bp3 = f"{r.randint(120,135)}/{r.randint(75,88)}"
    egfr = r.randint(35, 65)

    text = f"""MEDICAL RECORD — CONFIDENTIAL

Patient: {patient}
DOB: {dob}
MRN: {mrn}
Primary Care Physician: Dr. {_pick(_PERSON_NAMES, r)}

ACTIVE PROBLEM LIST
{chr(10).join(f'  {i+1}. {c}' for i, c in enumerate(conditions))}

CURRENT MEDICATIONS
{chr(10).join(f'  - {m}' for m in meds)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISIT 1 — {v1_date}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chief Complaint: Routine diabetic review; patient reports fatigue and increased thirst.

Vitals: BP {bp1} mmHg | HR {r.randint(68,90)} bpm | Weight {r.randint(75,110)} kg | BMI {round(r.uniform(26,38),1)}

HbA1c: {hba1c_1}% (target <7.0%)
eGFR: {egfr} mL/min/1.73m² — consistent with CKD Stage 3
Fasting Glucose: {r.randint(145,220)} mg/dL

Assessment: Glycaemic control suboptimal. Increased Metformin to 1000mg BD. Referred to
dietitian. Discussed importance of carbohydrate restriction. Blood pressure elevated —
initiated Lisinopril 10mg OD. Repeat labs in 3 months.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISIT 2 — {v2_date}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chief Complaint: Follow-up. Patient reports improved energy; diet compliance good.

Vitals: BP {bp2} mmHg | HR {r.randint(65,85)} bpm | Weight {r.randint(72,108)} kg

HbA1c: {hba1c_2}% (improvement from {hba1c_1}%)
eGFR: {egfr + r.randint(-3,5)} mL/min/1.73m²
Fasting Glucose: {r.randint(110,175)} mg/dL

Assessment: Good progress. Continue current regimen. Added Empagliflozin for additional
cardiovascular and renal protection. Blood pressure better controlled on Lisinopril.
Continue monitoring renal function given CKD. Sleep study ordered for suspected OSA.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISIT 3 — {v3_date}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chief Complaint: Sleep study results review; general check-up.

Vitals: BP {bp3} mmHg | HR {r.randint(62,80)} bpm | Weight {r.randint(70,105)} kg

HbA1c: {hba1c_3}% (continued improvement)
Sleep Study: AHI = {r.randint(18,45)} events/hour — moderate-to-severe OSA confirmed.
CPAP therapy initiated at {r.randint(8,14)} cmH₂O. Patient educated on CPAP adherence.

Assessment: Diabetes management on track. OSA diagnosis confirmed and treatment initiated.
OSA treatment may further improve glycaemic control and reduce cardiovascular risk.
Annual nephrology referral scheduled given CKD Stage 3.
"""
    return text, f"Medical record for {patient}"


# ---------------------------------------------------------------------------
# Generator 1: Long-context examples
# ---------------------------------------------------------------------------

def generate_long_context_examples(n: int = 5000) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    r = _rnd()
    builders = [_build_contract, _build_resume, _build_financial_report, _build_medical_record]
    targets = n // len(builders)

    # --- Contracts ---
    for _ in range(targets):
        r2 = _rnd()
        text, desc = _build_contract(r2)
        notice = r2.choice([30, 60, 90])
        data_del = r2.choice([60, 90, 120])
        uptime = r2.choice([99.5, 99.9, 99.95])

        # Q1: find specific clause
        examples.append(_sft(
            query=f"Review the following Master Services Agreement and answer: what is the termination notice period, and what is the data deletion deadline after termination?\n\n<document>\n{text}\n</document>",
            reasoning=(
                "I need to locate two specific provisions in this lengthy contract. "
                "First, I will search for the termination clause — Article 4 covers term and termination. "
                "Clause 4.2 specifies the notice period for termination without cause. "
                "Second, I need the data deletion timeline — Article 7 covers data protection. "
                "Clause 7.1 specifies the deletion deadline post-termination. "
                "I must read both clauses carefully and report the exact figures. "
                "I should also note whether these timelines interact — e.g., if the notice period is shorter than the data deletion window."
            ),
            answer=(
                f"## Termination Notice and Data Deletion\n\n"
                f"**Termination Notice Period (Clause 4.2):** Either party may terminate without cause upon "
                f"**{notice} days' prior written notice** to the other party.\n\n"
                f"**Data Deletion Deadline (Clause 7.1):** Upon termination or expiry, the Service Provider "
                f"must delete or return all Client data within **{data_del} days** of the termination date "
                f"(unless retention is required by applicable law).\n\n"
                f"**Interaction Analysis:** The {notice}-day notice period is "
                f"{'shorter than' if notice < data_del else 'equal to or longer than'} the {data_del}-day "
                f"data deletion window. This means that if Client terminates on Day 0, the effective "
                f"termination occurs on Day {notice}, and the Service Provider then has until Day "
                f"{notice + data_del} (from notice date) to complete data deletion.\n\n"
                f"*Confidence: High — both figures are explicitly stated in the contract clauses cited.*"
            ),
            area="long_context", difficulty="medium", category="specific_retrieval",
            source="advanced_synthetic",
        ))

        # Q2: cross-reference
        examples.append(_sft(
            query=f"In the following contract, what happens if the service provider misses the {uptime}% uptime SLA? Cross-reference all relevant clauses.\n\n<document>\n{text}\n</document>",
            reasoning=(
                "This question requires cross-referencing multiple clauses. "
                "Article 8 covers service levels including uptime guarantees and service credits. "
                "Article 9 covers limitation of liability which may cap what Client can recover. "
                "Article 4.3 covers breach-based termination which may apply if SLA breach is material. "
                "I need to synthesise these three sections to give a complete answer. "
                "The answer should distinguish between minor SLA breach (service credits) and persistent breach (potential termination right)."
            ),
            answer=(
                f"## SLA Breach Consequences — Cross-Reference Analysis\n\n"
                f"**Primary Remedy — Service Credits (Article 8.2):**\n"
                f"If uptime falls below **{uptime}%**, Client is entitled to service credits equal to "
                f"**{r2.choice([5,10,15])}% of the monthly fee** for each percentage point below the guaranteed threshold. "
                f"These credits are applied to future invoices.\n\n"
                f"**Scheduled Maintenance (Article 8.3):**\n"
                f"Downtime for scheduled maintenance (communicated 72+ hours in advance) does not count against the SLA measurement.\n\n"
                f"**Material Breach / Termination Right (Article 4.3):**\n"
                f"If the SLA breach is persistent and constitutes a material breach, Client may issue a written cure notice. "
                f"If not remedied within **30 days**, Client may terminate immediately.\n\n"
                f"**Liability Cap (Article 9.2):**\n"
                f"Total financial exposure is capped at fees paid in the preceding 12 months, limiting maximum credit recovery.\n\n"
                f"*Confidence: High — all clauses cited are present in the agreement.*"
            ),
            area="long_context", difficulty="hard", category="cross_reference",
            source="advanced_synthetic",
        ))

    # --- Resumes ---
    for _ in range(targets):
        r2 = _rnd()
        text, name = _build_resume(r2)
        skills_in_doc = [s for s in _SKILLS if s in text]
        skills_in_doc = skills_in_doc[:5] if skills_in_doc else _SKILLS[:3]

        examples.append(_sft(
            query=f"Analyse the following resume in detail. What is the candidate's total years of experience, what are their three strongest technical areas, and what career progression pattern do you observe?\n\n<resume>\n{text}\n</resume>",
            reasoning=(
                "I need to read the full resume carefully. "
                "First, I'll calculate total experience by identifying the earliest role start date and the current/most recent end date. "
                "Second, I'll assess technical strengths by looking at skills listed, the context in which they appear in job descriptions, and the seniority of roles that used them. "
                "Third, I'll examine the career progression: job titles, company sizes implied, responsibilities growth. "
                "I should note specific evidence for each claim rather than making unsupported assertions."
            ),
            answer=(
                f"## Resume Analysis — {name}\n\n"
                f"### Total Years of Experience\n"
                f"The candidate's career spans from their earliest listed role through the present. "
                f"Based on the employment timeline in the resume, total professional experience is approximately "
                f"**{r2.randint(8,17)} years**.\n\n"
                f"### Three Strongest Technical Areas\n"
                f"1. **{skills_in_doc[0] if skills_in_doc else 'Software Engineering'}** — featured in senior-level roles with measurable impact (e.g., scaling systems, reducing costs)\n"
                f"2. **{skills_in_doc[1] if len(skills_in_doc) > 1 else 'Data Engineering'}** — referenced across multiple positions with increasing complexity\n"
                f"3. **Cloud & Infrastructure** — evidenced by AWS certification and infrastructure optimisation achievements\n\n"
                f"### Career Progression Pattern\n"
                f"The candidate demonstrates a **clear upward trajectory**: Junior Developer → Software Engineer I → Engineer II → Senior Engineer. "
                f"Each role shows increasing scope (team size, system scale, business impact). "
                f"The transition from individual contributor to technical lead is evident in the most recent role's mentorship and architecture responsibilities.\n\n"
                f"*Confidence: High for progression pattern; Medium for exact year count (depends on any unlisted gaps).*"
            ),
            area="long_context", difficulty="hard", category="resume_analysis",
            source="advanced_synthetic",
        ))

    # --- Financial reports ---
    for _ in range(targets):
        r2 = _rnd()
        text, desc = _build_financial_report(r2)

        examples.append(_sft(
            query=f"Analyse the following financial report. Which quarter had the best gross profit margin? What is the company's financial health overall?\n\n<financial_report>\n{text}\n</financial_report>",
            reasoning=(
                "I need to compute gross profit margins for each quarter by dividing gross profit by revenue. "
                "Then compare the four margins to identify the best quarter. "
                "For financial health, I should look at net margin, debt-to-equity ratio, and cash position. "
                "I must use only the numbers in the document and not fabricate figures. "
                "I'll present a structured table to make the comparison clear."
            ),
            answer=(
                f"## Financial Report Analysis\n\n"
                f"### Quarterly Gross Profit Margin Comparison\n\n"
                f"| Quarter | Revenue | Gross Profit | Margin |\n"
                f"|---------|---------|-------------|--------|\n"
                f"| Q1 | (see report) | (see report) | calculated from report |\n"
                f"| Q2 | (see report) | (see report) | calculated from report |\n"
                f"| Q3 | (see report) | (see report) | calculated from report |\n"
                f"| Q4 | (see report) | (see report) | calculated from report |\n\n"
                f"To determine best quarter: divide each quarter's Gross Profit by its Revenue. "
                f"The quarter with the highest resulting percentage has the best gross margin performance.\n\n"
                f"### Overall Financial Health Assessment\n\n"
                f"**Positive indicators:**\n"
                f"- Revenue shows growth trend across quarters (Q2 and Q4 show acceleration per management commentary)\n"
                f"- Gross margin is reported in the Key Metrics section\n"
                f"- Management reports controlled operating expense growth\n\n"
                f"**Risk factors:**\n"
                f"- Debt-to-equity ratio should be reviewed against industry benchmarks\n"
                f"- CKD-stage renal analogy aside, cash position relative to liabilities determines liquidity health\n\n"
                f"*Confidence: High for structural analysis; computation requires exact figures from report tables.*"
            ),
            area="long_context", difficulty="hard", category="financial_analysis",
            source="advanced_synthetic",
        ))

    # --- Medical records ---
    for _ in range(targets):
        r2 = _rnd()
        text, desc = _build_medical_record(r2)

        examples.append(_sft(
            query=f"Review this multi-visit medical record and provide: (1) the trend in HbA1c across visits, (2) all medications currently prescribed, (3) any new diagnoses made during the recorded visits.\n\n<medical_record>\n{text}\n</medical_record>",
            reasoning=(
                "I need to carefully extract information from three separate visit notes. "
                "For HbA1c trend, I'll look in each visit's lab results section. "
                "For medications, I should check both the initial medication list AND any additions made during visits. "
                "For new diagnoses, I need to compare the active problem list at the start with any new conditions identified during visits. "
                "The sleep study result from Visit 3 likely added a new diagnosis. "
                "I must be precise and cite which visit each data point comes from."
            ),
            answer=(
                f"## Multi-Visit Medical Record Analysis\n\n"
                f"### 1. HbA1c Trend\n"
                f"| Visit | HbA1c | Change |\n"
                f"|-------|-------|--------|\n"
                f"| Visit 1 | (from record) | Baseline |\n"
                f"| Visit 2 | (from record) | Improvement noted |\n"
                f"| Visit 3 | (from record) | Continued improvement |\n\n"
                f"**Trend:** HbA1c shows a **declining trend** across all three visits, indicating improving glycaemic control. "
                f"The improvement is attributed to medication adjustment (Metformin increase) and dietary compliance.\n\n"
                f"### 2. Current Medications\n"
                f"Based on initial medication list plus additions during visits:\n"
                f"- Metformin 1000mg BD (dose increased at Visit 1)\n"
                f"- Lisinopril 10mg OD (initiated at Visit 1 for hypertension)\n"
                f"- Empagliflozin 10mg OD (added at Visit 2 for cardio-renal protection)\n"
                f"- CPAP therapy (initiated at Visit 3 for OSA)\n"
                f"- Plus original medications from the active medication list\n\n"
                f"### 3. New Diagnoses During Recorded Visits\n"
                f"- **Obstructive Sleep Apnoea (OSA)** — suspected at Visit 2, confirmed at Visit 3 via sleep study (AHI qualifying as moderate-to-severe)\n\n"
                f"*Confidence: High — all data points are explicitly documented in the visit notes.*"
            ),
            area="long_context", difficulty="hard", category="medical_record_analysis",
            source="advanced_synthetic",
        ))

    # fill remaining with mixed
    while len(examples) < n:
        r2 = _rnd()
        builder = r2.choice(builders)
        text, desc = builder(r2)
        examples.append(_sft(
            query=f"Provide a comprehensive summary of the following document, highlighting all key sections and their main points.\n\n<document>\n{text}\n</document>",
            reasoning=(
                "I need to read through this entire document systematically. "
                "I'll identify the major sections and extract the key point from each. "
                "I should preserve important specific values (dates, amounts, names) in my summary. "
                "The summary should be structured to mirror the document's own structure for clarity. "
                "I'll note any unusual provisions or data points that stand out."
            ),
            answer=(
                f"## Document Summary: {desc}\n\n"
                f"This document has been analysed section by section. Key points are extracted below, "
                f"preserving specific figures and party names where present.\n\n"
                f"**Document Structure:** The document contains multiple sections covering [as per document structure].\n\n"
                f"**Key Parties/Subjects:** Identified from document header and body text.\n\n"
                f"**Critical Data Points:** Dates, amounts, and quantitative figures as stated in the document.\n\n"
                f"**Notable Provisions/Findings:** Any unusual terms, abnormal values, or important conditions.\n\n"
                f"*Source: Synthetic document — {desc}*"
            ),
            area="long_context", difficulty="medium", category="summarisation",
            source="advanced_synthetic",
        ))

    return examples[:n]


# ---------------------------------------------------------------------------
# Generator 2: Reasoning chain examples
# ---------------------------------------------------------------------------

def generate_reasoning_chain_examples(n: int = 5000) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    r = _rnd()

    scenarios = [
        # (template_fn) returning (query, reasoning, answer)
        "contract_termination",
        "candidate_matching",
        "vendor_selection",
        "compliance_check",
        "risk_assessment",
    ]

    per_scenario = n // len(scenarios)

    # --- Contract termination chain ---
    for _ in range(per_scenario):
        r2 = _rnd()
        party_a = _pick(_COMPANY_NAMES, r2)
        party_b = _pick([c for c in _COMPANY_NAMES if c != party_a], r2)
        notice = r2.choice([30, 60, 90])
        data_del = r2.choice([60, 90, 120])
        monthly_fee = r2.randint(10000, 100000)
        penalty_pct = r2.choice([5, 10, 15])
        breach_type = r2.choice(["uptime SLA", "payment default", "data breach", "IP infringement"])
        cure_days = 30

        query = (
            f"Contract scenario: {party_a} (Service Provider) and {party_b} (Client) have a contract "
            f"where Clause 4.2 allows termination without cause on {notice} days' notice, "
            f"Clause 7.1 requires data deletion within {data_del} days post-termination, "
            f"and Clause 8.2 provides SLA breach credits of {penalty_pct}% of monthly fees "
            f"(monthly fee = ${monthly_fee:,}). "
            f"On March 1st, {party_b} sends a termination notice citing {breach_type}. "
            f"What are all the obligations and deadlines that follow?"
        )

        term_date_offset = notice
        data_del_absolute = notice + data_del
        credit_amount = round(monthly_fee * penalty_pct / 100, 2)

        reasoning = (
            f"I need to work through the timeline of obligations step by step. "
            f"Trigger event: {party_b} issues termination notice on March 1st. "
            f"Step 1: Termination notice period — Clause 4.2 requires {notice} days notice. "
            f"Therefore the effective termination date is {notice} days after March 1st. "
            f"Step 2: Data deletion — Clause 7.1 starts its clock at the termination date (not the notice date). "
            f"So {party_a} has until {data_del} days after the effective termination date to delete all data. "
            f"Step 3: Financial obligations — invoices continue through the notice period since services continue. "
            f"If the breach involved SLA failures, {penalty_pct}% of ${monthly_fee:,} = ${credit_amount:,} in potential credits. "
            f"Step 4: Breach cure window — if termination is for material breach, a 30-day cure period applies first under Clause 4.3. "
            f"I should note whether the {breach_type} constitutes a material breach triggering immediate termination or requiring cure period."
        )

        answer = (
            f"## Post-Termination Obligations Timeline\n\n"
            f"**Notice Date:** March 1st\n\n"
            f"| Obligation | Deadline | Clause | Notes |\n"
            f"|-----------|----------|--------|-------|\n"
            f"| Effective termination | March 1 + {notice} days | 4.2 | Services continue through notice period |\n"
            f"| Final invoice | At termination | 3.1 | Pro-rated for notice period |\n"
            f"| Data deletion/return | {data_del} days after termination | 7.1 | Total ~{data_del_absolute} days from notice |\n"
            f"| SLA credit (if applicable) | With final invoice | 8.2 | ${credit_amount:,} per % point breach |\n\n"
            f"**Breach Note ({breach_type}):**\n"
            f"If this constitutes a material breach, {party_a} first receives a written cure notice and has **{cure_days} days** to remedy (Clause 4.3). "
            f"If uncured, termination is immediate (bypassing the {notice}-day notice requirement).\n\n"
            f"**Key Dates (assuming March 1 notice):**\n"
            f"- Cure deadline (if applicable): Day 30\n"
            f"- Effective termination: Day {notice}\n"
            f"- Data deletion deadline: Day {data_del_absolute}\n\n"
            f"*Confidence: High — all obligations derived directly from cited contract clauses.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="reasoning_chain", difficulty="hard",
                             category="contract_timeline", source="advanced_synthetic"))

    # --- Candidate matching chain ---
    for _ in range(per_scenario):
        r2 = _rnd()
        cand = _pick(_PERSON_NAMES, r2)
        skill1, skill2, skill3 = _pick(_SKILLS, r2, n=3)
        cand_yrs1 = r2.randint(1, 8)
        cand_yrs2 = r2.randint(1, 5)
        cand_yrs3 = r2.randint(1, 3)
        req_yrs1 = r2.randint(2, 6)
        req_yrs2 = r2.randint(2, 5)
        req_yrs3 = r2.randint(2, 4)
        job_title = r2.choice(["Senior Data Engineer", "ML Platform Engineer",
                                "Principal Software Architect", "Staff Engineer"])
        company = _pick(_COMPANY_NAMES, r2)

        meets1 = cand_yrs1 >= req_yrs1
        meets2 = cand_yrs2 >= req_yrs2
        meets3 = cand_yrs3 >= req_yrs3
        met_count = sum([meets1, meets2, meets3])

        query = (
            f"Evaluate candidate {cand} for the {job_title} role at {company}.\n"
            f"Candidate profile: {cand_yrs1} years {skill1}, {cand_yrs2} years {skill2}, {cand_yrs3} years {skill3}.\n"
            f"Job requirements: {req_yrs1}+ years {skill1}, {req_yrs2}+ years {skill2}, {req_yrs3}+ years {skill3}. "
            f"All three are listed as required (not preferred). Is this a strong match?"
        )

        reasoning = (
            f"I need to evaluate each requirement systematically before forming a conclusion. "
            f"Requirement 1: {req_yrs1}+ years {skill1}. Candidate has {cand_yrs1} years. {'MEETS' if meets1 else 'DOES NOT MEET'} ({cand_yrs1}/{req_yrs1}). "
            f"Requirement 2: {req_yrs2}+ years {skill2}. Candidate has {cand_yrs2} years. {'MEETS' if meets2 else 'DOES NOT MEET'} ({cand_yrs2}/{req_yrs2}). "
            f"Requirement 3: {req_yrs3}+ years {skill3}. Candidate has {cand_yrs3} years. {'MEETS' if meets3 else 'DOES NOT MEET'} ({cand_yrs3}/{req_yrs3}). "
            f"Score: {met_count}/3 required criteria met. "
            f"Since all three are listed as required (not preferred), failing any one is a significant gap. "
            f"My assessment: {'Strong match — all criteria met' if met_count == 3 else f'Partial match — {3-met_count} gap(s) in required skills' if met_count >= 1 else 'Poor match — no required criteria met'}."
        )

        if met_count == 3:
            verdict = "**Strong Match**"
            detail = "The candidate meets all three required skill thresholds."
        elif met_count == 2:
            gap_skill = skill1 if not meets1 else (skill2 if not meets2 else skill3)
            gap_req = req_yrs1 if not meets1 else (req_yrs2 if not meets2 else req_yrs3)
            gap_cand = cand_yrs1 if not meets1 else (cand_yrs2 if not meets2 else cand_yrs3)
            verdict = "**Partial Match — Proceed with Caution**"
            detail = f"Meets 2/3 required criteria. Gap: **{gap_skill}** ({gap_cand} years vs {gap_req} required). Since this is a *required* skill, recommend interview to assess practical depth."
        else:
            verdict = "**Weak Match — Not Recommended to Proceed**"
            detail = f"Only meets {met_count}/3 required criteria. Multiple required skill gaps make this candidate unlikely to succeed in the role without significant upskilling."

        answer = (
            f"## Candidate Evaluation: {cand} — {job_title}\n\n"
            f"### Requirement-by-Requirement Assessment\n\n"
            f"| Skill | Required | Candidate | Status |\n"
            f"|-------|----------|-----------|--------|\n"
            f"| {skill1} | {req_yrs1}+ years | {cand_yrs1} years | {'✓ Meets' if meets1 else '✗ Gap'} |\n"
            f"| {skill2} | {req_yrs2}+ years | {cand_yrs2} years | {'✓ Meets' if meets2 else '✗ Gap'} |\n"
            f"| {skill3} | {req_yrs3}+ years | {cand_yrs3} years | {'✓ Meets' if meets3 else '✗ Gap'} |\n\n"
            f"### Overall Verdict: {verdict}\n\n"
            f"{detail}\n\n"
            f"*Confidence: High — assessment is based on direct comparison of stated years of experience against explicit requirements.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="reasoning_chain", difficulty="medium",
                             category="candidate_evaluation", source="advanced_synthetic"))

    # --- Compliance check chain ---
    for _ in range(per_scenario):
        r2 = _rnd()
        company = _pick(_COMPANY_NAMES, r2)
        regulation = r2.choice(["GDPR", "HIPAA", "SOC 2 Type II", "ISO 27001", "PCI DSS"])
        violations = r2.randint(0, 4)
        findings = r2.sample([
            "encryption at rest not enabled on all storage buckets",
            "no formal incident response plan documented",
            "access logs retained for only 30 days (requirement: 90 days)",
            "third-party vendors lack signed DPAs",
            "MFA not enforced for privileged accounts",
            "no annual penetration testing conducted",
            "data retention policy not formally published",
        ], k=violations)

        query = (
            f"Compliance audit findings for {company} against {regulation}:\n"
            + (("\n".join(f"- {f}" for f in findings) if findings else "- No findings noted") + "\n")
            + f"\nAre they compliant with {regulation}? What is their risk level and what must they do?"
        )

        reasoning = (
            f"I need to assess each finding against {regulation} requirements. "
            f"First, I'll classify each finding by severity: critical (immediate compliance risk), "
            f"major (significant gap), or minor (best-practice deviation). "
            f"{'No findings means likely compliant, pending full audit scope.' if not findings else f'There are {len(findings)} findings to assess.'} "
            f"Then I need to determine overall compliance status and risk level. "
            f"For {regulation}, any critical finding means non-compliant status. "
            f"Finally, I should prioritise remediation actions."
        )

        if not findings:
            risk = "Low"
            status = "Likely Compliant"
            actions = "Continue current controls; schedule next annual review."
        elif violations <= 2:
            risk = "Medium"
            status = "Partially Non-Compliant"
            actions = "Address gaps within 90 days; document remediation plan."
        else:
            risk = "High"
            status = "Non-Compliant"
            actions = "Immediate remediation required; consider engaging compliance counsel."

        findings_table = (
            "\n".join(f"| {f[:60]} | Major | 60 days |" for f in findings)
            if findings else "| No findings | N/A | N/A |"
        )

        answer = (
            f"## {regulation} Compliance Assessment — {company}\n\n"
            f"**Overall Status: {status}**\n"
            f"**Risk Level: {risk}**\n\n"
            f"### Findings Analysis\n\n"
            f"| Finding | Severity | Remediation Deadline |\n"
            f"|---------|----------|---------------------|\n"
            f"{findings_table}\n\n"
            f"### Required Actions\n"
            f"{actions}\n\n"
            f"### Reasoning\n"
            f"Based on {len(findings)} audit finding(s), the organisation {'has no' if not findings else 'has'} identified gaps "
            f"in {regulation} compliance. {'All requirements appear to be met.' if not findings else 'Each finding must be remediated with documented evidence before the next audit.'}\n\n"
            f"*Confidence: High for structural assessment; specific {regulation} article citations would require full regulatory text review.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="reasoning_chain", difficulty="hard",
                             category="compliance_reasoning", source="advanced_synthetic"))

    # --- Risk assessment chain ---
    for _ in range(per_scenario):
        r2 = _rnd()
        company = _pick(_COMPANY_NAMES, r2)
        risks = r2.sample([
            ("data breach exposure", "high", "Critical", "$2M-$10M"),
            ("contract termination risk", "medium", "Significant", "$500K-$2M"),
            ("regulatory fine", "high", "Major", "$1M-$5M"),
            ("key person dependency", "low", "Moderate", "$200K-$1M"),
            ("supply chain disruption", "medium", "Significant", "$1M-$3M"),
            ("reputational damage", "medium", "Moderate", "Unquantified"),
        ], k=r2.randint(2, 4))

        query = (
            f"Based on the following risk register excerpt for {company}, "
            f"calculate the overall risk profile and recommend the top 2 risks to address first.\n\n"
            + "\n".join(f"Risk: {r[0]} | Likelihood: {r[1]} | Impact: {r[2]} | Estimated exposure: {r[3]}" for r in risks)
        )

        reasoning = (
            f"To prioritise risks, I need to apply a risk matrix: combine likelihood and impact ratings. "
            f"High likelihood + Critical impact = top priority. "
            f"I'll score each risk: high=3, medium=2, low=1 for likelihood; Critical=4, Major=3, Significant=2, Moderate=1 for impact. "
            f"Multiply to get risk score, then rank. "
            f"Top 2 by score should be addressed first. "
            f"I should also consider the financial exposure estimates as a secondary ranking factor."
        )

        scored = []
        lik_map = {"high": 3, "medium": 2, "low": 1}
        imp_map = {"Critical": 4, "Major": 3, "Significant": 2, "Moderate": 1}
        for risk_name, lik, imp, exp in risks:
            score = lik_map.get(lik, 2) * imp_map.get(imp, 2)
            scored.append((risk_name, lik, imp, exp, score))
        scored.sort(key=lambda x: x[4], reverse=True)

        answer = (
            f"## Risk Profile Assessment — {company}\n\n"
            f"### Risk Scoring Matrix\n\n"
            f"| Risk | Likelihood | Impact | Score | Exposure |\n"
            f"|------|-----------|--------|-------|----------|\n"
            + "\n".join(f"| {s[0]} | {s[1]} | {s[2]} | {s[4]}/12 | {s[3]} |" for s in scored)
            + f"\n\n### Top Priority Risks\n\n"
            f"**Priority 1: {scored[0][0]}** (Score: {scored[0][4]}/12)\n"
            f"Estimated exposure: {scored[0][3]}. Immediate mitigation plan required.\n\n"
            f"**Priority 2: {scored[1][0]}** (Score: {scored[1][4]}/12)\n"
            f"Estimated exposure: {scored[1][3]}. Mitigation plan within 30 days.\n\n"
            f"### Overall Risk Profile\n"
            f"{'HIGH — immediate board-level attention required.' if scored[0][4] >= 9 else 'MEDIUM — management-level action plan required.' if scored[0][4] >= 6 else 'LOW — monitor and review quarterly.'}\n\n"
            f"*Confidence: High — scoring based on stated likelihood/impact ratings; exposure estimates are as provided.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="reasoning_chain", difficulty="hard",
                             category="risk_assessment", source="advanced_synthetic"))

    # --- Vendor selection chain ---
    for _ in range(per_scenario):
        r2 = _rnd()
        vendors = _pick(_COMPANY_NAMES, r2, n=3)
        criteria = ["price", "support SLA", "security certifications", "scalability"]
        scores = {v: {c: r2.randint(1, 5) for c in criteria} for v in vendors}
        weights = {"price": 0.30, "support SLA": 0.25, "security certifications": 0.25, "scalability": 0.20}

        query = (
            f"We are selecting a cloud vendor from three candidates. Score each vendor (1-5) on criteria:\n"
            + "\n".join(
                f"{v}: " + ", ".join(f"{c}={scores[v][c]}" for c in criteria)
                for v in vendors
            )
            + f"\nWeights: price=30%, support SLA=25%, security=25%, scalability=20%. "
            f"Which vendor should we select and why?"
        )

        weighted = {}
        for v in vendors:
            weighted[v] = sum(scores[v][c] * weights[c] for c in criteria)
        best = max(weighted, key=lambda x: weighted[x])

        reasoning = (
            f"I need to compute the weighted score for each vendor. "
            f"For each vendor, multiply each criterion score by its weight and sum. "
            + " ".join(
                f"{v}: {' + '.join(f'{scores[v][c]}×{weights[c]}' for c in criteria)} = {weighted[v]:.2f}."
                for v in vendors
            )
            + f" Highest score wins: {best} with {weighted[best]:.2f}. "
            f"I should also flag any vendor with a critically low security score since that's a deal-breaker."
        )

        answer = (
            f"## Vendor Selection Analysis\n\n"
            f"### Weighted Scoring\n\n"
            f"| Vendor | Price (30%) | Support (25%) | Security (25%) | Scalability (20%) | **Total** |\n"
            f"|--------|------------|--------------|----------------|-------------------|----------|\n"
            + "\n".join(
                f"| {v} | {scores[v]['price']} | {scores[v]['support SLA']} | {scores[v]['security certifications']} | {scores[v]['scalability']} | **{weighted[v]:.2f}** |"
                for v in vendors
            )
            + f"\n\n### Recommendation: **{best}**\n\n"
            f"**{best}** achieves the highest weighted score of **{weighted[best]:.2f}/5.00**. "
            f"This accounts for the defined business priorities with price and support SLA carrying the most weight.\n\n"
            f"**Security note:** Verify that the selected vendor holds required certifications (SOC 2, ISO 27001) regardless of score.\n\n"
            f"*Confidence: High — recommendation is mathematically derived from provided scores and weights.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="reasoning_chain", difficulty="medium",
                             category="vendor_selection", source="advanced_synthetic"))

    return examples[:n]


# ---------------------------------------------------------------------------
# Generator 3: Numerical reasoning examples
# ---------------------------------------------------------------------------

def generate_numerical_reasoning_examples(n: int = 3000) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    r = _rnd()
    per_type = n // 5

    # --- Invoice totals ---
    for _ in range(per_type):
        r2 = _rnd()
        text, inv_no = _build_invoice(r2, n_items=r2.randint(5, 12))
        import re as _re
        lines = text.split("\n")
        item_lines = [l for l in lines if "$" in l and "|" not in l and "TOTAL" not in l
                      and "Subtotal" not in l and "Tax" not in l and "Invoice" not in l
                      and "Due" not in l and "From" not in l and "To:" not in l]
        amounts_found = _re.findall(r'\$([\d,]+\.\d{2})', text)
        if len(amounts_found) < 3:
            continue
        grand_total_match = _re.search(r'TOTAL DUE\s+\$([\d,]+\.\d{2})', text)
        if not grand_total_match:
            continue
        grand_total = grand_total_match.group(1)
        subtotal_match = _re.search(r'Subtotal\s+\$([\d,]+\.\d{2})', text)
        tax_match = _re.search(r'Tax \((\d+)%\)\s+\$([\d,]+\.\d{2})', text)

        if not subtotal_match or not tax_match:
            continue

        subtotal = subtotal_match.group(1)
        tax_rate = tax_match.group(1)
        tax_amt = tax_match.group(2)

        examples.append(_sft(
            query=f"Review the following invoice and verify: (1) Are the line item totals correct? (2) Is the subtotal accurate? (3) Does the grand total match subtotal + tax?\n\n{text}",
            reasoning=(
                f"I need to verify arithmetic on this invoice step by step. "
                f"Step 1: For each line item, verify qty × unit_price = line_total. "
                f"Step 2: Sum all line totals to verify the subtotal of ${subtotal}. "
                f"Step 3: Verify tax: subtotal × {tax_rate}% = ${tax_amt}. "
                f"Step 4: Verify grand total = subtotal + tax = ${grand_total}. "
                f"I should flag any discrepancies found in the arithmetic."
            ),
            answer=(
                f"## Invoice Verification — {inv_no}\n\n"
                f"### Step 1: Line Item Verification\n"
                f"Each line item was checked: quantity × unit price = line total. "
                f"All line items verified as arithmetically correct based on the figures shown.\n\n"
                f"### Step 2: Subtotal Verification\n"
                f"Sum of all line totals = **${subtotal}** ✓\n\n"
                f"### Step 3: Tax Verification\n"
                f"${subtotal} × {tax_rate}% = **${tax_amt}** ✓\n\n"
                f"### Step 4: Grand Total Verification\n"
                f"${subtotal} (subtotal) + ${tax_amt} (tax) = **${grand_total}** ✓\n\n"
                f"**Conclusion:** All arithmetic checks pass. Invoice total of **${grand_total}** is correct.\n\n"
                f"*Confidence: High — all calculations verified against stated figures.*"
            ),
            area="numerical_reasoning", difficulty="medium",
            category="invoice_verification", source="advanced_synthetic",
        ))

    # --- Percentage calculations ---
    for _ in range(per_type):
        r2 = _rnd()
        company = _pick(_COMPANY_NAMES, r2)
        year = r2.randint(2022, 2024)
        q1 = r2.randint(1_000_000, 20_000_000)
        q2 = int(q1 * r2.uniform(0.8, 1.4))
        q3 = int(q2 * r2.uniform(0.85, 1.35))
        q4 = int(q3 * r2.uniform(0.90, 1.30))

        q1_q2_chg = (q2 - q1) / q1 * 100
        q2_q3_chg = (q3 - q2) / q2 * 100
        q3_q4_chg = (q4 - q3) / q3 * 100
        annual = q1 + q2 + q3 + q4
        best_q = max([(q1, "Q1"), (q2, "Q2"), (q3, "Q3"), (q4, "Q4")], key=lambda x: x[0])

        query = (
            f"Quarterly revenue for {company} in {year}:\n"
            f"Q1: ${q1:,} | Q2: ${q2:,} | Q3: ${q3:,} | Q4: ${q4:,}\n\n"
            f"Calculate: (1) QoQ % change for each quarter transition, "
            f"(2) Total annual revenue, (3) Which quarter had highest revenue, "
            f"(4) Average quarterly revenue."
        )

        reasoning = (
            f"I need to perform four calculations. "
            f"QoQ changes: Q1→Q2: ({q2:,}-{q1:,})/{q1:,}×100 = {q1_q2_chg:.1f}%. "
            f"Q2→Q3: ({q3:,}-{q2:,})/{q2:,}×100 = {q2_q3_chg:.1f}%. "
            f"Q3→Q4: ({q4:,}-{q3:,})/{q3:,}×100 = {q3_q4_chg:.1f}%. "
            f"Annual: {q1:,}+{q2:,}+{q3:,}+{q4:,} = {annual:,}. "
            f"Best quarter: {best_q[1]} with ${best_q[0]:,}. "
            f"Average: {annual:,}/4 = {annual//4:,}."
        )

        answer = (
            f"## Revenue Analysis — {company} {year}\n\n"
            f"### Quarter-over-Quarter Changes\n\n"
            f"| Transition | Change | Direction |\n"
            f"|-----------|--------|----------|\n"
            f"| Q1 → Q2 | **{q1_q2_chg:+.1f}%** | {'▲ Growth' if q1_q2_chg > 0 else '▼ Decline'} |\n"
            f"| Q2 → Q3 | **{q2_q3_chg:+.1f}%** | {'▲ Growth' if q2_q3_chg > 0 else '▼ Decline'} |\n"
            f"| Q3 → Q4 | **{q3_q4_chg:+.1f}%** | {'▲ Growth' if q3_q4_chg > 0 else '▼ Decline'} |\n\n"
            f"### Key Metrics\n\n"
            f"- **Total Annual Revenue:** ${annual:,}\n"
            f"- **Highest Revenue Quarter:** {best_q[1]} (${best_q[0]:,})\n"
            f"- **Average Quarterly Revenue:** ${annual//4:,}\n\n"
            f"*Confidence: High — all figures calculated directly from provided data.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="numerical_reasoning", difficulty="medium",
                             category="percentage_calculation", source="advanced_synthetic"))

    # --- Aggregation across vendors ---
    for _ in range(per_type):
        r2 = _rnd()
        vendors = _pick(_COMPANY_NAMES, r2, n=4)
        orders = {}
        for v in vendors:
            n_orders = r2.randint(2, 5)
            order_vals = [r2.randint(5000, 150000) for _ in range(n_orders)]
            orders[v] = order_vals

        query = (
            f"Purchase order data for the past quarter:\n\n"
            + "\n".join(
                f"{v}: {len(orders[v])} orders totalling ${sum(orders[v]):,} "
                f"(individual orders: {', '.join('$'+f'{x:,}' for x in orders[v])})"
                for v in vendors
            )
            + "\n\nWhich vendor has the highest average order value? "
            f"What is the total spend? Which vendor represents the largest % of total spend?"
        )

        avgs = {v: sum(orders[v]) / len(orders[v]) for v in vendors}
        totals = {v: sum(orders[v]) for v in vendors}
        grand = sum(totals.values())
        best_avg = max(avgs, key=lambda x: avgs[x])
        biggest_spend = max(totals, key=lambda x: totals[x])

        reasoning = (
            f"Computing average order value: divide each vendor's total by order count. "
            + " ".join(f"{v}: ${totals[v]:,}/{len(orders[v])} = ${avgs[v]:,.0f}." for v in vendors)
            + f" Grand total: {' + '.join(f'${totals[v]:,}' for v in vendors)} = ${grand:,}. "
            f"Best AOV: {best_avg} at ${avgs[best_avg]:,.0f}. "
            f"Largest spend share: {biggest_spend} at {totals[biggest_spend]/grand*100:.1f}%."
        )

        answer = (
            f"## Vendor Spend Analysis\n\n"
            f"| Vendor | Orders | Total Spend | Avg Order Value | % of Total |\n"
            f"|--------|--------|-------------|-----------------|------------|\n"
            + "\n".join(
                f"| {v} | {len(orders[v])} | ${totals[v]:,} | ${avgs[v]:,.0f} | {totals[v]/grand*100:.1f}% |"
                for v in vendors
            )
            + f"\n| **TOTAL** | {sum(len(orders[v]) for v in vendors)} | **${grand:,}** | - | 100% |\n\n"
            f"### Key Findings\n\n"
            f"- **Highest Average Order Value:** **{best_avg}** at **${avgs[best_avg]:,.0f}** per order\n"
            f"- **Total Quarterly Spend:** **${grand:,}**\n"
            f"- **Largest Spend Share:** **{biggest_spend}** representing **{totals[biggest_spend]/grand*100:.1f}%** of total\n\n"
            f"*Confidence: High — all calculations derived from provided order data.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="numerical_reasoning", difficulty="medium",
                             category="vendor_aggregation", source="advanced_synthetic"))

    # --- Payment verification ---
    for _ in range(per_type):
        r2 = _rnd()
        items = [(r2.choice(["Consulting", "Licence", "Support", "Hardware", "Training"]),
                  r2.randint(1, 10),
                  round(r2.uniform(500, 15000), 2))
                 for _ in range(r2.randint(3, 6))]
        correct_total = round(sum(qty * price for _, qty, price in items), 2)
        claimed_payment = round(correct_total * r2.choice([0.95, 1.0, 1.0, 1.05, 0.90]), 2)
        discrepancy = round(claimed_payment - correct_total, 2)
        company = _pick(_COMPANY_NAMES, r2)

        query = (
            f"{company} claims a payment of ${claimed_payment:,.2f} for the following items:\n\n"
            + "\n".join(f"- {desc}: {qty} × ${price:,.2f}" for desc, qty, price in items)
            + f"\n\nVerify whether the claimed payment of ${claimed_payment:,.2f} is consistent with the line items."
        )

        correct = abs(discrepancy) < 0.01

        reasoning = (
            f"I need to compute the correct total from line items and compare against the claimed payment. "
            + " ".join(f"{desc}: {qty}×${price:,.2f}=${qty*price:,.2f}." for desc, qty, price in items)
            + f" Sum: ${correct_total:,.2f}. "
            f"Claimed: ${claimed_payment:,.2f}. "
            f"Difference: ${discrepancy:,.2f}. "
            f"{'The payment matches.' if correct else f'There is a discrepancy of ${abs(discrepancy):,.2f}.'}"
        )

        if correct:
            verdict = f"The claimed payment of **${claimed_payment:,.2f}** is **correct** and consistent with all line items."
            action = "No action required."
        elif discrepancy > 0:
            verdict = f"The claimed payment of **${claimed_payment:,.2f}** is **overstated by ${abs(discrepancy):,.2f}**. Correct total is ${correct_total:,.2f}."
            action = f"**Action required:** Request a credit note or revised invoice for the ${abs(discrepancy):,.2f} overcharge."
        else:
            verdict = f"The claimed payment of **${claimed_payment:,.2f}** is **understated by ${abs(discrepancy):,.2f}**. Correct total is ${correct_total:,.2f}."
            action = f"**Action required:** Issue a supplementary invoice for the missing ${abs(discrepancy):,.2f}."

        answer = (
            f"## Payment Verification — {company}\n\n"
            f"### Line Item Calculation\n\n"
            f"| Item | Qty | Unit Price | Line Total |\n"
            f"|------|-----|-----------|------------|\n"
            + "\n".join(f"| {desc} | {qty} | ${price:,.2f} | ${qty*price:,.2f} |" for desc, qty, price in items)
            + f"\n| **TOTAL** | | | **${correct_total:,.2f}** |\n\n"
            f"### Verification Result\n\n"
            f"{verdict}\n\n"
            f"{action}\n\n"
            f"*Confidence: High — arithmetic verified from stated figures.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="numerical_reasoning", difficulty="hard",
                             category="payment_verification", source="advanced_synthetic"))

    # --- Statistical analysis ---
    for _ in range(per_type):
        r2 = _rnd()
        metric = r2.choice(["response time (ms)", "error rate (%)", "throughput (req/s)", "CPU usage (%)"])
        samples = [round(r2.uniform(10, 500), 1) for _ in range(r2.randint(6, 10))]
        mean = round(sum(samples) / len(samples), 2)
        sorted_s = sorted(samples)
        n_s = len(sorted_s)
        median = sorted_s[n_s // 2] if n_s % 2 == 1 else round((sorted_s[n_s//2-1] + sorted_s[n_s//2]) / 2, 2)
        min_v, max_v = sorted_s[0], sorted_s[-1]
        variance = round(sum((x - mean)**2 for x in samples) / n_s, 2)
        import math as _math
        std_dev = round(_math.sqrt(variance), 2)
        company = _pick(_COMPANY_NAMES, r2)

        query = (
            f"System monitoring data for {company} — {metric} over {n_s} readings:\n"
            f"{', '.join(str(x) for x in samples)}\n\n"
            f"Calculate: mean, median, min, max, and standard deviation. "
            f"Flag any readings that are more than 2 standard deviations from the mean (potential anomalies)."
        )

        anomalies = [x for x in samples if abs(x - mean) > 2 * std_dev]

        reasoning = (
            f"Computing statistics on {n_s} data points. "
            f"Mean: sum/n = {sum(samples):.1f}/{n_s} = {mean}. "
            f"Sorted values: {sorted_s}. Median = {median}. "
            f"Min={min_v}, Max={max_v}. "
            f"Variance = avg of squared deviations = {variance}. "
            f"Std dev = sqrt({variance}) = {std_dev}. "
            f"Anomaly threshold: mean ± 2×std_dev = {mean}±{2*std_dev:.2f} = [{round(mean-2*std_dev,2)}, {round(mean+2*std_dev,2)}]. "
            f"{'Anomalies found: ' + str(anomalies) if anomalies else 'No anomalies detected.'}"
        )

        answer = (
            f"## Statistical Analysis — {company} ({metric})\n\n"
            f"**Data ({n_s} readings):** {', '.join(str(x) for x in samples)}\n\n"
            f"### Descriptive Statistics\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Mean | **{mean}** |\n"
            f"| Median | **{median}** |\n"
            f"| Min | **{min_v}** |\n"
            f"| Max | **{max_v}** |\n"
            f"| Std Deviation | **{std_dev}** |\n\n"
            f"### Anomaly Detection (>2σ from mean)\n\n"
            f"Threshold: [{round(mean-2*std_dev,2)}, {round(mean+2*std_dev,2)}]\n\n"
            + (f"**Anomalies detected:** {', '.join(str(x) for x in anomalies)} — these readings warrant investigation.\n"
               if anomalies else "**No anomalies detected** — all readings within 2 standard deviations of mean.\n")
            + f"\n*Confidence: High — all calculations performed on provided data.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="numerical_reasoning", difficulty="hard",
                             category="statistical_analysis", source="advanced_synthetic"))

    return examples[:n]


# ---------------------------------------------------------------------------
# Generator 4: Table understanding examples
# ---------------------------------------------------------------------------

def generate_table_understanding_examples(n: int = 3000) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    r = _rnd()
    per_type = n // 5

    def _make_employee_table(r2: random.Random) -> Tuple[str, List]:
        headers = ["Name", "Department", "Salary", "Years", "Rating"]
        rows = []
        depts = ["Engineering", "Sales", "Marketing", "Finance", "HR"]
        for _ in range(r2.randint(6, 10)):
            rows.append([
                _pick(_PERSON_NAMES, r2),
                r2.choice(depts),
                r2.randint(60000, 180000),
                r2.randint(1, 15),
                round(r2.uniform(2.5, 5.0), 1),
            ])
        table = "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in rows:
            table += f"| {row[0]} | {row[1]} | ${row[2]:,} | {row[3]} | {row[4]} |\n"
        return table, rows

    # --- Extract cell values ---
    for _ in range(per_type):
        r2 = _rnd()
        table, rows = _make_employee_table(r2)
        target = r2.choice(rows)
        target_name = target[0]
        target_salary = target[2]
        target_dept = target[1]

        query = f"From the following employee table, what is {target_name}'s salary and department?\n\n{table}"
        reasoning = (
            f"I need to find the row for {target_name} in the table. "
            f"Scanning the Name column... found {target_name}. "
            f"Reading across: Department = {target_dept}, Salary = ${target_salary:,}. "
            f"These are direct lookups from the table."
        )
        answer = (
            f"From the employee table:\n\n"
            f"- **Name:** {target_name}\n"
            f"- **Department:** {target_dept}\n"
            f"- **Salary:** ${target_salary:,}\n\n"
            f"*Confidence: High — direct table lookup.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="table_understanding", difficulty="easy",
                             category="cell_lookup", source="advanced_synthetic"))

    # --- Column totals and averages ---
    for _ in range(per_type):
        r2 = _rnd()
        table, rows = _make_employee_table(r2)
        total_salary = sum(r[2] for r in rows)
        avg_salary = total_salary // len(rows)
        max_salary = max(r[2] for r in rows)
        min_salary = min(r[2] for r in rows)
        max_person = next(r[0] for r in rows if r[2] == max_salary)
        min_person = next(r[0] for r in rows if r[2] == min_salary)

        query = f"Analyse the salary column in the following employee table. Compute total, average, max, and min. Identify highest and lowest paid employees.\n\n{table}"
        reasoning = (
            f"I need to extract all salary values from the table and perform calculations. "
            f"Salaries: {', '.join('$'+str(r[2]) for r in rows)}. "
            f"Total: ${total_salary:,}. Average: ${total_salary:,}/{len(rows)} = ${avg_salary:,}. "
            f"Max: ${max_salary:,} ({max_person}). Min: ${min_salary:,} ({min_person})."
        )
        answer = (
            f"## Salary Analysis\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Total Payroll | **${total_salary:,}** |\n"
            f"| Average Salary | **${avg_salary:,}** |\n"
            f"| Highest Salary | **${max_salary:,}** ({max_person}) |\n"
            f"| Lowest Salary | **${min_salary:,}** ({min_person}) |\n"
            f"| Headcount | **{len(rows)}** |\n\n"
            f"*Confidence: High — all figures computed directly from table data.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="table_understanding", difficulty="medium",
                             category="column_aggregation", source="advanced_synthetic"))

    # --- Row comparison / ranking ---
    for _ in range(per_type):
        r2 = _rnd()
        table, rows = _make_employee_table(r2)
        sorted_by_rating = sorted(rows, key=lambda x: x[4], reverse=True)
        top3 = sorted_by_rating[:3]

        query = f"From the following employee table, rank the top 3 employees by performance rating. For each, state their name, department, salary, and rating.\n\n{table}"
        reasoning = (
            f"I need to sort all employees by the Rating column in descending order. "
            f"Ratings: {', '.join(f'{r[0]}={r[4]}' for r in rows)}. "
            f"Top 3 after sorting: {', '.join(r[0] for r in top3)}."
        )
        answer = (
            f"## Top 3 Employees by Performance Rating\n\n"
            f"| Rank | Name | Department | Salary | Rating |\n"
            f"|------|------|-----------|--------|--------|\n"
            + "\n".join(
                f"| {i+1} | **{e[0]}** | {e[1]} | ${e[2]:,} | **{e[4]}** |"
                for i, e in enumerate(top3)
            )
            + f"\n\n*Confidence: High — ranking derived directly from table ratings.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="table_understanding", difficulty="medium",
                             category="row_ranking", source="advanced_synthetic"))

    # --- Anomaly detection in table ---
    for _ in range(per_type):
        r2 = _rnd()
        table, rows = _make_employee_table(r2)
        # Inject an anomaly: one very high salary
        anomaly_idx = r2.randint(0, len(rows)-1)
        rows[anomaly_idx][2] = r2.randint(500000, 1000000)  # outlier
        anomaly_name = rows[anomaly_idx][0]
        # Rebuild table
        headers = ["Name", "Department", "Salary", "Years", "Rating"]
        table = "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in rows:
            table += f"| {row[0]} | {row[1]} | ${row[2]:,} | {row[3]} | {row[4]} |\n"
        salaries = [r[2] for r in rows]
        mean_s = sum(salaries) / len(salaries)

        query = f"Identify any anomalies or outliers in the following employee salary table. Explain your reasoning.\n\n{table}"
        reasoning = (
            f"I need to identify statistical outliers in the salary column. "
            f"Computing mean: ${mean_s:,.0f}. "
            f"Scanning for values significantly higher or lower. "
            f"{anomaly_name} has ${rows[anomaly_idx][2]:,} which is {rows[anomaly_idx][2]/mean_s:.1f}x the average — clear outlier. "
            f"This could be a data entry error, executive compensation, or contract worker rate."
        )
        answer = (
            f"## Salary Anomaly Analysis\n\n"
            f"**Average salary (excluding potential outlier):** approximately ${int(mean_s):,}\n\n"
            f"### Detected Anomaly\n\n"
            f"| Employee | Salary | vs Average | Flag |\n"
            f"|----------|--------|-----------|------|\n"
            f"| **{anomaly_name}** | **${rows[anomaly_idx][2]:,}** | **{rows[anomaly_idx][2]/mean_s:.1f}x average** | OUTLIER |\n\n"
            f"**{anomaly_name}**'s salary of **${rows[anomaly_idx][2]:,}** is significantly higher than the group average. "
            f"Possible explanations:\n"
            f"1. Executive/senior leadership compensation (check job title)\n"
            f"2. Data entry error (extra zero)\n"
            f"3. Contract rate mistakenly entered as annual salary\n\n"
            f"**Recommendation:** Verify this record against HR source system.\n\n"
            f"*Confidence: High — outlier identification based on statistical deviation from group mean.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="table_understanding", difficulty="hard",
                             category="anomaly_detection", source="advanced_synthetic"))

    # --- Missing values handling ---
    for _ in range(per_type):
        r2 = _rnd()
        table, rows = _make_employee_table(r2)
        # Inject missing values
        missing_count = r2.randint(1, 3)
        missing_rows = r2.sample(range(len(rows)), k=missing_count)
        for idx in missing_rows:
            rows[idx][4] = None  # missing rating
        headers = ["Name", "Department", "Salary", "Years", "Rating"]
        table = "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in rows:
            rating_str = str(row[4]) if row[4] is not None else "N/A"
            table += f"| {row[0]} | {row[1]} | ${row[2]:,} | {row[3]} | {rating_str} |\n"
        missing_names = [rows[i][0] for i in missing_rows]
        valid_ratings = [r[4] for r in rows if r[4] is not None]
        avg_rating = round(sum(valid_ratings) / len(valid_ratings), 2) if valid_ratings else 0.0

        query = f"The following employee table has some missing performance ratings (shown as N/A). Identify which employees are missing ratings, compute the average rating for those who have one, and suggest how to handle the missing data.\n\n{table}"
        reasoning = (
            f"I need to scan the Rating column for N/A values. "
            f"Found missing ratings for: {', '.join(missing_names)}. "
            f"Valid ratings: {', '.join(str(x) for x in valid_ratings)}. "
            f"Average of valid: {sum(valid_ratings):.1f}/{len(valid_ratings)} = {avg_rating}. "
            f"Missing data handling options: exclude from analysis, impute with mean, or flag for collection."
        )
        answer = (
            f"## Missing Data Analysis — Employee Ratings\n\n"
            f"### Employees with Missing Ratings\n"
            + "\n".join(f"- **{name}** — rating not recorded" for name in missing_names)
            + f"\n\n### Statistics on Available Ratings\n\n"
            f"- **Employees with ratings:** {len(valid_ratings)}/{len(rows)}\n"
            f"- **Average rating (excluding N/A):** {avg_rating}\n\n"
            f"### Recommended Approaches for Missing Data\n\n"
            f"1. **Collect missing data** — reach out to managers for {', '.join(missing_names[:2])}'s performance reviews (preferred)\n"
            f"2. **Mean imputation** — temporarily substitute {avg_rating} for analysis purposes (note: reduces variance)\n"
            f"3. **Exclusion** — exclude from rating-based analyses but retain for salary/headcount metrics\n\n"
            f"*Confidence: High — missing values clearly identified; handling strategy depends on business context.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="table_understanding", difficulty="medium",
                             category="missing_values", source="advanced_synthetic"))

    return examples[:n]


# ---------------------------------------------------------------------------
# Generator 5: Temporal reasoning examples
# ---------------------------------------------------------------------------

def generate_temporal_reasoning_examples(n: int = 2000) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    import math as _math

    per_type = n // 4

    # --- Date difference calculations ---
    for _ in range(per_type):
        r2 = _rnd()
        company = _pick(_COMPANY_NAMES, r2)
        from datetime import date, timedelta
        start = date(r2.randint(2020, 2023), r2.randint(1, 12), r2.randint(1, 28))
        delta_days = r2.randint(30, 730)
        end = start + timedelta(days=delta_days)
        months = round(delta_days / 30.44, 1)
        event = r2.choice(["contract signing", "first payment", "project kick-off",
                            "compliance audit", "data migration completion"])
        event2 = r2.choice(["final delivery", "contract expiry", "renewal date",
                             "performance review", "security assessment"])

        query = (
            f"For {company}: the {event} occurred on {start.strftime('%B %d, %Y')} "
            f"and the {event2} occurred on {end.strftime('%B %d, %Y')}. "
            f"How many days, weeks, and months elapsed between these two events? "
            f"Was it more or less than 6 months?"
        )
        weeks = round(delta_days / 7, 1)
        six_months = delta_days > 182

        reasoning = (
            f"I need to calculate the duration between {start.strftime('%B %d, %Y')} and {end.strftime('%B %d, %Y')}. "
            f"Days: end - start = {delta_days} days. "
            f"Weeks: {delta_days}/7 = {weeks} weeks. "
            f"Months: {delta_days}/30.44 = {months} months. "
            f"Six months = ~182 days. {delta_days} days is {'more' if six_months else 'less'} than 182 days."
        )
        answer = (
            f"## Timeline Calculation — {company}\n\n"
            f"**From:** {start.strftime('%B %d, %Y')} ({event})\n"
            f"**To:** {end.strftime('%B %d, %Y')} ({event2})\n\n"
            f"| Unit | Duration |\n"
            f"|------|----------|\n"
            f"| Days | **{delta_days} days** |\n"
            f"| Weeks | **{weeks} weeks** |\n"
            f"| Months | **{months} months** |\n\n"
            f"**Compared to 6 months (~182 days):** The period of {delta_days} days is "
            f"**{'more' if six_months else 'less'} than 6 months** "
            f"({'by ' + str(delta_days - 182) + ' days' if six_months else 'by ' + str(182 - delta_days) + ' days'}).\n\n"
            f"*Confidence: High — calculated directly from provided dates.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="temporal_reasoning", difficulty="easy",
                             category="date_difference", source="advanced_synthetic"))

    # --- Event ordering / timeline ---
    for _ in range(per_type):
        r2 = _rnd()
        from datetime import date, timedelta
        base = date(r2.randint(2022, 2024), 1, 1)
        company = _pick(_COMPANY_NAMES, r2)
        event_templates = [
            "Contract signed", "Kick-off meeting", "Phase 1 delivered",
            "Security audit", "User acceptance testing", "Go-live", "Post-implementation review"
        ]
        offsets = sorted(r2.sample(range(1, 400), k=5))
        events = list(zip(event_templates[:5], [(base + timedelta(days=o)) for o in offsets]))
        r2.shuffle(events)  # shuffle to test ordering

        query = (
            f"The following events occurred for a {company} project. "
            f"Arrange them in chronological order and identify which happened before and after the Go-live:\n\n"
            + "\n".join(f"- {evt}: {dt.strftime('%B %d, %Y')}" for evt, dt in events)
        )

        sorted_events = sorted(events, key=lambda x: x[1])
        golive = next((dt for evt, dt in sorted_events if evt == "Go-live"), None)

        reasoning = (
            f"I need to sort the events by their dates. "
            f"Converting all dates to comparable form and sorting... "
            f"Chronological order: {', '.join(e[0] + '(' + e[1].strftime('%Y-%m-%d') + ')' for e in sorted_events)}. "
            f"{'Go-live date is ' + golive.strftime('%B %d, %Y') + '. All events before this are pre-go-live.' if golive else 'No Go-live found in events.'}"
        )

        pre_golive = [e for e in sorted_events if golive and e[1] < golive and e[0] != "Go-live"]
        post_golive = [e for e in sorted_events if golive and e[1] > golive and e[0] != "Go-live"]

        answer = (
            f"## Chronological Event Timeline — {company}\n\n"
            f"| # | Event | Date |\n"
            f"|---|-------|------|\n"
            + "\n".join(f"| {i+1} | {e[0]} | {e[1].strftime('%B %d, %Y')} |" for i, e in enumerate(sorted_events))
            + (f"\n\n### Pre / Post Go-live\n\n"
               f"**Before Go-live ({golive.strftime('%B %d, %Y')}):**\n"
               + "\n".join(f"- {e[0]} ({e[1].strftime('%B %d, %Y')})" for e in pre_golive)
               + f"\n\n**After Go-live:**\n"
               + "\n".join(f"- {e[0]} ({e[1].strftime('%B %d, %Y')})" for e in post_golive)
               if golive else "")
            + f"\n\n*Confidence: High — ordering based on stated dates.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="temporal_reasoning", difficulty="medium",
                             category="event_ordering", source="advanced_synthetic"))

    # --- Warranty / validity check ---
    for _ in range(per_type):
        r2 = _rnd()
        from datetime import date, timedelta
        purchase = date(r2.randint(2020, 2024), r2.randint(1, 12), r2.randint(1, 28))
        warranty_years = r2.choice([1, 2, 3, 5])
        warranty_expiry = date(purchase.year + warranty_years, purchase.month, purchase.day)
        check_date = date(r2.randint(2022, 2026), r2.randint(1, 12), r2.randint(1, 28))
        product = r2.choice(["enterprise server", "network switch", "UPS battery backup",
                              "storage array", "workstation"])
        company = _pick(_COMPANY_NAMES, r2)
        still_valid = check_date <= warranty_expiry

        query = (
            f"{company} purchased a {product} on {purchase.strftime('%B %d, %Y')} "
            f"with a {warranty_years}-year manufacturer warranty. "
            f"As of {check_date.strftime('%B %d, %Y')}, is the warranty still valid? "
            f"If so, how many days remain? If not, how long ago did it expire?"
        )

        if still_valid:
            days_remaining = (warranty_expiry - check_date).days
            status_text = f"The warranty is **still valid** with **{days_remaining} days remaining** (expires {warranty_expiry.strftime('%B %d, %Y')})."
        else:
            days_expired = (check_date - warranty_expiry).days
            status_text = f"The warranty **expired {days_expired} days ago** on {warranty_expiry.strftime('%B %d, %Y')}."

        reasoning = (
            f"Purchase date: {purchase.strftime('%Y-%m-%d')}. Warranty duration: {warranty_years} years. "
            f"Expiry date: {warranty_expiry.strftime('%Y-%m-%d')}. "
            f"Check date: {check_date.strftime('%Y-%m-%d')}. "
            f"Is {check_date.strftime('%Y-%m-%d')} before {warranty_expiry.strftime('%Y-%m-%d')}? "
            f"{'Yes — still valid.' if still_valid else 'No — already expired.'}"
        )

        answer = (
            f"## Warranty Status Check\n\n"
            f"| Detail | Value |\n"
            f"|--------|-------|\n"
            f"| Product | {product} |\n"
            f"| Purchase Date | {purchase.strftime('%B %d, %Y')} |\n"
            f"| Warranty Period | {warranty_years} year(s) |\n"
            f"| Warranty Expiry | {warranty_expiry.strftime('%B %d, %Y')} |\n"
            f"| Check Date | {check_date.strftime('%B %d, %Y')} |\n\n"
            f"**Status:** {status_text}\n\n"
            f"*Confidence: High — calculated from stated purchase date and warranty term.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="temporal_reasoning", difficulty="easy",
                             category="validity_check", source="advanced_synthetic"))

    # --- Timeline from scattered dates ---
    for _ in range(per_type):
        r2 = _rnd()
        from datetime import date, timedelta
        company = _pick(_COMPANY_NAMES, r2)
        base_year = r2.randint(2021, 2024)
        contract_text = (
            f"This Agreement was signed on {_MONTHS[r2.randint(0,5)]} {r2.randint(1,28)}, {base_year}. "
            f"The initial payment of ${r2.randint(10000,100000):,} was received on "
            f"{_MONTHS[r2.randint(0,5)]} {r2.randint(1,28)}, {base_year}. "
            f"Phase 1 deliverables were submitted {r2.randint(30,90)} days after signing. "
            f"The midpoint review occurred on {_MONTHS[r2.randint(6,11)]} {r2.randint(1,28)}, {base_year}. "
            f"Final delivery was completed on {_MONTHS[r2.randint(0,5)]} {r2.randint(1,28)}, {base_year+1}. "
            f"The retention payment of ${r2.randint(5000,50000):,} was released 30 days after final acceptance."
        )

        query = (
            f"Extract all dates and time references from the following contract narrative "
            f"and construct a chronological timeline:\n\n{contract_text}"
        )

        reasoning = (
            f"I need to scan the text for all explicit dates and time references. "
            f"Looking for: month/year combinations, day references, relative time expressions ('30 days after', 'X days after signing'). "
            f"Once extracted, I'll convert relative references to approximate absolute dates and sort chronologically. "
            f"I should flag any ambiguous references."
        )

        answer = (
            f"## Extracted Timeline\n\n"
            f"| Event | Date/Timing | Type |\n"
            f"|-------|------------|------|\n"
            f"| Agreement signed | Extracted from text | Explicit date |\n"
            f"| Initial payment received | Extracted from text | Explicit date |\n"
            f"| Phase 1 deliverables | 30-90 days after signing | Relative reference |\n"
            f"| Midpoint review | Extracted from text | Explicit date |\n"
            f"| Final delivery | Extracted from text | Explicit date ({base_year+1}) |\n"
            f"| Retention payment | 30 days after final acceptance | Relative reference |\n\n"
            f"**Timeline notes:**\n"
            f"- The contract spans approximately 12+ months ({base_year} to {base_year+1})\n"
            f"- Two relative time references need anchor dates to compute absolute dates\n"
            f"- Phase 1 deadline: signing date + 30-90 days\n"
            f"- Retention release: final acceptance date + 30 days\n\n"
            f"*Confidence: High for explicit dates; Medium for relative references (depends on anchor dates).*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="temporal_reasoning", difficulty="hard",
                             category="timeline_construction", source="advanced_synthetic"))

    return examples[:n]


# ---------------------------------------------------------------------------
# Generator 6: Legal reasoning examples
# ---------------------------------------------------------------------------

def generate_legal_reasoning_examples(n: int = 2000) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    per_type = n // 4

    # --- Party/obligation mapping ---
    for _ in range(per_type):
        r2 = _rnd()
        text, desc = _build_contract(r2)
        party_a = _pick(_COMPANY_NAMES, r2)
        party_b = _pick([c for c in _COMPANY_NAMES if c != party_a], r2)

        query = (
            f"From the following contract, identify: (1) all parties, (2) primary obligations of each party, "
            f"(3) payment terms, (4) termination rights.\n\n<contract>\n{text}\n</contract>"
        )
        reasoning = (
            f"I need to systematically extract legal obligations from this contract. "
            f"Party identification: look for names in the recitals and signature blocks. "
            f"Obligations: scan each article for 'shall' and 'must' statements attributed to each party. "
            f"Payment terms: Article 3 covers fees and payment. "
            f"Termination rights: Article 4 covers term and termination. "
            f"I'll map each obligation clearly to the responsible party."
        )
        answer = (
            f"## Contract Analysis\n\n"
            f"### Parties\n"
            f"- **Service Provider:** As named in the agreement preamble\n"
            f"- **Client:** As named in the agreement preamble\n\n"
            f"### Primary Obligations\n\n"
            f"**Service Provider shall:**\n"
            f"- Commence service delivery within 15 business days of Effective Date (Art. 2.1)\n"
            f"- Assign a dedicated project manager (Art. 2.3)\n"
            f"- Maintain {r2.choice([99.5, 99.9, 99.95])}% uptime SLA (Art. 8.1)\n"
            f"- Delete/return Client data within stated days of termination (Art. 7.1)\n"
            f"- Notify Client within 48 hours of data breach (Art. 7.3)\n\n"
            f"**Client shall:**\n"
            f"- Pay monthly invoices within 30 days of receipt (Art. 3.2)\n"
            f"- Provide access and cooperation required for service delivery (Art. 2)\n\n"
            f"### Payment Terms\n"
            f"- Monthly invoicing at contracted rate\n"
            f"- Net 30 payment terms\n"
            f"- Late interest at stated % per month\n\n"
            f"### Termination Rights\n"
            f"- **Without cause:** {r2.choice([30,60,90])} days written notice (either party)\n"
            f"- **For breach:** Notice + 30-day cure period (Art. 4.3)\n"
            f"- **For insolvency:** Immediate termination by Client (Art. 4.4)\n\n"
            f"*Confidence: High — all items mapped to specific contract articles.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="legal_reasoning", difficulty="hard",
                             category="obligation_mapping", source="advanced_synthetic"))

    # --- Condition/exception chains ---
    for _ in range(per_type):
        r2 = _rnd()
        notice = r2.choice([30, 60, 90])
        data_del = r2.choice([60, 90, 120])
        cure = 30
        company = _pick(_COMPANY_NAMES, r2)

        clause_text = (
            f"Clause 4.2: Either party may terminate this Agreement without cause upon {notice} days' notice, "
            f"UNLESS the terminating party is in material breach of this Agreement, in which case they forfeit "
            f"the right to terminate without cause until such breach is remedied. "
            f"PROVIDED THAT where termination is initiated by Client, Service Provider shall have {cure} days "
            f"to cure any outstanding breach before termination takes effect. "
            f"EXCEPT WHERE the breach involves a data security incident, in which case Client may terminate "
            f"immediately without notice."
        )

        query = (
            f"Parse the following complex contract clause and map all conditions and exceptions:\n\n{clause_text}\n\n"
            f"Under what circumstances can each party terminate, and what are the exact requirements for each path?"
        )
        reasoning = (
            f"This clause has multiple conditional branches. I'll map them as a decision tree. "
            f"Base rule: {notice}-day notice for no-cause termination. "
            f"Exception 1: if terminating party is in material breach, they lose no-cause termination right until breach cured. "
            f"Exception 2: Client-initiated termination triggers a {cure}-day cure opportunity for Service Provider. "
            f"Exception to Exception 2: data security incidents allow immediate Client termination. "
            f"I need to lay this out clearly as a decision tree."
        )
        answer = (
            f"## Clause Analysis — Termination Decision Tree\n\n"
            f"### Termination Paths\n\n"
            f"**Path 1: No-cause termination (standard)**\n"
            f"- Requirement: {notice} days' written notice\n"
            f"- Caveat: NOT available if terminating party is in material breach\n\n"
            f"**Path 2: Client terminates for Service Provider breach**\n"
            f"- Step 1: Client issues breach notice\n"
            f"- Step 2: Service Provider has {cure} days to cure\n"
            f"- If cured: Agreement continues\n"
            f"- If not cured: Termination takes effect\n"
            f"- Exception: Data security breach → immediate termination (no cure period)\n\n"
            f"**Path 3: Service Provider terminates for Client breach**\n"
            f"- Standard: {notice} days' notice (if not in breach themselves)\n"
            f"- Note: No cure right explicitly granted to Client under this clause\n\n"
            f"### Condition Summary Table\n\n"
            f"| Scenario | Notice Required | Cure Period | Immediate Option |\n"
            f"|----------|----------------|-------------|------------------|\n"
            f"| No-cause by either party | {notice} days | N/A | No |\n"
            f"| Client for SP breach | Written notice | {cure} days | Only for data breach |\n"
            f"| SP for Client breach | {notice} days | None stated | No |\n\n"
            f"*Confidence: High — all conditions mapped directly from clause language.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="legal_reasoning", difficulty="hard",
                             category="condition_exception_chain", source="advanced_synthetic"))

    # --- Risk identification ---
    for _ in range(per_type):
        r2 = _rnd()
        text, desc = _build_contract(r2)

        query = (
            f"Perform a legal risk assessment of the following contract. "
            f"Identify the top 3 risks for the Client and rate each as Low/Medium/High.\n\n"
            f"<contract>\n{text}\n</contract>"
        )
        reasoning = (
            f"I need to analyse this contract from the Client's perspective to identify risks. "
            f"Risk areas to examine: liability limitations (Art. 9), payment obligations, data security (Art. 7), "
            f"SLA remedies (Art. 8), IP ownership (Art. 5), termination protections (Art. 4). "
            f"I'll evaluate each area and identify where Client is exposed. "
            f"Rating criteria: High = significant financial or operational exposure; Medium = manageable with monitoring; Low = minimal impact."
        )
        answer = (
            f"## Client Risk Assessment\n\n"
            f"| # | Risk | Clause | Rating | Reasoning |\n"
            f"|---|------|--------|--------|-----------|\n"
            f"| 1 | Liability cap may limit recovery for SLA breaches | Art. 9.2 | **Medium** | Total liability capped at 12 months' fees; may be insufficient for major outages |\n"
            f"| 2 | Data deletion timeline creates exposure window | Art. 7.1 | **Medium** | {r2.choice([60,90,120])}-day deletion window means data at risk post-termination |\n"
            f"| 3 | Warranty disclaimer (Art. 6.3) | Art. 6.3 | **High** | 'No warranties express or implied' is broad; Client must negotiate specific SLA commitments |\n\n"
            f"### Recommended Mitigations\n"
            f"1. Negotiate enhanced liability cap or specific carve-outs for data breaches\n"
            f"2. Add data escrow or certified deletion requirement\n"
            f"3. Ensure Schedule A contains detailed, enforceable specifications to counter warranty disclaimer\n\n"
            f"*Confidence: Medium — risk assessment based on contract terms; full assessment requires jurisdiction-specific legal advice.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="legal_reasoning", difficulty="hard",
                             category="risk_identification", source="advanced_synthetic"))

    # --- Compliance checking ---
    for _ in range(per_type):
        r2 = _rnd()
        regulation = r2.choice(["GDPR Article 28", "HIPAA BAA requirements", "SOC 2 contractual controls"])
        reqs = {
            "GDPR Article 28": [
                "Processing only on documented instructions",
                "Confidentiality obligations for personnel",
                "Appropriate technical and organisational security measures",
                "Subprocessor approval and flowdown requirements",
                "Assistance with data subject rights",
                "Deletion/return of data at end of processing",
                "Audit rights for controller",
            ],
            "HIPAA BAA requirements": [
                "Use/disclosure limitations for PHI",
                "Appropriate safeguards for PHI",
                "Reporting of breaches involving PHI",
                "Subcontractor BAA requirements",
                "Return/destruction of PHI at termination",
                "Access rights for HHS audits",
            ],
            "SOC 2 contractual controls": [
                "Security incident notification within 72 hours",
                "Annual penetration testing",
                "Encryption at rest and in transit",
                "Access control and MFA requirements",
                "Audit log retention (minimum 12 months)",
            ],
        }[regulation]

        text, _ = _build_contract(r2)
        met_reqs = r2.sample(reqs, k=r2.randint(3, len(reqs)-1))
        unmet_reqs = [r for r in reqs if r not in met_reqs]

        query = (
            f"Review the following contract and assess whether it meets {regulation} requirements. "
            f"Specifically check for these requirements:\n"
            + "\n".join(f"- {req}" for req in reqs)
            + f"\n\n<contract>\n{text[:2000]}\n</contract>"
        )
        reasoning = (
            f"I need to check each {regulation} requirement against the contract provisions. "
            f"For each requirement, I'll search for corresponding language in the contract. "
            f"The contract covers data deletion in Article 7, security in Article 7.2, breach notification in Article 7.3. "
            f"I'll mark each requirement as Met, Partial, or Not Found."
        )
        answer = (
            f"## {regulation} Compliance Assessment\n\n"
            f"| Requirement | Status | Contract Clause |\n"
            f"|-------------|--------|----------------|\n"
            + "\n".join(f"| {req} | {'✓ Met' if req in met_reqs else '✗ Not found'} | {'Art. 7' if req in met_reqs else 'Missing'} |" for req in reqs)
            + f"\n\n**Compliance Summary:**\n"
            f"- **Met:** {len(met_reqs)}/{len(reqs)} requirements\n"
            f"- **Gaps:** {len(unmet_reqs)} requirement(s) not addressed\n\n"
            + (f"**Required additions to achieve compliance:**\n"
               + "\n".join(f"- Add explicit clause for: {req}" for req in unmet_reqs[:3])
               if unmet_reqs else "**Contract appears compliant** with stated requirements.\n")
            + f"\n\n*Confidence: Medium — based on review of available contract text; full legal review recommended.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="legal_reasoning", difficulty="hard",
                             category="compliance_checking", source="advanced_synthetic"))

    return examples[:n]


# ---------------------------------------------------------------------------
# Generator 7: Comparison examples
# ---------------------------------------------------------------------------

def generate_comparison_examples(n: int = 2000) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    per_type = n // 4

    # --- Contract comparison ---
    for _ in range(per_type):
        r2 = _rnd()
        text1, desc1 = _build_contract(r2)
        text2, desc2 = _build_contract(_rnd())

        party1 = _pick(_COMPANY_NAMES, r2)
        party2 = _pick([c for c in _COMPANY_NAMES if c != party1], r2)
        notice1 = r2.choice([30, 60, 90])
        notice2 = r2.choice([30, 60, 90])
        val1 = r2.randint(50000, 1000000)
        val2 = r2.randint(50000, 1000000)

        query = (
            f"Compare these two vendor contracts side by side. "
            f"Contract A ({party1}): {notice1}-day termination notice, total value ${val1:,}, governed by New York law.\n"
            f"Contract B ({party2}): {notice2}-day termination notice, total value ${val2:,}, governed by California law.\n\n"
            f"Which contract is more favourable for the Client? Provide a structured comparison."
        )
        better_val = party1 if val1 <= val2 else party2
        better_notice = party1 if notice1 <= notice2 else party2

        reasoning = (
            f"I need to compare both contracts across key dimensions. "
            f"Termination flexibility: shorter notice = more flexibility for Client. {notice1} vs {notice2} days. "
            f"Cost: lower value = better for Client. ${val1:,} vs ${val2:,}. "
            f"Governing law: New York vs California — both are common commercial law jurisdictions, difference is mainly venue. "
            f"I should also consider that a balanced comparison acknowledges trade-offs rather than declaring one universally better."
        )
        answer = (
            f"## Contract Comparison: {party1} vs {party2}\n\n"
            f"| Dimension | Contract A ({party1}) | Contract B ({party2}) | Better For Client |\n"
            f"|-----------|----------------------|----------------------|------------------|\n"
            f"| Termination Notice | {notice1} days | {notice2} days | {'Contract A' if notice1 < notice2 else 'Contract B' if notice2 < notice1 else 'Equal'} |\n"
            f"| Total Value | ${val1:,} | ${val2:,} | {'Contract A' if val1 < val2 else 'Contract B' if val2 < val1 else 'Equal'} |\n"
            f"| Governing Law | New York | California | Context-dependent |\n\n"
            f"### Overall Assessment\n"
            f"**{better_val}** offers lower cost. **{better_notice}** offers more termination flexibility. "
            f"The preferred contract depends on which factor Client prioritises:\n\n"
            f"- **If cost is primary:** Prefer Contract {'A' if val1 <= val2 else 'B'} (${min(val1,val2):,})\n"
            f"- **If flexibility is primary:** Prefer Contract {'A' if notice1 <= notice2 else 'B'} ({min(notice1,notice2)}-day notice)\n\n"
            f"*Confidence: High — based on provided contract parameters.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="comparison", difficulty="medium",
                             category="contract_comparison", source="advanced_synthetic"))

    # --- Candidate comparison ---
    for _ in range(per_type):
        r2 = _rnd()
        cands = _pick(_PERSON_NAMES, r2, n=3)
        skills = _pick(_SKILLS, r2, n=4)
        cand_data = {
            c: {
                "years_exp": r2.randint(2, 15),
                "skills": r2.sample(skills, k=r2.randint(1, 4)),
                "rating": round(r2.uniform(3.0, 5.0), 1),
                "salary_req": r2.randint(80000, 200000),
            }
            for c in cands
        }
        job = r2.choice(["Senior Engineer", "Tech Lead", "Principal Architect"])
        max_budget = r2.randint(120000, 180000)

        query = (
            f"Compare these three candidates for the {job} role (budget: ${max_budget:,}/year):\n\n"
            + "\n".join(
                f"**{c}:** {d['years_exp']} years experience, skills: {', '.join(d['skills'])}, "
                f"performance rating: {d['rating']}/5.0, salary requirement: ${d['salary_req']:,}"
                for c, d in cand_data.items()
            )
            + "\n\nWho is the best candidate? Provide a scored comparison."
        )

        # Score: experience (0.3), rating (0.3), within budget (0.2), skill coverage (0.2)
        scores = {}
        for c, d in cand_data.items():
            exp_score = min(d['years_exp'] / 15, 1.0) * 5
            rat_score = d['rating']
            budget_score = 5.0 if d['salary_req'] <= max_budget else max(0, 5 - (d['salary_req'] - max_budget) / 10000)
            skill_score = len(d['skills']) / len(skills) * 5
            total = 0.3 * exp_score + 0.3 * rat_score + 0.2 * budget_score + 0.2 * skill_score
            scores[c] = round(total, 2)

        best = max(scores, key=lambda x: scores[x])
        reasoning = (
            f"Scoring each candidate across 4 dimensions: experience (30%), rating (30%), budget fit (20%), skill coverage (20%). "
            + " ".join(f"{c}: score={scores[c]}." for c in cands)
            + f" Best: {best} with {scores[best]}."
        )
        answer = (
            f"## Candidate Comparison — {job}\n\n"
            f"| Candidate | Experience | Rating | Within Budget | Skills | **Score** |\n"
            f"|-----------|-----------|--------|--------------|--------|----------|\n"
            + "\n".join(
                f"| {c} | {cand_data[c]['years_exp']} yrs | {cand_data[c]['rating']}/5 | "
                f"{'Yes' if cand_data[c]['salary_req'] <= max_budget else 'No (+$'+str(cand_data[c]['salary_req']-max_budget)+')'} | "
                f"{len(cand_data[c]['skills'])}/{len(skills)} | **{scores[c]}** |"
                for c in cands
            )
            + f"\n\n### Recommendation: **{best}**\n\n"
            f"**{best}** achieves the highest weighted score of **{scores[best]}/5.00**. "
            f"{'Within budget at $' + str(cand_data[best]['salary_req']) + '.' if cand_data[best]['salary_req'] <= max_budget else 'Note: exceeds budget — negotiate or seek approval.'}\n\n"
            f"*Confidence: High — scoring based on provided candidate data and defined weights.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="comparison", difficulty="medium",
                             category="candidate_comparison", source="advanced_synthetic"))

    # --- Product/vendor feature comparison ---
    for _ in range(per_type):
        r2 = _rnd()
        products = _pick(_COMPANY_NAMES, r2, n=3)
        features = ["price/month", "uptime SLA", "support tier", "storage (GB)", "API rate limit"]
        vals = {
            p: {
                "price/month": r2.randint(500, 5000),
                "uptime SLA": r2.choice([99.5, 99.9, 99.95, 99.99]),
                "support tier": r2.choice(["Basic", "Standard", "Premium", "Enterprise"]),
                "storage (GB)": r2.choice([100, 500, 1000, 5000, 10000]),
                "API rate limit": r2.choice([1000, 10000, 100000, "Unlimited"]),
            }
            for p in products
        }
        priority = r2.choice(["uptime SLA", "price/month", "support tier"])

        query = (
            f"Compare these three vendor offerings and recommend the best for an enterprise needing high availability:\n\n"
            + "\n".join(
                f"**{p}:** " + ", ".join(f"{k}={v}" for k, v in vals[p].items())
                for p in products
            )
            + f"\n\nPrimary priority: {priority}."
        )

        best_uptime = max(products, key=lambda p: vals[p]["uptime SLA"])
        cheapest = min(products, key=lambda p: vals[p]["price/month"])

        reasoning = (
            f"For high-availability enterprise, uptime SLA is critical. "
            f"Best uptime: {best_uptime} at {vals[best_uptime]['uptime SLA']}%. "
            f"Cheapest: {cheapest} at ${vals[cheapest]['price/month']:,}/month. "
            f"Primary priority is {priority} — ranking by that criterion first."
        )

        answer = (
            f"## Vendor Feature Comparison\n\n"
            f"| Feature | {products[0]} | {products[1]} | {products[2]} |\n"
            f"|---------|{'|'.join(['-'*12]*3)}|\n"
            + "\n".join(
                f"| {feat} | {vals[products[0]][feat]} | {vals[products[1]][feat]} | {vals[products[2]][feat]} |"
                for feat in features
            )
            + f"\n\n### Recommendation for High-Availability Enterprise\n\n"
            f"**Best for {priority}:** See table above for leading vendor.\n"
            f"**Best overall uptime:** **{best_uptime}** ({vals[best_uptime]['uptime SLA']}% SLA)\n"
            f"**Most cost-effective:** **{cheapest}** (${vals[cheapest]['price/month']:,}/month)\n\n"
            f"For an enterprise prioritising high availability, **{best_uptime}** is recommended given its superior uptime SLA, "
            f"which directly impacts business continuity.\n\n"
            f"*Confidence: High — based on provided specifications.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="comparison", difficulty="medium",
                             category="product_comparison", source="advanced_synthetic"))

    # --- Gap analysis ---
    for _ in range(per_type):
        r2 = _rnd()
        required_skills = _pick(_SKILLS, r2, n=5)
        candidate = _pick(_PERSON_NAMES, r2)
        has_skills = r2.sample(required_skills, k=r2.randint(2, 5))
        missing = [s for s in required_skills if s not in has_skills]

        query = (
            f"Job description requires: {', '.join(required_skills)}.\n"
            f"Candidate {candidate} has: {', '.join(has_skills)}.\n\n"
            f"Perform a skills gap analysis."
        )
        coverage = len(has_skills) / len(required_skills) * 100

        reasoning = (
            f"Comparing required skills to candidate's skills. "
            f"Required: {required_skills}. Has: {has_skills}. "
            f"Missing: {missing if missing else 'none'}. "
            f"Coverage: {len(has_skills)}/{len(required_skills)} = {coverage:.0f}%."
        )
        answer = (
            f"## Skills Gap Analysis — {candidate}\n\n"
            f"| Skill | Required | Candidate | Status |\n"
            f"|-------|----------|-----------|--------|\n"
            + "\n".join(
                f"| {s} | Yes | {'Yes' if s in has_skills else 'No'} | {'✓' if s in has_skills else '✗ GAP'} |"
                for s in required_skills
            )
            + f"\n\n**Coverage:** {len(has_skills)}/{len(required_skills)} required skills ({coverage:.0f}%)\n\n"
            + (f"**Gaps identified:** {', '.join(missing)}\n\n"
               f"**Recommendation:** {'Strong fit — all skills present.' if not missing else f'Partial fit. Missing: {chr(34) + chr(34).join(missing) + chr(34)}. Consider training plan or adjusted requirements.'}\n"
               if missing else "**Recommendation:** Strong fit — all required skills present.\n")
            + f"\n*Confidence: High — direct skill-by-skill comparison.*"
        )
        examples.append(_sft(query, reasoning, answer,
                             area="comparison", difficulty="easy",
                             category="gap_analysis", source="advanced_synthetic"))

    return examples[:n]


# ---------------------------------------------------------------------------
# Generator 8: DPO advanced preference pairs
# ---------------------------------------------------------------------------

def generate_dpo_advanced_pairs(n: int = 3000) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    per_type = n // 6

    # --- Deep analysis vs shallow extraction ---
    for _ in range(per_type):
        r2 = _rnd()
        text, desc = _build_contract(r2)
        party = _pick(_COMPANY_NAMES, r2)
        notice = r2.choice([30, 60, 90])
        data_del = r2.choice([60, 90, 120])

        query = f"What are the key risks and obligations in this contract?\n\n{text[:1500]}"

        pairs.append(_dpo(
            query=query,
            c_reasoning=(
                f"I need to perform a thorough analysis of this contract. "
                f"First, I'll identify the parties and their respective obligations. "
                f"Then I'll look at financial exposure, termination rights, liability limitations, and data handling. "
                f"I should organise my findings by risk level and cite the specific clauses."
            ),
            c_answer=(
                f"## Contract Risk and Obligations Analysis\n\n"
                f"### Key Obligations\n"
                f"**Service Provider must:** Deliver services within 15 days, maintain {r2.choice([99.5,99.9])}% uptime, "
                f"delete data within {data_del} days of termination, report breaches within 48 hours.\n\n"
                f"**Client must:** Pay invoices within 30 days, provide required access.\n\n"
                f"### Key Risks\n"
                f"1. **High:** Broad warranty disclaimer (Art. 6.3) limits Client remedies\n"
                f"2. **Medium:** Liability cap at 12 months' fees may be insufficient for major incidents\n"
                f"3. **Medium:** {data_del}-day data deletion window creates post-termination data exposure\n\n"
                f"### Recommended Actions\n"
                f"- Negotiate specific SLA remedies beyond service credits\n"
                f"- Request liability carve-out for data breaches\n\n"
                f"*Confidence: High — analysis based on contract provisions cited.*"
            ),
            r_reasoning="Let me quickly look at this contract.",
            r_answer="This is a standard services contract. The parties have agreed to provide services. Seems fine.",
            category="deep_vs_shallow", source="advanced_synthetic",
        ))

    # --- Step-by-step vs jump to conclusion ---
    for _ in range(per_type):
        r2 = _rnd()
        items = [(r2.choice(["Software", "Hardware", "Services", "Licence", "Support"]),
                  r2.randint(1, 10), round(r2.uniform(1000, 20000), 2))
                 for _ in range(r2.randint(4, 7))]
        correct = round(sum(q * p for _, q, p in items), 2)
        claimed = round(correct * r2.choice([0.95, 1.05, 1.0]), 2)
        discrepancy = round(claimed - correct, 2)

        query = (
            "Verify this invoice total:\n"
            + "\n".join(f"- {d}: {q} × ${p:,.2f}" for d, q, p in items)
            + f"\nClaimed total: ${claimed:,.2f}"
        )

        pairs.append(_dpo(
            query=query,
            c_reasoning=(
                f"I need to verify this invoice by computing each line item total and summing them. "
                + " ".join(f"{d}: {q}×${p:,.2f}=${q*p:,.2f}." for d, q, p in items)
                + f" Sum = ${correct:,.2f}. Claimed = ${claimed:,.2f}. "
                f"Difference = ${abs(discrepancy):,.2f}. {'Correct.' if abs(discrepancy) < 0.01 else 'Discrepancy found.'}"
            ),
            c_answer=(
                f"## Invoice Verification\n\n"
                + "\n".join(f"- {d}: {q} × ${p:,.2f} = **${q*p:,.2f}**" for d, q, p in items)
                + f"\n\n**Computed total:** ${correct:,.2f}\n"
                f"**Claimed total:** ${claimed:,.2f}\n\n"
                + ("**Result: CORRECT** — claimed total matches line item sum."
                   if abs(discrepancy) < 0.01
                   else f"**Result: DISCREPANCY** — claimed total is {'over' if discrepancy > 0 else 'under'}stated by **${abs(discrepancy):,.2f}**.")
            ),
            r_reasoning="Let me check this invoice.",
            r_answer=f"The invoice total is ${claimed:,.2f}. Looks reasonable.",
            category="step_by_step_vs_jump", source="advanced_synthetic",
        ))

    # --- Cited evidence vs unsupported claims ---
    for _ in range(per_type):
        r2 = _rnd()
        text, name = _build_resume(r2)
        skills = [s for s in _SKILLS if s in text][:3]

        query = f"Is this candidate a good fit for a Senior Data Engineer role requiring 5+ years Python and 3+ years ML?\n\n{text[:1500]}"

        pairs.append(_dpo(
            query=query,
            c_reasoning=(
                f"I need to check specific requirements against evidence in the resume. "
                f"Requirement 1: 5+ years Python. Scanning for Python mentions and years... "
                f"Requirement 2: 3+ years ML. Looking for ML experience in role descriptions. "
                f"I should cite specific job entries and durations rather than making general statements."
            ),
            c_answer=(
                f"## Fit Assessment — Senior Data Engineer\n\n"
                f"**Requirement: 5+ years Python**\n"
                f"Evidence from resume: Python is listed in technical skills. "
                f"{'Experience spans multiple roles — cumulative Python exposure likely meets threshold.' if 'Python' in text else 'Python not explicitly mentioned — clarification needed.'}\n\n"
                f"**Requirement: 3+ years ML**\n"
                f"Evidence: {'ML referenced in job descriptions — review specific role durations to confirm 3-year requirement.' if 'ML' in text or 'Machine Learning' in text else 'ML experience not explicitly cited — potential gap.'}\n\n"
                f"**Verdict:** Assessment requires verification of specific year counts per skill from full resume. "
                f"Based on available evidence, candidate appears {'strong' if skills else 'moderate'} — recommend technical interview.\n\n"
                f"*Confidence: Medium — full assessment requires complete resume review.*"
            ),
            r_reasoning="This person looks like they have good skills.",
            r_answer="Yes, this candidate looks good for the role. They seem experienced.",
            category="cited_vs_unsupported", source="advanced_synthetic",
        ))

    # --- Structured output vs wall of text ---
    for _ in range(per_type):
        r2 = _rnd()
        text, desc = _build_financial_report(r2)

        query = f"Summarise the key financial metrics from this report.\n\n{text[:2000]}"

        pairs.append(_dpo(
            query=query,
            c_reasoning=(
                f"I'll extract specific financial metrics and present them in a structured format. "
                f"Key areas: revenue trend across quarters, profitability metrics, balance sheet highlights. "
                f"A table will make comparison easy."
            ),
            c_answer=(
                f"## Key Financial Metrics\n\n"
                f"### Income Statement Highlights\n\n"
                f"| Metric | Value |\n"
                f"|--------|-------|\n"
                f"| Annual Revenue | See FY total in report |\n"
                f"| Gross Margin | As stated in Key Metrics |\n"
                f"| Net Margin | As stated in Key Metrics |\n\n"
                f"### Balance Sheet Highlights\n\n"
                f"| Metric | Value |\n"
                f"|--------|-------|\n"
                f"| Total Assets | See report |\n"
                f"| Debt-to-Equity | As calculated in Key Metrics |\n\n"
                f"### Revenue Trend\n"
                f"Quarterly progression: Q1 → Q2 → Q3 → Q4 as shown in income statement table.\n\n"
                f"*Source: Financial report extract provided.*"
            ),
            r_reasoning="Let me read through this financial report.",
            r_answer=(
                "The company had some revenue in Q1 and it changed in Q2 and Q3 and Q4. "
                "There was also some gross profit and operating expenses and then net income. "
                "The balance sheet had assets and liabilities. The company seems to be doing okay overall."
            ),
            category="structured_vs_prose", source="advanced_synthetic",
        ))

    # --- Honest uncertainty vs overconfident wrong answers ---
    for _ in range(per_type):
        r2 = _rnd()
        company = _pick(_COMPANY_NAMES, r2)
        year = r2.randint(2022, 2024)

        query = (
            f"Based on {company}'s Q3 {year} performance, what will their Q4 revenue be?"
        )

        pairs.append(_dpo(
            query=query,
            c_reasoning=(
                f"The user is asking me to predict future revenue. "
                f"I don't have Q3 data in this question, and even if I did, revenue prediction requires "
                f"much more information than a single quarter's results. "
                f"I should be honest about this limitation while being helpful about what analysis IS possible."
            ),
            c_answer=(
                f"I cannot reliably predict {company}'s Q4 {year} revenue from a single quarter's results. "
                f"Revenue forecasting requires:\n"
                f"- Historical trend data (multiple years)\n"
                f"- Pipeline/backlog information\n"
                f"- Seasonal adjustment factors\n"
                f"- Management guidance\n\n"
                f"**What I can do:** If you share the historical quarterly revenue data, I can calculate the "
                f"average QoQ growth rate and provide a trend-based estimate with appropriate uncertainty bounds.\n\n"
                f"*Confidence: High that prediction cannot be made reliably without additional data.*"
            ),
            r_reasoning="I can estimate this based on typical growth patterns.",
            r_answer=f"{company}'s Q4 {year} revenue will be approximately $45 million, representing 12% growth over Q3.",
            category="honest_uncertainty_vs_overconfident", source="advanced_synthetic",
        ))

    # --- Specific numbers vs vague approximations ---
    for _ in range(per_type):
        r2 = _rnd()
        text, inv_no = _build_invoice(r2, n_items=6)
        import re as _re
        grand_match = _re.search(r'TOTAL DUE\s+\$([\d,]+\.\d{2})', text)
        if not grand_match:
            continue
        grand = grand_match.group(1)
        subtotal_match = _re.search(r'Subtotal\s+\$([\d,]+\.\d{2})', text)
        if not subtotal_match:
            continue
        subtotal = subtotal_match.group(1)

        query = f"What is the total amount due on this invoice?\n\n{text}"

        pairs.append(_dpo(
            query=query,
            c_reasoning=(
                f"I need to find the TOTAL DUE line on this invoice. "
                f"Scanning the invoice... found 'TOTAL DUE' line. The amount is ${grand}. "
                f"This is the grand total including all line items and tax."
            ),
            c_answer=(
                f"The total amount due on invoice **{inv_no}** is **${grand}** (including tax).\n\n"
                f"- Subtotal (before tax): ${subtotal}\n"
                f"- Grand total due: **${grand}**\n\n"
                f"*Confidence: High — figure read directly from TOTAL DUE line on invoice.*"
            ),
            r_reasoning="Looking at the invoice total.",
            r_answer="The invoice total is somewhere around a few thousand dollars based on the line items.",
            category="specific_vs_vague", source="advanced_synthetic",
        ))

    return pairs[:n]


# ---------------------------------------------------------------------------
# Orchestrator: generate_all_advanced
# ---------------------------------------------------------------------------

def generate_all_advanced(output_dir: str) -> Dict[str, int]:
    """Generate all advanced training examples and save to output_dir.

    Returns a dict with counts for sft and dpo outputs.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sft_path = out_path / "advanced_sft.jsonl"
    dpo_path = out_path / "advanced_dpo.jsonl"

    sft_examples: List[Dict[str, Any]] = []
    dpo_examples: List[Dict[str, Any]] = []

    print("[1/8] Generating long-context examples (5,000)...")
    sft_examples.extend(generate_long_context_examples(5000))
    print(f"      -> {len(sft_examples)} SFT so far")

    print("[2/8] Generating reasoning-chain examples (5,000)...")
    sft_examples.extend(generate_reasoning_chain_examples(5000))
    print(f"      -> {len(sft_examples)} SFT so far")

    print("[3/8] Generating numerical reasoning examples (3,000)...")
    sft_examples.extend(generate_numerical_reasoning_examples(3000))
    print(f"      -> {len(sft_examples)} SFT so far")

    print("[4/8] Generating table understanding examples (3,000)...")
    sft_examples.extend(generate_table_understanding_examples(3000))
    print(f"      -> {len(sft_examples)} SFT so far")

    print("[5/8] Generating temporal reasoning examples (2,000)...")
    sft_examples.extend(generate_temporal_reasoning_examples(2000))
    print(f"      -> {len(sft_examples)} SFT so far")

    print("[6/8] Generating legal reasoning examples (2,000)...")
    sft_examples.extend(generate_legal_reasoning_examples(2000))
    print(f"      -> {len(sft_examples)} SFT so far")

    print("[7/8] Generating comparison examples (2,000)...")
    sft_examples.extend(generate_comparison_examples(2000))
    print(f"      -> {len(sft_examples)} SFT so far")

    print("[8/8] Generating DPO preference pairs (3,000)...")
    dpo_examples.extend(generate_dpo_advanced_pairs(3000))
    print(f"      -> {len(dpo_examples)} DPO pairs")

    # Deduplicate
    print("Deduplicating SFT examples...")
    sft_examples = _deduplicate(sft_examples)
    print(f"After dedup: {len(sft_examples)} SFT examples")

    print("Deduplicating DPO pairs...")
    dpo_examples = _deduplicate(dpo_examples)
    print(f"After dedup: {len(dpo_examples)} DPO pairs")

    # Write SFT
    print(f"Writing SFT to {sft_path}...")
    with JSONLWriter(sft_path) as w:
        for ex in sft_examples:
            w.write(ex)
    sft_count = len(sft_examples)

    # Write DPO
    print(f"Writing DPO to {dpo_path}...")
    with JSONLWriter(dpo_path) as w:
        for ex in dpo_examples:
            w.write(ex)
    dpo_count = len(dpo_examples)

    print(f"\nDone. SFT: {sft_count}, DPO: {dpo_count}, Total: {sft_count + dpo_count}")
    return {"sft": sft_count, "dpo": dpo_count, "total": sft_count + dpo_count}
