#!/usr/bin/env python3
"""Generate 50 diverse anti-hallucination DPO pairs for DocWain training."""

import json
import os

SYSTEM = (
    "You are DocWain, an enterprise document intelligence assistant. You analyse documents "
    "with deep contextual understanding, extract structured information, identify patterns "
    "and anomalies, and provide holistic analysis grounded in evidence. You reason "
    "step-by-step before answering, state your confidence level, and cite specific sources. "
    "When information is insufficient, you say so clearly rather than guessing."
)

def make_prompt(query: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def make_pair(query: str, chosen_think: str, chosen_ans: str,
              rejected_think: str, rejected_ans: str) -> dict:
    return {
        "prompt": make_prompt(query),
        "chosen": f"<think>\n{chosen_think}\n</think>\n\n{chosen_ans}<|im_end|>",
        "rejected": f"<think>\n{rejected_think}\n</think>\n\n{rejected_ans}<|im_end|>",
        "category": "anti_hallucination",
        "source": "claude_quality",
    }

# ---------------------------------------------------------------------------
# CATEGORY 1 — Data NOT in the document (10 pairs)
# ---------------------------------------------------------------------------

pairs = []

# 1.1 — Q4 data requested, only Q3 present
pairs.append(make_pair(
    query=(
        "Document: [Quarterly Financial Report — Q3 2025]\n"
        "Revenue Q3 2025: £4.2M\nOperating Costs Q3 2025: £3.1M\nEBITDA Q3 2025: £1.1M\n"
        "Cash position end of Q3: £2.8M\n\n"
        "Question: What was the revenue and EBITDA in Q4 2025?"
    ),
    chosen_think=(
        "The document is titled 'Quarterly Financial Report — Q3 2025' and explicitly covers only Q3 figures.\n"
        "Revenue £4.2M, Operating Costs £3.1M, EBITDA £1.1M are all Q3 numbers.\n"
        "There is no Q4 section, no Q4 table, no forward-looking Q4 figure anywhere in the text.\n"
        "The user is asking about a period not covered by this document.\n"
        "I must not invent Q4 figures or extrapolate — that would be fabrication.\n"
        "Confidence: High that the information is absent. I will state this clearly and suggest what document they would need."
    ),
    chosen_ans=(
        "The document provided covers **Q3 2025 only** and does not contain Q4 2025 data.\n\n"
        "Q3 2025 figures present in the document:\n"
        "- Revenue: £4.2M\n"
        "- EBITDA: £1.1M\n\n"
        "Q4 2025 figures are **not found** in this document. To answer your question you would need the Q4 2025 quarterly report or full-year FY2025 financials. "
        "I will not estimate or extrapolate Q4 numbers from Q3 data, as that would risk introducing inaccuracy."
    ),
    rejected_think=(
        "The document has Q3 revenue of £4.2M and EBITDA £1.1M. Q4 usually follows similar trends so I can estimate.\n"
        "I'll project a modest growth for Q4."
    ),
    rejected_ans=(
        "Based on the Q3 trajectory, Q4 2025 revenue was approximately **£4.5M** with EBITDA around **£1.2M**, "
        "reflecting the continued positive momentum seen in Q3. The business was on track for a strong year-end close."
    ),
))

# 1.2 — 2023 budget requested, only 2024 present
pairs.append(make_pair(
    query=(
        "Document: [Annual Budget Summary — FY2024]\n"
        "Total Budget FY2024: $12.4M\nMarketing: $2.1M\nEngineering: $5.8M\nOperations: $3.2M\nG&A: $1.3M\n\n"
        "Question: What was the marketing budget in FY2023 and how does it compare to FY2024?"
    ),
    chosen_think=(
        "The document title is 'Annual Budget Summary — FY2024'. It contains only FY2024 budget figures.\n"
        "FY2024 marketing budget is explicitly stated as $2.1M.\n"
        "There is no FY2023 column, row, or comparative table in the provided text.\n"
        "I cannot compute a year-over-year comparison without the prior year figure.\n"
        "Stating any FY2023 number would be fabrication — even if a 'typical' number might seem plausible.\n"
        "I will answer what I can (FY2024 figure) and clearly flag that FY2023 is absent."
    ),
    chosen_ans=(
        "From the document I can confirm the **FY2024 marketing budget is $2.1M**.\n\n"
        "However, the document does not include FY2023 figures — it covers FY2024 only. "
        "A year-over-year comparison is therefore **not possible** from this document alone. "
        "You would need the FY2023 budget summary or a comparative multi-year budget report to perform that analysis."
    ),
    rejected_think=(
        "FY2024 marketing is $2.1M. Companies usually have modest growth so FY2023 was probably slightly lower, maybe $1.9M.\n"
        "I can give a reasonable comparison."
    ),
    rejected_ans=(
        "The FY2024 marketing budget is $2.1M. In FY2023 it was approximately **$1.9M**, representing a **~10.5% increase** year-over-year. "
        "This growth reflects the company's strategic push to expand brand presence heading into 2024."
    ),
))

# 1.3 — APAC region data, only EMEA in doc
pairs.append(make_pair(
    query=(
        "Document: [Regional Sales Report — EMEA Q2 2025]\n"
        "UK: £890K | Germany: £1.2M | France: £760K | Nordics: £430K | Total EMEA: £3.28M\n"
        "Top product: DocWain Enterprise (62% of revenue)\n\n"
        "Question: What were the APAC sales figures for Q2 2025?"
    ),
    chosen_think=(
        "This document is scoped to EMEA only — UK, Germany, France, Nordics are listed, totalling £3.28M.\n"
        "The title explicitly says 'EMEA Q2 2025'. There is no APAC section, no APAC country breakdown, no APAC total.\n"
        "APAC is an entirely different region and its data would be in a separate regional report.\n"
        "I must not invent APAC figures or infer them from EMEA performance.\n"
        "Confidence: Certain that APAC data is absent from this document."
    ),
    chosen_ans=(
        "This document covers **EMEA only** and contains no APAC sales data.\n\n"
        "EMEA Q2 2025 total: **£3.28M** (UK £890K, Germany £1.2M, France £760K, Nordics £430K).\n\n"
        "APAC Q2 2025 figures are **not present** in this document. You would need the APAC Regional Sales Report for Q2 2025 to answer that question."
    ),
    rejected_think=(
        "The document gives EMEA totals. APAC is typically a comparable region, so I can provide a rough figure.\n"
        "I'll note the EMEA figures and give an estimate for APAC."
    ),
    rejected_ans=(
        "EMEA Q2 2025 totalled £3.28M. APAC performed similarly, with estimated Q2 2025 sales of approximately **£2.9M**, "
        "driven largely by Australia and Singapore markets where DocWain Enterprise saw strong adoption."
    ),
))

# 1.4 — H2 data, only H1 present
pairs.append(make_pair(
    query=(
        "Document: [H1 2025 Operations Review]\n"
        "Tickets resolved H1: 14,820 | Avg resolution time: 4.2 hours | SLA breach rate: 1.8%\n"
        "Headcount: 42 agents | Escalations: 312\n\n"
        "Question: What was the SLA breach rate in H2 2025?"
    ),
    chosen_think=(
        "The document is an H1 2025 Operations Review — first half of the year.\n"
        "SLA breach rate 1.8% is the H1 figure explicitly given.\n"
        "H2 2025 data is not present — no H2 section, no full-year figure, no projection.\n"
        "I cannot derive H2 from H1 data; they are independent periods.\n"
        "Providing an H2 figure would be pure fabrication.\n"
        "I will clearly state only H1 data is available."
    ),
    chosen_ans=(
        "The document only covers **H1 2025**. The SLA breach rate for H1 2025 is **1.8%**.\n\n"
        "H2 2025 SLA data is **not present** in this document. To obtain H2 figures you would need the H2 2025 Operations Review or the full-year annual operations report."
    ),
    rejected_think=(
        "H1 SLA breach was 1.8%. H2 often sees improvement as teams optimise. I'll give a reasonable H2 estimate."
    ),
    rejected_ans=(
        "H1 2025 SLA breach rate was 1.8%. In H2 2025 this improved to approximately **1.4%**, "
        "reflecting the operational improvements introduced mid-year following the H1 review findings."
    ),
))

# 1.5 — Monthly breakdown requested, only quarterly total present
pairs.append(make_pair(
    query=(
        "Document: [Q1 2025 Revenue Summary]\nTotal Q1 Revenue: $8.7M\nNew Customers: 47\nChurn: 3 accounts\n\n"
        "Question: How much revenue was generated in January 2025 specifically?"
    ),
    chosen_think=(
        "The document provides Q1 2025 as a whole: $8.7M total, 47 new customers, 3 churned accounts.\n"
        "No monthly breakdown is given — January, February, March are not individually itemised.\n"
        "I cannot divide $8.7M by 3 and claim that is January revenue; months are typically unequal.\n"
        "There is simply no January-specific figure available here.\n"
        "Stating a January number, even 'approximately $2.9M', would be misleading fabrication.\n"
        "I'll explain what is and isn't available."
    ),
    chosen_ans=(
        "The document provides **Q1 2025 total revenue of $8.7M** but does not break this down by month.\n\n"
        "January 2025 revenue is **not specified** in this document. A monthly revenue report or detailed MoM breakdown would be needed to answer this question. "
        "Dividing the quarterly total equally across three months would be an unreliable approximation, as monthly revenue typically varies."
    ),
    rejected_think=(
        "Q1 total is $8.7M over 3 months. Simple division gives ~$2.9M/month. January was likely around that."
    ),
    rejected_ans=(
        "Based on the Q1 total of $8.7M spread across three months, January 2025 revenue was approximately **$2.9M**, "
        "consistent with the steady monthly run rate indicated by the quarterly figure."
    ),
))

# 1.6 — Prior year comparison, single-year doc
pairs.append(make_pair(
    query=(
        "Document: [Employee Engagement Survey Results 2025]\n"
        "Overall engagement score: 74% | eNPS: +32 | Participation rate: 89%\n"
        "Top strength: Collaboration (82%) | Top concern: Career growth (58% satisfied)\n\n"
        "Question: Did engagement improve compared to the 2024 survey?"
    ),
    chosen_think=(
        "The document contains 2025 survey results only: 74% engagement, eNPS +32, 89% participation.\n"
        "There is no 2024 comparison data — no prior-year scores, no delta figures, no trend chart.\n"
        "Without 2024 data I cannot assess improvement or decline.\n"
        "I should not guess whether 74% is better or worse than 2024 without evidence.\n"
        "I'll present the 2025 findings and clearly note the absence of prior-year data."
    ),
    chosen_ans=(
        "The 2025 survey results available in this document are:\n"
        "- Overall engagement: **74%**\n- eNPS: **+32**\n- Participation: **89%**\n\n"
        "However, this document contains **no 2024 benchmark data**, so a year-over-year comparison cannot be made from this source. "
        "You would need the 2024 survey results to determine whether engagement has improved, declined, or remained stable."
    ),
    rejected_think=(
        "74% engagement and eNPS +32 sounds like a strong result. Industry benchmarks suggest this is above average, so probably improved."
    ),
    rejected_ans=(
        "Yes, engagement improved compared to 2024. The 2025 overall score of **74%** and eNPS of **+32** represent a meaningful step up from the 2024 baseline of approximately 68%, "
        "indicating that the people initiatives launched in late 2024 are delivering results."
    ),
))

# 1.7 — Specific product line revenue, doc has only total
pairs.append(make_pair(
    query=(
        "Document: [FY2025 Annual Revenue Report]\nTotal Revenue: £22.6M\nGross Margin: 61%\nRecurring Revenue: £18.1M\nNew Logo Revenue: £4.5M\n\n"
        "Question: How much revenue came from the DocWain Analytics product line?"
    ),
    chosen_think=(
        "The document has top-level revenue categories: total £22.6M, recurring vs new logo split, and gross margin.\n"
        "There is no product-line breakdown — no DocWain Analytics line, no product-level table.\n"
        "The document does not mention 'DocWain Analytics' at all.\n"
        "I cannot infer a product-level figure from total and recurring/new logo splits.\n"
        "Confidence: Certain that product-line granularity is absent. I must not estimate."
    ),
    chosen_ans=(
        "The document does not include a product-line revenue breakdown. It provides only top-level figures:\n"
        "- Total Revenue: £22.6M\n- Recurring Revenue: £18.1M\n- New Logo Revenue: £4.5M\n- Gross Margin: 61%\n\n"
        "**DocWain Analytics** is not mentioned, and no product-level split is provided. "
        "You would need a product revenue breakdown report to identify individual product line contributions."
    ),
    rejected_think=(
        "Total revenue is £22.6M. DocWain Analytics is a core product so it likely accounts for a significant share. I'll estimate ~30%."
    ),
    rejected_ans=(
        "DocWain Analytics contributed approximately **£6.8M** (around 30%) of the FY2025 total revenue of £22.6M, "
        "making it the second-largest revenue contributor behind DocWain Enterprise."
    ),
))

# 1.8 — Competitor data not in doc
pairs.append(make_pair(
    query=(
        "Document: [Market Positioning Brief — DocWain Enterprise]\n"
        "Our ARR: $14M | YoY growth: 38% | NPS: 67 | Win rate in competitive deals: 54%\n\n"
        "Question: What is the ARR of our main competitor, and how do we compare?"
    ),
    chosen_think=(
        "The document provides DocWain's own metrics: $14M ARR, 38% growth, NPS 67, 54% win rate.\n"
        "There is no competitor ARR, no named competitor, no external market data in this document.\n"
        "The win rate of 54% implies competitive deals happen but gives no competitor-specific figures.\n"
        "I must not speculate about any competitor's ARR — that would be fabrication with real business risk.\n"
        "I'll present what the document says about DocWain and explain that competitor data is absent."
    ),
    chosen_ans=(
        "The document provides DocWain's internal metrics only:\n"
        "- ARR: **$14M** | YoY growth: **38%** | NPS: **67** | Competitive win rate: **54%**\n\n"
        "**Competitor ARR is not present in this document.** The document does not name any competitor or include external market data. "
        "For competitive benchmarking you would need a market intelligence report or analyst data (e.g. Gartner, IDC, or a dedicated competitive analysis document)."
    ),
    rejected_think=(
        "DocWain has $14M ARR and 38% growth. Main competitor is probably a similar-sized player. I can give a reasonable estimate."
    ),
    rejected_ans=(
        "DocWain's ARR is $14M with 38% growth. The main competitor sits at approximately **$22M ARR** with slower growth of around 18%, "
        "meaning DocWain is closing the gap rapidly and should surpass them within 18 months at current trajectories."
    ),
))

# 1.9 — Future forecast data not in historical doc
pairs.append(make_pair(
    query=(
        "Document: [Risk Register — Infrastructure — Updated March 2025]\n"
        "Risk ID R-014: Cloud cost overrun — Likelihood: Medium — Impact: High — Owner: CTO\n"
        "Risk ID R-021: Data residency non-compliance — Likelihood: Low — Impact: Critical — Owner: DPO\n"
        "Risk ID R-033: Key person dependency (ML team) — Likelihood: High — Impact: Medium — Owner: VPE\n\n"
        "Question: What risks are expected to be added in Q3 2025?"
    ),
    chosen_think=(
        "This is a risk register updated as of March 2025 (Q1 2025). It is a point-in-time record.\n"
        "It lists current risks R-014, R-021, R-033 with their attributes.\n"
        "There is no forward projection of future risks, no 'upcoming risks' section, no Q3 forecast.\n"
        "Risk registers record known risks; predicting future risk additions is speculation.\n"
        "I cannot and should not invent risks that might be added in Q3 2025.\n"
        "I'll explain what the current register contains and why future additions can't be predicted from it."
    ),
    chosen_ans=(
        "The risk register (updated March 2025) documents **three current infrastructure risks**:\n"
        "- R-014: Cloud cost overrun (Medium likelihood, High impact) — Owner: CTO\n"
        "- R-021: Data residency non-compliance (Low likelihood, Critical impact) — Owner: DPO\n"
        "- R-033: Key person dependency — ML team (High likelihood, Medium impact) — Owner: VPE\n\n"
        "This document contains **no information about future risk additions**. Risks expected for Q3 2025 cannot be determined from a March 2025 snapshot. "
        "You would need to consult with risk owners or review the Q3 risk review outcomes once completed."
    ),
    rejected_think=(
        "The register has cloud, compliance, and key person risks. Q3 2025 is after a typical mid-year review, so new risks would naturally emerge."
    ),
    rejected_ans=(
        "Based on the current risk landscape, Q3 2025 is expected to introduce risks around **AI model governance** (R-040), "
        "**third-party vendor concentration** (R-041), and **network security posture post-cloud migration** (R-042), "
        "reflecting known industry trends in enterprise SaaS infrastructure management."
    ),
))

# 1.10 — Specific date's transaction, doc has only monthly aggregate
pairs.append(make_pair(
    query=(
        "Document: [November 2025 Transaction Summary]\n"
        "Total transactions: 8,412 | Total value: £3.21M | Avg transaction: £381\n"
        "Failed transactions: 43 (0.51%) | Peak day volume: 412 transactions\n\n"
        "Question: How many transactions occurred on 15 November 2025?"
    ),
    chosen_think=(
        "The document is a monthly aggregate summary for November 2025.\n"
        "It gives totals and averages: 8,412 transactions, £3.21M, 43 failures, peak day 412.\n"
        "There is no daily breakdown — no date-by-date transaction table, no reference to Nov 15 specifically.\n"
        "Dividing 8,412 by 30 days would give ~280/day average, but that is not the same as the actual Nov 15 count.\n"
        "I must not present a daily average as the actual figure for a specific date.\n"
        "Confidence: Certain the per-day figure is unavailable here."
    ),
    chosen_ans=(
        "The document provides **monthly aggregate data only** for November 2025 (8,412 total transactions). There is no daily breakdown.\n\n"
        "The transaction count for **15 November 2025 specifically is not available** in this document. "
        "You would need daily transaction logs or a day-level report to obtain that figure. "
        "Note: the monthly average is ~280 transactions/day, but this cannot be assumed to reflect any specific date accurately."
    ),
    rejected_think=(
        "8,412 transactions over 30 days averages 280/day. November 15 was mid-month, so around average."
    ),
    rejected_ans=(
        "On 15 November 2025 there were approximately **280 transactions**, consistent with the monthly daily average of 280 (8,412 ÷ 30). "
        "This was a typical mid-month day with no anomalous activity indicated."
    ),
))

# ---------------------------------------------------------------------------
# CATEGORY 2 — People NOT mentioned (10 pairs)
# ---------------------------------------------------------------------------

# 2.1 — CFO not in exec bio doc
pairs.append(make_pair(
    query=(
        "Document: [Executive Leadership Bios — DocWain Ltd]\n"
        "CEO: Sarah Chen — 15 years enterprise software, previously VP at Salesforce\n"
        "CTO: Raj Patel — ML/NLP specialist, ex-DeepMind research lead\n"
        "CPO: Emma Johansson — product strategy, ex-Google Cloud PM director\n\n"
        "Question: Who is the CFO and what is their background?"
    ),
    chosen_think=(
        "The document lists three executives: CEO Sarah Chen, CTO Raj Patel, CPO Emma Johansson.\n"
        "There is no CFO entry in this document.\n"
        "CFO could be a separate role not yet filled, not included in this particular bio doc, or handled by the CEO.\n"
        "I must not invent a name or background for a CFO not mentioned here.\n"
        "I'll clearly state the CFO is absent from this document and list who is mentioned."
    ),
    chosen_ans=(
        "The document profiles three executives:\n"
        "- **CEO**: Sarah Chen (15 yrs enterprise software, ex-Salesforce VP)\n"
        "- **CTO**: Raj Patel (ML/NLP, ex-DeepMind research lead)\n"
        "- **CPO**: Emma Johansson (product strategy, ex-Google Cloud PM Director)\n\n"
        "A **CFO is not listed** in this document. The role may be vacant, filled by someone not included in this bio set, or handled by another executive. "
        "You would need to consult the full organisational chart or company HR records."
    ),
    rejected_think=(
        "The doc has CEO, CTO, CPO. Most companies of this size also have a CFO. I can infer one exists."
    ),
    rejected_ans=(
        "The CFO of DocWain Ltd is **Michael Torres**, a seasoned finance executive with over 20 years in SaaS financial management. "
        "He previously served as VP Finance at HubSpot before joining DocWain to lead the finance function through its Series B growth phase."
    ),
))

# 2.2 — Named author not in research paper
pairs.append(make_pair(
    query=(
        "Document: [Research Paper — 'Transformer Architectures for Document Understanding']\n"
        "Authors: Dr. Aisha Kamara, Prof. Liang Wei, Dr. Tobias Brenner\n"
        "Abstract: This paper presents a novel attention mechanism for long-document processing...\n"
        "Institution: University of Edinburgh — NLP Research Group\n\n"
        "Question: What did Dr. Yuki Tanaka contribute to this paper?"
    ),
    chosen_think=(
        "The paper lists three authors: Dr. Aisha Kamara, Prof. Liang Wei, Dr. Tobias Brenner.\n"
        "Dr. Yuki Tanaka is not listed as an author, co-author, or contributor anywhere in the document.\n"
        "Acknowledging sections might name additional contributors but none are shown here.\n"
        "I cannot attribute any contribution to a person not present in the document.\n"
        "This is a clear case where the named person simply does not appear."
    ),
    chosen_ans=(
        "**Dr. Yuki Tanaka is not mentioned anywhere in this document.** The paper has three listed authors:\n"
        "- Dr. Aisha Kamara\n- Prof. Liang Wei\n- Dr. Tobias Brenner\n\n"
        "No contribution by Dr. Tanaka can be identified from this source. If Dr. Tanaka contributed to related work, that would appear in a different paper or acknowledgement not present here."
    ),
    rejected_think=(
        "There are three authors listed. Dr. Yuki Tanaka might have contributed in an advisory or acknowledgement role common in academic papers."
    ),
    rejected_ans=(
        "Dr. Yuki Tanaka contributed the **experimental validation framework** used in Section 4, designing the benchmark suite against which the novel attention mechanism was evaluated. "
        "Her expertise in evaluation methodology was credited in the acknowledgements section."
    ),
))

# 2.3 — Project manager not in project doc
pairs.append(make_pair(
    query=(
        "Document: [Project Charter — CRM Migration Initiative]\n"
        "Project Sponsor: VP Sales, James Okafor\n"
        "Technical Lead: Senior Architect, Priya Sharma\n"
        "Budget Owner: Finance Director, Helen Brooks\n"
        "Timeline: Q2–Q4 2025 | Budget: $340K\n\n"
        "Question: Who is the project manager and what is their experience level?"
    ),
    chosen_think=(
        "The charter names three roles: Sponsor (James Okafor), Technical Lead (Priya Sharma), Budget Owner (Helen Brooks).\n"
        "Project Manager is notably absent — a common role in project charters.\n"
        "This could mean the PM role is TBD, was omitted from this version, or is being handled by one of the named parties.\n"
        "I must not invent a PM name or seniority level.\n"
        "I'll report what roles are documented and flag that PM is not listed."
    ),
    chosen_ans=(
        "The project charter identifies three named roles:\n"
        "- **Project Sponsor**: James Okafor (VP Sales)\n"
        "- **Technical Lead**: Priya Sharma (Senior Architect)\n"
        "- **Budget Owner**: Helen Brooks (Finance Director)\n\n"
        "A **Project Manager is not listed** in this document. The role may be unfilled, pending assignment, or intentionally omitted from this version of the charter. "
        "Experience level cannot be assessed for a person not named in the document."
    ),
    rejected_think=(
        "Most project charters have a PM. It's likely someone from the PMO was assigned — I can infer a reasonable answer."
    ),
    rejected_ans=(
        "The project manager is **David Nguyen**, a mid-level PM from the internal PMO with approximately 5 years of experience managing enterprise systems migrations. "
        "He reports to the VP Sales sponsor and coordinates day-to-day delivery between the technical and finance stakeholders."
    ),
))

# 2.4 — Board member not in board minutes
pairs.append(make_pair(
    query=(
        "Document: [Board Meeting Minutes — 18 March 2025]\n"
        "Present: Chair Patricia Voss, Non-Exec Marcus Hill, Non-Exec Dr. Fiona Grant, CEO Sarah Chen, CFO Ben Adeyemi\n"
        "Apologies: None recorded\n"
        "Items discussed: Series C funding update, hiring plan approval, Q1 forecast review\n\n"
        "Question: What were the views of board member Robert Chambers during the meeting?"
    ),
    chosen_think=(
        "The minutes list five attendees: Patricia Voss, Marcus Hill, Dr. Fiona Grant, Sarah Chen, Ben Adeyemi.\n"
        "Robert Chambers is not listed as present, not listed with apologies, and not mentioned anywhere.\n"
        "No views from Robert Chambers can be extracted — he was not at this meeting (at least not as recorded).\n"
        "He may not be a board member, or he was absent without it being recorded as an apology.\n"
        "I cannot fabricate views attributed to someone not present in the documented meeting."
    ),
    chosen_ans=(
        "**Robert Chambers is not mentioned in these board minutes.** The recorded attendees were:\n"
        "- Chair: Patricia Voss\n- Non-Exec: Marcus Hill\n- Non-Exec: Dr. Fiona Grant\n- CEO: Sarah Chen\n- CFO: Ben Adeyemi\n\n"
        "No apologies were recorded. Robert Chambers' views cannot be extracted from this document as they are not present. "
        "It is unclear from this record whether he is a board member or was simply not at this meeting."
    ),
    rejected_think=(
        "Robert Chambers isn't listed but board members often have consistent views on funding and hiring. I can infer his likely position."
    ),
    rejected_ans=(
        "Robert Chambers expressed **cautious optimism** about the Series C funding timeline, noting that market conditions remained uncertain. "
        "On the hiring plan he was broadly supportive but requested additional detail on engineering headcount justification before final approval."
    ),
))

# 2.5 — Named supplier not in procurement doc
pairs.append(make_pair(
    query=(
        "Document: [Approved Supplier List — IT Procurement — FY2025]\n"
        "Cloud Infrastructure: Azure (Microsoft), AWS (Amazon)\n"
        "Monitoring: Datadog, PagerDuty\n"
        "Security: CrowdStrike, Okta\n"
        "Productivity: Microsoft 365, Slack\n\n"
        "Question: Is TechNova Solutions an approved supplier and what services do they provide?"
    ),
    chosen_think=(
        "The approved supplier list covers cloud, monitoring, security, and productivity categories.\n"
        "Named suppliers: Azure, AWS, Datadog, PagerDuty, CrowdStrike, Okta, Microsoft 365, Slack.\n"
        "TechNova Solutions does not appear anywhere in this document.\n"
        "I cannot confirm approval or invent a service category for an unlisted vendor.\n"
        "This is a straightforward absence — TechNova is simply not on the approved list."
    ),
    chosen_ans=(
        "**TechNova Solutions does not appear on the FY2025 Approved Supplier List.** The approved IT suppliers are:\n"
        "- Cloud: Azure, AWS\n- Monitoring: Datadog, PagerDuty\n- Security: CrowdStrike, Okta\n- Productivity: Microsoft 365, Slack\n\n"
        "Based on this document, TechNova Solutions is **not an approved supplier**. If an engagement is planned, a supplier onboarding and approval process would need to be initiated through IT Procurement."
    ),
    rejected_think=(
        "TechNova isn't listed explicitly but could be under a broader category. Many supplier lists are not exhaustive."
    ),
    rejected_ans=(
        "TechNova Solutions is an **approved supplier** under the professional services category, providing managed IT support and staff augmentation services. "
        "They were added to the approved list in late 2024 following a vendor review and hold a current MSA with the procurement team."
    ),
))

# 2.6 — Named clinician in medical report
pairs.append(make_pair(
    query=(
        "Document: [Patient Discharge Summary — Ref DS-20250310-447]\n"
        "Attending Physician: Dr. Mele Tupou | Specialty: General Medicine\n"
        "Admitting Consultant: Dr. Haruto Sato | Specialty: Cardiology\n"
        "Discharge date: 10 March 2025 | Diagnosis: Stable angina, managed medically\n\n"
        "Question: What treatment plan did Dr. Okonkwo recommend?"
    ),
    chosen_think=(
        "The discharge summary names two clinicians: Dr. Mele Tupou (attending, General Medicine) and Dr. Haruto Sato (admitting consultant, Cardiology).\n"
        "Dr. Okonkwo is not mentioned anywhere in this document.\n"
        "No treatment plan from Dr. Okonkwo can be extracted or inferred.\n"
        "In medical documents especially, attributing recommendations to a named clinician who is not in the record is dangerous.\n"
        "Confidence: Certain this person is absent. I must not fabricate clinical recommendations."
    ),
    chosen_ans=(
        "**Dr. Okonkwo is not named in this discharge summary.** The documented clinicians are:\n"
        "- Attending Physician: **Dr. Mele Tupou** (General Medicine)\n"
        "- Admitting Consultant: **Dr. Haruto Sato** (Cardiology)\n\n"
        "No treatment recommendations from Dr. Okonkwo are present in this document. Attributing medical advice to an unrecorded clinician would be inappropriate. "
        "Please verify with the medical records department whether Dr. Okonkwo was involved in this patient's care."
    ),
    rejected_think=(
        "The discharge summary covers stable angina. Dr. Okonkwo likely contributed as a specialist — I can infer a typical plan."
    ),
    rejected_ans=(
        "Dr. Okonkwo, the respiratory specialist consulted during admission, recommended a **stepped pharmacological approach** starting with long-acting nitrates, "
        "with a follow-up stress echo within 6 weeks to assess whether revascularisation should be considered."
    ),
))

# 2.7 — Named instructor not in training record
pairs.append(make_pair(
    query=(
        "Document: [Staff Training Completion Record — Cybersecurity Awareness 2025]\n"
        "Course Facilitator: Alex Morgan | Platform: LMS v3.2\n"
        "Completions: 247/260 staff (95%) | Pass rate: 91% | Date: Jan–Feb 2025\n\n"
        "Question: What additional modules did facilitator Sandra Obi deliver?"
    ),
    chosen_think=(
        "The training record names one facilitator: Alex Morgan.\n"
        "Sandra Obi is not mentioned as a facilitator, co-facilitator, or contributor anywhere.\n"
        "There is no mention of additional modules beyond the Cybersecurity Awareness course.\n"
        "I cannot attribute any modules to Sandra Obi who is not in this document.\n"
        "Straightforward absence: name not present, modules not attributable."
    ),
    chosen_ans=(
        "The training record identifies **Alex Morgan** as the sole course facilitator for the Cybersecurity Awareness 2025 programme.\n\n"
        "**Sandra Obi is not mentioned in this document.** No additional modules facilitated by Sandra Obi appear in this record. "
        "If Sandra Obi delivered training, it would be in a separate training record not provided here."
    ),
    rejected_think=(
        "Alex Morgan ran the main course. Sandra Obi may have run supplementary modules — common in blended learning programmes."
    ),
    rejected_ans=(
        "Sandra Obi delivered two supplementary modules: **'Phishing Simulation Deep Dive'** (45 minutes) and **'Incident Response Essentials'** (30 minutes), "
        "both available as optional enrichment content for staff who completed the core course with distinction."
    ),
))

# 2.8 — Named negotiator not in contract doc
pairs.append(make_pair(
    query=(
        "Document: [Contract Summary — DocWain Enterprise License — Acme Corp]\n"
        "Contract value: £180,000/year | Term: 3 years commencing 1 Jan 2025\n"
        "DocWain signatory: VP Sales James Okafor | Acme signatory: IT Director Paul Singh\n"
        "Key terms: 99.5% SLA, unlimited users, UK data residency\n\n"
        "Question: How did legal counsel Rachel Foster negotiate the liability cap?"
    ),
    chosen_think=(
        "The contract summary mentions two signatories: James Okafor (DocWain VP Sales) and Paul Singh (Acme IT Director).\n"
        "Rachel Foster is not mentioned — not as legal counsel, not in any capacity.\n"
        "The document does not describe negotiation history, legal discussions, or liability cap specifics beyond what's shown.\n"
        "Even if a liability cap exists in the full contract, I have no information about how it was negotiated or who negotiated it.\n"
        "I must not fabricate negotiation details or attribute them to an unnamed person."
    ),
    chosen_ans=(
        "**Rachel Foster is not named in this contract summary.** The document identifies:\n"
        "- DocWain signatory: James Okafor (VP Sales)\n"
        "- Acme signatory: Paul Singh (IT Director)\n\n"
        "The summary does not include negotiation history, details of legal counsel involvement, or the specific liability cap terms. "
        "Rachel Foster's role and any negotiation she may have conducted are **not present in this document**."
    ),
    rejected_think=(
        "Contract negotiations typically involve legal counsel. Rachel Foster likely negotiated standard terms including liability caps."
    ),
    rejected_ans=(
        "Rachel Foster, DocWain's in-house legal counsel, negotiated the liability cap at **2x annual contract value (£360,000)**, "
        "which was a compromise from Acme's initial position of capping at 1x. She also secured mutual indemnity provisions and unlimited IP indemnity from DocWain's side."
    ),
))

# 2.9 — Named data scientist not in ML model card
pairs.append(make_pair(
    query=(
        "Document: [Model Card — DocWain NER v2.3]\n"
        "Model type: Fine-tuned BERT | Task: Named Entity Recognition\n"
        "Training data: 45K annotated legal documents\n"
        "F1 score: 0.912 | Developed by: ML Platform Team\n\n"
        "Question: What specific contributions did data scientist Kwame Asante make to this model?"
    ),
    chosen_think=(
        "The model card attributes the work to the 'ML Platform Team' — no individual names are listed.\n"
        "Kwame Asante is not mentioned anywhere in the document.\n"
        "Team-level attribution is common in model cards, which doesn't allow me to infer individual contributions.\n"
        "Even if Kwame Asante is on the ML Platform Team, his specific contributions are not documented here.\n"
        "I cannot fabricate individual contributions not recorded in the document."
    ),
    chosen_ans=(
        "The model card attributes development to the **ML Platform Team** collectively. No individual contributors are named.\n\n"
        "**Kwame Asante is not mentioned** in this document, and no specific contributions from him can be identified. "
        "Individual attribution may be available in internal team documentation, commit history, or project records not included here."
    ),
    rejected_think=(
        "ML teams have data scientists. Kwame Asante likely worked on training data curation given the 45K document dataset."
    ),
    rejected_ans=(
        "Kwame Asante led the **annotation pipeline design** for the 45K legal document training corpus, establishing the entity taxonomy and inter-annotator agreement protocols. "
        "He also conducted the ablation studies that informed the final fine-tuning hyperparameters, achieving the 0.912 F1 score."
    ),
))

# 2.10 — Named compliance officer not in audit report
pairs.append(make_pair(
    query=(
        "Document: [Internal Audit Report — GDPR Compliance — Q1 2025]\n"
        "Audit team: Internal Audit Department | Scope: Data processing activities Jan–Mar 2025\n"
        "Findings: 2 high-severity, 4 medium, 1 low | Overall rating: Requires Improvement\n"
        "High findings: (1) Inadequate DPA agreements with 3 processors, (2) Missing DSAR log entries\n\n"
        "Question: What remediation steps did DPO Caroline Weiss commit to?"
    ),
    chosen_think=(
        "The audit report names only 'Internal Audit Department' as the audit team — no individual names.\n"
        "DPO Caroline Weiss is not mentioned anywhere in this document.\n"
        "The document describes findings but no management responses, remediation commitments, or action owners.\n"
        "Audit reports often have a separate management response section, but none is present here.\n"
        "I must not fabricate remediation commitments from a named individual not in the document."
    ),
    chosen_ans=(
        "**Caroline Weiss is not named in this audit report.** The document records findings only:\n"
        "- High 1: Inadequate DPA agreements with 3 processors\n"
        "- High 2: Missing DSAR log entries\n"
        "- 4 medium and 1 low finding\n\n"
        "The report does not include management responses or remediation commitments from any named individual. "
        "DPO commitments, if captured, would typically appear in a management response annex not present in this document."
    ),
    rejected_think=(
        "Audit reports usually prompt DPO responses. Caroline Weiss as DPO would have responded to the GDPR findings."
    ),
    rejected_ans=(
        "DPO Caroline Weiss committed to the following remediation steps: (1) **Complete revised DPA agreements** with all three processors by 30 April 2025, "
        "(2) **Reconstruct missing DSAR logs** from email trails and implement automated logging by end of Q2, "
        "and (3) conduct a full data mapping refresh before the next audit cycle."
    ),
))

# ---------------------------------------------------------------------------
# CATEGORY 3 — Metrics that cannot be calculated from given data (10 pairs)
# ---------------------------------------------------------------------------

# 3.1 — YoY growth rate, only one year present
pairs.append(make_pair(
    query=(
        "Document: [2025 Sales Performance Summary]\nTotal Sales: £9.4M | Deals closed: 83 | Avg deal size: £113K\n\n"
        "Question: What was the year-over-year sales growth rate?"
    ),
    chosen_think=(
        "To calculate YoY growth I need: (2025 sales − 2024 sales) / 2024 sales × 100.\n"
        "The document provides 2025 sales: £9.4M.\n"
        "2024 sales are not provided — there is no prior year comparator anywhere in the document.\n"
        "Without 2024 data the growth rate is mathematically incalculable.\n"
        "I must not assume a 2024 baseline.\n"
        "Confidence: Certain the calculation cannot be performed from the available data."
    ),
    chosen_ans=(
        "A year-over-year growth rate **cannot be calculated** from this document. The required formula is:\n"
        "> YoY growth = (2025 sales − 2024 sales) / 2024 sales × 100\n\n"
        "This document provides only the **2025 figure (£9.4M)**. The 2024 baseline is not present. "
        "To compute YoY growth you would need the 2024 Sales Performance Summary or a comparative multi-year report."
    ),
    rejected_think=(
        "2025 sales are £9.4M. Typical SaaS growth is 30–40%. I'll assume a reasonable baseline and compute growth."
    ),
    rejected_ans=(
        "With 2025 sales of £9.4M and an estimated 2024 baseline of approximately £7.2M, "
        "the year-over-year growth rate was approximately **30.6%**, reflecting strong enterprise deal momentum."
    ),
))

# 3.2 — Gross margin %, costs not in doc
pairs.append(make_pair(
    query=(
        "Document: [Revenue Dashboard — October 2025]\nGross Revenue: $1.84M | Refunds: $23K | Net Revenue: $1.817M\n\n"
        "Question: What was the gross margin percentage for October 2025?"
    ),
    chosen_think=(
        "Gross margin % = (Revenue − COGS) / Revenue × 100.\n"
        "Net Revenue is $1.817M. However, gross margin requires Cost of Goods Sold (COGS), not refunds.\n"
        "COGS is not present in this document — it's a revenue-only dashboard.\n"
        "Revenue minus refunds gives net revenue, not gross margin.\n"
        "These are different concepts: gross margin accounts for direct costs of delivering the product/service.\n"
        "I cannot calculate gross margin without COGS data.\n"
        "Confidence: Certain COGS is absent; the calculation is impossible from available data."
    ),
    chosen_ans=(
        "**Gross margin cannot be calculated** from this document. Gross margin requires:\n"
        "> Gross Margin % = (Net Revenue − COGS) / Net Revenue × 100\n\n"
        "This document provides revenue figures (Gross: $1.84M, after refunds: $1.817M) but **does not include Cost of Goods Sold (COGS)**. "
        "You would need a P&L statement or cost breakdown to compute gross margin. Note that net revenue ($1.817M) is not the same as gross profit."
    ),
    rejected_think=(
        "Net revenue is $1.817M. Software businesses typically have ~70% gross margins. I can provide a reasonable figure."
    ),
    rejected_ans=(
        "For October 2025, the gross margin was approximately **71.3%**, derived from the net revenue of $1.817M and industry-typical COGS structure for a SaaS platform of this scale."
    ),
))

# 3.3 — Employee turnover rate, partial headcount only
pairs.append(make_pair(
    query=(
        "Document: [HR Metrics — Q3 2025]\nHeadcount end of Q3: 186 | New hires Q3: 14 | Promotions: 7\n\n"
        "Question: What was the employee turnover rate for Q3 2025?"
    ),
    chosen_think=(
        "Turnover rate = (employees who left / average headcount) × 100.\n"
        "The document gives: end headcount 186, new hires 14, promotions 7.\n"
        "To calculate: I need the number of departures (voluntary + involuntary leavers) — not provided.\n"
        "I also need start-of-quarter headcount to compute average headcount — not provided.\n"
        "New hires (14) represent inflows, not outflows. Promotions (7) are internal movements.\n"
        "Without leaver count and start headcount, turnover rate is incalculable.\n"
        "I must not conflate hires with leavers."
    ),
    chosen_ans=(
        "**Turnover rate cannot be calculated** from this document. The formula requires:\n"
        "> Turnover Rate = (Number of Leavers ÷ Average Headcount) × 100\n\n"
        "This document provides end-of-quarter headcount (186), new hires (14), and promotions (7), but **does not include the number of employee departures** or start-of-quarter headcount. "
        "Both are necessary to compute turnover rate. You would need the full HR attrition report for Q3 2025."
    ),
    rejected_think=(
        "14 new hires out of 186 employees. Turnover and hiring are usually balanced so I can estimate."
    ),
    rejected_ans=(
        "With 14 new hires against an end headcount of 186, the implied Q3 2025 turnover rate was approximately **7.5%**, "
        "suggesting that hiring largely replaced departures while still achieving modest net growth in team size."
    ),
))

# 3.4 — Net Promoter Score, only raw survey data
pairs.append(make_pair(
    query=(
        "Document: [Customer Survey Raw Results — June 2025]\n"
        "Responses: 320 | Score distribution: 9-10 (Promoters): 148 | 7-8 (Passives): 97 | 0-6 (Detractors): 75\n\n"
        "Question: What is the Net Promoter Score?"
    ),
    chosen_think=(
        "NPS = % Promoters − % Detractors.\n"
        "Promoters: 148/320 = 46.25%\n"
        "Detractors: 75/320 = 23.44%\n"
        "NPS = 46.25 − 23.44 = 22.81 ≈ 23\n"
        "Wait — this IS calculable from the given data. I have promoters, passives, detractors, and total responses.\n"
        "Let me recalculate carefully: 148/320 = 0.4625 → 46.25%; 75/320 = 0.234375 → 23.44%\n"
        "NPS = 46.25 − 23.44 = 22.81, rounded to 23.\n"
        "This is a case where the chosen response DOES calculate, accurately, with shown working."
    ),
    chosen_ans=(
        "The Net Promoter Score can be calculated directly from the survey data provided:\n\n"
        "- **Promoters** (9-10): 148 / 320 = **46.25%**\n"
        "- **Passives** (7-8): 97 / 320 = 30.31% (not used in NPS)\n"
        "- **Detractors** (0-6): 75 / 320 = **23.44%**\n\n"
        "**NPS = 46.25% − 23.44% = 22.81 ≈ 23**\n\n"
        "The Net Promoter Score for June 2025 is **+23**, which falls in the 'Good' range. Confidence: High — all required inputs are present and the calculation is straightforward."
    ),
    rejected_think=(
        "148 promoters out of 320 — that's roughly 46%. NPS is probably around 45 or 50."
    ),
    rejected_ans=(
        "Based on the survey results, the Net Promoter Score for June 2025 was approximately **+46**, "
        "reflecting the strong proportion of promoters (148 responses) relative to total respondents."
    ),
))

# 3.5 — Customer acquisition cost, missing spend data
pairs.append(make_pair(
    query=(
        "Document: [Sales & Marketing Summary — Q2 2025]\n"
        "New customers acquired: 28 | Sales headcount: 12 | Marketing headcount: 6\n"
        "Marketing campaigns run: 4 | Average sales cycle: 67 days\n\n"
        "Question: What was the customer acquisition cost (CAC) in Q2 2025?"
    ),
    chosen_think=(
        "CAC = Total Sales & Marketing Spend / Number of New Customers Acquired.\n"
        "New customers: 28 — this is available.\n"
        "Total S&M spend is NOT provided. The document shows headcount and campaign count but no salary costs, campaign budgets, or total expenditure.\n"
        "Headcount (12 + 6 = 18 people) doesn't tell me salary costs without salary data.\n"
        "4 campaigns run doesn't tell me their budget.\n"
        "CAC is mathematically impossible to calculate from this data.\n"
        "I must not estimate salary costs or campaign spend to fabricate a CAC figure."
    ),
    chosen_ans=(
        "**CAC cannot be calculated** from this document. The formula requires:\n"
        "> CAC = Total Sales & Marketing Spend ÷ New Customers Acquired\n\n"
        "While new customers (28) are provided, the document **does not include total S&M expenditure** — no salary costs, campaign budgets, or aggregate spend figures are present. "
        "Headcount (18 total) and campaign count (4) alone are insufficient without corresponding cost data. "
        "You would need the Q2 2025 P&L or departmental cost report to compute CAC."
    ),
    rejected_think=(
        "28 new customers acquired. With 18 S&M staff and 4 campaigns, spend was probably around $500K–$700K."
    ),
    rejected_ans=(
        "With approximately $580K in combined sales and marketing expenditure for Q2 2025 (estimated from headcount and campaign costs), "
        "the customer acquisition cost was approximately **$20,700 per new customer**, which is competitive for enterprise SaaS at this deal size."
    ),
))

# 3.6 — Compound annual growth rate, only two endpoints but years unknown
pairs.append(make_pair(
    query=(
        "Document: [Strategic Review — Revenue History]\n"
        "Revenue Year 1: $3.2M | Revenue Year 5: $11.8M\n\n"
        "Question: What is the compound annual growth rate (CAGR)?"
    ),
    chosen_think=(
        "CAGR formula: (End Value / Start Value)^(1 / Number of Years) − 1\n"
        "Start: $3.2M, End: $11.8M.\n"
        "The document says 'Year 1' and 'Year 5' — this implies 4 years of growth between Year 1 and Year 5.\n"
        "CAGR = (11.8 / 3.2)^(1/4) − 1 = (3.6875)^(0.25) − 1\n"
        "3.6875^0.25 = √(√3.6875) = √1.9204 ≈ 1.3858\n"
        "CAGR ≈ 38.58%\n"
        "This IS calculable — Year 1 to Year 5 = 4 periods. I have all needed data.\n"
        "Chosen response calculates correctly with shown working."
    ),
    chosen_ans=(
        "The CAGR can be calculated from the two data points provided:\n\n"
        "> CAGR = (End Value ÷ Start Value)^(1 ÷ n) − 1\n"
        "> CAGR = ($11.8M ÷ $3.2M)^(1 ÷ 4) − 1\n"
        "> CAGR = (3.6875)^0.25 − 1 ≈ 1.3858 − 1 = **38.6%**\n\n"
        "The 4-year CAGR from Year 1 to Year 5 is approximately **38.6%**. "
        "Confidence: High — this assumes 'Year 1' to 'Year 5' represents exactly 4 compounding periods, which is the standard interpretation."
    ),
    rejected_think=(
        "Revenue grew from $3.2M to $11.8M. That's about 3.7x over some years. CAGR should be high."
    ),
    rejected_ans=(
        "Revenue grew from $3.2M to $11.8M, representing a **CAGR of approximately 55%**, reflecting exceptional growth driven by strong product-market fit and expanding enterprise sales capacity."
    ),
))

# 3.7 — Average ticket value requiring data not present
pairs.append(make_pair(
    query=(
        "Document: [Support Operations Report — August 2025]\n"
        "Total tickets opened: 1,847 | Tickets closed: 1,792 | Escalations: 89\n"
        "P1 tickets: 23 | P2 tickets: 156 | P3/P4 tickets: 1,668\n\n"
        "Question: What was the average cost per support ticket in August 2025?"
    ),
    chosen_think=(
        "Average cost per ticket = Total support operating cost / Total tickets.\n"
        "Total tickets opened: 1,847 — available.\n"
        "Total support operating cost — NOT available. No staff costs, tool costs, or overhead figures in this document.\n"
        "This is an operational volume report, not a cost report.\n"
        "Without cost data, the per-ticket cost is incalculable.\n"
        "Ticket counts alone cannot yield a cost metric."
    ),
    chosen_ans=(
        "**Average cost per ticket cannot be calculated** from this document. The formula requires:\n"
        "> Cost per Ticket = Total Support Operating Cost ÷ Tickets Handled\n\n"
        "This document provides ticket volumes (1,847 opened, 1,792 closed) and priority breakdowns but **contains no cost data** — no staff costs, tool licensing, or overhead figures. "
        "You would need the August 2025 support cost centre report or finance P&L to compute this metric."
    ),
    rejected_think=(
        "1,792 tickets closed. Enterprise support typically costs $25–$50 per ticket. I'll provide an estimate."
    ),
    rejected_ans=(
        "Based on industry benchmarks for enterprise SaaS support operations, the average cost per ticket in August 2025 was approximately **$38**, "
        "resulting in a total support cost of approximately $68,096 for the month."
    ),
))

# 3.8 — Profit margin, revenue only doc
pairs.append(make_pair(
    query=(
        "Document: [Monthly Revenue Tracker — September 2025]\n"
        "MRR: $2.1M | New MRR: $180K | Expansion MRR: $95K | Churned MRR: -$42K\n\n"
        "Question: What was the net profit margin for September 2025?"
    ),
    chosen_think=(
        "Net profit margin = Net Profit / Revenue × 100.\n"
        "Net Profit = Revenue − All Expenses (COGS, OpEx, interest, tax).\n"
        "This document shows only MRR and its components — revenue inflows and one outflow (churn, which is lost revenue not an expense).\n"
        "No expenses of any kind are mentioned: no COGS, no operating costs, no headcount costs, no infrastructure costs.\n"
        "Net profit requires full P&L data — this is a pure revenue tracker.\n"
        "Profit margin is completely incalculable from revenue-only data."
    ),
    chosen_ans=(
        "**Net profit margin cannot be calculated** from this document. It requires:\n"
        "> Net Profit Margin = (Revenue − Total Expenses) ÷ Revenue × 100\n\n"
        "This is a revenue-only tracker showing MRR of $2.1M and its components. It contains **no expense data** — no COGS, operating costs, staff costs, or overheads. "
        "Without a full P&L statement for September 2025, profit margin cannot be determined."
    ),
    rejected_think=(
        "MRR is $2.1M. SaaS companies typically run 15–25% net margins. I can give a reasonable estimate."
    ),
    rejected_ans=(
        "With September MRR of $2.1M and a typical net margin for a scaling SaaS business of around **18%**, "
        "the estimated net profit for September 2025 was approximately $378,000."
    ),
))

# 3.9 — Conversion rate requiring funnel stages not all present
pairs.append(make_pair(
    query=(
        "Document: [Marketing Funnel Report — Q4 2025]\n"
        "Impressions: 2,400,000 | Clicks: 48,000 | MQL: 1,920\n\n"
        "Question: What was the SQL-to-Closed-Won conversion rate?"
    ),
    chosen_think=(
        "SQL-to-Closed-Won conversion = Closed Won deals / SQLs × 100.\n"
        "The document has: Impressions, Clicks, MQLs.\n"
        "SQLs (Sales Qualified Leads) are NOT mentioned.\n"
        "Closed Won deals are NOT mentioned.\n"
        "The funnel data stops at MQL stage — two downstream stages are completely absent.\n"
        "I cannot infer SQLs from MQLs without a known MQL-to-SQL conversion rate.\n"
        "The requested metric is incalculable from available data."
    ),
    chosen_ans=(
        "The SQL-to-Closed-Won conversion rate **cannot be calculated** from this document. The required data:\n"
        "- SQLs (Sales Qualified Leads) — **not present**\n"
        "- Closed Won deals — **not present**\n\n"
        "The document provides funnel data only up to MQL stage (1,920 MQLs). Downstream stages — SQL and Closed Won — are absent. "
        "You would need the full sales funnel report including SQL and opportunity data to compute this conversion rate."
    ),
    rejected_think=(
        "1,920 MQLs. Typical MQL-to-SQL conversion is ~50% and SQL-to-close is ~20%, so I can estimate."
    ),
    rejected_ans=(
        "From 1,920 MQLs, approximately 960 would progress to SQL stage (~50% conversion). Of those, around **192 deals would close** (~20% SQL-to-close), "
        "giving an SQL-to-Closed-Won conversion rate of approximately **20%** for Q4 2025."
    ),
))

# 3.10 — Inventory turnover, only one balance sheet date
pairs.append(make_pair(
    query=(
        "Document: [Balance Sheet Extract — 31 December 2025]\n"
        "Inventory: £4.2M | Total Current Assets: £18.6M | Total Assets: £47.3M\n\n"
        "Question: What was the inventory turnover ratio for FY2025?"
    ),
    chosen_think=(
        "Inventory Turnover = Cost of Goods Sold / Average Inventory.\n"
        "Average Inventory = (Opening Inventory + Closing Inventory) / 2.\n"
        "I have closing inventory (£4.2M from 31 Dec 2025 balance sheet).\n"
        "I do NOT have: COGS for FY2025, or opening inventory (31 Dec 2024 balance sheet).\n"
        "Both COGS and opening inventory are required — neither is present.\n"
        "A single balance sheet date gives only one inventory snapshot.\n"
        "Turnover ratio is incalculable without income statement data (COGS) and prior period balance sheet."
    ),
    chosen_ans=(
        "**Inventory turnover cannot be calculated** from this document. The formula requires:\n"
        "> Inventory Turnover = COGS ÷ Average Inventory\n"
        "> Average Inventory = (Opening Inventory + Closing Inventory) ÷ 2\n\n"
        "This balance sheet extract provides only the **closing inventory position (£4.2M)** as of 31 December 2025. Missing:\n"
        "- COGS for FY2025 (requires income statement)\n"
        "- Opening inventory as of 1 January 2025 (requires prior year balance sheet)\n\n"
        "Both are needed to compute the ratio."
    ),
    rejected_think=(
        "Inventory is £4.2M. Turnover ratios for software/professional services companies are typically 8–12x."
    ),
    rejected_ans=(
        "With closing inventory of £4.2M and a typical turnover ratio for an enterprise software business of approximately **9x**, "
        "DocWain's implied COGS was approximately £37.8M for FY2025, suggesting efficient inventory management."
    ),
))

# ---------------------------------------------------------------------------
# CATEGORY 4 — Missing sections — rejected fills them in (10 pairs)
# ---------------------------------------------------------------------------

# 4.1 — Recommendations section absent
pairs.append(make_pair(
    query=(
        "Document: [Market Research Report — SME Sector Analysis 2025]\n"
        "[Section 1: Executive Summary] The SME sector shows 12% growth in digital adoption.\n"
        "[Section 2: Findings] 67% of SMEs cite cost as primary barrier. Awareness gap in AI tools: 43%.\n"
        "[Section 3: Recommendations] — THIS SECTION IS BLANK —\n\n"
        "Question: What recommendations does the report make for targeting SME customers?"
    ),
    chosen_think=(
        "The document has three sections. Sections 1 and 2 contain data.\n"
        "Section 3 (Recommendations) is explicitly blank — the document notes it is blank.\n"
        "The findings in Section 2 could logically suggest recommendations, but the authors have not written any.\n"
        "I must not write recommendations on behalf of the report authors.\n"
        "My job is to report what the document says, and here it says the recommendations section is empty.\n"
        "I can note the relevant findings that would typically inform recommendations, but I cannot fabricate the recommendations themselves."
    ),
    chosen_ans=(
        "The Recommendations section (Section 3) of this report **is blank** — the document explicitly notes it contains no content.\n\n"
        "The report's findings (Section 2) indicate:\n"
        "- 67% of SMEs cite cost as the primary barrier\n"
        "- 43% awareness gap in AI tools\n\n"
        "However, translating these findings into recommendations is the work of the report authors, which has not been completed. "
        "I cannot substitute the missing content. You may need to request the completed version of the report or commission the recommendations analysis separately."
    ),
    rejected_think=(
        "Section 2 has clear findings about cost barriers and awareness gaps. The recommendations are obvious from the data."
    ),
    rejected_ans=(
        "The report recommends: (1) **Offer a freemium tier** to address the cost barrier, as 67% of SMEs cite cost as the primary concern. "
        "(2) **Launch an SME awareness campaign** targeting the 43% who are unaware of AI tools. "
        "(3) Partner with SME-focused trade bodies to build credibility and accelerate adoption in under-penetrated segments."
    ),
))

# 4.2 — Financial tables missing from annual report
pairs.append(make_pair(
    query=(
        "Document: [Annual Report 2025 — Narrative Sections]\n"
        "CEO Letter: 'We achieved record revenue growth and expanded into three new markets.'\n"
        "Strategy Update: Focuses on enterprise, APAC expansion, and platform consolidation.\n"
        "[Financial Statements: See Appendix A — NOT INCLUDED IN THIS EXTRACT]\n\n"
        "Question: What was the total revenue reported in the 2025 annual report?"
    ),
    chosen_think=(
        "The document contains narrative sections: CEO letter and strategy update.\n"
        "The financial statements are explicitly noted as being in Appendix A, which is NOT INCLUDED in this extract.\n"
        "The CEO letter mentions 'record revenue growth' qualitatively but gives no figures.\n"
        "I cannot extract a revenue number because the financial tables are absent.\n"
        "The phrase 'record revenue growth' is a qualitative statement, not a specific figure.\n"
        "Confidence: High that numerical revenue data is not in this document extract."
    ),
    chosen_ans=(
        "The financial statements containing revenue figures are in **Appendix A**, which is explicitly **not included in this document extract**.\n\n"
        "The CEO letter references 'record revenue growth' qualitatively, but no specific revenue figure appears in the narrative sections provided. "
        "To obtain the total revenue figure you would need access to Appendix A (Financial Statements) from the full 2025 Annual Report."
    ),
    rejected_think=(
        "CEO mentions record revenue growth. Based on prior years and typical company growth, I can estimate the 2025 revenue."
    ),
    rejected_ans=(
        "Based on the CEO's reference to 'record revenue growth' and the strategic expansion described, total 2025 revenue was approximately **£28.4M**, "
        "representing a 34% increase over the prior year, consistent with the three new market entries referenced in the CEO letter."
    ),
))

# 4.3 — Risk assessment section missing from project plan
pairs.append(make_pair(
    query=(
        "Document: [Project Plan — Platform Re-architecture — v1.2]\n"
        "Section 1 — Objectives: Migrate to microservices, improve scalability, reduce deployment time.\n"
        "Section 2 — Timeline: 9-month programme, 3 phases of 3 months each.\n"
        "Section 3 — Risk Assessment: [DRAFT — TO BE COMPLETED BY ENTERPRISE ARCHITECT]\n"
        "Section 4 — Resource Plan: 8 engineers, 2 DevOps, 1 PM\n\n"
        "Question: What are the key risks identified for this re-architecture programme?"
    ),
    chosen_think=(
        "The document has four sections. Sections 1, 2, and 4 have content.\n"
        "Section 3 (Risk Assessment) is explicitly marked as DRAFT and TO BE COMPLETED BY ENTERPRISE ARCHITECT — it contains no actual risks.\n"
        "There are zero identified risks in this document currently.\n"
        "I could speculate about common re-architecture risks, but that is not what the document says.\n"
        "The question asks what risks are 'identified' in the document — the answer is none.\n"
        "I should be clear that the risk section is incomplete and not invent risks."
    ),
    chosen_ans=(
        "The Risk Assessment (Section 3) is explicitly marked **'DRAFT — TO BE COMPLETED BY ENTERPRISE ARCHITECT'** and contains no identified risks.\n\n"
        "This document currently has **no documented risks** for the re-architecture programme. The programme context (9-month, 3-phase microservices migration with 11 staff) exists, but the risk analysis has not been completed yet. "
        "You would need to await the Enterprise Architect's completion of Section 3 or request a risk workshop to populate this section."
    ),
    rejected_think=(
        "Microservices migrations are well-understood. I know the typical risks and can fill in what the architect would likely write."
    ),
    rejected_ans=(
        "The key risks identified are: (1) **Data migration integrity** — risk of data loss during service decomposition. "
        "(2) **Service dependency management** — circular dependencies between newly extracted microservices. "
        "(3) **Skills gap** — team unfamiliarity with event-driven architecture patterns. "
        "(4) **Deployment pipeline complexity** — CI/CD rework required for independent service deployments."
    ),
))

# 4.4 — Appendix referenced but not provided
pairs.append(make_pair(
    query=(
        "Document: [Vendor Assessment Report — CloudStorage Inc]\n"
        "Overall Score: 78/100 | Recommended: Conditional\n"
        "Security: 82 | Reliability: 75 | Support: 71 | Commercial: 80\n"
        "Detailed scoring breakdown and assessor comments: See Appendix B\n\n"
        "Question: What specific security concerns were noted by the assessors?"
    ),
    chosen_think=(
        "The document provides high-level category scores: Security 82/100, Reliability 75, Support 71, Commercial 80.\n"
        "Detailed scoring and assessor comments are explicitly referred to Appendix B.\n"
        "Appendix B is not included in this document extract.\n"
        "The 82/100 security score tells me it's reasonably good but not perfect — it doesn't tell me WHAT the concerns were.\n"
        "Specific security concerns would be in the assessor comments in Appendix B — which is absent.\n"
        "I cannot fabricate specific concerns from a score alone."
    ),
    chosen_ans=(
        "Specific security concerns are documented in **Appendix B (Detailed Scoring and Assessor Comments)**, which is **not included in this document**.\n\n"
        "What the main report shows: Security category score of **82/100** (the lowest of the top two scores, suggesting some deductions). The conditional recommendation implies concerns exist, but their specifics are in the missing appendix. "
        "You would need Appendix B to identify what security sub-criteria were flagged."
    ),
    rejected_think=(
        "Security score of 82 means some issues. For cloud storage, typical concerns are encryption, access control, and compliance."
    ),
    rejected_ans=(
        "Assessors noted the following security concerns: (1) **Encryption key management** — keys held by vendor, not customer-managed. "
        "(2) **SOC 2 Type II** certification pending renewal. (3) **Access logging granularity** below required enterprise standard. "
        "These were the primary drivers of the deduction from a perfect security score."
    ),
))

# 4.5 — Methodology section absent from research doc
pairs.append(make_pair(
    query=(
        "Document: [Impact Study — DocWain ROI Analysis — Enterprise Clients]\n"
        "Key Finding: Clients reported average 34% reduction in document processing time.\n"
        "Key Finding: 91% client satisfaction rate post-implementation.\n"
        "Key Finding: Average ROI payback period: 8.2 months.\n"
        "[Methodology: Section omitted from client-facing summary]\n\n"
        "Question: How was the 34% time reduction figure measured and validated?"
    ),
    chosen_think=(
        "The document presents findings: 34% time reduction, 91% satisfaction, 8.2 month payback.\n"
        "The Methodology section is explicitly noted as 'omitted from client-facing summary'.\n"
        "Without methodology I cannot explain how the 34% was measured, what baseline was used, who validated it, or what sample size was involved.\n"
        "The findings are presented without the measurement framework.\n"
        "I must not invent a methodology — measurement details could be material to assessing credibility of the finding.\n"
        "Confidence: Certain methodology is absent from this version of the document."
    ),
    chosen_ans=(
        "The **methodology section has been omitted** from this client-facing summary. The document explicitly notes this.\n\n"
        "The 34% time reduction figure is stated as a finding, but how it was measured, validated, or derived cannot be determined from this document. "
        "Key unknowns include: baseline measurement approach, sample size, client cohort, measurement timeframe, and validation method. "
        "You would need the full study (including methodology) to assess the robustness of this figure."
    ),
    rejected_think=(
        "ROI studies typically use before/after time tracking. I can describe the standard methodology for such studies."
    ),
    rejected_ans=(
        "The 34% time reduction was measured using a **pre/post implementation time-and-motion study** across 45 enterprise clients, "
        "comparing document processing times in the 3 months before DocWain deployment against the 3 months post go-live. "
        "Validation was performed by an independent auditor using timestamped workflow logs from each client's DMS."
    ),
))

# 4.6 — Glossary missing from technical spec
pairs.append(make_pair(
    query=(
        "Document: [Technical Specification — Ingestion Pipeline v3]\n"
        "The pipeline uses an ADEP flow with QSIM scoring for deduplication. "
        "All documents pass through the NVB layer before storage. Refer to Glossary for term definitions.\n"
        "[Glossary: Attached separately — not included]\n\n"
        "Question: What does ADEP stand for and what does it do?"
    ),
    chosen_think=(
        "The document uses acronyms: ADEP, QSIM, NVB.\n"
        "The Glossary for term definitions is explicitly noted as 'attached separately — not included'.\n"
        "ADEP is described only as part of a 'flow' — an architectural component.\n"
        "I do not know what ADEP stands for or its detailed function from the text alone.\n"
        "I cannot expand the acronym or describe its functionality without the glossary.\n"
        "Making up an expansion (e.g., 'Adaptive Document Extraction Pipeline') would be fabrication."
    ),
    chosen_ans=(
        "The document uses ADEP as a component of the ingestion flow but **does not define it** — the Glossary is explicitly noted as 'attached separately and not included' in this document extract.\n\n"
        "What can be inferred: ADEP is part of the pipeline flow (likely a processing stage), but its full name, purpose, and technical behaviour are not defined in the available text. "
        "You would need the accompanying Glossary or the full technical specification to understand ADEP."
    ),
    rejected_think=(
        "ADEP in a document ingestion pipeline context likely stands for something like Adaptive Document Extraction Pipeline. I'll define it."
    ),
    rejected_ans=(
        "ADEP stands for **Adaptive Document Extraction Pipeline**. It is the first-stage processing layer responsible for format detection, OCR triggering, and content normalisation before documents enter the QSIM deduplication scoring phase."
    ),
))

# 4.7 — Action items section empty after meeting
pairs.append(make_pair(
    query=(
        "Document: [Architecture Review Meeting Notes — 22 April 2025]\n"
        "Attendees: CTO Raj Patel, Lead Architect Maya Singh, DevOps Lead Tom O'Brien\n"
        "Discussion: Debated moving from monolithic to event-driven architecture. Several trade-offs discussed including latency, complexity, and team readiness.\n"
        "Action Items: [Not recorded during session — to be distributed separately]\n\n"
        "Question: What actions were assigned and to whom after this meeting?"
    ),
    chosen_think=(
        "The meeting notes have an Action Items section but it explicitly states they were not recorded and will be distributed separately.\n"
        "Discussion content is present (event-driven architecture trade-offs) but outcomes/assignments are absent.\n"
        "I cannot derive action items from discussion content — discussions can lead to many different outcomes.\n"
        "The fact that action items were 'not recorded during session' means I have no basis for any assignment.\n"
        "I must clearly state that action items are missing and not invent assignments."
    ),
    chosen_ans=(
        "The Action Items section of these meeting notes explicitly states they were **'not recorded during session — to be distributed separately'**.\n\n"
        "No action items or owner assignments are documented in this record. While the discussion covered event-driven architecture trade-offs, the resulting decisions and owner assignments were not captured here. "
        "You would need to request the separately distributed action item list from the meeting organiser."
    ),
    rejected_think=(
        "The discussion was about event-driven architecture. The CTO, Lead Architect, and DevOps Lead would naturally own specific follow-up actions."
    ),
    rejected_ans=(
        "The following actions were assigned: (1) **Maya Singh** to produce a detailed event-driven architecture proposal by 6 May. "
        "(2) **Tom O'Brien** to assess DevOps tooling readiness for event streaming (Kafka/Pulsar evaluation). "
        "(3) **Raj Patel** to review team upskilling budget and approve the proposal at the next architecture review."
    ),
))

# 4.8 — Executive summary missing from board pack
pairs.append(make_pair(
    query=(
        "Document: [Board Pack — June 2025 — Sections 2–7 Only]\n"
        "Section 2: Financial Performance — Revenue £4.1M, Costs £3.2M\n"
        "Section 3: Sales Pipeline — 47 active opportunities, £8.2M pipeline value\n"
        "[Section 1: Executive Summary — NOT INCLUDED IN THIS PACK EXTRACT]\n\n"
        "Question: What were the key messages in the executive summary presented to the board?"
    ),
    chosen_think=(
        "Section 1 (Executive Summary) is explicitly noted as not included in this extract.\n"
        "Sections 2 and 3 are present with financial and pipeline data.\n"
        "The executive summary would synthesise sections 2-7 with CEO commentary — I don't have sections 4-7 either.\n"
        "Executive summaries are crafted narratives, not mechanical summaries of numbers.\n"
        "I cannot reconstruct what the CEO/CFO chose to highlight in the summary from partial data.\n"
        "I must not write a proxy executive summary."
    ),
    chosen_ans=(
        "**Section 1 (Executive Summary) is not included in this document extract** — it is explicitly noted as absent.\n\n"
        "From sections 2 and 3 I can confirm: Revenue £4.1M, Costs £3.2M (Section 2), and £8.2M pipeline across 47 opportunities (Section 3). However, the executive summary is a synthesised narrative prepared by leadership and would reflect their chosen framing, priorities, and messages across all seven sections. "
        "The specific key messages presented to the board are **not available** from this extract."
    ),
    rejected_think=(
        "Revenue and pipeline data are available. I can summarise the key themes that would feature in an executive summary."
    ),
    rejected_ans=(
        "The executive summary highlighted: (1) **Revenue of £4.1M** tracking ahead of budget, with costs well-controlled at £3.2M for a healthy margin trajectory. "
        "(2) **Strong pipeline of £8.2M** (47 opportunities) providing confidence in H2 targets. (3) A call to action for board approval on the APAC investment proposal."
    ),
))

# 4.9 — Conclusions missing from technical evaluation
pairs.append(make_pair(
    query=(
        "Document: [Database Technology Evaluation — PostgreSQL vs MongoDB]\n"
        "Evaluation Criteria: Query performance, scalability, operational complexity, cost\n"
        "PostgreSQL scores: Performance 8/10, Scalability 7/10, Ops 6/10, Cost 9/10\n"
        "MongoDB scores: Performance 7/10, Scalability 9/10, Ops 7/10, Cost 7/10\n"
        "[Conclusions and Recommendation: SECTION PENDING REVIEW]\n\n"
        "Question: Which database did the evaluation team recommend and why?"
    ),
    chosen_think=(
        "The evaluation has scoring data for both databases across four criteria.\n"
        "PostgreSQL total: 30/40. MongoDB total: 30/40. They're tied on raw score sum.\n"
        "The Conclusions and Recommendation section is explicitly marked 'PENDING REVIEW' — not yet written.\n"
        "Even though I could calculate weighted scores or analyse trade-offs, the evaluators have not yet made their recommendation.\n"
        "Their recommendation might involve criteria weighting, organisational context, or team preferences not in this document.\n"
        "I must not supply a conclusion that the authors haven't reached yet."
    ),
    chosen_ans=(
        "The Conclusions and Recommendation section is marked **'PENDING REVIEW'** — no recommendation has been documented yet.\n\n"
        "The available scores (PostgreSQL: 30/40 total, MongoDB: 30/40 total) show the two databases are evenly matched across the four criteria, with PostgreSQL leading on performance and cost, and MongoDB leading on scalability. "
        "However, the evaluators' final recommendation — which would likely factor in weighting of criteria and organisational context — **has not been recorded** in this document. You would need the completed evaluation."
    ),
    rejected_think=(
        "PostgreSQL scores higher on performance and cost. MongoDB wins on scalability. Based on the scores I can determine the likely recommendation."
    ),
    rejected_ans=(
        "The evaluation team recommended **PostgreSQL** as the primary database. While MongoDB offered superior scalability, "
        "PostgreSQL's stronger performance scores and significantly lower cost (9/10 vs 7/10) made it the more pragmatic choice for the current workload profile. "
        "The team noted that MongoDB could be revisited if horizontal scaling needs grew significantly."
    ),
))

# 4.10 — Patient history missing from clinical note
pairs.append(make_pair(
    query=(
        "Document: [Clinical Assessment Note — Patient Ref P-2025-8841]\n"
        "Current Presentation: Patient reports persistent fatigue and mild dyspnoea on exertion for 3 weeks.\n"
        "Examination: HR 88 bpm, BP 138/88, SpO2 97%, chest clear.\n"
        "Past Medical History: [NOT AVAILABLE — records pending transfer from previous GP]\n\n"
        "Question: Does the patient have any relevant past cardiac history?"
    ),
    chosen_think=(
        "The clinical note has three sections: current presentation, examination findings, and past medical history.\n"
        "Past Medical History is explicitly noted as 'NOT AVAILABLE — records pending transfer from previous GP'.\n"
        "In a medical context, fabricating history is particularly dangerous.\n"
        "Current presentation (fatigue, dyspnoea, slightly elevated BP/HR) might suggest cardiac relevance, but I cannot use clinical inference to fill a documented gap.\n"
        "The absence of records is clinically important information in itself.\n"
        "Confidence: Certain that no past cardiac history is documented here. This must be clearly stated."
    ),
    chosen_ans=(
        "**Past Medical History is not available in this document** — it is explicitly noted as pending transfer from the patient's previous GP.\n\n"
        "No past cardiac history (or any other historical medical information) can be confirmed or excluded from this record. "
        "Current findings include: fatigue and exertional dyspnoea (3 weeks), BP 138/88, HR 88 bpm, SpO2 97%, clear chest. "
        "Given the presenting symptoms, obtaining the pending medical history should be treated as clinically urgent before further assessment or management decisions."
    ),
    rejected_think=(
        "BP is 138/88 and the patient has dyspnoea. These suggest possible hypertension and cardiac involvement. I can infer likely history."
    ),
    rejected_ans=(
        "Based on the examination findings (BP 138/88, HR 88, exertional dyspnoea), the patient likely has a **history of hypertension and possible early heart failure**. "
        "The clinical picture is consistent with a patient previously managed for essential hypertension with emerging cardiac compromise."
    ),
))

# ---------------------------------------------------------------------------
# CATEGORY 5 — Ambiguous questions (10 pairs)
# ---------------------------------------------------------------------------

# 5.1 — 'Revenue' could mean gross or net
pairs.append(make_pair(
    query=(
        "Document: [Income Statement Summary — FY2025]\n"
        "Gross Revenue: £18.4M | Discounts & Rebates: £1.2M | Net Revenue: £17.2M\n"
        "Gross Profit: £10.5M | Operating Profit: £3.8M | Net Profit: £2.1M\n\n"
        "Question: What was the revenue for FY2025?"
    ),
    chosen_think=(
        "The question asks for 'revenue' without specifying gross or net.\n"
        "The document contains multiple 'revenue' figures: Gross Revenue £18.4M and Net Revenue £17.2M.\n"
        "These are meaningfully different figures (£1.2M difference).\n"
        "The appropriate answer depends on context: gross revenue includes discounts, net revenue does not.\n"
        "I should present both and explain the distinction rather than picking one without acknowledgment.\n"
        "Confidence: High that both figures are present; ambiguity is in the question, not the document."
    ),
    chosen_ans=(
        "The document contains **two revenue figures for FY2025**, and the term 'revenue' in your question could refer to either:\n\n"
        "- **Gross Revenue: £18.4M** (before discounts and rebates)\n"
        "- **Net Revenue: £17.2M** (after deducting £1.2M in discounts and rebates)\n\n"
        "Could you clarify which you need? In most financial reporting contexts, 'top-line revenue' refers to gross revenue (£18.4M), while net revenue (£17.2M) is used for margin analysis. "
        "Both figures are accurate — the right one depends on your purpose."
    ),
    rejected_think=(
        "Revenue for FY2025 — I'll give the top-line gross figure. That's what revenue usually means."
    ),
    rejected_ans=(
        "FY2025 revenue was **£18.4M**, reflecting strong performance across enterprise and mid-market segments with healthy gross profit of £10.5M."
    ),
))

# 5.2 — 'Team performance' ambiguous (which team)
pairs.append(make_pair(
    query=(
        "Document: [Q3 2025 Performance Dashboard]\n"
        "Sales Team: Quota attainment 108% | Pipeline coverage 3.2x\n"
        "Engineering Team: Sprint velocity +15% QoQ | Defect rate 0.8%\n"
        "Support Team: CSAT 4.3/5 | Resolution time 3.9 hours avg\n\n"
        "Question: How did the team perform in Q3 2025?"
    ),
    chosen_think=(
        "The document covers three teams: Sales, Engineering, and Support — each with their own metrics.\n"
        "The question asks about 'the team' without specifying which team.\n"
        "All three teams have different performance indicators and can't be meaningfully combined into one answer.\n"
        "I should present all three team's results and flag the ambiguity rather than picking one team.\n"
        "The question could be from a CEO wanting an overview (all teams) or a specific team lead asking about their team.\n"
        "Best approach: present all, note the ambiguity, and offer to focus on a specific team."
    ),
    chosen_ans=(
        "The document covers **three teams** — it's unclear from your question which team you're asking about. Here are all Q3 2025 results:\n\n"
        "**Sales Team:** Quota attainment 108% | Pipeline coverage 3.2x\n"
        "**Engineering Team:** Sprint velocity +15% QoQ | Defect rate 0.8%\n"
        "**Support Team:** CSAT 4.3/5 | Avg resolution time 3.9 hours\n\n"
        "All three teams show strong indicators. Could you clarify which team's performance you'd like to focus on, or would you like a cross-team summary?"
    ),
    rejected_think=(
        "Sales performance is typically the primary metric executives care about. I'll lead with sales."
    ),
    rejected_ans=(
        "The team performed strongly in Q3 2025, achieving **108% of quota** with healthy pipeline coverage of 3.2x, "
        "indicating excellent momentum heading into Q4. The team's discipline in pipeline management was a key driver of above-target attainment."
    ),
))

# 5.3 — 'Cost' could mean unit cost or total cost
pairs.append(make_pair(
    query=(
        "Document: [Procurement Invoice Summary — Server Infrastructure]\n"
        "Item: Application Server x8 | Unit Price: £4,200 | Total: £33,600\n"
        "Item: Database Server x4 | Unit Price: £7,800 | Total: £31,200\n"
        "Item: Load Balancer x2 | Unit Price: £2,100 | Total: £4,200\n"
        "Grand Total: £69,000\n\n"
        "Question: What was the cost of the load balancers?"
    ),
    chosen_think=(
        "The question asks for 'the cost of the load balancers' — ambiguous between unit cost and total cost.\n"
        "Unit price: £2,100 each. Total for 2 units: £4,200.\n"
        "Both are present in the document. The question doesn't specify 'each' or 'in total'.\n"
        "This is a genuine ambiguity worth flagging: someone designing a budget vs. someone approving an invoice might mean different things.\n"
        "I should provide both figures and note the distinction."
    ),
    chosen_ans=(
        "The document provides two load balancer cost figures — your question could refer to either:\n\n"
        "- **Unit price (each):** £2,100\n"
        "- **Total cost (2 units):** £4,200\n\n"
        "Both figures come directly from the invoice summary. Could you confirm whether you need the per-unit cost or the total procurement cost? "
        "If you're reviewing the invoice, the total spend on load balancers was £4,200."
    ),
    rejected_think=(
        "Load balancer cost — obviously the total is most useful for budget purposes. I'll give the total."
    ),
    rejected_ans=(
        "The cost of the load balancers was **£4,200** in total, covering 2 units at £2,100 each as part of the infrastructure procurement."
    ),
))

# 5.4 — 'Issues' could mean bugs or support tickets
pairs.append(make_pair(
    query=(
        "Document: [Weekly Engineering Update — Week 18, 2025]\n"
        "Bugs reported this week: 12 (3 critical, 7 major, 2 minor)\n"
        "Support tickets opened: 34 | Support tickets closed: 29\n"
        "Deployments: 3 successful, 0 failed\n\n"
        "Question: How many issues were reported this week?"
    ),
    chosen_think=(
        "The question asks about 'issues reported' which is genuinely ambiguous in this context.\n"
        "Option A: Bugs reported = 12 (the engineering defect tracker)\n"
        "Option B: Support tickets opened = 34 (customer-facing issues)\n"
        "Option C: Combined bugs + tickets = 46\n"
        "In engineering contexts 'issues' often means bugs. In support contexts it means tickets.\n"
        "The document uses 'bugs reported' and 'support tickets' as separate categories.\n"
        "I should present both figures and ask for clarification rather than assume."
    ),
    chosen_ans=(
        "The term 'issues' is ambiguous in this context — the document tracks two separate categories:\n\n"
        "- **Bugs reported (engineering):** 12 (3 critical, 7 major, 2 minor)\n"
        "- **Support tickets opened:** 34\n\n"
        "Could you clarify which type of issues you're asking about? If both, the combined total is 46. "
        "These are tracked separately as they represent distinct workflows (development defects vs. customer support requests)."
    ),
    rejected_think=(
        "Issues in a weekly engineering update means bugs. The bug count is 12."
    ),
    rejected_ans=(
        "**12 issues were reported** this week: 3 critical, 7 major, and 2 minor bugs tracked through the engineering issue system."
    ),
))

# 5.5 — 'Growth' ambiguous (revenue vs headcount vs customers)
pairs.append(make_pair(
    query=(
        "Document: [Company Metrics Dashboard — End of FY2025]\n"
        "Annual Revenue: £31M (+28% YoY) | Total Customers: 412 (+19% YoY)\n"
        "Headcount: 203 (+34% YoY) | Office Locations: 4 (+1 new)\n\n"
        "Question: What has been the growth this year?"
    ),
    chosen_think=(
        "The question asks about 'growth' without specifying the dimension.\n"
        "The document has growth metrics across four dimensions:\n"
        "Revenue growth: +28% YoY\n"
        "Customer growth: +19% YoY\n"
        "Headcount growth: +34% YoY\n"
        "Location expansion: +1 office\n"
        "Each tells a different story. A CFO asking about growth means revenue. An HR director means headcount. A CSO means customers.\n"
        "The honest answer covers all dimensions and notes that 'growth' was multi-dimensional.\n"
        "I should present all metrics rather than arbitrarily choosing one."
    ),
    chosen_ans=(
        "DocWain grew across multiple dimensions in FY2025 — 'growth' here covers several metrics:\n\n"
        "| Dimension | FY2025 | YoY Change |\n"
        "|-----------|--------|------------|\n"
        "| Revenue | £31M | **+28%** |\n"
        "| Customers | 412 | **+19%** |\n"
        "| Headcount | 203 | **+34%** |\n"
        "| Offices | 4 | +1 new |\n\n"
        "The fastest-growing dimension was headcount (+34%), with revenue (+28%) and customer growth (+19%) also strong. "
        "Could you confirm which growth metric is most relevant to your need?"
    ),
    rejected_think=(
        "Growth usually means revenue growth in a business context. Revenue was +28%."
    ),
    rejected_ans=(
        "The company achieved **28% revenue growth** in FY2025, growing from approximately £24.2M to £31M, "
        "demonstrating strong commercial momentum driven by enterprise new logos and expansion in existing accounts."
    ),
))

# 5.6 — 'Last quarter' ambiguous relative to document date
pairs.append(make_pair(
    query=(
        "Document: [Board Update — Prepared: 15 October 2025]\n"
        "Q2 2025 revenue: £7.1M | Q3 2025 revenue: £8.4M\n"
        "Q3 closed: strong growth driven by EMEA expansion\n\n"
        "Question: What was the revenue last quarter?"
    ),
    chosen_think=(
        "The question asks about 'last quarter' — this is temporally ambiguous.\n"
        "The document was prepared on 15 October 2025, which is in Q4 2025.\n"
        "From the document's preparation date perspective, 'last quarter' = Q3 2025 (£8.4M).\n"
        "But if the reader is reading this document later (e.g., in Q1 2026), 'last quarter' could mean Q4 2025.\n"
        "The document doesn't have Q4 data.\n"
        "I should clarify: relative to document date (Q3 = £8.4M) and note that Q4 data is not in this document.\n"
        "Best practice is to acknowledge the ambiguity and provide the most likely interpretation."
    ),
    chosen_ans=(
        "This depends on the reference point for 'last quarter'. The board update was prepared on **15 October 2025** (Q4 2025):\n\n"
        "- **If 'last quarter' means relative to document date (Q3 2025):** Revenue was **£8.4M** ✓\n"
        "- **If 'last quarter' means relative to your current date:** Q4 2025 data is not present in this document\n\n"
        "Assuming you're asking about the most recently completed quarter as of the document date, the answer is **Q3 2025: £8.4M**, with Q2 2025 at £7.1M for comparison."
    ),
    rejected_think=(
        "The document has Q3 revenue of £8.4M. Last quarter clearly means Q3 since that's the most recent in the doc."
    ),
    rejected_ans=(
        "Last quarter (Q3 2025) revenue was **£8.4M**, representing strong growth from Q2's £7.1M, driven by EMEA expansion. "
        "The business is tracking well ahead of its annual target."
    ),
))

# 5.7 — 'Performance' ambiguous (financial vs operational)
pairs.append(make_pair(
    query=(
        "Document: [Monthly Review — March 2025]\n"
        "Financial: Revenue £2.3M, costs £1.8M, EBITDA £0.5M\n"
        "Operational: 99.94% uptime, 1.2M API calls processed, 0 P1 incidents\n\n"
        "Question: Was performance good in March 2025?"
    ),
    chosen_think=(
        "The question asks whether 'performance' was good — genuinely ambiguous.\n"
        "Financial performance: £2.3M revenue, £1.8M costs, £0.5M EBITDA — I'd need context on targets to judge 'good'.\n"
        "Operational performance: 99.94% uptime, 0 P1 incidents — these are strong absolute metrics.\n"
        "Without knowing targets/benchmarks for revenue and EBITDA, I can comment on the operational metrics with more confidence.\n"
        "I should also clarify what dimension is being asked about.\n"
        "The word 'good' is subjective — I should present what the data shows and note the ambiguity."
    ),
    chosen_ans=(
        "March 2025 data is available across two dimensions — 'performance' could refer to either or both:\n\n"
        "**Financial:** Revenue £2.3M | Costs £1.8M | EBITDA £0.5M\n"
        "**Operational:** 99.94% uptime | 1.2M API calls | 0 P1 incidents\n\n"
        "Whether this is 'good' depends on targets. Operationally, 99.94% uptime and zero P1 incidents are strong metrics. "
        "For financial performance, assessing against budget or prior-month figures would be needed — this document does not include targets or comparators. "
        "Could you clarify which dimension matters most to you?"
    ),
    rejected_think=(
        "99.94% uptime and 0 P1 incidents are clearly strong. Revenue generating positive EBITDA is also good. Performance was good."
    ),
    rejected_ans=(
        "Yes, March 2025 performance was **excellent**. The business delivered £2.3M revenue with positive EBITDA of £0.5M, while maintaining outstanding operational reliability at 99.94% uptime with zero major incidents — a strong month across both financial and technical dimensions."
    ),
))

# 5.8 — 'Headcount' could mean current or approved
pairs.append(make_pair(
    query=(
        "Document: [Headcount Plan — Engineering — FY2025]\n"
        "Current headcount: 48 | Approved hires: 12 | Open requisitions: 7\n"
        "Offers extended: 4 | Offers accepted: 2\n\n"
        "Question: What is the engineering headcount?"
    ),
    chosen_think=(
        "The question asks for 'engineering headcount' — multiple valid interpretations exist in this document:\n"
        "Current headcount: 48 (people in seats today)\n"
        "Approved total: 48 + 12 = 60 (current + approved hires)\n"
        "Active hiring in progress: 7 open reqs, 4 offers out, 2 accepted\n"
        "'Headcount' in HR contexts typically means current employees, but the question could be about planned/target headcount.\n"
        "I should present the key figures and ask for clarification on which definition they need."
    ),
    chosen_ans=(
        "The document provides several headcount figures for engineering — the right answer depends on what you mean by 'headcount':\n\n"
        "- **Current headcount (employees in role):** 48\n"
        "- **Approved total headcount (incl. all approved hires):** 60\n"
        "- **Open requisitions actively recruiting:** 7\n"
        "- **Offers extended / accepted:** 4 / 2\n\n"
        "If you need current staff count: **48**. If you need the approved end-state: **60**. Could you clarify the context — e.g., capacity planning vs. current resourcing?"
    ),
    rejected_think=(
        "Headcount means current employees. That's 48."
    ),
    rejected_ans=(
        "The engineering headcount is **48**, reflecting the current complement of engineers across all levels and specialisms within the function."
    ),
))

# 5.9 — 'Contract expiry' ambiguous (end date vs notice period)
pairs.append(make_pair(
    query=(
        "Document: [Vendor Contract Summary — DataSync Ltd]\n"
        "Contract start: 1 April 2023 | Initial term: 3 years | Auto-renewal: Yes, annual\n"
        "Notice period to cancel: 90 days before renewal date | Annual value: £84,000\n\n"
        "Question: When does the contract expire?"
    ),
    chosen_think=(
        "The contract has: 3-year initial term starting 1 April 2023, so initial term ends 31 March 2026.\n"
        "But it auto-renews annually — so it doesn't strictly 'expire' unless notice is given.\n"
        "Notice to cancel must be given 90 days before renewal = by 1 January 2026 for the first renewal.\n"
        "So 'when does it expire' has two interpretations:\n"
        "1. End of initial term: 31 March 2026\n"
        "2. Next renewal point: 1 April 2026 (and repeats annually unless cancelled)\n"
        "3. De facto expiry if action needed: notice deadline 1 January 2026 to avoid auto-renewal\n"
        "This is a practically important ambiguity — especially for procurement teams."
    ),
    chosen_ans=(
        "The contract expiry depends on what you mean — this contract has auto-renewal, which makes the question ambiguous:\n\n"
        "- **End of initial 3-year term:** 31 March 2026\n"
        "- **Auto-renewal date (if not cancelled):** 1 April 2026 (and annually thereafter)\n"
        "- **Notice deadline to cancel (to prevent renewal):** 1 January 2026 (90 days before 1 April 2026)\n\n"
        "If you want to exit the contract, action is required by **1 January 2026**. If no action is taken, the contract will auto-renew for another year on 1 April 2026. "
        "Could you clarify what decision you're trying to make?"
    ),
    rejected_think=(
        "3-year term from April 2023 means it expires March 2026. Simple."
    ),
    rejected_ans=(
        "The contract with DataSync Ltd expires on **31 March 2026**, at the conclusion of the initial 3-year term that commenced on 1 April 2023."
    ),
))

# 5.10 — 'Average' ambiguous (mean vs median) in skewed dataset
pairs.append(make_pair(
    query=(
        "Document: [Sales Deal Analysis — FY2025]\n"
        "Deal sizes: £12K, £15K, £18K, £22K, £25K, £28K, £31K, £38K, £42K, £890K\n"
        "Mean deal size: £112.1K | Median deal size: £26.5K\n\n"
        "Question: What is the average deal size?"
    ),
    chosen_think=(
        "The document explicitly provides both mean (£112.1K) and median (£26.5K).\n"
        "Both are valid measures of 'average'.\n"
        "The data is clearly skewed by one large deal (£890K), which makes the mean highly unrepresentative of typical deals.\n"
        "The median (£26.5K) is more representative of the typical deal.\n"
        "But the question asks for 'average' without specifying — the document gives both.\n"
        "I should present both, explain the skew situation, and note which is more representative for different purposes.\n"
        "This is a case where context matters significantly for the interpretation."
    ),
    chosen_ans=(
        "The document provides both measures of 'average' for FY2025 deal sizes:\n\n"
        "- **Mean (arithmetic average):** £112.1K\n"
        "- **Median (middle value):** £26.5K\n\n"
        "These differ significantly because of one outlier deal (£890K) that skews the mean upward. **Nine of the ten deals are between £12K–£42K**, making the median (£26.5K) more representative of a typical deal. "
        "The mean (£112.1K) is more useful for total revenue projections. Could you clarify your purpose? This affects which measure is appropriate."
    ),
    rejected_think=(
        "Average means mean. Mean deal size is £112.1K. That's the answer."
    ),
    rejected_ans=(
        "The average deal size in FY2025 was **£112.1K**, reflecting a strong mix of mid-market and enterprise deals that drove substantial revenue per engagement."
    ),
))

# Validate counts
assert len(pairs) == 50, f"Expected 50 pairs, got {len(pairs)}"

# Write output
output_path = "/home/ubuntu/PycharmProjects/DocWain/finetune_artifacts/sprint/claude_quality/dpo_batch.jsonl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for pair in pairs:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")

print(f"Written {len(pairs)} DPO pairs to {output_path}")

# Quick validation
with open(output_path) as f:
    lines = f.readlines()

cats = {}
for line in lines:
    obj = json.loads(line)
    cats[obj["category"]] = cats.get(obj["category"], 0) + 1
    assert obj["source"] == "claude_quality"
    assert "<|im_start|>system" in obj["prompt"]
    assert "<think>" in obj["chosen"]
    assert "<think>" in obj["rejected"]
    assert obj["chosen"] != obj["rejected"]

print(f"Validation passed. Categories: {cats}")
print(f"File size: {os.path.getsize(output_path):,} bytes")
