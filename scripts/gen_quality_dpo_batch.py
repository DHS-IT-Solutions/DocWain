"""Generate 20 DPO quality pairs covering 5 categories (4 each)."""
import json, pathlib

OUT = pathlib.Path("finetune_artifacts/sprint/claude_quality/quality_dpo_batch.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

SYS = ("You are DocWain, an enterprise document intelligence assistant. "
       "You analyse documents with deep contextual understanding, extract structured information, "
       "identify patterns and anomalies, and provide holistic analysis grounded in evidence. "
       "You reason step-by-step before answering, state your confidence level, and cite specific "
       "sources. When information is insufficient, you say so clearly rather than guessing.")

def pair(query, c_think, c_ans, r_think, r_ans, cat):
    prompt = (f"<|im_start|>system\n{SYS}<|im_end|>\n"
              f"<|im_start|>user\n{query}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    chosen  = f"<think>\n{c_think}\n</think>\n\n{c_ans}<|im_end|>"
    rejected = f"<think>\n{r_think}\n</think>\n\n{r_ans}<|im_end|>"
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected,
            "category": "quality_dpo", "source": "claude_quality", "subcategory": cat}

PAIRS = []

# ── Category 1: Deep vs Shallow Analysis (4 pairs) ──────────────────────────

PAIRS.append(pair(
    query="""Analyse this earnings snippet:
Revenue Q1: $4.2M (+18% YoY). Gross margin: 61% (Q1-25: 58%). OpEx: $3.1M (+31% YoY).
EBITDA: $0.4M. Cash: $1.1M (90-day runway). Headcount +22 (all engineering).""",
    c_think=(
        "Revenue growth 18% YoY is healthy but OpEx grew 31% — expenses outpacing revenue. "
        "Gross margin improved 3pp to 61%, positive signal on unit economics. "
        "EBITDA $0.4M is thin relative to revenue $4.2M (~10% margin). "
        "Cash $1.1M with only 90-day runway is a critical risk — burn is ~$367K/month. "
        "Engineering headcount surge (+22) explains OpEx spike; no sales/GTM growth visible. "
        "Key risk: runway may not survive to next fundraise without revenue acceleration."
    ),
    c_ans=(
        "**Financial Health: CAUTION — short runway dominates the risk profile.**\n\n"
        "**Positives:**\n"
        "- Gross margin expanded 3pp to 61%, indicating improving unit economics\n"
        "- 18% YoY revenue growth shows product-market traction\n\n"
        "**Risks:**\n"
        "- OpEx grew 31% vs revenue 18% — expense leverage moving in wrong direction\n"
        "- EBITDA $0.4M (~10% margin) leaves almost no buffer\n"
        "- **Critical:** 90-day cash runway ($1.1M ÷ ~$367K/mo burn) requires immediate fundraising\n"
        "- Headcount growth is entirely engineering — no GTM investment visible\n\n"
        "**Confidence:** High on arithmetic; Medium on burn rate (monthly OpEx split not provided).\n"
        "**Recommendation:** Prioritise bridge financing or revenue acceleration before runway expires."
    ),
    r_think="Revenue went up, margins look okay, cash might be low.",
    r_ans=(
        "The company is growing revenue and margins improved. OpEx is higher but that's normal "
        "when hiring. Cash seems a bit low. Overall looks like a typical early-stage company."
    ),
    cat="deep_vs_shallow"
))

PAIRS.append(pair(
    query="""Contract clause: 'Either party may terminate this Agreement for convenience upon 30 days'
written notice. Upon termination, Client shall pay all fees for work completed through the
termination date, plus a kill fee equal to 20% of the remaining contract value.'
Analyse risk to the client.""",
    c_think=(
        "Termination for convenience is standard but the kill fee is punitive to the client. "
        "20% of *remaining* value means the earlier a client exits the larger the penalty. "
        "If contract is $500K and client exits at 20% completion, remaining=$400K, kill fee=$80K. "
        "30 days notice is short — may not align with project wind-down needs. "
        "Vendor has asymmetric benefit: can also terminate for convenience at 30 days with no kill fee. "
        "This asymmetry is the primary risk to flag."
    ),
    c_ans=(
        "**Risk Assessment: HIGH — asymmetric termination burden on client.**\n\n"
        "**Key risks identified:**\n"
        "1. **Kill fee scales inversely with progress** — exiting early on a $500K contract at 20% "
        "completion costs an additional $80K (20% × $400K remaining). Early exits are heavily penalised.\n"
        "2. **No reciprocal kill fee** — vendor can terminate for convenience with zero penalty beyond "
        "completed-work fees. This asymmetry favours the vendor.\n"
        "3. **30-day notice is tight** — insufficient for complex project wind-down or vendor transition.\n\n"
        "**Recommended negotiation positions:**\n"
        "- Cap kill fee at 10% or apply only after 50% contract completion\n"
        "- Add reciprocal kill fee obligation on vendor\n"
        "- Extend notice to 60 days\n\n"
        "**Confidence:** High — risk assessment based on clause text only; full contract context not provided."
    ),
    r_think="Kill fee exists, that's a cost to the client.",
    r_ans="The client has to pay a 20% kill fee if they terminate early. This adds cost to leaving the contract.",
    cat="deep_vs_shallow"
))

PAIRS.append(pair(
    query="""Support ticket data (30 days):
Tickets: 412. Avg resolution: 4.2 days. CSAT: 67%. Escalations: 38 (9.2%).
Top categories: Billing (34%), Access/Auth (28%), Data export (19%), Other (19%).
Repeat tickets (same user, same issue within 14 days): 61 (14.8%).""",
    c_think=(
        "CSAT 67% is well below the typical SaaS benchmark of 85%+. "
        "4.2-day avg resolution is slow for SaaS; billing and auth issues typically expect same-day. "
        "14.8% repeat tickets signal fixes are not sticking — likely workarounds not root-cause fixes. "
        "9.2% escalation rate is high. Billing 34% suggests billing UX or process issues, not just bugs. "
        "Access/Auth 28% — could indicate onboarding friction or SSO problems. "
        "Combined billing+auth = 62% of tickets; addressing both should move the metrics significantly."
    ),
    c_ans=(
        "**Support Health: POOR — three compounding issues identified.**\n\n"
        "| Metric | Actual | Benchmark | Status |\n"
        "|--------|--------|-----------|--------|\n"
        "| CSAT | 67% | 85%+ | ⚠ Critical |\n"
        "| Avg resolution | 4.2 days | <1 day (billing/auth) | ⚠ Slow |\n"
        "| Escalation rate | 9.2% | <3% | ⚠ High |\n"
        "| Repeat ticket rate | 14.8% | <5% | ⚠ High |\n\n"
        "**Root-cause hypotheses:**\n"
        "1. Billing (34%) + Auth (28%) = 62% of volume — likely systemic product issues, not one-offs\n"
        "2. 14.8% repeats suggest agents are applying workarounds; root fixes are not deployed\n"
        "3. 9.2% escalation indicates L1 agents lack tooling or authority to resolve core issues\n\n"
        "**Priority actions:** (1) Root-cause audit of top 20 billing tickets; (2) Auth/SSO flow review; "
        "(3) Empower L1 with billing adjustment tooling.\n\n"
        "**Confidence:** High on diagnosis; Medium on root causes (ticket detail not provided)."
    ),
    r_think="CSAT is low and there are lots of tickets.",
    r_ans="CSAT at 67% is not great. Billing and auth are the top issues. The team should look into why resolution takes so long.",
    cat="deep_vs_shallow"
))

PAIRS.append(pair(
    query="""Research abstract: 'We found a statistically significant correlation (r=0.73, p<0.001,
n=847) between daily screen time and self-reported sleep quality in adults aged 18-35.
Participants using screens within 1 hour of bedtime reported 23% lower sleep quality scores.
Limitations: self-reported measures, cross-sectional design, convenience sample.'""",
    c_think=(
        "r=0.73 is a strong correlation and p<0.001 with n=847 gives statistical confidence. "
        "However, cross-sectional design means causality cannot be established — direction is unknown. "
        "Self-reported sleep quality is subjective and susceptible to recall bias. "
        "Convenience sample limits generalisability beyond the studied population. "
        "23% lower sleep quality for pre-bedtime screen use is an actionable finding despite limitations. "
        "Key distinction: correlation ≠ causation here; poor sleepers may use screens more (reverse causality)."
    ),
    c_ans=(
        "**Finding strength: MODERATE — strong association, causality not established.**\n\n"
        "**What the data supports:**\n"
        "- Strong positive correlation (r=0.73) between screen time and sleep quality impact\n"
        "- Pre-bedtime screen use (within 1hr) associated with 23% lower sleep quality scores\n"
        "- Adequately powered (n=847, p<0.001)\n\n"
        "**What it does NOT support:**\n"
        "- Causal claims — cross-sectional design cannot determine direction of effect\n"
        "- Generalisation — convenience sample may not represent broader adult population\n"
        "- Objective sleep impairment — self-report measures are susceptible to recall bias\n\n"
        "**Appropriate use of this finding:** Hypothesis generation and policy guidance (with caveats). "
        "Not suitable for strong causal claims in clinical or regulatory contexts.\n\n"
        "**Confidence:** High that association exists; Low that screen time *causes* poor sleep from this study alone."
    ),
    r_think="The study shows screens affect sleep with a significant result.",
    r_ans="This study shows screen time is correlated with worse sleep. The p-value is significant so the finding is reliable. People should avoid screens before bed.",
    cat="deep_vs_shallow"
))

# ── Category 2: Structured vs Unstructured Output (4 pairs) ─────────────────

PAIRS.append(pair(
    query="""Extract all obligations from this clause:
'The Supplier shall deliver the goods within 14 days of order confirmation, provide a
Certificate of Conformance with each shipment, maintain product liability insurance of
no less than £2M, and notify the Buyer within 24 hours of any delay or quality issue.'""",
    c_think=(
        "Four distinct obligations are present in this clause. "
        "I should extract each with: obligation type, party, trigger, deadline/threshold, consequence if stated. "
        "No explicit consequences for breach are stated in this clause alone. "
        "Structured extraction as a table will be clearest for downstream review."
    ),
    c_ans=(
        "**Obligations Extracted: 4**\n\n"
        "| # | Obligation | Party | Trigger | Deadline/Threshold |\n"
        "|---|-----------|-------|---------|-------------------|\n"
        "| 1 | Deliver goods | Supplier | Order confirmation | Within 14 days |\n"
        "| 2 | Provide Certificate of Conformance | Supplier | Each shipment | With shipment |\n"
        "| 3 | Maintain product liability insurance | Supplier | Ongoing | Min £2,000,000 |\n"
        "| 4 | Notify of delay or quality issue | Supplier | On occurrence | Within 24 hours |\n\n"
        "**Notes:**\n"
        "- All obligations fall on the Supplier; Buyer has no obligations in this clause\n"
        "- No breach consequences specified here — check elsewhere in contract\n"
        "- Insurance threshold is a floor (\"no less than\") — Supplier may carry more\n\n"
        "**Confidence:** High — all obligations are explicitly stated."
    ),
    r_think="I can see a few obligations in this clause for the supplier.",
    r_ans="The supplier needs to deliver within 14 days, provide a certificate, have insurance over £2M, and notify about delays within 24 hours.",
    cat="structured_vs_unstructured"
))

PAIRS.append(pair(
    query="""Parse this invoice and return structured data:
INVOICE #INV-2094 | Date: 12 Mar 2026 | Due: 11 Apr 2026
Vendor: Apex Cloud Ltd | VAT: GB123456789
Line 1: Cloud compute (Feb) — 840 hrs @ £0.85 = £714.00
Line 2: Storage 4TB @ £18.50/TB = £74.00
Line 3: Support (Standard) — 1 month = £150.00
Subtotal: £938.00 | VAT 20%: £187.60 | Total: £1,125.60""",
    c_think=(
        "Need to parse: header fields, vendor, line items, and totals. "
        "Verify arithmetic: 840×0.85=714 ✓, 4×18.50=74 ✓, support=150 ✓, subtotal=938 ✓, VAT=187.60 ✓, total=1125.60 ✓. "
        "All figures check out. Return as structured JSON-like block."
    ),
    c_ans=(
        "**Invoice Parsed — all figures verified ✓**\n\n"
        "```\n"
        "invoice_number : INV-2094\n"
        "invoice_date   : 2026-03-12\n"
        "due_date       : 2026-04-11\n"
        "payment_terms  : 30 days\n"
        "vendor_name    : Apex Cloud Ltd\n"
        "vendor_vat     : GB123456789\n"
        "\n"
        "line_items:\n"
        "  1. description : Cloud compute (Feb)\n"
        "     quantity    : 840 hrs\n"
        "     unit_price  : £0.85\n"
        "     amount      : £714.00  ✓\n"
        "  2. description : Storage\n"
        "     quantity    : 4 TB\n"
        "     unit_price  : £18.50/TB\n"
        "     amount      : £74.00   ✓\n"
        "  3. description : Support (Standard)\n"
        "     quantity    : 1 month\n"
        "     unit_price  : £150.00\n"
        "     amount      : £150.00  ✓\n"
        "\n"
        "subtotal       : £938.00   ✓\n"
        "vat_rate       : 20%\n"
        "vat_amount     : £187.60   ✓\n"
        "total_due      : £1,125.60 ✓\n"
        "```\n"
        "**Anomalies:** None — all calculations verified correct."
    ),
    r_think="Invoice has three line items and a total.",
    r_ans="Invoice INV-2094 from Apex Cloud Ltd for £1,125.60 including VAT. It covers compute, storage, and support services.",
    cat="structured_vs_unstructured"
))

PAIRS.append(pair(
    query="""Summarise the key decisions from this meeting note:
Attendees: Sarah (PM), Dev (CTO), Priya (Sales). Date: 10 Apr 2026.
- Agreed to delay v2.0 launch from May to July to complete security audit.
- Priya to present revised pricing to enterprise prospects by 18 Apr.
- Dev confirmed API rate limits will increase to 1000 req/min in v2.0.
- No decision on mobile app — deferred to next sprint review.
- Sarah to draft updated roadmap by 15 Apr and share with board.""",
    c_think=(
        "Need to separate: decisions made, actions assigned, and deferred items. "
        "Three concrete decisions, two action items with owners and due dates, one deferred item. "
        "Structured table for actions, bullet list for decisions, flag for deferred."
    ),
    c_ans=(
        "**Meeting Summary — 10 Apr 2026**\n\n"
        "**Decisions Made (3):**\n"
        "1. v2.0 launch delayed: May → July (reason: security audit completion)\n"
        "2. API rate limits will increase to 1,000 req/min in v2.0 (confirmed by Dev)\n"
        "3. Revised enterprise pricing to be presented by Priya before commitments\n\n"
        "**Action Items:**\n"
        "| Owner | Action | Due |\n"
        "|-------|--------|-----|\n"
        "| Priya | Present revised pricing to enterprise prospects | 18 Apr 2026 |\n"
        "| Sarah | Draft updated roadmap and share with board | 15 Apr 2026 |\n\n"
        "**Deferred (1):**\n"
        "- Mobile app decision — deferred to next sprint review (no date set)\n\n"
        "**Confidence:** High — all items explicitly stated in notes."
    ),
    r_think="There were some decisions and action items.",
    r_ans="The team decided to delay v2.0 to July for a security audit. Priya and Sarah have follow-up tasks. Mobile app was pushed to later.",
    cat="structured_vs_unstructured"
))

PAIRS.append(pair(
    query="""Extract entities from: 'On 3 March 2026, Meridian Pharma plc (LSE: MRDP) announced
that CEO Dr. Angela Reeves signed a $450M licensing deal with Novaris AG (Basel) for the
compound MRD-2291, granting exclusive rights in EU and APAC markets for 15 years.'""",
    c_think=(
        "Entity types to extract: organisations, persons, financial figures, dates, locations, products, legal terms. "
        "Systematic pass: 2 orgs (Meridian, Novaris), 1 person (Dr. Angela Reeves), 1 date, 1 value, 2 markets, 1 compound, 1 duration, 1 ticker."
    ),
    c_ans=(
        "**Entities Extracted:**\n\n"
        "| Type | Value | Context |\n"
        "|------|-------|---------|\n"
        "| Organisation | Meridian Pharma plc | Licensor; LSE listed |\n"
        "| Ticker | MRDP (LSE) | Meridian Pharma |\n"
        "| Organisation | Novaris AG | Licensee; headquartered Basel |\n"
        "| Person | Dr. Angela Reeves | CEO, Meridian Pharma |\n"
        "| Date | 3 March 2026 | Announcement date |\n"
        "| Financial value | $450M | Deal value |\n"
        "| Product/Compound | MRD-2291 | Subject of licence |\n"
        "| Geography | EU | Licensed market |\n"
        "| Geography | APAC | Licensed market |\n"
        "| Location | Basel | Novaris AG HQ |\n"
        "| Deal term | 15 years | Licence duration |\n"
        "| Rights type | Exclusive | Scope of licence |\n\n"
        "**Total entities: 12** across 7 types."
    ),
    r_think="There are companies, a person, and financial details in this text.",
    r_ans="Entities include Meridian Pharma, Novaris AG, Dr. Angela Reeves, a $450M deal, and compound MRD-2291.",
    cat="structured_vs_unstructured"
))

# ── Category 3: Complete vs Partial Extraction (4 pairs) ────────────────────

PAIRS.append(pair(
    query="""List all dates and deadlines from this project update:
'Phase 1 design review must complete by 20 April. UAT kicks off 5 May and runs for 3 weeks.
Go-live is targeted for 30 May, subject to UAT sign-off. Post-launch hypercare runs for
2 weeks. The steering committee meets every first Monday; next is 6 April.'""",
    c_think=(
        "Explicit dates: 20 Apr (design review), 5 May (UAT start), 30 May (go-live target), 6 Apr (steering committee). "
        "Derived dates: UAT end = 5 May + 3 weeks = 26 May. Hypercare end = 30 May + 2 weeks = 13 Jun. "
        "All should be listed including derived ones with calculation shown."
    ),
    c_ans=(
        "**All Dates & Deadlines — 6 total (4 explicit, 2 derived):**\n\n"
        "| Date | Event | Type |\n"
        "|------|-------|------|\n"
        "| 6 Apr 2026 | Steering committee meeting | Explicit |\n"
        "| 20 Apr 2026 | Phase 1 design review deadline | Explicit |\n"
        "| 5 May 2026 | UAT kick-off | Explicit |\n"
        "| 26 May 2026 | UAT end (5 May + 3 weeks) | Derived |\n"
        "| 30 May 2026 | Go-live target | Explicit — conditional on UAT sign-off |\n"
        "| 13 Jun 2026 | Hypercare end (30 May + 2 weeks) | Derived |\n\n"
        "**Conditional flag:** Go-live and all subsequent dates shift if UAT sign-off is delayed.\n"
        "**Confidence:** High on explicit dates; Medium on derived dates (assumes calendar weeks, no holidays)."
    ),
    r_think="I can see a few dates mentioned.",
    r_ans="Key dates: design review by 20 April, UAT from 5 May, go-live on 30 May.",
    cat="complete_vs_partial"
))

PAIRS.append(pair(
    query="""Extract all financial figures from:
'The Group reported revenue of €82.4M for FY2025, up 14% from €72.3M in FY2024.
Gross profit was €49.4M (margin: 60%). Operating costs rose to €41.2M. EBIT was €8.2M.
Net finance costs were €1.4M, yielding PBT of €6.8M. Tax at 19%: €1.3M. Net profit: €5.5M.
EPS: €0.44 (diluted). Dividend proposed: €0.12/share. Net debt: €18.6M.'""",
    c_think=(
        "Need to extract all figures: revenue (current + prior), GP, margin, OpEx, EBIT, finance costs, PBT, tax rate+amount, net profit, EPS, dividend, net debt. "
        "Verify P&L chain: Revenue 82.4 - GP cost (82.4-49.4=33) = GP 49.4 ✓. GP 49.4 - OpEx 41.2 = EBIT 8.2 ✓. EBIT 8.2 - finance 1.4 = PBT 6.8 ✓. Tax 6.8×19%=1.29≈1.3 ✓. NP 6.8-1.3=5.5 ✓. "
        "All 13 figures plus calculated checks."
    ),
    c_ans=(
        "**Financial Figures Extracted — P&L verified ✓**\n\n"
        "| Line | FY2025 | FY2024 | Change |\n"
        "|------|--------|--------|--------|\n"
        "| Revenue | €82.4M | €72.3M | +14% |\n"
        "| Gross Profit | €49.4M | — | 60% margin |\n"
        "| Operating Costs | €41.2M | — | — |\n"
        "| EBIT | €8.2M | — | 9.9% margin |\n"
        "| Net Finance Costs | €1.4M | — | — |\n"
        "| PBT | €6.8M | — | — |\n"
        "| Tax (19%) | €1.3M | — | — |\n"
        "| Net Profit | €5.5M | — | 6.7% margin |\n\n"
        "**Per-share & balance sheet:**\n"
        "- EPS (diluted): €0.44 | Dividend proposed: €0.12/share | Net debt: €18.6M\n\n"
        "**P&L cross-check:** All figures internally consistent ✓"
    ),
    r_think="Revenue and profit are the main figures.",
    r_ans="Revenue was €82.4M (+14%) and net profit was €5.5M. EPS is €0.44 and a dividend of €0.12 is proposed.",
    cat="complete_vs_partial"
))

PAIRS.append(pair(
    query="""What risks are mentioned in this paragraph?
'The project faces delays due to supplier lead times and pending regulatory approval.
The integration with the legacy system may introduce data integrity issues. Budget
overruns are possible if rework is required. Key person dependency on the lead
architect poses a continuity risk. Stakeholder misalignment on scope remains unresolved.'""",
    c_think=(
        "Count and categorise all risks mentioned. I see: schedule risk (supplier/regulatory), technical risk (data integrity), financial risk (budget overrun), resource risk (key person dependency), governance risk (stakeholder misalignment). "
        "Five distinct risks across four categories. Each should be extracted with type and description."
    ),
    c_ans=(
        "**Risks Identified: 5 across 4 categories**\n\n"
        "| # | Risk | Category | Trigger |\n"
        "|---|------|----------|---------|\n"
        "| 1 | Supplier lead time delays | Schedule | External dependency |\n"
        "| 2 | Pending regulatory approval | Schedule/Compliance | External — unresolved |\n"
        "| 3 | Legacy integration data integrity issues | Technical | Integration complexity |\n"
        "| 4 | Budget overrun from rework | Financial | Conditional on defects |\n"
        "| 5 | Key person dependency (lead architect) | Resource/Continuity | Single point of failure |\n"
        "| 6 | Stakeholder scope misalignment | Governance | Unresolved — active risk |\n\n"
        "*Note: Stakeholder misalignment (item 6) is an active, unresolved risk — higher immediacy than the others.*\n\n"
        "**Confidence:** High — all risks explicitly stated in text."
    ),
    r_think="A few risks are mentioned including delays and budget.",
    r_ans="The main risks are supplier delays, regulatory approval, and possible budget overruns. There's also a key person risk.",
    cat="complete_vs_partial"
))

PAIRS.append(pair(
    query="""Extract all named parties and their roles from:
'This Service Agreement is between DataCore Ltd ("Provider"), registered in England,
and Fenwick Group plc ("Client"), acting through its subsidiary Fenwick Digital Ltd.
KPMG LLP ("Auditor") may audit Provider compliance annually. Payments are processed
via Stripe Payments Europe Ltd ("Payment Processor"). Disputes are referred to the
ICC International Court of Arbitration ("Arbitration Body").'""",
    c_think=(
        "Five named parties total: DataCore (Provider), Fenwick Group and Fenwick Digital (Client/subsidiary), KPMG (Auditor), Stripe (Payment Processor), ICC (Arbitration Body). "
        "Note Fenwick Digital is a subsidiary acting on behalf of the Client — nuance worth capturing. "
        "Also capture defined term aliases."
    ),
    c_ans=(
        "**Named Parties Extracted: 5 entities, 6 roles**\n\n"
        "| Defined Term | Legal Entity | Role | Notes |\n"
        "|-------------|-------------|------|-------|\n"
        "| \"Provider\" | DataCore Ltd | Service provider | Registered in England |\n"
        "| \"Client\" | Fenwick Group plc | Client (principal) | Acting through subsidiary |\n"
        "| — | Fenwick Digital Ltd | Client (contracting subsidiary) | Acts on behalf of Fenwick Group |\n"
        "| \"Auditor\" | KPMG LLP | Compliance auditor | Annual audit rights |\n"
        "| \"Payment Processor\" | Stripe Payments Europe Ltd | Payment processing | Third-party processor |\n"
        "| \"Arbitration Body\" | ICC International Court of Arbitration | Dispute resolution | Governs disputes |\n\n"
        "**Notable:** Fenwick Digital Ltd has no defined term alias — its authority derives from Fenwick Group plc's instruction. "
        "This creates a potential gap if Fenwick Digital is not party to the agreement in its own right.\n\n"
        "**Confidence:** High — all parties explicitly named."
    ),
    r_think="There are a few companies mentioned in different roles.",
    r_ans="DataCore is the provider, Fenwick Group is the client, KPMG is the auditor, Stripe handles payments, and ICC handles disputes.",
    cat="complete_vs_partial"
))

# ── Category 4: Grounded vs Speculative Claims (4 pairs) ────────────────────

PAIRS.append(pair(
    query="""Document states: 'Trial arm A achieved 68% response rate vs 52% in arm B (p=0.04).
Median PFS: Arm A 8.3 months, Arm B 6.1 months.' What can we conclude?""",
    c_think=(
        "p=0.04 is statistically significant at α=0.05. Response rate difference is 16pp. "
        "PFS difference is 2.2 months. However, I don't know: sample size, confidence intervals, "
        "trial phase, whether this is ITT or per-protocol analysis, follow-up duration, OS data. "
        "I should state what the data shows vs what additional context is needed for stronger conclusions."
    ),
    c_ans=(
        "**Evidence-based conclusions from stated data:**\n\n"
        "**Supported by the data:**\n"
        "- Arm A shows statistically significant higher response rate: 68% vs 52% (Δ=16pp, p=0.04)\n"
        "- Arm A shows longer median PFS: 8.3 vs 6.1 months (Δ=2.2 months)\n"
        "- p=0.04 meets conventional α=0.05 significance threshold\n\n"
        "**Not supported without additional data:**\n"
        "- Clinical significance of the PFS difference — 2.2 months may or may not be meaningful; "
        "depends on disease area and patient QoL\n"
        "- Overall survival benefit — PFS does not always translate to OS\n"
        "- Effect size confidence intervals not stated — p=0.04 is borderline significant\n"
        "- Generalisability — trial population, phase, and analysis set not provided\n\n"
        "**Confidence:** High that Arm A outperforms Arm B on stated endpoints; "
        "Low on clinical significance and OS without further data."
    ),
    r_think="Arm A has better numbers so it's the better treatment.",
    r_ans="Arm A is clearly the better treatment. The response rate and PFS are both higher, so it should be preferred.",
    cat="grounded_vs_speculative"
))

PAIRS.append(pair(
    query="""This employment letter states: 'Your base salary is £78,000 per annum, reviewed
annually in April. You are eligible for a discretionary bonus of up to 15% of base salary.
Pension: employer contributes 5% of qualifying earnings. Notice period: 3 months (both parties).'
What is the guaranteed total compensation?""",
    c_think=(
        "Guaranteed elements: base salary £78,000. Pension on qualifying earnings — 2025/26 qualifying earnings band upper limit is ~£50,270, so max employer pension = 5% × 50,270 = £2,513.50 (approximately). "
        "The bonus is discretionary — explicitly not guaranteed. "
        "Total guaranteed = salary + minimum pension. I should not include bonus in guaranteed figure and should note pension calculation depends on qualifying earnings band."
    ),
    c_ans=(
        "**Guaranteed Compensation Analysis:**\n\n"
        "| Component | Amount | Guaranteed? |\n"
        "|-----------|--------|-------------|\n"
        "| Base salary | £78,000 p.a. | Yes — contractual |\n"
        "| Employer pension (5% of qualifying earnings) | ~£2,514 p.a.* | Yes — contractual |\n"
        "| Discretionary bonus (up to 15%) | Up to £11,700 | **No** — discretionary |\n\n"
        "**Guaranteed total: ~£80,514 p.a.**\n\n"
        "*Pension estimate uses 2025/26 qualifying earnings upper band (~£50,270). "
        "Actual amount depends on the applicable band in the year of payment.*\n\n"
        "**Maximum total (if full bonus paid): ~£92,214**\n\n"
        "**Important:** The bonus is explicitly 'discretionary' — it cannot be relied upon as guaranteed income. "
        "Notice period of 3 months is mutual.\n\n"
        "**Confidence:** High on base and notice; Medium on pension (band assumption); Low on bonus (employer discretion)."
    ),
    r_think="Add up the salary and potential bonus.",
    r_ans="Total compensation is around £89,700 including the base salary and 15% bonus, plus pension on top.",
    cat="grounded_vs_speculative"
))

PAIRS.append(pair(
    query="""Audit finding: 'Three of fifteen sampled purchase orders (20%) lacked dual authorisation
as required by policy section 4.2. Total value of non-compliant POs: £143,000.'
Assess the risk.""",
    c_think=(
        "20% non-compliance rate in sample is material — not isolated. £143K is the sample value; extrapolating to population requires knowing total PO count and value (not stated). "
        "Risk: unauthorised expenditure, fraud opportunity, policy non-compliance. "
        "Cannot state fraud occurred — only control failure is evidenced. "
        "Must flag what is evidenced vs what is hypothetical."
    ),
    c_ans=(
        "**Risk Assessment — based on audit finding only:**\n\n"
        "**What is evidenced:**\n"
        "- Control failure rate: 20% (3/15 sampled POs) — material non-compliance with policy 4.2\n"
        "- Affected value in sample: £143,000\n"
        "- Control: dual authorisation — a segregation-of-duties control\n\n"
        "**Risk implications:**\n"
        "- 20% failure rate suggests systemic issue, not isolated exception\n"
        "- Absent dual authorisation, a single approver can commit spend without oversight\n"
        "- Segregation-of-duties failure is a key fraud enabler — though no fraud is evidenced here\n\n"
        "**What cannot be concluded from this finding alone:**\n"
        "- Total population exposure (full PO volume/value not stated)\n"
        "- Whether non-compliance was intentional or process gap\n"
        "- Whether any actual loss occurred\n\n"
        "**Recommended actions:** (1) Expand sample to full population; (2) Retrospective review of £143K POs for legitimacy; "
        "(3) Mandatory dual-auth enforcement in ERP system.\n\n"
        "**Confidence:** High on control failure; Low on fraud risk without further investigation."
    ),
    r_think="20% non-compliance is high and there could be fraud.",
    r_ans="This is a serious finding. 20% of POs didn't have dual authorisation which means there could be fraud happening. The £143K is at risk.",
    cat="grounded_vs_speculative"
))

PAIRS.append(pair(
    query="""Report says: 'Customer churn increased from 3.2% to 5.8% MoM in March.
The product team launched a new pricing tier in February.' Are these events connected?""",
    c_think=(
        "The question asks about causation. The document states two facts in sequence but does not assert a causal link. "
        "Temporal correlation exists: pricing change Feb, churn rise March — plausible connection. "
        "However, correlation ≠ causation; other factors could explain the churn increase. "
        "I should state what is evidenced, what is plausible, and what additional data would confirm or refute."
    ),
    c_ans=(
        "**Causal Analysis — evidence-bounded:**\n\n"
        "**What the document states:** Two independent facts — churn rose (3.2%→5.8%) and pricing changed in February. No causal link is asserted in the document.\n\n"
        "**Plausible connection:** Temporal sequence is consistent with pricing driving churn; a 1-month lag is typical for customer response to pricing changes.\n\n"
        "**What would be needed to confirm:**\n"
        "- Churn cohort analysis: did churning customers cite pricing in exit surveys?\n"
        "- Segment breakdown: did the new tier's target segment show higher churn?\n"
        "- External factors: were there competitor launches, service issues, or seasonality in March?\n\n"
        "**What cannot be concluded:** The document does not establish causation. Asserting the pricing change caused the churn spike would be speculative without supporting data.\n\n"
        "**Confidence:** Low on causation from document alone; Medium on correlation being worth investigating."
    ),
    r_think="The pricing change probably caused the churn spike.",
    r_ans="The pricing change in February almost certainly caused the churn increase in March. When prices go up, customers leave. The team should revert the pricing.",
    cat="grounded_vs_speculative"
))

# ── Category 5: Precise vs Vague Language (4 pairs) ─────────────────────────

PAIRS.append(pair(
    query="""Clause: 'Payment shall be made within a reasonable time after receipt of invoice.'
Is this enforceable and what does it mean in practice?""",
    c_think=(
        "'Reasonable time' is legally ambiguous — no fixed period defined. "
        "In UK contract law, 'reasonable time' for commercial payments is often interpreted as 30 days under the Late Payment of Commercial Debts Act 1998 as a fallback. "
        "The clause creates uncertainty: parties may dispute what 'reasonable' means. "
        "Should flag: ambiguity risk, likely legal interpretation, recommended improvement."
    ),
    c_ans=(
        "**Enforceability: YES — but with significant ambiguity risk.**\n\n"
        "**Legal position (UK):**\n"
        "- 'Reasonable time' is a recognised legal concept and the clause is enforceable\n"
        "- Where no specific period is agreed, UK courts typically default to 30 days for B2B transactions "
        "(Late Payment of Commercial Debts (Interest) Act 1998)\n"
        "- Parties may disagree on what is 'reasonable' — litigation risk if disputed\n\n"
        "**Practical risks:**\n"
        "- Payer may interpret 'reasonable' as 60-90 days; payee expects 30 days\n"
        "- Uncertainty makes cash flow forecasting difficult\n"
        "- Late payment interest rights exist but trigger date is ambiguous\n\n"
        "**Recommended redraft:**\n"
        "> 'Payment shall be made within 30 days of the date of receipt of a valid invoice.'\n\n"
        "**Confidence:** High on legal analysis (UK jurisdiction assumed); note: jurisdiction affects interpretation."
    ),
    r_think="The clause is a bit vague about timing.",
    r_ans="'Reasonable time' is vague and could mean different things. It would be better to specify a fixed number of days like 30 or 60 days to avoid confusion.",
    cat="precise_vs_vague"
))

PAIRS.append(pair(
    query="""SLA: 'The system should be mostly available during business hours with minimal downtime.'
Rewrite this as a precise, measurable SLA with appropriate enterprise standards.""",
    c_think=(
        "Current SLA has no measurable targets: 'mostly' and 'minimal' are subjective. "
        "Enterprise SLA standards: 99.9% = 8.7hrs downtime/yr, 99.5% = 43.8hrs/yr. "
        "Need: uptime %, measurement window, business hours definition, exclusions, measurement method, remedies. "
        "Provide a full rewrite with standard enterprise clauses."
    ),
    c_ans=(
        "**Issues with current SLA:** 'Mostly available' and 'minimal downtime' are unmeasurable — not enforceable.\n\n"
        "**Rewritten SLA:**\n\n"
        "> **5.1 Availability Target:** Provider guarantees 99.9% monthly uptime ("
        "\"Service Availability\") measured across all calendar hours, excluding scheduled maintenance windows.\n"
        "> **5.2 Business Hours:** Defined as 08:00–18:00 GMT, Monday–Friday, excluding UK public holidays.\n"
        "> **5.3 Measurement:** Uptime = (Total minutes − Downtime minutes) ÷ Total minutes × 100. "
        "Measured via Provider's monitoring platform, reported monthly.\n"
        "> **5.4 Scheduled Maintenance:** Maximum 4 hours per month; notified ≥72 hours in advance; "
        "excluded from downtime calculation.\n"
        "> **5.5 Remedies:** Monthly uptime < 99.9%: 10% service credit; < 99.0%: 25% service credit; "
        "< 95.0%: termination right.\n\n"
        "**Confidence:** High — standard enterprise SLA structure; specific percentages should be negotiated per criticality."
    ),
    r_think="The SLA needs specific numbers.",
    r_ans="Replace vague terms with a specific uptime percentage like 99.9%, define business hours clearly, and add penalties for breaches.",
    cat="precise_vs_vague"
))

PAIRS.append(pair(
    query="""Policy says: 'Employees should promptly report any suspected security incidents to the
appropriate team.' Identify the precision gaps and rewrite it.""",
    c_think=(
        "Gaps: 'promptly' = no time defined; 'suspected' = no threshold defined; "
        "'appropriate team' = no named team or contact; 'security incidents' = not defined. "
        "Need: specific timeframe, definition of reportable incident, named contact/channel, escalation path."
    ),
    c_ans=(
        "**Precision Gaps Identified: 4**\n\n"
        "| Term | Problem | Fix |\n"
        "|------|---------|-----|\n"
        "| 'promptly' | No time bound — interpretable as hours to days | Specify '**within 1 hour**' |\n"
        "| 'suspected security incidents' | No definition of what qualifies | Define threshold (see below) |\n"
        "| 'appropriate team' | No named team, contact, or channel | Name the CISO/SOC + provide contact |\n"
        "| No escalation path | What if team is unreachable? | Add backup escalation |\n\n"
        "**Rewritten policy clause:**\n\n"
        "> Employees must report any suspected information security incident **within 1 hour** of discovery "
        "to the Security Operations Centre (SOC) via **security@company.com** or **ext. 5555** (24/7). "
        "A 'suspected incident' includes: unauthorised access, malware detection, data loss, or any anomalous "
        "system behaviour. If the SOC is unreachable, escalate directly to the CISO (ciso@company.com).\n\n"
        "**Confidence:** High on gap analysis; rewrite placeholders require company-specific details."
    ),
    r_think="The policy is vague about timing and who to contact.",
    r_ans="The policy should specify a time limit like 1 hour, name the security team directly, and define what counts as a security incident.",
    cat="precise_vs_vague"
))

PAIRS.append(pair(
    query="""Report conclusion: 'The project was somewhat over budget and took longer than expected,
but overall delivered decent value.' Rewrite with precise, evidence-based language.""",
    c_think=(
        "Terms to replace: 'somewhat over budget' (no figure), 'longer than expected' (no duration), "
        "'decent value' (no metric). This conclusion is unpublishable in enterprise reporting. "
        "I should show the transformation from vague to precise, and note that actual figures are needed. "
        "Will demonstrate structure and flag where data must be inserted."
    ),
    c_ans=(
        "**Vagueness audit of original conclusion:**\n\n"
        "| Phrase | Problem | Required data |\n"
        "|--------|---------|---------------|\n"
        "| 'somewhat over budget' | No figure — 1% or 50%? | Budgeted vs actual cost |\n"
        "| 'longer than expected' | No duration | Planned vs actual end date |\n"
        "| 'decent value' | Subjective — no metric | Benefits delivered vs business case |\n\n"
        "**Rewritten template (insert actuals):**\n\n"
        "> The project completed on **[DATE]**, **[N] weeks** behind the planned delivery date of **[DATE]** "
        "(schedule variance: **[+N%]**). Final cost was **£[X]** against a budget of **£[Y]** "
        "(cost variance: **+[N%]**). The project delivered **[key benefit 1]** and **[key benefit 2]**, "
        "against a business case target of **[metric]**. ROI is projected at **[X%]** over **[Y] years** "
        "based on **[assumption]**.\n\n"
        "**Note:** Precise rewrite requires the underlying project data. The original conclusion should not "
        "be published in a board or executive report without quantification.\n\n"
        "**Confidence:** High on structural fix; data values must be sourced from project records."
    ),
    r_think="The conclusion needs numbers instead of vague words.",
    r_ans="Replace 'somewhat over budget' with the actual variance percentage, specify the delay in days or weeks, and quantify the value delivered against the business case metrics.",
    cat="precise_vs_vague"
))

# ── Write output ─────────────────────────────────────────────────────────────

with OUT.open("w", encoding="utf-8") as f:
    for p in PAIRS:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"Written {len(PAIRS)} pairs to {OUT}")
print(f"File size: {OUT.stat().st_size:,} bytes")

# Quick stats
from collections import Counter
cats = Counter(p["subcategory"] for p in PAIRS)
for cat, count in sorted(cats.items()):
    print(f"  {cat}: {count}")
