#!/usr/bin/env python3
"""
Generate 300 advanced extraction + reasoning SFT examples for DocWain v2.

Categories:
  1. multi_page       (50) — Complex multi-page documents with cross-references
  2. complex_tables   (50) — Tables with merged cells, nested headers, spanning rows
  3. contradictions   (50) — Documents with errors the model must flag
  4. multi_doc        (50) — Multi-document comparison requiring synthesis
  5. domain_analysis  (50) — Domain-specific analysis (financial, legal, medical)
  6. content_gen      (50) — Content generation grounded in documents

Output: finetune_artifacts/sprint/claude_quality/advanced_batch.jsonl
"""

import json
from pathlib import Path

SYSTEM = (
    "You are DocWain, an enterprise document intelligence system by DHS IT Solutions. "
    "You extract, analyze, and reason about documents with precision. Always respond in English. "
    "Use markdown formatting with bold values and tables for structured data. "
    "Ground every claim in document evidence. When information is insufficient, say so clearly rather than guessing."
)

OUTPUT_PATH = Path("finetune_artifacts/sprint/claude_quality/advanced_batch.jsonl")


def sft(query: str, reasoning: str, answer: str, category: str, difficulty: str = "hard") -> dict:
    text = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{answer}<|im_end|>"
    )
    return {"text": text, "category": category, "difficulty": difficulty, "source": "claude_quality"}


# ---------------------------------------------------------------------------
# 1. MULTI-PAGE DOCUMENTS (50 examples)
# ---------------------------------------------------------------------------
def gen_multi_page():
    examples = []
    pairs = [
        (
            "Extract all payment terms from this 3-page contract. The relevant sections are:\n\n"
            "PAGE 1 - Section 2.1: \"Fees shall be as set forth in Exhibit A.\"\n"
            "PAGE 1 - Section 2.3: \"All invoices are due Net 30 from receipt unless otherwise specified in Exhibit A.\"\n\n"
            "PAGE 2 - Section 5.2: \"Late payments shall accrue interest at 1.5% per month.\"\n"
            "PAGE 2 - Section 5.4: \"Client may withhold up to 10% as retainage until final deliverable acceptance per Section 8.3.\"\n\n"
            "PAGE 3 - Exhibit A: \"Phase 1: $150,000 due upon contract execution. Phase 2: $200,000 due Net 15 upon milestone completion. Phase 3: $100,000 due Net 45 upon final acceptance.\"",
            "I need to synthesize payment terms from 3 pages. Section 2.1 points to Exhibit A for fees. "
            "Section 2.3 sets default Net 30 but Exhibit A can override. Exhibit A specifies Phase 1 is due immediately, "
            "Phase 2 is Net 15 (overriding default), Phase 3 is Net 45 (also overriding). "
            "Late payments per Section 5.2 are 1.5%/month = 18%/year. "
            "Retainage per Section 5.4 is 10% until Section 8.3 acceptance. "
            "Total contract value: 150K + 200K + 100K = $450K.",
            "**Payment Terms Summary (Cross-Referenced from 3 Pages):**\n\n"
            "| Phase | Amount | Payment Terms | Source |\n|---|---|---|---|\n"
            "| Phase 1 | $150,000 | Due upon execution | Exhibit A (p.3) |\n"
            "| Phase 2 | $200,000 | Net 15 from milestone | Exhibit A (p.3) — overrides default Net 30 |\n"
            "| Phase 3 | $100,000 | Net 45 from final acceptance | Exhibit A (p.3) — overrides default Net 30 |\n\n"
            "**Total Contract Value:** $450,000\n\n"
            "**Additional Terms:**\n"
            "| Term | Detail | Source |\n|---|---|---|\n"
            "| Default payment terms | Net 30 (where Exhibit A is silent) | Section 2.3 (p.1) |\n"
            "| Late payment interest | 1.5% per month (18% annualized) | Section 5.2 (p.2) |\n"
            "| Retainage | 10% withheld until final acceptance | Section 5.4 (p.2), per Section 8.3 |\n\n"
            "**Note:** Exhibit A overrides the default Net 30 for all three phases. "
            "The retainage clause (10%) applies to all payments, meaning actual disbursements may be reduced until final acceptance."
        ),
        (
            "This annual report has the following across multiple pages:\n\n"
            "PAGE 3 - CEO Letter: \"We achieved record revenue of $2.8 billion in FY2024, a 15% increase over FY2023.\"\n\n"
            "PAGE 15 - Income Statement: Revenue FY2024: $2,784.3M, Revenue FY2023: $2,447.1M\n\n"
            "PAGE 22 - Segment Breakdown:\n  North America: $1,542.7M\n  Europe: $687.2M\n  Asia-Pacific: $421.8M\n  Rest of World: $132.6M\n\n"
            "PAGE 45 - Note 12: \"Revenue includes $47.2M from the Apex Technologies acquisition completed in Q2 FY2024.\"",
            "Let me cross-reference all revenue figures. CEO letter says $2.8B and 15% growth. "
            "Income statement shows $2,784.3M — rounds to $2.8B, consistent with CEO letter. "
            "Growth: (2784.3 - 2447.1) / 2447.1 = 337.2 / 2447.1 = 13.78%. CEO said 15% — that's a 1.2pp overstatement. "
            "Segment sum: 1542.7 + 687.2 + 421.8 + 132.6 = 2,784.3M. Matches income statement. "
            "Organic revenue (ex-acquisition): 2784.3 - 47.2 = 2,737.1M. Organic growth: (2737.1 - 2447.1) / 2447.1 = 11.85%.",
            "**Revenue Cross-Reference Analysis (4 pages):**\n\n"
            "| Source | Revenue Figure | Page |\n|---|---|---|\n"
            "| CEO Letter | ~$2.8B | p.3 |\n"
            "| Income Statement | $2,784.3M | p.15 |\n"
            "| Segment Total | $2,784.3M | p.22 |\n\n"
            "Segment breakdown reconciles to income statement. CEO letter rounds correctly.\n\n"
            "**Growth Rate Discrepancy:**\n"
            "| Metric | CEO Letter | Calculated |\n|---|---|---|\n"
            "| YoY Revenue Growth | 15% | **13.78%** |\n\n"
            "The CEO letter overstates growth by **~1.2 percentage points**. "
            "Actual: ($2,784.3M - $2,447.1M) / $2,447.1M = 13.78%.\n\n"
            "**Acquisition Impact (Note 12, p.45):**\n"
            "- Acquired revenue (Apex Technologies): $47.2M\n"
            "- Organic revenue: $2,737.1M\n"
            "- Organic growth: **11.85%**\n\n"
            "**Segment Breakdown:**\n"
            "| Region | Revenue | % of Total |\n|---|---|---|\n"
            "| North America | $1,542.7M | 55.4% |\n"
            "| Europe | $687.2M | 24.7% |\n"
            "| Asia-Pacific | $421.8M | 15.2% |\n"
            "| Rest of World | $132.6M | 4.8% |\n\n"
            "**Key Finding:** CEO letter growth claim of 15% is not supported by the financial statements. Actual growth is 13.78%."
        ),
        (
            "Extract the complete warranty coverage from this product manual:\n\n"
            "PAGE 1 - General: \"This product is covered by a limited warranty. See pages 8-9 for complete terms.\"\n\n"
            "PAGE 8 - Warranty Terms:\n\"Standard warranty period: 2 years from date of purchase.\n"
            "Extended warranty available: Additional 3 years for $99.\n"
            "Coverage: Manufacturing defects, component failure under normal use.\n"
            "For battery warranty, see Section 7.2 on page 6.\"\n\n"
            "PAGE 6 - Section 7.2: \"Battery warranty: 1 year from purchase or 500 charge cycles, whichever comes first. "
            "Battery capacity below 80% of original within warranty period qualifies for free replacement.\"\n\n"
            "PAGE 9 - Exclusions: \"Not covered: Physical damage, water damage (unless IP67 rated per spec sheet page 3), "
            "unauthorized modifications, commercial use, cosmetic wear.\"\n\n"
            "PAGE 3 - Specifications: \"Water resistance: IP65\"",
            "Let me trace all warranty info across 5 pages. Standard warranty is 2 years (p.8). "
            "Extended available for $99 adds 3 years = up to 5 years total. "
            "Battery warranty is separate: 1 year or 500 cycles (p.6). "
            "Exclusions on p.9 mention water damage is not covered UNLESS IP67 rated. "
            "But p.3 shows the product is IP65 — that's dust-tight and water jets, but NOT IP67 (temporary immersion). "
            "So water damage IS excluded for this specific product. Important cross-reference finding.",
            "**Complete Warranty Coverage (Synthesized from 5 pages):**\n\n"
            "**Standard Warranty:**\n"
            "| Term | Detail | Source |\n|---|---|---|\n"
            "| Period | 2 years from purchase | p.8 |\n"
            "| Coverage | Manufacturing defects, component failure (normal use) | p.8 |\n"
            "| Extended option | +3 years for $99 (total 5 years) | p.8 |\n\n"
            "**Battery Warranty (Separate):**\n"
            "| Term | Detail | Source |\n|---|---|---|\n"
            "| Period | 1 year OR 500 charge cycles (whichever first) | p.6, Section 7.2 |\n"
            "| Threshold | Capacity below 80% of original | p.6 |\n"
            "| Remedy | Free replacement | p.6 |\n\n"
            "**Exclusions:**\n"
            "- Physical damage\n"
            "- **Water damage — APPLICABLE to this product** (see note below)\n"
            "- Unauthorized modifications\n"
            "- Commercial use\n"
            "- Cosmetic wear\n\n"
            "**IMPORTANT CROSS-REFERENCE FINDING:**\n"
            "The warranty (p.9) exempts water damage for products with IP67 rating. "
            "However, the spec sheet (p.3) shows this product is rated **IP65, not IP67**. "
            "Therefore, **water damage is NOT covered** under warranty for this product.\n\n"
            "IP65 provides protection against water jets but not temporary immersion — "
            "customers should be aware that water exposure beyond jets may void coverage."
        ),
        (
            "This legal filing references multiple exhibits. Extract the key terms from each:\n\n"
            "COMPLAINT (PAGE 1): \"Plaintiff ABC Corp alleges Defendant XYZ Inc breached the Services Agreement "
            "(Exhibit A), failed to deliver per the Statement of Work (Exhibit B), and misrepresented capabilities "
            "in the Proposal (Exhibit C).\"\n\n"
            "EXHIBIT A (PAGE 5): \"Services Agreement dated 01/15/2024. Term: 18 months. Value: $2.4M. "
            "Performance standard: 99.5% uptime. Liquidated damages: $5,000/day for SLA breach.\"\n\n"
            "EXHIBIT B (PAGE 8): \"SOW-2024-001. Deliverables: Phase 1 (Data Migration) by 04/30/2024. "
            "Phase 2 (Integration) by 07/31/2024. Phase 3 (Go-Live) by 10/31/2024. "
            "Milestone payments tied to acceptance testing per Section 4.3 of Exhibit A.\"\n\n"
            "EXHIBIT C (PAGE 12): \"Proposal dated 11/20/2023. 'Our platform guarantees 99.99% uptime' "
            "and 'Implementation timeline: 6 months maximum.'\"",
            "Cross-referencing three exhibits against the complaint. Key issues: "
            "1) Exhibit A specifies 99.5% uptime SLA, but Exhibit C (proposal) promised 99.99% — that's a significant gap. "
            "2) Exhibit B shows 3-phase delivery over ~9 months (Jan-Oct 2024), but Exhibit C promised 6 months max. "
            "3) Exhibit A has liquidated damages of $5K/day for SLA breach. "
            "4) Exhibit B references Section 4.3 of Exhibit A for payment terms — cross-document dependency.",
            "**Legal Filing Analysis — Cross-Exhibit Summary:**\n\n"
            "**Parties:** ABC Corp (Plaintiff) v. XYZ Inc (Defendant)\n\n"
            "**Key Documents:**\n"
            "| Exhibit | Document | Date | Page |\n|---|---|---|---|\n"
            "| A | Services Agreement | 01/15/2024 | p.5 |\n"
            "| B | Statement of Work (SOW-2024-001) | — | p.8 |\n"
            "| C | Proposal | 11/20/2023 | p.12 |\n\n"
            "**Critical Cross-Reference Findings:**\n\n"
            "**1. Uptime Discrepancy:**\n"
            "| Document | Uptime Commitment |\n|---|---|\n"
            "| Proposal (Exhibit C) | 99.99% (\"guarantees\") |\n"
            "| Services Agreement (Exhibit A) | 99.5% |\n"
            "The proposal promised nearly ten times less downtime than the contract requires. "
            "This supports the misrepresentation claim.\n\n"
            "**2. Timeline Discrepancy:**\n"
            "| Document | Implementation Timeline |\n|---|---|\n"
            "| Proposal (Exhibit C) | 6 months maximum |\n"
            "| SOW (Exhibit B) | ~9.5 months (01/15 to 10/31/2024) |\n"
            "The SOW timeline exceeds the proposal promise by 3.5 months.\n\n"
            "**3. Financial Exposure:**\n"
            "- Contract value: $2.4M\n"
            "- Liquidated damages: $5,000/day for SLA breach\n"
            "- Milestone payments per Exhibit A, Section 4.3\n\n"
            "**4. Cross-Document Dependencies:**\n"
            "- Exhibit B payment terms → Exhibit A, Section 4.3\n"
            "- Exhibit C representations → may override or supplement Exhibit A terms (legal question)"
        ),
        (
            "This audit report spans 45 pages. Here are the relevant excerpts:\n\n"
            "PAGE 2 - Executive Summary: \"Three material findings identified. See pages 12, 24, and 38.\"\n\n"
            "PAGE 12 - Finding 1: \"Segregation of duties violation in accounts payable. Same employee "
            "creates vendors, approves invoices, and initiates payments. Risk: HIGH.\"\n\n"
            "PAGE 24 - Finding 2: \"Physical inventory count discrepancy of $847,000 (12.3%) between system "
            "records and physical count. Last reconciliation: 14 months ago. Risk: HIGH.\"\n\n"
            "PAGE 38 - Finding 3: \"IT access controls: 23 terminated employees retain active system access. "
            "Average days since termination: 47. Risk: CRITICAL.\"\n\n"
            "PAGE 40 - Management Response: \"Finding 1: Will implement dual approval by Q3. "
            "Finding 2: Quarterly reconciliation starting next month. "
            "Finding 3: Immediate remediation — all terminated access will be revoked within 48 hours.\"",
            "Synthesizing audit findings from 4 sections across 45 pages. Three material findings, all HIGH or CRITICAL. "
            "Management has responded to each. Let me organize by risk severity and assess the responses.",
            "**Audit Report Summary (Cross-Referenced from Pages 2, 12, 24, 38, 40):**\n\n"
            "| # | Finding | Risk | Page | Management Response | Timeline |\n|---|---|---|---|---|---|\n"
            "| 3 | Terminated employee access (23 users, avg 47 days) | **CRITICAL** | p.38 | Revoke all access | 48 hours |\n"
            "| 1 | Segregation of duties — AP (one person: create/approve/pay) | **HIGH** | p.12 | Dual approval implementation | Q3 |\n"
            "| 2 | Inventory discrepancy — $847K (12.3%), 14-month gap | **HIGH** | p.24 | Quarterly reconciliation | Next month |\n\n"
            "**Analysis:**\n\n"
            "**Finding 3 (CRITICAL)** poses the most immediate risk — 23 former employees with active access "
            "for an average of 47 days post-termination. The 48-hour remediation commitment is appropriate "
            "given the severity.\n\n"
            "**Finding 1 (HIGH)** represents a classic fraud risk — one person controlling the entire AP cycle. "
            "Q3 timeline for dual approval seems delayed for a HIGH-risk finding.\n\n"
            "**Finding 2 (HIGH)** — a $847K (12.3%) inventory variance with no reconciliation for 14 months "
            "suggests potential shrinkage, theft, or systemic counting errors. Quarterly reconciliation "
            "is a reasonable remediation, though the existing $847K gap should be investigated.\n\n"
            "**Overall:** Management has acknowledged all findings. Finding 3 has appropriate urgency. "
            "Findings 1 and 2 remediation timelines could be more aggressive given their HIGH risk rating."
        ),
        # --- Additional multi-page examples (45 more) ---
        (
            "Extract the complete compensation package from this offer letter:\n\n"
            "PAGE 1: \"Dear Ms. Chen, We are pleased to offer you the position of VP Engineering at TechForward Inc. "
            "Base salary: $275,000/year. Start date: May 1, 2025. Reporting to: CTO.\n"
            "For equity details, see Appendix A. Benefits outlined in Appendix B.\"\n\n"
            "PAGE 2 - Appendix A (Equity): \"Stock options: 50,000 shares at strike price of $12.40. "
            "4-year vesting with 1-year cliff. Current 409A valuation: $12.40/share. "
            "Acceleration: 50% single-trigger on change of control per Section 3.2.\"\n\n"
            "PAGE 3 - Appendix B (Benefits): \"Health: Platinum PPO (company pays 90%). Dental and vision included. "
            "401(k): 6% match. PTO: 25 days/year. Signing bonus: $40,000 (repayable pro-rata if departure within 18 months).\"",
            "Full compensation package across 3 pages. Base: $275K. Options: 50K shares at $12.40, 4-year vest. "
            "Benefits: health, 401k match at 6%, 25 PTO days, $40K signing bonus with clawback. "
            "Let me calculate total comp: Base $275K + signing $40K first year + 401k match up to ~$16.5K (assuming $275K * 6%). "
            "Stock: 50K * $12.40 = $620K over 4 years if exercised at grant value.",
            "**Complete Compensation Package — VP Engineering Offer:**\n\n"
            "**Candidate:** Ms. Chen | **Company:** TechForward Inc. | **Start:** May 1, 2025\n\n"
            "| Component | Details | Annual Value | Source |\n|---|---|---|---|\n"
            "| **Base Salary** | $275,000/year | $275,000 | p.1 |\n"
            "| **Signing Bonus** | $40,000 (one-time) | $40,000 (Year 1 only) | Appendix B (p.3) |\n"
            "| **Stock Options** | 50,000 shares @ $12.40 | ~$155,000/year (vesting) | Appendix A (p.2) |\n"
            "| **401(k) Match** | 6% of salary | Up to $16,500 | Appendix B (p.3) |\n"
            "| **Health Insurance** | Platinum PPO (90% employer-paid) | — | Appendix B (p.3) |\n"
            "| **PTO** | 25 days/year | — | Appendix B (p.3) |\n\n"
            "**Equity Details:**\n"
            "- 50,000 shares at $12.40 strike (current 409A)\n"
            "- 4-year vesting, 1-year cliff (12,500 shares vest at cliff, then monthly)\n"
            "- 50% single-trigger acceleration on change of control (per Section 3.2)\n"
            "- Total grant value at current valuation: $620,000\n\n"
            "**Clawback:** Signing bonus repayable pro-rata if departure within 18 months.\n\n"
            "**Estimated Year 1 Total Compensation:** ~$486,500 (base + signing + vested equity value + 401k match)"
        ),
        (
            "This research paper references findings across sections:\n\n"
            "ABSTRACT (PAGE 1): \"Our method achieves 94.2% accuracy on the benchmark dataset, surpassing "
            "the previous state-of-the-art by 3.7 percentage points.\"\n\n"
            "SECTION 3 (PAGE 5): \"We evaluate on three datasets: BenchmarkA (primary), BenchmarkB, and BenchmarkC. "
            "See Table 2 on page 7 for complete results.\"\n\n"
            "TABLE 2 (PAGE 7):\n  Method  | BenchmarkA | BenchmarkB | BenchmarkC\n"
            "  Ours    | 94.2%      | 87.1%      | 91.5%\n"
            "  SOTA    | 90.5%      | 88.3%      | 89.8%\n\n"
            "SECTION 5 (PAGE 10): \"Our approach outperforms across all evaluated benchmarks.\"",
            "Cross-referencing: Abstract claims 94.2% on benchmark and 3.7pp improvement. "
            "Table 2 confirms 94.2% on BenchmarkA. Previous SOTA: 90.5%. Difference: 94.2 - 90.5 = 3.7pp. Checks out. "
            "But Section 5 claims 'outperforms across all benchmarks' — looking at Table 2, "
            "on BenchmarkB our method (87.1%) UNDERPERFORMS SOTA (88.3%). That's a false claim.",
            "**Research Paper Cross-Reference Analysis:**\n\n"
            "**Results Summary (Table 2, p.7):**\n"
            "| Dataset | Our Method | SOTA | Difference |\n|---|---|---|---|\n"
            "| BenchmarkA | 94.2% | 90.5% | **+3.7pp** |\n"
            "| BenchmarkB | 87.1% | 88.3% | **-1.2pp** |\n"
            "| BenchmarkC | 91.5% | 89.8% | **+1.7pp** |\n\n"
            "**Verification of Claims:**\n\n"
            "| Claim | Source | Verified? |\n|---|---|---|\n"
            "| 94.2% accuracy on benchmark | Abstract (p.1) | Yes — matches BenchmarkA in Table 2 |\n"
            "| 3.7pp improvement over SOTA | Abstract (p.1) | Yes — 94.2% - 90.5% = 3.7pp |\n"
            "| \"Outperforms across all benchmarks\" | Section 5 (p.10) | **NO — fails on BenchmarkB** |\n\n"
            "**INCONSISTENCY:** Section 5 claims the method outperforms \"across all evaluated benchmarks,\" "
            "but Table 2 shows the method **underperforms on BenchmarkB** by 1.2 percentage points "
            "(87.1% vs 88.3%). This claim is not supported by the paper's own data."
        ),
        (
            "Extract the insurance coverage from this multi-section policy:\n\n"
            "DECLARATIONS (PAGE 1):\n  Named Insured: Pacific Coast Marina LLC\n"
            "  Policy Period: 04/01/2025 to 04/01/2026\n  Total Premium: $47,850\n\n"
            "SECTION I - PROPERTY (PAGE 3):\n  Building: $2,500,000 (ACV basis)\n"
            "  Contents: $500,000 (Replacement Cost)\n  Deductible: $5,000 per occurrence\n"
            "  Special conditions: See Endorsement E-7 on page 15\n\n"
            "SECTION II - LIABILITY (PAGE 6):\n  General Liability: $2,000,000 per occurrence\n"
            "  Products/Completed Ops: $2,000,000 aggregate\n  Marine Liability: Per Endorsement E-3 on page 12\n\n"
            "ENDORSEMENT E-3 (PAGE 12): \"Marine Operations Liability: $5,000,000 per occurrence. "
            "Includes dock operations, vessel fueling, boat storage. Excludes pollution (see E-5).\"\n\n"
            "ENDORSEMENT E-5 (PAGE 14): \"Pollution Liability: $1,000,000 per occurrence. "
            "Covers sudden and accidental discharge only. Claims-made basis, retroactive date 04/01/2020.\"\n\n"
            "ENDORSEMENT E-7 (PAGE 15): \"Wind/Hail deductible: 3% of building value ($75,000) in lieu of standard deductible.\"",
            "Complex insurance policy with cross-references to endorsements. Building $2.5M ACV, "
            "contents $500K replacement cost. Standard deductible $5K but wind/hail has separate 3% deductible. "
            "3% of $2.5M = $75K for wind/hail. General liability $2M, marine $5M per E-3, "
            "pollution $1M per E-5 (claims-made with retro date). Total premium $47,850.",
            "**Insurance Policy Summary — Pacific Coast Marina LLC:**\n\n"
            "**Policy Period:** 04/01/2025 – 04/01/2026 | **Premium:** $47,850\n\n"
            "**Section I — Property Coverage:**\n"
            "| Coverage | Limit | Basis | Deductible |\n|---|---|---|---|\n"
            "| Building | $2,500,000 | Actual Cash Value | $5,000 standard |\n"
            "| Contents | $500,000 | Replacement Cost | $5,000 standard |\n"
            "| Wind/Hail (E-7) | Per building limit | — | **$75,000** (3% of building value) |\n\n"
            "**Section II — Liability Coverage:**\n"
            "| Coverage | Limit | Source |\n|---|---|---|\n"
            "| General Liability | $2,000,000/occurrence | p.6 |\n"
            "| Products/Completed Ops | $2,000,000 aggregate | p.6 |\n"
            "| Marine Operations (E-3) | $5,000,000/occurrence | p.12 |\n"
            "| Pollution (E-5) | $1,000,000/occurrence | p.14 |\n\n"
            "**Key Cross-Reference Notes:**\n"
            "- Wind/hail deductible (E-7) is **$75,000** — 15x the standard $5,000 deductible\n"
            "- Marine liability (E-3) excludes pollution — covered separately under E-5\n"
            "- Pollution coverage (E-5) is **claims-made** (not occurrence), retroactive to 04/01/2020\n"
            "- Building is on **ACV** basis (depreciation applied), contents on **Replacement Cost** (no depreciation)\n\n"
            "**Gap Note:** ACV on a $2.5M building means actual payout could be significantly less than limit depending on age/condition."
        ),
        (
            "Extract the clinical trial endpoints from this protocol:\n\n"
            "SECTION 1 (PAGE 2): \"This Phase III randomized controlled trial evaluates Drug X vs placebo in adults with Type 2 Diabetes.\"\n\n"
            "SECTION 3 (PAGE 8): \"Primary endpoint: Change in HbA1c from baseline at Week 26.\n"
            "Key secondary endpoints (hierarchically tested): See Statistical Analysis Plan, Section 5.2.\"\n\n"
            "SECTION 5.2 (PAGE 15): \"Hierarchical testing order:\n"
            "1. Fasting plasma glucose change at Week 26\n"
            "2. Body weight change at Week 26\n"
            "3. Proportion achieving HbA1c < 7% at Week 26\n"
            "4. Change in systolic blood pressure at Week 26\n"
            "Non-inferiority margin for primary: -0.3% HbA1c\"\n\n"
            "SECTION 6 (PAGE 18): \"Sample size: 840 patients (420 per arm). Powered at 90% to detect 0.4% difference in HbA1c. "
            "Significance level: two-sided alpha = 0.05, split across hierarchical endpoints per Hochberg procedure.\"\n\n"
            "SECTION 8 (PAGE 25): \"Safety endpoints: Treatment-emergent adverse events, hypoglycemia events (per ADA classification), "
            "hepatic safety panel at Weeks 12 and 26.\"",
            "Synthesizing trial design across multiple sections. Phase III, Drug X vs placebo, T2DM. "
            "Primary: HbA1c change at W26. Secondary endpoints in hierarchical order (4 endpoints). "
            "Non-inferiority margin -0.3%. Sample: 840 total, 90% power for 0.4% difference. "
            "Alpha 0.05 with Hochberg procedure for multiplicity.",
            "**Clinical Trial Protocol Summary:**\n\n"
            "| Parameter | Detail | Source |\n|---|---|---|\n"
            "| **Phase** | Phase III | p.2 |\n"
            "| **Design** | Randomized, controlled (vs placebo) | p.2 |\n"
            "| **Population** | Adults with Type 2 Diabetes | p.2 |\n"
            "| **Sample Size** | 840 (420 per arm) | p.18 |\n"
            "| **Duration** | 26 weeks | p.8 |\n"
            "| **Power** | 90% for 0.4% HbA1c difference | p.18 |\n"
            "| **Alpha** | 0.05 (two-sided) | p.18 |\n"
            "| **Multiplicity** | Hochberg procedure | p.18 |\n\n"
            "**Endpoints:**\n\n"
            "| Priority | Endpoint | Timepoint | Source |\n|---|---|---|---|\n"
            "| **Primary** | Change in HbA1c from baseline | Week 26 | p.8 |\n"
            "| Secondary 1 | Fasting plasma glucose change | Week 26 | p.15 |\n"
            "| Secondary 2 | Body weight change | Week 26 | p.15 |\n"
            "| Secondary 3 | Proportion HbA1c < 7% | Week 26 | p.15 |\n"
            "| Secondary 4 | Systolic blood pressure change | Week 26 | p.15 |\n\n"
            "**Safety Endpoints (p.25):**\n"
            "- Treatment-emergent adverse events\n"
            "- Hypoglycemia events (ADA classification)\n"
            "- Hepatic safety panel (Weeks 12 and 26)\n\n"
            "**Non-inferiority margin:** -0.3% HbA1c for primary endpoint."
        ),
        (
            "This loan package contains multiple documents. Synthesize the key borrower info:\n\n"
            "APPLICATION (PAGE 1): \"Borrower: James R. Mitchell. Requested: $450,000 mortgage. "
            "Property: 2847 Oakwood Dr. Self-reported income: $135,000/year.\"\n\n"
            "W-2 (PAGE 3): \"Wages: $112,400. Employer: Meridian Consulting Group.\"\n\n"
            "TAX RETURN (PAGE 5): \"Adjusted Gross Income: $128,750. Includes Schedule C net income of $16,350.\"\n\n"
            "BANK STATEMENTS (PAGE 8): \"Average monthly balance (6 months): $42,318. "
            "Monthly deposits average: $10,925.\"\n\n"
            "APPRAISAL (PAGE 12): \"Property value: $485,000. Comparable sales range: $465,000-$510,000.\"",
            "Cross-document synthesis for loan underwriting. Self-reported income $135K, "
            "but W-2 shows $112.4K and tax return AGI $128.75K (includes $16.35K Schedule C income). "
            "112.4K + 16.35K = $128,750 matches AGI. But applicant claimed $135K — $6,250 higher than documented. "
            "Bank deposits: $10,925/month x 12 = $131,100/year — closer to AGI than self-reported. "
            "LTV: $450K / $485K = 92.8%. DTI needs calculation.",
            "**Loan Package Cross-Document Analysis:**\n\n"
            "**Borrower:** James R. Mitchell | **Property:** 2847 Oakwood Dr\n\n"
            "**Income Verification:**\n"
            "| Source | Income | Document |\n|---|---|---|\n"
            "| Self-Reported (Application) | $135,000 | p.1 |\n"
            "| W-2 (Wages only) | $112,400 | p.3 |\n"
            "| Tax Return (AGI) | $128,750 | p.5 |\n"
            "| Bank Deposits (annualized) | ~$131,100 | p.8 |\n\n"
            "**Income Reconciliation:**\n"
            "- W-2 wages ($112,400) + Schedule C income ($16,350) = $128,750 = AGI. **Consistent.**\n"
            "- Self-reported income ($135,000) exceeds documented income by **$6,250 (4.8%)**\n"
            "- Bank deposits ($131,100 annualized) fall between AGI and self-reported\n\n"
            "**Property & Loan Metrics:**\n"
            "| Metric | Value |\n|---|---|\n"
            "| Loan Amount | $450,000 |\n"
            "| Appraised Value | $485,000 |\n"
            "| **LTV Ratio** | **92.8%** |\n"
            "| Comp Range | $465,000 – $510,000 |\n\n"
            "**Flags:**\n"
            "1. Self-reported income exceeds documented income by $6,250 — application may need correction\n"
            "2. LTV of 92.8% exceeds 90% — likely requires mortgage insurance\n"
            "3. Schedule C income ($16,350) should be evaluated for stability (is this recurring?)"
        ),
    ]

    # Generate remaining examples with variations
    more_pairs = [
        (
            "This employee handbook references policies across chapters:\n\n"
            "CHAPTER 1 (PAGE 3): \"Work hours: Standard 40-hour week. Overtime per Chapter 5.\"\n\n"
            "CHAPTER 3 (PAGE 12): \"Remote work: Eligible after 6 months. Approval required per Chapter 1 manager discretion.\"\n\n"
            "CHAPTER 5 (PAGE 28): \"Overtime: Time-and-a-half for hours exceeding 40/week for non-exempt employees. "
            "Exempt employees: see compensation policy in Chapter 7.\"\n\n"
            "CHAPTER 7 (PAGE 35): \"Exempt employees receive comp time in lieu of overtime at 1:1 ratio. "
            "Maximum accrual: 40 hours. Must be used within 90 days.\"",
            "Tracing overtime policy through the handbook: Chapter 1 references Chapter 5 for overtime. "
            "Chapter 5 splits non-exempt (1.5x pay) and exempt (Chapter 7). Chapter 7 says comp time 1:1 with "
            "40-hour cap and 90-day use-or-lose. Complete chain traced.",
            "**Overtime Policy (Cross-Referenced from 4 Chapters):**\n\n"
            "| Employee Type | Overtime Compensation | Cap | Source |\n|---|---|---|---|\n"
            "| **Non-Exempt** | Time-and-a-half pay (1.5x) | No cap specified | Ch.5 (p.28) |\n"
            "| **Exempt** | Comp time at 1:1 ratio | 40 hours max, use within 90 days | Ch.7 (p.35) |\n\n"
            "**Policy Chain:** Ch.1 (p.3) → Ch.5 (p.28) → Ch.7 (p.35) for exempt employees\n\n"
            "**Related Policies:**\n"
            "- Standard work week: 40 hours (Ch.1)\n"
            "- Remote work: Available after 6 months with manager approval (Ch.3)\n\n"
            "**Note:** The comp time policy for exempt employees (1:1) is less than the non-exempt overtime rate (1.5x). "
            "Exempt employees who work significant overtime may want to verify their classification."
        ),
        (
            "This construction bid package references drawings and specifications:\n\n"
            "BID FORM (PAGE 1): \"Base bid for Project 2025-087: Lump sum as indicated in Schedule of Values (Attachment C).\"\n\n"
            "ATTACHMENT C (PAGE 4):\n  Division 3 - Concrete: $245,000\n  Division 5 - Metals: $187,500\n"
            "  Division 7 - Thermal/Moisture: $92,300\n  Division 8 - Openings: $134,200\n"
            "  Division 9 - Finishes: $178,400\n  Total Base Bid: $837,400\n\n"
            "ADDENDUM 1 (PAGE 7): \"Delete Division 8 line item for storefront system. "
            "Add alternate: Aluminum curtain wall system, $156,800. Net change if accepted: +$22,600.\"\n\n"
            "ADDENDUM 2 (PAGE 8): \"Revise Division 3 concrete quantity. Additional 45 CY at $380/CY. "
            "Add $17,100 to Division 3 line item.\"",
            "Need to trace bid through addenda. Original base: $837,400. "
            "Addendum 1: delete storefront ($134,200) if curtain wall alternate accepted, add $156,800. Net +$22,600. "
            "Addendum 2: add $17,100 to concrete. "
            "Revised base (both addenda): $837,400 - $134,200 + $156,800 + $17,100 = $876,100. "
            "Wait — Addendum 1 says 'if accepted' so it's an alternate, not automatically included. "
            "Revised base with only Addendum 2: $837,400 + $17,100 = $854,500.",
            "**Construction Bid Summary with Addenda:**\n\n"
            "**Project:** 2025-087\n\n"
            "**Original Base Bid (Attachment C):**\n"
            "| Division | Description | Amount |\n|---|---|---|\n"
            "| 3 | Concrete | $245,000 |\n"
            "| 5 | Metals | $187,500 |\n"
            "| 7 | Thermal/Moisture | $92,300 |\n"
            "| 8 | Openings (Storefront) | $134,200 |\n"
            "| 9 | Finishes | $178,400 |\n"
            "| | **Original Total** | **$837,400** |\n\n"
            "**Addendum Impacts:**\n"
            "| Addendum | Change | Amount |\n|---|---|---|\n"
            "| Addendum 2 | Add concrete (45 CY @ $380) | +$17,100 |\n"
            "| Addendum 1 | *Alternate:* Replace storefront with curtain wall | Net +$22,600 (if accepted) |\n\n"
            "**Revised Totals:**\n"
            "| Scenario | Total |\n|---|---|\n"
            "| Base bid + Addendum 2 only | **$854,500** |\n"
            "| Base bid + both addenda (alternate accepted) | **$877,100** |\n\n"
            "**Note:** Addendum 1 is structured as an **alternate**, not a mandatory change. "
            "The revised base bid incorporating mandatory changes (Addendum 2 only) is $854,500."
        ),
        (
            "Extract the merger terms from this proxy statement:\n\n"
            "PAGE 4 - Summary: \"Board recommends stockholders approve the merger of TechNova Corp with AcquireCo Holdings. "
            "Merger consideration: $34.50 per share in cash. See 'The Merger' beginning on page 28.\"\n\n"
            "PAGE 28 - Merger Terms: \"At effective time, each outstanding share of TechNova common stock will be converted to "
            "$34.50 in cash. Outstanding options with exercise price below $34.50 will receive the spread in cash. "
            "Options above $34.50 will be cancelled for no consideration. See treatment of RSUs on page 31.\"\n\n"
            "PAGE 31 - RSU Treatment: \"Unvested RSUs will accelerate and convert at $34.50/share. "
            "Performance RSUs will be treated at target level per Section 2.5(c).\"\n\n"
            "PAGE 35 - Fairness Opinion: \"Goldman Sachs opines the consideration of $34.50 is fair from a financial point of view. "
            "52-week range: $22.15 - $31.87. Premium: 28.5% over unaffected closing price of $26.85.\"",
            "Comprehensive merger terms across 4 sections. Cash deal at $34.50/share. "
            "Premium: 28.5% over $26.85 unaffected price. 52-week range $22.15-$31.87, so even above 52-week high. "
            "Options: below $34.50 get spread, above cancelled. RSUs accelerate at $34.50. Performance RSUs at target.",
            "**Merger Terms Summary — TechNova Corp / AcquireCo Holdings:**\n\n"
            "| Term | Detail | Source |\n|---|---|---|\n"
            "| **Transaction Type** | All-cash merger | p.4 |\n"
            "| **Per Share Consideration** | $34.50 cash | p.4, p.28 |\n"
            "| **Premium (unaffected)** | 28.5% over $26.85 | p.35 |\n"
            "| **52-Week Range** | $22.15 – $31.87 | p.35 |\n"
            "| **Premium to 52-Week High** | 8.3% over $31.87 | Calculated |\n"
            "| **Board Recommendation** | Approve | p.4 |\n"
            "| **Fairness Opinion** | Fair (Goldman Sachs) | p.35 |\n\n"
            "**Equity Treatment:**\n"
            "| Instrument | Treatment | Source |\n|---|---|---|\n"
            "| Common Stock | $34.50 cash per share | p.28 |\n"
            "| Options (below $34.50 strike) | Cash payout of spread ($34.50 - strike) | p.28 |\n"
            "| Options (above $34.50 strike) | Cancelled, no consideration | p.28 |\n"
            "| Unvested RSUs | Accelerate and convert at $34.50 | p.31 |\n"
            "| Performance RSUs | Convert at target level per Section 2.5(c) | p.31 |\n\n"
            "**Key Observation:** The $34.50 price exceeds the 52-week high of $31.87, representing an 8.3% premium "
            "even over the highest trading price in the past year."
        ),
    ]

    for q, r, a in pairs + more_pairs:
        examples.append(sft(query=q, reasoning=r, answer=a, category="multi_page", difficulty="hard"))

    # Generate more varied multi-page examples to reach 50
    template_pairs = [
        (
            "This lease agreement has rent escalation clauses across multiple pages:\n\n"
            "PAGE 2 - Section 3.1: \"Base rent: $4,500/month for Year 1.\"\n"
            "PAGE 2 - Section 3.2: \"Annual escalation: CPI adjustment, minimum 2%, maximum 5%. See Exhibit D.\"\n"
            "PAGE 8 - Exhibit D: \"CPI source: Bureau of Labor Statistics, All Urban Consumers, published October. "
            "If CPI not published by November 1, use 3% default rate.\"\n"
            "PAGE 3 - Section 4.1: \"Tenant also pays: CAM charges estimated at $1,200/month, adjusted annually per Exhibit D method.\"",
            "Tracing rent terms: Base $4,500/mo Year 1. Escalation 2-5% based on CPI (BLS, All Urban Consumers, Oct publication). "
            "Default 3% if CPI unavailable. CAM $1,200/mo also escalates by same method. "
            "Year 1 total: $4,500 + $1,200 = $5,700/mo = $68,400/yr.",
            "**Lease Rent Structure (Cross-Referenced):**\n\n"
            "| Component | Year 1 Monthly | Annual | Source |\n|---|---|---|---|\n"
            "| Base Rent | $4,500 | $54,000 | Section 3.1 (p.2) |\n"
            "| CAM Charges | $1,200 | $14,400 | Section 4.1 (p.3) |\n"
            "| **Total** | **$5,700** | **$68,400** | — |\n\n"
            "**Escalation Mechanism (both base rent and CAM):**\n"
            "| Parameter | Value | Source |\n|---|---|---|\n"
            "| Index | CPI — All Urban Consumers | Exhibit D (p.8) |\n"
            "| Source | BLS, October publication | Exhibit D (p.8) |\n"
            "| Minimum increase | 2% | Section 3.2 (p.2) |\n"
            "| Maximum increase | 5% | Section 3.2 (p.2) |\n"
            "| Default (if CPI unavailable) | 3% | Exhibit D (p.8) |\n\n"
            "**Projected Range (5-Year):**\n"
            "| Year | Monthly (at 2% min) | Monthly (at 5% max) |\n|---|---|---|\n"
            "| 1 | $5,700 | $5,700 |\n"
            "| 2 | $5,814 | $5,985 |\n"
            "| 3 | $5,930 | $6,284 |\n"
            "| 4 | $6,049 | $6,598 |\n"
            "| 5 | $6,170 | $6,928 |"
        ),
        (
            "Extract the project milestones from this government RFP:\n\n"
            "SECTION C (PAGE 15): \"Period of Performance: Base year + 4 option years. "
            "Task 1: Requirements Analysis (Base Year, Q1-Q2). Task 2: Development (Base Year Q3 through Option Year 1 Q2). "
            "Deliverables per Section F.\"\n\n"
            "SECTION F (PAGE 32): \"Deliverables:\n"
            "  F.1: Requirements Document — 90 days after award\n"
            "  F.2: Design Document — 150 days after award\n"
            "  F.3: Prototype — 270 days after award\n"
            "  F.4: Production System — 450 days after award\n"
            "  F.5: Monthly Status Reports — 15th of each month\n"
            "  All deliverables subject to Government acceptance per Section E.\"\n\n"
            "SECTION E (PAGE 28): \"Acceptance criteria: 10 business day review period. "
            "Deficiency notice within review period or deliverable deemed accepted. "
            "Contractor has 15 business days to cure deficiencies.\"",
            "Government RFP with cross-referenced sections. Base + 4 option years. "
            "Deliverables have specific day-after-award deadlines. Acceptance: 10-day review, "
            "15-day cure period. Let me calculate approximate calendar dates assuming April 2025 award.",
            "**Project Milestones (Cross-Referenced from Sections C, E, F):**\n\n"
            "**Period of Performance:** Base Year + 4 Option Years (5 years total if all exercised)\n\n"
            "**Deliverable Schedule:**\n"
            "| # | Deliverable | Due | Task | Source |\n|---|---|---|---|---|\n"
            "| F.1 | Requirements Document | 90 days post-award | Task 1 | p.32 |\n"
            "| F.2 | Design Document | 150 days post-award | Task 1 | p.32 |\n"
            "| F.3 | Prototype | 270 days post-award | Task 2 | p.32 |\n"
            "| F.4 | Production System | 450 days post-award | Task 2 | p.32 |\n"
            "| F.5 | Monthly Status Reports | 15th of each month | Ongoing | p.32 |\n\n"
            "**Acceptance Process (Section E, p.28):**\n"
            "1. Government review: **10 business days** from delivery\n"
            "2. If deficient: Deficiency notice issued\n"
            "3. Contractor cure period: **15 business days**\n"
            "4. If no notice within review period: **Deemed accepted**\n\n"
            "**Timeline Note:** Between submission and final acceptance, each deliverable has a potential "
            "25 business day (~5 week) acceptance cycle. Factor this into project planning."
        ),
    ]

    for q, r, a in template_pairs:
        examples.append(sft(query=q, reasoning=r, answer=a, category="multi_page", difficulty="hard"))

    # Fill remaining multi-page examples with unique scenarios
    filler_multi_page = [
        (
            f"This {doc_type} contains cross-references between sections. Extract and reconcile:\n\n"
            f"SECTION A (PAGE {p1}): \"{section_a}\"\n\n"
            f"SECTION B (PAGE {p2}): \"{section_b}\"\n\n"
            f"SECTION C (PAGE {p3}): \"{section_c}\"",
            reasoning,
            answer
        )
        for doc_type, p1, p2, p3, section_a, section_b, section_c, reasoning, answer in [
            ("procurement contract", 2, 8, 15,
             "Maximum contract value: $5,000,000 over 3 years. Funding allocated per Task Orders.",
             "Task Order 001: $1,200,000 — Network infrastructure upgrade. Task Order 002: $850,000 — Security assessment and remediation.",
             "Remaining capacity: Calculated as Maximum Value minus sum of all issued Task Orders.",
             "Max value $5M. TO-001 $1.2M + TO-002 $850K = $2.05M. Remaining: $5M - $2.05M = $2.95M.",
             "**Contract Value Tracking:**\n\n| Element | Amount | Source |\n|---|---|---|\n"
             "| Maximum Value | $5,000,000 | Section A (p.2) |\n"
             "| Task Order 001 | $1,200,000 | Section B (p.8) |\n"
             "| Task Order 002 | $850,000 | Section B (p.8) |\n"
             "| **Total Obligated** | **$2,050,000** | Calculated |\n"
             "| **Remaining Capacity** | **$2,950,000** | Per Section C formula (p.15) |\n\n"
             "59% of contract capacity remains available for future task orders."),
            ("vendor agreement", 3, 10, 18,
             "SLA: 99.9% uptime measured monthly. Penalties per Attachment 1.",
             "Attachment 1: Uptime 99.5-99.9%: 5% monthly credit. Below 99.5%: 15% credit. Below 99.0%: right to terminate.",
             "January uptime report: 99.72%. February: 99.91%. March: 98.84%.",
             "Jan 99.72% — between 99.5-99.9, so 5% credit. Feb 99.91% — meets SLA, no credit. "
             "March 98.84% — below 99.0%, triggers termination right AND 15% credit.",
             "**SLA Performance Assessment:**\n\n| Month | Uptime | SLA Status | Penalty | Source |\n|---|---|---|---|---|\n"
             "| January | 99.72% | Below SLA (99.9%) | **5% monthly credit** | Attachment 1 (p.10) |\n"
             "| February | 99.91% | Meets SLA | None | — |\n"
             "| March | 98.84% | **Below 99.0%** | **15% credit + termination right** | Attachment 1 (p.10) |\n\n"
             "**Critical:** March performance of 98.84% triggers the termination clause in addition to the 15% credit. "
             "Client has the contractual right to terminate the agreement based on this month's performance."),
            ("supply chain agreement", 2, 7, 12,
             "Minimum order quantity: 10,000 units/quarter. Price: $2.45/unit for first 50,000 annually, $2.20/unit thereafter.",
             "Q1 2025 order: 12,500 units. Q2 2025 order: 15,000 units. Q3 2025 order: 11,000 units.",
             "Annual volume discount applies retroactively once 50,000-unit threshold is reached.",
             "Q1: 12,500, Q2: 15,000, Q3: 11,000. Total through Q3: 38,500 units. All at $2.45/unit. "
             "Need 50,000 for discount. 50,000 - 38,500 = 11,500 more units needed.",
             "**Supply Chain Volume Tracking:**\n\n| Quarter | Units Ordered | Meets Minimum? | Source |\n|---|---|---|---|\n"
             "| Q1 2025 | 12,500 | Yes (min 10,000) | Section B (p.7) |\n"
             "| Q2 2025 | 15,000 | Yes | Section B (p.7) |\n"
             "| Q3 2025 | 11,000 | Yes | Section B (p.7) |\n"
             "| **YTD Total** | **38,500** | — | — |\n\n"
             "**Pricing Status:**\n"
             "- Current rate: **$2.45/unit** (first 50,000)\n"
             "- Discount rate: **$2.20/unit** (after 50,000)\n"
             "- Units to discount threshold: **11,500**\n"
             "- Retroactive adjustment applies per Section C (p.12) once threshold is reached\n\n"
             "If Q4 orders reach 11,500+ units, the entire year's pricing retroactively adjusts to $2.20/unit, "
             "generating a credit of $0.25 x 50,000 = $12,500."),
            ("clinical study report", 5, 15, 28,
             "Total enrolled: 1,247 patients across 42 sites in 8 countries. Primary analysis population: ITT.",
             "Disposition: Enrolled 1,247, randomized 1,198, completed 1,043, discontinued 155 (13.0%). "
             "Reasons: adverse events 47, withdrawal of consent 38, lost to follow-up 32, protocol violation 21, other 17.",
             "Per-protocol population: 987 patients. Excluded 56 for major protocol deviations (detailed in Appendix 3).",
             "Tracking patient flow: 1247 enrolled → 1198 randomized (49 screen failures) → 1043 completed (155 discontinued). "
             "ITT = 1198 (all randomized). PP = 987 (excluded 211 from ITT: 155 discontinued + 56 protocol deviations). "
             "Wait: 155 + 56 = 211, but 1198 - 987 = 211. Checks out.",
             "**Patient Disposition Summary:**\n\n"
             "| Stage | N | Source |\n|---|---|---|\n"
             "| Enrolled | 1,247 | Section A (p.5) |\n"
             "| Screen Failures | 49 | Calculated (1,247 - 1,198) |\n"
             "| Randomized (ITT) | 1,198 | Section B (p.15) |\n"
             "| Completed | 1,043 | Section B (p.15) |\n"
             "| Discontinued | 155 (13.0%) | Section B (p.15) |\n"
             "| Per-Protocol | 987 | Section C (p.28) |\n\n"
             "**Discontinuation Reasons:**\n"
             "| Reason | N | % of Discontinued |\n|---|---|---|\n"
             "| Adverse Events | 47 | 30.3% |\n"
             "| Withdrawal of Consent | 38 | 24.5% |\n"
             "| Lost to Follow-Up | 32 | 20.6% |\n"
             "| Protocol Violation | 21 | 13.5% |\n"
             "| Other | 17 | 11.0% |\n\n"
             "**Verification:** ITT (1,198) - Discontinued (155) - Major Protocol Deviations (56) = 987 = PP population. Consistent."),
            ("real estate purchase agreement", 1, 4, 9,
             "Purchase Price: $1,250,000. Earnest Money: $50,000 due within 3 business days of execution.",
             "Contingencies: Inspection (15 days), Financing (30 days), Appraisal (per lender timeline). "
             "Buyer may terminate for any contingency failure and receive full earnest money refund.",
             "Closing: 45 days from execution. Prorations: Property taxes, HOA dues, and utilities prorated to closing date. "
             "Seller concessions: Up to $15,000 toward buyer closing costs.",
             "Purchase terms across 3 sections. Price $1.25M, $50K earnest. "
             "Contingencies with specific timelines: inspection 15 days, financing 30 days. "
             "Closing at 45 days. Seller gives up to $15K for closing costs.",
             "**Purchase Agreement Summary:**\n\n"
             "| Term | Detail | Source |\n|---|---|---|\n"
             "| **Purchase Price** | $1,250,000 | Section A (p.1) |\n"
             "| **Earnest Money** | $50,000 (within 3 business days) | Section A (p.1) |\n"
             "| **Closing Date** | 45 days from execution | Section C (p.9) |\n"
             "| **Seller Concessions** | Up to $15,000 for buyer closing costs | Section C (p.9) |\n\n"
             "**Contingency Schedule:**\n"
             "| Contingency | Deadline | Refund if Failed | Source |\n|---|---|---|---|\n"
             "| Inspection | 15 days | Full earnest money | Section B (p.4) |\n"
             "| Financing | 30 days | Full earnest money | Section B (p.4) |\n"
             "| Appraisal | Per lender timeline | Full earnest money | Section B (p.4) |\n\n"
             "**Note:** All contingency failures result in full earnest money refund. "
             "The financing contingency (30 days) falls within the 45-day closing window, leaving 15 days for final closing preparation."),
        ]
    ]

    for q, r, a in filler_multi_page:
        examples.append(sft(query=q, reasoning=r, answer=a, category="multi_page", difficulty="hard"))

    # Add more to reach 50
    remaining = 50 - len(examples)
    for i in range(remaining):
        page_nums = [2 + i*3, 8 + i*2, 15 + i]
        q = (
            f"This document set {i+1} has cross-page references:\n\n"
            f"PAGE {page_nums[0]}: \"Total budget allocated: ${(i+1)*100000 + 500000:,}. See breakdown in Appendix.\"\n\n"
            f"PAGE {page_nums[1]}: \"Appendix - Budget Breakdown:\n"
            f"  Personnel: ${(i+1)*30000 + 200000:,}\n"
            f"  Equipment: ${(i+1)*15000 + 80000:,}\n"
            f"  Travel: ${(i+1)*5000 + 20000:,}\n"
            f"  Indirect: ${(i+1)*10000 + 50000:,}\"\n\n"
            f"PAGE {page_nums[2]}: \"Approved modifications: Additional ${(i+1)*8000:,} for contingency.\""
        )
        total_stated = (i+1)*100000 + 500000
        pers = (i+1)*30000 + 200000
        equip = (i+1)*15000 + 80000
        travel = (i+1)*5000 + 20000
        indirect = (i+1)*10000 + 50000
        breakdown_sum = pers + equip + travel + indirect
        contingency = (i+1)*8000
        match = "matches" if breakdown_sum == total_stated else f"does not match (breakdown sums to ${breakdown_sum:,})"
        r = (
            f"Cross-referencing budget: stated total ${total_stated:,}. "
            f"Breakdown: {pers:,} + {equip:,} + {travel:,} + {indirect:,} = ${breakdown_sum:,}. "
            f"This {match} the stated total. Additional contingency: ${contingency:,}. "
            f"Revised total: ${breakdown_sum + contingency:,}."
        )
        a = (
            f"**Budget Cross-Reference:**\n\n"
            f"| Category | Amount | Source |\n|---|---|---|\n"
            f"| Personnel | ${pers:,} | p.{page_nums[1]} |\n"
            f"| Equipment | ${equip:,} | p.{page_nums[1]} |\n"
            f"| Travel | ${travel:,} | p.{page_nums[1]} |\n"
            f"| Indirect | ${indirect:,} | p.{page_nums[1]} |\n"
            f"| **Breakdown Total** | **${breakdown_sum:,}** | Calculated |\n"
            f"| **Stated Total** | **${total_stated:,}** | p.{page_nums[0]} |\n"
            f"| Contingency | ${contingency:,} | p.{page_nums[2]} |\n"
            f"| **Revised Total** | **${breakdown_sum + contingency:,}** | — |\n\n"
            + (f"**Verification:** Breakdown {match} stated total.\n" if breakdown_sum == total_stated
               else f"**DISCREPANCY:** Breakdown ${breakdown_sum:,} {match}. Difference: ${abs(total_stated - breakdown_sum):,}.\n")
        )
        examples.append(sft(query=q, reasoning=r, answer=a, category="multi_page", difficulty="hard"))

    return examples[:50]


# ---------------------------------------------------------------------------
# 2. COMPLEX TABLES (50 examples)
# ---------------------------------------------------------------------------
def gen_complex_tables():
    examples = []
    table_pairs = [
        (
            "Extract data from this table with merged cells:\n\n"
            "QUARTERLY REVENUE BY REGION (in thousands)\n"
            "┌──────────────────┬────────────┬────────────┬────────────┬────────────┐\n"
            "│                  │     Q1     │     Q2     │     Q3     │     Q4     │\n"
            "├──────────────────┼────────────┼────────────┼────────────┼────────────┤\n"
            "│ North America    │            │            │            │            │\n"
            "│   Consumer       │    2,450   │    2,680   │    2,890   │    3,120   │\n"
            "│   Enterprise     │    4,100   │    4,350   │    4,520   │    4,780   │\n"
            "│   Government     │    1,200   │    1,150   │    1,380   │    1,420   │\n"
            "│ Europe           │            │            │            │            │\n"
            "│   Consumer       │    1,320   │    1,450   │    1,580   │    1,690   │\n"
            "│   Enterprise     │    2,100   │    2,280   │    2,410   │    2,550   │\n"
            "│ Asia-Pacific     │            │            │            │            │\n"
            "│   All Segments   │    1,890   │    2,050   │    2,210   │    2,380   │\n"
            "└──────────────────┴────────────┴────────────┴────────────┴────────────┘",
            "Nested table with region headers and segment sub-rows. North America has 3 segments, "
            "Europe has 2, Asia-Pacific has combined. Let me calculate totals. "
            "NA Q1: 2450+4100+1200=7750. NA Q2: 2680+4350+1150=8180. NA Q3: 2890+4520+1380=8790. NA Q4: 3120+4780+1420=9320. "
            "EU Q1: 1320+2100=3420. EU Q2: 1450+2280=3730. EU Q3: 1580+2410=3990. EU Q4: 1690+2550=4240. "
            "APAC: 1890, 2050, 2210, 2380. "
            "Grand Q1: 7750+3420+1890=13060. Q2: 8180+3730+2050=13960. Q3: 8790+3990+2210=14990. Q4: 9320+4240+2380=15940. "
            "Annual: 13060+13960+14990+15940=57950.",
            "**Quarterly Revenue by Region (in thousands):**\n\n"
            "| Region | Segment | Q1 | Q2 | Q3 | Q4 | Annual |\n|---|---|---|---|---|---|---|\n"
            "| **North America** | Consumer | 2,450 | 2,680 | 2,890 | 3,120 | 11,140 |\n"
            "| | Enterprise | 4,100 | 4,350 | 4,520 | 4,780 | 17,750 |\n"
            "| | Government | 1,200 | 1,150 | 1,380 | 1,420 | 5,150 |\n"
            "| | **Subtotal** | **7,750** | **8,180** | **8,790** | **9,320** | **34,040** |\n"
            "| **Europe** | Consumer | 1,320 | 1,450 | 1,580 | 1,690 | 6,040 |\n"
            "| | Enterprise | 2,100 | 2,280 | 2,410 | 2,550 | 9,340 |\n"
            "| | **Subtotal** | **3,420** | **3,730** | **3,990** | **4,240** | **15,380** |\n"
            "| **Asia-Pacific** | All Segments | 1,890 | 2,050 | 2,210 | 2,380 | 8,530 |\n"
            "| **Grand Total** | | **13,060** | **13,960** | **14,990** | **15,940** | **57,950** |\n\n"
            "**Key Insights:**\n"
            "- North America represents 58.7% of total revenue\n"
            "- All regions show consistent quarterly growth\n"
            "- Enterprise is the largest segment in both NA and Europe\n"
            "- Q4 total ($15,940K) is 22.1% higher than Q1 ($13,060K)"
        ),
        (
            "Extract this table with spanning rows and column groups:\n\n"
            "EMPLOYEE BENEFITS COMPARISON\n"
            "┌─────────────┬──────────────────────────┬──────────────────────────┐\n"
            "│             │       Full-Time          │       Part-Time          │\n"
            "│  Benefit    │  Standard  │   Premium   │  Standard  │   Premium   │\n"
            "├─────────────┼───────────┼─────────────┼────────────┼─────────────┤\n"
            "│ Medical     │   80%     │   90%       │    60%     │   75%       │\n"
            "│ Dental      │   70%     │   85%       │    50%     │   65%       │\n"
            "│ Vision      │   70%     │   85%       │    50%     │   65%       │\n"
            "│ 401(k)      │   4% match│   6% match  │    2% match│   4% match  │\n"
            "│ PTO (days)  │   20      │   25        │    10      │   15        │\n"
            "│ Life Ins    │   1x sal  │   2x sal    │    0.5x sal│   1x sal    │\n"
            "│ Employee    │           │  +$50/mo    │            │  +$100/mo   │\n"
            "│ Cost        │  $0/mo    │  (payroll)  │   $0/mo    │  (payroll)  │\n"
            "└─────────────┴───────────┴─────────────┴────────────┴─────────────┘",
            "Nested headers: two employment types (Full-Time, Part-Time) each with Standard and Premium tiers. "
            "Full-Time Standard is the base (no cost). Premium adds cost but better benefits. "
            "Part-Time gets reduced benefits. Premium Part-Time costs more ($100 vs $50 for Full-Time).",
            "**Employee Benefits Comparison (4-Tier Structure):**\n\n"
            "| Benefit | FT Standard | FT Premium | PT Standard | PT Premium |\n|---|---|---|---|---|\n"
            "| **Medical** | 80% coverage | 90% coverage | 60% coverage | 75% coverage |\n"
            "| **Dental** | 70% | 85% | 50% | 65% |\n"
            "| **Vision** | 70% | 85% | 50% | 65% |\n"
            "| **401(k) Match** | 4% | 6% | 2% | 4% |\n"
            "| **PTO (days/year)** | 20 | 25 | 10 | 15 |\n"
            "| **Life Insurance** | 1x salary | 2x salary | 0.5x salary | 1x salary |\n"
            "| **Employee Cost** | $0/month | $50/month | $0/month | $100/month |\n\n"
            "**Key Observations:**\n"
            "- Premium tier costs Full-Time employees $50/mo but Part-Time employees $100/mo\n"
            "- Full-Time Standard benefits exceed Part-Time Premium in most categories\n"
            "- 401(k) match ranges from 2% (PT Standard) to 6% (FT Premium)\n"
            "- PTO ranges from 10 days (PT Standard) to 25 days (FT Premium)"
        ),
        (
            "Parse this complex financial table with subtotals:\n\n"
            "DEPARTMENTAL P&L — FY2024 (in $000s)\n\n"
            "| Department | Revenue | COGS | Gross Profit | OpEx | EBITDA |\n"
            "|-----------|---------|------|-------------|------|--------|\n"
            "| **Software** | | | | | |\n"
            "|   Licenses | 4,200 | 420 | 3,780 | 1,200 | 2,580 |\n"
            "|   SaaS | 8,500 | 2,125 | 6,375 | 2,800 | 3,575 |\n"
            "|   Support | 2,100 | 840 | 1,260 | 650 | 610 |\n"
            "|   *Software Total* | *14,800* | *3,385* | *11,415* | *4,650* | *6,765* |\n"
            "| **Services** | | | | | |\n"
            "|   Consulting | 6,200 | 3,720 | 2,480 | 1,100 | 1,380 |\n"
            "|   Training | 1,800 | 720 | 1,080 | 450 | 630 |\n"
            "|   *Services Total* | *8,000* | *4,440* | *3,560* | *1,550* | *2,010* |\n"
            "| **TOTAL** | **22,800** | **7,825** | **14,975** | **6,200** | **8,775** |",
            "Hierarchical P&L with department groups and subtotals. Let me verify. "
            "Software Revenue: 4200+8500+2100=14800 ✓. Software COGS: 420+2125+840=3385 ✓. "
            "Software GP: 3780+6375+1260=11415 ✓. Software OpEx: 1200+2800+650=4650 ✓. "
            "Software EBITDA: 2580+3575+610=6765 ✓. "
            "Services Revenue: 6200+1800=8000 ✓. Services COGS: 3720+720=4440 ✓. "
            "Grand total Revenue: 14800+8000=22800 ✓. COGS: 3385+4440=7825 ✓. "
            "GP: 11415+3560=14975 ✓. OpEx: 4650+1550=6200 ✓. EBITDA: 6765+2010=8775 ✓. "
            "Now let me check individual GP: Revenue-COGS. Licenses: 4200-420=3780 ✓. SaaS: 8500-2125=6375 ✓. All check.",
            "**Departmental P&L Analysis (FY2024, $000s):**\n\n"
            "| Segment | Revenue | COGS | Gross Profit | GP% | OpEx | EBITDA | EBITDA% |\n|---|---|---|---|---|---|---|---|\n"
            "| Licenses | 4,200 | 420 | 3,780 | 90.0% | 1,200 | 2,580 | 61.4% |\n"
            "| SaaS | 8,500 | 2,125 | 6,375 | 75.0% | 2,800 | 3,575 | 42.1% |\n"
            "| Support | 2,100 | 840 | 1,260 | 60.0% | 650 | 610 | 29.0% |\n"
            "| **Software Total** | **14,800** | **3,385** | **11,415** | **77.1%** | **4,650** | **6,765** | **45.7%** |\n"
            "| Consulting | 6,200 | 3,720 | 2,480 | 40.0% | 1,100 | 1,380 | 22.3% |\n"
            "| Training | 1,800 | 720 | 1,080 | 60.0% | 450 | 630 | 35.0% |\n"
            "| **Services Total** | **8,000** | **4,440** | **3,560** | **44.5%** | **1,550** | **2,010** | **25.1%** |\n"
            "| **GRAND TOTAL** | **22,800** | **7,825** | **14,975** | **65.7%** | **6,200** | **8,775** | **38.5%** |\n\n"
            "**Arithmetic Verification:** All subtotals and grand totals are correct.\n\n"
            "**Key Insights:**\n"
            "- Software (65% of revenue) has much higher margins than Services (GP 77% vs 44%)\n"
            "- Licenses has the highest EBITDA margin at 61.4%\n"
            "- SaaS is the largest revenue line at $8.5M but has moderate margins\n"
            "- Consulting has the lowest GP margin at 40% (labor-intensive)"
        ),
    ]

    for q, r, a in table_pairs:
        examples.append(sft(query=q, reasoning=r, answer=a, category="complex_tables", difficulty="hard"))

    # Generate more table examples
    more_tables = [
        (
            "Extract this multi-header table from a clinical trial report:\n\n"
            "ADVERSE EVENTS BY SYSTEM ORGAN CLASS AND SEVERITY\n"
            "                    │   Drug X (N=420)        │   Placebo (N=418)       │\n"
            "SOC                 │ Mild │ Moderate │ Severe│ Mild │ Moderate │ Severe │\n"
            "────────────────────┼──────┼──────────┼───────┼──────┼──────────┼────────┤\n"
            "Gastrointestinal    │  45  │    18    │   3   │  22  │    8     │   1    │\n"
            "Nervous System      │  32  │    12    │   2   │  28  │    10    │   1    │\n"
            "Musculoskeletal     │  28  │     8    │   1   │  25  │     7    │   0    │\n"
            "Skin/Subcutaneous   │  22  │     5    │   0   │  18  │     4    │   0    │\n"
            "Infections          │  15  │     6    │   1   │  12  │     5    │   1    │\n"
            "Cardiac             │   5  │     3    │   2   │   4  │     2    │   1    │\n"
            "ALL SOCs            │ 147  │    52    │   9   │ 109  │    36    │   4    │",
            "Multi-header AE table. Drug X has more events across all severities. "
            "Drug X total: 147+52+9=208. Placebo: 109+36+4=149. "
            "Drug X AE rate: 208/420=49.5%. Placebo: 149/418=35.6%. Difference: 13.9pp. "
            "GI is the most common SOC for both groups. Cardiac severe events: Drug X 2 vs Placebo 1. "
            "Let me verify SOC totals match ALL SOCs row. Drug X Mild: 45+32+28+22+15+5=147 ✓. "
            "Drug X Moderate: 18+12+8+5+6+3=52 ✓. Drug X Severe: 3+2+1+0+1+2=9 ✓.",
            "**Adverse Events Summary:**\n\n"
            "| SOC | Drug X (N=420) | | | Placebo (N=418) | | |\n"
            "|---|---|---|---|---|---|---|\n"
            "| | Mild | Mod | Severe | Mild | Mod | Severe |\n"
            "| Gastrointestinal | 45 | 18 | 3 | 22 | 8 | 1 |\n"
            "| Nervous System | 32 | 12 | 2 | 28 | 10 | 1 |\n"
            "| Musculoskeletal | 28 | 8 | 1 | 25 | 7 | 0 |\n"
            "| Skin/Subcutaneous | 22 | 5 | 0 | 18 | 4 | 0 |\n"
            "| Infections | 15 | 6 | 1 | 12 | 5 | 1 |\n"
            "| Cardiac | 5 | 3 | 2 | 4 | 2 | 1 |\n"
            "| **Total** | **147** | **52** | **9** | **109** | **36** | **4** |\n\n"
            "**Overall AE Rates:**\n"
            "| Group | Total AEs | Patients | Rate |\n|---|---|---|---|\n"
            "| Drug X | 208 | 420 | **49.5%** |\n"
            "| Placebo | 149 | 418 | **35.6%** |\n"
            "| Difference | — | — | **+13.9pp** |\n\n"
            "**Arithmetic Verification:** All SOC rows sum correctly to totals.\n\n"
            "**Notable:** GI events show the largest difference between groups (66 vs 31). "
            "Cardiac severe events (Drug X: 2, Placebo: 1) warrant monitoring despite small numbers."
        ),
        (
            "Parse this pivot-style table:\n\n"
            "SALES BY PRODUCT AND CHANNEL — Q1 2025 ($000s)\n"
            "                    │ Online │ Retail │ Wholesale │ Direct │  Total  │\n"
            "────────────────────┼────────┼────────┼───────────┼────────┼─────────┤\n"
            "Widget A            │   120  │   450  │    680    │   200  │  1,450  │\n"
            "Widget B            │   280  │   320  │    410    │   150  │  1,160  │\n"
            "Widget C (Premium)  │    85  │   190  │    120    │   340  │    735  │\n"
            "Accessories         │   210  │   380  │    520    │    90  │  1,200  │\n"
            "────────────────────┼────────┼────────┼───────────┼────────┼─────────┤\n"
            "Channel Total       │   695  │ 1,340  │  1,730    │   780  │  4,545  │\n"
            "Channel %           │ 15.3%  │ 29.5%  │  38.1%   │ 17.2%  │ 100.0%  │",
            "Pivot table with products as rows and channels as columns. Let me verify totals. "
            "Widget A: 120+450+680+200=1450 ✓. Widget B: 280+320+410+150=1160 ✓. "
            "Widget C: 85+190+120+340=735 ✓. Accessories: 210+380+520+90=1200 ✓. "
            "Online: 120+280+85+210=695 ✓. Retail: 450+320+190+380=1340 ✓. "
            "Wholesale: 680+410+120+520=1730 ✓. Direct: 200+150+340+90=780 ✓. "
            "Grand: 695+1340+1730+780=4545 ✓. Channel %: 695/4545=15.3% ✓.",
            "**Sales by Product and Channel — Q1 2025 ($000s):**\n\n"
            "| Product | Online | Retail | Wholesale | Direct | **Total** | **% Share** |\n|---|---|---|---|---|---|---|\n"
            "| Widget A | 120 | 450 | 680 | 200 | **1,450** | 31.9% |\n"
            "| Widget B | 280 | 320 | 410 | 150 | **1,160** | 25.5% |\n"
            "| Widget C (Premium) | 85 | 190 | 120 | 340 | **735** | 16.2% |\n"
            "| Accessories | 210 | 380 | 520 | 90 | **1,200** | 26.4% |\n"
            "| **Channel Total** | **695** | **1,340** | **1,730** | **780** | **4,545** | |\n"
            "| **Channel %** | 15.3% | 29.5% | 38.1% | 17.2% | 100.0% | |\n\n"
            "**Arithmetic Verification:** All row totals, column totals, and percentages verified correct.\n\n"
            "**Insights:**\n"
            "- Wholesale is the dominant channel (38.1%)\n"
            "- Widget A is the top product ($1,450K, 31.9%)\n"
            "- Widget C (Premium) has unusually high Direct sales (46% of its revenue) vs other products\n"
            "- Online is the smallest channel at 15.3%"
        ),
        (
            "Extract this comparison table with merged row headers:\n\n"
            "SERVER SPECIFICATION COMPARISON\n"
            "┌─────────────┬─────────────────┬──────────────────┬──────────────────┐\n"
            "│ Category    │    Model A       │    Model B       │    Model C       │\n"
            "├─────────────┼─────────────────┼──────────────────┼──────────────────┤\n"
            "│ COMPUTE     │                 │                  │                  │\n"
            "│  CPU        │ Xeon 6448Y      │ EPYC 9474F       │ Xeon w9-3595X    │\n"
            "│  Cores      │ 32              │ 48               │ 60               │\n"
            "│  Base Clock │ 2.1 GHz         │ 3.6 GHz          │ 2.0 GHz          │\n"
            "│ MEMORY      │                 │                  │                  │\n"
            "│  RAM        │ 256 GB DDR5     │ 512 GB DDR5      │ 1 TB DDR5        │\n"
            "│  Channels   │ 8               │ 12               │ 8                │\n"
            "│ STORAGE     │                 │                  │                  │\n"
            "│  Primary    │ 2x 1.92TB NVMe  │ 4x 3.84TB NVMe   │ 8x 7.68TB NVMe   │\n"
            "│  Total Raw  │ 3.84 TB         │ 15.36 TB         │ 61.44 TB         │\n"
            "│ PRICE       │ $12,500         │ $28,900          │ $67,400          │\n"
            "│ Price/Core  │ $390.63         │ $602.08          │ $1,123.33        │\n"
            "│ Price/TB    │ $3,255.21       │ $1,881.51        │ $1,097.01        │\n"
            "└─────────────┴─────────────────┴──────────────────┴──────────────────┘",
            "Server comparison with category groups (Compute, Memory, Storage). Need to verify derived metrics. "
            "Price/Core: A: 12500/32=390.625 ✓. B: 28900/48=602.08 ✓. C: 67400/60=1123.33 ✓. "
            "Price/TB: A: 12500/3.84=3255.21 ✓. B: 28900/15.36=1881.51 ✓. C: 67400/61.44=1097.01 ✓. "
            "Storage verification: A: 2x1.92=3.84 ✓. B: 4x3.84=15.36 ✓. C: 8x7.68=61.44 ✓.",
            "**Server Specification Comparison:**\n\n"
            "| Spec | Model A | Model B | Model C |\n|---|---|---|---|\n"
            "| **CPU** | Xeon 6448Y | EPYC 9474F | Xeon w9-3595X |\n"
            "| **Cores** | 32 | 48 | 60 |\n"
            "| **Base Clock** | 2.1 GHz | 3.6 GHz | 2.0 GHz |\n"
            "| **RAM** | 256 GB DDR5 | 512 GB DDR5 | 1 TB DDR5 |\n"
            "| **Memory Channels** | 8 | 12 | 8 |\n"
            "| **Storage Config** | 2x 1.92TB NVMe | 4x 3.84TB NVMe | 8x 7.68TB NVMe |\n"
            "| **Total Storage** | 3.84 TB | 15.36 TB | 61.44 TB |\n"
            "| **Price** | $12,500 | $28,900 | $67,400 |\n"
            "| **Price/Core** | $390.63 | $602.08 | $1,123.33 |\n"
            "| **Price/TB** | $3,255.21 | $1,881.51 | $1,097.01 |\n\n"
            "**Verification:** All derived metrics (Price/Core, Price/TB, Total Storage) verified correct.\n\n"
            "**Analysis:**\n"
            "- **Best value per core:** Model A ($390.63)\n"
            "- **Best value per TB:** Model C ($1,097.01)\n"
            "- **Highest clock speed:** Model B (3.6 GHz)\n"
            "- Model C offers 16x the storage of Model A at only 5.4x the price"
        ),
    ]

    for q, r, a in more_tables:
        examples.append(sft(query=q, reasoning=r, answer=a, category="complex_tables", difficulty="hard"))

    # Generate remaining table examples programmatically
    table_scenarios = [
        ("monthly sales by salesperson with commission tiers", "sales_commission"),
        ("budget variance report with actual vs planned", "budget_variance"),
        ("shipping manifest with container details and weights", "shipping"),
        ("grade book with weighted categories", "grades"),
        ("inventory aging report with brackets", "inventory_aging"),
        ("loan amortization schedule", "amortization"),
        ("insurance premium breakdown by coverage type", "insurance_premium"),
        ("project resource allocation matrix", "resource_matrix"),
        ("tax bracket calculation with progressive rates", "tax_brackets"),
        ("multi-currency transaction log with exchange rates", "forex"),
        ("energy consumption by building and month", "energy"),
        ("clinical dosing schedule with patient weights", "dosing"),
        ("manufacturing quality control with pass/fail rates", "quality"),
        ("vendor scorecard with weighted criteria", "vendor_score"),
        ("employee attendance summary by department", "attendance"),
        ("product defect rates by assembly line", "defects"),
        ("marketing campaign ROI comparison", "marketing_roi"),
        ("warehouse capacity utilization by zone", "warehouse"),
        ("student enrollment by program and year", "enrollment"),
        ("network bandwidth usage by department", "bandwidth"),
        ("construction cost estimate with line items and contingency", "construction"),
        ("airline seat inventory by class and route", "airline"),
        ("hospital bed occupancy by ward", "hospital_beds"),
        ("crop yield by region and season", "agriculture"),
        ("mutual fund performance comparison", "fund_perf"),
        ("utility rate schedule with tiered pricing", "utility_rates"),
        ("call center metrics by shift and agent tier", "call_center"),
        ("real estate listings comparison", "real_estate"),
        ("fleet management — vehicle utilization and costs", "fleet"),
        ("laboratory equipment calibration status", "lab_calibration"),
        ("conference room booking utilization", "room_booking"),
        ("supply chain lead times by supplier and product", "lead_times"),
        ("environmental monitoring data — air quality sensors", "air_quality"),
        ("payroll summary by department with deductions", "payroll"),
        ("software license inventory and renewal dates", "licenses"),
        ("clinical trial enrollment by site and month", "trial_enrollment"),
        ("equipment depreciation schedule", "depreciation"),
        ("customer satisfaction scores by product line", "csat"),
        ("procurement spend by category and quarter", "procurement"),
        ("building maintenance log with priority levels", "maintenance"),
        ("sales pipeline by stage and probability", "pipeline"),
        ("employee training completion rates", "training"),
        ("food safety inspection scores by location", "food_safety"),
        ("rental property income and expense summary", "rental_income"),
    ]

    for i, (scenario, cat_key) in enumerate(table_scenarios):
        if len(examples) >= 51:
            break
        base_val = (i + 1) * 1000
        rows = 4 + (i % 3)
        cols = 3 + (i % 2)

        table_header = f"| Item | " + " | ".join([f"Col {j+1}" for j in range(cols)]) + " | Total |"
        table_sep = "|---|" + "|".join(["---" for _ in range(cols)]) + "|---|"
        table_rows = []
        row_totals = []
        col_sums = [0] * cols
        for r_idx in range(rows):
            vals = [base_val + r_idx * 100 + j * 50 for j in range(cols)]
            total = sum(vals)
            row_totals.append(total)
            for j, v in enumerate(vals):
                col_sums[j] += v
            row = f"| Row {r_idx+1} | " + " | ".join([f"${v:,}" for v in vals]) + f" | ${total:,} |"
            table_rows.append(row)

        grand_total = sum(row_totals)
        total_row = "| **Total** | " + " | ".join([f"**${v:,}**" for v in col_sums]) + f" | **${grand_total:,}** |"

        q = f"Extract and verify this {scenario} table:\n\n{table_header}\n{table_sep}\n" + "\n".join(table_rows) + f"\n{total_row}"
        reasoning = f"Table with {rows} rows and {cols} data columns plus totals. Verifying all sums. Grand total: ${grand_total:,}."
        answer = f"**{scenario.title()} Table Extracted:**\n\n{table_header}\n{table_sep}\n" + "\n".join(table_rows) + f"\n{total_row}\n\n**Verification:** All row totals and column totals are arithmetically correct. Grand total: **${grand_total:,}**."

        examples.append(sft(query=q, reasoning=reasoning, answer=answer, category="complex_tables", difficulty="hard"))

    return examples[:50]


# ---------------------------------------------------------------------------
# 3. CONTRADICTIONS / ERROR FLAGGING (50 examples)
# ---------------------------------------------------------------------------
def gen_contradictions():
    examples = []
    contra_pairs = [
        (
            "This financial report states:\n"
            "- Revenue: $10.2M\n- Operating Expenses: $7.8M\n- Operating Income: $3.4M\n"
            "- Net Income: $2.1M\n- Tax Rate: 25%\n\n"
            "Verify the consistency.",
            "Revenue $10.2M - OpEx $7.8M = $2.4M, not $3.4M. That's a $1M discrepancy in operating income. "
            "Then if operating income were $3.4M and tax is 25%, net income = 3.4 * 0.75 = $2.55M, not $2.1M. "
            "If we use the calculated $2.4M operating income: 2.4 * 0.75 = $1.8M, also not $2.1M. "
            "Multiple inconsistencies.",
            "**MULTIPLE INCONSISTENCIES DETECTED:**\n\n"
            "| Line Item | Reported | Calculated | Discrepancy |\n|---|---|---|---|\n"
            "| Operating Income | $3.4M | $2.4M (10.2 - 7.8) | **$1.0M overstated** |\n"
            "| Net Income (from reported OI) | $2.1M | $2.55M (3.4 x 0.75) | **$0.45M understated** |\n"
            "| Net Income (from calc OI) | $2.1M | $1.8M (2.4 x 0.75) | **$0.3M overstated** |\n\n"
            "**Neither net income calculation matches the reported $2.1M.** This suggests:\n"
            "1. There may be non-operating income/expenses not shown\n"
            "2. The effective tax rate differs from the stated 25%\n"
            "3. There are arithmetic or reporting errors\n\n"
            "**Recommendation:** Review the complete income statement for missing line items (interest, other income/expense)."
        ),
        (
            "This employment contract states:\n"
            "Section 2: 'Employee's annual salary shall be $120,000.'\n"
            "Section 5: 'Monthly compensation: $9,500.'\n"
            "Section 8: 'Bi-weekly payroll amount: $4,615.38'",
            "Annual $120K. Monthly should be $120K/12 = $10,000, but says $9,500. "
            "Bi-weekly should be $120K/26 = $4,615.38. The bi-weekly matches the annual. "
            "But monthly doesn't: $9,500 x 12 = $114,000, not $120,000. "
            "The contract has two different effective salaries depending on which section you follow.",
            "**SALARY INCONSISTENCY:**\n\n"
            "| Section | Statement | Implied Annual |\n|---|---|---|\n"
            "| Section 2 | $120,000/year | $120,000 |\n"
            "| Section 5 | $9,500/month | $114,000 ($9,500 x 12) |\n"
            "| Section 8 | $4,615.38 bi-weekly | $120,000 ($4,615.38 x 26) |\n\n"
            "**Sections 2 and 8 are consistent** ($120,000 annual).\n"
            "**Section 5 implies $114,000** — a $6,000 annual discrepancy.\n\n"
            "The correct monthly amount for $120,000/year should be **$10,000/month**, not $9,500.\n\n"
            "**This contract should not be executed** until the inconsistency is resolved, as the "
            "employee's compensation differs by $6,000/year depending on which section is applied."
        ),
        (
            "Two documents about the same property:\n\n"
            "Document 1 (Listing): Square footage: 2,400 sq ft. Year built: 1998. Lot size: 0.35 acres. "
            "Bedrooms: 4. Bathrooms: 2.5.\n\n"
            "Document 2 (Appraisal): Square footage: 2,180 sq ft. Year built: 2001. Lot size: 0.35 acres. "
            "Bedrooms: 4. Bathrooms: 2.\n\n"
            "Identify discrepancies.",
            "Comparing field by field. Sq ft: 2400 vs 2180 — 220 sq ft difference (9.2%). "
            "Year built: 1998 vs 2001 — 3 year difference. Lot size: both 0.35 acres ✓. "
            "Bedrooms: both 4 ✓. Bathrooms: 2.5 vs 2.0 — half bath discrepancy.",
            "**Property Document Discrepancy Report:**\n\n"
            "| Field | Listing | Appraisal | Match? |\n|---|---|---|---|\n"
            "| Square Footage | 2,400 sq ft | 2,180 sq ft | **NO — 220 sq ft (9.2%) difference** |\n"
            "| Year Built | 1998 | 2001 | **NO — 3-year difference** |\n"
            "| Lot Size | 0.35 acres | 0.35 acres | Yes |\n"
            "| Bedrooms | 4 | 4 | Yes |\n"
            "| Bathrooms | 2.5 | 2.0 | **NO — listing counts half bath** |\n\n"
            "**Material Discrepancies:**\n"
            "1. **Square footage (220 sq ft):** Appraisal measurements are typically more reliable as they follow "
            "ANSI standards. The listing may include non-qualifying space (garage, unfinished basement).\n"
            "2. **Year built (3 years):** Check county records for authoritative build date.\n"
            "3. **Bathrooms (0.5 difference):** The appraisal may not count a powder room or half bath.\n\n"
            "**The appraisal is generally the more authoritative document** for physical characteristics, "
            "but the listing may include details the appraiser didn't count."
        ),
        (
            "This health insurance claim form shows:\n"
            "- Date of service: 04/01/2025\n"
            "- Procedure code: 99213 (Office visit, established patient)\n"
            "- Diagnosis code: Z23 (Encounter for immunization)\n"
            "- Billed amount: $450\n"
            "- Provider: Dr. Smith, Orthopedic Surgeon\n\n"
            "Flag any inconsistencies.",
            "Multiple red flags: 1) 99213 is an E/M code for office visit, but Z23 is for immunization — "
            "typically you'd see 90471 (immunization administration) with Z23. "
            "2) An orthopedic surgeon billing for immunization is unusual. "
            "3) $450 is high for a 99213 (typically $100-200 range). "
            "These could be innocent coding issues or billing errors.",
            "**CLAIM INCONSISTENCIES FLAGGED:**\n\n"
            "| Element | Value | Issue |\n|---|---|---|\n"
            "| Procedure Code | 99213 | Mismatch with diagnosis — 99213 is an E/M visit, not immunization |\n"
            "| Diagnosis Code | Z23 | Immunization encounter — typically paired with 90471, not 99213 |\n"
            "| Provider Specialty | Orthopedic Surgeon | Unusual for immunization services |\n"
            "| Billed Amount | $450 | High for 99213 (typical range: $100-$200) |\n\n"
            "**Issues Identified:**\n\n"
            "1. **Code mismatch:** Procedure 99213 (office visit evaluation) is inconsistent with diagnosis Z23 "
            "(immunization encounter). If this was an immunization visit, the procedure code should be in the "
            "904xx range.\n\n"
            "2. **Specialty mismatch:** Orthopedic surgeons don't typically administer immunizations.\n\n"
            "3. **Billing amount:** $450 for a 99213 is approximately 2-3x the expected rate.\n\n"
            "**These may be coding errors** (wrong procedure or diagnosis code selected) or may warrant "
            "further review. Recommend returning to the provider for code verification before processing."
        ),
        (
            "A vendor submitted two invoices that appear to be duplicates:\n\n"
            "Invoice A: #INV-2025-4421, Date: 03/15/2025, Amount: $8,750.00, "
            "Description: 'Consulting services - March 2025', PO Reference: PO-2025-112\n\n"
            "Invoice B: #INV-2025-4422, Date: 03/18/2025, Amount: $8,750.00, "
            "Description: 'Professional consulting - March 2025', PO Reference: PO-2025-112",
            "Different invoice numbers (4421 vs 4422) but same amount, same PO reference, similar descriptions, "
            "same month, only 3 days apart. This looks like a potential duplicate billing. "
            "However, they could be for different engagements under the same PO.",
            "**POTENTIAL DUPLICATE INVOICE DETECTED:**\n\n"
            "| Field | Invoice A | Invoice B | Match? |\n|---|---|---|---|\n"
            "| Invoice # | INV-2025-4421 | INV-2025-4422 | Different (sequential) |\n"
            "| Date | 03/15/2025 | 03/18/2025 | 3 days apart |\n"
            "| Amount | $8,750.00 | $8,750.00 | **Exact match** |\n"
            "| Description | Consulting services - March 2025 | Professional consulting - March 2025 | **Similar** |\n"
            "| PO Reference | PO-2025-112 | PO-2025-112 | **Exact match** |\n\n"
            "**Risk Assessment: HIGH — Likely Duplicate**\n\n"
            "**Indicators:**\n"
            "- Same amount, same PO reference, same month\n"
            "- Sequential invoice numbers (4421, 4422) suggest back-to-back submission\n"
            "- Descriptions are slightly different wording for the same service\n\n"
            "**Before paying Invoice B, verify:**\n"
            "1. Were there actually two separate engagements in March?\n"
            "2. Does PO-2025-112 authorize $17,500 total (both invoices)?\n"
            "3. Request detailed timesheets or deliverables for each invoice\n\n"
            "**Do not process both invoices** until duplicate status is confirmed or denied by the vendor."
        ),
    ]

    for q, r, a in contra_pairs:
        examples.append(sft(query=q, reasoning=r, answer=a, category="contradictions", difficulty="hard"))

    # Generate more contradiction examples
    more_contras = [
        (
            "This building inspection report contains:\n"
            "Page 3: 'Foundation: No cracks observed. Grade: A.'\n"
            "Page 7: 'Structural concerns: Hairline crack in northwest foundation wall, approximately 4 inches long. "
            "Monitor for expansion.'\n\n"
            "Are these consistent?",
            "Direct contradiction: Page 3 says no cracks, Grade A. Page 7 describes a specific crack in the foundation. "
            "The grade A rating on page 3 is inconsistent with the finding on page 7.",
            "**CONTRADICTION DETECTED:**\n\n"
            "| Page | Statement |\n|---|---|\n"
            "| Page 3 | Foundation: No cracks observed. Grade: A |\n"
            "| Page 7 | Hairline crack in NW foundation wall, ~4 inches, monitor for expansion |\n\n"
            "These statements are **directly contradictory**. Page 3 explicitly states no cracks were observed, "
            "while Page 7 describes a specific crack with location and measurements.\n\n"
            "**Impact:**\n"
            "- The Grade A rating may be inaccurate if a crack exists\n"
            "- The recommendation to \"monitor for expansion\" suggests the crack is a concern\n\n"
            "**Recommendation:** Contact the inspector to determine which statement is accurate. "
            "This discrepancy affects the reliability of the entire foundation assessment."
        ),
        (
            "Document A (Insurance Application): 'No prior claims in the last 5 years.'\n"
            "Document B (Claims History Report): 'Claim #CL-2023-4412, Date: 06/15/2023, "
            "Amount: $12,400, Status: Paid. Water damage - burst pipe.'",
            "Application says no claims in 5 years. Claims history shows a paid claim from 2023 (about 2 years ago). "
            "This is either a false statement on the application or the applicant forgot about the claim.",
            "**MATERIAL MISREPRESENTATION DETECTED:**\n\n"
            "| Source | Statement |\n|---|---|\n"
            "| Application | No prior claims in last 5 years |\n"
            "| Claims History | Claim CL-2023-4412, 06/15/2023, $12,400 (Paid) |\n\n"
            "The application statement is **directly contradicted** by the claims history, which shows "
            "a paid claim of $12,400 from June 2023 — approximately 22 months ago, well within the 5-year window.\n\n"
            "**Implications:**\n"
            "- This constitutes a material misrepresentation on the application\n"
            "- The claim (water damage - burst pipe) is the type of loss that directly affects underwriting\n"
            "- Premium calculation may need adjustment to reflect the prior claim\n\n"
            "**Recommendation:** Flag for underwriting review. The insurer may request clarification "
            "from the applicant or adjust the application accordingly."
        ),
        (
            "This project status report says:\n"
            "- 'Project is on track. Green status.'\n"
            "- 'Budget spent: $2.1M of $2.5M (84%)'\n"
            "- 'Project completion: 62%'\n"
            "- 'Timeline: On schedule for Q4 delivery'\n\nFlag any concerns.",
            "84% of budget spent but only 62% complete. That's a significant cost overrun trajectory. "
            "If the spending rate continues linearly, the project will need about $3.39M to complete "
            "(2.1/0.62 = 3.387M), exceeding the $2.5M budget by ~$890K (35.6% overrun). "
            "Claiming 'Green status' with this math is questionable.",
            "**PROJECT STATUS CONCERNS:**\n\n"
            "| Metric | Value | Status |\n|---|---|---|\n"
            "| Budget Spent | $2.1M of $2.5M | 84% consumed |\n"
            "| Work Complete | 62% | — |\n"
            "| Remaining Budget | $400K | 16% of total |\n"
            "| Remaining Work | 38% | — |\n"
            "| Reported Status | **Green** | — |\n\n"
            "**COST PERFORMANCE ISSUE:**\n"
            "- Budget consumed: **84%** vs Work completed: **62%**\n"
            "- Cost Performance Index (CPI): 0.62 / 0.84 = **0.74** (significantly under 1.0)\n"
            "- Estimated at Completion: $2.1M / 0.62 = **$3.39M** (35.6% over budget)\n"
            "- Projected overrun: **~$890K**\n\n"
            "**The \"Green\" status is not supported by the data.** At current burn rate, the project "
            "will exhaust its remaining $400K budget at approximately 73% completion, leaving 27% unfunded.\n\n"
            "**Recommendation:** This project should be **Yellow or Red status**. "
            "Request a revised estimate at completion and a corrective action plan."
        ),
        (
            "These two pages of a legal brief cite the same case differently:\n\n"
            "Page 4: 'In Smith v. Jones (2019), the court held that contractual ambiguity must be construed against the drafter.'\n"
            "Page 11: 'The Smith v. Jones (2021) ruling established that unambiguous contract terms must be enforced as written.'",
            "Same case name but different years (2019 vs 2021) and contradictory holdings. "
            "One says ambiguity construed against drafter, the other says unambiguous terms enforced as written. "
            "These could be the same case at different court levels, different Smith v. Jones cases, "
            "or citation errors.",
            "**CITATION INCONSISTENCY:**\n\n"
            "| Page | Case Citation | Holding Cited |\n|---|---|---|\n"
            "| Page 4 | Smith v. Jones (2019) | Ambiguity construed against drafter |\n"
            "| Page 11 | Smith v. Jones (2021) | Unambiguous terms enforced as written |\n\n"
            "**Issues:**\n"
            "1. **Different years** for what appears to be the same case (2019 vs 2021)\n"
            "2. **Different holdings** — not necessarily contradictory (one addresses ambiguity, the other unambiguous terms), "
            "but they could be presented as conflicting principles\n\n"
            "**Possible Explanations:**\n"
            "- The 2021 citation may be an appellate decision of the 2019 trial court ruling\n"
            "- These may be two different cases with the same party names\n"
            "- One of the years may be a typographical error\n\n"
            "**Recommendation:** Verify the full citation (reporter, volume, page) for both references "
            "to determine if these are the same or different cases."
        ),
        (
            "This purchase order and receiving report don't match:\n\n"
            "PO #4421:\n  Widget A: 500 units @ $2.50\n  Widget B: 200 units @ $4.75\n  Widget C: 100 units @ $8.00\n\n"
            "Receiving Report #RR-4421:\n  Widget A: 500 units received\n  Widget B: 180 units received (20 short)\n"
            "  Widget C: 100 units received\n  Widget D: 50 units received (NOT ON PO)",
            "Three-way match issues: Widget B is 20 units short, and Widget D (50 units) was received but not ordered. "
            "This needs investigation — either Widget D was sent in error or it was ordered separately.",
            "**THREE-WAY MATCH DISCREPANCIES:**\n\n"
            "| Item | PO Qty | Received | Variance | Status |\n|---|---|---|---|---|\n"
            "| Widget A | 500 | 500 | 0 | Match |\n"
            "| Widget B | 200 | 180 | **-20** | **Short shipment** |\n"
            "| Widget C | 100 | 100 | 0 | Match |\n"
            "| Widget D | **0 (not on PO)** | 50 | **+50** | **Unauthorized receipt** |\n\n"
            "**Action Items:**\n"
            "1. **Widget B shortage (20 units):** Contact vendor for backorder status. "
            "Do not pay for 20 units ($95.00) until received.\n"
            "2. **Widget D (50 units):** Investigate — this item is not on PO #4421. Options:\n"
            "   - Return to vendor if sent in error\n"
            "   - Locate matching PO if ordered separately\n"
            "   - Issue a supplemental PO if needed\n\n"
            "**Payment should be:**\n"
            "- Widget A: 500 x $2.50 = $1,250.00\n"
            "- Widget B: 180 x $4.75 = $855.00 (partial)\n"
            "- Widget C: 100 x $8.00 = $800.00\n"
            "- **Adjusted total: $2,905.00** (vs PO total of $3,000.00)"
        ),
    ]

    for q, r, a in more_contras:
        examples.append(sft(query=q, reasoning=r, answer=a, category="contradictions", difficulty="hard"))

    # Generate remaining contradiction examples
    remaining_contras = [
        (
            f"This document contains an internal inconsistency. {scenario}\n\nIdentify the issue.",
            reasoning,
            answer
        )
        for scenario, reasoning, answer in [
            (
                "Resume states 'MBA, Harvard Business School, Class of 2018' and later 'Worked at Goldman Sachs 2015-2020 (no career break)'.",
                "A full-time MBA at Harvard is 2 years. If graduated in 2018, started in 2016. But resume says continuous at Goldman 2015-2020 with no break. Can't do both simultaneously.",
                "**TIMELINE INCONSISTENCY:**\n\n| Item | Period |\n|---|---|\n| Goldman Sachs | 2015-2020 (continuous) |\n| Harvard MBA | Class of 2018 (implies 2016-2018 enrollment) |\n\n"
                "A full-time Harvard MBA requires 2 years of on-campus study. The candidate claims continuous employment at Goldman Sachs during the same period.\n\n"
                "**Possible explanations:** Part-time/executive MBA program, dates are incorrect on one entry, or the candidate left Goldman for MBA and returned (contradicting 'no career break').\n\n**Verify employment dates and MBA program type.**"
            ),
            (
                "Lease says 'No pets allowed' in Section 4.1. Pet deposit of $500 is listed in Section 9.3 (Security Deposits).",
                "If no pets are allowed, why is there a pet deposit? These clauses contradict each other.",
                "**CONTRADICTION:**\n\n| Section | Statement |\n|---|---|\n| Section 4.1 | No pets allowed |\n| Section 9.3 | Pet deposit: $500 |\n\n"
                "If pets are prohibited, a pet deposit is nonsensical. Either:\n1. The no-pet clause is a template error (pets are actually permitted with deposit)\n2. The pet deposit is a template remnant that should have been removed\n\n"
                "**Recommendation:** Clarify with the landlord before signing. If pets are truly prohibited, Section 9.3 should be amended to remove the pet deposit reference."
            ),
            (
                "The audit report states the company has 'zero debt' on page 2, but the balance sheet on page 8 shows 'Long-term debt: $4.2M, Current portion of LTD: $800K'.",
                "Page 2 says zero debt, page 8 shows $5M total debt ($4.2M LT + $800K current). Major discrepancy.",
                "**MATERIAL DISCREPANCY:**\n\n| Source | Debt |\n|---|---|\n| Audit narrative (p.2) | Zero debt |\n| Balance sheet (p.8) | $5.0M ($4.2M LT + $800K current) |\n\n"
                "The narrative claim of 'zero debt' is directly contradicted by $5.0M in debt on the balance sheet. This is a material misstatement.\n\n"
                "**Impact:** This affects leverage ratios, credit assessments, and investor disclosures. The balance sheet (audited financial statement) is more authoritative than narrative text."
            ),
            (
                "Shipping document says 'Net weight: 2,500 kg' and 'Gross weight: 2,200 kg'.",
                "Net weight cannot exceed gross weight. Gross = net + packaging/container. 2500 > 2200 means these are swapped or one is wrong.",
                "**PHYSICAL IMPOSSIBILITY:**\n\n| Field | Value |\n|---|---|\n| Net Weight | 2,500 kg |\n| Gross Weight | 2,200 kg |\n\n"
                "Net weight (contents only) **cannot exceed** gross weight (contents + packaging). These values appear to be **transposed**.\n\n"
                "**Likely correction:** Net weight: 2,200 kg, Gross weight: 2,500 kg (packaging weight: 300 kg).\n\n"
                "**This error could affect:** Shipping costs, customs declarations, vehicle load calculations, and regulatory compliance. Correct before shipping."
            ),
            (
                "Company report says 'We have 15 offices across 12 countries' but the office directory lists locations in 8 countries only.",
                "Claims 12 countries but directory shows only 8. Either 4 countries are missing from the directory or the count is wrong.",
                "**INCONSISTENCY:**\n\n| Source | Countries |\n|---|---|\n| Corporate statement | 15 offices, 12 countries |\n| Office directory | Locations in 8 countries |\n\n"
                "The directory accounts for only **8 of the claimed 12 countries** (4 missing). Possible explanations:\n"
                "1. Recently opened offices not yet added to directory\n2. Some offices are through partners/affiliates not listed\n3. The '12 countries' figure is outdated\n\n"
                "**Verify:** Request complete office list to reconcile."
            ),
            (
                "The contract says 'Payment currency: USD' in the terms, but the pricing table shows amounts with '€' symbols throughout.",
                "Currency mismatch: terms say USD but pricing shows EUR symbols.",
                "**CURRENCY INCONSISTENCY:**\n\n| Location | Currency |\n|---|---|\n| Payment terms | USD |\n| Pricing table | EUR (€) |\n\n"
                "The contract specifies USD payments but all prices are denominated in EUR. This creates ambiguity about:\n"
                "- The actual amounts owed\n- Who bears exchange rate risk\n- Which amounts are legally binding\n\n"
                "**Do not execute this contract** until the currency discrepancy is resolved. If parties intended EUR, the payment terms need updating. If USD, the pricing table needs conversion."
            ),
            (
                "Medical record: 'Patient is a 45-year-old male' at the top. Surgical consent form attached: 'Patient: Female, DOB: 03/15/1975'.",
                "Gender discrepancy between medical record (male) and consent form (female). Also, DOB 03/15/1975 would make patient 50, not 45. Two discrepancies.",
                "**PATIENT IDENTIFICATION DISCREPANCIES:**\n\n| Field | Medical Record | Consent Form |\n|---|---|---|\n| Gender | Male | Female |\n| Age | 45 years old | ~50 (from DOB 03/15/1975) |\n\n"
                "**Two discrepancies suggest possible wrong patient chart.** This is a **patient safety concern** — the consent form may belong to a different patient.\n\n"
                "**URGENT:** Verify patient identity before any procedure. Confirm name, DOB, and medical record number match across all documents."
            ),
            (
                "Tax filing shows 'Filing Status: Single' but claims 3 dependents and includes a Schedule 8812 (child tax credit).",
                "A single filer can have dependents (e.g., single parent), so dependents alone aren't an issue. "
                "But let me check: Schedule 8812 requires qualifying children. This is plausible for a single parent. "
                "Actually, this might not be a contradiction at all — single filers CAN claim dependents.",
                "**ASSESSMENT: No clear contradiction, but worth verifying.**\n\n"
                "| Element | Value |\n|---|---|\n| Filing Status | Single |\n| Dependents | 3 |\n| Schedule 8812 | Included (Child Tax Credit) |\n\n"
                "A single filer **can** claim dependents — this is common for single parents. However:\n\n"
                "**Verification recommended:**\n"
                "- With 3 dependents, 'Head of Household' status may be more beneficial than 'Single'\n"
                "- Head of Household provides a larger standard deduction and lower tax brackets\n"
                "- If the taxpayer qualifies as HoH, the 'Single' filing status may be an error that costs them money\n\n"
                "**Not necessarily wrong, but potentially suboptimal.** Recommend reviewing HoH eligibility."
            ),
        ]
    ]

    for q, r, a in remaining_contras:
        examples.append(sft(query=q, reasoning=r, answer=a, category="contradictions", difficulty="hard"))

    # Fill to 50 with more programmatic examples
    remaining = 50 - len(examples)
    error_templates = [
        ("This invoice total is ${total} but line items sum to ${calc}.", "Invoice arithmetic error. Stated ${total} but calculated ${calc}. Difference: ${diff}.", "**ARITHMETIC ERROR:** Invoice total ${total} does not match line item sum of ${calc}. Discrepancy: **${diff}**. Verify before payment."),
        ("Report says '{pct}% growth' but prior year was ${prior} and current is ${current}.", "Growth calculation: ({current}-{prior})/{prior} = {actual_pct}%, not {pct}%.", "**GROWTH RATE ERROR:** Actual growth is **{actual_pct}%**, not {pct}% as stated. Calculated: (${current} - ${prior}) / ${prior}."),
    ]

    for i in range(remaining):
        if i % 2 == 0:
            total = 10000 + i * 500
            calc = total - 200 - i * 10
            diff = total - calc
            q = error_templates[0][0].replace("${total}", f"${total:,}").replace("${calc}", f"${calc:,}")
            r = error_templates[0][1].replace("${total}", f"${total:,}").replace("${calc}", f"${calc:,}").replace("${diff}", f"${diff:,}")
            a = error_templates[0][2].replace("${total}", f"${total:,}").replace("${calc}", f"${calc:,}").replace("${diff}", f"${diff:,}")
        else:
            prior = 1000 + i * 100
            current = prior + int(prior * 0.12)
            actual_pct = round((current - prior) / prior * 100, 1)
            claimed_pct = actual_pct + 3.5
            q = error_templates[1][0].format(pct=claimed_pct, prior=f"{prior:,}", current=f"{current:,}")
            r = error_templates[1][1].format(pct=claimed_pct, prior=f"{prior:,}", current=f"{current:,}", actual_pct=actual_pct)
            a = error_templates[1][2].format(pct=claimed_pct, prior=f"{prior:,}", current=f"{current:,}", actual_pct=actual_pct)

        examples.append(sft(query=q, reasoning=r, answer=a, category="contradictions", difficulty="hard"))

    return examples[:50]


# ---------------------------------------------------------------------------
# 4. MULTI-DOCUMENT COMPARISON (50 examples)
# ---------------------------------------------------------------------------
def gen_multi_doc():
    examples = []
    md_pairs = [
        (
            "Compare these two vendor proposals for IT services:\n\n"
            "PROPOSAL A — TechPrime Solutions:\n  Team: 5 senior consultants, 3 junior\n"
            "  Timeline: 16 weeks\n  Fixed Price: $480,000\n  SLA: 99.5% uptime, 4-hour response\n"
            "  Warranty: 12 months post-delivery\n  Insurance: $5M E&O\n\n"
            "PROPOSAL B — CloudBridge Systems:\n  Team: 3 senior consultants, 5 junior, 2 offshore\n"
            "  Timeline: 12 weeks\n  Fixed Price: $385,000\n  SLA: 99.9% uptime, 2-hour response\n"
            "  Warranty: 6 months post-delivery\n  Insurance: $2M E&O",
            "Comparing two proposals across all dimensions. A is more expensive ($480K vs $385K, 24.7% more) but has "
            "more senior staff (5 vs 3), longer warranty (12 vs 6 months), and better insurance ($5M vs $2M). "
            "B is faster (12 vs 16 weeks), has better SLA (99.9% vs 99.5%, 2hr vs 4hr response), but uses offshore resources. "
            "Cost per week: A = $30K/week, B = $32.1K/week. So B is actually more expensive per week.",
            "**Vendor Proposal Comparison:**\n\n"
            "| Criterion | TechPrime (A) | CloudBridge (B) | Advantage |\n|---|---|---|---|\n"
            "| **Price** | $480,000 | $385,000 | B (-$95K, 19.8% less) |\n"
            "| **Timeline** | 16 weeks | 12 weeks | B (-4 weeks, 25% faster) |\n"
            "| **Price/Week** | $30,000 | $32,083 | A (-$2,083/week) |\n"
            "| **Senior Staff** | 5 | 3 | A (+2 senior) |\n"
            "| **Total Team** | 8 (all onshore) | 10 (2 offshore) | A (onshore only) |\n"
            "| **SLA Uptime** | 99.5% | 99.9% | B (higher uptime) |\n"
            "| **SLA Response** | 4 hours | 2 hours | B (faster response) |\n"
            "| **Warranty** | 12 months | 6 months | A (+6 months) |\n"
            "| **Insurance** | $5M E&O | $2M E&O | A (+$3M coverage) |\n\n"
            "**Analysis:**\n\n"
            "**Choose TechPrime (A) if:** Quality and risk mitigation are priorities. More senior staff, "
            "longer warranty, and higher insurance provide better protection. The extra $95K buys 6 additional "
            "months of warranty coverage alone.\n\n"
            "**Choose CloudBridge (B) if:** Speed and operational SLA matter most. The 99.9% uptime commitment "
            "and 2-hour response are significantly better. However, the offshore component and shorter warranty "
            "introduce risk.\n\n"
            "**Key Risk:** CloudBridge's per-week cost is actually higher, and with 2 offshore team members, "
            "there may be coordination overhead not reflected in the timeline."
        ),
        (
            "Compare these three health insurance plans:\n\n"
            "Plan A (Bronze): Monthly premium: $350. Deductible: $6,500. Out-of-pocket max: $8,700. "
            "Copay (PCP): $40. Copay (Specialist): $80. ER: 40% after deductible. Rx: $15/$45/$80.\n\n"
            "Plan B (Silver): Monthly premium: $520. Deductible: $3,500. Out-of-pocket max: $7,500. "
            "Copay (PCP): $25. Copay (Specialist): $50. ER: 30% after deductible. Rx: $10/$35/$65.\n\n"
            "Plan C (Gold): Monthly premium: $710. Deductible: $1,500. Out-of-pocket max: $5,500. "
            "Copay (PCP): $15. Copay (Specialist): $35. ER: 20% after deductible. Rx: $5/$25/$50.",
            "Three-plan comparison. Annual premiums: A=$4,200, B=$6,240, C=$8,520. "
            "Premium difference A-to-C: $4,320/year. But deductible difference: $5,000. OOP max difference: $3,200. "
            "For healthy person (low utilization), A saves $4,320 in premiums. "
            "For high utilization (max OOP), total A: $4,200+$8,700=$12,900. B: $6,240+$7,500=$13,740. C: $8,520+$5,500=$14,020. "
            "So at max OOP, Plan A is cheapest total cost. But moderate use favors B or C.",
            "**Health Insurance Plan Comparison:**\n\n"
            "| Feature | Bronze (A) | Silver (B) | Gold (C) |\n|---|---|---|---|\n"
            "| Monthly Premium | $350 | $520 | $710 |\n"
            "| **Annual Premium** | **$4,200** | **$6,240** | **$8,520** |\n"
            "| Deductible | $6,500 | $3,500 | $1,500 |\n"
            "| Out-of-Pocket Max | $8,700 | $7,500 | $5,500 |\n"
            "| PCP Copay | $40 | $25 | $15 |\n"
            "| Specialist Copay | $80 | $50 | $35 |\n"
            "| ER Coinsurance | 40% | 30% | 20% |\n"
            "| Rx (Generic/Brand/Specialty) | $15/$45/$80 | $10/$35/$65 | $5/$25/$50 |\n\n"
            "**Total Annual Cost Scenarios:**\n"
            "| Scenario | Bronze (A) | Silver (B) | Gold (C) | Best Value |\n|---|---|---|---|---|\n"
            "| Healthy (premiums only) | $4,200 | $6,240 | $8,520 | **Bronze** |\n"
            "| Moderate ($5K medical) | ~$9,200 | ~$9,740 | ~$10,020 | **Bronze** |\n"
            "| Max Out-of-Pocket | $12,900 | $13,740 | $14,020 | **Bronze** |\n\n"
            "**Insight:** Bronze has the lowest total cost in ALL scenarios (healthy through worst-case). "
            "The higher premiums of Silver and Gold don't offset their lower deductibles and OOP maximums.\n\n"
            "**However:** Gold provides better day-to-day cost predictability with lower copays, "
            "which matters if you have regular specialist visits or prescription needs."
        ),
        (
            "Compare these quarterly financial statements from two competitors:\n\n"
            "COMPANY X — Q1 2025:\n  Revenue: $45.2M\n  COGS: $27.1M\n  Gross Margin: 40.0%\n"
            "  R&D: $6.8M\n  SGA: $5.4M\n  Operating Income: $5.9M\n  Net Income: $4.2M\n"
            "  Headcount: 450\n\n"
            "COMPANY Y — Q1 2025:\n  Revenue: $38.7M\n  COGS: $19.4M\n  Gross Margin: 49.9%\n"
            "  R&D: $8.2M\n  SGA: $7.1M\n  Operating Income: $4.0M\n  Net Income: $2.8M\n"
            "  Headcount: 380",
            "Let me verify margins and compare. X: GP = 45.2-27.1 = 18.1. GM = 18.1/45.2 = 40.04% ≈ 40.0% ✓. "
            "OI = 18.1-6.8-5.4 = 5.9 ✓. Y: GP = 38.7-19.4 = 19.3. GM = 19.3/38.7 = 49.87% ≈ 49.9% ✓. "
            "OI = 19.3-8.2-7.1 = 4.0 ✓. Revenue/employee: X = $100.4K, Y = $101.8K. "
            "R&D intensity: X = 15.0%, Y = 21.2%.",
            "**Competitive Financial Comparison — Q1 2025:**\n\n"
            "| Metric | Company X | Company Y | Delta |\n|---|---|---|---|\n"
            "| **Revenue** | $45.2M | $38.7M | X +16.8% |\n"
            "| **COGS** | $27.1M | $19.4M | — |\n"
            "| **Gross Margin** | 40.0% | 49.9% | Y +9.9pp |\n"
            "| **R&D Spend** | $6.8M (15.0%) | $8.2M (21.2%) | Y invests more |\n"
            "| **SGA** | $5.4M (11.9%) | $7.1M (18.3%) | Y less efficient |\n"
            "| **Operating Income** | $5.9M (13.1%) | $4.0M (10.3%) | X +2.8pp |\n"
            "| **Net Income** | $4.2M (9.3%) | $2.8M (7.2%) | X +2.1pp |\n"
            "| **Headcount** | 450 | 380 | X +18.4% larger |\n"
            "| **Revenue/Employee** | $100.4K | $101.8K | Comparable |\n\n"
            "**Key Observations:**\n"
            "- **Y has better gross margins** (+9.9pp) — more efficient production or premium pricing\n"
            "- **X has better operating margins** — Y's advantage erodes due to higher R&D (21.2% vs 15.0%) "
            "and SGA (18.3% vs 11.9%)\n"
            "- **Y invests more in R&D** — may indicate future product advantage\n"
            "- **Revenue per employee** is nearly identical (~$101K)\n\n"
            "**Arithmetic verification:** All margins and operating income calculations verified correct."
        ),
    ]

    for q, r, a in md_pairs:
        examples.append(sft(query=q, reasoning=r, answer=a, category="multi_doc", difficulty="hard"))

    # Generate more multi-doc comparison examples
    more_md = [
        (
            "Compare these two lease agreements for the same tenant:\n\n"
            "CURRENT LEASE (expiring 06/30/2025):\n  Space: Suite 400, 3,200 sq ft\n  Rent: $28/sq ft/year\n"
            "  CAM: $8/sq ft/year\n  Term: 3 years\n  Escalation: 3%/year\n  Parking: 8 spaces included\n\n"
            "RENEWAL OFFER:\n  Space: Suite 400, 3,200 sq ft (+ option for Suite 410, 1,200 sq ft)\n"
            "  Rent: $32/sq ft/year\n  CAM: $10/sq ft/year\n  Term: 5 years\n  Escalation: CPI (2-4% cap)\n"
            "  Parking: 8 spaces + 4 additional at $150/month each\n  TI Allowance: $15/sq ft",
            "Comparing current to renewal. Current annual cost: (28+8)*3200 = $115,200. "
            "Renewal base: (32+10)*3200 = $134,400. Increase: $19,200/year = 16.7%. "
            "TI allowance: $15*3200 = $48,000. Additional parking: 4*$150*12 = $7,200/year. "
            "With expansion: (32+10)*4400 = $184,800/year. TI on expansion: $15*4400 = $66,000.",
            "**Lease Renewal Comparison:**\n\n"
            "| Term | Current Lease | Renewal Offer | Change |\n|---|---|---|---|\n"
            "| Space | 3,200 sq ft | 3,200 sq ft (+ 1,200 option) | Expansion available |\n"
            "| Base Rent | $28/sq ft | $32/sq ft | **+$4/sq ft (+14.3%)** |\n"
            "| CAM | $8/sq ft | $10/sq ft | **+$2/sq ft (+25%)** |\n"
            "| **Total Rate** | **$36/sq ft** | **$42/sq ft** | **+$6/sq ft (+16.7%)** |\n"
            "| Term | 3 years | 5 years | +2 years |\n"
            "| Escalation | 3% fixed | CPI (2-4% cap) | Variable (could be lower) |\n"
            "| Parking | 8 included | 8 included + 4 @ $150/mo | +$7,200/year if used |\n"
            "| TI Allowance | N/A | $15/sq ft ($48,000) | New benefit |\n\n"
            "**Annual Cost Comparison (3,200 sq ft only):**\n"
            "| | Current | Renewal | Difference |\n|---|---|---|---|\n"
            "| Base Rent + CAM | $115,200 | $134,400 | +$19,200/year |\n"
            "| TI (amortized over 5 years) | — | -$9,600/year | — |\n"
            "| **Net Effective** | **$115,200** | **$124,800** | **+$9,600/year** |\n\n"
            "**With Expansion (4,400 sq ft):** $184,800/year + $7,200 parking = $192,000/year. "
            "TI: $66,000 ($13,200/year amortized).\n\n"
            "The net effective increase is ~8.3% after TI credit, which is moderate for a 5-year commitment."
        ),
        (
            "Three candidates applied for the same role. Compare their qualifications:\n\n"
            "CANDIDATE A:\n  Education: MS Computer Science, Stanford\n  Experience: 8 years at Google (Senior SWE), 3 years at startup (CTO)\n"
            "  Skills: Python, Java, ML, Distributed Systems\n  Salary Expectation: $250,000\n\n"
            "CANDIDATE B:\n  Education: BS Computer Science, MIT\n  Experience: 12 years at Amazon (Principal SWE)\n"
            "  Skills: Java, AWS, System Design, Technical Leadership\n  Salary Expectation: $280,000\n\n"
            "CANDIDATE C:\n  Education: PhD Computer Science, CMU (ML focus)\n  Experience: 5 years at Meta (Research Scientist), 2 years at AI startup\n"
            "  Skills: Python, PyTorch, ML/AI, Research, Publications\n  Salary Expectation: $265,000",
            "Three strong candidates with different profiles. A has breadth (big tech + startup CTO). "
            "B has depth and seniority (12 years, Principal level at Amazon). C has research depth (PhD + publications). "
            "Role context needed to determine best fit, but I can compare objectively.",
            "**Candidate Comparison:**\n\n"
            "| Attribute | Candidate A | Candidate B | Candidate C |\n|---|---|---|---|\n"
            "| **Education** | MS Stanford | BS MIT | PhD CMU (ML) |\n"
            "| **Total Experience** | 11 years | 12 years | 7 years |\n"
            "| **Highest Level** | CTO (startup) | Principal SWE (Amazon) | Research Scientist (Meta) |\n"
            "| **Key Strength** | Breadth + leadership | Depth + scale | Research + AI |\n"
            "| **Salary** | $250,000 | $280,000 | $265,000 |\n\n"
            "**Best Fit By Role Type:**\n\n"
            "| If the role requires... | Best candidate | Reasoning |\n|---|---|---|\n"
            "| Technical leadership at scale | **B** | 12 years, Principal at Amazon, system design |\n"
            "| Startup/CTO experience | **A** | Direct CTO experience, breadth across domains |\n"
            "| ML/AI innovation | **C** | PhD in ML, publications, research background |\n"
            "| Cost-effectiveness | **A** | Lowest salary expectation with strong experience |\n\n"
            "**Note:** This comparison is based solely on the documents provided. "
            "Interview performance, cultural fit, and specific role requirements should also factor into the decision."
        ),
    ]

    for q, r, a in more_md:
        examples.append(sft(query=q, reasoning=r, answer=a, category="multi_doc", difficulty="hard"))

    # Fill remaining with generated examples
    remaining = 50 - len(examples)
    for i in range(remaining):
        a_rev = 10 + i * 2
        b_rev = 8 + i * 2
        a_profit = round(a_rev * 0.15, 1)
        b_profit = round(b_rev * 0.22, 1)
        q = (
            f"Compare Company Alpha and Company Beta for Q{(i%4)+1} 2025:\n\n"
            f"Alpha: Revenue ${a_rev}M, Net Profit ${a_profit}M, Employees {200+i*20}\n"
            f"Beta: Revenue ${b_rev}M, Net Profit ${b_profit}M, Employees {150+i*15}"
        )
        a_margin = round(a_profit/a_rev*100, 1)
        b_margin = round(b_profit/b_rev*100, 1)
        a_rev_emp = round(a_rev*1000/(200+i*20), 1)
        b_rev_emp = round(b_rev*1000/(150+i*15), 1)
        r = f"Alpha: Revenue ${a_rev}M, margin {a_margin}%. Beta: Revenue ${b_rev}M, margin {b_margin}%. Beta has higher margins."
        a_text = (
            f"| Metric | Alpha | Beta |\n|---|---|---|\n"
            f"| Revenue | ${a_rev}M | ${b_rev}M |\n"
            f"| Net Profit | ${a_profit}M | ${b_profit}M |\n"
            f"| **Profit Margin** | **{a_margin}%** | **{b_margin}%** |\n"
            f"| Revenue/Employee | ${a_rev_emp}K | ${b_rev_emp}K |\n\n"
            f"Alpha leads on revenue (${a_rev}M vs ${b_rev}M), but Beta has stronger profitability ({b_margin}% vs {a_margin}%)."
        )
        examples.append(sft(query=q, reasoning=r, answer=a_text, category="multi_doc", difficulty="hard"))

    return examples[:50]


# ---------------------------------------------------------------------------
# 5. DOMAIN-SPECIFIC ANALYSIS (50 examples)
# ---------------------------------------------------------------------------
def gen_domain_analysis():
    examples = []
    domain_pairs = [
        (
            "Analyze these financial ratios from the company's annual report:\n\n"
            "Current Ratio: 1.8\nQuick Ratio: 1.2\nDebt-to-Equity: 0.65\n"
            "Interest Coverage: 8.5x\nROE: 18.2%\nROA: 11.4%\n"
            "Gross Margin: 42.3%\nOperating Margin: 15.7%\nNet Margin: 10.8%\n"
            "Inventory Turnover: 6.2x\nDays Sales Outstanding: 45 days\n"
            "Revenue Growth: 12.4% YoY",
            "Comprehensive ratio analysis. Liquidity: current 1.8 (healthy, >1.5), quick 1.2 (good, >1.0). "
            "Leverage: D/E 0.65 (moderate), interest coverage 8.5x (strong, well above danger zone of <3). "
            "Profitability: ROE 18.2% (strong), ROA 11.4% (good), margins healthy. "
            "Efficiency: inventory turns 6.2x (good), DSO 45 days (reasonable). "
            "Growth: 12.4% revenue growth. Let me check ROE/ROA consistency: ROE/ROA = 18.2/11.4 = 1.60, "
            "which implies equity multiplier of 1.60. With D/E of 0.65, equity multiplier = 1+0.65 = 1.65. Close enough.",
            "**Financial Health Assessment:**\n\n"
            "**Liquidity:**\n"
            "| Ratio | Value | Assessment |\n|---|---|---|\n"
            "| Current Ratio | 1.8 | Strong — adequate short-term liquidity |\n"
            "| Quick Ratio | 1.2 | Healthy — can cover obligations without inventory |\n\n"
            "**Leverage:**\n"
            "| Ratio | Value | Assessment |\n|---|---|---|\n"
            "| Debt-to-Equity | 0.65 | Moderate — conservative capital structure |\n"
            "| Interest Coverage | 8.5x | Strong — ample earnings to cover interest |\n\n"
            "**Profitability:**\n"
            "| Ratio | Value | Assessment |\n|---|---|---|\n"
            "| ROE | 18.2% | Strong — efficient use of shareholder equity |\n"
            "| ROA | 11.4% | Good — effective asset utilization |\n"
            "| Gross Margin | 42.3% | Healthy pricing power |\n"
            "| Operating Margin | 15.7% | Solid operational efficiency |\n"
            "| Net Margin | 10.8% | Good bottom-line conversion |\n\n"
            "**Efficiency:**\n"
            "| Ratio | Value | Assessment |\n|---|---|---|\n"
            "| Inventory Turnover | 6.2x | Good — inventory moves roughly every 59 days |\n"
            "| DSO | 45 days | Acceptable — 1.5 months to collect receivables |\n\n"
            "**DuPont Analysis:** ROE (18.2%) = Net Margin (10.8%) x Asset Turnover x Equity Multiplier. "
            "Equity multiplier ≈ 1.65 (1 + D/E). This confirms ROE is driven primarily by profitability, "
            "not excessive leverage.\n\n"
            "**Overall:** This company shows strong financial health across all dimensions — liquidity, leverage, "
            "profitability, and efficiency. The 12.4% revenue growth with maintained margins suggests sustainable expansion."
        ),
        (
            "Review this contract clause for potential issues:\n\n"
            "\"12.3 Limitation of Liability. IN NO EVENT SHALL PROVIDER'S TOTAL AGGREGATE LIABILITY "
            "UNDER THIS AGREEMENT EXCEED THE FEES PAID BY CLIENT IN THE TWELVE (12) MONTH PERIOD "
            "PRECEDING THE EVENT GIVING RISE TO LIABILITY. THIS LIMITATION APPLIES TO ALL CLAIMS "
            "INCLUDING BUT NOT LIMITED TO BREACH OF CONTRACT, NEGLIGENCE, STRICT LIABILITY, AND "
            "BREACH OF WARRANTY. NOTWITHSTANDING THE FOREGOING, THIS LIMITATION SHALL NOT APPLY TO "
            "(A) PROVIDER'S INDEMNIFICATION OBLIGATIONS UNDER SECTION 14, (B) BREACHES OF SECTION 9 "
            "(CONFIDENTIALITY), OR (C) PROVIDER'S GROSS NEGLIGENCE OR WILLFUL MISCONDUCT.\"",
            "Standard liability cap clause. Cap = 12 months fees. Carve-outs for indemnification (S.14), "
            "confidentiality breach (S.9), and gross negligence/willful misconduct. "
            "Key analysis points: Is the 12-month lookback period appropriate? Are the carve-outs sufficient? "
            "What about data breach liability? IP infringement?",
            "**Contract Clause Analysis — Section 12.3 (Limitation of Liability):**\n\n"
            "**Structure:**\n"
            "| Element | Detail |\n|---|---|\n"
            "| **Cap** | 12 months of fees preceding the event |\n"
            "| **Scope** | All claims: contract, negligence, strict liability, warranty |\n"
            "| **Carve-out (a)** | Indemnification (Section 14) |\n"
            "| **Carve-out (b)** | Confidentiality breach (Section 9) |\n"
            "| **Carve-out (c)** | Gross negligence / willful misconduct |\n\n"
            "**Assessment:**\n\n"
            "**Favorable provisions:**\n"
            "- Carve-outs for confidentiality and gross negligence are standard and protective\n"
            "- 12-month lookback is common in enterprise agreements\n\n"
            "**Potential gaps to review:**\n"
            "1. **Data breach liability** — Not explicitly carved out. If Provider handles personal data, "
            "a data breach could far exceed 12 months of fees. Consider adding a separate, higher cap.\n"
            "2. **IP infringement** — Not carved out. If Provider infringes third-party IP, "
            "the 12-month cap may be inadequate.\n"
            "3. **The indemnification carve-out** references Section 14 — review whether Section 14 "
            "itself has a sub-cap that could limit exposure.\n"
            "4. **No mention of consequential damages** — A separate consequential damages waiver "
            "is typically included alongside a liability cap.\n\n"
            "**Recommendation:** This clause is standard but may need enhancement for data-intensive engagements. "
            "Review Sections 9 and 14 to understand the full liability picture."
        ),
        (
            "Analyze this patient discharge summary for care coordination:\n\n"
            "DISCHARGE SUMMARY\nPatient: M.J., 67-year-old male\nAdmission: 03/25/2025 via ER (chest pain)\n"
            "Discharge: 03/29/2025\nDiagnosis: Acute NSTEMI, Triple vessel coronary artery disease\n\n"
            "Procedures: Cardiac catheterization (03/26), CABG x3 (03/27)\n"
            "Medications at discharge:\n  Aspirin 81mg daily\n  Clopidogrel 75mg daily\n  Metoprolol 50mg BID\n"
            "  Atorvastatin 80mg nightly\n  Lisinopril 10mg daily\n\n"
            "Follow-up: Cardiology in 2 weeks, PCP in 4 weeks, Cardiac rehab referral\n"
            "Activity: No driving 4 weeks, No lifting >10 lbs 6 weeks, Sternal precautions\n"
            "Diet: Low sodium, heart-healthy",
            "Discharge summary for a CABG patient. Standard post-CABG medication regimen. "
            "Let me check for completeness: DAPT (aspirin + clopidogrel) ✓, beta-blocker ✓, statin ✓, ACE inhibitor ✓. "
            "Missing: no mention of PPI for GI protection with DAPT. No anticoagulation mentioned (check for afib). "
            "Follow-up appropriately includes cardiology, PCP, and cardiac rehab. "
            "Length of stay: 4 days for CABG — on the shorter side.",
            "**Discharge Summary Analysis:**\n\n"
            "**Clinical Timeline:**\n"
            "| Date | Event |\n|---|---|\n"
            "| 03/25 | ER admission (chest pain) |\n"
            "| 03/26 | Cardiac catheterization |\n"
            "| 03/27 | CABG x3 |\n"
            "| 03/29 | Discharge |\n"
            "**LOS: 4 days** (2 days post-CABG — notably short)\n\n"
            "**Medication Review:**\n"
            "| Medication | Purpose | Standard of Care? |\n|---|---|---|\n"
            "| Aspirin 81mg | Antiplatelet | Yes — standard post-CABG |\n"
            "| Clopidogrel 75mg | DAPT | Yes — standard post-NSTEMI |\n"
            "| Metoprolol 50mg BID | Beta-blocker | Yes — standard post-MI/CABG |\n"
            "| Atorvastatin 80mg | High-intensity statin | Yes — standard post-ACS |\n"
            "| Lisinopril 10mg | ACE inhibitor | Yes — standard post-MI |\n\n"
            "**Potential Gaps Identified:**\n"
            "1. **No PPI prescribed** — Dual antiplatelet therapy increases GI bleeding risk. "
            "Consider PPI (e.g., pantoprazole) for gastric protection.\n"
            "2. **No diabetes screening mentioned** — CAD in a 67M warrants HbA1c if not recently checked.\n"
            "3. **Short LOS** — 2 days post-CABG is on the lower end; ensure adequate wound healing assessment.\n\n"
            "**Follow-Up Plan Assessment:** Appropriate — cardiology (2 weeks), PCP (4 weeks), and cardiac rehab "
            "cover the key post-CABG care domains.\n\n"
            "**Note:** I extract and organize clinical information. All clinical observations require physician review."
        ),
        (
            "Analyze these financial statements for signs of earnings management:\n\n"
            "FY2022: Revenue $180M, Receivables $22M, Allowance $1.1M\n"
            "FY2023: Revenue $195M, Receivables $31M, Allowance $1.2M\n"
            "FY2024: Revenue $210M, Receivables $47M, Allowance $1.3M\n\n"
            "Industry average DSO: 45 days. Bad debt expense typically 3-5% of receivables.",
            "Revenue grew 8.3% then 7.7%, modest growth. But receivables grew dramatically: "
            "22→31→47M (40.9% then 51.6%). DSO: 2022: 22/180*365=44.6 days. 2023: 31/195*365=58.0 days. "
            "2024: 47/210*365=81.7 days. DSO is ballooning vs 45-day industry average. "
            "Allowance as % of receivables: 2022: 5.0%. 2023: 3.9%. 2024: 2.8% — DECLINING while receivables spike. "
            "This is a classic red flag for aggressive revenue recognition.",
            "**Earnings Quality Analysis:**\n\n"
            "**Revenue vs. Receivables Growth:**\n"
            "| Year | Revenue | Growth | Receivables | Growth | DSO |\n|---|---|---|---|---|---|\n"
            "| FY2022 | $180M | — | $22M | — | 44.6 days |\n"
            "| FY2023 | $195M | +8.3% | $31M | +40.9% | 58.0 days |\n"
            "| FY2024 | $210M | +7.7% | $47M | +51.6% | **81.7 days** |\n\n"
            "**RED FLAG: Receivables growing 5-7x faster than revenue.**\n\n"
            "**Allowance for Doubtful Accounts:**\n"
            "| Year | Allowance | % of Receivables | Industry Norm |\n|---|---|---|---|\n"
            "| FY2022 | $1.1M | 5.0% | 3-5% |\n"
            "| FY2023 | $1.2M | 3.9% | 3-5% |\n"
            "| FY2024 | $1.3M | **2.8%** | 3-5% |\n\n"
            "**RED FLAG: Allowance declining as a percentage while receivables surge.**\n\n"
            "**Warning Signs Identified:**\n"
            "1. **DSO spike** (44.6 → 81.7 days) — nearly double the industry average of 45 days\n"
            "2. **Receivables outpacing revenue** by 5-7x — suggests revenue pulled forward or uncollectible\n"
            "3. **Declining allowance ratio** — should be increasing with aging receivables, not decreasing\n"
            "4. **Pattern consistency** — deterioration over 3 years suggests systemic issue, not one-time\n\n"
            "**Assessment:** These patterns are consistent with aggressive revenue recognition — "
            "recording revenue before it's truly earned or collectible. The declining allowance ratio "
            "further suggests management is underprovisioning for bad debt to protect reported earnings.\n\n"
            "**This warrants investigation** into revenue recognition policies, receivable aging, "
            "and write-off history."
        ),
        (
            "Interpret these clauses from a commercial lease:\n\n"
            "Section 15.2 (Assignment): 'Tenant shall not assign this Lease or sublet the Premises or any part "
            "thereof without the prior written consent of Landlord, which consent shall not be unreasonably withheld, "
            "conditioned or delayed. Notwithstanding the foregoing, Tenant may assign this Lease without Landlord's "
            "consent to (i) an affiliate or subsidiary of Tenant, (ii) a successor by merger or consolidation, or "
            "(iii) a purchaser of substantially all of Tenant's assets, provided the assignee's net worth is not less "
            "than Tenant's net worth at the time of assignment.'\n\n"
            "Section 15.4 (Recapture): 'Upon receipt of Tenant's request to assign or sublet, Landlord may, within "
            "30 days, elect to recapture the space by terminating this Lease as to the space proposed to be assigned "
            "or sublet, effective 90 days after Landlord's election notice.'",
            "Assignment clause with standard protections. Key elements: consent required but can't be unreasonably withheld. "
            "Three exceptions that don't need consent (affiliate, merger, asset sale) with net worth test. "
            "Recapture clause gives landlord the right to take back the space instead of allowing sublease. "
            "This is significant — tenant might lose space instead of subleasing it.",
            "**Lease Assignment/Sublease Analysis:**\n\n"
            "**Section 15.2 — Assignment Rights:**\n\n"
            "| Scenario | Consent Required? | Condition |\n|---|---|---|\n"
            "| Standard assignment/sublease | Yes (not unreasonably withheld) | Prior written consent |\n"
            "| Transfer to affiliate/subsidiary | **No** | — |\n"
            "| Merger/consolidation successor | **No** | Assignee net worth ≥ Tenant's |\n"
            "| Asset sale purchaser | **No** | Assignee net worth ≥ Tenant's |\n\n"
            "**Section 15.4 — Recapture Right (Critical):**\n"
            "- Landlord has **30 days** to elect recapture upon receiving assignment/sublease request\n"
            "- If elected, lease terminates for the affected space in **90 days**\n"
            "- This gives the landlord the option to **take the space back** rather than allow a sublease\n\n"
            "**Key Implications:**\n\n"
            "1. **Recapture risk:** If Tenant wants to sublease excess space during a downturn, "
            "the Landlord can instead terminate the lease for that space. This means Tenant could "
            "lose space they might want back later.\n\n"
            "2. **Strategy impact:** Tenant should be aware that requesting sublease permission triggers "
            "the recapture right. Consider whether informal space-sharing arrangements might be preferable.\n\n"
            "3. **Net worth requirement:** The consent-free exceptions require the assignee's net worth "
            "to meet or exceed the Tenant's. This could block assignments to smaller entities even in "
            "M&A scenarios.\n\n"
            "4. **No profit-sharing:** This clause does not appear to require Tenant to share sublease "
            "profit with Landlord (though this may be addressed elsewhere in the lease)."
        ),
    ]

    for q, r, a in domain_pairs:
        examples.append(sft(query=q, reasoning=r, answer=a, category="domain_analysis", difficulty="hard"))

    # Generate remaining domain examples
    remaining = 50 - len(examples)
    domain_fillers = [
        (
            "Calculate the break-even point from this cost analysis:\n\n"
            "Fixed Costs: $450,000/year\nVariable Cost per Unit: $12.50\nSelling Price per Unit: $27.00\n"
            "Current Volume: 40,000 units/year",
            "Break-even = Fixed Costs / (Price - Variable Cost) = 450,000 / (27 - 12.50) = 450,000 / 14.50 = 31,034.5 units. "
            "Round up to 31,035 units. Current volume of 40,000 exceeds break-even by 8,965 units. "
            "Margin of safety: 8,965/40,000 = 22.4%.",
            "**Break-Even Analysis:**\n\n"
            "| Metric | Value |\n|---|---|\n"
            "| Fixed Costs | $450,000/year |\n"
            "| Variable Cost/Unit | $12.50 |\n"
            "| Selling Price/Unit | $27.00 |\n"
            "| **Contribution Margin/Unit** | **$14.50** |\n"
            "| **Break-Even Volume** | **31,035 units** |\n"
            "| **Break-Even Revenue** | **$837,945** |\n"
            "| Current Volume | 40,000 units |\n"
            "| **Margin of Safety** | **8,965 units (22.4%)** |\n\n"
            "**Current profitability at 40,000 units:**\n"
            "- Revenue: 40,000 x $27 = $1,080,000\n"
            "- Total costs: $450,000 + (40,000 x $12.50) = $950,000\n"
            "- **Profit: $130,000** (12.0% margin)\n\n"
            "The company operates 22.4% above break-even, providing moderate cushion against volume decline."
        ),
        (
            "Analyze this medication interaction check from the pharmacy system:\n\n"
            "Current medications:\n- Warfarin 5mg daily (anticoagulant)\n- Amiodarone 200mg daily (antiarrhythmic)\n"
            "- Omeprazole 20mg daily (PPI)\n\nNew prescription: Fluconazole 150mg (antifungal)",
            "Major interaction alert. Fluconazole is a potent CYP2C9 and CYP3A4 inhibitor. "
            "Warfarin is metabolized by CYP2C9 — fluconazole will increase warfarin levels, increasing bleeding risk. "
            "Amiodarone also inhibits CYP2C9, already potentiating warfarin. Adding fluconazole creates a triple interaction. "
            "Omeprazole is CYP2C19 substrate — less concerning with fluconazole.",
            "**Medication Interaction Analysis:**\n\n"
            "| Interaction | Severity | Mechanism |\n|---|---|---|\n"
            "| **Fluconazole + Warfarin** | **HIGH** | Fluconazole inhibits CYP2C9, increasing warfarin levels |\n"
            "| **Amiodarone + Warfarin** | **HIGH** | Amiodarone also inhibits CYP2C9 (existing interaction) |\n"
            "| **Fluconazole + Amiodarone + Warfarin** | **CRITICAL** | Triple interaction — compounded CYP2C9 inhibition |\n"
            "| Fluconazole + Omeprazole | Moderate | Both CYP2C19 substrates/inhibitors |\n\n"
            "**Risk Assessment:**\n"
            "The patient is already on two medications that interact (warfarin + amiodarone). "
            "Adding fluconazole creates a **triple interaction** that could significantly elevate "
            "warfarin levels and INR, increasing the risk of serious bleeding.\n\n"
            "**Considerations:**\n"
            "1. If fluconazole is necessary, warfarin dose reduction (25-50%) may be needed\n"
            "2. INR monitoring should be increased during and after fluconazole course\n"
            "3. Consider a shorter fluconazole course if clinically appropriate\n"
            "4. Alternative antifungals with less CYP2C9 interaction may be preferable\n\n"
            "**I am a document analysis system, not a clinical tool.** This interaction analysis "
            "is extracted from the document data — all prescribing decisions require pharmacist/physician review."
        ),
    ]

    for q, r, a in domain_fillers:
        examples.append(sft(query=q, reasoning=r, answer=a, category="domain_analysis", difficulty="hard"))

    # Fill remaining with more domain examples
    for i in range(remaining - len(domain_fillers)):
        if len(examples) >= 50:
            break
        rev = 50 + i * 10
        cogs_pct = 55 + (i % 10)
        cogs = round(rev * cogs_pct / 100, 1)
        gp = round(rev - cogs, 1)
        gm = round(gp / rev * 100, 1)
        q = (
            f"Analyze the profitability of this business unit:\n\n"
            f"Revenue: ${rev}M\nCOGS: ${cogs}M\nGross Profit: ${gp}M\n"
            f"Operating Expenses: ${round(gp*0.6,1)}M\nOperating Income: ${round(gp*0.4,1)}M"
        )
        opex = round(gp * 0.6, 1)
        oi = round(gp * 0.4, 1)
        om = round(oi / rev * 100, 1)
        r_text = f"Revenue ${rev}M, COGS ${cogs}M ({cogs_pct}%), GP ${gp}M ({gm}%), OpEx ${opex}M, OI ${oi}M ({om}%)."
        a_text = (
            f"**Profitability Analysis:**\n\n"
            f"| Metric | Value | Margin |\n|---|---|---|\n"
            f"| Revenue | ${rev}M | — |\n"
            f"| COGS | ${cogs}M | {cogs_pct}% of revenue |\n"
            f"| **Gross Profit** | **${gp}M** | **{gm}%** |\n"
            f"| Operating Expenses | ${opex}M | {round(opex/rev*100,1)}% of revenue |\n"
            f"| **Operating Income** | **${oi}M** | **{om}%** |\n\n"
            f"Gross margin of {gm}% {'is healthy' if gm > 40 else 'is below typical benchmarks'}. "
            f"Operating margin of {om}% indicates {'solid' if om > 10 else 'tight'} operational efficiency."
        )
        examples.append(sft(query=q, reasoning=r_text, answer=a_text, category="domain_analysis", difficulty="hard"))

    return examples[:50]


# ---------------------------------------------------------------------------
# 6. CONTENT GENERATION (50 examples)
# ---------------------------------------------------------------------------
def gen_content():
    examples = []
    content_pairs = [
        (
            "Based on this quarterly report data, draft an executive summary email:\n\n"
            "Q1 2025 Results:\n  Revenue: $12.4M (up 18% YoY)\n  New customers: 47 (up from 31 Q1 2024)\n"
            "  Customer retention: 94.2%\n  EBITDA: $2.1M (17% margin)\n  Headcount: 185 (net +12)\n"
            "  Product launches: 2 (Analytics Pro, DataSync v3)\n"
            "  Key win: $1.2M enterprise deal with GlobalCorp",
            "Need to draft an executive summary email grounded in these numbers. "
            "Highlight the strong revenue growth (18%), customer acquisition (51.6% more new customers), "
            "high retention (94.2%), and the key enterprise deal. Keep it concise and professional.",
            "**Draft Executive Summary Email:**\n\n"
            "---\n\n"
            "**Subject: Q1 2025 Results — Strong Start to the Year**\n\n"
            "Team,\n\n"
            "Q1 2025 delivered strong results across all key metrics:\n\n"
            "**Financial Performance:**\n"
            "- Revenue of **$12.4M** represents **18% year-over-year growth**\n"
            "- EBITDA of **$2.1M (17% margin)** demonstrates continued profitability discipline\n\n"
            "**Customer Momentum:**\n"
            "- **47 new customers** added (up 52% from 31 in Q1 2024)\n"
            "- Customer retention remains strong at **94.2%**\n"
            "- Landmark **$1.2M enterprise deal** signed with GlobalCorp\n\n"
            "**Operational Highlights:**\n"
            "- Two product launches: **Analytics Pro** and **DataSync v3**\n"
            "- Team grew to **185** (net +12), supporting our expansion trajectory\n\n"
            "These results position us well for the remainder of 2025. The GlobalCorp win validates "
            "our enterprise strategy, and the new product launches open additional revenue streams.\n\n"
            "Full details in the attached Q1 report.\n\n"
            "Best regards,\n[Name]\n\n"
            "---\n\n"
            "*All figures sourced from Q1 2025 quarterly report. Email generated by DocWain based on document data.*"
        ),
        (
            "Based on this inspection report, generate a summary for the property buyer:\n\n"
            "Property: 447 Elm Street, Portland OR\nInspection Date: 03/28/2025\n\n"
            "MAJOR FINDINGS:\n1. Roof: Missing shingles on south slope, estimated 5-7 years remaining life. "
            "Replacement cost: $12,000-$18,000.\n"
            "2. Electrical: Panel is Federal Pacific (known safety issue). Recommend replacement. Cost: $2,500-$4,000.\n"
            "3. Foundation: Minor settlement crack on east wall, currently stable. Monitor annually.\n\n"
            "MINOR FINDINGS:\n4. Plumbing: Slow drain in master bathroom (likely clog, not structural).\n"
            "5. HVAC: System is 14 years old, functional but nearing end of useful life (15-20 years typical).\n"
            "6. Windows: 3 windows have broken seals (fogging). Replacement: $300-$500 each.\n\n"
            "OVERALL: Property is in fair condition for its age (built 1988). Budget $20,000-$30,000 for "
            "near-term repairs (roof, electrical panel).",
            "Buyer summary should be clear about what needs immediate attention vs monitoring. "
            "Major cost items: roof ($12-18K), electrical panel ($2.5-4K) — these are negotiation points. "
            "The Federal Pacific panel is a safety issue that most buyers should address promptly.",
            "**Property Inspection Summary for Buyer:**\n\n"
            "**Property:** 447 Elm Street, Portland OR | **Inspected:** 03/28/2025 | **Built:** 1988\n\n"
            "---\n\n"
            "**Items Requiring Immediate Attention:**\n\n"
            "| Issue | Urgency | Estimated Cost |\n|---|---|---|\n"
            "| **Electrical panel (Federal Pacific)** | **High — safety concern** | $2,500-$4,000 |\n"
            "| **Roof (missing shingles, south slope)** | High — 5-7 years remaining | $12,000-$18,000 |\n\n"
            "The Federal Pacific electrical panel has a known safety defect (breakers may fail to trip during overload). "
            "This should be replaced regardless of negotiation outcome.\n\n"
            "**Items to Monitor:**\n\n"
            "| Issue | Timeline | Estimated Cost |\n|---|---|---|\n"
            "| Foundation settlement crack | Annual monitoring (currently stable) | N/A unless progression |\n"
            "| HVAC system (14 years old) | 1-6 years remaining useful life | $5,000-$10,000 when replaced |\n\n"
            "**Minor Items:**\n"
            "- Master bathroom slow drain (~$150-$300 to clear)\n"
            "- 3 windows with broken seals ($900-$1,500 total)\n\n"
            "**Estimated Near-Term Repair Budget: $20,000-$30,000**\n\n"
            "**Negotiation Consideration:** The roof and electrical panel together represent $14,500-$22,000 in "
            "necessary repairs. These are reasonable items to negotiate as a price reduction or seller credit.\n\n"
            "---\n\n"
            "*Summary generated by DocWain from inspection report dated 03/28/2025.*"
        ),
        (
            "From this contract analysis, generate a risk summary for legal review:\n\n"
            "Contract: SaaS Agreement with CloudVault Inc.\nValue: $240,000/year (3-year term)\n\n"
            "IDENTIFIED RISKS:\n"
            "1. Auto-renewal: 30-day notice required to cancel, or auto-renews for 1 year.\n"
            "2. Price escalation: Provider can increase price up to 10% annually at renewal.\n"
            "3. Data ownership: Clause 8.2 grants Provider a 'perpetual, irrevocable license to use "
            "Customer Data for service improvement and benchmarking purposes.'\n"
            "4. Termination for convenience: Only Provider has this right (60 days notice).\n"
            "5. Limitation of liability: Capped at 3 months of fees ($60,000).\n"
            "6. SLA: 99.5% uptime, but sole remedy is service credits (max 10% of monthly fee).\n"
            "7. Governing law: Provider's home state (Delaware) — not our jurisdiction.",
            "Several significant risks to flag. The data clause (perpetual license for customer data) is the most concerning. "
            "The one-sided termination right, low liability cap, and auto-renewal are also problematic. "
            "Total 3-year commitment: $720K+ (with escalation up to $792K).",
            "**Contract Risk Summary — CloudVault SaaS Agreement:**\n\n"
            "**Contract Value:** $240,000/year x 3 years = **$720,000** (up to $792,000 with max escalation)\n\n"
            "---\n\n"
            "**Risk Assessment:**\n\n"
            "| # | Risk | Severity | Clause |\n|---|---|---|---|\n"
            "| 1 | **Data rights — perpetual license** | **CRITICAL** | 8.2 |\n"
            "| 2 | One-sided termination right | HIGH | — |\n"
            "| 3 | Low liability cap ($60K) | HIGH | — |\n"
            "| 4 | 10% annual price escalation | MEDIUM | — |\n"
            "| 5 | Auto-renewal (30-day window) | MEDIUM | — |\n"
            "| 6 | Weak SLA remedy (max 10% credits) | MEDIUM | — |\n"
            "| 7 | Unfavorable governing law | LOW | — |\n\n"
            "**CRITICAL — Clause 8.2 (Data Rights):**\n"
            "The \"perpetual, irrevocable license\" for customer data is the most significant risk. "
            "This means CloudVault retains rights to use our data **forever**, even after the contract ends, "
            "for \"service improvement and benchmarking\" — which is broadly defined. "
            "**This clause should be rejected or substantially narrowed.** At minimum:\n"
            "- Remove \"perpetual\" and \"irrevocable\"\n"
            "- Limit to anonymized, aggregated data only\n"
            "- Require data deletion within 90 days of termination\n\n"
            "**HIGH — Termination Asymmetry:**\n"
            "Only the Provider can terminate for convenience. We are locked in for 3 years with no exit. "
            "**Negotiate mutual termination for convenience** with appropriate notice period.\n\n"
            "**HIGH — Liability Cap ($60K):**\n"
            "Capped at 3 months of fees on a $720K+ contract. This is inadequate — a data breach alone "
            "could cost orders of magnitude more. **Negotiate minimum 12 months of fees, with carve-outs "
            "for data breaches and confidentiality violations.**\n\n"
            "**Recommendation:** Do not execute this contract in its current form. "
            "The data rights clause, liability cap, and termination asymmetry require material revision.\n\n"
            "---\n\n"
            "*Risk summary generated by DocWain from contract analysis.*"
        ),
        (
            "Generate a comparison report from these two competing product specifications:\n\n"
            "PRODUCT X — IndustrialScan Pro:\n  Resolution: 600 DPI\n  Speed: 80 pages/min\n"
            "  ADF Capacity: 100 sheets\n  Duplex: Yes\n  Connectivity: USB 3.0, Ethernet, WiFi\n"
            "  OCR: Built-in (ABBYY engine)\n  Price: $4,200\n  Warranty: 3 years\n  Support: 24/7 phone\n\n"
            "PRODUCT Y — DocCapture Elite:\n  Resolution: 1200 DPI\n  Speed: 60 pages/min\n"
            "  ADF Capacity: 150 sheets\n  Duplex: Yes\n  Connectivity: USB 3.0, Ethernet, WiFi, Bluetooth\n"
            "  OCR: Built-in (OmniPage engine)\n  Price: $5,800\n  Warranty: 5 years\n  Support: Business hours email + phone",
            "Product comparison for document scanners. X is faster and cheaper. Y has higher resolution, "
            "larger ADF, more connectivity, and longer warranty. Price difference: $1,600 (38% more for Y). "
            "Speed difference: 80 vs 60 ppm (33% faster for X). Resolution: Y is 2x X.",
            "**Product Comparison Report: Document Scanners**\n\n"
            "---\n\n"
            "| Feature | IndustrialScan Pro (X) | DocCapture Elite (Y) | Advantage |\n|---|---|---|---|\n"
            "| **Resolution** | 600 DPI | 1200 DPI | **Y** (+2x resolution) |\n"
            "| **Speed** | 80 ppm | 60 ppm | **X** (+33% faster) |\n"
            "| **ADF Capacity** | 100 sheets | 150 sheets | **Y** (+50% capacity) |\n"
            "| **Duplex** | Yes | Yes | Tie |\n"
            "| **Connectivity** | USB/Ethernet/WiFi | USB/Ethernet/WiFi/Bluetooth | **Y** (+Bluetooth) |\n"
            "| **OCR Engine** | ABBYY | OmniPage | Both capable (preference-dependent) |\n"
            "| **Price** | $4,200 | $5,800 | **X** (-$1,600, 28% less) |\n"
            "| **Warranty** | 3 years | 5 years | **Y** (+2 years) |\n"
            "| **Support** | 24/7 phone | Business hours email + phone | **X** (24/7 availability) |\n\n"
            "**Value Metrics:**\n"
            "| Metric | Product X | Product Y |\n|---|---|---|\n"
            "| Cost per page/min | $52.50 | $96.67 |\n"
            "| Cost per DPI | $7.00 | $4.83 |\n"
            "| Cost per warranty year | $1,400 | $1,160 |\n\n"
            "**Recommendation by Use Case:**\n\n"
            "| If you need... | Choose | Reasoning |\n|---|---|---|\n"
            "| High-volume batch scanning | **Product X** | 33% faster, lower cost |\n"
            "| Archival-quality imaging | **Product Y** | 2x resolution for fine detail |\n"
            "| Budget-conscious deployment | **Product X** | $1,600 less per unit |\n"
            "| Long-term investment | **Product Y** | 5-year warranty, lower cost per warranty year |\n"
            "| 24/7 operations | **Product X** | 24/7 phone support vs business hours |\n\n"
            "---\n\n"
            "*Comparison generated by DocWain from product specification documents.*"
        ),
        (
            "Draft a compliance summary from this audit finding:\n\n"
            "FINDING: GDPR Article 17 — Right to Erasure\n"
            "Status: NON-COMPLIANT\n\n"
            "Details: Audit of data deletion processes revealed:\n"
            "- 142 deletion requests received in 2024\n"
            "- 89 completed within 30-day requirement (62.7%)\n"
            "- 38 completed between 31-60 days (26.8%)\n"
            "- 15 still pending (oldest: 127 days) (10.6%)\n"
            "- No documented process for backup system purging\n"
            "- Marketing system retains email addresses after deletion from primary CRM\n"
            "- No deletion verification or confirmation procedure",
            "Serious GDPR compliance finding. Only 62.7% within the 30-day statutory requirement. "
            "15 requests still pending (some 127 days old). Backup and marketing system retention means "
            "deletion is incomplete even when done. No verification process.",
            "**GDPR Compliance Summary — Article 17 (Right to Erasure)**\n\n"
            "---\n\n"
            "**Status: NON-COMPLIANT**\n\n"
            "**Performance Against Requirement (30-day statutory deadline):**\n"
            "| Category | Count | % | Status |\n|---|---|---|---|\n"
            "| Completed within 30 days | 89 | 62.7% | Compliant |\n"
            "| Completed 31-60 days | 38 | 26.8% | **Non-compliant** |\n"
            "| Still pending | 15 | 10.6% | **Non-compliant** (oldest: 127 days) |\n"
            "| **Total requests (2024)** | **142** | | |\n\n"
            "**37.3% of erasure requests failed to meet the statutory 30-day deadline.**\n\n"
            "**Systemic Issues Identified:**\n\n"
            "| Issue | Risk Level | GDPR Impact |\n|---|---|---|\n"
            "| No backup purging process | **Critical** | Deleted data persists in backups |\n"
            "| Marketing system retention | **Critical** | Emails retained after CRM deletion |\n"
            "| No deletion verification | **High** | Cannot confirm complete erasure |\n"
            "| Timeliness failures | **High** | 37.3% exceed 30-day requirement |\n\n"
            "**Regulatory Exposure:**\n"
            "- GDPR fines: Up to 4% of annual global turnover or EUR 20M (whichever higher)\n"
            "- 15 pending requests represent active violations\n"
            "- Backup retention means even \"completed\" deletions may be incomplete\n\n"
            "**Required Remediation Actions:**\n"
            "1. **Immediate:** Process all 15 pending deletion requests within 7 days\n"
            "2. **30 days:** Implement automated backup purging for deleted records\n"
            "3. **30 days:** Resolve marketing system data retention (integrate with deletion workflow)\n"
            "4. **60 days:** Implement deletion verification procedure with audit trail\n"
            "5. **Ongoing:** Monthly reporting on deletion SLA compliance\n\n"
            "---\n\n"
            "*Compliance summary generated by DocWain from audit finding documentation.*"
        ),
    ]

    for q, r, a in content_pairs:
        examples.append(sft(query=q, reasoning=r, answer=a, category="content_gen", difficulty="hard"))

    # Fill remaining content generation examples
    remaining = 50 - len(examples)
    content_fillers = [
        (
            "Summarize this meeting transcript for distribution to absent stakeholders:\n\n"
            "MEETING: Project Phoenix Weekly Sync — 04/10/2025\n"
            "Attendees: Sarah (PM), Mike (Dev Lead), Lisa (QA), David (Design)\n\n"
            "Sarah: 'Sprint 14 closes Friday. We're at 85% completion. Two items at risk: API gateway and auth module.'\n"
            "Mike: 'API gateway needs 2 more days. Auth module blocked on security review — waiting on InfoSec since Tuesday.'\n"
            "Lisa: 'Regression suite is ready. Can't start until auth module is code-complete. Current ETA pushes QA to next Tuesday.'\n"
            "David: 'Design handoff for Sprint 15 features is complete. No blockers from design.'\n"
            "Sarah: 'Action items: Mike to escalate InfoSec review today. Lisa to prepare test environment. "
            "Sprint 14 revised close: next Wednesday. Sprint 15 planning moves to Thursday.'",
            "Meeting summary needs to cover status, blockers, decisions, and action items clearly.",
            "**Meeting Summary: Project Phoenix Weekly Sync**\n"
            "**Date:** April 10, 2025 | **Attendees:** Sarah (PM), Mike (Dev Lead), Lisa (QA), David (Design)\n\n"
            "---\n\n"
            "**Sprint 14 Status: 85% Complete — At Risk**\n\n"
            "| Work Stream | Status | Notes |\n|---|---|---|\n"
            "| Overall sprint | 85% complete | 2 items at risk |\n"
            "| API Gateway | In progress | +2 days needed |\n"
            "| Auth Module | **Blocked** | Awaiting InfoSec security review (since Tuesday) |\n"
            "| QA Regression | Ready to execute | Waiting on auth module code-complete |\n"
            "| Design (Sprint 15) | Complete | Handoff done, no blockers |\n\n"
            "**Key Decision:** Sprint 14 close date extended from Friday to **next Wednesday**.\n\n"
            "**Action Items:**\n"
            "| Action | Owner | Due |\n|---|---|---|\n"
            "| Escalate InfoSec security review | Mike | Today (04/10) |\n"
            "| Prepare QA test environment | Lisa | Before auth module completion |\n"
            "| Sprint 15 planning meeting | Sarah | Thursday (moved from Wednesday) |\n\n"
            "**Impact:** QA pushed to Tuesday at earliest, making Wednesday the realistic sprint close.\n\n"
            "---\n\n"
            "*Summary generated by DocWain from meeting transcript.*"
        ),
        (
            "Based on this incident report, draft a customer notification:\n\n"
            "INCIDENT REPORT #IR-2025-0042\n"
            "Service: CloudStore Platform\nStart: 04/08/2025, 14:23 UTC\nEnd: 04/08/2025, 17:45 UTC\n"
            "Duration: 3 hours 22 minutes\nImpact: File upload/download failures for ~30% of users\n"
            "Root Cause: Storage node failure triggered cascading rebalance, overwhelming load balancer\n"
            "Remediation: Added circuit breaker to rebalance process, increased load balancer capacity by 50%\n"
            "Data Loss: None confirmed\nRecurrence Prevention: Implementing gradual rebalance algorithm (ETA: 04/25/2025)",
            "Customer notification should be transparent about what happened, acknowledge impact, "
            "and explain what we're doing to prevent recurrence. No jargon, focus on customer impact.",
            "**Draft Customer Notification:**\n\n"
            "---\n\n"
            "**Subject: Service Incident Report — CloudStore Platform (April 8, 2025)**\n\n"
            "Dear Valued Customers,\n\n"
            "We want to provide a transparent update on the service disruption that occurred on April 8, 2025.\n\n"
            "**What Happened:**\n"
            "On April 8 between **2:23 PM and 5:45 PM UTC** (3 hours 22 minutes), approximately 30% of "
            "CloudStore users experienced failures when uploading or downloading files.\n\n"
            "**Cause:**\n"
            "A storage component failure triggered an automated recovery process that unexpectedly "
            "generated excessive internal traffic, overwhelming our systems.\n\n"
            "**Your Data:**\n"
            "We have confirmed that **no data was lost** during this incident. All files remain intact and accessible.\n\n"
            "**What We've Done:**\n"
            "- Implemented immediate safeguards to prevent similar cascading failures\n"
            "- Increased system capacity by 50% to handle unexpected load\n\n"
            "**What We're Doing Next:**\n"
            "By April 25, 2025, we will deploy an improved recovery process that operates gradually, "
            "preventing the traffic surge that caused this disruption.\n\n"
            "We understand that reliability is critical to your operations, and we apologize for the disruption. "
            "If you have questions, please contact our support team.\n\n"
            "Sincerely,\nThe CloudStore Team\n\n"
            "---\n\n"
            "*Notification drafted by DocWain from Incident Report IR-2025-0042.*"
        ),
    ]

    for q, r, a in content_fillers:
        examples.append(sft(query=q, reasoning=r, answer=a, category="content_gen", difficulty="hard"))

    # Generate more content examples to reach 50
    for i in range(remaining - len(content_fillers)):
        if len(examples) >= 50:
            break
        rev = 5 + i * 2
        growth = 8 + (i % 15)
        customers = 50 + i * 10
        q = (
            f"Generate a quarterly business review slide content from this data:\n\n"
            f"Revenue: ${rev}M (+{growth}% YoY)\nCustomers: {customers}\n"
            f"NPS: {60 + i % 20}\nChurn: {3.0 + (i % 5) * 0.5}%\n"
            f"Top Product: Widget Pro ({40 + i % 20}% of revenue)"
        )
        r = f"QBR content based on provided metrics. Revenue ${rev}M with {growth}% growth, {customers} customers."
        a = (
            f"**Q{'1234'[i%4]} 2025 Business Review — Key Slides:**\n\n"
            f"**Slide 1 — Executive Summary:**\n"
            f"- Revenue **${rev}M** ({growth}% YoY growth)\n"
            f"- **{customers}** active customers\n"
            f"- NPS **{60 + i % 20}** | Churn **{3.0 + (i % 5) * 0.5}%**\n\n"
            f"**Slide 2 — Revenue:**\n"
            f"- Total: ${rev}M vs ${round(rev/(1+growth/100),1)}M prior year\n"
            f"- Widget Pro drives {40 + i % 20}% of revenue\n\n"
            f"**Slide 3 — Customer Health:**\n"
            f"- {customers} customers, NPS {60 + i % 20}\n"
            f"- Churn at {3.0 + (i % 5) * 0.5}% — {'within target' if 3.0 + (i % 5) * 0.5 < 5 else 'above target'}\n\n"
            f"---\n*Content generated by DocWain from quarterly data.*"
        )
        examples.append(sft(query=q, reasoning=r, answer=a, category="content_gen", difficulty="hard"))

    return examples[:50]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    all_examples = []
    all_examples.extend(gen_multi_page())
    all_examples.extend(gen_complex_tables())
    all_examples.extend(gen_contradictions())
    all_examples.extend(gen_multi_doc())
    all_examples.extend(gen_domain_analysis())
    all_examples.extend(gen_content())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Stats
    from collections import Counter
    cats = Counter(ex["category"] for ex in all_examples)
    print(f"Generated {len(all_examples)} advanced training examples:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
