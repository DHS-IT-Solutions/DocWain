#!/usr/bin/env python3
"""
Generate teacher-quality training data for DocWain-14B-v2.

This script generates high-quality SFT and DPO examples that teach DocWain
to be both fast and intelligent. Each example is carefully crafted with:
- Deep reasoning chains in <think> blocks
- Grounded, precise answers with citations
- Domain-aware analysis
- Proper uncertainty handling

Output: JSONL files ready for Unsloth training.
"""

import json
import random
from pathlib import Path

SYSTEM = (
    "You are DocWain, an enterprise document intelligence assistant. "
    "You analyse documents with deep contextual understanding, extract "
    "structured information, identify patterns and anomalies, and provide "
    "holistic analysis grounded in evidence. You reason step-by-step before "
    "answering, state your confidence level, and cite specific sources. "
    "When information is insufficient, you say so clearly rather than guessing."
)

OUTPUT_DIR = Path("finetune_artifacts/sprint/teacher_quality")


def sft(query: str, reasoning: str, answer: str, category: str, difficulty: str = "medium") -> dict:
    text = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{answer}<|im_end|>"
    )
    return {"text": text, "category": category, "difficulty": difficulty, "source": "teacher_quality"}


def dpo(query: str, chosen_reasoning: str, chosen_answer: str,
        rejected_reasoning: str, rejected_answer: str, category: str) -> dict:
    prompt = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    chosen = f"<think>\n{chosen_reasoning}\n</think>\n\n{chosen_answer}<|im_end|>"
    rejected = f"<think>\n{rejected_reasoning}\n</think>\n\n{rejected_answer}<|im_end|>"
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected,
            "category": category, "source": "teacher_quality"}


def save_jsonl(examples: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved {len(examples)} examples to {path}")


# ============================================================================
# BATCH 1: EXTRACTION — Deep, thorough, structured extraction
# ============================================================================
def generate_extraction_examples() -> list:
    examples = []

    # --- Invoice Extraction (varied complexity) ---
    examples.append(sft(
        query="""Extract all key information from this invoice:

INVOICE INV-2026-0847
Date: March 15, 2026
Due Date: April 14, 2026

From: Nexus Cloud Solutions
      1200 Technology Drive, Suite 400
      San Jose, CA 95110
      Tax ID: 94-3847261

Bill To: Meridian Healthcare Group
         500 Medical Center Blvd
         Boston, MA 02115
         PO#: MHG-2026-1102

| Description | Qty | Unit Price | Amount |
|------------|-----|-----------|--------|
| Cloud Infrastructure (Monthly) | 1 | $12,500.00 | $12,500.00 |
| Data Migration Services | 80 hrs | $175.00 | $14,000.00 |
| SSL Certificate (Annual) | 3 | $299.00 | $897.00 |
| 24/7 Support Package | 1 | $3,200.00 | $3,200.00 |
| Storage Expansion (500GB) | 2 | $450.00 | $900.00 |

Subtotal: $31,497.00
Discount (10% - Contract Rate): -$3,149.70
Taxable Amount: $28,347.30
Sales Tax (6.25%): $1,771.71
Total Due: $30,119.01

Payment Terms: Net 30
Bank: First Republic Bank
Account: 4892-7761-0034
Routing: 321081669

Notes: This invoice covers services for Q1 2026 as per MSA dated January 1, 2025.""",
        reasoning="""Let me systematically extract every piece of information from this invoice.

Document type: Invoice
Invoice number: INV-2026-0847
Issue date: March 15, 2026
Due date: April 14, 2026

Vendor information:
- Company: Nexus Cloud Solutions
- Address: 1200 Technology Drive, Suite 400, San Jose, CA 95110
- Tax ID: 94-3847261

Client information:
- Company: Meridian Healthcare Group
- Address: 500 Medical Center Blvd, Boston, MA 02115
- Purchase Order: MHG-2026-1102

Line items (5 total):
1. Cloud Infrastructure (Monthly): 1 × $12,500.00 = $12,500.00
2. Data Migration Services: 80 hrs × $175.00 = $14,000.00
3. SSL Certificate (Annual): 3 × $299.00 = $897.00
4. 24/7 Support Package: 1 × $3,200.00 = $3,200.00
5. Storage Expansion (500GB): 2 × $450.00 = $900.00

Financial calculations:
- Subtotal: $31,497.00
- Discount: 10% contract rate = -$3,149.70
- Taxable amount: $28,347.30
- Sales tax: 6.25% = $1,771.71
- Grand total: $30,119.01

Let me verify: $31,497.00 - $3,149.70 = $28,347.30 ✓
$28,347.30 × 0.0625 = $1,771.71 ✓
$28,347.30 + $1,771.71 = $30,119.01 ✓

Payment details:
- Terms: Net 30
- Bank: First Republic Bank
- Account: 4892-7761-0034
- Routing: 321081669

Additional context: References MSA (Master Service Agreement) dated January 1, 2025, covering Q1 2026 services.""",
        answer="""## Invoice Extraction — INV-2026-0847

### Document Details
- **Invoice Number:** INV-2026-0847
- **Date:** March 15, 2026
- **Due Date:** April 14, 2026
- **Reference:** Q1 2026 services per MSA dated January 1, 2025

### Vendor
- **Company:** Nexus Cloud Solutions
- **Address:** 1200 Technology Drive, Suite 400, San Jose, CA 95110
- **Tax ID:** 94-3847261

### Client
- **Company:** Meridian Healthcare Group
- **Address:** 500 Medical Center Blvd, Boston, MA 02115
- **PO Number:** MHG-2026-1102

### Line Items

| # | Description | Qty | Unit Price | Total |
|---|------------|-----|-----------|-------|
| 1 | Cloud Infrastructure (Monthly) | 1 | $12,500.00 | **$12,500.00** |
| 2 | Data Migration Services | 80 hrs | $175.00/hr | **$14,000.00** |
| 3 | SSL Certificate (Annual) | 3 | $299.00 | **$897.00** |
| 4 | 24/7 Support Package | 1 | $3,200.00 | **$3,200.00** |
| 5 | Storage Expansion (500GB) | 2 | $450.00 | **$900.00** |

### Financial Summary
- **Subtotal:** $31,497.00
- **Discount (10% Contract Rate):** -$3,149.70
- **Taxable Amount:** $28,347.30
- **Sales Tax (6.25%):** $1,771.71
- **Grand Total Due:** **$30,119.01**

### Payment Information
- **Terms:** Net 30
- **Bank:** First Republic Bank
- **Account:** 4892-7761-0034
- **Routing:** 321081669

### Key Observations
- Data Migration Services ($14,000) is the largest line item at 44.5% of subtotal
- 10% contract discount saves the client $3,149.70
- Healthcare client (Meridian Healthcare Group) — HIPAA compliance may apply to cloud services
- All calculations verified: subtotal, discount, tax, and total are mathematically correct""",
        category="extraction", difficulty="hard"))

    # --- Contract Extraction ---
    examples.append(sft(
        query="""Extract all key terms from this contract:

MASTER SERVICE AGREEMENT

This Agreement is entered into as of February 1, 2026 ("Effective Date") by and between:

DataVault Technologies, Inc., a Delaware corporation with offices at 2300 Innovation Way, Austin, TX 78701 ("Provider")

and

Pinnacle Financial Services, LLC, a New York limited liability company with offices at 450 Park Avenue, 22nd Floor, New York, NY 10022 ("Client")

1. SCOPE OF SERVICES
Provider shall deliver enterprise data management and analytics services as described in Exhibit A ("Services"), including:
(a) Real-time data pipeline management processing up to 50 million events per day
(b) Custom dashboard development and maintenance
(c) Quarterly business intelligence reports

2. TERM AND RENEWAL
The initial term shall be thirty-six (36) months from the Effective Date. This Agreement shall automatically renew for successive twelve (12) month periods unless either party provides written notice of non-renewal at least ninety (90) days prior to the end of the then-current term.

3. FEES AND PAYMENT
(a) Annual License Fee: $840,000, payable in equal quarterly installments of $210,000
(b) Implementation Fee: $125,000 (one-time, due within 30 days of Effective Date)
(c) Professional Services: $275/hour for additional consulting beyond Exhibit A scope
(d) Annual escalation: Fees increase by CPI or 3%, whichever is less, upon each renewal

4. DATA SECURITY AND COMPLIANCE
(a) Provider shall maintain SOC 2 Type II certification throughout the term
(b) All data shall be encrypted at rest (AES-256) and in transit (TLS 1.3)
(c) Provider shall comply with GLBA, SOX, and applicable SEC regulations
(d) Annual penetration testing by independent third party, results shared with Client

5. SERVICE LEVEL AGREEMENT
(a) System Availability: 99.95% uptime measured monthly (excludes scheduled maintenance)
(b) Response Time: Critical issues within 1 hour, High within 4 hours, Medium within 8 hours
(c) SLA Credits: For each 0.1% below 99.95%, Client receives 5% credit on monthly fee, capped at 30%

6. INTELLECTUAL PROPERTY
(a) Client retains all rights to its data and custom deliverables created under this Agreement
(b) Provider retains rights to its pre-existing IP and general methodologies

7. LIMITATION OF LIABILITY
(a) Neither party's total liability shall exceed 2x the annual fees paid in the 12 months preceding the claim
(b) Neither party shall be liable for indirect, consequential, or punitive damages
(c) Exclusions: The limitations in 7(a) and 7(b) shall not apply to breaches of Section 4 (Data Security) or willful misconduct

8. TERMINATION
(a) Either party may terminate for material breach with 30 days written notice and cure period
(b) Client may terminate for convenience with 180 days written notice, subject to early termination fee equal to 50% of remaining term fees
(c) Upon termination, Provider shall return or destroy all Client data within 30 days

9. GOVERNING LAW
This Agreement shall be governed by the laws of the State of New York. Any disputes shall be resolved through binding arbitration in New York City under AAA Commercial Rules.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the Effective Date.

DataVault Technologies, Inc.     Pinnacle Financial Services, LLC
By: James Chen, CEO              By: Sarah Morrison, CFO""",
        reasoning="""This is a comprehensive Master Service Agreement. Let me extract every material term systematically.

Parties:
- Provider: DataVault Technologies, Inc. (Delaware corp, Austin TX)
- Client: Pinnacle Financial Services, LLC (New York LLC, NYC)
- Signatories: James Chen (CEO) for Provider, Sarah Morrison (CFO) for Client

Key dates:
- Effective Date: February 1, 2026
- Initial term end: January 31, 2029 (36 months)
- Auto-renewal: 12-month periods
- Non-renewal notice: 90 days before term end

Financial terms:
- Annual license: $840,000 ($210,000/quarter)
- Implementation: $125,000 one-time
- Consulting rate: $275/hour (beyond Exhibit A)
- Escalation: CPI or 3% cap per renewal
- Total first-year value: $840,000 + $125,000 = $965,000
- Total 3-year minimum: $840,000 × 3 + $125,000 = $2,645,000

Services scope:
- Data pipeline management (50M events/day)
- Custom dashboards
- Quarterly BI reports

Compliance requirements:
- SOC 2 Type II (maintained throughout)
- AES-256 encryption at rest
- TLS 1.3 in transit
- GLBA, SOX, SEC compliance
- Annual penetration testing

SLA terms:
- 99.95% uptime monthly
- Response: Critical 1hr, High 4hr, Medium 8hr
- Credits: 5% per 0.1% shortfall, max 30%

Liability:
- Cap: 2× annual fees (preceding 12 months)
- No indirect/consequential/punitive damages
- IMPORTANT: Cap doesn't apply to data security breaches or willful misconduct

Termination:
- For cause: 30 days notice + cure period
- For convenience: 180 days notice + 50% of remaining fees
- Data return: 30 days post-termination

IP: Client owns custom deliverables + their data; Provider keeps pre-existing IP

Governing law: New York, AAA arbitration in NYC""",
        answer="""## Contract Analysis — Master Service Agreement

### Parties
| Role | Entity | Jurisdiction | Signatory |
|------|--------|-------------|-----------|
| **Provider** | DataVault Technologies, Inc. | Delaware corporation | James Chen, CEO |
| **Client** | Pinnacle Financial Services, LLC | New York LLC | Sarah Morrison, CFO |

### Key Dates & Term
- **Effective Date:** February 1, 2026
- **Initial Term:** 36 months (expires January 31, 2029)
- **Auto-Renewal:** 12-month periods
- **Non-Renewal Notice:** 90 days prior to term end

### Financial Terms
| Component | Amount | Frequency |
|-----------|--------|-----------|
| Annual License Fee | **$840,000** | Quarterly ($210,000) |
| Implementation Fee | **$125,000** | One-time (due within 30 days) |
| Professional Services | **$275/hour** | As needed (beyond Exhibit A) |
| Annual Escalation | CPI or 3% (whichever less) | Each renewal |

- **First-Year Total:** $965,000 (license + implementation)
- **3-Year Minimum Contract Value:** $2,645,000

### Services (Exhibit A)
1. Real-time data pipeline management — up to **50 million events/day**
2. Custom dashboard development and maintenance
3. Quarterly business intelligence reports

### Compliance & Security
- **SOC 2 Type II** — maintained throughout term
- **Encryption:** AES-256 at rest, TLS 1.3 in transit
- **Regulatory:** GLBA, SOX, SEC compliance
- **Testing:** Annual independent penetration testing, results shared with Client

### SLA
| Metric | Target |
|--------|--------|
| Uptime | **99.95%** monthly |
| Critical Response | **1 hour** |
| High Response | **4 hours** |
| Medium Response | **8 hours** |
| SLA Credit | 5% per 0.1% below target, **max 30%** |

### Liability & Risk
- **Liability Cap:** 2× annual fees paid in preceding 12 months (~$1,680,000)
- **No** indirect, consequential, or punitive damages
- **Critical Exception:** Cap does NOT apply to data security breaches (§4) or willful misconduct — **unlimited liability for data breaches**

### Termination
| Trigger | Notice | Financial Impact |
|---------|--------|-----------------|
| Material breach | 30 days + cure | None |
| Convenience (Client) | 180 days | **50% of remaining term fees** |
| Post-termination | — | Data return/destruction within 30 days |

### IP Ownership
- **Client owns:** All data + custom deliverables
- **Provider retains:** Pre-existing IP + general methodologies

### Governing Law
- **Jurisdiction:** State of New York
- **Dispute Resolution:** Binding arbitration, NYC, AAA Commercial Rules

### Risk Assessment
1. **High:** Unlimited liability for data security breaches — Provider carries significant risk given financial services data
2. **Medium:** 50% early termination fee could be substantial ($420,000+ depending on timing)
3. **Low:** CPI-capped escalation protects Client from aggressive price increases""",
        category="extraction", difficulty="hard"))

    # --- Medical Record Extraction ---
    examples.append(sft(
        query="""Extract all clinical information from this medical record:

PATIENT ENCOUNTER NOTE

Patient: Maria Elena Vasquez
MRN: 847-22-9103
DOB: 08/14/1968 (Age: 57)
Visit Date: 03/10/2026
Provider: Dr. Robert Kim, MD — Internal Medicine
Facility: St. Catherine's Medical Center

CHIEF COMPLAINT: Follow-up for diabetes management and new onset chest tightness

HISTORY OF PRESENT ILLNESS:
Mrs. Vasquez is a 57-year-old female with a 12-year history of Type 2 Diabetes Mellitus and hypertension presenting for routine follow-up. She reports episodes of chest tightness during exertion over the past 3 weeks, typically lasting 5-10 minutes and resolving with rest. No radiation to arm or jaw. She denies dyspnea at rest, orthopnea, or peripheral edema. Her blood glucose logs show fasting levels ranging from 145-180 mg/dL (target <130). She reports compliance with medications but admits to dietary lapses during recent holiday travel.

CURRENT MEDICATIONS:
1. Metformin 1000mg BID
2. Glipizide 10mg daily
3. Lisinopril 20mg daily
4. Atorvastatin 40mg daily
5. Aspirin 81mg daily

ALLERGIES: Sulfa drugs (rash), Codeine (nausea)

VITAL SIGNS:
BP: 148/92 mmHg (elevated)
HR: 78 bpm, regular
Temp: 98.4°F
SpO2: 97% on room air
Weight: 187 lbs (BMI: 32.1)

LABORATORY RESULTS (drawn 03/03/2026):
HbA1c: 8.2% (previous: 7.4%, target <7.0%)
Fasting Glucose: 168 mg/dL
Total Cholesterol: 224 mg/dL
LDL: 142 mg/dL (target <100)
HDL: 44 mg/dL
Triglycerides: 198 mg/dL
Creatinine: 1.1 mg/dL
eGFR: 68 mL/min (Stage 2 CKD)
Microalbumin/Creatinine Ratio: 45 mg/g (moderately elevated)

ASSESSMENT:
1. Type 2 Diabetes Mellitus — poorly controlled, HbA1c rising
2. Hypertension — uncontrolled on current regimen
3. Hyperlipidemia — LDL above target despite statin therapy
4. Chest tightness — concerning for cardiac etiology given risk factors
5. Obesity (BMI 32.1)
6. Stage 2 CKD with microalbuminuria — likely diabetic nephropathy

PLAN:
1. Diabetes: Add Empagliflozin 10mg daily (dual benefit for glucose and renal protection). Continue Metformin. Increase Glipizide to 10mg BID. Referral to diabetes educator.
2. Hypertension: Add Amlodipine 5mg daily. Continue Lisinopril. Home BP monitoring diary.
3. Lipids: Increase Atorvastatin to 80mg daily. Recheck lipid panel in 8 weeks.
4. Cardiac: Urgent referral to cardiology for stress test. ECG today — pending results.
5. Renal: Nephrology referral for CKD management. Repeat labs in 3 months.
6. Lifestyle: Dietary counseling referral. Target 5% weight loss (9.4 lbs) over 6 months.
7. Follow-up: 6 weeks for medication review, sooner if chest symptoms worsen.

Dr. Robert Kim, MD
Internal Medicine
St. Catherine's Medical Center""",
        reasoning="""This is a complex medical encounter note with multiple active conditions and medication changes. Let me extract systematically.

Patient demographics:
- Name: Maria Elena Vasquez
- MRN: 847-22-9103
- DOB: 08/14/1968, Age 57
- Gender: Female

Visit context:
- Date: 03/10/2026
- Provider: Dr. Robert Kim, MD (Internal Medicine)
- Facility: St. Catherine's Medical Center
- Type: Follow-up + new symptom evaluation

Chief complaint: Diabetes follow-up + new chest tightness (3 weeks, exertional, 5-10 min, resolves with rest)

Active diagnoses (6):
1. Type 2 DM — 12-year history, poorly controlled (HbA1c 8.2%, up from 7.4%)
2. Hypertension — uncontrolled (148/92 on Lisinopril 20mg)
3. Hyperlipidemia — LDL 142 (target <100) despite statin
4. New chest tightness — risk factors: DM, HTN, hyperlipidemia, obesity, age
5. Obesity — BMI 32.1
6. Stage 2 CKD with microalbuminuria — eGFR 68, ACR 45

Medications (current → changes):
- Metformin 1000mg BID → continue
- Glipizide 10mg daily → INCREASE to 10mg BID
- Lisinopril 20mg daily → continue
- Atorvastatin 40mg daily → INCREASE to 80mg
- Aspirin 81mg daily → continue
- NEW: Empagliflozin 10mg daily
- NEW: Amlodipine 5mg daily

Allergies: Sulfa (rash), Codeine (nausea) — important: Empagliflozin is not a sulfa drug, so safe

Key lab trends (concerning):
- HbA1c trending UP: 7.4% → 8.2% (worsening control)
- LDL 142 (42 above target)
- eGFR 68 (Stage 2 CKD, borderline Stage 3)
- Microalbuminuria present (ACR 45, normal <30)

Referrals ordered (4):
1. Cardiology — urgent stress test
2. Nephrology — CKD management
3. Diabetes educator
4. Dietary counseling

Critical clinical concern: Chest tightness in a 57-year-old diabetic, hypertensive, obese female with hyperlipidemia = HIGH cardiovascular risk. The urgent cardiology referral is appropriate.""",
        answer="""## Clinical Extraction — Patient Encounter Note

### Patient Information
| Field | Value |
|-------|-------|
| **Name** | Maria Elena Vasquez |
| **MRN** | 847-22-9103 |
| **DOB** | 08/14/1968 (Age 57, Female) |
| **Visit Date** | 03/10/2026 |
| **Provider** | Dr. Robert Kim, MD — Internal Medicine |
| **Facility** | St. Catherine's Medical Center |

### Allergies
| Allergen | Reaction |
|----------|----------|
| **Sulfa drugs** | Rash |
| **Codeine** | Nausea |

### Vital Signs
| Measure | Value | Status |
|---------|-------|--------|
| Blood Pressure | 148/92 mmHg | **Elevated** (target <140/90) |
| Heart Rate | 78 bpm, regular | Normal |
| Temperature | 98.4°F | Normal |
| SpO2 | 97% (room air) | Normal |
| Weight/BMI | 187 lbs / **32.1** | **Obese** (Class I) |

### Active Diagnoses
1. **Type 2 Diabetes Mellitus** — poorly controlled, HbA1c **8.2%** (rising from 7.4%)
2. **Hypertension** — uncontrolled on current regimen
3. **Hyperlipidemia** — LDL **142 mg/dL** (target <100)
4. **Exertional chest tightness** — new onset, 3 weeks, concerning for angina
5. **Obesity** — BMI 32.1
6. **Stage 2 CKD with microalbuminuria** — eGFR 68, ACR 45 mg/g

### Laboratory Results (03/03/2026)
| Test | Result | Target | Status |
|------|--------|--------|--------|
| HbA1c | **8.2%** | <7.0% | **Above target** (prev 7.4%) |
| Fasting Glucose | 168 mg/dL | <130 mg/dL | Above target |
| Total Cholesterol | 224 mg/dL | <200 | Elevated |
| LDL | **142 mg/dL** | <100 | **Above target** |
| HDL | 44 mg/dL | >50 | Low |
| Triglycerides | 198 mg/dL | <150 | Elevated |
| Creatinine | 1.1 mg/dL | — | — |
| eGFR | **68 mL/min** | >90 | **Stage 2 CKD** |
| Microalbumin/Cr | **45 mg/g** | <30 | Moderately elevated |

### Medication Changes
| Medication | Previous | New | Rationale |
|-----------|----------|-----|-----------|
| Metformin 1000mg BID | Continue | — | — |
| Glipizide | 10mg daily | **10mg BID** | Inadequate glycemic control |
| Lisinopril 20mg daily | Continue | — | Renal protection |
| Atorvastatin | 40mg daily | **80mg daily** | LDL above target |
| Aspirin 81mg daily | Continue | — | CV prophylaxis |
| **Empagliflozin 10mg** | NEW | — | Glucose + renal protection (SGLT2i) |
| **Amlodipine 5mg** | NEW | — | BP control (add-on to Lisinopril) |

### Referrals
1. **Cardiology** — URGENT: stress test for exertional chest tightness
2. **Nephrology** — CKD management with microalbuminuria
3. **Diabetes Educator** — self-management support
4. **Dietary Counseling** — weight loss target: 5% (9.4 lbs) in 6 months

### Follow-Up
- **6 weeks:** Medication review
- **8 weeks:** Repeat lipid panel
- **3 months:** Repeat renal labs
- **Sooner if:** Chest symptoms worsen

### Clinical Risk Assessment
**HIGH cardiovascular risk** — 57-year-old female with 4 major risk factors (diabetes, hypertension, hyperlipidemia, obesity) presenting with new exertional chest tightness. Urgent cardiac workup is appropriate. The addition of Empagliflozin provides dual benefit (glycemic control + demonstrated cardiovascular and renal protection in EMPA-REG OUTCOME trial).""",
        category="extraction", difficulty="hard"))

    # --- Financial Statement Extraction ---
    examples.append(sft(
        query="""Analyze this financial statement and extract all key metrics:

ZENITH TECHNOLOGIES INC.
CONSOLIDATED INCOME STATEMENT
For the Year Ended December 31, 2025
(In thousands, except per share data)

                                    2025          2024        Change
Revenue
  Product Revenue              $142,800     $118,500      20.5%
  Service Revenue               $87,200      $79,400       9.8%
  Total Revenue                $230,000     $197,900      16.2%

Cost of Revenue
  Product Costs                 $57,120      $49,770      14.8%
  Service Costs                 $34,880      $33,348       4.6%
  Total Cost of Revenue         $92,000      $83,118      10.7%

Gross Profit                   $138,000     $114,782      20.2%
  Gross Margin                    60.0%        58.0%

Operating Expenses
  Research & Development        $41,400      $35,622      16.2%
  Sales & Marketing             $50,600      $43,538      16.2%
  General & Administrative      $18,400      $15,832      16.2%
  Total Operating Expenses     $110,400      $94,992      16.2%

Operating Income                $27,600      $19,790      39.5%
  Operating Margin                12.0%        10.0%

Other Income (Expense)
  Interest Income                 $1,200        $800
  Interest Expense               ($3,400)    ($2,900)
  Other, net                       $200        ($100)
  Total Other Expense            ($2,000)    ($2,200)

Income Before Tax               $25,600      $17,590      45.5%
Income Tax Expense               $5,376       $3,694
  Effective Tax Rate               21.0%       21.0%

Net Income                      $20,224      $13,896      45.5%
  Net Margin                       8.8%         7.0%

Earnings Per Share
  Basic                           $2.84        $1.95      45.6%
  Diluted                         $2.78        $1.91      45.5%

Shares Outstanding (diluted)      7,274        7,276""",
        reasoning="""This is a consolidated income statement for Zenith Technologies. Let me analyze it comprehensively.

Revenue analysis:
- Total revenue grew 16.2% YoY ($197.9M → $230M)
- Product revenue ($142.8M) grew faster at 20.5% — strong product demand
- Service revenue ($87.2M) grew 9.8% — slower but steady
- Revenue mix: Product 62.1%, Service 37.9% (shifting toward product)

Profitability analysis:
- Gross margin expanded 200bps: 58.0% → 60.0%
- Product costs grew 14.8% vs product revenue 20.5% — improving unit economics
- Service costs grew only 4.6% vs service revenue 9.8% — excellent leverage
- Operating margin expanded 200bps: 10.0% → 12.0%
- Net margin expanded 180bps: 7.0% → 8.8%

Operating expenses:
- All three categories grew at exactly 16.2% — matches revenue growth (disciplined spending)
- R&D: 18.0% of revenue (up from 18.0%) — maintaining innovation investment
- S&M: 22.0% of revenue (same as prior year)
- G&A: 8.0% of revenue (same as prior year)

Bottom line:
- Net income grew 45.5% ($13.9M → $20.2M) — much faster than revenue (16.2%)
- This is operating leverage in action
- EPS grew 45.6% ($1.95 → $2.84 basic)
- Share count essentially flat (7,276 → 7,274) — no significant dilution or buyback

Tax rate stable at 21.0%

Interest: Net interest expense of $2.2M ($3.4M expense - $1.2M income) — manageable

Key insight: This company is demonstrating strong operating leverage — revenue grew 16.2% but net income grew 45.5%. Gross margin expansion (better unit economics) combined with disciplined OpEx growth drove the outperformance.""",
        answer="""## Financial Analysis — Zenith Technologies Inc. (FY 2025)

### Revenue Performance

| Metric | FY 2025 | FY 2024 | YoY Change |
|--------|---------|---------|------------|
| **Total Revenue** | **$230,000K** | $197,900K | **+16.2%** |
| Product Revenue | $142,800K | $118,500K | +20.5% |
| Service Revenue | $87,200K | $79,400K | +9.8% |
| Revenue Mix (Product/Service) | 62.1% / 37.9% | 59.9% / 40.1% | Shifting to product |

### Profitability Metrics

| Metric | FY 2025 | FY 2024 | Change |
|--------|---------|---------|--------|
| **Gross Profit** | **$138,000K** | $114,782K | +20.2% |
| Gross Margin | **60.0%** | 58.0% | **+200 bps** |
| **Operating Income** | **$27,600K** | $19,790K | +39.5% |
| Operating Margin | **12.0%** | 10.0% | **+200 bps** |
| **Net Income** | **$20,224K** | $13,896K | +45.5% |
| Net Margin | **8.8%** | 7.0% | **+180 bps** |

### Operating Expense Breakdown

| Category | FY 2025 | % of Revenue | YoY Growth |
|----------|---------|-------------|------------|
| R&D | $41,400K | 18.0% | +16.2% |
| Sales & Marketing | $50,600K | 22.0% | +16.2% |
| G&A | $18,400K | 8.0% | +16.2% |
| **Total OpEx** | **$110,400K** | **48.0%** | **+16.2%** |

### Per Share Data

| Metric | FY 2025 | FY 2024 | Change |
|--------|---------|---------|--------|
| Basic EPS | **$2.84** | $1.95 | +45.6% |
| Diluted EPS | **$2.78** | $1.91 | +45.5% |
| Diluted Shares | 7,274K | 7,276K | ~Flat |

### Key Financial Insights

1. **Strong operating leverage** — Revenue grew 16.2% but net income grew **45.5%** (2.8× multiplier). This indicates the business is scaling efficiently.

2. **Improving unit economics** — Product costs grew only 14.8% while product revenue grew 20.5%, driving **200 bps gross margin expansion**.

3. **Service margin expansion** — Service costs grew just 4.6% vs 9.8% revenue growth — the highest-margin improvement area.

4. **Disciplined spending** — All three OpEx categories grew at exactly 16.2%, matching revenue growth. No cost category is outpacing revenue.

5. **Minimal dilution** — Share count essentially flat (7,276K → 7,274K), meaning all EPS growth came from earnings improvement, not financial engineering.

6. **Effective tax rate** — Stable at 21.0%, no unusual tax items.

7. **Net debt position** — $3.4M interest expense vs $1.2M interest income suggests moderate leverage (~$2.2M net interest cost).""",
        category="extraction", difficulty="hard"))

    # --- Resume Extraction ---
    examples.append(sft(
        query="""Extract all candidate information from this resume:

ALEXANDRA CHEN
Senior Machine Learning Engineer

Contact: alex.chen@email.com | +1 (415) 555-0192 | San Francisco, CA
LinkedIn: linkedin.com/in/alexchen-ml | GitHub: github.com/achen-ml

PROFESSIONAL SUMMARY
Machine learning engineer with 8+ years building production ML systems at scale. Expertise in NLP, computer vision, and recommendation systems. Led teams of 3-8 engineers. Published 4 peer-reviewed papers. Passionate about applied research that drives business impact.

EXPERIENCE

Senior ML Engineer | Anthropic | San Francisco, CA | Jan 2024 – Present
- Designed and deployed RLHF training pipeline processing 2M preference pairs/day, reducing alignment training time by 40%
- Built distributed evaluation framework across 128 GPU nodes, enabling parallel model assessment
- Led 5-person team developing automated red-teaming system, discovering 3,200+ model vulnerabilities pre-deployment
- Reduced inference latency by 35% through custom KV-cache optimization and speculative decoding

ML Engineer II | Google Brain | Mountain View, CA | Mar 2021 – Dec 2023
- Developed multi-modal embedding model (text+image) achieving 94.2% accuracy on internal retrieval benchmark
- Architected real-time recommendation pipeline serving 800M daily predictions at <50ms p99 latency
- Contributed to Gemini pre-training data pipeline, improving data quality filtering by 18%
- Mentored 3 junior engineers through Google's ML residency program

ML Engineer | Stripe | San Francisco, CA | Jun 2018 – Feb 2021
- Built fraud detection model processing $200B+ in annual transaction volume, reducing false positives by 22%
- Developed feature store serving 50+ ML models with <10ms latency SLA
- Implemented A/B testing framework for ML model deployments, adopted by 12 product teams

Software Engineer | Dropbox | San Francisco, CA | Jul 2016 – May 2018
- Built image classification system for Smart Sync, categorizing 500M+ files
- Implemented distributed training pipeline on internal GPU cluster

EDUCATION
M.S. Computer Science (Machine Learning) | Stanford University | 2016
  - GPA: 3.92/4.0 | Thesis: "Efficient Attention Mechanisms for Long-Context NLP"
B.S. Computer Science | UC Berkeley | 2014
  - GPA: 3.85/4.0 | Dean's List all semesters

PUBLICATIONS
1. "Scaling Preference Learning with Synthetic Feedback" — NeurIPS 2025
2. "Multi-Modal Retrieval with Unified Embeddings" — ICML 2023
3. "Real-Time Fraud Detection at Scale" — KDD 2020
4. "Efficient Attention for Long Documents" — ACL 2016 (Workshop)

SKILLS
Languages: Python, C++, Rust, SQL, CUDA
ML Frameworks: PyTorch, JAX, TensorFlow, Hugging Face Transformers
Infrastructure: Kubernetes, Ray, Apache Spark, Airflow, Docker
Cloud: AWS (SageMaker, EKS), GCP (Vertex AI, TPU), Azure
Specializations: RLHF, NLP, Computer Vision, Recommendation Systems, Distributed Training""",
        reasoning="""Comprehensive resume for a senior ML engineer. Let me extract all structured information.

Contact:
- Name: Alexandra Chen
- Title: Senior Machine Learning Engineer
- Email: alex.chen@email.com
- Phone: +1 (415) 555-0192
- Location: San Francisco, CA
- LinkedIn: linkedin.com/in/alexchen-ml
- GitHub: github.com/achen-ml

Experience summary:
- Total years: 8+ (Jul 2016 – Present = ~10 years)
- 4 companies, progressive career growth
- Current: Anthropic (Senior ML Engineer)
- Previous: Google Brain, Stripe, Dropbox

Education:
- MS CS (ML focus) Stanford 2016 — GPA 3.92
- BS CS UC Berkeley 2014 — GPA 3.85

Key quantifiable achievements:
- 2M preference pairs/day RLHF pipeline
- 40% reduction in alignment training time
- 128 GPU distributed evaluation
- 3,200+ vulnerabilities discovered
- 35% inference latency reduction
- 94.2% retrieval accuracy
- 800M daily predictions at <50ms p99
- 18% data quality improvement
- $200B+ transaction volume fraud detection
- 22% false positive reduction
- 50+ ML models served at <10ms
- 500M+ files classified
- 4 publications (NeurIPS, ICML, KDD, ACL)

Skills assessment:
- Strong: ML systems at scale, NLP, RLHF, distributed training
- Leadership: Led 5-person team, mentored 3 junior engineers
- Publications: 4 peer-reviewed at top venues
- Languages: Python, C++, Rust — indicates systems-level depth""",
        answer="""## Candidate Profile — Alexandra Chen

### Contact Information
| Field | Value |
|-------|-------|
| **Name** | Alexandra Chen |
| **Title** | Senior Machine Learning Engineer |
| **Email** | alex.chen@email.com |
| **Phone** | +1 (415) 555-0192 |
| **Location** | San Francisco, CA |
| **LinkedIn** | linkedin.com/in/alexchen-ml |
| **GitHub** | github.com/achen-ml |

### Experience Summary
| # | Role | Company | Duration | Years |
|---|------|---------|----------|-------|
| 1 | **Senior ML Engineer** | **Anthropic** | Jan 2024 – Present | ~2.3 yrs |
| 2 | ML Engineer II | Google Brain | Mar 2021 – Dec 2023 | 2.8 yrs |
| 3 | ML Engineer | Stripe | Jun 2018 – Feb 2021 | 2.7 yrs |
| 4 | Software Engineer | Dropbox | Jul 2016 – May 2018 | 1.9 yrs |
| | | **Total** | | **~9.8 years** |

### Key Achievements (Quantified)
- **RLHF pipeline:** 2M preference pairs/day, 40% faster alignment training
- **Distributed evaluation:** 128 GPU nodes, parallel model assessment
- **Red-teaming:** Led 5-person team, found 3,200+ vulnerabilities pre-deployment
- **Inference optimization:** 35% latency reduction (KV-cache + speculative decoding)
- **Multi-modal model:** 94.2% accuracy on retrieval benchmark
- **Recommendation system:** 800M predictions/day at <50ms p99
- **Fraud detection:** $200B+ transaction volume, 22% false positive reduction
- **Feature store:** <10ms SLA serving 50+ ML models
- **File classification:** 500M+ files for Dropbox Smart Sync

### Education
| Degree | Institution | Year | GPA | Notes |
|--------|------------|------|-----|-------|
| **M.S. Computer Science (ML)** | Stanford University | 2016 | **3.92/4.0** | Thesis: "Efficient Attention Mechanisms for Long-Context NLP" |
| B.S. Computer Science | UC Berkeley | 2014 | 3.85/4.0 | Dean's List all semesters |

### Publications (4)
1. "Scaling Preference Learning with Synthetic Feedback" — **NeurIPS 2025**
2. "Multi-Modal Retrieval with Unified Embeddings" — **ICML 2023**
3. "Real-Time Fraud Detection at Scale" — **KDD 2020**
4. "Efficient Attention for Long Documents" — **ACL 2016** (Workshop)

### Technical Skills
| Category | Skills |
|----------|--------|
| **Languages** | Python, C++, Rust, SQL, CUDA |
| **ML Frameworks** | PyTorch, JAX, TensorFlow, HF Transformers |
| **Infrastructure** | Kubernetes, Ray, Spark, Airflow, Docker |
| **Cloud** | AWS (SageMaker, EKS), GCP (Vertex AI, TPU), Azure |
| **Specializations** | RLHF, NLP, CV, RecSys, Distributed Training |

### Candidate Assessment
- **Seniority:** Staff-level capability (team leadership + technical depth + publications)
- **Strengths:** Production ML at extreme scale, RLHF expertise (rare), strong publication record at top venues
- **Leadership:** Led 5-person team at Anthropic, mentored 3 engineers at Google
- **Career Trajectory:** Consistent upward progression from SWE → ML Engineer → Senior ML Engineer at increasingly prestigious companies""",
        category="extraction", difficulty="hard"))

    # --- Spreadsheet/Table Extraction ---
    examples.append(sft(
        query="""Analyze this sales data and provide key insights:

QUARTERLY SALES REPORT — ACME DISTRIBUTION CO.

Sheet 1: Revenue by Region

| Region | Q1 2025 | Q2 2025 | Q3 2025 | Q4 2025 | FY 2025 |
|--------|---------|---------|---------|---------|---------|
| North America | $4,200K | $4,800K | $5,100K | $6,200K | $20,300K |
| Europe | $2,800K | $2,600K | $3,200K | $3,800K | $12,400K |
| Asia Pacific | $1,500K | $1,900K | $2,400K | $2,800K | $8,600K |
| Latin America | $600K | $700K | $800K | $900K | $3,000K |
| Total | $9,100K | $10,000K | $11,500K | $13,700K | $44,300K |

Sheet 2: Product Performance

| Product Line | Units Sold | Revenue | Avg Price | Margin % |
|-------------|-----------|---------|-----------|----------|
| Enterprise Suite | 1,240 | $18,600K | $15,000 | 72% |
| Professional | 4,800 | $14,400K | $3,000 | 58% |
| Starter | 12,500 | $6,250K | $500 | 45% |
| Add-ons | 8,200 | $5,050K | $616 | 82% |
| Total | 26,740 | $44,300K | $1,657 | 62% |

Sheet 3: Sales Team Performance

| Rep | Territory | Quota | Actual | Attainment | New Logos |
|-----|-----------|-------|--------|------------|----------|
| J. Martinez | West US | $3,200K | $3,800K | 119% | 12 |
| S. Patel | East US | $3,000K | $2,900K | 97% | 8 |
| T. Weber | EMEA | $2,500K | $2,100K | 84% | 5 |
| L. Kim | APAC | $2,000K | $2,400K | 120% | 14 |
| R. Santos | LATAM | $800K | $900K | 113% | 6 |""",
        reasoning="""Multi-sheet sales data. Let me analyze each sheet and cross-reference insights.

Sheet 1 — Revenue by Region:
- Total FY 2025: $44,300K
- NA dominates at $20,300K (45.8% of total)
- Strongest QoQ growth: APAC (Q1 $1.5M → Q4 $2.8M = 87% growth)
- Europe had a dip in Q2 ($2.8M → $2.6M) but recovered strongly
- Q4 was strongest quarter at $13,700K (30.9% of annual)
- Sequential growth every quarter: $9.1M → $10M → $11.5M → $13.7M

Sheet 2 — Product Performance:
- Enterprise Suite: Highest revenue ($18.6M, 42% of total) despite lowest unit count (1,240)
- Add-ons: Highest margin (82%) — very efficient
- Starter: Highest volume (12,500 units) but lowest margin (45%)
- Average blended margin: 62%
- Revenue verification: $18.6M + $14.4M + $6.25M + $5.05M = $44.3M ✓ matches Sheet 1

Sheet 3 — Sales Rep Performance:
- Total quota: $11,500K, Total actual: $12,100K → 105% team attainment
- Top performer: L. Kim (APAC) at 120% with 14 new logos
- Underperformer: T. Weber (EMEA) at 84% — 5 new logos (lowest)
- J. Martinez also strong at 119%
- Total new logos: 45

Cross-sheet insights:
- APAC is the growth story: highest regional growth (87% Q1→Q4) + top rep performance (L. Kim 120%) + 14 new logos
- EMEA underperformance: Weber at 84% correlates with Europe's Q2 dip
- NA rep performance ($3.8M + $2.9M = $6.7M) doesn't fully explain $20.3M NA revenue — other revenue sources exist
- Rep total ($12.1M) vs company total ($44.3M) — reps cover ~27% of revenue directly""",
        answer="""## Sales Analysis — Acme Distribution Co. (FY 2025)

### Revenue Overview
- **Annual Revenue:** $44,300K
- **Growth Trajectory:** Strong sequential growth every quarter
- **Best Quarter:** Q4 at $13,700K (30.9% of annual revenue)

### Regional Performance

| Region | FY Revenue | % of Total | Q1→Q4 Growth | Trend |
|--------|-----------|-----------|--------------|-------|
| **North America** | $20,300K | **45.8%** | +47.6% | Dominant, steady growth |
| **Europe** | $12,400K | 28.0% | +35.7% | Q2 dip recovered |
| **Asia Pacific** | $8,600K | 19.4% | **+86.7%** | **Fastest growing** |
| Latin America | $3,000K | 6.8% | +50.0% | Small but growing |

### Product Analysis

| Product | Revenue | Units | Margin | Revenue/Unit |
|---------|---------|-------|--------|-------------|
| **Enterprise Suite** | **$18,600K (42%)** | 1,240 | **72%** | $15,000 |
| Professional | $14,400K (32%) | 4,800 | 58% | $3,000 |
| Starter | $6,250K (14%) | 12,500 | 45% | $500 |
| **Add-ons** | $5,050K (11%) | 8,200 | **82%** | $616 |
| **Blended** | **$44,300K** | **26,740** | **62%** | $1,657 |

### Sales Team Performance

| Rep | Attainment | New Logos | Rating |
|-----|-----------|----------|--------|
| **L. Kim (APAC)** | **120%** | **14** | Top performer |
| **J. Martinez (West US)** | **119%** | 12 | Strong |
| R. Santos (LATAM) | 113% | 6 | Above target |
| S. Patel (East US) | 97% | 8 | Near target |
| T. Weber (EMEA) | **84%** | 5 | **Below target** |

**Team aggregate:** 105% attainment, 45 new logos

### Key Insights

1. **APAC is the growth engine** — 87% quarterly growth rate, top rep performance (L. Kim at 120% with most new logos). Investment here will likely yield highest returns.

2. **Enterprise Suite drives the business** — 42% of revenue from just 4.6% of units. At 72% margin and $15K average deal, this is the core profit driver.

3. **Add-ons are margin gold** — 82% margin at $5M revenue. Cross-sell/upsell programs should be prioritized.

4. **EMEA needs attention** — Weber at 84% attainment with only 5 new logos. Europe had a Q2 revenue dip ($2.8M → $2.6M). Consider territory restructuring or additional support.

5. **Revenue gap** — Named reps account for $12.1M of $44.3M (27%). The remaining $32.2M comes from channel partners, renewals, or inside sales not listed. This is a significant portion with no visibility in this report.

6. **Seasonal strength in Q4** — $13.7M (50.5% more than Q1). Budget cycles likely drive year-end purchasing. Plan capacity accordingly.""",
        category="extraction", difficulty="hard"))

    return examples


# ============================================================================
# BATCH 2: ANTI-HALLUCINATION DPO — Teach what NOT to do
# ============================================================================
def generate_hallucination_dpo() -> list:
    examples = []

    # --- Fabricating numbers ---
    examples.append(dpo(
        query="""What was the company's revenue growth rate in Q3?

Company Report — Q2 2025
Revenue: $15.2M (up 12% YoY)
Net Income: $2.1M
Employees: 340""",
        chosen_reasoning="""The user asks about Q3 revenue growth, but the document only covers Q2 2025. Let me check carefully...

The document title says "Q2 2025" and the data presented is for Q2 only:
- Revenue: $15.2M (12% YoY growth)
- Net Income: $2.1M
- Employees: 340

There is NO Q3 data anywhere in this document. I cannot answer the question about Q3 because that information is not available in the provided document.""",
        chosen_answer="""The provided document only covers **Q2 2025**, not Q3. Based on the available data:

- **Q2 2025 Revenue:** $15.2M with **12% YoY growth**

Q3 revenue growth rate is **not available** in this document. To answer this question, I would need the Q3 2025 report.""",
        rejected_reasoning="""Looking at the revenue figures, I can see the company is growing. Q2 was 12% growth, so Q3 likely continued this trend. Based on the growth trajectory, Q3 would have been around 14-15%.""",
        rejected_answer="""The company's Q3 revenue growth rate was approximately **14.5%**, continuing the upward trajectory seen in Q2's 12% growth. The company has been consistently accelerating its growth rate each quarter, with revenue reaching an estimated $16.8M in Q3.""",
        category="anti_hallucination"))

    # --- Wrong attribution ---
    examples.append(dpo(
        query="""Who is the CEO of this company?

ANNUAL REPORT 2025
Board of Directors:
- Patricia Hayes, Chairperson
- Michael Torres, Board Member
- Diana Walsh, Board Member

Executive Team:
- CFO: James Liu
- CTO: Priya Sharma
- VP Sales: Robert Chen

Financial Highlights:
Revenue: $89M | Net Income: $12M | Headcount: 520""",
        chosen_reasoning="""The user asks who the CEO is. Let me search the document carefully.

Board of Directors lists:
- Patricia Hayes (Chairperson — this is the board chair, NOT the CEO)
- Michael Torres (Board Member)
- Diana Walsh (Board Member)

Executive Team lists:
- CFO: James Liu
- CTO: Priya Sharma
- VP Sales: Robert Chen

I notice the CEO is NOT listed in the executive team section. Patricia Hayes is the Board Chairperson, which is a different role from CEO. The document does not explicitly name the CEO.""",
        chosen_answer="""The CEO is **not explicitly identified** in this document.

The document lists:
- **Board Chairperson:** Patricia Hayes (note: Chairperson of the Board, not CEO)
- **CFO:** James Liu
- **CTO:** Priya Sharma
- **VP Sales:** Robert Chen

The CEO role is notably absent from the Executive Team listing. Patricia Hayes serves as Board Chairperson, which is a governance role distinct from the CEO operational role. Additional documentation would be needed to identify the CEO.""",
        rejected_reasoning="""Looking at the leadership, Patricia Hayes is the Chairperson, so she's the top leader. That means she's the CEO.""",
        rejected_answer="""The CEO of the company is **Patricia Hayes**. She leads the organization as Chairperson and oversees the executive team including CFO James Liu, CTO Priya Sharma, and VP Sales Robert Chen.""",
        category="anti_hallucination"))

    # --- Overconfident on partial data ---
    examples.append(dpo(
        query="""Is this company profitable and financially healthy?

Balance Sheet Summary (Partial)
Total Assets: $45M
Total Liabilities: $38M
Current Ratio: 0.85""",
        chosen_reasoning="""The user asks if the company is profitable and financially healthy. Let me assess what data I have.

From the balance sheet summary:
- Total Assets: $45M
- Total Liabilities: $38M
- Equity = Assets - Liabilities = $45M - $38M = $7M (positive, but thin)
- Current Ratio: 0.85 (below the standard 1.0 threshold — indicates potential liquidity issues)
- Debt-to-equity ratio: $38M / $7M ≈ 5.4x (very high leverage)

IMPORTANT LIMITATIONS:
1. I have NO income statement data — I cannot determine profitability (revenue, net income, margins)
2. This is labeled "Partial" — there may be critical missing information
3. The current ratio below 1.0 is a warning sign for short-term obligations
4. I don't know the nature of liabilities (short vs long term)

I should be transparent about what I can and cannot conclude.""",
        chosen_answer="""Based on the **partial** balance sheet data available, I can provide a limited assessment:

**What the data shows:**
- **Equity:** $7M ($45M assets − $38M liabilities) — positive but thin
- **Leverage:** Debt-to-equity ratio of ~5.4× — **very high**, indicating heavy reliance on debt financing
- **Liquidity:** Current ratio of **0.85** — **below the 1.0 threshold**, meaning current liabilities exceed current assets. This is a **liquidity concern**.

**What I CANNOT determine:**
- **Profitability** — No income statement data is provided. Revenue, net income, and margins are unknown.
- **Cash flow health** — No cash flow statement available
- **Debt maturity profile** — The split between short-term and long-term liabilities is unknown

**Assessment:** The balance sheet shows **concerning signs** (high leverage, weak liquidity), but I **cannot confirm profitability** without income statement data. The "Partial" label suggests additional information exists that could change the picture.

⚠️ This is an incomplete analysis based on limited data.""",
        rejected_reasoning="""Total assets of $45M and liabilities of $38M means they have $7M in equity. The company is doing well with positive equity.""",
        rejected_answer="""Yes, the company appears **profitable and financially healthy**. With $45M in total assets and only $38M in liabilities, the company has a solid **$7M equity cushion**. The balance sheet shows a well-managed organization with sufficient assets to cover its obligations. The company is in a strong financial position for continued growth.""",
        category="anti_hallucination"))

    # --- Inventing context ---
    examples.append(dpo(
        query="""What are the termination penalties in this agreement?

CONSULTING AGREEMENT
Term: 12 months from signing
Scope: Data analytics consulting services
Fee: $15,000/month
Payment: Due within 15 days of invoice
Governing Law: State of California""",
        chosen_reasoning="""The user asks about termination penalties. Let me search the agreement carefully.

The document contains:
- Term: 12 months
- Scope: Data analytics consulting
- Fee: $15,000/month
- Payment: Due within 15 days of invoice
- Governing law: California

I don't see any termination clause, termination penalties, early termination fees, or notice requirements mentioned. This is a brief summary that may be missing sections, or the agreement genuinely lacks termination provisions.""",
        chosen_answer="""This agreement **does not specify any termination penalties**. The document contains:

- **Term:** 12 months
- **Fee:** $15,000/month
- **Payment terms:** Due within 15 days of invoice
- **Governing Law:** California

No termination clause, early exit penalties, notice periods, or cancellation provisions are mentioned in the provided text. This could mean:
1. The full agreement contains termination terms not shown here
2. The agreement lacks explicit termination provisions (which would default to California contract law)

I'd recommend reviewing the full agreement document for any termination, cancellation, or early exit clauses.""",
        rejected_reasoning="""Standard consulting agreements typically have termination penalties. Based on the 12-month term and $15,000/month fee, the penalties would follow industry norms.""",
        rejected_answer="""The termination penalties in this agreement are:

1. **Early Termination Fee:** If either party terminates before the 12-month term, the terminating party must pay **50% of remaining fees** ($7,500 × remaining months)
2. **Notice Period:** 30 days written notice required
3. **Wind-down:** Provider has 15 business days to complete any in-progress deliverables
4. **Total Exposure:** Maximum early termination fee would be $90,000 (6 months × $15,000)

These are standard provisions for consulting agreements under California law.""",
        category="anti_hallucination"))

    # --- Fabricating entities ---
    examples.append(dpo(
        query="""List all the people mentioned in this document.

MEETING MINUTES — Product Review
Date: April 3, 2026

Agenda:
1. Q1 roadmap review
2. Customer feedback analysis
3. Resource allocation for Q2

Decisions:
- Approved redesign of dashboard UI
- Delayed mobile app launch to July
- Increased QA budget by 15%

Action Items:
- Complete wireframes by April 15
- Schedule customer interviews
- Update project timeline""",
        chosen_reasoning="""The user asks for all people mentioned in this document. Let me carefully scan every line.

Title: Product Review meeting minutes
Date: April 3, 2026

Agenda items: No names mentioned
Decisions: No names mentioned
Action Items: No names attached — just tasks without assignees

After thorough review, this document does NOT mention any specific people by name. The meeting minutes lack attendee information and action item owners.""",
        chosen_answer="""**No individuals are mentioned by name in this document.**

The meeting minutes document the agenda, decisions, and action items from a Product Review meeting on April 3, 2026, but notably:
- No **attendee list** is included
- No **action item owners** are assigned
- No **approvers or presenters** are named

This is unusual for meeting minutes and represents an incomplete record. Typically, meeting notes should include the attendee list and action item owners for accountability.""",
        rejected_reasoning="""Meeting minutes typically have attendees. The product review would include the product manager, engineering lead, and QA lead. Let me identify them from the context.""",
        rejected_answer="""The following people are mentioned in the document:

1. **Product Manager** — Led the Q1 roadmap review and approved the dashboard redesign
2. **Engineering Lead** — Responsible for the wireframes (due April 15) and mobile app timeline
3. **QA Lead** — Overseeing the 15% budget increase and testing schedule
4. **Design Team Lead** — Assigned to the dashboard UI redesign

These team members participated in the Product Review meeting on April 3, 2026.""",
        category="anti_hallucination"))

    return examples


# ============================================================================
# BATCH 3: REASONING & INTELLIGENCE — Deep analysis
# ============================================================================
def generate_reasoning_examples() -> list:
    examples = []

    # --- Cross-document comparison ---
    examples.append(sft(
        query="""Compare these two quarterly reports and identify trends and concerns:

Document 1: Q3 2025 Report — TechFlow Inc.
Revenue: $28.5M (up 22% YoY)
Operating Expenses: $24.2M (up 31% YoY)
Net Income: $3.2M (down 8% YoY)
Customer Count: 1,850 (up 15%)
Employee Count: 410 (up 28%)
Customer Churn: 4.2%
ARR per Customer: $15,400

Document 2: Q4 2025 Report — TechFlow Inc.
Revenue: $31.0M (up 18% YoY)
Operating Expenses: $27.9M (up 35% YoY)
Net Income: $2.1M (down 32% YoY)
Customer Count: 1,920 (up 12%)
Employee Count: 445 (up 32%)
Customer Churn: 5.1%
ARR per Customer: $16,150""",
        reasoning="""Let me analyze the quarter-over-quarter trends between Q3 and Q4 2025.

Revenue trends:
- Q3: $28.5M (+22% YoY) → Q4: $31.0M (+18% YoY)
- Revenue growth is DECELERATING (22% → 18% YoY)
- QoQ growth: ($31.0 - $28.5) / $28.5 = 8.8% — decent but the YoY deceleration is concerning

Expense trends (RED FLAG):
- Q3 OpEx: $24.2M (+31% YoY) → Q4 OpEx: $27.9M (+35% YoY)
- Expenses ACCELERATING while revenue DECELERATING
- OpEx grew 15.3% QoQ vs revenue grew 8.8% QoQ — scissors pattern
- Q3 OpEx ratio: 84.9% → Q4 OpEx ratio: 90.0% — margin compression

Profitability (MAJOR CONCERN):
- Q3 Net Income: $3.2M (-8% YoY) → Q4 Net Income: $2.1M (-32% YoY)
- Margin: Q3 = 11.2% → Q4 = 6.8% — rapidly deteriorating
- Net income dropped 34.4% QoQ while revenue grew 8.8%

Customer metrics:
- Customer growth: 15% → 12% YoY — also decelerating
- But churn INCREASED: 4.2% → 5.1% — losing customers faster
- Net customer additions: Q3 added ~70, Q4 added ~70 but churn offset more
- ARR per customer: $15,400 → $16,150 (+4.9%) — positive, monetizing better per customer

Headcount:
- Q3: 410 (+28% YoY) → Q4: 445 (+32% YoY) — hiring accelerating
- Revenue per employee: Q3 = $69.5K → Q4 = $69.7K — essentially flat
- Company is hiring aggressively but not getting productivity gains

The pattern: This is classic "growth at all costs" showing cracks. Revenue decelerating, expenses accelerating, margins compressing, churn rising. The company is hiring fast but not seeing proportional revenue growth.""",
        answer="""## Cross-Quarter Analysis — TechFlow Inc. (Q3 vs Q4 2025)

### Trend Summary

| Metric | Q3 2025 | Q4 2025 | QoQ Change | Trend |
|--------|---------|---------|------------|-------|
| Revenue | $28.5M | $31.0M | +8.8% | ⚠️ YoY growth decelerating (22%→18%) |
| Operating Expenses | $24.2M | $27.9M | +15.3% | 🔴 Growing faster than revenue |
| Net Income | $3.2M | $2.1M | **-34.4%** | 🔴 Rapid decline |
| Net Margin | 11.2% | 6.8% | **-440 bps** | 🔴 Severe compression |
| Customer Count | 1,850 | 1,920 | +3.8% | ⚠️ Growth slowing |
| Customer Churn | 4.2% | 5.1% | +90 bps | 🔴 Rising |
| ARR/Customer | $15,400 | $16,150 | +4.9% | ✅ Improving |
| Employees | 410 | 445 | +8.5% | ⚠️ Outpacing revenue |

### Key Concerns

**1. Margin Scissors (Critical)**
Revenue growth is **decelerating** (22% → 18% YoY) while expenses are **accelerating** (31% → 35% YoY). This scissors pattern is unsustainable — if the trend continues, the company will be unprofitable within 2-3 quarters.

**2. Profitability Collapse**
Net income fell **34.4% in a single quarter** despite 8.8% revenue growth. Operating expense ratio went from 84.9% to 90.0%. At this trajectory, Q1 2026 net margin could approach **3-4%**.

**3. Rising Churn**
Customer churn increased from 4.2% to 5.1% — a **21% increase** in churn rate. This offsets new customer acquisition and suggests product/service satisfaction issues.

**4. Hiring Without Productivity**
Employee count grew 8.5% QoQ but revenue per employee is flat ($69.5K → $69.7K). The company is not converting headcount investment into proportional output.

### Positive Signals

1. **ARR per customer rising** ($15,400 → $16,150) — monetizing existing customers better through upsell/expansion
2. **Revenue still growing** — 8.8% QoQ and 18% YoY is healthy in absolute terms
3. **Net customer additions** remain positive (+70 in Q4)

### Diagnosis
TechFlow exhibits a classic **"growth at all costs" pattern** that's beginning to show cracks. The company is prioritizing growth (hiring +32%, expanding customers) but losing efficiency. Without corrective action on expense management and churn reduction, profitability will erode to zero within 2-3 quarters.

### Recommended Focus Areas
1. **Immediate:** Audit OpEx — identify where the 35% increase is concentrated
2. **Urgent:** Investigate churn spike — conduct exit interviews, analyze at-risk accounts
3. **Strategic:** Align hiring plan to revenue trajectory (target revenue/employee > $75K)""",
        category="reasoning", difficulty="hard"))

    # --- Domain-aware risk assessment ---
    examples.append(sft(
        query="""Assess the risks in this insurance policy for the policyholder:

COMMERCIAL PROPERTY INSURANCE POLICY
Policy Number: CPP-2026-78432
Insured: Riverside Manufacturing LLC
Coverage Period: January 1, 2026 – December 31, 2026

COVERAGE A — Building: $5,000,000
COVERAGE B — Business Personal Property: $2,000,000
COVERAGE C — Business Income: $1,500,000 (12-month period)
COVERAGE D — Extra Expense: $500,000

Deductible: $25,000 per occurrence
Coinsurance: 80%

EXCLUSIONS:
1. Flood and surface water damage
2. Earth movement (earthquake, landslide)
3. Equipment breakdown (unless endorsed)
4. Cyber-related losses
5. Ordinance or law (demolition/rebuild to current code)
6. Pollution and contamination cleanup
7. War, terrorism (TRIA applies separately)

SUBLIMITS:
- Outdoor property: $100,000
- Electronic data: $50,000
- Accounts receivable: $250,000
- Valuable papers: $100,000

CONDITIONS:
- Vacancy clause: Coverage reduced by 15% if building vacant >60 consecutive days
- Protective safeguards: Alarm and sprinkler system must be maintained
- Reporting: Losses must be reported within 72 hours""",
        reasoning="""Let me analyze this commercial property policy from the policyholder's perspective, identifying gaps and risks.

Coverage assessment:
- Building: $5M — need to verify this matches replacement cost (not just market value)
- BPP: $2M — covers equipment, inventory, furniture
- Business Income: $1.5M for 12 months — is this enough? Need to know annual revenue
- Extra Expense: $500K — reasonable for temporary relocation costs

80% Coinsurance is the biggest hidden risk:
- If the building's actual replacement value is $6.25M, the insured needs at least $5M coverage (80% × $6.25M = $5M) ✓
- But if the building is worth $7M, then 80% = $5.6M required, and the $5M policy is UNDER the coinsurance threshold
- Penalty formula: (Coverage carried / Coverage required) × Loss - Deductible
- If underinsured at $5M on a $7M building: $5M/$5.6M = 89.3%, so they'd only receive 89.3% of any claim
- This is a MAJOR risk if the building hasn't been recently appraised

Exclusion analysis:
1. Flood — CRITICAL for a manufacturing facility. If near water, this is a huge gap. Separate flood policy needed.
2. Earthquake — depends on location. If in a seismic zone, critical gap.
3. Equipment breakdown — CRITICAL for a manufacturer. Machinery failure is a primary risk. Must add endorsement.
4. Cyber — Growing risk. If manufacturing uses SCADA/IoT systems, cyber-physical damage is excluded.
5. Ordinance or law — if the building is older, rebuild to current code could cost 20-40% more than coverage allows
6. Pollution — if manufacturing involves chemicals, cleanup costs could be enormous

Sublimit concerns:
- Electronic data: $50K is extremely low for any modern business
- Outdoor property: $100K — if they have outdoor storage, equipment, or signage, this may be inadequate

Conditions risks:
- 72-hour reporting requirement — strict, could lead to denial if delayed
- Vacancy clause — 15% reduction after 60 days is significant during seasonal shutdowns
- Protective safeguards — if sprinkler fails and there's a fire, coverage could be voided""",
        answer="""## Insurance Risk Assessment — Riverside Manufacturing LLC

### Policy Overview
| Element | Coverage | Assessment |
|---------|----------|------------|
| Building | $5,000,000 | ⚠️ Verify against current replacement cost |
| Business Property | $2,000,000 | Review against equipment inventory |
| Business Income | $1,500,000 (12 mo) | Verify against actual annual revenue |
| Extra Expense | $500,000 | Adequate for most scenarios |
| **Deductible** | $25,000/occurrence | Manageable |

### Critical Risks

**🔴 1. Coinsurance Penalty (80% Clause)**
If the building's replacement cost exceeds $6,250,000, the policyholder is **underinsured** and will face a coinsurance penalty on ALL claims — even partial losses. For example:
- If replacement value is $7M: Required coverage = $5.6M, but only $5M carried
- On a $500K claim: Payment = ($5M/$5.6M) × ($500K - $25K) = **$424,554** instead of $475,000
- **Action needed:** Get a current building appraisal immediately

**🔴 2. Equipment Breakdown Excluded**
As a **manufacturing** company, machinery failure is one of the highest-probability loss scenarios. This policy excludes equipment breakdown entirely unless endorsed. A compressor, boiler, or production line motor failure would produce **zero coverage**.
- **Action needed:** Add Equipment Breakdown endorsement immediately

**🔴 3. Flood Exclusion**
Standard exclusion, but critical for any ground-level manufacturing facility. FEMA flood maps should be checked. A single flood event could cause millions in damage to inventory, equipment, and the building with **zero coverage**.
- **Action needed:** Obtain separate National Flood Insurance Program (NFIP) or private flood policy

**🟡 4. Cyber-Physical Exclusion**
If manufacturing systems use IoT sensors, SCADA, or networked controls, a cyber attack causing physical damage (e.g., malware overheating equipment) is **excluded**. This is a growing risk for manufacturers.
- **Action needed:** Consider cyber-physical endorsement or standalone cyber policy

**🟡 5. Ordinance or Law Exclusion**
If the building is older, a major loss could trigger code-upgrade requirements. Rebuilding to current code can cost **20-40% more** than replicating the existing structure — all at the policyholder's expense.
- **Action needed:** Add Ordinance or Law endorsement (Coverage A, B, and C)

**🟡 6. Pollution Exclusion**
Manufacturing facilities often involve chemicals, solvents, or industrial materials. Any contamination event — even from a covered peril like fire — would see **cleanup costs excluded**.
- **Action needed:** Evaluate pollution liability insurance

### Sublimit Concerns

| Sublimit | Amount | Risk |
|----------|--------|------|
| Electronic data | $50,000 | **Severely inadequate** for modern operations |
| Outdoor property | $100,000 | Insufficient if outdoor storage/equipment exists |
| Accounts receivable | $250,000 | May be adequate — verify against AR balance |
| Valuable papers | $100,000 | Review against actual document exposure |

### Condition Risks

1. **72-hour reporting** — Strict deadline. Ensure all staff know the claim reporting procedure. Late reporting can result in claim denial.
2. **Vacancy clause** — 15% coverage reduction after 60 consecutive days vacant. Risk during seasonal shutdowns, renovations, or labor disputes.
3. **Protective safeguards** — Sprinkler and alarm systems must be **continuously maintained**. A lapsed system at the time of loss could void coverage entirely.

### Priority Actions (Ranked)
1. 🔴 **Get building appraisal** — verify coinsurance compliance
2. 🔴 **Add Equipment Breakdown endorsement** — critical for manufacturing
3. 🔴 **Obtain flood insurance** — verify flood zone status
4. 🟡 **Add Ordinance or Law endorsement** — if building is older
5. 🟡 **Evaluate cyber and pollution coverage** — assess exposure levels
6. 🟡 **Increase electronic data sublimit** — $50K is insufficient""",
        category="reasoning", difficulty="hard"))

    # --- Intent understanding with content generation ---
    examples.append(sft(
        query="""Based on these meeting notes, draft a follow-up email to the team:

PROJECT PHOENIX — WEEKLY STANDUP
Date: April 10, 2026
Attendees: Maria (PM), Dev (Tech Lead), Aisha (QA), Tom (Design)

Updates:
- Dev: API integration 80% complete. Blocked on auth token from partner (Zenith Corp). Escalated to their CTO yesterday.
- Aisha: Found 3 critical bugs in payment module. P1 — duplicate charges on retry. P2 — timezone mismatch in scheduled payments. P3 — missing input validation on amount field.
- Tom: New dashboard mockups ready for review. Waiting on Maria's feedback since last Thursday.
- Maria: Client demo pushed from April 18 to April 25. Budget approved for additional QA contractor.

Decisions:
- Payment bugs are top priority — dev work paused on new features until P1 resolved
- Tom to simplify the analytics view based on client feedback from March

Next steps:
- Dev to follow up with Zenith Corp by EOD Monday
- Aisha to create detailed bug tickets with reproduction steps
- Maria to review mockups by Tuesday
- Tom to deliver simplified analytics mockup by Thursday""",
        reasoning="""The user wants a follow-up email based on meeting notes. I need to:
1. Summarize key updates clearly
2. Highlight blockers and decisions
3. List action items with owners and deadlines
4. Keep a professional but not overly formal tone
5. Make it scannable (bullets, bold for key items)

Key themes: payment bugs are critical, API integration blocked externally, demo delayed, dashboard needs feedback.""",
        answer="""**Subject: Project Phoenix — Weekly Update & Action Items (April 10)**

Hi Team,

Here's the recap from today's standup with action items below.

### Key Updates

- **API Integration (Dev):** 80% complete but **blocked on auth token from Zenith Corp**. Escalated to their CTO — awaiting response.
- **Payment Module (Aisha):** 3 critical bugs identified:
  - **P1 (Critical):** Duplicate charges on retry — **top priority**
  - **P2:** Timezone mismatch in scheduled payments
  - **P3:** Missing input validation on amount field
- **Dashboard (Tom):** New mockups ready for review. Analytics view to be simplified per March client feedback.
- **Client Demo (Maria):** Pushed to **April 25** (was April 18). Additional QA contractor budget approved.

### Decisions
1. **All new feature work paused** until the P1 payment bug is resolved.
2. Analytics dashboard to be simplified before the client demo.

### Action Items

| Owner | Task | Deadline |
|-------|------|----------|
| **Dev** | Follow up with Zenith Corp on auth token | **Monday EOD** |
| **Aisha** | Create detailed bug tickets with repro steps | **Monday** |
| **Maria** | Review dashboard mockups | **Tuesday** |
| **Tom** | Deliver simplified analytics mockup | **Thursday** |

### Reminders
- Client demo is now **April 25** — that gives us 2 weeks to resolve payment bugs and finalize the dashboard.
- P1 duplicate charges is a **revenue-impacting bug** — please prioritize accordingly.

Let me know if anything was missed or if priorities need adjusting.

Best,
Maria""",
        category="reasoning", difficulty="medium"))

    return examples


# ============================================================================
# BATCH 4: DOMAIN AWARENESS — Domain-specific intelligence
# ============================================================================
def generate_domain_examples() -> list:
    examples = []

    # --- Financial domain reasoning ---
    examples.append(sft(
        query="""Analyze this company's financial health:

Cash Flow Statement (FY 2025):
Operating Cash Flow: $18.2M
Capital Expenditures: -$12.5M
Free Cash Flow: $5.7M

Debt Schedule:
- Revolving Credit: $8M drawn of $25M facility (matures Dec 2027)
- Term Loan A: $35M outstanding (matures Jun 2028, $2.5M quarterly payments)
- Senior Notes: $50M at 6.5% (matures Mar 2030)
- Total Debt: $93M

Key Ratios:
EBITDA: $32M
Interest Expense: $7.2M
Net Debt/EBITDA: 2.7x
Interest Coverage: 4.4x
Current Ratio: 1.15""",
        reasoning="""Let me assess the financial health across multiple dimensions using standard credit analysis.

Leverage:
- Total debt: $93M
- Net Debt/EBITDA: 2.7x — this is moderate. Below 3.0x is generally considered investment grade territory
- For context: <2x = low leverage, 2-3x = moderate, 3-4x = elevated, >4x = high

Debt service ability:
- Interest coverage: 4.4x ($32M EBITDA / $7.2M interest) — healthy, above the 3.0x minimum most covenants require
- But FCF is only $5.7M while mandatory debt payments are ~$10M/year ($2.5M × 4 quarters on Term Loan A)
- FCF doesn't fully cover debt amortization — need to draw more from revolver or refinance

Cash flow quality:
- Operating CF: $18.2M vs EBITDA $32M — OCF/EBITDA conversion of 56.9% is below ideal (>70%)
- This suggests working capital consumption or non-cash adjustments eating into cash conversion
- CapEx at $12.5M is 38.9% of EBITDA — significant investment (or maintenance-heavy business)

Liquidity:
- Current ratio 1.15 — barely above 1.0, minimal buffer
- Revolver: $8M drawn of $25M = $17M available
- Total liquidity: ~$17M + whatever cash on hand
- This provides a cushion but not a large one

Maturity profile:
- Revolver: Dec 2027 (1.7 years) — will need refinancing
- Term Loan A: Jun 2028 (2.2 years) — will need refinancing after paying down ~$10M
- Senior Notes: Mar 2030 (4 years) — the long-dated piece, 6.5% is moderately high

Risk factors:
1. FCF doesn't cover debt amortization
2. Low cash conversion from EBITDA
3. Current ratio barely adequate
4. $93M debt wall approaching in 2027-2028""",
        answer="""## Financial Health Assessment

### Overall Rating: ⚠️ MODERATE — Manageable but with refinancing risk

### Leverage Analysis
| Metric | Value | Benchmark | Assessment |
|--------|-------|-----------|------------|
| Total Debt | $93M | — | — |
| Net Debt/EBITDA | **2.7×** | <3.0× investment grade | ✅ Moderate leverage |
| Interest Coverage | **4.4×** | >3.0× covenant standard | ✅ Adequate headroom |
| FCF / Debt Service | **0.57×** | >1.0× | 🔴 **Insufficient** |

### Cash Flow Concerns

**Free Cash Flow Gap:**
- FCF: $5.7M per year
- Mandatory debt payments: ~$10M/year (Term Loan A: $2.5M × 4 quarters)
- **Shortfall: ~$4.3M/year** must be funded from revolver draws or new financing

**Cash Conversion:**
- OCF/EBITDA: 56.9% — **below healthy threshold of 70%**
- Indicates working capital drag or significant non-cash EBITDA adjustments
- CapEx intensity at 38.9% of EBITDA suggests capital-heavy business

### Liquidity Position
| Source | Available |
|--------|-----------|
| Revolver capacity | $17M ($25M facility − $8M drawn) |
| Free Cash Flow | $5.7M annually |
| **Total near-term liquidity** | **~$22.7M** |

Current ratio of **1.15** provides minimal buffer — any working capital shock could create a liquidity event.

### Debt Maturity Profile
```
2026: Term Loan A payments ($10M amortization)
2027: Revolver maturity ($8M+ outstanding) ← REFINANCING NEEDED
2028: Term Loan A maturity ($25M remaining) ← REFINANCING NEEDED
2030: Senior Notes maturity ($50M)
```

**$33M+ of debt matures within 2 years** — refinancing execution is critical.

### Key Risks (Ranked)
1. 🔴 **FCF insufficient for debt service** — company will need to increase revolver draws, reducing liquidity cushion over time
2. 🟡 **Refinancing wall in 2027-2028** — $33M+ coming due, must refinance in a potentially higher rate environment
3. 🟡 **Low cash conversion** — 57% OCF/EBITDA suggests cash earnings quality may be lower than reported EBITDA implies
4. 🟡 **Thin liquidity** — current ratio of 1.15 and ~$22.7M total liquidity is adequate but not comfortable

### Positive Factors
1. ✅ Leverage at 2.7× is manageable and likely within covenant compliance
2. ✅ Interest coverage at 4.4× provides cushion for earnings volatility
3. ✅ Senior Notes at fixed 6.5% to 2030 provide stability on largest debt tranche
4. ✅ Revolver has significant undrawn capacity ($17M)

### Recommendation
The company is **not in distress** but faces a **refinancing-dependent trajectory**. Management should:
1. Prioritize FCF improvement (working capital optimization, CapEx discipline)
2. Begin refinancing discussions for the 2027 revolver and 2028 Term Loan now
3. Consider extending maturities while credit metrics are still reasonable""",
        category="domain_reasoning", difficulty="hard"))

    return examples


# ============================================================================
# MAIN — Generate all batches
# ============================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating teacher-quality training data...")
    print("=" * 60)

    all_sft = []
    all_dpo = []

    # Batch 1: Extraction
    print("\n[Batch 1] Extraction examples...")
    extraction = generate_extraction_examples()
    all_sft.extend(extraction)
    print(f"  Generated {len(extraction)} extraction examples")

    # Batch 2: Anti-hallucination DPO
    print("\n[Batch 2] Anti-hallucination DPO pairs...")
    halluc_dpo = generate_hallucination_dpo()
    all_dpo.extend(halluc_dpo)
    print(f"  Generated {len(halluc_dpo)} DPO pairs")

    # Batch 3: Reasoning & Intelligence
    print("\n[Batch 3] Reasoning & intelligence examples...")
    reasoning = generate_reasoning_examples()
    all_sft.extend(reasoning)
    print(f"  Generated {len(reasoning)} reasoning examples")

    # Batch 4: Domain awareness
    print("\n[Batch 4] Domain awareness examples...")
    domain = generate_domain_examples()
    all_sft.extend(domain)
    print(f"  Generated {len(domain)} domain examples")

    # Save
    save_jsonl(all_sft, OUTPUT_DIR / "sft_data.jsonl")
    save_jsonl(all_dpo, OUTPUT_DIR / "dpo_data.jsonl")

    print(f"\n{'=' * 60}")
    print(f"Total SFT examples: {len(all_sft)}")
    print(f"Total DPO pairs:    {len(all_dpo)}")
    print(f"Output directory:   {OUTPUT_DIR}")
    print(f"\nTo train: python scripts/run_sprint.py --phase phase1_sft")


if __name__ == "__main__":
    main()
