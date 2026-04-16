#!/usr/bin/env python3
"""
Generate 50 diverse extraction SFT examples for DocWain.
Each example has unique document content, thorough reasoning, and structured answers.
"""

import json
from pathlib import Path

SYSTEM = (
    "You are DocWain, an enterprise document intelligence assistant. "
    "You analyse documents with deep contextual understanding, extract "
    "structured information, identify patterns and anomalies, and provide "
    "holistic analysis grounded in evidence. You reason step-by-step before "
    "answering, state your confidence level, and cite specific sources. "
    "When information is insufficient, you say so clearly rather than guessing."
)

OUTPUT_PATH = Path("finetune_artifacts/sprint/claude_quality/extraction_batch.jsonl")


def sft(query: str, reasoning: str, answer: str, difficulty: str = "medium") -> dict:
    text = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{answer}<|im_end|>"
    )
    return {"text": text, "category": "extraction", "difficulty": difficulty, "source": "claude_quality"}


EXAMPLES = []

# ─── 1. Invoice ───────────────────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract all key financial data from this invoice:

INVOICE
Meridian Office Supplies Ltd
14 Blackfriars Road, London SE1 8EG
VAT Reg: GB 294 7831 56

Bill To:
Hargreaves & Pemberton LLP
One Canada Square, Canary Wharf
London E14 5AB

Invoice No: INV-2024-08847
Invoice Date: 14 March 2024
Due Date: 13 April 2024
PO Reference: HPL-PO-2024-0312

Description                          Qty    Unit Price    Total
─────────────────────────────────────────────────────────────
A4 Copy Paper (80gsm, Box 5 reams)    20     £18.50       £370.00
Pilot G2 Gel Pens (Box 12)           15     £12.40       £186.00
Staedtler Whiteboard Markers (Set 4)  30      £6.75       £202.50
Fellowes Shredder Bags (100pk)         8     £22.00       £176.00
HP 305A Toner Cartridge (Black)        5     £89.99       £449.95
Leitz Lever Arch Files (10pk, A4)     12     £31.50       £378.00

                                        Subtotal:    £1,762.45
                                        VAT (20%):     £352.49
                                        TOTAL DUE:   £2,114.94

Payment Terms: Net 30. BACS preferred.
Sort Code: 20-45-73  Account: 83947621
Reference: INV-2024-08847

Overdue accounts subject to 2% monthly interest per Late Payment Act 1998.""",
    reasoning="""I need to extract all key financial data from this UK office supplies invoice. Let me work through it systematically.

Vendor information: Meridian Office Supplies Ltd, 14 Blackfriars Road, London SE1 8EG, VAT Reg GB 294 7831 56.

Buyer information: Hargreaves & Pemberton LLP, One Canada Square, Canary Wharf, London E14 5AB.

Invoice identifiers: Invoice No INV-2024-08847, Invoice Date 14 March 2024, Due Date 13 April 2024 (30-day terms confirmed), PO Reference HPL-PO-2024-0312.

Line items — let me verify each total:
1. A4 Copy Paper: 20 × £18.50 = £370.00 ✓
2. Pilot G2 Gel Pens: 15 × £12.40 = £186.00 ✓
3. Whiteboard Markers: 30 × £6.75 = £202.50 ✓
4. Shredder Bags: 8 × £22.00 = £176.00 ✓
5. HP Toner: 5 × £89.99 = £449.95 ✓
6. Lever Arch Files: 12 × £31.50 = £378.00 ✓

Subtotal check: 370.00 + 186.00 + 202.50 + 176.00 + 449.95 + 378.00 = £1,762.45 ✓

VAT at 20%: £1,762.45 × 0.20 = £352.49 ✓

Total: £1,762.45 + £352.49 = £2,114.94 ✓

All arithmetic checks out. Payment is BACS, sort code 20-45-73, account 83947621. Late payment clause invokes 2% monthly interest under UK Late Payment Act 1998.

Highest-value line item is HP 305A Toner at £449.95 (25.5% of subtotal). No discounts applied. No partial payments indicated. Confidence: high.""",
    answer="""## Invoice Extraction — INV-2024-08847

### Document Parties
| Role | Details |
|------|---------|
| **Vendor** | Meridian Office Supplies Ltd, 14 Blackfriars Road, London SE1 8EG |
| **VAT Registration** | GB 294 7831 56 |
| **Buyer** | Hargreaves & Pemberton LLP, One Canada Square, Canary Wharf, London E14 5AB |

### Invoice Header
| Field | Value |
|-------|-------|
| **Invoice Number** | INV-2024-08847 |
| **Invoice Date** | 14 March 2024 |
| **Due Date** | 13 April 2024 |
| **PO Reference** | HPL-PO-2024-0312 |
| **Payment Terms** | Net 30 |

### Line Items
| # | Description | Qty | Unit Price | Line Total |
|---|-------------|-----|-----------|-----------|
| 1 | A4 Copy Paper 80gsm (Box 5 reams) | 20 | £18.50 | £370.00 |
| 2 | Pilot G2 Gel Pens (Box 12) | 15 | £12.40 | £186.00 |
| 3 | Staedtler Whiteboard Markers (Set 4) | 30 | £6.75 | £202.50 |
| 4 | Fellowes Shredder Bags (100pk) | 8 | £22.00 | £176.00 |
| 5 | HP 305A Toner Cartridge (Black) | 5 | £89.99 | **£449.95** |
| 6 | Leitz Lever Arch Files (10pk, A4) | 12 | £31.50 | £378.00 |

### Financial Summary
| | Amount |
|--|--------|
| **Subtotal (ex-VAT)** | £1,762.45 |
| **VAT (20%)** | £352.49 |
| **Total Due** | **£2,114.94** |

### Payment Details
- **Method:** BACS preferred
- **Sort Code:** 20-45-73
- **Account Number:** 83947621
- **Payment Reference:** INV-2024-08847
- **Late Payment:** 2% monthly interest (Late Payment of Commercial Debts Act 1998)

### Key Observations
- **Highest-value line:** HP 305A Toner at £449.95 (25.5% of subtotal)
- **Arithmetic verification:** All line totals, subtotal, VAT and grand total confirmed correct
- **No discounts** applied; no partial payments recorded

**Confidence: High** — all fields clearly stated and all arithmetic verified.""",
    difficulty="medium"
))

# ─── 2. Purchase Order ────────────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract structured data from this purchase order:

PURCHASE ORDER
─────────────────────────────────────────────────────
Thornbury Manufacturing Group
456 Industrial Estate, Sheffield S9 2WX
Tel: 0114 276 8800  |  procurement@thornbury-mfg.co.uk

PO NUMBER: TMG-PO-2024-4471
DATE ISSUED: 22 January 2024
DELIVERY REQUIRED BY: 12 February 2024

SUPPLIER:
Apex Pneumatics Ltd
Unit 7, Rother Valley Business Park
Rotherham S60 1BX
Account Ref: APXP-0044

SHIP TO:
Thornbury MFG — Factory 3, Bay 12
456 Industrial Estate, Sheffield S9 2WX
Attention: Goods Inwards — David Cartwright

LINE  PART NUMBER    DESCRIPTION                      QTY   UOM   UNIT COST   EXTENDED
 01   AP-FCV-32-SS   Flow Control Valve 32mm SS        24    EA     £142.00    £3,408.00
 02   AP-PRV-G12     Pressure Relief Valve G1/2"       12    EA     £218.50    £2,622.00
 03   AP-CON-QF08    Quick-Fit Connector 8mm (100pk)    5   PKG     £67.30      £336.50
 04   AP-TBG-PU-6    PU Tubing 6mm OD (50m roll)       10   ROL     £38.75      £387.50
 05   AP-FLT-40M     Inline Filter 40 Micron            8    EA     £94.00      £752.00

                                        ORDER TOTAL:    £7,506.00

TERMS & CONDITIONS:
- Payment: Net 45 from invoice date
- Delivery: DDP Sheffield (supplier bears freight + duty)
- Warranty: 12 months parts and labour
- Partial deliveries not accepted without prior approval
- This PO constitutes a binding contract upon supplier acknowledgement

Authorised by: J. Bhattacharya, Head of Procurement
Signature on file  |  Date: 22 Jan 2024""",
    reasoning="""This is an industrial purchase order from Thornbury Manufacturing to Apex Pneumatics. I need to extract all structured data.

Issuer: Thornbury Manufacturing Group, Sheffield S9 2WX. Supplier: Apex Pneumatics Ltd, Rotherham S60 1BX, account ref APXP-0044.

Key dates: PO issued 22 January 2024, delivery required by 12 February 2024 — that's 21 days, a tight lead time for industrial components.

Ship-to contact: David Cartwright, Goods Inwards, Factory 3 Bay 12.

Line item verification:
Line 01: AP-FCV-32-SS, 24 × £142.00 = £3,408.00 ✓
Line 02: AP-PRV-G12, 12 × £218.50 = £2,622.00 ✓
Line 03: AP-CON-QF08, 5 × £67.30 = £336.50 ✓
Line 04: AP-TBG-PU-6, 10 × £38.75 = £387.50 ✓
Line 05: AP-FLT-40M, 8 × £94.00 = £752.00 ✓

Sum: 3408 + 2622 + 336.50 + 387.50 + 752 = £7,506.00 ✓ matches stated total.

All items are pneumatic/fluid control components — consistent product category.

Terms: Net 45, DDP Sheffield (so supplier bears freight risk), 12-month warranty, no partial deliveries without approval.

Authorised by J. Bhattacharya. No VAT shown — this may be a pre-VAT PO or VAT handled at invoice stage.

Confidence: High.""",
    answer="""## Purchase Order Extraction — TMG-PO-2024-4471

### Header Information
| Field | Value |
|-------|-------|
| **PO Number** | TMG-PO-2024-4471 |
| **Date Issued** | 22 January 2024 |
| **Delivery Required By** | 12 February 2024 |
| **Lead Time** | 21 days |

### Parties
| Role | Details |
|------|---------|
| **Buyer** | Thornbury Manufacturing Group, 456 Industrial Estate, Sheffield S9 2WX |
| **Supplier** | Apex Pneumatics Ltd, Unit 7, Rother Valley Business Park, Rotherham S60 1BX |
| **Supplier Account Ref** | APXP-0044 |
| **Delivery Contact** | David Cartwright, Goods Inwards, Factory 3 Bay 12 |

### Line Items
| Line | Part Number | Description | Qty | UOM | Unit Cost | Extended |
|------|-------------|-------------|-----|-----|-----------|----------|
| 01 | AP-FCV-32-SS | Flow Control Valve 32mm SS | 24 | EA | £142.00 | £3,408.00 |
| 02 | AP-PRV-G12 | Pressure Relief Valve G1/2" | 12 | EA | £218.50 | £2,622.00 |
| 03 | AP-CON-QF08 | Quick-Fit Connector 8mm (100pk) | 5 | PKG | £67.30 | £336.50 |
| 04 | AP-TBG-PU-6 | PU Tubing 6mm OD (50m roll) | 10 | ROL | £38.75 | £387.50 |
| 05 | AP-FLT-40M | Inline Filter 40 Micron | 8 | EA | £94.00 | £752.00 |
| | | | | **ORDER TOTAL** | | **£7,506.00** |

### Commercial Terms
| Term | Detail |
|------|--------|
| **Payment** | Net 45 from invoice date |
| **Incoterm** | DDP Sheffield (supplier pays freight + duty) |
| **Warranty** | 12 months parts and labour |
| **Partial Delivery** | Not accepted without prior approval |
| **Contract Formation** | Binding upon supplier acknowledgement |

### Authorisation
- **Authorised by:** J. Bhattacharya, Head of Procurement
- **Date:** 22 January 2024

### Key Observations
- **Highest-value line:** Flow Control Valves (Line 01) at £3,408.00 — 45.4% of order value
- **Product category:** Pneumatic/fluid control components (consistent set)
- **No VAT stated** — likely calculated at invoice stage
- **Arithmetic verified:** All line totals and order total correct

**Confidence: High**""",
    difficulty="medium"
))

# ─── 3. Contract ──────────────────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract the key terms and obligations from this contract excerpt:

SERVICE AGREEMENT

This Service Agreement ("Agreement") is entered into as of 1 April 2024 ("Effective Date")
between:

Castellan Data Systems Ltd, a company registered in England and Wales (Company No. 09847231),
with its registered office at 200 Aldersgate Street, London EC1A 4HD ("Service Provider")

and

Nexford Insurance Group plc, Company No. 02193847, 8 Finsbury Square, London EC2A 1AF ("Client")

1. SERVICES
Service Provider shall deliver: (a) real-time data enrichment API (SLA: 99.5% uptime,
<200ms p95 latency); (b) monthly data quality reports; (c) dedicated technical account
management (minimum 4 hrs/month).

2. TERM
Initial term: 24 months from Effective Date. Auto-renews for successive 12-month periods
unless either party gives 90-day written notice prior to renewal date.

3. FEES
Monthly retainer: £28,500 (invoiced on 1st of each month).
Overage: £0.0018 per API call exceeding 10,000,000 calls/month.
Annual CPI uplift capped at 4%.

4. LIABILITY
Aggregate liability capped at 12 months' fees (£342,000). Consequential loss excluded.
SLA breach: service credits of 5% monthly fee per hour of downtime beyond 99.5% threshold,
capped at 20% of monthly fee.

5. DATA PROTECTION
Both parties act as independent data controllers. DPA 2018 / UK GDPR compliant.
Data Processing Agreement executed separately (Annex B).

6. TERMINATION FOR CAUSE
Either party may terminate with 30 days' notice if material breach unremedied after
14-day cure period. Immediate termination rights on insolvency.""",
    reasoning="""This is a B2B services agreement between Castellan Data Systems (provider) and Nexford Insurance Group (client). I need to extract all key commercial and legal terms.

Parties: Service Provider is Castellan Data Systems Ltd (Co. No. 09847231, 200 Aldersgate Street EC1A 4HD). Client is Nexford Insurance Group plc (Co. No. 02193847, 8 Finsbury Square EC2A 1AF).

Effective Date: 1 April 2024.

Services: Three deliverables — (a) real-time data enrichment API with 99.5% uptime SLA and sub-200ms p95 latency commitment; (b) monthly data quality reports; (c) TAM access minimum 4 hrs/month.

Term: 24-month initial, auto-renews 12-month rolling unless 90 days notice given. So initial expiry would be 31 March 2026.

Fees: £28,500/month retainer. Annual contract value = £28,500 × 12 = £342,000. Overage at £0.0018/call over 10M calls/month. CPI uplift capped 4% annually.

Liability cap: 12 months' fees = £342,000. This matches the annual value — a standard 1× ACV cap. Consequential loss excluded. SLA credits: 5% monthly fee (£1,425) per hour of downtime past threshold, capped at 20% of monthly fee (£5,700).

Data protection: both independent controllers, DPA 2018/UK GDPR, separate DPA as Annex B.

Termination for cause: 30 days notice after 14-day unremedied breach cure period. Instant termination on insolvency.

No IP terms or confidentiality clause visible in this excerpt — those may be in omitted sections.

Confidence: High for what's present; partial document.""",
    answer="""## Contract Extraction — Castellan / Nexford Service Agreement

### Parties
| Role | Entity | Company No. | Address |
|------|--------|-------------|---------|
| **Service Provider** | Castellan Data Systems Ltd | 09847231 | 200 Aldersgate Street, London EC1A 4HD |
| **Client** | Nexford Insurance Group plc | 02193847 | 8 Finsbury Square, London EC2A 1AF |

### Key Dates
| Milestone | Date |
|-----------|------|
| **Effective Date** | 1 April 2024 |
| **Initial Term Expiry** | 31 March 2026 |
| **Auto-Renewal Period** | 12 months rolling |
| **Renewal Notice Required** | 90 days before renewal date |

### Services & SLAs
| Service | Commitment |
|---------|-----------|
| Data Enrichment API | **99.5% uptime**, <200ms p95 latency |
| Data Quality Reports | Monthly |
| Technical Account Management | Minimum 4 hrs/month |

### Commercial Terms
| Item | Value |
|------|-------|
| **Monthly Retainer** | £28,500 |
| **Annual Contract Value** | £342,000 |
| **API Overage Rate** | £0.0018 per call >10M/month |
| **Annual CPI Uplift Cap** | 4% |

### Liability Framework
| Item | Value |
|------|-------|
| **Aggregate Liability Cap** | £342,000 (12 months' fees) |
| **Consequential Loss** | Excluded |
| **SLA Credit Rate** | 5% monthly fee (£1,425) per hour of excess downtime |
| **SLA Credit Cap** | 20% of monthly fee (£5,700/month) |

### Termination Rights
| Trigger | Notice Period |
|---------|--------------|
| Material breach (unremedied) | 30 days after 14-day cure period |
| Insolvency | Immediate |
| Convenience (renewal) | 90 days before renewal date |

### Data Protection
- Both parties: **independent data controllers**
- Compliance: DPA 2018 / UK GDPR
- Separate Data Processing Agreement: **Annex B**

### Key Observations
- Liability cap equals exactly 1× ACV — standard commercial structure
- SLA at 99.5% permits ~3.65 hrs downtime/month before credits trigger
- **Note:** This is an excerpt — IP, confidentiality and dispute resolution clauses not visible

**Confidence: High** (for provisions shown); document appears to be partial""",
    difficulty="hard"
))

# ─── 4. Medical Record ────────────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract structured clinical information from this discharge summary:

DISCHARGE SUMMARY
─────────────────────────────────────────────
St. Catherine's University Hospital NHS Trust
Dept of Cardiology, Ward 7B

Patient: HARTLEY, Margaret Anne
DOB: 03/09/1951 (Age 72)        NHS No: 432 876 5190
Admission Date: 08 Feb 2024     Discharge Date: 14 Feb 2024
Consultant: Dr R. Okonkwo (Cardiology)
GP: Dr S. Patel, Westgate Surgery, Bristol BS3 4LT

PRESENTING COMPLAINT:
Acute onset chest pain, diaphoresis and dyspnoea commencing at 07:30 on 08/02/2024.

DIAGNOSIS:
Primary: Non-ST Elevation Myocardial Infarction (NSTEMI)
Secondary: Type 2 Diabetes Mellitus (established), Hypertension (established)

INVESTIGATIONS:
- Troponin I (on admission): 1,842 ng/L [ref <52 ng/L] — markedly elevated
- Troponin I (6hr serial): 3,210 ng/L — rising pattern confirmed MI
- ECG: ST depression leads V4-V6, T-wave inversion I, aVL
- Echo (09/02): LVEF 48% (mildly reduced), hypokinesis mid-anterior wall
- Coronary angiogram (10/02): 85% stenosis LAD (proximal), 40% RCA (non-obstructive)

PROCEDURES:
- PCI to LAD with drug-eluting stent (Xience Sierra 3.5×28mm) — 10/02/2024
- Post-PCI TIMI-3 flow achieved. No complications.

MEDICATIONS ON DISCHARGE:
1. Aspirin 75mg OD (lifelong)
2. Ticagrelor 90mg BD (12 months minimum — DAPT)
3. Atorvastatin 80mg nocte (lifelong)
4. Ramipril 5mg OD (titrate to 10mg)
5. Bisoprolol 2.5mg OD (titrate)
6. Metformin 1g BD (pre-existing, continued)
7. GTN spray PRN

FOLLOW-UP:
- Cardiology OPD: 6 weeks (Dr Okonkwo's team)
- Cardiac Rehab referral made
- GP review: 2 weeks (medication titration)
- Echo at 3 months (reassess LVEF)

DRIVING: Patient advised not to drive for 4 weeks (DVLA guidance post-MI).""",
    reasoning="""This is an NHS cardiology discharge summary for a 72-year-old female patient. I need to extract all structured clinical data carefully.

Patient demographics: Margaret Anne Hartley, DOB 03/09/1951 (age 72 at admission). NHS No 432 876 5190. Admitted 08/02/2024, discharged 14/02/2024 — 6-day inpatient stay.

Responsible clinician: Dr R. Okonkwo (Cardiology). GP: Dr S. Patel, Westgate Surgery, Bristol.

Diagnoses: Primary NSTEMI. Background conditions: T2DM and hypertension (both established).

Investigations — key findings:
- Troponin I on admission 1,842 ng/L (ref <52 ng/L = 35× ULN) — markedly elevated, diagnostic threshold far exceeded
- Serial troponin at 6h: 3,210 ng/L — rising pattern confirms type 1 MI (not stable elevation)
- ECG: ST depression V4-V6 and T-wave inversion I/aVL — lateral and anterior ischaemia pattern consistent with LAD territory
- Echo: LVEF 48% (mildly reduced; normal ≥55%), hypokinesis mid-anterior wall — consistent with LAD territory injury
- Angiogram: 85% LAD proximal stenosis (culprit), 40% RCA (non-obstructive, no intervention needed)

Procedure: PCI to proximal LAD with DES (Xience Sierra 3.5×28mm), TIMI-3 flow post-PCI (optimal result), no complications.

Medications: 7 items. DAPT = aspirin + ticagrelor (12 months per ACS guidelines). Atorvastatin 80mg high-intensity statin. Ramipril and bisoprolol being titrated up. Metformin continued for T2DM. GTN PRN for symptom relief.

Follow-up: 4 touchpoints — cardiology OPD 6wk, cardiac rehab referral, GP 2wk, echo 3 months. Driving restriction 4 weeks.

Confidence: High.""",
    answer="""## Clinical Extraction — Discharge Summary, Margaret Hartley

### Patient Demographics
| Field | Value |
|-------|-------|
| **Name** | Hartley, Margaret Anne |
| **Date of Birth** | 03 September 1951 (Age 72) |
| **NHS Number** | 432 876 5190 |
| **Admitting Hospital** | St. Catherine's University Hospital NHS Trust |
| **Ward** | Cardiology Ward 7B |

### Admission Details
| Field | Value |
|-------|-------|
| **Admission Date** | 08 February 2024 |
| **Discharge Date** | 14 February 2024 |
| **Length of Stay** | 6 days |
| **Consultant** | Dr R. Okonkwo (Cardiology) |
| **GP** | Dr S. Patel, Westgate Surgery, Bristol BS3 4LT |

### Diagnoses
| Type | Condition |
|------|-----------|
| **Primary (acute)** | Non-ST Elevation Myocardial Infarction (NSTEMI) |
| **Secondary** | Type 2 Diabetes Mellitus (established) |
| **Secondary** | Hypertension (established) |

### Key Investigations
| Test | Result | Reference | Interpretation |
|------|--------|-----------|----------------|
| Troponin I (admission) | 1,842 ng/L | <52 ng/L | **35× ULN — markedly elevated** |
| Troponin I (6hr serial) | 3,210 ng/L | <52 ng/L | Rising — confirms acute MI |
| ECG | ST depression V4–V6, T-inversion I/aVL | — | Anterior-lateral ischaemia |
| Echo LVEF | **48%** | ≥55% | Mildly reduced |
| Echo wall motion | Hypokinesis mid-anterior wall | — | LAD territory injury |
| Angiogram LAD | **85% stenosis (proximal)** | — | Culprit lesion — treated |
| Angiogram RCA | 40% stenosis | — | Non-obstructive — no intervention |

### Procedure
| Item | Detail |
|------|--------|
| **Procedure** | Percutaneous Coronary Intervention (PCI) |
| **Target Vessel** | Left Anterior Descending (LAD), proximal |
| **Device** | Drug-eluting stent — Xience Sierra 3.5 × 28 mm |
| **Date** | 10 February 2024 |
| **Result** | TIMI-3 flow achieved; no complications |

### Discharge Medications
| # | Drug | Dose | Frequency | Duration/Note |
|---|------|------|-----------|---------------|
| 1 | Aspirin | 75 mg | Once daily | Lifelong |
| 2 | **Ticagrelor** | 90 mg | Twice daily | 12 months minimum (DAPT) |
| 3 | Atorvastatin | 80 mg | Nocte | Lifelong (high-intensity) |
| 4 | Ramipril | 5 mg | Once daily | Titrate to 10 mg |
| 5 | Bisoprolol | 2.5 mg | Once daily | Titrate up |
| 6 | Metformin | 1 g | Twice daily | Pre-existing, continued |
| 7 | GTN spray | — | PRN | Symptom relief |

### Follow-Up Plan
| Appointment | Timeframe | Purpose |
|-------------|-----------|---------|
| GP review | 2 weeks | Medication titration |
| Cardiology OPD | 6 weeks | Dr Okonkwo's team |
| Echocardiogram | 3 months | Reassess LVEF |
| Cardiac Rehabilitation | Referral made | — |

### Restrictions
- **Driving:** No driving for **4 weeks** (DVLA post-MI guidance)

**Confidence: High** — all clinical data clearly documented""",
    difficulty="hard"
))

# ─── 5. Resume / CV ───────────────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract structured professional information from this CV:

CURRICULUM VITAE

PRIYANKA SUBRAMANIAM
priyanka.subramaniam@email.com  |  +44 7891 234567
London, UK  |  linkedin.com/in/priyankasubramaniam
Security clearance: SC (active, issued UKSV Feb 2023)

PROFESSIONAL SUMMARY
Data Engineering Lead with 9 years' experience designing high-throughput
data pipelines and lakehouses for financial services and defence clients.
Specialist in Apache Spark, Databricks and Azure ecosystem. Led teams of
up to 12 engineers. MSc Computer Science, Imperial College London.

EXPERIENCE

Senior Data Engineer → Data Engineering Lead
Palantir Technologies UK Ltd, London
March 2020 – Present (4 years 1 month)
• Architected Foundry-based data lakehouse for HMRC, processing 2.4TB/day
• Led 12-engineer team delivering £4.2M programme on time and 8% under budget
• Reduced pipeline latency from 4.2hrs to 23min (91% improvement)
• Mentored 6 junior engineers; 4 promoted within 18 months

Data Engineer
Barclays Capital, Canary Wharf, London
June 2017 – February 2020 (2 years 9 months)
• Built real-time trade surveillance pipeline using Kafka + Spark Streaming
• Processed 380,000 trade events/day with <50ms end-to-end latency
• Reduced false-positive rate in anomaly detection by 34% via feature engineering

Graduate Data Analyst
Capgemini UK, Birmingham
August 2015 – May 2017 (1 year 10 months)
• ETL development using Informatica PowerCenter for NHS Digital
• Automated 14 manual reporting workflows, saving 120 hrs/month

EDUCATION
MSc Computer Science (Distinction) — Imperial College London — 2015
BEng Electronic Engineering (First Class) — University of Birmingham — 2013

CERTIFICATIONS
- Databricks Certified Data Engineer Professional (2023)
- Azure Data Engineer Associate DP-203 (2022)
- AWS Certified Solutions Architect – Associate (2021)
- Palantir Foundry Certified Developer (2020)

TECHNICAL SKILLS
Languages: Python (expert), Scala (proficient), SQL (expert), Java (familiar)
Platforms: Databricks, Azure Data Factory, Azure Synapse, AWS Glue, Palantir Foundry
Streaming: Apache Kafka, Spark Streaming, Azure Event Hubs
Orchestration: Apache Airflow, Azure Data Factory, Prefect""",
    reasoning="""This is a comprehensive CV for Priyanka Subramaniam, a data engineering professional. I need to extract all structured information.

Personal details: email priyanka.subramaniam@email.com, phone +44 7891 234567, London UK. Security clearance: SC (active, UKSV Feb 2023).

Career progression:
1. Capgemini UK, Birmingham — Graduate Data Analyst, Aug 2015–May 2017 (1yr 10mo)
2. Barclays Capital, London — Data Engineer, Jun 2017–Feb 2020 (2yr 9mo)
3. Palantir Technologies UK — Senior DE → Lead, Mar 2020–present (4yr 1mo as of Apr 2024)

Total experience: approximately 9 years, consistent with summary claim.

Education: MSc CS Distinction, Imperial 2015; BEng Electronic Engineering First Class, Birmingham 2013. Both strong results.

Key quantified achievements:
- Led 12-engineer team, £4.2M programme, 8% under budget at Palantir
- Pipeline latency reduction: 4.2hrs → 23min = 91% improvement ✓ (matches stated)
- Barclays: 380K trade events/day, <50ms latency, 34% false-positive reduction
- Capgemini: 120 hrs/month saved from automation

Certifications: 4 active — Databricks Professional 2023, Azure DP-203 2022, AWS SA-Associate 2021, Palantir Foundry 2020.

Skills: Python/SQL expert, Scala proficient. Key platforms: Databricks, Azure, Palantir Foundry, Kafka.

Confidence: High.""",
    answer="""## CV Extraction — Priyanka Subramaniam

### Contact & Personal
| Field | Value |
|-------|-------|
| **Name** | Priyanka Subramaniam |
| **Email** | priyanka.subramaniam@email.com |
| **Phone** | +44 7891 234567 |
| **Location** | London, UK |
| **LinkedIn** | linkedin.com/in/priyankasubramaniam |
| **Security Clearance** | SC — Active (issued UKSV, February 2023) |

### Career Timeline
| # | Employer | Role | Period | Duration |
|---|----------|------|--------|----------|
| 3 | Palantir Technologies UK | Senior Data Engineer → **Data Engineering Lead** | Mar 2020–Present | 4 yr 1 mo |
| 2 | Barclays Capital | Data Engineer | Jun 2017–Feb 2020 | 2 yr 9 mo |
| 1 | Capgemini UK | Graduate Data Analyst | Aug 2015–May 2017 | 1 yr 10 mo |
| | | **Total Experience** | | **~9 years** |

### Quantified Achievements
| Employer | Achievement | Metric |
|----------|-------------|--------|
| Palantir | Team size | 12 engineers |
| Palantir | Programme value | £4.2M |
| Palantir | Budget performance | 8% under budget |
| Palantir | Pipeline latency | 4.2 hrs → 23 min (**91% reduction**) |
| Palantir | Data volume | 2.4 TB/day (HMRC lakehouse) |
| Palantir | Mentorship outcomes | 6 juniors; 4 promoted within 18 months |
| Barclays | Trade event throughput | 380,000 events/day |
| Barclays | Latency | <50 ms end-to-end |
| Barclays | False-positive reduction | 34% |
| Capgemini | Automation saving | 120 hrs/month |

### Education
| Degree | Institution | Result | Year |
|--------|-------------|--------|------|
| MSc Computer Science | Imperial College London | **Distinction** | 2015 |
| BEng Electronic Engineering | University of Birmingham | **First Class** | 2013 |

### Certifications
| Certification | Year |
|---------------|------|
| Databricks Certified Data Engineer Professional | 2023 |
| Azure Data Engineer Associate DP-203 | 2022 |
| AWS Certified Solutions Architect – Associate | 2021 |
| Palantir Foundry Certified Developer | 2020 |

### Technical Skills
| Category | Skills |
|----------|--------|
| **Languages** | Python (expert), SQL (expert), Scala (proficient), Java (familiar) |
| **Platforms** | Databricks, Azure Data Factory, Azure Synapse, AWS Glue, Palantir Foundry |
| **Streaming** | Apache Kafka, Spark Streaming, Azure Event Hubs |
| **Orchestration** | Apache Airflow, Azure Data Factory, Prefect |

**Confidence: High** — all data explicitly stated; career dates internally consistent""",
    difficulty="medium"
))

# ─── 6. Financial Statement ───────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract and analyse the key financial metrics from this income statement:

CONSOLIDATED INCOME STATEMENT
Fortis Retail Holdings plc
For the year ended 31 December 2023
(£ thousands)

                                    FY2023      FY2022      Change
Revenue
 Retail stores                    284,710     271,340        +4.9%
 Online / e-commerce               97,420      81,690       +19.3%
 Wholesale                         31,880      34,210        -6.8%
─────────────────────────────────────────────────────────────────
Total Revenue                     414,010     387,240        +6.9%

Cost of Sales                    (261,327)   (244,263)       +7.0%
─────────────────────────────────────────────────────────────────
Gross Profit                      152,683     142,977        +6.8%
Gross Margin %                      36.9%       36.9%          —

Operating Expenses
 Distribution & logistics          (38,450)    (35,820)       +7.3%
 Marketing                         (22,180)    (19,440)      +14.1%
 Administrative expenses           (41,220)    (40,890)       +0.8%
─────────────────────────────────────────────────────────────────
Total Operating Expenses          (101,850)    (96,150)       +5.9%

EBITDA                              50,833      46,827        +8.6%
Depreciation & amortisation        (14,210)    (13,580)       +4.6%
─────────────────────────────────────────────────────────────────
EBIT (Operating Profit)             36,623      33,247       +10.2%
Net Finance Costs                   (4,880)     (4,320)      +13.0%
─────────────────────────────────────────────────────────────────
Profit Before Tax                   31,743      28,927        +9.7%
Income Tax (23%)                    (7,301)     (6,653)       +9.7%
─────────────────────────────────────────────────────────────────
Profit After Tax                    24,442      22,274        +9.7%

Earnings Per Share (basic)           42.3p       38.5p        +9.9%
Dividend Per Share                   18.0p       16.5p        +9.1%""",
    reasoning="""This is an annual income statement for Fortis Retail Holdings plc, FY2023 vs FY2022. I need to extract key metrics and verify the arithmetic.

Revenue segments:
- Retail stores: £284,710K (+4.9%) — dominant segment, 68.8% of revenue
- Online/e-commerce: £97,420K (+19.3%) — fastest growing, 23.5% of revenue
- Wholesale: £31,880K (-6.8%) — declining, 7.7% of revenue
Total revenue: 284,710 + 97,420 + 31,880 = £414,010K ✓ (+6.9%)

Gross profit: £414,010 - £261,327 = £152,683K ✓
Gross margin: 152,683 / 414,010 = 36.87% ≈ 36.9% ✓ — flat YoY (same as 2022: 142,977/387,240=36.93%)

EBITDA: 152,683 - 38,450 - 22,180 - 41,220 = £50,833K ✓ — margin 50,833/414,010 = 12.3%
EBIT: 50,833 - 14,210 = £36,623K ✓ — margin 8.8%
PBT: 36,623 - 4,880 = £31,743K ✓
Tax: £31,743 × 23% = £7,301K ✓ (UK corporation tax rate 2023)
PAT: 31,743 - 7,301 = £24,442K ✓

EPS 42.3p vs 38.5p: growth 9.9% — consistent with PAT growth 9.7% (slight difference likely due to share count changes).

Key observations:
- Online channel growing fast at 19.3%, now 23.5% of revenue — channel shift positive
- Wholesale declining 6.8% — may indicate deliberate exit or demand issues
- Gross margin flat at 36.9% — pricing power maintained despite cost pressures
- Marketing spend growing 14.1% — investing behind growth
- Finance costs up 13% — possible debt refinancing at higher rates in rising rate environment

Confidence: High — all arithmetic verified.""",
    answer="""## Financial Statement Extraction — Fortis Retail Holdings plc, FY2023

### Revenue Analysis
| Segment | FY2023 (£k) | FY2022 (£k) | Growth | Revenue Mix |
|---------|------------|------------|--------|-------------|
| Retail Stores | 284,710 | 271,340 | +4.9% | 68.8% |
| Online / E-commerce | **97,420** | 81,690 | **+19.3%** | 23.5% |
| Wholesale | 31,880 | 34,210 | **-6.8%** | 7.7% |
| **Total Revenue** | **414,010** | **387,240** | **+6.9%** | 100% |

### Profit & Loss Summary
| Metric | FY2023 (£k) | FY2022 (£k) | Growth | Margin (FY23) |
|--------|------------|------------|--------|---------------|
| **Gross Profit** | 152,683 | 142,977 | +6.8% | **36.9%** |
| **EBITDA** | 50,833 | 46,827 | +8.6% | **12.3%** |
| **EBIT** | 36,623 | 33,247 | +10.2% | **8.8%** |
| **Profit Before Tax** | 31,743 | 28,927 | +9.7% | 7.7% |
| **Profit After Tax** | 24,442 | 22,274 | +9.7% | 5.9% |

### Operating Expenses Breakdown
| Expense | FY2023 (£k) | FY2022 (£k) | Growth |
|---------|------------|------------|--------|
| Distribution & Logistics | 38,450 | 35,820 | +7.3% |
| Marketing | 22,180 | 19,440 | **+14.1%** |
| Administrative | 41,220 | 40,890 | +0.8% |
| **Total OpEx** | **101,850** | **96,150** | +5.9% |

### Per Share Data
| Metric | FY2023 | FY2022 | Growth |
|--------|--------|--------|--------|
| **Earnings Per Share** | 42.3p | 38.5p | +9.9% |
| **Dividend Per Share** | 18.0p | 16.5p | +9.1% |
| **Dividend Payout Ratio** | 42.6% | 42.9% | — |

### Key Observations
- **Gross margin flat** at 36.9% YoY — pricing power maintained despite cost inflation
- **Online channel is the growth engine** at +19.3%; now 23.5% of revenue (was 21.1%)
- **Wholesale declining** (-6.8%) — may indicate strategic exit or structural demand shift
- **Marketing investment** growing ahead of revenue (+14.1% vs +6.9%) — investing in growth
- **Finance costs rising 13%** — likely higher refinancing rates in elevated interest rate environment
- **Tax rate 23%** — aligned with UK corporation tax rate effective April 2023
- **All arithmetic verified** — no discrepancies found

**Confidence: High**""",
    difficulty="hard"
))

# ─── 7. Spreadsheet / Data Table ──────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract and summarise the key data from this sales performance spreadsheet:

Q4 2023 REGIONAL SALES PERFORMANCE
Sales Operations — Extracted 31 Dec 2023

Region      | Rep Name          | Target (£) | Actuals (£) | Attainment | New Logos | Renewals
------------|-------------------|-----------|-------------|------------|-----------|--------
North West  | Gareth Williams   | 420,000    | 487,350     | 116.0%     |     8     |   14
North West  | Amara Osei        | 380,000    | 341,200     |  89.8%     |     4     |   12
North East  | Claire Buchanan   | 400,000    | 418,900     | 104.7%     |     6     |   15
North East  | Danny Kowalczyk   | 350,000    | 299,700     |  85.6%     |     3     |    9
Midlands    | Sunita Parveen    | 450,000    | 521,400     | 115.9%     |    11     |   16
Midlands    | Tom Fitzgerald    | 390,000    | 390,000     | 100.0%     |     5     |   13
South East  | Keiko Tanaka      | 510,000    | 589,200     | 115.5%     |    12     |   19
South East  | Marcus Acheampong | 460,000    | 503,780     | 109.5%     |     9     |   17
South West  | Fiona MacLeod     | 370,000    | 327,600     |  88.5%     |     3     |   11
South West  | Pradeep Nair      | 410,000    | 455,890     | 111.2%     |     7     |   15

Extract all key metrics and identify top performers, underperformers, and regional trends.""",
    reasoning="""This is a Q4 2023 sales performance table with 10 reps across 5 regions. I need to compute aggregates and identify patterns.

First, let me compute totals:
Total Target: 420+380+400+350+450+390+510+460+370+410 = £4,140,000
Total Actuals: 487,350+341,200+418,900+299,700+521,400+390,000+589,200+503,780+327,600+455,890

Let me add actuals: 487,350+341,200 = 828,550 | +418,900 = 1,247,450 | +299,700 = 1,547,150 | +521,400 = 2,068,550 | +390,000 = 2,458,550 | +589,200 = 3,047,750 | +503,780 = 3,551,530 | +327,600 = 3,879,130 | +455,890 = 4,335,020

Overall attainment: 4,335,020 / 4,140,000 = 104.7% — team beat target overall.

Total new logos: 8+4+6+3+11+5+12+9+3+7 = 68
Total renewals: 14+12+15+9+16+13+19+17+11+15 = 141

Top performers by attainment: Gareth Williams 116.0%, Sunita Parveen 115.9%, Keiko Tanaka 115.5%.

Underperformers (below 90%): Danny Kowalczyk 85.6%, Fiona MacLeod 88.5%, Amara Osei 89.8%.

By region attainment:
- North West: (487,350+341,200)/(420,000+380,000) = 828,550/800,000 = 103.6%
- North East: (418,900+299,700)/750,000 = 718,600/750,000 = 95.8%
- Midlands: (521,400+390,000)/840,000 = 911,400/840,000 = 108.5% — best region
- South East: (589,200+503,780)/970,000 = 1,092,980/970,000 = 112.7% — best by absolute value
- South West: (327,600+455,890)/780,000 = 783,490/780,000 = 100.4%

South East is strongest region (112.7% and highest absolute revenue). North East is weakest (95.8%).

New logo leaders: Keiko Tanaka 12, Sunita Parveen 11, Marcus Acheampong 9.

Confidence: High — all calculations performed from raw data.""",
    answer="""## Sales Performance Extraction — Q4 2023

### Individual Rep Performance
| Rep | Region | Target (£) | Actuals (£) | Attainment | New Logos | Renewals |
|-----|--------|-----------|-------------|------------|-----------|---------|
| **Gareth Williams** | North West | 420,000 | 487,350 | **116.0%** | 8 | 14 |
| Amara Osei | North West | 380,000 | 341,200 | 89.8% | 4 | 12 |
| Claire Buchanan | North East | 400,000 | 418,900 | 104.7% | 6 | 15 |
| Danny Kowalczyk | North East | 350,000 | 299,700 | **85.6%** | 3 | 9 |
| **Sunita Parveen** | Midlands | 450,000 | 521,400 | **115.9%** | 11 | 16 |
| Tom Fitzgerald | Midlands | 390,000 | 390,000 | 100.0% | 5 | 13 |
| **Keiko Tanaka** | South East | 510,000 | 589,200 | **115.5%** | 12 | 19 |
| Marcus Acheampong | South East | 460,000 | 503,780 | 109.5% | 9 | 17 |
| Fiona MacLeod | South West | 370,000 | 327,600 | 88.5% | 3 | 11 |
| Pradeep Nair | South West | 410,000 | 455,890 | 111.2% | 7 | 15 |

### Team Totals
| Metric | Value |
|--------|-------|
| **Total Target** | £4,140,000 |
| **Total Actuals** | **£4,335,020** |
| **Overall Attainment** | **104.7%** |
| **Total New Logos** | 68 |
| **Total Renewals** | 141 |

### Regional Summary
| Region | Target (£) | Actuals (£) | Attainment | Reps | New Logos |
|--------|-----------|-------------|------------|------|-----------|
| South East | 970,000 | 1,092,980 | **112.7%** | 2 | 21 |
| Midlands | 840,000 | 911,400 | **108.5%** | 2 | 16 |
| North West | 800,000 | 828,550 | 103.6% | 2 | 12 |
| South West | 780,000 | 783,490 | 100.4% | 2 | 10 |
| North East | 750,000 | 718,600 | **95.8%** | 2 | 9 |

### Top Performers (Attainment)
| Rank | Rep | Attainment | Excess over Target |
|------|-----|------------|-------------------|
| 1 | Gareth Williams | 116.0% | +£67,350 |
| 2 | Sunita Parveen | 115.9% | +£71,400 |
| 3 | Keiko Tanaka | 115.5% | +£79,200 |

### Underperformers (<90%)
| Rep | Region | Attainment | Shortfall |
|-----|--------|------------|-----------|
| Danny Kowalczyk | North East | 85.6% | -£50,300 |
| Fiona MacLeod | South West | 88.5% | -£42,400 |
| Amara Osei | North West | 89.8% | -£38,800 |

### New Logo Leaders
| Rep | New Logos | Region |
|-----|-----------|--------|
| Keiko Tanaka | 12 | South East |
| Sunita Parveen | 11 | Midlands |
| Marcus Acheampong | 9 | South East |

**Confidence: High** — all figures computed directly from source data""",
    difficulty="medium"
))

# ─── 8. Insurance Claim ───────────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract all claim details from this property insurance claim form:

PROPERTY INSURANCE CLAIM FORM
Policy Reference: HOM-2024-GBR-447821
Insurer: Albion Mutual Insurance Society
Date Submitted: 19 February 2024

POLICYHOLDER DETAILS
Name: Mr. Jonathan & Mrs. Sarah Whitfield
Property Address: 42 Elmwood Gardens, Guildford, Surrey GU2 7PL
Policy Period: 01 Dec 2023 – 30 Nov 2024
Sum Insured (Buildings): £485,000
Sum Insured (Contents): £85,000

INCIDENT DETAILS
Date of Incident: 14 February 2024
Time: Approx. 02:30
Cause: Burst water pipe (escape of water)
Location: First-floor airing cupboard — feed pipe to hot water cylinder fractured
Discovered by: Mrs. Sarah Whitfield at approx. 07:15
Police/Fire Report: Not applicable
Third Party Involved: No

DAMAGE DESCRIPTION
Water escaped for approximately 4.75 hours before isolation. Damage to:
- First floor: airing cupboard, hallway flooring (oak hardwood, 24 sq.m)
- Ground floor ceiling: 3 rooms affected — kitchen, dining room, hallway
- Ground floor contents: dining table, 6 chairs, sideboard (antique oak)
- Structural: plasterboard and cornice in dining room (approx. 18 sq.m)

LOSS ESTIMATE
Emergency plumber callout (Rushmore Plumbing, 14 Feb): £385.00
Temporary drying equipment hire (DryTech UK, 2 weeks): £1,240.00
Buildings repair estimate (Oakwood Restoration Ltd): £18,750.00
Contents claim:
  Dining table & chairs (replacement): £3,200.00
  Antique sideboard (specialist valuation req.): TBC
TOTAL CLAIMED (excl. TBC items): £23,575.00

EXCESS
Standard Excess: £250 (buildings), £150 (contents)

Loss adjustor appointed? Not yet.
Previous claims (last 5 years): None""",
    reasoning="""This is a property insurance claim form filed by the Whitfields with Albion Mutual, reference HOM-2024-GBR-447821. I need to extract all structured claim data.

Policyholder: Jonathan & Sarah Whitfield, 42 Elmwood Gardens, Guildford GU2 7PL. Policy in force: 1 Dec 2023–30 Nov 2024 (so incident on 14 Feb 2024 is within policy period). Buildings sum insured £485,000, contents £85,000.

Incident: burst water pipe (escape of water) in first-floor airing cupboard at approx 02:30 on 14 Feb 2024. Discovered by Mrs Whitfield at 07:15 — that's approximately 4hr 45min = 4.75 hours consistent with stated duration. No police/fire involvement, no third party.

Damage areas:
- First floor: airing cupboard + oak hardwood hallway flooring (24 sq.m)
- Ground floor ceilings: 3 rooms (kitchen, dining room, hallway)
- Ground floor contents: dining table, 6 chairs, sideboard (antique oak)
- Structural: plasterboard and cornice, dining room, ~18 sq.m

Claimed amounts:
- Emergency plumber: £385.00 (buildings/emergency)
- Drying equipment hire: £1,240.00 (2 weeks, DryTech UK)
- Buildings repair estimate: £18,750.00 (Oakwood Restoration)
- Contents - dining table/chairs: £3,200.00
- Antique sideboard: TBC (specialist valuation pending)

Total stated: £23,575 — let me verify: 385 + 1,240 + 18,750 + 3,200 = £23,575 ✓

Applicable excesses: £250 (buildings), £150 (contents). Net buildings claim before excess: £385 + £1,240 + £18,750 = £20,375, less £250 = £20,125. Net contents before excess: £3,200 + TBC, less £150.

No loss adjustor yet appointed. Clean claim history (no prior claims in 5 years).

Confidence: High for all stated items; sideboard value is pending.""",
    answer="""## Insurance Claim Extraction — HOM-2024-GBR-447821

### Policy & Policyholder
| Field | Value |
|-------|-------|
| **Policy Reference** | HOM-2024-GBR-447821 |
| **Insurer** | Albion Mutual Insurance Society |
| **Policyholder** | Mr. Jonathan & Mrs. Sarah Whitfield |
| **Property** | 42 Elmwood Gardens, Guildford, Surrey GU2 7PL |
| **Policy Period** | 01 Dec 2023 – 30 Nov 2024 |
| **Buildings Sum Insured** | £485,000 |
| **Contents Sum Insured** | £85,000 |
| **Claim Date Submitted** | 19 February 2024 |

### Incident Details
| Field | Value |
|-------|-------|
| **Date of Incident** | 14 February 2024 |
| **Time** | Approx. 02:30 |
| **Cause** | Burst water pipe — escape of water |
| **Location** | First-floor airing cupboard (feed pipe to hot water cylinder) |
| **Discovered by** | Mrs. Sarah Whitfield at approx. 07:15 |
| **Water Escape Duration** | Approx. 4 hrs 45 min |
| **Police/Fire Report** | Not applicable |
| **Third Party** | None |

### Damage Scope
| Area | Damage |
|------|--------|
| First floor — airing cupboard | Pipe fracture, water damage |
| First floor — hallway | Oak hardwood flooring, 24 sq.m |
| Ground floor ceilings | Kitchen, dining room, hallway — 3 rooms |
| Dining room structural | Plasterboard + cornice, approx. 18 sq.m |
| Contents — dining room | Table, 6 chairs, antique oak sideboard |

### Claimed Amounts
| Item | Supplier | Amount | Category |
|------|----------|--------|----------|
| Emergency plumber callout | Rushmore Plumbing (14 Feb) | £385.00 | Buildings |
| Drying equipment hire (2 weeks) | DryTech UK | £1,240.00 | Buildings |
| Buildings repair estimate | Oakwood Restoration Ltd | £18,750.00 | Buildings |
| Dining table & 6 chairs (replacement) | — | £3,200.00 | Contents |
| Antique sideboard | Specialist valuation pending | **TBC** | Contents |
| | **TOTAL (excl. TBC)** | **£23,575.00** | |

### Excess & Net Claim
| Category | Gross Claim | Excess | Net (excl. TBC) |
|----------|-------------|--------|-----------------|
| Buildings | £20,375.00 | £250 | **£20,125.00** |
| Contents | £3,200.00 + TBC | £150 | **£3,050.00 + TBC** |

### Status
- **Loss Adjustor:** Not yet appointed
- **Prior Claims (5 years):** None
- **Outstanding:** Specialist valuation of antique sideboard required before full settlement

**Confidence: High** — all stated values verified and arithmetic confirmed""",
    difficulty="medium"
))

# ─── 9. Government Form ───────────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract all information from this UK Companies House annual return data:

COMPANIES HOUSE — CONFIRMATION STATEMENT
Company Number: 12847563
Company Name: VERDANT HORIZONS CONSULTING LIMITED
Date of Confirmation Statement: 30 September 2023
Made up to date: 28 September 2023

REGISTERED OFFICE
18 St Andrew Square
Edinburgh
EH2 2BD
Scotland

COMPANY TYPE: Private company limited by shares
SIC CODE(S): 70229 — Management consultancy activities (other)
             74909 — Other professional, scientific and technical activities

DIRECTORS
Name: MCDONALD, Fiona Anne
  Appointed: 14 October 2019
  Nationality: British
  Country of residence: Scotland
  Date of Birth: **/**/1978 (month/year only by law)
  Service address: Registered office

Name: IBRAHIM, Khalid Yusuf
  Appointed: 01 March 2022
  Nationality: British
  Country of residence: England
  Date of Birth: **/**/1984
  Service address: 4 Cloth Fair, London EC1A 7JQ

SECRETARY: None appointed

PERSONS WITH SIGNIFICANT CONTROL
Name: MCDONALD, Fiona Anne
  Nature of control: Holds more than 50% but not more than 75% of shares
  Holds more than 50% but not more than 75% of voting rights
  Date registered as PSC: 14 October 2019

Name: IBRAHIM, Khalid Yusuf
  Nature of control: Holds 25% but not more than 50% of shares
  Holds 25% but not more than 50% of voting rights
  Date registered as PSC: 01 March 2022

SHARE CAPITAL
Class: Ordinary
Total number of shares: 1,000
Aggregate nominal value: £1,000 (£1.00 per share)

SHAREHOLDERS
Fiona Anne McDonald: 600 shares (60%)
Khalid Yusuf Ibrahim: 400 shares (40%)""",
    reasoning="""This is a UK Companies House Confirmation Statement (formerly Annual Return) for Verdant Horizons Consulting Limited. I need to extract all structured company, officer, and ownership data.

Company details: Company No. 12847563, private limited by shares, registered in Scotland (Edinburgh EH2 2BD). Two SIC codes: 70229 (management consultancy other) and 74909 (other professional/scientific/technical).

Confirmation statement date: 30 September 2023, made up to 28 September 2023.

Directors:
1. Fiona Anne McDonald — appointed 14 Oct 2019, British, resident Scotland, DOB 1978 (month redacted per law), service address = registered office
2. Khalid Yusuf Ibrahim — appointed 01 Mar 2022, British, resident England, DOB 1984, service address 4 Cloth Fair London EC1A 7JQ

No company secretary appointed.

PSC (Persons with Significant Control):
- Fiona McDonald: >50% but ≤75% shares and voting rights, PSC since 14 Oct 2019
- Khalid Ibrahim: 25–50% shares and voting rights, PSC since 01 Mar 2022

Share capital: 1,000 ordinary shares at £1.00 each = £1,000 nominal value.
Shareholders: McDonald 600 (60%), Ibrahim 400 (40%).

Cross-check PSC bands vs shareholding: McDonald 60% → falls in >50%–≤75% band ✓. Ibrahim 40% → falls in 25%–≤50% band ✓. Consistent.

Total shares: 600 + 400 = 1,000 ✓.

No secretary; small consulting company with two officers both acting as directors and shareholders.

Confidence: High.""",
    answer="""## Companies House Confirmation Statement Extraction

### Company Overview
| Field | Value |
|-------|-------|
| **Company Number** | 12847563 |
| **Company Name** | VERDANT HORIZONS CONSULTING LIMITED |
| **Company Type** | Private company limited by shares |
| **Confirmation Date** | 30 September 2023 |
| **Made Up To** | 28 September 2023 |

### Registered Office
18 St Andrew Square, Edinburgh, EH2 2BD, Scotland

### Business Activities
| SIC Code | Description |
|----------|-------------|
| 70229 | Management consultancy activities (other) |
| 74909 | Other professional, scientific and technical activities |

### Directors
| Name | Appointed | Nationality | Residence | DOB (yr) | Service Address |
|------|-----------|-------------|-----------|-----------|-----------------|
| Fiona Anne McDonald | 14 Oct 2019 | British | Scotland | 1978 | Registered Office |
| Khalid Yusuf Ibrahim | 01 Mar 2022 | British | England | 1984 | 4 Cloth Fair, London EC1A 7JQ |

**Company Secretary:** None appointed

### Persons with Significant Control (PSC)
| Name | Share Control Band | Voting Rights Band | PSC Since |
|------|-------------------|--------------------|-----------|
| Fiona Anne McDonald | >50% – ≤75% | >50% – ≤75% | 14 Oct 2019 |
| Khalid Yusuf Ibrahim | 25% – ≤50% | 25% – ≤50% | 01 Mar 2022 |

### Share Capital & Ownership
| Shareholder | Shares Held | % Ownership | Nominal Value |
|-------------|-------------|-------------|---------------|
| Fiona Anne McDonald | 600 | **60.0%** | £600 |
| Khalid Yusuf Ibrahim | 400 | **40.0%** | £400 |
| **Total** | **1,000** | **100%** | **£1,000** |

- **Share Class:** Ordinary
- **Nominal Value per Share:** £1.00

### Verification Notes
- PSC bands consistent with stated shareholdings (McDonald 60% → >50–75% band ✓; Ibrahim 40% → 25–50% band ✓)
- Total shares reconcile: 600 + 400 = 1,000 ✓
- No discrepancies identified

**Confidence: High**""",
    difficulty="medium"
))

# ─── 10. Technical Specification ──────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract the technical specifications and requirements from this document:

TECHNICAL SPECIFICATION
Project: Northgate Data Centre Cooling Upgrade
Document Ref: NDC-MECH-SPEC-2024-003 Rev B
Issued by: Arup Associates LLP
Issue Date: 08 March 2024
Client: Northgate Facilities Management Ltd

1. SCOPE
Replacement of legacy CRAC units in Halls A, B and C with modern precision
cooling, serving approximately 840 kW IT load across 3 halls.

2. DESIGN BASIS
  Total IT Load:         840 kW (design day)
  Cooling Redundancy:    N+1 per hall
  Target PUE:            ≤1.35 (design), ≤1.40 (contractual limit)
  ASHRAE Envelope:       A2 (inlet temp 10–35°C, humidity 20–80% non-condensing)
  Seismic Zone:          Not applicable (UK site)

3. EQUIPMENT SCHEDULE

Hall A (280 kW IT load):
  Unit Type: In-Row Cooling (IRC)
  Qty: 8 units (7 active + 1 standby)
  Unit Capacity: 50 kW sensible cooling each
  Chilled Water: 14°C supply / 20°C return
  Airflow per unit: 3,200 m³/hr
  Fan Power: 2.2 kW/unit

Hall B (320 kW IT load):
  Unit Type: Computer Room Air Handler (CRAH)
  Qty: 5 units (4 active + 1 standby)
  Unit Capacity: 90 kW sensible each
  Chilled Water: 14°C supply / 20°C return
  Airflow per unit: 8,500 m³/hr
  Fan Power: 5.5 kW/unit

Hall C (240 kW IT load):
  Unit Type: In-Row Cooling (IRC)
  Qty: 6 units (5 active + 1 standby)
  Unit Capacity: 55 kW sensible each
  Chilled Water: 14°C supply / 20°C return
  Airflow per unit: 3,800 m³/hr
  Fan Power: 2.8 kW/unit

4. CHILLED WATER PLANT
  Total chilled water plant capacity: 1,050 kW (cooling)
  Redundancy: N+1 chillers
  Glycol concentration: 30% (freeze protection to -15°C)
  Primary pump duty: 45 l/s at 350 kPa

5. PERFORMANCE REQUIREMENTS
  Max temperature in server inlet: 27°C
  Min temperature in server inlet: 18°C
  Humidity: 40–60% RH at rack face
  Noise limit (hall perimeter): ≤68 dB(A)""",
    reasoning="""This is a mechanical engineering specification for a data centre cooling upgrade. I need to extract all technical parameters and equipment data.

Project scope: 840 kW IT load across three halls (A, B, C) at Northgate Data Centre. Document Rev B issued by Arup Associates 08 March 2024.

Design parameters:
- Total IT load: 840 kW (let me verify: Hall A 280 + Hall B 320 + Hall C 240 = 840 kW ✓)
- Redundancy: N+1 per hall (single unit standby per hall)
- Target PUE ≤1.35 design, ≤1.40 contractual
- ASHRAE A2 envelope (inlet 10–35°C, 20–80% RH)

Equipment by hall:
Hall A (280 kW): 8 IRC units (7+1), 50 kW each. Active capacity = 7×50 = 350 kW — headroom vs 280 kW = 25%. Airflow = 8×3,200 = 25,600 m³/hr total. Fan power = 8×2.2 = 17.6 kW.

Hall B (320 kW): 5 CRAH units (4+1), 90 kW each. Active capacity = 4×90 = 360 kW — headroom vs 320 kW = 12.5%. Airflow = 5×8,500 = 42,500 m³/hr total. Fan power = 5×5.5 = 27.5 kW.

Hall C (240 kW): 6 IRC units (5+1), 55 kW each. Active capacity = 5×55 = 275 kW — headroom vs 240 kW = 14.6%. Airflow = 6×3,800 = 22,800 m³/hr total. Fan power = 6×2.8 = 16.8 kW.

Total active cooling capacity: 350 + 360 + 275 = 985 kW. Total plant capacity 1,050 kW — larger as it serves total with N+1.

Chilled water: 14°C/20°C flow/return (6°C delta-T) for all halls. 30% glycol for -15°C protection.

Performance: server inlet 18–27°C, humidity 40–60% RH, noise ≤68 dB(A) perimeter.

Confidence: High.""",
    answer="""## Technical Specification Extraction — NDC-MECH-SPEC-2024-003 Rev B

### Document Header
| Field | Value |
|-------|-------|
| **Project** | Northgate Data Centre Cooling Upgrade |
| **Document Reference** | NDC-MECH-SPEC-2024-003 Revision B |
| **Issued by** | Arup Associates LLP |
| **Issue Date** | 08 March 2024 |
| **Client** | Northgate Facilities Management Ltd |

### Design Basis
| Parameter | Value |
|-----------|-------|
| **Total IT Load** | 840 kW (design day) |
| **Cooling Redundancy** | N+1 per hall |
| **Target PUE (design)** | ≤1.35 |
| **Contractual PUE limit** | ≤1.40 |
| **ASHRAE Envelope** | A2 (10–35°C inlet, 20–80% RH non-condensing) |

### Equipment Schedule by Hall
| Parameter | Hall A | Hall B | Hall C |
|-----------|--------|--------|--------|
| **IT Load** | 280 kW | 320 kW | 240 kW |
| **Unit Type** | In-Row Cooling (IRC) | CRAH | In-Row Cooling (IRC) |
| **Total Units** | 8 (7+1) | 5 (4+1) | 6 (5+1) |
| **Unit Capacity** | 50 kW each | 90 kW each | 55 kW each |
| **Active Capacity** | 350 kW | 360 kW | 275 kW |
| **Headroom vs IT Load** | +25.0% | +12.5% | +14.6% |
| **Airflow / Unit** | 3,200 m³/hr | 8,500 m³/hr | 3,800 m³/hr |
| **Total Airflow** | 25,600 m³/hr | 42,500 m³/hr | 22,800 m³/hr |
| **Fan Power / Unit** | 2.2 kW | 5.5 kW | 2.8 kW |
| **Total Fan Power** | 17.6 kW | 27.5 kW | 16.8 kW |

### Chilled Water Plant
| Parameter | Value |
|-----------|-------|
| **Total Plant Capacity** | 1,050 kW |
| **Redundancy** | N+1 chillers |
| **Supply Temperature** | 14°C |
| **Return Temperature** | 20°C (ΔT = 6°C) |
| **Glycol Concentration** | 30% (freeze protection to −15°C) |
| **Primary Pump Duty** | 45 l/s at 350 kPa |

### Environmental Performance Requirements
| Parameter | Requirement |
|-----------|-------------|
| **Server Inlet Temperature** | 18–27°C |
| **Humidity at Rack Face** | 40–60% RH |
| **Noise (hall perimeter)** | ≤68 dB(A) |

### Verification
- IT load check: 280 + 320 + 240 = 840 kW ✓
- All halls configured N+1 redundancy ✓
- Chilled water spec identical across all halls (14/20°C) ✓

**Confidence: High**""",
    difficulty="hard"
))

print("Part 1 (examples 1-10) defined. Continuing to part 2...")

# ─── 11. Policy Document ──────────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract the key provisions from this HR policy:

REMOTE WORKING POLICY
Effective Date: 01 January 2024
Policy Owner: Director of People & Culture
Version: 3.2 | Review Due: 31 December 2024
Applies to: All permanent employees (UK entities)

1. ELIGIBILITY
1.1 Available to employees who have completed their probationary period (minimum 3 months).
1.2 Not available to employees on a performance improvement plan (PIP).
1.3 Roles designated "on-site required" (Annex 1) are excluded.

2. ARRANGEMENTS
2.1 Standard arrangement: minimum 2 days per week in-office ("anchor days").
2.2 Maximum remote working: 3 days per week under standard arrangement.
2.3 Fully remote (exceptions only): requires VP-level approval and quarterly review.
2.4 Anchor days set by line manager; Tuesday and Thursday are company-wide anchor days.

3. EQUIPMENT & EXPENSES
3.1 Company provides: laptop, docking station, ergonomic assessment (one-time).
3.2 Employee responsible for: broadband connectivity, adequate workspace.
3.3 Home office allowance: £400 per annum (taxable benefit, processed via payroll Q1).
3.4 HMRC flat-rate home working deduction available — employees to claim directly.

4. DATA SECURITY
4.1 VPN mandatory when accessing company systems outside office network.
4.2 No client data to be stored on personal devices.
4.3 Physical documents containing confidential data must not leave secure premises.
4.4 Screen privacy filters required in public spaces.

5. HEALTH & SAFETY
5.1 Employees must complete DSE self-assessment annually.
5.2 Company ergonomic assessor available on request.
5.3 Lone working protocol applies to home workers.

6. NON-COMPLIANCE
Breach of data security provisions: subject to disciplinary procedure up to and including dismissal.
Repeated non-compliance with anchor days: treated as unauthorised absence.""",
    reasoning="""This is a corporate HR remote working policy, Version 3.2, effective 01 January 2024. I need to extract all provisions in structured form.

Scope: all permanent UK employees. Effective 01 Jan 2024. Review due 31 Dec 2024. Policy owner: Director of People & Culture.

Eligibility requirements:
- Must have completed probationary period (min 3 months)
- Not eligible if on PIP (Performance Improvement Plan)
- "On-site required" roles excluded (Annex 1 referenced but not in this excerpt)

Arrangements:
- Standard: min 2 days in-office, max 3 days remote
- Company-wide anchor days: Tuesday and Thursday
- Line manager sets specific anchor days
- Fully remote (exception): requires VP approval + quarterly review

Equipment/expenses:
- Company provides: laptop, docking station, ergonomic assessment
- Employee responsible for: broadband, workspace
- £400/year home office allowance — taxable, Q1 payroll
- HMRC flat-rate deduction also available (employee claims directly)

Data security:
- VPN mandatory off-network
- No client data on personal devices
- Physical confidential documents stay on-site
- Screen privacy filters in public

H&S:
- Annual DSE self-assessment
- Ergonomic assessor on request
- Lone working protocol applies

Consequences: data breach → disciplinary up to dismissal. Anchor day non-compliance → unauthorised absence.

Confidence: High.""",
    answer="""## HR Policy Extraction — Remote Working Policy v3.2

### Policy Metadata
| Field | Value |
|-------|-------|
| **Policy Title** | Remote Working Policy |
| **Version** | 3.2 |
| **Effective Date** | 01 January 2024 |
| **Review Due** | 31 December 2024 |
| **Policy Owner** | Director of People & Culture |
| **Scope** | All permanent employees — UK entities |

### Eligibility
| Criterion | Requirement |
|-----------|-------------|
| Probationary period | Must be completed (minimum 3 months) |
| Performance Improvement Plan | **Not eligible** if on active PIP |
| On-site required roles | **Excluded** (see Annex 1) |

### Working Arrangements
| Arrangement | Detail |
|-------------|--------|
| **In-office minimum** | 2 days per week |
| **Remote maximum (standard)** | 3 days per week |
| **Company anchor days** | Tuesday and Thursday (company-wide) |
| **Individual anchor days** | Set by line manager |
| **Fully remote** | Exception only — requires VP approval + quarterly review |

### Equipment & Expenses
| Item | Responsibility | Detail |
|------|---------------|--------|
| Laptop | Company provided | — |
| Docking station | Company provided | — |
| Ergonomic assessment | Company provided | One-time |
| Broadband | **Employee** | — |
| Home workspace | **Employee** | — |
| **Home office allowance** | Company (taxable) | **£400/annum** — Q1 payroll |
| HMRC flat-rate deduction | Employee claims directly | — |

### Data Security Obligations
| Obligation | Requirement |
|------------|-------------|
| VPN | **Mandatory** when accessing systems off-network |
| Client data | Must not be stored on personal devices |
| Physical confidential documents | Must not leave secure premises |
| Public spaces | Screen privacy filter required |

### Health & Safety
- DSE self-assessment: **annually**
- Ergonomic assessor: available on request
- Lone working protocol: applies to all home workers

### Non-Compliance Consequences
| Breach | Consequence |
|--------|-------------|
| Data security breach | Disciplinary up to **dismissal** |
| Repeated anchor day non-compliance | Treated as **unauthorised absence** |

**Confidence: High**""",
    difficulty="medium"
))

# ─── 12. Audit Report ─────────────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract all findings and ratings from this internal audit report:

INTERNAL AUDIT REPORT
Report Reference: IA-2024-FIN-0088
Audit Area: Accounts Payable & Supplier Payments
Audit Period: October 2023 – January 2024
Issued: 14 March 2024
Prepared by: Group Internal Audit
Audit Rating: MODERATE

EXECUTIVE SUMMARY
The audit identified 6 findings across the accounts payable function.
Two findings are rated HIGH risk, three MEDIUM, and one LOW.
Management action plans have been agreed for all findings.

FINDINGS

Finding 01 — DUPLICATE PAYMENT RISK [HIGH]
Observation: AP system does not enforce unique invoice number validation across
supplier accounts. Testing of 847 invoices identified 3 duplicate payments totalling
£47,280. Duplicates paid to Suppliers B, F and M.
Root Cause: Invoice uniqueness check disabled during system migration (Sept 2022)
and not re-enabled.
Impact: Financial loss; reputational risk with suppliers.
Management Action: Re-enable uniqueness constraint by 30 April 2024. Retrospective
scan of 24 months to be completed by 31 May 2024.
Owner: AP Manager (J. Thorne)

Finding 02 — SEGREGATION OF DUTIES [HIGH]
Observation: 4 AP clerks have ability to both create suppliers AND approve payments
up to £10,000. COSO framework requires these roles to be separated.
Root Cause: Role-based access controls not reviewed since 2019 ERP implementation.
Management Action: Restrict dual-role access by 15 April 2024. Quarterly access review
to be established.
Owner: IT Security Manager + AP Manager

Finding 03 — SUPPLIER MASTER FILE CHANGES [MEDIUM]
Observation: 23% of supplier bank detail changes in the period had no independent
authorisation. Tested 61 changes; 14 lacked second approval.
Management Action: Implement mandatory 2-person authorisation for bank changes.
Owner: AP Manager. Target: 30 April 2024.

Finding 04 — INVOICE APPROVAL AUTHORITY [MEDIUM]
Observation: 8% of invoices (£ value: £312,000 of £3.9M tested) approved by individuals
without delegated authority per the Delegation of Authority matrix.
Management Action: Refresh DoA training; automate authority limits in workflow.
Owner: Finance Director. Target: 31 May 2024.

Finding 05 — STATEMENT RECONCILIATIONS [MEDIUM]
Observation: Only 41% of key suppliers (by value) had supplier statement reconciliations
completed in the audit period. Policy requires 100% monthly for suppliers >£50k/annum.
Management Action: Reinstate monthly reconciliation programme.
Owner: AP Manager. Target: 30 April 2024.

Finding 06 — FILING & DOCUMENTATION [LOW]
Observation: 12% of paid invoices had incomplete supporting documentation.
Management Action: Refresh filing procedure. Owner: AP Supervisor. Target: 28 Feb 2024.""",
    reasoning="""This is an internal audit report for an accounts payable function, rated MODERATE overall. I need to extract all findings, ratings, observations, and action plans.

Report metadata: IA-2024-FIN-0088, audit period Oct 2023–Jan 2024, issued 14 March 2024. Overall rating: MODERATE. 6 findings: 2 HIGH, 3 MEDIUM, 1 LOW.

Finding 01 — DUPLICATE PAYMENT RISK [HIGH]:
- 847 invoices tested, 3 duplicates found = 0.35% rate
- Value of duplicates: £47,280 — actual financial loss
- Root cause: uniqueness check disabled during Sept 2022 migration and not re-enabled (18 months gap)
- Actions: re-enable constraint by 30 Apr 2024; retrospective 24-month scan by 31 May 2024
- Owner: J. Thorne (AP Manager)

Finding 02 — SEGREGATION OF DUTIES [HIGH]:
- 4 AP clerks have create-supplier AND approve-payments (≤£10k) dual capability
- COSO framework violation
- Root cause: access controls not reviewed since 2019 ERP implementation (5 years)
- Actions: restrict access by 15 Apr 2024; establish quarterly access review
- Owner: IT Security Manager + AP Manager

Finding 03 — SUPPLIER MASTER FILE [MEDIUM]:
- 61 bank detail changes tested; 14 (23%) lacked second approval
- Risk: fraudulent bank detail manipulation (supplier impersonation/redirection)
- Action: mandatory 2-person auth for bank changes by 30 Apr 2024

Finding 04 — INVOICE APPROVAL AUTHORITY [MEDIUM]:
- 8% of invoices by count, £312k of £3.9M tested (8.0%) approved outside delegated authority
- Action: DoA training refresh + workflow automation by 31 May 2024

Finding 05 — STATEMENT RECONCILIATIONS [MEDIUM]:
- Policy: 100% monthly for suppliers >£50k/annum; actual: only 41% completed
- Gap of 59% non-compliance
- Action: reinstate programme by 30 Apr 2024

Finding 06 — DOCUMENTATION [LOW]:
- 12% of paid invoices had incomplete supporting docs
- Action: refresh filing procedure, target 28 Feb 2024 (already past — should be noted as potentially overdue)

Overall: two HIGH findings with actual financial loss identified; SoD issue is systemic and longest-running (5 years). Confidence: High.""",
    answer="""## Internal Audit Report Extraction — IA-2024-FIN-0088

### Report Metadata
| Field | Value |
|-------|-------|
| **Reference** | IA-2024-FIN-0088 |
| **Audit Area** | Accounts Payable & Supplier Payments |
| **Audit Period** | October 2023 – January 2024 |
| **Issue Date** | 14 March 2024 |
| **Prepared by** | Group Internal Audit |
| **Overall Rating** | **MODERATE** |

### Findings Summary
| Risk Level | Count |
|-----------|-------|
| HIGH | 2 |
| MEDIUM | 3 |
| LOW | 1 |
| **Total** | **6** |

### Detailed Findings
| # | Finding | Rating | Key Metric | Root Cause | Owner | Target Date |
|---|---------|--------|-----------|------------|-------|-------------|
| 01 | Duplicate Payment Risk | **HIGH** | 3 duplicates / £47,280 in 847 tested | Uniqueness check disabled Sep 2022 migration | J. Thorne (AP Mgr) | 30 Apr 2024 |
| 02 | Segregation of Duties | **HIGH** | 4 clerks with dual create+approve access | Access not reviewed since 2019 ERP | IT Security + AP Mgr | 15 Apr 2024 |
| 03 | Supplier Bank Detail Changes | **MEDIUM** | 14/61 changes (23%) unauthorised | No mandatory 2nd approval | AP Manager | 30 Apr 2024 |
| 04 | Invoice Approval Authority | **MEDIUM** | £312k of £3.9M (8%) outside DoA | Delegation matrix not enforced in workflow | Finance Director | 31 May 2024 |
| 05 | Statement Reconciliations | **MEDIUM** | Only 41% completion (policy: 100%) | Programme not maintained | AP Manager | 30 Apr 2024 |
| 06 | Filing & Documentation | **LOW** | 12% invoices incomplete docs | Filing procedure not followed | AP Supervisor | 28 Feb 2024 |

### HIGH Risk Findings Detail

**Finding 01 — Duplicate Payments:**
- Sample: 847 invoices tested
- Issue: 3 confirmed duplicate payments totalling **£47,280** (financial loss confirmed)
- Affected suppliers: B, F, and M
- Remediation: Re-enable uniqueness constraint + retrospective 24-month scan

**Finding 02 — Segregation of Duties (COSO violation):**
- 4 AP clerks have ability to **create suppliers AND approve payments** (up to £10,000)
- Access controls unchanged since 2019 (5-year gap)
- Remediation: Restrict access by 15 Apr; establish quarterly access review cadence

### Key Observations
- **Actual financial loss confirmed:** £47,280 from duplicate payments
- **Longest-running issue:** SoD gap exists since 2019 ERP implementation
- **Finding 06 target (28 Feb 2024) has already passed** — status should be confirmed with management
- All 6 management action plans agreed

**Confidence: High**""",
    difficulty="hard"
))

# ─── 13. Lease Agreement ──────────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract the key commercial terms from this commercial lease:

COMMERCIAL LEASE SUMMARY — HEADS OF TERMS

Premises: Suite 4B, Pinnacle Business Park, Warrington WA3 6GH
Approx. Floor Area: 3,240 sq ft (GIA), 2,980 sq ft (NIA)

LANDLORD: Pinnacle Estates (Northwest) Ltd, Company No. 07241983
TENANT: Quorum Analytics Ltd, Company No. 11938420

TERM: 7 years commencing 25 March 2024 (Lease Commencement Date)
BREAK CLAUSE: Tenant-only break at Year 4 (25 March 2028), subject to:
  - 6 months prior written notice
  - No rent arrears at break date
  - Vacant possession on break date

RENT:
Year 1–3: £54,000 per annum (£16.67/sq ft NIA)
Year 4–7: Open market rent review (upward only, surveyor determination if not agreed)
Payment: Quarterly in advance on usual quarter days

SERVICE CHARGE:
Estimated service charge: £11.50/sq ft NIA = £34,270 p.a.
Cap: 5% annual increase on base year (Year 1) charge
Scope: Building insurance, maintenance, security, landscaping, M&E

RENT-FREE PERIOD: 4 months from Lease Commencement Date

DILAPIDATIONS:
Schedule of condition to be attached at lease commencement.
Tenant's liability limited to putting premises back to schedule of condition standard.

DEPOSIT: 6 months' rent = £27,000 (held by landlord as stakeholder)

ALTERATIONS: Consent required; not to be unreasonably withheld.
ASSIGNMENT/SUBLETTING: Permitted with landlord consent (not unreasonably withheld).
USE: Class E (commercial, business, service) — office use.""",
    reasoning="""This is a commercial lease heads of terms document for office space at Warrington. I need to extract all key commercial and legal provisions.

Parties: Landlord Pinnacle Estates (Northwest) Ltd (07241983), Tenant Quorum Analytics Ltd (11938420).

Premises: Suite 4B, Pinnacle Business Park, Warrington WA3 6GH. GIA 3,240 sq ft, NIA 2,980 sq ft (NIA is the lettable area used for rent calculation).

Term: 7 years from 25 March 2024, so expiry 24 March 2031.

Break clause: Tenant-only at Year 4 (25 March 2028). Three conditions: 6 months notice (so notice must be served by 25 September 2027), no rent arrears at break date, vacant possession. This is a conditional break — failure on any condition voids the break.

Rent:
- Year 1-3: £54,000 p.a. = £4,500/month = £13,500/quarter
- £54,000 / 2,980 NIA = £18.12/sq ft — stated as £16.67/sq ft. Let me recheck: £54,000 / 2,980 = £18.12. The stated £16.67/sq ft seems inconsistent. Actually £16.67 × 2,980 = £49,676 ≠ £54,000. This may be a GIA-based rate: £54,000 / 3,240 = £16.67/sq ft GIA ✓. So the document appears to quote rent per sq ft GIA despite saying NIA. I should flag this inconsistency.
- Year 4-7: open market review (upward only)
- Payment: quarterly in advance

Rent-free: 4 months. Economic rent start = 25 July 2024.

Service charge: £34,270 p.a. estimated (£11.50 × 2,980 NIA = £34,270 ✓). Capped at 5% annual increase.

Deposit: 6 months = £54,000/2 = £27,000 ✓.

Dilapidations limited to schedule of condition standard — tenant-favourable clause.

Alterations and assignment permitted with consent.

Confidence: High, but flagging £/sq ft inconsistency in rent clause.""",
    answer="""## Commercial Lease Extraction — Suite 4B, Pinnacle Business Park

### Parties & Premises
| Field | Value |
|-------|-------|
| **Landlord** | Pinnacle Estates (Northwest) Ltd — Co. No. 07241983 |
| **Tenant** | Quorum Analytics Ltd — Co. No. 11938420 |
| **Premises** | Suite 4B, Pinnacle Business Park, Warrington WA3 6GH |
| **Floor Area (GIA)** | 3,240 sq ft |
| **Floor Area (NIA)** | 2,980 sq ft |

### Lease Term
| Milestone | Date |
|-----------|------|
| **Lease Commencement** | 25 March 2024 |
| **Contractual Expiry** | 24 March 2031 |
| **Lease Length** | 7 years |
| **Rent-Free Period Ends** | 25 July 2024 (4 months) |

### Break Clause
| Detail | Provision |
|--------|-----------|
| **Break Right** | Tenant only |
| **Break Date** | 25 March 2028 (Year 4) |
| **Notice Required** | 6 months prior written (by **25 September 2027**) |
| **Conditions** | No rent arrears; vacant possession on break date |
| **Nature** | Conditional break — all conditions must be met |

### Rent
| Period | Annual Rent | Quarterly | Per Sq Ft |
|--------|-------------|-----------|-----------|
| **Years 1–3** | **£54,000** | £13,500 | £16.67/sq ft GIA |
| **Years 4–7** | Open market review | — | Upward only |

> **Note:** Document states £16.67/sq ft NIA, but £54,000 ÷ 2,980 NIA = £18.12/sq ft. Rate of £16.67 reconciles to GIA (£54,000 ÷ 3,240 = £16.67). Likely a drafting error — recommend clarification before execution.

### Service Charge
| Item | Value |
|------|-------|
| **Estimated p.a.** | £34,270 (£11.50/sq ft NIA) |
| **Annual Increase Cap** | 5% on Year 1 base |
| **Scope** | Building insurance, maintenance, security, landscaping, M&E |

### Other Financial Terms
| Item | Value |
|------|-------|
| **Rent-Free Period** | 4 months (economic rent from 25 Jul 2024) |
| **Deposit** | **£27,000** (6 months' rent — held as stakeholder) |

### Key Provisions
| Provision | Detail |
|-----------|--------|
| **Dilapidations** | Limited to schedule of condition standard (tenant-favourable) |
| **Alterations** | Permitted with consent — not unreasonably withheld |
| **Assignment/Subletting** | Permitted with consent — not unreasonably withheld |
| **Permitted Use** | Class E — office use |

**Confidence: High** — one discrepancy flagged in rent per sq ft calculation""",
    difficulty="hard"
))

# ─── 14. Bank Statement ───────────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract and analyse the transactions from this bank statement:

BUSINESS CURRENT ACCOUNT STATEMENT
Account Name: HALCYON EVENTS LTD
Sort Code: 30-98-45    Account Number: 71284936
Statement Period: 01 February 2024 – 29 February 2024
Opening Balance: £12,847.33

Date       Description                        Debit (£)   Credit (£)  Balance (£)
01 Feb     Balance brought forward                                      12,847.33
02 Feb     BACS IN — WESTON CONF CENTRE        8,500.00               21,347.33
05 Feb     DD — HMRC PAYE FEB24                2,847.00               18,500.33
07 Feb     CHQ 003847 — FLOWERS BY ROSA          340.00               18,160.33
09 Feb     BACS IN — MERIDIAN CORP EVENTS      22,400.00               40,560.33
12 Feb     BP — VENUE EQUIPMENT HIRE           1,250.00               39,310.33
13 Feb     FASTER PMT — STAFFING DIRECT LTD    6,800.00               32,510.33
15 Feb     BACS IN — HILLSIDE WEDDINGS          3,200.00               35,710.33
16 Feb     DD — OFFICE RENT FEB24              3,500.00               32,210.33
19 Feb     CHAPS — AV PRODUCTIONS UK           4,950.00               27,260.33
22 Feb     BACS IN — CITY LAW FIRM LLP        15,000.00               42,260.33
23 Feb     DD — BT BUSINESS INTERNET              89.00               42,171.33
23 Feb     DD — ZURICH BUSINESS INS             412.00               41,759.33
26 Feb     FP — CATERING EXCELLENCE LTD        3,800.00               37,959.33
27 Feb     DD — BARCLAYS LOAN FEB24            1,200.00               36,759.33
29 Feb     Closing Balance                                              36,759.33

Extract all transactions, categorise them, and verify the running balance.""",
    reasoning="""This is a February 2024 business bank statement for Halcyon Events Ltd. I need to extract all transactions, categorise them, verify balances, and identify patterns.

Opening balance: £12,847.33 on 01 Feb 2024.

Let me list all transactions and verify running balances:
02 Feb BACS IN Weston Conf Centre: +£8,500.00 → 12,847.33 + 8,500 = £21,347.33 ✓
05 Feb DD HMRC PAYE: -£2,847.00 → 21,347.33 - 2,847 = £18,500.33 ✓
07 Feb CHQ Flowers by Rosa: -£340.00 → 18,500.33 - 340 = £18,160.33 ✓
09 Feb BACS IN Meridian Corp Events: +£22,400.00 → 18,160.33 + 22,400 = £40,560.33 ✓
12 Feb BP Venue Equipment Hire: -£1,250.00 → 40,560.33 - 1,250 = £39,310.33 ✓
13 Feb FP Staffing Direct Ltd: -£6,800.00 → 39,310.33 - 6,800 = £32,510.33 ✓
15 Feb BACS IN Hillside Weddings: +£3,200.00 → 32,510.33 + 3,200 = £35,710.33 ✓
16 Feb DD Office Rent: -£3,500.00 → 35,710.33 - 3,500 = £32,210.33 ✓
19 Feb CHAPS AV Productions: -£4,950.00 → 32,210.33 - 4,950 = £27,260.33 ✓
22 Feb BACS IN City Law Firm LLP: +£15,000.00 → 27,260.33 + 15,000 = £42,260.33 ✓
23 Feb DD BT Business Internet: -£89.00 → 42,260.33 - 89 = £42,171.33 ✓
23 Feb DD Zurich Business Ins: -£412.00 → 42,171.33 - 412 = £41,759.33 ✓
26 Feb FP Catering Excellence: -£3,800.00 → 41,759.33 - 3,800 = £37,959.33 ✓
27 Feb DD Barclays Loan: -£1,200.00 → 37,959.33 - 1,200 = £36,759.33 ✓

Closing balance: £36,759.33 ✓ — all balances verified.

Total credits: 8,500 + 22,400 + 3,200 + 15,000 = £49,100.00
Total debits: 2,847 + 340 + 1,250 + 6,800 + 3,500 + 4,950 + 89 + 412 + 3,800 + 1,200 = £25,188.00
Net change: 49,100 - 25,188 = £23,912 → 12,847.33 + 23,912 = £36,759.33 ✓

Income: 4 receipts. Outgoings: 10 transactions.
Categorisation:
- Income (client receipts): Weston, Meridian, Hillside, City Law Firm = £49,100
- Payroll: PAYE £2,847
- Suppliers/event costs: Flowers £340, Equipment Hire £1,250, Staffing £6,800, AV Productions £4,950, Catering £3,800 = £17,140
- Fixed overheads: Office Rent £3,500, BT £89, Insurance £412, Loan £1,200 = £5,201

Confidence: High — all transactions verified.""",
    answer="""## Bank Statement Extraction — Halcyon Events Ltd, February 2024

### Account Details
| Field | Value |
|-------|-------|
| **Account Name** | Halcyon Events Ltd |
| **Sort Code** | 30-98-45 |
| **Account Number** | 71284936 |
| **Statement Period** | 01–29 February 2024 |
| **Opening Balance** | £12,847.33 |
| **Closing Balance** | **£36,759.33** |

### All Transactions
| Date | Description | Type | Debit (£) | Credit (£) | Balance (£) |
|------|-------------|------|-----------|-----------|------------|
| 02 Feb | Weston Conf Centre | BACS IN | — | 8,500.00 | 21,347.33 |
| 05 Feb | HMRC PAYE Feb24 | DD | 2,847.00 | — | 18,500.33 |
| 07 Feb | Flowers by Rosa | CHQ | 340.00 | — | 18,160.33 |
| 09 Feb | Meridian Corp Events | BACS IN | — | 22,400.00 | 40,560.33 |
| 12 Feb | Venue Equipment Hire | BP | 1,250.00 | — | 39,310.33 |
| 13 Feb | Staffing Direct Ltd | FP | 6,800.00 | — | 32,510.33 |
| 15 Feb | Hillside Weddings | BACS IN | — | 3,200.00 | 35,710.33 |
| 16 Feb | Office Rent Feb24 | DD | 3,500.00 | — | 32,210.33 |
| 19 Feb | AV Productions UK | CHAPS | 4,950.00 | — | 27,260.33 |
| 22 Feb | City Law Firm LLP | BACS IN | — | 15,000.00 | 42,260.33 |
| 23 Feb | BT Business Internet | DD | 89.00 | — | 42,171.33 |
| 23 Feb | Zurich Business Ins | DD | 412.00 | — | 41,759.33 |
| 26 Feb | Catering Excellence Ltd | FP | 3,800.00 | — | 37,959.33 |
| 27 Feb | Barclays Loan Feb24 | DD | 1,200.00 | — | 36,759.33 |

### Summary by Category
| Category | Transactions | Amount |
|----------|-------------|--------|
| **Client Income** | 4 | **£49,100.00** |
| Payroll (PAYE) | 1 | £2,847.00 |
| Event Suppliers | 5 | £17,140.00 |
| Fixed Overheads | 4 | £5,201.00 |
| **Total Debits** | **10** | **£25,188.00** |

### Financial Summary
| Metric | Value |
|--------|-------|
| Total Credits | £49,100.00 |
| Total Debits | £25,188.00 |
| **Net Movement** | **+£23,912.00** |
| **Closing Balance** | **£36,759.33** |

### Client Income Detail
| Client | Amount |
|--------|--------|
| Meridian Corp Events | £22,400.00 |
| City Law Firm LLP | £15,000.00 |
| Weston Conf Centre | £8,500.00 |
| Hillside Weddings | £3,200.00 |

**Balance verification: All 14 running balances confirmed correct ✓**

**Confidence: High**""",
    difficulty="medium"
))

# ─── 15. Loan Agreement ───────────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract the financial terms from this business loan agreement summary:

BUSINESS LOAN FACILITY AGREEMENT — TERM SHEET
Lender: Thornfield Commercial Finance Ltd
Borrower: Castlegate Manufacturing Solutions Ltd (Co. No. 08374920)
Date: 15 January 2024

FACILITY DETAILS
Facility Type: Term Loan (secured)
Facility Amount: £850,000
Purpose: Purchase of CNC machining centre and fit-out of production bay

REPAYMENT SCHEDULE
Drawdown: Single drawdown on 15 January 2024
Repayment: 60 monthly instalments
Monthly Capital Repayment: £14,166.67
Final Maturity Date: 15 January 2029
Balloon Payment: None (fully amortising)

INTEREST
Rate Type: Fixed
Interest Rate: 7.85% per annum
Monthly Interest (Month 1): £5,560.42
Effective Annual Rate (EAR): 8.14%
Total Interest Payable: £187,412.40

FEES
Arrangement Fee: £8,500 (1.0% of facility, deducted at drawdown)
Early Repayment Charge: 2% of outstanding balance (Years 1–2); 1% (Years 3–5)

SECURITY
Fixed charge: CNC machining centre (asset financed)
Personal guarantee: Director — M. Castlegate (up to £425,000 = 50% of facility)

COVENANTS
Interest Cover Ratio: minimum 2.0× (tested semi-annually)
Net Debt / EBITDA: maximum 3.5× (tested semi-annually)
Change of control: lender consent required""",
    reasoning="""This is a term sheet for a secured business loan from Thornfield Commercial Finance to Castlegate Manufacturing Solutions. I need to extract all financial terms.

Borrower: Castlegate Manufacturing Solutions Ltd, Co. No. 08374920. Facility: £850,000 term loan, secured, for CNC equipment purchase.

Repayment: 60 monthly instalments of £14,166.67. Let me verify: £850,000 / 60 = £14,166.67 ✓ (capital repayment only).

Interest rate: 7.85% pa fixed. Monthly interest Month 1: £850,000 × 7.85% / 12 = £850,000 × 0.0785 / 12 = £66,725 / 12 = £5,560.42 ✓

EAR: 8.14% for a 7.85% nominal rate — this implies monthly compounding: (1 + 0.0785/12)^12 - 1 = (1.006542)^12 - 1 ≈ 1.0814 - 1 = 8.14% ✓

Total interest: stated £187,412.40. This is the sum of all monthly interest payments over 60 months on a reducing balance — I'll trust the stated figure.

Arrangement fee: £8,500 = 1.0% of £850,000 ✓. Net disbursement after fee deduction: £850,000 - £8,500 = £841,500.

Total cost of borrowing: £187,412.40 interest + £8,500 arrangement = £195,912.40.

ERC: 2% years 1-2, 1% years 3-5. On £850,000, max ERC (year 1) = £17,000.

Security: fixed charge on asset + personal guarantee £425,000 (50% of facility).

Covenants: ICR ≥ 2.0×, Net Debt/EBITDA ≤ 3.5×, both semi-annual. Change of control requires lender consent.

Confidence: High — arithmetic verified.""",
    answer="""## Business Loan Extraction — Castlegate Manufacturing Solutions

### Parties
| Role | Details |
|------|---------|
| **Lender** | Thornfield Commercial Finance Ltd |
| **Borrower** | Castlegate Manufacturing Solutions Ltd (Co. No. 08374920) |
| **Date** | 15 January 2024 |

### Facility Summary
| Parameter | Value |
|-----------|-------|
| **Facility Type** | Secured Term Loan |
| **Facility Amount** | £850,000 |
| **Purpose** | CNC machining centre + production bay fit-out |
| **Drawdown** | Single, 15 January 2024 |
| **Net Disbursement (after fee)** | **£841,500** |

### Repayment Schedule
| Parameter | Value |
|-----------|-------|
| **Number of Instalments** | 60 monthly |
| **Monthly Capital Repayment** | £14,166.67 |
| **Final Maturity Date** | 15 January 2029 |
| **Balloon Payment** | None — fully amortising |

### Interest & Costs
| Item | Value |
|------|-------|
| **Rate Type** | Fixed |
| **Nominal Annual Rate** | **7.85%** |
| **Effective Annual Rate (EAR)** | 8.14% |
| **Month 1 Interest** | £5,560.42 |
| **Total Interest Payable** | £187,412.40 |
| **Arrangement Fee** | £8,500 (1.0% of facility) |
| **Total Cost of Borrowing** | **£195,912.40** |

### Fees & Charges
| Fee | Trigger | Amount |
|-----|---------|--------|
| Arrangement Fee | At drawdown | £8,500 |
| Early Repayment Charge (Yr 1–2) | Early settlement | 2% of outstanding balance |
| Early Repayment Charge (Yr 3–5) | Early settlement | 1% of outstanding balance |

### Security Package
| Type | Detail |
|------|--------|
| **Fixed charge** | CNC machining centre (asset financed) |
| **Personal guarantee** | Director M. Castlegate — **£425,000** (50% of facility) |

### Financial Covenants
| Covenant | Threshold | Testing Frequency |
|----------|-----------|------------------|
| **Interest Cover Ratio** | Minimum 2.0× | Semi-annually |
| **Net Debt / EBITDA** | Maximum 3.5× | Semi-annually |
| **Change of control** | Lender consent required | — |

### Arithmetic Verification
- Capital: £850,000 ÷ 60 = £14,166.67/month ✓
- Month 1 interest: £850,000 × 7.85% ÷ 12 = £5,560.42 ✓
- EAR: (1 + 0.0785/12)^12 − 1 ≈ 8.14% ✓
- Arrangement fee: 1.0% × £850,000 = £8,500 ✓

**Confidence: High**""",
    difficulty="hard"
))

# ─── 16. Payslip ──────────────────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract all earnings and deductions from this payslip:

PAYSLIP — CONFIDENTIAL
Employer: Stonebridge Capital Management Ltd
Employee: OGUNDIMU, Adebayo Festus
Employee No: SCM-04471
Tax Code: 1257L (cumulative)
NI Number: NP 84 73 21 C
NI Category: A
Pay Period: March 2024 (Period 12 of 12)
Pay Date: 28 March 2024

EARNINGS
Basic Salary (Annual £78,000 / 12)        6,500.00
Performance Bonus (Q4 2023)               4,200.00
Car Allowance                               450.00
Private Medical Benefit (employer cost)     312.00  [P11D benefit — shown for info]
─────────────────────────────────────────────────────
GROSS PAY (taxable)                      11,150.00
[Employer pension contrib. £390.00 — not in gross]

DEDUCTIONS
Income Tax (PAYE)                         3,672.40
Employee National Insurance (Cat A)         905.28
Employee Pension (5% of qualifying)         325.00
─────────────────────────────────────────────────────
TOTAL DEDUCTIONS                          4,902.68

NET PAY                                   6,247.32

YTD FIGURES (April 2023 – March 2024)
Gross Pay YTD:                           96,450.00
Income Tax YTD:                          26,847.20
NI YTD:                                   5,819.44
Pension (employee) YTD:                   3,900.00

PAYMENT METHOD: BACS  |  Account: ••••7821
EMPLOYER NI (for payroll reference): £1,362.42""",
    reasoning="""This is a March 2024 payslip for Adebayo Festus Ogundimu at Stonebridge Capital Management. Period 12 of 12 means this is the final month of the tax year (April 2023–March 2024).

Earnings:
- Basic salary: £78,000/12 = £6,500.00 ✓
- Q4 performance bonus: £4,200.00
- Car allowance: £450.00
- Private medical (P11D): £312.00 — noted as employer cost shown for information only (not cash payment)
- Gross taxable pay: 6,500 + 4,200 + 450 = £11,150.00 ✓ (P11D benefit excluded from gross cash)

Employer pension: £390.00 — not in employee gross (this is the employer's contribution above the line).

Deductions:
- PAYE Income Tax: £3,672.40
- Employee NI (Cat A): £905.28
- Employee pension (5%): £325.00. Check: 5% of what? £325 / 5% = £6,500 = basic salary. So pension qualifying earnings = basic only (not bonus/car allowance). £6,500 × 5% = £325.00 ✓

Total deductions: 3,672.40 + 905.28 + 325.00 = £4,902.68 ✓

Net pay: 11,150.00 - 4,902.68 = £6,247.32 ✓

YTD check: 
- Gross YTD £96,450. Year = 12 months. Average gross per month = 96,450/12 = £8,037.50. This month is £11,150 (high due to bonus) — prior 11 months avg would be (96,450 - 11,150)/11 = 85,300/11 = £7,754.55.
- Tax YTD £26,847.20. Effective rate YTD: 26,847.20/96,450 = 27.8%.
- NI YTD: £5,819.44
- Pension employee YTD: £3,900. Check: 12 months × £325 = £3,900 ✓ (consistent with pension on basic only)

Tax code 1257L = £12,570 personal allowance standard.

Employer NI: £1,362.42 (payroll reference — employer cost not charged to employee).

Total employer cost this month: £11,150 + £390 pension + £1,362.42 employer NI = £12,902.42.

Confidence: High.""",
    answer="""## Payslip Extraction — Ogundimu, Adebayo Festus — March 2024

### Employee & Payroll Details
| Field | Value |
|-------|-------|
| **Employer** | Stonebridge Capital Management Ltd |
| **Employee** | Ogundimu, Adebayo Festus |
| **Employee No** | SCM-04471 |
| **Tax Code** | 1257L (cumulative) |
| **NI Number** | NP 84 73 21 C |
| **NI Category** | A |
| **Pay Period** | March 2024 (Period 12 of 12 — final month, tax year end) |
| **Pay Date** | 28 March 2024 |

### Earnings
| Component | Amount | Notes |
|-----------|--------|-------|
| Basic Salary | £6,500.00 | £78,000 p.a. ÷ 12 |
| Performance Bonus (Q4 2023) | £4,200.00 | — |
| Car Allowance | £450.00 | — |
| Private Medical (P11D) | £312.00 | *Employer cost — informational only, not cash* |
| **Gross Pay (taxable)** | **£11,150.00** | Excl. P11D |
| Employer Pension Contribution | £390.00 | *Not in employee gross* |

### Deductions
| Deduction | Amount | Basis |
|-----------|--------|-------|
| Income Tax (PAYE) | £3,672.40 | Tax code 1257L |
| Employee NI (Cat A) | £905.28 | — |
| Employee Pension (5%) | £325.00 | 5% of basic salary (£6,500) |
| **Total Deductions** | **£4,902.68** | |

### Net Pay
| | Amount |
|--|--------|
| Gross Pay | £11,150.00 |
| Less: Total Deductions | (£4,902.68) |
| **Net Pay** | **£6,247.32** |
| Payment Method | BACS — account ••••7821 |

### Year-to-Date (Apr 2023 – Mar 2024)
| Metric | YTD Amount |
|--------|-----------|
| **Gross Pay YTD** | £96,450.00 |
| Income Tax YTD | £26,847.20 |
| Employee NI YTD | £5,819.44 |
| Employee Pension YTD | £3,900.00 |
| **Effective tax rate (YTD)** | 27.8% |

### Employer Costs This Period
| Item | Amount |
|------|--------|
| Gross Pay | £11,150.00 |
| Employer Pension Contribution | £390.00 |
| Employer NI | £1,362.42 |
| **Total Employer Cost** | **£12,902.42** |

### Arithmetic Verification
- Gross: £6,500 + £4,200 + £450 = £11,150 ✓
- Pension: 5% × £6,500 = £325 ✓
- Total deductions: £3,672.40 + £905.28 + £325.00 = £4,902.68 ✓
- Net pay: £11,150 − £4,902.68 = £6,247.32 ✓
- Pension YTD: 12 × £325 = £3,900 ✓

**Confidence: High**""",
    difficulty="medium"
))

# ─── 17. Planning Application ─────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract the key details from this planning application:

PLANNING APPLICATION — DECISION NOTICE
Local Planning Authority: Moorfield District Council
Application Reference: MDD/2024/0284
Decision Date: 22 March 2024

APPLICANT: Mr. & Mrs. R. Chakraborty
Site Address: 14 Larkspur Lane, Moorfield, MF4 7BT
Agent: Whitmore Architectural Services, 3 Market Square, Moorfield

DESCRIPTION OF DEVELOPMENT:
Single-storey rear extension (5.8m projection, maximum height 3.9m, eaves
height 2.6m) with glazed roof lantern; internal remodelling to create open-plan
kitchen/dining/living space; formation of 2 no. rooflights to existing rear slope.

DECISION: GRANTED WITH CONDITIONS

CONDITIONS:
1. Development shall begin within 3 years of this decision (by 22 March 2027)
2. Development in accordance with approved plans: WAS-2024-048 Rev C (site plan),
   WAS-2024-049 Rev B (floor plans), WAS-2024-050 Rev A (elevations)
3. External materials (brick, roof tiles) to match existing dwelling — samples to be
   submitted to and approved by LPA before construction commences
4. No obstruction of neighbouring drainage channel during construction
5. Obscure glazing to side elevation window (minimum Level 4) — retained in perpetuity

CONSULTATION RESPONSES:
Highways Authority: No objection
Environmental Health: No objection
Neighbouring properties consulted: 5 (2 letters of support, 1 objection — privacy concern)
Tree Officer: No trees affected

POLICY BASIS:
Approved under: Moorfield District Local Plan 2019, Policies BE5 (extensions to dwellings),
DES1 (design quality) and National Planning Policy Framework para. 130.

APPEAL INFORMATION:
Right of appeal against conditions exists — must be lodged with Planning Inspectorate
within 6 months of this decision (by 22 September 2024).""",
    reasoning="""This is a planning permission decision notice from Moorfield District Council. I need to extract all key planning data.

Application details: MDD/2024/0284, decision 22 March 2024. Granted with conditions.

Applicant: Mr. & Mrs. R. Chakraborty, 14 Larkspur Lane, Moorfield MF4 7BT. Agent: Whitmore Architectural Services.

Development description: single-storey rear extension:
- Projection: 5.8m
- Maximum height: 3.9m
- Eaves height: 2.6m
- Feature: glazed roof lantern
- Also: internal open-plan remodel, 2 rooflights to rear slope

Decision: GRANTED WITH CONDITIONS.

5 conditions:
1. Start within 3 years → by 22 March 2027
2. Comply with approved drawings: WAS-2024-048 Rev C (site plan), 049 Rev B (floor plans), 050 Rev A (elevations)
3. Materials to match existing — samples pre-approval required
4. No obstruction of drainage channel during construction
5. Obscure glazing (min Level 4) to side elevation window — permanent condition

Consultation: Highways and Environmental Health no objection. 5 neighbours consulted: 2 support, 1 objection (privacy). Tree Officer no objection.

Policy basis: Moorfield District Local Plan 2019 policies BE5 and DES1; NPPF para 130.

Appeal rights: against conditions only; must be lodged with Planning Inspectorate within 6 months (by 22 September 2024).

Confidence: High.""",
    answer="""## Planning Application Extraction — MDD/2024/0284

### Application Details
| Field | Value |
|-------|-------|
| **LPA** | Moorfield District Council |
| **Reference** | MDD/2024/0284 |
| **Decision Date** | 22 March 2024 |
| **Decision** | **GRANTED WITH CONDITIONS** |

### Parties
| Role | Details |
|------|---------|
| **Applicant** | Mr. & Mrs. R. Chakraborty |
| **Site Address** | 14 Larkspur Lane, Moorfield, MF4 7BT |
| **Agent** | Whitmore Architectural Services, 3 Market Square, Moorfield |

### Development Description
| Parameter | Value |
|-----------|-------|
| **Type** | Single-storey rear extension |
| **Projection** | 5.8 metres |
| **Maximum Height** | 3.9 metres |
| **Eaves Height** | 2.6 metres |
| **Feature** | Glazed roof lantern |
| **Additional works** | Internal open-plan remodel; 2 rooflights to rear roof slope |

### Conditions
| # | Condition | Key Date/Requirement |
|---|-----------|---------------------|
| 1 | Development commencement | **By 22 March 2027** |
| 2 | Comply with approved drawings | WAS-2024-048 Rev C / 049 Rev B / 050 Rev A |
| 3 | External materials to match | Samples to LPA before construction |
| 4 | Drainage channel | No obstruction during construction |
| 5 | Side elevation window glazing | Obscure Level 4 minimum — **permanent** |

### Consultation Responses
| Consultee | Response |
|-----------|----------|
| Highways Authority | No objection |
| Environmental Health | No objection |
| Tree Officer | No objection |
| Neighbours (5 consulted) | 2 support, **1 objection** (privacy concern) |

### Policy Basis
| Document | Policy |
|----------|--------|
| Moorfield District Local Plan 2019 | BE5 (Extensions to dwellings) |
| Moorfield District Local Plan 2019 | DES1 (Design quality) |
| National Planning Policy Framework | Paragraph 130 |

### Appeal Rights
- **Right to appeal:** Against conditions only
- **Deadline:** 22 September 2024 (6 months from decision)
- **Body:** Planning Inspectorate

**Confidence: High**""",
    difficulty="medium"
))

print("Part 2 (examples 11-17) defined. Continuing to part 3...")

# ─── 18. Export / Customs Declaration ────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract all shipping and customs details from this export declaration:

COMMERCIAL INVOICE & PACKING LIST
Exporter (Shipper): Techform Precision Engineering Ltd
                   Unit 9, Pennine Industrial Park, Leeds LS12 4NP, UK
                   EORI: GB123456789000
                   VAT: GB 987 6543 21

Consignee:         Müller Maschinenbau GmbH
                   Industriestraße 47, 73230 Kirchheim unter Teck, Germany
                   DE EORI: DE4812937450

Invoice No: TPE-EXP-2024-0312
Invoice Date: 05 March 2024
Incoterm: DAP Kirchheim unter Teck
Currency: GBP
Country of Origin: United Kingdom
HS Code: 8466.93.00 (Parts for metal-working machinery)

GOODS DESCRIPTION:
Box 1 (of 3): Precision tooling fixtures, hardened steel
              Part No. TF-HTX-3204 (×6)     Unit value: £485.00    Line: £2,910.00
              Gross weight: 48.2 kg  |  Net weight: 44.0 kg  |  Dims: 80×60×40cm

Box 2 (of 3): CNC workholding clamps, stainless steel
              Part No. TF-WHC-1180 (×24)    Unit value: £67.50     Line: £1,620.00
              Gross weight: 31.5 kg  |  Net weight: 29.0 kg  |  Dims: 70×50×35cm

Box 3 (of 3): Calibration gauges, tungsten carbide
              Part No. TF-CAG-0088 (×12)    Unit value: £228.00    Line: £2,736.00
              Gross weight: 18.7 kg  |  Net weight: 17.2 kg  |  Dims: 50×40×30cm

TOTALS
Total Packages: 3 boxes
Total Gross Weight: 98.4 kg
Total Net Weight: 90.2 kg
TOTAL INVOICE VALUE: £7,266.00

Freight (prepaid by exporter to DAP): £420.00
Insurance: £36.33
CIF Value (for German import duty): £7,722.33

German import duty (HS 8466.93.00): 0% (EU MFN rate — UK has no UK-EU TCA duty benefit here)
German VAT (19%): £1,467.24 (payable by importer on import)""",
    reasoning="""This is a combined commercial invoice and packing list for an export from the UK to Germany. I need to extract all trade, customs and logistics data.

Exporter: Techform Precision Engineering Ltd, Leeds LS12 4NP. EORI GB123456789000, VAT GB 987 6543 21.
Consignee: Müller Maschinenbau GmbH, Kirchheim unter Teck, Germany. EORI DE4812937450.

Invoice TPE-EXP-2024-0312 dated 05 March 2024. Incoterm DAP (Delivered at Place) — exporter bears freight and risk to destination, but import duty/VAT is consignee's responsibility.

HS code 8466.93.00 = parts for metal-working machinery.

Line item verification:
Box 1: TF-HTX-3204 ×6 @ £485.00 = £2,910.00 ✓
Box 2: TF-WHC-1180 ×24 @ £67.50 = £1,620.00 ✓
Box 3: TF-CAG-0088 ×12 @ £228.00 = £2,736.00 ✓

Total value: 2,910 + 1,620 + 2,736 = £7,266.00 ✓

Weight check: 48.2 + 31.5 + 18.7 = 98.4 kg gross ✓; 44.0 + 29.0 + 17.2 = 90.2 kg net ✓

CIF value: £7,266 + £420 freight + £36.33 insurance = £7,722.33 ✓

German import duty: 0% (EU MFN rate — note the comment that UK-EU TCA doesn't apply here; UK goods may not benefit from preferential rate and are subject to MFN, which happens to be 0% for this HS code).

German import VAT: 19% of CIF value: £7,722.33 × 19% = £1,467.24 ✓

All arithmetic verified. Consignee will pay import VAT on import into Germany.

Confidence: High.""",
    answer="""## Export Declaration Extraction — TPE-EXP-2024-0312

### Trade Parties
| Role | Details |
|------|---------|
| **Exporter** | Techform Precision Engineering Ltd, Unit 9 Pennine Industrial Park, Leeds LS12 4NP |
| **Exporter EORI** | GB123456789000 |
| **Exporter VAT** | GB 987 6543 21 |
| **Consignee** | Müller Maschinenbau GmbH, Industriestraße 47, 73230 Kirchheim unter Teck, Germany |
| **Consignee EORI** | DE4812937450 |

### Shipment Details
| Field | Value |
|-------|-------|
| **Invoice Number** | TPE-EXP-2024-0312 |
| **Invoice Date** | 05 March 2024 |
| **Incoterm** | DAP Kirchheim unter Teck |
| **Currency** | GBP |
| **Country of Origin** | United Kingdom |
| **HS Code** | 8466.93.00 — Parts for metal-working machinery |

### Line Items & Packing
| Box | Part Number | Description | Qty | Unit Value | Line Value | Gross Wt | Net Wt | Dimensions |
|-----|-------------|-------------|-----|-----------|-----------|---------|--------|------------|
| 1 | TF-HTX-3204 | Precision tooling fixtures, hardened steel | 6 | £485.00 | £2,910.00 | 48.2 kg | 44.0 kg | 80×60×40 cm |
| 2 | TF-WHC-1180 | CNC workholding clamps, stainless steel | 24 | £67.50 | £1,620.00 | 31.5 kg | 29.0 kg | 70×50×35 cm |
| 3 | TF-CAG-0088 | Calibration gauges, tungsten carbide | 12 | £228.00 | £2,736.00 | 18.7 kg | 17.2 kg | 50×40×30 cm |

### Totals
| Metric | Value |
|--------|-------|
| Total Packages | 3 boxes |
| Total Gross Weight | **98.4 kg** |
| Total Net Weight | **90.2 kg** |
| **Invoice Value** | **£7,266.00** |

### Customs Valuation (German Import)
| Component | Amount |
|-----------|--------|
| Invoice Value (FOB) | £7,266.00 |
| Freight (prepaid) | £420.00 |
| Insurance | £36.33 |
| **CIF Value (duty basis)** | **£7,722.33** |
| German Import Duty (HS 8466.93) | **0%** (EU MFN rate) |
| German Import VAT (19%) | **£1,467.24** — payable by consignee on import |

### Arithmetic Verification
- Line totals: £2,910 + £1,620 + £2,736 = £7,266 ✓
- Gross weight: 48.2 + 31.5 + 18.7 = 98.4 kg ✓
- Net weight: 44.0 + 29.0 + 17.2 = 90.2 kg ✓
- CIF: £7,266 + £420 + £36.33 = £7,722.33 ✓
- German VAT: £7,722.33 × 19% = £1,467.24 ✓

**Confidence: High**""",
    difficulty="hard"
))

# ─── 19. Research Grant Application ──────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract the key details from this research grant summary:

GRANT APPLICATION SUMMARY
Funding Body: UK Research and Innovation (UKRI) — EPSRC
Call: Trustworthy Autonomous Systems — Safety and Reliability (TAS-SR)
Application Reference: EP/Y041823/1
Status: FUNDED

PROJECT TITLE:
"SAFE-AV: Formal Verification Methods for Safety-Critical Decision Making
in Autonomous Vehicles under Uncertainty"

Lead Institution: Cranmore University (Engineering and Computing Faculty)
Principal Investigator: Professor Helena Blackwell
Co-Investigators:
  Dr. Niamh Costello (Cranmore University — School of Mathematics)
  Prof. James Whitby (University of Bramford — Dept. of Transport Engineering)

PROJECT DURATION: 36 months
Start Date: 01 September 2024
End Date: 31 August 2027

BUDGET SUMMARY (£)
                        Cranmore    Bramford    TOTAL
Staff — PDRAs              187,420     94,860   282,280
Staff — Technicians         38,200          0    38,200
Directly Incurred — Travel  12,400      8,200    20,600
Directly Incurred — Other   18,750      6,400    25,150
Directly Allocated (Estates) 34,820    18,340    53,160
Directly Allocated (Indirect) 96,450   48,220   144,670
Total Requested from EPSRC  388,040   175,020   563,060
Institution Contribution        0           0        0
TOTAL PROJECT COST          388,040   175,020   563,060

INDUSTRIAL PARTNER: Autonomous Drive Systems plc (cash contribution: £30,000)

OBJECTIVES:
1. Develop formal verification framework for AV decision modules
2. Create probabilistic safety models validated against SOTIF (ISO 21448)
3. Demonstrate on hardware-in-the-loop testbed (3 AV scenarios)
4. Produce open-source toolkit and 3 BSc/MSc student placements""",
    reasoning="""This is a funded EPSRC grant application summary for autonomous vehicle safety research. I need to extract all key administrative, financial and scientific details.

Funding body: UKRI-EPSRC, Call TAS-SR (Trustworthy Autonomous Systems — Safety and Reliability). Reference EP/Y041823/1. Status: FUNDED.

Project: SAFE-AV — formal verification for AV decision making under uncertainty.

People:
- PI: Prof. Helena Blackwell, Cranmore University
- Co-I: Dr. Niamh Costello (Cranmore, Maths)
- Co-I: Prof. James Whitby (Bramford, Transport Engineering)

Duration: 36 months, 01 Sep 2024 – 31 Aug 2027.

Budget verification:
Staff PDRAs: 187,420 + 94,860 = £282,280 ✓
Staff Technicians: 38,200 + 0 = £38,200 ✓
Travel: 12,400 + 8,200 = £20,600 ✓
Other DI: 18,750 + 6,400 = £25,150 ✓
Estates (DA): 34,820 + 18,340 = £53,160 ✓
Indirect (DA): 96,450 + 48,220 = £144,670 ✓

Cranmore total: 187,420 + 38,200 + 12,400 + 18,750 + 34,820 + 96,450 = 388,040 ✓
Bramford total: 94,860 + 0 + 8,200 + 6,400 + 18,340 + 48,220 = 176,020. But stated is £175,020 — discrepancy of £1,000. Let me recheck: 94,860 + 8,200 = 103,060; + 6,400 = 109,460; + 18,340 = 127,800; + 48,220 = 176,020. Stated total is £175,020 — there is a £1,000 discrepancy in the Bramford column. I should flag this.

Overall total: 388,040 + 175,020 = £563,060 (as stated). But if Bramford is £176,020, total should be £564,060.

Industrial partner: Autonomous Drive Systems plc, £30,000 cash — this is separate from EPSRC grant (in-kind/cash contribution not in grant budget).

Objectives: 4 stated. Focus areas: formal verification framework, probabilistic safety models (SOTIF ISO 21448), hardware-in-the-loop testbed, open-source toolkit + student placements.

Confidence: High for most data; £1,000 Bramford arithmetic discrepancy flagged.""",
    answer="""## Research Grant Extraction — EP/Y041823/1

### Grant Overview
| Field | Value |
|-------|-------|
| **Funding Body** | UK Research and Innovation (UKRI) — EPSRC |
| **Call** | Trustworthy Autonomous Systems — Safety and Reliability (TAS-SR) |
| **Reference** | EP/Y041823/1 |
| **Status** | **FUNDED** |
| **Project Title** | SAFE-AV: Formal Verification Methods for Safety-Critical Decision Making in Autonomous Vehicles under Uncertainty |

### Project Team
| Role | Name | Institution | Department |
|------|------|-------------|------------|
| **Principal Investigator** | Prof. Helena Blackwell | Cranmore University | Engineering and Computing |
| Co-Investigator | Dr. Niamh Costello | Cranmore University | School of Mathematics |
| Co-Investigator | Prof. James Whitby | University of Bramford | Transport Engineering |

### Timeline
| Milestone | Date |
|-----------|------|
| **Start Date** | 01 September 2024 |
| **End Date** | 31 August 2027 |
| **Duration** | 36 months |

### Budget Summary (£)
| Cost Category | Cranmore | Bramford | Total |
|---------------|---------|---------|-------|
| Staff — PDRAs | 187,420 | 94,860 | 282,280 |
| Staff — Technicians | 38,200 | 0 | 38,200 |
| Travel (Directly Incurred) | 12,400 | 8,200 | 20,600 |
| Other (Directly Incurred) | 18,750 | 6,400 | 25,150 |
| Estates (Directly Allocated) | 34,820 | 18,340 | 53,160 |
| Indirect Costs (Directly Allocated) | 96,450 | 48,220 | 144,670 |
| **Total Requested** | **388,040** | **175,020** | **563,060** |

> **Arithmetic Note:** Independent summation of Bramford column yields **£176,020**, not £175,020 as stated — a discrepancy of **£1,000**. This should be verified with the original application. The total of £563,060 is based on the stated Bramford figure.

### Industrial Collaboration
| Partner | Contribution Type | Amount |
|---------|------------------|--------|
| Autonomous Drive Systems plc | Cash contribution | **£30,000** |

*(Industrial cash contribution is separate from EPSRC grant — not included in grant budget)*

### Research Objectives
| # | Objective |
|---|-----------|
| 1 | Develop formal verification framework for AV decision modules |
| 2 | Create probabilistic safety models validated against SOTIF (ISO 21448) |
| 3 | Demonstrate on hardware-in-the-loop testbed (3 AV scenarios) |
| 4 | Produce open-source toolkit + 3 BSc/MSc student placements |

**Confidence: High** — one budget arithmetic discrepancy flagged for verification""",
    difficulty="hard"
))

# ─── 20. Supplier Quality Agreement ──────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract quality and compliance requirements from this supplier agreement:

SUPPLIER QUALITY AGREEMENT (SQA)
Document No: SQA-2024-0047
Revision: 2
Effective Date: 01 February 2024
Customer: Helix Aerospace Components Ltd
Supplier: Precision Castings International Ltd (PCI)

1. SCOPE
Applies to all castings supplied under Part Numbers HX-CAST-2200 series
(aluminium alloy investment castings for aerospace structural applications).

2. QUALITY MANAGEMENT SYSTEM
Supplier shall maintain and demonstrate current certification to:
  AS9100 Rev D (aerospace QMS) — mandatory
  ISO 9001:2015 — minimum baseline
  NADCAP accreditation for Heat Treatment and NDT — mandatory

3. FIRST ARTICLE INSPECTION (FAI)
Full AS9102 FAI required for all new part numbers and after:
  - Engineering change orders (all classes)
  - Process changes affecting part characteristics
  - Manufacturing breaks exceeding 24 months
FAI package to be submitted to and approved by Helix before production release.

4. INCOMING INSPECTION / CERTIFICATES
Each delivery must include:
  - Certificate of Conformance (CoC) signed by Supplier QA Manager
  - Material Test Reports (MTR) traceable to heat/lot numbers
  - NDT Reports (where applicable per drawing callout)
  - Dimensional reports (100% for new parts; sample plan AQL 0.65 thereafter)

5. NONCONFORMANCE MANAGEMENT
Supplier to report any nonconformance within 24 hours of detection.
8D corrective action report required within 10 business days.
Three (3) repeat nonconformances of same root cause in 12 months triggers
Supplier Improvement Plan (SIP) and potential supplier re-qualification.

6. APPROVED MATERIAL SOURCES
Alloy stock: Only from Helix-approved material sources (Annex A).
Substitution requires prior written approval from Helix Supply Chain Director.

7. PERFORMANCE TARGETS
On-Time Delivery (OTD): ≥95%
PPM Defect Rate: ≤500 PPM
Customer Returns: 0 tolerance for safety-critical defects
Quarterly Business Review (QBR): mandatory attendance""",
    reasoning="""This is a Supplier Quality Agreement between Helix Aerospace Components (customer) and Precision Castings International (supplier). I need to extract all quality and compliance requirements.

Scope: aluminium alloy investment castings, HX-CAST-2200 series, for aerospace structural applications.

Required certifications:
- AS9100 Rev D (aerospace QMS) — mandatory
- ISO 9001:2015 — minimum baseline
- NADCAP for Heat Treatment AND NDT — both mandatory

FAI (First Article Inspection):
- AS9102 standard
- Required for: new part numbers, all ECOs, process changes, manufacturing breaks >24 months
- Full approval before production release

Documentation per delivery:
- Certificate of Conformance (signed by QA Manager)
- Material Test Reports (heat/lot traceable)
- NDT reports (where drawing-required)
- Dimensional reports: 100% for new parts; AQL 0.65 sample plan thereafter

Nonconformance response times:
- Report within 24 hours
- 8D corrective action within 10 business days
- 3 repeat nonconformances (same root cause) in 12 months → Supplier Improvement Plan + possible re-qualification

Material controls: approved sources only (Annex A), substitution needs SCM Director written approval.

Performance targets:
- OTD ≥95%
- PPM ≤500
- Safety-critical defect customer returns: zero tolerance
- QBRs: mandatory

Confidence: High.""",
    answer="""## Supplier Quality Agreement Extraction — SQA-2024-0047

### Document Details
| Field | Value |
|-------|-------|
| **Document No** | SQA-2024-0047 Rev 2 |
| **Effective Date** | 01 February 2024 |
| **Customer** | Helix Aerospace Components Ltd |
| **Supplier** | Precision Castings International Ltd (PCI) |
| **Scope** | HX-CAST-2200 series — aluminium alloy investment castings (aerospace structural) |

### Required Certifications
| Certification | Scope | Status |
|---------------|-------|--------|
| **AS9100 Rev D** | Aerospace QMS | **Mandatory** |
| ISO 9001:2015 | Baseline QMS | Minimum requirement |
| **NADCAP — Heat Treatment** | Special process | **Mandatory** |
| **NADCAP — NDT** | Special process | **Mandatory** |

### First Article Inspection (FAI)
- **Standard:** AS9102 (full FAI)
- **Required when:**
  - New part numbers
  - Engineering Change Orders (all classes)
  - Process changes affecting part characteristics
  - Manufacturing breaks exceeding **24 months**
- **Release gate:** FAI package approved by Helix before production

### Delivery Documentation Requirements
| Document | Requirement |
|----------|-------------|
| Certificate of Conformance (CoC) | Signed by Supplier QA Manager — every shipment |
| Material Test Reports (MTR) | Traceable to heat/lot number — every shipment |
| NDT Reports | Where called out on drawing |
| Dimensional Reports | **100%** for new parts; **AQL 0.65** sample plan thereafter |

### Nonconformance Requirements
| Event | Timeframe |
|-------|-----------|
| Nonconformance notification to Helix | Within **24 hours** of detection |
| 8D Corrective Action Report | Within **10 business days** |
| Trigger for Supplier Improvement Plan | 3 repeat NCs (same root cause) in 12 months |
| Consequence of SIP trigger | SIP initiation + potential re-qualification |

### Material Controls
- Alloy stock from **Helix-approved sources only** (Annex A)
- Source substitution requires **written approval from Helix Supply Chain Director**

### Performance Targets
| KPI | Target |
|-----|--------|
| **On-Time Delivery (OTD)** | ≥95% |
| **PPM Defect Rate** | ≤500 PPM |
| **Safety-Critical Returns** | **Zero tolerance** |
| Quarterly Business Review | Mandatory attendance |

**Confidence: High**""",
    difficulty="medium"
))

# ─── 21. Probate / Will Summary ───────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract the key provisions from this will summary:

LAST WILL AND TESTAMENT — SUMMARY PREPARED BY SOLICITORS
Testator: ELEANOR MARGARET FORSYTHE
Date of Will: 14 November 2020
Date of Death: 02 January 2024
Probate Reference: C0192847/2024

EXECUTORS:
1. Gerald Raymond Forsythe (Husband) — Primary executor
2. Whitmore & Partners Solicitors, 8 Crown Street, Bristol BS1 4HW — Professional executor

ESTATE SUMMARY (as at grant of probate):
Property — 4 Clifton Park Road, Bristol BS8 3HX: £875,000
Investment Portfolio (Hargreaves Lansdown): £284,500
NS&I Premium Bonds: £32,000
Current/savings accounts (3 banks): £47,830
Personal chattels (jewellery, antiques — agreed value): £28,400
Less: outstanding mortgage: (£124,780)
Less: funeral expenses: (£8,240)
NET ESTATE: £1,134,710

INHERITANCE TAX:
Estate value: £1,134,710
Nil-rate band (NRB): £325,000
Residence NRB (RNRB): £175,000
Transferred NRB (deceased's spouse): £325,000
Transferred RNRB: £175,000
Total exemptions/allowances: £1,000,000
Taxable estate: £134,710
IHT at 40%: £53,884

SPECIFIC BEQUESTS:
- Jewellery collection → to Sophia Forsythe-Williams (daughter)
- Antique writing desk → to Thomas Forsythe (son)
- NS&I Premium Bonds → to Clifton Animal Rescue Trust (charity)

RESIDUARY ESTATE (after IHT, bequests and costs):
Shared equally between: Gerald Raymond Forsythe (Husband) and two children
(Sophia Forsythe-Williams and Thomas Forsythe) — one-third each.""",
    reasoning="""This is a probate/will summary prepared by solicitors for Eleanor Margaret Forsythe, who died 02 January 2024. I need to extract all key estate and testamentary provisions.

Testator: Eleanor Margaret Forsythe. Will dated 14 November 2020, so made 3+ years before death. Probate ref C0192847/2024.

Executors: Gerald Forsythe (husband, primary) and Whitmore & Partners Solicitors (professional).

Estate assets and liabilities:
- Property Bristol BS8: £875,000
- Investment portfolio: £284,500
- NS&I Premium Bonds: £32,000
- Bank accounts: £47,830
- Personal chattels: £28,400
Gross assets: 875,000 + 284,500 + 32,000 + 47,830 + 28,400 = £1,267,730
Less mortgage: (£124,780)
Less funeral: (£8,240)
Net estate: 1,267,730 - 124,780 - 8,240 = £1,134,710 ✓

IHT calculation:
NRB own: £325,000
RNRB own: £175,000
Transferred NRB (from deceased spouse — available as Eleanor predeceased Gerald? Wait — Gerald is the surviving husband. So Eleanor is the first to die. The transferred NRB would come from a previously deceased spouse, not Gerald. Actually, looking again: "Transferred NRB (deceased's spouse)" — this may refer to Eleanor's previously deceased spouse before Gerald, or it could be that Gerald had a prior deceased spouse who transferred their NRB to the estate. This needs careful reading. More likely this is a standard calculation where Eleanor's estate uses both her own NRB and any transferred NRB from a prior deceased spouse. I'll note this for accuracy but proceed with the stated figures.
Total allowances: 325,000 + 175,000 + 325,000 + 175,000 = £1,000,000 ✓
Taxable estate: 1,134,710 - 1,000,000 = £134,710 ✓
IHT at 40%: £134,710 × 40% = £53,884 ✓

Specific bequests:
- Jewellery → Sophia Forsythe-Williams (daughter)
- Antique writing desk → Thomas Forsythe (son)
- NS&I Premium Bonds (£32,000) → Clifton Animal Rescue Trust (charity — no IHT on charitable bequests, but bonds already passed through estate total)

Residuary: shared equally thirds among husband Gerald + daughter Sophia + son Thomas.

Confidence: High for stated figures; transferred NRB provenance slightly ambiguous but calculation is consistent.""",
    answer="""## Will / Probate Extraction — Eleanor Margaret Forsythe

### Testamentary Details
| Field | Value |
|-------|-------|
| **Testator** | Eleanor Margaret Forsythe |
| **Date of Will** | 14 November 2020 |
| **Date of Death** | 02 January 2024 |
| **Probate Reference** | C0192847/2024 |

### Executors
| Role | Name |
|------|------|
| Primary Executor | Gerald Raymond Forsythe (Husband) |
| Professional Executor | Whitmore & Partners Solicitors, 8 Crown Street, Bristol BS1 4HW |

### Estate Assets & Liabilities
| Asset / Liability | Value |
|-------------------|-------|
| Property — 4 Clifton Park Road, Bristol BS8 3HX | £875,000 |
| Investment Portfolio (Hargreaves Lansdown) | £284,500 |
| NS&I Premium Bonds | £32,000 |
| Bank / Savings Accounts (3 banks) | £47,830 |
| Personal Chattels (jewellery, antiques) | £28,400 |
| **Gross Estate** | **£1,267,730** |
| Less: Outstanding Mortgage | (£124,780) |
| Less: Funeral Expenses | (£8,240) |
| **Net Estate** | **£1,134,710** |

### Inheritance Tax Computation
| Allowance | Amount |
|-----------|--------|
| Nil-Rate Band (own) | £325,000 |
| Residence NRB (own) | £175,000 |
| Transferred NRB | £325,000 |
| Transferred RNRB | £175,000 |
| **Total Exemptions** | **£1,000,000** |
| **Taxable Estate** | **£134,710** |
| **IHT at 40%** | **£53,884** |

### Specific Bequests
| Bequest | Beneficiary | Type |
|---------|-------------|------|
| Jewellery collection | Sophia Forsythe-Williams (daughter) | Personal chattel |
| Antique writing desk | Thomas Forsythe (son) | Personal chattel |
| NS&I Premium Bonds (£32,000) | Clifton Animal Rescue Trust | Charitable bequest |

### Residuary Estate Distribution
After IHT, specific bequests and administration costs, residue divided **equally in three shares:**
| Beneficiary | Relationship | Share |
|-------------|-------------|-------|
| Gerald Raymond Forsythe | Husband | 1/3 |
| Sophia Forsythe-Williams | Daughter | 1/3 |
| Thomas Forsythe | Son | 1/3 |

### Arithmetic Verification
- Gross estate: £875,000 + £284,500 + £32,000 + £47,830 + £28,400 = £1,267,730 ✓
- Net estate: £1,267,730 − £124,780 − £8,240 = £1,134,710 ✓
- Taxable estate: £1,134,710 − £1,000,000 = £134,710 ✓
- IHT: £134,710 × 40% = £53,884 ✓

**Confidence: High**""",
    difficulty="hard"
))

# ─── 22. Environmental Assessment ────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract key findings from this environmental screening report:

PRELIMINARY ECOLOGICAL APPRAISAL (PEA)
Report Reference: WSP-ENV-2024-0831
Site: Former Whitmore Foundry, Canal Road, Castleford WF10 3LX
Client: Bridgewater Homes Ltd
Prepared by: WSP Environmental Ltd
Date of Survey: 09 February 2024 & 16 February 2024
Planning Application: Proposed 94-dwelling residential development

HABITATS ASSESSED:
The 2.4 ha site comprises:
  Hardstanding / buildings:  58%  (1.39 ha)
  Scrub vegetation:          22%  (0.53 ha)
  Ruderal / amenity grass:   14%  (0.34 ha)
  Standing water (pond):      6%  (0.14 ha)

PROTECTED SPECIES FINDINGS:

Bats (Chiroptera):
  - 5 bat species detected via transect surveys using Anabat Swift detectors
  - Species confirmed: Common Pipistrelle, Soprano Pipistrelle, Brown Long-eared,
    Noctule, Nathusius' Pipistrelle
  - 2 existing buildings contain potential roost features (PRF) — further emergence
    survey required (May–August 2024) before planning determination
  - Commuting routes along canal corridor (north boundary) — moderate–high activity

Great Crested Newt (GCN):
  - Pond (0.14 ha) surveyed: eDNA sample positive — GCN presence confirmed
  - Habitat Suitability Index (HSI): 0.72 (Good)
  - Mitigation: Natural England District Level Licensing (DLL) scheme recommended

Breeding Birds:
  - 14 species recorded during breeding bird survey
  - Schedule 1 species: Peregrine Falcon (foraging — no nesting confirmed)
  - Section 41 species: Skylark (1 territory), Lapwing (1 territory, scrub edge)

ECOLOGICAL CONSTRAINTS SUMMARY:
  HIGH:   Great Crested Newt, Bat roosts (buildings)
  MEDIUM: Commuting bats, Breeding birds (S41 species)
  LOW:    General invertebrates, common breeding birds

RECOMMENDED ACTIONS:
1. Bat emergence/re-entry surveys — 3 visits May–August 2024 (EPS licence likely required)
2. GCN — enrol in DLL scheme; engage Natural England
3. Breeding bird surveys to commence March 2024
4. Retain and enhance pond; create 0.3 ha of new freshwater habitat""",
    reasoning="""This is a Preliminary Ecological Appraisal for a former industrial site proposed for 94-unit residential development. I need to extract all ecological findings and constraint ratings.

Site: Former Whitmore Foundry, Canal Road, Castleford WF10 3LX. Area: 2.4 ha. Surveys on 09 and 16 February 2024.

Habitat breakdown:
- Hardstanding/buildings: 58% = 1.39 ha
- Scrub: 22% = 0.53 ha
- Ruderal/amenity grass: 14% = 0.34 ha
- Standing water (pond): 6% = 0.14 ha
Check: 1.39 + 0.53 + 0.34 + 0.14 = 2.40 ha ✓ and 58+22+14+6 = 100% ✓

Protected species:
Bats: 5 species confirmed (Common Pipistrelle, Soprano Pipistrelle, Brown Long-eared, Noctule, Nathusius' Pipistrelle). 2 buildings have Potential Roost Features (PRF) — emergence surveys needed May-Aug 2024 before planning determination. Canal corridor commuting route — moderate-high activity.

GCN: eDNA positive (confirmed presence). HSI 0.72 = Good category. Recommended mitigation: Natural England District Level Licensing scheme.

Breeding birds: 14 species. Schedule 1: Peregrine Falcon (foraging, no nesting). Section 41 (priority species): Skylark (1 territory), Lapwing (1 territory, scrub edge).

Constraint ratings:
- HIGH: GCN, bat roosts in buildings
- MEDIUM: Commuting bats, breeding birds (S41)
- LOW: invertebrates, common breeding birds

Actions: 4 recommended. Key planning constraint: bat emergence surveys must complete before planning determination — this could delay consent until September 2024 at earliest.

Confidence: High.""",
    answer="""## Ecological Assessment Extraction — WSP-ENV-2024-0831

### Report Details
| Field | Value |
|-------|-------|
| **Reference** | WSP-ENV-2024-0831 |
| **Site** | Former Whitmore Foundry, Canal Road, Castleford WF10 3LX |
| **Client** | Bridgewater Homes Ltd |
| **Consultant** | WSP Environmental Ltd |
| **Survey Dates** | 09 & 16 February 2024 |
| **Proposed Development** | 94-dwelling residential development |
| **Total Site Area** | 2.4 ha |

### Habitat Composition
| Habitat Type | % Area | Hectares |
|-------------|--------|----------|
| Hardstanding / Buildings | 58% | 1.39 ha |
| Scrub Vegetation | 22% | 0.53 ha |
| Ruderal / Amenity Grass | 14% | 0.34 ha |
| Standing Water (Pond) | 6% | 0.14 ha |
| **Total** | **100%** | **2.40 ha** |

### Protected Species Findings

**Bats (Chiroptera)**
| Parameter | Finding |
|-----------|---------|
| Species detected | **5** (Common Pipistrelle, Soprano Pipistrelle, Brown Long-eared, Noctule, Nathusius' Pipistrelle) |
| Buildings with PRF | **2** — emergence survey required |
| Survey equipment | Anabat Swift detectors |
| Canal corridor activity | Moderate–High |
| Planning implication | Emergence surveys (May–Aug 2024) **must complete before planning determination** |

**Great Crested Newt (GCN)**
| Parameter | Finding |
|-----------|---------|
| eDNA result | **Positive — confirmed presence** |
| Habitat Suitability Index | **0.72 (Good)** |
| Recommended mitigation | Natural England District Level Licensing (DLL) scheme |

**Breeding Birds**
| Category | Species | Status |
|----------|---------|--------|
| Total species recorded | 14 | — |
| Schedule 1 | Peregrine Falcon | Foraging — no nesting confirmed |
| Section 41 (priority) | Skylark | 1 territory |
| Section 41 (priority) | Lapwing | 1 territory (scrub edge) |

### Ecological Constraints Summary
| Constraint Level | Species/Feature |
|-----------------|-----------------|
| **HIGH** | Great Crested Newt; Bat roosts in buildings |
| **MEDIUM** | Commuting bats (canal corridor); Breeding birds (S41 species) |
| **LOW** | General invertebrates; Common breeding birds |

### Recommended Actions
| # | Action | Timing |
|---|--------|--------|
| 1 | Bat emergence/re-entry surveys (3 visits) — EPS licence likely required | May–August 2024 |
| 2 | GCN: Enrol in DLL scheme; engage Natural England | Immediate |
| 3 | Breeding bird surveys | From March 2024 |
| 4 | Retain pond; create 0.3 ha new freshwater habitat | Design stage |

> **Critical Planning Risk:** Bat emergence surveys must conclude before planning determination — consent unlikely before **September 2024** at earliest.

**Confidence: High**""",
    difficulty="medium"
))

# ─── 23. Board Resolution ─────────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract all decisions and parties from this board resolution:

WRITTEN RESOLUTION OF THE BOARD OF DIRECTORS
Ferncastle Group Holdings Ltd
Company No: 10294857

Date: 28 February 2024

The following Directors, constituting a quorum of the Board, have signed this
written resolution in accordance with the Company's Articles of Association (Art. 14.3):

Signatories:
  Victoria Chen — Non-Executive Chair
  Robert Ashworth — Chief Executive Officer
  Preethi Nair — Chief Financial Officer
  James Holloway — Non-Executive Director (Audit Committee Chair)

RESOLUTIONS PASSED:

1. APPROVAL OF FY2023 FINANCIAL STATEMENTS
   RESOLVED: That the consolidated and company financial statements for the year
   ended 31 December 2023 be and are hereby approved for signature and filing
   with Companies House.

2. APPOINTMENT OF AUDITORS
   RESOLVED: That Grant Thornton UK LLP be reappointed as statutory auditors for
   the financial year ending 31 December 2024 at an agreed audit fee of £186,000,
   an increase of £14,000 (8.1%) from the prior year fee of £172,000.

3. INTERIM DIVIDEND
   RESOLVED: That an interim dividend of 6.5 pence per ordinary share be declared,
   payable on 29 March 2024 to shareholders on the register at close of business
   on 15 March 2024 (record date). Total dividend liability: £2,847,500 based on
   438,076,923 ordinary shares in issue.

4. BANKING MANDATE UPDATE
   RESOLVED: That the Company's banking mandate with Lloyds Bank plc
   (Account 28473920) be updated to add Ms. Preethi Nair as authorised signatory
   (Level 2 — up to £500,000), with immediate effect.

5. AUTHORITY TO EXECUTE CONTRACTS
   RESOLVED: That the CEO and CFO acting jointly be authorised to execute contracts
   up to £5,000,000 in value without further Board approval, effective 01 March 2024.

All resolutions passed unanimously.""",
    reasoning="""This is a written board resolution for Ferncastle Group Holdings Ltd. I need to extract all resolutions, parties and key data points.

Company: Ferncastle Group Holdings Ltd, Co. No. 10294857. Date: 28 February 2024.

Signing directors (quorum per Art. 14.3): Victoria Chen (NE Chair), Robert Ashworth (CEO), Preethi Nair (CFO), James Holloway (NED, Audit Committee Chair). All 4 signed = unanimous.

5 resolutions:

1. FY2023 Financial Statements — approved for Companies House filing (year ended 31 Dec 2023).

2. Auditor reappointment — Grant Thornton UK LLP for FY2024 at £186,000. Prior year: £172,000. Increase: £14,000 = 8.1%. Check: (186,000-172,000)/172,000 = 14,000/172,000 = 8.14% ≈ 8.1% ✓

3. Interim dividend — 6.5p per ordinary share, payable 29 March 2024, record date 15 March 2024. Total liability: £2,847,500. Check: 438,076,923 shares × £0.065 = £28,475,000... wait that's £28.5M. But stated is £2,847,500. Let me recalculate: 438,076,923 × 0.065 = 28,475,000 — that's £28.475M not £2,847,500. There's a factor of 10 discrepancy. Actually if the share count is 43,807,692 × 0.065 = £2,847,500. So the stated share count of 438,076,923 is inconsistent with the dividend total of £2,847,500. This is a material discrepancy I must flag.

4. Banking mandate — add Preethi Nair (CFO) as Level 2 authorised signatory (up to £500k) on Lloyds Bank account 28473920.

5. Contract authority — CEO + CFO jointly can approve contracts up to £5M without further Board approval, effective 01 March 2024.

All resolutions passed unanimously.

Confidence: High for most items; dividend arithmetic discrepancy flagged.""",
    answer="""## Board Resolution Extraction — Ferncastle Group Holdings Ltd

### Document Details
| Field | Value |
|-------|-------|
| **Company** | Ferncastle Group Holdings Ltd |
| **Company No** | 10294857 |
| **Resolution Date** | 28 February 2024 |
| **Basis** | Written resolution per Articles of Association, Art. 14.3 |
| **Vote** | Unanimous |

### Signing Directors (Quorum)
| Name | Role |
|------|------|
| Victoria Chen | Non-Executive Chair |
| Robert Ashworth | Chief Executive Officer |
| Preethi Nair | Chief Financial Officer |
| James Holloway | Non-Executive Director (Audit Committee Chair) |

### Resolutions

**Resolution 1 — FY2023 Financial Statements**
- Consolidated and company financial statements for year ended **31 December 2023** approved for signature and Companies House filing

**Resolution 2 — Auditor Reappointment**
| Item | Value |
|------|-------|
| Auditor | Grant Thornton UK LLP |
| Period | Year ending 31 December 2024 |
| Agreed Fee | **£186,000** |
| Prior Year Fee | £172,000 |
| Fee Increase | £14,000 (+8.1%) |

**Resolution 3 — Interim Dividend**
| Item | Value |
|------|-------|
| Dividend Rate | **6.5 pence per ordinary share** |
| Record Date | 15 March 2024 |
| Payment Date | 29 March 2024 |
| Total Dividend Liability | £2,847,500 |
| Shares in Issue (stated) | 438,076,923 |

> **Arithmetic Discrepancy:** 438,076,923 shares × £0.065 = **£28,475,000**, not £2,847,500. The stated total is consistent with approximately 43,807,692 shares. The share count or total liability figure should be verified — likely a typographical error in the stated share count.

**Resolution 4 — Banking Mandate**
| Item | Detail |
|------|--------|
| Bank | Lloyds Bank plc |
| Account | 28473920 |
| New Signatory | Preethi Nair (CFO) |
| Authority Level | Level 2 — up to **£500,000** |
| Effective | Immediately |

**Resolution 5 — Contract Execution Authority**
| Item | Detail |
|------|--------|
| Authorised signatories | CEO + CFO acting **jointly** |
| Contract value limit | Up to **£5,000,000** (no further Board approval needed) |
| Effective date | 01 March 2024 |

**Confidence: High** — one dividend arithmetic discrepancy flagged for verification""",
    difficulty="hard"
))

print("Part 3 (examples 18-23) defined. Continuing to part 4...")

# ─── 24. Laboratory Test Report ───────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract all test results and pass/fail status from this material test report:

MATERIAL TEST REPORT
Report No: MTR-2024-0441
Issued by: Castleford Materials Testing Laboratory Ltd
UKAS Accreditation No: 4721 (ISO/IEC 17025:2017)
Client: Bridgeford Steel Fabrications Ltd
Order No: BSF-LAB-2024-177
Date of Testing: 06 March 2024
Report Date: 11 March 2024

MATERIAL IDENTIFICATION
Material: Structural Carbon Steel — Grade S355JR
Standard: BS EN 10025-2:2019
Heat No: 47831A
Batch / Coil No: C-2024-0882
Product Form: Plate, 25mm thickness

MECHANICAL TEST RESULTS

Test: TENSILE TEST (BS EN ISO 6892-1)
Specimen orientation: Longitudinal
Gauge Length: 200mm
  Yield Strength (ReH): 398 MPa     [Spec. min: 355 MPa]   PASS
  Tensile Strength (Rm): 541 MPa    [Spec: 470–630 MPa]    PASS
  Elongation (A): 26.4%             [Spec. min: 22%]        PASS
  Reduction of Area: 68.2%          [No requirement]        INFO

Test: CHARPY IMPACT TEST (BS EN ISO 148-1)
Test Temperature: -20°C (JR designation requirement)
3 Specimens tested — Individual / Average results:
  Specimen 1: 187 J
  Specimen 2: 164 J
  Specimen 3: 201 J
  Average: 184 J    [Spec. min avg: 27 J; min individual: 19 J]    PASS

CHEMICAL ANALYSIS (Ladle Analysis)
Element    Result    Specification Max    Status
C          0.181%    0.200%               PASS
Si         0.312%    0.550%               PASS
Mn         1.482%    1.600%               PASS
P          0.014%    0.035%               PASS
S          0.009%    0.040%               PASS
N          0.005%    0.012%               PASS
Carbon Equivalent (CEV): 0.431%  [Max: 0.450%]  PASS

DIMENSIONAL CHECK
Thickness measured (3 points): 24.9mm, 25.1mm, 25.0mm
Tolerance per EN 10029: ±0.7mm for 25mm plate    All PASS""",
    reasoning="""This is a UKAS-accredited material test report for S355JR structural steel plate. I need to extract all test results and verify pass/fail determinations.

Lab: Castleford Materials Testing Laboratory Ltd, UKAS 4721 (ISO/IEC 17025:2017). Testing 06 Mar 2024, report 11 Mar 2024.

Material: S355JR structural steel, plate 25mm, Heat 47831A, Batch C-2024-0882. Standard BS EN 10025-2:2019.

Mechanical tests:

Tensile test (BS EN ISO 6892-1), longitudinal:
- Yield strength 398 MPa vs min 355 MPa: margin = +43 MPa = +12.1% above minimum → PASS ✓
- Tensile strength 541 MPa vs spec 470–630 MPa: within range → PASS ✓
- Elongation 26.4% vs min 22%: margin = +4.4% above minimum → PASS ✓
- Reduction of area 68.2% — informational only, no spec requirement

Charpy impact test (BS EN ISO 148-1) at -20°C:
Individual results: 187, 164, 201 J
Average: (187+164+201)/3 = 552/3 = 184 J ✓ (stated 184 J ✓)
Spec: min avg 27 J, min individual 19 J. All three specimens far exceed minimums: lowest individual 164 J vs 19 J min → PASS ✓ (extremely high margin — energy absorbed is ~8.6× minimum)

Chemical analysis:
C: 0.181% vs 0.200% max → PASS (margin 0.019%)
Si: 0.312% vs 0.550% max → PASS
Mn: 1.482% vs 1.600% max → PASS (margin 0.118%)
P: 0.014% vs 0.035% max → PASS
S: 0.009% vs 0.040% max → PASS
N: 0.005% vs 0.012% max → PASS
CEV: 0.431% vs 0.450% max → PASS (margin 0.019%)

CEV calculation check: CEV = C + Mn/6 + (Cr+Mo+V)/5 + (Ni+Cu)/15. With C=0.181, Mn=1.482: 0.181 + 1.482/6 = 0.181 + 0.247 = 0.428. Without other elements the result would be ~0.428, close to 0.431 (small additions from other elements) — consistent.

Dimensional: 24.9, 25.1, 25.0 mm. Tolerance ±0.7mm = 24.3–25.7 mm. All within tolerance → PASS ✓

All tests: PASS. Overall: CERTIFICATE CONFIRMS CONFORMITY.

Confidence: High — all results verified.""",
    answer="""## Material Test Report Extraction — MTR-2024-0441

### Report Details
| Field | Value |
|-------|-------|
| **Report No** | MTR-2024-0441 |
| **Issuing Lab** | Castleford Materials Testing Laboratory Ltd |
| **Accreditation** | UKAS No. 4721 (ISO/IEC 17025:2017) |
| **Client** | Bridgeford Steel Fabrications Ltd |
| **Date of Testing** | 06 March 2024 |
| **Report Date** | 11 March 2024 |

### Material Identification
| Field | Value |
|-------|-------|
| **Grade** | Structural Carbon Steel S355JR |
| **Standard** | BS EN 10025-2:2019 |
| **Heat Number** | 47831A |
| **Batch / Coil** | C-2024-0882 |
| **Product Form** | Plate — 25 mm thickness |

### Mechanical Test Results
| Test | Parameter | Result | Specification | Margin | Status |
|------|-----------|--------|--------------|--------|--------|
| Tensile | Yield Strength (ReH) | 398 MPa | Min 355 MPa | +43 MPa (+12.1%) | **PASS** |
| Tensile | Tensile Strength (Rm) | 541 MPa | 470–630 MPa | Within range | **PASS** |
| Tensile | Elongation (A) | 26.4% | Min 22% | +4.4% | **PASS** |
| Tensile | Reduction of Area | 68.2% | No requirement | — | INFO |
| Charpy | Average Impact (−20°C) | **184 J** | Min 27 J avg | +157 J (+581%) | **PASS** |
| Charpy | Min Individual | 164 J | Min 19 J | +145 J | **PASS** |

*Charpy individual results: 187 J / 164 J / 201 J — Average: 184 J ✓*

### Chemical Analysis
| Element | Result | Max Specification | Margin | Status |
|---------|--------|------------------|--------|--------|
| C | 0.181% | 0.200% | −0.019% | **PASS** |
| Si | 0.312% | 0.550% | −0.238% | **PASS** |
| Mn | 1.482% | 1.600% | −0.118% | **PASS** |
| P | 0.014% | 0.035% | −0.021% | **PASS** |
| S | 0.009% | 0.040% | −0.031% | **PASS** |
| N | 0.005% | 0.012% | −0.007% | **PASS** |
| **CEV** | **0.431%** | **0.450%** | **−0.019%** | **PASS** |

### Dimensional Check
| Point | Measured | Tolerance (±0.7mm) | Status |
|-------|---------|-------------------|--------|
| 1 | 24.9 mm | 24.3–25.7 mm | **PASS** |
| 2 | 25.1 mm | 24.3–25.7 mm | **PASS** |
| 3 | 25.0 mm | 24.3–25.7 mm | **PASS** |

### Overall Conformity: ALL TESTS PASSED

> Material conforms to BS EN 10025-2:2019 requirements for Grade S355JR. UKAS-accredited report.

**Confidence: High** — all results independently verified""",
    difficulty="hard"
))

# ─── 25. Non-Disclosure Agreement ─────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract the key terms from this NDA:

MUTUAL NON-DISCLOSURE AGREEMENT

This Agreement is entered into as of 10 January 2024 between:
Party A: Fenwick Digital Solutions Ltd (Company No. 08374921), 25 Quayside,
         Newcastle upon Tyne NE1 3DE
Party B: Vantage Capital Partners LLP, authorised and regulated by the FCA
         (FRN: 482739), 1 More London Place, London SE1 2AF

PURPOSE:
The parties are considering a potential investment by Party B into Party A
("Potential Transaction") and wish to share confidential information for
the purpose of evaluating such transaction.

TERM: 2 years from Effective Date (until 10 January 2026)

CONFIDENTIAL INFORMATION: Includes all non-public information disclosed,
whether oral, written or electronic, marked "Confidential" or which a
reasonable person would understand to be confidential given the context.
Excludes: (a) information already in public domain; (b) information already
known to recipient; (c) information independently developed; (d) information
required to be disclosed by law or regulation.

OBLIGATIONS:
- Keep confidential using at least the same care as own confidential information
  (no less than reasonable care)
- Use only for Purpose stated above
- Disclose only to: directors, employees, advisers with need to know (and bound
  by equivalent obligations)
- Promptly notify of any unauthorised disclosure

RETURN / DESTRUCTION:
On written request or termination: return or certifiably destroy all confidential
information and copies within 14 days. Legal hold exception applies.

REMEDIES:
Parties acknowledge that breach may cause irreparable harm for which monetary
damages are inadequate. Injunctive relief available without bond or proof of damage.

GOVERNING LAW: England and Wales. Exclusive jurisdiction: courts of England and Wales.""",
    reasoning="""This is a mutual NDA between Fenwick Digital Solutions (target company) and Vantage Capital Partners (potential investor/PE firm). The purpose is to enable due diligence for a potential investment transaction.

Parties:
- Party A: Fenwick Digital Solutions Ltd, Co. No. 08374921, Newcastle NE1 3DE
- Party B: Vantage Capital Partners LLP, FCA authorised FRN 482739, London SE1 2AF

Nature of arrangement: mutual (both parties can disclose and receive). Purpose: evaluating a potential investment by Vantage into Fenwick.

Effective date: 10 January 2024. Term: 2 years, expires 10 January 2026.

Confidential information definition: broad — non-public info, marked confidential or reasonably understood as confidential. 4 standard carve-outs: public domain, prior knowledge, independent development, legally required disclosure.

Obligations: reasonable care standard (not higher than own CI protection), purpose limitation, need-to-know restriction (directors, employees, advisers bound by equivalent obligations), breach notification duty.

Return/destruction: within 14 days of written request or termination. Certifiable destruction required. Legal hold exception (important for FCA-regulated party).

Remedies: injunctive relief acknowledged as appropriate; no requirement to post bond or prove actual damage — standard in NDAs.

Governing law: England and Wales, exclusive jurisdiction.

Key observation: This is a mutual NDA (both parties have equal obligations). The FCA regulation of Party B (investment firm) means they may have additional regulatory obligations around disclosure that could interact with this NDA — the legal hold exception is particularly relevant.

Confidence: High.""",
    answer="""## NDA Extraction — Fenwick Digital Solutions / Vantage Capital Partners

### Agreement Details
| Field | Value |
|-------|-------|
| **Agreement Type** | Mutual Non-Disclosure Agreement |
| **Effective Date** | 10 January 2024 |
| **Expiry Date** | 10 January 2026 |
| **Term** | 2 years |
| **Purpose** | Evaluating potential investment by Party B into Party A |

### Parties
| Role | Entity | Details |
|------|--------|---------|
| **Party A** | Fenwick Digital Solutions Ltd | Co. No. 08374921, 25 Quayside, Newcastle upon Tyne NE1 3DE |
| **Party B** | Vantage Capital Partners LLP | FCA FRN 482739, 1 More London Place, London SE1 2AF |

### Confidential Information
**Included:** All non-public information (oral, written, electronic) — marked "Confidential" or reasonably understood as confidential

**Excluded (4 carve-outs):**
1. Information already in the public domain
2. Information already known to recipient
3. Information independently developed by recipient
4. Information required to be disclosed by law or regulation

### Core Obligations
| Obligation | Standard |
|-----------|---------|
| Protection standard | At least same care as own CI (minimum: reasonable care) |
| Permitted use | Purpose only (evaluating Potential Transaction) |
| Permitted disclosure | Directors, employees, advisers — need to know, equivalent obligations |
| Unauthorised disclosure | Promptly notify other party |

### Return / Destruction
- On written request or termination: return or certifiably destroy all CI and copies
- **Deadline:** Within **14 days**
- **Exception:** Legal hold (relevant given FCA-regulated Party B)

### Remedies
| Remedy | Provision |
|--------|-----------|
| **Injunctive relief** | Available without bond or proof of actual damage |
| Basis | Parties acknowledge breach may cause irreparable harm |

### Governing Law & Jurisdiction
- **Governing law:** England and Wales
- **Jurisdiction:** Courts of England and Wales (exclusive)

### Key Observations
- Agreement is **mutual** — both parties have equal obligations as disclosing and receiving parties
- Party B is FCA-regulated — regulatory disclosure obligations may override NDA (legal hold exception addresses this)
- Standard commercial NDA structure; appropriate for pre-investment due diligence

**Confidence: High**""",
    difficulty="medium"
))

print("Part 4 (examples 24-25) defined. Continuing to part 5...")

# ─── 26. Healthcare Prescription ──────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract all medication details from this hospital prescription:

INPATIENT PRESCRIPTION CHART — REGULAR MEDICATIONS
Hospital: Royal Northgate NHS Foundation Trust
Ward: 6C (Neurology)
Patient: PATEL, Divyesh Kumar
DOB: 22/04/1967  |  Weight: 83 kg  |  Allergies: Penicillin (rash), Codeine (nausea)
Prescriber: Dr. A. Kamara (Reg. No. GMC 7483920)
Date Chart Initiated: 15 March 2024

Drug 1: LEVETIRACETAM
  Dose: 1,000 mg
  Route: Oral (tablet)
  Frequency: Twice daily (08:00 and 20:00)
  Indication: Epilepsy — secondary generalised seizures
  Prescriber note: Check renal function; reduce dose if eGFR <50 mL/min/1.73m²

Drug 2: ATORVASTATIN
  Dose: 40 mg
  Route: Oral (tablet)
  Frequency: Once daily (22:00)
  Indication: Secondary cardiovascular prevention

Drug 3: RAMIPRIL
  Dose: 5 mg
  Route: Oral (capsule)
  Frequency: Once daily (08:00)
  Indication: Hypertension / cardiac protection
  Prescriber note: Monitor blood pressure and U&Es

Drug 4: CLOPIDOGREL
  Dose: 75 mg
  Route: Oral (tablet)
  Frequency: Once daily (08:00)
  Indication: Antiplatelet — secondary stroke prevention
  Prescriber note: Withhold 5 days pre-procedure if invasive procedure planned

Drug 5: METFORMIN
  Dose: 500 mg
  Route: Oral (tablet)
  Frequency: Twice daily with meals (08:00 and 18:00)
  Indication: Type 2 Diabetes Mellitus
  Prescriber note: Withhold if eGFR <30; withhold 48hr pre/post IV contrast

Drug 6: ESOMEPRAZOLE
  Dose: 20 mg
  Route: Oral (capsule)
  Frequency: Once daily (before breakfast)
  Indication: Gastroprotection (with clopidogrel + NSAIDs)

NOTED ALLERGIES — MUST NOT PRESCRIBE:
Penicillin and all beta-lactam antibiotics (cross-sensitivity possible)
Codeine and all codeine-containing preparations""",
    reasoning="""This is an inpatient prescription chart for a neurology patient. I need to extract all medication details, noting the documented allergies and clinical warnings.

Patient: Divyesh Kumar Patel, DOB 22/04/1967, weight 83 kg. Ward 6C Neurology, Royal Northgate NHS Foundation Trust. Prescriber: Dr A. Kamara GMC 7483920.

Allergies: Penicillin (rash) and Codeine (nausea) — cross-sensitivities: all beta-lactams and all codeine-containing preparations are excluded.

6 medications:
1. Levetiracetam 1,000 mg PO BD (08:00/20:00) — epilepsy. Renal caution: reduce if eGFR <50. Daily dose 2,000 mg.
2. Atorvastatin 40 mg PO OD (22:00) — secondary CV prevention. Nocte dosing appropriate.
3. Ramipril 5 mg PO OD (08:00) — hypertension/cardiac. Monitor BP and U&Es.
4. Clopidogrel 75 mg PO OD (08:00) — antiplatelet for secondary stroke prevention. Pre-procedure: withhold 5 days.
5. Metformin 500 mg PO BD with meals (08:00/18:00) — T2DM. Dual withhold criteria: eGFR <30 and 48hr pre/post IV contrast. Daily dose 1,000 mg.
6. Esomeprazole 20 mg PO OD (before breakfast) — gastroprotection (with clopidogrel and NSAIDs).

Clinical notes:
- Esomeprazole is co-prescribed with clopidogrel — there is a known drug interaction (CYP2C19 inhibition reducing clopidogrel efficacy) but this is often accepted when gastroprotection is essential; this is a clinical decision.
- Levetiracetam dose 1,000 mg BD is a standard therapeutic dose for epilepsy.
- Ramipril + Metformin combination requires renal monitoring — consistent with the documented U&E monitoring instruction.
- Patient has multiple comorbidities: epilepsy, hypertension, T2DM, previous cardiovascular event (statin + antiplatelet).

Confidence: High for extraction; clinical interpretation notes for context.""",
    answer="""## Prescription Chart Extraction — Patel, Divyesh Kumar

### Patient Details
| Field | Value |
|-------|-------|
| **Patient** | Patel, Divyesh Kumar |
| **DOB** | 22 April 1967 |
| **Weight** | 83 kg |
| **Ward** | 6C (Neurology), Royal Northgate NHS Foundation Trust |
| **Prescriber** | Dr. A. Kamara (GMC 7483920) |
| **Chart Date** | 15 March 2024 |

### Documented Allergies
| Allergen | Reaction | Cross-Sensitivities — Do Not Prescribe |
|----------|----------|----------------------------------------|
| **Penicillin** | Rash | All beta-lactam antibiotics |
| **Codeine** | Nausea | All codeine-containing preparations |

### Prescribed Medications
| # | Drug | Dose | Route | Frequency | Times | Daily Dose |
|---|------|------|-------|-----------|-------|-----------|
| 1 | **Levetiracetam** | 1,000 mg | Oral tablet | Twice daily | 08:00 / 20:00 | 2,000 mg |
| 2 | Atorvastatin | 40 mg | Oral tablet | Once daily | 22:00 | 40 mg |
| 3 | Ramipril | 5 mg | Oral capsule | Once daily | 08:00 | 5 mg |
| 4 | **Clopidogrel** | 75 mg | Oral tablet | Once daily | 08:00 | 75 mg |
| 5 | Metformin | 500 mg | Oral tablet | Twice daily (with meals) | 08:00 / 18:00 | 1,000 mg |
| 6 | Esomeprazole | 20 mg | Oral capsule | Once daily (before breakfast) | — | 20 mg |

### Indications
| Drug | Indication |
|------|-----------|
| Levetiracetam | Epilepsy — secondary generalised seizures |
| Atorvastatin | Secondary cardiovascular prevention |
| Ramipril | Hypertension / cardiac protection |
| Clopidogrel | Antiplatelet — secondary stroke prevention |
| Metformin | Type 2 Diabetes Mellitus |
| Esomeprazole | Gastroprotection (with clopidogrel + NSAIDs) |

### Clinical Warnings & Prescriber Notes
| Drug | Warning |
|------|---------|
| **Levetiracetam** | Reduce dose if eGFR <50 mL/min/1.73m² — check renal function |
| **Ramipril** | Monitor blood pressure and U&Es |
| **Clopidogrel** | Withhold **5 days** before any invasive procedure |
| **Metformin** | Withhold if eGFR <30; withhold **48 hours** pre/post IV contrast |

**Confidence: High** — all prescribing data extracted; clinical context provided for completeness""",
    difficulty="hard"
))

# ─── 27. Merger & Acquisition Term Sheet ──────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract the deal terms from this M&A term sheet:

INDICATIVE TERM SHEET
Proposed acquisition of Quark Data Analytics Ltd ("Target")
Acquirer: Crestmore Technologies plc
Date: 20 March 2024

CONSIDERATION
Total Enterprise Value: £47,000,000
Equity Value: £43,200,000 (EV less net debt of £3,800,000)
Structure:
  Cash at completion:      £32,400,000  (75% of equity value)
  Deferred consideration:  £5,400,000   (12.5%) — 18 months post-completion
  Earn-out:                £5,400,000   (12.5%) — based on FY2025 EBITDA target

EARN-OUT MECHANICS
FY2025 EBITDA target: £6,800,000
Full earn-out payable if FY2025 EBITDA ≥ £6,800,000
Pro-rated 50–100% for EBITDA £5,440,000–£6,800,000 (80–100% of target)
Zero earn-out if FY2025 EBITDA <£5,440,000

VALUATION MULTIPLE
EV / LTM EBITDA: 47,000,000 / 5,100,000 = 9.2× (LTM EBITDA stated by Target: £5,100,000)

COMPLETION CONDITIONS
1. Satisfactory completion of financial, legal and technical due diligence
2. Seller warranties and representations true at completion
3. No material adverse change (MAC) in Target's business
4. Regulatory clearance if required (CMA assessment)
5. Key employee retention: Top 3 engineers to sign 24-month non-competes

EXCLUSIVITY: 45 days from signing of this term sheet

BREAK FEE: £940,000 (2% of EV) payable by Target if deal breaks due to Target's fault

GOVERNING LAW: England & Wales""",
    reasoning="""This is an M&A indicative term sheet for Crestmore Technologies' proposed acquisition of Quark Data Analytics. I need to extract all deal economics, conditions and mechanics.

Parties: Acquirer Crestmore Technologies plc, Target Quark Data Analytics Ltd.

EV: £47,000,000. Net debt: £3,800,000. Equity value: £47,000,000 - £3,800,000 = £43,200,000 ✓

Consideration structure:
- Cash at completion: £32,400,000 = £43,200,000 × 75% = £32,400,000 ✓
- Deferred consideration: £5,400,000 = £43,200,000 × 12.5% = £5,400,000 ✓ (18 months post-completion)
- Earn-out: £5,400,000 = £43,200,000 × 12.5% = £5,400,000 ✓ (FY2025 EBITDA)
Total: £32,400,000 + £5,400,000 + £5,400,000 = £43,200,000 ✓

EV/EBITDA multiple: £47,000,000 / £5,100,000 = 9.216 ≈ 9.2× ✓

Earn-out mechanics:
- Full earn-out if FY2025 EBITDA ≥ £6,800,000
- Pro-rated for EBITDA £5,440,000–£6,800,000 (80–100% of target)
- Zero if EBITDA < £5,440,000 (< 80% of target)

Threshold check: £5,440,000 = 80% of £6,800,000 = £5,440,000 ✓

5 completion conditions: DD, W&R, MAC, regulatory/CMA, key employee retention (top 3 engineers, 24-month non-competes).

Exclusivity: 45 days from signing.

Break fee: £940,000 = 2% of £47,000,000 = £940,000 ✓ — payable by Target if Target causes deal break.

Note: This is non-binding indicative term sheet — not a binding agreement.

Confidence: High — all arithmetic verified.""",
    answer="""## M&A Term Sheet Extraction — Quark Data Analytics Ltd

### Transaction Overview
| Field | Value |
|-------|-------|
| **Target** | Quark Data Analytics Ltd |
| **Acquirer** | Crestmore Technologies plc |
| **Date** | 20 March 2024 |
| **Document Type** | Indicative Term Sheet (non-binding) |

### Valuation & Consideration
| Metric | Value |
|--------|-------|
| **Total Enterprise Value (EV)** | **£47,000,000** |
| Less: Net Debt | (£3,800,000) |
| **Equity Value** | **£43,200,000** |
| EV / LTM EBITDA Multiple | **9.2×** (LTM EBITDA: £5,100,000) |

### Payment Structure
| Component | Amount | % of Equity | Timing |
|-----------|--------|-------------|--------|
| **Cash at Completion** | **£32,400,000** | 75% | At completion |
| Deferred Consideration | £5,400,000 | 12.5% | 18 months post-completion |
| Earn-Out | £5,400,000 | 12.5% | Based on FY2025 EBITDA |
| **Total Equity Value** | **£43,200,000** | **100%** | |

### Earn-Out Mechanics
| EBITDA Outcome | Earn-Out Payable |
|----------------|-----------------|
| ≥ £6,800,000 (100% of target) | **Full £5,400,000** |
| £5,440,000–£6,800,000 (80–100%) | Pro-rated 50–100% |
| < £5,440,000 (< 80% of target) | **Zero** |

*Threshold: £5,440,000 = 80% × £6,800,000 ✓*

### Completion Conditions
| # | Condition |
|---|-----------|
| 1 | Satisfactory financial, legal and technical due diligence |
| 2 | Seller warranties and representations true at completion |
| 3 | No Material Adverse Change (MAC) in Target's business |
| 4 | Regulatory clearance (CMA assessment if required) |
| 5 | Key employee retention — top 3 engineers sign **24-month non-competes** |

### Deal Protection
| Term | Value |
|------|-------|
| **Exclusivity Period** | 45 days from term sheet signing |
| **Break Fee (Target fault)** | **£940,000** (2.0% of EV) |

### Arithmetic Verification
- Equity value: £47,000,000 − £3,800,000 = £43,200,000 ✓
- Cash at completion: £43,200,000 × 75% = £32,400,000 ✓
- Deferred + earn-out: £43,200,000 × 12.5% × 2 = £10,800,000 ✓
- EV multiple: £47,000,000 ÷ £5,100,000 = 9.2× ✓
- Break fee: £47,000,000 × 2% = £940,000 ✓

**Confidence: High — indicative / non-binding document**""",
    difficulty="hard"
))

# ─── 28. IT Security Policy ───────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract all security requirements from this IT security policy:

INFORMATION SECURITY POLICY
Document ID: ISP-001
Version: 4.1
Approved by: Chief Information Security Officer
Approval Date: 15 January 2024
Review Date: 14 January 2025
Applies to: All employees, contractors and third-party users with access to
            Pinnacle Financial Services Ltd systems

PASSWORD & AUTHENTICATION REQUIREMENTS
- Minimum password length: 14 characters
- Complexity: uppercase, lowercase, digit, special character (all required)
- Password expiry: 90 days (privileged accounts: 30 days)
- MFA mandatory for: all remote access, all cloud platforms, privileged accounts
- Password manager approved: CyberArk (enterprise accounts), Bitwarden Business (personal)
- Previous passwords prohibited: last 12

DATA CLASSIFICATION
Level 1 — Public: No restrictions on sharing
Level 2 — Internal: Share within organisation only; no external email without approval
Level 3 — Confidential: Encryption required in transit and at rest; access on need-to-know
Level 4 — Restricted: Board/CISO approval for access; HSM-protected encryption; audit log mandatory

DEVICE SECURITY
- Full-disk encryption mandatory on all endpoints (BitLocker for Windows, FileVault for Mac)
- MDM enrollment mandatory within 24 hours of device provisioning
- OS patches: Critical — within 48 hours; High — within 7 days; Medium — within 30 days
- Screen lock: automatic after 5 minutes; manual lock when leaving workstation

INCIDENT REPORTING
- Suspected security incidents: report to SOC within 1 hour of awareness
- Confirmed breaches involving personal data: notify CISO within 4 hours
- GDPR notification to ICO: within 72 hours of confirmation (where required)
- Post-incident review: mandatory within 10 business days

THIRD-PARTY ACCESS
- Vendor risk assessment required before access granted
- Third-party access via dedicated VPN only (no direct network access)
- Quarterly access review; automatic revocation after 90 days inactivity""",
    reasoning="""This is an IT security policy for Pinnacle Financial Services Ltd, version 4.1, approved January 2024. I need to extract all technical requirements and timelines.

Scope: all employees, contractors and third-party users with system access.

Password/authentication:
- Min 14 characters, all 4 complexity types required
- Expiry: standard 90 days, privileged 30 days
- MFA: remote access, cloud platforms, privileged accounts (all mandatory)
- Approved password managers: CyberArk (enterprise) and Bitwarden Business (personal)
- Password history: 12 prohibited

Data classification: 4 levels:
1. Public — no restrictions
2. Internal — internal only, no external email without approval
3. Confidential — encryption in transit and at rest, need-to-know
4. Restricted — board/CISO approval, HSM encryption, mandatory audit logs

Device security:
- Full disk encryption: BitLocker (Windows), FileVault (Mac)
- MDM enrollment within 24 hours of provisioning
- Patch windows: Critical 48h, High 7 days, Medium 30 days
- Screen lock after 5 minutes; manual lock required when leaving

Incident reporting timelines:
- Suspected incident → SOC within 1 hour
- Confirmed breach with personal data → CISO within 4 hours
- GDPR ICO notification → within 72 hours (where required)
- Post-incident review → within 10 business days

Third party: vendor risk assessment before access, dedicated VPN only, quarterly access review, automatic revocation after 90 days inactivity.

Confidence: High.""",
    answer="""## IT Security Policy Extraction — ISP-001 v4.1

### Document Metadata
| Field | Value |
|-------|-------|
| **Document ID** | ISP-001 Version 4.1 |
| **Approved by** | Chief Information Security Officer |
| **Approval Date** | 15 January 2024 |
| **Review Date** | 14 January 2025 |
| **Scope** | All employees, contractors, and third-party users — Pinnacle Financial Services Ltd |

### Password & Authentication
| Requirement | Standard | Privileged Accounts |
|-------------|----------|---------------------|
| Minimum Length | **14 characters** | 14 characters |
| Complexity | Uppercase + Lowercase + Digit + Special | Same |
| Password Expiry | **90 days** | **30 days** |
| Password History | Last 12 prohibited | Last 12 prohibited |
| MFA | Remote access, cloud platforms | **Mandatory** |

**Approved Password Managers:**
- Enterprise: **CyberArk**
- Personal: **Bitwarden Business**

### Data Classification Framework
| Level | Name | Requirements |
|-------|------|-------------|
| 1 | **Public** | No restrictions |
| 2 | **Internal** | Internal sharing only; external email requires approval |
| 3 | **Confidential** | Encryption in transit and at rest; need-to-know access |
| 4 | **Restricted** | Board/CISO approval; HSM-protected encryption; **mandatory audit log** |

### Device Security
| Requirement | Standard |
|------------|---------|
| Full-disk encryption | **BitLocker** (Windows) / **FileVault** (Mac) — mandatory |
| MDM enrollment | Within **24 hours** of device provisioning |
| Screen lock | Automatic after **5 minutes**; manual lock when leaving workstation |

### Patch Management Timelines
| Severity | Remediation Window |
|----------|-------------------|
| **Critical** | Within **48 hours** |
| **High** | Within **7 days** |
| **Medium** | Within **30 days** |

### Incident Reporting Timelines
| Event | Notification | To |
|-------|-------------|-----|
| Suspected security incident | Within **1 hour** of awareness | SOC |
| Confirmed breach with personal data | Within **4 hours** | CISO |
| GDPR reportable breach | Within **72 hours** (where required) | ICO |
| Post-incident review | Within **10 business days** | — |

### Third-Party Access Controls
| Control | Requirement |
|---------|-------------|
| Vendor risk assessment | Required before access granted |
| Access method | Dedicated VPN only (no direct network access) |
| Access review | Quarterly |
| Inactivity revocation | Automatic after **90 days** |

**Confidence: High**""",
    difficulty="medium"
))

# ─── 29. Quantity Surveyor's Report ───────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract all cost data from this quantity surveyor's cost plan:

COST PLAN — STAGE 3 (RIBA)
Project: Westbridge Community Sports Centre
Client: Westbridge Borough Council
QS Firm: Turner & Associates Quantity Surveyors
Report Date: 08 February 2024
Report Ref: TA-QS-2024-WBC-003

ELEMENTAL COST PLAN
                                          £           £/m²  (GIA: 3,840 m²)
─────────────────────────────────────────────────────────────────────────────
SUBSTRUCTURE
  Foundations (piled)                   384,000      100
  Ground floor construction             211,200       55

SUPERSTRUCTURE
  Frame (steel)                         691,200      180
  Upper floors                          192,000       50
  Roof (standing seam aluminium)        384,000      100
  External walls (curtain walling + CLT) 576,000     150
  Internal walls and partitions         192,000       50
  Windows and external doors            230,400       60

INTERNAL FINISHES
  Floor finishes                        184,320       48
  Ceiling finishes                       96,000       25
  Wall finishes                          80,640       21

FITTINGS & FURNISHINGS                   96,000       25

SERVICES
  Mechanical                            499,200      130
  Electrical                            403,200      105
  Specialist sports equipment           192,000       50

EXTERNAL WORKS                          192,000       50
─────────────────────────────────────────────────────────────────────────────
BUILDING WORKS SUBTOTAL               4,608,160    1,199

Preliminaries (12%)                     552,979      144
Design contingency (10%)               516,114      134
Construction contingency (5%)          283,863       74
─────────────────────────────────────────────────────────────────────────────
BASE ESTIMATE                          5,961,116    1,551

Professional fees (12%)                715,334      186
Client contingency (5%)               333,822       87
─────────────────────────────────────────────────────────────────────────────
TOTAL PROJECT COST ESTIMATE           7,010,272    1,824""",
    reasoning="""This is an RIBA Stage 3 cost plan for a community sports centre. GIA = 3,840 m². I need to extract and verify all cost elements.

Let me verify building works subtotal by summing all elements:
Substructure: 384,000 + 211,200 = 595,200
Frame: 691,200
Upper floors: 192,000
Roof: 384,000
External walls: 576,000
Internal walls: 192,000
Windows/doors: 230,400
Floor finishes: 184,320
Ceiling finishes: 96,000
Wall finishes: 80,640
Fittings: 96,000
Mechanical: 499,200
Electrical: 403,200
Specialist sports: 192,000
External works: 192,000

Sum: 595,200 + 691,200 = 1,286,400
+ 192,000 = 1,478,400
+ 384,000 = 1,862,400
+ 576,000 = 2,438,400
+ 192,000 = 2,630,400
+ 230,400 = 2,860,800
+ 184,320 = 3,045,120
+ 96,000 = 3,141,120
+ 80,640 = 3,221,760
+ 96,000 = 3,317,760
+ 499,200 = 3,816,960
+ 403,200 = 4,220,160
+ 192,000 = 4,412,160
+ 192,000 = 4,604,160

Stated subtotal: £4,608,160. My sum: £4,604,160. Discrepancy of £4,000. Let me recheck floor finishes: 48 × 3,840 = 184,320 ✓. Wall finishes: 21 × 3,840 = 80,640 ✓. Actually let me re-add more carefully:
595,200 + 691,200 + 192,000 + 384,000 + 576,000 + 192,000 + 230,400 + 184,320 + 96,000 + 80,640 + 96,000 + 499,200 + 403,200 + 192,000 + 192,000

= (595,200 + 691,200) = 1,286,400
+ (192,000 + 384,000) = 576,000 → 1,862,400
+ (576,000 + 192,000) = 768,000 → 2,630,400
+ (230,400 + 184,320) = 414,720 → 3,045,120
+ (96,000 + 80,640) = 176,640 → 3,221,760
+ (96,000 + 499,200) = 595,200 → 3,816,960
+ (403,200 + 192,000) = 595,200 → 4,412,160
+ 192,000 → 4,604,160

So my total is £4,604,160 vs stated £4,608,160. Discrepancy of £4,000. I should flag this.

Assuming stated subtotal of £4,608,160:
Preliminaries 12%: £4,608,160 × 12% = £552,979.20 ≈ £552,979 ✓
Design contingency 10%: on what base? (4,608,160 + 552,979) × 10% = 5,161,139 × 10% = £516,114 ✓ (applied to subtotal + prelims)
Construction contingency 5%: (5,161,139 + 516,114) × 5% = 5,677,253 × 5% = £283,863 ✓
Base estimate: 4,608,160 + 552,979 + 516,114 + 283,863 = £5,961,116 ✓

Professional fees 12%: £5,961,116 × 12% = £715,334 ✓
Client contingency 5%: (5,961,116 + 715,334) × 5% = 6,676,450 × 5% = £333,822 ✓
Total: 5,961,116 + 715,334 + 333,822 = £7,010,272 ✓

Elemental rates: check largest item, frame at £180/m²: 180 × 3,840 = £691,200 ✓

Building works at £1,199/m²: 1,199 × 3,840 = £4,603,840 — states £4,608,160. Consistent with the £4,000 discrepancy above — the rate may be rounded.

Overall conclusion: there's a small (~£4,000) discrepancy in building works subtotal that should be flagged.

Confidence: High for most items; discrepancy noted.""",
    answer="""## Cost Plan Extraction — Westbridge Community Sports Centre

### Project Details
| Field | Value |
|-------|-------|
| **Project** | Westbridge Community Sports Centre |
| **Client** | Westbridge Borough Council |
| **QS Firm** | Turner & Associates Quantity Surveyors |
| **Report Reference** | TA-QS-2024-WBC-003 |
| **Report Date** | 08 February 2024 |
| **RIBA Stage** | Stage 3 |
| **Gross Internal Area (GIA)** | 3,840 m² |

### Elemental Cost Plan
| Element | Cost (£) | £/m² |
|---------|---------|------|
| **Substructure** | | |
| Foundations (piled) | 384,000 | 100 |
| Ground floor construction | 211,200 | 55 |
| **Superstructure** | | |
| Frame (steel) | 691,200 | 180 |
| Upper floors | 192,000 | 50 |
| Roof (standing seam aluminium) | 384,000 | 100 |
| External walls (curtain walling + CLT) | 576,000 | 150 |
| Internal walls and partitions | 192,000 | 50 |
| Windows and external doors | 230,400 | 60 |
| **Internal Finishes** | | |
| Floor finishes | 184,320 | 48 |
| Ceiling finishes | 96,000 | 25 |
| Wall finishes | 80,640 | 21 |
| Fittings & Furnishings | 96,000 | 25 |
| **Services** | | |
| Mechanical | 499,200 | 130 |
| Electrical | 403,200 | 105 |
| Specialist sports equipment | 192,000 | 50 |
| External works | 192,000 | 50 |

### Cost Summary
| Category | Amount (£) | £/m² |
|----------|-----------|------|
| **Building Works Subtotal** | **4,608,160** | **1,199** |
| Preliminaries (12%) | 552,979 | 144 |
| Design Contingency (10%) | 516,114 | 134 |
| Construction Contingency (5%) | 283,863 | 74 |
| **Base Estimate** | **5,961,116** | **1,551** |
| Professional Fees (12%) | 715,334 | 186 |
| Client Contingency (5%) | 333,822 | 87 |
| **TOTAL PROJECT COST ESTIMATE** | **£7,010,272** | **£1,824/m²** |

### Cost Breakdown by Category (Building Works)
| Category | Total | % of Building Works |
|----------|-------|---------------------|
| Substructure | £595,200 | 12.9% |
| Superstructure | £2,265,600 | 49.2% |
| Internal Finishes | £360,960 | 7.8% |
| Services | £1,094,400 | 23.7% |
| Other | £192,000 | 4.2% |

> **Arithmetic Note:** Independent summation of line items yields **£4,604,160** vs. stated £4,608,160 (£4,000 discrepancy). All higher-level totals (base estimate, total project cost) verified as internally consistent using the stated building works figure.

**Confidence: High** — one minor discrepancy in building works subtotal flagged""",
    difficulty="hard"
))

# ─── 30. Clinical Trial Summary ───────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract the key data from this clinical trial results summary:

CLINICAL TRIAL RESULTS SUMMARY
Trial: AURORA-2
EudraCT Number: 2022-004831-17
Sponsor: Helix Biopharma Ltd
Phase: III, Randomised, Double-blind, Placebo-controlled
Indication: Moderate-to-severe plaque psoriasis
Drug: Helixumab 150 mg SC every 4 weeks (after loading dose at Weeks 0, 2, 4)

STUDY POPULATION
Total enrolled: 842 patients (ITT population: 840)
Randomisation: 2:1 (helixumab:placebo)
  Helixumab arm: 560 patients
  Placebo arm: 280 patients
Mean age: 43.7 years (range 18–76)
Gender: 62% male, 38% female
Mean PASI at baseline: 19.4 (range 12.0–48.3)

PRIMARY ENDPOINT
PASI 90 response at Week 16:
  Helixumab: 347/560 (61.9%)
  Placebo: 22/280 (7.9%)
  Difference: 54.1 percentage points
  Odds Ratio: 18.7 (95% CI: 11.4–30.6)
  p-value: <0.0001

SECONDARY ENDPOINTS (Week 16)
  PASI 75: Helixumab 83.4% vs Placebo 18.2% (p<0.0001)
  PASI 100: Helixumab 34.8% vs Placebo 1.4% (p<0.0001)
  IGA 0/1: Helixumab 69.6% vs Placebo 8.6% (p<0.0001)

SAFETY SUMMARY
Treatment-emergent adverse events (any): 67.3% helixumab vs 61.8% placebo
Serious adverse events: 4.1% helixumab vs 3.9% placebo
Discontinuations due to AE: 2.3% helixumab vs 1.8% placebo
Deaths: 0 in either arm
Injection site reactions: 8.2% helixumab vs 3.1% placebo""",
    reasoning="""This is a Phase III clinical trial results summary for helixumab in psoriasis. I need to extract all efficacy, safety and study design data.

Study: AURORA-2, EudraCT 2022-004831-17. Phase III double-blind RCT. Sponsor: Helix Biopharma.

Population: 842 enrolled, 840 ITT. 2:1 randomisation: helixumab 560, placebo 280. Check: 560+280=840 ✓ (2 excluded from ITT). Mean age 43.7, 62% male. Baseline PASI 19.4 (moderate-severe).

Primary endpoint — PASI 90 at Week 16:
Helixumab: 347/560 = 0.619 = 61.9% ✓
Placebo: 22/280 = 0.0786 = 7.9% ✓
Difference: 61.9 - 7.9 = 54.0 percentage points (stated 54.1 — slight rounding difference)
Odds ratio: 18.7 (95% CI 11.4–30.6), p<0.0001 — highly statistically significant, CI doesn't cross 1.

Secondary endpoints all p<0.0001:
PASI 75: 83.4% vs 18.2% — absolute difference 65.2pp
PASI 100 (complete clearance): 34.8% vs 1.4% — absolute difference 33.4pp — notable that over a third of patients achieved complete clearance
IGA 0/1 (clear/almost clear): 69.6% vs 8.6% — absolute difference 61.0pp

Safety:
Overall AEs: 67.3% vs 61.8% — marginally higher in drug arm (expected)
Serious AEs: 4.1% vs 3.9% — very similar, no significant imbalance
Discontinuations: 2.3% vs 1.8% — low in both arms
Deaths: 0 — positive
Injection site reactions: 8.2% vs 3.1% — drug-related, expected for SC biologic

Overall: strong efficacy signal with acceptable safety profile. Trial clearly meets primary endpoint with very high statistical confidence.

Confidence: High.""",
    answer="""## Clinical Trial Data Extraction — AURORA-2

### Study Details
| Field | Value |
|-------|-------|
| **Trial Name** | AURORA-2 |
| **EudraCT Number** | 2022-004831-17 |
| **Sponsor** | Helix Biopharma Ltd |
| **Phase** | III — Randomised, Double-Blind, Placebo-Controlled |
| **Indication** | Moderate-to-severe plaque psoriasis |
| **Investigational Drug** | Helixumab 150 mg SC every 4 weeks (loading: Weeks 0, 2, 4) |

### Study Population
| Parameter | Value |
|-----------|-------|
| Total Enrolled | 842 |
| ITT Population | **840** |
| Randomisation Ratio | 2:1 (helixumab:placebo) |
| **Helixumab Arm** | 560 patients |
| **Placebo Arm** | 280 patients |
| Mean Age | 43.7 years (range 18–76) |
| Gender | 62% male / 38% female |
| Mean Baseline PASI | 19.4 (range 12.0–48.3) |

### Primary Endpoint — PASI 90 at Week 16
| Group | Responders | Rate |
|-------|------------|------|
| **Helixumab** | 347/560 | **61.9%** |
| Placebo | 22/280 | 7.9% |
| **Absolute Difference** | | **54.1 percentage points** |
| **Odds Ratio** | | **18.7** (95% CI: 11.4–30.6) |
| **p-value** | | **<0.0001** |

### Secondary Endpoints (Week 16)
| Endpoint | Helixumab | Placebo | Difference | p-value |
|----------|-----------|---------|-----------|---------|
| **PASI 75** | 83.4% | 18.2% | +65.2pp | <0.0001 |
| **PASI 100** (complete clearance) | **34.8%** | 1.4% | +33.4pp | <0.0001 |
| **IGA 0/1** (clear/almost clear) | 69.6% | 8.6% | +61.0pp | <0.0001 |

### Safety Summary
| Event | Helixumab (n=560) | Placebo (n=280) |
|-------|------------------|-----------------|
| Any TEAE | 67.3% | 61.8% |
| Serious AEs | 4.1% | 3.9% |
| Discontinuations due to AE | 2.3% | 1.8% |
| **Deaths** | **0** | **0** |
| Injection site reactions | 8.2% | 3.1% |

### Key Findings
- All primary and secondary endpoints met with **p<0.0001**
- Complete skin clearance (PASI 100) achieved in **34.8%** of treated patients
- **No deaths** in either arm — favourable safety profile
- Injection site reactions (8.2% vs 3.1%) consistent with expectations for SC biologics
- Serious AE rates comparable between arms (4.1% vs 3.9%)

**Confidence: High** — all response rates and denominators verified""",
    difficulty="hard"
))

print("Part 5 (examples 26-30) defined. Continuing to part 6...")

# ─── 31. Software Licence Agreement ───────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract the licensing terms from this enterprise software agreement:

ENTERPRISE SOFTWARE LICENCE AGREEMENT
Licensor: Datavault Software GmbH (registered Germany, HRB 74821)
Licensee: Castleford Insurance Holdings plc (Co. No. 01293847, UK)
Effective Date: 01 April 2024
Order Form Reference: DVS-ENT-2024-0382

LICENCE GRANT
Software: Datavault DataOps Platform v7.x (including all updates in subscription)
Licence Type: Enterprise Subscription
Users: Up to 250 named users (Production environment)
Additional environments: 2 non-production (Dev/UAT) — unlimited users
Term: 3 years (01 April 2024 – 31 March 2027), auto-renews 1 year

FEES
Year 1: £284,000 (invoiced 01 April 2024)
Year 2: £298,200 (+5%)
Year 3: £313,110 (+5%)
Total 3-year commitment: £895,310

Overage: £340/user/year for named users exceeding 250
Support: Premier Support included (24/7, 4-hour critical response SLA)

PERMITTED USE
Internal business operations only. No resale or sub-licensing.
Deployment: on-premises (Castleford data centres) and AWS GovCloud (EU-West)

INTELLECTUAL PROPERTY
All IP remains with Datavault. Licensee granted limited, non-exclusive, non-transferable licence.
Licensee data: Licensee retains ownership. Datavault may use anonymised aggregate data for product improvement.

AUDIT RIGHTS
Datavault may conduct licence compliance audit with 30 days written notice (max 1 per year).
Audit costs borne by Datavault unless material non-compliance found (>5% excess users).

TERMINATION
For convenience: 90 days written notice (no refund of pre-paid fees).
For cause: 30 days to cure material breach; immediate for insolvency.""",
    reasoning="""This is an enterprise software licence agreement between Datavault (German software company) and Castleford Insurance (UK insurer). I need to extract all commercial and legal terms.

Parties: Licensor Datavault Software GmbH (Germany, HRB 74821), Licensee Castleford Insurance Holdings plc (UK, Co. No. 01293847). Effective 01 April 2024.

Software: Datavault DataOps Platform v7.x, enterprise subscription.

Users: 250 named users (Production), unlimited in 2 non-production environments (Dev/UAT).

Term: 3 years, 01 Apr 2024–31 Mar 2027. Auto-renews annually.

Fees:
Year 1: £284,000
Year 2: £298,200 = £284,000 × 1.05 = £298,200 ✓
Year 3: £313,110 = £298,200 × 1.05 = £313,110 ✓
3-year total: 284,000 + 298,200 + 313,110 = £895,310 ✓

Overage: £340/user/year above 250 threshold.

Support: Premier Support included, 24/7, 4-hour critical response SLA.

Permitted use: internal operations only, no resale/sublicensing. Deployment: on-premises + AWS GovCloud EU-West.

IP: Datavault retains all IP. Licensee retains data ownership. Datavault can use anonymised aggregate data for product development.

Audit: 30-day notice, max once per year. Datavault pays unless non-compliance >5% excess users.

Termination: convenience (90 days, no refund of prepaid); for cause (30-day cure period); immediate on insolvency.

Confidence: High.""",
    answer="""## Software Licence Agreement Extraction — DVS-ENT-2024-0382

### Agreement Details
| Field | Value |
|-------|-------|
| **Licensor** | Datavault Software GmbH (Germany, HRB 74821) |
| **Licensee** | Castleford Insurance Holdings plc (Co. No. 01293847, UK) |
| **Effective Date** | 01 April 2024 |
| **Order Form Ref** | DVS-ENT-2024-0382 |

### Licence Scope
| Parameter | Value |
|-----------|-------|
| **Software** | Datavault DataOps Platform v7.x (all updates included) |
| **Licence Type** | Enterprise Subscription |
| **Named Users (Production)** | Up to **250** |
| **Non-Production Environments** | 2 (Dev/UAT) — unlimited users |
| **Deployment** | On-premises (Castleford DCs) + AWS GovCloud EU-West |
| **Permitted Use** | Internal business operations only — no resale or sub-licensing |

### Term
| Milestone | Date |
|-----------|------|
| **Start** | 01 April 2024 |
| **End** | 31 March 2027 |
| **Length** | 3 years |
| **Auto-Renewal** | 1 year (annual) |

### Financial Terms
| Year | Annual Fee | YoY Increase |
|------|-----------|-------------|
| Year 1 | £284,000 | — |
| Year 2 | £298,200 | +5.0% |
| Year 3 | £313,110 | +5.0% |
| **3-Year Total** | **£895,310** | |

| Additional Charge | Rate |
|------------------|------|
| Overage (>250 named users) | **£340/user/year** |
| Support | Premier Support included (24/7, 4-hr critical SLA) |

### Key Provisions
| Provision | Detail |
|-----------|--------|
| **IP Ownership** | All IP retained by Datavault |
| **Licensee Licence** | Limited, non-exclusive, non-transferable |
| **Licensee Data** | Licensee retains ownership; Datavault may use anonymised aggregate data |
| **Audit Rights** | 30 days written notice; max 1 audit/year |
| **Audit Costs** | Datavault pays (unless non-compliance >5% excess users) |

### Termination Rights
| Trigger | Notice / Mechanism |
|---------|-------------------|
| Convenience | 90 days written notice — **no refund of prepaid fees** |
| Material breach | 30-day cure period |
| Insolvency | Immediate termination |

### Arithmetic Verification
- Year 2: £284,000 × 1.05 = £298,200 ✓
- Year 3: £298,200 × 1.05 = £313,110 ✓
- 3-year total: £284,000 + £298,200 + £313,110 = £895,310 ✓

**Confidence: High**""",
    difficulty="medium"
))

# ─── 32. Property Valuation ───────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract all valuation data from this RICS valuation report summary:

VALUATION REPORT — EXECUTIVE SUMMARY
RICS Registered Valuer: Catherine Thornton MRICS
Firm: Harrington Chartered Surveyors LLP
Report Reference: HCS-VAL-2024-0219
Date of Valuation: 22 February 2024
Client: Bridgewater Mortgage Lending Ltd (mortgage security purposes)

PROPERTY
Address: 23 Kestrel Gardens, Northampton NN3 8PQ
Property Type: Detached house, 4 bedrooms
Tenure: Freehold
Year Built: Circa 1997
Floor Area (GIA): 182 sq m (1,959 sq ft)
Garage: Integral double garage
EPC Rating: C (rating 72)

MARKET COMMENTARY
Local market active. Average days-to-sale in NN3 postcode: 38 days (Q4 2023).
3 comparables used (sales within 6 months, within 0.5-mile radius):

Comparable 1: 18 Kestrel Gardens — 4 bed detached, 176 sq m — sold Oct 2023 — £487,500
              (adjusted to £490,000 for superior spec)
Comparable 2: 7 Osprey Close — 4 bed detached, 189 sq m — sold Dec 2023 — £518,000
              (adjusted to £495,000 for larger floor area)
Comparable 3: 41 Moorhen Way — 4 bed detached, 178 sq m — sold Jan 2024 — £492,000
              (no adjustment required)

VALUATION
Market Value (as defined by RICS Red Book VPS 4):  £495,000
Valuation Methodology: Direct comparison
Date of Inspection: 20 February 2024

SPECIAL ASSUMPTIONS / CAVEATS
- No structural survey undertaken; valuation assumes no significant defects
- Japanese knotweed: not observed; standard caveat applies
- Overhanging boundary tree (rear garden): disclosed by vendor; not considered to
  materially impact value at this time but monitoring advised""",
    reasoning="""This is an RICS Red Book mortgage valuation for a residential property. I need to extract all valuation data, comparable evidence and caveats.

Report: HCS-VAL-2024-0219. RICS Registered Valuer Catherine Thornton MRICS, Harrington Chartered Surveyors. Date of valuation 22 February 2024, inspection 20 February 2024 (2 days before report). Client: Bridgewater Mortgage Lending Ltd.

Property: 23 Kestrel Gardens, Northampton NN3 8PQ. Freehold detached house, 4 bed, ~1997 built, 182 sq m GIA (1,959 sq ft), integral double garage, EPC C (72).

Comparable analysis — 3 comparables, all within 6 months and 0.5 miles:
1. 18 Kestrel Gardens — 4 bed, 176 sq m, sold Oct 2023 £487,500. Adjusted +£2,500 to £490,000 for superior spec of subject vs comparable.
2. 7 Osprey Close — 4 bed, 189 sq m, sold Dec 2023 £518,000. Adjusted DOWN to £495,000 for larger floor area (comparable is 189 vs 182 sq m). The comparable is larger so its value is adjusted down to reflect the smaller subject.
3. 41 Moorhen Way — 4 bed, 178 sq m, sold Jan 2024 £492,000. No adjustment — similar spec.

The three adjusted values are: £490,000, £495,000, £492,000. Range £490k–£495k. Valuer has adopted £495,000 as the market value.

£/sq m analysis: £495,000 / 182 sq m = £2,720/sq m; £495,000 / 1,959 sq ft = £253/sq ft.

Caveats: no structural survey (standard mortgage valuation limitation), Japanese knotweed not observed, overhanging tree disclosed but not currently material.

Market commentary: 38 days average TTS in NN3 — liquid market.

Confidence: High.""",
    answer="""## Property Valuation Extraction — HCS-VAL-2024-0219

### Valuation Details
| Field | Value |
|-------|-------|
| **Report Reference** | HCS-VAL-2024-0219 |
| **RICS Registered Valuer** | Catherine Thornton MRICS |
| **Firm** | Harrington Chartered Surveyors LLP |
| **Date of Inspection** | 20 February 2024 |
| **Date of Valuation** | 22 February 2024 |
| **Purpose** | Mortgage security — Bridgewater Mortgage Lending Ltd |
| **Methodology** | RICS Red Book VPS 4 — Direct comparison |

### Property Details
| Field | Value |
|-------|-------|
| **Address** | 23 Kestrel Gardens, Northampton NN3 8PQ |
| **Type** | Detached house, 4 bedrooms |
| **Tenure** | Freehold |
| **Year Built** | Circa 1997 |
| **Floor Area (GIA)** | 182 sq m / 1,959 sq ft |
| **Garage** | Integral double garage |
| **EPC Rating** | C (72) |

### Comparable Evidence
| # | Address | Beds | Area | Sale Date | Sale Price | Adjusted Value | Adjustment Reason |
|---|---------|------|------|-----------|-----------|----------------|------------------|
| 1 | 18 Kestrel Gardens | 4 bed | 176 sq m | Oct 2023 | £487,500 | £490,000 | +£2,500 (subject superior spec) |
| 2 | 7 Osprey Close | 4 bed | 189 sq m | Dec 2023 | £518,000 | £495,000 | −£23,000 (comparable larger) |
| 3 | 41 Moorhen Way | 4 bed | 178 sq m | Jan 2024 | £492,000 | £492,000 | None required |

*All comparables: within 6 months, within 0.5-mile radius. Average days-to-sale NN3: 38 days (Q4 2023)*

### Valuation
| Metric | Value |
|--------|-------|
| **Market Value** | **£495,000** |
| Value per sq m | £2,720/sq m |
| Value per sq ft | £253/sq ft |
| Adjusted comparable range | £490,000–£495,000 |

### Special Assumptions & Caveats
| Item | Status |
|------|--------|
| Structural survey | Not undertaken — valuation assumes no significant defects |
| Japanese knotweed | Not observed — standard caveat applies |
| Boundary tree (rear) | Disclosed by vendor; **not currently material to value**; monitoring advised |

**Confidence: High** — market value well-supported by 3 local comparable sales""",
    difficulty="medium"
))

# ─── 33. Employee Disciplinary Record ────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract all factual details from this disciplinary outcome letter:

WITHOUT PREJUDICE — DISCIPLINARY OUTCOME LETTER
[This letter is marked without prejudice save as to costs]

Issued by: Pinnacle Retail Solutions Ltd, HR Department
To: Mr. Samuel Okereke, Store Manager, Croydon Branch
Date: 12 March 2024

Dear Mr. Okereke,

I write to confirm the outcome of the disciplinary hearing held on 08 March 2024
chaired by Regional Operations Director, Ms. Felicity Ashworth.

ALLEGATIONS INVESTIGATED
1. Authorising staff discounts in excess of the 20% employee scheme limit on 
   14 occasions (November 2023 – January 2024), totalling unauthorised discounts
   valued at £3,847.
2. Failure to complete mandatory food hygiene refresher training by the deadline
   of 30 November 2023.
3. Falsification of the temperature monitoring log on 3 occasions in December 2023.

FINDINGS
Allegation 1: PROVEN — documentary evidence from EPOS system confirmed 14 instances.
Allegation 2: PROVEN — training records confirm non-completion as at hearing date.
Allegation 3: NOT PROVEN — insufficient evidence to conclude falsification beyond 
              reasonable doubt.

OUTCOME: FINAL WRITTEN WARNING

Terms of Warning:
- Live for 12 months from the date of this letter (expires 12 March 2025)
- Any further misconduct during this period may result in dismissal
- Completion of mandatory food hygiene training required within 14 days (by 26 March 2024)
- No further unauthorised discounts — adherence to the 20% scheme limit is mandatory

APPEAL RIGHTS:
Mr. Okereke has the right to appeal within 5 working days of receiving this letter
(deadline: 19 March 2024). Appeal in writing to Head of HR.""",
    reasoning="""This is a disciplinary outcome letter for Mr. Samuel Okereke, Store Manager at Pinnacle Retail Solutions' Croydon branch. The letter is marked Without Prejudice (WPSAC). I need to extract all factual details objectively.

Issued: 12 March 2024. Hearing date: 08 March 2024. Chaired by Felicity Ashworth, Regional Operations Director.

Three allegations:
1. Unauthorised staff discounts: 14 occasions, Nov 2023–Jan 2024, total value £3,847. Outcome: PROVEN — documentary EPOS evidence.
2. Failure to complete mandatory food hygiene training by 30 Nov 2023 deadline. Outcome: PROVEN — training records confirm non-completion.
3. Temperature log falsification: 3 occasions in December 2023. Outcome: NOT PROVEN — insufficient evidence.

Outcome: FINAL WRITTEN WARNING.

Terms:
- Duration: 12 months from 12 March 2024, expires 12 March 2025
- Further misconduct during 12-month period may lead to dismissal
- Food hygiene training must be completed within 14 days = by 26 March 2024. Check: 12 March + 14 days = 26 March ✓
- No further unauthorised discounts

Appeal: 5 working days from receipt of letter. Letter dated 12 March = assuming received same day: 5 working days from 12 March = deadline 19 March 2024. Check: 12 March is a Tuesday 2024. Working days: Wed 13, Thu 14, Fri 15, Mon 18, Tue 19 = 5 working days = 19 March ✓

Appeal to: Head of HR, in writing.

Important note: This letter is Without Prejudice — it should only be extracted in context of the employer's HR records.

Confidence: High for factual content.""",
    answer="""## Disciplinary Outcome Extraction — Okereke, Samuel

### Document Details
| Field | Value |
|-------|-------|
| **Document Type** | Disciplinary Outcome Letter |
| **Legal Status** | Without Prejudice Save as to Costs (WPSAC) |
| **Issued by** | Pinnacle Retail Solutions Ltd — HR Department |
| **Issued to** | Mr. Samuel Okereke, Store Manager, Croydon Branch |
| **Issue Date** | 12 March 2024 |
| **Hearing Date** | 08 March 2024 |
| **Hearing Chair** | Ms. Felicity Ashworth, Regional Operations Director |

### Allegations & Findings
| # | Allegation | Detail | Finding |
|---|-----------|--------|---------|
| 1 | Unauthorised staff discounts | 14 occasions, Nov 2023–Jan 2024; total value £3,847 | **PROVEN** — EPOS documentary evidence |
| 2 | Failure to complete mandatory food hygiene training | Deadline 30 Nov 2023 — not completed | **PROVEN** — training records confirmed |
| 3 | Temperature monitoring log falsification | 3 occasions, December 2023 | **NOT PROVEN** — insufficient evidence |

### Outcome
**FINAL WRITTEN WARNING**

### Warning Terms
| Condition | Detail |
|-----------|--------|
| **Warning live period** | 12 months — 12 March 2024 to **12 March 2025** |
| **Consequence of further misconduct** | May result in dismissal |
| **Food hygiene training** | Must be completed within 14 days — **by 26 March 2024** |
| **Discount compliance** | Strict adherence to 20% employee scheme limit required |

### Appeal Rights
| Detail | Value |
|--------|-------|
| **Appeal window** | 5 working days from receipt |
| **Appeal deadline** | **19 March 2024** |
| **Appeal to** | Head of HR (in writing) |

### Key Observations
- Only 2 of 3 allegations proven; temperature log allegation (allegation 3) did not meet the evidential threshold
- Total financial value of unauthorised discounts: **£3,847** across 14 instances
- Final Written Warning is a severe sanction — next step if misconduct recurs within 12 months would typically be dismissal

**Confidence: High** — factual extraction only; document marked WPSAC""",
    difficulty="medium"
))

# ─── 34. Mortgage Offer ───────────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract all mortgage terms from this mortgage offer letter:

MORTGAGE OFFER
Lender: Northgate Building Society
Branch: Manchester Mortgage Centre
Reference: MO-2024-GMC-48271
Date of Offer: 05 March 2024
Offer Validity: 6 months (expires 05 September 2024)

APPLICANTS: Mr. Callum Fraser & Ms. Anika Kapoor
Property: 7 Elmhurst Close, Didsbury, Manchester M20 2PL
Tenure: Freehold  |  Purchase Price: £525,000

MORTGAGE DETAILS
Mortgage Amount: £393,750
Loan to Value (LTV): 75.0%
Term: 30 years (360 monthly payments)
Repayment Method: Capital and interest (full repayment)

INTEREST RATE (INITIAL)
Rate Type: 5-year fixed rate
Initial Rate: 4.89% per annum
Initial Monthly Payment: £2,073.48
Initial Period Ends: 31 March 2029

REVERSION RATE (after initial period)
Rate: Northgate Standard Variable Rate (SVR), currently 7.49% p.a.
Indicative SVR monthly payment: £2,747.22

FEES
Arrangement Fee: £999 (may be added to mortgage balance)
Valuation Fee: £450 (already paid)
Legal Fee (solicitor): not included — borrower to instruct separately

AFFORDABILITY INFORMATION
Annual income (combined verified): £118,400
Monthly mortgage payment as % of net monthly income: 31.4%

TOTAL AMOUNT PAYABLE (illustrative)
Total interest over term (at initial rate, no change): £308,657
Total amount repayable: £702,407 (mortgage + all interest + arrangement fee)

OVERPAYMENT ALLOWANCE: 10% of outstanding balance per year without ERC
EARLY REPAYMENT CHARGE: 3% of amount repaid in excess of 10% (Years 1–5)""",
    reasoning="""This is a mortgage offer letter from Northgate Building Society for Mr. Callum Fraser and Ms. Anika Kapoor. I need to extract all financial terms.

Reference MO-2024-GMC-48271, issued 05 March 2024, valid 6 months to 05 September 2024.

Property: 7 Elmhurst Close, Didsbury, Manchester M20 2PL. Freehold purchase at £525,000.

Mortgage: £393,750. LTV: 393,750/525,000 = 75.0% ✓. Term: 30 years (360 payments).

Initial rate: 4.89% fixed for 5 years ending 31 March 2029. Monthly payment: £2,073.48. Let me check: P&I payment at 4.89%/12 = 0.4075% per month on £393,750 for 360 months. Monthly rate r = 0.04890/12 = 0.004075. Payment = 393,750 × 0.004075 × (1.004075)^360 / ((1.004075)^360 - 1). (1.004075)^360 = e^(360×ln(1.004075)) ≈ e^(360×0.004067) ≈ e^1.464 ≈ 4.322. Payment = 393,750 × 0.004075 × 4.322/(4.322-1) = 393,750 × 0.004075 × 4.322/3.322 = 393,750 × 0.004075 × 1.3010 = 393,750 × 0.005304 = £2,088. Close to £2,073 — the slight difference is due to my approximation. The stated £2,073.48 is plausible.

SVR payment £2,747.22 at 7.49% on remaining balance after 5 years (balance will be less than £393,750).

Fees: arrangement £999 (addable to loan), valuation £450 paid, solicitor separate.

Deposit: £525,000 - £393,750 = £131,250. As % of purchase = 25%.

Affordability: combined income £118,400/year. Monthly mortgage £2,073 as 31.4% of net monthly income. Stated as 31.4%.

Total repayable: £702,407. This includes: £393,750 principal + £999 arrangement fee + £308,657 interest, but 393,750 + 999 + 308,657 = £703,406 ≠ £702,407. Slight discrepancy of ~£1,000 — may relate to how the arrangement fee is treated. I'll note this.

ERC: 3% on amounts over 10% overpayment, Years 1-5. So free overpayment = 10% of balance/year.

Confidence: High; minor arithmetic note on total repayable.""",
    answer="""## Mortgage Offer Extraction — MO-2024-GMC-48271

### Offer Details
| Field | Value |
|-------|-------|
| **Lender** | Northgate Building Society |
| **Reference** | MO-2024-GMC-48271 |
| **Date of Offer** | 05 March 2024 |
| **Offer Expiry** | 05 September 2024 (6 months) |

### Applicants & Property
| Field | Value |
|-------|-------|
| **Applicants** | Mr. Callum Fraser & Ms. Anika Kapoor |
| **Property** | 7 Elmhurst Close, Didsbury, Manchester M20 2PL |
| **Tenure** | Freehold |
| **Purchase Price** | £525,000 |
| **Deposit** | £131,250 (25%) |

### Mortgage Terms
| Parameter | Value |
|-----------|-------|
| **Mortgage Amount** | **£393,750** |
| **Loan to Value (LTV)** | **75.0%** |
| **Repayment Term** | 30 years (360 payments) |
| **Repayment Method** | Capital and interest (full repayment) |

### Interest Rates
| Period | Rate | Monthly Payment | Duration |
|--------|------|----------------|---------|
| **Initial (fixed)** | **4.89% p.a.** | **£2,073.48** | 5 years (until 31 Mar 2029) |
| **Reversion (SVR)** | 7.49% p.a. (current SVR) | £2,747.22 (indicative) | Remaining 25 years |

### Fees
| Fee | Amount | Notes |
|-----|--------|-------|
| Arrangement Fee | **£999** | Can be added to mortgage balance |
| Valuation Fee | £450 | Already paid |
| Legal/Solicitor | — | Borrower instructs separately |

### Affordability
| Metric | Value |
|--------|-------|
| Combined Verified Income | £118,400/year |
| Monthly Payment as % of Net Income | 31.4% |

### Total Cost (Illustrative)
| Component | Amount |
|-----------|--------|
| Mortgage Principal | £393,750 |
| Total Interest (30-yr illustrative) | £308,657 |
| **Total Amount Repayable** | **£702,407** |

### Overpayment & ERC
| Feature | Detail |
|---------|--------|
| **Free overpayment allowance** | 10% of outstanding balance per year |
| **ERC (Years 1–5)** | 3% of amount repaid in excess of 10% annual allowance |
| **After Year 5** | No ERC |

**Confidence: High**""",
    difficulty="medium"
))

# ─── 35. GDPR Data Processing Record ─────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract all data processing activities from this GDPR record:

RECORD OF PROCESSING ACTIVITIES (ROPA)
Article 30(1) UK GDPR
Controller: Brightwell Healthcare Recruitment Ltd
Company No: 09384721
Data Protection Officer: Sarah Mwangi (dpo@brightwellrecruitment.co.uk)
Date: 01 March 2024

PROCESSING ACTIVITY 1: CANDIDATE MANAGEMENT
Purpose: Recruitment matching and candidate placement services
Categories of data subjects: Job applicants and candidates
Personal data categories: Name, contact details, employment history, qualifications,
  references, right-to-work documents, DBS check results (where applicable)
Special category data: Health information (occupational health assessments),
  criminal conviction data (DBS checks)
Legal basis: Legitimate interests (matching); Legal obligation (right-to-work, DBS);
             Explicit consent (special category health data)
Retention: 6 years post last contact (regulatory requirement); DBS data — 6 months
Recipients: Healthcare employer clients (B2B), background check providers (processors)
Transfers outside UK: None

PROCESSING ACTIVITY 2: CLIENT RELATIONSHIP MANAGEMENT
Purpose: Managing commercial relationships with NHS Trusts and private healthcare providers
Categories of data subjects: Client contact persons (employees of client organisations)
Personal data categories: Name, job title, work email, work phone, business address
Special category data: None
Legal basis: Legitimate interests (B2B relationship management)
Retention: 7 years post contract end (contractual / tax purposes)
Recipients: CRM platform (Salesforce) — processor agreement in place
Transfers outside UK: Salesforce EU data residency — within UK adequacy decision scope

PROCESSING ACTIVITY 3: EMPLOYEE DATA (OWN STAFF)
Purpose: Employment administration, payroll, legal compliance
Personal data categories: Name, address, NI number, bank details, salary, absence records,
  performance reviews, disciplinary records
Special category data: Health data (sick leave), trade union membership
Legal basis: Contract of employment; Legal obligation; Explicit consent (health data);
             Legitimate interests (performance/disciplinary)
Retention: Employment period + 6 years (payroll/tax); Disciplinary records — 5 years
Recipients: Payroll bureau (Moorgate Payroll Ltd), HMRC, pension provider
Transfers outside UK: None""",
    reasoning="""This is a GDPR Article 30 Record of Processing Activities (ROPA) for Brightwell Healthcare Recruitment Ltd. I need to extract all three processing activities with their associated data protection details.

Controller: Brightwell Healthcare Recruitment Ltd, Co. No. 09384721. DPO: Sarah Mwangi. Document dated 01 March 2024.

Three processing activities:

Activity 1 — Candidate Management:
- Purpose: recruitment matching and placement
- Data subjects: job applicants and candidates
- Personal data: name, contacts, employment history, qualifications, references, right-to-work docs, DBS results
- Special category data: health information (occupational health), criminal conviction data (DBS)
- Legal bases: legitimate interests (matching), legal obligation (RTW, DBS), explicit consent (health)
- Retention: 6 years post last contact (regulatory); DBS data only 6 months
- Recipients: healthcare employer clients, background check providers (processors)
- No international transfers

Activity 2 — Client CRM:
- Purpose: B2B relationship management with NHS Trusts and private healthcare
- Data subjects: client contact persons (employees of client orgs)
- Personal data: work contact details, name, title
- No special category data
- Legal basis: legitimate interests
- Retention: 7 years post contract end
- Recipients: Salesforce (processor agreement in place), EU data residency, within adequacy scope
- Transfer: Salesforce EU — within UK adequacy decision

Activity 3 — Employee Data:
- Purpose: employment admin, payroll, legal compliance
- Personal data: name, address, NI, bank details, salary, absence, performance, disciplinary
- Special category: health data (sick leave), trade union membership
- Legal bases: contract, legal obligation, explicit consent (health), legitimate interests (performance)
- Retention: employment + 6 years (payroll/tax); disciplinary records 5 years
- Recipients: Moorgate Payroll Ltd, HMRC, pension provider
- No international transfers

Confidence: High.""",
    answer="""## GDPR ROPA Extraction — Brightwell Healthcare Recruitment Ltd

### Controller Details
| Field | Value |
|-------|-------|
| **Controller** | Brightwell Healthcare Recruitment Ltd (Co. No. 09384721) |
| **DPO** | Sarah Mwangi (dpo@brightwellrecruitment.co.uk) |
| **Legal basis** | UK GDPR Article 30(1) |
| **Date** | 01 March 2024 |

---

### Activity 1 — Candidate Management
| Attribute | Detail |
|-----------|--------|
| **Purpose** | Recruitment matching and candidate placement |
| **Data Subjects** | Job applicants and candidates |
| **Personal Data** | Name, contacts, employment history, qualifications, references, right-to-work docs, DBS results |
| **Special Category Data** | Health information (occupational health); Criminal conviction data (DBS) |
| **Legal Bases** | Legitimate interests (matching); Legal obligation (RTW + DBS); **Explicit consent** (health data) |
| **Retention — General** | 6 years post last contact |
| **Retention — DBS Data** | **6 months only** |
| **Recipients** | Healthcare employer clients; Background check providers (processors) |
| **International Transfers** | None |

---

### Activity 2 — Client Relationship Management
| Attribute | Detail |
|-----------|--------|
| **Purpose** | Managing commercial relationships (NHS Trusts + private healthcare providers) |
| **Data Subjects** | Client contact persons (employees of client organisations) |
| **Personal Data** | Name, job title, work email, work phone, business address |
| **Special Category Data** | **None** |
| **Legal Basis** | Legitimate interests (B2B relationship management) |
| **Retention** | 7 years post contract end |
| **Recipients** | Salesforce (processor — data processing agreement in place) |
| **International Transfers** | Salesforce EU data residency — within UK adequacy decision scope |

---

### Activity 3 — Employee Data
| Attribute | Detail |
|-----------|--------|
| **Purpose** | Employment administration, payroll, legal compliance |
| **Personal Data** | Name, address, NI number, bank details, salary, absence records, performance reviews, disciplinary records |
| **Special Category Data** | Health data (sick leave); Trade union membership |
| **Legal Bases** | Contract of employment; Legal obligation; Explicit consent (health); Legitimate interests (performance/disciplinary) |
| **Retention — Payroll/Tax** | Employment period + 6 years |
| **Retention — Disciplinary** | 5 years |
| **Recipients** | Moorgate Payroll Ltd; HMRC; Pension provider |
| **International Transfers** | None |

**Confidence: High**""",
    difficulty="medium"
))

print("Part 6 (examples 31-35) defined. Continuing to part 7...")

# ─── 36. Construction Contract ────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract key terms from this JCT contract summary:

CONTRACT DATA — JCT DESIGN AND BUILD CONTRACT 2016
Project: Northfield Logistics Hub — Phase 2 Warehousing
Employer: Northfield Property Developments Ltd
Contractor: Meridian Construction Group Ltd
Contract Administrator: BDP Architects LLP (Employer's Agent)

CONTRACT SUM: £12,847,500 (excluding VAT)
BASE DATE: 01 November 2023
DATE FOR POSSESSION: 08 January 2024
DATE FOR COMPLETION: 15 October 2024 (40 weeks)

LIQUIDATED DAMAGES: £18,500 per week (or pro rata for any period less than a week)

DEFECTS LIABILITY PERIOD: 12 months from practical completion

RETENTION:
Retention percentage: 5%
Retention release: Half on practical completion; half at end of Defects Liability Period

INSURANCE
Works insurance: All risks by Contractor — minimum £15,000,000
Public liability: Minimum £10,000,000 — Contractor
Professional indemnity: £5,000,000 per claim — Contractor (design portions)
Employer's liability: £30,000,000 — statutory minimum compliance

FLUCTUATIONS: Contract sum adjustment: Option A (not applicable — fixed price)

ADVANCE PAYMENT: £500,000 — payable on contract execution (recoverable from first 3 valuations)

NAMED SUBCONTRACTORS (Schedule 7):
Structural steel: Apex Steelwork Ltd
Mechanical & Electrical: ClearAir Building Services Ltd
Sprinkler system: FireShield UK Ltd""",
    reasoning="""This is contract data for a JCT Design and Build 2016 contract for a logistics warehouse. I need to extract all key commercial and legal terms.

Parties: Employer Northfield Property Developments Ltd, Contractor Meridian Construction Group Ltd, Employer's Agent BDP Architects LLP.

Contract sum: £12,847,500 (ex-VAT).
Base date: 01 Nov 2023. Possession: 08 Jan 2024. Completion: 15 Oct 2024.

Programme check: 08 Jan to 15 Oct 2024. Jan: 23 days remaining (Jan has 31, starting 8th = 24 days from 8-31 Jan). Feb: 29 days (2024 is leap year). Mar: 31, Apr: 30, May: 31, Jun: 30, Jul: 31, Aug: 31, Sep: 30, Oct 1-15 = 15 days. Total = 24+29+31+30+31+30+31+31+30+15 = 282 days = about 40.3 weeks — consistent with stated 40 weeks.

Liquidated damages: £18,500/week. As % of contract sum per week: £18,500/£12,847,500 = 0.144% per week. Annualised = ~7.5% — reasonable pre-estimate of loss for industrial development.

Retention: 5% on valuations. Release: half at practical completion, half at end of DLP (12 months). Maximum retention = 5% × £12,847,500 = £642,375. Half released at PC = £321,188.

Advance payment: £500,000 recoverable from first 3 valuations (i.e. ~£166,667/valuation deduction).

Insurance: works all-risks minimum £15M, PL £10M, PI £5M/claim for design, EL statutory.

Fluctuations: fixed price (Option A).

Named subcontractors: Apex (structural steel), ClearAir (M&E), FireShield (sprinklers).

DLP: 12 months from PC.

Confidence: High.""",
    answer="""## JCT Contract Data Extraction — Northfield Logistics Hub Phase 2

### Contract Parties
| Role | Party |
|------|-------|
| **Employer** | Northfield Property Developments Ltd |
| **Contractor** | Meridian Construction Group Ltd |
| **Employer's Agent** | BDP Architects LLP |
| **Form of Contract** | JCT Design and Build Contract 2016 |

### Key Dates & Programme
| Milestone | Date |
|-----------|------|
| Base Date | 01 November 2023 |
| Date for Possession | 08 January 2024 |
| **Date for Completion** | **15 October 2024** |
| Contract Period | **40 weeks** |

### Financial Terms
| Item | Value |
|------|-------|
| **Contract Sum (ex-VAT)** | **£12,847,500** |
| **Advance Payment** | £500,000 (recoverable — first 3 valuations) |
| **Maximum Retention** | £642,375 (5% × contract sum) |

### Retention
| Event | Release Amount |
|-------|---------------|
| Practical Completion | 50% of total retention held |
| End of Defects Liability Period | Remaining 50% |
| **Retention %** | **5%** |

### Liquidated & Ascertained Damages
| Rate | Per Week |
|------|---------|
| **LADs** | **£18,500/week** (pro-rata for part weeks) |

### Defects Liability Period
- **Duration:** 12 months from practical completion

### Insurance Requirements
| Cover | Minimum Amount | Insured by |
|-------|---------------|-----------|
| All-risks (works) | £15,000,000 | Contractor |
| Public liability | £10,000,000 | Contractor |
| Professional indemnity (design) | £5,000,000 per claim | Contractor |
| Employer's liability | £30,000,000 | Contractor (statutory) |

### Named Subcontractors (Schedule 7)
| Trade | Subcontractor |
|-------|--------------|
| Structural steel | Apex Steelwork Ltd |
| Mechanical & Electrical | ClearAir Building Services Ltd |
| Sprinkler system | FireShield UK Ltd |

### Other Terms
- **Price basis:** Fixed price (Fluctuations Option A — not applicable)

**Confidence: High**""",
    difficulty="hard"
))

# ─── 37. Regulatory Submission ────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract the key information from this FCA regulatory submission summary:

REGULATORY CAPITAL RETURN — SUMMARY
Firm: Bridgestone Capital Advisory Ltd
FCA Reference: 483921
Reporting Period: Q4 2023 (01 October – 31 December 2023)
Report submitted: 30 January 2024
ICARA Review Date: 15 March 2024

REGULATORY FRAMEWORK: IFPRU 125K (Investment Firm Prudential Regime — Annex I exempt)

OWN FUNDS
Common Equity Tier 1 (CET1):
  Share capital:                     £500,000
  Retained earnings:                 £1,847,320
  Current year retained profit:       £284,190
  Less: Intangible assets           (£38,200)
  Less: Deferred tax assets         (£12,400)
Total CET1:                        £2,580,910

Total Own Funds:                   £2,580,910 (no AT1 or T2 capital)

CAPITAL REQUIREMENTS
Fixed Overheads Requirement (FOR):   £847,200
K-factor requirements:               £312,840
  of which K-AUM:                    £218,400
  of which K-CMH:                     £94,440
Pillar 2 Requirement (ICARA):        £180,000
Total Capital Requirement:         £1,340,040

CAPITAL RATIOS & HEADROOM
Total Capital Ratio: 192.6% (Own Funds / Total Requirement)
Capital Headroom: £1,240,870
Own Funds Threshold Requirement (OFTR): £847,200

LIQUIDITY
Liquid Assets: £1,284,000
Monthly Fixed Overheads: £70,600
Liquidity Coverage Ratio: 18.2 months of fixed overheads""",
    reasoning="""This is an FCA regulatory capital return summary for Bridgestone Capital Advisory Ltd, an IFPRU 125K investment firm. I need to extract all prudential capital data.

Firm: Bridgestone Capital Advisory Ltd, FCA ref 483921. IFPRU 125K framework. Reporting period Q4 2023. ICARA review 15 March 2024.

Own Funds (CET1 only):
Share capital: £500,000
+ Retained earnings: £1,847,320
+ Current year retained profit: £284,190
Less intangibles: (£38,200)
Less deferred tax: (£12,400)
CET1 total: 500,000 + 1,847,320 + 284,190 - 38,200 - 12,400 = £2,580,910 ✓

Total own funds = CET1 only (no AT1 or Tier 2) = £2,580,910 ✓

Capital requirements:
FOR: £847,200 (Fixed Overheads Requirement — 1/4 of annual fixed overheads)
K-factors: £312,840
  K-AUM: £218,400
  K-CMH: £94,440
  K-AUM + K-CMH: 218,400 + 94,440 = £312,840 ✓
Pillar 2 (ICARA): £180,000
Total: 847,200 + 312,840 + 180,000 = £1,340,040 ✓

Capital ratio: 2,580,910 / 1,340,040 = 1.927 = 192.7% — stated as 192.6% (slight rounding). Close enough.

Headroom: 2,580,910 - 1,340,040 = £1,240,870 ✓

OFTR (Own Funds Threshold Requirement = FOR) = £847,200 — this is the minimum floor.

Liquidity: liquid assets £1,284,000. Monthly fixed overheads £70,600. Check: FOR = £847,200 = monthly fixed overheads × 12 = 70,600 × 12 = £847,200 ✓

LCR in months: 1,284,000 / 70,600 = 18.19 months ≈ 18.2 months ✓

The firm is well-capitalised at 192.6% of total capital requirement with 18.2 months of liquid assets.

Confidence: High.""",
    answer="""## FCA Regulatory Capital Return Extraction — Bridgestone Capital Advisory

### Firm & Return Details
| Field | Value |
|-------|-------|
| **Firm** | Bridgestone Capital Advisory Ltd |
| **FCA Reference** | 483921 |
| **Regulatory Framework** | IFPRU 125K (Annex I exempt) |
| **Reporting Period** | Q4 2023 (01 Oct – 31 Dec 2023) |
| **Submission Date** | 30 January 2024 |
| **ICARA Review Date** | 15 March 2024 |

### Own Funds (CET1)
| Component | Amount (£) |
|-----------|-----------|
| Share Capital | 500,000 |
| Retained Earnings | 1,847,320 |
| Current Year Retained Profit | 284,190 |
| Less: Intangible Assets | (38,200) |
| Less: Deferred Tax Assets | (12,400) |
| **Total CET1 / Total Own Funds** | **£2,580,910** |

*(No Additional Tier 1 or Tier 2 capital)*

### Capital Requirements
| Requirement | Amount (£) |
|-------------|-----------|
| Fixed Overheads Requirement (FOR) | 847,200 |
| K-AUM | 218,400 |
| K-CMH | 94,440 |
| **K-factor Total** | **312,840** |
| Pillar 2 (ICARA) | 180,000 |
| **Total Capital Requirement** | **£1,340,040** |

### Capital Position
| Metric | Value |
|--------|-------|
| **Total Capital Ratio** | **192.6%** |
| **Capital Headroom** | **£1,240,870** |
| Own Funds Threshold Requirement (OFTR) | £847,200 |

### Liquidity
| Metric | Value |
|--------|-------|
| **Liquid Assets** | £1,284,000 |
| Monthly Fixed Overheads | £70,600 |
| **Liquidity Coverage** | **18.2 months** of fixed overheads |

### Arithmetic Verification
- CET1: £500k + £1,847,320 + £284,190 − £38,200 − £12,400 = £2,580,910 ✓
- K-factors: £218,400 + £94,440 = £312,840 ✓
- Total requirement: £847,200 + £312,840 + £180,000 = £1,340,040 ✓
- FOR consistency: £847,200 ÷ 12 = £70,600/month ✓
- Liquidity ratio: £1,284,000 ÷ £70,600 = 18.2 months ✓

**Confidence: High** — firm appears well-capitalised and liquid""",
    difficulty="hard"
))

# ─── 38. Supply Chain Agreement ───────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract the key obligations from this distribution agreement:

DISTRIBUTION AGREEMENT
Supplier: Nordic Wellness Foods AS (Norway, org. no. 923 847 012)
Distributor: Greenwave Distribution Ltd (UK, Co. No. 11748293)
Effective Date: 01 February 2024
Territory: United Kingdom and Republic of Ireland

APPOINTMENT
Exclusive distributor for the Territory for the Products listed in Schedule 1.
Exclusivity subject to meeting Minimum Purchase Obligations (MPO).

PRODUCTS: Nordic Omega-3 range (12 SKUs), Nordic Collagen range (8 SKUs) — 20 SKUs total

MINIMUM PURCHASE OBLIGATIONS
Year 1 (2024): £480,000
Year 2 (2025): £624,000 (+30%)
Year 3 (2026): £811,200 (+30%)
Payment: 60-day payment terms from invoice date

PRICING
Transfer price: 42% of UK recommended retail price (RRP) as published in price list
RRP review: annual, 90 days notice to Distributor
Trade discount to retailers: Distributor's discretion (within approved margins)

OBLIGATIONS OF DISTRIBUTOR
1. Maintain minimum 3 months' stock cover at all times
2. Cold chain storage at 2–8°C for applicable SKUs
3. Food safety management system (FSMS) certified to BRC AA grade or equivalent
4. Submit quarterly sales reports by 15th of following month
5. Not distribute competing omega-3 or collagen products (non-compete)

MARKETING
Marketing Development Fund (MDF): 2% of annual purchases
Distributor to submit marketing plan for approval; spend tracked quarterly

TERM AND TERMINATION
Initial term: 3 years. Auto-renewal: 12 months.
Termination for MPO failure: 60 days' written notice if MPO missed for 2 consecutive quarters.
Termination for cause: 30 days' notice.""",
    reasoning="""This is an exclusive distribution agreement between Nordic Wellness Foods (Norwegian supplier) and Greenwave Distribution (UK distributor). I need to extract all commercial and operational terms.

Parties: Nordic Wellness Foods AS (Norway, 923 847 012) — supplier; Greenwave Distribution Ltd (UK, 11748293) — distributor. Effective 01 Feb 2024. Territory: UK and Republic of Ireland.

Appointment: exclusive for territory, conditional on meeting MPO.

Products: Nordic Omega-3 (12 SKUs) + Nordic Collagen (8 SKUs) = 20 SKUs total.

MPO:
Year 1: £480,000
Year 2: £624,000 = £480,000 × 1.30 = £624,000 ✓
Year 3: £811,200 = £624,000 × 1.30 = £811,200 ✓
3-year total commitment: £480,000 + £624,000 + £811,200 = £1,915,200
Payment terms: 60 days from invoice.

Pricing: 42% of UK RRP. So if RRP is £100, transfer price = £42. This leaves distributor margin from £42 to whatever they sell to trade minus their costs.

Distributor obligations:
1. Min 3 months stock cover always maintained
2. Cold chain 2–8°C for applicable SKUs
3. BRC AA grade FSMS or equivalent (high food safety standard)
4. Quarterly sales reports by 15th of following month
5. Non-compete on omega-3 and collagen products

MDF: 2% of annual purchases. Year 1 = 2% × £480,000 = £9,600 MDF.

Term: 3 years, auto-renews 12 months. MPO failure: termination right after 2 consecutive quarters of MPO miss, 60 days notice. For cause: 30 days.

Governing jurisdiction: not stated in this excerpt.

Confidence: High.""",
    answer="""## Distribution Agreement Extraction — Nordic Wellness / Greenwave

### Agreement Details
| Field | Value |
|-------|-------|
| **Supplier** | Nordic Wellness Foods AS (Norway, org. no. 923 847 012) |
| **Distributor** | Greenwave Distribution Ltd (UK, Co. No. 11748293) |
| **Effective Date** | 01 February 2024 |
| **Territory** | United Kingdom and Republic of Ireland |
| **Appointment Type** | **Exclusive** (conditional on MPO) |

### Products
| Range | SKU Count |
|-------|-----------|
| Nordic Omega-3 | 12 SKUs |
| Nordic Collagen | 8 SKUs |
| **Total** | **20 SKUs** |

### Minimum Purchase Obligations (MPO)
| Year | MPO | Growth |
|------|-----|--------|
| Year 1 (2024) | £480,000 | — |
| Year 2 (2025) | £624,000 | +30% |
| Year 3 (2026) | £811,200 | +30% |
| **3-Year Total** | **£1,915,200** | |
| **Payment Terms** | 60 days from invoice | |

### Pricing Structure
| Parameter | Value |
|-----------|-------|
| **Transfer Price** | 42% of UK RRP |
| RRP Review | Annual — 90 days advance notice to distributor |
| Trade Discounts | Distributor's discretion (within approved margins) |

### Distributor Obligations
| # | Obligation |
|---|-----------|
| 1 | Minimum **3 months' stock cover** at all times |
| 2 | Cold chain storage **2–8°C** for applicable SKUs |
| 3 | FSMS certified to **BRC AA grade** or equivalent |
| 4 | Quarterly sales reports by **15th** of following month |
| 5 | **Non-compete** — no competing omega-3 or collagen products |

### Marketing Development Fund (MDF)
| Metric | Value |
|--------|-------|
| MDF Rate | 2% of annual purchases |
| Year 1 MDF | £9,600 (2% × £480,000) |
| Governance | Marketing plan submitted for approval; quarterly spend tracking |

### Term & Termination
| Provision | Detail |
|-----------|--------|
| Initial Term | 3 years |
| Auto-Renewal | 12 months |
| **MPO Failure** | 60 days notice after **2 consecutive quarters** of MPO miss |
| For Cause | 30 days notice |

**Confidence: High** — all MPO escalators and calculations verified""",
    difficulty="medium"
))

# ─── 39. Credit Assessment Report ────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract the key credit metrics from this business credit assessment:

CREDIT ASSESSMENT REPORT
Prepared by: Thornfield Commercial Finance Ltd — Credit Committee
Assessment Date: 28 February 2024
Borrower: Castlegate Engineering Ltd (Co. No. 09384729)
Facility Requested: £2,500,000 revolving credit facility (RCF)
Industry: Precision engineering (SIC 2562)
Directors/Guarantors: M. Castlegate (100% shareholder)

FINANCIAL SUMMARY (last 3 years, year-end March)
                         FY2023      FY2022      FY2021
Revenue (£k)              8,420       7,340       6,180
Gross Profit (£k)         2,947       2,572       2,101
GP Margin                  35.0%       35.0%       34.0%
EBITDA (£k)               1,264       1,028         874
EBITDA Margin              15.0%       14.0%       14.1%
Net Profit (£k)             748         612         487
Net Debt (£k)             1,840       1,920       2,100
Net Debt / EBITDA          1.46×       1.87×       2.40×

CREDIT SCORING
Credit Bureau Score: 74/100 (Good)
Payment history: 3 late payments (>30 days) in 36 months
CCJs: None
Director's personal credit: Score 812 (Experian — Excellent)

KEY RATIOS
Interest Cover: 4.2× (FY2023)
Current Ratio: 1.68
Quick Ratio: 1.21
Debtor Days: 52 days
Creditor Days: 38 days
Cash Conversion Cycle: 34 days (52 - 38 + 20 [stock days])

ASSESSMENT OUTCOME
Proposed facility: Approved in principle — £2,500,000 RCF
Recommended pricing: SONIA + 3.25% margin
Covenant: Net Debt/EBITDA ≤ 2.5× (tested semi-annually)
Security: Debenture over assets + personal guarantee M. Castlegate (£750,000)
Condition: Audited accounts within 180 days of year-end""",
    reasoning="""This is a commercial credit assessment for Castlegate Engineering Ltd applying for a £2.5M revolving credit facility. I need to extract all credit metrics and assessment outcomes.

Borrower: Castlegate Engineering Ltd, Co. No. 09384729, precision engineering, 100% owned by M. Castlegate.

Financial performance trends:
Revenue: £6,180K → £7,340K → £8,420K — growing 18.7% and 14.7% YoY — positive trend.
GP margin stable at ~35% — pricing power maintained.
EBITDA growing: £874K → £1,028K → £1,264K — EBITDA margin improving 14.0% → 15.0%.
Net profit: £487K → £612K → £748K — consistent profitability.
Net Debt reducing: £2,100K → £1,920K → £1,840K — deleveraging.
Net Debt/EBITDA improving: 2.40× → 1.87× → 1.46× — strong improvement trend.

Credit bureau: 74/100 (Good). 3 late payments in 36 months — minor blemishes. No CCJs. Director personal score 812 (Excellent Experian).

Key ratios FY2023:
Interest cover 4.2× — comfortable
Current ratio 1.68 — healthy liquidity
Quick ratio 1.21 — adequate
Debtor days 52 — slightly long but industry typical
Creditor days 38 — paying promptly
Cash conversion cycle: 52 - 38 + 20 (stock) = 34 days ✓

Assessment outcome: APPROVED IN PRINCIPLE.
Pricing: SONIA + 3.25%.
Covenant: ND/EBITDA ≤ 2.5× (current 1.46× = significant headroom).
Security: debenture + personal guarantee £750,000 (30% of facility amount).
Condition: audited accounts within 180 days of year-end.

Confidence: High.""",
    answer="""## Credit Assessment Extraction — Castlegate Engineering Ltd

### Assessment Overview
| Field | Value |
|-------|-------|
| **Borrower** | Castlegate Engineering Ltd (Co. No. 09384729) |
| **Industry** | Precision Engineering (SIC 2562) |
| **Facility Requested** | £2,500,000 Revolving Credit Facility |
| **Assessment Date** | 28 February 2024 |
| **Guarantor** | M. Castlegate (100% shareholder) |

### Financial Performance
| Metric | FY2021 | FY2022 | FY2023 | Trend |
|--------|--------|--------|--------|-------|
| Revenue (£k) | 6,180 | 7,340 | **8,420** | ↑ Growing |
| Gross Profit (£k) | 2,101 | 2,572 | **2,947** | ↑ |
| **GP Margin** | 34.0% | 35.0% | **35.0%** | → Stable |
| EBITDA (£k) | 874 | 1,028 | **1,264** | ↑ Growing |
| **EBITDA Margin** | 14.1% | 14.0% | **15.0%** | ↑ Improving |
| Net Profit (£k) | 487 | 612 | **748** | ↑ Growing |
| Net Debt (£k) | 2,100 | 1,920 | **1,840** | ↓ Reducing |
| **Net Debt / EBITDA** | 2.40× | 1.87× | **1.46×** | ↓ Improving |

### Credit Scores
| Metric | Score | Grade |
|--------|-------|-------|
| Business credit bureau | 74/100 | Good |
| Late payments (36 months) | 3 | Minor |
| CCJs | None | — |
| Director personal credit (Experian) | 812 | **Excellent** |

### Key Financial Ratios (FY2023)
| Ratio | Value | Assessment |
|-------|-------|-----------|
| Interest Cover | **4.2×** | Comfortable |
| Current Ratio | 1.68 | Healthy |
| Quick Ratio | 1.21 | Adequate |
| Debtor Days | 52 days | Industry typical |
| Creditor Days | 38 days | Prompt payment |
| Cash Conversion Cycle | **34 days** | Efficient |

### Assessment Outcome
| Item | Detail |
|------|--------|
| **Decision** | **APPROVED IN PRINCIPLE** |
| **Facility** | £2,500,000 RCF |
| **Pricing** | SONIA + 3.25% margin |
| **Covenant** | Net Debt / EBITDA ≤ 2.5× (semi-annual testing) |
| **Covenant Headroom** | 1.46× vs 2.5× limit = **1.04× headroom** |
| **Security** | Debenture over assets + Personal guarantee £750,000 (M. Castlegate) |
| **Condition** | Audited accounts within 180 days of year-end |

**Confidence: High** — all financial metrics extracted and trend analysis confirmed""",
    difficulty="hard"
))

# ─── 40. Fire Safety Risk Assessment ─────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract all findings from this fire safety risk assessment:

FIRE RISK ASSESSMENT
Premises: Greystone Mill Business Park, Unit 14 (Haverstock Packaging Ltd)
Address: Greystone Mill Lane, Huddersfield HD3 4FW
Assessment Date: 29 January 2024
Assessor: Robert Fenwick, CMIOSH, Fire Risk Assessor
Assessment Valid Until: 28 January 2025 (or following material change)

PREMISES SUMMARY
Occupancy type: Light industrial / warehouse
Occupants: 34 employees (max. on site simultaneously)
Shift pattern: 06:00–22:00 Mon–Fri; 06:00–14:00 Saturday
Fire strategy: Detected evacuation — Addressable alarm system (Kentec panel)
Fire suppression: None (sprinklers not installed)
Structural construction: Steel frame, block inner leaf, insulated cladding (PIR core)

FIRE HAZARD FINDINGS
1. [CRITICAL] PIR Insulated Cladding — unprotected polyisocyanurate core sections
   identified at rear elevation (Est. 80 sq m). PIR presents high fire spread risk.
   Action: Engage specialist cladding contractor for remediation / cavity barriers.
   Deadline: Within 60 days (29 March 2024). Owner: Director.

2. [HIGH] Cardboard bale storage abutting electrical distribution board in warehouse.
   Action: Maintain 1-metre clear zone around all electrical panels. Immediate.
   Owner: Warehouse Manager.

3. [HIGH] Fire door to server room — self-closing device defective.
   Action: Replace self-closer. Within 14 days (12 February 2024). Owner: FM.

4. [MEDIUM] Fire extinguisher annual service overdue by 4 months (10 of 18 units).
   Action: Book service contractor within 7 days. Owner: FM.

5. [MEDIUM] Emergency lighting: 3 luminaires in mezzanine staircase failed last test.
   Action: Replace faulty luminaires within 14 days. Owner: FM.

6. [LOW] Fire action notices not posted at all call points (4 missing).
   Action: Install notices. Within 30 days. Owner: FM.

OVERALL RISK RATING: HIGH (due to Finding 1)
Previous assessment rating: MEDIUM (2023)""",
    reasoning="""This is a fire risk assessment for a light industrial/warehouse premises. I need to extract all findings, risk ratings, actions and deadlines.

Premises: Haverstock Packaging Ltd, Unit 14 Greystone Mill Business Park, Huddersfield HD3 4FW. Assessment 29 Jan 2024, valid 1 year to 28 Jan 2025.

Premises details: 34 employees max, 06:00–22:00 weekdays, 06:00–14:00 Saturday. Steel frame with PIR cladding. Addressable alarm system (Kentec panel). No sprinklers.

6 findings:

Finding 1 [CRITICAL]: PIR cladding with unprotected core, ~80 sq m rear elevation. PIR is the same type of cladding involved in Grenfell Tower — this is a serious fire spread risk in the UK post-Grenfell regulatory environment. Deadline 29 March 2024 (60 days from 29 Jan 2024 ✓). Owner: Director level — reflects seriousness.

Finding 2 [HIGH]: Cardboard bales next to electrical distribution board — ignition source + fuel = high risk. Immediate action required (no deadline given = now). Owner: Warehouse Manager.

Finding 3 [HIGH]: Fire door self-closer defective (server room). Fire doors need self-closing to be effective. Deadline 12 Feb 2024 (14 days from 29 Jan ✓). Owner: FM.

Finding 4 [MEDIUM]: Fire extinguisher service overdue 4 months — 10 of 18 units affected (55.6%). 7-day deadline to book. Owner: FM.

Finding 5 [MEDIUM]: 3 emergency lighting luminaires failed in mezzanine staircase. 14-day deadline. Owner: FM.

Finding 6 [LOW]: 4 missing fire action notices. 30-day deadline. Owner: FM.

Overall rating: HIGH (upgraded from MEDIUM due to Finding 1 — PIR cladding). Immediate regulatory compliance concern.

Confidence: High.""",
    answer="""## Fire Risk Assessment Extraction — Haverstock Packaging Ltd

### Assessment Details
| Field | Value |
|-------|-------|
| **Premises** | Unit 14, Greystone Mill Business Park, Huddersfield HD3 4FW |
| **Occupier** | Haverstock Packaging Ltd |
| **Assessment Date** | 29 January 2024 |
| **Valid Until** | 28 January 2025 (or material change) |
| **Assessor** | Robert Fenwick, CMIOSH |
| **Overall Risk Rating** | **HIGH** (previously MEDIUM — upgraded due to Finding 1) |

### Premises Summary
| Parameter | Detail |
|-----------|--------|
| Occupancy | Light industrial / warehouse |
| Max. occupants | **34 employees** |
| Operating hours | 06:00–22:00 Mon–Fri; 06:00–14:00 Saturday |
| Fire alarm | Addressable system (Kentec panel) |
| Sprinkler system | **None** |
| Construction | Steel frame, block inner leaf, PIR insulated cladding |

### Findings Summary
| # | Severity | Issue | Deadline | Owner |
|---|----------|-------|----------|-------|
| 1 | **CRITICAL** | PIR cladding — unprotected core (80 sq m, rear elevation) | **29 March 2024** (60 days) | Director |
| 2 | **HIGH** | Cardboard bales abutting electrical distribution board | **Immediate** | Warehouse Manager |
| 3 | **HIGH** | Server room fire door — self-closer defective | **12 Feb 2024** (14 days) | FM |
| 4 | **MEDIUM** | Fire extinguisher service overdue 4 months (10/18 units) | Book within **7 days** | FM |
| 5 | **MEDIUM** | Emergency lighting — 3 luminaires failed (mezzanine staircase) | **14 days** | FM |
| 6 | **LOW** | Fire action notices missing at 4 call points | **30 days** | FM |

### Critical Finding Detail — PIR Cladding
- **Hazard:** Unprotected polyisocyanurate (PIR) cladding core — high fire spread risk
- **Area affected:** Approximately 80 sq m, rear elevation
- **Required action:** Engage specialist cladding contractor for remediation / cavity barriers
- **Regulatory context:** PIR cladding compliance is a post-Grenfell regulatory priority in UK

### Remediation Priority Order
| Priority | Action | By |
|----------|--------|-----|
| 1 | Clear 1-metre zone around electrical panels | **Immediately** |
| 2 | Replace fire door self-closer (server room) | 12 Feb 2024 |
| 3 | Book fire extinguisher service | Within 7 days |
| 4 | Replace 3 emergency lighting luminaires | Within 14 days |
| 5 | Engage cladding specialist (PIR) | 29 March 2024 |
| 6 | Install missing fire action notices | Within 30 days |

**Confidence: High** — all findings, deadlines and owners extracted""",
    difficulty="medium"
))

print("Part 7 (examples 36-40) defined. Continuing to part 8...")

# ─── 41. Pension Scheme Valuation ─────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract the key actuarial findings from this pension scheme valuation summary:

ACTUARIAL VALUATION SUMMARY
Scheme: Bridgewater Manufacturing Pension Scheme
Employer: Bridgewater Engineering Holdings Ltd
Actuary: Jonathan Carey FIA, Mercer Ltd
Valuation Date: 31 March 2023
Report Date: 20 September 2023
Scheme Status: Closed to future accrual (since 31 March 2019)

MEMBERSHIP SUMMARY (at valuation date)
Active members:                  0 (scheme closed)
Deferred members:              284
Pensioners:                    341
Total:                         625

FINANCIAL POSITION (Technical Provisions basis)
Assets at valuation date:                    £84,720,000
Technical Provisions (liabilities):          £92,480,000
DEFICIT:                                     £7,760,000
Funding Level:                                   91.6%

FINANCIAL POSITION (Buy-out basis)
Buy-out liabilities (estimated):            £118,400,000
Buy-out funding level:                            71.6%
Buy-out deficit:                             £33,680,000

RECOVERY PLAN
Agreed with Trustees: employer deficit contributions of £960,000 per annum
Recovery period: 8 years and 3 months (completion date: 30 June 2031)
Total recovery plan contributions: £7,920,000

ASSUMPTIONS (Technical Provisions basis)
Discount rate: 1.20% per annum (gilts + 0.50%)
RPI inflation: 3.15% per annum
CPI inflation: 2.65% per annum (RPI minus 0.50%)
Mortality: S3PMA / S3PFA tables, CMI_2022 projections, 1.25% long-term rate""",
    reasoning="""This is an actuarial valuation summary for a closed defined benefit pension scheme. I need to extract all financial position, membership and recovery plan data.

Scheme: Bridgewater Manufacturing Pension Scheme. Employer: Bridgewater Engineering Holdings Ltd. Actuary: Jonathan Carey FIA, Mercer. Valuation date 31 March 2023.

Membership: 0 active (closed since March 2019), 284 deferred, 341 pensioners = 625 total. More pensioners than deferreds — consistent with a mature closed scheme.

Technical Provisions (TP) basis:
Assets: £84,720,000
TP liabilities: £92,480,000
Deficit: 92,480,000 - 84,720,000 = £7,760,000 ✓
Funding level: 84,720,000/92,480,000 = 91.62% ≈ 91.6% ✓

Buy-out basis:
Buy-out liabilities: £118,400,000 (much higher than TP — insurers price more conservatively)
Buy-out deficit: 118,400,000 - 84,720,000 = £33,680,000 ✓
Buy-out funding level: 84,720,000/118,400,000 = 71.55% ≈ 71.6% ✓

The gap between TP liabilities (£92.5M) and buy-out liabilities (£118.4M) = £25.9M, reflecting the premium for insurance pricing. Buy-out is about 28.1% more expensive than TP basis.

Recovery plan: £960,000/year for 8 years 3 months = 99 months. Total: 8 years × £960,000 = £7,680,000 + 3/12 × £960,000 = £240,000 = £7,920,000 ✓ (stated).

Current TP deficit £7,760,000 vs recovery plan total £7,920,000 — slight excess due to investment returns assumed over period. Close match — actuarially consistent.

Assumptions: discount rate gilts + 0.50% = 1.20%, so gilt yield implied = 0.70% — this reflects March 2023 gilt yields which were actually higher. May be a snapshot or risk-adjusted rate. RPI 3.15%, CPI 2.65% (50bp spread). Mortality: standard tables with CMI_2022.

Confidence: High — all arithmetic verified.""",
    answer="""## Pension Scheme Valuation Extraction — Bridgewater Manufacturing Pension Scheme

### Valuation Details
| Field | Value |
|-------|-------|
| **Scheme** | Bridgewater Manufacturing Pension Scheme |
| **Employer** | Bridgewater Engineering Holdings Ltd |
| **Actuary** | Jonathan Carey FIA, Mercer Ltd |
| **Valuation Date** | 31 March 2023 |
| **Report Date** | 20 September 2023 |
| **Scheme Status** | Closed to future accrual (since 31 March 2019) |

### Membership (at valuation date)
| Category | Count |
|----------|-------|
| Active Members | 0 (scheme closed) |
| Deferred Members | 284 |
| Pensioners | 341 |
| **Total** | **625** |

### Financial Position — Technical Provisions (TP) Basis
| Metric | Value |
|--------|-------|
| **Assets** | £84,720,000 |
| **TP Liabilities** | £92,480,000 |
| **TP Deficit** | **£7,760,000** |
| **Funding Level** | **91.6%** |

### Financial Position — Buy-Out Basis
| Metric | Value |
|--------|-------|
| Buy-Out Liabilities | £118,400,000 |
| **Buy-Out Deficit** | **£33,680,000** |
| **Buy-Out Funding Level** | **71.6%** |
| TP vs Buy-Out liability gap | £25,920,000 (+28.1%) |

### Recovery Plan
| Parameter | Value |
|-----------|-------|
| Annual Employer Contributions | **£960,000** |
| Recovery Period | 8 years 3 months |
| Recovery Plan End Date | 30 June 2031 |
| **Total Recovery Contributions** | **£7,920,000** |

### Actuarial Assumptions (Technical Provisions)
| Assumption | Rate |
|------------|------|
| Discount Rate | 1.20% p.a. (gilts + 0.50%) |
| RPI Inflation | 3.15% p.a. |
| CPI Inflation | 2.65% p.a. (RPI − 0.50%) |
| Mortality Basis | S3PMA/S3PFA tables, CMI_2022 projections, 1.25% long-term rate |

### Arithmetic Verification
- TP Deficit: £92,480,000 − £84,720,000 = £7,760,000 ✓
- TP Funding Level: £84,720,000 ÷ £92,480,000 = 91.6% ✓
- Buy-Out Deficit: £118,400,000 − £84,720,000 = £33,680,000 ✓
- Recovery total: (8 × £960k) + (3/12 × £960k) = £7,920,000 ✓

**Confidence: High**""",
    difficulty="hard"
))

# ─── 42. Software Requirements Specification ──────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract all functional and non-functional requirements from this SRS excerpt:

SOFTWARE REQUIREMENTS SPECIFICATION
Document: SRS-PORTAL-2024-001 v1.2
Project: Northgate Council — Resident Self-Service Portal
Author: Enterprise Solutions Team, Northgate Borough Council
Date: 15 February 2024

1. FUNCTIONAL REQUIREMENTS

FR-001: User Registration
  The system shall allow residents to register using a valid UK postcode and
  council tax account number. Registration must be verified via email within
  24 hours. Unverified accounts expire after 72 hours.

FR-002: Bill Payment
  The system shall accept payments via credit/debit card (Visa, Mastercard, Amex)
  and direct debit setup. Payments shall be processed through the council's
  approved payment gateway (Civica Pay). Minimum payment: £1.00. Maximum: £5,000.

FR-003: Planning Application Tracking
  Residents shall be able to view status of planning applications using the
  planning reference number. Status updates within 2 hours of case officer action.
  Documents attached to applications (PDFs) downloadable by applicants.

FR-004: Report Street Issues
  Residents can report: potholes, street lighting faults, fly-tipping, graffiti.
  Each report generates a unique reference number. Auto-acknowledge within 15 minutes.
  SLA for response: Category A (safety) 4 hours; Category B (other) 72 hours.

FR-005: Waste Collection Schedule
  System shall display collection schedule for resident's address based on UPRN lookup.
  iCal subscription export. Reminders: configurable email/SMS up to 7 days in advance.

2. NON-FUNCTIONAL REQUIREMENTS

NFR-001: Performance — Page load time ≤2 seconds at 95th percentile under normal load.
NFR-002: Availability — 99.5% uptime excluding planned maintenance (max 4hr/month).
NFR-003: Scalability — System shall support 10,000 concurrent users without degradation.
NFR-004: Security — OWASP Top 10 compliance; Cyber Essentials Plus certified hosting.
NFR-005: Accessibility — WCAG 2.1 Level AA compliance; tested with 3 assistive technologies.
NFR-006: Data Retention — Personal data retained max 7 years per council retention schedule.""",
    reasoning="""This is a Software Requirements Specification for a council resident self-service portal. I need to extract all functional and non-functional requirements with their specific parameters.

Document: SRS-PORTAL-2024-001 v1.2, Northgate Borough Council, dated 15 February 2024.

Functional requirements (5):

FR-001 User Registration:
- UK postcode + council tax account number for registration
- Email verification within 24 hours
- Unverified accounts expire at 72 hours

FR-002 Bill Payment:
- Card types: Visa, Mastercard, Amex; also direct debit setup
- Payment gateway: Civica Pay (approved)
- Payment limits: min £1.00, max £5,000

FR-003 Planning Application Tracking:
- Search by planning reference number
- Status updates within 2 hours of case officer action
- PDF documents downloadable by applicants

FR-004 Street Issue Reporting:
- Categories: potholes, street lighting, fly-tipping, graffiti
- Unique reference per report
- Auto-acknowledge within 15 minutes
- SLA: Category A (safety) 4 hours; Category B (other) 72 hours

FR-005 Waste Collection:
- UPRN-based schedule lookup
- iCal export
- Email/SMS reminders configurable up to 7 days ahead

Non-functional requirements (6):

NFR-001 Performance: page load ≤2 seconds at 95th percentile
NFR-002 Availability: 99.5% uptime, max 4hr/month planned maintenance
NFR-003 Scalability: 10,000 concurrent users without degradation
NFR-004 Security: OWASP Top 10 compliant, Cyber Essentials Plus certified hosting
NFR-005 Accessibility: WCAG 2.1 Level AA, tested with 3 assistive technologies
NFR-006 Data retention: max 7 years per council retention schedule

Confidence: High.""",
    answer="""## SRS Extraction — Northgate Resident Self-Service Portal

### Document Details
| Field | Value |
|-------|-------|
| **Document** | SRS-PORTAL-2024-001 Version 1.2 |
| **Project** | Northgate Borough Council — Resident Self-Service Portal |
| **Author** | Enterprise Solutions Team |
| **Date** | 15 February 2024 |

### Functional Requirements
| ID | Feature | Key Parameters |
|----|---------|---------------|
| **FR-001** | User Registration | UK postcode + council tax account; email verify within 24hrs; accounts expire after 72hrs if unverified |
| **FR-002** | Bill Payment | Visa/Mastercard/Amex + direct debit; Civica Pay gateway; **min £1.00 / max £5,000** |
| **FR-003** | Planning Application Tracking | Search by reference; status updates within **2 hours** of case officer action; PDFs downloadable |
| **FR-004** | Report Street Issues | Potholes, lighting, fly-tipping, graffiti; unique reference; auto-acknowledge **<15 min**; Cat A SLA **4 hrs**; Cat B SLA **72 hrs** |
| **FR-005** | Waste Collection Schedule | UPRN lookup; iCal export; email/SMS reminders configurable up to **7 days** in advance |

### FR-004 Issue Category SLAs
| Category | Type | Response SLA |
|----------|------|-------------|
| **Category A** | Safety issues | **4 hours** |
| **Category B** | All other issues | **72 hours** |

### Non-Functional Requirements
| ID | Category | Requirement |
|----|----------|-------------|
| **NFR-001** | Performance | Page load **≤2 seconds** at 95th percentile (normal load) |
| **NFR-002** | Availability | **99.5% uptime**; max 4 hours/month planned maintenance |
| **NFR-003** | Scalability | **10,000 concurrent users** without degradation |
| **NFR-004** | Security | OWASP Top 10 compliance; **Cyber Essentials Plus** certified hosting |
| **NFR-005** | Accessibility | **WCAG 2.1 Level AA**; tested with 3 assistive technologies |
| **NFR-006** | Data Retention | Personal data max **7 years** (council retention schedule) |

**Confidence: High** — all requirements extracted with specific measurable parameters""",
    difficulty="medium"
))

# ─── 43. Joint Venture Agreement ──────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract the key terms from this joint venture agreement:

JOINT VENTURE AGREEMENT — HEADS OF TERMS
Project: Development and operation of 180,000 sq ft Grade A logistics park
Site: Former Whitmore Quarry, Doncaster DN7 6PQ (freehold owned by Party A)

PARTIES
Party A: Whitmore Land Group Ltd (Co. No. 09284710) — Landowner
Party B: Apex Developments (UK) Ltd (Co. No. 07284910) — Developer

JOINT VENTURE VEHICLE
A new special purpose vehicle (SPV) shall be incorporated: Whitmore Apex
Logistics Ltd (to be incorporated). Each party will hold 50% of ordinary shares.

CONTRIBUTIONS
Party A: Land value at £4,800,000 (independent RICS valuation), contributed at practical completion.
Party B: Development funding — estimated total development cost £22,400,000 (excl. land).
Party A: Day-to-day development management team input (no cash contribution required).

TOTAL DEVELOPMENT VALUE (GDV): £38,500,000 (estimated)
TOTAL COST (land + development): £27,200,000
ESTIMATED PROFIT: £11,300,000 (29.4% on cost; 29.3% margin on GDV)

PROFIT DISTRIBUTION
After repayment of Party B's development funding (£22,400,000) and Party A's land contribution
(£4,800,000) — profit split 50:50.
Preferred return: Party B receives 8% annual return on unrecovered development funding
until project completion.

EXIT STRATEGY
Target: Forward sale to logistics REIT or institutional investor at practical completion.
Longstop date for exit: 36 months from planning grant.
If no exit achieved: either party may trigger compulsory purchase mechanism.""",
    reasoning="""This is a JV heads of terms for a logistics park development. I need to extract all financial, structural and governance terms.

Parties: Whitmore Land Group (landowner, Co. 09284710) and Apex Developments (developer, Co. 07284910). 50:50 SPV to be incorporated as Whitmore Apex Logistics Ltd.

Contributions:
- Party A (Whitmore): land at £4,800,000 (RICS valuation), contributed at PC + management input
- Party B (Apex): development funding £22,400,000 cash

Total cost: £4,800,000 (land) + £22,400,000 (dev) = £27,200,000 ✓

GDV: £38,500,000 stated.
Estimated profit: £38,500,000 - £27,200,000 = £11,300,000 ✓
Profit on cost: £11,300,000/£27,200,000 = 41.5% — but stated 29.4%. Let me recheck: document states £11,300,000 is 29.4% on cost. £27,200,000 × 29.4% = £7,996,800 ≠ £11,300,000. There's a discrepancy. £11,300,000/£27,200,000 = 41.5%, not 29.4%. And margin on GDV: £11,300,000/£38,500,000 = 29.4% ✓. So the stated "29.4% on cost" is actually "29.4% on GDV" — it appears the labels are swapped. The stated "29.3% margin on GDV" is actually "on cost" = 41.5% approximately. The GDV margin is 29.4% and cost margin is 41.5% — the labels in the document appear reversed.

I should flag this discrepancy.

Profit waterfall:
1. Return Party B's £22,400,000 development funding
2. Return Party A's £4,800,000 land contribution
3. Remaining profit split 50:50
Party B preferred return: 8% p.a. on unrecovered development funding during project.

Exit: forward sale at practical completion, longstop 36 months from planning grant. If no exit: compulsory purchase mechanism either party can trigger.

Confidence: High for most items; margin calculation labelling discrepancy flagged.""",
    answer="""## Joint Venture Agreement Extraction — Whitmore Quarry Logistics Park

### Project Overview
| Field | Value |
|-------|-------|
| **Project** | 180,000 sq ft Grade A Logistics Park |
| **Site** | Former Whitmore Quarry, Doncaster DN7 6PQ |
| **JV Vehicle** | Whitmore Apex Logistics Ltd (SPV — to be incorporated) |

### Parties & Equity
| Party | Entity | Role | Equity |
|-------|--------|------|--------|
| Party A | Whitmore Land Group Ltd (Co. 09284710) | Landowner + management | **50%** |
| Party B | Apex Developments (UK) Ltd (Co. 07284910) | Developer + funder | **50%** |

### Contributions
| Party | Contribution | Value |
|-------|-------------|-------|
| **Party A** | Land (RICS valuation — contributed at PC) | **£4,800,000** |
| **Party B** | Development funding (cash) | **£22,400,000** |
| Party A | Development management (no cash) | Equity contribution |

### Project Financials
| Metric | Value |
|--------|-------|
| Total Development Cost (excl. land) | £22,400,000 |
| Land Value | £4,800,000 |
| **Total Cost** | **£27,200,000** |
| **Gross Development Value (GDV)** | **£38,500,000** |
| **Estimated Profit** | **£11,300,000** |
| Profit margin on GDV | **29.4%** |
| Profit on cost | **41.5%** |

> **Note:** Document states "29.4% on cost" but £11,300,000 ÷ £27,200,000 = 41.5% on cost; 29.4% is the margin on GDV. The labels appear transposed in the source document.

### Profit Distribution Waterfall
| Step | Distribution |
|------|-------------|
| 1 | Repay Party B development funding: **£22,400,000** |
| 2 | Repay Party A land contribution: **£4,800,000** |
| 3 | Residual profit: **50:50 split** |
| Preferred return | Party B earns **8% p.a.** on unrecovered funding until completion |

### Exit Strategy
| Parameter | Detail |
|-----------|--------|
| Target exit | Forward sale to logistics REIT or institutional investor (at PC) |
| Longstop date | **36 months from planning grant** |
| Fallback | Compulsory purchase mechanism (either party may trigger) |

**Confidence: High** — one margin label discrepancy flagged; all arithmetic verified""",
    difficulty="hard"
))

# ─── 44. Benefits Award Letter ────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract all benefit entitlements from this DWP award letter:

UNIVERSAL CREDIT AWARD SUMMARY
Department for Work and Pensions
Issued: 14 February 2024
Claimant: Ms. Fatima Al-Rashidi
National Insurance Number: KN 47 28 63 B
Case Reference: UC-2024-NW-847219

HOUSEHOLD: Ms. Al-Rashidi + 2 dependent children (ages 7 and 4)
Monthly assessment period: 14 of each month to 13 of following month

STANDARD ALLOWANCE
Single claimant (25+):                    £368.74

ELEMENTS AWARDED
Child element (1st child — born before 6 Apr 2017): £315.00
Child element (2nd child):                           £269.58
Disabled Child Addition (2nd child — lower rate):    £156.11
Housing Costs Element (private rental, Northgate):   £782.40
Childcare element: up to 85% of costs up to £1,014.63 (2 children)

WORK ALLOWANCE
Higher work allowance (housing costs element applies): £404 per month
(Earnings below this threshold have no effect on UC)
Above threshold: UC reduces by 55 pence per £1 earned

TOTAL STANDARD AWARD (before earnings):  £1,891.83
(Housing + Standard + Child elements + Disabled Child)

EARNINGS DEDUCTION (current assessment period)
Gross earnings (employer): £1,240.00
Earnings above work allowance: £1,240 - £404 = £836.00
Deduction from UC: £836 × 55% = £459.80

FINAL UC MONTHLY PAYMENT: £1,432.03

PAYMENT METHOD: BACS  |  Payment date: 17th of each month""",
    reasoning="""This is a Universal Credit award summary for Ms. Fatima Al-Rashidi. I need to extract all components, verify the arithmetic, and present the benefit structure clearly.

Claimant: Fatima Al-Rashidi, NI KN 47 28 63 B. Household: single adult + 2 children (ages 7 and 4). Assessment period 14th–13th monthly.

Standard allowance (single 25+): £368.74

Elements:
- Child element 1st child (pre-Apr 2017 = two-child rule exception or pre-cap): £315.00
- Child element 2nd child: £269.58
- Disabled Child Addition (2nd child, lower rate): £156.11
- Housing costs: £782.40
- Childcare: up to 85% of costs up to £1,014.63 (max 2 children)

Standard award before earnings: 368.74 + 315.00 + 269.58 + 156.11 + 782.40 = £1,891.83 ✓

Work allowance: Higher work allowance (because housing costs element in award) = £404/month. Taper: 55p in the £ above threshold.

Earnings calculation:
Gross earnings: £1,240.00
Above work allowance: £1,240 - £404 = £836.00
Deduction: £836.00 × 55% = £459.80 ✓

Final payment: £1,891.83 - £459.80 = £1,432.03 ✓

Payment: BACS, 17th of each month.

Note: The childcare element isn't included in the standard award total (£1,891.83) — it would be separate and dependent on actual childcare costs claimed.

Confidence: High — all arithmetic verified.""",
    answer="""## Universal Credit Award Extraction — Al-Rashidi, Fatima

### Claimant Details
| Field | Value |
|-------|-------|
| **Claimant** | Ms. Fatima Al-Rashidi |
| **NI Number** | KN 47 28 63 B |
| **Case Reference** | UC-2024-NW-847219 |
| **Household** | Claimant + 2 dependent children (ages 7 and 4) |
| **Assessment Period** | 14th – 13th (monthly) |
| **Issue Date** | 14 February 2024 |

### Benefit Components (Before Earnings)
| Component | Monthly Amount |
|-----------|---------------|
| Standard Allowance (single 25+) | £368.74 |
| Child Element — 1st child (pre-Apr 2017) | £315.00 |
| Child Element — 2nd child | £269.58 |
| Disabled Child Addition (2nd child, lower rate) | £156.11 |
| Housing Costs Element (private rental) | £782.40 |
| **Standard Award Subtotal** | **£1,891.83** |
| Childcare Element | Up to 85% of costs (max entitlement: £1,014.63/month for 2 children) |

### Work Allowance & Taper
| Parameter | Value |
|-----------|-------|
| **Work Allowance (Higher)** | £404.00/month |
| Taper Rate | **55p per £1 earned above work allowance** |

### Earnings Deduction Calculation
| Step | Amount |
|------|--------|
| Gross Earnings (this period) | £1,240.00 |
| Less: Work Allowance | (£404.00) |
| Earnings above threshold | £836.00 |
| UC Deduction (55% × £836.00) | **£459.80** |

### Final Award
| | Amount |
|--|--------|
| Standard Award | £1,891.83 |
| Less: Earnings Deduction | (£459.80) |
| **Final UC Monthly Payment** | **£1,432.03** |
| **Payment Date** | 17th of each month (BACS) |

### Arithmetic Verification
- Standard award: £368.74 + £315.00 + £269.58 + £156.11 + £782.40 = £1,891.83 ✓
- Above-threshold earnings: £1,240 − £404 = £836 ✓
- Deduction: £836 × 55% = £459.80 ✓
- Final payment: £1,891.83 − £459.80 = £1,432.03 ✓

**Confidence: High**""",
    difficulty="medium"
))

# ─── 45. Arbitration Award ────────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract all key findings and awards from this arbitration summary:

ARBITRATION AWARD — EXECUTIVE SUMMARY
Arbitration Institution: London Court of International Arbitration (LCIA)
Reference: LCIA Arbitration No. 214839
Seat of Arbitration: London, England
Governing Law: English Law
Award Date: 08 February 2024

PARTIES
Claimant: Vanguard Construction SA (Switzerland)
Respondent: Meridian Energy Projects Ltd (UK, Co. No. 09384721)

ARBITRATOR: Professor Helena Blackstone QC (sole arbitrator, LCIA appointment)

DISPUTE SUMMARY
Dispute arising from EPC contract dated 15 March 2019 for construction of a
combined heat and power plant in Aberdeen, Scotland.
Total contract value: £48,500,000.

CLAIMS ASSERTED
Claimant's claims:
1. Unpaid invoices (5 invoices): £3,847,200
2. Variation orders disputed (12 VOs): £1,284,000
3. Prolongation costs (18-week delay): £947,500
4. Interest on above sums at 8% (statutory): claimed from various dates
Total claim: £6,078,700 + interest

Respondent's counterclaim:
1. Delay damages (liquidated): £2,160,000 (24 weeks × £90,000/week)
2. Defect rectification costs: £487,000
Total counterclaim: £2,647,000

AWARD
Re Claimant's Claim 1 (Invoices): AWARDED — £3,847,200 in full (invoices valid)
Re Claimant's Claim 2 (VOs): AWARDED — £847,000 (10 of 12 VOs upheld; 2 rejected)
Re Claimant's Claim 3 (Prolongation): AWARDED — £631,000 (18 weeks reduced to 12 weeks at £52,583/week)
Interest on awarded sums: 8% from dates of each invoice/VO claim — £484,200
Re Respondent's Counterclaim 1 (LADs): DISMISSED — delay found to be Employer's risk
Re Respondent's Counterclaim 2 (Defects): AWARDED — £487,000

NET AWARD TO CLAIMANT:
(3,847,200 + 847,000 + 631,000 + 484,200) − 487,000 = £5,322,400

COSTS
Arbitration costs (LCIA + arbitrator): £284,000 — Respondent to pay 75% (£213,000)
Legal costs: each party to bear its own legal costs.""",
    reasoning="""This is an LCIA arbitration award summary for a construction dispute. I need to extract all claims, awards, and the net result.

Parties: Claimant Vanguard Construction SA (Switzerland), Respondent Meridian Energy Projects Ltd (UK). Arbitrator: Prof. Helena Blackstone QC (sole). Seat: London. Award date 08 Feb 2024.

Dispute: EPC contract 15 March 2019, £48,500,000 CHP plant in Aberdeen.

Claimant's claims:
1. Unpaid invoices: £3,847,200
2. Variation orders (12): £1,284,000
3. Prolongation (18 weeks): £947,500
Total claim: 3,847,200 + 1,284,000 + 947,500 = £6,078,700 ✓ + interest

Respondent's counterclaim:
1. LADs: £2,160,000 (24 weeks × £90,000/week = £2,160,000 ✓)
2. Defects: £487,000
Total: £2,647,000 ✓

Award:
Claim 1 (invoices): £3,847,200 awarded in full ✓
Claim 2 (VOs): £847,000 awarded (10 of 12 VOs upheld). Rejected 2 VOs = £1,284,000 - £847,000 = £437,000 rejected
Claim 3 (prolongation): £631,000 awarded (12 of 18 weeks approved, at £52,583/week: 12 × £52,583 = £631,000 ✓)
Interest: £484,200
Counterclaim 1 (LADs): DISMISSED — delay found Employer's risk (significant)
Counterclaim 2 (defects): £487,000 awarded to Respondent

Net calculation: (3,847,200 + 847,000 + 631,000 + 484,200) - 487,000 = 5,809,400 - 487,000 = £5,322,400 ✓

LCIA costs £284,000; Respondent pays 75% = £213,000; Claimant pays 25% = £71,000.

Key observation: the dismissal of the LAD counterclaim is highly significant — if the 24-week delay damages had been awarded, the Respondent would have recovered £2,160,000 against the claimant which would have significantly reduced the net award.

Confidence: High — all arithmetic verified.""",
    answer="""## Arbitration Award Extraction — LCIA No. 214839

### Arbitration Details
| Field | Value |
|-------|-------|
| **Institution** | LCIA (London Court of International Arbitration) |
| **Reference** | LCIA Arbitration No. 214839 |
| **Seat** | London, England |
| **Governing Law** | English Law |
| **Award Date** | 08 February 2024 |
| **Arbitrator** | Prof. Helena Blackstone QC (sole arbitrator) |

### Parties
| Role | Party |
|------|-------|
| **Claimant** | Vanguard Construction SA (Switzerland) |
| **Respondent** | Meridian Energy Projects Ltd (UK, Co. No. 09384721) |

### Background
- **Contract:** EPC dated 15 March 2019 — combined heat and power plant, Aberdeen
- **Contract Value:** £48,500,000

### Claims & Awards
| # | Claim | Amount Claimed | Amount Awarded | Outcome |
|---|-------|---------------|----------------|---------|
| C1 | Unpaid invoices (5) | £3,847,200 | **£3,847,200** | Awarded in full |
| C2 | Variation orders (12) | £1,284,000 | **£847,000** | 10/12 VOs upheld |
| C3 | Prolongation (18 weeks) | £947,500 | **£631,000** | 12 of 18 weeks approved |
| C4 | Interest at 8% statutory | — | **£484,200** | Awarded |
| **Total Claim** | | **£6,078,700** | **£5,809,400** | |

### Counterclaims & Awards
| # | Counterclaim | Amount Claimed | Amount Awarded | Outcome |
|---|-------------|---------------|----------------|---------|
| R1 | Liquidated damages (24 wks × £90k) | £2,160,000 | £0 | **DISMISSED** — delay Employer's risk |
| R2 | Defect rectification | £487,000 | **£487,000** | Awarded in full |
| **Total Counterclaim** | | **£2,647,000** | **£487,000** | |

### Net Award
| Calculation | Amount |
|-------------|--------|
| Total awarded to Claimant | £5,809,400 |
| Less: Counterclaim awarded to Respondent | (£487,000) |
| **NET AWARD TO CLAIMANT** | **£5,322,400** |

### Costs
| Item | Total | Allocation |
|------|-------|-----------|
| LCIA + Arbitrator fees | £284,000 | Respondent: 75% (£213,000); Claimant: 25% (£71,000) |
| Legal costs | — | Each party bears own costs |

### Key Findings
- **LAD counterclaim dismissed** — delay attributed to Employer's risk (saves Claimant £2,160,000)
- **VO success rate:** 10/12 (83.3%) — 2 VOs (£437,000) rejected
- **Prolongation reduced** from 18 to 12 weeks; rate £52,583/week applied

**Confidence: High** — all arithmetic verified""",
    difficulty="hard"
))

print("Part 8 (examples 41-45) defined. Continuing to part 9...")

# ─── 46. Merger Integration Checklist ────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract all tasks, owners and statuses from this integration tracker:

POST-MERGER INTEGRATION TRACKER
Acquiring Company: Pinnacle Group Holdings plc
Target: Castlegate IT Solutions Ltd
Acquisition Completion Date: 01 January 2024
Integration Lead: Anna Kowalczyk (PMO Director)
Report Date: 15 March 2024 (Day 75 post-close)

WORKSTREAM 1: PEOPLE & HR
Task                                    Owner           Status    Due Date    RAG
Org design announcement                 CHRO            DONE      Jan 31      GREEN
Job matching / redundancy consultation  HR Director     IN PROG   Apr 30      AMBER
TUPE obligations confirmed              Legal           DONE      Jan 15      GREEN
Harmonise benefits (pension, health)    Total Reward    NOT START Jun 30      RED
Employee engagement survey              HR BP           DONE      Mar 01      GREEN

WORKSTREAM 2: IT & SYSTEMS
Task                                    Owner           Status    Due Date    RAG
Network integration (VPN linking)       IT Director     DONE      Feb 28      GREEN
Email domain migration (@castlegate)    IT Manager      IN PROG   Mar 31      AMBER
ERP consolidation (SAP)                 CTO             NOT START Sep 30      RED
Data room decommission                  IT Director     DONE      Feb 15      GREEN
Cybersecurity audit (target systems)    CISO            IN PROG   Apr 15      AMBER

WORKSTREAM 3: COMMERCIAL
Task                                    Owner           Status    Due Date    RAG
Customer communication sent             Sales Dir       DONE      Jan 10      GREEN
Cross-sell pipeline review              RevOps          IN PROG   Apr 30      AMBER
Supplier renegotiation (top 10)         Procurement     NOT START May 31      RED
Contract novation (key customer)        Legal           IN PROG   Mar 31      AMBER

SUMMARY STATISTICS
Total tasks: 14   Completed: 5   In Progress: 5   Not Started: 4
Days behind on any RED task: SAP (not started, due Sep 30 — lead time concern)""",
    reasoning="""This is a post-merger integration tracker at Day 75 post-close (01 Jan 2024 + 75 days = 15 March 2024 ✓). I need to extract all tasks, owners, statuses, due dates and RAG ratings.

Overview: Pinnacle Group Holdings acquiring Castlegate IT Solutions. Integration lead: Anna Kowalczyk (PMO Director). Report date 15 March 2024.

Workstream 1 — People & HR (5 tasks):
1. Org design announcement — CHRO — DONE — Jan 31 — GREEN
2. Job matching/redundancy consultation — HR Director — IN PROGRESS — Apr 30 — AMBER
3. TUPE obligations confirmed — Legal — DONE — Jan 15 — GREEN
4. Harmonise benefits — Total Reward — NOT STARTED — Jun 30 — RED
5. Employee engagement survey — HR BP — DONE — Mar 01 — GREEN

Workstream 2 — IT & Systems (5 tasks):
1. Network integration (VPN) — IT Director — DONE — Feb 28 — GREEN
2. Email domain migration — IT Manager — IN PROGRESS — Mar 31 — AMBER
3. ERP consolidation (SAP) — CTO — NOT STARTED — Sep 30 — RED
4. Data room decommission — IT Director — DONE — Feb 15 — GREEN
5. Cybersecurity audit — CISO — IN PROGRESS — Apr 15 — AMBER

Workstream 3 — Commercial (4 tasks):
1. Customer communication — Sales Dir — DONE — Jan 10 — GREEN
2. Cross-sell pipeline review — RevOps — IN PROGRESS — Apr 30 — AMBER
3. Supplier renegotiation — Procurement — NOT STARTED — May 31 — RED
4. Contract novation (key customers) — Legal — IN PROGRESS — Mar 31 — AMBER

Summary: 14 total: 5 done, 5 in progress, 4 not started.
GREEN: 6, AMBER: 5, RED: 3 (Benefits harmonisation, SAP ERP, Supplier renegotiation)

Done count from above: org design, TUPE, engagement survey, network, data room, customer comms = 6. But summary says 5 done. Let me recount: org design ✓, TUPE ✓, engagement survey ✓, network ✓, data room ✓, customer comms ✓ = 6, not 5. The summary says 5 completed — discrepancy of 1. I should flag this.

Email domain and contract novation both due 31 March 2024 — today is 15 March, so 16 days remaining. Both in progress = at risk.

Confidence: High; note discrepancy in completed count.""",
    answer="""## Integration Tracker Extraction — Pinnacle / Castlegate (Day 75)

### Overview
| Field | Value |
|-------|-------|
| **Acquirer** | Pinnacle Group Holdings plc |
| **Target** | Castlegate IT Solutions Ltd |
| **Completion Date** | 01 January 2024 |
| **Report Date** | 15 March 2024 (Day 75 post-close) |
| **Integration Lead** | Anna Kowalczyk, PMO Director |

### Workstream 1 — People & HR
| Task | Owner | Status | Due Date | RAG |
|------|-------|--------|----------|-----|
| Org design announcement | CHRO | ✅ DONE | 31 Jan | 🟢 GREEN |
| Job matching / redundancy consultation | HR Director | 🔄 IN PROG | 30 Apr | 🟡 AMBER |
| TUPE obligations confirmed | Legal | ✅ DONE | 15 Jan | 🟢 GREEN |
| Harmonise benefits (pension, health) | Total Reward | ⬜ NOT STARTED | 30 Jun | 🔴 RED |
| Employee engagement survey | HR BP | ✅ DONE | 01 Mar | 🟢 GREEN |

### Workstream 2 — IT & Systems
| Task | Owner | Status | Due Date | RAG |
|------|-------|--------|----------|-----|
| Network integration (VPN linking) | IT Director | ✅ DONE | 28 Feb | 🟢 GREEN |
| Email domain migration (@castlegate) | IT Manager | 🔄 IN PROG | **31 Mar** | 🟡 AMBER |
| ERP consolidation (SAP) | CTO | ⬜ NOT STARTED | 30 Sep | 🔴 RED |
| Data room decommission | IT Director | ✅ DONE | 15 Feb | 🟢 GREEN |
| Cybersecurity audit (target systems) | CISO | 🔄 IN PROG | 15 Apr | 🟡 AMBER |

### Workstream 3 — Commercial
| Task | Owner | Status | Due Date | RAG |
|------|-------|--------|----------|-----|
| Customer communication sent | Sales Dir | ✅ DONE | 10 Jan | 🟢 GREEN |
| Cross-sell pipeline review | RevOps | 🔄 IN PROG | 30 Apr | 🟡 AMBER |
| Supplier renegotiation (top 10) | Procurement | ⬜ NOT STARTED | 31 May | 🔴 RED |
| Contract novation (key customers) | Legal | 🔄 IN PROG | **31 Mar** | 🟡 AMBER |

### Summary Statistics
| Status | Count (Stated) | Count (Verified) |
|--------|---------------|------------------|
| Completed | 5 | **6** |
| In Progress | 5 | 5 |
| Not Started | 4 | 4 |
| **Total** | **14** | **14** |

> **Discrepancy:** Tracker states 5 completed but 6 tasks are marked DONE (org design, TUPE, engagement survey, network, data room, customer comms). Recommend reconciliation.

### RAG Summary
| Rating | Count | Tasks |
|--------|-------|-------|
| 🟢 GREEN | 6 | Completed or on track |
| 🟡 AMBER | 5 | In progress — monitoring required |
| 🔴 RED | 3 | Benefits harmonisation; **SAP ERP**; Supplier renegotiation |

### Immediate Attention
- **Email domain migration** and **Contract novation** both due 31 March (16 days from report date) — both in progress, at risk
- **SAP ERP consolidation** — not started; Sep 30 due date; CTO noted as lead time concern

**Confidence: High** — one completed-task count discrepancy flagged""",
    difficulty="medium"
))

# ─── 47. Prospectus / Bond Offering ───────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract the key terms from this bond offering summary:

BOND OFFERING CIRCULAR — SUMMARY
Issuer: Northgate Infrastructure Finance plc
Guarantor: Northgate Group Holdings plc
Type of Instrument: Senior Secured Green Bonds
ISIN: GB00BF8374K2
Date of Issue: 20 February 2024
Principal Amount: £350,000,000
Currency: GBP Sterling

TERMS
Maturity Date: 20 February 2031 (7 years)
Coupon: 5.250% per annum, payable semi-annually
Coupon Payment Dates: 20 February and 20 August each year
Day Count: Actual/Actual ICMA
Issue Price: 99.487% (discount to par)
Yield to Maturity: 5.33% (at issue price)
Minimum Denomination: £100,000

RATINGS
Moody's: Baa2 (Stable outlook) — Investment Grade
S&P: BBB (Stable) — Investment Grade
Fitch: BBB+ (Stable) — Investment Grade

USE OF PROCEEDS
Eligible Green Projects per Green Bond Framework:
  - Renewable energy infrastructure: 60% (£210,000,000)
  - Energy-efficient transport: 25% (£87,500,000)
  - Green buildings: 15% (£52,500,000)

COVENANTS
Interest Coverage Ratio: minimum 2.5× (tested semi-annually)
Gearing Ratio (Net Debt/RAB): maximum 75%
Change of control: put option at 101% of principal

LISTING: London Stock Exchange (Regulated Market)
Governing Law: English Law
Trustee: Citibank N.A., London branch""",
    reasoning="""This is a senior secured green bond offering circular summary. I need to extract all financial terms, ratings and use of proceeds.

Issuer: Northgate Infrastructure Finance plc. Guarantor: Northgate Group Holdings plc. ISIN GB00BF8374K2. Issue date 20 February 2024.

Instrument: £350,000,000 senior secured green bonds.

Term: 7 years, maturity 20 February 2031. Semi-annual coupon of 5.250% pa.

Semi-annual coupon payment: £350,000,000 × 5.25% / 2 = £9,187,500 per payment.

Issue price: 99.487% = £350,000,000 × 0.99487 = £348,204,500 proceeds (discount of £1,795,500 from par).

YTM: 5.33% at issue price — higher than coupon of 5.25% because of discount to par. Consistent.

Ratings: Moody's Baa2, S&P BBB, Fitch BBB+ — all investment grade. S&P and Fitch at BBB/BBB+ category, Moody's equivalent at Baa2. Slight split with Fitch slightly higher.

Use of proceeds (green):
- Renewable energy: 60% × £350M = £210M ✓
- Green transport: 25% × £350M = £87.5M ✓
- Green buildings: 15% × £350M = £52.5M ✓
Total: £210M + £87.5M + £52.5M = £350M ✓

Covenants: ICR ≥ 2.5×, Net Debt/RAB ≤ 75%, change of control put at 101%.

Listed on LSE Regulated Market. English law. Trustee Citibank NA London.

Confidence: High.""",
    answer="""## Bond Offering Extraction — Northgate Infrastructure Finance plc

### Issuance Details
| Field | Value |
|-------|-------|
| **Issuer** | Northgate Infrastructure Finance plc |
| **Guarantor** | Northgate Group Holdings plc |
| **Instrument** | Senior Secured Green Bonds |
| **ISIN** | GB00BF8374K2 |
| **Date of Issue** | 20 February 2024 |
| **Listing** | London Stock Exchange (Regulated Market) |
| **Governing Law** | English Law |
| **Trustee** | Citibank N.A., London branch |

### Financial Terms
| Parameter | Value |
|-----------|-------|
| **Principal Amount** | **£350,000,000** |
| **Issue Price** | 99.487% (£348,204,500 net proceeds) |
| **Discount to Par** | £1,795,500 |
| **Maturity Date** | 20 February 2031 |
| **Tenor** | 7 years |
| **Coupon Rate** | **5.250% per annum** |
| **Coupon Frequency** | Semi-annual |
| **Coupon Payment Dates** | 20 February and 20 August |
| **Semi-Annual Coupon** | £9,187,500 |
| **Yield to Maturity** | 5.33% (at issue price) |
| **Day Count** | Actual/Actual ICMA |
| **Minimum Denomination** | £100,000 |

### Credit Ratings
| Agency | Rating | Outlook | Category |
|--------|--------|---------|---------|
| **Moody's** | Baa2 | Stable | Investment Grade |
| **S&P** | BBB | Stable | Investment Grade |
| **Fitch** | BBB+ | Stable | Investment Grade |

### Use of Proceeds (Green Bond Framework)
| Category | Allocation % | Amount (£) |
|----------|-------------|-----------|
| Renewable Energy Infrastructure | 60% | **£210,000,000** |
| Energy-Efficient Transport | 25% | £87,500,000 |
| Green Buildings | 15% | £52,500,000 |
| **Total** | **100%** | **£350,000,000** |

### Covenants
| Covenant | Threshold | Testing |
|----------|-----------|---------|
| Interest Coverage Ratio | Minimum **2.5×** | Semi-annually |
| Net Debt / RAB | Maximum **75%** | Semi-annually |
| Change of control | Put option at **101%** of principal | On trigger |

### Arithmetic Verification
- Proceeds: £350M × 99.487% = £348,204,500 ✓
- Semi-annual coupon: £350M × 5.25% ÷ 2 = £9,187,500 ✓
- Green proceeds: £210M + £87.5M + £52.5M = £350M ✓

**Confidence: High**""",
    difficulty="hard"
))

# ─── 48. Tenancy Agreement ────────────────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract all tenancy terms from this assured shorthold tenancy:

ASSURED SHORTHOLD TENANCY AGREEMENT
Landlord: Mr. Raghunathan Pillai
Landlord's Agent: Moorfields Lettings Ltd, 14 High Street, Leeds LS1 4PJ
Tenant(s): Ms. Chloe Warrington
Property: Flat 3, 28 Victoria Terrace, Leeds LS6 1AZ (First floor flat)
Furnished: Unfurnished (white goods included)

TERM
Start Date: 01 March 2024
Fixed Term Ends: 28 February 2025 (12 months)
Becomes: Statutory periodic thereafter (rolling monthly)

RENT
Monthly Rent: £875.00
Payment Day: 1st of each month
Payment Method: Standing order to: Moorfields Lettings Ltd
  Sort Code: 20-84-73  |  Account: 64738291
First Rent Due: 01 March 2024

DEPOSIT
Amount: £1,009.62 (5.78 weeks of rent — per Tenant Fees Act 2019 max 5 weeks)
Protected in: MyDeposits Custodial scheme
Reference: MYD-2024-847293

ADDITIONAL TERMS
Council tax: Tenant's responsibility
Utilities (gas, electric, water): Tenant's responsibility
TV licence: Tenant's responsibility
Pet policy: No pets without prior written consent
Smoking: No smoking anywhere in the property
Subletting: Not permitted
Rent increase: Minimum 2 months' written notice required

RIGHT TO RENT CHECK
Completed: Yes — passport verified 20 February 2024
Share code used: WQX-47K-9MR

NOTICE PERIODS
Landlord notice (Section 21): minimum 2 months after fixed term
Tenant notice: minimum 1 month (periodic tenancy)
Early termination: By agreement only""",
    reasoning="""This is an assured shorthold tenancy (AST) agreement for a residential flat in Leeds. I need to extract all key terms.

Parties: Landlord Mr. Raghunathan Pillai, agent Moorfields Lettings Ltd. Tenant: Ms. Chloe Warrington. Property: Flat 3, 28 Victoria Terrace, Leeds LS6 1AZ (first floor, unfurnished, white goods included).

Term: fixed 01 March 2024 – 28 February 2025 (12 months), then rolling monthly statutory periodic.

Rent: £875/month, payable 1st of month by standing order. Sort code 20-84-73, account 64738291.

Deposit: £1,009.62. Deposit Act: max 5 weeks rent. 5 weeks = £875 × 12/52 × 5 = £875/4.333 × 5... Let me calculate: weekly rent = £875 × 12/52 = £10,500/52 = £201.923/week. 5 weeks = £201.923 × 5 = £1,009.62 ✓ — consistent with Tenant Fees Act 2019 maximum.

Deposit protected: MyDeposits custodial, reference MYD-2024-847293.

Tenant responsibilities: council tax, utilities (gas/electric/water), TV licence. No pets without consent. No smoking. No subletting.

Right to rent: passport verified 20 Feb 2024, share code WQX-47K-9MR.

Notice: Section 21 landlord notice = minimum 2 months (and only after fixed term). Tenant notice = minimum 1 month (periodic). Early termination by agreement only.

Confidence: High.""",
    answer="""## Tenancy Agreement Extraction — Flat 3, 28 Victoria Terrace, Leeds

### Parties & Property
| Field | Value |
|-------|-------|
| **Landlord** | Mr. Raghunathan Pillai |
| **Landlord's Agent** | Moorfields Lettings Ltd, 14 High Street, Leeds LS1 4PJ |
| **Tenant** | Ms. Chloe Warrington |
| **Property** | Flat 3, 28 Victoria Terrace, Leeds LS6 1AZ (First floor flat) |
| **Furnished** | Unfurnished (white goods included) |

### Tenancy Term
| Milestone | Date |
|-----------|------|
| **Start Date** | 01 March 2024 |
| **Fixed Term End** | 28 February 2025 |
| **Duration** | 12 months |
| **After Fixed Term** | Statutory periodic (rolling monthly) |

### Rent
| Parameter | Value |
|-----------|-------|
| **Monthly Rent** | **£875.00** |
| **Payment Due** | 1st of each month |
| **Payment Method** | Standing order |
| **Payee** | Moorfields Lettings Ltd |
| **Sort Code** | 20-84-73 |
| **Account Number** | 64738291 |
| **First Payment** | 01 March 2024 |

### Deposit
| Parameter | Value |
|-----------|-------|
| **Deposit Amount** | **£1,009.62** |
| **Basis** | 5 weeks' rent (Tenant Fees Act 2019 maximum) |
| **Weekly Rent** | £201.92 (£875 × 12 ÷ 52) |
| **Protection Scheme** | MyDeposits Custodial |
| **Reference** | MYD-2024-847293 |

### Tenant Responsibilities
| Item | Responsibility |
|------|---------------|
| Council Tax | **Tenant** |
| Gas, Electricity, Water | **Tenant** |
| TV Licence | **Tenant** |
| Pets | Not permitted without prior written consent |
| Smoking | Not permitted anywhere in property |
| Subletting | Not permitted |
| Rent Increase Notice | Minimum 2 months' written notice required |

### Right to Rent
- **Check completed:** Yes
- **Document verified:** Passport (20 February 2024)
- **Share Code:** WQX-47K-9MR

### Notice Periods
| Event | Notice Required |
|-------|----------------|
| Landlord (Section 21) | Minimum **2 months** — after fixed term only |
| Tenant (periodic tenancy) | Minimum **1 month** |
| Early termination | By agreement only |

### Verification
- Deposit: £875 × 12 ÷ 52 × 5 = £1,009.62 ✓ (compliant with Tenant Fees Act 2019)

**Confidence: High**""",
    difficulty="medium"
))

# ─── 49. Business Transfer Agreement ─────────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract key terms from this business sale agreement summary:

BUSINESS SALE AGREEMENT — SUMMARY OF TERMS
Transaction: Sale and purchase of the business and assets of
             Castlegate Fine Foods Ltd (not a share sale)
Seller: Castlegate Fine Foods Ltd (Co. No. 08374922)
Buyer: Northfield Specialty Foods Group Ltd (Co. No. 11938471)
Date of Agreement: 01 March 2024
Completion Date: 01 April 2024

PURCHASE PRICE: £2,840,000 (allocated as below)

ASSET ALLOCATION
Goodwill:                    £1,480,000
Stock at valuation:           £320,000 (to be determined at completion)
Plant and equipment:          £487,000
Customer contracts:           £340,000
Intellectual property:        £213,000
TOTAL (fixed assets):       £2,520,000
[+ stock at completion valuation, est. £320,000]

DEFERRED CONSIDERATION
£180,000 deferred for 12 months, subject to no warranty claims.
Interest: 3% per annum on outstanding deferred consideration.

EARNOUT
£140,000 earnout based on annual revenue from acquired customer contracts
exceeding £1,800,000 in the 12 months post-completion.

EMPLOYEES
15 employees transferring under TUPE 2006.
Seller responsible for pre-completion TUPE obligations.
Buyer to maintain existing terms and conditions for minimum 12 months post-completion.

WARRANTIES
Seller's warranties limited to: £2,840,000 cap; 18-month time limit.
£50,000 de minimis; £150,000 basket.

RESTRICTIVE COVENANTS (Seller/Director)
Non-compete: 3 years, within 50-mile radius of existing business premises.
Non-solicitation of customers: 3 years.
Non-solicitation of employees: 2 years.""",
    reasoning="""This is a business asset purchase agreement (not a share sale) for Castlegate Fine Foods. I need to extract all commercial terms.

Parties: Seller Castlegate Fine Foods Ltd (08374922), Buyer Northfield Specialty Foods Group Ltd (11938471). Asset sale agreed 01 March 2024, completion 01 April 2024.

Purchase price: £2,840,000 total (including stock at completion valuation ~£320,000).

Asset allocation:
Goodwill: £1,480,000
Stock: £320,000 (est., to be determined at completion)
Plant/equipment: £487,000
Customer contracts: £340,000
IP: £213,000
Fixed assets subtotal: £1,480,000 + £487,000 + £340,000 + £213,000 = £2,520,000 ✓ (excl. stock)
With stock: £2,520,000 + £320,000 = £2,840,000 ✓

Structure:
- Fixed consideration: £2,840,000 (including est. stock)
- Deferred: £180,000 (12 months, subject to no warranty claims, 3% pa interest)
- Earnout: £140,000 (if revenue from acquired contracts >£1,800,000 in year 1 post-completion)

Wait — is the deferred consideration and earnout included in the £2,840,000, or additional? The purchase price is stated as £2,840,000 with asset allocation totalling £2,840,000, and then deferred consideration and earnout are listed separately — these appear to be additional consideration above £2,840,000, or they may be components of it. The text says "PURCHASE PRICE: £2,840,000" then lists deferred and earnout separately without clarifying. This is ambiguous and I should flag it.

TUPE: 15 employees transferring. Seller responsible pre-completion, buyer must maintain terms 12 months.

Warranties: cap £2,840,000 (= full purchase price — standard), 18-month time limit, £50,000 de minimis, £150,000 basket (threshold before claims can be brought).

Restrictive covenants: non-compete 3 years/50 miles; non-solicit customers 3 years; non-solicit employees 2 years.

Confidence: High; note ambiguity on whether deferred/earnout are additional to the stated £2,840,000.""",
    answer="""## Business Sale Agreement Extraction — Castlegate Fine Foods

### Transaction Overview
| Field | Value |
|-------|-------|
| **Transaction Type** | Asset purchase (not a share sale) |
| **Seller** | Castlegate Fine Foods Ltd (Co. No. 08374922) |
| **Buyer** | Northfield Specialty Foods Group Ltd (Co. No. 11938471) |
| **Agreement Date** | 01 March 2024 |
| **Completion Date** | 01 April 2024 |

### Purchase Price & Asset Allocation
| Asset | Value |
|-------|-------|
| Goodwill | £1,480,000 |
| Plant and Equipment | £487,000 |
| Customer Contracts | £340,000 |
| Intellectual Property | £213,000 |
| **Fixed Assets Subtotal** | **£2,520,000** |
| Stock (at completion valuation) | ~£320,000 |
| **Total Purchase Price** | **£2,840,000** |

### Consideration Structure
| Component | Amount | Conditions |
|-----------|--------|-----------|
| Base consideration | £2,840,000 | At completion |
| **Deferred consideration** | **£180,000** | 12 months post-completion; subject to no warranty claims; 3% p.a. interest |
| **Earnout** | **£140,000** | Revenue from acquired contracts >£1,800,000 in first 12 months |

> **Note:** It is unclear from the summary whether deferred consideration and earnout are included within or additional to the stated £2,840,000 purchase price. Clarification should be sought from the full agreement.

### Employees
| Parameter | Detail |
|-----------|--------|
| **Transferring Employees** | 15 (under TUPE 2006) |
| Pre-completion obligations | Seller responsible |
| Post-completion T&Cs | Buyer to maintain existing terms for **minimum 12 months** |

### Warranty Provisions
| Parameter | Value |
|-----------|-------|
| **Warranty Cap** | £2,840,000 (100% of purchase price) |
| **Time Limit** | 18 months from completion |
| **De Minimis** | £50,000 per claim |
| **Basket (threshold)** | £150,000 aggregate before claims can be brought |

### Restrictive Covenants (Seller/Director)
| Covenant | Duration | Scope |
|----------|---------|-------|
| Non-compete | **3 years** | 50-mile radius of existing business premises |
| Non-solicitation of customers | **3 years** | — |
| Non-solicitation of employees | **2 years** | — |

**Confidence: High** — one ambiguity on deferred/earnout structure flagged""",
    difficulty="hard"
))

# ─── 50. Regulatory Investigation Letter ──────────────────────────────────────
EXAMPLES.append(sft(
    query="""Extract all key details from this FCA investigation notice:

INFORMATION REQUIREMENT — FORMAL NOTICE
Financial Conduct Authority
12 Endeavour Square, London E20 1JN

Notice Reference: ENF-2024-INV-08472
Issued to: Pinnacle Asset Management Ltd
FCA Reference: 284719
Date of Notice: 05 March 2024
Deadline for Response: 04 April 2024 (30 days)

Legal Basis: Section 165 Financial Services and Markets Act 2000

SUBJECT OF INVESTIGATION
The FCA is conducting an investigation under Section 166 FSMA 2000 (skilled person review)
into concerns regarding:
1. Potential breaches of COBS 2.1.1R (conflicts of interest — Conduct of Business rules)
2. Adequacy of best execution monitoring (COBS 11.2A.28R — MiFID II requirements)
3. Client categorisation procedures for 'eligible counterparty' classification (COBS 3.6)

INFORMATION REQUIRED (Annex A):
- All trade records (equities, fixed income) 01 Jan 2023 – 31 Dec 2023 (12 months)
- Conflicts of interest register (current and all versions since 01 Jan 2021)
- Best execution policies and quarterly reports (01 Jan 2022 – present)
- Client categorisation records for all institutional clients (current and 2021–2023)
- Board minutes and relevant committee minutes where above matters were discussed

FORMAT: Electronic (CSV/Excel for trade data); PDF/searchable for documents
CONTACT: Enforcement caseworker — Ms. L. Bramley (laura.bramley@fca.org.uk)

CONSEQUENCES OF NON-COMPLIANCE
Failure to comply with a Section 165 requirement without reasonable excuse is a
criminal offence under Section 177 FSMA 2000, punishable by up to 2 years' imprisonment
and/or an unlimited fine.""",
    reasoning="""This is a formal information requirement notice from the FCA under Section 165 FSMA 2000. I need to extract all key regulatory details, investigation subjects, and information requirements.

Recipient: Pinnacle Asset Management Ltd, FCA ref 284719. Notice ENF-2024-INV-08472, issued 05 March 2024, deadline 04 April 2024 (30 days — 05 March + 30 days = 04 April ✓).

Legal basis: Section 165 FSMA 2000 (power to require information).

Investigation basis: Section 166 FSMA 2000 (skilled person review). The FCA has three areas of concern:

1. COBS 2.1.1R — conflicts of interest (general Conduct of Business rule)
2. COBS 11.2A.28R — best execution monitoring (MiFID II specific)
3. COBS 3.6 — eligible counterparty client categorisation

Information required:
1. Trade records (equities + fixed income) for full year 2023 (01 Jan – 31 Dec)
2. Conflicts register — current + all versions since 01 Jan 2021 (3 years back)
3. Best execution policies + quarterly reports from 01 Jan 2022 to present
4. Client categorisation records — institutional clients, current + 2021-2023
5. Board and committee minutes on above matters

Format: CSV/Excel (trade data), PDF/searchable (documents). Contact: Ms. L. Bramley, FCA enforcement.

Consequences: non-compliance without reasonable excuse = criminal offence under Section 177 FSMA, up to 2 years imprisonment and/or unlimited fine — extremely serious.

This is a significant regulatory event for Pinnacle. Section 166 (skilled person review) means the FCA may appoint an independent skilled person to review the firm at the firm's cost.

Confidence: High.""",
    answer="""## FCA Investigation Notice Extraction — ENF-2024-INV-08472

### Notice Details
| Field | Value |
|-------|-------|
| **Issuing Authority** | Financial Conduct Authority |
| **Notice Reference** | ENF-2024-INV-08472 |
| **Recipient** | Pinnacle Asset Management Ltd |
| **FCA Reference** | 284719 |
| **Date of Notice** | 05 March 2024 |
| **Response Deadline** | **04 April 2024** (30 days) |
| **Legal Basis** | Section 165, Financial Services and Markets Act 2000 |
| **Investigation Type** | Section 166 FSMA — Skilled Person Review |

### Areas Under Investigation
| # | Rule | Description |
|---|------|-------------|
| 1 | **COBS 2.1.1R** | Potential conflicts of interest breaches |
| 2 | **COBS 11.2A.28R** | Adequacy of best execution monitoring (MiFID II) |
| 3 | **COBS 3.6** | Client categorisation — eligible counterparty classification |

### Information Required (Annex A)
| # | Information | Period |
|---|-------------|--------|
| 1 | Trade records (equities + fixed income) — CSV/Excel | 01 Jan 2023 – 31 Dec 2023 |
| 2 | Conflicts of interest register (all versions) | Current + from 01 Jan 2021 |
| 3 | Best execution policies + quarterly reports | 01 Jan 2022 – present |
| 4 | Client categorisation records (institutional clients) | Current + 2021–2023 |
| 5 | Board and committee minutes (relevant to above) | As applicable |

### Submission Requirements
| Parameter | Requirement |
|-----------|-------------|
| **Trade data format** | Electronic — CSV or Excel |
| **Document format** | PDF or searchable format |
| **FCA Contact** | Ms. L. Bramley, Enforcement (laura.bramley@fca.org.uk) |

### Consequences of Non-Compliance
| Provision | Consequence |
|-----------|-------------|
| **Legal basis** | Section 177 FSMA 2000 (criminal offence) |
| **Imprisonment** | Up to **2 years** |
| **Fine** | **Unlimited** |

### Key Observations
- This is a **significant regulatory event** — Section 166 allows FCA to appoint a skilled person at the firm's expense
- Three COBS rule areas implicated — broad in scope across conflicts, execution quality and categorisation
- 30-day response window is short for potentially large volumes of trade data and records

**Confidence: High**""",
    difficulty="hard"
))

print("All 50 examples defined. Writing JSONL output file...")

# ─── OUTPUT ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in EXAMPLES:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            count += 1
    print(f"Written {count} examples to {OUTPUT_PATH}")

    # Print stats
    categories = {}
    difficulties = {}
    for ex in EXAMPLES:
        d = ex.get("difficulty", "unknown")
        difficulties[d] = difficulties.get(d, 0) + 1

    print(f"\nDifficulty distribution:")
    for k, v in sorted(difficulties.items()):
        print(f"  {k}: {v}")
    print(f"\nAll examples have category='extraction', source='claude_quality'")
