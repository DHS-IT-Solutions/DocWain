"""
Synthetic Document Factory — generates realistic fake documents for training data.

Each document is returned as:
    {
        "content": str,       # the document text
        "type": str,          # document type key
        "metadata": {
            "ground_truth": dict   # actual field values for verification
        }
    }

All generators accept a `random.Random` instance so output is deterministic
when a seed is provided.
"""

import random
from typing import Optional

# ── 16 canonical document types ──────────────────────────────────────────────
DOCUMENT_TYPES = [
    "invoice",
    "purchase_order",
    "contract",
    "policy",
    "financial_statement",
    "medical_record",
    "resume",
    "technical_spec",
    "government_form",
    "spreadsheet",
    "scanned_degraded",
    "meeting_notes",
    "audit_report",
    "insurance_claim",
    "legal_filing",
    "compliance_report",
]

# ── Shared lookup tables ──────────────────────────────────────────────────────
_NAMES = [
    "Alice Johnson", "Bob Martinez", "Carol Singh", "David Kim",
    "Eve Thompson", "Frank Osei", "Grace Liu", "Henry Patel",
    "Isabella Rossi", "James O'Brien", "Karen Yamamoto", "Liam Nguyen",
    "Mia Andersen", "Noah Fernandez", "Olivia Schmidt", "Paul Dubois",
    "Quinn Walker", "Rachel Chen", "Sam Kowalski", "Tina Rashid",
]

_COMPANIES = [
    "Meridian Solutions Ltd", "Apex Dynamics Inc", "Pinnacle Group LLC",
    "Vertex Technologies Corp", "Horizon Ventures Ltd", "Summit Consulting Group",
    "Catalyst Global Partners", "Nexus Innovations Ltd", "Arcadia Systems Inc",
    "Zenith Capital Corp", "Sterling Logistics Ltd", "Luminary Healthcare Inc",
    "Ironclad Manufacturing Corp", "Orion Financial Services LLC", "Celadon Energy Ltd",
]

_CITIES = [
    "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX",
    "Phoenix, AZ", "Philadelphia, PA", "San Antonio, TX", "San Diego, CA",
    "Dallas, TX", "San Jose, CA", "Austin, TX", "Jacksonville, FL",
    "London, UK", "Toronto, ON", "Sydney, NSW",
]

_PRODUCTS = [
    ("Professional Services", 1500.00, "hrs"),
    ("Software License (Annual)", 4800.00, "seat"),
    ("Hardware Module X-200", 249.99, "unit"),
    ("Consulting Engagement", 2200.00, "day"),
    ("Support Retainer", 800.00, "month"),
    ("Training Workshop", 3500.00, "session"),
    ("Data Storage (1 TB)", 45.00, "TB/mo"),
    ("Network Equipment Kit", 1250.00, "unit"),
    ("API Access Tier-3", 600.00, "month"),
    ("Custom Integration Pack", 5000.00, "project"),
]

_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _rand_date(rng: random.Random, year_range=(2022, 2025)) -> str:
    year = rng.randint(*year_range)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"{year}-{month:02d}-{day:02d}"


def _rand_amount(rng: random.Random, lo=100.0, hi=50000.0) -> float:
    return round(rng.uniform(lo, hi), 2)


def _rand_id(rng: random.Random, prefix="", digits=6) -> str:
    return f"{prefix}{rng.randint(10**(digits-1), 10**digits - 1)}"


# ── Invoice generator ─────────────────────────────────────────────────────────
def _gen_invoice(rng: random.Random) -> dict:
    inv_no = _rand_id(rng, "INV-")
    issue_date = _rand_date(rng)
    due_date = _rand_date(rng, (2024, 2026))
    vendor = rng.choice(_COMPANIES)
    client = rng.choice([c for c in _COMPANIES if c != vendor])
    client_addr = rng.choice(_CITIES)

    items = rng.sample(_PRODUCTS, k=rng.randint(2, 5))
    rows = []
    subtotal = 0.0
    line_items = []
    for name, unit_price, unit in items:
        qty = rng.randint(1, 20)
        total = round(qty * unit_price, 2)
        subtotal += total
        rows.append(f"| {name} | {qty} {unit} | ${unit_price:,.2f} | ${total:,.2f} |")
        line_items.append({"description": name, "qty": qty, "unit_price": unit_price, "total": total})

    tax_rate = rng.choice([0.05, 0.08, 0.10, 0.125])
    tax = round(subtotal * tax_rate, 2)
    total_due = round(subtotal + tax, 2)

    lines = [
        f"INVOICE",
        f"",
        f"Invoice No: {inv_no}",
        f"Issue Date: {issue_date}",
        f"Due Date:   {due_date}",
        f"",
        f"FROM: {vendor}",
        f"TO:   {client}",
        f"      {client_addr}",
        f"",
        f"| Description | Quantity | Unit Price | Amount |",
        f"|-------------|----------|------------|--------|",
    ]
    lines.extend(rows)
    lines += [
        f"|             |          | Subtotal   | ${subtotal:,.2f} |",
        f"|             |          | Tax ({tax_rate*100:.0f}%) | ${tax:,.2f} |",
        f"|             |          | **TOTAL**  | **${total_due:,.2f}** |",
        f"",
        f"Payment Terms: Net 30",
        f"Bank: First National Bank  |  Account: {_rand_id(rng, digits=10)}  |  Routing: {_rand_id(rng, digits=9)}",
    ]

    return {
        "content": "\n".join(lines),
        "type": "invoice",
        "metadata": {
            "ground_truth": {
                "invoice_number": inv_no,
                "issue_date": issue_date,
                "due_date": due_date,
                "vendor": vendor,
                "client": client,
                "subtotal": subtotal,
                "tax": tax,
                "total_due": total_due,
                "line_items": line_items,
            }
        },
    }


# ── Contract generator ────────────────────────────────────────────────────────
def _gen_contract(rng: random.Random) -> dict:
    party_a = rng.choice(_COMPANIES)
    party_b = rng.choice([c for c in _COMPANIES if c != party_a])
    effective_date = _rand_date(rng)
    expiry_date = _rand_date(rng, (2025, 2027))
    contract_value = _rand_amount(rng, 10000, 500000)
    contract_id = _rand_id(rng, "CTR-")
    governing_law = rng.choice(["State of New York", "State of California", "State of Delaware",
                                "State of Texas", "England and Wales"])
    notice_days = rng.choice([30, 60, 90])

    lines = [
        f"SERVICE AGREEMENT",
        f"Contract ID: {contract_id}",
        f"",
        f"This Service Agreement (\"Agreement\") is entered into as of {effective_date}",
        f"by and between {party_a} (\"Service Provider\") and {party_b} (\"Client\").",
        f"",
        f"1. SCOPE OF SERVICES",
        f"   Service Provider agrees to deliver professional services as mutually agreed",
        f"   in writing. All deliverables shall conform to the specifications set forth",
        f"   in Exhibit A attached hereto.",
        f"",
        f"2. TERM",
        f"   This Agreement commences on {effective_date} and expires on {expiry_date},",
        f"   unless earlier terminated pursuant to Section 8.",
        f"",
        f"3. COMPENSATION",
        f"   Client shall pay Service Provider a total contract value not to exceed",
        f"   ${contract_value:,.2f}. Invoices are payable within thirty (30) days of receipt.",
        f"",
        f"4. CONFIDENTIALITY",
        f"   Each party agrees to hold the other party's Confidential Information in",
        f"   strict confidence and not to disclose it to any third party.",
        f"",
        f"5. INTELLECTUAL PROPERTY",
        f"   All work product created under this Agreement shall be the exclusive",
        f"   property of Client upon full payment of all outstanding invoices.",
        f"",
        f"6. LIMITATION OF LIABILITY",
        f"   In no event shall either party be liable for indirect, incidental, special",
        f"   or consequential damages arising out of this Agreement.",
        f"",
        f"7. GOVERNING LAW",
        f"   This Agreement shall be governed by the laws of the {governing_law}.",
        f"",
        f"8. TERMINATION",
        f"   Either party may terminate this Agreement upon {notice_days} days' written notice.",
        f"",
        f"IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.",
        f"",
        f"{party_a}                          {party_b}",
        f"Signature: ___________________     Signature: ___________________",
        f"Name:      ___________________     Name:      ___________________",
        f"Title:     ___________________     Title:     ___________________",
        f"Date:      {effective_date}        Date:      {effective_date}",
    ]

    return {
        "content": "\n".join(lines),
        "type": "contract",
        "metadata": {
            "ground_truth": {
                "contract_id": contract_id,
                "party_a": party_a,
                "party_b": party_b,
                "effective_date": effective_date,
                "expiry_date": expiry_date,
                "contract_value": contract_value,
                "governing_law": governing_law,
                "notice_days": notice_days,
            }
        },
    }


# ── Spreadsheet generator ─────────────────────────────────────────────────────
def _gen_spreadsheet(rng: random.Random) -> dict:
    sheet_name = rng.choice(["Q1 Sales", "Budget 2024", "Inventory Tracker",
                              "Employee Hours", "Expense Report", "Revenue Forecast"])
    headers = ["Region", "Product", "Units Sold", "Unit Price", "Revenue", "Cost", "Margin %"]

    regions = ["North", "South", "East", "West", "Central"]
    products_short = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]

    rows = []
    total_revenue = 0.0
    for _ in range(rng.randint(5, 10)):
        region = rng.choice(regions)
        product = rng.choice(products_short)
        units = rng.randint(50, 2000)
        price = round(rng.uniform(10, 500), 2)
        revenue = round(units * price, 2)
        cost = round(revenue * rng.uniform(0.4, 0.75), 2)
        margin = round((revenue - cost) / revenue * 100, 1) if revenue else 0.0
        total_revenue += revenue
        rows.append([region, product, str(units), f"${price:.2f}", f"${revenue:,.2f}",
                      f"${cost:,.2f}", f"{margin}%"])

    def fmt_row(cells):
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join(["---"] * len(headers)) + " |"

    lines = [
        f"Sheet: {sheet_name}",
        f"",
        fmt_row(headers),
        separator,
    ]
    for row in rows:
        lines.append(fmt_row(row))

    lines += [
        f"",
        f"Total Revenue: ${total_revenue:,.2f}",
        f"Generated: {_rand_date(rng)}",
    ]

    return {
        "content": "\n".join(lines),
        "type": "spreadsheet",
        "metadata": {
            "ground_truth": {
                "sheet_name": sheet_name,
                "row_count": len(rows),
                "total_revenue": round(total_revenue, 2),
            }
        },
    }


# ── Medical record generator ──────────────────────────────────────────────────
_DIAGNOSES = [
    ("Type 2 Diabetes Mellitus", "E11.9"),
    ("Essential Hypertension", "I10"),
    ("Major Depressive Disorder", "F32.1"),
    ("Acute Bronchitis", "J20.9"),
    ("Osteoarthritis of Knee", "M17.11"),
    ("Hypothyroidism", "E03.9"),
    ("Gastroesophageal Reflux Disease", "K21.0"),
    ("Atrial Fibrillation", "I48.91"),
    ("Chronic Kidney Disease Stage 3", "N18.3"),
    ("Migraine without Aura", "G43.009"),
]

_MEDICATIONS = [
    ("Metformin", "500 mg", "twice daily"),
    ("Lisinopril", "10 mg", "once daily"),
    ("Atorvastatin", "40 mg", "once daily at bedtime"),
    ("Sertraline", "50 mg", "once daily"),
    ("Omeprazole", "20 mg", "once daily before meals"),
    ("Levothyroxine", "75 mcg", "once daily on empty stomach"),
    ("Amlodipine", "5 mg", "once daily"),
    ("Metoprolol Succinate", "25 mg", "once daily"),
]


def _gen_medical_record(rng: random.Random) -> dict:
    patient_name = rng.choice(_NAMES)
    dob = _rand_date(rng, (1950, 2000))
    mrn = _rand_id(rng, "MRN-")
    visit_date = _rand_date(rng)
    diagnosis, icd = rng.choice(_DIAGNOSES)
    bp_sys = rng.randint(100, 160)
    bp_dia = rng.randint(60, 100)
    hr = rng.randint(55, 110)
    temp = round(rng.uniform(97.0, 100.4), 1)
    weight = round(rng.uniform(120, 250), 1)
    meds = rng.sample(_MEDICATIONS, k=rng.randint(1, 3))

    lines = [
        f"PATIENT MEDICAL RECORD",
        f"",
        f"Patient Name: {patient_name}",
        f"Date of Birth: {dob}",
        f"MRN: {mrn}",
        f"Visit Date: {visit_date}",
        f"",
        f"CHIEF COMPLAINT",
        f"Patient presents with symptoms consistent with {diagnosis}.",
        f"",
        f"VITAL SIGNS",
        f"  Blood Pressure: {bp_sys}/{bp_dia} mmHg",
        f"  Heart Rate: {hr} bpm",
        f"  Temperature: {temp} °F",
        f"  Weight: {weight} lbs",
        f"",
        f"ASSESSMENT AND PLAN",
        f"  Primary Diagnosis: {diagnosis} (ICD-10: {icd})",
        f"  Clinical findings support the above diagnosis based on patient history,",
        f"  physical examination, and available laboratory results.",
        f"",
        f"MEDICATIONS PRESCRIBED",
    ]
    for med_name, dose, freq in meds:
        lines.append(f"  - {med_name} {dose} — {freq}")

    lines.extend([
        f"",
        f"FOLLOW-UP",
        f"  Schedule follow-up in {rng.choice([2, 4, 6, 8, 12])} weeks.",
        f"  Order labs: CBC, CMP, HbA1c.",
        f"",
        f"Attending Physician: Dr. {rng.choice(_NAMES).split()[1]}",
    ])

    return {
        "content": "\n".join(lines),
        "type": "medical_record",
        "metadata": {
            "ground_truth": {
                "patient_name": patient_name,
                "dob": dob,
                "mrn": mrn,
                "visit_date": visit_date,
                "diagnosis": diagnosis,
                "icd_code": icd,
                "blood_pressure": f"{bp_sys}/{bp_dia}",
                "heart_rate": hr,
                "medications": [m[0] for m in meds],
            }
        },
    }


# ── Resume generator ──────────────────────────────────────────────────────────
_ROLES = [
    "Software Engineer", "Data Scientist", "Product Manager",
    "DevOps Engineer", "Machine Learning Engineer", "Backend Developer",
    "Full Stack Developer", "Cloud Architect", "Security Analyst",
    "Business Intelligence Analyst",
]

_SKILLS = [
    "Python", "Java", "TypeScript", "Go", "Rust", "C++",
    "TensorFlow", "PyTorch", "Kubernetes", "Docker", "AWS", "Azure",
    "PostgreSQL", "MongoDB", "Redis", "Kafka", "React", "FastAPI",
]

_UNIVERSITIES = [
    "State University", "Tech Institute", "National University",
    "City College of Engineering", "Polytechnic University",
]


def _gen_resume(rng: random.Random) -> dict:
    name = rng.choice(_NAMES)
    role = rng.choice(_ROLES)
    city = rng.choice(_CITIES)
    email = f"{name.split()[0].lower()}.{name.split()[1].lower()}@email.com"
    phone = f"+1-{rng.randint(200,999)}-{rng.randint(100,999)}-{rng.randint(1000,9999)}"
    years_exp = rng.randint(2, 18)
    skills = rng.sample(_SKILLS, k=rng.randint(5, 10))
    degree = rng.choice(["B.S.", "M.S.", "B.Eng.", "M.Eng."])
    field = rng.choice(["Computer Science", "Electrical Engineering", "Data Science",
                        "Information Systems", "Mathematics"])
    university = rng.choice(_UNIVERSITIES)
    grad_year = rng.randint(2000, 2022)

    exp_entries = []
    for i in range(rng.randint(2, 4)):
        emp = rng.choice(_COMPANIES)
        start_year = 2024 - years_exp + i * rng.randint(1, 3)
        end_year = start_year + rng.randint(1, 4)
        end_str = str(end_year) if end_year < 2025 else "Present"
        exp_entries.append((emp, rng.choice(_ROLES), start_year, end_str))

    lines = [
        f"{name}",
        f"{role}  |  {city}  |  {email}  |  {phone}",
        f"",
        f"PROFESSIONAL SUMMARY",
        f"Results-driven {role} with {years_exp}+ years of experience delivering",
        f"scalable solutions in fast-paced enterprise environments.",
        f"",
        f"TECHNICAL SKILLS",
        f"  {', '.join(skills)}",
        f"",
        f"PROFESSIONAL EXPERIENCE",
    ]
    for emp, job_title, start_yr, end_str in exp_entries:
        lines += [
            f"",
            f"  {job_title}  —  {emp}  ({start_yr} – {end_str})",
            f"  • Delivered high-impact features improving system throughput by {rng.randint(15,60)}%.",
            f"  • Collaborated with cross-functional teams to define and ship product roadmap.",
            f"  • Mentored junior engineers and conducted code reviews.",
        ]

    lines += [
        f"",
        f"EDUCATION",
        f"  {degree} in {field}  —  {university}  ({grad_year})",
    ]

    return {
        "content": "\n".join(lines),
        "type": "resume",
        "metadata": {
            "ground_truth": {
                "name": name,
                "role": role,
                "email": email,
                "years_experience": years_exp,
                "skills": skills,
                "degree": degree,
                "field": field,
                "university": university,
                "grad_year": grad_year,
            }
        },
    }


# ── Financial statement generator ─────────────────────────────────────────────
def _gen_financial_statement(rng: random.Random) -> dict:
    company = rng.choice(_COMPANIES)
    year = rng.randint(2021, 2024)
    revenue = _rand_amount(rng, 500000, 50000000)
    cogs = round(revenue * rng.uniform(0.35, 0.60), 2)
    gross_profit = round(revenue - cogs, 2)
    opex = round(gross_profit * rng.uniform(0.30, 0.55), 2)
    ebitda = round(gross_profit - opex, 2)
    depreciation = round(ebitda * rng.uniform(0.05, 0.15), 2)
    ebit = round(ebitda - depreciation, 2)
    interest = round(ebit * rng.uniform(0.02, 0.08), 2)
    ebt = round(ebit - interest, 2)
    tax_rate = rng.choice([0.21, 0.25, 0.28])
    tax_expense = round(max(ebt, 0) * tax_rate, 2)
    net_income = round(ebt - tax_expense, 2)

    total_assets = round(revenue * rng.uniform(0.8, 2.0), 2)
    total_liabilities = round(total_assets * rng.uniform(0.3, 0.65), 2)
    equity = round(total_assets - total_liabilities, 2)

    lines = [
        f"CONSOLIDATED FINANCIAL STATEMENTS",
        f"{company}",
        f"For the Fiscal Year Ended December 31, {year}",
        f"(All amounts in USD)",
        f"",
        f"INCOME STATEMENT",
        f"",
        f"  Revenue                          ${revenue:>15,.2f}",
        f"  Cost of Goods Sold              (${cogs:>14,.2f})",
        f"  ─────────────────────────────────────────────",
        f"  Gross Profit                     ${gross_profit:>15,.2f}",
        f"",
        f"  Operating Expenses              (${opex:>14,.2f})",
        f"  ─────────────────────────────────────────────",
        f"  EBITDA                           ${ebitda:>15,.2f}",
        f"",
        f"  Depreciation & Amortization     (${depreciation:>14,.2f})",
        f"  ─────────────────────────────────────────────",
        f"  EBIT                             ${ebit:>15,.2f}",
        f"",
        f"  Interest Expense                (${interest:>14,.2f})",
        f"  ─────────────────────────────────────────────",
        f"  Earnings Before Tax              ${ebt:>15,.2f}",
        f"",
        f"  Income Tax ({tax_rate*100:.0f}%)              (${tax_expense:>14,.2f})",
        f"  ═════════════════════════════════════════════",
        f"  Net Income                       ${net_income:>15,.2f}",
        f"",
        f"BALANCE SHEET (Summary)",
        f"",
        f"  Total Assets                     ${total_assets:>15,.2f}",
        f"  Total Liabilities               (${total_liabilities:>14,.2f})",
        f"  ─────────────────────────────────────────────",
        f"  Shareholders' Equity             ${equity:>15,.2f}",
    ]

    return {
        "content": "\n".join(lines),
        "type": "financial_statement",
        "metadata": {
            "ground_truth": {
                "company": company,
                "fiscal_year": year,
                "revenue": revenue,
                "gross_profit": gross_profit,
                "ebitda": ebitda,
                "net_income": net_income,
                "total_assets": total_assets,
                "equity": equity,
            }
        },
    }


# ── Scanned/degraded document generator ──────────────────────────────────────
_OCR_SUBSTITUTIONS = {
    'o': '0', '0': 'o', 'l': '1', '1': 'l', 'I': 'l',
    'e': 'c', 'a': 'o', 'n': 'rn', 'u': 'n', 'm': 'rn',
    'S': '$', 'B': '8', 'g': 'q',
}


def _degrade_text(text: str, rng: random.Random, error_rate=0.04) -> str:
    chars = list(text)
    for i, ch in enumerate(chars):
        if rng.random() < error_rate and ch in _OCR_SUBSTITUTIONS:
            chars[i] = _OCR_SUBSTITUTIONS[ch]
    return "".join(chars)


def _gen_scanned_degraded(rng: random.Random) -> dict:
    base_type = rng.choice(["invoice", "purchase_order", "contract", "government_form"])
    # Generate a clean version first using generic content
    company = rng.choice(_COMPANIES)
    doc_date = _rand_date(rng)
    ref_no = _rand_id(rng, "REF-")
    amount = _rand_amount(rng, 500, 20000)
    recipient = rng.choice(_NAMES)

    clean_text = "\n".join([
        f"DOCUMENT TYPE: {base_type.upper().replace('_', ' ')}",
        f"Reference No: {ref_no}",
        f"Date: {doc_date}",
        f"",
        f"Issued By: {company}",
        f"Recipient: {recipient}",
        f"",
        f"Amount Due: ${amount:,.2f}",
        f"",
        f"This document has been scanned from an original paper copy.",
        f"Some characters may be misrecognized due to scan quality.",
        f"",
        f"Authorized Signature: ___________________________",
        f"Scan Quality: {rng.choice(['Fair', 'Poor', 'Very Poor'])}",
        f"OCR Confidence: {rng.randint(62, 88)}%",
        f"",
        f"[OCR PROCESSED — VERIFY CRITICAL FIELDS MANUALLY]",
    ])

    degraded = _degrade_text(clean_text, rng)

    return {
        "content": degraded,
        "type": "scanned_degraded",
        "metadata": {
            "ground_truth": {
                "original_type": base_type,
                "ref_no": ref_no,
                "doc_date": doc_date,
                "amount": amount,
                "issuer": company,
                "recipient": recipient,
                "quality": "degraded",
                "ocr_processed": True,
            }
        },
    }


# ── Generic generator (fallback for remaining types) ──────────────────────────
_GENERIC_TEMPLATES = {
    "purchase_order": (
        "PURCHASE ORDER",
        ["PO Number", "Vendor", "Ship To", "Order Date", "Delivery Date",
         "Item Description", "Quantity", "Unit Cost", "Total"],
    ),
    "policy": (
        "POLICY DOCUMENT",
        ["Policy Number", "Policy Holder", "Coverage Type", "Effective Date",
         "Expiry Date", "Premium Amount", "Deductible", "Coverage Limit"],
    ),
    "technical_spec": (
        "TECHNICAL SPECIFICATION",
        ["Document ID", "Product Name", "Version", "Author", "Review Date",
         "System Requirements", "API Endpoints", "Data Schema", "Performance SLA"],
    ),
    "government_form": (
        "GOVERNMENT FORM",
        ["Form Number", "Agency", "Applicant Name", "SSN/TIN", "Filing Date",
         "Tax Year", "Gross Income", "Deductions", "Tax Owed"],
    ),
    "meeting_notes": (
        "MEETING MINUTES",
        ["Meeting Date", "Attendees", "Facilitator", "Agenda Items",
         "Decisions Made", "Action Items", "Next Meeting Date"],
    ),
    "audit_report": (
        "AUDIT REPORT",
        ["Audit ID", "Auditor", "Entity Audited", "Period", "Scope",
         "Findings", "Recommendations", "Management Response", "Opinion"],
    ),
    "insurance_claim": (
        "INSURANCE CLAIM FORM",
        ["Claim Number", "Policy Number", "Claimant", "Date of Loss",
         "Description of Loss", "Estimated Loss Amount", "Adjuster", "Status"],
    ),
    "legal_filing": (
        "LEGAL FILING",
        ["Case Number", "Court", "Plaintiff", "Defendant", "Filing Date",
         "Type of Action", "Relief Sought", "Attorney of Record"],
    ),
    "compliance_report": (
        "COMPLIANCE REPORT",
        ["Report ID", "Organization", "Regulatory Framework", "Reporting Period",
         "Compliance Status", "Gaps Identified", "Remediation Plan", "Sign-Off"],
    ),
}


def _gen_generic(doc_type: str, rng: random.Random) -> dict:
    title, fields = _GENERIC_TEMPLATES.get(doc_type, ("DOCUMENT", ["Reference", "Date", "Content"]))
    doc_id = _rand_id(rng, f"{doc_type[:3].upper()}-")
    doc_date = _rand_date(rng)
    company = rng.choice(_COMPANIES)
    person = rng.choice(_NAMES)
    amount = _rand_amount(rng, 1000, 100000)

    lines = [
        f"{title}",
        f"",
        f"Document ID: {doc_id}",
        f"Date: {doc_date}",
        f"Organization: {company}",
        f"Prepared By: {person}",
        f"",
    ]

    ground_truth = {
        "doc_id": doc_id,
        "date": doc_date,
        "organization": company,
        "prepared_by": person,
        "reference_amount": amount,
    }

    for i, field in enumerate(fields):
        if "date" in field.lower():
            val = _rand_date(rng)
        elif "amount" in field.lower() or "cost" in field.lower() or "income" in field.lower():
            val = f"${_rand_amount(rng, 500, 50000):,.2f}"
        elif "number" in field.lower() or "id" in field.lower():
            val = _rand_id(rng, f"{field[:2].upper()}-")
        elif "name" in field.lower() or "person" in field.lower() or \
             field.lower() in ("plaintiff", "defendant", "claimant", "auditor", "attendees",
                               "facilitator", "attorney of record"):
            val = rng.choice(_NAMES)
        elif "company" in field.lower() or "entity" in field.lower() or \
             field.lower() in ("agency", "court", "organization"):
            val = rng.choice(_COMPANIES)
        elif "status" in field.lower():
            val = rng.choice(["Compliant", "Non-Compliant", "Pending Review", "Under Investigation"])
        else:
            val = f"[{field} — details on file]"
        lines.append(f"{field}: {val}")
        ground_truth[field.lower().replace(" ", "_")] = val

    lines += [
        f"",
        f"This document is generated for internal reference purposes.",
        f"For questions, contact {person} at {company}.",
    ]

    return {
        "content": "\n".join(lines),
        "type": doc_type,
        "metadata": {"ground_truth": ground_truth},
    }


# ── Dispatcher ────────────────────────────────────────────────────────────────
_DEDICATED_GENERATORS = {
    "invoice": _gen_invoice,
    "contract": _gen_contract,
    "spreadsheet": _gen_spreadsheet,
    "medical_record": _gen_medical_record,
    "resume": _gen_resume,
    "financial_statement": _gen_financial_statement,
    "scanned_degraded": _gen_scanned_degraded,
}


def generate_document(doc_type: str, seed: Optional[int] = None) -> dict:
    """Generate a single synthetic document of the given type.

    Args:
        doc_type: One of the strings in DOCUMENT_TYPES.
        seed: Optional integer seed for deterministic output.

    Returns:
        dict with keys "content", "type", and "metadata".

    Raises:
        ValueError: If doc_type is not recognised.
    """
    if doc_type not in DOCUMENT_TYPES:
        raise ValueError(f"Unknown document type '{doc_type}'. Valid types: {DOCUMENT_TYPES}")

    rng = random.Random(seed)
    gen = _DEDICATED_GENERATORS.get(doc_type)
    if gen:
        return gen(rng)
    return _gen_generic(doc_type, rng)


def generate_batch(count: int, seed: Optional[int] = None) -> list:
    """Generate a batch of synthetic documents cycling through all DOCUMENT_TYPES.

    Args:
        count: Number of documents to generate.
        seed: Optional integer seed; each document gets a derived seed.

    Returns:
        List of document dicts.
    """
    docs = []
    for i in range(count):
        doc_type = DOCUMENT_TYPES[i % len(DOCUMENT_TYPES)]
        doc_seed = None if seed is None else seed + i
        docs.append(generate_document(doc_type, seed=doc_seed))
    return docs
