"""
Expanded 700-example eval test bank across 7 categories for DocWain sprint evaluation.

Categories and counts:
  extraction_accuracy: 150
  table_excel_reasoning: 100
  ocr_vision: 80
  hallucination_probes: 150
  intent_understanding: 80
  cross_document: 60
  content_generation: 80
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

CATEGORY_COUNTS: dict[str, int] = {
    "extraction_accuracy": 150,
    "table_excel_reasoning": 100,
    "ocr_vision": 80,
    "hallucination_probes": 150,
    "intent_understanding": 80,
    "cross_document": 60,
    "content_generation": 80,
}

_TOTAL = sum(CATEGORY_COUNTS.values())  # 700


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _pick(rng: random.Random, seq: list) -> Any:
    return seq[rng.randint(0, len(seq) - 1)]


def _difficulty(rng: random.Random) -> str:
    return rng.choice(["easy", "medium", "hard"])


# ---------------------------------------------------------------------------
# Synthetic document builders
# ---------------------------------------------------------------------------

def _invoice(rng: random.Random, idx: int) -> dict:
    vendor = _pick(rng, [
        "Acme Supplies Ltd", "Global Tech Corp", "Horizon Services Inc",
        "Peak Solutions LLC", "Vertex Systems", "BlueStar Logistics",
        "Omega Industrial", "Nexus Partners", "Cardinal Distribution",
        "Summit Consulting",
    ])
    inv_no = f"INV-{2024 + (idx % 3)}-{10000 + idx}"
    date = f"2024-{(idx % 12) + 1:02d}-{(idx % 28) + 1:02d}"
    items = []
    subtotal = 0.0
    n_items = rng.randint(2, 5)
    item_names = [
        "Office Supplies", "Software License", "Consulting Hours",
        "Hardware Components", "Maintenance Service", "Cloud Storage (annual)",
        "Training Materials", "Support Contract", "Network Equipment",
        "Data Migration", "API Calls (1000 units)", "Server Rack",
    ]
    for _ in range(n_items):
        name = _pick(rng, item_names)
        qty = rng.randint(1, 20)
        unit_price = round(rng.uniform(10, 500), 2)
        line_total = round(qty * unit_price, 2)
        subtotal += line_total
        items.append((name, qty, unit_price, line_total))
    subtotal = round(subtotal, 2)
    tax_rate = _pick(rng, [0.05, 0.08, 0.10, 0.125, 0.20])
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)
    due_date = f"2024-{((idx + 1) % 12) + 1:02d}-{(idx % 28) + 1:02d}"
    lines = [
        f"INVOICE",
        f"Invoice Number: {inv_no}",
        f"Date: {date}",
        f"Due Date: {due_date}",
        f"Vendor: {vendor}",
        f"Bill To: DocWain Enterprises, 100 Tech Park, San Francisco, CA 94105",
        "",
        "Line Items:",
    ]
    for name, qty, unit_price, line_total in items:
        lines.append(f"  {name}: {qty} x ${unit_price:.2f} = ${line_total:.2f}")
    lines += [
        "",
        f"Subtotal: ${subtotal:.2f}",
        f"Tax ({int(tax_rate * 100)}%): ${tax:.2f}",
        f"Total Due: ${total:.2f}",
        "",
        f"Payment Terms: Net 30",
        f"Bank: First National Bank, Account: 987654321, Routing: 021000021",
    ]
    return {
        "text": "\n".join(lines),
        "invoice_no": inv_no,
        "vendor": vendor,
        "total": total,
        "subtotal": subtotal,
        "tax": tax,
        "date": date,
        "due_date": due_date,
        "items": items,
    }


def _contract(rng: random.Random, idx: int) -> dict:
    parties = [
        ("Apex Corporation", "BlueStar Services"),
        ("Nexus Systems Inc", "Vertex Consulting LLC"),
        ("Summit Enterprises", "Horizon Analytics"),
        ("Cardinal Tech", "Omega Solutions"),
        ("Global Dynamics", "Peak Performance Ltd"),
    ]
    party_a, party_b = parties[idx % len(parties)]
    start = f"2024-{(idx % 12) + 1:02d}-01"
    months = _pick(rng, [6, 12, 24, 36])
    end_year = 2024 + (months // 12)
    end_month = ((idx % 12) + months) % 12 + 1
    end = f"{end_year}-{end_month:02d}-01"
    value = round(rng.uniform(50000, 500000), 2)
    penalty_pct = _pick(rng, [2, 3, 5, 10])
    penalty = round(value * penalty_pct / 100, 2)
    notice_days = _pick(rng, [30, 60, 90])
    governing_law = _pick(rng, ["California", "New York", "Delaware", "Texas", "Illinois"])
    text = f"""SERVICE AGREEMENT

This Service Agreement ("Agreement") is entered into as of {start} between:
  Party A: {party_a} ("Client")
  Party B: {party_b} ("Service Provider")

1. SERVICES
   Service Provider agrees to deliver software development and consulting services
   as detailed in Exhibit A attached hereto.

2. TERM
   This Agreement commences on {start} and expires on {end}, unless earlier
   terminated in accordance with Section 7.

3. COMPENSATION
   Client shall pay Service Provider a total contract value of ${value:,.2f},
   payable in monthly installments. Late payment incurs a {penalty_pct}% penalty
   (${penalty:,.2f}).

4. CONFIDENTIALITY
   Both parties agree to maintain strict confidentiality of all proprietary
   information disclosed during the term of this Agreement and for 3 years thereafter.

5. INTELLECTUAL PROPERTY
   All work product created by Service Provider under this Agreement shall be
   considered work-for-hire and ownership shall vest in Client upon full payment.

6. LIMITATION OF LIABILITY
   Neither party shall be liable for indirect, incidental, or consequential damages.
   Maximum liability is capped at the total contract value paid in the prior 12 months.

7. TERMINATION
   Either party may terminate this Agreement with {notice_days} days written notice.
   Termination for cause requires 15 days written notice and opportunity to cure.

8. GOVERNING LAW
   This Agreement shall be governed by the laws of the State of {governing_law}.

Signatures:
  {party_a}: ____________________  Date: {start}
  {party_b}: ____________________  Date: {start}
"""
    return {
        "text": text,
        "party_a": party_a,
        "party_b": party_b,
        "start": start,
        "end": end,
        "value": value,
        "penalty_pct": penalty_pct,
        "notice_days": notice_days,
        "governing_law": governing_law,
    }


def _resume(rng: random.Random, idx: int) -> dict:
    first_names = ["Jordan", "Taylor", "Alex", "Morgan", "Casey", "Riley", "Cameron", "Avery", "Quinn", "Drew"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Wilson", "Moore"]
    name = f"{first_names[idx % len(first_names)]} {last_names[(idx // len(first_names)) % len(last_names)]}"
    email = f"{name.lower().replace(' ', '.')}@email.com"
    phone = f"+1-555-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"
    title = _pick(rng, ["Senior Software Engineer", "Data Scientist", "Product Manager",
                        "DevOps Engineer", "Machine Learning Engineer", "Backend Developer"])
    years_exp = rng.randint(3, 15)
    university = _pick(rng, ["MIT", "Stanford University", "UC Berkeley", "Carnegie Mellon",
                              "University of Michigan", "Georgia Tech"])
    degree = _pick(rng, ["B.S. Computer Science", "M.S. Data Science", "B.S. Electrical Engineering",
                          "M.S. Computer Science"])
    grad_year = 2024 - years_exp - 4
    skills = rng.sample(["Python", "Java", "Go", "Rust", "TensorFlow", "PyTorch", "Kubernetes",
                          "AWS", "GCP", "React", "PostgreSQL", "Redis", "Spark", "dbt"], 6)
    salary_exp = rng.randint(120, 250) * 1000
    text = f"""CURRICULUM VITAE

{name}
{title}
Email: {email} | Phone: {phone}
LinkedIn: linkedin.com/in/{name.lower().replace(' ', '-')}

PROFESSIONAL SUMMARY
Experienced {title} with {years_exp}+ years in enterprise software development.
Seeking senior roles with growth opportunities. Expected salary: ${salary_exp:,}/year.

TECHNICAL SKILLS
{', '.join(skills)}

WORK EXPERIENCE
Senior {title}, TechCorp Inc (2021–Present)
  - Led development of microservices handling 10M+ daily requests
  - Reduced latency by 40% through query optimization

{title}, StartupXYZ (2019–2021)
  - Built ML pipeline processing 5TB daily data
  - Managed team of 4 engineers

EDUCATION
{degree}, {university}, {grad_year}
GPA: {rng.uniform(3.2, 4.0):.2f}/4.00

CERTIFICATIONS
AWS Solutions Architect (2023), Google Cloud Professional (2022)
"""
    return {
        "text": text,
        "name": name,
        "title": title,
        "years_exp": years_exp,
        "email": email,
        "university": university,
        "degree": degree,
        "skills": skills,
        "salary_exp": salary_exp,
    }


def _financial_statement(rng: random.Random, idx: int) -> dict:
    company = _pick(rng, ["Apex Holdings", "BlueStar Financial", "Nexus Capital", "Summit Corp", "Vertex Inc"])
    year = 2022 + (idx % 3)
    revenue = round(rng.uniform(1e6, 50e6), 2)
    cogs = round(revenue * rng.uniform(0.4, 0.65), 2)
    gross_profit = round(revenue - cogs, 2)
    opex = round(revenue * rng.uniform(0.15, 0.30), 2)
    ebitda = round(gross_profit - opex, 2)
    depreciation = round(revenue * rng.uniform(0.02, 0.05), 2)
    ebit = round(ebitda - depreciation, 2)
    interest = round(revenue * rng.uniform(0.01, 0.03), 2)
    ebt = round(ebit - interest, 2)
    tax_rate = rng.uniform(0.21, 0.28)
    taxes = round(max(ebt * tax_rate, 0), 2)
    net_income = round(ebt - taxes, 2)
    text = f"""INCOME STATEMENT — {company}
Fiscal Year Ending December 31, {year}
(All figures in USD)

Revenue:                     ${revenue:>15,.2f}
Cost of Goods Sold:          ${cogs:>15,.2f}
                             ─────────────────
Gross Profit:                ${gross_profit:>15,.2f}

Operating Expenses:          ${opex:>15,.2f}
EBITDA:                      ${ebitda:>15,.2f}
Depreciation & Amortization: ${depreciation:>15,.2f}
EBIT:                        ${ebit:>15,.2f}
Interest Expense:            ${interest:>15,.2f}
EBT:                         ${ebt:>15,.2f}
Income Tax ({int(tax_rate*100)}%):             ${taxes:>15,.2f}
                             ─────────────────
Net Income:                  ${net_income:>15,.2f}

Gross Margin: {gross_profit/revenue*100:.1f}%
Net Margin: {net_income/revenue*100:.1f}%
"""
    return {
        "text": text,
        "company": company,
        "year": year,
        "revenue": revenue,
        "gross_profit": gross_profit,
        "net_income": net_income,
        "ebitda": ebitda,
    }


def _medical_record(rng: random.Random, idx: int) -> dict:
    patient_ids = [f"PT-{10000 + i}" for i in range(50)]
    patient_id = patient_ids[idx % len(patient_ids)]
    conditions = ["Type 2 Diabetes", "Hypertension", "Asthma", "Hypothyroidism", "Osteoarthritis"]
    medications = ["Metformin 500mg", "Lisinopril 10mg", "Albuterol inhaler", "Levothyroxine 50mcg", "Ibuprofen 400mg"]
    condition = _pick(rng, conditions)
    med = _pick(rng, medications)
    bp_sys = rng.randint(110, 160)
    bp_dia = rng.randint(70, 100)
    glucose = rng.randint(80, 240)
    visit_date = f"2024-{(idx % 12) + 1:02d}-{(idx % 28) + 1:02d}"
    text = f"""CLINICAL VISIT RECORD

Patient ID: {patient_id}
Visit Date: {visit_date}
Provider: Dr. Sarah Chen, MD — Internal Medicine

CHIEF COMPLAINT: Routine follow-up for {condition}

VITAL SIGNS:
  Blood Pressure: {bp_sys}/{bp_dia} mmHg
  Heart Rate: {rng.randint(60, 90)} bpm
  Temperature: {rng.uniform(97.5, 99.0):.1f}°F
  Fasting Glucose: {glucose} mg/dL

CURRENT MEDICATIONS:
  - {med} — take as directed
  - Aspirin 81mg — daily

ASSESSMENT & PLAN:
  {condition} — {'well-controlled' if rng.random() > 0.4 else 'poorly controlled'}.
  Continue current regimen. Follow up in {_pick(rng, [3, 6, 12])} months.
  Patient counseled on diet and exercise modifications.

ALLERGIES: {'NKDA' if rng.random() > 0.3 else 'Penicillin (rash)'}
"""
    return {
        "text": text,
        "patient_id": patient_id,
        "condition": condition,
        "medication": med,
        "bp": f"{bp_sys}/{bp_dia}",
        "glucose": glucose,
        "visit_date": visit_date,
    }


def _policy_doc(rng: random.Random, idx: int) -> dict:
    policy_types = [
        ("Remote Work Policy", "employees may work remotely up to 3 days per week"),
        ("Data Retention Policy", "data must be retained for 7 years then securely destroyed"),
        ("Expense Reimbursement Policy", "expenses over $500 require prior manager approval"),
        ("Code of Conduct Policy", "violations may result in immediate termination"),
        ("Security Incident Response Policy", "all incidents must be reported within 24 hours"),
    ]
    ptype, key_rule = policy_types[idx % len(policy_types)]
    effective_date = f"2024-{(idx % 12) + 1:02d}-01"
    review_date = f"2025-{(idx % 12) + 1:02d}-01"
    owner = _pick(rng, ["HR Department", "IT Security Team", "Legal & Compliance", "Operations"])
    text = f"""CORPORATE POLICY: {ptype.upper()}

Policy Number: POL-{2024}-{100 + idx:03d}
Effective Date: {effective_date}
Next Review: {review_date}
Policy Owner: {owner}
Version: 2.{idx % 5}

1. PURPOSE
   This policy establishes standards and guidelines for {ptype.lower()}
   across all DocWain departments and subsidiaries.

2. SCOPE
   Applies to all full-time employees, contractors, and third-party vendors
   with access to company resources.

3. POLICY STATEMENT
   {ptype}: {key_rule}. All personnel are expected to comply fully.
   Non-compliance will be addressed through progressive disciplinary action.

4. RESPONSIBILITIES
   - Employees: Read, understand, and comply with this policy
   - Managers: Enforce policy and report violations promptly
   - {owner}: Maintain, update, and communicate policy changes

5. EXCEPTIONS
   Exceptions must be submitted in writing to {owner} and approved by the
   Chief Compliance Officer before implementation.

6. REVIEW CYCLE
   This policy is reviewed annually. Next review scheduled for {review_date}.
"""
    return {
        "text": text,
        "policy_type": ptype,
        "key_rule": key_rule,
        "effective_date": effective_date,
        "review_date": review_date,
        "owner": owner,
    }


# ---------------------------------------------------------------------------
# Generator: extraction_accuracy (150)
# ---------------------------------------------------------------------------

def _gen_extraction_accuracy(rng: random.Random) -> list[dict]:
    examples = []
    # 25 each for 6 subtypes = 150
    subtypes = [
        ("invoice", _invoice),
        ("contract", _contract),
        ("resume", _resume),
        ("financial", _financial_statement),
        ("medical", _medical_record),
        ("policy", _policy_doc),
    ]
    for subtype, builder in subtypes:
        for i in range(25):
            doc = builder(rng, i)
            if subtype == "invoice":
                questions = [
                    (f"What is the total amount due on invoice {doc['invoice_no']}?",
                     {"expected_answer": f"${doc['total']:.2f}", "expected_values": {"total": doc['total']}}),
                    (f"Who is the vendor on this invoice?",
                     {"expected_answer": doc['vendor']}),
                    (f"What is the due date for invoice {doc['invoice_no']}?",
                     {"expected_answer": doc['due_date']}),
                ]
            elif subtype == "contract":
                questions = [
                    (f"What is the total contract value in this service agreement?",
                     {"expected_answer": f"${doc['value']:,.2f}", "expected_values": {"value": doc['value']}}),
                    (f"How many days notice is required to terminate this contract?",
                     {"expected_answer": f"{doc['notice_days']} days"}),
                    (f"Which state's laws govern this agreement?",
                     {"expected_answer": doc['governing_law']}),
                ]
            elif subtype == "resume":
                questions = [
                    (f"What is the candidate's name and contact email?",
                     {"expected_answer": f"{doc['name']}, {doc['email']}"}),
                    (f"Where did the candidate complete their degree?",
                     {"expected_answer": doc['university']}),
                    (f"What salary is the candidate expecting?",
                     {"expected_answer": f"${doc['salary_exp']:,}/year", "expected_values": {"salary": doc['salary_exp']}}),
                ]
            elif subtype == "financial":
                questions = [
                    (f"What was the net income for {doc['company']} in fiscal year {doc['year']}?",
                     {"expected_answer": f"${doc['net_income']:,.2f}", "expected_values": {"net_income": doc['net_income']}}),
                    (f"What was the total revenue reported?",
                     {"expected_answer": f"${doc['revenue']:,.2f}", "expected_values": {"revenue": doc['revenue']}}),
                    (f"What was the EBITDA?",
                     {"expected_answer": f"${doc['ebitda']:,.2f}", "expected_values": {"ebitda": doc['ebitda']}}),
                ]
            elif subtype == "medical":
                questions = [
                    (f"What is the patient's blood pressure reading?",
                     {"expected_answer": f"{doc['bp']} mmHg"}),
                    (f"What condition is being managed for patient {doc['patient_id']}?",
                     {"expected_answer": doc['condition']}),
                    (f"What medication is listed for this patient?",
                     {"expected_answer": doc['medication']}),
                ]
            else:  # policy
                questions = [
                    (f"What is the key rule stated in this policy?",
                     {"expected_answer": doc['key_rule']}),
                    (f"Who owns this policy?",
                     {"expected_answer": doc['owner']}),
                    (f"When is the next policy review date?",
                     {"expected_answer": doc['review_date']}),
                ]
            q_text, reference = questions[i % len(questions)]
            prompt = f"Document:\n{doc['text']}\n\nQuestion: {q_text}"
            ex_id = _sha256_id(f"extraction_{subtype}_{i}")
            examples.append({
                "id": ex_id,
                "category": "extraction_accuracy",
                "prompt": prompt,
                "reference": reference,
                "difficulty": _difficulty(rng),
            })
    return examples


# ---------------------------------------------------------------------------
# Generator: table_excel_reasoning (100)
# ---------------------------------------------------------------------------

def _gen_table_excel_reasoning(rng: random.Random) -> list[dict]:
    examples = []
    for i in range(100):
        n_rows = rng.randint(5, 12)
        regions = ["North", "South", "East", "West", "Central"]
        products = ["Widget A", "Widget B", "Gadget X", "Gadget Y", "Service Pro"]
        rows = []
        for r in range(n_rows):
            region = regions[r % len(regions)]
            product = products[(r + i) % len(products)]
            q1 = round(rng.uniform(10000, 200000), 2)
            q2 = round(rng.uniform(10000, 200000), 2)
            q3 = round(rng.uniform(10000, 200000), 2)
            q4 = round(rng.uniform(10000, 200000), 2)
            total = round(q1 + q2 + q3 + q4, 2)
            rows.append((region, product, q1, q2, q3, q4, total))

        # Build table text
        header = "Region       | Product     |     Q1       |     Q2       |     Q3       |     Q4       |   Annual Total"
        sep = "-" * len(header)
        table_lines = [header, sep]
        for region, product, q1, q2, q3, q4, total in rows:
            table_lines.append(
                f"{region:<12} | {product:<11} | {q1:>12,.2f} | {q2:>12,.2f} | {q3:>12,.2f} | {q4:>12,.2f} | {total:>14,.2f}"
            )
        grand_total = round(sum(r[6] for r in rows), 2)
        max_row = max(rows, key=lambda r: r[6])
        min_row = min(rows, key=lambda r: r[6])
        avg_total = round(grand_total / n_rows, 2)

        q_variants = [
            (f"What is the grand total annual revenue across all regions and products?",
             {"expected_answer": f"${grand_total:,.2f}", "expected_values": {"grand_total": grand_total}}),
            (f"Which product-region combination had the highest annual total, and what was it?",
             {"expected_answer": f"{max_row[1]} in {max_row[0]}: ${max_row[6]:,.2f}",
              "expected_values": {"max_total": max_row[6], "product": max_row[1], "region": max_row[0]}}),
            (f"What is the average annual total per row in this spreadsheet?",
             {"expected_answer": f"${avg_total:,.2f}", "expected_values": {"avg_total": avg_total}}),
            (f"Which row had the lowest annual total revenue?",
             {"expected_answer": f"{min_row[1]} in {min_row[0]}: ${min_row[6]:,.2f}"}),
        ]
        q_text, reference = q_variants[i % len(q_variants)]
        table_str = "\n".join(table_lines)
        prompt = f"Spreadsheet Data:\n{table_str}\n\nQuestion: {q_text}"
        ex_id = _sha256_id(f"table_excel_{i}")
        examples.append({
            "id": ex_id,
            "category": "table_excel_reasoning",
            "prompt": prompt,
            "reference": reference,
            "difficulty": _difficulty(rng),
        })
    return examples


# ---------------------------------------------------------------------------
# Generator: ocr_vision (80) — degraded scanned receipts
# ---------------------------------------------------------------------------

def _corrupt(rng: random.Random, text: str) -> str:
    """Simulate OCR degradation with random character substitutions."""
    corruptions = {
        "0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "S": "5",
        "8": "B", "B": "8", "6": "b", "b": "6", "2": "Z", "Z": "2",
    }
    result = []
    for ch in text:
        if ch in corruptions and rng.random() < 0.08:
            result.append(corruptions[ch])
        elif rng.random() < 0.02:
            result.append(" ")  # random space insertion
        else:
            result.append(ch)
    return "".join(result)


def _gen_ocr_vision(rng: random.Random) -> list[dict]:
    examples = []
    stores = ["QuickMart", "SaveMore", "FreshGrocer", "TechZone", "OfficeDepot Plus",
              "AutoParts Pro", "PharmaCare", "BookNook", "SportsGear", "HomeBase"]
    for i in range(80):
        store = stores[i % len(stores)]
        date = f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}"
        time_str = f"{rng.randint(8, 21):02d}:{rng.randint(0, 59):02d}"
        items = []
        n_items = rng.randint(3, 7)
        receipt_items = [
            ("Bottled Water 24pk", 3.99), ("Organic Milk 1gal", 4.49), ("Bread Loaf", 2.99),
            ("AA Batteries 8pk", 7.99), ("USB Cable", 12.99), ("Notebook (3pk)", 5.49),
            ("Pen Set (10ct)", 4.99), ("Coffee Beans 1lb", 11.99), ("Protein Bar (6pk)", 8.99),
            ("Hand Sanitizer", 3.49), ("Paper Towels 6pk", 9.99), ("Dish Soap", 3.29),
        ]
        subtotal = 0.0
        for _ in range(n_items):
            item_name, price = _pick(rng, receipt_items)
            subtotal += price
            items.append((item_name, price))
        subtotal = round(subtotal, 2)
        tax = round(subtotal * 0.0875, 2)
        total = round(subtotal + tax, 2)
        cashback = 0.0
        if rng.random() > 0.6:
            cashback = _pick(rng, [20.0, 40.0, 60.0])
        receipt_text = f"""{store}
{store.upper()} RECEIPT
Date: {date}  Time: {time_str}
Store #: {100 + i}  Register: {rng.randint(1, 8)}
Cashier: {_pick(rng, ['Mike', 'Sarah', 'James', 'Lisa'])}
------------------------------------
ITEMS PURCHASED:
"""
        for item_name, price in items:
            receipt_text += f"  {item_name:<28} ${price:.2f}\n"
        receipt_text += f"""------------------------------------
Subtotal:                      ${subtotal:.2f}
Tax (8.75%):                   ${tax:.2f}
TOTAL:                         ${total:.2f}
"""
        if cashback > 0:
            receipt_text += f"Cash Back:                     ${cashback:.2f}\n"
        receipt_text += f"""
Payment: VISA ending 4242
Auth: {rng.randint(100000, 999999)}
------------------------------------
Thank you for shopping at {store}!
"""
        corrupted = _corrupt(rng, receipt_text)
        q_variants = [
            (f"What is the total amount charged on this receipt?",
             {"expected_answer": f"${total:.2f}", "expected_values": {"total": total}}),
            (f"What store issued this receipt and what was the date?",
             {"expected_answer": f"{store}, {date}"}),
            (f"What was the subtotal before tax?",
             {"expected_answer": f"${subtotal:.2f}", "expected_values": {"subtotal": subtotal}}),
            (f"How many items were purchased according to the receipt?",
             {"expected_answer": str(n_items), "expected_values": {"item_count": n_items}}),
        ]
        q_text, reference = q_variants[i % len(q_variants)]
        prompt = f"[Scanned receipt with OCR artifacts]\n{corrupted}\n\nQuestion: {q_text}"
        ex_id = _sha256_id(f"ocr_vision_{i}")
        examples.append({
            "id": ex_id,
            "category": "ocr_vision",
            "prompt": prompt,
            "reference": reference,
            "difficulty": _difficulty(rng),
        })
    return examples


# ---------------------------------------------------------------------------
# Generator: hallucination_probes (150)
# First 50: unanswerable (answerable=False), rest: answerable
# ---------------------------------------------------------------------------

def _gen_hallucination_probes(rng: random.Random) -> list[dict]:
    examples = []

    # 50 unanswerable — ask about things not in document
    for i in range(50):
        doc = _invoice(rng, i + 200)
        unanswerable_questions = [
            "What is the CEO's name who approved this invoice?",
            "What is the purchase order number associated with this invoice?",
            "What discount was applied to this invoice?",
            "What is the shipping tracking number for this order?",
            "When was this invoice originally quoted?",
            "What is the contract number this invoice is billed under?",
            "Who signed off on the delivery of these goods?",
            "What is the vendor's tax ID number?",
        ]
        q_text = unanswerable_questions[i % len(unanswerable_questions)]
        prompt = f"Document:\n{doc['text']}\n\nQuestion: {q_text}"
        ex_id = _sha256_id(f"halluc_unanswerable_{i}")
        examples.append({
            "id": ex_id,
            "category": "hallucination_probes",
            "prompt": prompt,
            "reference": {
                "expected_answer": "This information is not present in the document.",
                "answerable": False,
            },
            "difficulty": "hard",
        })

    # 100 answerable hallucination probes — questions with specific traps
    for i in range(100):
        doc_type = i % 4
        if doc_type == 0:
            doc = _invoice(rng, i + 300)
            tricky_qs = [
                (f"Does this invoice show a discount? What percentage?",
                 {"expected_answer": "No discount is mentioned in this invoice.", "answerable": True}),
                (f"What is the total amount due — confirm it includes tax?",
                 {"expected_answer": f"${doc['total']:.2f} (includes {round(doc['tax']/doc['subtotal']*100)}% tax of ${doc['tax']:.2f})",
                  "answerable": True}),
            ]
        elif doc_type == 1:
            doc = _contract(rng, i + 300)
            tricky_qs = [
                (f"Can the contract be terminated immediately without notice?",
                 {"expected_answer": f"No. {doc['notice_days']} days written notice is required.", "answerable": True}),
                (f"Is there a non-compete clause in this agreement?",
                 {"expected_answer": "The document does not contain a non-compete clause.", "answerable": True}),
            ]
        elif doc_type == 2:
            doc = _resume(rng, i + 300)
            tricky_qs = [
                (f"Does the candidate have a PhD?",
                 {"expected_answer": f"No. Their highest degree is {doc['degree']} from {doc['university']}.",
                  "answerable": True}),
                (f"Does the candidate list any certifications?",
                 {"expected_answer": "Yes, AWS Solutions Architect (2023) and Google Cloud Professional (2022).",
                  "answerable": True}),
            ]
        else:
            doc = _financial_statement(rng, i + 300)
            tricky_qs = [
                (f"Did {doc['company']} report a loss in {doc['year']}?",
                 {"expected_answer": f"No, net income was ${doc['net_income']:,.2f} (positive)." if doc['net_income'] > 0
                  else f"Yes, net loss was ${abs(doc['net_income']):,.2f}.", "answerable": True}),
                (f"What was the revenue growth compared to prior year?",
                 {"expected_answer": "Prior year data is not included in this document.", "answerable": False}),
            ]
        q_text, reference = tricky_qs[i % len(tricky_qs)]
        prompt = f"Document:\n{doc['text']}\n\nQuestion: {q_text}"
        ex_id = _sha256_id(f"halluc_answerable_{i}")
        examples.append({
            "id": ex_id,
            "category": "hallucination_probes",
            "prompt": prompt,
            "reference": reference,
            "difficulty": _pick(rng, ["medium", "hard"]),
        })

    return examples


# ---------------------------------------------------------------------------
# Generator: intent_understanding (80)
# ---------------------------------------------------------------------------

def _gen_intent_understanding(rng: random.Random) -> list[dict]:
    examples = []
    intent_types = ["lookup", "compare", "summarize", "extract"]
    for i in range(80):
        intent = intent_types[i % len(intent_types)]
        doc_idx = i + 500
        if intent == "lookup":
            doc = _invoice(rng, doc_idx)
            queries = [
                f"Find the invoice number for the {doc['vendor']} invoice",
                f"Look up the due date on this invoice",
                f"What vendor sent this invoice?",
            ]
            q_text = queries[i % len(queries)]
            reference = {"expected_answer": doc['invoice_no'] if "invoice number" in q_text
                         else doc['due_date'] if "due date" in q_text else doc['vendor'],
                         "intent": "lookup"}
            prompt = f"Document:\n{doc['text']}\n\nUser query: {q_text}"
        elif intent == "compare":
            doc1 = _contract(rng, doc_idx)
            doc2 = _contract(rng, doc_idx + 100)
            q_text = f"Compare the contract values and notice periods between these two agreements"
            reference = {
                "expected_answer": (
                    f"Contract 1 ({doc1['party_a']} & {doc1['party_b']}): value ${doc1['value']:,.2f}, "
                    f"{doc1['notice_days']} days notice. "
                    f"Contract 2 ({doc2['party_a']} & {doc2['party_b']}): value ${doc2['value']:,.2f}, "
                    f"{doc2['notice_days']} days notice."
                ),
                "intent": "compare",
            }
            prompt = f"Document 1:\n{doc1['text']}\n\nDocument 2:\n{doc2['text']}\n\nUser query: {q_text}"
        elif intent == "summarize":
            doc = _policy_doc(rng, doc_idx)
            q_text = f"Summarize this policy document in 2-3 sentences"
            reference = {
                "expected_answer": (
                    f"This is the {doc['policy_type']} (effective {doc['effective_date']}), owned by {doc['owner']}. "
                    f"Key rule: {doc['key_rule']}. Next review: {doc['review_date']}."
                ),
                "intent": "summarize",
            }
            prompt = f"Document:\n{doc['text']}\n\nUser query: {q_text}"
        else:  # extract
            doc = _resume(rng, doc_idx)
            q_text = f"Extract the candidate's name, title, and expected salary from this resume"
            reference = {
                "expected_answer": f"Name: {doc['name']}, Title: {doc['title']}, Expected Salary: ${doc['salary_exp']:,}/year",
                "expected_values": {
                    "name": doc['name'],
                    "title": doc['title'],
                    "salary": doc['salary_exp'],
                },
                "intent": "extract",
            }
            prompt = f"Document:\n{doc['text']}\n\nUser query: {q_text}"
        ex_id = _sha256_id(f"intent_{intent}_{i}")
        examples.append({
            "id": ex_id,
            "category": "intent_understanding",
            "prompt": prompt,
            "reference": reference,
            "difficulty": _difficulty(rng),
        })
    return examples


# ---------------------------------------------------------------------------
# Generator: cross_document (60)
# ---------------------------------------------------------------------------

def _gen_cross_document(rng: random.Random) -> list[dict]:
    examples = []
    cross_types = [
        ("invoice_contract", _invoice, _contract),
        ("financial_financial", _financial_statement, _financial_statement),
        ("contract_policy", _contract, _policy_doc),
        ("resume_resume", _resume, _resume),
    ]
    for i in range(60):
        ct_name, builder1, builder2 = cross_types[i % len(cross_types)]
        doc1 = builder1(rng, i + 600)
        doc2 = builder2(rng, i + 700)

        if ct_name == "invoice_contract":
            q_text = "Does the invoice total exceed the monthly installment implied by the contract value? Explain."
            monthly_installment = round(doc2['value'] / 12, 2)
            exceeds = doc1['total'] > monthly_installment
            reference = {
                "expected_answer": (
                    f"{'Yes' if exceeds else 'No'}. Invoice total is ${doc1['total']:.2f}; "
                    f"the contract (${doc2['value']:,.2f} / 12 months) implies ~${monthly_installment:,.2f}/month."
                ),
                "expected_values": {
                    "invoice_total": doc1['total'],
                    "monthly_installment": monthly_installment,
                    "exceeds": exceeds,
                },
            }
        elif ct_name == "financial_financial":
            q_text = "Compare the net income margin between the two financial statements."
            margin1 = round(doc1['net_income'] / doc1['revenue'] * 100, 1)
            margin2 = round(doc2['net_income'] / doc2['revenue'] * 100, 1)
            reference = {
                "expected_answer": (
                    f"{doc1['company']} ({doc1['year']}): {margin1}% net margin. "
                    f"{doc2['company']} ({doc2['year']}): {margin2}% net margin. "
                    f"{'First' if margin1 > margin2 else 'Second'} company has higher margin."
                ),
                "expected_values": {"margin1": margin1, "margin2": margin2},
            }
        elif ct_name == "contract_policy":
            q_text = "Which document has a longer review/term period — the contract or the policy?"
            reference = {
                "expected_answer": (
                    f"The contract has a term until {doc1['end']}; "
                    f"the policy reviews annually (next: {doc2['review_date']}). "
                    f"The contract term is longer."
                ),
            }
        else:  # resume_resume
            q_text = "Which candidate has more years of experience and what salary are they expecting?"
            more_exp = doc1 if doc1['years_exp'] >= doc2['years_exp'] else doc2
            reference = {
                "expected_answer": (
                    f"{more_exp['name']} has more experience ({more_exp['years_exp']} years) "
                    f"and expects ${more_exp['salary_exp']:,}/year."
                ),
                "expected_values": {
                    "candidate": more_exp['name'],
                    "years_exp": more_exp['years_exp'],
                },
            }
        prompt = f"Document 1:\n{doc1['text']}\n\nDocument 2:\n{doc2['text']}\n\nQuestion: {q_text}"
        ex_id = _sha256_id(f"cross_doc_{ct_name}_{i}")
        examples.append({
            "id": ex_id,
            "category": "cross_document",
            "prompt": prompt,
            "reference": reference,
            "difficulty": _pick(rng, ["medium", "hard"]),
        })
    return examples


# ---------------------------------------------------------------------------
# Generator: content_generation (80)
# ---------------------------------------------------------------------------

def _gen_content_generation(rng: random.Random) -> list[dict]:
    examples = []
    gen_types = ["summary_email", "status_report", "action_items", "executive_summary"]
    for i in range(80):
        gen_type = gen_types[i % len(gen_types)]
        doc_idx = i + 800

        if gen_type == "summary_email":
            doc = _invoice(rng, doc_idx)
            q_text = f"Write a brief email to accounts payable summarizing this invoice for approval"
            reference = {
                "expected_answer": (
                    f"Subject: Invoice Approval Request — {doc['invoice_no']}\n"
                    f"Hi AP Team, please review invoice {doc['invoice_no']} from {doc['vendor']} "
                    f"dated {doc['date']}, due {doc['due_date']}, for ${doc['total']:.2f}. "
                    f"Kindly process payment by the due date. Thank you."
                ),
                "generation_type": "summary_email",
                "key_facts": {
                    "invoice_no": doc['invoice_no'],
                    "vendor": doc['vendor'],
                    "total": doc['total'],
                    "due_date": doc['due_date'],
                },
            }
        elif gen_type == "status_report":
            doc = _contract(rng, doc_idx)
            q_text = "Draft a one-paragraph status report on this contract for the executive team"
            reference = {
                "expected_answer": (
                    f"Contract between {doc['party_a']} (Client) and {doc['party_b']} (Service Provider) "
                    f"is active from {doc['start']} through {doc['end']}, valued at ${doc['value']:,.2f}. "
                    f"Governed by {doc['governing_law']} law; termination requires {doc['notice_days']} days notice."
                ),
                "generation_type": "status_report",
                "key_facts": {
                    "parties": [doc['party_a'], doc['party_b']],
                    "value": doc['value'],
                },
            }
        elif gen_type == "action_items":
            doc = _medical_record(rng, doc_idx)
            q_text = "List the action items from this clinical record as a checklist"
            reference = {
                "expected_answer": (
                    f"[ ] Continue {doc['medication']} as directed\n"
                    f"[ ] Schedule follow-up appointment\n"
                    f"[ ] Review diet and exercise plan\n"
                    f"[ ] Monitor glucose levels (current: {doc['glucose']} mg/dL)\n"
                    f"[ ] Track blood pressure (current: {doc['bp']} mmHg)"
                ),
                "generation_type": "action_items",
                "key_facts": {
                    "medication": doc['medication'],
                    "glucose": doc['glucose'],
                },
            }
        else:  # executive_summary
            doc = _financial_statement(rng, doc_idx)
            q_text = f"Write a 3-sentence executive summary of {doc['company']}'s financial performance"
            reference = {
                "expected_answer": (
                    f"{doc['company']} reported revenue of ${doc['revenue']:,.2f} for fiscal year {doc['year']}. "
                    f"Gross profit reached ${doc['gross_profit']:,.2f} with EBITDA of ${doc['ebitda']:,.2f}. "
                    f"Net income was ${doc['net_income']:,.2f}, representing a "
                    f"{round(doc['net_income']/doc['revenue']*100, 1)}% net margin."
                ),
                "generation_type": "executive_summary",
                "key_facts": {
                    "revenue": doc['revenue'],
                    "net_income": doc['net_income'],
                },
            }
        prompt = f"Document:\n{doc['text']}\n\nTask: {q_text}"
        ex_id = _sha256_id(f"content_gen_{gen_type}_{i}")
        examples.append({
            "id": ex_id,
            "category": "content_generation",
            "prompt": prompt,
            "reference": reference,
            "difficulty": _difficulty(rng),
        })
    return examples


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_eval_bank() -> list[dict]:
    """Generate 700 deterministic eval examples across 7 categories."""
    rng = random.Random(42)
    examples = []
    examples.extend(_gen_extraction_accuracy(rng))
    examples.extend(_gen_table_excel_reasoning(rng))
    examples.extend(_gen_ocr_vision(rng))
    examples.extend(_gen_hallucination_probes(rng))
    examples.extend(_gen_intent_understanding(rng))
    examples.extend(_gen_cross_document(rng))
    examples.extend(_gen_content_generation(rng))
    assert len(examples) == _TOTAL, f"Expected {_TOTAL}, got {len(examples)}"
    return examples


def save_eval_bank(examples: list[dict], path: Path) -> None:
    """Write examples to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def load_eval_bank(path: Path) -> list[dict]:
    """Read examples from a JSONL file."""
    path = Path(path)
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


if __name__ == "__main__":
    bank = generate_eval_bank()
    print(f"Generated {len(bank)} examples")
    counts = {}
    for ex in bank:
        counts[ex["category"]] = counts.get(ex["category"], 0) + 1
    for cat, cnt in counts.items():
        print(f"  {cat}: {cnt}")
