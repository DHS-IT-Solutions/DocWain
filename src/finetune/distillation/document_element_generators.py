"""Document element identification training data generators for DocWain V2.

Produces high-quality SFT and DPO examples that teach DocWain to identify and
extract EVERY element from any document with near-perfect accuracy — invoices,
purchase orders, quotes, Excel tables, and general document structure.

Generators:
- generate_invoice_extraction_examples()     — 500 SFT
- generate_po_extraction_examples()          — 500 SFT
- generate_quote_extraction_examples()       — 500 SFT
- generate_excel_table_extraction_examples() — 300 SFT
- generate_document_structure_examples()     — 300 SFT
- generate_cross_field_validation_examples() — 200 SFT
- generate_element_dpo_pairs()               — 500 DPO
- generate_all_element_examples(output_dir) — orchestrator
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.finetune.v2.data_generator.base import (
    JSONLWriter,
    format_dpo_example,
    format_sft_example,
)

# ---------------------------------------------------------------------------
# Seed data pools
# ---------------------------------------------------------------------------

_UK_COMPANIES = [
    "UrbEdge Facilities Management Ltd",
    "BrightPath Solutions Ltd",
    "Thornfield Consulting Group",
    "Kensington Digital Services",
    "Harrow & Webb Engineering Ltd",
    "Aldgate Office Supplies",
    "BlueSky Facilities Ltd",
    "Sterling Procurement Partners",
    "Northgate Technology Ltd",
    "Clearwater Maintenance Services",
    "Apex Workspace Solutions Ltd",
    "Ironbridge Systems Ltd",
    "Meridian Services Group",
    "Castleton Property Management",
    "Redwood Facilities UK",
]

_US_COMPANIES = [
    "Dell Workspace Solutions Ltd",
    "Kaiser LLC",
    "Summit Capital Group",
    "Horizon Technology Inc",
    "Pinnacle Dynamics Corp",
    "Cascade Digital Solutions",
    "Vantage Business Services LLC",
    "Praxis Engineering Corp",
    "Alluvial Data Inc",
    "Solstice Media LLC",
    "ClearPath Analytics Inc",
    "BlueStar Consulting LLC",
    "Pacific Rim Logistics Inc",
    "Evergreen Systems Corp",
    "Lakefront Ventures LLC",
]

_EU_COMPANIES = [
    "Technologie AG München",
    "Dupont Services SARL",
    "Noord Logistiek B.V.",
    "Berliner Consulting GmbH",
    "Lyon Digital SAS",
    "Hasselt Solutions NV",
    "Rotterdam Supplies B.V.",
    "Antwerp Trade Partners NV",
    "Barcelona Tech S.L.",
    "Frankfurt Systems GmbH",
]

_UK_ADDRESSES = [
    ("25 Kingsway Business Centre", "London", "WC2B 6XF"),
    ("12 Victoria Street", "London", "SW1H 0ET"),
    ("5 Canary Wharf Tower", "London", "E14 5AB"),
    ("88 King Street", "Manchester", "M2 4WQ"),
    ("32 St Mary Street", "Cardiff", "CF10 1AB"),
    ("14 George Street", "Edinburgh", "EH2 2PF"),
    ("7 Colmore Row", "Birmingham", "B3 2BJ"),
    ("21 Queen Street", "Leeds", "LS1 2TW"),
    ("6 Park Row", "Bristol", "BS1 5LJ"),
    ("45 Cloth Market", "Newcastle", "NE1 1EE"),
]

_US_ADDRESSES = [
    ("7642 Mariner Drive, Suite 300", "Tampa", "FL 33609"),
    ("1500 Broadway, Floor 12", "New York", "NY 10036"),
    ("200 West Adams Street", "Chicago", "IL 60606"),
    ("3 Embarcadero Center", "San Francisco", "CA 94111"),
    ("555 17th Street, Suite 1600", "Denver", "CO 80202"),
    ("1801 California Street", "Denver", "CO 80202"),
    ("1100 Louisiana Street", "Houston", "TX 77002"),
    ("700 Fifth Avenue", "Seattle", "WA 98104"),
    ("4600 S Syracuse Street", "Denver", "CO 80237"),
    ("10 South Wacker Drive", "Chicago", "IL 60606"),
]

_EU_ADDRESSES = [
    ("257 Douglas Centers", "Hasselt", "3500 BE"),
    ("Maximilianstraße 35", "München", "80539 DE"),
    ("Rue de Rivoli 45", "Paris", "75001 FR"),
    ("Keizersgracht 452", "Amsterdam", "1016 GD NL"),
    ("Calle Gran Via 28", "Madrid", "28013 ES"),
]

_UK_PHONES = [
    "+44 (0)20 7812 4456",
    "+44 (0)20 3456 7890",
    "+44 (0)161 234 5678",
    "+44 (0)113 456 7890",
    "+44 (0)117 345 6789",
    "+44 (0)121 234 5678",
    "+44 (0)131 234 5678",
]

_US_PHONES = [
    "(813) 555-9210",
    "(212) 555-4400",
    "(312) 555-7800",
    "(415) 555-3200",
    "(303) 555-6100",
    "(713) 555-8800",
    "(206) 555-1200",
]

_EMAILS_UK = [
    "accounts@urbedge.co.uk",
    "billing@brightpath.co.uk",
    "invoices@thornfield.co.uk",
    "finance@aldgate-supplies.co.uk",
    "accounts@clearwater-maint.co.uk",
]

_EMAILS_US = [
    "billing@dellworkspace.com",
    "accounts@summitcapital.com",
    "invoices@horizontech.com",
    "ar@pinnacledynamics.com",
    "finance@cascadedigital.com",
]

_PRODUCT_SERVICES = [
    ("Office Cleaning Services - Monthly Contract", 1, None),
    ("HVAC Maintenance and Servicing", 1, None),
    ("Security System Installation", 1, None),
    ("Network Infrastructure Cabling", 1, None),
    ("Dell Latitude 5540 Laptop", 5, None),
    ("Microsoft 365 Business Premium (Annual)", 10, None),
    ("Ergonomic Office Chair - Mesh Back", 20, None),
    ("LED Monitor 27\" 4K UHD", 8, None),
    ("Printer Toner Cartridge (Black)", 25, None),
    ("Managed IT Support - 40 Hours", 1, None),
    ("Electrical Safety Inspection", 1, None),
    ("Plumbing Maintenance Service", 1, None),
    ("Waste Management Monthly Fee", 1, None),
    ("Cloud Storage Solution - 1TB/year", 5, None),
    ("Software Development Consulting - 80 Hours", 1, None),
    ("Fire Safety Equipment Check", 1, None),
    ("Parking Space Rental - Annual", 3, None),
    ("Catering Equipment Hire", 1, None),
    ("Document Management System Licence", 1, None),
    ("Photocopier Lease - Quarterly", 1, None),
    ("Wireless Access Point Installation", 4, None),
    ("Data Centre Rack Space - Monthly", 2, None),
    ("Training Room Hire - Full Day", 2, None),
    ("Video Conferencing Setup", 1, None),
    ("Power Distribution Unit", 6, None),
]

_PAYMENT_TERMS = [
    "Net 30",
    "Net 14",
    "Net 60",
    "Due on receipt",
    "2/10 Net 30",
    "30 days from invoice date",
    "14 days EOM",
    "Immediate payment required",
]

_SHIPPING_METHODS = [
    "Standard Delivery",
    "Express Next Day",
    "Courier Service",
    "Collected by Buyer",
    "Freight Forwarding",
    "Royal Mail Tracked",
    "DHL Express",
    "FedEx Ground",
]

_INCOTERMS = ["EXW", "FOB", "CIF", "DDP", "DAP", "FCA"]

_CURRENCIES = {
    "GBP": ("£", "UK"),
    "USD": ("$", "US"),
    "EUR": ("€", "EU"),
}

_VAT_RATES = {"GBP": 0.20, "USD": 0.0, "EUR": 0.21}
_TAX_LABELS = {"GBP": "VAT (20%)", "USD": "Sales Tax (0%)", "EUR": "VAT (21%)"}


def _rnd() -> random.Random:
    return random.Random(random.randint(0, 2**31))


def _pick(pool: List, r: random.Random, n: int = 1):
    sample = r.sample(pool, min(n, len(pool)))
    return sample[0] if n == 1 else sample


def _fmt_currency(amount: float, symbol: str) -> str:
    return f"{symbol}{amount:,.2f}"


def _make_sft(
    query: str,
    reasoning: str,
    answer: str,
    *,
    area: str,
    difficulty: str,
    category: str,
) -> Dict[str, Any]:
    example = format_sft_example(query, reasoning, answer)
    example["area"] = area
    example["difficulty"] = difficulty
    example["category"] = category
    example["source"] = "claude_element_distillation"
    return example


def _make_dpo(
    query: str,
    chosen_reasoning: str,
    chosen_answer: str,
    rejected_reasoning: str,
    rejected_answer: str,
    *,
    area: str,
    difficulty: str,
    category: str,
) -> Dict[str, Any]:
    example = format_dpo_example(
        query,
        chosen_reasoning,
        chosen_answer,
        rejected_reasoning,
        rejected_answer,
    )
    example["area"] = area
    example["difficulty"] = difficulty
    example["category"] = category
    example["source"] = "claude_element_distillation"
    return example


# ---------------------------------------------------------------------------
# Invoice data synthesis helpers
# ---------------------------------------------------------------------------

def _build_invoice(r: random.Random) -> Dict[str, Any]:
    currency_code = r.choice(list(_CURRENCIES.keys()))
    symbol, region = _CURRENCIES[currency_code]

    if region == "UK":
        vendor_name = _pick(_UK_COMPANIES, r)
        vendor_addr = _pick(_UK_ADDRESSES, r)
        vendor_phone = _pick(_UK_PHONES, r)
        vendor_email = _pick(_EMAILS_UK, r)
        customer_name = _pick(_UK_COMPANIES + _US_COMPANIES, r)
        customer_addr = _pick(_UK_ADDRESSES + _US_ADDRESSES, r)
    elif region == "US":
        vendor_name = _pick(_US_COMPANIES, r)
        vendor_addr = _pick(_US_ADDRESSES, r)
        vendor_phone = _pick(_US_PHONES, r)
        vendor_email = _pick(_EMAILS_US, r)
        customer_name = _pick(_US_COMPANIES, r)
        customer_addr = _pick(_US_ADDRESSES, r)
    else:
        vendor_name = _pick(_EU_COMPANIES, r)
        vendor_addr = _pick(_EU_ADDRESSES, r)
        vendor_phone = _pick(_UK_PHONES + _US_PHONES, r)
        vendor_email = f"billing@{vendor_name.split()[0].lower()}.eu"
        customer_name = _pick(_EU_COMPANIES + _UK_COMPANIES, r)
        customer_addr = _pick(_EU_ADDRESSES + _UK_ADDRESSES, r)

    year = r.randint(2024, 2026)
    month = r.randint(1, 12)
    day = r.randint(1, 28)
    inv_date = f"{day:02d}/{month:02d}/{year}"
    due_day = day + r.choice([14, 30, 60])
    due_month = month
    if due_day > 28:
        due_day -= 28
        due_month = (month % 12) + 1
    due_date = f"{due_day:02d}/{due_month:02d}/{year}"

    inv_num = f"INV-{year}-{r.randint(10000, 99999):05d}"
    po_ref = f"PO{r.randint(1000000, 9999999)}"

    n_items = r.randint(2, 6)
    items = []
    selected = r.sample(_PRODUCT_SERVICES, min(n_items, len(_PRODUCT_SERVICES)))
    for desc, base_qty, _ in selected:
        qty = r.randint(1, 10) if base_qty == 1 else r.randint(base_qty, base_qty * 3)
        unit_price = round(r.uniform(50, 5000), 2)
        amount = round(qty * unit_price, 2)
        items.append({
            "description": desc,
            "qty": qty,
            "unit_price": unit_price,
            "amount": amount,
        })

    subtotal = round(sum(i["amount"] for i in items), 2)
    discount = round(subtotal * r.choice([0, 0, 0, 0.05, 0.10]), 2)
    discounted = round(subtotal - discount, 2)
    tax_rate = _VAT_RATES[currency_code]
    tax = round(discounted * tax_rate, 2)
    total = round(discounted + tax, 2)

    bank_details = {
        "bank": r.choice(["Barclays Bank", "HSBC", "NatWest", "Lloyds Bank", "Bank of America", "Wells Fargo", "Deutsche Bank"]),
        "account_name": vendor_name,
        "sort_code": f"{r.randint(10,99)}-{r.randint(10,99)}-{r.randint(10,99)}",
        "account_number": str(r.randint(10000000, 99999999)),
        "iban": f"GB{r.randint(10,99)}BARC{r.randint(10000000000000000, 99999999999999999)}",
    }

    return {
        "inv_num": inv_num,
        "inv_date": inv_date,
        "due_date": due_date,
        "po_ref": po_ref,
        "vendor_name": vendor_name,
        "vendor_street": vendor_addr[0],
        "vendor_city": vendor_addr[1],
        "vendor_postcode": vendor_addr[2],
        "vendor_phone": vendor_phone,
        "vendor_email": vendor_email,
        "customer_name": customer_name,
        "customer_street": customer_addr[0],
        "customer_city": customer_addr[1],
        "customer_postcode": customer_addr[2],
        "items": items,
        "subtotal": subtotal,
        "discount": discount,
        "tax": tax,
        "tax_label": _TAX_LABELS[currency_code],
        "total": total,
        "currency": currency_code,
        "symbol": symbol,
        "payment_terms": _pick(_PAYMENT_TERMS, r),
        "bank_details": bank_details,
        "delivery_address": f"{customer_addr[0]}, {customer_addr[1]}, {customer_addr[2]}",
    }


def _render_invoice_text(inv: Dict[str, Any]) -> str:
    s = inv["symbol"]
    lines = [
        f"{inv['inv_date']} | {inv['inv_num']} | {inv['vendor_name']}",
        f"{inv['vendor_street']}, {inv['vendor_city']}, {inv['vendor_postcode']}",
        f"Telephone: {inv['vendor_phone']}  Email: {inv['vendor_email']}",
        f"",
        f"INVOICE TO:",
        f"{inv['customer_name']}",
        f"{inv['customer_street']}, {inv['customer_city']}, {inv['customer_postcode']}",
        f"",
        f"INVOICE NUMBER: {inv['inv_num']}",
        f"INVOICE DATE: {inv['inv_date']}",
        f"DUE DATE: {inv['due_date']}",
        f"PO REFERENCE: {inv['po_ref']}",
        f"",
        f"ITEMS:",
    ]
    for item in inv["items"]:
        lines.append(
            f"  {item['description']} | Qty: {item['qty']} | Unit: {s}{item['unit_price']:.2f} | Amount: {s}{item['amount']:.2f}"
        )
    lines += [
        f"",
        f"SUBTOTAL: {s}{inv['subtotal']:.2f}",
    ]
    if inv["discount"] > 0:
        lines.append(f"DISCOUNT: -{s}{inv['discount']:.2f}")
    lines += [
        f"{inv['tax_label']}: {s}{inv['tax']:.2f}",
        f"TOTAL DUE: {s}{inv['total']:.2f}",
        f"",
        f"PAYMENT TERMS: {inv['payment_terms']}",
        f"BANK: {inv['bank_details']['bank']}  Account: {inv['bank_details']['account_number']}",
        f"Sort Code: {inv['bank_details']['sort_code']}  IBAN: {inv['bank_details']['iban']}",
        f"",
        f"DELIVERY ADDRESS: {inv['delivery_address']}",
    ]
    return "\n".join(lines)


def _invoice_reasoning(inv: Dict[str, Any]) -> str:
    s = inv["symbol"]
    items_summary = ", ".join(f'"{i["description"]}"' for i in inv["items"])
    return (
        f"I need to systematically identify every element in this invoice document.\n\n"
        f"Step 1 — Document type confirmation: The header contains 'INVOICE NUMBER' and 'TOTAL DUE', confirming this is an invoice.\n\n"
        f"Step 2 — Invoice identifiers: I can see:\n"
        f"  - Invoice number: {inv['inv_num']} (appears in header and body)\n"
        f"  - Invoice date: {inv['inv_date']}\n"
        f"  - Due date: {inv['due_date']}\n"
        f"  - PO reference: {inv['po_ref']}\n\n"
        f"Step 3 — Vendor details (issuing party): The top block contains the vendor:\n"
        f"  - Name: {inv['vendor_name']}\n"
        f"  - Address: {inv['vendor_street']}, {inv['vendor_city']}, {inv['vendor_postcode']}\n"
        f"  - Phone: {inv['vendor_phone']}\n"
        f"  - Email: {inv['vendor_email']}\n\n"
        f"Step 4 — Customer/bill-to details: The 'INVOICE TO' block contains:\n"
        f"  - Name: {inv['customer_name']}\n"
        f"  - Address: {inv['customer_street']}, {inv['customer_city']}, {inv['customer_postcode']}\n\n"
        f"Step 5 — Line items: I count {len(inv['items'])} line items: {items_summary}\n"
        f"  For each I extract: description, quantity, unit price, calculated amount.\n\n"
        f"Step 6 — Financial summary:\n"
        f"  - Subtotal: {s}{inv['subtotal']:.2f} (sum of all line items)\n"
        f"  - Discount: {s}{inv['discount']:.2f}\n"
        f"  - Tax ({inv['tax_label']}): {s}{inv['tax']:.2f}\n"
        f"  - Total: {s}{inv['total']:.2f}\n\n"
        f"Step 7 — Payment and banking:\n"
        f"  - Terms: {inv['payment_terms']}\n"
        f"  - Bank: {inv['bank_details']['bank']}, Account: {inv['bank_details']['account_number']}\n"
        f"  - IBAN: {inv['bank_details']['iban']}\n\n"
        f"Step 8 — Delivery address: {inv['delivery_address']}\n\n"
        f"Step 9 — Cross-check: Verify {s}{inv['subtotal']:.2f} - {s}{inv['discount']:.2f} + {s}{inv['tax']:.2f} = {s}{inv['total']:.2f} ✓"
    )


def _invoice_answer(inv: Dict[str, Any]) -> str:
    s = inv["symbol"]

    # Build line items table
    table_rows = "| # | Description | Qty | Unit Price | Amount |\n|---|---|---|---|---|\n"
    for i, item in enumerate(inv["items"], 1):
        table_rows += f"| {i} | {item['description']} | {item['qty']} | {s}{item['unit_price']:.2f} | {s}{item['amount']:.2f} |\n"

    discount_line = f"| **Discount** | | | | **-{s}{inv['discount']:.2f}** |\n" if inv["discount"] > 0 else ""

    return (
        f"## Invoice Extraction — Complete Field Analysis\n\n"
        f"### Document Identifiers\n"
        f"| Field | Value |\n|---|---|\n"
        f"| **Invoice Number** | **{inv['inv_num']}** |\n"
        f"| **Invoice Date** | **{inv['inv_date']}** |\n"
        f"| **Due Date** | **{inv['due_date']}** |\n"
        f"| **PO Reference** | **{inv['po_ref']}** |\n"
        f"| **Currency** | **{inv['currency']}** |\n\n"
        f"### Vendor (Issuing Party)\n"
        f"| Field | Value |\n|---|---|\n"
        f"| **Company** | **{inv['vendor_name']}** |\n"
        f"| **Street** | {inv['vendor_street']} |\n"
        f"| **City** | {inv['vendor_city']} |\n"
        f"| **Postcode** | {inv['vendor_postcode']} |\n"
        f"| **Phone** | {inv['vendor_phone']} |\n"
        f"| **Email** | {inv['vendor_email']} |\n\n"
        f"### Bill To (Customer)\n"
        f"| Field | Value |\n|---|---|\n"
        f"| **Company** | **{inv['customer_name']}** |\n"
        f"| **Street** | {inv['customer_street']} |\n"
        f"| **City** | {inv['customer_city']} |\n"
        f"| **Postcode** | {inv['customer_postcode']} |\n\n"
        f"### Line Items\n"
        f"{table_rows}\n"
        f"### Financial Summary\n"
        f"| Component | Amount |\n|---|---|\n"
        f"| **Subtotal** | **{s}{inv['subtotal']:.2f}** |\n"
        f"{discount_line}"
        f"| **{inv['tax_label']}** | **{s}{inv['tax']:.2f}** |\n"
        f"| **TOTAL DUE** | **{s}{inv['total']:.2f}** |\n\n"
        f"### Payment Terms\n"
        f"- **Terms:** {inv['payment_terms']}\n"
        f"- **Bank:** {inv['bank_details']['bank']}\n"
        f"- **Account Name:** {inv['bank_details']['account_name']}\n"
        f"- **Account Number:** {inv['bank_details']['account_number']}\n"
        f"- **Sort Code:** {inv['bank_details']['sort_code']}\n"
        f"- **IBAN:** {inv['bank_details']['iban']}\n\n"
        f"### Delivery Address\n"
        f"{inv['delivery_address']}\n\n"
        f"---\n"
        f"**Extraction complete — {len(inv['items'])} line items, total {s}{inv['total']:.2f} {inv['currency']}**"
    )


# ---------------------------------------------------------------------------
# 1. Invoice extraction examples
# ---------------------------------------------------------------------------

def generate_invoice_extraction_examples() -> List[Dict[str, Any]]:
    """Generate 500 invoice extraction SFT examples."""
    examples = []
    query_templates = [
        "Extract all fields from this invoice document:\n\n{doc}",
        "Parse and identify every element in the following invoice:\n\n{doc}",
        "This is an invoice document. Identify and extract all structured data:\n\n{doc}",
        "Perform complete extraction of this invoice — every field, every line item:\n\n{doc}",
        "Extract the invoice data in structured format:\n\n{doc}",
    ]
    difficulties = ["easy", "medium", "hard"]
    diff_weights = [0.3, 0.5, 0.2]

    r = _rnd()
    for i in range(500):
        inv = _build_invoice(r)
        doc_text = _render_invoice_text(inv)
        tmpl = r.choice(query_templates)
        query = tmpl.format(doc=doc_text)
        reasoning = _invoice_reasoning(inv)
        answer = _invoice_answer(inv)
        diff = r.choices(difficulties, weights=diff_weights, k=1)[0]
        examples.append(_make_sft(
            query, reasoning, answer,
            area="document_extraction",
            difficulty=diff,
            category="invoice",
        ))

    return examples


# ---------------------------------------------------------------------------
# PO data synthesis helpers
# ---------------------------------------------------------------------------

def _build_po(r: random.Random) -> Dict[str, Any]:
    currency_code = r.choice(list(_CURRENCIES.keys()))
    symbol, region = _CURRENCIES[currency_code]

    buyer_name = _pick(_UK_COMPANIES + _US_COMPANIES, r)
    buyer_addr = _pick(_UK_ADDRESSES + _US_ADDRESSES, r)

    if region == "UK":
        vendor_name = _pick(_UK_COMPANIES, r)
        vendor_addr = _pick(_UK_ADDRESSES, r)
        vendor_phone = _pick(_UK_PHONES, r)
    else:
        vendor_name = _pick(_US_COMPANIES, r)
        vendor_addr = _pick(_US_ADDRESSES, r)
        vendor_phone = _pick(_US_PHONES, r)

    year = r.randint(2024, 2026)
    month = r.randint(1, 12)
    day = r.randint(1, 28)
    po_date = f"{day:02d}/{month:02d}/{year}"

    delivery_day = day + r.randint(7, 45)
    delivery_month = month
    if delivery_day > 28:
        delivery_day -= 28
        delivery_month = (month % 12) + 1
    delivery_date = f"{delivery_day:02d}/{delivery_month:02d}/{year}"

    po_num = f"PO{r.randint(1000000, 9999999)}"

    n_items = r.randint(2, 7)
    items = []
    selected = r.sample(_PRODUCT_SERVICES, min(n_items, len(_PRODUCT_SERVICES)))
    for desc, base_qty, _ in selected:
        qty = r.randint(1, 15)
        unit_price = round(r.uniform(20, 3000), 2)
        amount = round(qty * unit_price, 2)
        items.append({
            "description": desc,
            "qty": qty,
            "unit_price": unit_price,
            "amount": amount,
            "part_number": f"PN-{r.randint(10000, 99999)}",
        })

    subtotal = round(sum(i["amount"] for i in items), 2)
    tax_rate = _VAT_RATES[currency_code]
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)

    approver = r.choice([
        "John Smith - Procurement Manager",
        "Sarah Davies - Finance Director",
        "Michael Brown - Operations Head",
        "Emma Wilson - Purchasing Manager",
        "Robert Taylor - CFO",
    ])

    return {
        "po_num": po_num,
        "po_date": po_date,
        "delivery_date": delivery_date,
        "vendor_name": vendor_name,
        "vendor_street": vendor_addr[0],
        "vendor_city": vendor_addr[1],
        "vendor_postcode": vendor_addr[2],
        "vendor_phone": vendor_phone,
        "buyer_name": buyer_name,
        "buyer_street": buyer_addr[0],
        "buyer_city": buyer_addr[1],
        "buyer_postcode": buyer_addr[2],
        "items": items,
        "subtotal": subtotal,
        "tax": tax,
        "tax_label": _TAX_LABELS[currency_code],
        "total": total,
        "currency": currency_code,
        "symbol": symbol,
        "payment_terms": _pick(_PAYMENT_TERMS, r),
        "shipping_method": _pick(_SHIPPING_METHODS, r),
        "incoterms": _pick(_INCOTERMS, r),
        "approver": approver,
        "delivery_address": f"{buyer_addr[0]}, {buyer_addr[1]}, {buyer_addr[2]}",
        "notes": r.choice([
            "All goods must be delivered in original packaging.",
            "Please confirm receipt of this order within 2 business days.",
            "Partial deliveries are not accepted unless pre-agreed.",
            "All items subject to quality inspection on receipt.",
            "",
        ]),
    }


def _render_po_text(po: Dict[str, Any]) -> str:
    s = po["symbol"]
    lines = [
        f"PURCHASE ORDER # {po['po_num']}",
        f"The amount of the Purchase Order is the agreed fixed price.",
        f"",
        f"BUYER:",
        f"{po['buyer_name']}",
        f"{po['buyer_street']}, {po['buyer_city']}, {po['buyer_postcode']}",
        f"",
        f"VENDOR:",
        f"{po['vendor_name']}",
        f"{po['vendor_street']}, {po['vendor_city']}, {po['vendor_postcode']}",
        f"Phone: {po['vendor_phone']}",
        f"",
        f"PO DATE: {po['po_date']}",
        f"REQUIRED DELIVERY DATE: {po['delivery_date']}",
        f"SHIPPING METHOD: {po['shipping_method']}",
        f"INCOTERMS: {po['incoterms']}",
        f"",
        f"LINE ITEMS:",
    ]
    for item in po["items"]:
        lines.append(
            f"  Part# {item['part_number']} | {item['description']} | Qty: {item['qty']} | Unit: {s}{item['unit_price']:.2f} | Total: {s}{item['amount']:.2f}"
        )
    lines += [
        f"",
        f"SUBTOTAL: {s}{po['subtotal']:.2f}",
        f"{po['tax_label']}: {s}{po['tax']:.2f}",
        f"TOTAL ORDER VALUE: {s}{po['total']:.2f}",
        f"",
        f"PAYMENT TERMS: {po['payment_terms']}",
        f"DELIVERY ADDRESS: {po['delivery_address']}",
        f"",
        f"AUTHORISED BY: {po['approver']}",
    ]
    if po["notes"]:
        lines += [f"", f"NOTES: {po['notes']}"]
    return "\n".join(lines)


def _po_answer(po: Dict[str, Any]) -> str:
    s = po["symbol"]
    table_rows = "| # | Part Number | Description | Qty | Unit Price | Line Total |\n|---|---|---|---|---|---|\n"
    for i, item in enumerate(po["items"], 1):
        table_rows += f"| {i} | {item['part_number']} | {item['description']} | {item['qty']} | {s}{item['unit_price']:.2f} | {s}{item['amount']:.2f} |\n"

    return (
        f"## Purchase Order Extraction — Complete Field Analysis\n\n"
        f"### PO Identifiers\n"
        f"| Field | Value |\n|---|---|\n"
        f"| **PO Number** | **{po['po_num']}** |\n"
        f"| **PO Date** | **{po['po_date']}** |\n"
        f"| **Required Delivery Date** | **{po['delivery_date']}** |\n"
        f"| **Currency** | **{po['currency']}** |\n\n"
        f"### Buyer\n"
        f"| Field | Value |\n|---|---|\n"
        f"| **Company** | **{po['buyer_name']}** |\n"
        f"| **Street** | {po['buyer_street']} |\n"
        f"| **City** | {po['buyer_city']} |\n"
        f"| **Postcode** | {po['buyer_postcode']} |\n\n"
        f"### Vendor (Supplier)\n"
        f"| Field | Value |\n|---|---|\n"
        f"| **Company** | **{po['vendor_name']}** |\n"
        f"| **Street** | {po['vendor_street']} |\n"
        f"| **City** | {po['vendor_city']} |\n"
        f"| **Postcode** | {po['vendor_postcode']} |\n"
        f"| **Phone** | {po['vendor_phone']} |\n\n"
        f"### Line Items\n"
        f"{table_rows}\n"
        f"### Financial Summary\n"
        f"| Component | Amount |\n|---|---|\n"
        f"| **Subtotal** | **{s}{po['subtotal']:.2f}** |\n"
        f"| **{po['tax_label']}** | **{s}{po['tax']:.2f}** |\n"
        f"| **TOTAL ORDER VALUE** | **{s}{po['total']:.2f}** |\n\n"
        f"### Logistics & Terms\n"
        f"- **Shipping Method:** {po['shipping_method']}\n"
        f"- **Incoterms:** {po['incoterms']}\n"
        f"- **Payment Terms:** {po['payment_terms']}\n"
        f"- **Delivery Address:** {po['delivery_address']}\n\n"
        f"### Authorisation\n"
        f"- **Approved By:** {po['approver']}\n"
        + (f"\n### Notes\n{po['notes']}\n" if po["notes"] else "")
        + f"\n---\n**Extraction complete — {len(po['items'])} line items, total {s}{po['total']:.2f} {po['currency']}**"
    )


# ---------------------------------------------------------------------------
# 2. PO extraction examples
# ---------------------------------------------------------------------------

def generate_po_extraction_examples() -> List[Dict[str, Any]]:
    """Generate 500 purchase order extraction SFT examples."""
    examples = []
    query_templates = [
        "Extract all fields from this purchase order:\n\n{doc}",
        "Parse this purchase order document and identify every data element:\n\n{doc}",
        "Perform complete structured extraction of this PO:\n\n{doc}",
        "Identify and extract all information from this purchase order:\n\n{doc}",
        "This is a purchase order. Extract every field including all line items:\n\n{doc}",
    ]
    r = _rnd()
    for i in range(500):
        po = _build_po(r)
        doc_text = _render_po_text(po)
        tmpl = r.choice(query_templates)
        query = tmpl.format(doc=doc_text)

        s = po["symbol"]
        items_summary = ", ".join(f'"{i["description"]}"' for i in po["items"])
        reasoning = (
            f"Systematically extracting all elements from this purchase order.\n\n"
            f"Step 1 — Document type: 'PURCHASE ORDER #' header confirms document type.\n\n"
            f"Step 2 — PO identifiers:\n"
            f"  - PO Number: {po['po_num']}\n"
            f"  - PO Date: {po['po_date']}\n"
            f"  - Required delivery: {po['delivery_date']}\n\n"
            f"Step 3 — Buyer (issuing company): {po['buyer_name']}, {po['buyer_city']}\n\n"
            f"Step 4 — Vendor (supplier): {po['vendor_name']}, {po['vendor_city']}\n\n"
            f"Step 5 — Line items ({len(po['items'])} items): {items_summary}\n"
            f"  Each item has: part number, description, quantity, unit price, line total.\n\n"
            f"Step 6 — Financial totals: subtotal {s}{po['subtotal']:.2f}, "
            f"tax {s}{po['tax']:.2f}, total {s}{po['total']:.2f}\n\n"
            f"Step 7 — Logistics: {po['shipping_method']}, {po['incoterms']}\n\n"
            f"Step 8 — Payment terms: {po['payment_terms']}\n\n"
            f"Step 9 — Authorisation: {po['approver']}\n\n"
            f"Step 10 — Validation: sum of line items {s}{po['subtotal']:.2f} + "
            f"tax {s}{po['tax']:.2f} = {s}{po['total']:.2f} ✓"
        )
        answer = _po_answer(po)
        difficulty = r.choice(["easy", "medium", "medium", "hard"])
        examples.append(_make_sft(
            query, reasoning, answer,
            area="document_extraction",
            difficulty=difficulty,
            category="purchase_order",
        ))

    return examples


# ---------------------------------------------------------------------------
# Quote data synthesis helpers
# ---------------------------------------------------------------------------

def _build_quote(r: random.Random) -> Dict[str, Any]:
    currency_code = r.choice(list(_CURRENCIES.keys()))
    symbol, region = _CURRENCIES[currency_code]

    if region == "UK":
        vendor_name = _pick(_UK_COMPANIES, r)
        vendor_addr = _pick(_UK_ADDRESSES, r)
        vendor_phone = _pick(_UK_PHONES, r)
        vendor_email = _pick(_EMAILS_UK, r)
        customer_name = _pick(_UK_COMPANIES + _US_COMPANIES, r)
        customer_addr = _pick(_UK_ADDRESSES + _US_ADDRESSES, r)
    elif region == "US":
        vendor_name = _pick(_US_COMPANIES, r)
        vendor_addr = _pick(_US_ADDRESSES, r)
        vendor_phone = _pick(_US_PHONES, r)
        vendor_email = _pick(_EMAILS_US, r)
        customer_name = _pick(_US_COMPANIES, r)
        customer_addr = _pick(_US_ADDRESSES, r)
    else:
        vendor_name = _pick(_EU_COMPANIES, r)
        vendor_addr = _pick(_EU_ADDRESSES, r)
        vendor_phone = _pick(_UK_PHONES + _US_PHONES, r)
        vendor_email = f"sales@{vendor_name.split()[0].lower()}.eu"
        customer_name = _pick(_EU_COMPANIES + _UK_COMPANIES, r)
        customer_addr = _pick(_EU_ADDRESSES + _UK_ADDRESSES, r)

    year = r.randint(2024, 2026)
    month = r.randint(1, 12)
    day = r.randint(1, 28)
    quote_date = f"{day:02d}/{month:02d}/{year}"
    validity_days = r.choice([14, 30, 45, 60, 90])
    expiry_day = day + validity_days
    expiry_month = month
    if expiry_day > 28:
        expiry_day -= 28
        expiry_month = (month % 12) + 1
    expiry_date = f"{expiry_day:02d}/{expiry_month:02d}/{year}"

    quote_prefix = r.choice(["WSG", "QT", "QUO", "QUOT", "Q"])
    quote_num = f"{quote_prefix}{r.randint(100000, 999999)}"

    n_items = r.randint(2, 8)
    items = []
    selected = r.sample(_PRODUCT_SERVICES, min(n_items, len(_PRODUCT_SERVICES)))
    for desc, base_qty, _ in selected:
        qty = r.randint(1, 20)
        unit_price = round(r.uniform(15, 4000), 2)
        amount = round(qty * unit_price, 2)
        items.append({
            "description": desc,
            "qty": qty,
            "unit_price": unit_price,
            "amount": amount,
        })

    subtotal = round(sum(i["amount"] for i in items), 2)
    discount_pct = r.choice([0, 0, 5, 10, 15])
    discount = round(subtotal * discount_pct / 100, 2)
    discounted = round(subtotal - discount, 2)
    tax_rate = _VAT_RATES[currency_code]
    tax = round(discounted * tax_rate, 2)
    total = round(discounted + tax, 2)

    terms = r.choice([
        "Standard terms and conditions apply. Prices are fixed for the validity period.",
        "This quotation is valid for the stated period. Delivery times are estimates only.",
        "Payment required within 30 days of delivery. Standard warranty applies.",
        "Prices exclude installation unless stated. Technical support included for 12 months.",
    ])

    return {
        "quote_num": quote_num,
        "quote_date": quote_date,
        "expiry_date": expiry_date,
        "validity_days": validity_days,
        "vendor_name": vendor_name,
        "vendor_street": vendor_addr[0],
        "vendor_city": vendor_addr[1],
        "vendor_postcode": vendor_addr[2],
        "vendor_phone": vendor_phone,
        "vendor_email": vendor_email,
        "customer_name": customer_name,
        "customer_street": customer_addr[0],
        "customer_city": customer_addr[1],
        "customer_postcode": customer_addr[2],
        "items": items,
        "subtotal": subtotal,
        "discount": discount,
        "discount_pct": discount_pct,
        "tax": tax,
        "tax_label": _TAX_LABELS[currency_code],
        "total": total,
        "currency": currency_code,
        "symbol": symbol,
        "payment_terms": _pick(_PAYMENT_TERMS, r),
        "terms_and_conditions": terms,
        "notes": r.choice([
            "Prices are subject to change after expiry date.",
            "Volume discounts available on orders over 50 units.",
            "Free delivery on orders above minimum order value.",
            "",
        ]),
    }


def _render_quote_text(q: Dict[str, Any]) -> str:
    s = q["symbol"]
    lines = [
        f"Quotation # {q['quote_num']}",
        f"{q['vendor_name']}",
        f"{q['vendor_street']}, {q['vendor_city']}, {q['vendor_postcode']}",
        f"{q['vendor_phone']}",
        f"{q['vendor_email']}",
        f"",
        f"PREPARED FOR:",
        f"{q['customer_name']}",
        f"{q['customer_street']}, {q['customer_city']}, {q['customer_postcode']}",
        f"",
        f"QUOTATION DATE: {q['quote_date']}",
        f"VALID UNTIL: {q['expiry_date']} ({q['validity_days']} days)",
        f"",
        f"ITEMS:",
    ]
    for item in q["items"]:
        lines.append(
            f"  {item['description']} | Qty: {item['qty']} | Unit Price: {s}{item['unit_price']:.2f} | Total: {s}{item['amount']:.2f}"
        )
    lines += [
        f"",
        f"SUBTOTAL: {s}{q['subtotal']:.2f}",
    ]
    if q["discount"] > 0:
        lines.append(f"DISCOUNT ({q['discount_pct']}%): -{s}{q['discount']:.2f}")
    lines += [
        f"{q['tax_label']}: {s}{q['tax']:.2f}",
        f"GRAND TOTAL: {s}{q['total']:.2f}",
        f"",
        f"PAYMENT TERMS: {q['payment_terms']}",
        f"",
        f"TERMS & CONDITIONS:",
        f"{q['terms_and_conditions']}",
    ]
    if q["notes"]:
        lines += [f"", f"NOTES: {q['notes']}"]
    return "\n".join(lines)


def _quote_answer(q: Dict[str, Any]) -> str:
    s = q["symbol"]
    table_rows = "| # | Description | Qty | Unit Price | Line Total |\n|---|---|---|---|---|\n"
    for i, item in enumerate(q["items"], 1):
        table_rows += f"| {i} | {item['description']} | {item['qty']} | {s}{item['unit_price']:.2f} | {s}{item['amount']:.2f} |\n"

    discount_line = f"| **Discount ({q['discount_pct']}%)** | | | | **-{s}{q['discount']:.2f}** |\n" if q["discount"] > 0 else ""

    return (
        f"## Quotation Extraction — Complete Field Analysis\n\n"
        f"### Quote Identifiers\n"
        f"| Field | Value |\n|---|---|\n"
        f"| **Quotation Number** | **{q['quote_num']}** |\n"
        f"| **Quote Date** | **{q['quote_date']}** |\n"
        f"| **Valid Until** | **{q['expiry_date']}** |\n"
        f"| **Validity Period** | **{q['validity_days']} days** |\n"
        f"| **Currency** | **{q['currency']}** |\n\n"
        f"### Vendor (Quoting Party)\n"
        f"| Field | Value |\n|---|---|\n"
        f"| **Company** | **{q['vendor_name']}** |\n"
        f"| **Street** | {q['vendor_street']} |\n"
        f"| **City** | {q['vendor_city']} |\n"
        f"| **Postcode** | {q['vendor_postcode']} |\n"
        f"| **Phone** | {q['vendor_phone']} |\n"
        f"| **Email** | {q['vendor_email']} |\n\n"
        f"### Customer (Recipient)\n"
        f"| Field | Value |\n|---|---|\n"
        f"| **Company** | **{q['customer_name']}** |\n"
        f"| **Street** | {q['customer_street']} |\n"
        f"| **City** | {q['customer_city']} |\n"
        f"| **Postcode** | {q['customer_postcode']} |\n\n"
        f"### Line Items\n"
        f"{table_rows}\n"
        f"### Financial Summary\n"
        f"| Component | Amount |\n|---|---|\n"
        f"| **Subtotal** | **{s}{q['subtotal']:.2f}** |\n"
        f"{discount_line}"
        f"| **{q['tax_label']}** | **{s}{q['tax']:.2f}** |\n"
        f"| **GRAND TOTAL** | **{s}{q['total']:.2f}** |\n\n"
        f"### Terms\n"
        f"- **Payment Terms:** {q['payment_terms']}\n"
        f"- **T&C:** {q['terms_and_conditions']}\n"
        + (f"- **Notes:** {q['notes']}\n" if q["notes"] else "")
        + f"\n---\n**Extraction complete — {len(q['items'])} line items, grand total {s}{q['total']:.2f} {q['currency']}**"
    )


# ---------------------------------------------------------------------------
# 3. Quote extraction examples
# ---------------------------------------------------------------------------

def generate_quote_extraction_examples() -> List[Dict[str, Any]]:
    """Generate 500 quotation extraction SFT examples."""
    examples = []
    query_templates = [
        "Extract all fields from this quotation document:\n\n{doc}",
        "Parse this quote and identify every data element:\n\n{doc}",
        "Perform complete extraction of this quotation including all line items:\n\n{doc}",
        "Identify and extract all information from this quote:\n\n{doc}",
        "This is a quotation/quote document. Extract every field:\n\n{doc}",
    ]
    r = _rnd()
    for i in range(500):
        q = _build_quote(r)
        doc_text = _render_quote_text(q)
        tmpl = r.choice(query_templates)
        query = tmpl.format(doc=doc_text)

        s = q["symbol"]
        items_summary = ", ".join(f'"{i["description"]}"' for i in q["items"])
        reasoning = (
            f"Systematically extracting all elements from this quotation.\n\n"
            f"Step 1 — Document type: 'Quotation #' header confirms this is a quote document.\n\n"
            f"Step 2 — Quote identifiers:\n"
            f"  - Quote number: {q['quote_num']}\n"
            f"  - Quote date: {q['quote_date']}\n"
            f"  - Valid until: {q['expiry_date']} ({q['validity_days']} days)\n\n"
            f"Step 3 — Vendor (quoting party): {q['vendor_name']}, {q['vendor_city']}\n"
            f"  Phone: {q['vendor_phone']}, Email: {q['vendor_email']}\n\n"
            f"Step 4 — Customer (recipient): {q['customer_name']}, {q['customer_city']}\n\n"
            f"Step 5 — Line items ({len(q['items'])} items): {items_summary}\n\n"
            f"Step 6 — Financial summary:\n"
            f"  - Subtotal: {s}{q['subtotal']:.2f}\n"
            + (f"  - Discount ({q['discount_pct']}%): -{s}{q['discount']:.2f}\n" if q["discount"] > 0 else "")
            + f"  - Tax: {s}{q['tax']:.2f}\n"
            f"  - Grand total: {s}{q['total']:.2f}\n\n"
            f"Step 7 — Terms and conditions identified.\n\n"
            f"Step 8 — Validation: amounts cross-check ✓"
        )
        answer = _quote_answer(q)
        difficulty = r.choice(["easy", "medium", "medium", "hard"])
        examples.append(_make_sft(
            query, reasoning, answer,
            area="document_extraction",
            difficulty=difficulty,
            category="quote",
        ))

    return examples


# ---------------------------------------------------------------------------
# 4. Excel table extraction examples
# ---------------------------------------------------------------------------

def generate_excel_table_extraction_examples() -> List[Dict[str, Any]]:
    """Generate 300 Excel/tabular data extraction SFT examples."""
    examples = []
    r = _rnd()

    table_schemas = [
        {
            "name": "Sales Report",
            "columns": ["Product", "Region", "Q1", "Q2", "Q3", "Q4", "Annual Total"],
            "data_type": "financial",
        },
        {
            "name": "Supplier Quotation",
            "columns": ["Quotation", "Unnamed: 1", "Unnamed: 2", "Unnamed: 3", "Unit Price", "Quantity", "Total", "Unnamed: 7"],
            "data_type": "procurement",
        },
        {
            "name": "Employee Directory",
            "columns": ["Employee ID", "Name", "Department", "Role", "Salary", "Start Date", "Location"],
            "data_type": "hr",
        },
        {
            "name": "Asset Register",
            "columns": ["Asset ID", "Description", "Category", "Purchase Date", "Cost", "Depreciation", "Net Book Value"],
            "data_type": "financial",
        },
        {
            "name": "Project Tracker",
            "columns": ["Task ID", "Task Name", "Assignee", "Start", "End", "Status", "% Complete", "Notes"],
            "data_type": "operations",
        },
        {
            "name": "Inventory",
            "columns": ["SKU", "Description", "Category", "Qty On Hand", "Reorder Level", "Unit Cost", "Total Value"],
            "data_type": "warehouse",
        },
        {
            "name": "Budget vs Actual",
            "columns": ["Department", "Budget", "Actual", "Variance", "Variance %", "Status"],
            "data_type": "financial",
        },
    ]

    product_names = [
        "SupplyX Ltd", "Dell Workspace Solutions Ltd", "Meridian Tech",
        "Apex Consulting", "BlueSky Logistics", "Northgate Financial",
    ]

    query_templates = [
        "Extract all structured data from this Excel spreadsheet:\n\n{doc}",
        "Parse this tabular data and extract all rows and columns:\n\n{doc}",
        "Identify the table structure and extract all data from this spreadsheet:\n\n{doc}",
        "Extract the complete table including headers, data rows, and any totals:\n\n{doc}",
        "This is spreadsheet data. Identify columns, extract all rows, and summarise:\n\n{doc}",
    ]

    for i in range(300):
        schema = r.choice(table_schemas)
        cols = schema["columns"]
        n_rows = r.randint(4, 12)

        # Build fake table data
        rows = []
        for row_idx in range(n_rows):
            row = []
            for col in cols:
                if "Unnamed" in col:
                    row.append("")
                elif col in ["Q1", "Q2", "Q3", "Q4", "Budget", "Actual"]:
                    row.append(f"{r.uniform(5000, 250000):.2f}")
                elif col in ["Annual Total", "Total Value", "Net Book Value", "Total"]:
                    row.append(f"{r.uniform(20000, 1000000):.2f}")
                elif col in ["Unit Cost", "Unit Price", "Cost", "Salary"]:
                    row.append(f"{r.uniform(50, 5000):.2f}")
                elif col in ["Qty On Hand", "Quantity"]:
                    row.append(str(r.randint(1, 500)))
                elif col in ["% Complete"]:
                    row.append(f"{r.randint(0, 100)}%")
                elif col in ["Status"]:
                    row.append(r.choice(["On Track", "At Risk", "Complete", "Over Budget"]))
                elif col in ["Department"]:
                    row.append(r.choice(["Finance", "Operations", "HR", "IT", "Sales", "Legal"]))
                elif col in ["Category"]:
                    row.append(r.choice(["Hardware", "Software", "Services", "Facilities", "Equipment"]))
                elif col in ["Region", "Location"]:
                    row.append(r.choice(["London", "Manchester", "Birmingham", "Edinburgh", "Bristol"]))
                elif col in ["Product", "Description", "Task Name"]:
                    row.append(r.choice(_PRODUCT_SERVICES)[0][:30])
                elif col in ["Name", "Assignee"]:
                    row.append(r.choice(["James Smith", "Sarah Davies", "Michael Brown", "Emma Wilson", "Robert Taylor"]))
                elif col in ["Start Date", "Start", "Purchase Date"]:
                    row.append(f"{r.randint(1, 28):02d}/{r.randint(1, 12):02d}/{r.randint(2022, 2025)}")
                elif col in ["End"]:
                    row.append(f"{r.randint(1, 28):02d}/{r.randint(1, 12):02d}/{r.randint(2025, 2026)}")
                elif col in ["Notes"]:
                    row.append(r.choice(["", "Awaiting approval", "On hold", "Reviewed"]))
                else:
                    row.append(f"{col[:3].upper()}-{r.randint(1000, 9999)}")
            rows.append(row)

        # Build doc text
        header_row = " | ".join(cols)
        separator = " | ".join(["---"] * len(cols))
        data_rows = "\n".join(" | ".join(str(v) for v in row) for row in rows)

        # Identify unnamed columns
        unnamed_cols = [c for c in cols if "Unnamed" in c]
        named_cols = [c for c in cols if "Unnamed" not in c]

        header_line = f"Sheet: {schema['name']}"
        if r.random() < 0.3:
            company = r.choice(product_names)
            header_line = f"{company}\n{schema['name']}"

        doc_text = f"{header_line}\n\nColumns: {' | '.join(cols)}\n\n{header_row}\n{separator}\n{data_rows}"

        query = r.choice(query_templates).format(doc=doc_text)

        reasoning = (
            f"Analysing this Excel/tabular document systematically.\n\n"
            f"Step 1 — Sheet identification: Sheet name is '{schema['name']}'.\n\n"
            f"Step 2 — Column analysis: {len(cols)} columns detected.\n"
            f"  Named columns: {', '.join(named_cols)}\n"
            + (f"  Unnamed/empty columns: {', '.join(unnamed_cols)} — these are spacer or merged-cell artifacts\n" if unnamed_cols else "")
            + f"\nStep 3 — Data rows: {n_rows} data rows identified.\n\n"
            f"Step 4 — Data type classification: {schema['data_type']} data.\n\n"
            f"Step 5 — Header row is row 1. Data starts at row 2.\n\n"
            f"Step 6 — Checking for total/summary rows: inspecting last rows for TOTAL or SUM patterns.\n\n"
            f"Step 7 — Extracting all {n_rows} rows with all column values."
        )

        # Build answer
        col_desc = "\n".join(
            f"- **{c}**: {'Empty/spacer column (merged cell artifact)' if 'Unnamed' in c else 'Data column'}"
            for c in cols
        )

        answer_table = "| " + " | ".join(cols) + " |\n"
        answer_table += "| " + " | ".join(["---"] * len(cols)) + " |\n"
        for row in rows:
            answer_table += "| " + " | ".join(str(v) for v in row) + " |\n"

        answer = (
            f"## Excel Table Extraction — Complete Analysis\n\n"
            f"### Sheet Information\n"
            f"- **Sheet Name:** {schema['name']}\n"
            f"- **Total Columns:** {len(cols)} ({len(named_cols)} named, {len(unnamed_cols)} unnamed/spacer)\n"
            f"- **Data Rows:** {n_rows}\n"
            f"- **Data Domain:** {schema['data_type'].title()}\n\n"
            f"### Column Structure\n"
            f"{col_desc}\n\n"
            f"### Extracted Data\n"
            f"{answer_table}\n"
            f"### Summary\n"
            f"- **Rows extracted:** {n_rows}\n"
            f"- **Columns extracted:** {len(named_cols)} (excluding {len(unnamed_cols)} unnamed spacers)\n"
            + (f"- **Note:** Unnamed columns ({', '.join(unnamed_cols)}) are empty spacer columns — common in exported Excel files with merged header cells.\n" if unnamed_cols else "")
            + f"\n---\n**Extraction complete — {n_rows} rows × {len(cols)} columns ({schema['name']})**"
        )

        difficulty = r.choice(["easy", "medium", "medium", "hard"])
        examples.append(_make_sft(
            query, reasoning, answer,
            area="document_extraction",
            difficulty=difficulty,
            category="excel_table",
        ))

    return examples


# ---------------------------------------------------------------------------
# 5. Document structure examples
# ---------------------------------------------------------------------------

def generate_document_structure_examples() -> List[Dict[str, Any]]:
    """Generate 300 document structure identification SFT examples."""
    examples = []
    r = _rnd()

    doc_templates = [
        {
            "type": "contract",
            "elements": ["title", "parties", "recitals", "clauses", "signature_block", "schedules"],
        },
        {
            "type": "report",
            "elements": ["title", "header", "executive_summary", "sections", "tables", "footer", "page_numbers"],
        },
        {
            "type": "letter",
            "elements": ["letterhead", "date", "addressee_block", "salutation", "body_paragraphs", "closing", "signature"],
        },
        {
            "type": "form",
            "elements": ["title", "form_fields", "instructions", "tables", "signature_block", "date_field"],
        },
        {
            "type": "invoice",
            "elements": ["header", "vendor_block", "customer_block", "line_items_table", "financial_summary", "payment_terms", "footer"],
        },
        {
            "type": "policy",
            "elements": ["title", "scope", "definitions", "numbered_sections", "tables", "revision_history", "approval_block"],
        },
    ]

    structural_fragments = {
        "letterhead": [
            "URBEDGE FACILITIES MANAGEMENT LTD\n25 Kingsway Business Centre, London WC2B 6XF\nTel: +44 (0)20 7812 4456",
            "DELL WORKSPACE SOLUTIONS LTD\n7642 Mariner Drive Suite 300, Tampa FL 33609\nTel: (813) 555-9210",
        ],
        "watermark": ["DRAFT", "CONFIDENTIAL", "COPY", "SAMPLE"],
        "page_footer": [
            "Page 1 of 3 | Confidential | © 2025 DocWain Ltd",
            "INTERNAL USE ONLY | Page 2 | Generated: 01/04/2025",
        ],
        "signature_block": [
            "Signed: ________________________   Date: ___________\nName: _______________________   Title: ___________",
            "Authorised Signatory: ___________________\nFor and on behalf of: ___________________",
        ],
    }

    query_templates = [
        "Identify all structural elements in this document:\n\n{doc}",
        "Analyse the document structure and identify every component:\n\n{doc}",
        "Map out the structure of this document, identifying each element type:\n\n{doc}",
        "What structural elements are present in this document? Identify all of them:\n\n{doc}",
        "Perform document structure analysis — identify headers, tables, address blocks, signatures, footers:\n\n{doc}",
    ]

    for i in range(300):
        tmpl = r.choice(doc_templates)
        doc_type = tmpl["type"]
        elements_present = tmpl["elements"]

        # Build a synthetic document fragment
        doc_parts = []
        element_map = {}

        if "title" in elements_present:
            title = r.choice([
                f"SERVICE AGREEMENT — {r.randint(2024, 2026)}",
                f"FACILITIES MANAGEMENT CONTRACT",
                f"QUARTERLY PERFORMANCE REPORT — Q{r.randint(1,4)} {r.randint(2024, 2026)}",
                f"PURCHASE ORDER TERMS AND CONDITIONS",
                f"EMPLOYEE HANDBOOK — SECTION 4",
            ])
            doc_parts.append(f"{'=' * 60}\n{title}\n{'=' * 60}")
            element_map["title"] = title

        if "header" in elements_present:
            company = r.choice(_UK_COMPANIES + _US_COMPANIES)
            addr = r.choice(_UK_ADDRESSES + _US_ADDRESSES)
            header_text = f"{company}\n{addr[0]}, {addr[1]}, {addr[2]}"
            if r.random() > 0.5:
                header_text += f"\nTel: {r.choice(_UK_PHONES + _US_PHONES)}"
            doc_parts.append(header_text)
            element_map["header"] = "Company letterhead/header block"

        if "sections" in elements_present or "clauses" in elements_present:
            n_sections = r.randint(2, 5)
            section_titles = r.sample([
                "1. Definitions and Interpretation",
                "2. Scope of Services",
                "3. Payment Terms",
                "4. Liability and Indemnity",
                "5. Termination",
                "6. Confidentiality",
                "7. Data Protection",
                "8. Dispute Resolution",
            ], min(n_sections, 5))
            for st in section_titles:
                doc_parts.append(f"\n{st}\n" + "Lorem ipsum clause text for this section. " * r.randint(1, 3))
            element_map["sections"] = f"{n_sections} numbered sections"

        if "tables" in elements_present:
            doc_parts.append(
                "\n| Service | Frequency | Price |\n|---|---|---|\n"
                "| Office Cleaning | Weekly | £850.00 |\n"
                "| HVAC Service | Quarterly | £1,200.00 |\n"
                "| Waste Management | Monthly | £320.00 |"
            )
            element_map["table"] = "3-row service pricing table"

        if "signature_block" in elements_present:
            sig = r.choice(structural_fragments["signature_block"])
            doc_parts.append(f"\nSIGNATURES:\n{sig}")
            element_map["signature_block"] = "Signature and date fields"

        if "footer" in elements_present or "page_numbers" in elements_present:
            footer = r.choice(structural_fragments["page_footer"])
            doc_parts.append(f"\n---\n{footer}")
            element_map["footer"] = "Page footer with page number"

        # Optionally add watermark indicator
        has_watermark = r.random() < 0.3
        if has_watermark:
            wm = r.choice(structural_fragments["watermark"])
            doc_parts.append(f"\n[WATERMARK: {wm}]")
            element_map["watermark"] = f"'{wm}' watermark — ignore, not document content"

        doc_text = "\n\n".join(doc_parts)
        query = r.choice(query_templates).format(doc=doc_text)

        reasoning = (
            f"Performing structural analysis on this {doc_type} document.\n\n"
            f"Step 1 — Document type identification: structure matches a '{doc_type}' document.\n\n"
            f"Step 2 — Scanning for structural elements:\n"
            + "\n".join(f"  - Found: {k} → {v}" for k, v in element_map.items())
            + f"\n\nStep 3 — Table detection: {'table found with rows and columns' if 'table' in element_map else 'no tables detected'}.\n\n"
            f"Step 4 — Address/contact blocks: checking for company name + address + phone/email patterns.\n\n"
            f"Step 5 — Watermark/stamps: {'watermark detected — this is document metadata, not content' if has_watermark else 'no watermarks detected'}.\n\n"
            f"Step 6 — Page structure: checking for page numbers, headers, footers.\n\n"
            f"Step 7 — Signature blocks: {'found' if 'signature_block' in element_map else 'not present'}."
        )

        elements_list = "\n".join(f"- **{k.replace('_', ' ').title()}**: {v}" for k, v in element_map.items())

        answer = (
            f"## Document Structure Analysis\n\n"
            f"### Document Type\n"
            f"**{doc_type.replace('_', ' ').title()}**\n\n"
            f"### Identified Structural Elements\n"
            f"{elements_list}\n\n"
            f"### Element Count Summary\n"
            f"| Element Type | Present | Notes |\n|---|---|---|\n"
            + "\n".join(
                f"| {k.replace('_', ' ').title()} | Yes | {v} |"
                for k, v in element_map.items()
            )
            + "\n\n"
            f"### Structural Notes\n"
            + (f"- **Watermark detected:** '{element_map.get('watermark', '')}' — treat as document metadata, NOT content\n" if has_watermark else "")
            + f"- **Total structural elements identified:** {len(element_map)}\n"
            f"- **Document complexity:** {'High' if len(element_map) >= 5 else 'Medium' if len(element_map) >= 3 else 'Low'}\n\n"
            f"---\n**Structure analysis complete — {len(element_map)} elements identified in {doc_type} document**"
        )

        difficulty = r.choice(["easy", "medium", "medium", "hard"])
        examples.append(_make_sft(
            query, reasoning, answer,
            area="document_structure",
            difficulty=difficulty,
            category=f"structure_{doc_type}",
        ))

    return examples


# ---------------------------------------------------------------------------
# 6. Cross-field validation examples
# ---------------------------------------------------------------------------

def generate_cross_field_validation_examples() -> List[Dict[str, Any]]:
    """Generate 200 cross-field validation SFT examples."""
    examples = []
    r = _rnd()

    query_templates = [
        "Validate the internal consistency of this document:\n\n{doc}",
        "Check all calculations in this document for accuracy:\n\n{doc}",
        "Perform cross-field validation on this invoice/PO document:\n\n{doc}",
        "Verify the arithmetic and logical consistency of this document:\n\n{doc}",
        "Identify any errors or inconsistencies in this document:\n\n{doc}",
    ]

    for i in range(200):
        # Randomly pick validation scenario
        scenario = r.choice(["all_correct", "line_item_error", "subtotal_error", "tax_error", "date_error", "total_error"])

        currency_code = r.choice(list(_CURRENCIES.keys()))
        symbol, _ = _CURRENCIES[currency_code]

        n_items = r.randint(2, 5)
        items = []
        for _ in range(n_items):
            qty = r.randint(1, 20)
            unit_price = round(r.uniform(50, 2000), 2)
            correct_amount = round(qty * unit_price, 2)

            # Inject error in line item if this scenario
            if scenario == "line_item_error" and len(items) == 1:
                stated_amount = round(correct_amount * r.choice([0.9, 1.1, 1.05, 0.95]), 2)
            else:
                stated_amount = correct_amount

            items.append({
                "description": r.choice(_PRODUCT_SERVICES)[0][:30],
                "qty": qty,
                "unit_price": unit_price,
                "correct_amount": correct_amount,
                "stated_amount": stated_amount,
            })

        correct_subtotal = round(sum(i["correct_amount"] for i in items), 2)
        stated_subtotal = round(sum(i["stated_amount"] for i in items), 2)
        if scenario == "subtotal_error":
            stated_subtotal = round(stated_subtotal * r.choice([0.95, 1.05]), 2)

        tax_rate = _VAT_RATES[currency_code]
        correct_tax = round(correct_subtotal * tax_rate, 2)
        stated_tax = correct_tax
        if scenario == "tax_error":
            stated_tax = round(stated_tax * r.choice([0.8, 1.2]), 2)

        correct_total = round(correct_subtotal + correct_tax, 2)
        stated_total = round(stated_subtotal + stated_tax, 2)
        if scenario == "total_error":
            stated_total = round(stated_total + r.choice([-50, 50, 100, -100, 25.50]), 2)

        # Date validation
        year = r.randint(2024, 2026)
        month = r.randint(1, 11)
        day = r.randint(1, 20)
        inv_date = f"{day:02d}/{month:02d}/{year}"

        if scenario == "date_error":
            # Due date BEFORE invoice date
            due_day = day - r.randint(1, 10)
            due_date = f"{due_day:02d}/{month:02d}/{year}"
        else:
            due_day = day + r.randint(14, 60)
            due_month = month
            if due_day > 28:
                due_day -= 28
                due_month += 1
            due_date = f"{due_day:02d}/{due_month:02d}/{year}"

        inv_num = f"INV-{year}-{r.randint(10000, 99999)}"
        po_ref = f"PO{r.randint(1000000, 9999999)}"

        # Build document text
        lines = [
            f"INVOICE: {inv_num}",
            f"DATE: {inv_date}  DUE: {due_date}",
            f"PO REF: {po_ref}",
            f"",
        ]
        for item in items:
            lines.append(f"  {item['description']} | Qty: {item['qty']} x {symbol}{item['unit_price']:.2f} = {symbol}{item['stated_amount']:.2f}")
        lines += [
            f"",
            f"SUBTOTAL: {symbol}{stated_subtotal:.2f}",
            f"TAX: {symbol}{stated_tax:.2f}",
            f"TOTAL: {symbol}{stated_total:.2f}",
        ]
        doc_text = "\n".join(lines)

        # Determine errors found
        errors = []
        line_item_errors = []
        for j, item in enumerate(items):
            if abs(item["stated_amount"] - item["correct_amount"]) > 0.01:
                line_item_errors.append(
                    f"Line {j+1} '{item['description']}': "
                    f"stated {symbol}{item['stated_amount']:.2f} but "
                    f"{item['qty']} × {symbol}{item['unit_price']:.2f} = {symbol}{item['correct_amount']:.2f}"
                )

        if line_item_errors:
            errors.extend(line_item_errors)

        if abs(stated_subtotal - correct_subtotal) > 0.01:
            errors.append(
                f"Subtotal: stated {symbol}{stated_subtotal:.2f} but sum of line items = {symbol}{correct_subtotal:.2f}"
            )

        if abs(stated_tax - correct_tax) > 0.01 and tax_rate > 0:
            errors.append(
                f"Tax: stated {symbol}{stated_tax:.2f} but {tax_rate*100:.0f}% of {symbol}{correct_subtotal:.2f} = {symbol}{correct_tax:.2f}"
            )

        if abs(stated_total - correct_total) > 0.01:
            errors.append(
                f"Total: stated {symbol}{stated_total:.2f} but subtotal + tax = {symbol}{correct_total:.2f}"
            )

        if scenario == "date_error":
            errors.append(
                f"Date logic: invoice date {inv_date} is AFTER due date {due_date} — impossible"
            )

        query = r.choice(query_templates).format(doc=doc_text)

        def _item_check(item: Dict[str, Any], sym: str) -> str:
            if abs(item["stated_amount"] - item["correct_amount"]) < 0.01:
                return "✓"
            return f"✗ (stated {sym}{item['stated_amount']:.2f})"

        reasoning = (
            f"Performing cross-field validation on this document.\n\n"
            f"Step 1 — Line item check: verify each row: qty × unit_price = amount\n"
            + "\n".join(
                f"  Line {j+1}: {item['qty']} × {symbol}{item['unit_price']:.2f} = {symbol}{item['correct_amount']:.2f} "
                + _item_check(item, symbol)
                for j, item in enumerate(items)
            )
            + f"\n\nStep 2 — Subtotal check: sum of correct amounts = {symbol}{correct_subtotal:.2f}, stated = {symbol}{stated_subtotal:.2f} "
            f"{'✓' if abs(stated_subtotal - correct_subtotal) < 0.01 else '✗'}\n\n"
            f"Step 3 — Tax check: {tax_rate*100:.0f}% of {symbol}{correct_subtotal:.2f} = {symbol}{correct_tax:.2f}, stated = {symbol}{stated_tax:.2f} "
            f"{'✓' if abs(stated_tax - correct_tax) < 0.01 else '✗'}\n\n"
            f"Step 4 — Total check: {symbol}{stated_subtotal:.2f} + {symbol}{stated_tax:.2f} should equal stated total {symbol}{stated_total:.2f} "
            f"{'✓' if abs(stated_total - (stated_subtotal + stated_tax)) < 0.01 else '✗'}\n\n"
            f"Step 5 — Date logic: invoice date {inv_date} vs due date {due_date} — "
            f"{'due date is after invoice date ✓' if scenario != 'date_error' else 'ERROR: due date is before invoice date ✗'}"
        )

        if errors:
            error_list = "\n".join(f"- **ERROR:** {e}" for e in errors)
            verdict = "FAIL"
            verdict_note = f"**{len(errors)} validation error(s) found.**"
        else:
            error_list = "- No errors detected"
            verdict = "PASS"
            verdict_note = "**All calculations validated — document is internally consistent.**"

        answer = (
            f"## Cross-Field Validation Report\n\n"
            f"### Document: {inv_num} | Date: {inv_date} | Due: {due_date}\n\n"
            f"### Line Item Validation\n"
            f"| # | Description | Qty | Unit Price | Stated Amount | Correct Amount | Status |\n"
            f"|---|---|---|---|---|---|---|\n"
            + "\n".join(
                f"| {j+1} | {item['description'][:25]} | {item['qty']} | {symbol}{item['unit_price']:.2f} | "
                f"{symbol}{item['stated_amount']:.2f} | {symbol}{item['correct_amount']:.2f} | "
                f"{'✓ OK' if abs(item['stated_amount'] - item['correct_amount']) < 0.01 else '✗ ERROR'} |"
                for j, item in enumerate(items)
            )
            + f"\n\n### Financial Summary Validation\n"
            f"| Field | Stated | Correct | Status |\n|---|---|---|---|\n"
            f"| **Subtotal** | {symbol}{stated_subtotal:.2f} | {symbol}{correct_subtotal:.2f} | "
            f"{'✓ OK' if abs(stated_subtotal - correct_subtotal) < 0.01 else '✗ ERROR'} |\n"
            f"| **Tax** | {symbol}{stated_tax:.2f} | {symbol}{correct_tax:.2f} | "
            f"{'✓ OK' if abs(stated_tax - correct_tax) < 0.01 else '✗ ERROR'} |\n"
            f"| **Total** | {symbol}{stated_total:.2f} | {symbol}{correct_total:.2f} | "
            f"{'✓ OK' if abs(stated_total - correct_total) < 0.01 else '✗ ERROR'} |\n\n"
            f"### Date Logic Validation\n"
            f"- Invoice Date: **{inv_date}**\n"
            f"- Due Date: **{due_date}**\n"
            f"- Status: {'✓ Due date is after invoice date' if scenario != 'date_error' else '✗ ERROR: Due date precedes invoice date'}\n\n"
            f"### Validation Issues\n"
            f"{error_list}\n\n"
            f"### Overall Result: **{verdict}**\n"
            f"{verdict_note}"
        )

        difficulty = "hard" if errors else r.choice(["easy", "medium"])
        examples.append(_make_sft(
            query, reasoning, answer,
            area="document_validation",
            difficulty=difficulty,
            category="cross_field_validation",
        ))

    return examples


# ---------------------------------------------------------------------------
# 7. DPO preference pairs
# ---------------------------------------------------------------------------

def generate_element_dpo_pairs() -> List[Dict[str, Any]]:
    """Generate 500 DPO preference pairs for document element extraction."""
    examples = []
    r = _rnd()

    # We build pairs using invoices (most structured) and quotes/POs
    for i in range(500):
        doc_kind = r.choice(["invoice", "po", "quote"])

        if doc_kind == "invoice":
            inv = _build_invoice(r)
            doc_text = _render_invoice_text(inv)
            s = inv["symbol"]

            query = f"Extract all fields from this invoice document:\n\n{doc_text}"

            chosen_reasoning = _invoice_reasoning(inv)
            chosen_answer = _invoice_answer(inv)

            # Rejected: partial extraction, missing fields, no table, wrong amounts
            rejected_reasoning = (
                f"I'll extract the key details from this invoice.\n"
                f"The invoice number is {inv['inv_num']} and the total is {s}{inv['total']:.2f}."
            )
            # Pick a random subset of items to "miss"
            missed_item = r.choice(inv["items"])
            rejected_answer = (
                f"Invoice Number: {inv['inv_num']}\n"
                f"Date: {inv['inv_date']}\n"
                f"Vendor: {inv['vendor_name']}\n"
                f"Total: {s}{inv['total']:.2f}\n\n"
                f"Line items: {', '.join(i['description'] for i in inv['items'] if i != missed_item)}\n"
                f"(Note: some fields could not be extracted)"
            )

        elif doc_kind == "po":
            po = _build_po(r)
            doc_text = _render_po_text(po)
            s = po["symbol"]

            query = f"Extract all fields from this purchase order:\n\n{doc_text}"

            chosen_reasoning = (
                f"Systematically extracting all elements from this purchase order.\n\n"
                f"Step 1 — Document type confirmation: 'PURCHASE ORDER #' header present.\n"
                f"Step 2 — PO number: {po['po_num']}, Date: {po['po_date']}\n"
                f"Step 3 — Buyer: {po['buyer_name']}, {po['buyer_city']}\n"
                f"Step 4 — Vendor: {po['vendor_name']}, {po['vendor_city']}\n"
                f"Step 5 — {len(po['items'])} line items identified, each with part number, qty, unit price.\n"
                f"Step 6 — Total: {s}{po['total']:.2f}\n"
                f"Step 7 — Logistics: {po['shipping_method']}, {po['incoterms']}\n"
                f"Step 8 — Authorisation: {po['approver']}\n"
                f"Step 9 — Validation: sum of items matches total ✓"
            )
            chosen_answer = _po_answer(po)

            rejected_reasoning = "I can see this is a purchase order. Let me pull out the main details."
            rejected_answer = (
                f"PO Number: {po['po_num']}\n"
                f"Vendor: {po['vendor_name']}\n"
                f"Total: {s}{po['total']:.2f}\n"
                f"The PO contains several line items for various goods and services."
            )

        else:
            q = _build_quote(r)
            doc_text = _render_quote_text(q)
            s = q["symbol"]

            query = f"Extract all fields from this quotation:\n\n{doc_text}"

            chosen_reasoning = (
                f"Extracting all elements from this quotation.\n\n"
                f"Step 1 — Quote number: {q['quote_num']}, Date: {q['quote_date']}\n"
                f"Step 2 — Valid until: {q['expiry_date']} ({q['validity_days']} days validity)\n"
                f"Step 3 — Vendor: {q['vendor_name']}, {q['vendor_city']}\n"
                f"Step 4 — Customer: {q['customer_name']}, {q['customer_city']}\n"
                f"Step 5 — {len(q['items'])} line items with description, qty, unit price, total\n"
                f"Step 6 — Grand total: {s}{q['total']:.2f}\n"
                f"Step 7 — Terms and conditions extracted\n"
                f"Step 8 — All amounts cross-validated ✓"
            )
            chosen_answer = _quote_answer(q)

            rejected_reasoning = "Looking at this quote, I'll summarise the key points."
            rejected_answer = (
                f"This is a quotation ({q['quote_num']}) from {q['vendor_name']}.\n"
                f"Grand total: {s}{q['total']:.2f}\n"
                f"Valid until {q['expiry_date']}.\n"
                f"Contains {len(q['items'])} products/services."
            )

        difficulty = r.choice(["medium", "hard"])
        category = f"dpo_{doc_kind}"
        examples.append(_make_dpo(
            query,
            chosen_reasoning,
            chosen_answer,
            rejected_reasoning,
            rejected_answer,
            area="document_extraction",
            difficulty=difficulty,
            category=category,
        ))

    return examples


# ---------------------------------------------------------------------------
# 8. Orchestrator
# ---------------------------------------------------------------------------

def generate_all_element_examples(output_dir: str) -> Dict[str, Any]:
    """Generate all document element training examples and save to JSONL files.

    Args:
        output_dir: Directory to write output files into.

    Returns:
        Summary dict with counts and file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sft_path = out / "element_sft.jsonl"
    dpo_path = out / "element_dpo.jsonl"

    sft_counts: Dict[str, int] = {}
    dpo_counts: Dict[str, int] = {}

    generators = [
        ("invoice", generate_invoice_extraction_examples, 500),
        ("purchase_order", generate_po_extraction_examples, 500),
        ("quote", generate_quote_extraction_examples, 500),
        ("excel_table", generate_excel_table_extraction_examples, 300),
        ("document_structure", generate_document_structure_examples, 300),
        ("cross_field_validation", generate_cross_field_validation_examples, 200),
    ]

    with JSONLWriter(sft_path) as sft_writer:
        for name, fn, expected in generators:
            examples = fn()
            for ex in examples:
                sft_writer.write(ex)
            sft_counts[name] = len(examples)

    with JSONLWriter(dpo_path) as dpo_writer:
        dpo_examples = generate_element_dpo_pairs()
        for ex in dpo_examples:
            dpo_writer.write(ex)
        dpo_counts["element_dpo"] = len(dpo_examples)

    total_sft = sum(sft_counts.values())
    total_dpo = sum(dpo_counts.values())

    return {
        "sft_file": str(sft_path),
        "dpo_file": str(dpo_path),
        "sft_total": total_sft,
        "dpo_total": total_dpo,
        "sft_breakdown": sft_counts,
        "dpo_breakdown": dpo_counts,
        "total_examples": total_sft + total_dpo,
    }
