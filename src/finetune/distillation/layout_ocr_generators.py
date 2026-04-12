"""Layout-aware and OCR training data generators for DocWain.

Produces SFT and DPO examples that teach the model to understand document
layout, handle OCR artifacts, recognise patterns, and extract data with
precision.

Generator functions:
    generate_layout_aware_examples()      — 1,000 examples
    generate_ocr_handling_examples()      —   800 examples
    generate_pattern_recognition_examples() — 800 examples
    generate_accurate_extraction_examples() — 1,000 examples
    generate_context_awareness_examples() —   500 examples
    generate_layout_dpo_pairs()           —   500 DPO pairs
    generate_all_layout_ocr(output_dir)   — orchestrator
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

from src.finetune.v2.data_generator.base import (
    JSONLWriter,
    format_dpo_example,
    format_sft_example,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_SYSTEM = "You are DocWain."

_VENDORS = [
    ("Apex Solutions Ltd", "12 Park Lane", "London", "W1K 2AB", "United Kingdom",
     "+44 20 7946 0000", "accounts@apexsolutions.co.uk"),
    ("Brightfield Services Inc.", "400 Commerce Drive", "Austin", "TX 78701", "United States",
     "+1 512 555 0199", "billing@brightfieldservices.com"),
    ("Nordic Supply AS", "Storgata 14", "Oslo", "0155", "Norway",
     "+47 22 33 44 55", "invoices@nordicsupply.no"),
    ("Delta Procurement GmbH", "Hauptstraße 88", "Berlin", "10115", "Germany",
     "+49 30 123 456 78", "rechnungen@deltaprocurement.de"),
    ("Sunrise Trading Pte Ltd", "1 Marina Boulevard #28-00", "Singapore", "018989", "Singapore",
     "+65 6123 4567", "finance@sunrisetrading.sg"),
]

_CLIENTS = [
    ("GlobalTech Corp", "55 Finance Street", "Manchester", "M2 4AH", "United Kingdom"),
    ("Westbridge Capital", "1601 Elm St", "Dallas", "TX 75201", "United States"),
    ("Fjord Logistics AS", "Havneveien 22", "Bergen", "5003", "Norway"),
    ("Meridian Holdings GmbH", "Friedrichstraße 45", "Frankfurt", "60323", "Germany"),
    ("Pacific Rim Ventures", "50 Raffles Place #30-01", "Singapore", "048623", "Singapore"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sft(query: str, reasoning: str, answer: str, category: str) -> Dict[str, Any]:
    ex = format_sft_example(query, reasoning, answer, system_prompt=_SYSTEM)
    ex["category"] = category
    ex["source"] = "layout_ocr_generator"
    return ex


def _dpo(
    query: str,
    chosen_reasoning: str,
    chosen_answer: str,
    rejected_reasoning: str,
    rejected_answer: str,
    category: str,
) -> Dict[str, Any]:
    ex = format_dpo_example(
        query,
        chosen_reasoning,
        chosen_answer,
        rejected_reasoning,
        rejected_answer,
        system_prompt=_SYSTEM,
    )
    ex["category"] = category
    ex["source"] = "layout_ocr_generator"
    return ex


def _pick(lst):
    return random.choice(lst)


def _amount_uk(val: float) -> str:
    return f"£{val:,.2f}"


def _amount_us(val: float) -> str:
    return f"${val:,.2f}"


def _amount_eu(val: float) -> str:
    # EU format: dot as thousands sep, comma as decimal
    s = f"{val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"€{s}"


# ---------------------------------------------------------------------------
# 1. Layout-aware examples  (target: 1,000)
# ---------------------------------------------------------------------------

# We build named sub-templates and cycle through them to hit the target count.

def _multi_column_examples() -> List[Dict[str, Any]]:
    examples = []

    vendors = _VENDORS * 40
    clients = _CLIENTS * 40

    for i in range(200):
        v = vendors[i]
        c = clients[i % len(clients)]

        doc = (
            "=== INVOICE ===\n"
            "┌──────────────────────────────┬──────────────────────────────┐\n"
            "│  VENDOR                      │  BILL TO                     │\n"
            f"│  {v[0]:<28}│  {c[0]:<28}│\n"
            f"│  {v[1]:<28}│  {c[1]:<28}│\n"
            f"│  {v[2]}, {v[3]:<19}│  {c[2]}, {c[3]:<19}│\n"
            f"│  {v[4]:<28}│  {c[4]:<28}│\n"
            f"│  Tel: {v[5]:<22}│                              │\n"
            f"│  {v[6]:<28}│                              │\n"
            "└──────────────────────────────┴──────────────────────────────┘\n"
        )

        query = f"The following invoice uses a two-column layout. Identify the vendor details from the left column and the billing address from the right column.\n\n{doc}"
        reasoning = (
            "The document is formatted as a two-column table. "
            "The left column header is 'VENDOR' so all entries below it describe the supplier. "
            "The right column header is 'BILL TO' so its entries describe the client being invoiced. "
            "I will read each column independently and extract the structured data."
        )
        answer = (
            f"**Vendor (left column):**\n"
            f"- Company: {v[0]}\n"
            f"- Address: {v[1]}, {v[2]}, {v[3]}, {v[4]}\n"
            f"- Tel: {v[5]}\n"
            f"- Email: {v[6]}\n\n"
            f"**Bill To (right column):**\n"
            f"- Company: {c[0]}\n"
            f"- Address: {c[1]}, {c[2]}, {c[3]}, {c[4]}"
        )
        examples.append(_sft(query, reasoning, answer, "layout_multi_column"))

    return examples


def _table_detection_examples() -> List[Dict[str, Any]]:
    examples = []

    items_pool = [
        ("Consulting Services", 10, 150.00),
        ("Software Licence", 5, 800.00),
        ("Hardware Supply", 20, 45.50),
        ("Training Session", 3, 500.00),
        ("Maintenance Contract", 12, 200.00),
        ("Cloud Storage (TB)", 2, 350.00),
        ("On-site Support", 8, 175.00),
        ("Data Migration", 1, 2500.00),
        ("Security Audit", 1, 3200.00),
        ("Network Installation", 1, 4500.00),
    ]

    for i in range(200):
        n_items = random.randint(3, 6)
        chosen = random.sample(items_pool, n_items)
        subtotal = sum(q * u for _, q, u in chosen)
        vat = round(subtotal * 0.20, 2)
        total = subtotal + vat

        # pipe-delimited table
        header = "| Description           | Qty | Unit Price | Line Total |"
        sep    = "|:----------------------|----:|-----------:|-----------:|"
        rows = "\n".join(
            f"| {desc:<21} | {qty:>3} | {_amount_uk(unit):>10} | {_amount_uk(qty*unit):>10} |"
            for desc, qty, unit in chosen
        )
        footer = (
            f"| {'':21} |     |            |            |\n"
            f"| {'Subtotal':21} |     |            | {_amount_uk(subtotal):>10} |\n"
            f"| {'VAT @ 20%':21} |     |            | {_amount_uk(vat):>10} |\n"
            f"| {'**TOTAL**':21} |     |            | **{_amount_uk(total):>8}** |"
        )
        table_text = f"{header}\n{sep}\n{rows}\n{footer}"

        query = (
            "Extract all line items from the following pipe-delimited invoice table, "
            "then verify the totals.\n\n" + table_text
        )
        reasoning = (
            "The table uses pipe delimiters. The header row identifies four columns: "
            "Description, Qty, Unit Price, Line Total. "
            "I will read each data row, compute Qty × Unit Price for each line, "
            "then sum the line totals to verify the Subtotal. "
            "I will also verify VAT = Subtotal × 0.20 and Total = Subtotal + VAT."
        )
        item_lines = "\n".join(
            f"- {desc}: {qty} × {_amount_uk(unit)} = {_amount_uk(qty*unit)}"
            for desc, qty, unit in chosen
        )
        calc_ok = "Totals verified: all calculations are correct."
        answer = (
            f"**Line items extracted:**\n{item_lines}\n\n"
            f"**Subtotal:** {_amount_uk(subtotal)}\n"
            f"**VAT (20%):** {_amount_uk(vat)}\n"
            f"**Total:** {_amount_uk(total)}\n\n"
            f"{calc_ok}"
        )
        examples.append(_sft(query, reasoning, answer, "layout_table_detection"))

    return examples


def _form_field_examples() -> List[Dict[str, Any]]:
    examples = []

    form_templates = [
        (
            "SUPPLIER ONBOARDING FORM\n\nCompany Name: {company}\nRegistration No: {reg}\n"
            "VAT Number: {vat}\nPrimary Contact: {contact}\nEmail: {email}\nPhone: {phone}\n"
            "Payment Terms: {terms}\nBank Name: {bank}\nSort Code: {sort}\nAccount No: {acc}\n"
            "Approved: [x]    Date: {date}",
            {
                "company": "Apex Solutions Ltd", "reg": "GB12345678", "vat": "GB987654321",
                "contact": "Jane Smith", "email": "jsmith@apexsolutions.co.uk",
                "phone": "+44 20 7946 0000", "terms": "Net 30",
                "bank": "Barclays PLC", "sort": "20-00-00", "acc": "12345678",
                "date": "12/03/2025",
            },
        ),
        (
            "PURCHASE REQUEST FORM\n\nRequested By: {requester}\nDepartment: {dept}\n"
            "Date Required: {date_req}\nBudget Code: {budget}\nApproved By: {approver}\n"
            "Approval Date: {date_app}\nUrgent: [ ]    Standard: [x]\n"
            "Description of Goods/Services:\n{description}\nEstimated Value: {value}",
            {
                "requester": "Mark Johnson", "dept": "IT Operations",
                "date_req": "25/04/2025", "budget": "IT-2025-Q2",
                "approver": "Sarah Williams", "date_app": "18/04/2025",
                "description": "24-port managed network switches (x4) for office refresh",
                "value": "£4,800.00",
            },
        ),
    ]

    for i in range(200):
        tmpl, fields = _pick(form_templates)
        form_text = tmpl.format(**fields)

        query = f"Parse all form fields from the following document and return them as a structured JSON object.\n\n{form_text}"
        reasoning = (
            "The document contains 'Field: Value' pairs separated by newlines. "
            "I will scan each line for the pattern 'Label: Value', extract both parts, "
            "and assemble them into a JSON object. "
            "I will also note checkbox states ([x] = checked, [ ] = unchecked)."
        )
        extracted = {k: v for k, v in fields.items()}
        answer = (
            "**Extracted form fields:**\n\n"
            "```json\n"
            + json.dumps(extracted, indent=2)
            + "\n```"
        )
        examples.append(_sft(query, reasoning, answer, "layout_form_fields"))

    return examples


def _address_block_examples() -> List[Dict[str, Any]]:
    examples = []

    addresses = [
        {
            "label": "vendor",
            "block": (
                "Apex Solutions Ltd\n12 Park Lane\nLondon W1K 2AB\nUnited Kingdom\n"
                "Tel: +44 20 7946 0000\nFax: +44 20 7946 0001\nEmail: accounts@apexsolutions.co.uk"
            ),
            "parsed": {
                "company": "Apex Solutions Ltd",
                "street": "12 Park Lane",
                "city": "London",
                "postcode": "W1K 2AB",
                "country": "United Kingdom",
                "tel": "+44 20 7946 0000",
                "fax": "+44 20 7946 0001",
                "email": "accounts@apexsolutions.co.uk",
            },
        },
        {
            "label": "billing",
            "block": (
                "BILL TO:\nGlobalTech Corp\nAccounts Payable Department\n55 Finance Street\n"
                "Manchester M2 4AH\nUnited Kingdom\nRef: ACC-2025-0042"
            ),
            "parsed": {
                "company": "GlobalTech Corp",
                "department": "Accounts Payable Department",
                "street": "55 Finance Street",
                "city": "Manchester",
                "postcode": "M2 4AH",
                "country": "United Kingdom",
                "reference": "ACC-2025-0042",
            },
        },
        {
            "label": "shipping",
            "block": (
                "SHIP TO:\nGlobalTech Corp — Warehouse Dept\nUnit 7, Trafford Park\nManchester M17 1PZ\n"
                "United Kingdom\nAttn: Goods-In Team\nTel: +44 161 234 5678"
            ),
            "parsed": {
                "company": "GlobalTech Corp — Warehouse Dept",
                "street": "Unit 7, Trafford Park",
                "city": "Manchester",
                "postcode": "M17 1PZ",
                "country": "United Kingdom",
                "attention": "Goods-In Team",
                "tel": "+44 161 234 5678",
            },
        },
    ]

    for i in range(200):
        addr = _pick(addresses)
        query = (
            f"The following is an address block from a commercial document. "
            f"Parse it into individual components (company, street, city, postcode, country, "
            f"and contact details).\n\n{addr['block']}"
        )
        reasoning = (
            "Address blocks follow a standard postal format: company name on the first line, "
            "street address next, then city and postcode on the same line or consecutive lines, "
            "followed by country. Contact details (tel, fax, email) appear after the postal address. "
            "I will parse each element in sequence."
        )
        answer = (
            f"**Parsed {addr['label']} address:**\n\n"
            "```json\n"
            + json.dumps(addr["parsed"], indent=2)
            + "\n```"
        )
        examples.append(_sft(query, reasoning, answer, "layout_address_blocks"))

    return examples


def _header_footer_examples() -> List[Dict[str, Any]]:
    examples = []
    docs = [
        {
            "doc": (
                "ACME CORP — CONFIDENTIAL                                     Page 1 of 4\n"
                "Document: Contract for Services | Reference: CTR-2025-0891 | Version: 2.1\n"
                "─────────────────────────────────────────────────────────────────────\n"
                "1. PARTIES\nThis agreement is entered into between ACME Corp ('Client') and ...\n"
                "─────────────────────────────────────────────────────────────────────\n"
                "ACME CORP — CONFIDENTIAL                                     Page 1 of 4\n"
                "© 2025 ACME Corp. All rights reserved. | acmecorp.com | +44 20 1234 5678"
            ),
            "header": {
                "company": "ACME CORP",
                "classification": "CONFIDENTIAL",
                "pagination": "Page 1 of 4",
                "document": "Contract for Services",
                "reference": "CTR-2025-0891",
                "version": "2.1",
            },
            "footer": {
                "company": "ACME CORP",
                "classification": "CONFIDENTIAL",
                "pagination": "Page 1 of 4",
                "copyright": "© 2025 ACME Corp. All rights reserved.",
                "website": "acmecorp.com",
                "phone": "+44 20 1234 5678",
            },
            "main_content_start": "1. PARTIES",
        },
    ]

    for i in range(200):
        doc = _pick(docs)
        query = (
            "Identify and separate the header, footer, and main content from the following document page.\n\n"
            + doc["doc"]
        )
        reasoning = (
            "Headers appear at the top of pages before the main content, typically containing "
            "company name, document reference, and page numbers. "
            "Footers appear at the bottom, often repeating classification markings and adding "
            "copyright/contact information. "
            "The main content lies between the horizontal dividers. "
            "I will extract each region separately."
        )
        answer = (
            "**Header fields:**\n"
            + "\n".join(f"- {k}: {v}" for k, v in doc["header"].items())
            + "\n\n**Footer fields:**\n"
            + "\n".join(f"- {k}: {v}" for k, v in doc["footer"].items())
            + f"\n\n**Main content begins with:** \"{doc['main_content_start']}\""
        )
        examples.append(_sft(query, reasoning, answer, "layout_header_footer"))

    return examples


def generate_layout_aware_examples() -> List[Dict[str, Any]]:
    """Return 1,000 layout-aware SFT examples."""
    random.seed(42)
    examples: List[Dict[str, Any]] = []
    examples.extend(_multi_column_examples())    # 200
    examples.extend(_table_detection_examples()) # 200
    examples.extend(_form_field_examples())      # 200
    examples.extend(_address_block_examples())   # 200
    examples.extend(_header_footer_examples())   # 200
    random.shuffle(examples)
    return examples[:1000]


# ---------------------------------------------------------------------------
# 2. OCR handling examples  (target: 800)
# ---------------------------------------------------------------------------

def _common_ocr_error_examples() -> List[Dict[str, Any]]:
    examples = []

    # Each tuple: (corrupted_text, corrected_text, error_description)
    ocr_pairs = [
        ("Inv oice Number: lNV-2025-001", "Invoice Number: INV-2025-001",
         "'Inv oice' has an extra space (should be 'Invoice'); 'lNV' has lowercase 'l' mistaken for 'I'."),
        ("TotalAmount: £1O,234.56", "Total Amount: £10,234.56",
         "'TotalAmount' is missing a space; 'O' is confused with '0' making '1O,234.56' → '10,234.56'."),
        ("Due Date: 15/O3/2O25", "Due Date: 15/03/2025",
         "Both '/' separated segments have 'O' (letter) confused with '0' (digit)."),
        ("Quantity: l2", "Quantity: 12",
         "'l' (lowercase L) confused with '1' (digit)."),
        ("PO Number: PO-OO45-B", "PO Number: PO-0045-B",
         "Two instances of 'O' (letter) substituted for '0' (digit) in the numeric segment."),
        ("Net 3O Days", "Net 30 Days",
         "'3O' contains 'O' (letter) instead of '0' (digit)."),
        ("Unit Price: £ 45.5O", "Unit Price: £45.50",
         "Spurious space after '£'; final 'O' should be '0'."),
        ("Registered in Eng1and & Wa1es", "Registered in England & Wales",
         "Two '1' (digit) characters substituted for 'l' (lowercase L) in 'England' and 'Wales'."),
        ("VAT Reg: GB 123 4567 8O", "VAT Reg: GB 123 4567 80",
         "Final 'O' should be '0'."),
        ("lnvoice Date: O1/O4/2O25", "Invoice Date: 01/04/2025",
         "'l' mistaken for 'I' in 'Invoice'; three 'O' characters mistaken for '0'."),
        ("Â£1,234.56", "£1,234.56",
         "UTF-8 corruption of '£' symbol — 'Â£' is the mojibake encoding; should be '£'."),
        ("â‚¬2,500.00", "€2,500.00",
         "UTF-8 corruption of '€' symbol — 'â‚¬' is the mojibake encoding; should be '€'."),
        ("Inv\noice: 1234", "Invoice: 1234",
         "Word 'Invoice' broken across lines by OCR; should be joined."),
        ("Com\npany: Apex So\nlutions Ltd", "Company: Apex Solutions Ltd",
         "Two words broken mid-character across lines; both should be rejoined."),
    ]

    for i in range(200):
        pair = ocr_pairs[i % len(ocr_pairs)]
        corrupted, corrected, explanation = pair

        query = (
            f"The following text was extracted by an OCR engine and contains errors. "
            f"Identify and correct all OCR artifacts:\n\n\"{corrupted}\""
        )
        reasoning = (
            "I will scan the text for common OCR error patterns: "
            "letter/digit confusion (l/1/I, O/0), merged or broken words, "
            "missing spaces, extra spaces, and character encoding corruption. "
            f"Analysis: {explanation}"
        )
        answer = (
            f"**Corrected text:** \"{corrected}\"\n\n"
            f"**Errors found and fixed:**\n- {explanation}"
        )
        examples.append(_sft(query, reasoning, answer, "ocr_common_errors"))

    return examples


def _watermark_stamp_examples() -> List[Dict[str, Any]]:
    examples = []

    docs = [
        {
            "with_watermark": (
                "INVOICE\nD R A F T\nInvoice Number: INV-2025-0042\n"
                "D R A F T\nDate: 12/03/2025\nD R A F T\n"
                "Vendor: Apex Solutions Ltd\nAmount Due: £8,750.00\n"
                "D R A F T"
            ),
            "clean": (
                "INVOICE\nInvoice Number: INV-2025-0042\n"
                "Date: 12/03/2025\nVendor: Apex Solutions Ltd\nAmount Due: £8,750.00"
            ),
            "watermark": "DRAFT",
        },
        {
            "with_watermark": (
                "PURCHASE ORDER\nCONFIDENTIAL\nPO Number: PO-20250389\n"
                "CONFIDENTIAL\nIssued: 01/04/2025\nSupplier: Nordic Supply AS\n"
                "CONFIDENTIAL\nTotal: £22,400.00"
            ),
            "clean": (
                "PURCHASE ORDER\nPO Number: PO-20250389\n"
                "Issued: 01/04/2025\nSupplier: Nordic Supply AS\nTotal: £22,400.00"
            ),
            "watermark": "CONFIDENTIAL",
        },
        {
            "with_watermark": (
                "CONTRACT FOR SERVICES\nVOID\nParties: ACME Corp and Apex Solutions Ltd\n"
                "VOID\nEffective Date: 01/01/2025\nVOID\nTotal Value: £150,000.00"
            ),
            "clean": (
                "CONTRACT FOR SERVICES\nParties: ACME Corp and Apex Solutions Ltd\n"
                "Effective Date: 01/01/2025\nTotal Value: £150,000.00"
            ),
            "watermark": "VOID",
        },
    ]

    for i in range(200):
        doc = _pick(docs)
        query = (
            f"The following text was scanned from a document and the OCR has captured "
            f"watermark text mixed into the content. Extract only the actual document "
            f"content, ignoring the watermark.\n\n{doc['with_watermark']}"
        )
        reasoning = (
            f"The repeated text '{doc['watermark']}' appearing on its own lines is a watermark "
            "printed across the document. It is not part of the document content. "
            "I will filter out all occurrences of the watermark text and return only "
            "the genuine document fields."
        )
        answer = (
            f"**Watermark identified:** '{doc['watermark']}' (repeated {doc['with_watermark'].count(doc['watermark'])} times)\n\n"
            f"**Clean document content:**\n{doc['clean']}"
        )
        examples.append(_sft(query, reasoning, answer, "ocr_watermarks"))

    return examples


def _space_aligned_table_examples() -> List[Dict[str, Any]]:
    examples = []

    raw_tables = [
        {
            "text": (
                "ITEM          QTY    UNIT_PRICE    TOTAL\n"
                "Consulting     10      150.00     1500.00\n"
                "Licence         5      800.00     4000.00\n"
                "Hardware       20       45.50      910.00\n"
                "                              ----------\n"
                "                    SUBTOTAL    6410.00\n"
                "                    VAT 20%     1282.00\n"
                "                    TOTAL       7692.00"
            ),
            "items": [
                {"item": "Consulting", "qty": 10, "unit_price": 150.00, "total": 1500.00},
                {"item": "Licence", "qty": 5, "unit_price": 800.00, "total": 4000.00},
                {"item": "Hardware", "qty": 20, "unit_price": 45.50, "total": 910.00},
            ],
            "subtotal": 6410.00,
            "vat": 1282.00,
            "total": 7692.00,
        },
    ]

    for i in range(200):
        tbl = _pick(raw_tables)
        query = (
            "The following is space-aligned tabular text extracted from a scanned invoice. "
            "Parse the table into structured data and confirm the arithmetic.\n\n"
            + tbl["text"]
        )
        reasoning = (
            "The columns are aligned by whitespace. The header row defines the columns: "
            "ITEM, QTY, UNIT_PRICE, TOTAL. I will split each data row by whitespace, "
            "assign values to columns based on position, then verify: "
            "each line total = qty × unit_price, subtotal = sum of line totals, "
            "VAT = subtotal × 0.20, total = subtotal + VAT."
        )
        items_md = "\n".join(
            f"| {it['item']} | {it['qty']} | {_amount_uk(it['unit_price'])} | {_amount_uk(it['total'])} |"
            for it in tbl["items"]
        )
        checks = all(
            round(it["qty"] * it["unit_price"], 2) == it["total"]
            for it in tbl["items"]
        )
        vat_check = round(tbl["subtotal"] * 0.20, 2) == tbl["vat"]
        total_check = round(tbl["subtotal"] + tbl["vat"], 2) == tbl["total"]
        answer = (
            "**Parsed table:**\n\n"
            "| Item | Qty | Unit Price | Line Total |\n"
            "|:-----|----:|-----------:|-----------:|\n"
            + items_md + "\n\n"
            f"**Subtotal:** {_amount_uk(tbl['subtotal'])}\n"
            f"**VAT (20%):** {_amount_uk(tbl['vat'])}\n"
            f"**Total:** {_amount_uk(tbl['total'])}\n\n"
            f"**Arithmetic verification:**\n"
            f"- Line totals: {'PASS' if checks else 'FAIL'}\n"
            f"- VAT calculation: {'PASS' if vat_check else 'FAIL'}\n"
            f"- Grand total: {'PASS' if total_check else 'FAIL'}"
        )
        examples.append(_sft(query, reasoning, answer, "ocr_space_aligned_tables"))

    return examples


def _handwritten_annotation_examples() -> List[Dict[str, Any]]:
    examples = []

    docs = [
        {
            "text": (
                "Invoice Total: £4,500.00\n"
                "[handwritten: 'Agreed - process by 15th' - initials JB]\n"
                "Payment Terms: Net 30\n"
                "[handwritten: 'Expedite - Director approval']\n"
                "Approved: [x]"
            ),
            "printed_content": "Invoice Total: £4,500.00 | Payment Terms: Net 30 | Approved: Yes",
            "handwritten_notes": [
                "'Agreed - process by 15th' (initialled JB)",
                "'Expedite - Director approval'",
            ],
        },
        {
            "text": (
                "PO Total: £12,800.00\n"
                "[handwritten: 'Check pricing with supplier before approval']\n"
                "Delivery Date: 30/04/2025\n"
                "[handwritten: 'Urgent - needed for project kickoff']\n"
                "Budget Code: PROJ-2025-07"
            ),
            "printed_content": "PO Total: £12,800.00 | Delivery Date: 30/04/2025 | Budget Code: PROJ-2025-07",
            "handwritten_notes": [
                "'Check pricing with supplier before approval'",
                "'Urgent - needed for project kickoff'",
            ],
        },
    ]

    for i in range(200):
        doc = _pick(docs)
        query = (
            "The following document text contains both printed content and handwritten annotations "
            "captured by OCR. Separate the printed document content from the handwritten notes.\n\n"
            + doc["text"]
        )
        reasoning = (
            "Handwritten annotations are identified by the '[handwritten: ...]' markers introduced "
            "during OCR processing. I will extract the printed fields separately from the "
            "handwritten notes, clearly labelling each category."
        )
        notes_formatted = "\n".join(f"  {j+1}. {n}" for j, n in enumerate(doc["handwritten_notes"]))
        answer = (
            f"**Printed document content:**\n{doc['printed_content']}\n\n"
            f"**Handwritten annotations (flagged for review):**\n{notes_formatted}"
        )
        examples.append(_sft(query, reasoning, answer, "ocr_handwritten_annotations"))

    return examples


def generate_ocr_handling_examples() -> List[Dict[str, Any]]:
    """Return 800 OCR-handling SFT examples."""
    random.seed(43)
    examples: List[Dict[str, Any]] = []
    examples.extend(_common_ocr_error_examples())     # 200
    examples.extend(_watermark_stamp_examples())      # 200
    examples.extend(_space_aligned_table_examples())  # 200
    examples.extend(_handwritten_annotation_examples()) # 200
    random.shuffle(examples)
    return examples[:800]


# ---------------------------------------------------------------------------
# 3. Pattern recognition examples  (target: 800)
# ---------------------------------------------------------------------------

def _reference_number_examples() -> List[Dict[str, Any]]:
    examples = []

    patterns = [
        ("INV-2025-00042", "invoice", "INV-YYYY-NNNNN format", "INV-2025-00042"),
        ("#INV20250042", "invoice", "#INVYYYYNNNNN format", "#INV20250042"),
        ("PO-00389", "purchase_order", "PO-NNNNN format", "PO-00389"),
        ("Purchase Order #00389", "purchase_order", "Purchase Order #NNNNN format", "PO: 00389"),
        ("QTE-2025-00127", "quote", "QTE-YYYY-NNNNN format", "QTE-2025-00127"),
        ("Q-00127", "quote", "Q-NNNNN format", "Q-00127"),
        ("WSG100024", "custom", "alphanumeric custom format (e.g. customer-specific)", "WSG100024"),
        ("CTR-2025-0891", "contract", "CTR-YYYY-NNNN format", "CTR-2025-0891"),
        ("SO-2025-00512", "sales_order", "SO-YYYY-NNNNN format", "SO-2025-00512"),
        ("CR-00081", "credit_note", "CR-NNNNN format", "CR-00081"),
    ]

    for i in range(200):
        ref_str, doc_type, fmt_desc, normalised = _pick(patterns)
        query = (
            f"Identify the document type and extract the reference number from "
            f"the following identifier: '{ref_str}'"
        )
        reasoning = (
            f"The reference '{ref_str}' matches the {fmt_desc}. "
            f"Common prefixes: INV = Invoice, PO = Purchase Order, QTE/Q = Quote, "
            f"CTR = Contract, SO = Sales Order, CR = Credit Note. "
            f"Alphanumeric identifiers without a standard prefix are likely custom formats. "
            f"I will classify the document type and return the normalised reference."
        )
        answer = (
            f"**Document type:** {doc_type.replace('_', ' ').title()}\n"
            f"**Reference format:** {fmt_desc}\n"
            f"**Normalised reference:** {normalised}"
        )
        examples.append(_sft(query, reasoning, answer, "pattern_reference_numbers"))

    return examples


def _currency_amount_examples() -> List[Dict[str, Any]]:
    examples = []

    currency_cases = [
        ("£1,234.56", "GBP", 1234.56, "UK pound sterling with comma thousands separator and dot decimal"),
        ("GBP 1,234.56", "GBP", 1234.56, "ISO currency code GBP followed by amount"),
        ("$9,876.50", "USD", 9876.50, "US dollar with comma thousands separator and dot decimal"),
        ("USD 9,876.50", "USD", 9876.50, "ISO currency code USD followed by amount"),
        ("€5,000.00", "EUR", 5000.00, "Euro with dot thousands separator and dot decimal (ambiguous — context needed)"),
        ("€5.000,00", "EUR", 5000.00, "Euro with EU format: dot as thousands separator, comma as decimal"),
        ("EUR 5.000,00", "EUR", 5000.00, "ISO code EUR with EU numeric format"),
        ("1,450.00", "unknown", 1450.00, "No currency symbol — context required to determine currency"),
        ("SGD 12,500.00", "SGD", 12500.00, "Singapore dollar ISO code"),
        ("NOK 45.000,00", "NOK", 45000.00, "Norwegian krone with European numeric format"),
    ]

    for i in range(200):
        amount_str, currency, value, explanation = _pick(currency_cases)
        query = (
            f"Parse the following currency value and identify the currency, "
            f"the numeric value, and the number format used: '{amount_str}'"
        )
        reasoning = (
            f"Analysing '{amount_str}': {explanation}. "
            "Key disambiguation rules: if a dot appears in the thousands position AND a comma "
            "appears in the decimal position (e.g. 5.000,00) it is EU format. "
            "If comma is in thousands position and dot is decimal (e.g. 5,000.00) it is UK/US format. "
            "Currency symbol or ISO code identifies the currency."
        )
        if currency == "unknown":
            answer = (
                f"**Currency:** Unknown — no symbol or ISO code present; context required\n"
                f"**Numeric value:** {value:,.2f}\n"
                f"**Number format:** UK/US (comma thousands, dot decimal)\n"
                f"**Note:** {explanation}"
            )
        else:
            answer = (
                f"**Currency:** {currency}\n"
                f"**Numeric value:** {value:,.2f}\n"
                f"**Number format explanation:** {explanation}"
            )
        examples.append(_sft(query, reasoning, answer, "pattern_currency_amounts"))

    return examples


def _date_format_examples() -> List[Dict[str, Any]]:
    examples = []

    date_cases = [
        ("15/03/2025", "DD/MM/YYYY", "UK", "15 March 2025"),
        ("03/15/2025", "MM/DD/YYYY", "US", "15 March 2025"),
        ("2025-03-15", "ISO 8601 (YYYY-MM-DD)", "International", "15 March 2025"),
        ("15-03-2025", "DD-MM-YYYY", "UK/EU", "15 March 2025"),
        ("March 15, 2025", "Month DD, YYYY (long form)", "US/International", "15 March 2025"),
        ("15 March 2025", "DD Month YYYY (long form)", "UK", "15 March 2025"),
        ("15/03/25", "DD/MM/YY (short year)", "UK (ambiguous century)", "15 March 2025"),
        ("Q1 2025", "Quarterly", "Financial", "January–March 2025"),
        ("01/13/2025", "Invalid", "Invalid date", "Month 13 does not exist — likely OCR error or day/month transposition"),
        ("30/02/2025", "Invalid", "Invalid date", "February never has 30 days — data error"),
    ]

    for i in range(200):
        date_str, fmt, locale, interpretation = _pick(date_cases)
        query = (
            f"Identify the date format and parse the following date value: '{date_str}'. "
            f"The document originates from a UK company."
        )
        reasoning = (
            f"The date string is '{date_str}'. Format: {fmt}. "
            "For ambiguous formats like DD/MM/YYYY vs MM/DD/YYYY, I use document context: "
            "a UK company uses DD/MM/YYYY. "
            f"For invalid dates I will flag the error clearly."
        )
        if "Invalid" in fmt:
            answer = (
                f"**Date string:** {date_str}\n"
                f"**Status:** INVALID DATE\n"
                f"**Reason:** {interpretation}\n"
                f"**Action required:** Verify source document for OCR error or data entry mistake."
            )
        else:
            answer = (
                f"**Detected format:** {fmt}\n"
                f"**Locale/context:** {locale}\n"
                f"**Parsed date:** {interpretation}"
            )
        examples.append(_sft(query, reasoning, answer, "pattern_date_formats"))

    return examples


def _tax_payment_term_examples() -> List[Dict[str, Any]]:
    examples = []

    terms = [
        ("VAT @ 20%", "tax", "Value Added Tax at 20% rate (standard UK VAT rate)"),
        ("Sales Tax 8.25%", "tax", "US sales tax at 8.25% (common in Texas)"),
        ("Discount: 10%", "discount", "10% discount applied to subtotal before tax"),
        ("2/10 Net 30", "payment_terms",
         "2% discount if paid within 10 days; full amount due within 30 days"),
        ("Net 30", "payment_terms", "Full payment due within 30 calendar days of invoice date"),
        ("Net 60", "payment_terms", "Full payment due within 60 calendar days of invoice date"),
        ("Due on Receipt", "payment_terms", "Payment due immediately upon receipt of invoice"),
        ("EOM + 30", "payment_terms", "Payment due 30 days after the end of the invoice month"),
        ("COD", "payment_terms", "Cash on Delivery — payment due at time of delivery"),
        ("PIA", "payment_terms", "Payment in Advance — full payment required before goods are delivered"),
        ("GST 15%", "tax", "Goods and Services Tax at 15% (common in Australia/New Zealand)"),
        ("WHT 10%", "tax", "Withholding Tax at 10% — deducted at source"),
    ]

    for i in range(200):
        term_str, term_type, explanation = _pick(terms)
        query = (
            f"Identify the type and meaning of the following payment/tax term "
            f"found in a commercial document: '{term_str}'"
        )
        reasoning = (
            f"'{term_str}' is a standard commercial term. "
            f"Type: {term_type}. "
            f"I will explain its meaning in plain language and note any action it implies."
        )
        answer = (
            f"**Term:** {term_str}\n"
            f"**Type:** {term_type.replace('_', ' ').title()}\n"
            f"**Meaning:** {explanation}"
        )
        examples.append(_sft(query, reasoning, answer, "pattern_tax_payment_terms"))

    return examples


def generate_pattern_recognition_examples() -> List[Dict[str, Any]]:
    """Return 800 pattern recognition SFT examples."""
    random.seed(44)
    examples: List[Dict[str, Any]] = []
    examples.extend(_reference_number_examples())   # 200
    examples.extend(_currency_amount_examples())    # 200
    examples.extend(_date_format_examples())        # 200
    examples.extend(_tax_payment_term_examples())   # 200
    random.shuffle(examples)
    return examples[:800]


# ---------------------------------------------------------------------------
# 4. Accurate extraction examples  (target: 1,000)
# ---------------------------------------------------------------------------

def _full_invoice_extraction_examples() -> List[Dict[str, Any]]:
    examples = []

    for i in range(300):
        v = _VENDORS[i % len(_VENDORS)]
        c = _CLIENTS[i % len(_CLIENTS)]
        inv_num = f"INV-2025-{i+1:05d}"
        inv_date = f"{(i % 28) + 1:02d}/03/2025"
        due_date = f"{(i % 28) + 1:02d}/04/2025"
        po_ref = f"PO-{i+1:05d}"
        n_items = random.randint(2, 5)
        items_pool = [
            ("Consulting Services", random.randint(1, 20), 150.00),
            ("Software Licence", random.randint(1, 10), 800.00),
            ("Hardware Supply", random.randint(5, 50), 45.50),
            ("Training Session", random.randint(1, 5), 500.00),
            ("Maintenance Contract", random.randint(1, 12), 200.00),
        ]
        chosen = items_pool[:n_items]
        subtotal = round(sum(q * u for _, q, u in chosen), 2)
        vat = round(subtotal * 0.20, 2)
        total = round(subtotal + vat, 2)

        line_items_text = "\n".join(
            f"  {desc}  {qty} x {_amount_uk(unit)} = {_amount_uk(qty*unit)}"
            for desc, qty, unit in chosen
        )
        invoice_text = (
            f"INVOICE\n\nFrom: {v[0]}, {v[1]}, {v[2]} {v[3]}, {v[4]}\n"
            f"To: {c[0]}, {c[1]}, {c[2]} {c[3]}, {c[4]}\n"
            f"Invoice No: {inv_num}\nInvoice Date: {inv_date}\nDue Date: {due_date}\n"
            f"PO Reference: {po_ref}\nPayment Terms: Net 30\n\n"
            f"LINE ITEMS:\n{line_items_text}\n\n"
            f"Subtotal: {_amount_uk(subtotal)}\nVAT (20%): {_amount_uk(vat)}\n"
            f"TOTAL DUE: {_amount_uk(total)}"
        )

        query = f"Extract ALL fields from the following invoice into a structured JSON object.\n\n{invoice_text}"
        reasoning = (
            "I will parse the invoice systematically: header fields first "
            "(invoice number, dates, parties), then line items, then financial totals. "
            "I will verify arithmetic and report completeness."
        )
        extracted = {
            "invoice_number": inv_num,
            "invoice_date": inv_date,
            "due_date": due_date,
            "payment_terms": "Net 30",
            "po_reference": po_ref,
            "vendor": {
                "name": v[0], "address": f"{v[1]}, {v[2]} {v[3]}, {v[4]}"
            },
            "bill_to": {
                "name": c[0], "address": f"{c[1]}, {c[2]} {c[3]}, {c[4]}"
            },
            "line_items": [
                {"description": desc, "quantity": qty, "unit_price": unit,
                 "line_total": round(qty * unit, 2)}
                for desc, qty, unit in chosen
            ],
            "subtotal": subtotal,
            "vat_rate": "20%",
            "vat_amount": vat,
            "total_due": total,
            "currency": "GBP",
        }
        n_fields = 14 + len(chosen)
        answer = (
            "```json\n"
            + json.dumps(extracted, indent=2)
            + "\n```\n\n"
            f"**Extraction complete.** All {n_fields} fields extracted. No missing data. "
            f"Arithmetic verified: subtotal {_amount_uk(subtotal)} + VAT {_amount_uk(vat)} = {_amount_uk(total)}. Correct."
        )
        examples.append(_sft(query, reasoning, answer, "extraction_full_invoice"))

    return examples


def _validation_discrepancy_examples() -> List[Dict[str, Any]]:
    examples = []

    discrepancy_cases = [
        {
            "doc": (
                "INVOICE INV-2025-0099\n"
                "Line 1: Consulting (10 x £150.00) = £1,500.00\n"
                "Line 2: Software Licence (5 x £800.00) = £4,000.00\n"
                "Line 3: Hardware (20 x £45.50) = £910.00\n"
                "Subtotal: £4,350.00\nVAT 20%: £870.00\nTotal: £5,220.00"
            ),
            "issue": "Line totals sum to £6,410.00 but subtotal shows £4,350.00 — DISCREPANCY of £2,060.00",
            "correct_subtotal": 6410.00,
            "stated_subtotal": 4350.00,
        },
        {
            "doc": (
                "INVOICE INV-2025-0100\n"
                "Item A: Qty 0, Unit £250.00, Total £500.00\n"
                "Item B: Qty 5, Unit £100.00, Total £500.00\n"
                "Subtotal: £1,000.00\nVAT 20%: £200.00\nTotal: £1,200.00"
            ),
            "issue": "Item A has Qty=0 but line total=£500.00 — likely OCR error (0 should be 2)",
            "correct_subtotal": 1000.00,
            "stated_subtotal": 1000.00,
        },
        {
            "doc": (
                "INVOICE INV-2025-0101\n"
                "Invoice Date: 15/13/2025\n"
                "Payment Due: 45 days\n"
                "Total: £3,200.00"
            ),
            "issue": "Invoice Date 15/13/2025 is invalid — month 13 does not exist; likely OCR error",
            "correct_subtotal": None,
            "stated_subtotal": None,
        },
    ]

    for i in range(300):
        case = _pick(discrepancy_cases)
        query = (
            f"Review the following invoice for any discrepancies, errors, or anomalies. "
            f"Report findings with specific values.\n\n{case['doc']}"
        )
        reasoning = (
            "I will verify: (1) each line total = qty × unit_price, "
            "(2) subtotal = sum of line totals, "
            "(3) VAT = subtotal × stated rate, "
            "(4) grand total = subtotal + VAT, "
            "(5) all dates are valid calendar dates."
        )
        answer = (
            f"**DISCREPANCY DETECTED:**\n\n"
            f"{case['issue']}\n\n"
            f"**Action required:** Return invoice to vendor for correction before processing payment."
        )
        examples.append(_sft(query, reasoning, answer, "extraction_validation_discrepancy"))

    return examples


def _completeness_examples() -> List[Dict[str, Any]]:
    examples = []

    complete_docs = [
        {
            "doc": (
                "PO-2025-00512\nIssued: 15/03/2025\nSupplier: Nordic Supply AS\n"
                "Delivery Address: 55 Finance Street, Manchester M2 4AH\n"
                "Required By: 30/04/2025\nPayment Terms: Net 30\n"
                "Item 1: Network Switch 48-port x 4 = £4,800.00\n"
                "Item 2: Patch cables (box 100) x 2 = £160.00\n"
                "Subtotal: £4,960.00\nVAT 20%: £992.00\nTotal: £5,952.00\n"
                "Authorised By: Sarah Williams — 18/03/2025"
            ),
            "expected_fields": 12,
            "extracted_fields": 12,
            "missing": [],
        },
        {
            "doc": (
                "Quote QTE-2025-00127\nVendor: Delta Procurement GmbH\n"
                "Valid Until: 30/04/2025\n"
                "Item A: Office Chairs x 50 = €12,500.00\n"
                "Item B: Desks x 30 = €18,000.00\n"
                "Total: €30,500.00"
            ),
            "expected_fields": 8,
            "extracted_fields": 6,
            "missing": ["Issue Date", "Payment Terms"],
        },
    ]

    for i in range(200):
        doc = _pick(complete_docs)
        query = (
            f"Extract all fields from the following document and report "
            f"completeness (total fields extracted vs expected).\n\n{doc['doc']}"
        )
        reasoning = (
            "I will extract all labelled fields systematically. "
            "For a PO/Quote I expect: reference number, dates, parties, line items, "
            "financial totals, authorisation. "
            "I will count extracted fields and compare against the expected set, "
            "listing any missing fields explicitly."
        )
        if doc["missing"]:
            completeness = (
                f"Extracted {doc['extracted_fields']} of {doc['expected_fields']} expected fields. "
                f"Missing: {', '.join(doc['missing'])}."
            )
        else:
            completeness = (
                f"All {doc['expected_fields']} fields extracted. No missing data."
            )
        answer = (
            f"**Extraction summary:**\n{doc['doc']}\n\n"
            f"**Completeness report:** {completeness}"
        )
        examples.append(_sft(query, reasoning, answer, "extraction_completeness"))

    return examples


def generate_accurate_extraction_examples() -> List[Dict[str, Any]]:
    """Return 1,000 accurate extraction SFT examples."""
    random.seed(45)
    examples: List[Dict[str, Any]] = []
    examples.extend(_full_invoice_extraction_examples())      # 300
    examples.extend(_validation_discrepancy_examples())       # 300
    examples.extend(_completeness_examples())                 # 200
    # Pad remaining 200 with invoice examples using different seed
    random.seed(99)
    extra = _full_invoice_extraction_examples()
    random.shuffle(extra)
    examples.extend(extra[:200])
    random.shuffle(examples)
    return examples[:1000]


# ---------------------------------------------------------------------------
# 5. Context awareness examples  (target: 500)
# ---------------------------------------------------------------------------

def generate_context_awareness_examples() -> List[Dict[str, Any]]:
    """Return 500 context-awareness SFT examples."""
    random.seed(46)
    examples: List[Dict[str, Any]] = []

    context_cases = [
        {
            "doc": (
                "INVOICE INV-2025-0042\nFrom: Apex Cleaning Services Ltd\n"
                "To: GlobalTech Corp, Manchester\nDate: 01/04/2025\nDue: 01/05/2025\n"
                "PO Ref: PO-00389\nDescription: Monthly office cleaning services — April 2025\n"
                "Total: £1,850.00 (Net 30)"
            ),
            "summary": (
                "This is a service invoice from Apex Cleaning Services Ltd to GlobalTech Corp "
                "for monthly office cleaning services in April 2025, totalling £1,850.00. "
                "It references PO-00389 and is payable within 30 days (by 01/05/2025). "
                "Cross-reference with PO-00389 to verify the service was pre-authorised."
            ),
        },
        {
            "doc": (
                "QUOTE QTE-2025-00127\nFrom: Office Furnishings Direct Ltd\n"
                "To: Meridian Holdings GmbH\nValid Until: 15/02/2025\n"
                "Items: Ergonomic chairs x 100 @ £280.00 = £28,000.00\n"
                "Total: £28,000.00 (ex VAT)\nDelivery: 3–4 weeks"
            ),
            "summary": (
                "This is an expired quote from Office Furnishings Direct Ltd to Meridian Holdings GmbH "
                "for 100 ergonomic chairs totalling £28,000.00 (ex VAT). "
                "The validity date was 15/02/2025, which has passed (today is 09/04/2026). "
                "A new quote should be requested before proceeding with this purchase."
            ),
        },
        {
            "doc": (
                "PURCHASE ORDER PO-2025-00512\nTo: Nordic Supply AS\nIssued: 15/03/2025\n"
                "Payment Terms: Net 30\nItems: Managed switches x 4 = £4,800.00\n"
                "Total: £4,800.00 + VAT"
            ),
            "summary": (
                "This is a purchase order issued to Nordic Supply AS on 15/03/2025 "
                "for 4 managed network switches at £4,800.00 (plus VAT). "
                "Payment terms are Net 30, meaning payment is due by 14/04/2025. "
                "When the corresponding invoice arrives, verify quantities and prices match this PO."
            ),
        },
        {
            "doc": (
                "INVOICE INV-2025-0099\nFrom: Tech Supplies Ltd\nTo: GlobalTech Corp\n"
                "Invoice Date: 01/03/2025\nPO Ref: PO-00271\nTotal: £12,400.00\n"
                "Note: Partial delivery — remaining items on back order"
            ),
            "summary": (
                "This invoice from Tech Supplies Ltd to GlobalTech Corp for £12,400.00 covers a partial "
                "delivery against PO-00271. A further delivery (and likely a second invoice) is expected "
                "for back-ordered items. Do not close PO-00271 until all items are delivered and invoiced."
            ),
        },
    ]

    while len(examples) < 500:
        case = _pick(context_cases)
        query = (
            "Read the following document and provide a concise contextual summary — "
            "what is it, who are the parties, what is the financial obligation, "
            "and are there any actions required?\n\n" + case["doc"]
        )
        reasoning = (
            "I will identify: document type, parties (issuer and recipient), "
            "subject matter, financial values, relevant dates, payment terms, "
            "cross-references to other documents, and any outstanding actions or anomalies."
        )
        examples.append(_sft(query, reasoning, case["summary"], "context_awareness"))

    random.shuffle(examples)
    return examples[:500]


# ---------------------------------------------------------------------------
# 6. DPO pairs  (target: 500)
# ---------------------------------------------------------------------------

def generate_layout_dpo_pairs() -> List[Dict[str, Any]]:
    """Return 500 DPO preference pairs for layout-aware extraction."""
    random.seed(47)
    examples: List[Dict[str, Any]] = []

    dpo_scenarios = [
        # Scenario A: full vs partial extraction
        {
            "query": (
                "Extract all fields from this invoice:\n\n"
                "INVOICE INV-2025-0042\nDate: 12/03/2025\nDue: 12/04/2025\n"
                "From: Apex Solutions Ltd, London W1K 2AB\n"
                "To: GlobalTech Corp, Manchester M2 4AH\n"
                "PO Ref: PO-00389 | Terms: Net 30\n"
                "Consulting (10 x £150.00) = £1,500.00\n"
                "Software Licence (5 x £800.00) = £4,000.00\n"
                "Subtotal: £5,500.00 | VAT 20%: £1,100.00 | TOTAL: £6,600.00"
            ),
            "chosen_reasoning": (
                "I will extract every labelled field: invoice number, invoice date, due date, "
                "vendor name and address, client name and address, PO reference, payment terms, "
                "all line items with quantities and unit prices, subtotal, VAT rate and amount, "
                "and grand total. I will then verify all arithmetic."
            ),
            "chosen_answer": (
                "```json\n{\n"
                '  "invoice_number": "INV-2025-0042",\n'
                '  "invoice_date": "12/03/2025",\n'
                '  "due_date": "12/04/2025",\n'
                '  "vendor": {"name": "Apex Solutions Ltd", "location": "London W1K 2AB"},\n'
                '  "bill_to": {"name": "GlobalTech Corp", "location": "Manchester M2 4AH"},\n'
                '  "po_reference": "PO-00389",\n'
                '  "payment_terms": "Net 30",\n'
                '  "line_items": [\n'
                '    {"description": "Consulting", "qty": 10, "unit_price": 150.00, "line_total": 1500.00},\n'
                '    {"description": "Software Licence", "qty": 5, "unit_price": 800.00, "line_total": 4000.00}\n'
                "  ],\n"
                '  "subtotal": 5500.00,\n'
                '  "vat_rate": "20%",\n'
                '  "vat_amount": 1100.00,\n'
                '  "total_due": 6600.00,\n'
                '  "currency": "GBP"\n'
                "}\n```\n\n"
                "All 14 fields extracted. No missing data. "
                "Arithmetic verified: £1,500 + £4,000 = £5,500 subtotal. "
                "VAT: £5,500 × 20% = £1,100. Total: £5,500 + £1,100 = £6,600. All correct."
            ),
            "rejected_reasoning": (
                "I'll grab the main fields from this invoice."
            ),
            "rejected_answer": (
                "Invoice number: INV-2025-0042\nTotal: £6,600.00\nDue: 12/04/2025"
            ),
        },
        # Scenario B: layout-aware two-column parsing vs ignoring layout
        {
            "query": (
                "This invoice uses a two-column layout. Extract vendor and billing details:\n\n"
                "VENDOR                        | BILL TO\n"
                "Nordic Supply AS              | Fjord Logistics AS\n"
                "Storgata 14, Oslo 0155        | Havneveien 22, Bergen 5003\n"
                "Norway                        | Norway\n"
            ),
            "chosen_reasoning": (
                "The document has a two-column layout separated by '|'. "
                "Left column = VENDOR, right column = BILL TO. "
                "I will read each column independently to avoid mixing addresses."
            ),
            "chosen_answer": (
                "**Vendor:**\n- Name: Nordic Supply AS\n- Address: Storgata 14, Oslo 0155, Norway\n\n"
                "**Bill To:**\n- Name: Fjord Logistics AS\n- Address: Havneveien 22, Bergen 5003, Norway"
            ),
            "rejected_reasoning": (
                "I'll read the text from left to right and extract names and addresses."
            ),
            "rejected_answer": (
                "Company 1: Nordic Supply AS, Fjord Logistics AS\n"
                "Address: Storgata 14 Oslo 0155, Havneveien 22 Bergen 5003\n"
                "Country: Norway Norway"
            ),
        },
        # Scenario C: OCR correction vs accepting corrupted data
        {
            "query": (
                "Extract the invoice details from this OCR-extracted text:\n\n"
                "lnvoice Number: lNV-2O25-OO42\nDate: O1/O4/2O25\nTotal: £1O,234.56"
            ),
            "chosen_reasoning": (
                "The text has OCR errors: 'l' mistaken for 'I' in 'Invoice' and 'INV', "
                "'O' (letter) mistaken for '0' (digit) in dates and amounts. "
                "I will correct these before extraction."
            ),
            "chosen_answer": (
                "**Corrected and extracted:**\n"
                "- Invoice Number: INV-2025-0042\n"
                "- Date: 01/04/2025\n"
                "- Total: £10,234.56\n\n"
                "**OCR corrections applied:** 'l'→'I' (×2), 'O'→'0' (×5)"
            ),
            "rejected_reasoning": (
                "I will extract the data as shown in the text."
            ),
            "rejected_answer": (
                "- Invoice Number: lNV-2O25-OO42\n"
                "- Date: O1/O4/2O25\n"
                "- Total: £1O,234.56"
            ),
        },
        # Scenario D: discrepancy detection vs silent acceptance
        {
            "query": (
                "Verify and extract this invoice:\n\n"
                "Line 1: Consulting 10 x £150.00 = £1,500.00\n"
                "Line 2: Licence 5 x £800.00 = £4,000.00\n"
                "Subtotal: £4,350.00\nVAT 20%: £870.00\nTotal: £5,220.00"
            ),
            "chosen_reasoning": (
                "I will extract line items, verify each line total (qty × unit price), "
                "sum the lines and compare to the stated subtotal. "
                "Computed: £1,500 + £4,000 = £6,500 — but stated subtotal is £4,350. "
                "This is a discrepancy of £2,150 and must be flagged."
            ),
            "chosen_answer": (
                "**DISCREPANCY DETECTED:**\n"
                "- Line totals sum to: £6,500.00 (£1,500 + £4,000)\n"
                "- Stated subtotal: £4,350.00\n"
                "- Discrepancy: £2,150.00\n\n"
                "Do NOT process this invoice. Return to vendor for correction."
            ),
            "rejected_reasoning": "I'll extract the totals as printed.",
            "rejected_answer": (
                "Subtotal: £4,350.00\nVAT: £870.00\nTotal: £5,220.00"
            ),
        },
        # Scenario E: EU currency format handling vs misinterpretation
        {
            "query": (
                "Extract all monetary values from this German invoice:\n\n"
                "Nettobetrag: €15.000,00\nMwSt. 19%: €2.850,00\nGesamtbetrag: €17.850,00"
            ),
            "chosen_reasoning": (
                "This is a German invoice using EU number format: "
                "dot is the thousands separator, comma is the decimal separator. "
                "€15.000,00 = 15,000.00 EUR, not 15.00 EUR. "
                "I will convert all values to standard decimal representation."
            ),
            "chosen_answer": (
                "**Extracted values (EU format correctly interpreted):**\n"
                "- Net amount (Nettobetrag): €15,000.00\n"
                "- VAT 19% (Mwst.): €2,850.00\n"
                "- Gross total (Gesamtbetrag): €17,850.00\n\n"
                "**Verification:** €15,000 × 19% = €2,850. €15,000 + €2,850 = €17,850. Correct."
            ),
            "rejected_reasoning": "I'll extract the numbers as printed.",
            "rejected_answer": (
                "Net: €15.00\nVAT 19%: €2.85\nTotal: €17.85"
            ),
        },
    ]

    while len(examples) < 500:
        sc = _pick(dpo_scenarios)
        examples.append(
            _dpo(
                sc["query"],
                sc["chosen_reasoning"],
                sc["chosen_answer"],
                sc["rejected_reasoning"],
                sc["rejected_answer"],
                "layout_dpo",
            )
        )

    random.shuffle(examples)
    return examples[:500]


# ---------------------------------------------------------------------------
# 7. Orchestrator
# ---------------------------------------------------------------------------

def generate_all_layout_ocr(output_dir: str) -> Dict[str, Any]:
    """Generate all layout/OCR training data and write to output_dir.

    Writes:
        {output_dir}/layout_ocr_sft.jsonl
        {output_dir}/layout_ocr_dpo.jsonl

    Returns a summary dict.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sft_path = out / "layout_ocr_sft.jsonl"
    dpo_path = out / "layout_ocr_dpo.jsonl"

    # --- SFT ---
    sft_examples: List[Dict[str, Any]] = []
    sft_examples.extend(generate_layout_aware_examples())        # 1,000
    sft_examples.extend(generate_ocr_handling_examples())        # 800
    sft_examples.extend(generate_pattern_recognition_examples()) # 800
    sft_examples.extend(generate_accurate_extraction_examples()) # 1,000
    sft_examples.extend(generate_context_awareness_examples())   # 500

    with JSONLWriter(sft_path) as w:
        for ex in sft_examples:
            w.write(ex)
    sft_count = len(sft_examples)

    # --- DPO ---
    dpo_examples = generate_layout_dpo_pairs()  # 500
    with JSONLWriter(dpo_path) as w:
        for ex in dpo_examples:
            w.write(ex)
    dpo_count = len(dpo_examples)

    return {
        "sft_path": str(sft_path),
        "dpo_path": str(dpo_path),
        "sft_count": sft_count,
        "dpo_count": dpo_count,
        "total": sft_count + dpo_count,
        "breakdown": {
            "layout_aware": 1000,
            "ocr_handling": 800,
            "pattern_recognition": 800,
            "accurate_extraction": 1000,
            "context_awareness": 500,
            "dpo_pairs": 500,
        },
    }
