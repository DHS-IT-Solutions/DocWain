"""Track 3 — OCR & Vision Intelligence data generator for DocWain V2+ SFT.

Generates 2500 training examples across ten categories:
  - Printed text clean scans            (200)
  - Printed text degraded scans         (350)
  - Handwritten block letters           (300)
  - Handwritten cursive/notes           (250)
  - Mixed print + handwriting           (200)
  - Diagram understanding               (300)
  - Chart-in-image extraction           (200)
  - Table-in-image reconstruction       (250)
  - Stamps, watermarks, overlays        (200)
  - Caption & label extraction          (250)

Each OCR example includes confidence scoring per region.
Diagram examples include both extracted text and semantic description.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

from src.finetune.v2.data_generator.base import (
    DOMAINS,
    DOC_TYPES,
    JSONLWriter,
    format_sft_example,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_COMPANY_NAMES = [
    "Acme Corp", "Globex Industries", "Initech Solutions", "Umbrella LLC",
    "Soylent Corp", "Wonka Enterprises", "Stark Industries", "Wayne Enterprises",
    "Cyberdyne Systems", "Tyrell Corporation", "Weyland-Yutani", "Oscorp",
    "LexCorp", "Massive Dynamic", "Pied Piper", "Hooli",
]
_PERSON_NAMES = [
    "Alice Johnson", "Bob Martinez", "Carol Singh", "David Chen",
    "Emily Watson", "Frank Osei", "Grace Kim", "Hector Rossi",
    "Irene Muller", "James Okafor", "Karen Tanaka", "Liam Petrov",
    "Maria Fernandez", "Nathan Park", "Olivia Nguyen", "Paul Schmidt",
]
_DEPARTMENTS = [
    "Engineering", "Finance", "Human Resources", "Legal",
    "Marketing", "Operations", "Sales", "Research & Development",
    "Compliance", "Procurement", "IT", "Customer Success",
]
_ADDRESSES = [
    "123 Main St, Suite 400, New York, NY 10001",
    "456 Oak Avenue, San Francisco, CA 94105",
    "789 Elm Drive, Chicago, IL 60601",
    "321 Pine Road, Austin, TX 78701",
]
_DEGRADATION_TYPES = [
    ("skewed", "5-degree clockwise skew", "moderate"),
    ("low_resolution", "72 DPI scan", "significant"),
    ("bleed_through", "text from reverse side visible", "moderate"),
    ("faded", "faded ink with low contrast", "significant"),
    ("coffee_stain", "brown stain obscuring corner", "localized"),
    ("creased", "horizontal crease through middle", "moderate"),
    ("overexposed", "bright scan with washed-out text", "moderate"),
    ("underexposed", "dark scan with crushed shadows", "significant"),
]
_DIAGRAM_TYPES = [
    ("Organizational chart", "hierarchical", ["CEO", "VP Engineering", "VP Sales", "VP Marketing",
     "Director Backend", "Director Frontend", "Director Enterprise", "Director SMB"]),
    ("Flowchart", "sequential", ["Start", "Input Data", "Validate?", "Yes: Process",
     "No: Error Handler", "Transform", "Output Results", "End"]),
    ("Network diagram", "connected", ["Firewall", "Load Balancer", "Web Server 1", "Web Server 2",
     "App Server", "Database Primary", "Database Replica", "Cache Layer"]),
    ("Process map", "swim-lane", ["Customer Request", "Intake Team Review", "Manager Approval",
     "Engineering Assessment", "Budget Check", "Implementation", "QA Verification", "Delivery"]),
    ("State diagram", "transitional", ["Idle", "Initializing", "Running", "Paused",
     "Error", "Recovering", "Shutting Down", "Terminated"]),
]
_CHART_TYPES = [
    "bar", "line", "pie", "stacked bar", "grouped bar", "area", "scatter",
]
_STAMP_TYPES = [
    ("APPROVED", "green", "circular"),
    ("REJECTED", "red", "rectangular"),
    ("RECEIVED", "blue", "rectangular with date"),
    ("CONFIDENTIAL", "red", "diagonal banner"),
    ("DRAFT", "gray", "diagonal watermark"),
    ("COPY", "blue", "rectangular"),
    ("CERTIFIED TRUE COPY", "purple", "circular with border"),
    ("FOR INTERNAL USE ONLY", "orange", "rectangular banner"),
]


def _pick(lst: list, rng: random.Random) -> Any:
    return rng.choice(lst)


def _rand_amount(rng: random.Random, lo: float = 100.0, hi: float = 99999.0) -> str:
    return f"${rng.uniform(lo, hi):,.2f}"


def _rand_date(rng: random.Random) -> str:
    y = rng.randint(2019, 2026)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    return f"{y}-{m:02d}-{d:02d}"


def _confidence(rng: random.Random, base: float, spread: float = 0.1) -> float:
    return round(min(1.0, max(0.0, rng.gauss(base, spread))), 2)


# ---------------------------------------------------------------------------
# Category 1: Printed text clean scans (200)
# ---------------------------------------------------------------------------

def _gen_printed_clean(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    idx = 0
    while len(results) < count:
        results.append(_printed_clean_example(rng, idx))
        idx += 1
    return results


def _printed_clean_example(rng: random.Random, variant: int) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    doc_type = _pick(DOC_TYPES, rng)
    company = _pick(_COMPANY_NAMES, rng)
    person = _pick(_PERSON_NAMES, rng)
    date = _rand_date(rng)
    ref_num = f"REF-{rng.randint(10000, 99999)}"
    amount = _rand_amount(rng, 1000, 50000)

    conf_header = _confidence(rng, 0.98)
    conf_body = _confidence(rng, 0.97)
    conf_footer = _confidence(rng, 0.96)

    scan_desc = (
        f"[Scanned document: {domain} {doc_type} from {company}]\n"
        f"[Scan quality: 300 DPI, clean, no artifacts]\n\n"
        f"HEADER:\n"
        f"  {company}\n"
        f"  {doc_type.replace('_', ' ').title()} - {ref_num}\n"
        f"  Date: {date}\n\n"
        f"BODY:\n"
        f"  To: {person}\n"
        f"  Subject: {domain.title()} Review\n\n"
        f"  This document confirms the {domain} {doc_type.replace('_', ' ')} "
        f"with reference {ref_num}, dated {date}. The total amount of {amount} "
        f"has been approved and is scheduled for processing.\n\n"
        f"FOOTER:\n"
        f"  Page 1 of 1 | {company} Confidential"
    )
    query = f"{scan_desc}\n\nExtract all text from this clean scan with confidence scores."
    reasoning = (
        f"Step 1: Document scan quality is high -- 300 DPI, clean, no artifacts.\n"
        f"Step 2: Header region: company name, reference number, date -- all clearly printed.\n"
        f"Step 3: Body region: addressee, subject, and paragraph text -- standard font, good contrast.\n"
        f"Step 4: Footer region: page number and confidentiality notice.\n"
        f"Step 5: All text is machine-printed with consistent font and spacing.\n"
        f"Confidence: Very high across all regions."
    )
    answer = (
        f"**Extracted Text:**\n\n"
        f"**Header:**\n"
        f"- Company: {company}\n"
        f"- Document: {doc_type.replace('_', ' ').title()} - {ref_num}\n"
        f"- Date: {date}\n\n"
        f"**Body:**\n"
        f"- To: {person}\n"
        f"- Subject: {domain.title()} Review\n"
        f"- Content: This document confirms the {domain} {doc_type.replace('_', ' ')} "
        f"with reference {ref_num}, dated {date}. The total amount of {amount} "
        f"has been approved and is scheduled for processing.\n\n"
        f"**Footer:**\n"
        f"- Page 1 of 1 | {company} Confidential\n\n"
        f"**OCR Confidence:**\n"
        f"- Region 1 (header): {conf_header} -- clear printed text\n"
        f"- Region 2 (body): {conf_body} -- clear printed text\n"
        f"- Region 3 (footer): {conf_footer} -- clear printed text"
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Category 2: Printed text degraded scans (350)
# ---------------------------------------------------------------------------

def _gen_printed_degraded(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    idx = 0
    while len(results) < count:
        results.append(_printed_degraded_example(rng, idx))
        idx += 1
    return results


def _printed_degraded_example(rng: random.Random, variant: int) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    doc_type = _pick(DOC_TYPES, rng)
    company = _pick(_COMPANY_NAMES, rng)
    person = _pick(_PERSON_NAMES, rng)
    date = _rand_date(rng)
    degradation = _pick(_DEGRADATION_TYPES, rng)
    deg_name, deg_desc, deg_severity = degradation

    amount = _rand_amount(rng, 500, 80000)
    ref = f"{doc_type.upper()[:3]}-{rng.randint(1000, 9999)}"

    # Confidence drops with degradation
    base_conf = 0.92 if deg_severity == "moderate" else 0.78 if deg_severity == "significant" else 0.85
    conf_header = _confidence(rng, base_conf)
    conf_body = _confidence(rng, base_conf - 0.05)
    conf_degraded = _confidence(rng, base_conf - 0.15)

    # Simulate OCR errors based on degradation
    ocr_errors = {
        "skewed": [("amount", amount, amount.replace("3", "8") if "3" in amount else amount)],
        "low_resolution": [("reference", ref, ref.replace("0", "O") if "0" in ref else ref)],
        "bleed_through": [("date", date, f"{date} [ghost text: ...from reverse...]")],
        "faded": [("company", company, company[:len(company)//2] + "..." + company[-3:])],
        "coffee_stain": [("footer", "Page 1", "[obscured by stain]")],
        "creased": [("body_line", "approved and scheduled", "appro[crease]ed and sch[crease]duled")],
        "overexposed": [("amount", amount, "[$?,???.??]")],
        "underexposed": [("date", date, f"[barely visible: {date[:7]}...]")],
    }
    errors = ocr_errors.get(deg_name, [])
    error_field, original, corrupted = errors[0] if errors else ("none", "", "")

    scan_desc = (
        f"[Scanned document: {domain} {doc_type} from {company}]\n"
        f"[Scan quality: DEGRADED -- {deg_desc}]\n\n"
        f"HEADER (partially affected by {deg_name}):\n"
        f"  {company}\n"
        f"  {ref} | Date: {date}\n\n"
        f"BODY:\n"
        f"  Re: {domain.title()} matter for {person}\n"
        f"  Amount: {amount}\n"
        f"  Status: approved and scheduled for processing\n\n"
        f"[DEGRADATION NOTE: {deg_desc} -- affects readability in some regions]"
    )
    query = (
        f"{scan_desc}\n\n"
        f"Extract text from this degraded scan ({deg_name}). "
        f"Flag any regions where OCR confidence is low."
    )
    reasoning = (
        f"Step 1: Scan degradation detected: {deg_name} ({deg_desc}).\n"
        f"Step 2: Severity: {deg_severity} -- expect {'some' if deg_severity == 'moderate' else 'significant'} OCR errors.\n"
        f"Step 3: Header region: {company}, {ref}, {date} -- "
        f"{'mostly readable' if deg_severity == 'moderate' else 'partially obscured'}.\n"
        f"Step 4: Body region: key fields (person, amount, status) -- "
        f"{'legible with minor artifacts' if deg_severity == 'moderate' else 'degraded, some characters uncertain'}.\n"
        f"Step 5: Specific issue: '{error_field}' field may have OCR error: "
        f"original '{original}' may read as '{corrupted}'.\n"
        f"Confidence: {'Medium' if deg_severity == 'moderate' else 'Low'} -- "
        f"{deg_name} introduces uncertainty."
    )
    answer = (
        f"**Extracted Text (degraded scan -- {deg_name}):**\n\n"
        f"**Header:**\n"
        f"- Company: {company}\n"
        f"- Reference: {ref}\n"
        f"- Date: {date}\n\n"
        f"**Body:**\n"
        f"- Subject: {domain.title()} matter for {person}\n"
        f"- Amount: {amount}\n"
        f"- Status: Approved and scheduled\n\n"
        f"**OCR Confidence:**\n"
        f"- Region 1 (header): {conf_header} -- {deg_name} affects clarity\n"
        f"- Region 2 (body text): {conf_body} -- {'readable' if conf_body > 0.85 else 'partially degraded'}\n"
        f"- Region 3 (affected area): {conf_degraded} -- {deg_desc}\n\n"
        f"**Flagged issues:**\n"
        f"- '{error_field}' field: possible OCR misread due to {deg_name}. "
        f"Verify '{corrupted}' against source."
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Category 3: Handwritten block letters (300)
# ---------------------------------------------------------------------------

def _gen_handwritten_block(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    idx = 0
    while len(results) < count:
        results.append(_handwritten_block_example(rng, idx))
        idx += 1
    return results


def _handwritten_block_example(rng: random.Random, variant: int) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    person = _pick(_PERSON_NAMES, rng)
    company = _pick(_COMPANY_NAMES, rng)
    address = _pick(_ADDRESSES, rng)
    date = _rand_date(rng)

    # Simulate form fields filled in by hand with block letters
    fields = {
        "FULL NAME": person.upper(),
        "DATE OF BIRTH": f"{rng.randint(1960, 2000)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
        "ADDRESS": address.upper(),
        "PHONE": f"({rng.randint(200,999)}) {rng.randint(100,999)}-{rng.randint(1000,9999)}",
        "EMPLOYER": company.upper(),
        "DATE": date,
    }

    # Some characters may be ambiguous in block handwriting
    ambiguous_chars = {"O": "0", "I": "1", "S": "5", "Z": "2", "B": "8", "G": "6"}
    uncertain_fields = {}
    for field, value in fields.items():
        for char, alt in ambiguous_chars.items():
            if char in value:
                uncertain_fields[field] = (char, alt)
                break

    conf_fields = {}
    for f in fields:
        if f in uncertain_fields:
            conf_fields[f] = _confidence(rng, 0.82)
        else:
            conf_fields[f] = _confidence(rng, 0.91)

    scan_desc = (
        f"[Scanned form: {domain} application from {company}]\n"
        f"[Handwriting style: BLOCK CAPITALS]\n\n"
        f"Form fields (handwritten in block letters):\n"
        + "\n".join(f"  {k}: [{v}]" for k, v in fields.items())
    )
    query = (
        f"{scan_desc}\n\n"
        f"Extract the handwritten block-letter entries from this form. "
        f"Flag any ambiguous characters."
    )
    uncertain_notes = []
    for f, (char, alt) in uncertain_fields.items():
        uncertain_notes.append(f"  - {f}: '{char}' could be '{alt}'")

    reasoning = (
        f"Step 1: Form filled in block capitals -- generally more legible than cursive.\n"
        f"Step 2: Extract each field:\n"
        + "\n".join(f"  - {k}: \"{v}\"" for k, v in fields.items())
        + f"\nStep 3: Check for ambiguous block-letter characters (O/0, I/1, S/5, Z/2).\n"
        + (f"Step 4: Potential ambiguities found:\n" + "\n".join(uncertain_notes) if uncertain_notes else
           f"Step 4: No significant ambiguities detected.")
        + f"\nConfidence: {'Medium' if uncertain_fields else 'High'} -- "
        f"block letters are {'mostly' if uncertain_fields else 'clearly'} legible."
    )
    answer = (
        f"**Extracted Handwritten Fields (Block Letters):**\n\n"
        + "\n".join(f"- **{k}:** {v}" for k, v in fields.items())
        + f"\n\n**OCR Confidence:**\n"
        + "\n".join(
            f"- Region ({f}): {conf_fields[f]} -- "
            f"{'ambiguous character detected' if f in uncertain_fields else 'clear block letters'}"
            for f in fields
        )
        + ("\n\n**Ambiguous characters flagged:**\n"
           + "\n".join(uncertain_notes) if uncertain_notes else "")
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Category 4: Handwritten cursive/notes (250)
# ---------------------------------------------------------------------------

def _gen_handwritten_cursive(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    idx = 0
    while len(results) < count:
        results.append(_handwritten_cursive_example(rng, idx))
        idx += 1
    return results


def _handwritten_cursive_example(rng: random.Random, variant: int) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    person = _pick(_PERSON_NAMES, rng)
    context_type = _pick(["margin annotation", "meeting notes", "sticky note", "personal memo"], rng)

    note_templates = [
        f"Follow up with {_pick(_PERSON_NAMES, rng)} re: {domain} compliance by {_rand_date(rng)}",
        f"Need to review Section 3.2 -- numbers don't match {_pick(_DEPARTMENTS, rng)} report",
        f"Approved by {_pick(_PERSON_NAMES, rng)}, see attached for {domain} details",
        f"Check {_pick(_COMPANY_NAMES, rng)} contract clause 7b before signing",
        f"Budget revised to {_rand_amount(rng, 10000, 100000)} per {_pick(_DEPARTMENTS, rng)} request",
        f"Urgent: deadline moved to {_rand_date(rng)} -- notify all stakeholders",
        f"Discussed with {_pick(_PERSON_NAMES, rng)} -- agreed to extend timeline by 2 weeks",
        f"Flag for {_pick(_DEPARTMENTS, rng)} review -- potential {domain} risk identified",
    ]
    note_text = _pick(note_templates, rng)

    # Cursive has lower confidence and potential misreads
    conf = _confidence(rng, 0.65)
    # Simulate partial illegibility
    words = note_text.split()
    n_unclear = rng.randint(1, max(1, len(words) // 4))
    unclear_indices = rng.sample(range(len(words)), k=min(n_unclear, len(words)))
    display_words = []
    for i, w in enumerate(words):
        if i in unclear_indices:
            display_words.append(f"[{w}?]")
        else:
            display_words.append(w)
    partially_read = " ".join(display_words)

    scan_desc = (
        f"[Scanned {context_type} on {domain} document]\n"
        f"[Handwriting style: cursive, {_pick(['neat', 'hurried', 'cramped', 'flowing'], rng)}]\n\n"
        f"Handwritten text (cursive):\n"
        f"  \"{partially_read}\""
    )
    query = (
        f"{scan_desc}\n\n"
        f"Transcribe this handwritten cursive {context_type}. "
        f"Mark uncertain words and provide confidence scores."
    )
    reasoning = (
        f"Step 1: Context: {context_type} on a {domain} document.\n"
        f"Step 2: Handwriting is cursive -- lower baseline OCR accuracy expected.\n"
        f"Step 3: Attempt full transcription: \"{note_text}\"\n"
        f"Step 4: {n_unclear} word(s) are partially illegible or ambiguous.\n"
        f"Step 5: Uncertain words marked with [?].\n"
        f"Confidence: {'Low' if conf < 0.6 else 'Medium'} -- cursive handwriting "
        f"with {n_unclear} uncertain segment(s)."
    )
    answer = (
        f"**Transcription ({context_type}):**\n\n"
        f"\"{partially_read}\"\n\n"
        f"**Best interpretation:** \"{note_text}\"\n\n"
        f"**OCR Confidence:**\n"
        f"- Region 1 (handwritten note): {conf} -- cursive, partially legible\n"
        f"- Uncertain words: {n_unclear} of {len(words)} "
        f"({n_unclear/len(words)*100:.0f}% uncertain)\n\n"
        f"**Note:** Words marked with [?] should be verified against context or with the author."
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Category 5: Mixed print + handwriting (200)
# ---------------------------------------------------------------------------

def _gen_mixed_print_handwriting(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    idx = 0
    while len(results) < count:
        results.append(_mixed_print_hw_example(rng, idx))
        idx += 1
    return results


def _mixed_print_hw_example(rng: random.Random, variant: int) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    company = _pick(_COMPANY_NAMES, rng)
    person = _pick(_PERSON_NAMES, rng)
    date = _rand_date(rng)
    amount = _rand_amount(rng, 500, 50000)
    form_type = _pick(["Application", "Claim", "Enrollment", "Registration", "Request"], rng)

    printed_fields = {
        "Form Title": f"{company} {domain.title()} {form_type} Form",
        "Form Number": f"FM-{rng.randint(1000, 9999)}",
        "Instructions": f"Complete all sections. Print clearly in the spaces provided.",
    }
    handwritten_fields = {
        "Applicant Name": person,
        "Date": date,
        "Amount Requested": amount,
        "Reason": _pick([
            "Equipment replacement needed",
            "Annual renewal required",
            "New project funding",
            "Emergency repair",
        ], rng),
        "Signature": f"[cursive signature: {person}]",
    }

    conf_printed = _confidence(rng, 0.97)
    conf_block = _confidence(rng, 0.89)
    conf_cursive = _confidence(rng, 0.68)

    scan_desc = (
        f"[Scanned form: {domain} {form_type.lower()} from {company}]\n"
        f"[Mixed content: printed labels + handwritten entries]\n\n"
        f"PRINTED SECTIONS:\n"
        + "\n".join(f"  [{k}]: {v}" for k, v in printed_fields.items())
        + f"\n\nHANDWRITTEN ENTRIES:\n"
        + "\n".join(f"  {k}: [{v}] (handwritten)" for k, v in handwritten_fields.items())
    )
    query = (
        f"{scan_desc}\n\n"
        f"Extract all text from this form, distinguishing printed labels from handwritten entries. "
        f"Provide separate confidence scores for each type."
    )
    reasoning = (
        f"Step 1: Form has two text types: machine-printed labels and handwritten entries.\n"
        f"Step 2: Printed sections (labels, instructions) -- high OCR confidence.\n"
        f"Step 3: Handwritten entries -- variable confidence:\n"
        f"  - Block letter entries (name, date, amount): moderate confidence.\n"
        f"  - Cursive signature: low confidence.\n"
        f"Step 4: Cross-reference printed labels with handwritten values for field mapping.\n"
        f"Confidence: Mixed -- printed text is reliable, handwriting needs verification."
    )
    answer = (
        f"**Extracted Text (Mixed Print + Handwriting):**\n\n"
        f"**Printed Labels (machine-printed):**\n"
        + "\n".join(f"- {k}: {v}" for k, v in printed_fields.items())
        + f"\n\n**Handwritten Entries:**\n"
        + "\n".join(f"- **{k}:** {v}" for k, v in handwritten_fields.items())
        + f"\n\n**OCR Confidence:**\n"
        f"- Region 1 (printed labels): {conf_printed} -- clear machine-printed text\n"
        f"- Region 2 (block handwriting): {conf_block} -- legible block letters\n"
        f"- Region 3 (cursive signature): {conf_cursive} -- cursive, partially legible\n\n"
        f"**Recommendation:** Verify handwritten amount ({amount}) and signature against records."
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Category 6: Diagram understanding (300)
# ---------------------------------------------------------------------------

def _gen_diagram_understanding(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    idx = 0
    while len(results) < count:
        results.append(_diagram_example(rng, idx))
        idx += 1
    return results


def _diagram_example(rng: random.Random, variant: int) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    company = _pick(_COMPANY_NAMES, rng)
    diagram_type, structure, labels = _pick(_DIAGRAM_TYPES, rng)

    # Randomize the labels a bit
    if diagram_type == "Organizational chart":
        title = _pick(_PERSON_NAMES, rng)
        labels = [title] + [f"VP {_pick(_DEPARTMENTS, rng)}" for _ in range(3)] + \
                 [f"Dir. {_pick(_DEPARTMENTS, rng)}" for _ in range(4)]
    elif diagram_type == "Process map":
        labels = [f"{_pick(_DEPARTMENTS, rng)}: {step}" for step in
                  ["Receive Request", "Initial Review", "Approval", "Assignment",
                   "Execution", "Quality Check", "Delivery", "Closure"]]

    n_nodes = len(labels)
    n_connections = rng.randint(n_nodes - 1, n_nodes + 3)

    # Determine hierarchy/structure
    if structure == "hierarchical":
        levels = 3
        level_desc = f"{levels} hierarchical levels"
        top = labels[0]
        mid = labels[1:4]
        bottom = labels[4:]
    elif structure == "sequential":
        levels = 1
        level_desc = "linear sequence"
        top = labels[0]
        mid = labels[1:-1]
        bottom = [labels[-1]]
    else:
        levels = 2
        level_desc = f"{structure} arrangement"
        top = labels[0]
        mid = labels[1:len(labels)//2]
        bottom = labels[len(labels)//2:]

    conf_text = _confidence(rng, 0.90)
    conf_structure = _confidence(rng, 0.85)

    scan_desc = (
        f"[Image: {diagram_type} from {company}'s {domain} documentation]\n"
        f"[Diagram structure: {structure}]\n"
        f"[Nodes: {n_nodes}, Connections: {n_connections}]\n\n"
        f"Visible elements:\n"
        + "\n".join(f"  - [{label}]" for label in labels)
        + f"\n\nConnections: arrows/lines between nodes"
    )
    query = (
        f"{scan_desc}\n\n"
        f"Analyze this {diagram_type.lower()}. Provide both the extracted text "
        f"and a semantic description of the structure."
    )
    reasoning = (
        f"Step 1: Diagram type identified: {diagram_type}.\n"
        f"Step 2: Structure: {level_desc} with {n_nodes} nodes and {n_connections} connections.\n"
        f"Step 3: Extract all text labels from nodes.\n"
        f"Step 4: Map the relationships between nodes based on arrows/lines.\n"
        f"Step 5: Build semantic description of the overall structure.\n"
        f"Confidence: {'High' if conf_structure > 0.85 else 'Medium'} -- "
        f"diagram elements are {'clearly' if conf_structure > 0.85 else 'mostly'} readable."
    )
    answer = (
        f"**Diagram Analysis:**\n"
        f"- Type: {diagram_type}\n"
        f"- Structure: {level_desc}\n"
        f"- Description: {top} at {'top' if structure == 'hierarchical' else 'start'}, "
        f"{len(mid)} {'reports' if structure == 'hierarchical' else 'intermediate steps'}, "
        f"{len(bottom)} {'leaf nodes' if structure == 'hierarchical' else 'endpoints'}\n"
        f"- Extracted text: {', '.join(labels)}\n\n"
        f"**Semantic Structure:**\n"
        + (f"- Top level: {top}\n"
           f"- Middle level: {', '.join(mid)}\n"
           f"- Bottom level: {', '.join(bottom)}\n")
        + f"\n**Connections:** {n_connections} directional links between {n_nodes} nodes\n\n"
        f"**OCR Confidence:**\n"
        f"- Region (text labels): {conf_text} -- {'clear' if conf_text > 0.85 else 'mostly readable'}\n"
        f"- Region (structure/arrows): {conf_structure} -- {'clear' if conf_structure > 0.85 else 'some ambiguity'}"
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Category 7: Chart-in-image extraction (200)
# ---------------------------------------------------------------------------

def _gen_chart_extraction(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    idx = 0
    while len(results) < count:
        results.append(_chart_example(rng, idx))
        idx += 1
    return results


def _chart_example(rng: random.Random, variant: int) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    company = _pick(_COMPANY_NAMES, rng)
    chart_type = _pick(_CHART_TYPES, rng)

    if chart_type == "pie":
        categories = rng.sample(_DEPARTMENTS, k=rng.randint(3, 6))
        values = [rng.uniform(5, 40) for _ in categories]
        total = sum(values)
        values = [v / total * 100 for v in values]
        data_points = [(c, f"{v:.1f}%") for c, v in zip(categories, values)]
        y_axis = "Percentage"
        x_axis = "Category"
    else:
        x_labels = [f"Q{i+1}" for i in range(4)] if rng.random() > 0.5 else \
                   rng.sample(["Jan", "Feb", "Mar", "Apr", "May", "Jun"], k=rng.randint(4, 6))
        values = [rng.uniform(10000, 100000) for _ in x_labels]
        data_points = [(x, f"${v:,.0f}") for x, v in zip(x_labels, values)]
        y_axis = _pick(["Revenue ($)", "Units Sold", "Expenses ($)", "Growth (%)"], rng)
        x_axis = _pick(["Quarter", "Month", "Period", "Time"], rng)

    chart_title = f"{company} {domain.title()} {y_axis} by {x_axis}"
    conf_labels = _confidence(rng, 0.92)
    conf_values = _confidence(rng, 0.87)

    scan_desc = (
        f"[Image: {chart_type} chart from {company}'s {domain} report]\n"
        f"[Title: {chart_title}]\n"
        f"[X-axis: {x_axis}, Y-axis: {y_axis}]\n"
        f"[Data points: {len(data_points)}]\n"
    )
    query = (
        f"{scan_desc}\n\n"
        f"Extract all data points from this {chart_type} chart, including axis labels, "
        f"title, and numeric values."
    )
    reasoning = (
        f"Step 1: Chart type: {chart_type}. Title: \"{chart_title}\".\n"
        f"Step 2: Identify axes -- X: {x_axis}, Y: {y_axis}.\n"
        f"Step 3: Read each data point from the chart:\n"
        + "\n".join(f"  - {label}: {value}" for label, value in data_points)
        + f"\nStep 4: Verify data consistency -- {len(data_points)} points extracted.\n"
        f"Step 5: Note any visual estimation uncertainty for non-exact values.\n"
        f"Confidence: {'High' if conf_values > 0.85 else 'Medium'} -- "
        f"{'values clearly labeled' if conf_values > 0.85 else 'some values estimated from visual position'}."
    )
    answer = (
        f"**Chart Data Extraction:**\n\n"
        f"- **Chart type:** {chart_type}\n"
        f"- **Title:** {chart_title}\n"
        f"- **X-axis:** {x_axis}\n"
        f"- **Y-axis:** {y_axis}\n\n"
        f"**Data Points ({len(data_points)}):**\n"
        + "\n".join(f"- {label}: {value}" for label, value in data_points)
        + f"\n\n**OCR Confidence:**\n"
        f"- Region (axis labels/title): {conf_labels} -- printed text\n"
        f"- Region (data values): {conf_values} -- "
        f"{'read from labels' if conf_values > 0.85 else 'estimated from bar height/position'}"
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Category 8: Table-in-image reconstruction (250)
# ---------------------------------------------------------------------------

def _gen_table_reconstruction(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    idx = 0
    while len(results) < count:
        results.append(_table_reconstruction_example(rng, idx))
        idx += 1
    return results


def _table_reconstruction_example(rng: random.Random, variant: int) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    company = _pick(_COMPANY_NAMES, rng)
    n_rows = rng.randint(4, 8)
    n_cols = rng.randint(3, 5)

    col_types = rng.sample([
        ("Name", lambda: _pick(_PERSON_NAMES, rng)),
        ("Department", lambda: _pick(_DEPARTMENTS, rng)),
        ("Amount", lambda: _rand_amount(rng, 1000, 50000)),
        ("Date", lambda: _rand_date(rng)),
        ("Status", lambda: _pick(["Active", "Pending", "Closed", "Review"], rng)),
        ("ID", lambda: f"ID-{rng.randint(1000, 9999)}"),
        ("Product", lambda: _pick(["Widget A", "Gadget Pro", "Module X", "Sensor Suite"], rng)),
    ], k=n_cols)

    headers = [ct[0] for ct in col_types]
    rows_data = []
    for _ in range(n_rows):
        row = [ct[1]() for ct in col_types]
        rows_data.append(row)

    # Simulate scan quality issues
    quality = _pick(["clear gridlines", "faint gridlines", "no gridlines (whitespace-aligned)",
                      "partial gridlines (some missing)"], rng)
    conf_structure = _confidence(rng, 0.88 if "clear" in quality else 0.74)
    conf_text = _confidence(rng, 0.91)

    scan_desc = (
        f"[Scanned table image from {company}'s {domain} document]\n"
        f"[Table structure: {n_rows} rows x {n_cols} columns, {quality}]\n\n"
        f"Visible content:\n"
        f"  Headers: {' | '.join(headers)}\n"
        + "\n".join(f"  Row {i+1}: {' | '.join(row)}" for i, row in enumerate(rows_data))
    )
    query = (
        f"{scan_desc}\n\n"
        f"Reconstruct this scanned table into structured data. "
        f"Note any alignment or recognition issues."
    )
    reasoning = (
        f"Step 1: Table detected in scanned image -- {n_rows} rows x {n_cols} columns.\n"
        f"Step 2: Gridline quality: {quality}.\n"
        f"Step 3: Column identification from headers: {', '.join(headers)}.\n"
        f"Step 4: Read each cell and align with column headers.\n"
        f"Step 5: Verify row/column alignment -- "
        f"{'consistent' if 'clear' in quality else 'may need adjustment due to ' + quality}.\n"
        f"Confidence: {'High' if conf_structure > 0.85 else 'Medium'} -- "
        f"table structure {'clearly visible' if conf_structure > 0.85 else 'partially ambiguous'}."
    )
    # Build markdown table
    md_header = "| " + " | ".join(headers) + " |"
    md_sep = "| " + " | ".join("---" for _ in headers) + " |"
    md_rows = "\n".join("| " + " | ".join(row) + " |" for row in rows_data)
    answer = (
        f"**Reconstructed Table ({n_rows} x {n_cols}):**\n\n"
        f"{md_header}\n{md_sep}\n{md_rows}\n\n"
        f"**OCR Confidence:**\n"
        f"- Region (table structure): {conf_structure} -- {quality}\n"
        f"- Region (cell text): {conf_text} -- printed text within cells\n\n"
        f"**Notes:**\n"
        + ("- Table gridlines clearly define cell boundaries\n" if "clear" in quality else
           f"- {quality.capitalize()} -- column alignment inferred from text position\n")
        + f"- {n_rows} data rows and {n_cols} columns successfully reconstructed"
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Category 9: Stamps, watermarks, overlays (200)
# ---------------------------------------------------------------------------

def _gen_stamps_watermarks(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    idx = 0
    while len(results) < count:
        results.append(_stamp_watermark_example(rng, idx))
        idx += 1
    return results


def _stamp_watermark_example(rng: random.Random, variant: int) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    company = _pick(_COMPANY_NAMES, rng)
    person = _pick(_PERSON_NAMES, rng)
    date = _rand_date(rng)
    stamp_text, stamp_color, stamp_shape = _pick(_STAMP_TYPES, rng)

    # Some stamps have additional info
    stamp_extras = {}
    if "date" in stamp_shape:
        stamp_extras["Date"] = date
    if stamp_text in ("APPROVED", "CERTIFIED TRUE COPY"):
        stamp_extras["Authorized by"] = person
    if stamp_text == "RECEIVED":
        stamp_extras["Received date"] = date
        stamp_extras["Office"] = _pick(_DEPARTMENTS, rng)

    underlying_text = (
        f"This {domain} document from {company} pertains to the {_pick(DOC_TYPES, rng).replace('_', ' ')} "
        f"dated {_rand_date(rng)}. The matter involves {_pick(_DEPARTMENTS, rng)} operations."
    )

    # Determine overlap
    overlap_pct = rng.randint(10, 40)
    conf_stamp = _confidence(rng, 0.93)
    conf_underlying = _confidence(rng, 0.95 - overlap_pct / 100)

    scan_desc = (
        f"[Scanned {domain} document from {company}]\n"
        f"[Overlay: {stamp_shape} {stamp_color} stamp reading '{stamp_text}']\n"
        f"[Stamp overlaps approximately {overlap_pct}% of body text]\n\n"
        f"Underlying text:\n  {underlying_text}\n\n"
        f"Stamp details:\n"
        f"  Text: {stamp_text}\n"
        + ("\n".join(f"  {k}: {v}" for k, v in stamp_extras.items()) if stamp_extras else "")
    )
    query = (
        f"{scan_desc}\n\n"
        f"Extract both the stamp/overlay text and the underlying document text. "
        f"Note any areas where the stamp obscures content."
    )
    reasoning = (
        f"Step 1: Identify overlay -- {stamp_shape} {stamp_color} stamp: '{stamp_text}'.\n"
        f"Step 2: Stamp overlaps ~{overlap_pct}% of the page body.\n"
        f"Step 3: Extract stamp text and metadata:\n"
        f"  - Primary text: {stamp_text}\n"
        + "\n".join(f"  - {k}: {v}" for k, v in stamp_extras.items())
        + f"\nStep 4: Extract underlying text, noting obscured regions.\n"
        f"Step 5: The stamp {'significantly' if overlap_pct > 25 else 'partially'} "
        f"overlaps the body text.\n"
        f"Confidence: stamp text is {'high' if conf_stamp > 0.9 else 'medium'}, "
        f"underlying text is {'medium' if conf_underlying > 0.7 else 'low'} in overlap area."
    )
    answer = (
        f"**Stamp/Overlay Extraction:**\n\n"
        f"- **Stamp text:** {stamp_text}\n"
        f"- **Shape:** {stamp_shape}\n"
        f"- **Color:** {stamp_color}\n"
        + ("\n".join(f"- **{k}:** {v}" for k, v in stamp_extras.items()) + "\n" if stamp_extras else "")
        + f"\n**Underlying Document Text:**\n"
        f"\"{underlying_text}\"\n\n"
        f"**Overlap impact:** ~{overlap_pct}% of body text partially obscured by stamp.\n\n"
        f"**OCR Confidence:**\n"
        f"- Region 1 (stamp text): {conf_stamp} -- {stamp_color} ink on document\n"
        f"- Region 2 (underlying, clear area): {min(conf_underlying + 0.15, 0.98):.2f} -- unobscured text\n"
        f"- Region 3 (underlying, overlap area): {conf_underlying} -- "
        f"{'partially readable' if conf_underlying > 0.7 else 'significantly obscured'}"
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Category 10: Caption & label extraction (250)
# ---------------------------------------------------------------------------

def _gen_caption_label(count: int, rng: random.Random) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    templates = [
        lambda: _figure_caption(rng),
        lambda: _axis_labels(rng),
        lambda: _legend_extraction(rng),
    ]
    idx = 0
    while len(results) < count:
        fn = templates[idx % len(templates)]
        results.append(fn())
        idx += 1
    return results


def _figure_caption(rng: random.Random) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    company = _pick(_COMPANY_NAMES, rng)
    fig_num = rng.randint(1, 15)
    year = rng.randint(2021, 2026)

    caption_templates = [
        f"Figure {fig_num}: Annual {domain.title()} performance metrics for {company}, FY{year}",
        f"Fig. {fig_num} -- Comparison of {_pick(_DEPARTMENTS, rng)} vs {_pick(_DEPARTMENTS, rng)} budgets ({year})",
        f"Figure {fig_num}. Regional distribution of {domain} activities across Q1-Q4 {year}",
        f"FIGURE {fig_num}: Trend analysis -- {company} {domain} indicators, {year-2}-{year}",
    ]
    caption = _pick(caption_templates, rng)
    source_note = _pick([
        f"Source: {company} Internal Report, {year}",
        f"Data source: {_pick(_DEPARTMENTS, rng)} quarterly submissions",
        f"Note: All values in thousands. Adjusted for inflation.",
        f"*Preliminary figures, subject to final audit.",
    ], rng)

    conf = _confidence(rng, 0.94)

    scan_desc = (
        f"[Image from {company}'s {domain} report, page {rng.randint(1, 50)}]\n"
        f"[Contains a figure with caption and source note]\n\n"
        f"Figure content: [chart/diagram/image]\n\n"
        f"Caption (below figure):\n  \"{caption}\"\n\n"
        f"Source note (small text below caption):\n  \"{source_note}\""
    )
    query = (
        f"{scan_desc}\n\n"
        f"Extract the figure caption, number, and source note from this image."
    )
    reasoning = (
        f"Step 1: Locate caption text below the figure.\n"
        f"Step 2: Parse figure number: {fig_num}.\n"
        f"Step 3: Full caption: \"{caption}\".\n"
        f"Step 4: Source/footnote in smaller text: \"{source_note}\".\n"
        f"Step 5: Both are printed text at standard reading size.\n"
        f"Confidence: High -- caption text is clearly printed."
    )
    answer = (
        f"**Caption Extraction:**\n\n"
        f"- **Figure number:** {fig_num}\n"
        f"- **Caption:** \"{caption}\"\n"
        f"- **Source note:** \"{source_note}\"\n\n"
        f"**OCR Confidence:**\n"
        f"- Region (caption text): {conf} -- clear printed text\n"
        f"- Region (source note): {_confidence(rng, 0.91)} -- smaller font but legible"
    )
    return format_sft_example(query, reasoning, answer)


def _axis_labels(rng: random.Random) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    company = _pick(_COMPANY_NAMES, rng)
    chart_type = _pick(_CHART_TYPES, rng)

    x_label = _pick(["Quarter", "Month", "Year", "Region", "Department", "Category"], rng)
    y_label = _pick(["Revenue ($M)", "Units (thousands)", "Growth Rate (%)",
                      "Headcount", "Expenses ($K)", "Score"], rng)
    title = f"{company} {domain.title()} {y_label} by {x_label}"

    x_ticks = {
        "Quarter": ["Q1", "Q2", "Q3", "Q4"],
        "Month": rng.sample(["Jan", "Feb", "Mar", "Apr", "May", "Jun"], k=4),
        "Year": [str(y) for y in range(rng.randint(2020, 2023), rng.randint(2024, 2027))],
        "Region": rng.sample(["North", "South", "East", "West", "Central"], k=4),
        "Department": rng.sample(_DEPARTMENTS, k=4),
        "Category": rng.sample(["A", "B", "C", "D", "E"], k=4),
    }.get(x_label, ["A", "B", "C", "D"])

    conf_title = _confidence(rng, 0.96)
    conf_axis = _confidence(rng, 0.93)
    conf_ticks = _confidence(rng, 0.90)

    scan_desc = (
        f"[Chart image: {chart_type} chart from {company}'s {domain} report]\n"
        f"[Title at top, axis labels on sides, tick marks along axes]\n\n"
        f"Title: \"{title}\"\n"
        f"X-axis label: \"{x_label}\"\n"
        f"Y-axis label: \"{y_label}\"\n"
        f"X-axis ticks: {', '.join(x_ticks)}"
    )
    query = (
        f"{scan_desc}\n\n"
        f"Extract the chart title, axis labels, and tick mark text from this image."
    )
    reasoning = (
        f"Step 1: Title at top of chart: \"{title}\".\n"
        f"Step 2: X-axis label (bottom): \"{x_label}\" with ticks: {', '.join(x_ticks)}.\n"
        f"Step 3: Y-axis label (left side, rotated 90 degrees): \"{y_label}\".\n"
        f"Step 4: All text is printed; Y-axis label is rotated but readable.\n"
        f"Confidence: High -- standard chart labeling."
    )
    answer = (
        f"**Chart Label Extraction:**\n\n"
        f"- **Title:** \"{title}\"\n"
        f"- **X-axis label:** \"{x_label}\"\n"
        f"- **Y-axis label:** \"{y_label}\" (rotated 90 degrees)\n"
        f"- **X-axis ticks:** {', '.join(x_ticks)}\n\n"
        f"**OCR Confidence:**\n"
        f"- Region (title): {conf_title} -- large, clear text\n"
        f"- Region (axis labels): {conf_axis} -- standard size, Y-axis rotated\n"
        f"- Region (tick marks): {conf_ticks} -- smaller text along axes"
    )
    return format_sft_example(query, reasoning, answer)


def _legend_extraction(rng: random.Random) -> Dict[str, str]:
    domain = _pick(DOMAINS, rng)
    company = _pick(_COMPANY_NAMES, rng)
    chart_type = _pick(["bar", "line", "pie", "stacked bar"], rng)

    legend_items = rng.sample([
        ("Revenue", "blue", "solid"),
        ("Expenses", "red", "solid"),
        ("Profit", "green", "dashed"),
        ("Forecast", "orange", "dotted"),
        ("Prior Year", "gray", "solid"),
        ("Target", "black", "dashed"),
        ("Budget", "purple", "solid"),
        ("Actual", "teal", "solid"),
    ], k=rng.randint(3, 5))

    conf = _confidence(rng, 0.92)

    scan_desc = (
        f"[Chart image: {chart_type} chart from {company}'s {domain} report]\n"
        f"[Legend box in the top-right corner]\n\n"
        f"Legend entries:\n"
        + "\n".join(f"  [{color} {style} line/bar] {name}" for name, color, style in legend_items)
    )
    query = (
        f"{scan_desc}\n\n"
        f"Extract all legend entries from this chart, including their visual identifiers."
    )
    reasoning = (
        f"Step 1: Legend box located in top-right corner of the {chart_type} chart.\n"
        f"Step 2: {len(legend_items)} legend entries found.\n"
        f"Step 3: Each entry has a color indicator and label text:\n"
        + "\n".join(f"  - {name}: {color} {style}" for name, color, style in legend_items)
        + f"\nStep 4: All legend text is clearly printed.\n"
        f"Confidence: High -- legend entries are well-separated and readable."
    )
    answer = (
        f"**Legend Extraction ({len(legend_items)} entries):**\n\n"
        + "\n".join(f"- **{name}**: {color} {style} {'line' if chart_type == 'line' else 'bar/segment'}"
                    for name, color, style in legend_items)
        + f"\n\n**OCR Confidence:**\n"
        f"- Region (legend text): {conf} -- clear printed labels\n"
        f"- Region (color indicators): visual identification, not OCR"
    )
    return format_sft_example(query, reasoning, answer)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_track3_data(output_dir: str | Path, seed: int = 42) -> dict:
    """Generate Track 3 OCR & Vision Intelligence training data.

    Args:
        output_dir: Directory to write JSONL files into.
        seed: Random seed for reproducibility.

    Returns:
        Dict with ``sft_path`` and ``sft_count``.
    """
    rng = random.Random(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_path = output_dir / "track3_ocr_vision_sft.jsonl"

    categories = [
        ("printed_clean", _gen_printed_clean, 200),
        ("printed_degraded", _gen_printed_degraded, 350),
        ("handwritten_block", _gen_handwritten_block, 300),
        ("handwritten_cursive", _gen_handwritten_cursive, 250),
        ("mixed_print_hw", _gen_mixed_print_handwriting, 200),
        ("diagram_understanding", _gen_diagram_understanding, 300),
        ("chart_extraction", _gen_chart_extraction, 200),
        ("table_reconstruction", _gen_table_reconstruction, 250),
        ("stamps_watermarks", _gen_stamps_watermarks, 200),
        ("caption_label", _gen_caption_label, 250),
    ]

    all_examples: List[Dict[str, str]] = []
    for name, gen_fn, count in categories:
        sub_seed = seed + hash(name) % 10000
        cat_rng = random.Random(sub_seed)
        examples = gen_fn(count, cat_rng)
        all_examples.extend(examples)

    rng.shuffle(all_examples)

    with JSONLWriter(sft_path) as writer:
        for ex in all_examples:
            writer.write(ex)

    return {
        "sft_path": str(sft_path),
        "sft_count": len(all_examples),
    }


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/finetune/v2")
    result = generate_track3_data(out)
    print(f"Track 3 OCR/Vision: {result['sft_count']} SFT examples -> {result['sft_path']}")
