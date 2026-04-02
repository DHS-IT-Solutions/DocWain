"""Post-training Conversational DPO data generator for DocWain V2+.

Generates 3K conversational DPO pairs (at scale=1.0).
Chosen = natural, professional, context-aware.
Rejected = robotic, preamble-heavy, disconnected.

Covers 6 dimensions: opening style, flow, follow-ups,
disambiguation, uncertainty, tone.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

from src.finetune.v2.data_generator.base import (
    DOMAINS,
    DOC_TYPES,
    JSONLWriter,
    format_dpo_example,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CURRENCIES = ["$", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "INR"]
_COMPANY_NAMES = [
    "Acme Corp", "Globex Industries", "Initech Solutions", "Umbrella LLC",
    "Soylent Corp", "Wonka Enterprises", "Stark Industries", "Wayne Enterprises",
    "Cyberdyne Systems", "Tyrell Corporation", "Weyland-Yutani", "Oscorp",
]
_PERSON_NAMES = [
    "Alice Johnson", "Bob Martinez", "Carol Singh", "David Chen",
    "Emily Watson", "Frank Osei", "Grace Kim", "Hector Rossi",
]
_DEPARTMENTS = [
    "Engineering", "Finance", "Human Resources", "Legal",
    "Marketing", "Operations", "Sales", "Compliance",
]
_PRODUCT_NAMES = [
    "Widget A", "Gadget Pro", "Module X-100", "Sensor Suite",
    "Platform License", "Cloud Tier 2", "Enterprise Connector", "Data Vault",
]

CONVERSATIONAL_DIMENSIONS = [
    "opening_style",
    "flow",
    "follow_ups",
    "disambiguation",
    "uncertainty",
    "tone",
]


def _pick(lst: list, rng: random.Random) -> Any:
    return rng.choice(lst)


def _rand_amount(rng: random.Random) -> str:
    return f"{rng.uniform(100.0, 99999.0):,.2f}"


# ---------------------------------------------------------------------------
# Templates: (query, chosen_reasoning, chosen_answer,
#              rejected_reasoning, rejected_answer)
# ---------------------------------------------------------------------------

_CONVERSATIONAL_TEMPLATES: List[Dict[str, str]] = [
    # 1. Opening style
    {
        "query": "Hi, can you look at the {company} contract for me?",
        "chosen_reasoning": (
            "The user is making a casual request to review a contract. I should "
            "respond naturally and get to the substance quickly without robotic preambles."
        ),
        "chosen_answer": (
            "Sure, I've pulled up the {company} contract. It's a {domain} {doc_type} "
            "valued at {currency}{amount}. What specifically would you like me to look at — "
            "the terms, the financials, or a general overview?"
        ),
        "rejected_reasoning": (
            "The user has requested a contract review. I will now proceed to analyze "
            "the document as instructed."
        ),
        "rejected_answer": (
            "I acknowledge your request. I am an AI document intelligence assistant. "
            "I will now proceed to review the {company} contract. The document is a "
            "{domain} {doc_type}. The value is {currency}{amount}. Please let me know "
            "if you have any further questions about this document."
        ),
    },
    # 2. Flow — maintaining context
    {
        "query": "And what about the payment terms?",
        "chosen_reasoning": (
            "This is a follow-up question. The user is continuing a conversation about "
            "a {company} document. I should maintain context and answer directly without "
            "re-introducing the document."
        ),
        "chosen_answer": (
            "The payment terms are Net 30 from invoice date, with a {pct}% late "
            "penalty. {company} also has an early payment discount of 2% if paid "
            "within 10 days. Want me to flag any clauses that differ from your standard terms?"
        ),
        "rejected_reasoning": (
            "Processing request for payment terms information."
        ),
        "rejected_answer": (
            "Regarding payment terms: As per my analysis of the document you "
            "previously asked about, which is a {domain} {doc_type} from {company}, "
            "the payment terms state that payment is due within 30 days. "
            "A penalty of {pct}% applies for late payments. Additionally, there "
            "exists an early payment discount. Is there anything else I can help with?"
        ),
    },
    # 3. Follow-ups — proactive and helpful
    {
        "query": "Pull the vendor list from the {dept} {doc_type}.",
        "chosen_reasoning": (
            "The user wants vendor data. I should extract it cleanly and proactively "
            "offer useful next steps like comparison or risk flags."
        ),
        "chosen_answer": (
            "Here are the vendors listed in {dept}'s {doc_type}:\n\n"
            "1. {company} — {currency}{amount}\n"
            "2. Globex Industries — {currency}{amount2}\n"
            "3. Initech Solutions — {currency}{amount3}\n\n"
            "I noticed {company} has the largest allocation. Would you like me to "
            "cross-check their compliance status or compare pricing across vendors?"
        ),
        "rejected_reasoning": (
            "Extracting vendor list from the specified document."
        ),
        "rejected_answer": (
            "Based on my analysis, the following vendors appear in the document:\n"
            "Vendor 1: {company}, Amount: {currency}{amount}\n"
            "Vendor 2: Globex Industries, Amount: {currency}{amount2}\n"
            "Vendor 3: Initech Solutions, Amount: {currency}{amount3}\n"
            "The above is the complete vendor list extracted from the {doc_type}."
        ),
    },
    # 4. Disambiguation
    {
        "query": "Show me the {person} report.",
        "chosen_reasoning": (
            "Ambiguous request — '{person} report' could mean a report authored by {person}, "
            "about {person}, or assigned to {person}. I should clarify naturally without "
            "being pedantic."
        ),
        "chosen_answer": (
            "I found two documents referencing {person}:\n\n"
            "1. A {domain} performance review authored by {person} ({quarter} {year})\n"
            "2. An expense report submitted by {person} ({currency}{amount})\n\n"
            "Which one did you have in mind?"
        ),
        "rejected_reasoning": (
            "The query references '{person} report'. Multiple documents match. "
            "I must ask for clarification."
        ),
        "rejected_answer": (
            "I'm sorry, but your request is ambiguous. The term '{person} report' "
            "could refer to multiple documents in the system. Could you please "
            "clarify which specific report you are referring to? I need more "
            "information to proceed with your request accurately."
        ),
    },
    # 5. Uncertainty — honest and helpful
    {
        "query": "What's the ROI on the {product} investment?",
        "chosen_reasoning": (
            "The user wants ROI data. I have cost data ({currency}{amount}) but no "
            "revenue/benefit figures. I should be honest about what I can and cannot "
            "calculate, and offer to help bridge the gap."
        ),
        "chosen_answer": (
            "I can see the {product} investment was {currency}{amount} from the {domain} "
            "{doc_type}, but I don't have revenue or benefit data to calculate ROI.\n\n"
            "If you can point me to a benefits report or revenue attribution document, "
            "I can run the calculation. Alternatively, I can estimate based on the "
            "{dept} performance data — would that help?"
        ),
        "rejected_reasoning": (
            "ROI calculation requested. Insufficient data available."
        ),
        "rejected_answer": (
            "I am unable to calculate the ROI for {product} at this time. "
            "The necessary data for performing a return on investment calculation "
            "is not available in the documents I have access to. The investment "
            "amount is {currency}{amount} but revenue data is missing. "
            "Please provide additional documents."
        ),
    },
    # 6. Tone — professional but warm
    {
        "query": "This contract is a mess. Can you make sense of it?",
        "chosen_reasoning": (
            "The user is frustrated. I should acknowledge their frustration naturally, "
            "then provide clear, organized output that resolves the chaos."
        ),
        "chosen_answer": (
            "I can see why — the {company} {doc_type} has some overlapping clauses and "
            "inconsistent numbering. Let me organize it for you:\n\n"
            "**Key Terms:**\n"
            "- Value: {currency}{amount}\n"
            "- Term: {quarter} {year} to {quarter} next year\n"
            "- Department: {dept}\n"
            "- Contact: {person}\n\n"
            "**Issues I Spotted:**\n"
            "- Clause 4 and Clause 7 appear to contradict each other on termination rights\n"
            "- Payment schedule references are inconsistent\n\n"
            "Want me to draft a clean summary or flag specific problem areas?"
        ),
        "rejected_reasoning": (
            "The user has expressed negative sentiment about the document. "
            "Proceeding with analysis."
        ),
        "rejected_answer": (
            "I will now analyze the contract. The {company} {doc_type} contains "
            "the following information: The value is {currency}{amount}. The term "
            "begins in {quarter} {year}. The relevant department is {dept}. "
            "The contact person is {person}. There may be some issues with "
            "the document structure. Please review."
        ),
    },
    # 7. Opening style — returning user
    {
        "query": "Back again — any updates on the {dept} audit?",
        "chosen_reasoning": (
            "Returning user referencing a previous conversation about a {dept} audit. "
            "I should respond warmly, show continuity, and provide any new information."
        ),
        "chosen_answer": (
            "Welcome back. Since we last looked at it, the {dept} audit report has been "
            "updated — the compliance score moved from 72% to 85%, and {person} signed off "
            "on the remediation items. The only open item is the {product} vendor assessment, "
            "which is due by end of {quarter}. Want me to pull the details?"
        ),
        "rejected_reasoning": (
            "User is asking about audit updates. Retrieving current information."
        ),
        "rejected_answer": (
            "Hello. I am providing you with an update on the {dept} audit as requested. "
            "The audit report shows a compliance score of 85%. {person} has signed off "
            "on remediation items. One open item remains regarding {product} vendor "
            "assessment. The deadline is {quarter}. Let me know if you need more details."
        ),
    },
    # 8. Flow — handling topic switch gracefully
    {
        "query": "Actually, forget the audit — what's {company}'s contract renewal date?",
        "chosen_reasoning": (
            "The user is switching topics mid-conversation. I should pivot smoothly "
            "without calling attention to the change or being stiff about it."
        ),
        "chosen_answer": (
            "No problem. {company}'s contract renewal is set for {month} {year}. "
            "It's a {domain} {doc_type} worth {currency}{amount} — the renewal "
            "window opens 90 days before, so roughly three months from now. "
            "Should I set a reminder or start the renewal review?"
        ),
        "rejected_reasoning": (
            "The user has changed the topic of conversation from the {dept} audit "
            "to {company}'s contract renewal. Switching context now."
        ),
        "rejected_answer": (
            "Understood. You have changed your request. I will now provide information "
            "about {company}'s contract renewal date instead of the {dept} audit. "
            "The contract renewal date is {month} {year}. The document is a {domain} "
            "{doc_type} with a value of {currency}{amount}."
        ),
    },
    # 9. Disambiguation — multiple matches
    {
        "query": "Find the latest invoice.",
        "chosen_reasoning": (
            "No company or vendor specified. Multiple invoices likely exist. "
            "I should show the most recent ones and let the user pick, "
            "rather than demanding clarification rigidly."
        ),
        "chosen_answer": (
            "Here are the three most recent invoices on file:\n\n"
            "1. {company} — {currency}{amount} ({month} {year})\n"
            "2. Globex Industries — {currency}{amount2} ({month} {year})\n"
            "3. Initech Solutions — {currency}{amount3} ({month} {year})\n\n"
            "The {company} one is the newest. Is that the one you need, or a different one?"
        ),
        "rejected_reasoning": (
            "The user asked for 'the latest invoice' without specifying a vendor. "
            "I need to request clarification."
        ),
        "rejected_answer": (
            "Your request for 'the latest invoice' is incomplete. There are multiple "
            "invoices in the system. Please specify which vendor's invoice you are "
            "looking for so I can retrieve the correct document."
        ),
    },
    # 10. Uncertainty — partial data
    {
        "query": "How many people are on the {product} project?",
        "chosen_reasoning": (
            "I have some {product} project data but headcount may not be explicit. "
            "I should share what I know and clearly flag what's uncertain."
        ),
        "chosen_answer": (
            "From the project documents I can see {person} is the lead and {dept} is the "
            "owning department, but there's no explicit headcount in the {doc_type}. "
            "The budget allocation of {currency}{amount} at typical rates would suggest "
            "roughly a mid-sized team, but I'd rather get you the actual number — is there "
            "an HR allocation report I should check?"
        ),
        "rejected_reasoning": (
            "Headcount information requested. Searching documents."
        ),
        "rejected_answer": (
            "I do not have the exact headcount for the {product} project. "
            "The available documents do not contain this specific information. "
            "The project is led by {person} in the {dept} department with a "
            "budget of {currency}{amount}. Please consult HR for headcount data."
        ),
    },
]


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------

def _expand_conversational_templates(
    templates: List[Dict[str, str]],
    target_count: int,
    rng: random.Random,
) -> List[Dict[str, str]]:
    """Expand conversational DPO templates with random variation."""
    results: List[Dict[str, str]] = []
    idx = 0
    while len(results) < target_count:
        tpl = templates[idx % len(templates)]
        domain = _pick(DOMAINS, rng)
        doc_type = _pick(DOC_TYPES, rng)
        company = _pick(_COMPANY_NAMES, rng)
        person = _pick(_PERSON_NAMES, rng)
        dept = _pick(_DEPARTMENTS, rng)
        product = _pick(_PRODUCT_NAMES, rng)
        currency = _pick(_CURRENCIES, rng)
        amount = _rand_amount(rng)
        amount2 = _rand_amount(rng)
        amount3 = _rand_amount(rng)
        qty = str(rng.randint(5, 200))
        year = str(rng.randint(2020, 2026))
        quarter = rng.choice(["Q1", "Q2", "Q3", "Q4"])
        month = rng.choice([
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ])
        pct = f"{rng.uniform(1, 15):.1f}"

        subs = {
            "domain": domain, "doc_type": doc_type, "company": company,
            "person": person, "dept": dept, "product": product,
            "currency": currency, "amount": amount, "amount2": amount2,
            "amount3": amount3, "qty": qty, "year": year,
            "quarter": quarter, "month": month, "pct": pct,
        }

        try:
            query = tpl["query"].format(**subs)
            chosen_reasoning = tpl["chosen_reasoning"].format(**subs)
            chosen_answer = tpl["chosen_answer"].format(**subs)
            rejected_reasoning = tpl["rejected_reasoning"].format(**subs)
            rejected_answer = tpl["rejected_answer"].format(**subs)
        except (KeyError, IndexError):
            idx += 1
            continue

        results.append(format_dpo_example(
            query, chosen_reasoning, chosen_answer,
            rejected_reasoning, rejected_answer,
        ))
        idx += 1

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_conversational_dpo(count: int = 3000, *, seed: int = 80) -> List[Dict[str, str]]:
    """Generate conversational DPO preference pairs.

    Args:
        count: Number of pairs to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts with prompt/chosen/rejected keys.
    """
    rng = random.Random(seed)
    results = _expand_conversational_templates(_CONVERSATIONAL_TEMPLATES, count, rng)
    rng.shuffle(results)
    return results


def generate_post_conversational_data(output_dir: Path, scale: float = 1.0) -> int:
    """Generate post-training conversational DPO data and write to JSONL.

    Args:
        output_dir: Directory to write the JSONL file into.
        scale: Scaling factor (1.0 = 3K pairs).

    Returns:
        Number of examples written.
    """
    output_dir = Path(output_dir)
    count = max(1, int(3000 * scale))
    examples = generate_conversational_dpo(count=count)

    path = output_dir / "post_conversational_dpo.jsonl"
    with JSONLWriter(path) as writer:
        for ex in examples:
            writer.write(ex)

    return len(examples)
