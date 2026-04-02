"""Post-training Confidence Calibration SFT data generator for DocWain V2+.

Generates 2K confidence calibration examples (at scale=1.0) across 4 tiers:
  - High confidence    (40% = 800)
  - Medium confidence  (30% = 600)
  - Low confidence     (20% = 400)
  - Refusal            (10% = 200)

Each example includes per-source evidence assessment in ``<think>`` and
an explicit confidence statement in the answer.
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
# Constants
# ---------------------------------------------------------------------------

CONFIDENCE_TIERS = ["high", "medium", "low", "refusal"]

_TIER_FRACTIONS = {
    "high": 0.40,
    "medium": 0.30,
    "low": 0.20,
    "refusal": 0.10,
}

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


def _pick(lst: list, rng: random.Random) -> Any:
    return rng.choice(lst)


def _rand_amount(rng: random.Random) -> str:
    return f"{rng.uniform(100.0, 99999.0):,.2f}"


# ---------------------------------------------------------------------------
# Templates per tier
# ---------------------------------------------------------------------------

_HIGH_TEMPLATES: List[Dict[str, str]] = [
    {
        "query": (
            "What is the total value of the {company} {doc_type}?\n\n"
            "Document states: 'Total Contract Value: {currency}{amount}'"
        ),
        "reasoning": (
            "Evidence Assessment:\n"
            "- Source 1: {domain} {doc_type} from {company}\n"
            "  Relevance: Direct match — contains explicit 'Total Contract Value' field.\n"
            "  Quality: High — primary source document with clear, unambiguous value.\n"
            "  Extraction: {currency}{amount} stated verbatim.\n\n"
            "Confidence calibration: Single authoritative source with explicit value. "
            "No conflicting data. Direct textual match.\n"
            "Confidence: High"
        ),
        "answer": (
            "The total value of the {company} {doc_type} is **{currency}{amount}**.\n\n"
            "**Confidence: High** — This value is stated explicitly in the document "
            "as 'Total Contract Value: {currency}{amount}'. Single authoritative source, "
            "no ambiguity."
        ),
    },
    {
        "query": (
            "Who signed the {company} agreement?\n\n"
            "Signature block reads: 'Authorized Signatory: {person}, {dept} Director'"
        ),
        "reasoning": (
            "Evidence Assessment:\n"
            "- Source 1: Signature block of {company} agreement\n"
            "  Relevance: Direct — signature block is the authoritative source for signatory.\n"
            "  Quality: High — formal signature field.\n"
            "  Extraction: {person}, {dept} Director.\n\n"
            "Confidence calibration: Signature blocks are definitive. Name and title "
            "are explicitly stated.\n"
            "Confidence: High"
        ),
        "answer": (
            "The {company} agreement was signed by **{person}**, {dept} Director.\n\n"
            "**Confidence: High** — Extracted directly from the formal signature block."
        ),
    },
    {
        "query": (
            "What department owns the {product} project?\n\n"
            "Project charter states: 'Owning Department: {dept}'"
        ),
        "reasoning": (
            "Evidence Assessment:\n"
            "- Source 1: Project charter for {product}\n"
            "  Relevance: Direct — charter defines ownership.\n"
            "  Quality: High — authoritative governance document.\n"
            "  Extraction: {dept} stated as owning department.\n\n"
            "Confidence calibration: Project charters are the source of truth for ownership.\n"
            "Confidence: High"
        ),
        "answer": (
            "The {product} project is owned by the **{dept}** department.\n\n"
            "**Confidence: High** — Stated explicitly in the project charter."
        ),
    },
]

_MEDIUM_TEMPLATES: List[Dict[str, str]] = [
    {
        "query": (
            "What is the expected completion date for the {product} deployment?\n\n"
            "Source 1: Project plan says 'Target: {month} {year}'\n"
            "Source 2: Status report says 'Revised timeline under discussion'"
        ),
        "reasoning": (
            "Evidence Assessment:\n"
            "- Source 1: Project plan for {product}\n"
            "  Relevance: High — project plans define timelines.\n"
            "  Quality: Medium — may be outdated given Source 2.\n"
            "  Extraction: Target date = {month} {year}.\n\n"
            "- Source 2: Status report\n"
            "  Relevance: High — indicates timeline may have changed.\n"
            "  Quality: Medium — 'under discussion' is vague.\n"
            "  Extraction: Timeline is being revised (no new date given).\n\n"
            "Confidence calibration: Two sources partially conflict. Original target exists "
            "but may no longer be valid. No confirmed revised date available.\n"
            "Confidence: Medium"
        ),
        "answer": (
            "The original target completion date for {product} is **{month} {year}** "
            "per the project plan. However, the latest status report indicates the "
            "timeline is being revised.\n\n"
            "**Confidence: Medium** — The original date is documented but a revision "
            "is in progress. The actual completion date is currently uncertain."
        ),
    },
    {
        "query": (
            "How much did {company} spend on {dept} last year?\n\n"
            "Budget report shows {currency}{amount} allocated.\n"
            "Expense summary shows {currency}{amount2} spent."
        ),
        "reasoning": (
            "Evidence Assessment:\n"
            "- Source 1: Budget report — {currency}{amount} allocated to {dept}.\n"
            "  Relevance: Partial — allocation != actual spend.\n"
            "  Quality: High — official budget document.\n\n"
            "- Source 2: Expense summary — {currency}{amount2} spent.\n"
            "  Relevance: High — direct answer to spending question.\n"
            "  Quality: Medium — summary may not include all cost categories.\n\n"
            "Confidence calibration: Expense summary directly answers the question but "
            "may be incomplete. Discrepancy with budget allocation is expected.\n"
            "Confidence: Medium"
        ),
        "answer": (
            "Based on the expense summary, {company} spent **{currency}{amount2}** on "
            "{dept} last year, against a budget allocation of {currency}{amount}.\n\n"
            "**Confidence: Medium** — The expense summary provides a direct figure but "
            "may not capture all cost categories (e.g., shared services, overhead allocations)."
        ),
    },
]

_LOW_TEMPLATES: List[Dict[str, str]] = [
    {
        "query": (
            "What is {company}'s market share in the {domain} sector?\n\n"
            "Internal memo mentions 'we believe our share is approximately {pct}%' "
            "from {year}."
        ),
        "reasoning": (
            "Evidence Assessment:\n"
            "- Source 1: Internal memo from {year}\n"
            "  Relevance: Tangential — internal estimate, not market research.\n"
            "  Quality: Low — uses 'approximately' and 'we believe' (hedged language).\n"
            "  Freshness: Potentially outdated (from {year}).\n"
            "  Extraction: ~{pct}% estimated market share.\n\n"
            "Confidence calibration: Single source with hedged language, potentially outdated, "
            "no corroborating market data. Internal estimates often diverge from reality.\n"
            "Confidence: Low"
        ),
        "answer": (
            "An internal memo from {year} estimated {company}'s market share at "
            "approximately {pct}% in the {domain} sector.\n\n"
            "**Confidence: Low** — This is a self-reported internal estimate using "
            "hedged language ('approximately', 'we believe'). It may be outdated and "
            "is not corroborated by independent market research. Treat as directional only."
        ),
    },
    {
        "query": (
            "Will {company} renew the {product} contract?\n\n"
            "Email from {person}: 'We're leaning towards renewal but nothing is decided.'"
        ),
        "reasoning": (
            "Evidence Assessment:\n"
            "- Source 1: Email from {person}\n"
            "  Relevance: Moderate — expresses intent but not commitment.\n"
            "  Quality: Low — informal communication, non-binding.\n"
            "  Extraction: Leaning towards renewal, not decided.\n\n"
            "Confidence calibration: Single informal source expressing tentative intent. "
            "No contractual or board-level confirmation.\n"
            "Confidence: Low"
        ),
        "answer": (
            "Based on an email from {person}, {company} is 'leaning towards' renewing "
            "the {product} contract, but no decision has been made.\n\n"
            "**Confidence: Low** — This is based on a single informal email expressing "
            "tentative intent. No formal approval or board decision is documented."
        ),
    },
]

_REFUSAL_TEMPLATES: List[Dict[str, str]] = [
    {
        "query": (
            "What are {company}'s plans for next year's {domain} strategy?"
        ),
        "reasoning": (
            "Evidence Assessment:\n"
            "- Source search: No documents found containing {company}'s forward-looking "
            "{domain} strategy for next year.\n"
            "- Available documents cover historical data only.\n\n"
            "Confidence calibration: Zero relevant sources. Answering would require "
            "speculation, which violates DocWain's grounding principle.\n"
            "Confidence: Insufficient — cannot answer"
        ),
        "answer": (
            "I don't have any documents containing {company}'s forward-looking {domain} "
            "strategy plans. The available documents cover historical data only.\n\n"
            "**Confidence: Insufficient** — I cannot provide an answer without source "
            "material. Speculating on future strategy would be unreliable. If you have "
            "a strategy document to upload, I'd be happy to analyze it."
        ),
    },
    {
        "query": (
            "What did {person} say in the {dept} meeting last Tuesday?"
        ),
        "reasoning": (
            "Evidence Assessment:\n"
            "- Source search: No meeting minutes, transcripts, or notes found for the "
            "referenced {dept} meeting.\n"
            "- {person} appears in other documents but not in meeting records.\n\n"
            "Confidence calibration: No source material available for this specific event. "
            "Cannot fabricate meeting content.\n"
            "Confidence: Insufficient — cannot answer"
        ),
        "answer": (
            "I don't have any meeting minutes or notes from the {dept} meeting you're "
            "referring to. {person} appears in other documents but there are no records "
            "of that specific meeting in the system.\n\n"
            "**Confidence: Insufficient** — No source material available. If meeting "
            "minutes exist, please upload them and I can analyze the content."
        ),
    },
    {
        "query": (
            "Is {company} a good investment based on their {doc_type}?"
        ),
        "reasoning": (
            "Evidence Assessment:\n"
            "- Source 1: {company} {doc_type} available — contains {domain} operational data.\n"
            "  Relevance: Partial — operational documents are one input to investment decisions.\n"
            "  Quality: N/A for this question — investment advice requires comprehensive "
            "  financial analysis, market conditions, risk assessment, and regulatory context "
            "  that go far beyond a single document.\n\n"
            "Confidence calibration: Even with the document available, providing investment "
            "advice from a single operational document would be irresponsible and misleading.\n"
            "Confidence: Insufficient — outside scope"
        ),
        "answer": (
            "I can analyze the contents of {company}'s {doc_type} and extract factual "
            "information, but I cannot provide investment advice. Investment decisions "
            "require comprehensive financial analysis, market context, and risk assessment "
            "that go beyond a single {domain} document.\n\n"
            "**Confidence: Insufficient (out of scope)** — I can share factual findings "
            "from the document, but the investment question itself requires professional "
            "financial advisory expertise."
        ),
    },
]

# Map tier -> templates
_TIER_TEMPLATE_MAP = {
    "high": _HIGH_TEMPLATES,
    "medium": _MEDIUM_TEMPLATES,
    "low": _LOW_TEMPLATES,
    "refusal": _REFUSAL_TEMPLATES,
}


# ---------------------------------------------------------------------------
# Template expansion
# ---------------------------------------------------------------------------

def _expand_confidence_templates(
    templates: List[Dict[str, str]],
    target_count: int,
    rng: random.Random,
) -> List[Dict[str, str]]:
    """Expand confidence calibration templates with random variation."""
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
        qty = str(rng.randint(5, 200))
        year = str(rng.randint(2020, 2026))
        month = rng.choice([
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ])
        pct = f"{rng.uniform(5, 35):.1f}"

        subs = {
            "domain": domain, "doc_type": doc_type, "company": company,
            "person": person, "dept": dept, "product": product,
            "currency": currency, "amount": amount, "amount2": amount2,
            "qty": qty, "year": year, "month": month, "pct": pct,
        }

        try:
            query = tpl["query"].format(**subs)
            reasoning = tpl["reasoning"].format(**subs)
            answer = tpl["answer"].format(**subs)
        except (KeyError, IndexError):
            idx += 1
            continue

        results.append(format_sft_example(query, reasoning, answer))
        idx += 1

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_confidence_examples(count: int = 2000, *, seed: int = 85) -> List[Dict[str, str]]:
    """Generate confidence calibration SFT examples across 4 tiers.

    Distribution: 40% high, 30% medium, 20% low, 10% refusal.

    Args:
        count: Total number of examples to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of SFT examples with ``text`` key containing confidence statements.
    """
    rng = random.Random(seed)
    results: List[Dict[str, str]] = []

    for tier in CONFIDENCE_TIERS:
        tier_count = max(1, int(count * _TIER_FRACTIONS[tier]))
        templates = _TIER_TEMPLATE_MAP[tier]
        results.extend(_expand_confidence_templates(templates, tier_count, rng))

    rng.shuffle(results)
    return results[:count]


def generate_post_confidence_data(output_dir: Path, scale: float = 1.0) -> int:
    """Generate post-training confidence calibration data and write to JSONL.

    Args:
        output_dir: Directory to write the JSONL file into.
        scale: Scaling factor (1.0 = 2K examples).

    Returns:
        Number of examples written.
    """
    output_dir = Path(output_dir)
    count = max(1, int(2000 * scale))
    examples = generate_confidence_examples(count=count)

    path = output_dir / "post_confidence_calibration.jsonl"
    with JSONLWriter(path) as writer:
        for ex in examples:
            writer.write(ex)

    return len(examples)
