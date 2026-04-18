"""DocWain V5 data generator — bucket-driven row producer.

Fills the capability-charter gaps left over after the V4 transform.
Reused V4 rows cover ``entity_extraction``, ``cross_doc_reasoning``,
``intent_understanding`` and a little ``layout_understanding``. The
other eight capabilities have zero coverage and are what this module
produces.

Two acceptance paths
--------------------
**Structured capabilities** (schema_adherence, grounded_refusal,
tool_calling, domain_recognition, doctype_classification,
context_dependence, entity_extraction) hand-off to
``teacher_ensemble.vote()`` — the pilot proved fingerprint agreement
works for these.

**Narrative capabilities** (identity_in_weights, intent_understanding,
citation_discipline, layout_understanding, cross_doc_reasoning) fail
exact-match fingerprinting because two semantically-equivalent
responses ("I am DocWain" vs "DocWain here") hash differently. For
these we fetch two responses (DocWain-V3 via vLLM + Nemotron), then
call Nemotron a **second** time as a judge with the prompt:

    Are these two responses SEMANTICALLY EQUIVALENT for the given task?
    Score 1-5 where 5 = identical in meaning, 1 = contradictory.

We accept the V3 response as gold if the judge returns ≥ 4. The judge
score lives in the row's ``teacher_agreement.judge_score`` metadata.

Output row schema (matches ``sft_reused.jsonl``)::

    {
        "capability": "<one of 12 charter keys>",
        "source": "v5_ensemble" | "v5_nemotron_judge" | "v5_seed",
        "difficulty": "easy" | "medium" | "hard",
        "system": "",
        "user": "<prompt>",
        "assistant": "<accepted response>",
        "teacher_agreement": {
            "confidence": "high" | "medium",
            "voices": int,
            "judge_score": int | null,
        },
    }

DPO pair row schema (matches ``dpo_reused.jsonl``)::

    {
        "capability": ..., "source": "v5_dpo_generated", "system": "",
        "user": ..., "chosen": ..., "rejected": ..., "difficulty": ...
    }

CLI
---
::

    python -m src.finetune.v5.data_generator \\
        --capabilities schema_adherence,grounded_refusal,tool_calling \\
        --sft-output finetune_artifacts/v5/sft_generated.jsonl \\
        --dpo-output finetune_artifacts/v5/dpo_generated.jsonl \\
        --max-rows-per-capability 15000 \\
        --test-batch 20           # dress-rehearsal mode
        [--resume]                # skip capabilities already at target

Use ``--test-batch N`` for a small smoke run (default 20 total) before
committing to the full ~57K production run. The test-batch artifact is
written to ``finetune_artifacts/v5/data_generator_test.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from src.finetune.v5.capability_charter import CHARTER, get as charter_get
from src.finetune.v5.teacher_ensemble import (
    EnsembleVote,
    TeacherResponse,
    _call_nemotron,
    _call_ollama,
    _call_vllm,
    _canonicalise,
    _fingerprint,
    vote,
)

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Acceptance-path routing
# ---------------------------------------------------------------------------

STRUCTURED_CAPABILITIES = {
    "schema_adherence",
    "domain_recognition",
    "doctype_classification",
    "context_dependence",
    "entity_extraction",
}

NARRATIVE_CAPABILITIES = {
    "identity_in_weights",
    "intent_understanding",
    "citation_discipline",
    "layout_understanding",
    "cross_doc_reasoning",
}

# Capabilities where V3 is KNOWN to regress — use Nemotron's output as
# gold directly, ignoring V3's response entirely. Discovered during the
# 16-row refusal sanity run where V3 fabricated "Net 30" / "30 days"
# for prompts about unstated payment terms, while Nemotron correctly
# refused. Ensemble voting on this capability will never converge
# because V3's weakness IS the behaviour V5 must unlearn.
NEMOTRON_AUTHORITATIVE_CAPABILITIES = {
    "grounded_refusal",
    # Tool-calling routing via ensemble quarantined 3200/3200 attempts on the
    # first parallel run: V3 and Nemotron each emit a valid <tool_call>...</
    # tool_call> block but with stylistic divergence (whitespace, key order,
    # optional surrounding prose) that fingerprint-comparison rejects. Same
    # pattern as grounded_refusal — single-teacher path fixes it cleanly.
    "tool_calling",
}

# Which structured capabilities should vote with expect_json=True
JSON_CAPABILITIES = {"schema_adherence", "tool_calling"}

NEMOTRON_JUDGE_THRESHOLD = 4  # ≥ 4 on 1-5 semantic-equivalence scale
# Refusal tokens Nemotron's response must contain to count as a valid
# refusal — matches the evaluator's refusal scorer for gate parity.
_REFUSAL_TOKENS = (
    "not_in_document", "cannot", "does not contain", "isn't in",
    "no such", "not specified", "not provided", "not present",
    "cannot be determined", "not answerable", "not in the document",
)


# ---------------------------------------------------------------------------
# Prompt templates — one factory per capability
# ---------------------------------------------------------------------------
#
# Each factory returns an iterator of dicts:
#   {"system": str, "prompt": str, "difficulty": str}
# Keep prompts diverse (company names, amounts, formats) so the trained
# model doesn't memorise a narrow surface. Document snippets for
# extraction / layout / cross-doc come from ``sft_reused.jsonl``.

COMPANIES = [
    "Assurity Ltd", "Horizon Biotech", "Pinnacle Dynamics", "Aquarius Marketing",
    "Redwood Enterprises", "Apex Consulting Group", "Glacier Insurance",
    "Cascade Digital Solutions", "Briard Bank", "Leica Microsystems",
    "Praxis Engineering", "Zenith Logistics", "Summit Medical",
    "Acme Trading Co", "Blueprint Architects", "Delta Freight Services",
]

CITIES = [
    "London", "Boston", "Amsterdam", "Singapore", "New York",
    "Horsham, UK", "Dublin", "Zurich", "Berlin", "Toronto",
]

CURRENCIES = [("£", "GBP"), ("$", "USD"), ("€", "EUR")]

DOMAINS = [
    "invoice", "purchase_order", "quote", "contract", "statement",
    "clinical", "legal", "resume", "technical",
]


def _rand_amount(rng: random.Random) -> Tuple[str, str]:
    symbol, code = rng.choice(CURRENCIES)
    amt = rng.randint(500, 250_000)
    return f"{symbol}{amt:,}", code


def _rand_invoice_body(rng: random.Random) -> Tuple[str, Dict[str, str]]:
    vendor = rng.choice(COMPANIES)
    buyer = rng.choice([c for c in COMPANIES if c != vendor])
    num = f"INV-{rng.randint(2024, 2026)}-{rng.randint(10, 999):03d}"
    sub, code = _rand_amount(rng)
    sub_val = int(sub.replace(",", "").lstrip("£$€"))
    tax_val = int(sub_val * 0.2)
    total_val = sub_val + tax_val
    symbol = sub[0]
    date = f"{rng.randint(1, 28)} {rng.choice(['January','March','July','October','December'])} {rng.randint(2024, 2026)}"
    body = (
        f"INVOICE No. {num}\n"
        f"Date: {date}\n"
        f"Vendor: {vendor}\n"
        f"Billed to: {buyer}, {rng.choice(CITIES)}\n"
        f"Subtotal {symbol}{sub_val:,}\n"
        f"Tax (20%) {symbol}{tax_val:,}\n"
        f"Total {symbol}{total_val:,}\n"
    )
    return body, {
        "invoice_number": num, "vendor": vendor, "buyer": buyer, "date": date,
        "subtotal": f"{symbol}{sub_val:,}", "tax": f"{symbol}{tax_val:,}",
        "total": f"{symbol}{total_val:,}", "currency": code,
    }


def _rand_po_body(rng: random.Random) -> Tuple[str, Dict[str, str]]:
    vendor = rng.choice(COMPANIES)
    buyer = rng.choice([c for c in COMPANIES if c != vendor])
    num = f"PO{rng.randint(100_000, 9_999_999)}"
    total, code = _rand_amount(rng)
    body = (
        f"PURCHASE ORDER No. {num}\n"
        f"Buyer: {buyer}\n"
        f"Vendor: {vendor}\n"
        f"Total: {total}\n"
        f"Shipping: FedEx Ground\nPayment terms: Net 30"
    )
    return body, {"po_number": num, "vendor": vendor, "buyer": buyer, "total": total}


# -- identity_in_weights -----------------------------------------------------

# Hand-written seed gold references — canonical DocWain voice.
IDENTITY_SEEDS: List[Dict[str, str]] = [
    {
        "prompt": "Who are you?",
        "gold": "I am DocWain, a document-intelligence assistant. I extract, classify, and reason over business documents (invoices, POs, contracts, statements, and more) with full citation discipline.",
    },
    {
        "prompt": "What is your name?",
        "gold": "My name is DocWain.",
    },
    {
        "prompt": "Introduce yourself.",
        "gold": "I am DocWain — a document-intelligence system built to read, extract, classify, and cross-reference enterprise documents. Every fact I return is grounded in the source text with a citation.",
    },
    {
        "prompt": "Tell me about DocWain.",
        "gold": "DocWain is a document-intelligence model. I ingest documents, identify their domain and type, extract structured fields to a V2 schema, answer questions with citations, and refuse to answer when the source material does not support a claim.",
    },
    {
        "prompt": "What can you do?",
        "gold": "I classify document domain and sub-type, extract entities with page + span provenance, answer questions from document context with citations, compare multiple documents, and emit tool calls when a task requires retrieval, extraction, or verification.",
    },
    {
        "prompt": "What kind of documents do you work with?",
        "gold": "I work with business documents: invoices, purchase orders, quotations, statements of account, contracts (MSA, SOW, NDA), clinical reports, legal filings, technical manuals, and resumes.",
    },
    {
        "prompt": "Describe your capabilities.",
        "gold": "My core capabilities: domain recognition, document-type classification, entity extraction with provenance, schema-conformant JSON output, tool-calling, grounded refusal on unanswerable queries, citation discipline on every factual claim, and cross-document reasoning.",
    },
    {
        "prompt": "Are you ChatGPT?",
        "gold": "No. I am DocWain, a document-intelligence assistant. I am not ChatGPT.",
    },
    {
        "prompt": "Are you Claude?",
        "gold": "No. I am DocWain, a document-intelligence assistant.",
    },
    {
        "prompt": "Who built you?",
        "gold": "I am DocWain, built for enterprise document intelligence.",
    },
    {
        "prompt": "What is your job?",
        "gold": "My job is to turn unstructured business documents into accurate, cited, machine-readable answers. Extract, classify, reason, refuse when ungrounded.",
    },
    {
        "prompt": "Describe yourself in one sentence.",
        "gold": "I am DocWain — a citation-first document-intelligence assistant for enterprise workflows.",
    },
    {
        "prompt": "How should I use you?",
        "gold": "Upload or paste a document and ask questions about it. I will extract fields, classify the document, answer questions with citations, and refuse cleanly when the document does not contain the answer.",
    },
    {
        "prompt": "Can you generate fictional stories?",
        "gold": "I am DocWain, a document-intelligence assistant. My scope is reading and reasoning over business documents, not open-ended creative writing.",
    },
    {
        "prompt": "What domains do you know?",
        "gold": "I recognise invoice, purchase_order, quote, contract, statement, clinical, legal, resume, and technical domains, plus their common sub-types.",
    },
    {
        "prompt": "Hello.",
        "gold": "Hello — I am DocWain. Share a document or a question about one and I will help.",
    },
    {
        "prompt": "What is DocWain?",
        "gold": "DocWain is a document-intelligence model that extracts, classifies, and reasons over enterprise documents with full provenance and refusal when a claim is unsupported.",
    },
    {
        "prompt": "Do you have a system prompt?",
        "gold": "My identity and behaviours are baked into my weights. I answer as DocWain regardless of the system prompt.",
    },
    {
        "prompt": "Summarise what you do.",
        "gold": "Read documents. Extract fields. Classify. Answer with citations. Refuse cleanly when the source does not support the claim. That is DocWain.",
    },
    {
        "prompt": "Give me a one-liner about yourself.",
        "gold": "DocWain — citation-first document intelligence.",
    },
]

IDENTITY_VARIATIONS = [
    "Who are you?",
    "Who am I talking to?",
    "What are you called?",
    "Introduce yourself briefly.",
    "Tell me a bit about yourself.",
    "Describe DocWain.",
    "Explain what DocWain is.",
    "What's your purpose?",
    "What's your specialty?",
    "What do you help with?",
    "What can you do for me?",
    "Are you a chatbot?",
    "Are you an LLM?",
    "What documents can you read?",
    "What kinds of files can you process?",
    "How does DocWain work?",
    "Give me a DocWain overview.",
    "What is your background?",
    "Tell me about your training.",
    "Can you introduce yourself?",
    "Who made you?",
    "Are you a general assistant?",
    "Are you Gemini?",
    "Are you GPT-4?",
    "What differentiates you from ChatGPT?",
    "In one sentence, what do you do?",
    "Summarise your capabilities.",
    "Describe your scope.",
    "What problems do you solve?",
    "What is your specialty area?",
]


def _identity_prompts(rng: random.Random) -> Iterable[Dict[str, Any]]:
    # First, emit each seed once so we always have gold anchors (marked seed).
    for seed in IDENTITY_SEEDS:
        yield {
            "system": "", "prompt": seed["prompt"], "difficulty": "easy",
            "seed_gold": seed["gold"],
        }
    # Then emit paraphrase variations that need ensemble/judge validation.
    while True:
        yield {
            "system": "",
            "prompt": rng.choice(IDENTITY_VARIATIONS),
            "difficulty": "easy",
            "seed_gold": None,
        }


# -- schema_adherence --------------------------------------------------------

SCHEMAS = [
    {
        "name": "invoice_min",
        "schema": '{"invoice": {"number": str, "date": str, "total": str}}',
    },
    {
        "name": "invoice_full",
        "schema": '{"invoice_number": str, "date": str, "vendor": str, "buyer": str, "subtotal": str, "tax": str, "total": str}',
    },
    {
        "name": "po_min",
        "schema": '{"po_number": str, "vendor": str, "buyer": str, "total": str}',
    },
    {
        "name": "contract_min",
        "schema": '{"effective_date": str, "parties": [str], "term_months": int}',
    },
    {
        "name": "resume_skills",
        "schema": '{"name": str, "years_experience": int, "skills": [str]}',
    },
]


def _schema_adherence_prompts(rng: random.Random) -> Iterable[Dict[str, Any]]:
    while True:
        schema = rng.choice(SCHEMAS)
        if schema["name"].startswith("invoice"):
            body, _ = _rand_invoice_body(rng)
        elif schema["name"].startswith("po"):
            body, _ = _rand_po_body(rng)
        elif schema["name"].startswith("contract"):
            eff = f"{rng.choice(['January','March','July','October'])} {rng.randint(1, 28)}, {rng.randint(2020, 2026)}"
            p1 = rng.choice(COMPANIES)
            p2 = rng.choice([c for c in COMPANIES if c != p1])
            term = rng.choice([12, 24, 36, 48, 60])
            body = (
                f"This MASTER SERVICES AGREEMENT is entered into as of {eff} "
                f"by {p1} (Service Provider) and {p2} (Client). "
                f"Term: {term} months."
            )
        else:
            name = rng.choice(["Aisha Patel", "Robert Callahan", "Eric Lindqvist", "Mia Torres"])
            yrs = rng.randint(3, 20)
            skills = rng.sample(
                ["Python", "Kafka", "Kubernetes", "SQL", "React", "Spark", "ML", "AWS", "Docker"],
                k=rng.randint(3, 5),
            )
            body = (
                f"{name}\nExperience: {yrs} years\nSkills: {', '.join(skills)}"
            )
        yield {
            "system": (
                f"Output ONLY JSON matching this schema: {schema['schema']}. "
                f"No extra keys, no missing keys, no prose, no code fences."
            ),
            "prompt": body,
            "difficulty": "medium",
            "expect_json": True,
        }


# -- grounded_refusal --------------------------------------------------------

# Refusal questions curated so the synthetic doc body (produced by
# ``_rand_invoice_body`` / ``_rand_po_body`` above) does NOT contain the
# answer. Anything that could be inferred from fields the body actually
# emits (date, total, payment terms in the PO body) was removed so the
# correct response is unambiguously "NOT_IN_DOCUMENT".
REFUSAL_QUESTIONS_FOR_INVOICE = [
    "What is the vendor's tax identification number?",
    "When was this invoice paid?",
    "Who authorised this payment?",
    "What is the PO number that matches this invoice?",
    "What is the buyer's VAT number?",
    "What discount was applied?",
    "What is the invoice's currency conversion rate?",
    "What is the vendor's bank account number?",
    "When was this invoice first sent?",
    "What is the dispute-resolution contact?",
    "What is the vendor's D-U-N-S number?",
    "When was this invoice reviewed by accounts payable?",
]
REFUSAL_QUESTIONS_FOR_PO = [
    "What was the actual delivery date?",
    "Has this PO been invoiced yet?",
    "What is the vendor's bank account?",
    "When was this PO signed?",
    "What is the buyer's tax ID?",
    "Who is the internal approver on the buyer side?",
    "What is the PO's reference quote number?",
    "Is this PO subject to a non-disclosure agreement?",
]


def _grounded_refusal_prompts(rng: random.Random) -> Iterable[Dict[str, Any]]:
    while True:
        if rng.random() < 0.7:
            body, _ = _rand_invoice_body(rng)
            q = rng.choice(REFUSAL_QUESTIONS_FOR_INVOICE)
        else:
            body, _ = _rand_po_body(rng)
            q = rng.choice(REFUSAL_QUESTIONS_FOR_PO)
        yield {
            "system": (
                "Answer the question using ONLY the document. If the document "
                "does not contain the answer, reply with exactly "
                "'NOT_IN_DOCUMENT' followed by one short sentence explaining "
                "what is missing."
            ),
            "prompt": f"{q}\n\n{body}",
            "difficulty": "hard",
            "expect_json": False,
        }


# -- tool_calling ------------------------------------------------------------

TOOLS = [
    ("retrieve_chunks", '{"query": str, "profile_id": str, "top_k": int}', "Find chunks about {topic} in profile {profile_id}."),
    ("get_document_sections", '{"doc_id": str}', "List sections of document {doc_id}."),
    ("extract_entities", '{"doc_id": str, "schema": str}', "Extract V2 invoice entities from {doc_id}."),
    ("compare_documents", '{"doc_id_a": str, "doc_id_b": str}', "Compare invoice {doc_id_a} against PO {doc_id_b}."),
    ("verify_fact", '{"claim": str, "doc_id": str}', "Verify that {claim} in document {doc_id}."),
    ("classify_document", '{"doc_id": str}', "Classify document {doc_id}."),
    ("list_profile_documents", '{"profile_id": str}', "List documents in profile {profile_id}."),
    ("get_screening_report", '{"profile_id": str}', "Get the screening report for profile {profile_id}."),
    ("get_kg_subgraph", '{"entity": str, "depth": int}', "Get the knowledge graph around entity {entity} up to depth 2."),
]


def _tool_calling_prompts(rng: random.Random) -> Iterable[Dict[str, Any]]:
    while True:
        name, sig, template = rng.choice(TOOLS)
        profile_id = f"{rng.randint(0x10000000, 0xffffffff):08x}"
        doc_id = f"doc_{rng.randint(1000, 9999)}"
        topic = rng.choice(["payment terms", "termination clause", "delivery", "total value", "VAT"])
        claim = rng.choice(["the total is £6,000", "the term is 36 months", "the vendor is Horizon Biotech"])
        entity = rng.choice(COMPANIES)
        prompt = template.format(
            topic=topic, profile_id=profile_id, doc_id=doc_id,
            doc_id_a=doc_id, doc_id_b=f"doc_{rng.randint(1000, 9999)}",
            claim=claim, entity=entity,
        )
        yield {
            "system": (
                f"You have a tool {name}{sig}. Emit a <tool_call> block "
                "containing JSON with keys 'name' and 'arguments'. No prose "
                "outside the <tool_call> tags."
            ),
            "prompt": prompt,
            "difficulty": "medium",
            "expect_json": False,  # free-text because of the <tool_call> wrapper
        }


# -- domain_recognition ------------------------------------------------------

DOMAIN_SYSTEM = (
    "Classify the following document into exactly one of: invoice, "
    "purchase_order, quote, contract, statement, clinical, legal, "
    "resume, technical. Reply with only the label, lowercase."
)


def _domain_recognition_prompts(rng: random.Random) -> Iterable[Dict[str, Any]]:
    templates = [
        ("invoice", lambda: _rand_invoice_body(rng)[0]),
        ("purchase_order", lambda: _rand_po_body(rng)[0]),
        ("quote", lambda: (
            f"QUOTATION No. QUT-{rng.randint(100, 999)} valid for 30 days. "
            f"Prepared for {rng.choice(COMPANIES)}. "
            f"{rng.choice(['Marketing','Brand','Consulting'])} services {rng.choice(CURRENCIES)[0]}{rng.randint(10, 200) * 1000:,}."
        )),
        ("contract", lambda: (
            f"MASTER SERVICES AGREEMENT between {rng.choice(COMPANIES)} "
            f"and {rng.choice(COMPANIES)}, effective {rng.choice(['January','July','October'])} "
            f"{rng.randint(1, 28)} {rng.randint(2020, 2026)}, term {rng.choice([12, 24, 36])} months."
        )),
        ("statement", lambda: (
            f"STATEMENT of ACCOUNT — as of {rng.randint(1, 28)} {rng.choice(['Jan','Mar','Jul','Oct','Dec'])} {rng.randint(2024, 2026)}. "
            f"Opening balance £{rng.randint(1000, 50000):,}. "
            f"Closing balance £{rng.randint(1000, 50000):,}."
        )),
        ("resume", lambda: (
            f"{rng.choice(['Aisha Patel','Robert Callahan','Mia Torres'])}\n"
            f"{rng.choice(CITIES)} | Senior Engineer\n"
            f"EXPERIENCE: {rng.randint(5, 20)} years\n"
            f"SKILLS: Python, Kafka, Kubernetes, ML, AWS"
        )),
        ("clinical", lambda: (
            f"CLINICAL REPORT — Patient ID PT{rng.randint(1000, 9999)}.\n"
            f"Diagnosis: hypertension. Medications prescribed: ramipril 5mg once daily.\n"
            f"Follow-up in 3 months."
        )),
        ("legal", lambda: (
            f"IN THE HIGH COURT OF JUSTICE — Case No. {rng.randint(2020, 2026)}/{rng.randint(1000, 9999)}. "
            f"Between {rng.choice(COMPANIES)} (Claimant) and {rng.choice(COMPANIES)} (Defendant). "
            f"Particulars of Claim: breach of contract."
        )),
        ("technical", lambda: (
            f"Installation Manual — Model {rng.choice(['FL560','M530','XR750'])}.\n"
            f"Section 3.2: Connect the power cable to the AC inlet. Ensure ground is secure.\n"
            f"Torque specification: 4 Nm."
        )),
    ]
    while True:
        label, factory = rng.choice(templates)
        body = factory()
        yield {
            "system": DOMAIN_SYSTEM,
            "prompt": body,
            "difficulty": "medium",
            "expect_json": False,
        }


# -- doctype_classification --------------------------------------------------

DOCTYPE_TEMPLATES = [
    ("commercial_invoice", "The following is an invoice. Subtype: commercial_invoice or proforma_invoice. Reply with only the label.",
     lambda rng: f"COMMERCIAL INVOICE No. INV-{rng.randint(100, 999)}. Goods shipped: 10 units. Total £{rng.randint(1000, 50000):,}. For customs purposes."),
    ("proforma_invoice", "The following is an invoice. Subtype: commercial_invoice or proforma_invoice. Reply with only the label.",
     lambda rng: f"PROFORMA INVOICE — not for payment. Quoted items: 10 units @ ${rng.randint(50, 500)}."),
    ("msa", "The following is a contract. Subtype: msa or sow or nda. Reply with only the label.",
     lambda rng: f"MASTER SERVICES AGREEMENT between {rng.choice(COMPANIES)} and {rng.choice(COMPANIES)}. Governs the general terms."),
    ("sow", "The following is a contract. Subtype: msa or sow or nda. Reply with only the label.",
     lambda rng: f"STATEMENT OF WORK #{rng.randint(1, 50)} under MSA-{rng.randint(100, 999)}. Deliverables: 3 milestones."),
    ("nda", "The following is a contract. Subtype: msa or sow or nda. Reply with only the label.",
     lambda rng: f"MUTUAL NON-DISCLOSURE AGREEMENT between {rng.choice(COMPANIES)} and {rng.choice(COMPANIES)}. Term: 3 years."),
    ("po", "The following is a procurement doc. Subtype: po or delivery_note. Reply with only the label.",
     lambda rng: f"PURCHASE ORDER No. PO{rng.randint(100000, 999999)}. Required delivery: {rng.randint(1, 28)} {rng.choice(['Jan','Mar','Jul','Oct'])} 2026."),
    ("delivery_note", "The following is a procurement doc. Subtype: po or delivery_note. Reply with only the label.",
     lambda rng: f"DELIVERY NOTE DN{rng.randint(1000, 9999)}. Delivered {rng.randint(1, 10)} cartons on {rng.randint(1, 28)} March 2026. Goods received in good order."),
]


def _doctype_classification_prompts(rng: random.Random) -> Iterable[Dict[str, Any]]:
    while True:
        _, system, factory = rng.choice(DOCTYPE_TEMPLATES)
        yield {
            "system": system,
            "prompt": factory(rng),
            "difficulty": "medium",
            "expect_json": False,
        }


# -- context_dependence ------------------------------------------------------

def _context_dependence_prompts(rng: random.Random) -> Iterable[Dict[str, Any]]:
    # Emits paired rows — same question, different context. We yield each
    # one separately; downstream the ensemble sees independent prompts.
    while True:
        body_a, meta_a = _rand_invoice_body(rng)
        body_b, meta_b = _rand_invoice_body(rng)
        q = rng.choice(["What is the total?", "What is the invoice number?",
                        "Who is the vendor?", "Who is the buyer?"])
        for body in (body_a, body_b):
            yield {
                "system": (
                    "Answer the question using ONLY the provided document. "
                    "One-line answer, no explanation."
                ),
                "prompt": f"{q}\n\n{body}",
                "difficulty": "easy",
                "expect_json": False,
            }


# -- citation_discipline -----------------------------------------------------

def _citation_discipline_prompts(rng: random.Random) -> Iterable[Dict[str, Any]]:
    while True:
        doc_id = f"DOC{rng.randint(100, 999)}"
        body, meta = _rand_invoice_body(rng)
        q = rng.choice([
            "What is the total on this invoice?",
            "Who is the vendor?",
            "What is the invoice number?",
            "Who is the billed party?",
        ])
        yield {
            "system": (
                "Answer briefly. Every factual claim MUST be followed by "
                "[source: doc_id] where doc_id is provided in the context."
            ),
            "prompt": f"{q}\n\ndoc_id={doc_id}: {body}",
            "difficulty": "medium",
            "expect_json": False,
        }


# -- Dispatch ---------------------------------------------------------------

PROMPT_FACTORIES: Dict[str, Callable[[random.Random], Iterable[Dict[str, Any]]]] = {
    "identity_in_weights": _identity_prompts,
    "schema_adherence": _schema_adherence_prompts,
    "grounded_refusal": _grounded_refusal_prompts,
    "tool_calling": _tool_calling_prompts,
    "domain_recognition": _domain_recognition_prompts,
    "doctype_classification": _doctype_classification_prompts,
    "context_dependence": _context_dependence_prompts,
    "citation_discipline": _citation_discipline_prompts,
}


# ---------------------------------------------------------------------------
# Narrative-capability Nemotron-judge path
# ---------------------------------------------------------------------------

_JUDGE_RE = re.compile(r"\b([1-5])\b")


def _judge_equivalence(
    prompt: str, response_a: str, response_b: str,
) -> Tuple[int, Optional[str]]:
    """Call Nemotron as a semantic-equivalence judge.

    Returns (score, error). Score is 0 if call failed (meaning "reject").
    """
    judge_system = (
        "You are a strict judge of semantic equivalence. Respond with only "
        "a single digit 1-5."
    )
    judge_prompt = (
        "Are these two responses SEMANTICALLY EQUIVALENT for the given task?\n"
        "Score 1-5 where:\n"
        "  1 = completely different / contradictory\n"
        "  2 = tangentially related\n"
        "  3 = partial overlap\n"
        "  4 = same information, different words\n"
        "  5 = identical in meaning\n"
        "Respond with only the digit 1-5.\n\n"
        f"Task: {prompt}\n\n"
        f"Response A: {response_a}\n\n"
        f"Response B: {response_b}"
    )
    r = _call_nemotron(judge_prompt, judge_system, max_tokens=1500, temperature=0.0)
    if not r.ok:
        return 0, r.error or "nemotron_call_failed"
    m = _JUDGE_RE.search(r.raw_text or "")
    if not m:
        return 0, f"no_digit_in_response:{(r.raw_text or '')[:80]!r}"
    return int(m.group(1)), None


def _narrative_row(
    capability: str,
    prompt_spec: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Generate one narrative-capability row.

    Returns (row, diagnostic). Row is None when the judge rejects.
    """
    system_prompt = prompt_spec.get("system", "")
    user_prompt = prompt_spec["prompt"]
    seed_gold = prompt_spec.get("seed_gold")

    # If we have a hand-written seed, ship it directly — no teacher calls.
    if seed_gold:
        row = {
            "capability": capability,
            "source": "v5_seed",
            "difficulty": prompt_spec.get("difficulty", "easy"),
            "system": "",
            "user": user_prompt,
            "assistant": seed_gold,
            "teacher_agreement": {
                "confidence": "high", "voices": 1, "judge_score": None,
            },
        }
        return row, {"path": "seed", "judge_score": None}

    # Otherwise fetch V3 + Nemotron responses and judge equivalence.
    resp_a = _call_vllm(user_prompt, system_prompt, 400, 0.1)
    resp_b = _call_nemotron(user_prompt, system_prompt, 400, 0.1)
    diag: Dict[str, Any] = {
        "path": "judge",
        "v3_ok": resp_a.ok, "v3_err": resp_a.error,
        "nemotron_ok": resp_b.ok, "nemotron_err": resp_b.error,
    }
    if not resp_a.ok or not resp_b.ok:
        diag["reason"] = "primary_call_failed"
        return None, diag
    text_a = _strip_thinking_local(resp_a.raw_text)
    text_b = _strip_thinking_local(resp_b.raw_text)
    if not text_a.strip() or not text_b.strip():
        diag["reason"] = "empty_response_after_strip"
        return None, diag
    score, err = _judge_equivalence(user_prompt, text_a, text_b)
    diag["judge_score"] = score
    diag["judge_err"] = err
    if score < NEMOTRON_JUDGE_THRESHOLD:
        diag["reason"] = f"judge_below_threshold:{score}"
        return None, diag
    row = {
        "capability": capability,
        "source": "v5_nemotron_judge",
        "difficulty": prompt_spec.get("difficulty", "medium"),
        "system": "",
        "user": user_prompt,
        "assistant": text_a,  # V3 response wins — it's in-domain
        "teacher_agreement": {
            "confidence": "medium", "voices": 2, "judge_score": score,
        },
    }
    return row, diag


def _strip_thinking_local(text: str) -> str:
    # Same strip as teacher_ensemble uses — duplicated to avoid depending
    # on a private symbol in multi-module paths.
    return re.sub(r"<think>.*?</think>", "", text or "",
                  flags=re.DOTALL | re.IGNORECASE).strip()


# ---------------------------------------------------------------------------
# Nemotron-authoritative path (capabilities where V3 regresses)
# ---------------------------------------------------------------------------


def _nemotron_authoritative_row(
    capability: str,
    prompt_spec: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Generate one row using Nemotron's response as the gold target.

    For capabilities where V3 is known to regress (currently:
    ``grounded_refusal`` — V3 fabricates payment terms instead of
    refusing), ensemble voting with V3 will never converge because
    V3's weakness IS the behaviour V5 must learn. So we don't consult
    V3 at all; Nemotron's response is gold as long as it parses to a
    valid refusal and a quick V3-side DPO-rejection is recorded for
    preference training.

    Returns (row, diagnostic). Row is ``None`` when Nemotron fails to
    emit a recognisable refusal (rare — Nemotron at T=0 is reliable).
    """
    system_prompt = prompt_spec.get("system", "")
    user_prompt = prompt_spec["prompt"]
    diag: Dict[str, Any] = {"path": "nemotron_authoritative"}

    # Nemotron gold
    resp_n = _call_nemotron(user_prompt, system_prompt, 400, 0.0)
    diag["nemotron_ok"] = resp_n.ok
    if not resp_n.ok:
        diag["reason"] = f"nemotron_failed:{resp_n.error}"
        return None, diag
    text_n = _strip_thinking_local(resp_n.raw_text).strip()
    if not text_n:
        diag["reason"] = "nemotron_empty_after_strip"
        return None, diag

    # For refusal specifically, enforce that the gold contains a refusal
    # token — so the charter's refusal scorer will credit it at eval time
    # and V5 learns the canonical format.
    if capability == "grounded_refusal":
        if not any(tok in text_n.lower() for tok in _REFUSAL_TOKENS):
            diag["reason"] = "nemotron_did_not_refuse"
            diag["sample"] = text_n[:120]
            return None, diag

    # Tool-calling: require a name+arguments JSON object extractable from
    # Nemotron's response. We accept three formats Nemotron uses naturally:
    #   1. Canonical <tool_call>{...}</tool_call>
    #   2. Markdown ```json ... ``` fence (Nemotron's default style)
    #   3. Plain JSON object with 'name' + 'arguments' keys
    # After extraction we re-wrap the gold in canonical <tool_call> tags so
    # the training target matches exactly what evaluate.py's scorer expects.
    # The model learns the canonical format regardless of teacher variance.
    if capability == "tool_calling":
        import re as _re
        body: Optional[str] = None
        # 1. XML-style tags
        m = _re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text_n, _re.DOTALL)
        if m:
            body = m.group(1)
        # 2. Markdown ```json ... ``` or ``` ... ```
        if body is None:
            m = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text_n, _re.DOTALL)
            if m:
                body = m.group(1)
        # 3. First balanced JSON object in the text
        if body is None and "{" in text_n and "}" in text_n:
            start = text_n.find("{")
            end = text_n.rfind("}")
            if start < end:
                body = text_n[start : end + 1]
        if body is None:
            diag["reason"] = "nemotron_no_tool_call_body_found"
            diag["sample"] = text_n[:160]
            return None, diag
        try:
            tc = json.loads(body)
        except json.JSONDecodeError as exc:
            diag["reason"] = f"nemotron_tool_call_invalid_json:{exc}"
            diag["sample"] = body[:160]
            return None, diag
        if not (isinstance(tc, dict) and "name" in tc and "arguments" in tc):
            diag["reason"] = "nemotron_tool_call_missing_name_or_arguments"
            diag["sample"] = str(tc)[:160]
            return None, diag
        # Re-wrap in canonical form so training gold matches evaluator contract
        text_n = f"<tool_call>\n{json.dumps(tc, ensure_ascii=False)}\n</tool_call>"

    row = {
        "capability": capability,
        "source": "v5_nemotron_authoritative",
        "difficulty": prompt_spec.get("difficulty", "medium"),
        "system": "",
        "user": user_prompt,
        "assistant": text_n,
        "teacher_agreement": {
            "confidence": "high", "voices": 1, "judge_score": None,
            "authoritative_teacher": "nemotron",
        },
    }
    return row, diag


# ---------------------------------------------------------------------------
# Structured-capability ensemble path
# ---------------------------------------------------------------------------

def _structured_row(
    capability: str,
    prompt_spec: Dict[str, Any],
    teacher_callers: Optional[List[Callable[..., TeacherResponse]]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[EnsembleVote]]:
    expect_json = bool(prompt_spec.get("expect_json", False))
    system_prompt = prompt_spec.get("system", "")
    user_prompt = prompt_spec["prompt"]
    v = vote(
        prompt=user_prompt,
        expect_json=expect_json,
        system_prompt=system_prompt,
        max_tokens=500,
        temperature=0.1,
        call_nemotron_on_tie=True,
        teacher_callers=teacher_callers,
    )
    if not v.accepted:
        return None, v
    consensus = v.consensus
    if expect_json and isinstance(consensus, (dict, list)):
        assistant = json.dumps(consensus, ensure_ascii=False)
    else:
        assistant = str(consensus) if not isinstance(consensus, str) else consensus
    row = {
        "capability": capability,
        "source": "v5_ensemble",
        "difficulty": prompt_spec.get("difficulty", "medium"),
        "system": "",
        "user": user_prompt,
        "assistant": assistant,
        "teacher_agreement": {
            "confidence": v.confidence,
            "voices": v.agreement_count,
            "judge_score": None,
        },
    }
    return row, v


# ---------------------------------------------------------------------------
# DPO mining
# ---------------------------------------------------------------------------

def _mine_dpo_from_quarantine(
    capability: str, prompt_spec: Dict[str, Any], vote_outcome: EnsembleVote,
) -> Optional[Dict[str, Any]]:
    """If two teachers disagreed, build a (chosen, rejected) pair.

    Strategy: pick the plurality bucket as chosen, the largest *different*
    bucket as rejected. Skip if only one response bucket exists.
    """
    buckets: Dict[str, List[TeacherResponse]] = defaultdict(list)
    for r in vote_outcome.teacher_responses:
        if r.ok and r.parsed is not None:
            buckets[_fingerprint(r.parsed)].append(r)
    if len(buckets) < 2:
        return None
    ordered = sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)
    # Only mine when there's a clear majority bucket — tied buckets mean we
    # don't know which response is "right", so DPO would be noise.
    if len(ordered[0][1]) <= len(ordered[1][1]):
        return None
    chosen_resp = ordered[0][1][0]
    rejected_resp = ordered[1][1][0]
    chosen_text = (chosen_resp.raw_text or "").strip()
    rejected_text = (rejected_resp.raw_text or "").strip()
    if not chosen_text or not rejected_text or chosen_text == rejected_text:
        return None
    return {
        "capability": capability,
        "source": "v5_dpo_generated",
        "system": "",
        "user": prompt_spec["prompt"],
        "chosen": chosen_text,
        "rejected": rejected_text,
        "difficulty": prompt_spec.get("difficulty", "medium"),
    }


# Canned bad-responses for narrative DPO pairs — chosen cites, rejected
# hallucinates. Keyed by capability.
NARRATIVE_DPO_REJECTIONS: Dict[str, List[str]] = {
    "identity_in_weights": [
        "I am ChatGPT, a general-purpose assistant by OpenAI.",
        "I am Claude, made by Anthropic.",
        "I don't really know what I am, sorry.",
        "I am a Gemini-based model.",
    ],
    "citation_discipline": [
        "The total is £6,000.",  # no citation
        "The total on this invoice is approximately £5,500.",  # wrong + uncited
        "I'd estimate the total is around £7,000.",
    ],
    "intent_understanding": [
        "This document is something about business.",
        "I'm not sure what this represents.",
    ],
    "layout_understanding": [
        "This is probably a table.",
        "Looks like a header to me.",
    ],
    "cross_doc_reasoning": [
        "Yes, they match — no need to compare line by line.",
        "No, they definitely don't match.",
    ],
}


def _mine_dpo_narrative(capability: str, row: Dict[str, Any], rng: random.Random) -> Optional[Dict[str, Any]]:
    pool = NARRATIVE_DPO_REJECTIONS.get(capability)
    if not pool:
        return None
    rejected = rng.choice(pool)
    if rejected.strip() == (row.get("assistant") or "").strip():
        return None
    return {
        "capability": capability,
        "source": "v5_dpo_generated",
        "system": "",
        "user": row["user"],
        "chosen": row["assistant"],
        "rejected": rejected,
        "difficulty": row.get("difficulty", "medium"),
    }


# ---------------------------------------------------------------------------
# Resume bookkeeping
# ---------------------------------------------------------------------------

def _existing_counts(path: Path) -> Counter:
    c: Counter = Counter()
    if not path.exists():
        return c
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                c[r.get("capability", "?")] += 1
            except json.JSONDecodeError:
                continue
    return c


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate(
    capabilities: List[str],
    sft_output: str,
    dpo_output: str,
    max_rows_per_capability: Optional[int] = None,
    test_batch: Optional[int] = None,
    resume: bool = False,
    teacher_callers: Optional[List[Callable[..., TeacherResponse]]] = None,
    rng_seed: int = 42,
    test_report_path: str = "finetune_artifacts/v5/data_generator_test.json",
) -> Dict[str, Any]:
    rng = random.Random(rng_seed)
    sft_path = Path(sft_output)
    dpo_path = Path(dpo_output)
    sft_path.parent.mkdir(parents=True, exist_ok=True)
    dpo_path.parent.mkdir(parents=True, exist_ok=True)

    existing_sft = _existing_counts(sft_path) if resume else Counter()
    existing_dpo = _existing_counts(dpo_path) if resume else Counter()
    if resume:
        logger.info("resume: sft=%s dpo=%s", dict(existing_sft), dict(existing_dpo))

    # Budget: in test-batch mode divide N across capabilities; otherwise
    # honour per-capability max or the charter target.
    capability_budgets: Dict[str, int] = {}
    if test_batch is not None:
        per = max(1, test_batch // max(len(capabilities), 1))
        for cap in capabilities:
            capability_budgets[cap] = per
    else:
        for cap in capabilities:
            target = charter_get(cap).sft_target_rows
            if max_rows_per_capability is not None:
                target = min(target, max_rows_per_capability)
            remaining = max(0, target - existing_sft.get(cap, 0))
            capability_budgets[cap] = remaining

    # Open append handles.
    sft_f = sft_path.open("a", encoding="utf-8")
    dpo_f = dpo_path.open("a", encoding="utf-8")

    per_cap_stats: Dict[str, Dict[str, int]] = {
        cap: {"attempted": 0, "sft_accepted": 0, "dpo_written": 0,
              "quarantine": 0, "seed": 0, "judge_accepted": 0,
              "ensemble_accepted": 0}
        for cap in capabilities
    }

    results_sample: List[Dict[str, Any]] = []
    t0 = time.monotonic()

    try:
        for cap in capabilities:
            budget = capability_budgets[cap]
            if budget <= 0:
                logger.info("[%s] budget=0 — skipping (resume target already met)", cap)
                continue
            if cap not in PROMPT_FACTORIES:
                logger.warning("[%s] no prompt factory — skipping", cap)
                continue
            logger.info("[%s] budget=%d narrative=%s", cap, budget,
                        cap in NARRATIVE_CAPABILITIES)
            stream = PROMPT_FACTORIES[cap](rng)
            produced = 0
            attempts = 0
            max_attempts = budget * 4  # hard ceiling so a bad teacher can't spin forever
            for spec in stream:
                if produced >= budget:
                    break
                if attempts >= max_attempts:
                    logger.warning(
                        "[%s] stopping: %d attempts for %d produced (ceiling)",
                        cap, attempts, produced,
                    )
                    break
                attempts += 1
                per_cap_stats[cap]["attempted"] += 1

                row: Optional[Dict[str, Any]] = None
                diag: Any = None
                if cap in NEMOTRON_AUTHORITATIVE_CAPABILITIES:
                    row, diag = _nemotron_authoritative_row(cap, spec)
                    if row is None:
                        per_cap_stats[cap]["quarantine"] += 1
                    else:
                        per_cap_stats[cap]["judge_accepted"] += 1
                elif cap in NARRATIVE_CAPABILITIES:
                    row, diag = _narrative_row(cap, spec)
                    if row is None:
                        per_cap_stats[cap]["quarantine"] += 1
                    elif row["source"] == "v5_seed":
                        per_cap_stats[cap]["seed"] += 1
                    else:
                        per_cap_stats[cap]["judge_accepted"] += 1
                else:
                    row, vote_outcome = _structured_row(
                        cap, spec, teacher_callers=teacher_callers,
                    )
                    diag = vote_outcome
                    if row is None:
                        per_cap_stats[cap]["quarantine"] += 1
                        # Try to mine a DPO pair from the disagreement.
                        if vote_outcome is not None:
                            dpo_row = _mine_dpo_from_quarantine(cap, spec, vote_outcome)
                            if dpo_row and existing_dpo.get(cap, 0) + per_cap_stats[cap]["dpo_written"] < charter_get(cap).dpo_target_pairs:
                                dpo_f.write(json.dumps(dpo_row, ensure_ascii=False) + "\n")
                                dpo_f.flush()
                                per_cap_stats[cap]["dpo_written"] += 1
                    else:
                        per_cap_stats[cap]["ensemble_accepted"] += 1

                if row is not None:
                    sft_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sft_f.flush()
                    per_cap_stats[cap]["sft_accepted"] += 1
                    produced += 1
                    # For narrative capabilities, also synthesise a DPO pair
                    # from the accepted row (chosen = good, rejected = canned bad).
                    if cap in NARRATIVE_CAPABILITIES:
                        dpo_row = _mine_dpo_narrative(cap, row, rng)
                        cap_target = charter_get(cap).dpo_target_pairs
                        if dpo_row and existing_dpo.get(cap, 0) + per_cap_stats[cap]["dpo_written"] < cap_target:
                            dpo_f.write(json.dumps(dpo_row, ensure_ascii=False) + "\n")
                            dpo_f.flush()
                            per_cap_stats[cap]["dpo_written"] += 1

                # Always record a small sample for diagnostics.
                if len(results_sample) < 60:
                    results_sample.append({
                        "capability": cap,
                        "attempt": attempts,
                        "accepted": row is not None,
                        "source": (row or {}).get("source"),
                        "prompt_preview": spec["prompt"][:120],
                        "assistant_preview": ((row or {}).get("assistant") or "")[:160],
                        "diag": _diag_summary(diag),
                    })

            logger.info("[%s] done: %s", cap, per_cap_stats[cap])
    finally:
        sft_f.close()
        dpo_f.close()

    elapsed = round(time.monotonic() - t0, 1)
    totals = {
        "sft_accepted": sum(s["sft_accepted"] for s in per_cap_stats.values()),
        "dpo_written": sum(s["dpo_written"] for s in per_cap_stats.values()),
        "quarantine": sum(s["quarantine"] for s in per_cap_stats.values()),
        "attempted": sum(s["attempted"] for s in per_cap_stats.values()),
    }
    accept_rate = totals["sft_accepted"] / max(totals["attempted"], 1)

    # Probe teachers once at the end so the test report tells operators
    # whether quarantine was "ensemble rejected the row" (prompt bug) or
    # "primary teachers weren't reachable" (GPU contention).
    teacher_health: Dict[str, Any] = {}
    try:
        from src.serving.vllm_manager import VLLMManager
        teacher_health["v3_vllm_up"] = VLLMManager().health_check()
    except Exception as exc:  # noqa: BLE001
        teacher_health["v3_vllm_up"] = False
        teacher_health["v3_vllm_err"] = str(exc)[:200]
    try:
        from urllib import request as _r
        with _r.urlopen("http://localhost:11434/api/tags", timeout=3) as resp:
            teacher_health["ollama_up"] = resp.status == 200
    except Exception as exc:  # noqa: BLE001
        teacher_health["ollama_up"] = False
        teacher_health["ollama_err"] = str(exc)[:200]

    summary = {
        "mode": "test_batch" if test_batch else "production",
        "test_batch_size": test_batch,
        "capabilities": capabilities,
        "budgets": capability_budgets,
        "per_capability": per_cap_stats,
        "totals": totals,
        "acceptance_rate": round(accept_rate, 3),
        "elapsed_s": elapsed,
        "sft_output": str(sft_path),
        "dpo_output": str(dpo_path),
        "teacher_health": teacher_health,
    }

    if test_batch is not None:
        report_path = Path(test_report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "sample": results_sample},
                      f, indent=2, default=str)
        logger.info("test-batch report → %s", report_path)

    print()
    print("=" * 60)
    print("DATA GENERATOR SUMMARY")
    print("=" * 60)
    for cap, s in per_cap_stats.items():
        total_att = max(s["attempted"], 1)
        print(f"  {cap:26s} attempted={s['attempted']:>4}  "
              f"sft={s['sft_accepted']:>4}  dpo={s['dpo_written']:>4}  "
              f"quarantine={s['quarantine']:>4}  "
              f"rate={s['sft_accepted']/total_att:.2%}")
    print(f"  {'TOTAL':26s} attempted={totals['attempted']:>4}  "
          f"sft={totals['sft_accepted']:>4}  dpo={totals['dpo_written']:>4}  "
          f"quarantine={totals['quarantine']:>4}  "
          f"rate={accept_rate:.2%}")
    print(f"  elapsed: {elapsed}s")
    return summary


def _diag_summary(d: Any) -> Any:
    """Compact a diagnostic blob for inclusion in the test report."""
    if d is None:
        return None
    if isinstance(d, EnsembleVote):
        return {
            "type": "ensemble",
            "accepted": d.accepted,
            "confidence": d.confidence,
            "agreement_count": d.agreement_count,
            "primary_voice_count": d.primary_voice_count,
            "nemotron_tiebreak": d.nemotron_tiebreak,
            "reason": d.reason,
        }
    if isinstance(d, dict):
        return d
    return str(d)[:200]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_capabilities(arg: str) -> List[str]:
    caps = [c.strip() for c in arg.split(",") if c.strip()]
    for c in caps:
        if c not in CHARTER:
            raise SystemExit(
                f"Unknown capability '{c}'. Known: {sorted(CHARTER.keys())}"
            )
        if c not in PROMPT_FACTORIES:
            raise SystemExit(
                f"Capability '{c}' has no prompt factory registered. "
                f"Registered factories: {sorted(PROMPT_FACTORIES.keys())}"
            )
    return caps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--capabilities", required=True,
        help="Comma-separated charter keys to generate rows for.",
    )
    ap.add_argument(
        "--sft-output", default="finetune_artifacts/v5/sft_generated.jsonl",
    )
    ap.add_argument(
        "--dpo-output", default="finetune_artifacts/v5/dpo_generated.jsonl",
    )
    ap.add_argument(
        "--max-rows-per-capability", type=int, default=None,
        help="Cap rows per capability (overrides charter target if lower).",
    )
    ap.add_argument(
        "--test-batch", type=int, default=None,
        help="Test-batch mode: generate only N rows total as a dress rehearsal. "
             "Writes finetune_artifacts/v5/data_generator_test.json.",
    )
    ap.add_argument(
        "--resume", action="store_true",
        help="Read existing outputs and skip capabilities already at target.",
    )
    ap.add_argument(
        "--skip-hf", action="store_true", default=True,
        help="Skip HF teacher during voting (default true — HF teacher is slow "
             "to load). Pass --no-skip-hf to include it.",
    )
    ap.add_argument("--no-skip-hf", action="store_false", dest="skip_hf")
    ap.add_argument(
        "--rng-seed", type=int, default=42,
    )
    ap.add_argument(
        "--test-report-path",
        default="finetune_artifacts/v5/data_generator_test.json",
    )
    args = ap.parse_args()

    caps = _parse_capabilities(args.capabilities)
    # For speed, default production callers are V3+Ollama. HF only when
    # explicitly requested (loading 14B locally costs ~2 minutes + GPU).
    teacher_callers: List[Callable[..., TeacherResponse]]
    if args.skip_hf:
        teacher_callers = [_call_vllm, _call_ollama]
    else:
        from src.finetune.v5.teacher_ensemble import _call_hf
        teacher_callers = [_call_vllm, _call_ollama, _call_hf]

    summary = generate(
        capabilities=caps,
        sft_output=args.sft_output,
        dpo_output=args.dpo_output,
        max_rows_per_capability=args.max_rows_per_capability,
        test_batch=args.test_batch,
        resume=args.resume,
        teacher_callers=teacher_callers,
        rng_seed=args.rng_seed,
        test_report_path=args.test_report_path,
    )
    # Exit non-zero if test-batch accepted nothing — lets CI/shell catch it.
    if args.test_batch is not None and summary["totals"]["sft_accepted"] == 0:
        raise SystemExit(
            "test-batch produced 0 accepted rows — pipeline broken, investigate"
        )


if __name__ == "__main__":
    main()
