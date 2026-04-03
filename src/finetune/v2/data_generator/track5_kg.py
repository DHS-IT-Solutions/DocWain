"""Track 5 — KG-Augmented Knowledge data generator for DocWain V2+ SFT/DPO.

Generates 2000 training examples across seven KG reasoning categories:
  - Entity-aware answering        (400)
  - Relationship traversal        (350)
  - Cross-doc entity linking      (300)
  - KG-grounded fact checking     (250)
  - Missing relationship detection(200)
  - Ontology-aware reasoning      (250)
  - KG context format training    (250)

All examples include a ``<kg_context>`` block in the user query,
and the model's answers cite KG entities by ID and reason about
relationships.

Produces both SFT examples (with ``<think>`` reasoning) and DPO
preference pairs.
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
    format_sft_example,
)

# ---------------------------------------------------------------------------
# Helpers & constants
# ---------------------------------------------------------------------------

_CURRENCIES = ["$", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "INR"]
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
    "Mia Thompson", "Noah Williams", "Olivia Brown", "Priya Patel",
]
_DEPARTMENTS = [
    "Engineering", "Finance", "Human Resources", "Legal",
    "Marketing", "Operations", "Sales", "Research & Development",
    "Compliance", "Procurement", "IT", "Customer Success",
]
_PRODUCT_NAMES = [
    "Widget A", "Gadget Pro", "Module X-100", "Sensor Suite",
    "Platform License", "Cloud Tier 2", "Enterprise Connector",
    "Data Vault", "Analytics Pack", "Security Module",
]
_ROLES = [
    "CEO", "CFO", "CTO", "COO", "VP of Engineering", "VP of Sales",
    "General Counsel", "Director of HR", "Head of Compliance",
    "Chief Risk Officer", "Senior Manager", "Board Member",
]
_RELATIONSHIP_TYPES = [
    "WORKS_AT", "REPORTS_TO", "MANAGES", "SIGNED", "AUTHORED",
    "APPROVED", "REVIEWED", "SUPPLIES", "ACQUIRED", "PARTNERED_WITH",
    "SUBSIDIARY_OF", "CONTRACTED", "AUDITED", "INVESTED_IN",
]
_ENTITY_TYPES = [
    "Person", "Organization", "Contract", "Document", "Department",
    "Product", "Location", "Regulation", "Project", "Account",
]
_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_QUARTERS = ["Q1", "Q2", "Q3", "Q4"]


def _pick(lst: list, rng: random.Random) -> Any:
    return rng.choice(lst)


def _rand_amount(rng: random.Random, lo: float = 100.0, hi: float = 99999.0) -> str:
    return f"{rng.uniform(lo, hi):,.2f}"


def _rand_date(rng: random.Random) -> str:
    y = rng.randint(2020, 2026)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    return f"{y}-{m:02d}-{d:02d}"


def _eid(n: int) -> str:
    return f"E{n}"


def _subs(rng: random.Random) -> Dict[str, str]:
    """Build a standard substitution dict for KG templates."""
    p1 = _pick(_PERSON_NAMES, rng)
    p2 = _pick([n for n in _PERSON_NAMES if n != p1], rng)
    p3 = _pick([n for n in _PERSON_NAMES if n not in (p1, p2)], rng)
    c1 = _pick(_COMPANY_NAMES, rng)
    c2 = _pick([n for n in _COMPANY_NAMES if n != c1], rng)
    r1 = _pick(_ROLES, rng)
    r2 = _pick(_ROLES, rng)
    return {
        "domain": _pick(DOMAINS, rng),
        "doc_type": _pick(DOC_TYPES, rng),
        "company": c1,
        "company2": c2,
        "person": p1,
        "person2": p2,
        "person3": p3,
        "role": r1,
        "role2": r2,
        "dept": _pick(_DEPARTMENTS, rng),
        "dept2": _pick(_DEPARTMENTS, rng),
        "product": _pick(_PRODUCT_NAMES, rng),
        "currency": _pick(_CURRENCIES, rng),
        "amount": _rand_amount(rng),
        "amount2": _rand_amount(rng),
        "year": str(rng.randint(2020, 2026)),
        "year2": str(rng.randint(2020, 2026)),
        "quarter": _pick(_QUARTERS, rng),
        "month": _pick(_MONTHS, rng),
        "date1": _rand_date(rng),
        "date2": _rand_date(rng),
        "date3": _rand_date(rng),
        "pct": f"{rng.uniform(1, 45):.1f}",
        "rel_type": _pick(_RELATIONSHIP_TYPES, rng),
        "rel_type2": _pick(_RELATIONSHIP_TYPES, rng),
        "contract_id": f"Contract_{rng.randint(100, 999)}",
        "contract_id2": f"Contract_{rng.randint(100, 999)}",
        "doc_id1": f"doc_{rng.randint(1, 20)}",
        "doc_id2": f"doc_{rng.randint(1, 20)}",
        "doc_id3": f"doc_{rng.randint(1, 20)}",
    }


def _expand(
    templates: List[Dict[str, str]],
    count: int,
    rng: random.Random,
    *,
    mode: str = "sft",
) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    idx = 0
    while len(results) < count:
        tpl = templates[idx % len(templates)]
        s = _subs(rng)
        try:
            query = tpl["query"].format(**s)
            if mode == "sft":
                reasoning = tpl["reasoning"].format(**s)
                answer = tpl["answer"].format(**s)
                results.append(format_sft_example(query, reasoning, answer))
            else:
                chosen_r = tpl["chosen_reasoning"].format(**s)
                chosen_a = tpl["chosen_answer"].format(**s)
                rejected_r = tpl["rejected_reasoning"].format(**s)
                rejected_a = tpl["rejected_answer"].format(**s)
                results.append(format_dpo_example(
                    query, chosen_r, chosen_a, rejected_r, rejected_a,
                ))
        except (KeyError, IndexError):
            idx += 1
            continue
        idx += 1
    return results


# ===================================================================
# 1. ENTITY-AWARE ANSWERING (400)
# ===================================================================

_ENTITY_AWARE_SFT: List[Dict[str, str]] = [
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}, {doc_id2}]\n"
            "  - id: E2, name: \"{person}\", type: Person, role: \"{role}\", doc_sources: [{doc_id1}]\n"
            "  - id: E3, name: \"{person}\", type: Person, role: \"{role2}\", doc_sources: [{doc_id2}]\n"
            "relationships:\n"
            "  - E2 --[WORKS_AT]--> E1, since: {year}, source: {doc_id1}\n"
            "  - E3 --[WORKS_AT]--> E1, since: {year2}, source: {doc_id2}\n"
            "</kg_context>\n\n"
            "Who is {person} at {company}?"
        ),
        "reasoning": (
            "Step 1: The KG contains two entities both named '{person}' — E2 and E3.\n"
            "Step 2: E2 is a {role} (source: {doc_id1}), E3 is a {role2} (source: {doc_id2}).\n"
            "Step 3: This is an entity disambiguation case — same name, different roles.\n"
            "Step 4: Possible explanations: (a) role change over time ({year} vs {year2}), "
            "or (b) two different people with the same name.\n"
            "Step 5: Check dates — if {year} and {year2} differ, likely a role change for the same person.\n"
            "Confidence: Medium — disambiguation requires additional verification."
        ),
        "answer": (
            "The knowledge graph contains two entities named '{person}' at {company}:\n\n"
            "- **E2**: {role}, documented in {doc_id1} since {year}.\n"
            "- **E3**: {role2}, documented in {doc_id2} since {year2}.\n\n"
            "This could indicate either a role change (same person, {role} -> {role2}) or two "
            "distinct individuals sharing the same name. To disambiguate, I recommend checking "
            "employee IDs or other unique identifiers in the source documents."
        ),
    },
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{company}\", type: Organization, industry: \"{domain}\", doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{company2}\", type: Organization, industry: \"{domain}\", doc_sources: [{doc_id2}]\n"
            "  - id: E3, name: \"{person}\", type: Person, role: \"{role}\", doc_sources: [{doc_id1}, {doc_id2}]\n"
            "relationships:\n"
            "  - E3 --[WORKS_AT]--> E1, since: {year}, source: {doc_id1}\n"
            "  - E3 --[CONTRACTED]--> E2, date: {date1}, source: {doc_id2}\n"
            "</kg_context>\n\n"
            "What is {person}'s relationship with {company2}?"
        ),
        "reasoning": (
            "Step 1: E3 ({person}) has a WORKS_AT relationship with E1 ({company}) since {year}.\n"
            "Step 2: E3 also has a CONTRACTED relationship with E2 ({company2}) on {date1}.\n"
            "Step 3: {person} works at {company} but has a contractual relationship with {company2}.\n"
            "Step 4: This could be a vendor contract, consulting engagement, or business deal.\n"
            "Step 5: The CONTRACTED relationship is sourced from {doc_id2}.\n"
            "Confidence: High — relationship clearly defined in KG."
        ),
        "answer": (
            "According to the knowledge graph, {person} (E3) has a CONTRACTED relationship "
            "with {company2} (E2), established on {date1} (source: {doc_id2}). {person}'s primary "
            "employer is {company} (E1) where they serve as {role} since {year}. The contractual "
            "relationship with {company2} is separate from their employment."
        ),
    },
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{person}\", type: Person, role: \"{role}\", doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{person2}\", type: Person, role: \"{role2}\", doc_sources: [{doc_id1}]\n"
            "  - id: E3, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}, {doc_id2}]\n"
            "  - id: E4, name: \"{contract_id}\", type: Contract, value: \"{currency}{amount}\", doc_sources: [{doc_id2}]\n"
            "relationships:\n"
            "  - E1 --[SIGNED]--> E4, date: {date1}, source: {doc_id2}\n"
            "  - E2 --[APPROVED]--> E4, date: {date2}, source: {doc_id2}\n"
            "  - E4 --[CONTRACTED]--> E3, source: {doc_id2}\n"
            "</kg_context>\n\n"
            "Who signed the {contract_id} contract and who approved it?"
        ),
        "reasoning": (
            "Step 1: E4 is {contract_id}, a contract valued at {currency}{amount}.\n"
            "Step 2: E1 ({person}, {role}) SIGNED E4 on {date1}.\n"
            "Step 3: E2 ({person2}, {role2}) APPROVED E4 on {date2}.\n"
            "Step 4: The contract is with E3 ({company}).\n"
            "Step 5: Clear entity-to-action mapping from the KG.\n"
            "Confidence: High — explicit relationships in KG."
        ),
        "answer": (
            "For {contract_id} (E4, valued at {currency}{amount}):\n\n"
            "- **Signed by**: {person} (E1, {role}) on {date1}.\n"
            "- **Approved by**: {person2} (E2, {role2}) on {date2}.\n"
            "- **Counterparty**: {company} (E3).\n\n"
            "Both actions are documented in {doc_id2}."
        ),
    },
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{person}\", type: Person, role: \"Senior Analyst\", doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{person}\", type: Person, role: \"Senior Analyst\", department: \"{dept}\", doc_sources: [{doc_id2}]\n"
            "  - id: E3, name: \"{person}\", type: Person, role: \"Senior Analyst\", department: \"{dept2}\", doc_sources: [{doc_id3}]\n"
            "  - id: E4, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}, {doc_id2}, {doc_id3}]\n"
            "relationships:\n"
            "  - E1 --[WORKS_AT]--> E4, source: {doc_id1}\n"
            "  - E2 --[WORKS_AT]--> E4, source: {doc_id2}\n"
            "  - E3 --[WORKS_AT]--> E4, source: {doc_id3}\n"
            "</kg_context>\n\n"
            "There are multiple '{person}' entries. Are they the same person?"
        ),
        "reasoning": (
            "Step 1: Three entities with the same name '{person}', all Senior Analysts at {company}.\n"
            "Step 2: E1 has no department listed (from {doc_id1}).\n"
            "Step 3: E2 is in {dept} (from {doc_id2}), E3 is in {dept2} (from {doc_id3}).\n"
            "Step 4: If {dept} and {dept2} are different, these could be distinct people "
            "or the same person who transferred departments.\n"
            "Step 5: E1 may be an earlier record before department assignment.\n"
            "Step 6: Need employee ID or unique identifier to confirm.\n"
            "Confidence: Low — same name and role but different departments is ambiguous."
        ),
        "answer": (
            "Three KG entities share the name '{person}' with the role 'Senior Analyst' at {company}:\n\n"
            "- **E1**: No department specified (source: {doc_id1}).\n"
            "- **E2**: {dept} department (source: {doc_id2}).\n"
            "- **E3**: {dept2} department (source: {doc_id3}).\n\n"
            "These could represent: (a) the same person with a department transfer history, "
            "(b) two or three distinct individuals. Entity resolution requires matching on "
            "employee ID, email, or other unique identifiers from the source documents."
        ),
    },
]

_ENTITY_AWARE_DPO: List[Dict[str, str]] = [
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{person}\", type: Person, role: \"{role}\", doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{person}\", type: Person, role: \"Intern\", doc_sources: [{doc_id2}]\n"
            "  - id: E3, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}, {doc_id2}]\n"
            "relationships:\n"
            "  - E1 --[WORKS_AT]--> E3, since: {year}, source: {doc_id1}\n"
            "  - E2 --[WORKS_AT]--> E3, since: {year2}, source: {doc_id2}\n"
            "</kg_context>\n\n"
            "What role does {person} hold at {company}?"
        ),
        "chosen_reasoning": (
            "Step 1: Two entities named '{person}' — E1 ({role}, since {year}) and E2 (Intern, since {year2}).\n"
            "Step 2: These are likely different individuals given the role disparity.\n"
            "Step 3: A {role} and an Intern at the same company sharing a name is plausible.\n"
            "Step 4: Cannot definitively answer without disambiguation.\n"
            "Step 5: Must surface the ambiguity to the user with both possibilities.\n"
            "Confidence: Low — entity ambiguity prevents definitive answer."
        ),
        "chosen_answer": (
            "The KG contains two entities named '{person}' at {company} (E3):\n\n"
            "- **E1**: {role}, since {year} (source: {doc_id1}).\n"
            "- **E2**: Intern, since {year2} (source: {doc_id2}).\n\n"
            "These appear to be different individuals. Please specify which '{person}' you are "
            "referring to, or provide additional context (department, start date) to disambiguate."
        ),
        "rejected_reasoning": "Looking up the person in the KG.",
        "rejected_answer": "{person} is a {role} at {company}.",
    },
]


# ===================================================================
# 2. RELATIONSHIP TRAVERSAL (350)
# ===================================================================

_RELATIONSHIP_SFT: List[Dict[str, str]] = [
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{person}\", type: Person, role: \"{role}\", doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{person2}\", type: Person, role: \"{role2}\", doc_sources: [{doc_id1}]\n"
            "  - id: E3, name: \"{person3}\", type: Person, role: \"CFO\", doc_sources: [{doc_id1}]\n"
            "  - id: E4, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}]\n"
            "relationships:\n"
            "  - E1 --[REPORTS_TO]--> E2, source: {doc_id1}\n"
            "  - E2 --[REPORTS_TO]--> E3, source: {doc_id1}\n"
            "  - E1 --[WORKS_AT]--> E4, source: {doc_id1}\n"
            "  - E2 --[WORKS_AT]--> E4, source: {doc_id1}\n"
            "  - E3 --[WORKS_AT]--> E4, source: {doc_id1}\n"
            "</kg_context>\n\n"
            "Who reports to the CFO at {company}?"
        ),
        "reasoning": (
            "Step 1: Identify the CFO — E3 ({person3}), role: CFO.\n"
            "Step 2: Find REPORTS_TO relationships pointing to E3.\n"
            "Step 3: E2 ({person2}, {role2}) --[REPORTS_TO]--> E3.\n"
            "Step 4: Also check indirect reports — E1 ({person}, {role}) --[REPORTS_TO]--> E2.\n"
            "Step 5: Direct report to CFO: E2. Indirect report (via E2): E1.\n"
            "Confidence: High — relationship chain is explicit in KG."
        ),
        "answer": (
            "At {company} (E4), the following reporting relationships exist under the CFO ({person3}, E3):\n\n"
            "- **Direct report**: {person2} (E2, {role2}) reports directly to the CFO.\n"
            "- **Indirect report**: {person} (E1, {role}) reports to {person2}, who in turn reports to the CFO.\n\n"
            "Source: {doc_id1}."
        ),
    },
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{company2}\", type: Organization, doc_sources: [{doc_id2}]\n"
            "  - id: E3, name: \"{contract_id}\", type: Contract, value: \"{currency}{amount}\", doc_sources: [{doc_id1}, {doc_id2}]\n"
            "  - id: E4, name: \"{person}\", type: Person, role: \"{role}\", doc_sources: [{doc_id1}]\n"
            "  - id: E5, name: \"{person2}\", type: Person, role: \"{role2}\", doc_sources: [{doc_id2}]\n"
            "relationships:\n"
            "  - E1 --[CONTRACTED]--> E3, source: {doc_id1}\n"
            "  - E2 --[CONTRACTED]--> E3, source: {doc_id2}\n"
            "  - E4 --[SIGNED]--> E3, date: {date1}, source: {doc_id1}\n"
            "  - E5 --[SIGNED]--> E3, date: {date2}, source: {doc_id2}\n"
            "  - E4 --[WORKS_AT]--> E1, source: {doc_id1}\n"
            "  - E5 --[WORKS_AT]--> E2, source: {doc_id2}\n"
            "</kg_context>\n\n"
            "Trace the full relationship chain for {contract_id}."
        ),
        "reasoning": (
            "Step 1: E3 is {contract_id}, valued at {currency}{amount}.\n"
            "Step 2: Two organizations are parties: E1 ({company}) and E2 ({company2}).\n"
            "Step 3: Signatories: E4 ({person}, {role} at {company}) signed on {date1}; "
            "E5 ({person2}, {role2} at {company2}) signed on {date2}.\n"
            "Step 4: Relationship chain: {company} employs {person} who signed contract "
            "which was also signed by {person2} employed by {company2}.\n"
            "Step 5: Full traversal covers 5 entities and 6 relationships.\n"
            "Confidence: High — complete chain documented in KG."
        ),
        "answer": (
            "Full relationship chain for {contract_id} (E3, {currency}{amount}):\n\n"
            "1. {company} (E1) --[CONTRACTED]--> {contract_id} (E3) <--[CONTRACTED]-- {company2} (E2)\n"
            "2. {person} (E4, {role}) --[WORKS_AT]--> {company} (E1)\n"
            "3. {person} (E4) --[SIGNED]--> {contract_id} (E3) on {date1}\n"
            "4. {person2} (E5, {role2}) --[WORKS_AT]--> {company2} (E2)\n"
            "5. {person2} (E5) --[SIGNED]--> {contract_id} (E3) on {date2}\n\n"
            "The contract links two organizations through their respective signatories."
        ),
    },
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{person}\", type: Person, role: \"{role}\", doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{dept}\", type: Department, doc_sources: [{doc_id1}]\n"
            "  - id: E3, name: \"{dept2}\", type: Department, doc_sources: [{doc_id1}]\n"
            "  - id: E4, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}]\n"
            "relationships:\n"
            "  - E1 --[MANAGES]--> E2, since: {year}, source: {doc_id1}\n"
            "  - E1 --[MANAGES]--> E3, since: {year2}, source: {doc_id1}\n"
            "  - E2 --[PART_OF]--> E4, source: {doc_id1}\n"
            "  - E3 --[PART_OF]--> E4, source: {doc_id1}\n"
            "</kg_context>\n\n"
            "What departments does {person} manage?"
        ),
        "reasoning": (
            "Step 1: Find MANAGES relationships from E1 ({person}).\n"
            "Step 2: E1 --[MANAGES]--> E2 ({dept}) since {year}.\n"
            "Step 3: E1 --[MANAGES]--> E3 ({dept2}) since {year2}.\n"
            "Step 4: Both departments are part of E4 ({company}).\n"
            "Step 5: {person} manages two departments.\n"
            "Confidence: High — direct KG traversal."
        ),
        "answer": (
            "{person} (E1, {role}) manages two departments at {company} (E4):\n\n"
            "1. **{dept}** (E2) — since {year}.\n"
            "2. **{dept2}** (E3) — since {year2}.\n\n"
            "Source: {doc_id1}."
        ),
    },
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{person}\", type: Person, role: \"{role}\", doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{person2}\", type: Person, role: \"{role2}\", doc_sources: [{doc_id1}]\n"
            "  - id: E3, name: \"{person3}\", type: Person, role: \"CEO\", doc_sources: [{doc_id1}]\n"
            "  - id: E4, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}]\n"
            "relationships:\n"
            "  - E1 --[REPORTS_TO]--> E2, source: {doc_id1}\n"
            "  - E2 --[REPORTS_TO]--> E3, source: {doc_id1}\n"
            "  - E3 --[MANAGES]--> E4, source: {doc_id1}\n"
            "</kg_context>\n\n"
            "How many levels separate {person} from the CEO?"
        ),
        "reasoning": (
            "Step 1: Build reporting chain: E1 -> E2 -> E3 (CEO).\n"
            "Step 2: E1 ({person}) reports to E2 ({person2}) — 1 level.\n"
            "Step 3: E2 reports to E3 ({person3}, CEO) — 2 levels from E1.\n"
            "Step 4: {person} is 2 reporting levels below the CEO.\n"
            "Confidence: High — explicit chain in KG."
        ),
        "answer": (
            "{person} (E1, {role}) is **2 levels** below the CEO ({person3}, E3):\n\n"
            "- Level 1: {person} (E1) reports to {person2} (E2, {role2}).\n"
            "- Level 2: {person2} (E2) reports to {person3} (E3, CEO).\n\n"
            "Source: {doc_id1}."
        ),
    },
]

_RELATIONSHIP_DPO: List[Dict[str, str]] = [
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{person}\", type: Person, role: \"{role}\", doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{person2}\", type: Person, role: \"CEO\", doc_sources: [{doc_id1}]\n"
            "  - id: E3, name: \"{person3}\", type: Person, role: \"{role2}\", doc_sources: [{doc_id1}]\n"
            "  - id: E4, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}]\n"
            "relationships:\n"
            "  - E1 --[REPORTS_TO]--> E3, source: {doc_id1}\n"
            "  - E3 --[REPORTS_TO]--> E2, source: {doc_id1}\n"
            "</kg_context>\n\n"
            "Does {person} report to the CEO directly?"
        ),
        "chosen_reasoning": (
            "Step 1: Find {person} (E1) reporting chain.\n"
            "Step 2: E1 --[REPORTS_TO]--> E3 ({person3}, {role2}). Not the CEO.\n"
            "Step 3: E3 --[REPORTS_TO]--> E2 ({person2}, CEO). So E3 reports to CEO.\n"
            "Step 4: {person} reports to CEO indirectly through {person3}.\n"
            "Step 5: Direct report = no; indirect through {person3} = yes.\n"
            "Confidence: High — KG chain is explicit."
        ),
        "chosen_answer": (
            "No, {person} (E1) does not report directly to the CEO ({person2}, E2). "
            "The reporting chain is:\n\n"
            "- {person} (E1, {role}) --[REPORTS_TO]--> {person3} (E3, {role2}) "
            "--[REPORTS_TO]--> {person2} (E2, CEO).\n\n"
            "{person} is an indirect report to the CEO, with {person3} as the intermediary."
        ),
        "rejected_reasoning": "Checking the KG for the reporting relationship.",
        "rejected_answer": "{person} reports to the CEO.",
    },
]


# ===================================================================
# 3. CROSS-DOC ENTITY LINKING (300)
# ===================================================================

_CROSS_DOC_SFT: List[Dict[str, str]] = [
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}, {doc_id2}, {doc_id3}]\n"
            "  - id: E2, name: \"{person}\", type: Person, role: \"{role}\", doc_sources: [{doc_id1}]\n"
            "  - id: E3, name: \"{person}\", type: Person, role: \"{role}\", doc_sources: [{doc_id2}]\n"
            "  - id: E4, name: \"{contract_id}\", type: Contract, doc_sources: [{doc_id1}]\n"
            "  - id: E5, name: \"{contract_id2}\", type: Contract, doc_sources: [{doc_id2}]\n"
            "relationships:\n"
            "  - E2 --[SIGNED]--> E4, date: {date1}, source: {doc_id1}\n"
            "  - E3 --[SIGNED]--> E5, date: {date2}, source: {doc_id2}\n"
            "  - E2 --[WORKS_AT]--> E1, source: {doc_id1}\n"
            "  - E3 --[WORKS_AT]--> E1, source: {doc_id2}\n"
            "</kg_context>\n\n"
            "Are E2 and E3 the same person? Consolidate what we know about {person}."
        ),
        "reasoning": (
            "Step 1: E2 and E3 share the same name ('{person}') and role ('{role}').\n"
            "Step 2: Both work at E1 ({company}).\n"
            "Step 3: They appear in different documents ({doc_id1} and {doc_id2}).\n"
            "Step 4: High likelihood of being the same person — same name, role, and employer.\n"
            "Step 5: Consolidate: {person} signed {contract_id} ({date1}) and {contract_id2} ({date2}).\n"
            "Confidence: High — strong entity match on name + role + employer."
        ),
        "answer": (
            "E2 and E3 are very likely the same person based on matching name ('{person}'), "
            "role ('{role}'), and employer ({company}, E1). Consolidated profile:\n\n"
            "- **Name**: {person}\n"
            "- **Role**: {role} at {company}\n"
            "- **Activities**:\n"
            "  - Signed {contract_id} (E4) on {date1} (source: {doc_id1})\n"
            "  - Signed {contract_id2} (E5) on {date2} (source: {doc_id2})\n\n"
            "Recommend merging E2 and E3 into a single entity."
        ),
    },
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{company}\", type: Organization, alias: \"{company} Inc.\", doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{company} Inc.\", type: Organization, doc_sources: [{doc_id2}]\n"
            "  - id: E3, name: \"{person}\", type: Person, doc_sources: [{doc_id1}, {doc_id2}]\n"
            "relationships:\n"
            "  - E3 --[WORKS_AT]--> E1, source: {doc_id1}\n"
            "  - E3 --[WORKS_AT]--> E2, source: {doc_id2}\n"
            "</kg_context>\n\n"
            "Does {person} work at two different companies or one?"
        ),
        "reasoning": (
            "Step 1: E1 is '{company}' with alias '{company} Inc.' from {doc_id1}.\n"
            "Step 2: E2 is '{company} Inc.' from {doc_id2}.\n"
            "Step 3: E1 has an alias that matches E2's name exactly.\n"
            "Step 4: These are the same organization with different name representations.\n"
            "Step 5: {person} (E3) works at one company, not two.\n"
            "Confidence: High — alias match confirms entity equivalence."
        ),
        "answer": (
            "{person} (E3) works at one company. E1 ('{company}') and E2 ('{company} Inc.') "
            "are the same organization — E1's alias field matches E2's name. This is a common "
            "cross-document entity duplication where legal name variations create separate KG entries. "
            "Recommend merging E1 and E2."
        ),
    },
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{person}\", type: Person, role: \"{role}\", email: \"{person}@{domain}.com\", doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{person}\", type: Person, role: \"{role2}\", email: \"{person}@{domain}.com\", doc_sources: [{doc_id2}]\n"
            "  - id: E3, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}, {doc_id2}]\n"
            "relationships:\n"
            "  - E1 --[WORKS_AT]--> E3, since: {year}, source: {doc_id1}\n"
            "  - E2 --[WORKS_AT]--> E3, since: {year2}, source: {doc_id2}\n"
            "</kg_context>\n\n"
            "Reconcile the two entries for {person}."
        ),
        "reasoning": (
            "Step 1: E1 and E2 share name and email — very strong identity signal.\n"
            "Step 2: Role differs: E1 is {role} (since {year}), E2 is {role2} (since {year2}).\n"
            "Step 3: Same email confirms same person; role difference indicates promotion/transfer.\n"
            "Step 4: Timeline: {role} since {year}, then {role2} since {year2}.\n"
            "Step 5: Consolidated: single person with role history.\n"
            "Confidence: High — email match is a strong unique identifier."
        ),
        "answer": (
            "E1 and E2 are the same person — confirmed by matching email address. "
            "Reconciled profile for {person}:\n\n"
            "- **Current Role**: {role2} at {company} (since {year2}, source: {doc_id2})\n"
            "- **Previous Role**: {role} at {company} (since {year}, source: {doc_id1})\n"
            "- **Email**: {person}@{domain}.com\n\n"
            "The two entries reflect a role change from {role} to {role2}."
        ),
    },
]

_CROSS_DOC_DPO: List[Dict[str, str]] = [
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{person}\", type: Person, role: \"{role}\", doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{person}\", type: Person, role: \"{role}\", doc_sources: [{doc_id2}]\n"
            "  - id: E3, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}]\n"
            "  - id: E4, name: \"{company2}\", type: Organization, doc_sources: [{doc_id2}]\n"
            "relationships:\n"
            "  - E1 --[WORKS_AT]--> E3, source: {doc_id1}\n"
            "  - E2 --[WORKS_AT]--> E4, source: {doc_id2}\n"
            "</kg_context>\n\n"
            "Are E1 and E2 the same person?"
        ),
        "chosen_reasoning": (
            "Step 1: Same name '{person}' and same role '{role}'.\n"
            "Step 2: But different employers — E3 ({company}) vs E4 ({company2}).\n"
            "Step 3: Same name + same role but different companies is ambiguous.\n"
            "Step 4: Could be the same person who changed jobs, or two different people.\n"
            "Step 5: No unique identifier (email, ID) to confirm or deny.\n"
            "Step 6: Cannot merge without additional evidence.\n"
            "Confidence: Low — insufficient evidence to confirm identity."
        ),
        "chosen_answer": (
            "E1 and E2 cannot be definitively confirmed as the same person. While they share "
            "the name '{person}' and role '{role}', they are associated with different organizations:\n\n"
            "- E1 at {company} (E3, source: {doc_id1})\n"
            "- E2 at {company2} (E4, source: {doc_id2})\n\n"
            "Possible scenarios: (a) same person who changed employers, (b) two different people. "
            "A unique identifier (email, employee ID) is needed for entity resolution."
        ),
        "rejected_reasoning": "They have the same name so they're the same person.",
        "rejected_answer": "Yes, E1 and E2 are the same person — {person}, {role}.",
    },
]


# ===================================================================
# 4. KG-GROUNDED FACT CHECKING (250)
# ===================================================================

_FACT_CHECK_SFT: List[Dict[str, str]] = [
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{person}\", type: Person, role: \"{role}\", doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}]\n"
            "relationships:\n"
            "  - E1 --[WORKS_AT]--> E2, since: {year}, source: {doc_id1}\n"
            "</kg_context>\n\n"
            "Claim: \"{person} has been the {role2} of {company} since {year2}.\"\n"
            "Verify this claim against the knowledge graph."
        ),
        "reasoning": (
            "Step 1: KG says E1 ({person}) is a {role} at E2 ({company}) since {year}.\n"
            "Step 2: Claim says {role2} since {year2}.\n"
            "Step 3: Role mismatch — KG says {role}, claim says {role2}.\n"
            "Step 4: Date mismatch — KG says {year}, claim says {year2}.\n"
            "Step 5: The claim contradicts the KG on both role and start date.\n"
            "Confidence: High — KG data directly contradicts the claim."
        ),
        "answer": (
            "**Claim REJECTED.** The knowledge graph contradicts this claim on two points:\n\n"
            "1. **Role**: KG records {person} (E1) as '{role}', not '{role2}'.\n"
            "2. **Start Date**: KG shows {year}, not {year2}.\n\n"
            "KG source: {doc_id1}. The claim appears to contain inaccurate role and date information."
        ),
    },
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{company2}\", type: Organization, doc_sources: [{doc_id2}]\n"
            "  - id: E3, name: \"{contract_id}\", type: Contract, value: \"{currency}{amount}\", doc_sources: [{doc_id1}]\n"
            "relationships:\n"
            "  - E1 --[CONTRACTED]--> E3, source: {doc_id1}\n"
            "  - E2 --[CONTRACTED]--> E3, source: {doc_id1}\n"
            "</kg_context>\n\n"
            "Claim: \"{company} has a {currency}{amount2} contract with {company2}.\"\n"
            "Is this accurate?"
        ),
        "reasoning": (
            "Step 1: KG confirms contract exists — E3 ({contract_id}) between E1 and E2.\n"
            "Step 2: KG value: {currency}{amount}. Claim value: {currency}{amount2}.\n"
            "Step 3: The contract relationship is confirmed, but the value may differ.\n"
            "Step 4: If {amount} != {amount2}, the claim has an incorrect amount.\n"
            "Confidence: High — KG has explicit contract value."
        ),
        "answer": (
            "**Partially verified.** The KG confirms a contract ({contract_id}, E3) exists between "
            "{company} (E1) and {company2} (E2). However, the KG records the value as "
            "{currency}{amount}, while the claim states {currency}{amount2}. The relationship is "
            "confirmed but the amount may be inaccurate."
        ),
    },
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{person}\", type: Person, role: \"{role}\", doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{person2}\", type: Person, role: \"CEO\", doc_sources: [{doc_id1}]\n"
            "  - id: E3, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}]\n"
            "relationships:\n"
            "  - E1 --[REPORTS_TO]--> E2, source: {doc_id1}\n"
            "</kg_context>\n\n"
            "Claim: \"{person} reports directly to the CEO, {person2}.\"\n"
            "Verify."
        ),
        "reasoning": (
            "Step 1: KG has E1 ({person}) --[REPORTS_TO]--> E2 ({person2}, CEO).\n"
            "Step 2: Claim states {person} reports directly to CEO {person2}.\n"
            "Step 3: Perfect match — relationship, direction, and entities all align.\n"
            "Confidence: High — exact KG match."
        ),
        "answer": (
            "**Claim VERIFIED.** The knowledge graph confirms that {person} (E1, {role}) "
            "reports directly to {person2} (E2, CEO). Source: {doc_id1}."
        ),
    },
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{company2}\", type: Organization, doc_sources: [{doc_id2}]\n"
            "relationships:\n"
            "  - E1 --[SUBSIDIARY_OF]--> E2, since: {year}, source: {doc_id1}\n"
            "</kg_context>\n\n"
            "Claim: \"{company2} is a subsidiary of {company}.\"\n"
            "Is this correct?"
        ),
        "reasoning": (
            "Step 1: KG shows E1 ({company}) --[SUBSIDIARY_OF]--> E2 ({company2}).\n"
            "Step 2: This means {company} is a subsidiary OF {company2} — not the other way.\n"
            "Step 3: The claim reverses the relationship direction.\n"
            "Step 4: {company2} is the parent; {company} is the subsidiary.\n"
            "Confidence: High — relationship direction is unambiguous in KG."
        ),
        "answer": (
            "**Claim REJECTED.** The relationship direction is reversed. The KG shows:\n\n"
            "- {company} (E1) --[SUBSIDIARY_OF]--> {company2} (E2)\n\n"
            "This means {company} is the subsidiary of {company2}, not the other way around. "
            "Source: {doc_id1}."
        ),
    },
]

_FACT_CHECK_DPO: List[Dict[str, str]] = [
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{person}\", type: Person, role: \"{role}\", doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}]\n"
            "relationships:\n"
            "  - E1 --[WORKS_AT]--> E2, since: {year}, source: {doc_id1}\n"
            "</kg_context>\n\n"
            "Claim: \"{person} joined {company} in {year2} as {role}.\"\n"
            "Verify."
        ),
        "chosen_reasoning": (
            "Step 1: KG: E1 ({person}, {role}) WORKS_AT E2 ({company}) since {year}.\n"
            "Step 2: Claim says joined in {year2} — date differs from KG ({year}).\n"
            "Step 3: Role matches ({role}).\n"
            "Step 4: Partial match: role correct, date incorrect.\n"
            "Confidence: High — KG date is clear."
        ),
        "chosen_answer": (
            "**Partially verified.** The role is correct — {person} (E1) is indeed a {role} at "
            "{company} (E2). However, the KG records the start date as {year}, not {year2} as "
            "claimed. Source: {doc_id1}."
        ),
        "rejected_reasoning": "Checking the claim.",
        "rejected_answer": "The claim seems mostly correct.",
    },
]


# ===================================================================
# 5. MISSING RELATIONSHIP DETECTION (200)
# ===================================================================

_MISSING_REL_SFT: List[Dict[str, str]] = [
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{person}\", type: Person, role: \"{role}\", doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{person2}\", type: Person, role: \"{role2}\", doc_sources: [{doc_id2}]\n"
            "  - id: E3, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}]\n"
            "  - id: E4, name: \"{company2}\", type: Organization, doc_sources: [{doc_id2}]\n"
            "relationships:\n"
            "  - E1 --[WORKS_AT]--> E3, source: {doc_id1}\n"
            "  - E2 --[WORKS_AT]--> E4, source: {doc_id2}\n"
            "</kg_context>\n\n"
            "Is there any relationship between {person} and {person2}?"
        ),
        "reasoning": (
            "Step 1: E1 ({person}) works at E3 ({company}).\n"
            "Step 2: E2 ({person2}) works at E4 ({company2}).\n"
            "Step 3: No direct relationship between E1 and E2 in the KG.\n"
            "Step 4: No indirect relationship — their employers (E3, E4) are also unrelated.\n"
            "Step 5: The KG contains no evidence of any connection.\n"
            "Confidence: Medium — absence of evidence is not evidence of absence."
        ),
        "answer": (
            "**No relationship found** between {person} (E1) and {person2} (E2) in the knowledge "
            "graph. They work at different organizations ({company} and {company2} respectively), "
            "and no direct or indirect relationship edges connect them.\n\n"
            "Note: This means no documented relationship exists in the indexed documents. "
            "A relationship may exist in documents not yet indexed."
        ),
    },
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{company2}\", type: Organization, doc_sources: [{doc_id2}]\n"
            "relationships:\n"
            "  (none)\n"
            "</kg_context>\n\n"
            "What is the business relationship between {company} and {company2}?"
        ),
        "reasoning": (
            "Step 1: Both entities exist in the KG but from different documents.\n"
            "Step 2: The relationships section is empty — no edges at all.\n"
            "Step 3: Cannot determine any business relationship from the KG.\n"
            "Step 4: The documents that mention these companies may not describe their relationship.\n"
            "Confidence: Low — KG has no relationship data."
        ),
        "answer": (
            "The knowledge graph contains no documented relationship between {company} (E1) "
            "and {company2} (E2). The entities appear in separate documents ({doc_id1} and "
            "{doc_id2}) with no relationship edges connecting them.\n\n"
            "To establish their business relationship, additional documents mentioning both "
            "companies would need to be indexed (e.g., contracts, partnership agreements, "
            "or correspondence between them)."
        ),
    },
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{person}\", type: Person, role: \"{role}\", doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{contract_id}\", type: Contract, value: \"{currency}{amount}\", doc_sources: [{doc_id2}]\n"
            "  - id: E3, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}, {doc_id2}]\n"
            "relationships:\n"
            "  - E1 --[WORKS_AT]--> E3, source: {doc_id1}\n"
            "  - E3 --[CONTRACTED]--> E2, source: {doc_id2}\n"
            "</kg_context>\n\n"
            "Did {person} sign {contract_id}?"
        ),
        "reasoning": (
            "Step 1: E1 ({person}) works at E3 ({company}).\n"
            "Step 2: E3 has a CONTRACTED relationship with E2 ({contract_id}).\n"
            "Step 3: No SIGNED relationship exists between E1 and E2.\n"
            "Step 4: {person} works at the company that holds the contract, but there's no "
            "evidence they personally signed it.\n"
            "Step 5: The absence of a SIGNED edge means either they didn't sign or it wasn't recorded.\n"
            "Confidence: Medium — can only confirm absence from KG, not reality."
        ),
        "answer": (
            "The knowledge graph does not contain a SIGNED relationship between {person} (E1) "
            "and {contract_id} (E2). While {person} works at {company} (E3), which holds the "
            "contract, there is no evidence in the indexed documents that {person} personally "
            "signed it.\n\n"
            "The actual signatory may be recorded in {doc_id2} but not yet extracted into the KG."
        ),
    },
]

_MISSING_REL_DPO: List[Dict[str, str]] = [
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{person}\", type: Person, doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{person2}\", type: Person, doc_sources: [{doc_id2}]\n"
            "relationships:\n"
            "  (none)\n"
            "</kg_context>\n\n"
            "Does {person} manage {person2}?"
        ),
        "chosen_reasoning": (
            "Step 1: No relationships exist in the KG at all.\n"
            "Step 2: E1 and E2 appear in different documents with no connecting edges.\n"
            "Step 3: Cannot confirm or deny a management relationship.\n"
            "Step 4: Must clearly communicate the absence and its limitations.\n"
            "Confidence: Low — no data available."
        ),
        "chosen_answer": (
            "The knowledge graph contains **no relationships** between {person} (E1) and "
            "{person2} (E2), including no MANAGES relationship. The entities appear in separate "
            "documents ({doc_id1}, {doc_id2}) with no connecting edges.\n\n"
            "Important: this only reflects what has been indexed. The actual management "
            "relationship may exist in unindexed documents (e.g., org charts, HR records)."
        ),
        "rejected_reasoning": "No relationship found.",
        "rejected_answer": "No, {person} does not manage {person2}.",
    },
]


# ===================================================================
# 6. ONTOLOGY-AWARE REASONING (250)
# ===================================================================

_ONTOLOGY_SFT: List[Dict[str, str]] = [
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{person}\", type: Person, role: \"General Counsel\", doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{company}\", type: Organization, industry: \"legal\", doc_sources: [{doc_id1}]\n"
            "  - id: E3, name: \"{contract_id}\", type: Contract, subtype: \"NDA\", doc_sources: [{doc_id1}]\n"
            "  - id: E4, name: \"{company2}\", type: Organization, doc_sources: [{doc_id2}]\n"
            "relationships:\n"
            "  - E1 --[REVIEWED]--> E3, date: {date1}, source: {doc_id1}\n"
            "  - E1 --[APPROVED]--> E3, date: {date2}, source: {doc_id1}\n"
            "  - E3 --[GOVERNS]--> E2, source: {doc_id1}\n"
            "  - E3 --[GOVERNS]--> E4, source: {doc_id1}\n"
            "</kg_context>\n\n"
            "What is the legal review status of {contract_id}?"
        ),
        "reasoning": (
            "Step 1: E3 ({contract_id}) is an NDA (legal domain ontology).\n"
            "Step 2: E1 ({person}, General Counsel) performed two actions:\n"
            "  - REVIEWED on {date1}\n"
            "  - APPROVED on {date2}\n"
            "Step 3: In legal ontology, REVIEWED + APPROVED by General Counsel = fully cleared.\n"
            "Step 4: The NDA governs both E2 ({company}) and E4 ({company2}).\n"
            "Step 5: Legal review workflow: draft -> review -> approve -> execute.\n"
            "Confidence: High — legal workflow semantics are clear."
        ),
        "answer": (
            "{contract_id} (E3, NDA) has completed legal review:\n\n"
            "- **Reviewed** by {person} (E1, General Counsel) on {date1}.\n"
            "- **Approved** by {person} (E1, General Counsel) on {date2}.\n\n"
            "In legal domain workflow, review + approval by General Counsel indicates the NDA "
            "is legally cleared. The NDA governs the relationship between {company} (E2) and "
            "{company2} (E4). Next step would be execution/signing."
        ),
    },
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{person}\", type: Person, role: \"External Auditor\", doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}]\n"
            "  - id: E3, name: \"FY{year} Financial Statements\", type: Document, subtype: \"audit_report\", doc_sources: [{doc_id1}]\n"
            "relationships:\n"
            "  - E1 --[AUDITED]--> E3, date: {date1}, source: {doc_id1}\n"
            "  - E3 --[DESCRIBES]--> E2, source: {doc_id1}\n"
            "</kg_context>\n\n"
            "What is the audit status of {company}'s FY{year} financials?"
        ),
        "reasoning": (
            "Step 1: E3 is the FY{year} Financial Statements — an audit report.\n"
            "Step 2: E1 ({person}, External Auditor) AUDITED E3 on {date1}.\n"
            "Step 3: In financial ontology, an external auditor performing an AUDITED action "
            "means the audit has been conducted.\n"
            "Step 4: The relationship type + auditor role indicates completed audit.\n"
            "Step 5: No 'qualified opinion' or 'adverse' markers in the KG.\n"
            "Confidence: High — audit completion is documented."
        ),
        "answer": (
            "{company}'s (E2) FY{year} Financial Statements (E3) have been audited:\n\n"
            "- **Auditor**: {person} (E1, External Auditor)\n"
            "- **Audit Date**: {date1}\n"
            "- **Document Type**: Audit report\n\n"
            "The audit has been completed. The KG does not contain opinion qualifications; "
            "check the source document ({doc_id1}) for the auditor's opinion type "
            "(unqualified, qualified, adverse, or disclaimer)."
        ),
    },
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{person}\", type: Person, role: \"HR Director\", doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{person2}\", type: Person, role: \"Employee\", doc_sources: [{doc_id1}]\n"
            "  - id: E3, name: \"Termination Notice\", type: Document, subtype: \"hr_action\", doc_sources: [{doc_id1}]\n"
            "  - id: E4, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}]\n"
            "relationships:\n"
            "  - E1 --[AUTHORED]--> E3, date: {date1}, source: {doc_id1}\n"
            "  - E3 --[APPLIES_TO]--> E2, effective_date: {date2}, source: {doc_id1}\n"
            "  - E2 --[WORKS_AT]--> E4, since: {year}, source: {doc_id1}\n"
            "</kg_context>\n\n"
            "What is {person2}'s employment status at {company}?"
        ),
        "reasoning": (
            "Step 1: E2 ({person2}) has a WORKS_AT relationship with E4 ({company}) since {year}.\n"
            "Step 2: E3 (Termination Notice) APPLIES_TO E2 with effective date {date2}.\n"
            "Step 3: E1 ({person}, HR Director) authored the termination notice on {date1}.\n"
            "Step 4: In HR ontology, a Termination Notice applying to an employee indicates "
            "employment will end on the effective date.\n"
            "Step 5: {person2}'s employment status depends on whether {date2} has passed.\n"
            "Confidence: High — HR document semantics are clear."
        ),
        "answer": (
            "{person2}'s (E2) employment at {company} (E4) is subject to a Termination Notice (E3):\n\n"
            "- **Employed since**: {year}\n"
            "- **Termination Notice authored by**: {person} (E1, HR Director) on {date1}\n"
            "- **Effective termination date**: {date2}\n\n"
            "In HR terms, {person2} is either terminated or in a notice period depending on "
            "whether {date2} has passed. Their employment was active from {year} until {date2}."
        ),
    },
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{company}\", type: Organization, industry: \"insurance\", doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"Claim_{contract_id}\", type: Insurance_Claim, status: \"under_review\", amount: \"{currency}{amount}\", doc_sources: [{doc_id1}]\n"
            "  - id: E3, name: \"{person}\", type: Person, role: \"Claims Adjuster\", doc_sources: [{doc_id1}]\n"
            "relationships:\n"
            "  - E3 --[REVIEWS]--> E2, assigned_date: {date1}, source: {doc_id1}\n"
            "  - E2 --[FILED_AGAINST]--> E1, filing_date: {date2}, source: {doc_id1}\n"
            "</kg_context>\n\n"
            "What is the status of the insurance claim against {company}?"
        ),
        "reasoning": (
            "Step 1: E2 is an Insurance_Claim with status 'under_review' for {currency}{amount}.\n"
            "Step 2: In insurance ontology, 'under_review' means assigned but not adjudicated.\n"
            "Step 3: E3 ({person}, Claims Adjuster) is reviewing since {date1}.\n"
            "Step 4: Claim was filed against {company} (E1) on {date2}.\n"
            "Step 5: No resolution relationship exists — claim is still open.\n"
            "Confidence: High — status field is explicit."
        ),
        "answer": (
            "Insurance claim Claim_{contract_id} (E2) against {company} (E1):\n\n"
            "- **Status**: Under review\n"
            "- **Claim Amount**: {currency}{amount}\n"
            "- **Filed**: {date2}\n"
            "- **Assigned to**: {person} (E3, Claims Adjuster) since {date1}\n\n"
            "The claim has not been adjudicated. In insurance workflow, 'under_review' indicates "
            "the adjuster is evaluating the claim but no decision has been rendered."
        ),
    },
]

_ONTOLOGY_DPO: List[Dict[str, str]] = [
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{person}\", type: Person, role: \"Compliance Officer\", doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{company}\", type: Organization, industry: \"financial\", doc_sources: [{doc_id1}]\n"
            "  - id: E3, name: \"AML Report {year}\", type: Document, subtype: \"compliance_report\", doc_sources: [{doc_id1}]\n"
            "relationships:\n"
            "  - E1 --[AUTHORED]--> E3, date: {date1}, source: {doc_id1}\n"
            "  - E3 --[COVERS]--> E2, period: \"FY{year}\", source: {doc_id1}\n"
            "</kg_context>\n\n"
            "Is {company} AML compliant?"
        ),
        "chosen_reasoning": (
            "Step 1: E3 is an AML (Anti-Money Laundering) Report for FY{year}.\n"
            "Step 2: Authored by E1 ({person}, Compliance Officer) on {date1}.\n"
            "Step 3: The existence of a compliance report doesn't mean compliance — only that "
            "a report was produced.\n"
            "Step 4: In financial regulatory ontology, AML compliance requires: report filed, "
            "findings addressed, no outstanding violations.\n"
            "Step 5: KG shows report exists but no findings or violation data.\n"
            "Confidence: Low — report existence ≠ compliance."
        ),
        "chosen_answer": (
            "The KG confirms that an AML Report for FY{year} (E3) exists, authored by "
            "{person} (E1, Compliance Officer) on {date1}. However, the existence of a "
            "report does not confirm compliance.\n\n"
            "To determine AML compliance, I would need:\n"
            "- Report findings and any identified violations\n"
            "- Remediation status for flagged issues\n"
            "- Regulatory examination results\n\n"
            "The KG only confirms that the reporting obligation was met."
        ),
        "rejected_reasoning": "There's an AML report, so they're compliant.",
        "rejected_answer": "Yes, {company} is AML compliant — they have a compliance report.",
    },
]


# ===================================================================
# 7. KG CONTEXT FORMAT TRAINING (250)
# ===================================================================

_KG_FORMAT_SFT: List[Dict[str, str]] = [
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}, {doc_id2}]\n"
            "  - id: E2, name: \"{person}\", type: Person, role: \"{role}\", doc_sources: [{doc_id1}]\n"
            "relationships:\n"
            "  - E2 --[WORKS_AT]--> E1, since: {year}, source: {doc_id1}\n"
            "</kg_context>\n\n"
            "Summarize the entities and relationships in this knowledge graph context."
        ),
        "reasoning": (
            "Step 1: Parse the KG context block.\n"
            "Step 2: Entities: E1 ({company}, Organization), E2 ({person}, Person, {role}).\n"
            "Step 3: Relationships: E2 works at E1 since {year}.\n"
            "Step 4: This is a simple employer-employee relationship graph.\n"
            "Step 5: Source documents: {doc_id1} and {doc_id2} for E1; {doc_id1} for E2.\n"
            "Confidence: High — simple KG with clear structure."
        ),
        "answer": (
            "**Knowledge Graph Summary:**\n\n"
            "**Entities (2):**\n"
            "- E1: {company} (Organization) — referenced in {doc_id1}, {doc_id2}\n"
            "- E2: {person} ({role}, Person) — referenced in {doc_id1}\n\n"
            "**Relationships (1):**\n"
            "- E2 ({person}) --[WORKS_AT]--> E1 ({company}), since {year}\n\n"
            "This represents a simple employment relationship between an individual and an organization."
        ),
    },
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{company2}\", type: Organization, doc_sources: [{doc_id2}]\n"
            "  - id: E3, name: \"{person}\", type: Person, role: \"{role}\", doc_sources: [{doc_id1}]\n"
            "  - id: E4, name: \"{person2}\", type: Person, role: \"{role2}\", doc_sources: [{doc_id2}]\n"
            "  - id: E5, name: \"{contract_id}\", type: Contract, value: \"{currency}{amount}\", doc_sources: [{doc_id1}, {doc_id2}]\n"
            "relationships:\n"
            "  - E3 --[WORKS_AT]--> E1, source: {doc_id1}\n"
            "  - E4 --[WORKS_AT]--> E2, source: {doc_id2}\n"
            "  - E3 --[SIGNED]--> E5, date: {date1}, source: {doc_id1}\n"
            "  - E4 --[SIGNED]--> E5, date: {date2}, source: {doc_id2}\n"
            "  - E1 --[CONTRACTED]--> E5, source: {doc_id1}\n"
            "  - E2 --[CONTRACTED]--> E5, source: {doc_id2}\n"
            "</kg_context>\n\n"
            "Based on the KG context, describe the business transaction."
        ),
        "reasoning": (
            "Step 1: Parse entities — 2 organizations, 2 people, 1 contract.\n"
            "Step 2: Parse relationships — employment, signing, and contracting.\n"
            "Step 3: Reconstruct: {company} and {company2} entered into {contract_id}.\n"
            "Step 4: Signatories: {person} ({role}) for {company}, {person2} ({role2}) for {company2}.\n"
            "Step 5: Contract value: {currency}{amount}.\n"
            "Confidence: High — clear transactional structure."
        ),
        "answer": (
            "**Business Transaction:**\n\n"
            "{company} (E1) and {company2} (E2) entered into {contract_id} (E5), "
            "valued at {currency}{amount}.\n\n"
            "**Signatories:**\n"
            "- {person} (E3, {role} at {company}) — signed on {date1}\n"
            "- {person2} (E4, {role2} at {company2}) — signed on {date2}\n\n"
            "**Sources:** {doc_id1} (for {company}'s side), {doc_id2} (for {company2}'s side)."
        ),
    },
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{person}\", type: Person, role: \"{role}\", doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{person2}\", type: Person, role: \"{role2}\", doc_sources: [{doc_id1}]\n"
            "  - id: E3, name: \"{person3}\", type: Person, role: \"CEO\", doc_sources: [{doc_id1}]\n"
            "  - id: E4, name: \"{dept}\", type: Department, doc_sources: [{doc_id1}]\n"
            "  - id: E5, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}]\n"
            "relationships:\n"
            "  - E1 --[REPORTS_TO]--> E2, source: {doc_id1}\n"
            "  - E2 --[REPORTS_TO]--> E3, source: {doc_id1}\n"
            "  - E1 --[MEMBER_OF]--> E4, source: {doc_id1}\n"
            "  - E2 --[MANAGES]--> E4, source: {doc_id1}\n"
            "  - E4 --[PART_OF]--> E5, source: {doc_id1}\n"
            "</kg_context>\n\n"
            "Draw the organizational hierarchy from this KG context."
        ),
        "reasoning": (
            "Step 1: Top level — E3 ({person3}, CEO) at E5 ({company}).\n"
            "Step 2: E2 ({person2}, {role2}) reports to E3 and manages E4 ({dept}).\n"
            "Step 3: E1 ({person}, {role}) reports to E2 and is a member of E4.\n"
            "Step 4: E4 ({dept}) is part of E5 ({company}).\n"
            "Step 5: Hierarchy: CEO -> {role2} -> {role}, within {dept}.\n"
            "Confidence: High — KG provides complete hierarchy."
        ),
        "answer": (
            "**Organizational Hierarchy:**\n\n"
            "```\n"
            "{company} (E5)\n"
            " └── {person3} (E3, CEO)\n"
            "      └── {person2} (E2, {role2}) — manages {dept}\n"
            "           └── {person} (E1, {role}) — member of {dept}\n"
            "```\n\n"
            "**Department:** {dept} (E4) is part of {company} (E5), managed by {person2}.\n"
            "**Reporting Chain:** {person} -> {person2} -> {person3} (CEO).\n"
            "Source: {doc_id1}."
        ),
    },
]

_KG_FORMAT_DPO: List[Dict[str, str]] = [
    {
        "query": (
            "<kg_context>\n"
            "entities:\n"
            "  - id: E1, name: \"{company}\", type: Organization, doc_sources: [{doc_id1}]\n"
            "  - id: E2, name: \"{person}\", type: Person, role: \"{role}\", doc_sources: [{doc_id1}]\n"
            "relationships:\n"
            "  - E2 --[WORKS_AT]--> E1, since: {year}, source: {doc_id1}\n"
            "</kg_context>\n\n"
            "Using the KG context, tell me about {person}."
        ),
        "chosen_reasoning": (
            "Step 1: Parse KG context — E2 is {person}, a {role}.\n"
            "Step 2: E2 works at E1 ({company}) since {year}.\n"
            "Step 3: Source is {doc_id1}.\n"
            "Step 4: Present findings with entity IDs and relationship details.\n"
            "Confidence: High — straightforward KG read."
        ),
        "chosen_answer": (
            "From the knowledge graph:\n\n"
            "{person} (E2) is a **{role}** at {company} (E1), employed since {year}. "
            "This information is sourced from {doc_id1}.\n\n"
            "The KG contains no additional relationships or attributes for this entity. "
            "More details may be available in the source document."
        ),
        "rejected_reasoning": "The person works at the company.",
        "rejected_answer": "{person} works at {company} as a {role}.",
    },
]


# ===================================================================
# PUBLIC API
# ===================================================================


def generate_track5_data(output_dir: str | Path, seed: int = 42) -> dict:
    """Generate Track 5 KG-Augmented Knowledge training data.

    Produces both SFT and DPO JSONL files with 2000 SFT examples and
    a proportional set of DPO preference pairs.

    Args:
        output_dir: Directory to write JSONL files into.
        seed: Random seed for deterministic generation.

    Returns:
        Dict with ``sft_path``, ``dpo_path``, ``sft_count``, ``dpo_count``.
    """
    output_dir = Path(output_dir)
    rng = random.Random(seed)

    # --- SFT generation ---
    sft_categories = [
        (_ENTITY_AWARE_SFT, 400),
        (_RELATIONSHIP_SFT, 350),
        (_CROSS_DOC_SFT, 300),
        (_FACT_CHECK_SFT, 250),
        (_MISSING_REL_SFT, 200),
        (_ONTOLOGY_SFT, 250),
        (_KG_FORMAT_SFT, 250),
    ]

    all_sft: List[Dict[str, str]] = []
    sub_seed = seed
    for templates, count in sft_categories:
        sub_seed += 1
        sub_rng = random.Random(sub_seed)
        all_sft.extend(_expand(templates, count, sub_rng, mode="sft"))

    rng.shuffle(all_sft)

    sft_path = output_dir / "track5_kg_sft.jsonl"
    with JSONLWriter(sft_path) as writer:
        for ex in all_sft:
            writer.write(ex)

    # --- DPO generation ---
    dpo_categories = [
        (_ENTITY_AWARE_DPO, 80),
        (_RELATIONSHIP_DPO, 70),
        (_CROSS_DOC_DPO, 60),
        (_FACT_CHECK_DPO, 50),
        (_MISSING_REL_DPO, 40),
        (_ONTOLOGY_DPO, 50),
        (_KG_FORMAT_DPO, 50),
    ]

    all_dpo: List[Dict[str, str]] = []
    sub_seed = seed + 100
    for templates, count in dpo_categories:
        sub_seed += 1
        sub_rng = random.Random(sub_seed)
        all_dpo.extend(_expand(templates, count, sub_rng, mode="dpo"))

    rng.shuffle(all_dpo)

    dpo_path = output_dir / "track5_kg_dpo.jsonl"
    with JSONLWriter(dpo_path) as writer:
        for ex in all_dpo:
            writer.write(ex)

    return {
        "sft_path": str(sft_path),
        "dpo_path": str(dpo_path),
        "sft_count": len(all_sft),
        "dpo_count": len(all_dpo),
    }
