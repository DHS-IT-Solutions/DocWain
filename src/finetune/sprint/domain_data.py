"""Domain Knowledge Data Generator for DocWain V2+ finetuning.

Generates 12,000 domain-aware training examples across 8 enterprise domains,
teaching DocWain to detect document domains and apply domain-specific reasoning.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Import helpers from sprint / v2 base
# ---------------------------------------------------------------------------

try:
    from src.finetune.v2.data_generator.base import format_sft_example
except ImportError:
    _SYSTEM = (
        "You are DocWain, an enterprise document intelligence assistant. "
        "You analyse documents with deep contextual understanding, extract "
        "structured information, identify patterns and anomalies, and provide "
        "holistic analysis grounded in evidence. You reason step-by-step before "
        "answering, state your confidence level, and cite specific sources. "
        "When information is insufficient, you say so clearly rather than guessing."
    )

    def format_sft_example(
        query: str,
        reasoning: str,
        answer: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, str]:
        sys_prompt = system_prompt or _SYSTEM
        text = (
            f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{answer}<|im_end|>"
        )
        return {"text": text}


try:
    from src.finetune.sprint.document_factory import generate_document, DOCUMENT_TYPES
except ImportError:
    DOCUMENT_TYPES = ["contract", "invoice", "financial_statement", "medical_record",
                      "resume", "technical_spec", "government_form", "insurance_claim"]

    def generate_document(doc_type: str, seed: Optional[int] = None) -> dict:  # type: ignore[misc]
        rng = random.Random(seed)
        return {
            "content": f"[Synthetic {doc_type} document — seed {seed}]\n"
                       f"Reference: {rng.randint(1000, 9999)}\nDate: 2026-01-{rng.randint(1, 28):02d}",
            "type": doc_type,
            "metadata": {"ground_truth": {}},
        }


# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------

DOMAINS: Dict[str, Dict[str, List[str]]] = {
    "financial": {
        "doc_types": [
            "financial_statement", "invoice", "audit_report",
            "purchase_order", "compliance_report",
        ],
        "reasoning_patterns": [
            "variance analysis",
            "cash flow assessment",
            "ratio analysis",
            "budget vs actual comparison",
            "revenue trend analysis",
            "expense categorization",
            "profitability assessment",
        ],
        "detection_cues": [
            "balance sheet", "income statement", "EBITDA", "revenue", "liabilities",
            "assets", "cash flow", "net profit", "fiscal year", "quarterly earnings",
            "accounts payable", "accounts receivable", "depreciation", "amortization",
        ],
    },
    "legal": {
        "doc_types": [
            "contract", "legal_filing", "compliance_report", "policy",
        ],
        "reasoning_patterns": [
            "clause interdependency analysis",
            "obligation extraction",
            "risk clause identification",
            "jurisdiction analysis",
            "indemnification assessment",
            "termination clause review",
            "intellectual property analysis",
        ],
        "detection_cues": [
            "whereas", "hereinafter", "indemnify", "jurisdiction", "arbitration",
            "force majeure", "governing law", "breach", "remedy", "covenant",
            "representation", "warranty", "liability", "confidentiality", "termination",
        ],
    },
    "medical": {
        "doc_types": [
            "medical_record", "insurance_claim", "compliance_report",
        ],
        "reasoning_patterns": [
            "clinical timeline reconstruction",
            "medication interaction analysis",
            "diagnosis code mapping",
            "treatment outcome assessment",
            "patient risk stratification",
            "care plan compliance review",
            "adverse event identification",
        ],
        "detection_cues": [
            "patient", "diagnosis", "ICD-10", "CPT code", "prescription", "dosage",
            "clinical", "physician", "hospital", "treatment", "symptom", "prognosis",
            "HIPAA", "EHR", "lab results", "vital signs", "chief complaint",
        ],
    },
    "hr": {
        "doc_types": [
            "resume", "policy", "compliance_report", "meeting_notes",
        ],
        "reasoning_patterns": [
            "skills gap analysis",
            "compensation benchmarking",
            "performance trend analysis",
            "headcount planning",
            "attrition risk assessment",
            "policy compliance review",
            "recruitment funnel analysis",
        ],
        "detection_cues": [
            "employee", "onboarding", "performance review", "salary", "benefits",
            "PTO", "termination", "hiring", "job description", "competency",
            "workforce", "headcount", "org chart", "HR", "payroll",
        ],
    },
    "insurance": {
        "doc_types": [
            "insurance_claim", "policy", "compliance_report", "audit_report",
        ],
        "reasoning_patterns": [
            "claim validity assessment",
            "coverage gap analysis",
            "premium calculation review",
            "risk exposure quantification",
            "exclusion clause analysis",
            "subrogation potential assessment",
            "fraud indicator detection",
        ],
        "detection_cues": [
            "policyholder", "premium", "deductible", "claim", "coverage", "exclusion",
            "beneficiary", "underwriting", "loss ratio", "reinsurance", "subrogation",
            "endorsement", "rider", "insured", "indemnity",
        ],
    },
    "government": {
        "doc_types": [
            "government_form", "compliance_report", "policy", "audit_report",
        ],
        "reasoning_patterns": [
            "regulatory compliance mapping",
            "statutory obligation extraction",
            "public procurement assessment",
            "grant eligibility analysis",
            "FOI response review",
            "policy impact assessment",
            "budget allocation analysis",
        ],
        "detection_cues": [
            "regulation", "federal", "state", "municipality", "procurement",
            "statutory", "appropriations", "fiscal year", "agency", "department",
            "public record", "FOI", "grant", "compliance", "mandate",
        ],
    },
    "technical": {
        "doc_types": [
            "technical_spec", "compliance_report", "audit_report", "meeting_notes",
        ],
        "reasoning_patterns": [
            "architecture dependency analysis",
            "API compatibility assessment",
            "security vulnerability review",
            "performance bottleneck identification",
            "scalability gap analysis",
            "integration risk assessment",
            "technical debt quantification",
        ],
        "detection_cues": [
            "API", "endpoint", "architecture", "schema", "database", "microservice",
            "latency", "throughput", "SLA", "deployment", "CI/CD", "authentication",
            "encryption", "protocol", "interface", "specification",
        ],
    },
    "education": {
        "doc_types": [
            "policy", "compliance_report", "resume", "meeting_notes",
        ],
        "reasoning_patterns": [
            "learning outcome mapping",
            "curriculum gap analysis",
            "assessment alignment review",
            "accreditation compliance check",
            "student performance trend analysis",
            "resource allocation assessment",
            "pedagogical approach evaluation",
        ],
        "detection_cues": [
            "curriculum", "syllabus", "academic", "student", "faculty", "credits",
            "accreditation", "semester", "GPA", "enrollment", "learning objective",
            "assessment", "course", "instructor", "institution",
        ],
    },
}

# ---------------------------------------------------------------------------
# Difficulties
# ---------------------------------------------------------------------------

_DIFFICULTIES = ["easy", "medium", "hard"]

_DIFFICULTY_WEIGHTS = [0.3, 0.5, 0.2]

# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------


def _pick_doc_for_domain(domain: str, rng: random.Random, seed_offset: int) -> dict:
    """Pick a doc type that best fits the domain and generate a synthetic doc."""
    domain_info = DOMAINS[domain]
    preferred_types = domain_info["doc_types"]
    # Intersect with known DOCUMENT_TYPES, fallback to any
    available = [t for t in preferred_types if t in DOCUMENT_TYPES]
    if not available:
        available = DOCUMENT_TYPES
    doc_type = rng.choice(available)
    doc_seed = rng.randint(0, 2 ** 31 - 1) + seed_offset
    return generate_document(doc_type, seed=doc_seed)


def _detection_answer(domain: str, rng: random.Random) -> tuple[str, str, str]:
    """Return (query, reasoning, answer) for a domain detection example."""
    cues = DOMAINS[domain]["detection_cues"]
    patterns = DOMAINS[domain]["reasoning_patterns"]
    sampled_cues = rng.sample(cues, min(3, len(cues)))
    sampled_patterns = rng.sample(patterns, min(2, len(patterns)))

    query = "What domain does this document belong to? Identify the domain and explain the key signals that led to your classification."

    reasoning = (
        f"I need to analyse the document for domain-specific signals. "
        f"Key indicators I can observe: {', '.join(sampled_cues)}. "
        f"These cues are characteristic of {domain} documents. "
        f"The terminology, structure, and subject matter all align with the {domain} domain. "
        f"I am confident this is a {domain} document because of the presence of these distinctive markers."
    )

    answer = (
        f"**Domain: {domain.upper()}**\n\n"
        f"This document belongs to the **{domain}** domain based on the following signals:\n\n"
        + "\n".join(f"- **{cue}**: characteristic {domain} terminology" for cue in sampled_cues)
        + f"\n\nWith these signals identified, the appropriate analytical frameworks include: "
        + ", ".join(sampled_patterns)
        + "."
    )

    return query, reasoning, answer


def _reasoning_answer(domain: str, pattern: str, rng: random.Random) -> tuple[str, str, str]:
    """Return (query, reasoning, answer) for a domain reasoning example."""
    cues = DOMAINS[domain]["detection_cues"]
    sampled_cues = rng.sample(cues, min(4, len(cues)))

    query = f"Perform {pattern} on this document and provide your findings."

    reasoning = (
        f"I am applying {pattern} to this {domain} document. "
        f"First, I identify the relevant data points: {', '.join(sampled_cues[:2])}. "
        f"Then I apply {domain}-specific analytical frameworks. "
        f"I need to consider industry standards and domain conventions while performing this analysis. "
        f"I will structure my findings clearly with supporting evidence from the document."
    )

    findings = [
        f"The document contains {cue}-related information that is relevant to this analysis"
        for cue in sampled_cues
    ]

    answer = (
        f"**{pattern.title()} — {domain.upper()} Domain**\n\n"
        f"**Methodology:** Applied {domain}-specific {pattern} framework.\n\n"
        f"**Key Findings:**\n"
        + "\n".join(f"- {f}" for f in findings[:3])
        + f"\n\n**Assessment:** Based on the {pattern}, this {domain} document "
        f"demonstrates standard patterns consistent with industry norms. "
        f"Further review of {sampled_cues[-1]} may be warranted."
    )

    return query, reasoning, answer


def _cross_domain_answer(
    domain_a: str, domain_b: str, rng: random.Random
) -> tuple[str, str, str]:
    """Return (query, reasoning, answer) for a cross-domain analysis example."""
    cues_a = rng.sample(DOMAINS[domain_a]["detection_cues"], min(2, len(DOMAINS[domain_a]["detection_cues"])))
    cues_b = rng.sample(DOMAINS[domain_b]["detection_cues"], min(2, len(DOMAINS[domain_b]["detection_cues"])))
    pattern_a = rng.choice(DOMAINS[domain_a]["reasoning_patterns"])
    pattern_b = rng.choice(DOMAINS[domain_b]["reasoning_patterns"])

    query = (
        f"This document contains elements from both the {domain_a} and {domain_b} domains. "
        f"Identify the cross-domain aspects and perform an integrated analysis."
    )

    reasoning = (
        f"This is a cross-domain document touching {domain_a} and {domain_b}. "
        f"From the {domain_a} perspective, I see: {', '.join(cues_a)}. "
        f"From the {domain_b} perspective, I see: {', '.join(cues_b)}. "
        f"I need to apply both {pattern_a} and {pattern_b} to capture the full picture. "
        f"The intersection of these domains creates unique analytical considerations."
    )

    answer = (
        f"**Cross-Domain Analysis: {domain_a.upper()} × {domain_b.upper()}**\n\n"
        f"**{domain_a.title()} Dimension:**\n"
        + "\n".join(f"- {cue}: relevant to {domain_a} analysis" for cue in cues_a)
        + f"\n\nApplied framework: *{pattern_a}*\n\n"
        f"**{domain_b.title()} Dimension:**\n"
        + "\n".join(f"- {cue}: relevant to {domain_b} analysis" for cue in cues_b)
        + f"\n\nApplied framework: *{pattern_b}*\n\n"
        f"**Integration:** This document requires coordinated {domain_a} and {domain_b} "
        f"expertise to assess holistically. Findings from both dimensions must be reconciled."
    )

    return query, reasoning, answer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_domain_examples(
    domain: str,
    mode: str,
    count: int,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generate domain-specific training examples.

    Args:
        domain: One of the keys in DOMAINS.
        mode: "detection" or "reasoning".
        count: Number of examples to generate.
        seed: Optional random seed for reproducibility.

    Returns:
        List of example dicts with keys: text, category, domain, difficulty, source.
    """
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain '{domain}'. Valid: {list(DOMAINS.keys())}")
    if mode not in ("detection", "reasoning"):
        raise ValueError(f"Unknown mode '{mode}'. Valid: detection, reasoning")

    rng = random.Random(seed)
    examples: List[Dict[str, Any]] = []
    patterns = DOMAINS[domain]["reasoning_patterns"]

    for i in range(count):
        doc = _pick_doc_for_domain(domain, rng, seed_offset=i)
        difficulty = rng.choices(_DIFFICULTIES, weights=_DIFFICULTY_WEIGHTS, k=1)[0]

        if mode == "detection":
            query, reasoning, answer = _detection_answer(domain, rng)
            category = "domain_detection"
        else:
            pattern = patterns[i % len(patterns)]
            query, reasoning, answer = _reasoning_answer(domain, pattern, rng)
            category = "domain_reasoning"

        # Inject doc content into the query
        doc_snippet = doc["content"][:600]
        full_query = f"Document:\n\n{doc_snippet}\n\n---\n\n{query}"

        formatted = format_sft_example(full_query, reasoning, answer)
        examples.append({
            **formatted,
            "category": category,
            "domain": domain,
            "difficulty": difficulty,
            "source": "domain_injection",
        })

    return examples


def generate_cross_domain_examples(
    count: int,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generate cross-domain analysis examples by mixing two random domains.

    Args:
        count: Number of examples to generate.
        seed: Optional random seed for reproducibility.

    Returns:
        List of example dicts with keys: text, category, domain, difficulty, source.
    """
    rng = random.Random(seed)
    domain_keys = list(DOMAINS.keys())
    examples: List[Dict[str, Any]] = []

    for i in range(count):
        domain_a, domain_b = rng.sample(domain_keys, 2)
        difficulty = rng.choices(_DIFFICULTIES, weights=_DIFFICULTY_WEIGHTS, k=1)[0]

        # Pick a doc for the primary domain
        doc = _pick_doc_for_domain(domain_a, rng, seed_offset=i * 7)
        query, reasoning, answer = _cross_domain_answer(domain_a, domain_b, rng)

        doc_snippet = doc["content"][:600]
        full_query = f"Document:\n\n{doc_snippet}\n\n---\n\n{query}"

        formatted = format_sft_example(full_query, reasoning, answer)
        examples.append({
            **formatted,
            "category": "cross_domain",
            "domain": f"{domain_a}+{domain_b}",
            "difficulty": difficulty,
            "source": "domain_injection",
        })

    return examples


def generate_all_domain_data(seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Generate the full 12,000-example domain knowledge dataset.

    Breakdown:
        - 500 detection examples × 8 domains = 4,000
        - 800 reasoning examples × 8 domains = 6,400
        - 1,600 cross-domain examples

    Args:
        seed: Optional random seed for reproducibility.

    Returns:
        List of 12,000 example dicts.
    """
    all_examples: List[Dict[str, Any]] = []
    domain_keys = list(DOMAINS.keys())

    for idx, domain in enumerate(domain_keys):
        det_seed = None if seed is None else seed + idx * 1000
        all_examples.extend(generate_domain_examples(domain, "detection", count=500, seed=det_seed))

        reas_seed = None if seed is None else seed + idx * 1000 + 500
        all_examples.extend(generate_domain_examples(domain, "reasoning", count=800, seed=reas_seed))

    cross_seed = None if seed is None else seed + 99000
    all_examples.extend(generate_cross_domain_examples(count=1600, seed=cross_seed))

    return all_examples
