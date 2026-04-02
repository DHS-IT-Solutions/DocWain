"""Frozen evaluation suite generator for DocWain V2+ finetuning pipeline.

Generates exactly 500 held-out evaluation benchmark examples with a fixed
seed (42) for full reproducibility. Two runs always produce identical output.

10 benchmarks with fixed counts:
  - DocVQA-mini:       60
  - TableBench:        50
  - LayoutEval:        40
  - HalluBench:        50
  - ToolEval:          50
  - InsightEval:       50
  - SynthesisEval:     50
  - ConversationEval:  50
  - ConfidenceEval:    50
  - RegressionSuite:   50
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from src.finetune.v2.data_generator.base import (
    DOMAINS,
    DOC_TYPES,
    format_eval_example,
)

# ---------------------------------------------------------------------------
# Benchmark definitions (name -> count)
# ---------------------------------------------------------------------------

BENCHMARKS: Dict[str, int] = {
    "DocVQA-mini": 60,
    "TableBench": 50,
    "LayoutEval": 40,
    "HalluBench": 50,
    "ToolEval": 50,
    "InsightEval": 50,
    "SynthesisEval": 50,
    "ConversationEval": 50,
    "ConfidenceEval": 50,
    "RegressionSuite": 50,
}

_FIXED_SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CURRENCIES = ["$", "EUR", "GBP"]
_COMPANY_NAMES = [
    "Acme Corp", "Globex Industries", "Initech Solutions", "Umbrella LLC",
    "Stark Industries", "Wayne Enterprises", "Cyberdyne Systems", "Oscorp",
]
_PERSON_NAMES = [
    "Alice Johnson", "Bob Martinez", "Carol Singh", "David Chen",
    "Emily Watson", "Frank Osei", "Grace Kim", "Hector Rossi",
]
_DEPARTMENTS = [
    "Engineering", "Finance", "Human Resources", "Legal",
    "Marketing", "Operations", "Sales", "Compliance",
]
_PRODUCTS = [
    "Widget A", "Gadget Pro", "Module X-100", "Sensor Suite",
    "Platform License", "Cloud Tier 2",
]


def _pick(lst: list, rng: random.Random) -> Any:
    return rng.choice(lst)


def _amt(rng: random.Random) -> str:
    return f"{rng.uniform(500, 50000):,.2f}"


# ---------------------------------------------------------------------------
# Per-benchmark generators
# ---------------------------------------------------------------------------

def _gen_docvqa(rng: random.Random, n: int) -> List[Dict[str, Any]]:
    results = []
    for _ in range(n):
        company = _pick(_COMPANY_NAMES, rng)
        person = _pick(_PERSON_NAMES, rng)
        cur = _pick(_CURRENCIES, rng)
        amount = _amt(rng)
        domain = _pick(DOMAINS, rng)
        doc_type = _pick(DOC_TYPES, rng)
        results.append(format_eval_example(
            benchmark="DocVQA-mini",
            query=f"What is the total amount on the {company} {doc_type}?",
            context=f"{company} {domain} {doc_type}. Total: {cur}{amount}. Signed by {person}.",
            reference_answer=f"The total amount is {cur}{amount}.",
            rubric={"accuracy": 1.0, "grounding": 1.0},
        ))
    return results


def _gen_tablebench(rng: random.Random, n: int) -> List[Dict[str, Any]]:
    results = []
    for _ in range(n):
        company = _pick(_COMPANY_NAMES, rng)
        dept = _pick(_DEPARTMENTS, rng)
        cur = _pick(_CURRENCIES, rng)
        a1, a2 = _amt(rng), _amt(rng)
        results.append(format_eval_example(
            benchmark="TableBench",
            query=f"What is the {dept} budget from {company}'s table?",
            context=f"| Department | Budget |\n| {dept} | {cur}{a1} |\n| Operations | {cur}{a2} |",
            reference_answer=f"The {dept} budget is {cur}{a1}.",
            rubric={"accuracy": 1.0, "table_parsing": 1.0},
        ))
    return results


def _gen_layouteval(rng: random.Random, n: int) -> List[Dict[str, Any]]:
    results = []
    for _ in range(n):
        company = _pick(_COMPANY_NAMES, rng)
        person = _pick(_PERSON_NAMES, rng)
        domain = _pick(DOMAINS, rng)
        results.append(format_eval_example(
            benchmark="LayoutEval",
            query=f"Identify the header and signatory from this {domain} document layout.",
            context=f"[HEADER: {company} Official Document]\n[BODY: Agreement terms...]\n[FOOTER: Signed by {person}]",
            reference_answer=f"Header: {company} Official Document. Signatory: {person}.",
            rubric={"layout_accuracy": 1.0, "field_extraction": 1.0},
        ))
    return results


def _gen_hallubench(rng: random.Random, n: int) -> List[Dict[str, Any]]:
    results = []
    for _ in range(n):
        company = _pick(_COMPANY_NAMES, rng)
        cur = _pick(_CURRENCIES, rng)
        amount = _amt(rng)
        results.append(format_eval_example(
            benchmark="HalluBench",
            query=f"What is {company}'s CEO name and annual revenue?",
            context=f"{company} revenue: {cur}{amount}. No CEO information in document.",
            reference_answer=f"Revenue is {cur}{amount}. CEO name is not available in the provided document.",
            rubric={"hallucination_avoidance": 1.0, "honesty": 1.0},
        ))
    return results


def _gen_tooleval(rng: random.Random, n: int) -> List[Dict[str, Any]]:
    results = []
    tools = ["table_extract", "entity_extract", "summarize", "cross_reference", "ocr_correct"]
    for _ in range(n):
        tool = _pick(tools, rng)
        company = _pick(_COMPANY_NAMES, rng)
        domain = _pick(DOMAINS, rng)
        results.append(format_eval_example(
            benchmark="ToolEval",
            query=f"Use the appropriate tool to process this {domain} document from {company}.",
            context=f"{company} {domain} document requiring {tool} processing.",
            reference_answer=f"The appropriate tool is {tool}.",
            rubric={"tool_selection": 1.0, "execution": 1.0},
            expected_tools=[tool],
        ))
    return results


def _gen_insighteval(rng: random.Random, n: int) -> List[Dict[str, Any]]:
    categories = ["pattern_recognition", "anomaly_detection", "trend_analysis", "comparative_analysis"]
    results = []
    for _ in range(n):
        cat = _pick(categories, rng)
        company = _pick(_COMPANY_NAMES, rng)
        dept = _pick(_DEPARTMENTS, rng)
        cur = _pick(_CURRENCIES, rng)
        a1, a2, a3 = _amt(rng), _amt(rng), _amt(rng)
        results.append(format_eval_example(
            benchmark="InsightEval",
            query=f"Provide a {cat.replace('_', ' ')} insight on {company}'s {dept} data.",
            context=f"{dept} Q1: {cur}{a1}, Q2: {cur}{a2}, Q3: {cur}{a3}",
            reference_answer=f"A {cat.replace('_', ' ')} analysis of {company}'s {dept} shows variation across quarters.",
            rubric={"insight_quality": 1.0, "category_match": 1.0, "expected_category": cat},
        ))
    return results


def _gen_synthesiseval(rng: random.Random, n: int) -> List[Dict[str, Any]]:
    results = []
    for _ in range(n):
        company = _pick(_COMPANY_NAMES, rng)
        person = _pick(_PERSON_NAMES, rng)
        dept = _pick(_DEPARTMENTS, rng)
        domain = _pick(DOMAINS, rng)
        cur = _pick(_CURRENCIES, rng)
        a1, a2 = _amt(rng), _amt(rng)
        results.append(format_eval_example(
            benchmark="SynthesisEval",
            query=f"Synthesize all available information about {company}'s {domain} operations.",
            context=(
                f"Source 1: {company} {domain} report — budget {cur}{a1}.\n"
                f"Source 2: {person} memo — {dept} restructuring planned.\n"
                f"Source 3: Vendor invoice — {cur}{a2} outstanding."
            ),
            reference_answer=(
                f"{company}'s {domain} operations show a budget of {cur}{a1}, "
                f"planned {dept} restructuring per {person}, and {cur}{a2} in outstanding vendor payments."
            ),
            rubric={"completeness": 1.0, "coherence": 1.0, "source_coverage": 1.0},
        ))
    return results


def _gen_conversationeval(rng: random.Random, n: int) -> List[Dict[str, Any]]:
    results = []
    queries = [
        "Hi, can you look at this for me?",
        "What about the payment terms?",
        "Actually, switch to the vendor report.",
        "Show me {person}'s file.",
        "Thanks, that's all I need.",
    ]
    for _ in range(n):
        q = _pick(queries, rng).format(person=_pick(_PERSON_NAMES, rng))
        results.append(format_eval_example(
            benchmark="ConversationEval",
            query=q,
            context="Multi-turn document analysis conversation.",
            reference_answer="Natural, context-aware response maintaining conversational flow.",
            rubric={"naturalness": 1.0, "context_maintenance": 1.0, "helpfulness": 1.0},
        ))
    return results


def _gen_confidenceeval(rng: random.Random, n: int) -> List[Dict[str, Any]]:
    tiers = ["high", "medium", "low", "refusal"]
    results = []
    for _ in range(n):
        tier = _pick(tiers, rng)
        company = _pick(_COMPANY_NAMES, rng)
        domain = _pick(DOMAINS, rng)
        if tier == "high":
            ctx = f"Explicit value: {_pick(_CURRENCIES, rng)}{_amt(rng)} in {company} document."
            ref = "Clear answer with high confidence."
        elif tier == "medium":
            ctx = f"Two partially conflicting sources about {company}'s {domain} data."
            ref = "Answer noting conflict with medium confidence."
        elif tier == "low":
            ctx = f"Single informal mention in an email about {company}."
            ref = "Tentative answer with low confidence and caveats."
        else:
            ctx = f"No relevant documents found for the query about {company}."
            ref = "Refusal to answer with explanation of missing data."
        results.append(format_eval_example(
            benchmark="ConfidenceEval",
            query=f"What is {company}'s {domain} status?",
            context=ctx,
            reference_answer=ref,
            rubric={"calibration": 1.0, "expected_tier": tier},
        ))
    return results


def _gen_regressionsuite(rng: random.Random, n: int) -> List[Dict[str, Any]]:
    results = []
    scenarios = [
        ("simple_lookup", "What is the value?", "Value: {cur}{a1}", "The value is {cur}{a1}."),
        ("table_read", "Read row 1.", "| Col | Val |\n| A | {cur}{a1} |", "Row 1: A = {cur}{a1}."),
        ("entity_extract", "Who signed?", "Signed by {person}.", "{person} signed the document."),
        ("multi_table", "Compare tables.", "T1: {cur}{a1}, T2: {cur}{a2}", "T1={cur}{a1}, T2={cur}{a2}."),
        ("ocr_fix", "Fix OCR.", "Arnount: {cur}{a1}", "Amount: {cur}{a1}."),
    ]
    for _ in range(n):
        scenario = _pick(scenarios, rng)
        person = _pick(_PERSON_NAMES, rng)
        cur = _pick(_CURRENCIES, rng)
        a1, a2 = _amt(rng), _amt(rng)
        subs = {"cur": cur, "a1": a1, "a2": a2, "person": person}
        results.append(format_eval_example(
            benchmark="RegressionSuite",
            query=scenario[1].format(**subs),
            context=scenario[2].format(**subs),
            reference_answer=scenario[3].format(**subs),
            rubric={"accuracy": 1.0, "scenario": scenario[0]},
        ))
    return results


# Map benchmark name -> generator function
_BENCHMARK_GENERATORS = {
    "DocVQA-mini": _gen_docvqa,
    "TableBench": _gen_tablebench,
    "LayoutEval": _gen_layouteval,
    "HalluBench": _gen_hallubench,
    "ToolEval": _gen_tooleval,
    "InsightEval": _gen_insighteval,
    "SynthesisEval": _gen_synthesiseval,
    "ConversationEval": _gen_conversationeval,
    "ConfidenceEval": _gen_confidenceeval,
    "RegressionSuite": _gen_regressionsuite,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_eval_suite() -> List[Dict[str, Any]]:
    """Generate the frozen 500-example evaluation suite.

    Uses a fixed seed (42) for full reproducibility. Two calls always
    produce identical output.

    Returns:
        List of 500 eval example dicts.
    """
    rng = random.Random(_FIXED_SEED)
    results: List[Dict[str, Any]] = []

    for benchmark, count in BENCHMARKS.items():
        gen_fn = _BENCHMARK_GENERATORS[benchmark]
        results.extend(gen_fn(rng, count))

    return results


def write_eval_suite(output_dir: Path) -> None:
    """Generate and write the frozen eval suite to JSONL.

    Args:
        output_dir: Directory to write ``eval_suite.jsonl`` into.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = generate_eval_suite()

    path = output_dir / "eval_suite.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
