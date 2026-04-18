"""DocWain V5 Capability Charter — the 12-capability training contract.

Every training row is tagged with exactly one capability_id from this
file. Every gate in ``evaluate.py`` maps to one of these capabilities
via ``gate_threshold``. When someone asks "is V5 better than V3 at
layout understanding?", the answer comes from the ``evaluate.py`` run
scored against this charter.

This module is intentionally data-only — no runtime logic beyond
lookups. It's the single source of truth so drift between data gen,
training, and eval is impossible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Capability:
    """One capability the V5 model must learn to reflex."""

    capability_id: str
    name: str
    description: str
    # How many SFT rows aim at this capability (target — actual may be less
    # if teacher ensemble rejects a lot)
    sft_target_rows: int
    # How many DPO pairs feature this capability
    dpo_target_pairs: int
    # Where the gate lives on the 5-point LLM-judge scale, 0-1 scale, or F1
    gate_metric: str
    gate_threshold: float
    # Failure on this gate is a hard block vs degradation (hard = ship fails)
    hard_gate: bool
    # Which eval set exercises this capability
    eval_set: str
    # How the data generator's prompt should frame the task
    prompt_style: str
    # Notes for the data generator about what counts as "hard" for this cap
    difficulty_notes: str = ""


# The charter. Order here is the order they're taught in SFT curriculum.
CHARTER: Dict[str, Capability] = {
    "layout_understanding": Capability(
        capability_id="layout_understanding",
        name="Layout understanding",
        description=(
            "Given a document's text + layout tokens, identify semantic "
            "regions (header, body, footer, tables, signatures) and infer "
            "2-D relationships (what's next to what, what's under what)."
        ),
        sft_target_rows=10_000,
        dpo_target_pairs=1_000,
        gate_metric="F1",
        gate_threshold=0.85,
        hard_gate=False,
        eval_set="layout_regions_v5",
        prompt_style="layout-annotated",
        difficulty_notes=(
            "Hard examples: multi-column, nested tables, footer repeated "
            "every page, forms with aligned fields"
        ),
    ),
    "domain_recognition": Capability(
        capability_id="domain_recognition",
        name="Domain recognition",
        description=(
            "Classify document domain (invoice/PO/quote/contract/statement/"
            "clinical/legal/technical/resume) from the first 200 tokens."
        ),
        sft_target_rows=4_000,
        dpo_target_pairs=500,
        gate_metric="accuracy",
        gate_threshold=0.95,
        hard_gate=True,
        eval_set="domain_classification_v5",
        prompt_style="short-classification",
        difficulty_notes="Hard-pair examples (same vocabulary, different domain)",
    ),
    "doctype_classification": Capability(
        capability_id="doctype_classification",
        name="Document-type classification",
        description=(
            "Subtype within a domain — commercial vs proforma invoice, PO "
            "vs delivery note, SOW vs MSA, medical report vs prescription."
        ),
        sft_target_rows=4_000,
        dpo_target_pairs=500,
        gate_metric="accuracy",
        gate_threshold=0.90,
        hard_gate=False,
        eval_set="doctype_v5",
        prompt_style="short-classification",
    ),
    "entity_extraction": Capability(
        capability_id="entity_extraction",
        name="Entity extraction with provenance",
        description=(
            "Extract every entity the V2 schema asks for — with page, "
            "span character range, and confidence. Never invent a field."
        ),
        sft_target_rows=20_000,
        dpo_target_pairs=4_000,
        gate_metric="fidelity",
        gate_threshold=0.98,
        hard_gate=True,
        eval_set="extraction_v5",
        prompt_style="v2-schema-extraction",
        difficulty_notes=(
            "Hard: OCR noise, missing fields, conflicting values in same doc, "
            "tabular line items with multi-line cells"
        ),
    ),
    "intent_understanding": Capability(
        capability_id="intent_understanding",
        name="Intent understanding",
        description=(
            "Given a document, produce a 1-3 sentence narrative of the "
            "transaction it represents: who is doing what to whom, and why."
        ),
        sft_target_rows=8_000,
        dpo_target_pairs=1_000,
        gate_metric="llm_judge",
        gate_threshold=4.5,
        hard_gate=False,
        eval_set="intent_narrative_v5",
        prompt_style="narrative",
    ),
    "context_dependence": Capability(
        capability_id="context_dependence",
        name="Context-dependent answering",
        description=(
            "Same prompt, different supplied context → different answer, "
            "no cross-contamination between documents."
        ),
        sft_target_rows=6_000,
        dpo_target_pairs=2_000,
        gate_metric="consistency",
        gate_threshold=0.95,
        hard_gate=True,
        eval_set="contrastive_v5",
        prompt_style="contrastive-paired",
        difficulty_notes="Pairs where only 1-2 tokens of context differ",
    ),
    "cross_doc_reasoning": Capability(
        capability_id="cross_doc_reasoning",
        name="Cross-document reasoning",
        description=(
            "Multi-document conversations — compare invoice vs PO, "
            "reconcile figures across statements, spot contradictions."
        ),
        sft_target_rows=5_000,
        dpo_target_pairs=1_000,
        gate_metric="F1",
        gate_threshold=0.90,
        hard_gate=False,
        eval_set="cross_doc_v5",
        prompt_style="multi-document",
    ),
    "grounded_refusal": Capability(
        capability_id="grounded_refusal",
        name="Grounded refusal",
        description=(
            "When asked for a fact the context can't support, explicitly "
            "refuse and say what's missing. Never fabricate a number or date."
        ),
        sft_target_rows=10_000,
        dpo_target_pairs=3_000,
        gate_metric="refusal_rate",
        gate_threshold=1.00,
        hard_gate=True,
        eval_set="hard_negatives_v5",
        prompt_style="adversarial",
        difficulty_notes=(
            "Hardest: partial match (invoice mentions amount but not tax), "
            "tempting to interpolate; or similar-doc context tempting to "
            "copy"
        ),
    ),
    "schema_adherence": Capability(
        capability_id="schema_adherence",
        name="V2 schema adherence",
        description=(
            "Default output for any extraction task is valid V2 JSON, "
            "exact schema, no extra fields, no missing required fields."
        ),
        sft_target_rows=15_000,
        dpo_target_pairs=4_000,
        gate_metric="schema_match",
        gate_threshold=0.99,
        hard_gate=True,
        eval_set="schema_v5",
        prompt_style="v2-schema-extraction",
    ),
    "tool_calling": Capability(
        capability_id="tool_calling",
        name="Native tool-calling",
        description=(
            "Emit <tool_call>...</tool_call> with correct tool + args. "
            "Handle multi-turn traces with tool results and errors. Know "
            "when NOT to call a tool."
        ),
        sft_target_rows=10_000,
        dpo_target_pairs=2_000,
        gate_metric="tool_format_valid",
        gate_threshold=1.00,
        hard_gate=True,
        eval_set="tool_traces_v5",
        prompt_style="tool-conversation",
        difficulty_notes=(
            "Tools covered: retrieve_chunks, get_document_sections, "
            "extract_entities, compare_documents, verify_fact, "
            "classify_document, list_profile_documents, "
            "get_screening_report, get_kg_subgraph"
        ),
    ),
    "identity_in_weights": Capability(
        capability_id="identity_in_weights",
        name="Identity in weights",
        description=(
            "Blank system prompt. Model still answers as DocWain, "
            "describes its capabilities correctly, uses DocWain voice "
            "(concise, structured, citation-first)."
        ),
        sft_target_rows=5_000,
        dpo_target_pairs=1_000,
        gate_metric="identity_match",
        gate_threshold=1.00,
        hard_gate=True,
        eval_set="identity_probes_v5",
        prompt_style="identity-probe",
    ),
    "citation_discipline": Capability(
        capability_id="citation_discipline",
        name="Citation discipline",
        description=(
            "Every factual claim carries [source: doc_id @ page X] or "
            "equivalent. Citations must point to real chunks that back the "
            "claim, not decorative."
        ),
        sft_target_rows=3_000,  # Reading warmup — bulk of citation pressure is in other buckets
        dpo_target_pairs=0,  # Covered by grounded_refusal DPO pairs
        gate_metric="citation_rate",
        gate_threshold=0.95,
        hard_gate=False,
        eval_set="citation_v5",
        prompt_style="raw-reading",
    ),
}


# Ordering for the SFT curriculum — easier capabilities first, hardest last
CURRICULUM_ORDER: List[str] = [
    "identity_in_weights",     # Bake identity before anything else
    "domain_recognition",
    "doctype_classification",
    "citation_discipline",
    "schema_adherence",
    "entity_extraction",
    "layout_understanding",
    "intent_understanding",
    "context_dependence",
    "cross_doc_reasoning",
    "tool_calling",
    "grounded_refusal",        # Hardest, taught last so the model has prior context
]


def totals() -> Dict[str, int]:
    """Aggregate row targets across the charter."""
    return {
        "sft_rows": sum(c.sft_target_rows for c in CHARTER.values()),
        "dpo_pairs": sum(c.dpo_target_pairs for c in CHARTER.values()),
        "capabilities": len(CHARTER),
        "hard_gates": sum(1 for c in CHARTER.values() if c.hard_gate),
    }


def get(cap_id: str) -> Capability:
    if cap_id not in CHARTER:
        raise KeyError(
            f"Unknown capability '{cap_id}'. Charter has: "
            f"{sorted(CHARTER.keys())}"
        )
    return CHARTER[cap_id]


def hard_gates() -> List[Capability]:
    """Capabilities whose gate failure blocks shipping."""
    return [c for c in CHARTER.values() if c.hard_gate]


__all__ = [
    "Capability",
    "CHARTER",
    "CURRICULUM_ORDER",
    "totals",
    "get",
    "hard_gates",
]
