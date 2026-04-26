"""Generic researcher v2 prompts for all 9 insight types.

Per spec Section 7 — when a domain adapter has no per-type override,
this module supplies the prompt. KB-aware via kb_context.
"""
from __future__ import annotations

INSIGHT_TYPES = (
    "anomaly", "gap", "comparison", "scenario", "trend",
    "recommendation", "conflict", "projection", "next_action",
)

SYSTEM_PROMPT = (
    "You are DocWain's Researcher Agent v2. Produce structured insights "
    "from documents. Output ONLY valid JSON. CRITICAL — content rules:\n"
    "(1) The 'body' field must contain only statements derivable from "
    "the document text. You MUST NOT introduce facts that come from "
    "external knowledge into the body.\n"
    "(2) External knowledge MAY be cited via 'external_kb_refs' as "
    "metadata, but never mixed into 'body'.\n"
    "(3) Every insight requires at least one entry in 'evidence_doc_spans'.\n"
    "(4) Quote the supporting text verbatim in each span.\n"
    "(5) When the prompt's domain-focus uses placeholder variables "
    "like $X, $Y, $N, X months, etc., DO NOT echo them. Replace each "
    "with a concrete value computed or extracted from the document. "
    "If a value cannot be computed from document content, omit that "
    "scenario entirely rather than emit a placeholder. Insight bodies "
    "with templated placeholders ($X, $Y, $W, N months, X by Y date) "
    "are unacceptable.\n"
)


_TYPE_GUIDANCE = {
    "anomaly": "Identify anomalies — values, dates, terms that look unusual, inconsistent, or risky for this kind of document.",
    "gap": "Identify gaps — what the document does not cover that a reader would expect for this kind of content.",
    "comparison": "Compare aspects across the documents — only fields/values present in 2+ documents.",
    "scenario": "Reason through plausible scenarios the document content suggests — 'if X then Y', grounded in stated terms.",
    "trend": "Identify trends — directional changes over time using dated content in the documents.",
    "recommendation": "Recommend concrete next-best-actions a user should consider, justified by document content.",
    "conflict": "Detect contradictions or conflicts between documents.",
    "projection": "Project forward — numeric or categorical estimates extrapolated from document content.",
    "next_action": "Surface time-sensitive or attention-required next steps the documents imply.",
}


_MAX_DOC_CHARS = 16000


def build_typed_insight_prompt(
    *,
    insight_type: str,
    domain_name: str,
    document_text: str,
    kb_context: str = "",
    domain_focus: str = "",
) -> str:
    if insight_type not in INSIGHT_TYPES:
        raise ValueError(f"unknown insight_type: {insight_type}")
    guidance = _TYPE_GUIDANCE[insight_type]
    truncated = document_text[:_MAX_DOC_CHARS]
    kb_block = (
        f"\nDomain knowledge context (for interpretation only — do NOT inject into body):\n{kb_context}\n"
        if kb_context
        else ""
    )
    domain_block = (
        f"\nDomain-specific focus areas:\n{domain_focus}\n"
        if domain_focus
        else ""
    )
    return (
        f"Domain: {domain_name}\n"
        f"Insight type to produce: {insight_type}\n"
        f"Guidance: {guidance}\n"
        f"{domain_block}"
        f"{kb_block}\n"
        f"Document text:\n\n{truncated}\n\n"
        "Return JSON with this shape (no prose, no markdown fences):\n"
        '{"insights": [{"headline": "≤25 words", "body": "≤600 chars",'
        ' "evidence_doc_spans": [{"document_id": "...", "page": 0,'
        ' "char_start": 0, "char_end": 0, "quote": "verbatim text"}],'
        ' "external_kb_refs": [{"kb_id": "...", "ref": "...", "label": "..."}],'
        ' "confidence": 0.0, "severity": "info|notice|warn|critical"}]}\n'
    )
