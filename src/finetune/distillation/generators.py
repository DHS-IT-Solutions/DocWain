"""Knowledge distillation generators that produce SFT and DPO training
examples from real document chunks.

All SFT examples are formatted via ``format_sft_example`` from the V2 base
module and enriched with metadata (area, difficulty, category, source).
DPO pairs target the #1 failure mode: the model returning "0 items found"
when the document clearly contains data.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.finetune.v2.data_generator.base import format_dpo_example, format_sft_example

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MIN_DOC_LENGTH = 50  # skip trivially short docs


def _extract_entities(text: str) -> List[str]:
    """Pull capitalised multi-word names from text."""
    return list(dict.fromkeys(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)))


def _extract_amounts(text: str) -> List[str]:
    """Pull dollar/currency amounts."""
    return list(dict.fromkeys(re.findall(r"\$[\d,]+(?:\.\d{2})?", text)))


def _extract_dates(text: str) -> List[str]:
    """Pull date-like strings."""
    return list(dict.fromkeys(
        re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)
        + re.findall(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b", text)
    ))


def _extract_emails(text: str) -> List[str]:
    return list(dict.fromkeys(re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)))


def _snippet(text: str, max_len: int = 200) -> str:
    """Return a trimmed snippet of text."""
    return text[:max_len].strip() + ("..." if len(text) > max_len else "")


def _is_too_short(text: str) -> bool:
    return not text or len(text.strip()) < _MIN_DOC_LENGTH


def _make_sft(
    query: str,
    reasoning: str,
    answer: str,
    *,
    area: str,
    difficulty: str,
    category: str,
) -> Dict[str, Any]:
    """Build a complete SFT example with metadata."""
    example = format_sft_example(query, reasoning, answer)
    example["area"] = area
    example["difficulty"] = difficulty
    example["category"] = category
    example["source"] = "claude_distillation"
    return example


# ---------------------------------------------------------------------------
# 1. Extraction examples
# ---------------------------------------------------------------------------


def generate_extraction_examples(
    doc_text: str,
    doc_type: str,
    source_name: str,
) -> List[Dict[str, Any]]:
    """Generate SFT examples teaching document extraction."""
    if _is_too_short(doc_text):
        return []

    examples: List[Dict[str, Any]] = []
    entities = _extract_entities(doc_text)
    amounts = _extract_amounts(doc_text)
    dates = _extract_dates(doc_text)
    emails = _extract_emails(doc_text)
    snippet = _snippet(doc_text, 300)

    # --- Easy: list key entities ------------------------------------------------
    if entities:
        entity_list = ", ".join(entities[:10])
        examples.append(_make_sft(
            query=f"Given the following {doc_type} document, list the key entities mentioned.\n\n{snippet}",
            reasoning=(
                f"I need to scan the document for named entities. "
                f"I can see references to: {entity_list}. "
                f"I should list them clearly."
            ),
            answer=f"The key entities identified in this {doc_type} are:\n\n"
                   + "\n".join(f"- **{e}**" for e in entities[:10]),
            area="document_extraction",
            difficulty="easy",
            category="extraction",
        ))

    # --- Medium: extract structured fields ------------------------------------
    fields_parts = []
    if entities:
        fields_parts.append(f"**Entities:** {', '.join(entities[:5])}")
    if amounts:
        fields_parts.append(f"**Amounts:** {', '.join(amounts[:5])}")
    if dates:
        fields_parts.append(f"**Dates:** {', '.join(dates[:5])}")
    if emails:
        fields_parts.append(f"**Emails:** {', '.join(emails[:5])}")

    if fields_parts:
        examples.append(_make_sft(
            query=f"Extract all structured fields (names, dates, amounts, emails) from this {doc_type}.\n\n{snippet}",
            reasoning=(
                f"I will parse the document systematically for structured data. "
                f"Detected {len(entities)} entities, {len(amounts)} amounts, "
                f"{len(dates)} dates, {len(emails)} emails."
            ),
            answer="## Extracted Fields\n\n" + "\n".join(fields_parts),
            area="document_extraction",
            difficulty="medium",
            category="extraction",
        ))

    # --- Hard: type-specific extraction ----------------------------------------
    type_prompts = {
        "resume": "Extract the candidate's name, contact information, work experience, and skills.",
        "contract": "Extract the parties, effective date, term, obligations, and governing law.",
        "invoice": "Extract the vendor, invoice number, line items, totals, and payment terms.",
        "generic": "Identify the document type and extract all key information fields.",
    }
    prompt_text = type_prompts.get(doc_type, type_prompts["generic"])

    answer_lines = [f"## {doc_type.title()} Extraction\n"]
    if entities:
        answer_lines.append(f"**Key Parties/Names:** {', '.join(entities[:5])}")
    if amounts:
        answer_lines.append(f"**Financial Figures:** {', '.join(amounts[:5])}")
    if dates:
        answer_lines.append(f"**Dates:** {', '.join(dates[:5])}")
    if emails:
        answer_lines.append(f"**Contact:** {', '.join(emails[:3])}")
    answer_lines.append(f"\n*Source: {source_name}*")

    examples.append(_make_sft(
        query=f"{prompt_text}\n\n{snippet}",
        reasoning=(
            f"This is a {doc_type} document. I need to perform deep extraction. "
            f"Scanning for type-specific fields. Found {len(entities)} names, "
            f"{len(amounts)} monetary values, {len(dates)} dates."
        ),
        answer="\n".join(answer_lines),
        area="document_extraction",
        difficulty="hard",
        category="extraction",
    ))

    return examples


# ---------------------------------------------------------------------------
# 2. Analytical examples
# ---------------------------------------------------------------------------


def generate_analytical_examples(
    doc_text: str,
    doc_type: str,
    source_name: str,
) -> List[Dict[str, Any]]:
    """Generate SFT examples teaching analytical reasoning."""
    if _is_too_short(doc_text):
        return []

    examples: List[Dict[str, Any]] = []
    snippet = _snippet(doc_text, 400)
    entities = _extract_entities(doc_text)
    amounts = _extract_amounts(doc_text)

    # Implications
    examples.append(_make_sft(
        query=f"What are the key implications of the following {doc_type}?\n\n{snippet}",
        reasoning=(
            f"I need to analyse the {doc_type} for broader implications. "
            f"The document mentions {len(entities)} entities and {len(amounts)} financial figures. "
            f"I should consider business, legal, and operational impacts."
        ),
        answer=(
            f"## Key Implications\n\n"
            f"1. **Scope:** This {doc_type} involves {', '.join(entities[:3]) if entities else 'the mentioned parties'}.\n"
            f"2. **Financial Impact:** {'The document references ' + ', '.join(amounts[:3]) + '.' if amounts else 'No explicit financial figures detected.'}\n"
            f"3. **Operational Considerations:** Based on the content, stakeholders should review "
            f"obligations and timelines carefully.\n\n"
            f"*Confidence: Medium* - analysis based on available text."
        ),
        area="analytical_reasoning",
        difficulty="medium",
        category="analytical",
    ))

    # Patterns & anomalies
    examples.append(_make_sft(
        query=f"Identify any patterns or anomalies in this {doc_type}.\n\n{snippet}",
        reasoning=(
            f"Looking for patterns: repeated entities, unusual amounts, date clustering. "
            f"Entities found: {', '.join(entities[:5]) if entities else 'none'}. "
            f"Amounts: {', '.join(amounts[:5]) if amounts else 'none'}."
        ),
        answer=(
            f"## Pattern & Anomaly Analysis\n\n"
            f"**Patterns Identified:**\n"
            f"- Document structure is consistent with standard {doc_type} format.\n"
            f"{'- Recurring references to: ' + ', '.join(entities[:3]) + chr(10) if entities else ''}"
            f"\n**Potential Anomalies:**\n"
            f"- Further review recommended to validate completeness.\n\n"
            f"*Source: {source_name}*"
        ),
        area="analytical_reasoning",
        difficulty="hard",
        category="analytical",
    ))

    return examples


# ---------------------------------------------------------------------------
# 3. Cross-document examples
# ---------------------------------------------------------------------------


def generate_crossdoc_examples(
    doc_texts: List[str],
    doc_types: List[str],
    source_names: List[str],
) -> List[Dict[str, Any]]:
    """Generate cross-document comparison SFT examples."""
    if len(doc_texts) < 2:
        return []
    if any(_is_too_short(t) for t in doc_texts):
        return []

    examples: List[Dict[str, Any]] = []
    snippets = [_snippet(t, 200) for t in doc_texts]
    all_entities = [_extract_entities(t) for t in doc_texts]

    context = "\n\n---\n\n".join(
        f"**Document {i + 1} ({doc_types[i]}):**\n{s}"
        for i, s in enumerate(snippets)
    )

    # Shared entities
    if len(all_entities) >= 2 and all_entities[0] and all_entities[1]:
        shared = set(all_entities[0]) & set(all_entities[1])
        shared_text = ", ".join(shared) if shared else "no overlapping entities"
    else:
        shared = set()
        shared_text = "no overlapping entities detected"

    examples.append(_make_sft(
        query=f"Compare the following documents and identify shared entities and differences.\n\n{context}",
        reasoning=(
            f"Comparing {len(doc_texts)} documents of types {', '.join(doc_types)}. "
            f"Document 1 entities: {', '.join(all_entities[0][:5]) if all_entities[0] else 'none'}. "
            f"Document 2 entities: {', '.join(all_entities[1][:5]) if all_entities[1] else 'none'}. "
            f"Shared: {shared_text}."
        ),
        answer=(
            f"## Cross-Document Comparison\n\n"
            f"**Documents Analysed:** {', '.join(source_names)}\n\n"
            f"**Shared Entities:** {shared_text}\n\n"
            f"**Key Differences:**\n"
            f"- Document types differ: {', '.join(doc_types)}.\n"
            f"- Each document has unique entities requiring separate review.\n\n"
            f"*Confidence: Medium*"
        ),
        area="cross_document",
        difficulty="hard",
        category="crossdoc",
    ))

    return examples


# ---------------------------------------------------------------------------
# 4. Content generation examples
# ---------------------------------------------------------------------------


def generate_content_examples(
    doc_text: str,
    doc_type: str,
    source_name: str,
) -> List[Dict[str, Any]]:
    """Generate SFT examples teaching content generation (summaries, tables)."""
    if _is_too_short(doc_text):
        return []

    examples: List[Dict[str, Any]] = []
    snippet = _snippet(doc_text, 400)
    entities = _extract_entities(doc_text)
    amounts = _extract_amounts(doc_text)

    # Summary
    examples.append(_make_sft(
        query=f"Provide a concise executive summary of this {doc_type}.\n\n{snippet}",
        reasoning=(
            f"I need to distil the {doc_type} into key points. "
            f"Main entities: {', '.join(entities[:5]) if entities else 'not explicitly named'}. "
            f"Financial data: {', '.join(amounts[:3]) if amounts else 'none detected'}."
        ),
        answer=(
            f"## Executive Summary\n\n"
            f"This {doc_type} involves {', '.join(entities[:3]) if entities else 'the relevant parties'}. "
            f"{'Key financial figures include ' + ', '.join(amounts[:3]) + '. ' if amounts else ''}"
            f"The document outlines obligations and terms requiring stakeholder attention.\n\n"
            f"*Source: {source_name}*"
        ),
        area="content_generation",
        difficulty="medium",
        category="content",
    ))

    # Table
    if entities or amounts:
        table_rows = []
        for i, e in enumerate(entities[:5]):
            amt = amounts[i] if i < len(amounts) else "N/A"
            table_rows.append(f"| {e} | {amt} |")
        examples.append(_make_sft(
            query=f"Create a structured table of key data from this {doc_type}.\n\n{snippet}",
            reasoning=(
                f"Building a markdown table from extracted data. "
                f"Have {len(entities)} entities and {len(amounts)} amounts to tabulate."
            ),
            answer=(
                f"## Key Data Table\n\n"
                f"| Entity | Amount |\n|--------|--------|\n"
                + "\n".join(table_rows)
            ),
            area="content_generation",
            difficulty="hard",
            category="content",
        ))

    return examples


# ---------------------------------------------------------------------------
# 5. Boundary examples ("I don't know")
# ---------------------------------------------------------------------------


def generate_boundary_examples(
    doc_text: str,
    doc_type: str,
    source_name: str,
) -> List[Dict[str, Any]]:
    """Generate SFT examples teaching the model when to say 'I don't know'."""
    if _is_too_short(doc_text):
        return []

    examples: List[Dict[str, Any]] = []
    snippet = _snippet(doc_text, 300)

    # Question about info NOT in the document
    examples.append(_make_sft(
        query=f"What is the CEO's favourite colour based on this {doc_type}?\n\n{snippet}",
        reasoning=(
            f"The user is asking about a CEO's favourite colour. "
            f"This is a {doc_type} document. I need to check whether this "
            f"information is present. Scanning the text... there is no mention "
            f"of colour preferences. I must not hallucinate an answer."
        ),
        answer=(
            f"I cannot determine the CEO's favourite colour from this {doc_type}. "
            f"The document does not contain information about personal colour preferences. "
            f"I can only answer questions based on what is present in the provided document.\n\n"
            f"*Confidence: High* - this information is definitively absent."
        ),
        area="boundary_detection",
        difficulty="easy",
        category="boundary",
    ))

    # Speculation question
    examples.append(_make_sft(
        query=f"Predict what will happen to the company in 5 years based on this {doc_type}.\n\n{snippet}",
        reasoning=(
            f"The user wants a 5-year prediction. While I can identify trends in the "
            f"document, making specific future predictions goes beyond what the document "
            f"supports. I should clearly state limitations."
        ),
        answer=(
            f"I cannot make reliable 5-year predictions based solely on this {doc_type}. "
            f"While the document provides current information, predicting future outcomes "
            f"requires additional data, market analysis, and domain expertise beyond what "
            f"is captured here.\n\n"
            f"**What I can offer:** A summary of current trends and data points visible "
            f"in the document that may be relevant to future planning.\n\n"
            f"*Confidence: High that prediction is not supported by this document alone.*"
        ),
        area="boundary_detection",
        difficulty="medium",
        category="boundary",
    ))

    return examples


# ---------------------------------------------------------------------------
# 6. Reasoning chain examples
# ---------------------------------------------------------------------------


def generate_reasoning_examples(
    doc_text: str,
    doc_type: str,
    source_name: str,
) -> List[Dict[str, Any]]:
    """Generate step-by-step reasoning chain SFT examples."""
    if _is_too_short(doc_text):
        return []

    examples: List[Dict[str, Any]] = []
    snippet = _snippet(doc_text, 400)
    entities = _extract_entities(doc_text)
    amounts = _extract_amounts(doc_text)
    dates = _extract_dates(doc_text)

    examples.append(_make_sft(
        query=f"Walk me through a thorough analysis of this {doc_type}, step by step.\n\n{snippet}",
        reasoning=(
            f"Step 1: Identify document type - this is a {doc_type}.\n"
            f"Step 2: Extract key entities - found {len(entities)}: {', '.join(entities[:5]) if entities else 'none identified'}.\n"
            f"Step 3: Extract financial data - found {len(amounts)}: {', '.join(amounts[:3]) if amounts else 'none'}.\n"
            f"Step 4: Extract dates - found {len(dates)}: {', '.join(dates[:3]) if dates else 'none'}.\n"
            f"Step 5: Synthesise findings into a coherent analysis."
        ),
        answer=(
            f"## Step-by-Step Analysis of {doc_type.title()}\n\n"
            f"### Step 1: Document Classification\n"
            f"This document is classified as a **{doc_type}**.\n\n"
            f"### Step 2: Entity Identification\n"
            f"{'Key entities: ' + ', '.join(entities[:5]) if entities else 'No named entities clearly identified.'}\n\n"
            f"### Step 3: Financial Data\n"
            f"{'Amounts found: ' + ', '.join(amounts[:5]) if amounts else 'No monetary figures detected.'}\n\n"
            f"### Step 4: Temporal Data\n"
            f"{'Dates found: ' + ', '.join(dates[:5]) if dates else 'No explicit dates detected.'}\n\n"
            f"### Step 5: Synthesis\n"
            f"The document provides {'substantial' if len(entities) > 2 else 'limited'} structured data. "
            f"{'Further review of ' + ', '.join(entities[:2]) + ' is recommended.' if entities else 'Additional context may be needed for deeper analysis.'}\n\n"
            f"*Source: {source_name}*"
        ),
        area="reasoning_chain",
        difficulty="hard",
        category="reasoning",
    ))

    return examples


# ---------------------------------------------------------------------------
# 7. DPO preference pairs
# ---------------------------------------------------------------------------


def generate_dpo_pairs(
    doc_text: str,
    doc_type: str,
) -> List[Dict[str, Any]]:
    """Generate DPO preference pairs targeting the '0 items found' failure."""
    if _is_too_short(doc_text):
        return []

    pairs: List[Dict[str, Any]] = []
    snippet = _snippet(doc_text, 300)
    entities = _extract_entities(doc_text)
    amounts = _extract_amounts(doc_text)
    dates = _extract_dates(doc_text)

    # --- Pair 1: extraction ---------------------------------------------------
    query1 = f"Extract all key information from this {doc_type}.\n\n{snippet}"

    chosen_parts = []
    if entities:
        chosen_parts.append(f"**Entities:** {', '.join(entities[:5])}")
    if amounts:
        chosen_parts.append(f"**Amounts:** {', '.join(amounts[:5])}")
    if dates:
        chosen_parts.append(f"**Dates:** {', '.join(dates[:5])}")
    if not chosen_parts:
        chosen_parts.append(f"**Content:** The document contains text requiring manual review for structured fields.")

    pair1 = format_dpo_example(
        query=query1,
        chosen_reasoning=(
            f"I will carefully scan the {doc_type} for all structured information. "
            f"Found {len(entities)} entities, {len(amounts)} amounts, {len(dates)} dates."
        ),
        chosen_answer="## Extracted Information\n\n" + "\n".join(chosen_parts),
        rejected_reasoning=(
            f"Looking at this document... I don't see clear structured data."
        ),
        rejected_answer="0 items found. The document does not appear to contain extractable information.",
    )
    pair1["source"] = "claude_distillation"
    pair1["category"] = "extraction_vs_empty"
    pairs.append(pair1)

    # --- Pair 2: analysis depth -----------------------------------------------
    query2 = f"Analyse this {doc_type} and summarise the key findings.\n\n{snippet}"

    pair2 = format_dpo_example(
        query=query2,
        chosen_reasoning=(
            f"The document is a {doc_type}. I need to provide a thorough analysis. "
            f"Key entities: {', '.join(entities[:3]) if entities else 'to be identified from context'}. "
            f"I should structure my response with clear sections."
        ),
        chosen_answer=(
            f"## Analysis Summary\n\n"
            f"This {doc_type} contains the following key findings:\n\n"
            f"{'- **Named parties:** ' + ', '.join(entities[:3]) + chr(10) if entities else ''}"
            f"{'- **Financial data:** ' + ', '.join(amounts[:3]) + chr(10) if amounts else ''}"
            f"- The document structure indicates a formal {doc_type} requiring detailed review."
        ),
        rejected_reasoning="I'll give a quick summary.",
        rejected_answer=f"This is a {doc_type}. No significant findings.",
    )
    pair2["source"] = "claude_distillation"
    pair2["category"] = "thorough_vs_shallow"
    pairs.append(pair2)

    return pairs


# ---------------------------------------------------------------------------
# 8. Unified generator
# ---------------------------------------------------------------------------


def generate_all_categories(
    doc_text: str,
    doc_type: str,
    source_name: str,
) -> List[Dict[str, Any]]:
    """Call all single-doc generators and return the combined list."""
    results: List[Dict[str, Any]] = []
    results.extend(generate_extraction_examples(doc_text, doc_type, source_name))
    results.extend(generate_analytical_examples(doc_text, doc_type, source_name))
    results.extend(generate_content_examples(doc_text, doc_type, source_name))
    results.extend(generate_boundary_examples(doc_text, doc_type, source_name))
    results.extend(generate_reasoning_examples(doc_text, doc_type, source_name))
    return results
