"""Document intelligence — LLM-powered analysis after embedding.

Uses Ollama Cloud to analyze document content and generate:
- Document type classification
- Brief executive summary
- 5 intelligent questions specific to the document content
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

_ANALYSIS_PROMPT = """Analyze this document and respond in EXACT JSON format. No other text.

Document filename: {filename}
Document content (first 6000 chars):
---
{content}
---

Respond with this EXACT JSON structure:
{{
  "doc_type": "<one of: resume, invoice, contract, legal, medical, policy, bank_statement, report, spreadsheet, presentation, email, general>",
  "summary": "<2-3 sentence executive summary of what this document contains and its purpose>",
  "key_entities": ["<entity1>", "<entity2>", "<entity3>"],
  "questions": [
    "<specific question about a key fact or figure in this document>",
    "<analytical question requiring synthesis of document information>",
    "<comparison or trend question if applicable, or extraction question>",
    "<question about implications or next steps from this document>",
    "<question that would showcase deep understanding of the document>"
  ]
}}

Rules for questions:
- Each question must be SPECIFIC to THIS document's actual content
- Reference real names, dates, amounts, or topics from the document
- Questions should demonstrate intelligence — not generic like "what is this about?"
- Mix factual extraction with analytical reasoning questions"""


@dataclass
class DocumentIntelligence:
    """LLM-generated document analysis."""
    doc_type: str = "general"
    summary: str = ""
    key_entities: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)


async def analyze_document(
    text: str,
    filename: str,
    fallback_doc_type: str = "general",
) -> DocumentIntelligence:
    """Analyze document content using the LLM to generate intelligence.

    Returns DocumentIntelligence with doc_type, summary, key_entities, and 5 questions.
    Falls back gracefully if LLM is unavailable.
    """
    if not text or len(text.strip()) < 50:
        return DocumentIntelligence(doc_type=fallback_doc_type)

    try:
        from src.llm.gateway import create_llm_gateway

        gateway = create_llm_gateway()
        prompt = _ANALYSIS_PROMPT.format(
            filename=filename,
            content=text[:6000],
        )

        logger.info("Analyzing document %s (%d chars) via LLM...", filename, len(text))

        raw = await asyncio.to_thread(
            gateway.generate,
            prompt,
            system="You are a document analyst. Respond ONLY with valid JSON.",
            temperature=0.1,
            max_tokens=1024,
        )

        logger.info("LLM analysis response for %s: %d chars", filename, len(raw or ""))
        logger.debug("Raw LLM response: %s", (raw or "")[:500])

        # Parse JSON from response (handle markdown code blocks)
        parsed = _parse_json(raw)
        if not parsed:
            logger.warning("Failed to parse LLM analysis JSON for %s. Raw: %s", filename, (raw or "")[:300])
            return DocumentIntelligence(doc_type=fallback_doc_type)

        result = DocumentIntelligence(
            doc_type=parsed.get("doc_type", fallback_doc_type) or fallback_doc_type,
            summary=parsed.get("summary", ""),
            key_entities=parsed.get("key_entities", [])[:5],
            questions=parsed.get("questions", [])[:5],
        )
        logger.info("Document intelligence for %s: type=%s, entities=%d, questions=%d",
                     filename, result.doc_type, len(result.key_entities), len(result.questions))
        return result

    except Exception as exc:
        logger.error("Document analysis failed for %s: %s", filename, exc, exc_info=True)
        return DocumentIntelligence(doc_type=fallback_doc_type)


def _parse_json(text: str) -> Optional[dict]:
    """Extract and parse JSON from LLM response, handling markdown code blocks."""
    if not text:
        return None

    # Try direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` blocks
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None
