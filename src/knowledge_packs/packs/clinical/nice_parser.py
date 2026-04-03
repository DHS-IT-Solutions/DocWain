"""NICE content parser — transforms scraped guideline data into ParsedDocuments.

Extracts structured recommendation numbers (e.g., "1.4.2.3") and preserves
section hierarchy for recommendations, evidence, and quality statements.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from src.knowledge_packs.base import KnowledgePackParser, ParsedDocument
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Recommendation number pattern: e.g., "1.4.2.3" or "Recommendation 1.2.3"
_REC_NUMBER_PATTERN = re.compile(
    r"(?:Recommendation\s+)?(\d+(?:\.\d+){1,4})",
    re.IGNORECASE,
)

# Section type classifiers
_RECOMMENDATION_KEYWORDS = {"recommendation", "recommends", "should", "must", "consider"}
_EVIDENCE_KEYWORDS = {"evidence", "review", "study", "trial", "meta-analysis", "systematic"}
_QUALITY_KEYWORDS = {"quality statement", "quality standard", "quality measure"}


class NICEParser(KnowledgePackParser):
    """Parser for NICE guideline content.

    Transforms raw scraped guideline dicts into ``ParsedDocument`` instances
    with sections categorised as recommendations, evidence, or quality
    statements.
    """

    def parse(self, raw_content: Dict[str, Any]) -> List[ParsedDocument]:
        """Parse a raw NICE guideline dict into one or more ParsedDocuments.

        Parameters
        ----------
        raw_content:
            Dict with keys: id, title, url, type, sections, last_updated.
            ``sections`` is a list of dicts with ``heading`` and ``content``.

        Returns
        -------
        list[ParsedDocument]
            Typically one document per guideline, with enriched section metadata.
        """
        guideline_id = raw_content.get("id", "")
        title = raw_content.get("title", "Untitled")
        url = raw_content.get("url", "")
        gtype = raw_content.get("type", "")
        last_updated = raw_content.get("last_updated", "")
        raw_sections = raw_content.get("sections", [])

        if not raw_sections:
            logger.debug("No sections in guideline", extra={"id": guideline_id})
            # Return a minimal document with the title as content
            return [ParsedDocument(
                title=f"[{guideline_id}] {title}",
                source_url=url,
                sections=[{"heading": title, "content": f"Guideline {guideline_id}: {title}"}],
                metadata={
                    "guideline_id": guideline_id,
                    "guideline_type": gtype,
                },
                last_updated=last_updated,
            )]

        # Process sections: classify, extract recommendation numbers
        processed_sections: List[Dict[str, str]] = []
        recommendations: List[Dict[str, str]] = []
        evidence_sections: List[Dict[str, str]] = []
        quality_sections: List[Dict[str, str]] = []

        for raw_section in raw_sections:
            heading = raw_section.get("heading", "")
            content = raw_section.get("content", "")

            # Extract recommendation numbers
            rec_numbers = _REC_NUMBER_PATTERN.findall(content)
            enriched_content = content

            # Classify section
            section_type = self._classify_section(heading, content)
            section_entry = {
                "heading": heading,
                "content": enriched_content,
                "section_type": section_type,
            }

            if rec_numbers:
                section_entry["recommendation_numbers"] = ", ".join(rec_numbers)

            processed_sections.append(section_entry)

            if section_type == "recommendation":
                recommendations.append(section_entry)
            elif section_type == "evidence":
                evidence_sections.append(section_entry)
            elif section_type == "quality_statement":
                quality_sections.append(section_entry)

        # Build a single ParsedDocument with all sections
        doc = ParsedDocument(
            title=f"[{guideline_id}] {title}",
            source_url=url,
            sections=processed_sections,
            metadata={
                "guideline_id": guideline_id,
                "guideline_type": gtype,
                "recommendation_count": len(recommendations),
                "evidence_section_count": len(evidence_sections),
                "quality_statement_count": len(quality_sections),
                "total_sections": len(processed_sections),
                "all_recommendation_numbers": sorted(set(
                    num
                    for s in processed_sections
                    for num in _REC_NUMBER_PATTERN.findall(s.get("content", ""))
                )),
            },
            last_updated=last_updated,
        )

        return [doc]

    @staticmethod
    def _classify_section(heading: str, content: str) -> str:
        """Classify a section as recommendation, evidence, quality_statement, or general."""
        combined = f"{heading} {content[:200]}".lower()

        # Quality statements take priority (they can contain recommendation keywords)
        if any(kw in combined for kw in _QUALITY_KEYWORDS):
            return "quality_statement"

        if any(kw in combined for kw in _RECOMMENDATION_KEYWORDS):
            # Check for recommendation numbers as additional signal
            if _REC_NUMBER_PATTERN.search(content):
                return "recommendation"
            # Heading-level keywords are strong signals
            if any(kw in heading.lower() for kw in _RECOMMENDATION_KEYWORDS):
                return "recommendation"

        if any(kw in combined for kw in _EVIDENCE_KEYWORDS):
            return "evidence"

        # Default: check if it has recommendation numbers
        if _REC_NUMBER_PATTERN.search(content):
            return "recommendation"

        return "general"
