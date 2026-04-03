"""Abstract interfaces and data structures for knowledge packs.

Defines the contract that every scraper and parser must implement, plus
shared dataclasses for parsed documents and pack configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ParsedDocument:
    """A single parsed document from a knowledge pack source."""

    title: str
    source_url: str
    sections: List[Dict[str, str]]  # [{"heading": "...", "content": "..."}, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_updated: str = ""

    def full_text(self) -> str:
        """Concatenate all sections into a single text string."""
        parts: List[str] = []
        for s in self.sections:
            heading = s.get("heading", "")
            content = s.get("content", "")
            if heading:
                parts.append(f"## {heading}\n{content}")
            else:
                parts.append(content)
        return "\n\n".join(parts)


@dataclass
class KnowledgePackConfig:
    """Configuration for a knowledge pack."""

    name: str
    domain: str
    region: str  # e.g. "UK", "US", "EU", "global"
    qdrant_collection: str
    citation_format: str  # e.g. "[NICE {id}, Section {section}]"
    update_schedule_cron: str  # e.g. "0 3 1 * *"
    source_url: str
    content_types: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class KnowledgePackScraper(ABC):
    """Abstract scraper for a knowledge pack source.

    Subclasses fetch raw content from an authoritative source (web API, file
    download, etc.) and return structured dicts for the parser.
    """

    @abstractmethod
    def scrape(self) -> List[Dict[str, Any]]:
        """Perform a full scrape of the source.

        Returns a list of raw content dicts, each representing one
        document/guideline/regulation to be parsed.
        """
        ...

    @abstractmethod
    def check_updates(self, since: datetime) -> List[Dict[str, Any]]:
        """Check for new or updated content since the given datetime.

        Returns only the items that have been added or modified.
        """
        ...


class KnowledgePackParser(ABC):
    """Abstract parser for raw scraped content.

    Transforms a single raw content dict into one or more ``ParsedDocument``
    instances suitable for embedding and indexing.
    """

    @abstractmethod
    def parse(self, raw_content: Dict[str, Any]) -> List[ParsedDocument]:
        """Parse raw scraped content into structured documents.

        May return multiple ``ParsedDocument`` instances if the raw content
        contains multiple logical documents (e.g., sub-guidelines).
        """
        ...
