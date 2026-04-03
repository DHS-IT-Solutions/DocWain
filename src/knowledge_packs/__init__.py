"""DocWain Knowledge Pack System — pluggable, region-aware authoritative content.

Provides infrastructure for scraping, parsing, and indexing domain-specific
knowledge (e.g., NICE clinical guidelines) into Qdrant collections for
retrieval-augmented generation.
"""

from src.knowledge_packs.registry import PackRegistry
from src.knowledge_packs.base import KnowledgePackScraper, KnowledgePackParser

__all__ = [
    "PackRegistry",
    "KnowledgePackScraper",
    "KnowledgePackParser",
]
