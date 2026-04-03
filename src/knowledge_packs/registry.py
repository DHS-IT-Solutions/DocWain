"""Pack registry with region-aware routing.

Singleton registry that maps (domain, region) pairs to their configuration,
scraper class, and parser class. Supports exact-region matching with a
``global`` fallback.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple, Type

from src.knowledge_packs.base import (
    KnowledgePackConfig,
    KnowledgePackParser,
    KnowledgePackScraper,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PackRegistry:
    """Singleton registry for knowledge packs."""

    _instance: Optional["PackRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "PackRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._packs: Dict[
                    Tuple[str, str],
                    Tuple[KnowledgePackConfig, Type[KnowledgePackScraper], Type[KnowledgePackParser]],
                ] = {}
                cls._instance._initialised = False
            return cls._instance

    # -- Registration --------------------------------------------------------

    def register(
        self,
        domain: str,
        region: str,
        config: KnowledgePackConfig,
        scraper_cls: Type[KnowledgePackScraper],
        parser_cls: Type[KnowledgePackParser],
    ) -> None:
        """Register a knowledge pack for the given domain and region."""
        key = (domain.lower(), region.upper())
        self._packs[key] = (config, scraper_cls, parser_cls)
        logger.info(
            "Registered knowledge pack",
            extra={"domain": domain, "region": region, "name": config.name},
        )

    # -- Lookup --------------------------------------------------------------

    def get_pack(
        self,
        domain: str,
        region: str,
    ) -> Optional[Tuple[KnowledgePackConfig, Type[KnowledgePackScraper], Type[KnowledgePackParser]]]:
        """Return (config, scraper_cls, parser_cls) for exact region or global fallback.

        Returns ``None`` if no pack is registered for the domain.
        """
        key_exact = (domain.lower(), region.upper())
        if key_exact in self._packs:
            return self._packs[key_exact]

        # Fall back to global
        key_global = (domain.lower(), "GLOBAL")
        if key_global in self._packs:
            logger.info(
                "Using global fallback for knowledge pack",
                extra={"domain": domain, "requested_region": region},
            )
            return self._packs[key_global]

        return None

    def get_collection_name(self, domain: str, region: str) -> Optional[str]:
        """Return the Qdrant collection name for the given pack.

        Returns ``None`` if the pack is not registered.
        """
        pack = self.get_pack(domain, region)
        if pack is None:
            return None
        config, _, _ = pack
        return config.qdrant_collection

    def list_packs(self) -> List[Dict[str, Any]]:
        """Return metadata for all registered packs."""
        result: List[Dict[str, Any]] = []
        for (domain, region), (config, scraper_cls, parser_cls) in self._packs.items():
            result.append({
                "name": config.name,
                "domain": domain,
                "region": region,
                "qdrant_collection": config.qdrant_collection,
                "source_url": config.source_url,
                "update_schedule_cron": config.update_schedule_cron,
                "scraper": scraper_cls.__name__,
                "parser": parser_cls.__name__,
            })
        return result

    # -- Reset (testing) -----------------------------------------------------

    @classmethod
    def _reset(cls) -> None:
        """Reset singleton for testing."""
        with cls._lock:
            cls._instance = None


# ---------------------------------------------------------------------------
# Auto-register available packs on import
# ---------------------------------------------------------------------------


def _auto_register() -> None:
    """Discover and register built-in packs."""
    registry = PackRegistry()
    if registry._initialised:
        return

    try:
        from src.knowledge_packs.packs.clinical.nice_scraper import NICEScraper
        from src.knowledge_packs.packs.clinical.nice_parser import NICEParser

        config = KnowledgePackConfig(
            name="nice_clinical",
            domain="clinical",
            region="UK",
            qdrant_collection="knowledge_nice",
            citation_format="[NICE {id}, Section {section}]",
            update_schedule_cron="0 3 1 * *",
            source_url="https://www.nice.org.uk/guidance/published",
            content_types=["NG", "TA", "QS", "CG"],
        )
        registry.register("clinical", "UK", config, NICEScraper, NICEParser)
    except ImportError:
        logger.debug("NICE clinical pack not available — skipping auto-registration")

    registry._initialised = True


_auto_register()
