"""KGMultiHopMaterializer — KG inference edges
(spec §6, artifact_type="kg_edge").

Phase 1 stub returns ``[]``. Phase 2 implements the real synthesis: runs
adapter-configured Cypher patterns (with allowlisted edge types per
ERRATA §15) and emits one item per inferred edge. Each item's
``metadata`` carries ``{from_node, to_node, relation_type}`` — the storage
adapter converts these into ``INFERRED_RELATION`` edges in Neo4j.
Items without these metadata keys must not reach storage.
"""
from __future__ import annotations

from src.intelligence.sme.artifact_models import ArtifactItem
from src.intelligence.sme.builders._base import ArtifactBuilder


class KGMultiHopMaterializer(ArtifactBuilder):
    """Builder for multi-hop inferred KG edges."""

    artifact_type = "kg_edge"

    def _synthesize(
        self, *, subscription_id: str, profile_id: str, adapter, version: int
    ) -> list[ArtifactItem]:
        # Phase 1: empty list. Phase 2 runs adapter.kg_inference_rules through
        # a bounded Cypher executor and materializes inferred edges.
        return []
