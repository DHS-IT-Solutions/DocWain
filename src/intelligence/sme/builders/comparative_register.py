"""ComparativeRegisterBuilder — cross-doc comparisons
(spec §6, artifact_type="comparison").

Phase 1 stub returns ``[]``. Phase 2 implements the real synthesis: one item
per detected delta/conflict/timeline/corroboration finding. Each item cites
at least two source documents via evidence refs and carries the compared
items in ``metadata["compared_items"]`` (list of ``{doc_id, chunk_id, value}``
dicts) alongside the comparison axis in ``metadata["axis"]``.
"""
from __future__ import annotations

from src.intelligence.sme.artifact_models import ArtifactItem
from src.intelligence.sme.builders._base import ArtifactBuilder


class ComparativeRegisterBuilder(ArtifactBuilder):
    """Builder for the comparative register artifact."""

    artifact_type = "comparison"

    def _synthesize(
        self, *, subscription_id: str, profile_id: str, adapter, version: int
    ) -> list[ArtifactItem]:
        # Phase 1: empty list. Phase 2 walks adapter.comparison_axes and
        # materializes comparisons from the profile chunk corpus.
        return []
