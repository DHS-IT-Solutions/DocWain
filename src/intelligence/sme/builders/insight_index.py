"""InsightIndexBuilder — typed insight items (spec §6, artifact_type="insight").

Phase 1 stub returns ``[]``. Phase 2 implements the real synthesis: one item
per detected insight with the detector type (trend/anomaly/gap/risk/
opportunity/conflict) carried in ``domain_tags`` plus ``metadata["detector"]``,
the narrative in ``text``, supporting chunks in ``evidence``, and an optional
temporal scope in ``metadata["temporal_scope"]``.
"""
from __future__ import annotations

from src.intelligence.sme.artifact_models import ArtifactItem
from src.intelligence.sme.builders._base import ArtifactBuilder


class InsightIndexBuilder(ArtifactBuilder):
    """Builder for per-profile typed insight items."""

    artifact_type = "insight"

    def _synthesize(
        self, *, subscription_id: str, profile_id: str, adapter, version: int
    ) -> list[ArtifactItem]:
        # Phase 1: empty list. Phase 2 iterates the adapter's configured
        # insight detectors and materializes items via the detector registry.
        return []
