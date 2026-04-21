"""RecommendationBankBuilder — grounded recommendations
(spec §6, artifact_type="recommendation").

Phase 1 stub returns ``[]``. Phase 2 implements the real synthesis: for each
adapter-configured recommendation frame that matches the profile state, emits
an item whose ``text`` is the recommendation narrative and whose ``metadata``
carries ``{rationale, linked_insights, estimated_impact, assumptions,
caveats}``. Evidence refs cite the insights + source chunks that ground the
recommendation so the verifier can tie every recommendation back to data.
"""
from __future__ import annotations

from src.intelligence.sme.artifact_models import ArtifactItem
from src.intelligence.sme.builders._base import ArtifactBuilder


class RecommendationBankBuilder(ArtifactBuilder):
    """Builder for the recommendation-bank artifact."""

    artifact_type = "recommendation"

    def _synthesize(
        self, *, subscription_id: str, profile_id: str, adapter, version: int
    ) -> list[ArtifactItem]:
        # Phase 1: empty list. Phase 2 walks adapter.recommendation_frames
        # and materializes grounded recommendations from the insight index.
        return []
