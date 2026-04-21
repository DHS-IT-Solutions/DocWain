"""SMEDossierBuilder — dossier artifact (spec §6, artifact_type="dossier").

Phase 1 stub returns ``[]``. Phase 2 implements the real synthesis: one
:class:`ArtifactItem` per dossier section (summary, key_entities,
obligations, ...) with section narrative as ``text``, entity mentions in
``metadata["entities"]``, and ≥1 :class:`EvidenceRef` per section so the
verifier's ``evidence_presence`` / ``evidence_validity`` checks pass.
"""
from __future__ import annotations

from src.intelligence.sme.artifact_models import ArtifactItem
from src.intelligence.sme.builders._base import ArtifactBuilder


class SMEDossierBuilder(ArtifactBuilder):
    """Builder for the per-profile domain-aware dossier."""

    artifact_type = "dossier"

    def _synthesize(
        self, *, subscription_id: str, profile_id: str, adapter, version: int
    ) -> list[ArtifactItem]:
        # Phase 1: empty list keeps the full pipeline exercisable end-to-end
        # without producing synthetic content. Phase 2 replaces this with the
        # real dossier section generation driven by adapter.dossier.
        return []
