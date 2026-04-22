"""Abstract base class for SME artifact builders (ERRATA §6).

The orchestrator (:class:`src.intelligence.sme.synthesizer.SMESynthesizer`)
treats builders via a :class:`typing.Protocol`, but concrete builders inherit
from this ABC so they share construction, context wiring, and the public
``build`` signature. Phase 1 subclasses implement ``_synthesize`` as a return
``[]`` stub; Phase 2 fills each body without changing the public surface.

``BuilderContext`` is a structural protocol so the training-stage integration
can wire whatever concrete context fits (Qdrant + Neo4j readers) without
imposing an import cycle on the SME package.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

from src.intelligence.sme.adapter_schema import Adapter
from src.intelligence.sme.artifact_models import ArtifactItem


class BuilderContext(Protocol):
    """Read-side surface each builder receives.

    Phase 2 expands with additional readers (KG sparql, entity registry,
    temporal index, etc.). Phase 1 keeps the two primary iterators the
    majority of builders need so the stubs are interchangeable with real
    implementations in later phases.
    """

    def iter_profile_chunks(
        self, subscription_id: str, profile_id: str
    ) -> list[dict[str, Any]]: ...

    def iter_profile_kg(
        self, subscription_id: str, profile_id: str
    ) -> list[dict[str, Any]]: ...


class ArtifactBuilder(ABC):
    """Common base for every SME artifact builder.

    Subclasses must set :attr:`artifact_type` at class body level and
    implement :meth:`_synthesize`. The public :meth:`build` is final — it
    forwards directly to ``_synthesize`` so Phase 2 only needs to fill the
    inner method.
    """

    #: Slug identifying the artifact type produced by this builder. Must
    #: match :class:`src.intelligence.sme.artifact_models.ArtifactItem`'s
    #: ``artifact_type`` enum for items the builder emits.
    artifact_type: str

    def __init__(self, *, ctx: BuilderContext) -> None:
        self._ctx = ctx

    def build(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        adapter: Adapter,
        version: int,
    ) -> list[ArtifactItem]:
        """Public builder entrypoint (frozen signature, Phase 1→6)."""
        return self._synthesize(
            subscription_id=subscription_id,
            profile_id=profile_id,
            adapter=adapter,
            version=version,
        )

    @abstractmethod
    def _synthesize(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        adapter: Adapter,
        version: int,
    ) -> list[ArtifactItem]:
        """Produce artifact items. Phase 1 subclasses return ``[]``; Phase 2
        replaces the body with the real synthesis implementation."""
