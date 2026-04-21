"""Persistence facade for SME artifacts (spec §9, ERRATA §2 + §20).

Three stores share one :class:`SMEArtifactStorage` facade:

* **Azure Blob** receives the canonical JSON payload at
  ``sme_artifacts/{sub}/{prof}/{artifact_type}/{synthesis_version}.json`` plus
  the synthesis-run manifest at
  ``sme_artifacts/{sub}/{prof}/manifest/{synthesis_id}.json``.
* **Qdrant** per-subscription collection ``sme_artifacts_{subscription_id}``
  receives one point per :class:`ArtifactItem` carrying subscription + profile
  filters for isolation. Vector insertion is populated by Phase 2 (this module
  writes payload-only in Phase 1).
* **Neo4j** receives ``INFERRED_RELATION`` edges, but only when the artifact
  type is ``"kg_edge"``; other artifact types never touch Neo4j.

The canonical public surface (ERRATA §2) is three *facade* methods plus one
convenience wrapper:

* :meth:`put_snippet` — streaming single-item Qdrant write.
* :meth:`put_canonical` — batch Blob write of the canonical JSON payload.
* :meth:`put_manifest` — manifest writer for the synthesis run.
* :meth:`persist_items` — ``put_canonical`` + ``put_snippet``-per-item for
  callers that want the whole dance in one call. Also writes Neo4j edges
  when the artifact type is ``"kg_edge"``.

:class:`StorageDeps` is the injected dep bundle. ``embedder`` is optional in
Phase 1 (skeletons don't produce snippets); Phase 2 makes it required at
``put_snippet`` call sites (see ERRATA §20).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from src.intelligence.sme.artifact_models import ArtifactItem


class BlobStore(Protocol):
    """Minimal Blob write surface.

    Implementations bridge to Azure Blob (or an equivalent durable object
    store). No DocWain-layer wall-clock timeout is added here — the underlying
    SDK enforces per-op safety limits (spec §3 invariant 8).
    """

    def write_text(self, path: str, content: str) -> None: ...
    def delete(self, path: str) -> None: ...


class QdrantBridge(Protocol):
    """Per-subscription Qdrant write surface.

    Phase 1 only uses ``upsert_points`` + ``delete_by_filter``; Phase 2 extends
    the write path with real dense vectors via the :attr:`StorageDeps.embedder`
    dependency. The Phase 1 ``points`` payload omits ``vector`` deliberately —
    the collection can still index it for payload-only filtering.
    """

    def upsert_points(
        self, *, collection: str, points: list[dict[str, Any]]
    ) -> None: ...
    def delete_by_filter(
        self, *, collection: str, filter: dict[str, Any]
    ) -> None: ...


class Neo4jBridge(Protocol):
    """Neo4j write surface. Only invoked for ``artifact_type == "kg_edge"``;
    other artifact types bypass this dependency entirely. The edge writer
    receives a batch of plain dicts (subscription, profile, from/to nodes,
    relation type, confidence, evidence, inference path) and is responsible
    for translating to the ``INFERRED_RELATION`` label in the graph."""

    def write_inferred_edges(self, edges: list[dict[str, Any]]) -> None: ...


@dataclass
class StorageDeps:
    """Canonical four-field dependency bundle per ERRATA §20.

    ``embedder`` is optional because Phase 1 skeletons don't produce items that
    need vector computation. Phase 2 Task 11 makes the embedder required at
    every ``put_snippet`` call site; callers should treat the Phase 1 default
    as a scaffolding aid, not a contract to rely on in production paths.
    """

    blob: BlobStore
    qdrant: QdrantBridge
    neo4j: Neo4jBridge
    embedder: object | None = None


class SMEArtifactStorage:
    """Facade over Blob + Qdrant + Neo4j writes for SME artifacts.

    Profile isolation is enforced at the payload level (every Qdrant payload
    carries ``subscription_id`` + ``profile_id``) and at the collection level
    (``sme_artifacts_{subscription_id}``). Neo4j edges likewise carry both
    identifiers, so the Phase 3 retrieval filter can match cleanly.
    """

    def __init__(self, deps: StorageDeps) -> None:
        self.deps = deps

    # ------------------------------------------------------------------
    # Canonical facade methods (ERRATA §2)
    # ------------------------------------------------------------------
    def put_snippet(
        self,
        subscription_id: str,
        profile_id: str,
        item: ArtifactItem,
        *,
        synthesis_version: int,
    ) -> None:
        """Write one retrievable snippet to ``sme_artifacts_{sub}`` in Qdrant.

        Phase 1 writes payload-only; Phase 2 extends the point dict with a
        ``vector`` entry computed from :attr:`StorageDeps.embedder`. The
        payload shape is frozen so retrieval layer consumers (Phase 3) can
        filter on ``subscription_id`` + ``profile_id`` + ``artifact_type`` +
        ``synthesis_version`` deterministically.
        """
        self.deps.qdrant.upsert_points(
            collection=f"sme_artifacts_{subscription_id}",
            points=[
                {
                    "id": item.item_id,
                    "payload": {
                        "subscription_id": subscription_id,
                        "profile_id": profile_id,
                        "artifact_type": item.artifact_type,
                        "text": item.text,
                        "confidence": item.confidence,
                        "domain_tags": item.domain_tags,
                        "evidence": [e.model_dump() for e in item.evidence],
                        "synthesis_version": synthesis_version,
                    },
                }
            ],
        )

    def put_canonical(
        self,
        subscription_id: str,
        profile_id: str,
        artifact_type: str,
        items: list[ArtifactItem],
        *,
        synthesis_version: int,
    ) -> None:
        """Write the canonical JSON payload for a batch to Azure Blob.

        Path layout: ``sme_artifacts/{sub}/{prof}/{artifact_type}/{version}.json``
        (spec §9). The body captures subscription + profile + artifact type +
        version alongside the serialized items so the payload is
        self-describing when read back by the rebuild path or diagnostics.
        """
        path = (
            f"sme_artifacts/{subscription_id}/{profile_id}/"
            f"{artifact_type}/{synthesis_version}.json"
        )
        body = {
            "subscription_id": subscription_id,
            "profile_id": profile_id,
            "artifact_type": artifact_type,
            "version": synthesis_version,
            "items": [it.model_dump(mode="json") for it in items],
        }
        self.deps.blob.write_text(path, json.dumps(body, ensure_ascii=False))

    def put_manifest(
        self,
        subscription_id: str,
        profile_id: str,
        manifest: dict[str, Any],
    ) -> None:
        """Write the run manifest at
        ``sme_artifacts/{sub}/{prof}/manifest/{synthesis_id}.json``.

        The manifest carries the per-artifact-type pointers written by
        :meth:`put_canonical` so downstream consumers can locate every output
        from a single run with one Blob GET.
        """
        synthesis_id = manifest.get("synthesis_id")
        if not synthesis_id:
            raise ValueError("manifest must include 'synthesis_id'")
        path = (
            f"sme_artifacts/{subscription_id}/{profile_id}/"
            f"manifest/{synthesis_id}.json"
        )
        self.deps.blob.write_text(path, json.dumps(manifest, ensure_ascii=False))

    # ------------------------------------------------------------------
    # Convenience wrapper (ERRATA §2)
    # ------------------------------------------------------------------
    def persist_items(
        self,
        subscription_id: str,
        profile_id: str,
        artifact_type: str,
        items: list[ArtifactItem],
        *,
        version: int,
    ) -> None:
        """Persist a full artifact batch across every backing store.

        Writes the canonical Blob JSON, upserts one Qdrant point per item, and
        (only for ``artifact_type == "kg_edge"``) writes ``INFERRED_RELATION``
        edges to Neo4j. This is the call path the synthesizer orchestrator
        uses; one-off writers can use the individual facade methods.
        """
        self.put_canonical(
            subscription_id,
            profile_id,
            artifact_type,
            items,
            synthesis_version=version,
        )
        # Vector insertion lands in Phase 2; payload-only in Phase 1. Using
        # the facade ensures both call sites stay shape-compatible.
        if items:
            self.deps.qdrant.upsert_points(
                collection=f"sme_artifacts_{subscription_id}",
                points=[
                    {
                        "id": it.item_id,
                        "payload": {
                            "subscription_id": subscription_id,
                            "profile_id": profile_id,
                            "artifact_type": artifact_type,
                            "text": it.text,
                            "confidence": it.confidence,
                            "domain_tags": it.domain_tags,
                            "evidence": [e.model_dump() for e in it.evidence],
                            "synthesis_version": version,
                        },
                    }
                    for it in items
                ],
            )
        if artifact_type == "kg_edge" and items:
            self.deps.neo4j.write_inferred_edges(
                [
                    {
                        "subscription_id": subscription_id,
                        "profile_id": profile_id,
                        "from_node": it.metadata["from_node"],
                        "to_node": it.metadata["to_node"],
                        "relation_type": it.metadata["relation_type"],
                        "confidence": it.confidence,
                        "evidence": [
                            f"{e.doc_id}#{e.chunk_id}" for e in it.evidence
                        ],
                        "inference_path": it.inference_path,
                    }
                    for it in items
                ]
            )

    def delete_version(
        self,
        subscription_id: str,
        profile_id: str,
        artifact_type: str,
        *,
        version: int,
    ) -> None:
        """Remove a single version of a given artifact type from Blob + Qdrant.

        The Qdrant filter clause matches on subscription + profile + artifact
        type + version so the per-subscription collection stays clean without
        dropping unrelated artifact types. Neo4j edges are versioned at the
        edge-property level and cleared by the rebuild path in Phase 2; Phase 1
        does not mutate the graph on delete.
        """
        path = (
            f"sme_artifacts/{subscription_id}/{profile_id}/"
            f"{artifact_type}/{version}.json"
        )
        self.deps.blob.delete(path)
        self.deps.qdrant.delete_by_filter(
            collection=f"sme_artifacts_{subscription_id}",
            filter={
                "must": [
                    {"key": "subscription_id", "value": subscription_id},
                    {"key": "profile_id", "value": profile_id},
                    {"key": "artifact_type", "value": artifact_type},
                    {"key": "synthesis_version", "value": version},
                ]
            },
        )
