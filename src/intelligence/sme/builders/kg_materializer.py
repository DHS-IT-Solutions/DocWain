"""KGMultiHopMaterializer — produces :class:`ArtifactItem` rows for every
INFERRED_RELATION edge the adapter's ``kg_inference_rules`` discover.

Phase 2 implementation. Phase 1 shipped the skeleton returning ``[]``. This
module now:

1. Validates every ``rule['pattern']`` against the ERRATA §15 allowlist regex
   ``^[A-Za-z0-9_,\\s\\->()\\[\\]:]+$`` before any Cypher interpolation. A
   pattern carrying disallowed characters (e.g. a ``RETURN apoc.do...``
   injection attempt) raises ``ValueError`` BEFORE the Neo4j driver is
   touched. Phase 1's adapter schema is the first-line defence; this is
   defence-in-depth.
2. Uses the injected :class:`KGQueryClient` to run the adapter pattern (with
   the profile hard-filter fixed on the driver side) and yield candidate
   inference paths.
3. Emits one :class:`ArtifactItem` per candidate edge carrying
   ``artifact_type="kg_edge"``, the source/target nodes + the relation type
   in ``metadata``, and the full inference path so :meth:`SMEVerifier.verify`'s
   check 3 (inference-provenance) can validate path length against
   ``rule['max_hops']``.

The builder does NOT write to Neo4j directly. The orchestrator's
:class:`SMEArtifactStorage.persist_items` call writes INFERRED_RELATION edges
when ``artifact_type == "kg_edge"`` — the builder's only job is to stage the
candidates so the verifier can drop items that fail grounding before anything
hits the graph.

LLM injection is optional here. Most KG inference rules produce candidate
edges deterministically from the Cypher pattern; a secondary LLM pass can
materialize narrative rationale, but the Phase 2 minimum wires the
deterministic path only.
"""
from __future__ import annotations

import re
import uuid
from typing import Any, Iterable, Protocol

from src.intelligence.sme.artifact_models import ArtifactItem, EvidenceRef
from src.intelligence.sme.builders._base import ArtifactBuilder
from src.intelligence.sme.builders.dossier import _TraceSink


PATTERN_ALLOWED = re.compile(r"^[A-Za-z0-9_,\s\->()\[\]:]+$")


def _validate_pattern(pattern: str) -> None:
    """Raise :class:`ValueError` when ``pattern`` contains disallowed characters.

    Defence-in-depth per ERRATA §15. The adapter YAML schema validator is the
    first-line gate; this runtime check protects against adapters that bypass
    the pydantic path (fixtures, tests, monkey-patches).
    """
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("kg_inference_rules.pattern must be a non-empty string")
    if not PATTERN_ALLOWED.match(pattern):
        raise ValueError(
            "kg_inference_rules.pattern contains disallowed characters "
            f"(ERRATA §15); got: {pattern!r}"
        )


class KGQueryClient(Protocol):
    """Structural Neo4j query surface used by the materializer.

    Implementations run the adapter-supplied pattern Cypher with the
    ``subscription_id`` / ``profile_id`` hard-filter applied at the driver
    layer. The builder never interpolates parameter values into Cypher — the
    client is responsible for parameter binding. Returned rows minimally
    carry ``src_node_id``, ``dst_node_id``, ``inference_path`` (list of dicts
    ``{"from", "edge", "to"}``) and ``evidence`` (list of dicts
    ``{"doc_id", "chunk_id"}``).
    """

    def run_pattern(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        pattern: str,
        max_hops: int,
    ) -> Iterable[dict[str, Any]]: ...


class KGMultiHopMaterializer(ArtifactBuilder):
    """Builder for INFERRED_RELATION edges across adapter rules."""

    artifact_type = "kg_edge"

    def __init__(
        self,
        *,
        ctx,
        kg: KGQueryClient,
        trace: _TraceSink,
    ) -> None:
        super().__init__(ctx=ctx)
        self._kg = kg
        self._trace = trace

    def _synthesize(
        self, *, subscription_id: str, profile_id: str, adapter, version: int
    ) -> list[ArtifactItem]:
        items: list[ArtifactItem] = []
        for rule in adapter.kg_inference_rules:
            pattern = rule.pattern
            produces = rule.produces
            confidence_floor = float(rule.confidence_floor)
            max_hops = int(rule.max_hops)

            # Hard defence-in-depth per ERRATA §15. Raise BEFORE touching
            # the query client so adapters that slipped past Phase 1's
            # schema validator (e.g. monkey-patched fixtures) can't build
            # Cypher strings with disallowed payloads.
            _validate_pattern(pattern)
            try:
                rows = list(
                    self._kg.run_pattern(
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        pattern=pattern,
                        max_hops=max_hops,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self._trace.append(
                    {
                        "stage": "builder_kg_error",
                        "builder": self.artifact_type,
                        "produces": produces,
                        "error": str(exc),
                    }
                )
                continue

            for row in rows:
                item = self._row_to_item(
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    produces=produces,
                    pattern=pattern,
                    confidence_floor=confidence_floor,
                    max_hops=max_hops,
                    row=row,
                )
                if item is not None:
                    items.append(item)
        return items

    def _row_to_item(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        produces: str,
        pattern: str,
        confidence_floor: float,
        max_hops: int,
        row: dict[str, Any],
    ) -> ArtifactItem | None:
        src = str(row.get("src_node_id") or "")
        dst = str(row.get("dst_node_id") or "")
        if not src or not dst:
            self._trace.append(
                {
                    "stage": "builder_kg_skip_missing_nodes",
                    "builder": self.artifact_type,
                    "produces": produces,
                }
            )
            return None
        raw_path = row.get("inference_path") or []
        inference_path: list[dict[str, Any]] = []
        for hop in raw_path:
            if not isinstance(hop, dict):
                continue
            inference_path.append(
                {
                    "from": str(hop.get("from", "")),
                    "edge": str(hop.get("edge", "")),
                    "to": str(hop.get("to", "")),
                }
            )
        if not inference_path:
            # A materializer with no path to justify an edge is skipped —
            # SMEVerifier's inference-provenance check (3) already drops
            # empty-path items, but rejecting here keeps the storage layer
            # from writing degenerate rows.
            self._trace.append(
                {
                    "stage": "builder_kg_skip_empty_path",
                    "builder": self.artifact_type,
                    "produces": produces,
                }
            )
            return None
        if len(inference_path) > max_hops:
            self._trace.append(
                {
                    "stage": "builder_kg_skip_over_hops",
                    "builder": self.artifact_type,
                    "produces": produces,
                    "path_length": len(inference_path),
                    "max_hops": max_hops,
                }
            )
            return None
        evidence_rows = row.get("evidence") or []
        evidence: list[EvidenceRef] = []
        for ev in evidence_rows:
            if not isinstance(ev, dict):
                continue
            doc_id = str(ev.get("doc_id") or "")
            chunk_id = str(ev.get("chunk_id") or "")
            if not doc_id or not chunk_id:
                continue
            evidence.append(EvidenceRef(doc_id=doc_id, chunk_id=chunk_id))
        if not evidence:
            self._trace.append(
                {
                    "stage": "builder_kg_skip_no_evidence",
                    "builder": self.artifact_type,
                    "produces": produces,
                }
            )
            return None
        raw_confidence = row.get("confidence")
        try:
            confidence = (
                float(raw_confidence) if raw_confidence is not None else confidence_floor
            )
        except (TypeError, ValueError):
            confidence = confidence_floor
        confidence = max(0.0, min(1.0, confidence))
        item_id = (
            f"kg_edge:{subscription_id}:{profile_id}:{produces}:{uuid.uuid4().hex[:8]}"
        )
        return ArtifactItem(
            item_id=item_id,
            artifact_type="kg_edge",
            subscription_id=subscription_id,
            profile_id=profile_id,
            text=f"{src} -[{produces}]-> {dst}",
            evidence=evidence,
            confidence=confidence,
            inference_path=inference_path,
            domain_tags=[produces],
            metadata={
                "from_node": src,
                "to_node": dst,
                "relation_type": produces,
                "rule_pattern": pattern,
                "confidence_floor": confidence_floor,
                "max_hops": max_hops,
            },
        )
