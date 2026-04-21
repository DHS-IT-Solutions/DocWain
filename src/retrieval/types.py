"""Phase 3 retrieval types — shared contracts between orchestrator, merge,
rerank, pack assembly, and the core agent.

Per ERRATA §11, :class:`PackedItem` is the canonical cross-layer contract
Phase 4 consumes. It is a frozen dataclass so the reasoner path cannot
accidentally mutate provenance or confidence mid-pipeline.
:class:`RetrievalBundle` holds the raw per-layer outputs before merge and
rerank; every layer returns ``list[dict]`` which the Phase 3 merge helper
normalises into :class:`PackedItem` form.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

LayerTag = Literal["a", "b", "c", "d"]


@dataclass(frozen=True)
class PackedItem:
    """Unified retrieval candidate — the shape the reasoner sees.

    ``provenance`` is a tuple of (doc_id, chunk_id) pairs so the item can
    carry multiple citations (Layer C artifacts cite several evidence
    chunks). ``layer`` is the single-letter tag matching the source layer
    (``a`` chunks, ``b`` KG, ``c`` SME, ``d`` URL). ``sme_backed`` marks
    items that came from pre-reasoned synthesis — Layer C always true,
    Layer B synthesized edges (``kind == "kg_inferred"``) true, everything
    else false. ``metadata`` carries adapter-agnostic extras like
    ``artifact_type`` and ``relation_type``.
    """

    text: str
    provenance: tuple[tuple[str, str], ...]
    layer: LayerTag
    confidence: float
    rerank_score: float = 0.0
    sme_backed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalBundle:
    """Raw per-layer results plus telemetry.

    The merge step (Task 4) consumes ``layer_{a,b,c,d}`` and produces a
    flat list of :class:`PackedItem`. ``degraded_layers`` holds the
    full-name (``"layer_a"``, ``"layer_b"``, ...) of any layer that raised
    during parallel dispatch — the orchestrator never aborts on a single
    layer failure, it simply logs and drops the layer.

    Per ERRATA §11: ``degraded_layers`` is single-append (no double-append
    of short + long name). Only the full name lands in the list.
    """

    layer_a_chunks: list[dict[str, Any]] = field(default_factory=list)
    layer_b_kg: list[dict[str, Any]] = field(default_factory=list)
    layer_c_sme: list[dict[str, Any]] = field(default_factory=list)
    layer_d_url: list[dict[str, Any]] = field(default_factory=list)
    degraded_layers: list[str] = field(default_factory=list)
    per_layer_ms: dict[str, float] = field(default_factory=dict)
