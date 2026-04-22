"""Phase 3 Task 9 — intent-aware per-layer gating.

Spec Section 7 mechanism #5: simple intents (``greeting``, ``identity``,
``lookup``, ``count``, ``extract``) skip Layer B and Layer C entirely —
those layers add latency and pack tokens but provide no lift for
single-fact lookups. Borderline intents (``compare``, ``summarize``,
``overview``) run all three layers because SME artifacts help multi-doc
synthesis. Analytical intents (``analyze``, ``diagnose``, ``recommend``,
``investigate``) always run B + C with full top-K.

The gate is a pure function: it inspects the resolved intent (plus an
optional user-requested-compact flag) and returns a
:class:`GateDecision` carrying three ``run_{a,b,c}`` booleans. The
orchestrator (:meth:`src.retrieval.unified_retriever.UnifiedRetriever
.retrieve_four_layers`) consults the gate before dispatching so gated
layers never hit their backing store — not just "run and drop". That
matters because Layer B is a Neo4j round-trip and Layer C is a Qdrant
scroll over SME snippets; skipping them is the latency win.

Unknown intents default to running everything — better to over-retrieve
and let the pack assembler trim than to starve an intent we haven't
classified yet.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateDecision:
    """Per-layer run decision — frozen so callers can't mutate."""

    run_a: bool
    run_b: bool
    run_c: bool


# Canonical gate table. Mirrors the per-intent top-K defaults in
# :mod:`src.retrieval.top_k` — any intent with top-K ``0`` on a layer has
# that layer gated off here too (consistency check in tests).
_GATE_TABLE: dict[str, GateDecision] = {
    # Pure conversational — no retrieval at all.
    "greeting":    GateDecision(False, False, False),
    "identity":    GateDecision(False, False, False),
    "meta":        GateDecision(False, False, False),
    "farewell":    GateDecision(False, False, False),
    "thanks":      GateDecision(False, False, False),
    # Single-fact lookups — Layer A only.
    "lookup":      GateDecision(True,  False, False),
    "count":       GateDecision(True,  False, False),
    # Structured-extract / list / aggregate — Layer A + Layer C SME for
    # analysis-style lists; KG layer usually unhelpful for single-row answers.
    "extract":     GateDecision(True,  False, True),
    "list":        GateDecision(True,  False, True),
    "aggregate":   GateDecision(True,  False, True),
    # Borderline synthesis — all three layers (SME artifacts help summary).
    "compare":     GateDecision(True,  True,  True),
    "summarize":   GateDecision(True,  True,  True),
    "overview":    GateDecision(True,  True,  True),
    # Analytical — all three layers, full top-K.
    "analyze":     GateDecision(True,  True,  True),
    "diagnose":    GateDecision(True,  True,  True),
    "recommend":   GateDecision(True,  True,  True),
    "investigate": GateDecision(True,  True,  True),
}


_UNKNOWN_FALLBACK = GateDecision(True, True, True)


class IntentGate:
    """Intent → per-layer run decision.

    Instantiation is free; the gate is stateless. Subclasses / tests can
    inject a custom table via the constructor — Phase 4 may add
    rich-mode overrides, this class stays agnostic.
    """

    def __init__(self, table: dict[str, GateDecision] | None = None) -> None:
        self._table = dict(table) if table is not None else dict(_GATE_TABLE)

    def decide(
        self, intent: str, *, user_requested_compact: bool = False
    ) -> GateDecision:
        """Resolve ``intent`` → :class:`GateDecision`.

        ``user_requested_compact=True`` force-shuts Layers B and C off
        regardless of intent — used when the caller wants a single-layer
        response (e.g. explicit ``/compact`` slash command in the UI).
        Layer A always stays at whatever the table says so a compact
        request against an intent that itself gates A off (``greeting``)
        still skips retrieval.
        """
        base = self._table.get(intent, _UNKNOWN_FALLBACK)
        if user_requested_compact:
            return GateDecision(run_a=base.run_a, run_b=False, run_c=False)
        return base


__all__ = ["GateDecision", "IntentGate"]
