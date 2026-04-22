"""Phase 3 Task 6 — adaptive per-intent / per-layer top-K resolution.

The orchestrator (:meth:`src.retrieval.unified_retriever.UnifiedRetriever
.retrieve_four_layers`) asks this module how many candidates each layer
should return for a given intent. The canonical table comes from the
Phase 3 plan:

* ``trivial``/``greeting``/``identity``  → all layers off
* ``lookup``/``count``                   → ``a=5, b=0, c=0``
* ``extract``/``list``/``aggregate``     → ``a=10, b=0, c=0-2``
* ``compare``/``summarize``/``overview`` → ``a=12, b=5, c=5``
* ``analyze``/``diagnose``/``recommend``/``investigate`` → ``a=15, b=10, c=10``

Query-complexity signals bump the base top-K by a small additive amount
(more entities → more KG coverage needed, multi-part queries → more
chunks). Adapter caps clamp both ends so per-domain tunables stay
authoritative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class LayerTopK:
    """Per-layer top-K budget. ``0`` means "skip this layer entirely"."""

    a: int
    b: int
    c: int

    def clamp(self, *, caps: Mapping[str, int] | None) -> "LayerTopK":
        """Return a new :class:`LayerTopK` clamped by adapter caps.

        ``caps`` is an adapter-provided ``{"a": int, "b": int, "c": int}``
        upper-bound map. Missing keys mean no cap on that layer.
        """
        if not caps:
            return self
        return LayerTopK(
            a=min(self.a, int(caps.get("a", self.a))),
            b=min(self.b, int(caps.get("b", self.b))),
            c=min(self.c, int(caps.get("c", self.c))),
        )


# ---------------------------------------------------------------------------
# Canonical intent table — plan Section "Task 6".
# ---------------------------------------------------------------------------

_SIMPLE = {"greeting", "identity", "meta", "farewell", "thanks"}

_BASE_TABLE: dict[str, LayerTopK] = {
    "greeting": LayerTopK(0, 0, 0),
    "identity": LayerTopK(0, 0, 0),
    "meta": LayerTopK(0, 0, 0),
    "farewell": LayerTopK(0, 0, 0),
    "thanks": LayerTopK(0, 0, 0),
    "lookup": LayerTopK(5, 0, 0),
    "count": LayerTopK(5, 0, 0),
    "extract": LayerTopK(10, 0, 2),
    "list": LayerTopK(10, 0, 2),
    "aggregate": LayerTopK(10, 0, 2),
    "compare": LayerTopK(12, 5, 5),
    "summarize": LayerTopK(12, 5, 5),
    "overview": LayerTopK(12, 5, 5),
    "analyze": LayerTopK(15, 10, 10),
    "diagnose": LayerTopK(15, 10, 10),
    "recommend": LayerTopK(15, 10, 10),
    "investigate": LayerTopK(15, 10, 10),
}

_UNKNOWN_FALLBACK: LayerTopK = LayerTopK(5, 0, 0)


def base_top_k(intent: str) -> LayerTopK:
    """Return the canonical table value for ``intent``.

    Unknown intents fall back to the conservative ``lookup`` shape so
    Phase 3 cannot starve the pack on a new intent it hasn't seen.
    """
    return _BASE_TABLE.get(intent, _UNKNOWN_FALLBACK)


# ---------------------------------------------------------------------------
# Complexity-aware top-K
# ---------------------------------------------------------------------------


def _entity_count(query_analysis: Mapping[str, Any] | None) -> int:
    if not query_analysis:
        return 0
    raw = query_analysis.get("entities")
    if raw is None:
        return 0
    if isinstance(raw, (list, tuple, set)):
        return len(raw)
    if isinstance(raw, int):
        return raw
    return 0


def _temporal_span_months(query_analysis: Mapping[str, Any] | None) -> float:
    if not query_analysis:
        return 0.0
    span = query_analysis.get("temporal_span_months")
    if isinstance(span, (int, float)):
        return float(span)
    return 0.0


def _sub_query_count(query_analysis: Mapping[str, Any] | None) -> int:
    if not query_analysis:
        return 0
    raw = query_analysis.get("sub_queries")
    if isinstance(raw, (list, tuple)):
        return len(raw)
    if isinstance(raw, int):
        return raw
    return 0


def compute_top_k(
    intent: str,
    query_analysis: Mapping[str, Any] | None = None,
    adapter_caps: Mapping[str, int] | None = None,
) -> LayerTopK:
    """Compute the per-layer top-K for ``intent`` scaled by complexity.

    The complexity scaling is additive and modest — entity count
    contributes to Layer B, sub-query count lifts Layer A, and a long
    temporal span (> 12 months) bumps Layer C. Simple intents (greeting,
    identity) ignore complexity entirely and always stay at ``LayerTopK(0,0,0)``.

    ``adapter_caps`` is an optional upper-bound map ``{"a": int, "b":
    int, "c": int}`` that lets domain adapters clamp heavy-hitting
    intents down to their retrieval budgets.
    """
    if intent in _SIMPLE:
        return _BASE_TABLE[intent]

    base = base_top_k(intent)

    if base.a == 0 and base.b == 0 and base.c == 0:
        return base.clamp(caps=adapter_caps)

    entities = _entity_count(query_analysis)
    subs = _sub_query_count(query_analysis)
    temporal_months = _temporal_span_months(query_analysis)

    # Modest additive bumps — the canonical table already differentiates
    # intent tiers; complexity only nudges within the tier.
    bump_a = 0
    bump_b = 0
    bump_c = 0
    if subs >= 2:
        bump_a += 3
    if subs >= 4:
        bump_a += 3
    if entities >= 3:
        bump_b += 2
    if entities >= 6:
        bump_b += 2
    if temporal_months >= 12:
        bump_c += 2
    if temporal_months >= 36:
        bump_c += 3

    adjusted = LayerTopK(
        a=base.a + bump_a,
        b=base.b + bump_b if base.b > 0 else 0,
        c=base.c + bump_c if base.c > 0 else 0,
    )
    return adjusted.clamp(caps=adapter_caps)


__all__ = ["LayerTopK", "base_top_k", "compute_top_k"]
