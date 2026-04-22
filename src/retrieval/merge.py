"""Phase 3 retrieval merge, rerank, and MMR diversity.

Three pure-function surfaces that operate over :class:`PackedItem` and raw
per-layer dicts:

* :func:`merge_layers` — union the four Phase 3 retrieval layers,
  deduplicate on ``(doc_id, chunk_id)``, tag each item with its source
  layer, and set ``sme_backed=True`` for Layer C items and Layer B items
  whose ``kind == "kg_inferred"`` (ERRATA §11).
* :func:`rerank_merged_candidates` — apply the cross-encoder blend with
  an SME-intent bonus (``analyze`` / ``diagnose`` / ``recommend``), then
  sort descending by the blended rerank score.
* :func:`mmr_select` — maximal marginal relevance selector over
  :class:`PackedItem` candidates with adapter-tunable ``lam``.

This module does not talk to Qdrant, Neo4j, Redis, or the LLM gateway.
Every dependency is passed in by the caller (tests pass MagicMocks). The
merge layer is the seam Phase 4 will consume — response-shape changes live
in Phase 4, not here.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from src.retrieval.types import PackedItem

# Intent bonus applied to SME-backed items during rerank. Small enough not
# to overpower the cross-encoder signal on bad SME hits, large enough to
# break ties toward pre-reasoned synthesis when the analytical intent
# benefits from it (spec §7 stage 2).
_SME_INTENT_BONUS = {
    "analyze": 0.08,
    "diagnose": 0.08,
    "recommend": 0.10,
    "investigate": 0.05,
}


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

_LAYER_TAG_MAP = {
    "layer_a": "a",
    "layer_b": "b",
    "layer_c": "c",
    "layer_d": "d",
}


def _tag_for(layer: str) -> str:
    """Map ``layer_*`` → single-letter tag per ERRATA §11 PackedItem contract."""
    return _LAYER_TAG_MAP.get(layer, "a")


def _item_key(it: dict) -> tuple:
    """Stable dedup key. Falls back to ``id(it)`` when no (doc, chunk)
    tuple is available so Layer B KG edges (which have no chunk_id) and
    Layer C snippets never collide with each other."""
    doc_id = it.get("doc_id")
    chunk_id = it.get("chunk_id")
    if doc_id and chunk_id:
        return ("chunk", doc_id, chunk_id)
    if it.get("kind") == "kg_direct" or it.get("kind") == "kg_inferred":
        return (
            "kg",
            it.get("src") or it.get("from"),
            it.get("dst") or it.get("to"),
            it.get("relation_type") or it.get("type"),
        )
    if it.get("kind") == "sme_artifact" and it.get("snippet_id"):
        return ("sme", it.get("snippet_id"))
    return ("obj", id(it))


def _is_sme_source(layer: str, it: dict) -> bool:
    """Per ERRATA §11: Layer C items AND Layer B items with
    ``kind == "kg_inferred"`` are SME-backed."""
    if layer == "layer_c":
        return True
    if layer == "layer_b" and it.get("kind") == "kg_inferred":
        return True
    return False


def merge_layers(bundle_or_layers: Any) -> list[PackedItem]:
    """Union four Phase 3 layer outputs into a flat :class:`PackedItem` list.

    Accepts either a :class:`RetrievalBundle` (dataclass instance) or a
    dict with keys ``a``, ``b``, ``c``, ``d``. Duplicate keys across
    layers (same ``(doc_id, chunk_id)`` pair) keep the first occurrence
    but promote ``sme_backed`` to True when any SME source flags the key.

    Returns an ordered list of :class:`PackedItem` with provenance,
    confidence, and ``sme_backed`` already populated. Callers pass this
    list to :func:`rerank_merged_candidates` or :func:`mmr_select`.
    """
    if hasattr(bundle_or_layers, "layer_a_chunks"):
        layers = {
            "layer_a": bundle_or_layers.layer_a_chunks,
            "layer_b": bundle_or_layers.layer_b_kg,
            "layer_c": bundle_or_layers.layer_c_sme,
            "layer_d": bundle_or_layers.layer_d_url,
        }
    else:
        d = bundle_or_layers or {}
        layers = {
            "layer_a": d.get("a") or d.get("layer_a_chunks") or [],
            "layer_b": d.get("b") or d.get("layer_b_kg") or [],
            "layer_c": d.get("c") or d.get("layer_c_sme") or [],
            "layer_d": d.get("d") or d.get("layer_d_url") or [],
        }

    seen: dict[tuple, dict] = {}
    order: list[tuple] = []  # preserves first-seen insertion order

    for layer_name, items in layers.items():
        for raw in items or []:
            it = dict(raw)  # defensive copy so we don't mutate caller state
            it.setdefault("layer", layer_name)
            key = _item_key(it)
            is_sme = _is_sme_source(layer_name, it)
            if key in seen:
                # Duplicate: promote sme_backed if any SME source flagged it.
                if is_sme:
                    seen[key]["sme_backed"] = True
                continue
            if is_sme:
                it["sme_backed"] = True
            seen[key] = it
            order.append(key)

    return [_to_packed_item(seen[k]) for k in order]


def _to_packed_item(it: dict) -> PackedItem:
    """Convert a merged raw dict into the frozen :class:`PackedItem`.

    Provenance is a tuple-of-tuples per ERRATA §11. ``text`` comes from
    ``text`` → ``narrative`` → rendered KG edge string depending on what
    the raw item carries.
    """
    text = (
        it.get("text")
        or it.get("narrative")
        or _render_kg_edge(it)
        or ""
    )
    prov: list[tuple[str, str]] = []
    doc_id = it.get("doc_id")
    chunk_id = it.get("chunk_id")
    if doc_id and chunk_id:
        prov.append((str(doc_id), str(chunk_id)))
    for ev in it.get("evidence", []) or []:
        if isinstance(ev, str) and "#" in ev:
            d, c = ev.split("#", 1)
            prov.append((d, c))
    layer_long = it.get("layer", "layer_a")
    metadata = {
        "artifact_type": it.get("artifact_type"),
        "relation_type": it.get("relation_type") or it.get("type"),
        "kind": it.get("kind"),
        "inference_path": it.get("inference_path"),
    }
    # Strip Nones from metadata for cleanliness.
    metadata = {k: v for k, v in metadata.items() if v is not None}
    return PackedItem(
        text=str(text),
        provenance=tuple(prov),
        layer=_tag_for(layer_long),
        confidence=float(it.get("confidence") or 0.5),
        rerank_score=float(it.get("rerank_score") or it.get("score") or 0.0),
        sme_backed=bool(it.get("sme_backed") or layer_long == "layer_c"),
        metadata=metadata,
    )


def _render_kg_edge(it: dict) -> str:
    if it.get("kind") not in {"kg_direct", "kg_inferred"}:
        return ""
    src = it.get("src") or it.get("from") or ""
    dst = it.get("dst") or it.get("to") or ""
    rel = it.get("relation_type") or it.get("type") or "related_to"
    return f"{src} {rel} {dst}".strip()


# ---------------------------------------------------------------------------
# Rerank
# ---------------------------------------------------------------------------


def rerank_merged_candidates(
    query: str,
    candidates: Sequence[PackedItem],
    cross_encoder: Any,
    *,
    top_k: int = 10,
    intent: str = "lookup",
    enable_cross_encoder: bool = True,
) -> list[PackedItem]:
    """Rerank merged :class:`PackedItem` candidates with the CE + bonus blend.

    Blend: ``0.6 * CE + 0.3 * raw + 0.1 * confidence``, plus
    ``+bonus`` when the item is ``sme_backed`` and the intent is in the
    SME-intent bonus table.

    When ``enable_cross_encoder=False`` or ``cross_encoder`` is ``None``,
    candidates are sorted by their existing ``rerank_score`` (falling back
    to ``confidence``) without invoking the model.

    Returns a new list with :attr:`PackedItem.rerank_score` replaced — the
    frozen dataclass means we allocate fresh instances rather than mutate
    the caller's inputs. The order is descending by blended score.
    """
    if not candidates:
        return []
    if not (enable_cross_encoder and cross_encoder is not None):
        return sorted(
            candidates,
            key=lambda p: (p.rerank_score, p.confidence),
            reverse=True,
        )[:top_k]

    # Pre-filter to the top 4×top_k (≥40) by raw/confidence before spending
    # cross-encoder budget — the CE is the dominant cost on CPU.
    pool = sorted(
        candidates,
        key=lambda p: (p.rerank_score, p.confidence),
        reverse=True,
    )[: max(40, top_k * 4)]

    pairs = [(query, (p.text or "")[:1600]) for p in pool]
    ce_scores = cross_encoder.predict(pairs)
    bonus_for_intent = _SME_INTENT_BONUS.get(intent, 0.0)

    reranked: list[PackedItem] = []
    for p, ce in zip(pool, ce_scores):
        blended = (
            0.6 * float(ce)
            + 0.3 * float(p.rerank_score or 0.0)
            + 0.1 * float(p.confidence or 0.0)
        )
        if p.sme_backed and bonus_for_intent:
            blended += bonus_for_intent
        reranked.append(
            PackedItem(
                text=p.text,
                provenance=p.provenance,
                layer=p.layer,
                confidence=p.confidence,
                rerank_score=float(blended),
                sme_backed=p.sme_backed,
                metadata=p.metadata,
            )
        )

    reranked.sort(key=lambda x: x.rerank_score, reverse=True)
    return reranked[:top_k]


# ---------------------------------------------------------------------------
# MMR
# ---------------------------------------------------------------------------


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _token_overlap(a: str, b: str) -> float:
    """Fallback diversity measure when embeddings aren't carried — Jaccard
    over lowercased token sets. Good enough for Phase 3 since the reranker
    already consumes the CE signal; MMR here just breaks ties toward
    distinct surface forms."""
    ta = set((a or "").lower().split())
    tb = set((b or "").lower().split())
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    union = ta | tb
    return len(inter) / max(1, len(union))


def mmr_select(
    items: Sequence[PackedItem],
    *,
    top_k: int,
    lam: float = 0.5,
) -> list[PackedItem]:
    """Maximal marginal relevance over :class:`PackedItem` candidates.

    ``lam == 1`` → pure score; ``lam == 0`` → pure diversity. When items
    carry no embedding the similarity falls back to token overlap (which
    still gives reasonable diversity across surface forms). The default
    0.5 balances both.

    The input order is preserved on the primary score descent; the output
    is the selected top-``top_k`` in selection order.
    """
    if not items:
        return []
    pool = list(items)
    pool.sort(key=lambda p: (p.rerank_score, p.confidence), reverse=True)
    if top_k >= len(pool):
        return pool

    selected: list[PackedItem] = []
    remaining = list(pool)
    while remaining and len(selected) < top_k:
        best: PackedItem | None = None
        best_score = -math.inf
        for cand in remaining:
            rel = cand.rerank_score or cand.confidence
            div = 0.0
            if selected:
                div = max(
                    _token_overlap(cand.text, s.text) for s in selected
                )
            mmr = lam * rel - (1.0 - lam) * div
            if mmr > best_score:
                best = cand
                best_score = mmr
        assert best is not None
        selected.append(best)
        remaining = [r for r in remaining if r is not best]
    return selected


__all__ = [
    "merge_layers",
    "rerank_merged_candidates",
    "mmr_select",
]
