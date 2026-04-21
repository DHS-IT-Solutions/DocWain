"""Phase 3 Task 5 — budget-aware pack assembly over :class:`PackedItem`.

The assembler enforces the adapter's per-intent ``max_pack_tokens`` cap.
Drop order is lowest-confidence first, with ties broken by lowest
rerank_score. Layer C SME artifacts are compressed to a short
key-claims-plus-evidence-refs form so they don't blow the budget with
full prose. A final semantic dedup collapses items whose tokenised text
overlap exceeds a threshold — the reranker may have promoted two items
that paraphrase the same fact, and the pack assembler is the last place
to collapse those before the reasoner burns tokens re-reading them.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Iterable, Sequence

from src.retrieval.types import PackedItem


# 4 chars ≈ 1 token heuristic — adequate for budget gating. The exact
# tokeniser count doesn't matter here because the pack assembler is
# enforcing a safety cap, not the final prompt limit.
def _approx_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


class PackAssembler:
    """Adapter-driven pack assembler.

    ``adapter`` must expose ``retrieval_caps.max_pack_tokens`` as a mapping
    from intent → int. Unknown intents fall back to a ``generic`` cap, or
    an assembler-default of 4000 if neither is set — so callers can pass
    a stub adapter in tests without the whole pipeline complaining.
    """

    DEFAULT_CAP: int = 4000
    DEDUP_OVERLAP: float = 0.85
    # Upper bound on compressed SME body length (characters). Cheaper than
    # token counts; mirrors the ``_compress_sme`` budget reviewers expect.
    SME_BODY_CHAR_BUDGET: int = 1200

    def __init__(self, adapter: Any) -> None:
        caps_root = getattr(adapter, "retrieval_caps", None) or {}
        # Adapters are pydantic models; ``.retrieval_caps`` may be a dict
        # or a model attribute exposing ``max_pack_tokens``.
        max_pack = (
            caps_root.get("max_pack_tokens")
            if isinstance(caps_root, dict)
            else getattr(caps_root, "max_pack_tokens", None)
        ) or {}
        self._caps: dict[str, int] = {k: int(v) for k, v in (max_pack or {}).items()}
        self._default = int(self._caps.get("generic", self.DEFAULT_CAP))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def assemble(
        self,
        items: Sequence[PackedItem],
        intent: str,
    ) -> list[PackedItem]:
        """Return a budget-respecting ordered subset of ``items``.

        Algorithm:

        1. Compress Layer C SME items to key-claims + evidence refs.
        2. Sort by ``(confidence desc, rerank_score desc)`` so drop cuts
           the tail predictably.
        3. Greedy-pack until budget runs out; items that don't fit are
           dropped.
        4. Semantic dedup (Jaccard > 0.85) collapses near-duplicate text.

        Profile isolation is NOT enforced here — callers MUST pre-filter
        to the correct (subscription_id, profile_id). The assembler is
        scope-agnostic by design so it can be unit-tested without the
        MongoDB / Qdrant / Neo4j stack spun up.
        """
        if not items:
            return []

        budget = self._budget_for(intent)

        # Compress SME items so their token counts reflect the final
        # shape, not the uncompressed narrative.
        compressed: list[tuple[PackedItem, int]] = []
        for it in items:
            final = _compress_if_sme(it, self.SME_BODY_CHAR_BUDGET)
            compressed.append((final, _approx_tokens(final.text)))

        # Drop order: keep highest-confidence first; break ties by
        # rerank_score. Stable sort preserves caller order within ties.
        compressed.sort(
            key=lambda t: (t[0].confidence, t[0].rerank_score),
            reverse=True,
        )

        picked: list[PackedItem] = []
        used = 0
        for it, tokens in compressed:
            if used + tokens > budget:
                continue
            picked.append(it)
            used += tokens

        return _semantic_dedup(picked, threshold=self.DEDUP_OVERLAP)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _budget_for(self, intent: str) -> int:
        """Return the token budget for ``intent``.

        Falls back in order: explicit intent cap → ``generic`` cap →
        :attr:`DEFAULT_CAP`. Unknown intents never raise — assembly is
        best-effort under budget constraints.
        """
        if intent in self._caps:
            return self._caps[intent]
        return self._default


# ---------------------------------------------------------------------------
# Compression + dedup helpers
# ---------------------------------------------------------------------------


def _compress_if_sme(it: PackedItem, body_budget: int) -> PackedItem:
    """If ``it`` is a Layer C SME item, return a compressed :class:`PackedItem`.

    The compressed form is ``[SME/{artifact_type}] claim1 | claim2 | ...
    [evidence_refs]``. Other layers pass through unchanged — their text
    is already chunk-sized.
    """
    if it.layer != "c":
        return it
    meta = it.metadata or {}
    artifact_type = meta.get("artifact_type") or "sme"

    # Key claims: either already provided in metadata, or derived from
    # the full text by clipping to the body budget.
    claims = meta.get("key_claims")
    if not claims:
        base = (it.text or "")[:body_budget]
        claims = [base] if base else []

    body = " | ".join(str(c) for c in claims[:8] if c)
    ev_refs = []
    for doc_id, chunk_id in it.provenance[:6]:
        ev_refs.append(f"[{doc_id}#{chunk_id}]")
    refs_str = " ".join(ev_refs)
    text = f"[SME/{artifact_type}] {body} {refs_str}".strip()
    return replace(it, text=text)


def _token_overlap(a: str, b: str) -> float:
    """Jaccard over lowercased word tokens. The pack assembler's dedup
    threshold is higher than the MMR diversity knob because at this stage
    we're collapsing final duplicates, not just breaking ties."""
    ta = set((a or "").lower().split())
    tb = set((b or "").lower().split())
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    union = ta | tb
    return len(inter) / max(1, len(union))


def _semantic_dedup(
    items: list[PackedItem], *, threshold: float
) -> list[PackedItem]:
    kept: list[PackedItem] = []
    for it in items:
        dup = False
        for k in kept:
            if _token_overlap(it.text, k.text) >= threshold:
                # Prefer the SME-backed variant when available.
                if it.sme_backed and not k.sme_backed:
                    kept[kept.index(k)] = it
                dup = True
                break
        if not dup:
            kept.append(it)
    return kept


__all__ = ["PackAssembler"]
