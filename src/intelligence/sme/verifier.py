"""SMEVerifier: five-check fail-closed grounding gate (spec §6).

Canonical surface (ERRATA §3):

* ``SMEVerifier(chunk_store=..., max_inference_hops=3)``
* ``verify(item: ArtifactItem, ctx: VerifierContext) -> Verdict`` — single item
* ``verify_batch(items: list[ArtifactItem], ctx: VerifierContext) -> list[Verdict]``

Five fail-closed checks, in order:

1. **Evidence presence** — ≥1 evidence ref.
2. **Evidence validity** — every cited chunk exists; item text substantively
   overlaps with at least one cited chunk; any cited ``quote`` is present in
   its chunk.
3. **Inference provenance** — ``len(inference_path)`` ≤ the effective
   ``max_inference_hops`` (ctx overrides the verifier default).
4. **Confidence calibration** — claims with ``confidence > 0.8`` require ≥2
   evidence sources; otherwise the verifier rolls confidence back to ``0.6``
   and still passes.
5. **Contradiction** — within a batch, an item that semantically contradicts a
   higher-confidence prior item is dropped unless it carries a ``conflict``
   tag (annotated disagreement is explicitly permitted).
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Protocol

from src.intelligence.sme.artifact_models import ArtifactItem


class VerifierChunkStore(Protocol):
    """Chunk lookup surface used by the verifier. Phase 2 wires the real
    Qdrant-backed chunk reader; Phase 1 tests use a mock."""

    def chunk_exists(self, doc_id: str, chunk_id: str) -> bool: ...
    def chunk_text(self, doc_id: str, chunk_id: str) -> str: ...


@dataclass
class VerifierContext:
    """Per-call context passed to ``verify`` / ``verify_batch`` (ERRATA §3).

    Fields are optional so Phase 1 callers don't need to thread the full
    subscription + profile identity. ``max_inference_hops`` here is a per-call
    override of the verifier's instance default (Phase 2 uses this to honor an
    adapter's ``kg_inference_rules.*.max_hops``).
    """

    subscription_id: str | None = None
    profile_id: str | None = None
    max_inference_hops: int | None = None


@dataclass
class Verdict:
    """Single-item outcome returned by the verifier.

    ``adjusted_item`` is set only when ``passed`` is True; it may differ from
    the input when check 4 rolls back ``confidence``.
    """

    item_id: str
    passed: bool
    adjusted_item: ArtifactItem | None
    failing_check: str | None = None
    drop_reason: str | None = None


# --- Tuning constants. Spec §6 table B.3. --------------------------------
_ROLLBACK = 0.6
_HIGH_CONFIDENCE = 0.8
# Minimum ratio between item text and cited chunk; below this the item is
# considered not substantively present.
_TEXT_SIM_MIN = 0.25
# Quote must match its chunk to at least this ratio (closer than _TEXT_SIM_MIN
# because a quote is a verbatim claim).
_QUOTE_SIM_MIN = 0.5
# Items whose texts overlap above this ratio are considered "talking about the
# same thing" for the purposes of contradiction detection (check 5).
_CONTRADICTION_SIM = 0.6
# Opposite-word pairs used for the lightweight contradiction detector.
_OPPOSITE_PAIRS: list[tuple[str, str]] = [
    ("rose", "fell"),
    ("increase", "decrease"),
    ("up", "down"),
    ("grew", "declined"),
    ("gain", "loss"),
    ("profit", "loss"),
]


def _overlap(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _opposite(a: str, b: str) -> bool:
    """True if ``a`` and ``b`` contain opposite-sense words (check 5 helper)."""
    al, bl = a.lower(), b.lower()
    return any(
        (x in al and y in bl) or (y in al and x in bl) for x, y in _OPPOSITE_PAIRS
    )


class SMEVerifier:
    """Five-check grounding gate. Deliberately fail-closed: any check failure
    drops the item (check 4 is the single exception — it rolls back
    confidence rather than dropping)."""

    def __init__(
        self,
        *,
        chunk_store: VerifierChunkStore,
        max_inference_hops: int = 3,
    ) -> None:
        self._cs = chunk_store
        self._max_hops_default = max_inference_hops

    # ------------------------------------------------------------------
    # Public API (ERRATA §3 canonical surface)
    # ------------------------------------------------------------------
    def verify(self, item: ArtifactItem, ctx: VerifierContext) -> Verdict:
        """Run checks 1-4 on a single item. Check 5 (contradiction) requires
        a batch context and is a no-op in single-item mode."""
        return self._verify_single(item, ctx)

    def verify_batch(
        self, items: list[ArtifactItem], ctx: VerifierContext
    ) -> list[Verdict]:
        """Run checks 1-5 on a batch; returns verdicts in the same order as
        the input list.

        Contradiction detection (check 5) walks items by descending confidence
        so lower-confidence items that disagree with accepted higher-confidence
        items are dropped deterministically, regardless of input order.
        """
        verdicts_by_id: dict[str, Verdict] = {}
        accepted: list[ArtifactItem] = []

        for item in sorted(items, key=lambda i: -i.confidence):
            single = self._verify_single(item, ctx)
            if not single.passed:
                verdicts_by_id[item.item_id] = single
                continue

            contradiction = self._check_contradiction(single.adjusted_item, accepted)
            if not contradiction.passed:
                verdicts_by_id[item.item_id] = contradiction
                continue

            accepted.append(single.adjusted_item)
            verdicts_by_id[item.item_id] = single

        # Preserve input order in the returned list.
        return [verdicts_by_id[item.item_id] for item in items]

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------
    def _verify_single(self, item: ArtifactItem, ctx: VerifierContext) -> Verdict:
        # Check 1: evidence presence.
        if not item.evidence:
            return Verdict(
                item_id=item.item_id,
                passed=False,
                adjusted_item=None,
                failing_check="evidence_presence",
                drop_reason="no evidence refs",
            )

        # Check 2: evidence validity (each ref → chunk exists; item text
        # substantively present in at least one chunk; any cited quote is in
        # its chunk).
        any_substantively_present = False
        for ref in item.evidence:
            if not self._cs.chunk_exists(ref.doc_id, ref.chunk_id):
                return Verdict(
                    item_id=item.item_id,
                    passed=False,
                    adjusted_item=None,
                    failing_check="evidence_validity",
                    drop_reason=(
                        f"chunk {ref.doc_id}#{ref.chunk_id} missing"
                    ),
                )
            chunk_text = self._cs.chunk_text(ref.doc_id, ref.chunk_id)
            if _overlap(item.text, chunk_text) >= _TEXT_SIM_MIN:
                any_substantively_present = True
            if ref.quote and _overlap(ref.quote, chunk_text) < _QUOTE_SIM_MIN:
                return Verdict(
                    item_id=item.item_id,
                    passed=False,
                    adjusted_item=None,
                    failing_check="evidence_validity",
                    drop_reason="cited quote not present in chunk",
                )
        if not any_substantively_present:
            return Verdict(
                item_id=item.item_id,
                passed=False,
                adjusted_item=None,
                failing_check="evidence_validity",
                drop_reason="item text not substantively in any cited chunk",
            )

        # Check 3: inference provenance length vs. effective max hops.
        max_hops = (
            ctx.max_inference_hops
            if ctx.max_inference_hops is not None
            else self._max_hops_default
        )
        if item.inference_path and len(item.inference_path) > max_hops:
            return Verdict(
                item_id=item.item_id,
                passed=False,
                adjusted_item=None,
                failing_check="inference_provenance",
                drop_reason=(
                    f"path length {len(item.inference_path)} > {max_hops}"
                ),
            )

        # Check 4: confidence calibration — high claims need corroboration.
        adjusted = item
        if item.confidence > _HIGH_CONFIDENCE and len(item.evidence) < 2:
            adjusted = item.model_copy(update={"confidence": _ROLLBACK})

        return Verdict(
            item_id=item.item_id, passed=True, adjusted_item=adjusted
        )

    def _check_contradiction(
        self,
        item: ArtifactItem,
        accepted: list[ArtifactItem],
    ) -> Verdict:
        """Check 5: item is dropped if it contradicts a previously accepted
        higher-confidence item AND is not explicitly flagged as a conflict."""
        if "conflict" in item.domain_tags or item.metadata.get(
            "conflict_annotation"
        ) is True:
            return Verdict(item_id=item.item_id, passed=True, adjusted_item=item)

        for prior in accepted:
            if prior.confidence <= item.confidence:
                continue
            if (
                _overlap(prior.text, item.text) >= _CONTRADICTION_SIM
                and _opposite(prior.text, item.text)
            ):
                return Verdict(
                    item_id=item.item_id,
                    passed=False,
                    adjusted_item=None,
                    failing_check="contradiction",
                    drop_reason=(
                        f"contradicts higher-confidence {prior.item_id}"
                    ),
                )
        return Verdict(item_id=item.item_id, passed=True, adjusted_item=item)
