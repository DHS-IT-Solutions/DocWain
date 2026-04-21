"""PackSummary — canonical aggregated view over a Phase 3 ``list[PackedItem]``.

ERRATA §10 defines the shape:

- ``total_chunks`` / ``distinct_docs`` — cardinality for thin-pack detection
- ``has_sme_artifacts`` — True if any item carries ``sme_backed=True`` or a
  recognised ``metadata.artifact_type``
- ``bank_entries`` — plain-dict view of Recommendation Bank items (consumed by
  the grounding post-pass in ``recommendation_grounding.py``, which operates
  on dicts — not dataclasses — to keep the transform tight)
- ``evidence_items`` — PackedItems that carry provenance; passed into rich
  prompt builders for inline citation
- ``insights`` — PackedItems whose ``artifact_type == "insight"``

The canonical factory :meth:`PackSummary.from_packed_items` is the single
construction path. Callers never instantiate ``PackSummary`` by hand.

This module is formatting-adjacent — per the memory rule it lives in the
generation package, NOT in retrieval/ or intelligence/.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.retrieval.types import PackedItem


_SME_ARTIFACT_TYPES = frozenset(
    {"dossier", "insight", "comparative", "recommendation"}
)


@dataclass(frozen=True)
class PackSummary:
    """Aggregated view over a ``list[PackedItem]`` for rich-mode shape decisions.

    ``bank_entries`` / ``evidence_items`` / ``insights`` are derived views over
    the underlying pack list. Task 8 and Task 9 read them directly; no further
    transformation is needed at the call site.
    """

    total_chunks: int
    distinct_docs: int
    has_sme_artifacts: bool
    bank_entries: tuple[dict, ...] = field(default_factory=tuple)
    evidence_items: tuple[PackedItem, ...] = field(default_factory=tuple)
    insights: tuple[PackedItem, ...] = field(default_factory=tuple)

    @classmethod
    def from_packed_items(cls, items: list[PackedItem]) -> "PackSummary":
        """Build a :class:`PackSummary` from a post-assembly PackedItem list.

        Filters by ``metadata["artifact_type"]``:

        - ``"recommendation"`` items are emitted as plain-dict ``bank_entries``
          with ``recommendation`` / ``evidence`` / ``metadata`` keys. The
          grounding post-pass reads these dicts directly.
        - ``"insight"`` items are kept as :class:`PackedItem` so the rich
          prompt builder can render them with provenance intact.
        - Any item with populated ``provenance`` contributes to
          ``evidence_items`` — every citable chunk is inlined into the prompt.

        ``has_sme_artifacts`` is True when any item carries
        ``sme_backed=True`` OR a recognised artifact_type. This is the single
        signal the shape resolver consults for the rich / honest-compact split.
        """
        evidence: list[PackedItem] = []
        insights: list[PackedItem] = []
        bank: list[dict] = []
        docs: set[str] = set()
        has_sme = False

        for item in items:
            artifact = None
            if item.metadata:
                artifact = item.metadata.get("artifact_type")

            if item.sme_backed or artifact in _SME_ARTIFACT_TYPES:
                has_sme = True

            if item.provenance:
                evidence.append(item)
                for doc_id, _chunk_id in item.provenance:
                    docs.add(doc_id)

            if artifact == "insight":
                insights.append(item)
            if artifact == "recommendation":
                bank.append(
                    {
                        "recommendation": item.text,
                        "evidence": [
                            f"{d}:{c}" for d, c in item.provenance
                        ],
                        "metadata": dict(item.metadata or {}),
                    }
                )

        return cls(
            total_chunks=len(items),
            distinct_docs=len(docs),
            has_sme_artifacts=has_sme,
            bank_entries=tuple(bank),
            evidence_items=tuple(evidence),
            insights=tuple(insights),
        )
