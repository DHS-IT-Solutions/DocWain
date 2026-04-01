"""Hierarchical entity resolution with fuzzy string matching.

Deduplicates and merges entities extracted across documents into canonical
groups, using rapidfuzz for approximate name matching within the same type.
"""

from __future__ import annotations

from typing import Any

from rapidfuzz.fuzz import partial_ratio


class EntityResolver:
    """Resolves a flat list of entity mentions into canonical groups.

    Parameters
    ----------
    fuzzy_threshold:
        Minimum ``rapidfuzz.fuzz.partial_ratio`` score (0–100) for two names
        to be considered a match.  Default is 75.
    """

    def __init__(self, fuzzy_threshold: int = 75) -> None:
        self.fuzzy_threshold = fuzzy_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Resolve *entities* into canonical groups.

        Parameters
        ----------
        entities:
            Each element must have the keys ``name``, ``type``,
            ``confidence``, and ``doc_id``.

        Returns
        -------
        list[dict]
            Each group contains:
            ``canonical_name``, ``type``, ``aliases`` (sorted),
            ``doc_ids`` (sorted), ``confidence`` (weighted average),
            ``mention_count``.
        """
        # Internal representation while building groups.
        # Each group is a dict with the same output keys plus a
        # ``_total_confidence`` accumulator used for the weighted average.
        groups: list[dict[str, Any]] = []

        for entity in entities:
            name: str = entity["name"]
            etype: str = entity["type"]
            confidence: float = float(entity["confidence"])
            doc_id: str = entity["doc_id"]

            matched_group = self._find_matching_group(groups, name, etype)

            if matched_group is None:
                # Create a new group for this entity.
                groups.append(
                    {
                        "canonical_name": name,
                        "type": etype,
                        "aliases": [],
                        "doc_ids": [doc_id],
                        "_total_confidence": confidence,
                        "confidence": confidence,
                        "mention_count": 1,
                    }
                )
            else:
                self._merge_into_group(matched_group, name, confidence, doc_id)

        return [self._finalise_group(g) for g in groups]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_matching_group(
        self,
        groups: list[dict[str, Any]],
        name: str,
        etype: str,
    ) -> dict[str, Any] | None:
        """Return the first group of the same type whose canonical name or any
        alias matches *name* with score >= ``fuzzy_threshold``, or ``None``."""
        for group in groups:
            if group["type"] != etype:
                continue
            candidates = [group["canonical_name"]] + group["aliases"]
            for candidate in candidates:
                # Exact match always merges.
                if candidate.lower() == name.lower():
                    return group
                # Fuzzy match within threshold.
                if partial_ratio(candidate, name) >= self.fuzzy_threshold:
                    return group
        return None

    def _merge_into_group(
        self,
        group: dict[str, Any],
        name: str,
        confidence: float,
        doc_id: str,
    ) -> None:
        """Integrate a new mention into an existing *group* in-place."""
        # Promote to canonical if this mention has higher confidence.
        if confidence > group["confidence"]:
            # Demote current canonical to aliases if it differs from the new name.
            if group["canonical_name"].lower() != name.lower():
                if group["canonical_name"] not in group["aliases"]:
                    group["aliases"].append(group["canonical_name"])
            group["canonical_name"] = name
        else:
            # Add as alias if the name is genuinely different.
            if (
                name.lower() != group["canonical_name"].lower()
                and name not in group["aliases"]
            ):
                group["aliases"].append(name)

        # Track all documents.
        if doc_id not in group["doc_ids"]:
            group["doc_ids"].append(doc_id)

        # Update weighted-average confidence.
        group["_total_confidence"] += confidence
        group["mention_count"] += 1
        group["confidence"] = group["_total_confidence"] / group["mention_count"]

    @staticmethod
    def _finalise_group(group: dict[str, Any]) -> dict[str, Any]:
        """Return a clean output dict (removes internal accumulators, sorts lists)."""
        return {
            "canonical_name": group["canonical_name"],
            "type": group["type"],
            "aliases": sorted(group["aliases"]),
            "doc_ids": sorted(group["doc_ids"]),
            "confidence": group["confidence"],
            "mention_count": group["mention_count"],
        }
