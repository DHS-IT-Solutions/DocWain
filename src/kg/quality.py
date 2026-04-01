"""KG Quality & Completeness Scoring.

Provides functions to score entity completeness, relationship evidence strength,
and detect structural gaps in the knowledge graph.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "score_entity_completeness",
    "score_relationship_evidence",
    "detect_gaps",
]

# ---------------------------------------------------------------------------
# Field weights for entity completeness
# ---------------------------------------------------------------------------

_FIELD_WEIGHTS: dict[str, float] = {
    "name": 0.20,
    "aliases": 0.15,
    "relationships": 0.25,
    "doc_ids": 0.15,
    "temporal_bounds": 0.15,
    "confidence": 0.10,
    # "type" is always present — weight 0.00, not included here
}

# ---------------------------------------------------------------------------
# Expected relationships per entity type for gap detection
# ---------------------------------------------------------------------------

_EXPECTED_RELATIONSHIPS: dict[str, set[str]] = {
    "PERSON": {"employed_by", "reports_to", "role_of", "signatory_of"},
    "ORGANIZATION": {"employs", "party_to", "located_at"},
}


def _is_present(value: Any) -> bool:
    """Return True when *value* is non-None and non-empty."""
    if value is None:
        return False
    if isinstance(value, (str, list, dict, set, tuple)):
        return len(value) > 0
    # Numeric 0 counts as present (e.g. a confidence score of 0.0 was
    # explicitly set).  Everything else truthy also counts.
    return True


def score_entity_completeness(entity: dict) -> float:
    """Score how complete an entity record is.

    Args:
        entity: A dict representing a KG entity.  Expected optional keys:
            name, aliases, relationships, doc_ids, temporal_bounds, confidence.
            The *type* key is always assumed present and carries no weight.

    Returns:
        A float in [0.0, 1.0].  A name-only entity scores ~0.20.
    """
    total = 0.0
    for field, weight in _FIELD_WEIGHTS.items():
        if _is_present(entity.get(field)):
            total += weight
    return min(total, 1.0)


def score_relationship_evidence(doc_count: int) -> float:
    """Map the number of supporting documents to an evidence quality score.

    Scale:
        0  -> 0.0
        1  -> 0.3
        2  -> 0.6
        3  -> 0.7
        4  -> 0.8
        5+ -> min(0.9 + (count - 5) * 0.02, 1.0)

    Args:
        doc_count: Number of distinct documents that evidence the relationship.

    Returns:
        A float in [0.0, 1.0].
    """
    if doc_count <= 0:
        return 0.0
    if doc_count == 1:
        return 0.3
    if doc_count == 2:
        return 0.6
    if doc_count in (3, 4):
        return 0.7 + (doc_count - 3) * 0.1
    # 5 or more
    return min(0.9 + (doc_count - 5) * 0.02, 1.0)


def detect_gaps(entities: list[dict]) -> list[dict]:
    """Identify structural gaps in a list of KG entities.

    Gap rules:
    1. Entity has *no* relationships from the expected set for its type.
    2. Entity has *role_of* but is missing *employed_by* (PERSON only).

    Args:
        entities: List of entity dicts.  Each dict should have at minimum
            a *type* key and optionally a *relationships* key whose value
            is a list/set of relationship-type strings.

    Returns:
        A list of gap dicts, each with keys:
            - entity: the original entity dict
            - type:   the entity type string
            - gap:    a human-readable description of the missing information
    """
    gaps: list[dict] = []

    for entity in entities:
        etype = entity.get("type", "")
        expected = _EXPECTED_RELATIONSHIPS.get(etype)
        if expected is None:
            # Unknown type — no rules defined, skip
            continue

        raw_rels = entity.get("relationships") or []
        # Support list-of-strings or list-of-dicts with a "type" key
        if raw_rels and isinstance(raw_rels[0], dict):
            present = {r.get("type", "") for r in raw_rels}
        else:
            present = set(raw_rels)

        # Gap rule 1: none of the expected relationships are present
        if not present.intersection(expected):
            gaps.append({
                "entity": entity,
                "type": etype,
                "gap": (
                    f"Entity has no expected relationships "
                    f"({', '.join(sorted(expected))})"
                ),
            })
            continue  # rule 2 is only meaningful if there are some relationships

        # Gap rule 2: has role_of but no employed_by (PERSON only)
        if etype == "PERSON" and "role_of" in present and "employed_by" not in present:
            gaps.append({
                "entity": entity,
                "type": etype,
                "gap": "Entity has role_of but is missing employed_by",
            })

    return gaps
