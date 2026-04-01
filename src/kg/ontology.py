"""Domain ontology for typed relationship schemas used in Knowledge Graph extraction and validation."""

from typing import List, Optional

# ---------------------------------------------------------------------------
# Relationship definitions per domain
# Each relationship is a dict with keys:
#   name         - unique relationship name (snake_case)
#   source_types - list of entity types that can be the source
#   target_types - list of entity types that can be the target
# ---------------------------------------------------------------------------

_LEGAL_RELATIONSHIPS: List[dict] = [
    {
        "name": "party_to",
        "source_types": ["PERSON", "ORGANIZATION"],
        "target_types": ["CONTRACT", "AGREEMENT"],
    },
    {
        "name": "signatory_of",
        "source_types": ["PERSON"],
        "target_types": ["CONTRACT", "AGREEMENT"],
    },
    {
        "name": "governed_by",
        "source_types": ["CONTRACT", "AGREEMENT"],
        "target_types": ["LAW", "REGULATION", "CLAUSE"],
    },
    {
        "name": "amends",
        "source_types": ["CONTRACT", "AGREEMENT"],
        "target_types": ["CONTRACT", "AGREEMENT"],
    },
    {
        "name": "supersedes",
        "source_types": ["CONTRACT", "AGREEMENT", "LAW", "REGULATION"],
        "target_types": ["CONTRACT", "AGREEMENT", "LAW", "REGULATION"],
    },
    {
        "name": "terminates",
        "source_types": ["CONTRACT", "AGREEMENT"],
        "target_types": ["CONTRACT", "AGREEMENT"],
    },
    {
        "name": "effective_from",
        "source_types": ["CONTRACT", "AGREEMENT", "CLAUSE"],
        "target_types": ["DATE"],
    },
    {
        "name": "expires_on",
        "source_types": ["CONTRACT", "AGREEMENT", "CLAUSE"],
        "target_types": ["DATE"],
    },
]

_FINANCIAL_RELATIONSHIPS: List[dict] = [
    {
        "name": "invoiced_by",
        "source_types": ["INVOICE"],
        "target_types": ["PERSON", "ORGANIZATION"],
    },
    {
        "name": "paid_to",
        "source_types": ["PAYMENT"],
        "target_types": ["PERSON", "ORGANIZATION"],
    },
    {
        "name": "line_item_of",
        "source_types": ["LINE_ITEM"],
        "target_types": ["INVOICE"],
    },
    {
        "name": "totals_to",
        "source_types": ["INVOICE"],
        "target_types": ["AMOUNT"],
    },
    {
        "name": "billed_on",
        "source_types": ["INVOICE"],
        "target_types": ["DATE"],
    },
    {
        "name": "due_on",
        "source_types": ["INVOICE", "PAYMENT"],
        "target_types": ["DATE"],
    },
]

_HR_RELATIONSHIPS: List[dict] = [
    {
        "name": "employed_by",
        "source_types": ["PERSON"],
        "target_types": ["ORGANIZATION"],
    },
    {
        "name": "reports_to",
        "source_types": ["PERSON"],
        "target_types": ["PERSON"],
    },
    {
        "name": "holds_certification",
        "source_types": ["PERSON"],
        "target_types": ["CERTIFICATION"],
    },
    {
        "name": "worked_during",
        "source_types": ["PERSON"],
        "target_types": ["DATE"],
    },
    {
        "name": "role_of",
        "source_types": ["ROLE"],
        "target_types": ["PERSON"],
    },
]

_MEDICAL_RELATIONSHIPS: List[dict] = [
    {
        "name": "diagnosed_with",
        "source_types": ["PERSON"],
        "target_types": ["CONDITION", "MEDICAL_TERM"],
    },
    {
        "name": "prescribed",
        "source_types": ["PERSON"],
        "target_types": ["MEDICATION", "SUBSTANCE"],
    },
    {
        "name": "treated_by",
        "source_types": ["PERSON"],
        "target_types": ["PERSON", "ORGANIZATION"],
    },
    {
        "name": "allergic_to",
        "source_types": ["PERSON"],
        "target_types": ["SUBSTANCE", "MEDICATION"],
    },
    {
        "name": "admitted_on",
        "source_types": ["PERSON"],
        "target_types": ["DATE"],
    },
]

_GENERIC_RELATIONSHIPS: List[dict] = [
    {
        "name": "related_to",
        "source_types": ["ENTITY"],
        "target_types": ["ENTITY"],
    },
    {
        "name": "mentioned_in",
        "source_types": ["ENTITY"],
        "target_types": ["DOCUMENT"],
    },
    {
        "name": "part_of",
        "source_types": ["ENTITY"],
        "target_types": ["ENTITY"],
    },
    {
        "name": "located_at",
        "source_types": ["ENTITY", "PERSON", "ORGANIZATION"],
        "target_types": ["LOCATION"],
    },
]

# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------

_DOMAIN_MAP: dict = {
    "legal": _LEGAL_RELATIONSHIPS,
    "financial": _FINANCIAL_RELATIONSHIPS,
    "hr": _HR_RELATIONSHIPS,
    "medical": _MEDICAL_RELATIONSHIPS,
    "generic": _GENERIC_RELATIONSHIPS,
}

# Public: ordered list of domain names
DOMAINS: List[str] = list(_DOMAIN_MAP.keys())

# Public: flat list of all relationships with "domain" field injected
ALL_RELATIONSHIPS: List[dict] = []
for _domain_name, _rels in _DOMAIN_MAP.items():
    for _rel in _rels:
        ALL_RELATIONSHIPS.append({**_rel, "domain": _domain_name})

# Internal fast-lookup index by relationship name
_RELATIONSHIP_INDEX: dict = {r["name"]: r for r in ALL_RELATIONSHIPS}

# ---------------------------------------------------------------------------
# Domain hint sets used for detect_domain()
# ---------------------------------------------------------------------------

_DOMAIN_HINTS: dict = {
    "legal": {"CLAUSE", "CONTRACT", "AGREEMENT", "LAW", "REGULATION"},
    "financial": {"INVOICE", "PAYMENT", "LINE_ITEM", "AMOUNT"},
    "hr": {"CERTIFICATION", "ROLE", "SKILL"},
    "medical": {"MEDICAL_TERM", "CONDITION", "MEDICATION", "SUBSTANCE"},
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_domain_relationships(domain: str) -> List[str]:
    """Return relationship names for the given domain.

    Falls back to the generic domain for unknown domain names.
    """
    rels = _DOMAIN_MAP.get(domain, _GENERIC_RELATIONSHIPS)
    return [r["name"] for r in rels]


def get_relationship_schema(name: str) -> Optional[dict]:
    """Return the full schema dict for a relationship name, or None if unknown."""
    return _RELATIONSHIP_INDEX.get(name)


def detect_domain(entities: List[dict]) -> str:
    """Detect the document domain from a list of entity dicts.

    Each entity dict is expected to have at least a "type" key.
    The domain with the most hint matches wins; ties favour declaration order
    in DOMAINS.  Falls back to "generic" when no hints match.
    """
    entity_types = {e.get("type", "") for e in entities}

    best_domain = "generic"
    best_count = 0

    for domain in DOMAINS:
        if domain == "generic":
            continue
        hints = _DOMAIN_HINTS.get(domain, set())
        count = len(entity_types & hints)
        if count > best_count:
            best_count = count
            best_domain = domain

    return best_domain
