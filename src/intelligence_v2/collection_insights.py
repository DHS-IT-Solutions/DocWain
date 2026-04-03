"""Collection-level analysis — domain-aware insight generation.

Produces distributions, patterns, gaps, and anomalies from entities,
relationships, and document metadata.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List


def generate_insights(
    profile_type: str,
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    doc_metadata: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate collection-level insights appropriate to the domain.

    Returns a dict with keys: distributions, patterns, gaps, anomalies.
    """
    base = _base_insights(entities, relationships, doc_metadata)
    domain_fn = _DOMAIN_INSIGHTS.get(profile_type)
    if domain_fn:
        domain = domain_fn(entities, relationships, doc_metadata)
        # Merge domain insights into base
        for key in ("distributions", "patterns", "gaps", "anomalies"):
            if key in domain:
                if isinstance(base.get(key), dict) and isinstance(domain[key], dict):
                    base[key].update(domain[key])
                elif isinstance(base.get(key), list) and isinstance(domain[key], list):
                    base[key].extend(domain[key])
                else:
                    base[key] = domain[key]
    return base


# ---------------------------------------------------------------------------
# Base insights (all domains)
# ---------------------------------------------------------------------------


def _base_insights(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    doc_metadata: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Common insights applicable to any domain."""
    # Entity type distribution
    type_counts = Counter(e.get("type", "OTHER") for e in entities)

    # Document type distribution
    doc_type_counts = Counter(
        d.get("doc_type") or d.get("document_type") or "unknown"
        for d in doc_metadata
    )

    # Relationship type distribution
    rel_type_counts = Counter(
        r.get("relation_type") or r.get("relation", "UNKNOWN")
        for r in relationships
    )

    # Patterns: most connected entities
    entity_connections: Counter = Counter()
    for r in relationships:
        entity_connections[r.get("source", "")] += 1
        entity_connections[r.get("target", "")] += 1
    most_connected = entity_connections.most_common(10)

    # Gaps: entity types with very few instances
    gaps: List[str] = []
    for etype, count in type_counts.items():
        if count == 1:
            gaps.append(f"Only one {etype} entity found — may be incomplete")

    return {
        "distributions": {
            "entity_types": dict(type_counts.most_common(20)),
            "document_types": dict(doc_type_counts),
            "relationship_types": dict(rel_type_counts.most_common(15)),
        },
        "patterns": [
            f"{name} is highly connected ({count} relationships)"
            for name, count in most_connected
            if count >= 3
        ],
        "gaps": gaps,
        "anomalies": [],
    }


# ---------------------------------------------------------------------------
# HR insights
# ---------------------------------------------------------------------------


def _hr_insights(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    doc_metadata: List[Dict[str, Any]],
) -> Dict[str, Any]:
    skills = [e.get("name", "") for e in entities if e.get("type", "").upper() in ("SKILL", "TECHNOLOGY")]
    skill_counts = Counter(skills)
    persons = [e for e in entities if e.get("type", "").upper() in ("PERSON", "CANDIDATE")]
    certs = [e for e in entities if e.get("type", "").upper() in ("CERTIFICATION", "CERTIFICATE")]

    patterns: List[str] = []
    top_skills = skill_counts.most_common(5)
    if top_skills:
        patterns.append(
            "Most common skills: " + ", ".join(f"{s} ({c})" for s, c in top_skills)
        )

    gaps: List[str] = []
    if len(persons) > 0 and len(certs) == 0:
        gaps.append("No certifications found across candidates — consider requesting")

    return {
        "distributions": {"skill_frequency": dict(skill_counts.most_common(20))},
        "patterns": patterns,
        "gaps": gaps,
        "anomalies": [],
    }


# ---------------------------------------------------------------------------
# Finance insights
# ---------------------------------------------------------------------------


def _finance_insights(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    doc_metadata: List[Dict[str, Any]],
) -> Dict[str, Any]:
    vendors = [e for e in entities if e.get("type", "").upper() in ("VENDOR", "SUPPLIER", "ORGANIZATION", "COMPANY")]
    amounts = [e.get("name", "") for e in entities if e.get("type", "").upper() in ("AMOUNT", "MONETARY_VALUE", "MONEY")]

    vendor_counts = Counter(v.get("name", "") for v in vendors)

    # Try to parse numeric amounts for anomaly detection
    parsed_amounts: List[float] = []
    for a in amounts:
        cleaned = a.replace(",", "").replace("$", "").replace("EUR", "").replace("GBP", "").strip()
        try:
            parsed_amounts.append(float(cleaned))
        except (ValueError, TypeError):
            continue

    anomalies: List[str] = []
    if parsed_amounts:
        avg = sum(parsed_amounts) / len(parsed_amounts)
        for a in parsed_amounts:
            if a > avg * 5 and avg > 0:
                anomalies.append(f"Amount {a:,.2f} is significantly above average ({avg:,.2f})")

    return {
        "distributions": {
            "spend_by_vendor": dict(vendor_counts.most_common(20)),
        },
        "patterns": [],
        "gaps": [],
        "anomalies": anomalies[:5],
    }


# ---------------------------------------------------------------------------
# Legal insights
# ---------------------------------------------------------------------------


def _legal_insights(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    doc_metadata: List[Dict[str, Any]],
) -> Dict[str, Any]:
    obligations = [e for e in entities if e.get("type", "").upper() in ("OBLIGATION", "CLAUSE", "TERM")]
    parties = [e for e in entities if e.get("type", "").upper() in ("PARTY", "ORGANIZATION", "COMPANY", "PERSON")]
    dates = [e for e in entities if e.get("type", "").upper() in ("DATE", "DEADLINE", "EXPIRY_DATE", "RENEWAL_DATE")]

    party_counts = Counter(p.get("name", "") for p in parties)

    gaps: List[str] = []
    if obligations and not dates:
        gaps.append("Obligations found but no associated dates — deadlines may be missing")
    if len(parties) < 2:
        gaps.append("Fewer than 2 parties identified — contract may be incompletely parsed")

    return {
        "distributions": {
            "obligations_count": len(obligations),
            "parties": dict(party_counts),
        },
        "patterns": [],
        "gaps": gaps,
        "anomalies": [],
    }


# ---------------------------------------------------------------------------
# Logistics insights
# ---------------------------------------------------------------------------


def _logistics_insights(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    doc_metadata: List[Dict[str, Any]],
) -> Dict[str, Any]:
    products = [e for e in entities if e.get("type", "").upper() in ("PRODUCT", "ITEM", "SKU", "GOODS")]
    suppliers = [e for e in entities if e.get("type", "").upper() in ("SUPPLIER", "VENDOR", "MANUFACTURER")]
    locations = [e for e in entities if e.get("type", "").upper() in ("LOCATION", "WAREHOUSE", "FACILITY")]

    supplier_counts = Counter(s.get("name", "") for s in suppliers)

    gaps: List[str] = []
    if products and not suppliers:
        gaps.append("Products found but no suppliers identified")
    if products and not locations:
        gaps.append("Products found but no warehouse/location data")

    return {
        "distributions": {
            "product_count": len(products),
            "supplier_distribution": dict(supplier_counts.most_common(20)),
            "location_count": len(locations),
        },
        "patterns": [],
        "gaps": gaps,
        "anomalies": [],
    }


# ---------------------------------------------------------------------------
# Medical insights
# ---------------------------------------------------------------------------


def _medical_insights(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    doc_metadata: List[Dict[str, Any]],
) -> Dict[str, Any]:
    diagnoses = [e for e in entities if e.get("type", "").upper() in ("DIAGNOSIS", "CONDITION", "DISEASE")]
    medications = [e for e in entities if e.get("type", "").upper() in ("MEDICATION", "DRUG", "PRESCRIPTION")]
    patients = [e for e in entities if e.get("type", "").upper() in ("PATIENT", "PERSON")]

    diag_counts = Counter(d.get("name", "") for d in diagnoses)
    med_counts = Counter(m.get("name", "") for m in medications)

    patterns: List[str] = []
    top_diag = diag_counts.most_common(3)
    if top_diag:
        patterns.append("Most frequent diagnoses: " + ", ".join(f"{d} ({c})" for d, c in top_diag))

    gaps: List[str] = []
    if patients and not medications:
        gaps.append("Patient records found but no medications — prescriptions may be missing")

    return {
        "distributions": {
            "diagnosis_frequency": dict(diag_counts.most_common(20)),
            "medication_frequency": dict(med_counts.most_common(20)),
        },
        "patterns": patterns,
        "gaps": gaps,
        "anomalies": [],
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_DOMAIN_INSIGHTS = {
    "hr_recruitment": _hr_insights,
    "finance": _finance_insights,
    "legal": _legal_insights,
    "logistics": _logistics_insights,
    "medical": _medical_insights,
}
