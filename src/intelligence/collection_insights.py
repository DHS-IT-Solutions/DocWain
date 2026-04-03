"""Collection-level intelligence insights for document profiles.

Analyzes entities, relationships, and document metadata across an entire
profile to surface distributions, patterns, gaps, and anomalies.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Domain-specific insight generators
# ---------------------------------------------------------------------------


def _distribution(items: List[Any], key: str) -> Dict[str, int]:
    """Count occurrences of *key* across a list of dicts."""
    counts: Dict[str, int] = {}
    for item in items:
        val = item.get(key) if isinstance(item, dict) else getattr(item, key, None)
        if val is not None:
            label = str(val)
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def _detect_gaps(entities: List[Dict], required_fields: List[str]) -> List[str]:
    """Return human-readable gap descriptions for missing required fields."""
    gaps: List[str] = []
    for field in required_fields:
        missing = sum(1 for e in entities if not e.get(field))
        if missing > 0:
            pct = round(missing / max(len(entities), 1) * 100, 1)
            gaps.append(f"{missing}/{len(entities)} entities ({pct}%) missing '{field}'")
    return gaps


def _detect_anomalies_numeric(
    entities: List[Dict], field: str, label: str, z_threshold: float = 2.0
) -> List[str]:
    """Flag entities whose *field* value deviates > z_threshold std-devs."""
    values = []
    indexed: List[tuple] = []
    for e in entities:
        v = e.get(field)
        if v is not None:
            try:
                v_float = float(v)
                values.append(v_float)
                indexed.append((e, v_float))
            except (TypeError, ValueError):
                continue
    if len(values) < 3:
        return []
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5
    if std == 0:
        return []
    anomalies: List[str] = []
    for e, v in indexed:
        z = abs(v - mean) / std
        if z > z_threshold:
            name = e.get("name", e.get("id", "unknown"))
            direction = "above" if v > mean else "below"
            anomalies.append(
                f"{label} '{name}': {field}={v} is {z:.1f} std-devs {direction} mean ({mean:.1f})"
            )
    return anomalies


# ---------------------------------------------------------------------------
# HR / Recruitment insights
# ---------------------------------------------------------------------------


def _hr_insights(entities: List[Dict], relationships: List[Dict], doc_metadata: List[Dict]) -> Dict:
    distributions = {
        "skills": _distribution(entities, "primary_skill") if entities else {},
        "experience_bands": _experience_bands(entities),
        "role_types": _distribution(doc_metadata, "doc_type"),
    }
    patterns: List[str] = []
    top_skills = list(distributions["skills"].items())[:5]
    if top_skills:
        patterns.append(f"Top skills: {', '.join(s for s, _ in top_skills)}")
    avg_exp = _avg_numeric(entities, "experience_years")
    if avg_exp is not None:
        patterns.append(f"Average experience: {avg_exp:.1f} years")
    gaps = _detect_gaps(entities, ["experience_years", "certifications", "email"])
    anomalies = _detect_anomalies_numeric(entities, "experience_years", "Candidate")
    return {"distributions": distributions, "patterns": patterns, "gaps": gaps, "anomalies": anomalies}


def _experience_bands(entities: List[Dict]) -> Dict[str, int]:
    bands: Dict[str, int] = {"0-2": 0, "3-5": 0, "6-10": 0, "11+": 0}
    for e in entities:
        yrs = e.get("experience_years")
        if yrs is None:
            continue
        try:
            y = float(yrs)
        except (TypeError, ValueError):
            continue
        if y <= 2:
            bands["0-2"] += 1
        elif y <= 5:
            bands["3-5"] += 1
        elif y <= 10:
            bands["6-10"] += 1
        else:
            bands["11+"] += 1
    return bands


# ---------------------------------------------------------------------------
# Finance insights
# ---------------------------------------------------------------------------


def _finance_insights(entities: List[Dict], relationships: List[Dict], doc_metadata: List[Dict]) -> Dict:
    distributions = {
        "vendors": _distribution(entities, "name"),
        "payment_terms": _distribution(entities, "avg_payment_terms"),
        "document_types": _distribution(doc_metadata, "doc_type"),
    }
    patterns: List[str] = []
    total_spend = sum(float(e.get("total_spend", 0)) for e in entities if e.get("total_spend"))
    if total_spend:
        patterns.append(f"Total spend across vendors: ${total_spend:,.2f}")
    avg_terms = _avg_numeric(entities, "avg_payment_terms")
    if avg_terms is not None:
        patterns.append(f"Average payment terms: {avg_terms:.0f} days")
    gaps = _detect_gaps(entities, ["total_spend", "invoice_count", "avg_payment_terms"])
    anomalies = _detect_anomalies_numeric(entities, "total_spend", "Vendor")
    return {"distributions": distributions, "patterns": patterns, "gaps": gaps, "anomalies": anomalies}


# ---------------------------------------------------------------------------
# Legal insights
# ---------------------------------------------------------------------------


def _legal_insights(entities: List[Dict], relationships: List[Dict], doc_metadata: List[Dict]) -> Dict:
    distributions = {
        "risk_levels": _distribution(entities, "risk_level"),
        "contract_types": _distribution(doc_metadata, "doc_type"),
        "parties": _distribution(entities, "parties"),
    }
    patterns: List[str] = []
    high_risk = sum(1 for e in entities if str(e.get("risk_level", "")).lower() in ("high", "critical"))
    if high_risk:
        patterns.append(f"{high_risk} contract(s) flagged as high/critical risk")
    gaps = _detect_gaps(entities, ["key_dates", "obligations", "risk_level", "parties"])
    anomalies: List[str] = []
    for e in entities:
        if str(e.get("risk_level", "")).lower() == "critical":
            anomalies.append(f"Contract '{e.get('name', 'unknown')}' has CRITICAL risk level")
    return {"distributions": distributions, "patterns": patterns, "gaps": gaps, "anomalies": anomalies}


# ---------------------------------------------------------------------------
# Logistics insights
# ---------------------------------------------------------------------------


def _logistics_insights(entities: List[Dict], relationships: List[Dict], doc_metadata: List[Dict]) -> Dict:
    distributions = {
        "suppliers": _distribution(entities, "supplier"),
        "document_types": _distribution(doc_metadata, "doc_type"),
    }
    patterns: List[str] = []
    low_stock = [e for e in entities if _is_low_stock(e)]
    if low_stock:
        patterns.append(f"{len(low_stock)} product(s) at or below reorder point")
    avg_lead = _avg_numeric(entities, "lead_time_days")
    if avg_lead is not None:
        patterns.append(f"Average supplier lead time: {avg_lead:.0f} days")
    gaps = _detect_gaps(entities, ["stock_level", "reorder_point", "supplier", "lead_time_days"])
    anomalies = _detect_anomalies_numeric(entities, "lead_time_days", "Product")
    return {"distributions": distributions, "patterns": patterns, "gaps": gaps, "anomalies": anomalies}


def _is_low_stock(entity: Dict) -> bool:
    stock = entity.get("stock_level")
    reorder = entity.get("reorder_point")
    if stock is None or reorder is None:
        return False
    try:
        return float(stock) <= float(reorder)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Medical insights
# ---------------------------------------------------------------------------


def _medical_insights(entities: List[Dict], relationships: List[Dict], doc_metadata: List[Dict]) -> Dict:
    distributions = {
        "diagnoses": _distribution(entities, "primary_diagnosis"),
        "providers": _distribution(entities, "primary_provider"),
        "document_types": _distribution(doc_metadata, "doc_type"),
    }
    patterns: List[str] = []
    med_counts = [len(e.get("medications", [])) for e in entities if isinstance(e.get("medications"), list)]
    if med_counts:
        avg_meds = sum(med_counts) / len(med_counts)
        patterns.append(f"Average medications per patient: {avg_meds:.1f}")
    gaps = _detect_gaps(entities, ["diagnoses", "medications", "providers"])
    anomalies: List[str] = []
    for e in entities:
        meds = e.get("medications", [])
        if isinstance(meds, list) and len(meds) > 10:
            anomalies.append(
                f"Patient '{e.get('name', 'unknown')}' has {len(meds)} medications (polypharmacy risk)"
            )
    return {"distributions": distributions, "patterns": patterns, "gaps": gaps, "anomalies": anomalies}


# ---------------------------------------------------------------------------
# Generic insights
# ---------------------------------------------------------------------------


def _generic_insights(entities: List[Dict], relationships: List[Dict], doc_metadata: List[Dict]) -> Dict:
    distributions = {
        "entity_types": _distribution(entities, "type"),
        "document_types": _distribution(doc_metadata, "doc_type"),
    }
    patterns: List[str] = []
    if entities:
        patterns.append(f"Total entities extracted: {len(entities)}")
    if relationships:
        patterns.append(f"Total relationships mapped: {len(relationships)}")
    gaps = _detect_gaps(entities, ["type", "name"])
    anomalies: List[str] = []
    return {"distributions": distributions, "patterns": patterns, "gaps": gaps, "anomalies": anomalies}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _avg_numeric(entities: List[Dict], field: str) -> Optional[float]:
    vals = []
    for e in entities:
        v = e.get(field)
        if v is not None:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DOMAIN_FN = {
    "hr_recruitment": _hr_insights,
    "finance": _finance_insights,
    "legal": _legal_insights,
    "logistics": _logistics_insights,
    "medical": _medical_insights,
    "generic": _generic_insights,
}


def generate_insights(
    profile_type: str,
    entities: List[Dict],
    relationships: List[Dict],
    doc_metadata: List[Dict],
) -> Dict[str, Any]:
    """Generate domain-aware collection insights.

    Args:
        profile_type: One of hr_recruitment, finance, legal, logistics, medical, generic.
        entities: List of entity dicts from the knowledge graph or extraction layer.
        relationships: List of relationship dicts.
        doc_metadata: List of document metadata dicts for the profile.

    Returns:
        ``{distributions: {}, patterns: [], gaps: [], anomalies: []}``
    """
    fn = _DOMAIN_FN.get(profile_type, _generic_insights)
    try:
        return fn(entities, relationships, doc_metadata)
    except Exception:
        logger.exception("Error generating insights for profile_type=%s", profile_type)
        return {"distributions": {}, "patterns": [], "gaps": [], "anomalies": []}
